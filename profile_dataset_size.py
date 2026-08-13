"""Profile per-step compute cost of the SINDySz + FullRes conv-AE pipeline and
derive how large a dataset can be trained at a target epochs/day.

Why this exists
---------------
An Optuna trial's per-epoch wall-clock is::

    per_epoch_sec = train_steps * sec_per_train_step + valid_steps * sec_per_valid_step

The dominant per-step cost is the two ``torch.autograd.functional.jacobian``
calls (``vectorize=False``) in ``SINDySz.forward``: the decoder Jacobian runs
``F = H*W`` backward passes and the encoder Jacobian runs ``L =
latent_features`` backward passes per forward, over ``B*T`` sequences. Cost
therefore scales ~ ``B * T * (F + L)``. Dataset size does NOT change per-step
time; it changes the *number* of steps. So we measure per-step time on-GPU with
synthetic batches (timing is data-value-independent here) and then derive the
maximum dataset size analytically.

The harness reproduces the real ``training_step`` / ``validation_step`` by
running a minimal ``lightning.Trainer.fit`` over a synthetic dataset, so the
timed path is byte-for-byte the production dual-optimizer path (manual backward,
grad clipping, Jacobians, validation-with-Jacobians).

Output: a CSV (one row per HP combo) plus a printed summary highlighting the
worst case and the binding dataset-size budget for the whole search.

Run
---
    .venv/bin/python profile_dataset_size.py            # default 27-combo sweep
    .venv/bin/python profile_dataset_size.py --full     # all searched HPs

Key empirical finding (25 Hz / 19x19 cache, TITAN RTX)
------------------------------------------------------
The vectorize=False per-example Jacobian is extremely expensive and scales
SUPERLINEARLY in B*T (measured ~3 s/step at B*T=8 up to ~28-54 s/step at
B*T=32-96). Consequently the per-map_time_step cache size is often NOT the
binding constraint at map_step>=3; wall-clock is. The largest cache
(map_step=1, ~161.8k windows) is the one most likely to be time-limited.
Peak GPU memory is tiny (<0.1 GB) at this grid, so OOM is not the constraint at
25 Hz -- time is.

Tractability
------------
- ``--slow-step-sec`` (default 20): once a completed step exceeds this, record a
  single measurement and stop that combo.
- ``--skip-bt`` (default 48): combos with batch_size*time_dim above this are NOT
  measured; their per-step time is EXTRAPOLATED (flagged ``extrapolated``) from
  the fastest measured combo of the same map_step, scaled by (B*T ratio)^1.15.
  Set ``--skip-bt 0`` to measure everything (can take many minutes per large
  combo -- a single B*T=96 step is ~5 min).

Caveats
-------
- Profiled at the existing cache (f_max=25 Hz -> 19x19 grid, F=361). A 50 Hz
  cache (~37x37, F~1369) is estimated analytically in the summary (Jacobian-bound
  term scales ~ F); confirm by re-profiling on a regenerated cache.
- Disk I/O is NOT profiled (synthetic batches). With cached reads + persistent
  workers, GPU compute is expected to dominate.
- One trial per GPU at a time (matches main.py devices=[gpu_id], n_jobs=N_GPUS).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import time
import warnings

import h5py
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import lightning as L

from datasets import group_name_for_epoch_size
from model import SINDySz, ConvSINDyEncoder, ConvSINDyDecoder
from fullres_autoencoder import FullResAutoencoder


# ── Constants mirroring main.py ───────────────────────────────────────────────
BICOH_CACHE_FILE = os.getenv(
    "BICOH_CACHE_FILE", "/app/Data/WR/WR5_Run4_bicoh.hdf5"
)
SECONDS_PER_DAY = 86_400

# Fraction of sequences that end up in the training split in main.py:
#   trv = 0.8 of all; train = 0.8 of trv  -> 0.64 train, 0.16 valid, 0.20 test.
TRAIN_FRAC = 0.64
VALID_FRAC = 0.16

# Dummy LRs; irrelevant to timing.
_LR = 1e-3


# ── Cache helpers ─────────────────────────────────────────────────────────────
def read_cache_group(map_time_step: float):
    """Return (H, W, mask[1,H,W] float tensor, num_windows) for a map_time_step.

    Reads the ``es_{map_time_step:g}`` group written by precompute_bicoherence.py.
    """
    group = group_name_for_epoch_size(map_time_step)
    with h5py.File(BICOH_CACHE_FILE, "r") as f:
        if group not in f:
            raise KeyError(
                f"Cache {BICOH_CACHE_FILE} has no group '{group}' for "
                f"map_time_step={map_time_step}."
            )
        g = f[group]
        H = int(g.attrs["height"])
        W = int(g.attrs["width"])
        num_windows = int(g.attrs["num_windows"])
        mask = np.asarray(g["mask"][()], dtype=np.float32).reshape(1, H, W)
    mask_t = torch.from_numpy(mask).to(torch.get_default_dtype())
    return H, W, mask_t, num_windows


def num_sequences(num_windows: int, time_dim: int) -> int:
    """Non-overlapping sequences (stride == seq_len == time_dim), as in the dataset."""
    if num_windows < time_dim:
        return 0
    return (num_windows - time_dim) // time_dim + 1


# ── Synthetic dataset matching PrecomputedBicoherenceSequenceDataset contract ──
class _SyntheticMapDataset(data.Dataset):
    """Yields (maps[T,1,H,W], mask[1,H,W], label) with random map values.

    Timing is independent of the actual map values (no data-dependent branching
    in the hot path), so random data is valid for compute profiling.
    """

    def __init__(self, n: int, time_dim: int, H: int, W: int, mask: torch.Tensor):
        self.n = int(n)
        self.time_dim = int(time_dim)
        self.H = int(H)
        self.W = int(W)
        self.mask = mask  # [1,H,W]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        maps = torch.rand(
            self.time_dim, 1, self.H, self.W, dtype=torch.get_default_dtype()
        )
        # Apply mask so inputs live in the valid region (mirrors real data).
        maps = maps * self.mask.unsqueeze(0)
        label = torch.tensor(-1, dtype=torch.long)
        return maps, self.mask.clone(), label


def build_model(H, W, time_dim, latent_features, poly_order, map_time_step):
    """Construct SINDySz exactly as main.py does (dual-optimizer, conv AE)."""
    shared_ae = FullResAutoencoder(height=H, width=W, latent_dim=latent_features)
    encoder = ConvSINDyEncoder(height=H, width=W, latent_dim=latent_features, ae=shared_ae)
    decoder = ConvSINDyDecoder(height=H, width=W, latent_dim=latent_features, ae=shared_ae)
    model = SINDySz(
        time_dim=time_dim,
        system_features=H * W,
        latent_features=latent_features,
        poly_order=poly_order,
        encoder=encoder,
        decoder=decoder,
        lr=_LR,
        sindy_lr=_LR,
        decoder_lr=_LR,
        nan_check=False,
        use_dual_optimizers=True,
        sample_rate=(1.0 / map_time_step),
        reinit=False,
    ).to(torch.get_default_dtype())
    return model


class _StepTimer(L.Callback):
    """Time train/val steps with CUDA synchronization, excluding warmup steps.

    Adaptive early-stop: if a single measured train step exceeds
    ``slow_step_sec``, we treat the combo as "slow" and collect only ONE
    measured step (instead of ``warmup+N``) so a single pathological combo
    (e.g. large B*T with the vectorize=False per-example Jacobian) cannot stall
    the whole sweep for many minutes. The recorded time is still a valid median
    of the measured step(s).
    """

    def __init__(self, warmup_train: int, warmup_val: int, slow_step_sec: float):
        super().__init__()
        self.warmup_train = int(warmup_train)
        self.warmup_val = int(warmup_val)
        self.slow_step_sec = float(slow_step_sec)
        self._train_i = 0
        self._val_i = 0
        self._t0 = None
        self._v0 = None
        self.train_times: list[float] = []
        self.val_times: list[float] = []
        self.slow = False

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        dt = time.perf_counter() - self._t0
        self._train_i += 1
        if self._train_i > self.warmup_train:
            self.train_times.append(dt)
        # Slow-guard: as soon as ANY completed step (warmup or measured) exceeds
        # slow_step_sec, mark the combo slow, keep this step's time as the
        # measurement, and stop. This caps pathological large-B*T combos at ~1
        # warmup + 1 measured step instead of warmup_train + N steps.
        if dt > self.slow_step_sec:
            self.slow = True
            if not self.train_times:
                self.train_times.append(dt)
            trainer.should_stop = True

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._sync()
        self._v0 = time.perf_counter()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._sync()
        dt = time.perf_counter() - self._v0
        self._val_i += 1
        if self._val_i > self.warmup_val:
            self.val_times.append(dt)
        if dt > self.slow_step_sec:
            self.slow = True
            if not self.val_times:
                self.val_times.append(dt)
            trainer.should_stop = True


def profile_combo(
    *,
    batch_size,
    time_dim,
    map_time_step,
    latent_features,
    poly_order,
    train_steps,
    val_steps,
    warmup,
    slow_step_sec,
):
    """Time one HP combo. Returns a dict of measurements (or oom marker)."""
    H, W, mask, num_windows = read_cache_group(map_time_step)
    F = H * W

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Validation has no cuDNN/autograd first-step warmup benefit like training's
    # first batch, so a single warmup val step is enough.
    warmup_val = 1

    # Enough synthetic sequences to cover warmup + measured steps for both loaders.
    n_train = batch_size * (train_steps + warmup + 1)
    n_val = batch_size * (val_steps + warmup_val + 1)
    train_ds = _SyntheticMapDataset(n_train, time_dim, H, W, mask)
    val_ds = _SyntheticMapDataset(n_val, time_dim, H, W, mask)
    # num_workers=0: we are profiling compute, not I/O.
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    model = build_model(H, W, time_dim, latent_features, poly_order, map_time_step)
    timer = _StepTimer(
        warmup_train=warmup, warmup_val=warmup_val, slow_step_sec=slow_step_sec
    )

    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=train_steps + warmup,
        limit_val_batches=val_steps + warmup_val,
        num_sanity_val_steps=0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else 1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[timer],
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(model, train_loader, val_loader)
    except torch.cuda.OutOfMemoryError:
        del model, trainer
        torch.cuda.empty_cache()
        return {
            "batch_size": batch_size, "time_dim": time_dim,
            "map_time_step": map_time_step, "latent_features": latent_features,
            "poly_order": poly_order, "H": H, "W": W, "F": F,
            "oom": True, "num_windows": num_windows,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            del trainer
            try:
                del model
            except Exception:
                pass
            torch.cuda.empty_cache()
            return {
                "batch_size": batch_size, "time_dim": time_dim,
                "map_time_step": map_time_step, "latent_features": latent_features,
                "poly_order": poly_order, "H": H, "W": W, "F": F,
                "oom": True, "num_windows": num_windows,
            }
        raise

    peak_mem_gb = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available() else float("nan")
    )

    sec_train = float(np.median(timer.train_times)) if timer.train_times else float("nan")
    sec_val = float(np.median(timer.val_times)) if timer.val_times else float("nan")

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "batch_size": batch_size, "time_dim": time_dim,
        "map_time_step": map_time_step, "latent_features": latent_features,
        "poly_order": poly_order, "H": H, "W": W, "F": F,
        "oom": False, "num_windows": num_windows,
        "sec_per_train_step": sec_train,
        "sec_per_valid_step": sec_val,
        "peak_gpu_mem_gb": peak_mem_gb,
        "slow": bool(timer.slow),
    }


def derive_limits(row: dict, epochs_per_day: float) -> dict:
    """Given measured per-step times, solve for the max dataset size that hits
    ``epochs_per_day``, accounting for BOTH train and validation cost.

    Let S = total sequences. train_seq = TRAIN_FRAC*S, valid_seq = VALID_FRAC*S.
      train_steps = train_seq / B ; valid_steps = valid_seq / B
      per_epoch = train_steps*t_tr + valid_steps*t_val
                = (S/B) * (TRAIN_FRAC*t_tr + VALID_FRAC*t_val)
    Budget: per_epoch <= SECONDS_PER_DAY / epochs_per_day
      => S <= budget * B / (TRAIN_FRAC*t_tr + VALID_FRAC*t_val)
    Then source_windows = S * time_dim (stride == time_dim).
    """
    if row.get("oom"):
        return {"max_total_sequences": 0, "max_source_windows": 0,
                "capped": False, "time_or_data": "OOM"}
    t_tr = row["sec_per_train_step"]
    t_val = row["sec_per_valid_step"]
    B = row["batch_size"]
    td = row["time_dim"]
    if not (t_tr > 0):
        return {"max_total_sequences": 0, "max_source_windows": 0,
                "capped": False, "time_or_data": "NO_TIMING"}
    if not (t_val > 0):
        t_val = 0.0  # tolerate missing val timing (shouldn't happen)

    budget_sec = SECONDS_PER_DAY / float(epochs_per_day)
    per_seq_cost = (TRAIN_FRAC * t_tr + VALID_FRAC * t_val) / B
    max_total_seq_time = budget_sec / per_seq_cost

    available_seq = num_sequences(row["num_windows"], td)

    if max_total_seq_time >= available_seq:
        # Data-limited: even the full cache fits the time budget.
        max_total_seq = available_seq
        max_source_windows = row["num_windows"]
        capped = True
        which = "DATA"
    else:
        max_total_seq = int(math.floor(max_total_seq_time))
        max_source_windows = int(max_total_seq * td)
        capped = False
        which = "TIME"

    # How fast would a FULL-cache epoch be? (independent of the target E)
    full_epoch_sec = available_seq * per_seq_cost
    full_epochs_per_day = (
        SECONDS_PER_DAY / full_epoch_sec if full_epoch_sec > 0 else float("inf")
    )

    return {
        "max_total_sequences": int(max_total_seq),
        "max_train_sequences": int(max_total_seq * TRAIN_FRAC),
        "max_source_windows": int(max_source_windows),
        "available_windows": int(row["num_windows"]),
        "capped": capped,
        "time_or_data": which,
        "full_epoch_hours": full_epoch_sec / 3600.0,
        "full_epochs_per_day": full_epochs_per_day,
    }


def _extrapolate_row(bs, td, ms, lat, poly, measured, skip_bt):
    """Estimate per-step times for a combo too large to measure directly.

    Picks the fastest measured combo (prefer same map_step, then same
    latent/poly) as a baseline and scales its per-step times by the B*T ratio.
    A mild superlinear factor (exponent 1.15) is applied because the
    vectorize=False per-example Jacobian showed superlinear growth in B*T in
    corner measurements. This is a rough estimate, flagged ``extrapolated`` and
    should be confirmed by a direct measurement (--skip-bt 0) if it lands near
    a decision boundary.
    """
    H, W, mask, num_windows = read_cache_group(ms)
    F = H * W
    bt = bs * td

    # Prefer a baseline with the same map_step; fall back to any measured combo.
    candidates = [r for r in measured if r["map_time_step"] == ms] or list(measured)
    base = min(candidates, key=lambda r: r["sec_per_train_step"]) if candidates else None

    row = {
        "batch_size": bs, "time_dim": td, "map_time_step": ms,
        "latent_features": lat, "poly_order": poly, "H": H, "W": W, "F": F,
        "oom": False, "slow": True, "extrapolated": True,
        "num_windows": num_windows,
        "peak_gpu_mem_gb": float("nan"),
    }
    if base is None:
        # No baseline yet — cannot extrapolate; leave times as NaN.
        row["sec_per_train_step"] = float("nan")
        row["sec_per_valid_step"] = float("nan")
        return row

    base_bt = base["batch_size"] * base["time_dim"]
    ratio = (bt / base_bt) ** 1.15
    row["sec_per_train_step"] = base["sec_per_train_step"] * ratio
    row["sec_per_valid_step"] = base["sec_per_valid_step"] * ratio
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true",
                    help="Sweep all searched HPs (972 combos).")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--time-dims", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--map-steps", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--latent", type=int, nargs="+", default=None,
                    help="latent_features values (default: [12], or full set with --full)")
    ap.add_argument("--poly", type=int, nargs="+", default=None,
                    help="poly_order values (default: [2], or full set with --full)")
    ap.add_argument("--epochs-per-day", type=float, nargs="+", default=[3.0, 4.0])
    ap.add_argument("--train-steps", type=int, default=6)
    ap.add_argument("--val-steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--slow-step-sec", type=float, default=20.0,
                    help="If one measured train step exceeds this, record a "
                         "single measurement and move on (keeps the sweep "
                         "tractable for pathologically slow combos).")
    ap.add_argument("--skip-bt", type=int, default=48,
                    help="Do NOT measure combos with batch_size*time_dim above "
                         "this; extrapolate their per-step time from the "
                         "fastest measured combo of the same map_step (scaled "
                         "by the B*T ratio). Set 0/negative to measure all "
                         "(can take many minutes per large combo).")
    ap.add_argument("--out", type=str, default="profile_results.csv")
    args = ap.parse_args()

    if args.skip_bt is not None and args.skip_bt <= 0:
        args.skip_bt = None

    if args.full:
        latent_vals = args.latent or [6, 9, 12, 16]
        poly_vals = args.poly or [1, 2, 3]
    else:
        latent_vals = args.latent or [12]
        poly_vals = args.poly or [2]

    combos = list(itertools.product(
        args.batch_sizes, args.time_dims, args.map_steps, latent_vals, poly_vals
    ))
    # Primary and secondary epochs/day targets.
    e_primary = min(args.epochs_per_day)  # conservative (largest per-epoch budget uses smallest E)
    print(f"Profiling {len(combos)} HP combos on "
          f"{'GPU' if torch.cuda.is_available() else 'CPU'}. "
          f"Primary target = {e_primary} epochs/day (worst-case budget).\n")

    # Measure combos in increasing B*T order so that, when a large combo is
    # skipped (--skip-bt), a smaller measured combo of the same map_step exists
    # to extrapolate from.
    combos.sort(key=lambda c: (c[0] * c[1]))

    rows = []
    measured: list[dict] = []
    for i, (bs, td, ms, lat, poly) in enumerate(combos, 1):
        bt = bs * td
        print(f"[{i}/{len(combos)}] bs={bs} T={td} map_step={ms} "
              f"latent={lat} poly={poly} (B*T={bt}) ...", flush=True)

        if args.skip_bt is not None and bt > args.skip_bt:
            row = _extrapolate_row(
                bs, td, ms, lat, poly, measured, args.skip_bt
            )
        else:
            row = profile_combo(
                batch_size=bs, time_dim=td, map_time_step=ms,
                latent_features=lat, poly_order=poly,
                train_steps=args.train_steps, val_steps=args.val_steps,
                warmup=args.warmup, slow_step_sec=args.slow_step_sec,
            )
            if not row.get("oom") and row.get("sec_per_train_step", 0) > 0:
                measured.append(row)

        for e in args.epochs_per_day:
            lim = derive_limits(row, e)
            tag = f"E{e:g}"
            row[f"max_source_windows_{tag}"] = lim["max_source_windows"]
            row[f"max_total_sequences_{tag}"] = lim["max_total_sequences"]
            row[f"capped_{tag}"] = lim["capped"]
            row[f"limit_{tag}"] = lim["time_or_data"]
            # full-cache figures do not depend on E; store once from any target.
            row["full_epoch_hours"] = lim["full_epoch_hours"]
            row["full_epochs_per_day"] = lim["full_epochs_per_day"]
        if row.get("oom"):
            print("    -> OOM (skipped)")
        elif row.get("sec_per_train_step", 0) <= 0 or math.isnan(row.get("sec_per_train_step", float("nan"))):
            print("    -> no timing (skipped, no baseline to extrapolate from)")
        else:
            tags = []
            if row.get("slow"):
                tags.append("slow")
            if row.get("extrapolated"):
                tags.append("EXTRAPOLATED")
            tag = (" [" + ",".join(tags) + "]") if tags else ""
            print(f"    -> {row['sec_per_train_step']*1000:.1f} ms/train_step, "
                  f"{row['sec_per_valid_step']*1000:.1f} ms/val_step, "
                  f"peak {row.get('peak_gpu_mem_gb', float('nan')):.2f} GB | "
                  f"FULL cache ({row['num_windows']:,} win): "
                  f"{row['full_epoch_hours']:.1f} h/epoch = "
                  f"{row['full_epochs_per_day']:.1f} epochs/day | "
                  f"max_windows(E{e_primary:g})="
                  f"{row[f'max_source_windows_E{e_primary:g}']:,}{tag}")
        rows.append(row)

    # ── CSV ───────────────────────────────────────────────────────────────────
    fieldnames = [
        "batch_size", "time_dim", "map_time_step", "latent_features", "poly_order",
        "H", "W", "F", "oom", "slow", "extrapolated",
        "sec_per_train_step", "sec_per_valid_step",
        "peak_gpu_mem_gb", "num_windows",
        "full_epoch_hours", "full_epochs_per_day",
    ]
    for e in args.epochs_per_day:
        tag = f"E{e:g}"
        fieldnames += [f"max_source_windows_{tag}", f"max_total_sequences_{tag}",
                       f"capped_{tag}", f"limit_{tag}"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # ── Summary ─────────────────────────────────────────────────────────────
    ok = [r for r in rows if not r.get("oom") and r.get("sec_per_train_step", 0) > 0]
    oom = [r for r in rows if r.get("oom")]

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if oom:
        print(f"OOM combos ({len(oom)}) — these cap practical B*T regardless of "
              f"dataset size:")
        for r in oom:
            print(f"  bs={r['batch_size']} T={r['time_dim']} "
                  f"map_step={r['map_time_step']} latent={r['latent_features']} "
                  f"poly={r['poly_order']}")
    if not ok:
        print("No successfully-timed combos.")
        return

    prim = f"E{e_primary:g}"
    worst = max(ok, key=lambda r: r["sec_per_train_step"])
    print(f"\nWorst-case (slowest) timed combo:")
    print(f"  bs={worst['batch_size']} T={worst['time_dim']} "
          f"map_step={worst['map_time_step']} latent={worst['latent_features']} "
          f"poly={worst['poly_order']} -> "
          f"{worst['sec_per_train_step']*1000:.1f} ms/train_step, "
          f"peak {worst['peak_gpu_mem_gb']:.2f} GB")

    # Full-cache achievability: the slowest combo on the full cache determines
    # whether the WHOLE search can hit the target without any dataset cap.
    worst_full = min(ok, key=lambda r: r["full_epochs_per_day"])
    print(f"\nSlowest combo on the FULL cache:")
    print(f"  bs={worst_full['batch_size']} T={worst_full['time_dim']} "
          f"map_step={worst_full['map_time_step']} "
          f"({worst_full['num_windows']:,} windows) -> "
          f"{worst_full['full_epoch_hours']:.1f} h/epoch = "
          f"{worst_full['full_epochs_per_day']:.2f} epochs/day")

    # Binding budget: smallest max_source_windows among time-limited combos.
    time_limited = [r for r in ok if r.get(f"limit_{prim}") == "TIME"]
    if time_limited:
        binding = min(time_limited, key=lambda r: r[f"max_source_windows_{prim}"])
        print(f"\nBinding worst-case dataset budget at {e_primary:g} epochs/day "
              f"(safe for the WHOLE Optuna search):")
        print(f"  <= {binding[f'max_source_windows_{prim}']:,} source windows "
              f"(binding combo bs={binding['batch_size']} T={binding['time_dim']} "
              f"map_step={binding['map_time_step']}, "
              f"{binding['sec_per_train_step']*1000:.0f} ms/train_step)")
        print(f"  NOTE: this is per-map_time_step source windows; a combo's own "
              f"cache size ({binding['num_windows']:,} for map_step="
              f"{binding['map_time_step']}) may already be below/above it.")
    else:
        print(f"\nAll profiled combos are DATA-limited at {e_primary:g} "
              f"epochs/day: even the full available cache trains at >= "
              f"{e_primary:g} epochs/day, so dataset size is NOT the "
              f"constraint for the profiled combos.")
        print(f"  (Slowest full-cache combo above still does "
              f"{worst_full['full_epochs_per_day']:.2f} epochs/day.)")

    # Caveat about which caches were profiled: the map_step=1 cache is the
    # largest (most windows) and therefore the most likely to be time-limited.
    profiled_steps = sorted({r["map_time_step"] for r in ok})
    print(f"\nProfiled map_time_step caches: {profiled_steps}. "
          f"Reminder: smaller map_time_step => MORE windows "
          f"(es_1~161.8k, es_3~53.9k, es_5~32.4k), so profile map_step=1 to "
          f"stress the time budget.")

    # 50 Hz analytic extrapolation note.
    F_now = ok[0]["F"]
    F_50 = 37 * 37
    scale = F_50 / F_now
    print("\n50 Hz extrapolation (analytic, CONFIRM by re-profiling a 50 Hz cache):")
    print(f"  grid 19x19 (F={F_now}) -> ~37x37 (F~{F_50}); Jacobian-bound "
          f"sec/train_step grows up to ~{scale:.1f}x, so divide the "
          f"max_source_windows budgets above by ~{scale:.1f}.")
    print("=" * 78)


if __name__ == "__main__":
    main()
