"""Precompute per-window bicoherence maps into an HDF5 cache.

Training is slow because ``RawBicoherenceSequenceDataset`` recomputes the
expensive ``compute_bicoherence`` for every window on every access (~23 ms
each), and 4 Optuna trials do this in parallel on the CPU, starving the GPUs.

This script computes every per-window bicoherence map **once** (parallelised
across CPU cores) and writes them to a single HDF5 file, one group per
``epoch_size``. Downstream, ``PrecomputedBicoherenceSequenceDataset`` reads the
maps straight from disk so all trials/workers share the same cheap lookups.

The per-window map depends only on the signal slice and the bicoherence
parameters (not on ``seq_len``/``stride``/``batch_size``), so one map set per
``epoch_size`` serves every trial.

Layout
------
    /
      es_1.0/
        maps  [num_windows, H, W] float32   (NaNs already replaced with 0)
        mask  [H, W] float32
        f1s   [H]   float64
        f2s   [W]   float64
        attrs: epoch_size, segment_seconds, segment_overlap, f_max,
               smooth_sigma, sample_rate, channel, num_windows, height, width,
               epoch_num_samples, target_fs, complete
      es_3.0/ ...
      es_5.0/ ...

``complete=True`` is written only after a group is fully populated, so an
interrupted run is detectable and safe to overwrite.

Usage
-----
    .venv/bin/python precompute_bicoherence.py \
        --data-file /app/Data/WR/WR5_Run4.hdf5 \
        --out-file /app/Data/WR/WR5_Run4_bicoh.hdf5 \
        --epoch-sizes 1.0 3.0 5.0
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np

from datasets import RawBicoherenceSequenceDataset, group_name_for_epoch_size


# ---------------------------------------------------------------------------
# Worker plumbing (module globals so params are not re-pickled per task).
# ---------------------------------------------------------------------------

_WORKER_DATASET: RawBicoherenceSequenceDataset | None = None


def _init_worker(kwargs: dict) -> None:
    """Initialise a per-worker RawBicoherenceSequenceDataset probe.

    Each worker process builds its own dataset object (which opens the source
    HDF5 read-only inside ``_segment_signal``/``_bicoherence_map`` on demand).
    Handles are never shared across processes.
    """
    global _WORKER_DATASET
    _WORKER_DATASET = RawBicoherenceSequenceDataset(**kwargs)
    # Prime grid/mask so the first real task doesn't pay the probe cost.
    _WORKER_DATASET._ensure_grid_and_mask()


def _compute_window(window_idx: int):
    """Compute one window's bicoherence map (NaN->0), returned as float32."""
    assert _WORKER_DATASET is not None
    bmap = _WORKER_DATASET._bicoherence_map(window_idx)
    return window_idx, np.nan_to_num(bmap, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-epoch_size precompute
# ---------------------------------------------------------------------------

def _group_params(probe: RawBicoherenceSequenceDataset) -> dict:
    return {
        "epoch_size": float(probe.epoch_size),
        "segment_seconds": float(probe.segment_seconds),
        "segment_overlap": float(probe.segment_overlap),
        "f_max": float(probe.f_max),
        "smooth_sigma": float(probe.smooth_sigma),
        "sample_rate": int(probe.sample_rate),
        "channel": str(probe.channel),
    }


def _group_is_complete_and_matching(f: h5py.File, name: str, probe) -> bool:
    if name not in f:
        return False
    g = f[name]
    if not bool(g.attrs.get("complete", False)):
        return False
    want = _group_params(probe)
    for k, v in want.items():
        got = g.attrs.get(k, None)
        if isinstance(v, str):
            got_s = got.decode() if isinstance(got, bytes) else got
            if got_s != v:
                return False
        else:
            if got is None or not np.isclose(float(got), float(v)):
                return False
    if int(g.attrs.get("num_windows", -1)) != int(probe.num_windows):
        return False
    return True


def precompute_epoch_size(
    out_file: str,
    data_file: str,
    epoch_size: float,
    segment_seconds: float,
    segment_overlap: float,
    f_max: float,
    smooth_sigma: float,
    sample_rate: int,
    channel: str,
    n_jobs: int,
    force: bool,
    chunk_tasks: int = 256,
) -> None:
    import multiprocessing as mp

    name = group_name_for_epoch_size(epoch_size)

    # Build a probe (main process) to discover num_windows, grid, mask, freqs.
    ds_kwargs = dict(
        data_file=data_file,
        seq_len=2,  # any legal value; only used to satisfy the constructor
        epoch_size=epoch_size,
        f_max=f_max,
        segment_seconds=segment_seconds,
        segment_overlap=segment_overlap,
        smooth_sigma=smooth_sigma,
        sample_rate=sample_rate,
        channel=channel,
    )
    probe = RawBicoherenceSequenceDataset(**ds_kwargs)
    probe._ensure_grid_and_mask()
    H, W = probe.get_grid_size()
    num_windows = probe.num_windows
    mask = probe.get_mask().numpy()[0]  # [H, W]
    f1s, f2s = probe.get_freq_axes()

    # Skip if already complete and matching (unless --force).
    # If the file exists but is unreadable (e.g. stuck SWMR flags from a
    # previous crashed run), treat it as incomplete and fall through to
    # overwrite it.
    if os.path.exists(out_file) and not force:
        skip = False
        try:
            with h5py.File(out_file, "r") as f:
                skip = _group_is_complete_and_matching(f, name, probe)
        except OSError:
            print(
                f"[{name}] cache file exists but is unreadable (stuck SWMR "
                "flags from a previous interrupted run). Recreating it."
            )
            # Delete and start fresh so the write open below is clean.
            os.remove(out_file)
        if skip:
            print(
                f"[{name}] already complete ({num_windows} windows) — skipping."
            )
            return

    print(
        f"[{name}] computing {num_windows} windows, grid {H}x{W}, "
        f"n_jobs={n_jobs} ..."
    )
    t0 = time.time()

    # Open the cache for writing. Use 'a' so other groups are preserved.
    # Do NOT use libver="latest" here: that enables SWMR-write mode which sets
    # a consistency flag that prevents subsequent "r" opens (e.g. the skip-check
    # for the next epoch_size group) even after this context closes.
    # Readers open with swmr=True + libver="latest" on their own handles, which
    # is correct for concurrent read-only access after precompute completes.
    with h5py.File(out_file, "a") as f:
        if name in f:
            del f[name]  # overwrite a stale/partial group
        g = f.create_group(name)
        g.create_dataset("mask", data=mask.astype(np.float32))
        g.create_dataset("f1s", data=f1s.astype(np.float64))
        g.create_dataset("f2s", data=f2s.astype(np.float64))
        maps_ds = g.create_dataset(
            "maps",
            shape=(num_windows, H, W),
            dtype=np.float32,
            chunks=(1, H, W),
        )
        # Store the full cache key as attrs (complete written last).
        for k, v in _group_params(probe).items():
            g.attrs[k] = v
        g.attrs["num_windows"] = int(num_windows)
        g.attrs["height"] = int(H)
        g.attrs["width"] = int(W)
        g.attrs["epoch_num_samples"] = int(probe.epoch_num_samples)
        g.attrs["target_fs"] = float(RawBicoherenceSequenceDataset._TARGET_FS)
        g.attrs["source_file"] = str(data_file)
        g.attrs["schema_version"] = 1
        g.attrs["complete"] = False
        f.flush()

        # Parallel compute -> single (main-process) writer.
        done = 0
        report_every = max(1, num_windows // 100)
        if n_jobs <= 1:
            _init_worker(ds_kwargs)
            for widx in range(num_windows):
                _, m = _compute_window(widx)
                maps_ds[widx] = m
                done += 1
                if done % report_every == 0 or done == num_windows:
                    _progress(name, done, num_windows, t0)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=n_jobs,
                initializer=_init_worker,
                initargs=(ds_kwargs,),
            ) as pool:
                for widx, m in pool.imap_unordered(
                    _compute_window, range(num_windows), chunksize=chunk_tasks
                ):
                    maps_ds[widx] = m
                    done += 1
                    if done % report_every == 0 or done == num_windows:
                        _progress(name, done, num_windows, t0)

        maps_ds.flush()
        g.attrs["complete"] = True
        f.flush()

    dt = time.time() - t0
    print(f"[{name}] done: {num_windows} windows in {dt/60:.1f} min.")


def _progress(name: str, done: int, total: int, t0: float) -> None:
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    pct = 100.0 * done / total
    print(
        f"[{name}] {done}/{total} ({pct:5.1f}%)  "
        f"{rate:6.1f} win/s  ETA {eta/60:5.1f} min",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default="/app/Data/WR/WR5_Run4.hdf5")
    p.add_argument("--out-file", default="/app/Data/WR/WR5_Run4_bicoh.hdf5")
    p.add_argument(
        "--epoch-sizes", type=float, nargs="+", default=[1.0, 3.0, 5.0]
    )
    p.add_argument("--segment-seconds", type=float, default=0.75)
    p.add_argument("--segment-overlap", type=float, default=0.5)
    p.add_argument("--f-max", type=float, default=25.0)
    p.add_argument("--smooth-sigma", type=float, default=0.0)
    p.add_argument("--sample-rate", type=int, default=5000)
    p.add_argument("--channel", default="Ch.1")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes (default: all CPUs).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a matching complete group already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    print(
        f"Precomputing bicoherence cache -> {args.out_file}\n"
        f"  source={args.data_file} channel={args.channel} "
        f"epoch_sizes={args.epoch_sizes} n_jobs={args.n_jobs}"
    )
    for es in args.epoch_sizes:
        precompute_epoch_size(
            out_file=args.out_file,
            data_file=args.data_file,
            epoch_size=es,
            segment_seconds=args.segment_seconds,
            segment_overlap=args.segment_overlap,
            f_max=args.f_max,
            smooth_sigma=args.smooth_sigma,
            sample_rate=args.sample_rate,
            channel=args.channel,
            n_jobs=args.n_jobs,
            force=args.force,
        )
    print("All requested epoch_sizes complete.")


if __name__ == "__main__":
    main()
