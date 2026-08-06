"""Stage 3: Build and inspect a bicoherence dataset from a preprocessed HDF5.

This script is the third stage in the three-stage offline pipeline:

    Stage 1  beacon stats        raw HDF5 -> global stats + config attrs
    Stage 2  beacon preprocess   raw HDF5 -> preprocessed 100 Hz HDF5
    Stage 3  beacon bicoherence  preprocessed HDF5 -> bicoherence HDF5  <- this stage

``--data_file`` must be a **preprocessed** HDF5 file produced by Stage 2.
The script validates this by checking for the ``preprocessed=1`` root
attribute.  If you supply a raw file, a clear error is printed.

No preprocessing parameters are needed or accepted — all filter settings and
statistics are baked into the preprocessed file's attrs and read automatically.

Two modes are supported via ``--mode``:

seizure_windows (default)
    Windows are drawn from seizure epochs defined in ``--epoch_csv``
    (produced by ``generate_epochs.py``).  Each window is labelled
    ``"pre-ictal"``, ``"ictal"``, or ``"post-ictal"``.

full_recording
    Non-overlapping windows are tiled across the **entire** recording.
    ``--epoch_csv`` is optional; when supplied, windows whose centre falls
    inside a known seizure epoch are annotated with the appropriate phase
    label; all other windows are labelled ``"interictal"``.

Both modes print summary statistics, optionally write a metadata JSON, load a
verification sample / DataLoader batch, and pre-compute all bicoherence maps
to an HDF5 file via ``--save_hdf5``.  The HDF5 file can be loaded instantly
with :class:`~bicoherence_dataset.PrecomputedBicoherenceDataset`.

Example
-------
    # Full three-stage workflow:

    # Stage 1 - compute global stats (run once per raw file)
    uv run beacon stats \\
        --data_file Data/WR12_NewAmp1_0_subset_28800_36000.hdf5 \\
        --all_channels

    # Stage 2 - produce preprocessed 100 Hz file
    uv run beacon preprocess \\
        --data_file Data/WR12_NewAmp1_0_subset_28800_36000.hdf5 \\
        --output Data/WR12_NewAmp1_0_subset_28800_36000_preprocessed.hdf5 \\
        --all_channels

    # Stage 3 - build bicoherence dataset (full recording mode)
    uv run beacon bicoherence \\
        --mode full_recording \\
        --data_file Data/WR12_NewAmp1_0_subset_28800_36000_preprocessed.hdf5 \\
        --save_hdf5 results/bicoh_full.h5

    # Stage 3 - seizure-windows mode with epoch CSV
    uv run beacon bicoherence \\
        --mode seizure_windows \\
        --epoch_csv results/epochs/epoch_boundaries_20260329_120000.csv \\
        --data_file Data/WR12_NewAmp1_0_subset_28800_36000_preprocessed.hdf5 \\
        --save_hdf5 results/bicoh_seizure.h5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from beacon.bispectral.dataset import (
    BicoherenceWindowDataset,
    FullRecordingBicoherenceDataset,
    PrecomputedBicoherenceDataset,
)

BICOHERENCE_DESC = __doc__


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_preprocessed_file(data_file: str, channel: str) -> dict:
    """Validate that ``data_file`` is a Stage 2 preprocessed file.

    Returns the channel's attr dict (for config reporting).

    Raises
    ------
    SystemExit
        If the file is missing or does not have ``preprocessed=1``.
    """
    if not os.path.exists(data_file):
        print(f"ERROR: File not found: {data_file!r}", file=sys.stderr)
        raise SystemExit(1)

    try:
        with h5py.File(data_file, "r") as f:
            # Check root-level preprocessed flag first.
            root_flag = int(f.attrs.get("preprocessed", 0))
            if not root_flag:
                # Also accept per-channel flag (for partial writes).
                if channel in f:
                    ch_flag = int(f[channel].attrs.get("preprocessed", 0))
                    if not ch_flag:
                        print(
                            f"ERROR: {data_file!r} does not appear to be a "
                            "Stage 2 preprocessed file.\n"
                            "       The 'preprocessed' root attribute is absent or 0.\n"
                            "       Run Stage 2 first:\n"
                            f"           uv run beacon preprocess "
                            f"--data_file <raw_file> --output {data_file} "
                            "--all_channels",
                            file=sys.stderr,
                        )
                        raise SystemExit(1)

            # Return channel attrs for reporting.
            if channel in f:
                return dict(f[channel].attrs)
            return dict(f.attrs)

    except OSError as exc:
        print(f"ERROR opening {data_file!r}: {exc}", file=sys.stderr)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "bicoherence",
        help="Stage 3: build a bicoherence dataset from a preprocessed HDF5.",
        description=BICOHERENCE_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["seizure_windows", "full_recording"],
        default="seizure_windows",
        help=(
            "Dataset mode. 'seizure_windows' (default): windows drawn from "
            "seizure epochs in --epoch_csv. 'full_recording': non-overlapping "
            "windows tiled across the entire recording; --epoch_csv is optional "
            "and used only for phase annotation."
        ),
    )
    p.add_argument(
        "--epoch_csv", default=None,
        help=(
            "Path to epoch_boundaries_<timestamp>.csv from generate_epochs.py. "
            "Required for --mode seizure_windows. Optional for --mode "
            "full_recording (enables seizure phase annotation)."
        ),
    )
    p.add_argument(
        "--data_file", required=True,
        help=(
            "Path to the Stage 2 PREPROCESSED HDF5 file "
            "(beacon preprocess output). Must have "
            "preprocessed=1 attr. Do NOT supply a raw file here — run "
            "beacon preprocess first."
        ),
    )
    p.add_argument(
        "--channel", default="Ch.1",
        help="EEG channel dataset name (default: Ch.1).",
    )
    p.add_argument(
        "--output_info", default=None,
        help="Optional path to write dataset metadata JSON.",
    )
    p.add_argument(
        "--window_seconds", type=float, default=3.0,
        help="Window size in seconds (default: 3.0).",
    )
    p.add_argument(
        "--window_step", type=float, default=None,
        help=(
            "Step between consecutive window starts in seconds. "
            "Defaults to --window_seconds (non-overlapping). "
            "Set smaller for overlapping windows."
        ),
    )
    p.add_argument(
        "--segment_seconds", type=float, default=0.75,
        help="Sub-segment size for bicoherence averaging (default: 0.75).",
    )
    p.add_argument(
        "--segment_overlap", type=float, default=0.5,
        help="Sub-segment overlap fraction in [0, 1) (default: 0.5).",
    )
    p.add_argument(
        "--f_max", type=float, default=25.0,
        help="Maximum bicoherence frequency in Hz (default: 25.0).",
    )
    p.add_argument(
        "--smooth_sigma", type=float, default=0.0,
        help="Bicoherence Gaussian smoothing sigma in bins (default: 0.0).",
    )
    p.add_argument(
        "--blank_seconds", type=float, default=0.0,
        help=(
            "Exclude windows whose centre falls within this many seconds of "
            "the seizure onset or offset (blanking zone). Default: 0.0."
        ),
    )
    p.add_argument(
        "--min_phase_windows", type=int, default=2,
        help=(
            "Minimum number of pre-ictal AND post-ictal windows an epoch must "
            "retain after blanking. Default: 2."
        ),
    )
    p.add_argument(
        "--include_invalid", action="store_true",
        help=(
            "Include epochs flagged invalid in the CSV (default: valid only)."
        ),
    )
    p.add_argument(
        "--cache_epochs", action="store_true",
        help=(
            "Cache epoch signals in memory for speed "
            "(seizure_windows mode only)."
        ),
    )
    p.add_argument(
        "--batch_size", type=int, default=16,
        help="DataLoader batch size for the verification batch (default: 16).",
    )
    p.add_argument(
        "--no_sample", action="store_true",
        help="Skip loading a verification sample / batch.",
    )
    p.add_argument(
        "--save_hdf5", default=None, metavar="PATH",
        help=(
            "Pre-compute all bicoherence maps and write them to this HDF5 "
            "file. The file can be reloaded instantly with "
            "PrecomputedBicoherenceDataset (no EEG I/O or recomputation)."
        ),
    )
    p.add_argument(
        "--include_biphase", action="store_true",
        help=(
            "Also compute and store the mean biphase map alongside the "
            "bicoherence map. The HDF5 file will contain a /biphase dataset "
            "of the same shape [n_windows, n_f1, n_f2]."
        ),
    )
    p.set_defaults(func=main)


def register(subparsers) -> None:
    """Register the ``bicoherence`` subcommand."""
    _add_parser(subparsers)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _phase_counts(dataset: Dataset) -> Counter[str]:
    return Counter(w["phase"] for w in dataset.windows)  # type: ignore[attr-defined]


def _epoch_counts(dataset: Dataset) -> Counter[str]:
    return Counter(w["epoch_id"] for w in dataset.windows)  # type: ignore[attr-defined]


def _format_config_from_attrs(ch_attrs: dict) -> str:
    """Format a one-line summary of the filter/rejection config from attrs."""
    parts = []
    if "filt_target_fs" in ch_attrs:
        parts.append(f"fs={int(ch_attrs['filt_target_fs'])} Hz")
    if "filt_lowcut" in ch_attrs and "filt_highcut" in ch_attrs:
        parts.append(
            f"bandpass=[{float(ch_attrs['filt_lowcut'])}, "
            f"{float(ch_attrs['filt_highcut'])}] Hz"
        )
    if "filt_order" in ch_attrs:
        parts.append(f"order={int(ch_attrs['filt_order'])}")
    if "reject_amp_min" in ch_attrs and "reject_amp_max" in ch_attrs:
        parts.append(
            f"amp=[{float(ch_attrs['reject_amp_min'])}, "
            f"{float(ch_attrs['reject_amp_max'])}]"
        )
    if "reject_deriv_z" in ch_attrs:
        parts.append(f"deriv_z={float(ch_attrs['reject_deriv_z'])}")
    zscore = ch_attrs.get("zscore_applied", None)
    if zscore is not None:
        parts.append(f"zscore={'yes' if int(zscore) else 'no'}")
    return ", ".join(parts) if parts else "(no config attrs found)"


def print_summary(
    args: argparse.Namespace,
    dataset: Dataset,
    ch_attrs: dict,
    bicoh_shape: tuple[int, int] | None,
    freq_res: float,
    biphase_sample: torch.Tensor | None = None,
) -> None:
    phases = _phase_counts(dataset)
    epochs = _epoch_counts(dataset)
    total = len(dataset)
    is_full = args.mode == "full_recording"

    def pct(n: int) -> str:
        return f"{(100.0 * n / total):.1f}%" if total else "0.0%"

    title = (
        "Full-Recording Bicoherence Dataset Summary"
        if is_full else
        "Bicoherence Window Dataset Summary"
    )
    print(title)
    print("=" * len(title))
    if args.epoch_csv:
        print(f"Epoch CSV: {args.epoch_csv}")
    print(f"Preprocessed data file: {args.data_file}")
    print(f"Preprocessing config (from file): {_format_config_from_attrs(ch_attrs)}")

    effective_step = args.window_step if args.window_step is not None else args.window_seconds
    overlap_label = (
        f"step {effective_step}s"
        if effective_step < args.window_seconds
        else "non-overlapping"
    )
    print(f"Window size: {args.window_seconds}s ({overlap_label})")
    print(
        f"Segment size: {args.segment_seconds}s "
        f"({int(args.segment_overlap * 100)}% overlap)"
    )
    print()
    if is_full:
        rec_start = getattr(dataset, "_rec_start", 0.0)
        rec_stop = getattr(dataset, "_rec_stop", None)
        if rec_stop is not None:
            print(f"Recording span: {rec_start:.2f}s — {rec_stop:.2f}s "
                  f"({rec_stop - rec_start:.1f}s total)")
        print(f"Total windows: {total}")
        print()
        print("Windows by phase:")
        print(f"  Pre-ictal:   {phases.get('pre-ictal', 0):>6} ({pct(phases.get('pre-ictal', 0))})")
        print(f"  Ictal:       {phases.get('ictal', 0):>6} ({pct(phases.get('ictal', 0))})")
        print(f"  Post-ictal:  {phases.get('post-ictal', 0):>6} ({pct(phases.get('post-ictal', 0))})")
        print(f"  Interictal:  {phases.get('interictal', 0):>6} ({pct(phases.get('interictal', 0))})")
    else:
        print(f"Total epochs: {len(epochs)}")
        print(f"Total windows: {total}")
        print()
        print("Windows by phase:")
        print(f"  Pre-ictal:  {phases.get('pre-ictal', 0)} ({pct(phases.get('pre-ictal', 0))})")
        print(f"  Ictal:      {phases.get('ictal', 0)} ({pct(phases.get('ictal', 0))})")
        print(f"  Post-ictal: {phases.get('post-ictal', 0)} ({pct(phases.get('post-ictal', 0))})")
    print()
    if bicoh_shape is not None:
        print(f"Bicoherence shape: [{bicoh_shape[0]}, {bicoh_shape[1]}]")
    print(f"Frequency range: 0.0 - {args.f_max} Hz")
    print(f"Frequency resolution: ~{freq_res:.2f} Hz/bin")
    if args.include_biphase:
        print("Biphase: enabled (mean biphase map will be stored in HDF5 /biphase)")
        if biphase_sample is not None:
            import numpy as _np
            bp = biphase_sample.numpy() if hasattr(biphase_sample, "numpy") else _np.asarray(biphase_sample)
            finite = bp[_np.isfinite(bp)]
            if finite.size > 0:
                print(f"  Biphase sample — mean: {finite.mean():.4f} rad, "
                      f"std: {finite.std():.4f} rad, "
                      f"valid pixels: {finite.size}/{bp.size}")


def build_metadata(
    args: argparse.Namespace,
    dataset: Dataset,
    ch_attrs: dict,
    bicoh_shape: tuple[int, int] | None,
    freq_res: float,
) -> dict[str, Any]:
    phases = _phase_counts(dataset)
    epochs = _epoch_counts(dataset)
    is_full = args.mode == "full_recording"
    shape_list = [int(bicoh_shape[0]), int(bicoh_shape[1])] if bicoh_shape is not None else None

    cfg: dict = {
        "mode": args.mode,
        "preprocessed_data_file": args.data_file,
        "epoch_csv": args.epoch_csv,
        "window_seconds": args.window_seconds,
        "window_step": args.window_step if args.window_step is not None else args.window_seconds,
        "segment_seconds": args.segment_seconds,
        "segment_overlap": args.segment_overlap,
        "f_max": args.f_max,
        "channel": args.channel,
        "smooth_sigma": args.smooth_sigma,
        "blank_seconds": args.blank_seconds,
        "min_phase_windows": args.min_phase_windows,
        "valid_only": not args.include_invalid,
        "include_biphase": getattr(args, "include_biphase", False),
        # Preprocessing config sourced from file attrs.
        "filt_target_fs": int(ch_attrs.get("filt_target_fs", 0)),
        "filt_lowcut": float(ch_attrs.get("filt_lowcut", 0)),
        "filt_highcut": float(ch_attrs.get("filt_highcut", 0)),
        "filt_order": int(ch_attrs.get("filt_order", 0)),
        "reject_amp_min": float(ch_attrs.get("reject_amp_min", 0)),
        "reject_amp_max": float(ch_attrs.get("reject_amp_max", 0)),
        "reject_deriv_z": float(ch_attrs.get("reject_deriv_z", 0)),
        "zscore_applied": bool(int(ch_attrs.get("zscore_applied", 1))),
    }
    if is_full:
        cfg["rec_start"] = getattr(dataset, "_rec_start", None)
        cfg["rec_stop"] = getattr(dataset, "_rec_stop", None)

    stats: dict = {
        "n_windows_total": len(dataset),
        "n_pre_ictal": phases.get("pre-ictal", 0),
        "n_ictal": phases.get("ictal", 0),
        "n_post_ictal": phases.get("post-ictal", 0),
        "bicoherence_shape": shape_list,
        "frequency_resolution_hz": round(freq_res, 4),
    }
    if is_full:
        stats["n_interictal"] = phases.get("interictal", 0)
    else:
        stats["n_epochs"] = len(epochs)

    return {
        "dataset_config": cfg,
        "dataset_stats": stats,
        "windows": [
            {
                "global_idx": w["global_window_idx"],
                "epoch_id": w["epoch_id"],
                "window_idx": w["window_idx"],
                "window_start": round(w["window_start"], 4),
                "window_stop": round(w["window_stop"], 4),
                "phase": w["phase"],
            }
            for w in dataset.windows
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> int:
    # --- mode-specific validation -----------------------------------------
    if args.mode == "seizure_windows" and args.epoch_csv is None:
        print(
            "ERROR: --epoch_csv is required for --mode seizure_windows.\n"
            "       Supply a path to the epoch_boundaries CSV from generate_epochs.py."
        )
        return 1

    # --- validate that data_file is a preprocessed (Stage 2) file ---------
    print(f"Validating preprocessed file: {args.data_file}")
    ch_attrs = _validate_preprocessed_file(args.data_file, args.channel)
    print(f"  OK — preprocessing config: {_format_config_from_attrs(ch_attrs)}")
    print()

    # --- dataset construction ---------------------------------------------
    if args.mode == "full_recording":
        print(f"Mode: full_recording  |  data_file: {args.data_file}")
        if args.epoch_csv:
            print(f"Phase annotation enabled via: {args.epoch_csv}")
        else:
            print("No epoch CSV supplied — all windows will be labelled 'interictal'.")
        print()
        dataset = FullRecordingBicoherenceDataset(
            data_file=args.data_file,
            epoch_boundaries_csv=args.epoch_csv,
            window_seconds=args.window_seconds,
            window_step=args.window_step,
            segment_seconds=args.segment_seconds,
            segment_overlap=args.segment_overlap,
            f_max=args.f_max,
            channel=args.channel,
            smooth_sigma=args.smooth_sigma,
            include_invalid_epochs=args.include_invalid,
            blank_seconds=args.blank_seconds,
            min_phase_windows=args.min_phase_windows,
            return_biphase=args.include_biphase,
        )
    else:  # seizure_windows
        dataset = BicoherenceWindowDataset(
            epoch_boundaries_csv=args.epoch_csv,
            data_file=args.data_file,
            window_seconds=args.window_seconds,
            window_step=args.window_step,
            segment_seconds=args.segment_seconds,
            segment_overlap=args.segment_overlap,
            f_max=args.f_max,
            channel=args.channel,
            valid_only=not args.include_invalid,
            smooth_sigma=args.smooth_sigma,
            cache_epochs=args.cache_epochs,
            blank_seconds=args.blank_seconds,
            min_phase_windows=args.min_phase_windows,
            return_biphase=args.include_biphase,
        )

    if len(dataset) == 0:
        print("No windows generated. Check the data file / epoch CSV and window size.")
        return 1

    # Frequency axes / resolution (computed from a probe signal, no I/O).
    f1_vec, f2_vec = dataset.get_frequency_axes()
    freq_res = float(f1_vec[1] - f1_vec[0]) if len(f1_vec) > 1 else 0.0
    bicoh_shape = (len(f1_vec), len(f2_vec))

    bicoh_shape_sample = None
    biphase_sample = None
    if not args.no_sample:
        sample = dataset[0]
        if args.include_biphase:
            bicoh_shape_sample = tuple(sample[0].shape)
            biphase_sample = sample[1]
        else:
            bicoh_shape_sample = tuple(sample.shape)

    print_summary(
        args, dataset, ch_attrs, bicoh_shape_sample or bicoh_shape, freq_res,
        biphase_sample=biphase_sample,
    )

    if not args.no_sample:
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
        batch = next(iter(loader))
        print()
        if args.include_biphase:
            print(f"Verification batch bicoh shape:   {tuple(batch[0].shape)}")
            print(f"Verification batch biphase shape: {tuple(batch[1].shape)}")
        else:
            print(f"Verification batch shape: {tuple(batch.shape)}")

    if args.output_info:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_info)), exist_ok=True)
        metadata = build_metadata(
            args, dataset, ch_attrs, bicoh_shape_sample or bicoh_shape, freq_res
        )
        with open(args.output_info, "w") as fh:
            json.dump(metadata, fh, indent=2)
        print()
        print(f"Wrote dataset metadata to {args.output_info}")

    if args.save_hdf5:
        total = len(dataset)
        print()
        print(f"Pre-computing {total} bicoherence maps -> {args.save_hdf5}")
        if args.include_biphase:
            print("  (biphase maps will also be stored)")

        def _progress(done: int, total: int) -> None:
            pct = 100.0 * done / total
            bar = "#" * (done * 40 // total)
            print(f"\r  [{bar:<40}] {done}/{total} ({pct:.0f}%)", end="", flush=True)

        dataset.save_to_hdf5(args.save_hdf5, progress_cb=_progress)
        print()  # newline after progress bar

        # Quick round-trip check.
        precomputed = PrecomputedBicoherenceDataset(args.save_hdf5)
        sample_pre = precomputed[0]
        print(
            f"Saved. Round-trip check: shape={tuple(sample_pre.shape)}, "
            f"n_windows={len(precomputed)}"
        )
        if args.include_biphase:
            if precomputed.has_biphase:
                precomputed_bp = PrecomputedBicoherenceDataset(
                    args.save_hdf5, return_biphase=True
                )
                bicoh_rt, biphase_rt = precomputed_bp[0]
                print(
                    f"Biphase round-trip check: shape={tuple(biphase_rt.shape)}, "
                    f"has_biphase={precomputed.has_biphase}"
                )
            else:
                print(
                    "WARNING: --include_biphase was set but HDF5 has_biphase=False "
                    "(biphase dataset may be missing)."
                )
        print(f"Reload with: PrecomputedBicoherenceDataset({args.save_hdf5!r})")

    return 0
