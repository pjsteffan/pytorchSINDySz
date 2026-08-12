"""Save a few bicoherence-map sequences from ``RawBicoherenceSequenceDataset``.

This is a *verification* script: every map it plots is pulled straight out of
``RawBicoherenceSequenceDataset.__getitem__`` (the exact ``[T, 1, H, W]`` tensor
the model consumes), so what you see is what the model sees. Frequency axes and
the valid-region mask also come from the dataset's own helpers.

For each chosen sequence it writes one PNG with ``seq_len`` panels (one per
window), laid out in a grid, annotated with each window's start time.

Usage
-----
    .venv/bin/python visualize_raw_bicoherence.py \
        --data-file /app/Data/WR/WR5_Run4.hdf5 \
        --out-dir bicoherence_previews \
        --num-sequences 3 --seq-len 8 --epoch-size 1.0 \
        --segment-seconds 1 --f-max 25.0 --sample-rate 5000
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from datasets import (
    PrecomputedBicoherenceSequenceDataset,
    RawBicoherenceSequenceDataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-file", default="/app/Data/WR/WR5_Run4.hdf5")
    p.add_argument(
        "--cache-file",
        default=None,
        help="If given, read precomputed maps from this HDF5 cache "
        "(PrecomputedBicoherenceSequenceDataset) instead of computing on the fly.",
    )
    p.add_argument("--out-dir", default="bicoherence_previews")
    p.add_argument("--num-sequences", type=int, default=3,
                   help="How many sequences to render.")
    p.add_argument("--start-index", type=int, default=0,
                   help="Index of the first sequence to render.")
    p.add_argument("--stride-sequences", type=int, default=None,
                   help="Gap between rendered sequence indices "
                        "(default: spread evenly across the dataset).")
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--epoch-size", type=float, default=1.0)
    p.add_argument("--segment-seconds", type=float, default=1.0)
    p.add_argument("--segment-overlap", type=float, default=0.5)
    p.add_argument("--smooth-sigma", type=float, default=0.0)
    p.add_argument("--f-max", type=float, default=25.0)
    p.add_argument("--sample-rate", type=int, default=5000)
    p.add_argument("--channel", default="Ch.1")
    p.add_argument("--dpi", type=int, default=120)
    return p.parse_args()


def render_sequence(dataset, seq_idx, f1s, f2s, mask_np, out_path, dpi):
    """Render one sequence (all windows) to a single PNG."""
    maps, mask, label = dataset[seq_idx]  # <-- actual dataset output
    maps_np = maps.numpy()  # [T, 1, H, W]
    T = maps_np.shape[0]
    starts = dataset.window_start_seconds(seq_idx)

    ncols = min(4, T)
    nrows = math.ceil(T / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False
    )

    # Extent so pixels map to real frequencies (Hz) on both axes.
    extent = [float(f2s[0]), float(f2s[-1]), float(f1s[0]), float(f1s[-1])]
    # Shared colour scale across the sequence for honest comparison.
    valid = maps_np[:, 0][:, mask_np.astype(bool)]
    vmax = float(np.nanmax(valid)) if valid.size else 1.0
    vmax = vmax if vmax > 0 else 1.0

    for t in range(nrows * ncols):
        ax = axes[t // ncols][t % ncols]
        if t >= T:
            ax.axis("off")
            continue
        bmap = maps_np[t, 0].copy()
        # Grey-out the invalid triangular region using the dataset mask.
        bmap_masked = np.ma.masked_where(mask_np < 0.5, bmap)
        im = ax.imshow(
            bmap_masked, origin="lower", extent=extent, aspect="equal",
            vmin=0.0, vmax=vmax, cmap="viridis", interpolation="nearest",
        )
        ax.set_title(f"win {t}  t={starts[t]:.1f}s", fontsize=9)
        ax.set_xlabel("f2 (Hz)", fontsize=8)
        ax.set_ylabel("f1 (Hz)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"RawBicoherenceSequenceDataset  seq_idx={seq_idx}  "
        f"label={int(label)}  T={T}  grid={maps_np.shape[2]}x{maps_np.shape[3]}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.cache_file:
        dataset = PrecomputedBicoherenceSequenceDataset(
            cache_file=args.cache_file,
            epoch_size=args.epoch_size,
            seq_len=args.seq_len,
            f_max=args.f_max,
            segment_seconds=args.segment_seconds,
            segment_overlap=args.segment_overlap,
            smooth_sigma=args.smooth_sigma,
            sample_rate=args.sample_rate,
            channel=args.channel,
        )
    else:
        dataset = RawBicoherenceSequenceDataset(
            data_file=args.data_file,
            seq_len=args.seq_len,
            epoch_size=args.epoch_size,
            f_max=args.f_max,
            segment_seconds=args.segment_seconds,
            segment_overlap=args.segment_overlap,
            smooth_sigma=args.smooth_sigma,
            sample_rate=args.sample_rate,
            channel=args.channel,
        )

    n_total = len(dataset)
    H, W = dataset.get_grid_size()
    f1s, f2s = dataset.get_freq_axes()
    mask_np = dataset.get_mask().numpy()[0]  # [H, W]
    print(
        f"Dataset: {n_total} sequences | grid {H}x{W} | "
        f"num_windows={dataset.num_windows} | f in [{f1s[0]:.2f},{f1s[-1]:.2f}] Hz"
    )

    n = min(args.num_sequences, n_total)
    if args.stride_sequences is not None:
        indices = [
            args.start_index + i * args.stride_sequences for i in range(n)
        ]
    else:
        # Spread evenly across the whole recording so you sample different times.
        if n == 1:
            indices = [args.start_index]
        else:
            span = n_total - 1 - args.start_index
            step = max(1, span // (n - 1)) if span > 0 else 1
            indices = [args.start_index + i * step for i in range(n)]
    indices = [i for i in indices if 0 <= i < n_total]

    for seq_idx in indices:
        out_path = os.path.join(args.out_dir, f"sequence_{seq_idx:06d}.png")
        render_sequence(dataset, seq_idx, f1s, f2s, mask_np, out_path, args.dpi)
        print(f"  saved {out_path}")

    print(f"Done. Wrote {len(indices)} sequence figure(s) to {args.out_dir}/")


if __name__ == "__main__":
    main()
