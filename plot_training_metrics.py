"""Visualize training and validation loss components from Lightning CSV metrics.

This helper script reads a PyTorch Lightning metrics CSV produced by training
runs in this project and plots the total loss along with each loss component
(reconstruction, SINDy regularization, SINDy xdot, and SINDy zdot) as a
function of epoch. Training metrics, which are logged multiple times per
epoch, are averaged within each epoch; validation metrics, which are already
logged once per epoch, are used directly.

Example usages
--------------
Plot from a specific metrics CSV, writing to the default path
(``./training_metrics.png``)::

    uv run python plot_training_metrics.py \\
        --csv lightning_logs/shallow_fan_gru/lightning_logs/version_0/metrics.csv

Auto-discover the most recently modified ``metrics.csv`` under
``lightning_logs/``::

    uv run python plot_training_metrics.py --auto-discover

Write to a custom path (parent directories are created automatically), bump
the DPI, and also open the figure in an interactive window::

    uv run python plot_training_metrics.py \\
        --csv lightning_logs/shallow_fan_gru/lightning_logs/version_0/metrics.csv \\
        --output plots/run0.png \\
        --dpi 300 \\
        --show

Use it as a library from another script or notebook::

    from pathlib import Path
    from plot_training_metrics import plot_training_metrics

    plot_training_metrics(
        csv_path=Path("lightning_logs/.../metrics.csv"),
        output_path=Path("plots/run0.png"),
        show=False,
        dpi=200,
    )
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_COLUMNS = [
    "train_recon_loss",
    "train_sindyreg_loss",
    "train_sindyxdot_loss",
    "train_sindyzdot_loss",
    "train_total_loss",
]

VALID_COLUMNS = [
    "valid_recon_loss",
    "valid_sindyreg_loss",
    "valid_sindyxdot_loss",
    "valid_sindyzdot_loss",
    "valid_loss",
]

# Pairs of (train_col, valid_col, title, ylabel) for each subplot in order.
LOSS_PANELS = [
    (
        "train_total_loss",
        "valid_loss",
        "Total Loss vs Epoch",
        "Total Loss",
    ),
    (
        "train_recon_loss",
        "valid_recon_loss",
        "Reconstruction Loss vs Epoch",
        "Reconstruction Loss (λ1)",
    ),
    (
        "train_sindyreg_loss",
        "valid_sindyreg_loss",
        "SINDy Regularization vs Epoch",
        "SINDy Regularization (λ4)",
    ),
    (
        "train_sindyxdot_loss",
        "valid_sindyxdot_loss",
        "SINDy xdot Loss vs Epoch",
        "SINDy xdot Loss (λ2)",
    ),
    (
        "train_sindyzdot_loss",
        "valid_sindyzdot_loss",
        "SINDy zdot Loss vs Epoch",
        "SINDy zdot Loss (λ3)",
    ),
]


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load the Lightning metrics CSV into a DataFrame.

    Validates that the file exists and contains an `epoch` column. Missing
    loss columns are tolerated (they will produce empty plots for that
    component) but a warning is printed.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "epoch" not in df.columns:
        raise ValueError(
            f"Expected an 'epoch' column in {csv_path}, found columns: "
            f"{list(df.columns)}"
        )

    expected = set(TRAIN_COLUMNS + VALID_COLUMNS)
    missing = expected - set(df.columns)
    if missing:
        print(
            f"Warning: metrics CSV is missing expected columns: "
            f"{sorted(missing)}",
            file=sys.stderr,
        )

    return df


def aggregate_training_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Average training loss columns across all steps within each epoch.

    Rows that contain only validation metrics (i.e. all training loss columns
    are NaN) are excluded before averaging so they do not affect the mean.
    """
    train_cols = [c for c in TRAIN_COLUMNS if c in df.columns]
    if not train_cols:
        return pd.DataFrame(columns=["epoch"]).set_index("epoch")

    # Keep only rows that have at least one training value.
    train_rows = df.dropna(subset=train_cols, how="all")
    if train_rows.empty:
        return pd.DataFrame(columns=["epoch"]).set_index("epoch")

    grouped = (
        train_rows.groupby("epoch")[train_cols]
        .mean()
        .sort_index()
    )
    return grouped


def extract_validation_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one validation row per epoch.

    Validation entries are identified as rows that contain at least one
    non-NaN value in the validation columns. If multiple validation rows
    exist per epoch (unusual), they are averaged together.
    """
    valid_cols = [c for c in VALID_COLUMNS if c in df.columns]
    if not valid_cols:
        return pd.DataFrame(columns=["epoch"]).set_index("epoch")

    valid_rows = df.dropna(subset=valid_cols, how="all")
    if valid_rows.empty:
        return pd.DataFrame(columns=["epoch"]).set_index("epoch")

    grouped = (
        valid_rows.groupby("epoch")[valid_cols]
        .mean()
        .sort_index()
    )
    return grouped


def plot_loss_component(
    ax: plt.Axes,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    train_col: str,
    valid_col: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot a single loss component on the supplied Axes."""
    plotted_any = False

    if train_col in train_df.columns and not train_df.empty:
        series = train_df[train_col].dropna()
        if not series.empty:
            ax.plot(
                series.index.to_numpy(),
                series.to_numpy(),
                color="tab:blue",
                linestyle="-",
                marker="o",
                markersize=4,
                label="train",
            )
            plotted_any = True

    if valid_col in valid_df.columns and not valid_df.empty:
        series = valid_df[valid_col].dropna()
        if not series.empty:
            ax.plot(
                series.index.to_numpy(),
                series.to_numpy(),
                color="tab:orange",
                linestyle="--",
                marker="s",
                markersize=5,
                label="valid",
            )
            plotted_any = True

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted_any:
        ax.legend(loc="best")
    else:
        ax.text(
            0.5,
            0.5,
            "no data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )


def print_summary(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    """Print a brief summary of min / max / final values for each loss."""
    def _fmt(value: float) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value:.6g}"

    print("\nLoss summary (per-epoch values):")
    header = f"  {'component':<28} {'min':>14} {'max':>14} {'final':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for train_col, valid_col, _, _ in LOSS_PANELS:
        for label, frame, col in (
            (f"train/{train_col}", train_df, train_col),
            (f"valid/{valid_col}", valid_df, valid_col),
        ):
            if col in frame.columns and not frame.empty:
                series = frame[col].dropna()
            else:
                series = pd.Series(dtype=float)

            if series.empty:
                row = f"  {label:<28} {'n/a':>14} {'n/a':>14} {'n/a':>14}"
            else:
                row = (
                    f"  {label:<28} "
                    f"{_fmt(series.min()):>14} "
                    f"{_fmt(series.max()):>14} "
                    f"{_fmt(series.iloc[-1]):>14}"
                )
            print(row)


def find_latest_metrics(
    root: Path = Path("lightning_logs"),
) -> Path:
    """Find the most recently modified `metrics.csv` under `root`."""
    if not root.exists():
        raise FileNotFoundError(
            f"Cannot auto-discover metrics: directory {root} does not exist."
        )
    candidates = list(root.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.csv files were found under {root}."
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def plot_training_metrics(
    csv_path: Path,
    output_path: Path = Path("training_metrics.png"),
    show: bool = False,
    dpi: int = 150,
) -> None:
    """Build and save the five-panel loss figure."""
    df = load_metrics(csv_path)
    train_df = aggregate_training_by_epoch(df)
    valid_df = extract_validation_by_epoch(df)

    fig, axes = plt.subplots(
        nrows=len(LOSS_PANELS),
        ncols=1,
        figsize=(10, 14),
        sharex=True,
    )
    if len(LOSS_PANELS) == 1:
        axes = [axes]

    for ax, (train_col, valid_col, title, ylabel) in zip(axes, LOSS_PANELS):
        plot_loss_component(
            ax=ax,
            train_df=train_df,
            valid_df=valid_df,
            train_col=train_col,
            valid_col=valid_col,
            title=title,
            ylabel=ylabel,
        )

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"Training Metrics: {csv_path}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    print(f"Saved figure to {output_path}")

    print_summary(train_df, valid_df)

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training/validation loss components from a Lightning "
            "metrics.csv file."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help=(
            "Path to the metrics CSV file. Mutually exclusive with "
            "--auto-discover."
        ),
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help=(
            "Automatically use the most recently modified metrics.csv under "
            "the `lightning_logs/` directory."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_metrics.png"),
        help="Output path for the saved figure (default: training_metrics.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for the saved figure (default: 150).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.csv is None and not args.auto_discover:
        print(
            "Error: must supply either --csv PATH or --auto-discover.",
            file=sys.stderr,
        )
        return 2

    if args.csv is not None and args.auto_discover:
        print(
            "Error: --csv and --auto-discover are mutually exclusive.",
            file=sys.stderr,
        )
        return 2

    try:
        csv_path = (
            find_latest_metrics() if args.auto_discover else args.csv
        )
        plot_training_metrics(
            csv_path=csv_path,
            output_path=args.output,
            show=args.show,
            dpi=args.dpi,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
