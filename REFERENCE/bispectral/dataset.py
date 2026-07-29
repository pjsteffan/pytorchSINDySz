"""PyTorch dataset of windowed bicoherence maps.

Each dataset entry is one **non-overlapping** window (default 3 s) within an
epoch or recording tile.  The bicoherence for a window is estimated by
averaging the bispectrum across shorter **overlapping** sub-segments (default
0.75 s, 50 % overlap) inside the window, reusing
:func:`bispectral_analysis.compute_bicoherence`.

Three-stage offline pipeline
-----------------------------
Stage 1  ``beacon stats``          raw HDF5 → global stats + config attrs
Stage 2  ``beacon preprocess``   raw HDF5 → preprocessed 100 Hz HDF5
Stage 3  ``beacon bicoherence``    preprocessed HDF5 → bicoherence HDF5  ← uses these classes

:class:`BicoherenceWindowDataset` and :class:`FullRecordingBicoherenceDataset`
now consume the **preprocessed** HDF5 produced by Stage 2.  They perform
**no preprocessing** — they read the already-clean 100 Hz signal, window it,
and compute bicoherence.  The sample rate is read from the preprocessed
file's ``filt_target_fs`` attr.

Two dataset classes are provided:

* :class:`BicoherenceWindowDataset` — windows inside seizure epochs (existing
  behaviour, requires an epoch-boundaries CSV).
* :class:`FullRecordingBicoherenceDataset` — windows tiled across the **entire**
  recording without requiring an epoch CSV.  Optionally annotates each window
  with a seizure phase (``"pre-ictal"`` / ``"ictal"`` / ``"post-ictal"`` /
  ``"interictal"``) when an epoch CSV is supplied.

Pre-computed HDF5 workflow
--------------------------
1. Compute and save::

       dataset = BicoherenceWindowDataset(epoch_csv, preprocessed_file)
       dataset.save_to_hdf5("bicoh_dataset.h5")

       full_dataset = FullRecordingBicoherenceDataset(preprocessed_file)
       full_dataset.save_to_hdf5("bicoh_full.h5")

2. Load back instantly (no signal I/O or bicoherence computation)::

       fast_dataset = PrecomputedBicoherenceDataset("bicoh_dataset.h5")

HDF5 layout
-----------
``/bicoherence``          float32 ``[n_windows, n_f1, n_f2]`` – the maps
``/f1s``, ``/f2s``        float64 frequency axes
``/metadata/epoch_id``    bytes   seizure ID strings (or ``"full_recording"``)
``/metadata/window_idx``  int32   pre-blanking index within the epoch;
                                  may contain gaps when ``blank_seconds > 0``
                                  because transition windows are dropped
                                  without re-numbering.  Use
                                  ``global_window_idx`` (reconstructed as the
                                  HDF5 row index on load) for contiguous
                                  indexing.
``/metadata/window_start``  float64
``/metadata/window_stop``   float64
``/metadata/phase``       bytes   ``"pre-ictal"`` / ``"ictal"`` / ``"post-ictal"``
                                  / ``"interictal"``
Root attributes           all dataset config parameters
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from beacon.bispectral.analysis import compute_bicoherence


# ---------------------------------------------------------------------------
# Provenance attr keys forwarded from preprocessed file into bicoherence HDF5
# ---------------------------------------------------------------------------

# These keys are read from the preprocessed channel's attr dict (written by
# Stage 2 / beacon stats) and forwarded into every bicoherence
# HDF5 file written by save_to_hdf5.  Centralised here so both dataset
# classes stay in sync and future additions only need one edit.
PROVENANCE_ATTRS: frozenset[str] = frozenset({
    "raw_mean", "raw_std", "raw_diff_mean", "raw_diff_std",
    "filt_mean", "filt_std", "filt_target_fs",
    "filt_lowcut", "filt_highcut", "filt_order",
    "reject_amp_min", "reject_amp_max", "reject_deriv_z",
    "stats_schema_version",
    "zscore_applied",
})


# ---------------------------------------------------------------------------
# Window planning helpers
# ---------------------------------------------------------------------------

def assign_phase(
    window_center: float,
    pre_ictal_stop: float,
    ictal_stop: float,
    blank_seconds: float = 0.0,
) -> str:
    """Assign a phase from a window centre time (metadata only, not a label).

    Boundaries are the epoch's ``pre_ictal_stop`` and ``ictal_stop`` from the
    epoch CSV. A window whose centre falls before ``pre_ictal_stop`` is
    ``"pre-ictal"``, before ``ictal_stop`` is ``"ictal"``, otherwise
    ``"post-ictal"``.

    When ``blank_seconds > 0``, windows whose centre falls within
    ``blank_seconds`` of either the seizure onset (``pre_ictal_stop``) or
    seizure offset (``ictal_stop``) are labelled ``"transition"``.  The caller
    is responsible for deciding how to handle ``"transition"`` windows (e.g.
    dropping them from the dataset).
    """
    if blank_seconds > 0.0:
        if pre_ictal_stop - blank_seconds <= window_center < pre_ictal_stop:
            return "transition"
        if ictal_stop <= window_center < ictal_stop + blank_seconds:
            return "transition"
    if window_center < pre_ictal_stop:
        return "pre-ictal"
    if window_center < ictal_stop:
        return "ictal"
    return "post-ictal"


def generate_windows_for_epoch(
    epoch_start: float,
    epoch_stop: float,
    window_seconds: float,
    window_step: float | None = None,
) -> list[tuple[float, float]]:
    """Generate ``(start, stop)`` window boundaries across an epoch.

    Partial windows at the end of the epoch are discarded so every window has
    an identical duration.  By default ``window_step = window_seconds``
    (non-overlapping).  Pass a smaller ``window_step`` (e.g. ``0.5``) to
    produce overlapping windows with higher temporal resolution.
    """
    if window_step is None:
        window_step = window_seconds
    windows: list[tuple[float, float]] = []
    t0 = float(epoch_start)
    while t0 + window_seconds <= epoch_stop:
        windows.append((t0, t0 + window_seconds))
        t0 += window_step
    return windows


# ---------------------------------------------------------------------------
# Preprocessed-file helpers
# ---------------------------------------------------------------------------

def _read_preprocessed_fs(hdf5_path: str, channel: str) -> int:
    """Read ``filt_target_fs`` from a preprocessed HDF5 channel dataset.

    Raises
    ------
    ValueError
        If the file is missing the ``preprocessed=1`` flag or the
        ``filt_target_fs`` attr.
    """
    with h5py.File(hdf5_path, "r") as f:
        if channel not in f:
            available = [k for k in f.keys() if k != "Info"]
            raise ValueError(
                f"Channel {channel!r} not found in {hdf5_path!r}. "
                f"Available: {available}"
            )
        ds = f[channel]
        # Validate that this is a preprocessed file.
        if not int(f.attrs.get("preprocessed", ds.attrs.get("preprocessed", 0))):
            raise ValueError(
                f"{hdf5_path!r} does not appear to be a Stage 2 preprocessed "
                "file (missing 'preprocessed=1' attr).  Run Stage 2 first:\n"
                "    uv run beacon preprocess "
                f"--data_file <raw_file> --output {hdf5_path} --all_channels"
            )
        if "filt_target_fs" not in ds.attrs:
            raise ValueError(
                f"Channel {channel!r} in {hdf5_path!r} is missing the "
                "'filt_target_fs' attribute.  Re-run Stage 1 and Stage 2."
            )
        return int(ds.attrs["filt_target_fs"])


def _load_preprocessed_window(
    hdf5_path: str,
    channel: str,
    start_time: float,
    stop_time: float,
    fs: int,
) -> np.ndarray:
    """Load a time slice from a preprocessed HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to the Stage 2 preprocessed HDF5 file.
    channel : str
        Channel dataset name.
    start_time, stop_time : float
        Slice boundaries in seconds.
    fs : int
        Sample rate of the preprocessed file (used for index math).

    Returns
    -------
    np.ndarray
        float64 signal slice.
    """
    start_idx = int(round(start_time * fs))
    stop_idx = int(round(stop_time * fs))
    with h5py.File(hdf5_path, "r") as f:
        ds = f[channel]
        n_total = ds.shape[0]
        s = max(0, start_idx)
        e = min(n_total, stop_idx)
        if e <= s:
            raise ValueError(
                f"Empty window t=[{start_time:.3f}, {stop_time:.3f}]s "
                f"(indices {s}..{e}, dataset length {n_total})"
            )
        return ds[s:e].astype(np.float64, copy=False)


def _get_preprocessed_duration(hdf5_path: str, channel: str, fs: int) -> tuple[float, float]:
    """Return (start, stop) time bounds of a preprocessed channel."""
    with h5py.File(hdf5_path, "r") as f:
        n_total = f[channel].shape[0]
    return 0.0, n_total / float(fs)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BicoherenceWindowDataset(Dataset):
    """Lazy PyTorch dataset of per-window bicoherence maps.

    Reads from a **preprocessed** 100 Hz HDF5 file produced by Stage 2
    (``beacon preprocess``).  No preprocessing is performed at
    item access time — the dataset reads already-clean signal slices and
    computes bicoherence only.

    Parameters
    ----------
    epoch_boundaries_csv : str
        Path to the epoch-boundaries CSV produced by ``generate_epochs.py``.
    data_file : str
        Path to the **preprocessed** HDF5 file (Stage 2 output).  Must have
        ``preprocessed=1`` attr and ``filt_target_fs`` on each channel
        dataset.
    window_seconds : float
        Size of each analysis window (default ``3.0``).
    window_step : float, optional
        Step size between consecutive window starts in seconds.  Defaults
        to ``window_seconds`` (non-overlapping).
    segment_seconds : float
        Size of the overlapping sub-segments used to average the bicoherence
        within a window (default ``0.75``).
    segment_overlap : float
        Fractional overlap of the sub-segments in ``[0, 1)`` (default ``0.5``).
    f_max : float
        Maximum frequency for the bicoherence grid (default ``25.0``).
    channel : str
        EEG channel dataset name (default ``"Ch.1"``).
    valid_only : bool
        Keep only epochs flagged ``valid`` in the CSV (default ``True``).
    smooth_sigma : float
        Optional NaN-aware Gaussian smoothing of the bicoherence (in grid
        bins, default ``0.0`` = no smoothing).
    cache_epochs : bool
        If ``True``, cache epoch signals in memory (trading RAM for speed
        since many windows share an epoch).  Default ``False``.
    blank_seconds : float
        Windows whose centre falls within this many seconds of the seizure
        onset or offset are labelled ``"transition"`` and **dropped** from
        the dataset.  Default ``0.0`` (no blanking).
    min_phase_windows : int
        Minimum number of pre-ictal **and** post-ictal windows an epoch must
        retain (after blanking) to be included in the dataset.  Default ``2``.
    return_biphase : bool
        If ``True``, ``__getitem__`` returns a tuple
        ``(bicoh_tensor, biphase_tensor)`` instead of a single tensor.
        Default ``False``.
    """

    def __init__(
        self,
        epoch_boundaries_csv: str,
        data_file: str,
        window_seconds: float = 3.0,
        window_step: float | None = None,
        segment_seconds: float = 0.75,
        segment_overlap: float = 0.5,
        f_max: float = 25.0,
        channel: str = "Ch.1",
        valid_only: bool = True,
        smooth_sigma: float = 0.0,
        cache_epochs: bool = False,
        blank_seconds: float = 0.0,
        min_phase_windows: int = 2,
        return_biphase: bool = False,
    ) -> None:
        super().__init__()

        # Default window_step to window_seconds (non-overlapping).
        if window_step is None:
            window_step = window_seconds

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if window_step <= 0:
            raise ValueError("window_step must be positive")
        if window_step > window_seconds:
            raise ValueError("window_step cannot exceed window_seconds")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if not (0.0 <= segment_overlap < 1.0):
            raise ValueError("segment_overlap must be in [0, 1)")
        if segment_seconds > window_seconds:
            raise ValueError("segment_seconds cannot exceed window_seconds")
        if blank_seconds < 0.0:
            raise ValueError("blank_seconds must be >= 0")
        if min_phase_windows < 0:
            raise ValueError("min_phase_windows must be >= 0")

        self.epoch_boundaries_csv = epoch_boundaries_csv
        self.data_file = data_file
        self.window_seconds = float(window_seconds)
        self.window_step = float(window_step)
        self.segment_seconds = float(segment_seconds)
        self.segment_overlap = float(segment_overlap)
        self.f_max = float(f_max)
        self.channel = channel
        self.valid_only = bool(valid_only)
        self.smooth_sigma = float(smooth_sigma)
        self.cache_epochs = bool(cache_epochs)
        self.blank_seconds = float(blank_seconds)
        self.min_phase_windows = int(min_phase_windows)
        self.return_biphase = bool(return_biphase)

        # Read sample rate from preprocessed file attrs (validates it's a Stage 2 file).
        self.fs: int = _read_preprocessed_fs(data_file, channel)

        # Optional cache of epoch signals, keyed by seizure_id.
        self._epoch_cache: dict[str, np.ndarray] = {}
        # Cached (f1, f2) frequency axes, computed lazily once.
        self._freq_axes: tuple[np.ndarray, np.ndarray] | None = None

        self.windows: list[dict] = self._build_window_index()

    # -- index construction -------------------------------------------------

    def _build_window_index(self) -> list[dict]:
        df = pd.read_csv(self.epoch_boundaries_csv)

        required = {
            "seizure_id", "valid", "epoch_start", "epoch_stop",
            "pre_ictal_stop", "ictal_stop",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Epoch CSV {self.epoch_boundaries_csv!r} missing columns: "
                f"{sorted(missing)}"
            )

        if self.valid_only:
            df = df[df["valid"].astype(bool)].copy()
        # Drop rows lacking epoch boundaries (e.g. invalid seizures with NaNs).
        df = df.dropna(subset=["epoch_start", "epoch_stop"]).reset_index(drop=True)

        windows: list[dict] = []
        global_idx = 0
        for _, row in df.iterrows():
            seizure_id = str(row["seizure_id"])
            epoch_start = float(row["epoch_start"])
            epoch_stop = float(row["epoch_stop"])
            pre_ictal_stop = float(row["pre_ictal_stop"])
            ictal_stop = float(row["ictal_stop"])

            bounds = generate_windows_for_epoch(
                epoch_start, epoch_stop, self.window_seconds, self.window_step
            )

            epoch_windows: list[dict] = []
            for window_idx, (w_start, w_stop) in enumerate(bounds):
                w_center = 0.5 * (w_start + w_stop)
                phase = assign_phase(
                    w_center, pre_ictal_stop, ictal_stop, self.blank_seconds
                )
                if phase == "transition":
                    continue  # drop blanked windows entirely
                epoch_windows.append({
                    "epoch_id": seizure_id,
                    "window_idx": window_idx,
                    "global_window_idx": -1,  # placeholder; filled below
                    "epoch_start": epoch_start,
                    "epoch_stop": epoch_stop,
                    "window_start": w_start,
                    "window_stop": w_stop,
                    "phase": phase,
                })

            # Enforce minimum pre- and post-ictal window counts.
            n_pre  = sum(1 for w in epoch_windows if w["phase"] == "pre-ictal")
            n_post = sum(1 for w in epoch_windows if w["phase"] == "post-ictal")
            if n_pre < self.min_phase_windows or n_post < self.min_phase_windows:
                warnings.warn(
                    f"Epoch {seizure_id!r} dropped: has {n_pre} pre-ictal and "
                    f"{n_post} post-ictal window(s) after blanking "
                    f"(minimum required: {self.min_phase_windows}). "
                    "Consider increasing the epoch size or reducing blank_seconds.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            for w in epoch_windows:
                w["global_window_idx"] = global_idx
                global_idx += 1
            windows.extend(epoch_windows)

        return windows

    # -- epoch loading ------------------------------------------------------

    def _load_epoch_signal(
        self, epoch_id: str, epoch_start: float, epoch_stop: float,
    ) -> np.ndarray:
        """Load a full epoch signal from the preprocessed file, using cache if enabled."""
        if self.cache_epochs and epoch_id in self._epoch_cache:
            return self._epoch_cache[epoch_id]

        signal = _load_preprocessed_window(
            self.data_file, self.channel, epoch_start, epoch_stop, self.fs
        )

        if self.cache_epochs:
            self._epoch_cache[epoch_id] = signal
        return signal

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        meta = self.windows[idx]

        epoch_signal = self._load_epoch_signal(
            meta["epoch_id"], meta["epoch_start"], meta["epoch_stop"]
        )

        # Convert absolute window times to sample indices relative to the epoch.
        start_idx = int(round((meta["window_start"] - meta["epoch_start"]) * self.fs))
        stop_idx = int(round((meta["window_stop"] - meta["epoch_start"]) * self.fs))
        start_idx = max(0, start_idx)
        stop_idx = min(len(epoch_signal), stop_idx)
        window_signal = epoch_signal[start_idx:stop_idx]

        bres = compute_bicoherence(
            signal=window_signal,
            fs=self.fs,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )

        # Cache the frequency axes the first time we see a real result.
        if self._freq_axes is None:
            self._freq_axes = (
                np.asarray(bres.f1s, dtype=np.float64),
                np.asarray(bres.f2s, dtype=np.float64),
            )

        bicoh = torch.from_numpy(np.ascontiguousarray(bres.bicoherence)).float()
        bicoh = torch.nan_to_num(bicoh, nan=0.0)

        if self.return_biphase:
            biphase = torch.from_numpy(np.ascontiguousarray(bres.biphase)).float()
            biphase = torch.nan_to_num(biphase, nan=0.0)
            return bicoh, biphase

        return bicoh

    # -- introspection helpers ---------------------------------------------

    def get_window_metadata(self, idx: int) -> dict:
        """Return the metadata dict for the window at ``idx``."""
        return dict(self.windows[idx])

    def get_frequency_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(f1_vec, f2_vec)`` axes of the bicoherence grid.

        Computed once by running the bicoherence on the first window (using a
        zero signal of the correct length if no window is available), then
        cached.
        """
        if self._freq_axes is not None:
            return self._freq_axes

        seg_len = max(4, int(round(self.segment_seconds * self.fs)))
        probe = np.zeros(max(seg_len, int(round(self.window_seconds * self.fs))))
        bres = compute_bicoherence(
            signal=probe,
            fs=self.fs,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )
        self._freq_axes = (
            np.asarray(bres.f1s, dtype=np.float64),
            np.asarray(bres.f2s, dtype=np.float64),
        )
        return self._freq_axes

    def clear_cache(self, epoch_id: str | None = None) -> None:
        """Clear cached epoch signals."""
        if epoch_id is None:
            self._epoch_cache.clear()
        else:
            self._epoch_cache.pop(epoch_id, None)

    # -- HDF5 persistence ---------------------------------------------------

    def _compute_both(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return ``(bicoh_tensor, biphase_tensor)`` for window ``idx``."""
        meta = self.windows[idx]
        epoch_signal = self._load_epoch_signal(
            meta["epoch_id"], meta["epoch_start"], meta["epoch_stop"]
        )
        start_idx = max(0, int(round((meta["window_start"] - meta["epoch_start"]) * self.fs)))
        stop_idx = min(len(epoch_signal), int(round((meta["window_stop"] - meta["epoch_start"]) * self.fs)))
        window_signal = epoch_signal[start_idx:stop_idx]

        bres = compute_bicoherence(
            signal=window_signal,
            fs=self.fs,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )

        if self._freq_axes is None:
            self._freq_axes = (
                np.asarray(bres.f1s, dtype=np.float64),
                np.asarray(bres.f2s, dtype=np.float64),
            )

        bicoh = torch.from_numpy(np.ascontiguousarray(bres.bicoherence)).float()
        bicoh = torch.nan_to_num(bicoh, nan=0.0)
        biphase = torch.from_numpy(np.ascontiguousarray(bres.biphase)).float()
        biphase = torch.nan_to_num(biphase, nan=0.0)
        return bicoh, biphase

    def _read_provenance_attrs(self) -> dict:
        """Read RecordingConfig attrs from the preprocessed channel dataset."""
        with h5py.File(self.data_file, "r") as f:
            return dict(f[self.channel].attrs)

    def save_to_hdf5(
        self,
        path: str,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> None:
        """Compute all bicoherence maps and write them to an HDF5 file.

        The resulting file can be loaded instantly via
        :class:`PrecomputedBicoherenceDataset`.

        HDF5 layout
        -----------
        ``/bicoherence``         float32 ``[n_windows, n_f1, n_f2]``
        ``/biphase``             float32 ``[n_windows, n_f1, n_f2]`` (mean biphase, radians)
                                 — only written when ``self.return_biphase=True``
        ``/f1s``, ``/f2s``       float64 frequency axes
        ``/metadata/epoch_id``   variable-length UTF-8 strings
        ``/metadata/window_idx`` int32
        ``/metadata/window_start``, ``/metadata/window_stop`` float64
        ``/metadata/phase``      variable-length UTF-8 strings
        Root attributes          config + provenance from preprocessed file;
                                 ``has_biphase=1`` when biphase is stored.
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("Dataset is empty — nothing to save.")

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        first_bicoh, first_biphase = self._compute_both(0)
        n_f1, n_f2 = first_bicoh.shape
        f1_vec, f2_vec = self.get_frequency_axes()

        # Read provenance attrs from preprocessed file for storage.
        prov = self._read_provenance_attrs()

        str_dtype = h5py.special_dtype(vlen=str)

        with h5py.File(path, "w") as f:
            # --- bicoherence maps ---
            bicoh_ds = f.create_dataset(
                "bicoherence",
                shape=(n, n_f1, n_f2),
                dtype=np.float32,
                chunks=(1, n_f1, n_f2),
                compression="gzip",
                compression_opts=4,
            )
            bicoh_ds[0] = first_bicoh.numpy()

            # --- biphase maps (only when return_biphase=True) ---
            if self.return_biphase:
                biphase_ds = f.create_dataset(
                    "biphase",
                    shape=(n, n_f1, n_f2),
                    dtype=np.float32,
                    chunks=(1, n_f1, n_f2),
                    compression="gzip",
                    compression_opts=4,
                )
                biphase_ds[0] = first_biphase.numpy()

            # --- frequency axes ---
            f.create_dataset("f1s", data=f1_vec, dtype=np.float64)
            f.create_dataset("f2s", data=f2_vec, dtype=np.float64)

            # --- metadata arrays ---
            meta_grp = f.create_group("metadata")
            epoch_id_ds   = meta_grp.create_dataset("epoch_id",    shape=(n,), dtype=str_dtype)
            win_idx_ds    = meta_grp.create_dataset("window_idx",  shape=(n,), dtype=np.int32)
            win_start_ds  = meta_grp.create_dataset("window_start",shape=(n,), dtype=np.float64)
            win_stop_ds   = meta_grp.create_dataset("window_stop", shape=(n,), dtype=np.float64)
            phase_ds      = meta_grp.create_dataset("phase",       shape=(n,), dtype=str_dtype)

            # Fill index 0 (already computed above).
            m = self.windows[0]
            epoch_id_ds[0]  = m["epoch_id"]
            win_idx_ds[0]   = m["window_idx"]
            win_start_ds[0] = m["window_start"]
            win_stop_ds[0]  = m["window_stop"]
            phase_ds[0]     = m["phase"]

            if progress_cb is not None:
                progress_cb(1, n)

            # --- compute remaining windows ---
            for i in range(1, n):
                bicoh_t, biphase_t = self._compute_both(i)
                bicoh_ds[i] = bicoh_t.numpy()
                if self.return_biphase:
                    biphase_ds[i] = biphase_t.numpy()
                m = self.windows[i]
                epoch_id_ds[i]  = m["epoch_id"]
                win_idx_ds[i]   = m["window_idx"]
                win_start_ds[i] = m["window_start"]
                win_stop_ds[i]  = m["window_stop"]
                phase_ds[i]     = m["phase"]
                if progress_cb is not None:
                    progress_cb(i + 1, n)

            # --- root config attributes ---
            f.attrs["epoch_boundaries_csv"]   = self.epoch_boundaries_csv
            f.attrs["preprocessed_data_file"] = self.data_file
            f.attrs["window_seconds"]         = self.window_seconds
            f.attrs["window_step"]            = self.window_step
            f.attrs["segment_seconds"]        = self.segment_seconds
            f.attrs["segment_overlap"]        = self.segment_overlap
            f.attrs["f_max"]                  = self.f_max
            f.attrs["channel"]                = self.channel
            f.attrs["fs"]                     = self.fs
            f.attrs["smooth_sigma"]           = self.smooth_sigma
            f.attrs["blank_seconds"]          = self.blank_seconds
            f.attrs["min_phase_windows"]      = self.min_phase_windows
            f.attrs["n_windows"]              = n
            f.attrs["n_f1"]                   = n_f1
            f.attrs["n_f2"]                   = n_f2
            f.attrs["has_biphase"]            = int(self.return_biphase)
            # Provenance: copy RecordingConfig attrs from the preprocessed file.
            for k in PROVENANCE_ATTRS:
                if k in prov:
                    f.attrs[k] = prov[k]


# ---------------------------------------------------------------------------
# Full-recording dataset
# ---------------------------------------------------------------------------

class FullRecordingBicoherenceDataset(Dataset):
    """Bicoherence dataset tiled across an **entire** EEG recording.

    Reads from a **preprocessed** 100 Hz HDF5 file produced by Stage 2
    (``beacon preprocess``).  No preprocessing is performed at
    item access time.

    Slides windows of ``window_seconds`` across the full recording without
    requiring a seizure epoch CSV.  An optional epoch CSV may be supplied to
    annotate each window with a seizure phase.

    Parameters
    ----------
    data_file : str
        Path to the **preprocessed** HDF5 file (Stage 2 output).
    epoch_boundaries_csv : str, optional
        Path to epoch-boundaries CSV for phase annotation.
    window_seconds : float
        Size of each analysis window (default ``3.0``).
    window_step : float, optional
        Step size between consecutive window starts in seconds.  Defaults
        to ``window_seconds`` (non-overlapping).
    segment_seconds : float
        Sub-segment size for bicoherence averaging (default ``0.75``).
    segment_overlap : float
        Sub-segment overlap fraction in ``[0, 1)`` (default ``0.5``).
    f_max : float
        Maximum bicoherence frequency in Hz (default ``25.0``).
    channel : str
        EEG channel dataset name (default ``"Ch.1"``).
    smooth_sigma : float
        Optional Gaussian smoothing of bicoherence maps (default ``0.0``).
    include_invalid_epochs : bool
        When an epoch CSV is supplied and ``True``, invalid epochs are still
        used for phase annotation.  Default ``False``.
    blank_seconds : float
        Windows within this many seconds of a seizure boundary are labelled
        ``"transition"`` and dropped.  Default ``0.0``.
    min_phase_windows : int
        Minimum pre-ictal and post-ictal windows per epoch (after blanking);
        epochs below the threshold have their labels relabelled
        ``"interictal"``.  Default ``2``.
    return_biphase : bool
        If ``True``, ``__getitem__`` returns ``(bicoh_tensor, biphase_tensor)``.
        Default ``False``.
    """

    def __init__(
        self,
        data_file: str,
        epoch_boundaries_csv: str | None = None,
        window_seconds: float = 3.0,
        window_step: float | None = None,
        segment_seconds: float = 0.75,
        segment_overlap: float = 0.5,
        f_max: float = 25.0,
        channel: str = "Ch.1",
        smooth_sigma: float = 0.0,
        include_invalid_epochs: bool = False,
        blank_seconds: float = 0.0,
        min_phase_windows: int = 2,
        return_biphase: bool = False,
    ) -> None:
        super().__init__()

        if window_step is None:
            window_step = window_seconds

        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if window_step <= 0:
            raise ValueError("window_step must be positive")
        if window_step > window_seconds:
            raise ValueError("window_step cannot exceed window_seconds")
        if segment_seconds <= 0:
            raise ValueError("segment_seconds must be positive")
        if not (0.0 <= segment_overlap < 1.0):
            raise ValueError("segment_overlap must be in [0, 1)")
        if segment_seconds > window_seconds:
            raise ValueError("segment_seconds cannot exceed window_seconds")
        if blank_seconds < 0.0:
            raise ValueError("blank_seconds must be >= 0")
        if min_phase_windows < 0:
            raise ValueError("min_phase_windows must be >= 0")

        self.data_file = data_file
        self.epoch_boundaries_csv = epoch_boundaries_csv
        self.window_seconds = float(window_seconds)
        self.window_step = float(window_step)
        self.segment_seconds = float(segment_seconds)
        self.segment_overlap = float(segment_overlap)
        self.f_max = float(f_max)
        self.channel = channel
        self.smooth_sigma = float(smooth_sigma)
        self.include_invalid_epochs = bool(include_invalid_epochs)
        self.blank_seconds = float(blank_seconds)
        self.min_phase_windows = int(min_phase_windows)
        self.return_biphase = bool(return_biphase)

        # Cached (f1, f2) frequency axes, computed lazily once.
        self._freq_axes: tuple[np.ndarray, np.ndarray] | None = None

        # Read sample rate from preprocessed file attrs.
        self.fs: int = _read_preprocessed_fs(data_file, channel)

        # Load recording bounds.
        self._rec_start, self._rec_stop = _get_preprocessed_duration(
            data_file, channel=channel, fs=self.fs
        )

        # Build phase annotation intervals from the epoch CSV (if provided).
        self._epoch_intervals: list[tuple[float, float, float, float, str]] = []
        if epoch_boundaries_csv is not None:
            self._epoch_intervals = self._load_epoch_intervals(epoch_boundaries_csv)

        # Build flat window list.
        self.windows: list[dict] = self._build_window_index()

    # -- epoch interval loading ---------------------------------------------

    def _load_epoch_intervals(
        self, csv_path: str
    ) -> list[tuple[float, float, float, float, str]]:
        """Load valid epoch intervals for phase annotation."""
        df = pd.read_csv(csv_path)

        required = {
            "seizure_id", "valid", "epoch_start", "epoch_stop",
            "pre_ictal_stop", "ictal_stop",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Epoch CSV {csv_path!r} missing columns: {sorted(missing)}"
            )

        if not self.include_invalid_epochs:
            df = df[df["valid"].astype(bool)].copy()

        df = df.dropna(
            subset=["epoch_start", "epoch_stop", "pre_ictal_stop", "ictal_stop"]
        ).reset_index(drop=True)

        intervals = []
        for _, row in df.iterrows():
            intervals.append((
                float(row["epoch_start"]),
                float(row["pre_ictal_stop"]),
                float(row["ictal_stop"]),
                float(row["epoch_stop"]),
                str(row["seizure_id"]),
            ))
        return intervals

    # -- phase annotation ---------------------------------------------------

    def _annotate_window(
        self, window_center: float
    ) -> tuple[str, str]:
        """Return ``(phase, epoch_id)`` for a window centre time."""
        for epoch_start, pre_ictal_stop, ictal_stop, epoch_stop, seizure_id in self._epoch_intervals:
            if epoch_start <= window_center < epoch_stop:
                return assign_phase(
                    window_center, pre_ictal_stop, ictal_stop, self.blank_seconds
                ), seizure_id
        return "interictal", "full_recording"

    # -- index construction -------------------------------------------------

    def _build_window_index(self) -> list[dict]:
        """Tile the full recording with windows of ``window_seconds``."""
        windows: list[dict] = []
        t0 = self._rec_start
        global_idx = 0

        while t0 + self.window_seconds <= self._rec_stop:
            w_start = t0
            w_stop = t0 + self.window_seconds
            w_center = 0.5 * (w_start + w_stop)

            phase, epoch_id = self._annotate_window(w_center)

            t0 += self.window_step
            if phase == "transition":
                continue  # drop blanked windows entirely

            windows.append({
                "epoch_id": epoch_id,
                "window_idx": global_idx,
                "global_window_idx": global_idx,
                "window_start": w_start,
                "window_stop": w_stop,
                "phase": phase,
            })
            global_idx += 1

        # Enforce min_phase_windows per epoch — relabel rather than drop
        # so recording coverage stays complete.
        if self.min_phase_windows > 0 and self._epoch_intervals:
            for _, pre_ictal_stop, ictal_stop, _, seizure_id in self._epoch_intervals:
                epoch_wins = [w for w in windows if w["epoch_id"] == seizure_id]
                n_pre  = sum(1 for w in epoch_wins if w["phase"] == "pre-ictal")
                n_post = sum(1 for w in epoch_wins if w["phase"] == "post-ictal")
                if n_pre < self.min_phase_windows or n_post < self.min_phase_windows:
                    warnings.warn(
                        f"Epoch {seizure_id!r}: has {n_pre} pre-ictal and "
                        f"{n_post} post-ictal window(s) after blanking "
                        f"(minimum required: {self.min_phase_windows}). "
                        "Phase labels for this epoch relabelled 'interictal'. "
                        "Consider increasing the epoch size or reducing blank_seconds.",
                        UserWarning,
                        stacklevel=2,
                    )
                    for w in epoch_wins:
                        w["phase"] = "interictal"
                        w["epoch_id"] = "full_recording"

        return windows

    # -- signal loading -----------------------------------------------------

    def _load_window_signal(self, window_start: float, window_stop: float) -> np.ndarray:
        """Load the signal for a single window from the preprocessed file."""
        return _load_preprocessed_window(
            self.data_file, self.channel, window_start, window_stop, self.fs
        )

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        meta = self.windows[idx]

        window_signal = self._load_window_signal(
            meta["window_start"], meta["window_stop"]
        )

        bres = compute_bicoherence(
            signal=window_signal,
            fs=self.fs,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )

        if self._freq_axes is None:
            self._freq_axes = (
                np.asarray(bres.f1s, dtype=np.float64),
                np.asarray(bres.f2s, dtype=np.float64),
            )

        bicoh = torch.from_numpy(np.ascontiguousarray(bres.bicoherence)).float()
        bicoh = torch.nan_to_num(bicoh, nan=0.0)

        if self.return_biphase:
            biphase = torch.from_numpy(np.ascontiguousarray(bres.biphase)).float()
            biphase = torch.nan_to_num(biphase, nan=0.0)
            return bicoh, biphase

        return bicoh

    # -- introspection helpers ----------------------------------------------

    def get_window_metadata(self, idx: int) -> dict:
        """Return the metadata dict for the window at ``idx``."""
        return dict(self.windows[idx])

    def get_frequency_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(f1_vec, f2_vec)`` axes of the bicoherence grid."""
        if self._freq_axes is not None:
            return self._freq_axes

        seg_len = max(4, int(round(self.segment_seconds * self.fs)))
        probe = np.zeros(max(seg_len, int(round(self.window_seconds * self.fs))))
        bres = compute_bicoherence(
            signal=probe,
            fs=self.fs,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )
        self._freq_axes = (
            np.asarray(bres.f1s, dtype=np.float64),
            np.asarray(bres.f2s, dtype=np.float64),
        )
        return self._freq_axes

    # -- HDF5 persistence ---------------------------------------------------

    def _compute_both(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute and return ``(bicoh_tensor, biphase_tensor)`` for window ``idx``."""
        meta = self.windows[idx]
        window_signal = self._load_window_signal(
            meta["window_start"], meta["window_stop"]
        )

        bres = compute_bicoherence(
            signal=window_signal,
            fs=self.fs,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )

        if self._freq_axes is None:
            self._freq_axes = (
                np.asarray(bres.f1s, dtype=np.float64),
                np.asarray(bres.f2s, dtype=np.float64),
            )

        bicoh = torch.from_numpy(np.ascontiguousarray(bres.bicoherence)).float()
        bicoh = torch.nan_to_num(bicoh, nan=0.0)
        biphase = torch.from_numpy(np.ascontiguousarray(bres.biphase)).float()
        biphase = torch.nan_to_num(biphase, nan=0.0)
        return bicoh, biphase

    def _read_provenance_attrs(self) -> dict:
        """Read RecordingConfig attrs from the preprocessed channel dataset."""
        with h5py.File(self.data_file, "r") as f:
            return dict(f[self.channel].attrs)

    def save_to_hdf5(
        self,
        path: str,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> None:
        """Compute all bicoherence maps and write them to an HDF5 file.

        The resulting file is layout-compatible with
        :meth:`BicoherenceWindowDataset.save_to_hdf5` and can be loaded
        instantly via :class:`PrecomputedBicoherenceDataset`.
        """
        n = len(self)
        if n == 0:
            raise RuntimeError("Dataset is empty — nothing to save.")

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        first_bicoh, first_biphase = self._compute_both(0)
        n_f1, n_f2 = first_bicoh.shape
        f1_vec, f2_vec = self.get_frequency_axes()

        # Read provenance attrs from preprocessed file.
        prov = self._read_provenance_attrs()

        str_dtype = h5py.special_dtype(vlen=str)

        with h5py.File(path, "w") as f:
            bicoh_ds = f.create_dataset(
                "bicoherence",
                shape=(n, n_f1, n_f2),
                dtype=np.float32,
                chunks=(1, n_f1, n_f2),
                compression="gzip",
                compression_opts=4,
            )
            bicoh_ds[0] = first_bicoh.numpy()

            if self.return_biphase:
                biphase_ds = f.create_dataset(
                    "biphase",
                    shape=(n, n_f1, n_f2),
                    dtype=np.float32,
                    chunks=(1, n_f1, n_f2),
                    compression="gzip",
                    compression_opts=4,
                )
                biphase_ds[0] = first_biphase.numpy()

            f.create_dataset("f1s", data=f1_vec, dtype=np.float64)
            f.create_dataset("f2s", data=f2_vec, dtype=np.float64)

            meta_grp = f.create_group("metadata")
            epoch_id_ds  = meta_grp.create_dataset("epoch_id",    shape=(n,), dtype=str_dtype)
            win_idx_ds   = meta_grp.create_dataset("window_idx",  shape=(n,), dtype=np.int32)
            win_start_ds = meta_grp.create_dataset("window_start",shape=(n,), dtype=np.float64)
            win_stop_ds  = meta_grp.create_dataset("window_stop", shape=(n,), dtype=np.float64)
            phase_ds     = meta_grp.create_dataset("phase",       shape=(n,), dtype=str_dtype)

            m = self.windows[0]
            epoch_id_ds[0]  = m["epoch_id"]
            win_idx_ds[0]   = m["window_idx"]
            win_start_ds[0] = m["window_start"]
            win_stop_ds[0]  = m["window_stop"]
            phase_ds[0]     = m["phase"]

            if progress_cb is not None:
                progress_cb(1, n)

            for i in range(1, n):
                bicoh_t, biphase_t = self._compute_both(i)
                bicoh_ds[i] = bicoh_t.numpy()
                if self.return_biphase:
                    biphase_ds[i] = biphase_t.numpy()
                m = self.windows[i]
                epoch_id_ds[i]  = m["epoch_id"]
                win_idx_ds[i]   = m["window_idx"]
                win_start_ds[i] = m["window_start"]
                win_stop_ds[i]  = m["window_stop"]
                phase_ds[i]     = m["phase"]
                if progress_cb is not None:
                    progress_cb(i + 1, n)

            # Root config attributes.
            f.attrs["mode"]                   = "full_recording"
            f.attrs["preprocessed_data_file"] = self.data_file
            f.attrs["epoch_boundaries_csv"]   = self.epoch_boundaries_csv or ""
            f.attrs["window_seconds"]         = self.window_seconds
            f.attrs["window_step"]            = self.window_step
            f.attrs["segment_seconds"]        = self.segment_seconds
            f.attrs["segment_overlap"]        = self.segment_overlap
            f.attrs["f_max"]                  = self.f_max
            f.attrs["channel"]                = self.channel
            f.attrs["fs"]                     = self.fs
            f.attrs["smooth_sigma"]           = self.smooth_sigma
            f.attrs["blank_seconds"]          = self.blank_seconds
            f.attrs["min_phase_windows"]      = self.min_phase_windows
            f.attrs["rec_start"]              = self._rec_start
            f.attrs["rec_stop"]               = self._rec_stop
            f.attrs["n_windows"]              = n
            f.attrs["n_f1"]                   = n_f1
            f.attrs["n_f2"]                   = n_f2
            f.attrs["has_biphase"]            = int(self.return_biphase)
            # Provenance: copy RecordingConfig attrs from the preprocessed file.
            for k in PROVENANCE_ATTRS:
                if k in prov:
                    f.attrs[k] = prov[k]


# ---------------------------------------------------------------------------
# Pre-computed dataset (reads from HDF5 written by save_to_hdf5)
# ---------------------------------------------------------------------------

class PrecomputedBicoherenceDataset(Dataset):
    """Fast dataset backed by a pre-computed HDF5 file.

    Reads bicoherence maps written by
    :meth:`BicoherenceWindowDataset.save_to_hdf5` without any signal I/O or
    bicoherence computation. Each ``__getitem__`` call opens the HDF5 file,
    reads one ``[n_f1, n_f2]`` slice, and closes it — safe for DataLoader
    workers (``num_workers > 0``).

    Parameters
    ----------
    hdf5_path : str
        Path to the ``.h5`` file produced by ``save_to_hdf5``.
    return_biphase : bool
        If ``True``, ``__getitem__`` returns a tuple
        ``(bicoh_tensor, biphase_tensor)`` instead of a single tensor.
        Raises :exc:`RuntimeError` if the HDF5 file does not contain a
        ``/biphase`` dataset.  Default ``False``.
    """

    def __init__(self, hdf5_path: str, return_biphase: bool = False) -> None:
        super().__init__()
        self.hdf5_path = hdf5_path
        self.return_biphase = bool(return_biphase)

        with h5py.File(hdf5_path, "r") as f:
            self.config: dict = dict(f.attrs)
            self.f1s = np.asarray(f["f1s"], dtype=np.float64)
            self.f2s = np.asarray(f["f2s"], dtype=np.float64)
            n = int(self.config["n_windows"])
            # Detect biphase dataset presence via root attribute or key existence.
            self.has_biphase: bool = bool(
                int(self.config.get("has_biphase", 0))
            ) or ("biphase" in f)

            meta = f["metadata"]
            epoch_ids   = [meta["epoch_id"][i]   for i in range(n)]
            win_idxs    = meta["window_idx"][:]
            win_starts  = meta["window_start"][:]
            win_stops   = meta["window_stop"][:]
            phases      = [meta["phase"][i]      for i in range(n)]

        if self.return_biphase and not self.has_biphase:
            raise RuntimeError(
                f"return_biphase=True but the HDF5 file {hdf5_path!r} does "
                "not contain a /biphase dataset. Re-run save_to_hdf5 to "
                "generate a file with biphase data."
            )

        self.windows: list[dict] = [
            {
                "epoch_id":          epoch_ids[i],
                "window_idx":        int(win_idxs[i]),
                "global_window_idx": i,
                "window_start":      float(win_starts[i]),
                "window_stop":       float(win_stops[i]),
                "phase":             phases[i],
            }
            for i in range(n)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        with h5py.File(self.hdf5_path, "r") as f:
            arr = f["bicoherence"][idx]
            if self.return_biphase:
                biphase_arr = f["biphase"][idx]

        bicoh = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if self.return_biphase:
            biphase = torch.from_numpy(np.ascontiguousarray(biphase_arr)).float()
            return bicoh, biphase
        return bicoh

    def get_window_metadata(self, idx: int) -> dict:
        """Return the metadata dict for the window at ``idx``."""
        return dict(self.windows[idx])

    def get_frequency_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(f1s, f2s)`` frequency axes stored in the HDF5 file."""
        return self.f1s, self.f2s
