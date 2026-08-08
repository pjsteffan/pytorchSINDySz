import os
import torch
import numpy as np
import h5py
import pickle
from torch.utils.data import Dataset
from collections import Counter
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, welch




class WRsmallepoch(Dataset):
    def __init__(self, data_file: str, annotation_file: str, epoch_size: float, single_channel_flag: bool =True, psd_flag: bool = True, epoch_id_restriction: int = None, sample_rate: int = 5000):
        self.data_file = data_file
        self.annotation_file = annotation_file
        self.annotations = self.load_annotations(epoch_id_restriction)
        self.epoch_size = epoch_size
        self.sample_rate = sample_rate
        self.epoch_num_samples = self.epoch_size * self.sample_rate
        self.frequencies = self.compute_frequency_vector()
        self.freq_weights = torch.Tensor(np.roll(np.unique(self.frequencies),1))
        self.single_channel_flag = single_channel_flag
        self.psd_flag = psd_flag


    def compute_frequency_vector(self):
        # Example vector
        epochs = self.annotations['epoch_id'].to_list()
        # Step 1: Count occurrences of each number
        counts = Counter(epochs)

        # Step 2: Calculate relative frequency
        total_count = len(epochs)
        relative_frequency = {num: count / total_count for num, count in counts.items()}

        # Step 3: Replace each number with its relative frequency
        result_vector = [relative_frequency[num] for num in epochs]
        return torch.Tensor(result_vector)
    
    def load_annotations(self, epoch_id_restriction):
        with open(self.annotation_file, 'rb') as f:
            annotations = pickle.load(f)
        
        if epoch_id_restriction is not None:
            annotations = annotations[annotations['epoch_id'] == epoch_id_restriction]
        
        
        return annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations.iloc[idx]
        start_time = annotation['start_time']
        end_time = annotation['stop_time']
        label = annotation['epoch_id']

        start_index = int(start_time * self.sample_rate)
        
        if self.single_channel_flag:
            with h5py.File(self.data_file, 'r') as f:
                ch1_data = f['Ch.1'][start_index:int(start_index+self.epoch_num_samples)]
                ch1_mean = f['Ch.1'].attrs['mean']
                ch1_std = f['Ch.1'].attrs['std']
            
            ch1_data = self.downsample(ch1_data, original_fs=self.sample_rate, target_fs=100)
            ch1_data = self.filter_data(ch1_data, lowcut=5, highcut=30, fs=100.0, order=5)
            ch1_data = (ch1_data - ch1_mean) / (ch1_std + 1e-6)
            if self.psd_flag:
                _, ch1_data = self.power_spectrum(ch1_data, fs=100.0)
            data_tensor = torch.as_tensor(ch1_data.copy(), dtype=torch.get_default_dtype())
            label_tensor = torch.tensor(label, dtype=torch.long)
            return (data_tensor, label_tensor)
        else:
            with h5py.File(self.data_file, 'r') as f:
                ch1_data = f['Ch.1'][start_index:int(start_index+self.epoch_num_samples)]
                ch2_data = f['Ch.2'][start_index:int(start_index+self.epoch_num_samples)]
                
                ch1_mean = f['Ch.1'].attrs['mean']
                ch1_std = f['Ch.1'].attrs['std']
                ch2_mean = f['Ch.2'].attrs['mean']
                ch2_std = f['Ch.2'].attrs['std']
            
            ch1_data = self.downsample(ch1_data, original_fs=self.sample_rate, target_fs=100)
            ch2_data = self.downsample(ch2_data, original_fs=self.sample_rate, target_fs=100)

            ch1_data = self.filter_data(ch1_data, lowcut=5, highcut=30, fs=100.0, order=5)
            ch2_data = self.filter_data(ch2_data, lowcut=5, highcut=30, fs=100.0, order=5)

            if self.psd_flag:
                _, ch1_data = self.power_spectrum(ch1_data, fs=100.0)
                _, ch2_data = self.power_spectrum(ch2_data, fs=100.0)

            #normalize each channel separately to zero mean and unit variance
            ch1_data = (ch1_data - ch1_mean) / (ch1_std + 1e-10)  # add small value to avoid division by zero
            ch2_data = (ch2_data - ch2_mean) / (ch2_std + 1e-10)


            
            epoch_data = np.stack([ch1_data, ch2_data], axis=0)  # Shape: (2, num_samples)
            epoch_data = epoch_data.transpose(1, 0)  # Shape: (num_samples, 2)

            data_tensor = torch.as_tensor(epoch_data.copy(), dtype=torch.get_default_dtype())
            label_tensor = torch.tensor(label, dtype=torch.long)

            return (data_tensor, label_tensor)

    def downsample(self, data, original_fs=5000, target_fs=100):
        """
        Downsample the data to the target frequency using 1D interpolation.

        Parameters:
        - data: The original data array.
        - original_fs: The original sampling frequency (default is 5000 Hz).
        - target_fs: The target sampling frequency (default is 100 Hz).

        Returns:
        - downsampled_data: The data resampled to the target frequency.
        """
        duration = len(data) / original_fs
        time_original = np.linspace(0, duration, len(data))
        time_target = np.linspace(0, duration, int(duration * target_fs))

        interpolator = interp1d(time_original, data, kind='linear')
        downsampled_data = interpolator(time_target)

        return downsampled_data
    
    def filter_data(self, data, lowcut=5, highcut=30, fs=100.0, order=5):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    
    def power_spectrum(self, data, fs=100.0):
        freqs, psd = welch(data, fs,nperseg=150)
        return freqs, np.log1p(psd) 


class BicoherenceSequenceDataset(Dataset):
    """Sequences of per-window bicoherence maps for the convolutional SINDy AE.

    Each **sample** is one contiguous sequence of ``seq_len`` annotation
    segments that share the same ``epoch_id``. Every segment in the sequence is
    preprocessed (downsample -> band-pass -> z-score, mirroring
    :class:`WRsmallepoch`) and turned into a single bicoherence map
    ``b^2(f1, f2)`` via :func:`compute_bicoherence`. The stack of maps forms the
    SINDy time axis ``T = seq_len``.

    ``__getitem__`` returns ``(maps, mask, label)`` where:

    - ``maps``: float tensor ``[T, 1, H, W]`` (NaNs replaced with 0), ordered by
      segment start time.
    - ``mask``: float tensor ``[1, H, W]`` marking the valid lower-triangular
      region (``f1 <= f2`` and ``f1 + f2 <= f_max``). The frequency grid is
      constant across all windows of an epoch, so a single per-sample mask is
      sufficient (derived from the finite-value pattern of the maps).
    - ``label``: the ``epoch_id`` (kept for parity with :class:`WRsmallepoch`).

    The grid size ``(H, W)`` is determined by ``compute_bicoherence`` (it depends
    on ``sample_rate``, ``segment_seconds`` and ``f_max``) and is exposed via the
    :attr:`height`/:attr:`width` attributes and :meth:`get_grid_size` so the
    convolutional autoencoder can be sized to match.

    Parameters
    ----------
    data_file, annotation_file : str
        Same HDF5 signal file and pickled annotation table used by
        :class:`WRsmallepoch`.
    seq_len : int
        Number of consecutive same-epoch segments per sample (the SINDy T).
    epoch_size : float
        Duration (seconds) of each annotation segment window.
    f_max : float
        Maximum bicoherence frequency (default ``25.0``).
    segment_seconds, segment_overlap : float
        Sub-segment size / overlap used by :func:`compute_bicoherence` to
        average the bispectrum within one window.
    smooth_sigma : float
        Optional NaN-aware Gaussian smoothing of each bicoherence map.
    sample_rate : int
        Native sampling rate of the HDF5 signal (downsampled to 100 Hz, as in
        :class:`WRsmallepoch`).
    epoch_id_restriction : int, optional
        If given, keep only annotations with this ``epoch_id``.
    stride : int, optional
        Step (in segments) between consecutive sequence samples. Defaults to
        ``seq_len`` (non-overlapping sequences).
    """

    # Bicoherence is computed on the 100 Hz preprocessed signal.
    _TARGET_FS = 100.0

    def __init__(
        self,
        data_file: str,
        annotation_file: str,
        seq_len: int = 8,
        epoch_size: float = 5.0,
        f_max: float = 25.0,
        segment_seconds: float = 0.75,
        segment_overlap: float = 0.5,
        smooth_sigma: float = 0.0,
        sample_rate: int = 5000,
        epoch_id_restriction: int | None = None,
        stride: int | None = None,
    ):
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2 (SINDy needs a time axis)")

        self.data_file = data_file
        self.annotation_file = annotation_file
        self.seq_len = int(seq_len)
        self.epoch_size = float(epoch_size)
        self.f_max = float(f_max)
        self.segment_seconds = float(segment_seconds)
        self.segment_overlap = float(segment_overlap)
        self.smooth_sigma = float(smooth_sigma)
        self.sample_rate = int(sample_rate)
        self.stride = int(stride) if stride is not None else int(seq_len)

        # Reuse WRsmallepoch for signal loading + preprocessing of one segment.
        self._base = WRsmallepoch(
            data_file=data_file,
            annotation_file=annotation_file,
            single_channel_flag=True,
            psd_flag=False,  # we need the time-domain signal for bicoherence
            epoch_id_restriction=epoch_id_restriction,
            epoch_size=epoch_size,
            sample_rate=sample_rate,
        )

        # Build the list of sequence start indices (contiguous same-epoch runs).
        self._sequences = self._build_sequences()

        # Grid size + mask are discovered lazily on first access and cached.
        self._grid: tuple[int, int] | None = None
        self._mask: torch.Tensor | None = None

    # -- sequence index construction ---------------------------------------

    def _build_sequences(self) -> list[list[int]]:
        annotations = self._base.annotations.reset_index(drop=True)
        epoch_ids = annotations["epoch_id"].to_list()

        sequences: list[list[int]] = []
        n = len(epoch_ids)
        i = 0
        # Group contiguous rows with identical epoch_id, then window each run.
        while i < n:
            j = i
            while j < n and epoch_ids[j] == epoch_ids[i]:
                j += 1
            run = list(range(i, j))  # indices for this epoch run
            # Slide fixed-length windows across the run.
            for start in range(0, len(run) - self.seq_len + 1, self.stride):
                sequences.append(run[start : start + self.seq_len])
            i = j
        if not sequences:
            raise ValueError(
                "No sequences of length "
                f"{self.seq_len} found; reduce seq_len or check the "
                "annotation table / epoch_id_restriction."
            )
        return sequences

    # -- bicoherence for one segment ---------------------------------------

    def _segment_signal(self, idx: int) -> np.ndarray:
        """Return the preprocessed 100 Hz time-domain signal for one segment."""
        annotation = self._base.annotations.iloc[idx]
        start_time = annotation["start_time"]
        start_index = int(start_time * self.sample_rate)
        with h5py.File(self.data_file, "r") as f:
            ch1 = f["Ch.1"][
                start_index : int(start_index + self._base.epoch_num_samples)
            ]
            ch1_mean = f["Ch.1"].attrs["mean"]
            ch1_std = f["Ch.1"].attrs["std"]
        ch1 = self._base.downsample(
            ch1, original_fs=self.sample_rate, target_fs=self._TARGET_FS
        )
        ch1 = self._base.filter_data(
            ch1, lowcut=5, highcut=30, fs=self._TARGET_FS, order=5
        )
        ch1 = (ch1 - ch1_mean) / (ch1_std + 1e-6)
        return np.ascontiguousarray(ch1, dtype=np.float64)

    def _bicoherence_map(self, idx: int) -> np.ndarray:
        """Compute the bicoherence map for one segment (NaNs preserved)."""
        # Imported lazily so the base dataset can be used without pybispectra.
        from REFERENCE.bispectral.analysis import compute_bicoherence

        signal = self._segment_signal(idx)
        bres = compute_bicoherence(
            signal=signal,
            fs=self._TARGET_FS,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )
        return np.asarray(bres.bicoherence, dtype=np.float64)

    # -- grid / mask helpers -----------------------------------------------

    def _ensure_grid_and_mask(self) -> None:
        if self._grid is not None:
            return
        # Use the first segment of the first sequence to probe the grid + mask.
        probe_idx = self._sequences[0][0]
        bmap = self._bicoherence_map(probe_idx)
        H, W = bmap.shape
        self._grid = (int(H), int(W))
        # The invalid triangular region is NaN in the raw bicoherence; the mask
        # is the finite (valid) region and is constant across all windows since
        # the frequency grid depends only on fs/seg_seconds/f_max.
        mask = np.isfinite(bmap).astype(np.float32)
        self._mask = torch.from_numpy(mask).reshape(1, H, W)

    @property
    def height(self) -> int:
        self._ensure_grid_and_mask()
        return self._grid[0]

    @property
    def width(self) -> int:
        self._ensure_grid_and_mask()
        return self._grid[1]

    def get_grid_size(self) -> tuple[int, int]:
        self._ensure_grid_and_mask()
        return self._grid

    def get_mask(self) -> torch.Tensor:
        """Return the shared valid-region mask, shape ``[1, H, W]``."""
        self._ensure_grid_and_mask()
        return self._mask.clone()

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        self._ensure_grid_and_mask()
        H, W = self._grid
        seg_indices = self._sequences[idx]

        maps = np.empty((self.seq_len, 1, H, W), dtype=np.float32)
        for t, seg_idx in enumerate(seg_indices):
            bmap = self._bicoherence_map(seg_idx)
            maps[t, 0] = np.nan_to_num(bmap, nan=0.0).astype(np.float32)

        maps_t = torch.from_numpy(maps).to(torch.get_default_dtype())
        mask_t = self._mask.to(torch.get_default_dtype())
        # Sequence label = epoch_id of the run (shared across the sequence).
        label = int(self._base.annotations.iloc[seg_indices[0]]["epoch_id"])
        label_t = torch.tensor(label, dtype=torch.long)
        return (maps_t, mask_t, label_t)


class RawBicoherenceSequenceDataset(Dataset):
    """Annotation-agnostic variant of :class:`BicoherenceSequenceDataset`.

    Produces the exact same per-window bicoherence-map sequences, but tiles
    windows **contiguously across the entire raw recording** instead of using an
    annotation table to place windows or to group them into sequences. The
    bicoherence windowing/math is identical to
    :class:`BicoherenceSequenceDataset`.

    Window ``k`` starts at ``k * epoch_size`` seconds and is
    ``epoch_size`` seconds long. The whole recording is treated as one
    contiguous run of ``num_windows = floor(total_samples / epoch_num_samples)``
    windows, sliced into fixed-length ``seq_len`` sequences with ``stride``
    (default ``seq_len``, i.e. non-overlapping sequences). The trailing partial
    window (if any) is dropped so every window is full length.

    ``__getitem__`` returns ``(maps, mask, label)`` matching the annotated class:

    - ``maps``: float tensor ``[T, 1, H, W]`` (NaNs replaced with 0).
    - ``mask``: float tensor ``[1, H, W]`` marking the valid lower-triangular
      region.
    - ``label``: constant ``-1`` placeholder (kept only for API parity; there is
      no annotation to derive a real label from).

    Preprocessing per window mirrors :class:`WRsmallepoch` exactly:
    ``downsample`` -> band-pass ``filter_data`` (5-30 Hz) -> z-score using the
    recording-wide ``mean``/``std`` HDF5 attributes.

    Parameters
    ----------
    data_file : str
        HDF5 signal file (same format used by :class:`WRsmallepoch`).
    seq_len : int
        Number of consecutive windows per sample (the SINDy T).
    epoch_size : float
        Duration (seconds) of each window.
    f_max, segment_seconds, segment_overlap, smooth_sigma :
        Bicoherence parameters, forwarded to :func:`compute_bicoherence`.
    sample_rate : int
        Native sampling rate of the HDF5 signal (downsampled to 100 Hz).
    stride : int, optional
        Step (in windows) between consecutive sequence samples. Defaults to
        ``seq_len`` (non-overlapping sequences).
    channel : str
        HDF5 dataset key to read (default ``"Ch.1"``).
    """

    # Bicoherence is computed on the 100 Hz preprocessed signal.
    _TARGET_FS = 100.0

    def __init__(
        self,
        data_file: str,
        seq_len: int = 8,
        epoch_size: float = 5.0,
        f_max: float = 25.0,
        segment_seconds: float = 0.75,
        segment_overlap: float = 0.5,
        smooth_sigma: float = 0.0,
        sample_rate: int = 5000,
        stride: int | None = None,
        channel: str = "Ch.1",
    ):
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2 (SINDy needs a time axis)")

        self.data_file = data_file
        self.seq_len = int(seq_len)
        self.epoch_size = float(epoch_size)
        self.f_max = float(f_max)
        self.segment_seconds = float(segment_seconds)
        self.segment_overlap = float(segment_overlap)
        self.smooth_sigma = float(smooth_sigma)
        self.sample_rate = int(sample_rate)
        self.stride = int(stride) if stride is not None else int(seq_len)
        self.channel = str(channel)

        self.epoch_num_samples = int(self.epoch_size * self.sample_rate)

        # Total window count comes from the raw recording length (HDF5 shape).
        with h5py.File(self.data_file, "r") as f:
            n_samples = f[self.channel].shape[0]
        self.num_windows = int(n_samples // self.epoch_num_samples)
        if self.num_windows < self.seq_len:
            raise ValueError(
                f"Recording has only {self.num_windows} full windows of "
                f"{self.epoch_size}s, but seq_len={self.seq_len}. Reduce "
                "seq_len or epoch_size."
            )

        # Contiguous tiling: one run over all windows, sliced into sequences.
        self._sequences = self._build_sequences()

        # Grid size + mask are discovered lazily on first access and cached.
        self._grid: tuple[int, int] | None = None
        self._mask: torch.Tensor | None = None

    # -- sequence index construction ---------------------------------------

    def _build_sequences(self) -> list[list[int]]:
        run = list(range(self.num_windows))
        sequences: list[list[int]] = [
            run[start : start + self.seq_len]
            for start in range(0, self.num_windows - self.seq_len + 1, self.stride)
        ]
        if not sequences:
            raise ValueError(
                f"No sequences of length {self.seq_len} could be formed from "
                f"{self.num_windows} windows with stride {self.stride}."
            )
        return sequences

    # -- preprocessing / bicoherence for one window ------------------------

    def _segment_signal(self, window_idx: int) -> np.ndarray:
        """Return the preprocessed 100 Hz time-domain signal for one window."""
        start_index = window_idx * self.epoch_num_samples
        with h5py.File(self.data_file, "r") as f:
            ch = f[self.channel][start_index : start_index + self.epoch_num_samples]
            ch_mean = f[self.channel].attrs["mean"]
            ch_std = f[self.channel].attrs["std"]
        # Stateless helpers reused from WRsmallepoch (no instance needed).
        ch = WRsmallepoch.downsample(
            self, ch, original_fs=self.sample_rate, target_fs=self._TARGET_FS
        )
        ch = WRsmallepoch.filter_data(
            self, ch, lowcut=5, highcut=30, fs=self._TARGET_FS, order=5
        )
        ch = (ch - ch_mean) / (ch_std + 1e-6)
        return np.ascontiguousarray(ch, dtype=np.float64)

    def _bicoherence_map(self, window_idx: int) -> np.ndarray:
        """Compute the bicoherence map for one window (NaNs preserved)."""
        from REFERENCE.bispectral.analysis import compute_bicoherence

        signal = self._segment_signal(window_idx)
        bres = compute_bicoherence(
            signal=signal,
            fs=self._TARGET_FS,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )
        return np.asarray(bres.bicoherence, dtype=np.float64)

    # -- grid / mask helpers -----------------------------------------------

    def _ensure_grid_and_mask(self) -> None:
        if self._grid is not None:
            return
        from REFERENCE.bispectral.analysis import compute_bicoherence

        probe_idx = self._sequences[0][0]
        signal = self._segment_signal(probe_idx)
        bres = compute_bicoherence(
            signal=signal,
            fs=self._TARGET_FS,
            f_max=self.f_max,
            seg_seconds=self.segment_seconds,
            overlap=self.segment_overlap,
            smooth_sigma=self.smooth_sigma,
        )
        bmap = np.asarray(bres.bicoherence, dtype=np.float64)
        H, W = bmap.shape
        self._grid = (int(H), int(W))
        mask = np.isfinite(bmap).astype(np.float32)
        self._mask = torch.from_numpy(mask).reshape(1, H, W)
        # Cache frequency axes (Hz) for plotting/labelling; same grid for all
        # windows since it depends only on fs/seg_seconds/f_max.
        self._f1s = np.asarray(bres.f1s, dtype=np.float64)
        self._f2s = np.asarray(bres.f2s, dtype=np.float64)

    @property
    def height(self) -> int:
        self._ensure_grid_and_mask()
        return self._grid[0]

    @property
    def width(self) -> int:
        self._ensure_grid_and_mask()
        return self._grid[1]

    def get_grid_size(self) -> tuple[int, int]:
        self._ensure_grid_and_mask()
        return self._grid

    def get_mask(self) -> torch.Tensor:
        """Return the shared valid-region mask, shape ``[1, H, W]``."""
        self._ensure_grid_and_mask()
        return self._mask.clone()

    def get_freq_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(f1s, f2s)`` frequency axes (Hz) of the bicoherence grid."""
        self._ensure_grid_and_mask()
        return self._f1s.copy(), self._f2s.copy()

    def window_start_seconds(self, seq_idx: int) -> list[float]:
        """Start time (s) of each window in sequence ``seq_idx`` (for labels)."""
        return [w * self.epoch_size for w in self._sequences[seq_idx]]

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        self._ensure_grid_and_mask()
        H, W = self._grid
        win_indices = self._sequences[idx]

        maps = np.empty((self.seq_len, 1, H, W), dtype=np.float32)
        for t, win_idx in enumerate(win_indices):
            bmap = self._bicoherence_map(win_idx)
            maps[t, 0] = np.nan_to_num(bmap, nan=0.0).astype(np.float32)

        maps_t = torch.from_numpy(maps).to(torch.get_default_dtype())
        mask_t = self._mask.to(torch.get_default_dtype())
        # Annotation-agnostic: constant placeholder label for API parity.
        label_t = torch.tensor(-1, dtype=torch.long)
        return (maps_t, mask_t, label_t)


def group_name_for_epoch_size(epoch_size: float) -> str:
    """Return the HDF5 group name used to store maps for ``epoch_size``.

    Kept as a module-level helper so the precompute script and the dataset
    agree on the naming convention (``f"es_{epoch_size:g}"``).
    """
    return f"es_{float(epoch_size):g}"


class PrecomputedBicoherenceSequenceDataset(Dataset):
    """Drop-in replacement for :class:`RawBicoherenceSequenceDataset` that reads
    per-window bicoherence maps from a precomputed HDF5 cache instead of
    computing them on the fly.

    The expensive :func:`compute_bicoherence` call is run **once** by
    ``precompute_bicoherence.py`` and every trial/worker simply reads the
    resulting maps from disk. Because the per-window map depends only on the
    signal slice and the bicoherence parameters (not on ``seq_len``/``stride``/
    ``batch_size``), a single cached map set per ``epoch_size`` serves every
    trial regardless of those hyperparameters.

    ``__getitem__`` returns ``(maps, mask, label)`` identical in shape/dtype to
    :class:`RawBicoherenceSequenceDataset`:

    - ``maps``: float tensor ``[T, 1, H, W]`` (NaNs already replaced with 0 in
      the cache).
    - ``mask``: float tensor ``[1, H, W]`` marking the valid lower-triangular
      region.
    - ``label``: constant ``-1`` placeholder for API parity.

    Concurrency
    -----------
    The cache is opened **read-only** with ``swmr=True``. Each process (each
    Optuna trial) and each DataLoader worker opens its **own** handle lazily on
    first access; the handle is never shared across processes and is dropped from
    the pickled state (see :meth:`__getstate__`) so forked/spawned workers
    re-open it themselves. This is the standard safe pattern for many concurrent
    readers of a static HDF5 file, so multiple trials can read simultaneously
    without conflict (the writer, ``precompute_bicoherence.py``, runs once and
    finishes before any trial starts).

    Parameters
    ----------
    cache_file : str
        Path to the precomputed HDF5 cache (produced by
        ``precompute_bicoherence.py``).
    epoch_size : float
        Window duration (seconds); selects the ``es_{epoch_size:g}`` group.
    seq_len : int
        Number of consecutive windows per sample (the SINDy T).
    stride : int, optional
        Step (in windows) between consecutive sequences. Defaults to ``seq_len``
        (non-overlapping sequences).
    segment_seconds, segment_overlap, f_max, smooth_sigma, sample_rate, channel :
        Bicoherence parameters. These are **validated** against the cache group's
        stored attributes; a mismatch raises with instructions to re-run the
        precompute script. They do not trigger any computation here.
    """

    # Bicoherence is computed on the 100 Hz preprocessed signal.
    _TARGET_FS = 100.0

    def __init__(
        self,
        cache_file: str,
        epoch_size: float,
        seq_len: int = 8,
        stride: int | None = None,
        segment_seconds: float = 0.75,
        segment_overlap: float = 0.5,
        f_max: float = 25.0,
        smooth_sigma: float = 0.0,
        sample_rate: int = 5000,
        channel: str = "Ch.1",
    ):
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2 (SINDy needs a time axis)")

        self.cache_file = str(cache_file)
        self.epoch_size = float(epoch_size)
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else int(seq_len)
        self.segment_seconds = float(segment_seconds)
        self.segment_overlap = float(segment_overlap)
        self.f_max = float(f_max)
        self.smooth_sigma = float(smooth_sigma)
        self.sample_rate = int(sample_rate)
        self.channel = str(channel)

        self._group_name = group_name_for_epoch_size(self.epoch_size)

        # Read metadata (tiny) once in the constructing process, validating the
        # cache against the requested parameters. The read handle used here is
        # closed immediately; per-worker handles are opened lazily in __getitem__.
        self._read_metadata_and_validate()

        # Build the sequence index using the same windowing logic as
        # RawBicoherenceSequenceDataset._build_sequences.
        if self.num_windows < self.seq_len:
            raise ValueError(
                f"Cached group '{self._group_name}' has only {self.num_windows} "
                f"windows, but seq_len={self.seq_len}. Reduce seq_len or "
                "epoch_size."
            )
        self._sequences = self._build_sequences()

        # Per-worker HDF5 handle, opened lazily; never pickled (see __getstate__).
        self._h5 = None
        self._maps_ds = None

    # -- metadata / validation ---------------------------------------------

    def _read_metadata_and_validate(self) -> None:
        if not os.path.exists(self.cache_file):
            raise FileNotFoundError(
                f"Bicoherence cache not found: {self.cache_file}. Run "
                "precompute_bicoherence.py to generate it."
            )
        # Use a plain "r" open for the metadata check: the file is a completed
        # (non-SWMR) HDF5 written by precompute_bicoherence.py. Per-worker
        # reads in __getitem__ use swmr=True + libver="latest" which is
        # compatible with a non-SWMR file for concurrent read-only access.
        with h5py.File(self.cache_file, "r") as f:
            if self._group_name not in f:
                raise KeyError(
                    f"Cache '{self.cache_file}' has no group "
                    f"'{self._group_name}' for epoch_size={self.epoch_size}. "
                    "Re-run precompute_bicoherence.py with this epoch_size."
                )
            g = f[self._group_name]
            if not bool(g.attrs.get("complete", False)):
                raise RuntimeError(
                    f"Cache group '{self._group_name}' is incomplete (a previous "
                    "precompute run may have been interrupted). Re-run "
                    "precompute_bicoherence.py (optionally with --force)."
                )

            # Validate the bicoherence parameters stored on the group against
            # what this dataset was constructed with. A mismatch means the cache
            # was built for different settings and must not be used silently.
            expected = {
                "epoch_size": self.epoch_size,
                "segment_seconds": self.segment_seconds,
                "segment_overlap": self.segment_overlap,
                "f_max": self.f_max,
                "smooth_sigma": self.smooth_sigma,
                "sample_rate": self.sample_rate,
                "channel": self.channel,
            }
            mismatches = []
            for k, want in expected.items():
                got = g.attrs.get(k, None)
                if isinstance(want, str):
                    got_s = got.decode() if isinstance(got, bytes) else got
                    if got_s != want:
                        mismatches.append(f"{k}: cache={got_s!r} requested={want!r}")
                else:
                    if got is None or not np.isclose(float(got), float(want)):
                        mismatches.append(f"{k}: cache={got} requested={want}")
            if mismatches:
                raise ValueError(
                    "Bicoherence cache parameter mismatch for group "
                    f"'{self._group_name}':\n  " + "\n  ".join(mismatches) + "\n"
                    "Re-run precompute_bicoherence.py with matching parameters."
                )

            self.num_windows = int(g.attrs["num_windows"])
            H = int(g.attrs["height"])
            W = int(g.attrs["width"])
            self._grid = (H, W)
            mask = np.asarray(g["mask"][()], dtype=np.float32)
            self._mask = torch.from_numpy(mask).reshape(1, H, W)
            self._f1s = np.asarray(g["f1s"][()], dtype=np.float64)
            self._f2s = np.asarray(g["f2s"][()], dtype=np.float64)

    # -- sequence index construction ---------------------------------------

    def _build_sequences(self) -> list[list[int]]:
        run = list(range(self.num_windows))
        sequences: list[list[int]] = [
            run[start : start + self.seq_len]
            for start in range(0, self.num_windows - self.seq_len + 1, self.stride)
        ]
        if not sequences:
            raise ValueError(
                f"No sequences of length {self.seq_len} could be formed from "
                f"{self.num_windows} windows with stride {self.stride}."
            )
        return sequences

    # -- lazy per-worker handle --------------------------------------------

    def _ensure_open(self) -> None:
        """Open this process's own read-only handle on first access."""
        if self._h5 is None:
            self._h5 = h5py.File(
                self.cache_file, "r", swmr=True, libver="latest"
            )
            self._maps_ds = self._h5[self._group_name]["maps"]

    def __getstate__(self):
        # Do not pickle the open HDF5 handle: each worker must open its own.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_maps_ds"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._h5 = None
        self._maps_ds = None

    # -- grid / mask helpers -----------------------------------------------

    @property
    def height(self) -> int:
        return self._grid[0]

    @property
    def width(self) -> int:
        return self._grid[1]

    def get_grid_size(self) -> tuple[int, int]:
        return self._grid

    def get_mask(self) -> torch.Tensor:
        """Return the shared valid-region mask, shape ``[1, H, W]``."""
        return self._mask.clone()

    def get_freq_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(f1s, f2s)`` frequency axes (Hz) of the bicoherence grid."""
        return self._f1s.copy(), self._f2s.copy()

    def window_start_seconds(self, seq_idx: int) -> list[float]:
        """Start time (s) of each window in sequence ``seq_idx`` (for labels)."""
        return [w * self.epoch_size for w in self._sequences[seq_idx]]

    # -- Dataset protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int):
        self._ensure_open()
        H, W = self._grid
        win_indices = self._sequences[idx]

        # h5py fancy indexing requires a sorted, unique index list. With the
        # default stride == seq_len windows are disjoint and already sorted, but
        # sort defensively and restore the requested order afterwards.
        order = np.argsort(win_indices)
        sorted_idx = list(np.asarray(win_indices)[order])
        sorted_maps = np.asarray(self._maps_ds[sorted_idx], dtype=np.float32)
        # Undo the sort to match the sequence's temporal order.
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        raw = sorted_maps[inv]  # [T, H, W]

        maps = raw.reshape(self.seq_len, 1, H, W)
        maps_t = torch.from_numpy(np.ascontiguousarray(maps)).to(
            torch.get_default_dtype()
        )
        mask_t = self._mask.to(torch.get_default_dtype())
        label_t = torch.tensor(-1, dtype=torch.long)
        return (maps_t, mask_t, label_t)
