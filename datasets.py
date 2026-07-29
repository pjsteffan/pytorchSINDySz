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
