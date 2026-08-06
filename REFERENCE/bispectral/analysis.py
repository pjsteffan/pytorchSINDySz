"""Core bispectral analysis functions for Method 1.

Implements the per-seizure workflow described in WhitePaper.md sections
4.1-4.4 and 6.3:

* Power spectrum via Welch's method.
* Bicoherence (normalised bispectrum) using pybispectra's :class:`Bispectrum`
  combined with :class:`Threenorm` to obtain values bounded in [0, 1].
* Phase-randomised surrogates (Theiler et al. 1992) that preserve the
  power spectrum but destroy phase relationships.
* Per-frequency-pair significance testing against the surrogate null
  distribution.
* Set-based accumulation of frequencies participating in significant
  quadratic phase coupling and the resulting coupling power ratio R.
* Biphase consistency analysis across overlapping sub-segments.
* Harmonic vs cross-frequency classification and the sanity checks from
  WhitePaper section 6.3.

The functions are designed to operate on a *single* preprocessed seizure
signal at a time. Multi-epoch averaging that bicoherence needs is handled
internally by splitting the seizure into overlapping sub-epochs before
calling pybispectra.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import welch

# pybispectra emits its own progress bars + UserWarnings on every call.
# We import once and silence the chattier paths during repeated calls.
from pybispectra import Bispectrum, Threenorm, compute_fft


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _suppressed():
    """Silence pybispectra's stdout / stderr / warnings inside a tight loop.

    .. warning:: **Not thread-safe.**
        ``sys.stdout`` and ``sys.stderr`` are module-level globals.  Concurrent
        calls from multiple *threads* would corrupt each other's saved stdio
        state.  This is safe under the current usage pattern (single-threaded
        main loop; DataLoader workers are separate *processes* on Linux) but
        must be revisited before introducing thread-based parallelism.
    """
    devnull = open(os.devnull, "w")
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = devnull
        sys.stderr = devnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        devnull.close()


def _segment_signal(
    signal: np.ndarray,
    fs: float,
    seg_seconds: float = 2.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """Split a 1-D signal into overlapping sub-epochs.

    Returns an array of shape ``[n_segments, 1, n_samples_per_segment]``,
    matching pybispectra's expected ``[epochs, channels, times]`` layout.

    If the signal is shorter than one segment the whole signal is returned
    as a single epoch (the bispectrum estimate will be noisier but still
    defined).
    """
    n = len(signal)
    seg_len = int(round(seg_seconds * fs))
    if seg_len < 4:
        seg_len = max(4, n)
    if n <= seg_len:
        return signal.reshape(1, 1, n).astype(np.float64, copy=False)

    step = max(1, int(round(seg_len * (1.0 - overlap))))
    starts = list(range(0, n - seg_len + 1, step))
    if not starts:
        starts = [0]
    segs = np.stack([signal[s : s + seg_len] for s in starts], axis=0)
    return segs[:, np.newaxis, :].astype(np.float64, copy=False)


def find_freq_idx(freq_vec: np.ndarray, target: float) -> int:
    """Return the index of the bin nearest to ``target`` in ``freq_vec``."""
    return int(np.argmin(np.abs(np.asarray(freq_vec) - float(target))))


def _smooth_bicoh(b: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware 2D Gaussian smoothing of a bicoherence map.

    Pybispectra fills the lower triangle (``f2 < f1``) and the
    ``f1 + f2 > fs/2`` region with NaN. A naive Gaussian filter would
    propagate those NaNs into the valid principal domain and erode its
    boundary. We instead smooth ``(values * mask)`` and ``mask`` separately
    and divide, which is equivalent to a Gaussian-weighted average over
    only the valid neighbours of each bin. NaNs are restored in their
    original positions and the result is clipped back to [0, 1] to remain
    commensurate with the unsmoothed bicoherence (so downstream
    significance comparisons stay meaningful).

    ``sigma`` is in **bins**. With the default ``seg_seconds=2.0`` the
    grid step is ~0.5 Hz/bin, so ``sigma=1`` ≈ 0.5 Hz and ``sigma=2`` ≈
    1 Hz of smoothing. ``sigma <= 0`` is a no-op.
    """
    if sigma is None or sigma <= 0:
        return b
    nan = ~np.isfinite(b)
    if not nan.any():
        out = gaussian_filter(b, sigma=sigma, mode="nearest")
        return np.clip(out, 0.0, 1.0)
    filled = np.where(nan, 0.0, b)
    weight = (~nan).astype(np.float64)
    num = gaussian_filter(filled, sigma=sigma, mode="nearest")
    den = gaussian_filter(weight, sigma=sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    out[nan] = np.nan
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 1. Power spectrum
# ---------------------------------------------------------------------------

def compute_power_spectrum(
    signal: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectrum.

    Consistent with ``datasets.WRsmallepoch.power_spectrum`` (nperseg=150
    by default) but clamps to signal length for short seizures.
    """
    if nperseg is None:
        nperseg = min(len(signal), 150)
    nperseg = max(8, min(nperseg, len(signal)))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return np.asarray(freqs), np.asarray(psd)


# ---------------------------------------------------------------------------
# 2. Bicoherence
# ---------------------------------------------------------------------------

@dataclass
class BicoherenceResult:
    bicoherence: np.ndarray  # shape [n_f1, n_f2]
    f1s: np.ndarray
    f2s: np.ndarray
    coeffs: np.ndarray | None = None  # FFT coefficients used (for surrogates)
    fft_freqs: np.ndarray | None = None
    biphase: np.ndarray | None = None  # shape [n_f1, n_f2], radians in (-π, π]


def _bicoherence_from_coeffs(
    coeffs: np.ndarray,
    fft_freqs: np.ndarray,
    fs: float,
    f_max: float,
    smooth_sigma: float = 0.0,
) -> BicoherenceResult:
    """Run pybispectra's Bispectrum + Threenorm on supplied FFT coefficients.

    If ``smooth_sigma > 0``, a NaN-aware 2D Gaussian filter (sigma in bins)
    is applied to the final bicoherence map. The same value must be used
    for observed *and* surrogate bicoherences (see
    :func:`compute_significance_mask`) so that the per-bin significance
    threshold is computed on a matched null distribution.
    """
    with _suppressed():
        bs = Bispectrum(coeffs, fft_freqs, fs, verbose=False)
        bs.compute(f1s=(0.0, f_max), f2s=(0.0, f_max), n_jobs=1)
        bs_arr = bs.results.get_results()

        tn = Threenorm(coeffs, fft_freqs, fs, verbose=False)
        tn.compute(f1s=(0.0, f_max), f2s=(0.0, f_max), n_jobs=1)
        tn_arr = tn.results.get_results()

        f1_vec = np.asarray(bs.results.f1s)
        f2_vec = np.asarray(bs.results.f2s)

        # Extract biphase from the complex bispectrum while still inside the
        # suppressed block (the bispectrum object is still valid here).
        # bs_arr[0] is the single-epoch complex bispectrum, shape [n_f1, n_f2].
        # np.angle returns values in (-π, π].
        biphase_raw = np.angle(np.asarray(bs_arr[0]))

    bs2d = np.asarray(bs_arr[0])
    tn2d = np.asarray(tn_arr[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        bicoh = np.abs(bs2d) / tn2d
    # Capture pre-smoothing NaN mask to apply the same mask to biphase.
    bicoh_pre_clip = np.where(np.isfinite(bicoh), bicoh, np.nan)
    bicoh = np.clip(bicoh_pre_clip, 0.0, 1.0)
    bicoh = _smooth_bicoh(bicoh, smooth_sigma)

    # NaN biphase wherever the raw bicoherence is NaN (invalid triangular
    # region and the mask from pybispectra). Biphase is NOT smoothed because
    # Gaussian smoothing in rectilinear space is incorrect for circular data.
    biphase = np.where(np.isfinite(bicoh_pre_clip), biphase_raw, np.nan)

    return BicoherenceResult(
        bicoherence=bicoh, f1s=f1_vec, f2s=f2_vec,
        coeffs=coeffs, fft_freqs=fft_freqs,
        biphase=biphase,
    )


def compute_bicoherence(
    signal: np.ndarray,
    fs: float,
    f_max: float = 25.0,
    seg_seconds: float = 2.0,
    overlap: float = 0.5,
    smooth_sigma: float = 0.0,
) -> BicoherenceResult:
    """Compute the bicoherence ``b^2(f1, f2)`` for a single seizure.

    The seizure is segmented into overlapping sub-epochs so that the
    bispectrum and its normalising threenorm can be averaged, yielding a
    bicoherence bounded in [0, 1].

    ``smooth_sigma`` (in bicoherence-grid bins) optionally applies a
    NaN-aware 2D Gaussian smoother to the resulting map. The same value
    must be passed to :func:`compute_significance_mask` so the surrogate
    null distribution is smoothed identically.
    """
    data_3d = _segment_signal(signal, fs, seg_seconds=seg_seconds, overlap=overlap)
    with _suppressed():
        coeffs, fft_freqs = compute_fft(data_3d, fs, verbose=False)
    return _bicoherence_from_coeffs(coeffs, fft_freqs, fs, f_max, smooth_sigma=smooth_sigma)


def randomise_segment_phases(
    coeffs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Phase-randomise pybispectra FFT coefficients per segment.

    ``coeffs`` has shape ``[n_segments, n_channels, n_freqs]`` (complex).
    Each segment's per-bin magnitude is preserved (so each segment's PSD
    is preserved) while the phases are independently randomised, which
    destroys phase coupling **within each segment** — exactly the null
    hypothesis we want for surrogate-based bicoherence testing.

    The DC bin is kept real (zero phase) and the Nyquist bin (if present
    as the final coefficient) is also kept real.
    """
    mag = np.abs(coeffs)
    new_phases = rng.uniform(0.0, 2.0 * np.pi, size=coeffs.shape)
    new_phases[..., 0] = 0.0
    # pybispectra's rFFT keeps the Nyquist bin only for even-length signals;
    # forcing its phase to zero is harmless in either case.
    new_phases[..., -1] = 0.0
    return mag * np.exp(1j * new_phases)


# ---------------------------------------------------------------------------
# 3. Phase-randomised surrogates
# ---------------------------------------------------------------------------

def generate_phase_surrogate(
    signal: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one phase-randomised surrogate (Theiler et al. 1992).

    Preserves the magnitude spectrum (and therefore the power spectrum)
    exactly while destroying phase-coupling structure.
    """
    n = len(signal)
    X = np.fft.rfft(signal)
    mag = np.abs(X)

    # Randomise phases of interior bins; keep DC (and Nyquist if even n) real
    # to ensure the resulting time series stays real-valued.
    new_phases = rng.uniform(0.0, 2.0 * np.pi, size=X.shape)
    new_phases[0] = 0.0
    if n % 2 == 0:
        new_phases[-1] = 0.0

    X_surr = mag * np.exp(1j * new_phases)
    return np.fft.irfft(X_surr, n=n)


def generate_phase_surrogates(
    signal: np.ndarray,
    n_surrogates: int = 100,
    seed: int | None = None,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [generate_phase_surrogate(signal, rng) for _ in range(n_surrogates)]


# ---------------------------------------------------------------------------
# 4. Surrogate-based significance mask
# ---------------------------------------------------------------------------

def compute_significance_mask(
    bicoherence_result: BicoherenceResult,
    fs: float,
    n_surrogates: int = 100,
    alpha: float = 0.05,
    f_max: float = 25.0,
    seed: int | None = None,
    progress_cb=None,
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a boolean significance mask against phase-randomised surrogates.

    The surrogates are constructed by independently randomising the phase
    of every (segment, channel, frequency) bin in the **already-segmented**
    FFT coefficients used for the observed bicoherence. This is the
    standard "phase-randomised null" for averaged bicoherence: each
    segment retains its magnitude spectrum, so the average power spectrum
    is preserved, but inter-frequency phase relations within every
    segment are destroyed.

    Parameters
    ----------
    bicoherence_result : BicoherenceResult
        Result from :func:`compute_bicoherence` for the observed signal.
        Must carry the segmented FFT coefficients (it always does when
        produced by ``compute_bicoherence``).

    Returns
    -------
    mask : ndarray of bool, shape [n_f1, n_f2]
        ``True`` where the observed bicoherence exceeds the
        (1 - alpha) percentile of the surrogate null distribution.
    threshold : ndarray of float, shape [n_f1, n_f2]
        Per-bin significance threshold.
    """
    if bicoherence_result.coeffs is None or bicoherence_result.fft_freqs is None:
        raise ValueError(
            "BicoherenceResult does not carry FFT coefficients; rebuild it "
            "via compute_bicoherence() to enable surrogate testing."
        )

    bicoh_obs = bicoherence_result.bicoherence
    null = np.empty((n_surrogates, *bicoh_obs.shape), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for i in range(n_surrogates):
        surr_coeffs = randomise_segment_phases(bicoherence_result.coeffs, rng)
        res = _bicoherence_from_coeffs(
            surr_coeffs, bicoherence_result.fft_freqs, fs, f_max,
            smooth_sigma=smooth_sigma,
        )
        null[i] = res.bicoherence
        if progress_cb is not None:
            progress_cb(i + 1, n_surrogates)

    threshold = np.nanpercentile(null, 100.0 * (1.0 - alpha), axis=0)
    with np.errstate(invalid="ignore"):
        mask = (
            (bicoh_obs > threshold)
            & np.isfinite(bicoh_obs)
            & np.isfinite(threshold)
        )
    return mask, threshold


# ---------------------------------------------------------------------------
# 5. Coupled frequencies (set-based accumulation)
# ---------------------------------------------------------------------------

def extract_coupled_frequencies(
    bicoherence: np.ndarray,
    f1_vec: np.ndarray,
    f2_vec: np.ndarray,
    significance_mask: np.ndarray,
    fs: float | None = None,
    f_band: Tuple[float, float] | None = None,
) -> Tuple[Set[float], List[Tuple[float, float, float, float]]]:
    """Collect the unique frequencies participating in significant QPC.

    Iterates over the principal domain (``f1 <= f2``, ``f1 + f2 <= fs/2``)
    used by pybispectra, deduplicating the participating frequencies via
    a Python ``set`` so the same bin is never counted twice when it
    appears in multiple triads.

    Parameters
    ----------
    f_band : (low, high) tuple, optional
        If given, only triads whose **all three** frequencies fall in
        ``[low, high]`` are kept. This is important when the input was
        bandpass-filtered: components outside the passband have been
        attenuated and any apparent coupling involving them is an
        artefact of the bicoherence estimator rather than real physiology.

    Returns
    -------
    coupled_freqs : set of float
        Frequencies (Hz) that participate in at least one significant triad.
    triads : list of (f1, f2, f3, bicoherence_value)
        Significant triads, all within the principal domain (and band if
        ``f_band`` was supplied).
    """
    n1, n2 = bicoherence.shape
    nyquist = fs / 2.0 if fs is not None else math.inf
    if f_band is not None:
        f_lo, f_hi = float(f_band[0]), float(f_band[1])
    else:
        f_lo, f_hi = 0.0, math.inf
    coupled: Set[float] = set()
    triads: List[Tuple[float, float, float, float]] = []
    for i in range(n1):
        f1 = float(f1_vec[i])
        for j in range(n2):
            f2 = float(f2_vec[j])
            # Pybispectra principal domain: f1 <= f2 (lower triangle is NaN).
            if f2 < f1:
                continue
            f3 = f1 + f2
            if f3 > nyquist:
                continue
            if not (f_lo <= f1 <= f_hi and f_lo <= f2 <= f_hi and f_lo <= f3 <= f_hi):
                continue
            if not significance_mask[i, j]:
                continue
            val = float(bicoherence[i, j])
            if not np.isfinite(val):
                continue
            coupled.update({round(f1, 6), round(f2, 6), round(f3, 6)})
            triads.append((f1, f2, f3, val))
    return coupled, triads


# ---------------------------------------------------------------------------
# 6. Coupling power ratio R
# ---------------------------------------------------------------------------

def compute_coupling_power_ratio(
    psd: np.ndarray,
    psd_freqs: np.ndarray,
    coupled_freqs: Iterable[float],
    f_min: float = 0.0,
    f_max: float | None = None,
) -> Tuple[float, float, float]:
    """Compute R = P(coupled freqs) / P(all freqs in band).

    Total power is integrated over the same ``[f_min, f_max]`` band
    (defaults: full PSD) so that the ratio is meaningful for the analysis
    band used in bicoherence (typically 5-30 Hz here).

    Powers at coupled frequencies are obtained by linear interpolation of
    the PSD because Welch's frequency grid is generally finer than the
    bicoherence grid.
    """
    psd_freqs = np.asarray(psd_freqs)
    psd = np.asarray(psd)
    if f_max is None:
        f_max = float(psd_freqs[-1])

    band = (psd_freqs >= f_min) & (psd_freqs <= f_max)
    p_total = float(np.sum(psd[band]))
    if p_total <= 0.0:
        return 0.0, 0.0, 0.0

    coupled_freqs = sorted({float(f) for f in coupled_freqs if f_min <= f <= f_max})
    if not coupled_freqs:
        return 0.0, 0.0, p_total

    # Match each coupled frequency to its nearest PSD bin (which it will
    # never align with exactly because the grids differ). Using nearest-bin
    # power keeps the ratio commensurate with the discrete sum used for
    # p_total above.
    p_coupled = 0.0
    used: Set[int] = set()
    for f in coupled_freqs:
        idx = int(np.argmin(np.abs(psd_freqs - f)))
        if idx in used:
            continue
        used.add(idx)
        p_coupled += float(psd[idx])
    return p_coupled / p_total, p_coupled, p_total


# ---------------------------------------------------------------------------
# 7. Biphase analysis
# ---------------------------------------------------------------------------

def _circular_mean_std(angles: np.ndarray) -> Tuple[float, float]:
    """Circular mean and circular std of angles in radians."""
    if len(angles) == 0:
        return float("nan"), float("nan")
    c = np.mean(np.cos(angles))
    s = np.mean(np.sin(angles))
    mean = math.atan2(s, c)
    R = math.sqrt(c * c + s * s)
    # Circular std (Mardia): sqrt(-2 ln R).
    std = math.sqrt(-2.0 * math.log(R)) if R > 0 else float("inf")
    return float(mean), float(std)


def compute_biphase(
    signal: np.ndarray,
    fs: float,
    triads: Sequence[Tuple[float, float, float, float]],
    seg_seconds: float = 4.0,
    overlap: float = 0.5,
) -> Dict[Tuple[float, float], Dict[str, float]]:
    """Estimate biphase ``psi = phi(f1) + phi(f2) - phi(f3)`` per triad.

    The seizure is split into overlapping windows; the FFT phase at each
    of the three frequencies is read per window and the circular mean /
    std of psi across windows is reported. ``consistency`` is a 0..1
    metric (1 = perfectly locked phase relationship).

    For seizures shorter than ``seg_seconds`` (or with no triads) the
    function returns an empty dict / NaNs.
    """
    out: Dict[Tuple[float, float], Dict[str, float]] = {}
    if len(triads) == 0:
        return out

    n = len(signal)
    seg_len = int(round(seg_seconds * fs))
    if seg_len < 16 or seg_len > n:
        # Not enough data for sub-segmentation -> single-window biphase only.
        seg_len = n
    step = max(1, int(round(seg_len * (1.0 - overlap))))
    starts = list(range(0, n - seg_len + 1, step)) or [0]

    # Pre-compute FFTs of each window.
    segs = [signal[s : s + seg_len] for s in starts]
    fft_freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    fft_phases = [np.angle(np.fft.rfft(seg)) for seg in segs]

    def _phase_at(phases: np.ndarray, f: float) -> float:
        idx = int(np.argmin(np.abs(fft_freqs - f)))
        return float(phases[idx])

    seen: Set[Tuple[float, float]] = set()
    for f1, f2, f3, _bicoh_val in triads:
        key = (round(f1, 4), round(f2, 4))
        if key in seen:
            continue
        seen.add(key)
        psis = np.array(
            [
                _phase_at(ph, f1) + _phase_at(ph, f2) - _phase_at(ph, f3)
                for ph in fft_phases
            ]
        )
        # Wrap to [-pi, pi].
        psis = (psis + math.pi) % (2.0 * math.pi) - math.pi
        mean, std = _circular_mean_std(psis)
        consistency = max(0.0, 1.0 - std / math.pi) if math.isfinite(std) else 0.0
        out[key] = {
            "mean_biphase": mean,
            "circular_std": std,
            "consistency": consistency,
            "n_windows": len(psis),
        }
    return out


# ---------------------------------------------------------------------------
# 8. Harmonic vs cross-frequency classification
# ---------------------------------------------------------------------------

def classify_coupling_type(f1: float, f2: float, harmonic_tolerance: float = 0.5) -> str:
    """Return ``"harmonic"`` for diagonal triads (f1 ~= f2), else ``"cross-frequency"``."""
    return "harmonic" if abs(float(f1) - float(f2)) <= harmonic_tolerance else "cross-frequency"


# ---------------------------------------------------------------------------
# 9. Sanity check (WhitePaper section 6.3)
# ---------------------------------------------------------------------------

def _interp_power(psd: np.ndarray, psd_freqs: np.ndarray, f: float) -> float:
    psd_freqs = np.asarray(psd_freqs)
    if f <= psd_freqs[0] or f >= psd_freqs[-1]:
        idx = int(np.argmin(np.abs(psd_freqs - f)))
        return float(psd[idx])
    return float(np.interp(f, psd_freqs, psd))


def sanity_check_coupling(
    f1: float,
    f2: float,
    f3: float,
    psd: np.ndarray,
    psd_freqs: np.ndarray,
    bicoherence_value: float,
    bicoherence_high: float = 0.3,
    power_floor_ratio: float = 0.05,
) -> Dict[str, object]:
    """Apply WhitePaper section 6.3 sanity checks to one significant triad."""
    warnings_: List[str] = []
    p1 = _interp_power(psd, psd_freqs, f1)
    p2 = _interp_power(psd, psd_freqs, f2)
    p3 = _interp_power(psd, psd_freqs, f3)
    p_max = max(p1, p2, p3, 1e-30)
    floor = p_max * power_floor_ratio

    if p1 < floor:
        warnings_.append(f"low_power_at_f1({f1:.2f}Hz)")
    if p2 < floor:
        warnings_.append(f"low_power_at_f2({f2:.2f}Hz)")
    if p3 < floor and bicoherence_value > bicoherence_high:
        warnings_.append("ARTIFACT_high_bicoh_no_sum_power")
    if p3 > p_max * 0.5 and bicoherence_value < bicoherence_high:
        warnings_.append("INDEPENDENT_oscillator_at_f3")

    return {
        "passed": len(warnings_) == 0,
        "warnings": warnings_,
        "power_f1": p1,
        "power_f2": p2,
        "power_f3": p3,
    }
