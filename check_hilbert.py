"""
Self-check script: Compare PyTorch Hilbert (from model.pytorch_hilbert) with SciPy's scipy.signal.hilbert.

Usage:
  python Repos/pytorchSINDySz/check_hilbert.py --length 1024 --batch 4 --tol 1e-6 [--plot] [--save plot.png]

Behavior:
  - Builds a real-valued test signal with shape (batch, length), then transposes to (length, batch)
    to validate transforms along axis=0 on both implementations.
  - Compares complex analytic signals and reports max absolute error for complex, real, and imaginary parts.
    Exits with nonzero status if any exceeds tolerance.
  - With --plot, computes and overlays the amplitude envelopes (|analytic signal|) from both methods for
    the first column (batch element 0) as a function of time along axis=0.

Recommended tolerances:
  - float32 CPU: --tol 1e-6 (default)
  - float64 CPU: --tol 1e-9
  - If you adapt this to CUDA, consider relaxing by ~10x due to FFT numeric differences.
"""

import argparse
import sys
import numpy as np
import torch
from scipy.signal import hilbert as scipy_hilbert
import matplotlib.pyplot as plt

# Import the PyTorch implementation from this repo
from model import pytorch_hilbert  # type: ignore


def make_test_signal(length: int, batch: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, steps=length)
    # Create a mixture of sinusoids and some noise to better exercise the transform
    sig1 = torch.sin(2 * torch.pi * 5 * t)
    sig2 = 0.5 * torch.cos(2 * torch.pi * 11.3 * t + 0.3)
    base = sig1 + sig2
    noise = 0.05 * torch.randn(length)
    x1d = base + noise
    if batch == 1:
        return x1d.unsqueeze(0)  # shape (1, N)
    # Create batched variants with slight frequency offsets
    xs = []
    for b in range(batch):
        f = 5 + 0.1 * b
        g = 11.3 + 0.07 * b
        sigb = torch.sin(2 * torch.pi * f * t) + 0.5 * torch.cos(2 * torch.pi * g * t + 0.3)
        xs.append(sigb + 0.05 * torch.randn(length))
    x = torch.stack(xs, dim=0)  # (B, N)
    return x


def to_numpy_complex(x: torch.Tensor) -> np.ndarray:
    # torch.ifft returns complex dtype; ensure contiguous cpu tensor and convert to numpy complex128
    xc = x.detach().cpu().numpy()
    # Upcast to complex128 for stable comparison regardless of torch default dtype
    return xc.astype(np.complex128, copy=False)


def run_check(length: int, batch: int, tol: float, do_plot: bool = False, save_path: str | None = None) -> int:
    # Build real input (B, N). We will test axis along dim=0 by transposing to (N, B)
    xb = make_test_signal(length=length, batch=batch)  # shape (B, N)
    x = xb.T  # shape (N, B), transform axis=0

    # Torch forward: test axis=0 explicitly on shape (N, B)
    xt = x.to(torch.get_default_dtype())
    torch_analytic = pytorch_hilbert(xt, axis=0)

    # SciPy reference: transform along axis=0 on the same (N, B) shaped array
    x_np = x.numpy().astype(np.float64, copy=False)
    scipy_analytic_np = scipy_hilbert(x_np, axis=0)

    # Align dtypes for comparison
    torch_analytic_np = to_numpy_complex(torch_analytic)
    scipy_analytic_np = scipy_analytic_np.astype(np.complex128, copy=False)

    # Compute error metrics
    complex_err = np.max(np.abs(torch_analytic_np - scipy_analytic_np))
    real_err = np.max(np.abs(torch_analytic_np.real - scipy_analytic_np.real))
    imag_err = np.max(np.abs(torch_analytic_np.imag - scipy_analytic_np.imag))

    print(f"Length={length} Batch={batch}")
    print(f"Max abs error: complex={complex_err:.3e} real={real_err:.3e} imag={imag_err:.3e}")

    # Optional plotting of signal and envelopes for first batch element
    if do_plot:
        # After transpose, dim0 is time of length N. Choose first column (batch element 0) across time
        col = 0
        time = np.linspace(0.0, 1.0, num=length)
        x_np_plot = x[:, col].numpy()
        env_torch = np.abs(torch_analytic_np[:, col])
        env_scipy = np.abs(scipy_analytic_np[:, col])

        plt.figure(figsize=(10, 5))
        plt.plot(time, x_np_plot, label="Signal", color="#444444", linewidth=1.0)
        plt.plot(time, env_torch, label="Envelope (PyTorch)", color="#1f77b4", linewidth=1.5)
        plt.plot(time, env_scipy, label="Envelope (SciPy)", color="#ff7f0e", linestyle="--", linewidth=1.5)
        plt.title("Hilbert Envelope Comparison (axis=0, column 0)")
        plt.xlabel("Time (normalized)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    if not np.isfinite(complex_err):
        print("Non-finite error encountered.")
        return 2

    if complex_err > tol or real_err > tol or imag_err > tol:
        print(f"FAIL: errors exceed tolerance {tol}")
        return 1

    print("PASS: PyTorch Hilbert matches SciPy within tolerance")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=1024, help="Signal length N")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (1 for 1D)")
    parser.add_argument("--tol", type=float, default=1e-6, help="Tolerance for max abs error")
    parser.add_argument("--plot", action="store_true", help="Plot original signal and both envelopes for batch[0]")
    parser.add_argument("--save", type=str, default=None, help="If provided, save plot to this path instead of showing")
    args = parser.parse_args(argv)
    return run_check(length=args.length, batch=args.batch, tol=args.tol, do_plot=args.plot, save_path=args.save)


if __name__ == "__main__":
    sys.exit(main())
