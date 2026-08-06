import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L

# Avoid shadowing torch.nn.functional imported as F.
import torch.nn.functional as F

from itertools import combinations_with_replacement
import numpy as np
from typing import Any


def _finite_summary(
    t: torch.Tensor, *, max_print: int = 8
) -> tuple[int, int, int, str, list[tuple[int, ...]], list[str]]:
    """Return (n_nan, n_posinf, n_neginf, finite_minmax_str, bad_idx, bad_vals_str)."""

    if t.numel() == 0:
        return 0, 0, 0, "finite_min=nan finite_max=nan", [], []

    if t.is_complex():
        real = t.real
        imag = t.imag
        fin = torch.isfinite(real) & torch.isfinite(imag)
        nan_mask = torch.isnan(real) | torch.isnan(imag)
        posinf_mask = torch.isposinf(real) | torch.isposinf(imag)
        neginf_mask = torch.isneginf(real) | torch.isneginf(imag)
    else:
        fin = torch.isfinite(t)
        nan_mask = torch.isnan(t)
        posinf_mask = torch.isposinf(t)
        neginf_mask = torch.isneginf(t)

    n_nan = int(nan_mask.sum().item())
    n_posinf = int(posinf_mask.sum().item())
    n_neginf = int(neginf_mask.sum().item())

    finite_minmax_str = "finite_min=nan finite_max=nan"
    if bool(fin.any().item()):
        if t.is_complex():
            # report min/max over magnitude for complex
            mag = torch.abs(t)
            mag_f = mag[fin]
            finite_minmax_str = f"finite_min={mag_f.min().item():.6g} finite_max={mag_f.max().item():.6g}"
        else:
            tf = t[fin]
            finite_minmax_str = (
                f"finite_min={tf.min().item():.6g} finite_max={tf.max().item():.6g}"
            )

    bad_mask = ~fin
    bad_idx_t = bad_mask.nonzero(as_tuple=False)
    if bad_idx_t.numel() == 0:
        return n_nan, n_posinf, n_neginf, finite_minmax_str, [], []

    k = min(max_print, bad_idx_t.shape[0])
    bad_idx = [tuple(int(x) for x in row.tolist()) for row in bad_idx_t[:k]]

    # Extract a small sample of values; format for readability.
    bad_vals = []
    for idx in bad_idx:
        v = t[idx]
        if torch.is_complex(v):
            bad_vals.append(f"({v.real.item():.6g}+{v.imag.item():.6g}j)")
        else:
            bad_vals.append(f"{v.item()}")

    return n_nan, n_posinf, n_neginf, finite_minmax_str, bad_idx, bad_vals


def check_finite(
    t: Any,
    name: str,
    *,
    max_print: int = 8,
) -> None:
    """Raise FloatingPointError if tensor contains NaN/Inf.

    Intended for pinpointing where non-finite values first appear.
    """

    if not isinstance(t, torch.Tensor):
        return
    if t.numel() == 0:
        return

    if t.is_complex():
        ok = bool((torch.isfinite(t.real) & torch.isfinite(t.imag)).all().item())
    else:
        ok = bool(torch.isfinite(t).all().item())
    if ok:
        return

    n_nan, n_posinf, n_neginf, finite_minmax_str, bad_idx, bad_vals = _finite_summary(
        t, max_print=max_print
    )
    raise FloatingPointError(
        "non-finite detected in "
        f"{name} shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"nan={n_nan} +inf={n_posinf} -inf={n_neginf} {finite_minmax_str} "
        f"sample_idx={bad_idx} sample_val={bad_vals}"
    )


def check_module_params_finite(module: nn.Module, name: str) -> None:
    """Raise FloatingPointError if any module parameter contains NaN/Inf."""

    for pname, p in module.named_parameters(recurse=True):
        if p is None:
            continue
        check_finite(p.data, f"{name}/param:{pname}")


def equal_var_init(model: nn.Module) -> None:
    """Equal-variance init for Linear/GRU-style params.

    - Biases -> 0
    - Everything else -> Normal(0, 1/sqrt(fan_in))
    """

    import math

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or ".gru.bias" in name:
            param.data.fill_(0)
        else:
            # Expect weight-like tensors to be [..., fan_in]
            fan_in = int(param.shape[-1])
            param.data.normal_(std=1.0 / math.sqrt(fan_in))


class FANLayer(nn.Module):
    """FAN layer from https://arxiv.org/abs/2410.02675.

    Splits outputs into (cos(p), sin(p), g) where p is a linear projection and
    g is an activated linear projection.
    """

    def __init__(
        self, input_dim, output_dim, p_ratio=0.45, activation="gelu", use_p_bias=True
    ):
        super().__init__()
        if not (0.0 < p_ratio < 0.5):
            raise ValueError("p_ratio must be between 0 and 0.5")

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2

        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else (lambda x: x)

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        return torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)




class ShallowFANGRUEncoder(nn.Module):
    """Encoder that replaces the MLP bottleneck with a GRU.

    Original: Win -> FAN -> fc1 -> ReLU -> fc2 -> ReLU
    New:      Win -> FAN -> GRU -> Linear(proj to bottleneck)
    """

    def __init__(
        self,
        input_dim: int,
        p_ratio: float = 0.45,
        use_p_bias: bool = True,
        *,
        gru_hidden_dim: int | None = None,
        gru_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = self.input_dim // 10

        self.Win = nn.Linear(self.input_dim, self.input_dim)
        self.fan_layer1 = FANLayer(
            self.input_dim, self.input_dim, p_ratio=p_ratio, use_p_bias=use_p_bias
        )

        if gru_hidden_dim is None:
            gru_hidden_dim = max(1, self.input_dim // 5)
        self.gru_hidden_dim = int(gru_hidden_dim)

        self.gru = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=int(gru_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

        gru_out_dim = self.gru_hidden_dim * (2 if bidirectional else 1)
        self.proj = nn.Linear(gru_out_dim, self.bottleneck_dim)
        self.activate = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out = self.Win(x)
        out = self.fan_layer1(out)
        out, _ = self.gru(out)
        out = self.proj(out)
        out = self.activate(out)
        return out



class ShallowFANGRUDecoder(nn.Module):
    """Decoder that replaces the MLP expander with a GRU.

    Original: fc3 -> ReLU -> fc4 -> ReLU -> FAN -> Wout
    New:      Linear(up-proj) -> ReLU -> GRU -> FAN -> Wout
    """

    def __init__(
        self,
        output_dim: int,
        p_ratio: float = 0.45,
        use_p_bias: bool = True,
        *,
        gru_hidden_dim: int | None = None,
        gru_layers: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        in_dim = self.output_dim // 10

        if gru_hidden_dim is None:
            gru_hidden_dim = max(1, self.output_dim // 5)
        self.gru_hidden_dim = int(gru_hidden_dim)

        # First expand from bottleneck to a reasonable working width.
        self.fc_in = nn.Linear(in_dim, self.output_dim)
        self.activate = nn.ReLU()

        self.gru = nn.GRU(
            input_size=self.output_dim,
            hidden_size=self.gru_hidden_dim,
            num_layers=int(gru_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
        )

        gru_out_dim = self.gru_hidden_dim * (2 if bidirectional else 1)
        self.fc_out = nn.Linear(gru_out_dim, self.output_dim)

        self.fan_layer2 = FANLayer(
            self.output_dim, self.output_dim, p_ratio=p_ratio, use_p_bias=use_p_bias
        )
        self.Wout = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, bottleneck]
        out = self.fc_in(z)
        out = self.activate(out)
        out, _ = self.gru(out)
        out = self.fc_out(out)
        out = self.activate(out)
        out = self.fan_layer2(out)
        out = self.Wout(out)
        return out




#
# Convolutional (masked) encoder / decoder wrappers
#
#
# These adapt the ``FullResAutoencoder`` (a 2D masked convolutional autoencoder)
# to the SINDy pipeline's sequence contract and to the single-argument function
# signature required by ``torch.autograd.functional.jacobian``.
#
# Contract:
#   - Encoder ``forward(x)``: x is a batched *sequence of maps* with shape
#     ``[B, T, 1, H, W]`` and returns latent ``z`` of shape ``[B, T, d]``.
#   - Decoder ``forward(z)``: z is ``[B, T, d]`` and returns reconstructed maps
#     ``x_hat`` of shape ``[B, T, 1, H, W]``.
#
# The valid-region mask is *not* passed as an argument (which would make the
# Jacobian try to differentiate w.r.t. it). Instead the current batch's mask is
# stored on the module via ``set_mask`` before each forward pass. The mask is a
# plain (non-grad) attribute expanded to the flattened ``[N, 1, H, W]`` batch at
# call time. This keeps ``jacobian(fn, x)`` single-argument.


class ConvSINDyEncoder(nn.Module):
    """Sequence-aware, mask-on-module convolutional encoder.

    Wraps a :class:`FullResAutoencoder` and exposes ``forward(x)`` mapping
    ``[B, T, 1, H, W]`` -> ``[B, T, d]`` so it drops into the SINDy pipeline in
    place of the GRU/FAN encoders.

    Args:
        height: spatial height of the input maps.
        width: spatial width of the input maps.
        latent_dim: dimensionality of the latent space.
        ae: optional pre-built :class:`FullResAutoencoder` to reuse. When
            provided the same object should be passed to the paired
            :class:`ConvSINDyDecoder` so that encoder and decoder share weights
            and no parameters are left dead. If ``None`` a new instance is
            created (legacy behaviour, results in dead decoder-half weights).
    """

    def __init__(self, height: int, width: int, latent_dim: int, ae=None):
        super().__init__()
        from fullres_autoencoder import FullResAutoencoder

        self.height = int(height)
        self.width = int(width)
        self.latent_dim = int(latent_dim)
        if ae is not None:
            self.ae = ae
        else:
            self.ae = FullResAutoencoder(
                height=self.height, width=self.width, latent_dim=self.latent_dim
            )
        # Non-parameter mask storage. Registered as a buffer so it follows
        # ``.to(device)`` / dtype casts with the module.
        self.register_buffer(
            "_mask", torch.ones(1, 1, self.height, self.width), persistent=False
        )

    def set_mask(self, mask: torch.Tensor) -> None:
        """Store the valid-region mask for subsequent forward passes.

        Args:
            mask: tensor broadcastable to ``[N, 1, H, W]``. Common shapes are
                ``[1, 1, H, W]``, ``[B, 1, H, W]`` or ``[B, T, 1, H, W]``. The
                mask is flattened/reduced to ``[?, 1, H, W]`` and broadcast at
                call time.
        """
        m = mask.detach()
        # Collapse any leading dims down to a single batch dim of [*,1,H,W].
        m = m.reshape(-1, 1, self.height, self.width)
        self._mask = m.to(dtype=self.ae.latent_projection.weight.dtype)

    def _expand_mask(self, n: int, device, dtype) -> torch.Tensor:
        m = self._mask.to(device=device, dtype=dtype)
        if m.shape[0] == n:
            return m
        if m.shape[0] == 1:
            return m.expand(n, 1, self.height, self.width)
        # Fallback: if a per-sample mask of size B was provided but we now see
        # N = B*T flattened maps, tile it. Requires N divisible by m.shape[0].
        if n % m.shape[0] == 0:
            reps = n // m.shape[0]
            return m.repeat_interleave(reps, dim=0)
        raise ValueError(
            f"Cannot broadcast stored mask of batch {m.shape[0]} to {n} maps."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 1, H, W] (or with extra leading vmap dim during jacobian).
        # Flatten all leading dims except the trailing (1, H, W) into one batch.
        lead = x.shape[:-3]
        n = 1
        for s in lead:
            n *= int(s)
        x_flat = x.reshape(n, 1, self.height, self.width)
        mask = self._expand_mask(n, x_flat.device, x_flat.dtype)
        z = self.ae.encode(x_flat, mask)  # [n, d]
        return z.reshape(*lead, self.latent_dim)


class ConvSINDyDecoder(nn.Module):
    """Sequence-aware, mask-on-module convolutional decoder.

    Wraps a :class:`FullResAutoencoder` and exposes ``forward(z)`` mapping
    ``[B, T, d]`` -> ``[B, T, 1, H, W]``. The reconstruction is masked to the
    valid triangular region using the stored mask.

    Args:
        height: spatial height of the output maps.
        width: spatial width of the output maps.
        latent_dim: dimensionality of the latent space.
        ae: optional pre-built :class:`FullResAutoencoder` to reuse. Pass the
            same object that was given to the paired :class:`ConvSINDyEncoder`
            so that encoder and decoder share weights and all parameters
            participate in the computation graph. If ``None`` a new instance is
            created (legacy behaviour, results in dead encoder-half weights).
    """

    def __init__(self, height: int, width: int, latent_dim: int, ae=None):
        super().__init__()
        from fullres_autoencoder import FullResAutoencoder

        self.height = int(height)
        self.width = int(width)
        self.latent_dim = int(latent_dim)
        if ae is not None:
            self.ae = ae
        else:
            self.ae = FullResAutoencoder(
                height=self.height, width=self.width, latent_dim=self.latent_dim
            )
        self.register_buffer(
            "_mask", torch.ones(1, 1, self.height, self.width), persistent=False
        )

    def set_mask(self, mask: torch.Tensor) -> None:
        """Store the valid-region mask for subsequent forward passes."""
        m = mask.detach()
        m = m.reshape(-1, 1, self.height, self.width)
        self._mask = m.to(dtype=self.ae.latent_projection.weight.dtype)

    def _expand_mask(self, n: int, device, dtype) -> torch.Tensor:
        m = self._mask.to(device=device, dtype=dtype)
        if m.shape[0] == n:
            return m
        if m.shape[0] == 1:
            return m.expand(n, 1, self.height, self.width)
        if n % m.shape[0] == 0:
            reps = n // m.shape[0]
            return m.repeat_interleave(reps, dim=0)
        raise ValueError(
            f"Cannot broadcast stored mask of batch {m.shape[0]} to {n} maps."
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, d] (or with extra leading vmap dim during jacobian).
        lead = z.shape[:-1]
        n = 1
        for s in lead:
            n *= int(s)
        z_flat = z.reshape(n, self.latent_dim)
        x_hat = self.ae.decode(z_flat)  # [n, 1, H, W]
        mask = self._expand_mask(n, x_hat.device, x_hat.dtype)
        x_hat = x_hat * mask
        return x_hat.reshape(*lead, 1, self.height, self.width)


#
# Utility Functions for SINDy Fitting
#


def pytorch_hilbert(signal, axis=1):
    """Batch-aware Hilbert transform along the given axis (time).

    Args:
        signal (Tensor): arbitrary shape, batch-safe; expects time dimension at ``axis``.
        axis (int): axis along which to apply the transform (default: 1 for [B, T, ...]).
    Returns:
        Tensor: analytic signal with same shape as ``signal``.
    """
    N = signal.size(axis)
    signal_fft = torch.fft.fft(signal, dim=axis)

    H = signal.new_zeros(N)
    H[0] = 0  # DC component
    if N % 2 == 0:
        H[1 : N // 2] = 2  # Positive frequencies
        H[N // 2] = 0  # Nyquist frequency if N is even
    else:
        H[1 : (N + 1) // 2] = 2  # Positive frequencies

    # reshape for broadcasting on the target axis
    view_shape = [1] * signal.dim()
    view_shape[axis] = N
    H = H.view(*view_shape)

    analytic_signal_fft = signal_fft * H
    analytic_signal = torch.fft.ifft(analytic_signal_fft, dim=axis)
    return analytic_signal


def _debug_nan_guard(out: torch.Tensor, name: str, **context: torch.Tensor) -> None:
    """Inspect ``out`` for non-finite values and raise with diagnostics.

    Set a breakpoint on the ``raise`` line below to inspect ``out`` and any
    intermediate tensors passed via ``context`` (e.g. ``re``, ``im``, ``mag``)
    in the debugger right before the function would return non-finite values.
    """

    if not torch.isfinite(out).all():
        n_nan = int(torch.isnan(out).sum().item())
        n_posinf = int(torch.isposinf(out).sum().item())
        n_neginf = int(torch.isneginf(out).sum().item())
        ctx_summary = []
        for cname, ct in context.items():
            if not isinstance(ct, torch.Tensor):
                continue
            if ct.is_complex():
                fin = torch.isfinite(ct.real) & torch.isfinite(ct.imag)
            else:
                fin = torch.isfinite(ct)
            ctx_summary.append(
                f"{cname}: shape={tuple(ct.shape)} dtype={ct.dtype} "
                f"finite={int(fin.sum().item())}/{ct.numel()}"
            )
        ctx_str = " | ".join(ctx_summary) if ctx_summary else "(no context)"
        # <-- BREAKPOINT HERE: inspect `out` and **context tensors above.
        raise FloatingPointError(
            f"non-finite about to be returned from {name}: "
            f"shape={tuple(out.shape)} dtype={out.dtype} device={out.device} "
            f"nan={n_nan} +inf={n_posinf} -inf={n_neginf} :: {ctx_str}"
        )



def hilbert_features(x, eps: float = 1e-6):
    """Return (mag, cos_phase, sin_phase) along time axis 1."""
    a = pytorch_hilbert(x, axis=1)
    re, im = a.real, a.imag
    mag = torch.sqrt(re*re + im*im + eps*eps)   # eps inside sqrt → grad-safe
    cos_p = re / mag
    sin_p = im / mag
    return (mag, cos_p, sin_p)


def reshape_time_to_feature_blocks(
    x: torch.Tensor, time_dim: int = 500, block_size: int = 50
):
    """Reshape a [B, T] or [B, T, 1] time series to [B, T/block, block].

    Assumes T == ``time_dim`` and splits the time dimension into evenly sized
    blocks that become features at each (reduced) time step.
    """

    if x.dim() == 2:
        b, t = x.shape
        if t != time_dim:
            raise ValueError(f"Expected time_dim={time_dim} for reshape, got {t}")
        x = x.unsqueeze(-1)
    elif x.dim() == 3:
        b, t, f = x.shape
        if f != 1:
            raise ValueError(
                "reshape_time_to_feature_blocks expects last dim == 1 when x.dim()==3"
            )
        if t != time_dim:
            raise ValueError(f"Expected time_dim={time_dim} for reshape, got {t}")
    else:
        raise ValueError(f"Expected x with 2 or 3 dims, got {tuple(x.shape)}")

    if time_dim % block_size != 0:
        raise ValueError(
            f"time_dim={time_dim} must be divisible by block_size={block_size}"
        )

    new_time = time_dim // block_size
    return x.reshape(b, new_time, block_size)


#
# SINDy Model definitions
#


class SINDyModel(nn.Module):
    def __init__(
        self,
        time_dim,
        system_features,
        latent_features,
        poly_order,
        sindy_predict: nn.Module | None = None,
        nan_check: bool = False,
        nan_check_level: str = "basic",
    ):
        super(SINDyModel, self).__init__()
        """SINDy model operating on batched latent sequences.

        Expected input shape: [batch, time_dim, latent_features].

        This module is purely responsible for computing the sparse system
        output from latent encodings. Encoder/decoder orchestration and
        Jacobian computations live in the Lightning module `SINDySz`.

        `system_features` and `time_dim` are retained for metadata and
        documentation purposes; they are not used in the forward pass.
        """
        self.time_dim = time_dim
        self.system_features = system_features
        self.latent_features = latent_features
        self.poly_order = poly_order
        self.library_dim = self.compute_library_dim()

        self.nan_check = bool(nan_check)
        self.nan_check_level = str(nan_check_level).lower()
        if self.nan_check_level not in {"off", "basic", "full"}:
            raise ValueError(
                "nan_check_level must be one of: off, basic, full; "
                f"got {nan_check_level!r}"
            )

        self.SINDy_predict = (
            sindy_predict
            if sindy_predict is not None
            else nn.Linear(self.library_dim, latent_features)  # SINDy prediction layer
        )

    def compute_library_dim(self):
        self_features = self.latent_features
        hilbert_features = 3 * self.latent_features

        poly_features = 0
        for n in range(1, self.poly_order + 1):
            list_combinations = list(
                combinations_with_replacement(range(self.latent_features), n)
            )
            poly_features += len(list_combinations)

        return self_features + hilbert_features + poly_features

    def compute_library(self, z):
        """Build library features for batched latent states.

        Args:
            z (Tensor): shape [B, T, latent_features]
        Returns:
            Tensor: shape [B, T, library_dim]
        """
        B, T, lat_dim = z.shape
        L = lat_dim
        library = []

        if self.nan_check and self.nan_check_level != "off":
            check_finite(z, "compute_library/z")

        latent_indices = range(L)

        # Polynomial features over latent dimension per time step
        for n in range(1, self.poly_order + 1):
            list_combinations = list(combinations_with_replacement(latent_indices, n))
            for combination in list_combinations:
                # z[..., combination] -> [B, T, n]; prod over last -> [B, T]
                feat = torch.prod(z[..., combination], dim=-1, keepdim=True)
                if self.nan_check and self.nan_check_level == "full":
                    check_finite(feat, "compute_library/poly_feat")
                library.append(feat)

        # Linear latent features
        library.append(z)
        if self.nan_check and self.nan_check_level == "full":
            check_finite(z, "compute_library/linear_z")

        # Hilbert-derived features (real/imag parts) along time axis
        mag, cos_p, sin_p = hilbert_features(z)
        library.append(mag)
        library.append(cos_p)
        library.append(sin_p)
        
        
        #Add append

        if self.nan_check and self.nan_check_level != "off":
            check_finite(library[-2], "compute_library/hilbert_real")
            check_finite(library[-1], "compute_library/hilbert_phase")

        theta = torch.cat(library, dim=-1)
        if theta.shape[-1] != self.library_dim:
            raise RuntimeError(
                f"library_dim mismatch: expected {self.library_dim}, got {theta.shape[-1]}"
            )
        if self.nan_check and self.nan_check_level != "off":
            check_finite(theta, "compute_library/theta")
        return theta

    def forward(self, z):
        """Forward pass over batched latent sequences.

        Args:
            z (Tensor): shape [B, T, latent_features]
        Returns:
            tuple: (y_hat, SINDy_weights)
                y_hat: predicted latent time-derivatives, [B, T, latent_features]
                SINDy_weights: coefficient matrix, [latent_features, library_dim]
        """
        if z.dim() != 3:
            raise ValueError(
                f"Expected z shape [B, T, latent_features]; got {tuple(z.shape)}"
            )
        if z.shape[-1] != self.latent_features:
            raise ValueError(
                f"Expected z last dim == latent_features ({self.latent_features}); "
                f"got {z.shape[-1]}"
            )

        # Hard fail on dtype mismatch rather than silently casting:
        # `.to(dtype)` produces a leaf tensor that does not preserve
        # `requires_grad`, which would silently break the autograd path
        # the loss relies on (Jacobian/x_dot/z_dot computations).
        # Callers (e.g. `SINDySz.forward`) are responsible for casting `x`
        # before encoding; that cast propagates to `z` naturally.
        param_dtype = next(self.parameters()).dtype
        if z.dtype != param_dtype:
            raise TypeError(
                f"SINDyModel.forward: z dtype {z.dtype} != param dtype "
                f"{param_dtype}. Cast the input upstream (before encoding) "
                "rather than relying on an implicit cast here."
            )

        if self.nan_check and self.nan_check_level != "off":
            check_finite(z, "sindy_model.forward/z")

        theta_z = self.compute_library(z)  # [B, T, library_dim]
        if self.nan_check and self.nan_check_level != "off":
            check_finite(theta_z, "sindy_model.forward/theta_z")

        y_hat = self.SINDy_predict(theta_z)  # [B, T, latent_features]
        if self.nan_check and self.nan_check_level != "off":
            check_finite(y_hat, "sindy_model.forward/y_hat")
            check_finite(
                self.SINDy_predict.weight, "sindy_model.forward/SINDy_predict.weight"
            )

        return y_hat, self.SINDy_predict.weight





def _apply_finite_difference_batch(
    x: torch.Tensor,
    z: torch.Tensor,
    *,
    dt: float | None = None,
    fs: float | None = None,
    time_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-aware first derivatives via finite differences.

    Shared helper used by ``SINDyLoss`` and ``SINDyPathLoss``. Computes
    derivatives along the time axis using forward/backward differences at the
    boundaries and central differences in the interior.

    Args:
        x: Tensor [B, T, F]
        z: Tensor [B, T, L]
        dt: time step in seconds (preferred)
        fs: sampling frequency in Hz (used if dt is None)
        time_dim: time dimension (default 1)

    Returns:
        (x_dot, z_dot) with shapes [B, T, F] and [B, T, L]
    """

    if dt is None:
        if fs is None:
            dt = 1.0
        else:
            dt = 1.0 / float(fs)
    else:
        dt = float(dt)

    if x.dim() != 3 or z.dim() != 3:
        raise ValueError(
            f"Expected x [B,T,F] and z [B,T,L]; got x={tuple(x.shape)} z={tuple(z.shape)}"
        )
    if x.shape[0] != z.shape[0] or x.shape[1] != z.shape[1]:
        raise ValueError(
            f"Batch/time dims must match; got x={tuple(x.shape)} z={tuple(z.shape)}"
        )

    T = int(x.shape[time_dim])
    if T < 2:
        raise ValueError("Need at least two time steps for finite differences")

    def fd(t: torch.Tensor) -> torch.Tensor:
        # t: [B, T, C] (with time_dim==1) -> out same shape
        if time_dim != 1:
            t = t.transpose(time_dim, 1)

        out = torch.empty_like(t)
        out[:, 0, :] = (t[:, 1, :] - t[:, 0, :]) / dt
        out[:, -1, :] = (t[:, -1, :] - t[:, -2, :]) / dt
        out[:, 1:-1, :] = (t[:, 2:, :] - t[:, :-2, :]) / (2.0 * dt)

        if time_dim != 1:
            out = out.transpose(time_dim, 1)
        return out

    return fd(x), fd(z)


# NOTE: Used in single-optimizer mode. Computes all four loss components and
# returns their sum. For dual-optimizer mode see ``SINDyPathLoss`` and
# ``DecoderPathLoss`` below.
class SINDyLoss(nn.Module):
    def __init__(self, *, nan_check: bool = False, sample_rate: float = 100.0):
        super(SINDyLoss, self).__init__()
        # Lambda weights for UNNORMALIZED MSE losses.
        # Variance normalization removed to prevent explosion when derivatives
        # have near-zero variance (valid when latent codes change slowly).
        # MODIFIED: Increased lambda3 from 2.0 to 20.0 to give more weight to
        # learning accurate latent dynamics (fixing flat trajectory issue).
        # Decreased lambda4 from 0.01 to 0.001 to reduce over-regularization.
        self.lambda1 = 0.75    # reconstruction weight
        self.lambda2 = 50.0   # xdot weight (scale ~0.002)
        self.lambda3 = 40.0   # zdot weight (increased from 2.0)
        self.lambda4 = 0.01  # regularization (decreased from 0.01)
        self.nan_check = bool(nan_check)
        self.sample_rate = float(sample_rate)

    def apply_finite_difference_batch(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        *,
        dt: float | None = None,
        fs: float | None = None,
        time_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch-aware first derivatives via finite differences.

        Thin wrapper around the module-level ``_apply_finite_difference_batch``
        helper, kept as an instance method for backward compatibility.
        """
        return _apply_finite_difference_batch(
            x, z, dt=dt, fs=fs, time_dim=time_dim
        )

    def forward(self, x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights):
        """Batched SINDy loss with variance-normalized MSE terms.

        Each MSE term is divided by the variance of its target, making the
        loss components dimensionless and naturally O(1) (representing the
        "fraction of variance unexplained" = ``1 - R²``). The ``lambda*``
        parameters therefore act as **priority weights** rather than scale
        corrections. The L1 regularization term is not normalized.

        Normalization targets (denominators):
            - ``recon_loss``: ``var(x)`` (the input signal).
            - ``sindy_loss_xdot``: ``var(x_dot)`` (finite-difference derivative
              of ``x``).
            - ``sindy_loss_zdot``: ``var(z_dot)`` (finite-difference derivative
              of the latent ``z``). This is the *target* of the SINDy z-dot
              prediction; normalizing by the prediction ``var(y_hat)`` instead
              would create an unstable optimizer-controllable denominator.

        Args:
            x: [B, T, F]
            y_hat: [B, T, L]
            x_hat: [B, T, F]
            z: [B, T, L]
            jac_z_x: [B, T, L, F]
            jac_x_z: [B, T, F, L]
            SINDy_weights: [L, library_dim]
        Returns:
            Tuple ``(total_loss, recon_loss, sindy_loss_xdot,
            sindy_loss_zdot, sindy_regularization, diagnostics)`` where
            ``diagnostics`` is a dict of Python scalars containing target
            variances, unnormalized MSE values, and per-term R² values.
        """

        if self.nan_check:
            check_finite(x, "loss/x")
            check_finite(y_hat, "loss/y_hat")
            check_finite(x_hat, "loss/x_hat")
            check_finite(z, "loss/z")
            check_finite(jac_z_x, "loss/jac_z_x")
            check_finite(jac_x_z, "loss/jac_x_z")
            check_finite(SINDy_weights, "loss/SINDy_weights")

        # Finite differences along time dimension (no trimming needed)
        x_dot, z_dot = self.apply_finite_difference_batch(
            x, z, time_dim=1, fs=self.sample_rate
        )  # [B, T, F], [B, T, L]
        y_hat_trim = y_hat
        jac_trim = jac_z_x
        jac_xz_trim = jac_x_z

        if self.nan_check:
            check_finite(x_dot, "loss/x_dot")
            check_finite(z_dot, "loss/z_dot")
            check_finite(y_hat_trim, "loss/y_hat_trim")
            check_finite(jac_trim, "loss/jac_trim")
            check_finite(jac_xz_trim, "loss/jac_xz_trim")

        # Predicted x_dot from y_hat via decoder Jacobian
        x_dot_pred = torch.einsum("btfl,btl->btf", jac_xz_trim, y_hat_trim)
        if self.nan_check:
            check_finite(x_dot_pred, "loss/x_dot_pred")

        # z_dot predicted via autograd Jacobian * x_dot
        z_dot_pred = torch.einsum("btlf,btf->btl", jac_trim, x_dot)
        if self.nan_check:
            check_finite(z_dot_pred, "loss/z_dot_pred")

        # Compute variances for diagnostics/monitoring only (not used in loss).
        x_var = x.detach().var()
        x_dot_var = x_dot.detach().var()
        z_dot_var = z_dot.detach().var()

        if self.nan_check:
            check_finite(x_var, "loss/x_var")
            check_finite(x_dot_var, "loss/x_dot_var")
            check_finite(z_dot_var, "loss/z_dot_var")

        # Compute UNNORMALIZED MSE losses (no variance normalization).
        # Variance normalization removed because it causes explosion when
        # derivatives have near-zero variance. Lambda weights balance terms.
        recon_mse = F.mse_loss(x, x_hat)
        xdot_mse = F.mse_loss(x_dot, x_dot_pred)
        zdot_mse = F.mse_loss(z_dot_pred, y_hat_trim)

        recon_loss = self.lambda1 * recon_mse
        sindy_loss_xdot = self.lambda2 * xdot_mse
        sindy_loss_zdot = self.lambda3 * zdot_mse
        sindy_regularization = self.lambda4 * SINDy_weights.abs().sum()

        if self.nan_check:
            check_finite(recon_loss, "loss/recon_loss")
            check_finite(sindy_loss_xdot, "loss/sindy_loss_xdot")
            check_finite(sindy_loss_zdot, "loss/sindy_loss_zdot")
            check_finite(sindy_regularization, "loss/sindy_regularization")

        total_loss = (
            recon_loss + sindy_loss_xdot + sindy_loss_zdot + sindy_regularization
        )

        if self.nan_check:
            check_finite(total_loss, "loss/total_loss")

        diagnostics = {
            "x_var": x_var.item(),
            "x_dot_var": x_dot_var.item(),
            "z_dot_var": z_dot_var.item(),
            "y_hat_var": y_hat_trim.detach().var().item(),
            "recon_mse_unnorm": recon_mse.item(),
            "xdot_mse_unnorm": xdot_mse.item(),
            "zdot_mse_unnorm": zdot_mse.item(),
            # R² computed from unnormalized MSE relative to target variance.
            # When variance is near-zero, R² is not meaningful, but we compute
            # it anyway for monitoring. Clamp variance to avoid division by zero.
            "R2_recon": (
                1.0 - recon_mse.item() / max(x_var.item(), 1e-9)
            ),
            "R2_xdot": (
                1.0 - xdot_mse.item() / max(x_dot_var.item(), 1e-9)
            ),
            "R2_zdot": (
                1.0 - zdot_mse.item() / max(z_dot_var.item(), 1e-9)
            ),
        }

        return (
            total_loss,
            recon_loss,
            sindy_loss_xdot,
            sindy_loss_zdot,
            sindy_regularization,
            diagnostics,
        )


class SINDyPathLoss(nn.Module):
    """Loss for the SINDy optimization path (encoder + sindy_model).

    Computes the three SINDy-related loss components with variance-normalized
    MSE terms (dimensionless, naturally O(1)):
      - ``sindy_loss_xdot`` (λ2): MSE between ``x_dot`` (finite-difference)
        and ``x_dot_pred`` produced by mapping ``y_hat`` through the decoder
        Jacobian, divided by ``var(x_dot)``.
      - ``sindy_loss_zdot`` (λ3): MSE between ``z_dot_pred`` (encoder Jacobian
        times ``x_dot``) and ``y_hat``, divided by ``var(z_dot)``. The
        denominator is the *target* variance (signal-determined) rather than
        the prediction variance, which prevents the optimizer from minimizing
        the loss by inflating ``y_hat`` magnitude.
      - ``sindy_regularization`` (λ4): L1 penalty on the SINDy weight matrix
        (not normalized).

    With normalization, the ``lambda*`` parameters act as **priority weights**
    rather than scale corrections.
    """

    def __init__(self, *, nan_check: bool = False, sample_rate: float = 100.0):
        super(SINDyPathLoss, self).__init__()
        # Lambda weights for UNNORMALIZED MSE losses.
        # Variance normalization removed because it causes explosion when
        # z_dot has near-zero variance (latent codes change slowly over 3s).
        # Weights tuned based on typical unnormalized MSE scales:
        #   xdot_mse ~0.002, zdot_mse ~0.05
        # MODIFIED: Increased lambda3 from 2.0 to 20.0 to give more weight to
        # learning accurate latent dynamics (fixing flat trajectory issue).
        # Decreased lambda4 from 0.01 to 0.001 to reduce over-regularization.
        self.lambda2 = 50.0   # xdot weight
        self.lambda3 = 20.0   # zdot weight (increased from 2.0)
        self.lambda4 = 0.001  # regularization (decreased from 0.01)
        self.nan_check = bool(nan_check)
        self.sample_rate = float(sample_rate)

    def forward(self, x, y_hat, z, jac_z_x, jac_x_z, SINDy_weights):
        """Batched SINDy-path loss with variance-normalized MSE terms.

        Args:
            x: [B, T, F]
            y_hat: [B, T, L]
            z: [B, T, L]
            jac_z_x: [B, T, L, F]
            jac_x_z: [B, T, F, L]
            SINDy_weights: [L, library_dim]

        Returns:
            Tuple ``(sindy_path_loss, sindy_loss_xdot, sindy_loss_zdot,
            sindy_regularization, diagnostics)`` where ``diagnostics`` is a
            dict of Python scalars containing target variances, unnormalized
            MSE values, and per-term R² values.
        """

        if self.nan_check:
            check_finite(x, "sindy_path_loss/x")
            check_finite(y_hat, "sindy_path_loss/y_hat")
            check_finite(z, "sindy_path_loss/z")
            check_finite(jac_z_x, "sindy_path_loss/jac_z_x")
            check_finite(jac_x_z, "sindy_path_loss/jac_x_z")
            check_finite(SINDy_weights, "sindy_path_loss/SINDy_weights")

        x_dot, z_dot = _apply_finite_difference_batch(
            x, z, time_dim=1, fs=self.sample_rate
        )  # [B, T, F], [B, T, L]
        y_hat_trim = y_hat
        jac_trim = jac_z_x
        jac_xz_trim = jac_x_z

        if self.nan_check:
            check_finite(x_dot, "sindy_path_loss/x_dot")
            check_finite(z_dot, "sindy_path_loss/z_dot")

        # Predicted x_dot from y_hat via decoder Jacobian
        x_dot_pred = torch.einsum("btfl,btl->btf", jac_xz_trim, y_hat_trim)
        if self.nan_check:
            check_finite(x_dot_pred, "sindy_path_loss/x_dot_pred")

        # z_dot predicted via encoder Jacobian * x_dot
        z_dot_pred = torch.einsum("btlf,btf->btl", jac_trim, x_dot)
        if self.nan_check:
            check_finite(z_dot_pred, "sindy_path_loss/z_dot_pred")

        # Compute variances for diagnostics/monitoring only (not used in loss).
        x_dot_var = x_dot.detach().var()
        z_dot_var = z_dot.detach().var()

        if self.nan_check:
            check_finite(x_dot_var, "sindy_path_loss/x_dot_var")
            check_finite(z_dot_var, "sindy_path_loss/z_dot_var")

        # Compute UNNORMALIZED MSE losses (no variance normalization).
        # Variance normalization removed because it causes explosion when
        # derivatives have near-zero variance (which is valid when latent
        # codes change slowly). Lambda weights are tuned to balance the terms.
        xdot_mse = F.mse_loss(x_dot, x_dot_pred)
        zdot_mse = F.mse_loss(z_dot_pred, y_hat_trim)

        sindy_loss_xdot = self.lambda2 * xdot_mse
        sindy_loss_zdot = self.lambda3 * zdot_mse
        sindy_regularization = self.lambda4 * SINDy_weights.abs().sum()

        if self.nan_check:
            check_finite(sindy_loss_xdot, "sindy_path_loss/sindy_loss_xdot")
            check_finite(sindy_loss_zdot, "sindy_path_loss/sindy_loss_zdot")
            check_finite(sindy_regularization, "sindy_path_loss/sindy_regularization")

        sindy_path_loss = sindy_loss_xdot + sindy_loss_zdot + sindy_regularization

        if self.nan_check:
            check_finite(sindy_path_loss, "sindy_path_loss/total")

        diagnostics = {
            "x_dot_var": x_dot_var.item(),
            "z_dot_var": z_dot_var.item(),
            "y_hat_var": y_hat_trim.detach().var().item(),
            "xdot_mse_unnorm": xdot_mse.item(),
            "zdot_mse_unnorm": zdot_mse.item(),
            # R² computed from unnormalized MSE relative to target variance.
            # When variance is near-zero, R² is not meaningful, but we compute
            # it anyway for monitoring. Clamp variance to avoid division by zero.
            "R2_xdot": (
                1.0 - xdot_mse.item() / max(x_dot_var.item(), 1e-9)
            ),
            "R2_zdot": (
                1.0 - zdot_mse.item() / max(z_dot_var.item(), 1e-9)
            ),
        }

        return (
            sindy_path_loss,
            sindy_loss_xdot,
            sindy_loss_zdot,
            sindy_regularization,
            diagnostics,
        )


class DecoderPathLoss(nn.Module):
    """Loss for the Decoder optimization path (encoder + decoder).

    Computes the variance-normalized reconstruction loss ``recon_loss`` (λ1):sindy_lr
    MSE between ``x`` and ``x_hat`` divided by ``var(x)``. The result is
    dimensionless and naturally O(1) (fraction of variance unexplained).
    With normalization ``lambda1`` acts as a priority weight rather than a
    scale correction.
    """

    def __init__(self, *, nan_check: bool = False):
        super(DecoderPathLoss, self).__init__()
        self.lambda1 = 1
        self.nan_check = bool(nan_check)

    def forward(self, x, x_hat):
        """Batched decoder-path loss with variance-normalized MSE.

        Args:
            x: [B, T, F]
            x_hat: [B, T, F]

        Returns:
            Tuple ``(decoder_path_loss, recon_loss, diagnostics)``.
            ``decoder_path_loss`` and ``recon_loss`` are equal (returned for
            symmetry with ``SINDyPathLoss``); ``diagnostics`` is a dict of
            Python scalars containing ``x_var``, ``recon_mse_unnorm``, and
            ``R2_recon``.
        """
        if self.nan_check:
            check_finite(x, "decoder_path_loss/x")
            check_finite(x_hat, "decoder_path_loss/x_hat")

        # Compute variance for diagnostics/monitoring only (not used in loss).
        x_var = x.detach().var()

        if self.nan_check:
            check_finite(x_var, "decoder_path_loss/x_var")

        # Compute UNNORMALIZED MSE loss (no variance normalization).
        recon_mse = F.mse_loss(x, x_hat)
        recon_loss = self.lambda1 * recon_mse

        if self.nan_check:
            check_finite(recon_loss, "decoder_path_loss/recon_loss")

        diagnostics = {
            "x_var": x_var.item(),
            "recon_mse_unnorm": recon_mse.item(),
            # R² computed from unnormalized MSE relative to target variance.
            "R2_recon": (
                1.0 - recon_mse.item() / max(x_var.item(), 1e-9)
            ),
        }

        return recon_loss, recon_loss, diagnostics


#
# PyTorch Lightning Module for SINDy Training
#


class SINDySz(L.LightningModule):
    def __init__(
        self,
        model: SINDyModel | None = None,
        *,
        time_dim: int | None = None,
        system_features: int | None = None,
        latent_features: int | None = None,
        poly_order: int | None = None,
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        sindy_predict: nn.Module | None = None,
        lr: float = 0.001,
        nan_check: bool = False,
        nan_check_level: str = "basic",
        reinit: bool = True,
        use_dual_optimizers: bool = False,
        sindy_lr: float | None = None,
        decoder_lr: float | None = None,
        sample_rate: float = 100.0,
    ):
        """Lightning module that orchestrates the encode/SINDy/decode pipeline.

        Args:
            model: optional pre-built SINDyModel. Legacy instances that carry
                their own ``encoder``/``decoder`` are accepted and rewritten
                into the new layout.
            time_dim, system_features, latent_features, poly_order:
                required when ``model`` is None.
            encoder, decoder: required if not extractable from ``model``.
            sindy_predict: optional override for the SINDy library->latent
                projection layer.
            lr: AdamW learning rate.
            nan_check, nan_check_level: see ``SINDyModel``.
            reinit: when True (default), apply ``equal_var_init`` to the
                encoder, decoder, and sindy_model after assembly. Set to
                False to preserve weights of caller-supplied modules (e.g.
                a pretrained encoder/decoder).
            use_dual_optimizers: when True, train with two separate optimizers
                (one for the SINDy path = encoder + sindy_model, one for the
                Decoder path = encoder + decoder). The encoder is shared and
                receives gradients from both optimizers. Requires manual
                optimization.
            sindy_lr: learning rate for the SINDy-path optimizer. Defaults to
                ``lr`` when ``None``. Only used when ``use_dual_optimizers``.
            decoder_lr: learning rate for the Decoder-path optimizer. Defaults
                to ``lr`` when ``None``. Only used when ``use_dual_optimizers``.
            sample_rate: sampling rate (Hz) of the input time series. Passed
                to the loss criteria so that finite-difference derivatives
                are computed with the correct time step (``dt = 1/sample_rate``).
        """
        super(SINDySz, self).__init__()

        # Manual optimization is required for the dual-optimizer training loop.
        # Set this before module assembly so Lightning sees it during fit().
        if use_dual_optimizers:
            self.automatic_optimization = False

        # Resolve encoder/decoder/sindy_model from inputs. Two supported paths:
        #   1) `model` (a SINDyModel) is provided, optionally with encoder/decoder
        #      attributes (legacy layout); also takes encoder/decoder kwargs.
        #   2) `model` is None: build SINDyModel from scalars; encoder/decoder must
        #      be supplied separately.
        if model is not None:
            # Legacy: model may carry its own encoder/decoder.
            legacy_encoder = getattr(model, "encoder", None)
            legacy_decoder = getattr(model, "decoder", None)
            resolved_encoder = encoder if encoder is not None else legacy_encoder
            resolved_decoder = decoder if decoder is not None else legacy_decoder

            # Rebuild a clean SINDyModel without encoder/decoder if the legacy
            # model had them attached. Otherwise reuse the provided model directly.
            if legacy_encoder is not None or legacy_decoder is not None:
                legacy_predict = getattr(model, "SINDy_predict", None)

                # Build the new SINDyModel; this also computes library_dim
                # from the scalar params we hand it.
                sindy_model = SINDyModel(
                    time_dim=getattr(model, "time_dim", time_dim),
                    system_features=getattr(model, "system_features", system_features),
                    latent_features=getattr(model, "latent_features", latent_features),
                    poly_order=getattr(model, "poly_order", poly_order),
                    sindy_predict=legacy_predict,
                    nan_check=getattr(model, "nan_check", nan_check),
                    nan_check_level=getattr(model, "nan_check_level", nan_check_level),
                )

                # Defensive: if the legacy SINDy_predict layer has an
                # `in_features` that disagrees with the freshly computed
                # library_dim, refuse — silently swapping a randomly
                # initialized layer in would corrupt training state.
                if legacy_predict is not None and hasattr(
                    legacy_predict, "in_features"
                ):
                    if int(legacy_predict.in_features) != int(sindy_model.library_dim):
                        raise ValueError(
                            "Legacy SINDy_predict.in_features="
                            f"{legacy_predict.in_features} does not match the "
                            f"computed library_dim={sindy_model.library_dim}. "
                            "Refusing to silently discard or rebind weights."
                        )

                if legacy_predict is None:
                    import warnings

                    warnings.warn(
                        "Legacy SINDyModel was passed without a `SINDy_predict` "
                        "layer; SINDySz built a freshly initialized one. Any "
                        "prior training state for the SINDy layer is lost.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                sindy_model = model
        else:
            missing = [
                name
                for name, val in (
                    ("time_dim", time_dim),
                    ("system_features", system_features),
                    ("latent_features", latent_features),
                    ("poly_order", poly_order),
                )
                if val is None
            ]
            if missing:
                raise TypeError(
                    "SINDySz requires either `model` or all of: "
                    "time_dim, system_features, latent_features, poly_order. "
                    f"Missing: {', '.join(missing)}"
                )
            sindy_model = SINDyModel(
                time_dim=time_dim,
                system_features=system_features,
                latent_features=latent_features,
                poly_order=poly_order,
                sindy_predict=sindy_predict,
                nan_check=nan_check,
                nan_check_level=nan_check_level,
            )
            resolved_encoder = encoder
            resolved_decoder = decoder

        missing_modules = [
            name
            for name, val in (
                ("encoder", resolved_encoder),
                ("decoder", resolved_decoder),
            )
            if val is None
        ]
        if missing_modules:
            raise TypeError(
                "SINDySz requires both `encoder` and `decoder` modules; "
                f"missing: {', '.join(missing_modules)}."
            )

        self.encoder = resolved_encoder
        self.decoder = resolved_decoder
        self.sindy_model = sindy_model
        self.lr = float(lr)

        # Convolutional-map mode: when the encoder/decoder are the masked conv
        # wrappers, the pipeline operates on sequences of 2D maps shaped
        # [B, T, 1, H, W] rather than feature vectors [B, T, F]. Detect this by
        # the presence of the ``set_mask`` hook so the forward pass and Jacobian
        # helpers can branch and flatten the pixel dims for the SINDy losses.
        self.conv_mode = hasattr(self.encoder, "set_mask") and hasattr(
            self.decoder, "set_mask"
        )
        # Height/width of the conv maps (used to flatten x/x_hat -> [B,T,H*W]
        # for the loss criteria). Populated from the encoder when in conv mode.
        self._map_h = getattr(self.encoder, "height", None)
        self._map_w = getattr(self.encoder, "width", None)

        # Configure loss criteria and per-path learning rates based on mode.
        self.sample_rate = float(sample_rate)
        self.use_dual_optimizers = bool(use_dual_optimizers)
        if self.use_dual_optimizers:
            self.sindy_criterion = SINDyPathLoss(
                nan_check=nan_check, sample_rate=self.sample_rate
            )
            self.decoder_criterion = DecoderPathLoss(nan_check=nan_check)
            self.sindy_lr = float(sindy_lr if sindy_lr is not None else lr)
            self.decoder_lr = float(decoder_lr if decoder_lr is not None else lr)
        else:
            self.criterion = SINDyLoss(
                nan_check=nan_check, sample_rate=self.sample_rate
            )
            self.sindy_lr = float(sindy_lr if sindy_lr is not None else lr)
            self.decoder_lr = float(decoder_lr if decoder_lr is not None else lr)

        # Initialize each component separately. NOTE: this re-initializes
        # caller-supplied encoder/decoder modules. Pass ``reinit=False`` if
        # you are providing pretrained weights you want to preserve.
        if reinit:
            equal_var_init(self.encoder)
            equal_var_init(self.decoder)
            equal_var_init(self.sindy_model)

        # Counter incremented by on_after_backward so the dual-optimizer path
        # can identify which of the two per-step backward passes just completed.
        # Reset to 0 at the start of each training_step.
        self._backward_pass = 0

    def compute_jacobian_z_wrt_x(self, x):
        """Compute per-example Jacobian ∂z/∂x for batched inputs via ``self.encoder``.

        Args:
            x (Tensor): shape [B, T, F] (feature mode) or [B, T, 1, H, W]
                (conv-map mode), with requires_grad=True
        Returns:
            Tensor: Jacobian of shape [B, T, latent_features, F]. In conv mode
                ``F = H*W`` (pixel dims are flattened).
        """
        nan_check = bool(getattr(self.sindy_model, "nan_check", False))
        nan_check_level = str(
            getattr(self.sindy_model, "nan_check_level", "off")
        ).lower()

        if nan_check and nan_check_level != "off":
            check_finite(x, "jac_z_x/x")

        def encoder_bt(x_in: torch.Tensor) -> torch.Tensor:
            # feature mode: [B, T, F] -> [B, T, L]
            # conv mode:    [B, T, 1, H, W] -> [B, T, L]
            return self.encoder(x_in)

        # Full Jacobian over batch+time to support sequence models (e.g. GRU)
        # and per-map conv encoders.
        #   feature mode -> jac shape [B, T, L, B, T, F]
        #   conv mode    -> jac shape [B, T, L, B, T, 1, H, W]
        # Disable cuDNN RNN here: `vectorize=True` uses vmap, and cuDNN's
        # `_cudnn_rnn_backward` has no batching rule. The native (non-cuDNN)
        # GRU backward does, so disabling cuDNN makes the jacobian work.
        with torch.backends.cudnn.flags(enabled=False):
            jac = torch.autograd.functional.jacobian(
                encoder_bt,
                x,
                vectorize=False,
                create_graph=False,
            )

        if nan_check and nan_check_level == "full":
            check_finite(jac, "jac_z_x/raw")

        if self.conv_mode:
            # jac: [B, T, L, B, T, 1, H, W]. Flatten input pixel dims (1,H,W)
            # into a single feature axis F=H*W, then select the per-(b,t)
            # block diagonal ∂z[b,t,:]/∂x[b,t,:].
            B, T, L = jac.shape[0], jac.shape[1], jac.shape[2]
            jac = jac.reshape(B, T, L, B, T, -1)  # [B,T,L,B,T,F]

        # Select the per-(b,t) block diagonal: ∂z[b,t,:] / ∂x[b,t,:]
        # First diagonal picks matching batch index -> [T, L, T, F, B]
        # Second diagonal picks matching time index  -> [L, F, B, T]
        # Permute to [B, T, L, F]
        jac_diag = jac.diagonal(dim1=0, dim2=3).diagonal(dim1=0, dim2=2)
        jac_btlf = jac_diag.permute(2, 3, 0, 1).contiguous()

        if nan_check and nan_check_level != "off":
            check_finite(jac_btlf, "jac_z_x/out")
        return jac_btlf

    def compute_jacobian_x_wrt_z(self, z):
        """Compute per-example Jacobian ∂x/∂z for batched latents via ``self.decoder``.

        Args:
            z (Tensor): shape [B, T, latent_features]
        Returns:
            Tensor: Jacobian of shape [B, T, system_features, latent_features].
                In conv mode ``system_features = H*W`` (pixel dims flattened).
        """
        nan_check = bool(getattr(self.sindy_model, "nan_check", False))
        nan_check_level = str(
            getattr(self.sindy_model, "nan_check_level", "off")
        ).lower()

        if nan_check and nan_check_level != "off":
            check_finite(z, "jac_x_z/z")

        def decoder_bt(z_in: torch.Tensor) -> torch.Tensor:
            # feature mode: [B, T, L] -> [B, T, F]
            # conv mode:    [B, T, L] -> [B, T, 1, H, W]
            return self.decoder(z_in)

        # Full Jacobian over batch+time to support sequence models (e.g. GRU)
        # and per-map conv decoders.
        #   feature mode -> jac shape [B, T, F, B, T, L]
        #   conv mode    -> jac shape [B, T, 1, H, W, B, T, L]
        # See note in `compute_jacobian_z_wrt_x`: cuDNN RNN backward has no
        # vmap batching rule, so disable cuDNN for the jacobian call.
        with torch.backends.cudnn.flags(enabled=False):
            jac = torch.autograd.functional.jacobian(
                decoder_bt,
                z,
                vectorize=False,
                create_graph=False,
            )

        if nan_check and nan_check_level == "full":
            check_finite(jac, "jac_x_z/raw")

        if self.conv_mode:
            # jac: [B, T, 1, H, W, B, T, L]. Flatten output pixel dims (1,H,W)
            # into a single feature axis F=H*W so the layout matches feature
            # mode ([B, T, F, B, T, L]) before the block-diagonal selection.
            B, T = jac.shape[0], jac.shape[1]
            L = jac.shape[-1]
            jac = jac.reshape(B, T, -1, B, T, L)  # [B,T,F,B,T,L]

        # Per-(b,t) block diagonal: ∂x[b,t,:] / ∂z[b,t,:]
        # After two diagonals, permute to [B, T, F, L]
        jac_diag = jac.diagonal(dim1=0, dim2=3).diagonal(dim1=0, dim2=2)
        jac_btfl = jac_diag.permute(2, 3, 0, 1).contiguous()

        if nan_check and nan_check_level != "off":
            check_finite(jac_btfl, "jac_x_z/out")
        return jac_btfl

    def apply_finite_difference(self, filtered_data, fs):
        """Compute first derivative via finite differences (NumPy).

        Supports:
        - 1D input: [T]
        - 2D batched input: [B, T]

        Uses forward/backward differences at boundaries and central differences
        in the interior. Output has the same shape as input.
        """

        filtered_data = np.asarray(filtered_data)
        if filtered_data.ndim not in (1, 2):
            raise ValueError(
                f"filtered_data must be 1D [T] or 2D [B,T]; got shape {filtered_data.shape}"
            )

        dt = 1.0 / float(fs)

        if filtered_data.shape[-1] < 2:
            raise ValueError("filtered_data must contain at least two samples")

        deriv = np.empty_like(filtered_data, dtype=float)

        if filtered_data.ndim == 1:
            deriv[0] = (filtered_data[1] - filtered_data[0]) / dt
            deriv[-1] = (filtered_data[-1] - filtered_data[-2]) / dt
            deriv[1:-1] = (filtered_data[2:] - filtered_data[:-2]) / (2.0 * dt)
            return deriv

        # filtered_data: [B, T]
        deriv[:, 0] = (filtered_data[:, 1] - filtered_data[:, 0]) / dt
        deriv[:, -1] = (filtered_data[:, -1] - filtered_data[:, -2]) / dt
        deriv[:, 1:-1] = (filtered_data[:, 2:] - filtered_data[:, :-2]) / (2.0 * dt)
        return deriv

    def forward(self, x, mask=None):
        """Orchestrate the full pipeline: encode -> SINDy -> decode (+ Jacobians).

        Args:
            x (Tensor): shape ``[B, T, system_features]`` in feature mode, or
                ``[B, T, 1, H, W]`` in conv-map mode.
            mask (Tensor, optional): valid-region mask for conv-map mode,
                broadcastable to ``[N, 1, H, W]``. Stored on the conv
                encoder/decoder before encode/decode so the autograd Jacobian
                remains single-argument. Ignored in feature mode.
        Returns:
            tuple: (y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights). In conv
                mode ``x_hat`` is returned flattened as ``[B, T, H*W]`` (the
                pixel dims collapsed) so it is directly consumable by the loss
                criteria; the Jacobians are likewise flattened over pixels.
        """
        if self.conv_mode:
            if x.dim() != 5:
                raise ValueError(
                    "conv-map mode expects x shape [B, T, 1, H, W]; "
                    f"got {tuple(x.shape)}"
                )
        elif x.dim() != 3:
            raise ValueError(
                f"Expected x shape [B, T, F]; got {tuple(x.shape)}"
            )

        nan_check = bool(getattr(self.sindy_model, "nan_check", False))
        nan_check_level = str(
            getattr(self.sindy_model, "nan_check_level", "off")
        ).lower()
        nan_active = nan_check and nan_check_level != "off"

        param_dtype = next(self.parameters()).dtype
        if x.dtype != param_dtype:
            x = x.to(param_dtype)

        # Conv mode: publish the current batch's mask to the encoder/decoder so
        # their single-arg forward (and thus the Jacobian) uses it consistently.
        if self.conv_mode and mask is not None:
            self.encoder.set_mask(mask)
            self.decoder.set_mask(mask)

        if nan_active:
            check_finite(x, "forward/x")
            # If x is finite but z becomes non-finite, parameters are a likely culprit.
            check_module_params_finite(self.encoder, "forward/encoder")

        x = x.requires_grad_(True)

        # Encode: [B, T, F] or [B, T, 1, H, W] -> [B, T, latent_features]
        z = self.encoder(x).requires_grad_(True)
        if nan_active:
            check_finite(z, "forward/z")

        # SINDy prediction: latent derivatives + sparse coefficient matrix
        y_hat, SINDy_weights = self.sindy_model(z)
        if nan_active:
            check_finite(y_hat, "forward/y_hat")

        # Decode: [B, T, latent_features] -> [B, T, F] or [B, T, 1, H, W]
        x_hat = self.decoder(z)
        if nan_active:
            check_finite(x_hat, "forward/x_hat")

        # Per-example Jacobian ∂z/∂x: [B, T, L, F]
        jac_z_x = self.compute_jacobian_z_wrt_x(x)
        # Per-example Jacobian ∂x/∂z: [B, T, F, L]
        jac_x_z = self.compute_jacobian_x_wrt_z(z)

        # In conv mode flatten x_hat's pixel dims so downstream losses (which
        # expect [B, T, F]) can consume it. The Jacobians are already flattened
        # over pixels by the helpers above.
        if self.conv_mode:
            B, T = x_hat.shape[0], x_hat.shape[1]
            x_hat = x_hat.reshape(B, T, -1)

        if nan_active:
            check_finite(jac_z_x, "forward/jac_z_x")
            check_finite(jac_x_z, "forward/jac_x_z")
            check_finite(SINDy_weights, "forward/SINDy_predict.weight")

        return y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights

    def _unpack_batch(self, batch):
        """Return ``(x_model, x_loss, mask)`` from a dataloader batch.

        Supports both pipeline modes:
          - feature mode: batch is ``x`` (or ``(x, label)``) of shape
            ``[B, T, F]`` (or ``[B, T]`` which is reshaped). ``x_model`` and
            ``x_loss`` are identical; ``mask`` is ``None``.
          - conv-map mode: batch is ``(maps, mask, label)`` (label optional)
            with ``maps`` shaped ``[B, T, 1, H, W]``. ``x_model`` is the 5D map
            tensor fed to ``forward``; ``x_loss`` is the pixel-flattened
            ``[B, T, H*W]`` view consumed by the loss criteria; ``mask`` is the
            valid-region mask passed to ``forward``.
        """
        if self.conv_mode:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError(
                    "conv-map mode expects batches of (maps, mask[, label]); "
                    f"got {type(batch).__name__}"
                )
            maps, mask = batch[0], batch[1]
            if maps.dim() == 4:  # [B, 1, H, W] -> add singleton time axis
                maps = maps.unsqueeze(1)
            B, T = maps.shape[0], maps.shape[1]
            x_loss = maps.reshape(B, T, -1)
            return maps, x_loss, mask

        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        return x, x, None

    def training_step(self, batch, batch_idx):
        self._backward_pass = 0
        x, x_loss, mask = self._unpack_batch(batch)

        if not self.use_dual_optimizers:
            # Single-optimizer (automatic optimization) path.
            y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x, mask)
            (
                loss,
                recon_loss,
                sindy_loss_xdot,
                sindy_loss_zdot,
                sindy_regularization,
                diagnostics,
            ) = self.criterion(x_loss, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
            self.log("train_total_loss", loss)
            self.log("train_recon_loss", recon_loss)
            self.log("train_sindyxdot_loss", sindy_loss_xdot)
            self.log("train_sindyzdot_loss", sindy_loss_zdot)
            self.log("train_sindyreg_loss", sindy_regularization)
            self._log_loss_diagnostics(diagnostics, prefix="train_")
            return loss

        # Dual-optimizer (manual optimization) path.
        opt_sindy, opt_decoder = self.optimizers()

        # --- Train SINDy Path (encoder + sindy_model) ---
        self.toggle_optimizer(opt_sindy)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x, mask)
        (
            sindy_loss,
            sindy_loss_xdot,
            sindy_loss_zdot,
            sindy_regularization,
            sindy_diagnostics,
        ) = self.sindy_criterion(x_loss, y_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        self.manual_backward(sindy_loss)
        # Gradient clipping to prevent derivative explosion
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.sindy_model.parameters(), max_norm=1.0)
        opt_sindy.step()
        opt_sindy.zero_grad()
        self.untoggle_optimizer(opt_sindy)

        # --- Train Decoder Path (encoder + decoder) ---
        # Re-run the forward pass so the decoder optimizer sees a fresh graph
        # built from the (now-updated) encoder weights.
        self.toggle_optimizer(opt_decoder)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x, mask)
        decoder_loss, recon_loss, decoder_diagnostics = self.decoder_criterion(
            x_loss, x_hat
        )
        self.manual_backward(decoder_loss)
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        opt_decoder.step()
        opt_decoder.zero_grad()
        self.untoggle_optimizer(opt_decoder)

        # Detach for logging; both losses are scalars but we don't want to
        # retain their computation graphs after the optimizer steps.
        total_loss = sindy_loss.detach() + decoder_loss.detach()
        self.log("train_total_loss", total_loss)
        self.log("train_sindy_loss", sindy_loss.detach())
        self.log("train_decoder_loss", decoder_loss.detach())
        self.log("train_recon_loss", recon_loss.detach())
        self.log("train_sindyxdot_loss", sindy_loss_xdot.detach())
        self.log("train_sindyzdot_loss", sindy_loss_zdot.detach())
        self.log("train_sindyreg_loss", sindy_regularization.detach())
        self._log_loss_diagnostics(sindy_diagnostics, prefix="train_")
        self._log_loss_diagnostics(decoder_diagnostics, prefix="train_")

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, x_loss, mask = self._unpack_batch(batch)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x, mask)

        if self.use_dual_optimizers:
            (
                sindy_loss,
                sindy_loss_xdot,
                sindy_loss_zdot,
                sindy_regularization,
                sindy_diagnostics,
            ) = self.sindy_criterion(x_loss, y_hat, z, jac_z_x, jac_x_z, SINDy_weights)
            decoder_loss, recon_loss, decoder_diagnostics = self.decoder_criterion(
                x_loss, x_hat
            )
            total_loss = sindy_loss + decoder_loss

            self.log("valid_loss", total_loss)
            self.log("valid_sindy_loss", sindy_loss)
            self.log("valid_decoder_loss", decoder_loss)
            self.log("valid_recon_loss", recon_loss)
            self.log("valid_sindyxdot_loss", sindy_loss_xdot)
            self.log("valid_sindyzdot_loss", sindy_loss_zdot)
            self.log("valid_sindyreg_loss", sindy_regularization)
            self._log_loss_diagnostics(sindy_diagnostics, prefix="valid_")
            self._log_loss_diagnostics(decoder_diagnostics, prefix="valid_")
            return total_loss

        (
            total_loss,
            recon_loss,
            sindy_loss_xdot,
            sindy_loss_zdot,
            sindy_regularization,
            diagnostics,
        ) = self.criterion(x_loss, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        self.log("valid_loss", total_loss)
        self.log("valid_recon_loss", recon_loss)
        self.log("valid_sindyxdot_loss", sindy_loss_xdot)
        self.log("valid_sindyzdot_loss", sindy_loss_zdot)
        self.log("valid_sindyreg_loss", sindy_regularization)
        self._log_loss_diagnostics(diagnostics, prefix="valid_")
        return total_loss

    def test_step(self, batch, batch_idx):
        x, x_loss, mask = self._unpack_batch(batch)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x, mask)

        if self.use_dual_optimizers:
            (
                sindy_loss,
                sindy_loss_xdot,
                sindy_loss_zdot,
                sindy_regularization,
                sindy_diagnostics,
            ) = self.sindy_criterion(x_loss, y_hat, z, jac_z_x, jac_x_z, SINDy_weights)
            decoder_loss, recon_loss, decoder_diagnostics = self.decoder_criterion(
                x_loss, x_hat
            )
            total_loss = sindy_loss + decoder_loss

            self.log("test_loss", total_loss)
            self.log("test_sindy_loss", sindy_loss)
            self.log("test_decoder_loss", decoder_loss)
            self.log("test_recon_loss", recon_loss)
            self.log("test_sindyxdot_loss", sindy_loss_xdot)
            self.log("test_sindyzdot_loss", sindy_loss_zdot)
            self.log("test_sindyreg_loss", sindy_regularization)
            self._log_loss_diagnostics(sindy_diagnostics, prefix="test_")
            self._log_loss_diagnostics(decoder_diagnostics, prefix="test_")
            return total_loss

        (
            total_loss,
            recon_loss,
            sindy_loss_xdot,
            sindy_loss_zdot,
            sindy_regularization,
            diagnostics,
        ) = self.criterion(x_loss, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        self.log("test_loss", total_loss)
        self.log("test_recon_loss", recon_loss)
        self.log("test_sindyxdot_loss", sindy_loss_xdot)
        self.log("test_sindyzdot_loss", sindy_loss_zdot)
        self.log("test_sindyreg_loss", sindy_regularization)
        self._log_loss_diagnostics(diagnostics, prefix="test_")
        return total_loss

    def _log_loss_diagnostics(self, diagnostics: dict, *, prefix: str) -> None:
        """Log entries of a loss-diagnostics dict under ``<prefix><key>``.

        ``diagnostics`` is a dict of Python scalars (target variances,
        unnormalized MSE values and per-term R² values) produced by the loss
        modules. Keys vary per loss class -- this helper logs whatever keys
        are present so it works uniformly for ``SINDyLoss``,
        ``SINDyPathLoss`` and ``DecoderPathLoss`` diagnostics.
        """
        if not diagnostics:
            return
        for key, value in diagnostics.items():
            self.log(f"{prefix}{key}", value)

    def on_after_backward(self):
        """Report which model sections have non-finite gradients.

        Walks parameters of each top-level section of this Lightning module
        (encoder, decoder, sindy_model, plus any other direct submodules)
        and prints, per section:
            - which parameters have non-finite gradients (NaN / +Inf / -Inf counts)
            - which parameters have ``grad is None`` (not part of the graph)
            - a section-level summary

        In dual-optimizer mode Lightning calls this hook after *each*
        ``manual_backward``. The first call (SINDy path) fires before the
        decoder backward, so decoder-half parameters legitimately have
        ``grad=None`` at that point — that is expected, not a bug. We
        therefore skip reporting on the first pass and only print the full
        picture after the second (decoder) backward, when every parameter
        that is reachable from either loss has had a chance to accumulate a
        gradient.
        """

        self._backward_pass += 1

        # In dual-optimizer mode: silently skip the first (SINDy-path) backward.
        # The decoder-half params have no gradient yet at that point, which
        # would produce misleading grad_none reports.
        if self.use_dual_optimizers and self._backward_pass < 2:
            return

        # Build the list of (section_name, module) to inspect. Cover the named
        # top-level components explicitly, plus catch anything else hanging off
        # this LightningModule so nothing is silently ignored.
        sections: list[tuple[str, nn.Module]] = []
        seen_ids: set[int] = set()
        for sect_name in ("encoder", "decoder", "sindy_model"):
            sub = getattr(self, sect_name, None)
            if isinstance(sub, nn.Module):
                sections.append((sect_name, sub))
                seen_ids.add(id(sub))

        for child_name, child in self.named_children():
            if id(child) in seen_ids:
                continue
            # Skip the loss module; it has no parameters in this codebase but
            # could in principle, and reporting on it is not useful here.
            if child_name in ("criterion", "sindy_criterion", "decoder_criterion"):
                continue
            sections.append((child_name, child))
            seen_ids.add(id(child))

        any_bad = False
        lines: list[str] = []
        # Track parameter tensor ids already reported so that shared parameters
        # (e.g. a FullResAutoencoder whose ``ae`` object is held by both the
        # encoder and decoder wrappers) are counted and reported only once per
        # section — duplicates across sections are skipped rather than
        # double-reported.
        global_seen_param_ids: set[int] = set()
        for sect_name, module in sections:
            sect_bad: list[str] = []
            sect_none: list[str] = []
            n_params = 0
            sect_seen: set[int] = set()
            for pname, p in module.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                # Skip parameters already reported in a previous section
                # (happens when encoder and decoder share the same ae object).
                if id(p) in global_seen_param_ids:
                    continue
                if id(p) in sect_seen:
                    continue
                sect_seen.add(id(p))
                global_seen_param_ids.add(id(p))
                n_params += 1
                g = p.grad
                if g is None:
                    sect_none.append(pname)
                    continue
                if g.is_complex():
                    finite_mask = torch.isfinite(g.real) & torch.isfinite(g.imag)
                    nan_mask = torch.isnan(g.real) | torch.isnan(g.imag)
                    posinf_mask = torch.isposinf(g.real) | torch.isposinf(g.imag)
                    neginf_mask = torch.isneginf(g.real) | torch.isneginf(g.imag)
                else:
                    finite_mask = torch.isfinite(g)
                    nan_mask = torch.isnan(g)
                    posinf_mask = torch.isposinf(g)
                    neginf_mask = torch.isneginf(g)
                if bool(finite_mask.all().item()):
                    continue
                n_nan = int(nan_mask.sum().item())
                n_pinf = int(posinf_mask.sum().item())
                n_ninf = int(neginf_mask.sum().item())
                sect_bad.append(
                    f"{pname} shape={tuple(g.shape)} "
                    f"nan={n_nan} +inf={n_pinf} -inf={n_ninf}"
                )

            status = "OK" if not sect_bad else "NONFINITE"
            header = (
                f"[on_after_backward] section={sect_name} status={status} "
                f"params={n_params} bad={len(sect_bad)} grad_none={len(sect_none)}"
            )
            lines.append(header)
            for entry in sect_bad:
                lines.append(f"  - {sect_name}/{entry}")
            if sect_none:
                lines.append(
                    f"  (grad is None for: {', '.join(sect_none)})"
                )
            if sect_bad:
                any_bad = True

        if any_bad:
            lines.insert(0, "[on_after_backward] non-finite gradients detected:")
        else:
            lines.insert(0, "[on_after_backward] all gradients finite")

        # Single print to keep output coherent across distributed/async logs.
        print("\n".join(lines))
    

    def on_train_batch_end(self, outputs, batch, batch_idx):

        # Prune small SINDy weights after optimizer step so zeros persist into next iteration
        with torch.no_grad():
            w = self.sindy_model.SINDy_predict.weight
            w.data.masked_fill_(w.abs() < 1e-8, 0.0)

        # Optional: catch parameter corruption immediately after optimizer step.
        if (
            getattr(self.sindy_model, "nan_check", False)
            and getattr(self.sindy_model, "nan_check_level", "off") != "off"
        ):
            check_module_params_finite(self.encoder, "post_step/encoder")

    def configure_optimizers(self):
        if not self.use_dual_optimizers:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)
            return optimizer

        # Dual-optimizer setup: encoder is intentionally shared between both
        # parameter groups so it receives gradient updates from both paths.
        sindy_params = (
            list(self.encoder.parameters())
            + list(self.sindy_model.parameters())
        )
        decoder_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )

        
        opt_sindy = optim.AdamW(sindy_params, lr=self.sindy_lr)
        opt_decoder = optim.AdamW(decoder_params, lr=self.decoder_lr)

        warmup_epochs = 5  # widen the window so the ramp is meaningful

        # SINDy: ramp from 0 → full LR over warmup_epochs, then hold constant
        sched_sindy = torch.optim.lr_scheduler.SequentialLR(
            opt_sindy,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt_sindy,
                    start_factor=0.25,   # begins at 0 * sindy_lr
                    end_factor=1.0,     # ends at 1 * sindy_lr
                    total_iters=warmup_epochs,
                ),
                torch.optim.lr_scheduler.ConstantLR(opt_sindy, factor=1.0, total_iters=1000),
            ],
            milestones=[warmup_epochs],
        )

        # Decoder: ramp from full LR → 0 over warmup_epochs, then hold at a low floor
        sched_decoder = torch.optim.lr_scheduler.SequentialLR(
            opt_decoder,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    opt_decoder,
                    start_factor=1.0,   # begins at 1 * decoder_lr
                    end_factor=0.025,     # ends at 0 * decoder_lr
                    total_iters=warmup_epochs,
                ),
                torch.optim.lr_scheduler.ConstantLR(opt_decoder, factor=0.0, total_iters=1000),
            ],
            milestones=[warmup_epochs],
        )

        return [
            {"optimizer": opt_sindy, "lr_scheduler": sched_sindy},
            {"optimizer": opt_decoder, "lr_scheduler": sched_decoder},
        ]




if __name__ == "__main__":
    validate_capacity_match_shallow_mlp_vs_fan(50)
