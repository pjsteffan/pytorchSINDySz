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


class ShallowFANEncoder(nn.Module):
    """Encoder portion of shallowFAN_Sz: Win -> FAN -> fc1 -> ReLU -> fc2 -> ReLU."""

    def __init__(self, input_dim, p_ratio=0.45, use_p_bias=True):
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = self.input_dim // 10

        self.Win = nn.Linear(self.input_dim, self.input_dim)
        self.fan_layer1 = FANLayer(
            self.input_dim, self.input_dim, p_ratio=p_ratio, use_p_bias=use_p_bias
        )
        self.fc1 = nn.Linear(self.input_dim, self.input_dim // 5)
        self.fc2 = nn.Linear(self.input_dim // 5, self.bottleneck_dim)
        self.activate = nn.ReLU()

    def forward(self, x):
        out = self.Win(x)
        out = self.fan_layer1(out)
        out = self.fc1(out)
        out = self.activate(out)
        out = self.fc2(out)
        out = self.activate(out)
        return out


class ShallowFANDecoder(nn.Module):
    """Decoder portion of shallowFAN_Sz: fc3 -> ReLU -> fc4 -> ReLU -> FAN -> Wout."""

    def __init__(self, output_dim, p_ratio=0.45, use_p_bias=True):
        super().__init__()
        self.output_dim = int(output_dim)

        in_dim = self.output_dim // 10
        self.fc3 = nn.Linear(in_dim, self.output_dim // 5)
        self.fc4 = nn.Linear(self.output_dim // 5, self.output_dim)
        self.fan_layer2 = FANLayer(
            self.output_dim, self.output_dim, p_ratio=p_ratio, use_p_bias=use_p_bias
        )
        self.Wout = nn.Linear(self.output_dim, self.output_dim)
        self.activate = nn.ReLU()

    def forward(self, z):
        out = self.fc3(z)
        out = self.activate(out)
        out = self.fc4(out)
        out = self.activate(out)
        out = self.fan_layer2(out)
        out = self.Wout(out)
        return out


class ShallowFANGRUAutoencoder(nn.Module):
    """shallowFAN split into encoder/decoder.

    Matches the original shallowFAN_Sz architecture and enforces input_dim == output_dim
    due to the original implicit dimension coupling.
    """

    def __init__(self, input_dim, output_dim=None, p_ratio=0.45, use_p_bias=True):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if int(input_dim) != int(output_dim):
            raise ValueError("ShallowFANAutoencoder requires input_dim == output_dim")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.encoder = ShallowFANEncoder(
            self.input_dim, p_ratio=p_ratio, use_p_bias=use_p_bias
        )
        self.decoder = ShallowFANDecoder(
            self.output_dim, p_ratio=p_ratio, use_p_bias=use_p_bias
        )

    def forward(self, x):
        if x.dim() == 2:
            z = self.encoder(x)
            return self.decoder(z)
        if x.dim() == 3:
            b, t, f = x.shape
            x2 = x.reshape(b * t, f)
            z2 = self.encoder(x2)
            y2 = self.decoder(z2)
            return y2.reshape(b, t, f)
        raise ValueError(f"Expected x shape [N, F] or [B, T, F]; got {tuple(x.shape)}")


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def validate_capacity_match_shallow_mlp_vs_fan(
    feature_dim: int,
    p_ratio: float = 0.45,
    use_p_bias: bool = True,
    block_activation: str = "relu",
    block_bias: bool = True,
) -> dict:
    """Validate that capacity-matched MLP encoder/decoder match FAN counterparts.

    Instantiates:
      - ShallowFANEncoder/Decoder
      - CapacityMatchedShallowMLPEncoder/Decoder (via CapacityMatchedShallowMLPAutoencoder)

    Reports parameter count deltas:
      - encoder MLP vs FAN
      - decoder MLP vs FAN
      - residual blocks vs FAN layers (the intended matching budget)
      - overall total
    """

    fdim = int(feature_dim)

    fan_enc = ShallowFANEncoder(fdim, p_ratio=p_ratio, use_p_bias=use_p_bias)
    fan_dec = ShallowFANDecoder(fdim, p_ratio=p_ratio, use_p_bias=use_p_bias)

    mlp_ae = CapacityMatchedShallowMLPAutoencoder(
        fdim,
        p_ratio=p_ratio,
        use_p_bias=use_p_bias,
        block_activation=block_activation,
        block_bias=block_bias,
        verbose=False,
    )
    mlp_enc = mlp_ae.encoder
    mlp_dec = mlp_ae.decoder

    p_fan_enc = count_parameters(fan_enc)
    p_fan_dec = count_parameters(fan_dec)
    p_mlp_enc = count_parameters(mlp_enc)
    p_mlp_dec = count_parameters(mlp_dec)

    p_fan_layers = count_parameters(fan_enc.fan_layer1) + count_parameters(
        fan_dec.fan_layer2
    )
    p_blocks = count_parameters(mlp_enc.block1) + count_parameters(mlp_dec.block2)

    def pct(delta: int, base: int) -> float:
        if base == 0:
            return float("nan")
        return 100.0 * (float(delta) / float(base))

    out = {
        "feature_dim": fdim,
        "fan_encoder_params": p_fan_enc,
        "fan_decoder_params": p_fan_dec,
        "mlp_encoder_params": p_mlp_enc,
        "mlp_decoder_params": p_mlp_dec,
        "encoder_delta": p_mlp_enc - p_fan_enc,
        "decoder_delta": p_mlp_dec - p_fan_dec,
        "encoder_pct": pct(p_mlp_enc - p_fan_enc, p_fan_enc),
        "decoder_pct": pct(p_mlp_dec - p_fan_dec, p_fan_dec),
        "fan_layers_params": p_fan_layers,
        "mlp_blocks_params": p_blocks,
        "special_layers_delta": p_blocks - p_fan_layers,
        "special_layers_pct": pct(p_blocks - p_fan_layers, p_fan_layers),
        "fan_total": p_fan_enc + p_fan_dec,
        "mlp_total": p_mlp_enc + p_mlp_dec,
        "total_delta": (p_mlp_enc + p_mlp_dec) - (p_fan_enc + p_fan_dec),
        "total_pct": pct(
            (p_mlp_enc + p_mlp_dec) - (p_fan_enc + p_fan_dec),
            (p_fan_enc + p_fan_dec),
        ),
        "mlp_hidden_dim": getattr(mlp_ae, "hidden_dim", None),
    }

    print(
        "[validate_capacity_match] "
        f"F={fdim} H={out['mlp_hidden_dim']} "
        f"enc_delta={out['encoder_delta']} ({out['encoder_pct']:.3f}%) "
        f"dec_delta={out['decoder_delta']} ({out['decoder_pct']:.3f}%) "
        f"blocks_minus_fanlayers={out['special_layers_delta']} ({out['special_layers_pct']:.3f}%) "
        f"total_delta={out['total_delta']} ({out['total_pct']:.3f}%)"
    )
    return out


class ResidualFCBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation="relu",
        bias: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.dim, self.hidden_dim, bias=bias)
        self.fc2 = nn.Linear(self.hidden_dim, self.dim, bias=bias)

        if isinstance(activation, str):
            if activation.lower() == "relu":
                self.activation = nn.ReLU()
            elif activation.lower() == "gelu":
                self.activation = nn.GELU()
            elif activation.lower() == "tanh":
                self.activation = nn.Tanh()
            elif activation.lower() == "sigmoid":
                self.activation = nn.Sigmoid()
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation if activation else nn.Identity()

    def forward(self, x):
        return x + self.fc2(self.activation(self.fc1(x)))


class CapacityMatchedShallowMLPEncoder(nn.Module):
    """Encoder counterpart to ShallowFANEncoder with capacity-matched MLP block."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        block_activation="relu",
        block_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.bottleneck_dim = self.input_dim // 10
        self.hidden_dim = int(hidden_dim)

        self.Win = nn.Linear(self.input_dim, self.input_dim)
        self.block1 = ResidualFCBlock(
            self.input_dim,
            self.hidden_dim,
            activation=block_activation,
            bias=block_bias,
        )
        self.fc1 = nn.Linear(self.input_dim, self.input_dim // 5)
        self.fc2 = nn.Linear(self.input_dim // 5, self.bottleneck_dim)
        self.activate = nn.ReLU()

    def forward(self, x):
        if x.dim() == 2:
            out = self.Win(x)
            out = self.block1(out)
            out = self.fc1(out)
            out = self.activate(out)
            out = self.fc2(out)
            out = self.activate(out)
            return out
        if x.dim() == 3:
            b, t, f = x.shape
            x2 = x.reshape(b * t, f)
            z2 = self.forward(x2)
            return z2.reshape(b, t, z2.shape[-1])
        raise ValueError(f"Expected x shape [N, F] or [B, T, F]; got {tuple(x.shape)}")


class CapacityMatchedShallowMLPDecoder(nn.Module):
    """Decoder counterpart to ShallowFANDecoder with capacity-matched MLP block."""

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        block_activation="relu",
        block_bias: bool = True,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        in_dim = self.output_dim // 10

        self.fc3 = nn.Linear(in_dim, self.output_dim // 5)
        self.fc4 = nn.Linear(self.output_dim // 5, self.output_dim)
        self.block2 = ResidualFCBlock(
            self.output_dim,
            self.hidden_dim,
            activation=block_activation,
            bias=block_bias,
        )
        self.Wout = nn.Linear(self.output_dim, self.output_dim)
        self.activate = nn.ReLU()

    def forward(self, z):
        if z.dim() == 2:
            out = self.fc3(z)
            out = self.activate(out)
            out = self.fc4(out)
            out = self.activate(out)
            out = self.block2(out)
            out = self.Wout(out)
            return out
        if z.dim() == 3:
            b, t, l = z.shape
            z2 = z.reshape(b * t, l)
            x2 = self.forward(z2)
            return x2.reshape(b, t, x2.shape[-1])
        raise ValueError(f"Expected z shape [N, L] or [B, T, L]; got {tuple(z.shape)}")


class CapacityMatchedShallowMLPAutoencoder(nn.Module):
    """Capacity-matched MLP AE split into encoder/decoder objects."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        p_ratio: float = 0.45,
        use_p_bias: bool = True,
        block_activation="relu",
        block_bias: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        if int(input_dim) != int(output_dim):
            raise ValueError(
                "CapacityMatchedShallowMLPAutoencoder requires input_dim == output_dim"
            )

        fdim = int(input_dim)
        fan1 = FANLayer(fdim, fdim, p_ratio=p_ratio, use_p_bias=use_p_bias)
        fan2 = FANLayer(fdim, fdim, p_ratio=p_ratio, use_p_bias=use_p_bias)
        p_fan_layers = count_parameters(fan1) + count_parameters(fan2)

        def p_block(f: int, h: int, bias: bool) -> int:
            if bias:
                return 2 * f * h + (h + f)
            return 2 * f * h

        if block_bias:
            denom = 4 * fdim + 2
            h0 = int(max(1, round((p_fan_layers - 2 * fdim) / denom)))
        else:
            h0 = int(max(1, round(p_fan_layers / (4 * fdim))))

        best_h = 1
        best_delta = None
        window = 256
        lo = max(1, h0 - window)
        hi = h0 + window
        for h in range(lo, hi + 1):
            approx = 2 * p_block(fdim, h, block_bias)
            delta = abs(approx - p_fan_layers)
            if (
                best_delta is None
                or delta < best_delta
                or (delta == best_delta and h < best_h)
            ):
                best_delta = delta
                best_h = h

        self.hidden_dim = int(best_h)

        self.encoder = CapacityMatchedShallowMLPEncoder(
            fdim,
            self.hidden_dim,
            block_activation=block_activation,
            block_bias=block_bias,
        )
        self.decoder = CapacityMatchedShallowMLPDecoder(
            fdim,
            self.hidden_dim,
            block_activation=block_activation,
            block_bias=block_bias,
        )

        if verbose:
            p_total = count_parameters(self)
            print(
                "[CapacityMatchedShallowMLPAutoencoder] "
                f"F={fdim} H={self.hidden_dim} total={p_total} "
                f"fan_layers={p_fan_layers} blocks={count_parameters(self.encoder.block1) + count_parameters(self.decoder.block2)}"
            )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


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


def extract_real_component(x):
    # Expect x shape [B, T, *]; Hilbert along time (axis=1)
    return torch.abs(pytorch_hilbert(x, axis=1))


def extract_imaginary_component(x):
    # Expect x shape [B, T, *]; Hilbert along time (axis=1)
    return torch.angle(pytorch_hilbert(x, axis=1))


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
        encoder: nn.Module | None = None,
        decoder: nn.Module | None = None,
        sindy_predict: nn.Module | None = None,
        nan_check: bool = False,
        nan_check_level: str = "basic",
    ):
        super(SINDyModel, self).__init__()
        """SINDy model operating on batched sequences.

        Expected input shape: [batch, time_dim, system_features].
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

        # Allow caller to inject custom encoder/decoder/predictor modules.
        # Defaults preserve existing behavior.
        self.encoder = (
            encoder
            if encoder is not None
            else nn.Linear(system_features, latent_features)  # Example encoder
        )
        self.decoder = (
            decoder
            if decoder is not None
            else nn.Linear(latent_features, system_features)
        )
        self.SINDy_predict = (
            sindy_predict
            if sindy_predict is not None
            else nn.Linear(self.library_dim, latent_features)  # SINDy prediction layer
        )

    def compute_library_dim(self):
        self_features = self.latent_features
        hilbert_features = 2 * self.latent_features

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
        library.append(extract_real_component(z))
        library.append(extract_imaginary_component(z))

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

    def compute_jacobian_z_wrt_x(self, x):
        """Compute per-example Jacobian ∂z/∂x for batched inputs.

        Args:
            x (Tensor): shape [B, T, F] with requires_grad=True
        Returns:
            Tensor: Jacobian of shape [B, T, latent_features, F]
        """
        B, T, feat_dim = x.shape
        x_flat = x.reshape(-1, feat_dim)

        if self.nan_check and self.nan_check_level != "off":
            check_finite(x, "jac_z_x/x")

        def encoder_flat(x_in):
            # x_in shape [B*T, F]; returns [B*T, L]
            return self.encoder(x_in)

        jac = torch.autograd.functional.jacobian(
            encoder_flat,
            x_flat,
            vectorize=True,
            create_graph=False,
        )  # shape [B*T, L, B*T, F]

        if self.nan_check and self.nan_check_level == "full":
            check_finite(jac, "jac_z_x/raw")

        # take per-sample diagonal across the two batch/time axes
        jac_diag = jac.diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # [B*T, L, F]
        jac_btlf = jac_diag.reshape(B, T, self.latent_features, feat_dim)

        if self.nan_check and self.nan_check_level != "off":
            check_finite(jac_btlf, "jac_z_x/out")
        return jac_btlf

    def compute_jacobian_x_wrt_z(self, z):
        """Compute per-example Jacobian ∂x/∂z for batched latents.

        Args:
            z (Tensor): shape [B, T, latent_features]
        Returns:
            Tensor: Jacobian of shape [B, T, system_features, latent_features]
        """

        B, T, lat_dim = z.shape
        L = lat_dim
        z_flat = z.reshape(-1, L)

        if self.nan_check and self.nan_check_level != "off":
            check_finite(z, "jac_x_z/z")

        def decoder_flat(z_in):
            # z_in shape [B*T, L]; returns [B*T, F]
            return self.decoder(z_in)

        jac = torch.autograd.functional.jacobian(
            decoder_flat,
            z_flat,
            vectorize=True,
            create_graph=False,
        )  # shape [B*T, F, B*T, L]

        if self.nan_check and self.nan_check_level == "full":
            check_finite(jac, "jac_x_z/raw")

        jac_diag = jac.diagonal(dim1=0, dim2=2)  # [B*T, F, L]
        jac_btfl = jac_diag.reshape(B, T, self.system_features, L)

        if self.nan_check and self.nan_check_level != "off":
            check_finite(jac_btfl, "jac_x_z/out")
        return jac_btfl

    def forward(self, x):
        """Forward pass for batched sequences.

        Args:
            x (Tensor): shape [B, T, system_features]
        Returns:
            tuple: (y_hat, x_hat, z, SINDy_weights, decoder_weight)
                y_hat: predicted latent time-derivatives, [B, T, latent_features]
                x_hat: reconstruction, [B, T, system_features]
                z: latent states, [B, T, latent_features]
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape [B, T, F] or [B, T]; got {tuple(x.shape)}"
            )

        assert x.dim() == 3, "Expected x shape [B, T, F] or [B, T] (auto-unsqueezed)"
        # assert x.size(1) == self.time_dim, "Time dimension mismatch"

        param_dtype = next(self.parameters()).dtype
        if x.dtype != param_dtype:
            x = x.to(param_dtype)

        if self.nan_check and self.nan_check_level != "off":
            check_finite(x, "forward/x")
            # If x is finite but z becomes non-finite, parameters are a likely culprit.
            check_module_params_finite(self.encoder, "forward/encoder")

        x = x.requires_grad_(True)

        # Encoder/decoder act on the feature dimension (last dim) and are batch/time agnostic
        z = self.encoder(x).requires_grad_(True)  # [B, T, latent_features]
        if self.nan_check and self.nan_check_level != "off":
            check_finite(z, "forward/z")
        theta_x = self.compute_library(z)  # [B, T, library_dim]
        if self.nan_check and self.nan_check_level != "off":
            check_finite(theta_x, "forward/theta_x")
        y_hat = self.SINDy_predict(theta_x)  # [B, T, latent_features]
        if self.nan_check and self.nan_check_level != "off":
            check_finite(y_hat, "forward/y_hat")
        x_hat = self.decoder(z)  # [B, T, system_features]
        if self.nan_check and self.nan_check_level != "off":
            check_finite(x_hat, "forward/x_hat")

        # Per-example Jacobian ∂z/∂x: [B, T, L, F]
        jac_z_x = self.compute_jacobian_z_wrt_x(x)
        # Per-example Jacobian ∂x/∂z: [B, T, F, L]
        jac_x_z = self.compute_jacobian_x_wrt_z(z)

        if self.nan_check and self.nan_check_level != "off":
            check_finite(jac_z_x, "forward/jac_z_x")
            check_finite(jac_x_z, "forward/jac_x_z")
            check_finite(self.SINDy_predict.weight, "forward/SINDy_predict.weight")

        # Return weights needed for loss computations (constant across batch/time)
        return y_hat, x_hat, z, jac_z_x, jac_x_z, self.SINDy_predict.weight





class SINDyLoss(nn.Module):
    def __init__(self, *, nan_check: bool = False):
        super(SINDyLoss, self).__init__()
        self.lambda1 = 0.1
        self.lambda2 = 0.1
        self.lambda3 = 0.1
        self.lambda4 = 0.01
        self.nan_check = bool(nan_check)

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

        Computes derivatives along the time axis using forward/backward
        differences at the boundaries and central differences in the interior.

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

    def forward(self, x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights):
        """Batched SINDy loss.

        Args:
            x: [B, T, F]
            y_hat: [B, T, L]
            x_hat: [B, T, F]
            z: [B, T, L]
            jac_z_x: [B, T, L, F]
            jac_x_z: [B, T, F, L]
            SINDy_weights: [L, library_dim]
        Returns:
            scalar loss
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
            x, z, time_dim=1, fs=100
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

        recon_loss = self.lambda1 * F.mse_loss(x, x_hat)
        sindy_loss_xdot = self.lambda2 * F.mse_loss(x_dot, x_dot_pred)
        sindy_loss_zdot = self.lambda3 * F.mse_loss(z_dot_pred, y_hat_trim)
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

        return (
            total_loss,
            recon_loss,
            sindy_loss_xdot,
            sindy_loss_zdot,
            sindy_regularization,
        )


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
    ):
        super(SINDySz, self).__init__()

        if model is None:
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
            model = SINDyModel(
                time_dim=time_dim,
                system_features=system_features,
                latent_features=latent_features,
                poly_order=poly_order,
                encoder=encoder,
                decoder=decoder,
                sindy_predict=sindy_predict,
                nan_check=nan_check,
                nan_check_level=nan_check_level,
            )

        self.model = model
        self.criterion = SINDyLoss(nan_check=nan_check)
        self.lr = float(lr)

        equal_var_init(self.model)

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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x)
        loss, recon_loss, sindy_loss_xdot, sindy_loss_zdot, sindy_regularization = (
            self.criterion(x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        )
        self.log("train_total_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_sindyxdot_loss", sindy_loss_xdot)
        self.log("train_sindyzdot_loss", sindy_loss_zdot)
        self.log("train_sindyreg_loss", sindy_regularization)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x)
        loss, _, _, _, _ = self.criterion(
            x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights
        )
        self.log("valid_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x)
        loss, _, _, _, _ = self.criterion(
            x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights
        )
        self.log("test_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):

        # Prune small SINDy weights after optimizer step so zeros persist into next iteration
        with torch.no_grad():
            w = self.model.SINDy_predict.weight
            w.data.masked_fill_(w.abs() < 1e-8, 0.0)

        # Optional: catch parameter corruption immediately after optimizer step.
        if (
            getattr(self.model, "nan_check", False)
            and getattr(self.model, "nan_check_level", "off") != "off"
        ):
            check_module_params_finite(self.model.encoder, "post_step/encoder")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer



if __name__ == "__main__":
    validate_capacity_match_shallow_mlp_vs_fan(50)
