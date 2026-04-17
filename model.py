import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from itertools import combinations_with_replacement
import numpy as np


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


class ShallowFANAutoencoder(nn.Module):
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
        B, T, L = z.shape
        library = []

        latent_indices = range(L)

        # Polynomial features over latent dimension per time step
        for n in range(1, self.poly_order + 1):
            list_combinations = list(combinations_with_replacement(latent_indices, n))
            for combination in list_combinations:
                # z[..., combination] -> [B, T, n]; prod over last -> [B, T]
                feat = torch.prod(z[..., combination], dim=-1, keepdim=True)
                library.append(feat)

        # Linear latent features
        library.append(z)

        # Hilbert-derived features (real/imag parts) along time axis
        library.append(extract_real_component(z))
        library.append(extract_imaginary_component(z))

        return torch.cat(library, dim=-1)

    def compute_jacobian_z_wrt_x(self, x):
        """Compute per-example Jacobian ∂z/∂x for batched inputs.

        Args:
            x (Tensor): shape [B, T, F] with requires_grad=True
        Returns:
            Tensor: Jacobian of shape [B, T, latent_features, F]
        """
        B, T, F = x.shape
        x_flat = x.reshape(-1, F)

        def encoder_flat(x_in):
            # x_in shape [B*T, F]; returns [B*T, L]
            return self.encoder(x_in)

        jac = torch.autograd.functional.jacobian(
            encoder_flat,
            x_flat,
            vectorize=True,
            create_graph=True,
        )  # shape [B*T, L, B*T, F]

        # take per-sample diagonal across the two batch/time axes
        jac_diag = jac.diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # [B*T, L, F]
        jac_btlf = jac_diag.reshape(B, T, self.latent_features, F)
        return jac_btlf

    def compute_jacobian_x_wrt_z(self, z):
        """Compute per-example Jacobian ∂x/∂z for batched latents.

        Args:
            z (Tensor): shape [B, T, latent_features]
        Returns:
            Tensor: Jacobian of shape [B, T, system_features, latent_features]
        """

        B, T, L = z.shape
        z_flat = z.reshape(-1, L)

        def decoder_flat(z_in):
            # z_in shape [B*T, L]; returns [B*T, F]
            return self.decoder(z_in)

        jac = torch.autograd.functional.jacobian(
            decoder_flat,
            z_flat,
            vectorize=True,
            create_graph=True,
        )  # shape [B*T, F, B*T, L]

        jac_diag = jac.diagonal(dim1=0, dim2=2)  # [B*T, F, L]
        jac_btfl = jac_diag.reshape(B, T, self.system_features, L)
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

        x = x.requires_grad_(True)

        # Encoder/decoder act on the feature dimension (last dim) and are batch/time agnostic
        z = self.encoder(x).requires_grad_(True)  # [B, T, latent_features]
        theta_x = self.compute_library(z)  # [B, T, library_dim]
        y_hat = self.SINDy_predict(theta_x)  # [B, T, latent_features]
        x_hat = self.decoder(z)  # [B, T, system_features]

        # Per-example Jacobian ∂z/∂x: [B, T, L, F]
        jac_z_x = self.compute_jacobian_z_wrt_x(x)
        # Per-example Jacobian ∂x/∂z: [B, T, F, L]
        jac_x_z = self.compute_jacobian_x_wrt_z(z)

        # Return weights needed for loss computations (constant across batch/time)
        return y_hat, x_hat, z, jac_z_x, jac_x_z, self.SINDy_predict.weight


class SINDyLoss(nn.Module):
    def __init__(self):
        super(SINDyLoss, self).__init__()
        self.lambda1 = 1.0  # SINDy loss in x_dot
        self.lambda2 = 1.0  # SINDy loss in z_dot
        self.lambda3 = 1.0  # SINDy regularization loss
        self.lambda4 = 1.0  # z_dot via autograd Jacobian

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
        # Reconstruction over all time steps
        loss = F.mse_loss(x_hat, x)

        # Finite differences along time dimension
        x_dot = torch.diff(x, dim=1)  # [B, T-1, F]
        z_dot = torch.diff(z, dim=1)  # [B, T-1, L]
        y_hat_trim = y_hat[:, :-1, :]  # align to T-1
        jac_trim = jac_z_x[:, :-1, :, :]  # [B, T-1, L, F]
        jac_xz_trim = jac_x_z[:, :-1, :, :]  # [B, T-1, F, L]

        # Predicted x_dot from y_hat via decoder Jacobian
        x_dot_pred = torch.einsum("btfl,btl->btf", jac_xz_trim, y_hat_trim)

        # z_dot predicted via autograd Jacobian * x_dot
        z_dot_pred = torch.einsum("btlf,btf->btl", jac_trim, x_dot)

        loss += self.lambda1 * F.mse_loss(x_dot_pred, x_dot)
        loss += self.lambda2 * F.mse_loss(y_hat_trim, z_dot)
        loss += self.lambda4 * F.mse_loss(z_dot_pred, z_dot)
        loss += self.lambda3 * SINDy_weights.abs().sum()
        return loss


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
            )

        self.model = model
        self.criterion = SINDyLoss()
        self.lr = float(lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        self.log("validation_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = reshape_time_to_feature_blocks(x)
        y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights = self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, z, jac_z_x, jac_x_z, SINDy_weights)
        self.log("test_loss", loss)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Prune small SINDy weights after optimizer step so zeros persist into next iteration
        with torch.no_grad():
            weight = self.model.SINDy_predict.weight
            weight.masked_fill_(weight.abs() < 1e-3, 0.0)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    validate_capacity_match_shallow_mlp_vs_fan(50)
