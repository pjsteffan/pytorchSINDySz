import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from itertools import combinations_with_replacement
import numpy as np


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
        H[1:N//2] = 2  # Positive frequencies
        H[N//2] = 0  # Nyquist frequency if N is even
    else:
        H[1:(N+1)//2] = 2  # Positive frequencies

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

class SINDyModel(nn.Module):
    def __init__(self, time_dim, system_features, latent_features,poly_order):
        super(SINDyModel, self).__init__()
        """SINDy model operating on batched sequences.

        Expected input shape: [batch, time_dim, system_features].
        """
        self.time_dim = time_dim
        self.system_features = system_features
        self.latent_features = latent_features
        self.poly_order = poly_order
        self.library_dim = self.compute_library_dim()
        self.encoder = nn.Linear(system_features, latent_features)  # Example encoder
        self.decoder = nn.Linear(latent_features, system_features)
        self.SINDy_predict = nn.Linear(self.library_dim, latent_features)  # SINDy prediction layer

    def compute_library_dim(self):
        self_features = self.latent_features   
        hilbert_features = 2 * self.latent_features

        poly_features = 0
        for n in range(1, self.poly_order+1):
            list_combinations = list(combinations_with_replacement(range(self.latent_features), n))
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
            raise ValueError(f"Expected x shape [B, T, F] or [B, T]; got {tuple(x.shape)}")

        assert x.dim() == 3, "Expected x shape [B, T, F] or [B, T] (auto-unsqueezed)"
        assert x.size(1) == self.time_dim, "Time dimension mismatch"

        param_dtype = next(self.parameters()).dtype
        if x.dtype != param_dtype:
            x = x.to(param_dtype)

        x = x.requires_grad_(True)

        # Encoder/decoder act on the feature dimension (last dim) and are batch/time agnostic
        z = self.encoder(x)                 # [B, T, latent_features]
        theta_x = self.compute_library(z)   # [B, T, library_dim]
        y_hat = self.SINDy_predict(theta_x) # [B, T, latent_features]
        x_hat = self.decoder(z)             # [B, T, system_features]

        # Per-example Jacobian ∂z/∂x: [B, T, L, F]
        jac_z_x = self.compute_jacobian_z_wrt_x(x)

        # Return weights needed for loss computations (constant across batch/time)
        return y_hat, x_hat, z, jac_z_x, self.SINDy_predict.weight, self.decoder.weight
    

class SINDyLoss(nn.Module):
    def __init__(self):
        super(SINDyLoss, self).__init__()
        self.lambda1 = 1.0  # SINDy loss in x_dot
        self.lambda2 = 1.0  # SINDy loss in z_dot
        self.lambda3 = 5.0  # SINDy regularization loss 
        self.lambda4 = 1.0  # z_dot via autograd Jacobian

    def forward(self, x, y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight):
        """Batched SINDy loss.

        Args:
            x: [B, T, F]
            y_hat: [B, T, L]
            x_hat: [B, T, F]
            z: [B, T, L]
            jac_z_x: [B, T, L, F]
            SINDy_weights: [L, library_dim]
            decoder_weight: [F, L]
        Returns:
            scalar loss
        """
        # Reconstruction over all time steps
        loss = F.mse_loss(x_hat, x)

        # Finite differences along time dimension
        x_dot = torch.diff(x, dim=1)        # [B, T-1, F]
        z_dot = torch.diff(z, dim=1)        # [B, T-1, L]
        y_hat_trim = y_hat[:, :-1, :]       # align to T-1
        jac_trim = jac_z_x[:, :-1, :, :]    # [B, T-1, L, F]

        # Predicted x_dot from y_hat via decoder Jacobian
        x_dot_pred = torch.einsum('btl,fl->btf', y_hat_trim, decoder_weight)

        # z_dot predicted via autograd Jacobian * x_dot
        z_dot_pred = torch.einsum('btlf,btf->btl', jac_trim, x_dot)

        loss += self.lambda1 * F.mse_loss(x_dot_pred, x_dot)
        loss += self.lambda2 * F.mse_loss(y_hat_trim, z_dot)
        loss += self.lambda4 * F.mse_loss(z_dot_pred, z_dot)
        loss += self.lambda3 * SINDy_weights.abs().sum()
        return loss 




class SINDySz(L.LightningModule):
    def __init__(self, model):
        super(SINDySz, self).__init__()
        self.model = model
        self.criterion = SINDyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = x.unsqueeze(-1)
        y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight = self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = x.unsqueeze(-1)
        y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight = self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight)
        self.log('validation_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        if x.dim() == 2:  # allow [B, T] by treating it as single-feature
            x = x.unsqueeze(-1)
        y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight = self.forward(x)
        loss = self.criterion(x, y_hat, x_hat, z, jac_z_x, SINDy_weights, decoder_weight)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

