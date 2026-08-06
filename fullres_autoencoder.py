"""
FullResAutoencoder: A convolutional autoencoder for 31×31 lower-triangular masked inputs.

This network uses gated partial convolutions to handle irregular mask geometry in the encoder,
compresses to a 64-dimensional latent vector, and reconstructs back to full resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedPartialConv2d(nn.Module):
    """
    Gated Partial Convolution layer that handles masked inputs.
    
    Implements the approach from "Free-Form Image Inpainting with Gated Convolution"
    where both feature and gating convolutions are applied to handle irregular masks.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(GatedPartialConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Feature convolution
        self.conv_feature = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        
        # Gating convolution (learns to weight features based on mask)
        self.conv_gate = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        
        # Mask update convolution (fixed weights, no learnable parameters)
        self.mask_conv = nn.Conv2d(
            1, 1, kernel_size, stride, padding, bias=False
        )
        # Initialize mask convolution with all ones (counts valid neighbors)
        self.mask_conv.weight.data.fill_(1.0)
        self.mask_conv.weight.requires_grad = False
        
    def forward(self, x, mask):
        """
        Forward pass with gated partial convolution.
        
        Args:
            x: Input tensor (batch_size, in_channels, height, width)
            mask: Binary mask tensor (batch_size, 1, height, width) where 1=valid, 0=invalid
            
        Returns:
            output: Gated convolved features (batch_size, out_channels, height, width)
            mask_out: Updated mask (batch_size, 1, height, width)
        """
        # Apply feature convolution
        features = self.conv_feature(x * mask)
        
        # Apply gating convolution and sigmoid activation
        gate = torch.sigmoid(self.conv_gate(x * mask))
        
        # Gated features
        output = features * gate
        
        # Update mask: any pixel with at least one valid neighbor becomes valid
        with torch.no_grad():
            mask_out = self.mask_conv(mask)
            # Normalize by kernel size to get average valid ratio
            mask_out = torch.clamp(mask_out, 0, 1)
            # Binary mask: 1 if any valid input contributed. Preserve the
            # input mask's dtype (do NOT force float32) so mixed-precision /
            # float64 pipelines keep a consistent dtype through the stack.
            mask_out = (mask_out > 0).to(mask.dtype)
        
        return output, mask_out


def _conv_out_size(size, kernel_size=3, stride=2, padding=1):
    """Spatial output size of a Conv2d for one dimension.

    Uses the standard PyTorch formula ``floor((L + 2p - k) / s) + 1``.
    For ``kernel_size=3, stride=2, padding=1`` this equals ``ceil(L / 2)``.
    """
    return (size + 2 * padding - kernel_size) // stride + 1


class FullResAutoencoder(nn.Module):
    """
    Convolutional autoencoder for arbitrary ``H×W`` lower-triangular masked inputs.

    Architecture:
    - Encoder: 3 gated partial conv blocks (1→32→64→128 channels) with stride-2 downsampling
    - Latent: ``latent_dim``-dimensional vector
    - Decoder: 3 transposed conv blocks (128→64→32→32 channels) with stride-2 upsampling,
      followed by a bilinear resize back to the exact ``(H, W)`` and a 1-channel output conv.
    - Output: 1-channel reconstruction with ``H×W`` spatial dimensions

    The spatial size is fully parameterized: the encoder's downsampled spatial
    dimensions (and therefore the latent projection sizes) are computed from
    ``(height, width)`` rather than hardcoded to a 31×31 / 4×4 layout. This lets
    the same architecture ingest bicoherence maps of any grid size.

    All convolutions use 3×3 kernels. LeakyReLU(0.2) activations throughout.
    """

    def __init__(self, height=31, width=31, latent_dim=64):
        super(FullResAutoencoder, self).__init__()

        self.height = int(height)
        self.width = int(width)
        self.latent_dim = latent_dim

        # Spatial dims after 3 stride-2 encoder blocks (per dimension).
        h1 = _conv_out_size(self.height)
        h2 = _conv_out_size(h1)
        self._enc_h = _conv_out_size(h2)
        w1 = _conv_out_size(self.width)
        w2 = _conv_out_size(w1)
        self._enc_w = _conv_out_size(w2)
        self._flat_dim = 128 * self._enc_h * self._enc_w

        # ===== ENCODER =====
        # Block 1: 1 → 32 channels, H×W → H/2×W/2
        self.enc_block1 = GatedPartialConv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.enc_act1 = nn.LeakyReLU(0.2, inplace=True)

        # Block 2: 32 → 64 channels
        self.enc_block2 = GatedPartialConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_act2 = nn.LeakyReLU(0.2, inplace=True)

        # Block 3: 64 → 128 channels
        self.enc_block3 = GatedPartialConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_act3 = nn.LeakyReLU(0.2, inplace=True)

        # Latent projection: 128 * enc_h * enc_w → latent_dim
        self.latent_projection = nn.Linear(self._flat_dim, latent_dim)

        # ===== DECODER =====
        # Latent unprojection: latent_dim → 128 * enc_h * enc_w (reshape to 128 × enc_h × enc_w)
        self.latent_unprojection = nn.Linear(latent_dim, self._flat_dim)

        # Block 1: 128 → 64 channels, ×2 upsample
        self.dec_block1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_act1 = nn.LeakyReLU(0.2, inplace=True)

        # Block 2: 64 → 32 channels, ×2 upsample
        self.dec_block2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_act2 = nn.LeakyReLU(0.2, inplace=True)

        # Block 3: 32 → 32 channels, ×2 upsample
        self.dec_block3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_act3 = nn.LeakyReLU(0.2, inplace=True)

        # Output layer: 32 → 1 channel. A bilinear resize to the exact (H, W)
        # precedes this conv so the reconstruction matches the input size for
        # any grid dimensions (avoids brittle per-stage output_padding math).
        self.output_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def encode(self, x, mask):
        """
        Encode input to latent vector.
        
        Args:
            x: Input tensor (batch_size, 1, H, W)
            mask: Binary mask (batch_size, 1, H, W)
            
        Returns:
            latent: Latent vector (batch_size, latent_dim)
        """
        # Encoder Block 1: H×W → H/2×W/2
        x, mask = self.enc_block1(x, mask)
        x = self.enc_act1(x)
        
        # Encoder Block 2
        x, mask = self.enc_block2(x, mask)
        x = self.enc_act2(x)
        
        # Encoder Block 3
        x, mask = self.enc_block3(x, mask)
        x = self.enc_act3(x)
        
        # Flatten and project to latent space
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)  # (batch_size, 128 * enc_h * enc_w)
        latent = self.latent_projection(x)  # (batch_size, latent_dim)
        
        return latent
    
    def decode(self, latent):
        """
        Decode latent vector to reconstruction.
        
        Args:
            latent: Latent vector (batch_size, latent_dim)
            
        Returns:
            reconstruction: Reconstructed tensor (batch_size, 1, H, W)
        """
        # Unproject and reshape: latent_dim → 128 × enc_h × enc_w
        x = self.latent_unprojection(latent)
        batch_size = latent.size(0)
        x = x.reshape(batch_size, 128, self._enc_h, self._enc_w)
        
        # Decoder Block 1: ×2 upsample
        x = self.dec_block1(x)
        x = self.dec_act1(x)
        
        # Decoder Block 2: ×2 upsample
        x = self.dec_block2(x)
        x = self.dec_act2(x)
        
        # Decoder Block 3: ×2 upsample
        x = self.dec_block3(x)
        x = self.dec_act3(x)
        
        # Resize to the exact target (H, W) before the output conv so the
        # reconstruction matches the input size for arbitrary grid dimensions.
        if x.shape[-2:] != (self.height, self.width):
            x = F.interpolate(
                x, size=(self.height, self.width),
                mode="bilinear", align_corners=False,
            )
        
        # Output layer: H×W (no activation)
        reconstruction = self.output_conv(x)
        
        return reconstruction
    
    def forward(self, x, mask):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor (batch_size, 1, H, W)
            mask: Binary mask (batch_size, 1, H, W)
            
        Returns:
            reconstruction: Masked reconstruction (batch_size, 1, H, W)
            latent: Latent representation (batch_size, latent_dim)
        """
        # Encode to latent space
        latent = self.encode(x, mask)
        
        # Decode to reconstruction
        reconstruction = self.decode(latent)
        
        # Apply mask to ensure predictions only in valid triangular region
        reconstruction = reconstruction * mask
        
        return reconstruction, latent


def create_triangular_mask_from_frequencies(f1s, f2s, f_max, batch_size=1, device='cpu'):
    """
    Create a triangular mask based on the constraints f1 + f2 <= f_max AND f1 <= f2.
    
    This matches the mask generation in create_triangular_mask.py where the valid
    region is defined by the frequency constraints:
    - f1 + f2 <= f_max (triangular constraint from Nyquist)
    - f1 <= f2 (exploit swap symmetry b²(f1,f2) = b²(f2,f1), lower triangle only)
    
    Args:
        f1s: Array of f1 frequencies (length n_f1)
        f2s: Array of f2 frequencies (length n_f2)
        f_max: Maximum frequency threshold
        batch_size: Number of masks to create
        device: Device to create mask on
        
    Returns:
        mask: Binary mask tensor (batch_size, 1, n_f1, n_f2) where True = valid region
    """
    # Create meshgrid
    f1s_tensor = torch.tensor(f1s, device=device, dtype=torch.float32)
    f2s_tensor = torch.tensor(f2s, device=device, dtype=torch.float32)
    
    # meshgrid with indexing='ij' to match numpy behavior
    F1, F2 = torch.meshgrid(f1s_tensor, f2s_tensor, indexing='ij')
    
    # Create mask: valid where f1 + f2 <= f_max AND f1 <= f2
    # This exploits the symmetry b²(f1, f2) = b²(f2, f1) to restrict to lower triangle
    mask = ((F1 + F2) <= (f_max + 1e-9)) & (F1 <= F2)
    
    # Convert to float and add batch/channel dimensions
    mask = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, n_f1, n_f2)
    mask = mask.expand(batch_size, 1, -1, -1)  # (batch_size, 1, n_f1, n_f2)
    
    return mask


def create_lower_triangular_mask(size=31, batch_size=1, device='cpu'):
    """
    Create a simple lower triangular mask for testing.
    
    NOTE: For actual bicoherence data, use create_triangular_mask_from_frequencies()
    which properly implements the f1 + f2 <= f_max constraint.
    
    Args:
        size: Spatial dimension (default 31)
        batch_size: Number of masks to create
        device: Device to create mask on
        
    Returns:
        mask: Binary mask tensor (batch_size, 1, size, size)
    """
    mask = torch.tril(torch.ones(size, size, device=device))
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    mask = mask.expand(batch_size, 1, size, size)
    return mask


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test the model
    print("="*60)
    print("FullResAutoencoder Architecture Test")
    print("="*60)
    
    # Create model
    model = FullResAutoencoder(height=31, width=31, latent_dim=64)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create test input
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Test with frequency-based mask (matches create_triangular_mask.py)
    print("\n" + "-"*60)
    print("Test 1: Frequency-based triangular mask (f1 + f2 <= f_max)")
    print("-"*60)
    
    # Create frequency axes matching typical bicoherence data
    import numpy as np
    f_max = 25.0
    n_freqs = 31
    f1s = np.linspace(0, f_max, n_freqs)
    f2s = np.linspace(0, f_max, n_freqs)
    
    mask_freq = create_triangular_mask_from_frequencies(f1s, f2s, f_max, batch_size=batch_size, device=device)
    x_freq = torch.randn(batch_size, 1, 31, 31, device=device) * mask_freq
    
    print(f"Frequency mask - Valid pixels: {mask_freq[0].sum().item():.0f}")
    
    with torch.no_grad():
        reconstruction_freq, latent_freq = model(x_freq, mask_freq)
    
    print(f"Latent shape: {latent_freq.shape}")
    print(f"Reconstruction shape: {reconstruction_freq.shape}")
    compression_ratio_freq = mask_freq[0].sum().item() / latent_freq.shape[1]
    print(f"Compression ratio: {compression_ratio_freq:.2f}×")
    
    # Test with simple lower triangular mask
    print("\n" + "-"*60)
    print("Test 2: Simple lower triangular mask (for testing only)")
    print("-"*60)
    
    mask = create_lower_triangular_mask(size=31, batch_size=batch_size, device=device)
    
    # Random input in valid region
    x = torch.randn(batch_size, 1, 31, 31, device=device) * mask
    
    # Random input in valid region
    x = torch.randn(batch_size, 1, 31, 31, device=device) * mask
    
    print(f"Simple triangular mask - Valid pixels: {mask[0].sum().item():.0f}")
    
    with torch.no_grad():
        reconstruction, latent = model(x, mask)
    
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    compression_ratio = mask[0].sum().item() / latent.shape[1]
    print(f"Compression ratio: {compression_ratio:.2f}×")
    
    print("\n" + "="*60)
    print("Architecture Summary")
    print("="*60)
    print("\nEncoder:")
    print("  Block 1: 1 → 32 channels, 31×31 → 16×16")
    print("  Block 2: 32 → 64 channels, 16×16 → 8×8")
    print("  Block 3: 64 → 128 channels, 8×8 → 4×4")
    print("  Latent:  2048 → 64 dimensions")
    print("\nDecoder:")
    print("  Unproject: 64 → 2048 (128 × 4 × 4)")
    print("  Block 1: 128 → 64 channels, 4×4 → 8×8")
    print("  Block 2: 64 → 32 channels, 8×8 → 16×16")
    print("  Block 3: 32 → 32 channels, 16×16 → 31×31")
    print("  Output:  32 → 1 channel, 31×31 → 31×31")
    print("="*60)
