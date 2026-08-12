"""Minimum regression test — GRU/FAN + Convolutional paths.

Tests:
  1. forward(compute_jacobians=True)  -> Jacobians are tensors, correct shapes [GRU]
  2. forward(compute_jacobians=False) -> Jacobians are None, outputs still valid [GRU]
  3. Single-optimizer fast_dev_run [GRU]
  4. Dual-optimizer fast_dev_run [GRU]
  5. Conv forward(True) and forward(False) -> Jacobian shapes [B,T,L,H*W]/[B,T,H*W,L]
  6. Conv dual-optimizer fast_dev_run -> decoder path runs with None Jacobians
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from model import (
    SINDySz,
    ShallowFANGRUEncoder,
    ShallowFANGRUDecoder,
    ConvSINDyEncoder,
    ConvSINDyDecoder,
)
from fullres_autoencoder import FullResAutoencoder

# ── shared dims ───────────────────────────────────────────────────────────────
INPUT_DIM  = 50   # feature-mode: must be divisible by 10
TIME_STEPS = 10
BATCH      = 4
N_SAMPLES  = 16

H, W       = 16, 16   # small spatial grid; keeps Jacobian fast
LATENT_DIM = 8        # conv latent dim

# ── helpers ───────────────────────────────────────────────────────────────────
def make_gru_model(dual=False):
    enc = ShallowFANGRUEncoder(input_dim=INPUT_DIM)
    dec = ShallowFANGRUDecoder(output_dim=INPUT_DIM)
    return SINDySz(
        time_dim=TIME_STEPS, system_features=INPUT_DIM,
        latent_features=enc.bottleneck_dim, poly_order=2,
        encoder=enc, decoder=dec,
        use_dual_optimizers=dual, sindy_lr=1e-3, decoder_lr=1e-3,
    )

def make_conv_model(dual=False):
    ae = FullResAutoencoder(height=H, width=W, latent_dim=LATENT_DIM)
    enc = ConvSINDyEncoder(height=H, width=W, latent_dim=LATENT_DIM, ae=ae)
    dec = ConvSINDyDecoder(height=H, width=W, latent_dim=LATENT_DIM, ae=ae)
    return SINDySz(
        time_dim=TIME_STEPS, system_features=H * W,
        latent_features=LATENT_DIM, poly_order=2,
        encoder=enc, decoder=dec,
        use_dual_optimizers=dual, sindy_lr=1e-3, decoder_lr=1e-3,
    )

def make_gru_loader():
    x = torch.randn(N_SAMPLES, TIME_STEPS, INPUT_DIM)
    return DataLoader(TensorDataset(x), batch_size=BATCH, shuffle=False)

def make_conv_loader():
    # batch yields (maps [B,T,1,H,W], mask [B,1,H,W])
    maps = torch.randn(N_SAMPLES, TIME_STEPS, 1, H, W)
    mask = torch.ones(N_SAMPLES, 1, H, W)
    return DataLoader(TensorDataset(maps, mask), batch_size=BATCH, shuffle=False)

# ── Test 1: GRU forward, compute_jacobians=True ───────────────────────────────
print("Test 1: GRU forward(compute_jacobians=True) ...")
m = make_gru_model(); m.eval()
x = torch.randn(2, TIME_STEPS, INPUT_DIM)
_, _, _, jac_z_x, jac_x_z, _ = m.forward(x)
L_dim = m.sindy_model.latent_features
assert jac_z_x is not None and jac_x_z is not None
assert jac_z_x.shape == (2, TIME_STEPS, L_dim, INPUT_DIM), jac_z_x.shape
assert jac_x_z.shape == (2, TIME_STEPS, INPUT_DIM, L_dim), jac_x_z.shape
print(f"  PASS  jac_z_x={tuple(jac_z_x.shape)}  jac_x_z={tuple(jac_x_z.shape)}")

# ── Test 2: GRU forward, compute_jacobians=False ──────────────────────────────
print("Test 2: GRU forward(compute_jacobians=False) ...")
y_hat, x_hat, z, jac_z_x, jac_x_z, _ = m.forward(x, compute_jacobians=False)
assert jac_z_x is None and jac_x_z is None
assert y_hat.shape == (2, TIME_STEPS, L_dim)
assert x_hat.shape == (2, TIME_STEPS, INPUT_DIM)
print(f"  PASS  jac_z_x=None  jac_x_z=None  y_hat={tuple(y_hat.shape)}")

# ── Test 3: GRU single-optimizer fast_dev_run ─────────────────────────────────
print("Test 3: GRU single-optimizer fast_dev_run ...")
trainer = L.Trainer(fast_dev_run=True, accelerator="cpu", devices=1,
                    enable_progress_bar=False, logger=False)
trainer.fit(make_gru_model(dual=False), make_gru_loader())
print("  PASS")

# ── Test 4: GRU dual-optimizer fast_dev_run ───────────────────────────────────
print("Test 4: GRU dual-optimizer fast_dev_run ...")
trainer = L.Trainer(fast_dev_run=True, accelerator="cpu", devices=1,
                    enable_progress_bar=False, logger=False)
trainer.fit(make_gru_model(dual=True), make_gru_loader())
print("  PASS")

# ── Test 5: Conv forward, both jacobian modes ─────────────────────────────────
print("Test 5: Conv forward(compute_jacobians=True) and (False) ...")
mc = make_conv_model(); mc.eval()
maps = torch.randn(2, TIME_STEPS, 1, H, W)
mask = torch.ones(2, 1, H, W)

# True: Jacobians should be tensors with flattened pixel dims
_, _, _, jac_z_x, jac_x_z, _ = mc.forward(maps, mask, compute_jacobians=True)
F_flat = H * W
assert jac_z_x is not None and jac_x_z is not None, "Jacobians should not be None"
assert jac_z_x.shape == (2, TIME_STEPS, LATENT_DIM, F_flat), \
    f"bad jac_z_x shape: {jac_z_x.shape}, expected (2,{TIME_STEPS},{LATENT_DIM},{F_flat})"
assert jac_x_z.shape == (2, TIME_STEPS, F_flat, LATENT_DIM), \
    f"bad jac_x_z shape: {jac_x_z.shape}, expected (2,{TIME_STEPS},{F_flat},{LATENT_DIM})"
print(f"  PASS (True)  jac_z_x={tuple(jac_z_x.shape)}  jac_x_z={tuple(jac_x_z.shape)}")

# False: Jacobians should be None; x_hat should be [B,T,H*W] (flattened)
y_hat, x_hat, z, jac_z_x, jac_x_z, _ = mc.forward(maps, mask, compute_jacobians=False)
assert jac_z_x is None and jac_x_z is None, "Jacobians should be None"
assert y_hat.shape == (2, TIME_STEPS, LATENT_DIM), f"bad y_hat: {y_hat.shape}"
assert x_hat.shape == (2, TIME_STEPS, F_flat),     f"bad x_hat: {x_hat.shape}"
print(f"  PASS (False) jac_z_x=None  y_hat={tuple(y_hat.shape)}  x_hat={tuple(x_hat.shape)}")

# ── Test 6: Conv dual-optimizer fast_dev_run ──────────────────────────────────
print("Test 6: Conv dual-optimizer fast_dev_run ...")
trainer = L.Trainer(fast_dev_run=True, accelerator="cpu", devices=1,
                    enable_progress_bar=False, logger=False)
trainer.fit(make_conv_model(dual=True), make_conv_loader())
print("  PASS")

print("\nAll 6 regression tests passed.")
