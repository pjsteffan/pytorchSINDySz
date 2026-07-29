"""Comprehensive validation script for trained SINDySz convolutional autoencoder.

Performs three validation tasks:
1. Reconstruction validation: Visual comparison of input vs reconstructed bicoherence maps
2. SINDy equation extraction: Extract and visualize discovered dynamical equations
3. Latent dynamics simulation: Simulate dynamics in latent space and decode

Usage:
    python validate_model.py --checkpoint path/to/checkpoint.ckpt
    python validate_model.py --auto-discover  # Use latest checkpoint
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from pathlib import Path
from datetime import datetime
import argparse
import json
from itertools import combinations_with_replacement
import sys
from typing import Tuple, List, Dict, Any

from model import SINDySz, SINDyModel, ConvSINDyEncoder, ConvSINDyDecoder
from fullres_autoencoder import FullResAutoencoder
from datasets import BicoherenceSequenceDataset
from torch.utils.data import Subset, random_split


def find_latest_checkpoint(root: Path = Path("lightning_logs")) -> Path:
    """Find the most recently modified checkpoint file under root."""
    if not root.exists():
        raise FileNotFoundError(f"Directory {root} does not exist.")
    candidates = list(root.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found under {root}.")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def load_checkpoint_and_data(
    checkpoint_path: Path,
    data_file: str,
    annotation_file: str,
    device: str,
) -> Tuple[SINDySz, BicoherenceSequenceDataset, Subset, Subset, Subset]:
    """Load trained model and create dataset splits.
    
    Returns:
        model: Loaded SINDySz model in eval mode
        dataset: Full BicoherenceSequenceDataset
        train_set, valid_set, test_set: Dataset subsets
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # First, create dataset to get grid size (needed for model reconstruction)
    time_dim = 8
    latent_features = 5
    poly_order = 2
    print(f"\nCreating dataset from: {data_file}")
    dataset = BicoherenceSequenceDataset(
        data_file=data_file,
        annotation_file=annotation_file,
        seq_len=time_dim,
        epoch_size=5.0,
        f_max=25.0,
        epoch_id_restriction=None,
        sample_rate=5000,
    )
    
    H, W = dataset.get_grid_size()
    system_features = H * W
    print(f"Dataset created: {len(dataset)} sequences, grid size: {H}×{W}")
    
    # Create encoder and decoder (matching main.py's build_conv_masked_ae)
    shared_ae = FullResAutoencoder(height=H, width=W, latent_dim=latent_features)
    encoder = ConvSINDyEncoder(height=H, width=W, latent_dim=latent_features, ae=shared_ae)
    decoder = ConvSINDyDecoder(height=H, width=W, latent_dim=latent_features, ae=shared_ae)
    
    # Load model with explicit parameters (checkpoint doesn't save hyper_parameters)
    model = SINDySz.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        time_dim=time_dim,
        system_features=system_features,
        latent_features=latent_features,
        poly_order=poly_order,
        encoder=encoder,
        decoder=decoder,
        # The state_dict will be loaded and override the weights
    )
    model.eval()
    model.to(device)
    
    print(f"Model loaded successfully. Device: {device}")
    print(f"  Latent features: {model.sindy_model.latent_features}")
    print(f"  Time dim: {model.sindy_model.time_dim}")
    print(f"  Poly order: {model.sindy_model.poly_order}")
    print(f"  Library dim: {model.sindy_model.library_dim}")
    print(f"  Conv mode: {model.conv_mode}")
    
    # Create splits matching main.py
    trv_set_size = int(len(dataset) * 0.8)
    trv_indices = list(range(trv_set_size))
    test_indices = list(range(trv_set_size, len(dataset)))
    
    trv_set = Subset(dataset, trv_indices)
    test_set = Subset(dataset, test_indices)
    
    train_set_size = int(len(trv_set) * 0.8)
    valid_set_size = len(trv_set) - train_set_size
    
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(
        trv_set, [train_set_size, valid_set_size], generator=seed
    )
    
    print(f"Splits: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")
    
    return model, dataset, train_set, valid_set, test_set


def validate_reconstruction(
    model: SINDySz,
    valid_set: Subset,
    output_dir: Path,
    num_sequences: int = 3,
    device: str = "cuda",
) -> Dict[str, float]:
    """Visualize and compute metrics for reconstruction quality.
    
    Returns:
        metrics: Dictionary of reconstruction metrics
    """
    print("\n" + "="*60)
    print("TASK 1: RECONSTRUCTION VALIDATION")
    print("="*60)
    
    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Select evenly spaced sequences from validation set
    valid_size = len(valid_set)
    if num_sequences > valid_size:
        num_sequences = valid_size
    indices = np.linspace(0, valid_size - 1, num_sequences, dtype=int)
    
    all_mse = []
    all_r2 = []
    per_timestep_mse = [[] for _ in range(8)]
    
    with torch.no_grad():
        for seq_num, idx in enumerate(indices):
            # Get data
            maps, mask, label = valid_set[idx]
            maps = maps.unsqueeze(0).to(device)  # [1, T, 1, H, W]
            mask = mask.unsqueeze(0).to(device)  # [1, 1, H, W]
            
            # Forward pass
            _, x_hat, z, _, _, _ = model.forward(maps, mask)
            
            # Reshape x_hat from [1, T, H*W] back to [1, T, 1, H, W]
            B, T = 1, maps.shape[1]
            H, W = model._map_h, model._map_w
            x_hat = x_hat.reshape(B, T, 1, H, W)
            
            # Move to CPU for plotting
            maps_cpu = maps[0].cpu().numpy()  # [T, 1, H, W]
            x_hat_cpu = x_hat[0].cpu().numpy()  # [T, 1, H, W]
            mask_cpu = mask[0, 0].cpu().numpy()  # [H, W]
            
            # Apply mask
            maps_masked = maps_cpu * mask_cpu[None, None, :, :]
            x_hat_masked = x_hat_cpu * mask_cpu[None, None, :, :]
            
            # Compute metrics
            mse_per_t = []
            for t in range(T):
                mask_t = mask_cpu > 0
                mse_t = np.mean((maps_masked[t, 0, mask_t] - x_hat_masked[t, 0, mask_t]) ** 2)
                mse_per_t.append(mse_t)
                per_timestep_mse[t].append(mse_t)
            
            overall_mse = np.mean(mse_per_t)
            all_mse.append(overall_mse)
            
            # Compute R²
            mask_flat = mask_cpu > 0
            maps_flat = maps_masked[:, 0, mask_flat].flatten()
            x_hat_flat = x_hat_masked[:, 0, mask_flat].flatten()
            ss_res = np.sum((maps_flat - x_hat_flat) ** 2)
            ss_tot = np.sum((maps_flat - np.mean(maps_flat)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            all_r2.append(r2)
            
            # Create visualization
            fig = plt.figure(figsize=(20, 6))
            gs = gridspec.GridSpec(2, T, figure=fig, hspace=0.3, wspace=0.2)
            
            # Determine color scale
            vmin = np.min([maps_masked.min(), x_hat_masked.min()])
            vmax = np.max([maps_masked.max(), x_hat_masked.max()])
            
            for t in range(T):
                # Input row
                ax_in = fig.add_subplot(gs[0, t])
                im = ax_in.imshow(maps_masked[t, 0], cmap='viridis', vmin=vmin, vmax=vmax)
                ax_in.set_title(f't={t}\nMSE={mse_per_t[t]:.4f}', fontsize=9)
                ax_in.axis('off')
                
                # Reconstruction row
                ax_rec = fig.add_subplot(gs[1, t])
                ax_rec.imshow(x_hat_masked[t, 0], cmap='viridis', vmin=vmin, vmax=vmax)
                ax_rec.axis('off')
            
            # Add row labels
            fig.text(0.02, 0.75, 'Input', rotation=90, va='center', fontsize=12, weight='bold')
            fig.text(0.02, 0.25, 'Reconstruction', rotation=90, va='center', fontsize=12, weight='bold')
            
            # Colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            fig.suptitle(
                f'Sequence {idx} (Epoch ID: {label.item()}) - Overall MSE: {overall_mse:.6f}, R²: {r2:.4f}',
                fontsize=14, weight='bold'
            )
            
            output_path = recon_dir / f"sequence_{idx:03d}_reconstruction.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Sequence {idx}: MSE={overall_mse:.6f}, R²={r2:.4f}")
    
    # Aggregate metrics
    metrics = {
        'mean_mse': float(np.mean(all_mse)),
        'std_mse': float(np.std(all_mse)),
        'median_mse': float(np.median(all_mse)),
        'mean_r2': float(np.mean(all_r2)),
        'std_r2': float(np.std(all_r2)),
        'median_r2': float(np.median(all_r2)),
    }
    
    # Create summary plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # MSE distribution
    axes[0].hist(all_mse, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(metrics['mean_mse'], color='red', linestyle='--', label=f"Mean: {metrics['mean_mse']:.6f}")
    axes[0].set_xlabel('MSE')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Per-Sequence MSE Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Per-timestep MSE
    per_timestep_mean = [np.mean(mses) for mses in per_timestep_mse]
    per_timestep_std = [np.std(mses) for mses in per_timestep_mse]
    axes[1].errorbar(range(8), per_timestep_mean, yerr=per_timestep_std, marker='o', capsize=5)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Per-Time-Step MSE')
    axes[1].grid(alpha=0.3)
    
    # R² scores
    axes[2].hist(all_r2, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[2].axvline(metrics['mean_r2'], color='red', linestyle='--', label=f"Mean: {metrics['mean_r2']:.4f}")
    axes[2].set_xlabel('R² Score')
    axes[2].set_ylabel('Count')
    axes[2].set_title('R² Score Distribution')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(recon_dir / "reconstruction_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Save metrics
    with open(recon_dir / "reconstruction_metrics.txt", 'w') as f:
        f.write("RECONSTRUCTION VALIDATION METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of sequences evaluated: {len(all_mse)}\n\n")
        f.write("Mean Squared Error (MSE):\n")
        f.write(f"  Mean:   {metrics['mean_mse']:.6f}\n")
        f.write(f"  Std:    {metrics['std_mse']:.6f}\n")
        f.write(f"  Median: {metrics['median_mse']:.6f}\n\n")
        f.write("R² Score:\n")
        f.write(f"  Mean:   {metrics['mean_r2']:.6f}\n")
        f.write(f"  Std:    {metrics['std_r2']:.6f}\n")
        f.write(f"  Median: {metrics['median_r2']:.6f}\n\n")
        f.write("Per-Time-Step MSE (mean ± std):\n")
        for t, (mean_t, std_t) in enumerate(zip(per_timestep_mean, per_timestep_std)):
            f.write(f"  t={t}: {mean_t:.6f} ± {std_t:.6f}\n")
    
    print(f"\nReconstruction validation complete. Results saved to {recon_dir}")
    print(f"  Mean MSE: {metrics['mean_mse']:.6f}")
    print(f"  Mean R²:  {metrics['mean_r2']:.6f}")
    
    return metrics


def get_library_term_names(latent_features: int, poly_order: int) -> List[str]:
    """Generate library term names matching SINDyModel.compute_library order."""
    terms = []
    
    # Polynomial terms (order 1 to poly_order)
    for n in range(1, poly_order + 1):
        combinations = list(combinations_with_replacement(range(latent_features), n))
        for combo in combinations:
            if n == 1:
                terms.append(f"z{combo[0]}")
            elif n == 2:
                if combo[0] == combo[1]:
                    terms.append(f"z{combo[0]}²")
                else:
                    terms.append(f"z{combo[0]}·z{combo[1]}")
            else:
                # General case for higher orders
                term_str = "·".join([f"z{i}" for i in combo])
                terms.append(term_str)
    
    # Linear latent features (duplicates order-1 polynomials, but kept for library structure)
    for i in range(latent_features):
        terms.append(f"z{i}")
    
    # Hilbert features
    for i in range(latent_features):
        terms.append(f"mag(z{i})")
    for i in range(latent_features):
        terms.append(f"cos_φ(z{i})")
    for i in range(latent_features):
        terms.append(f"sin_φ(z{i})")
    
    return terms


def extract_sindy_equations(
    model: SINDySz,
    output_dir: Path,
    threshold: float = 1e-3,
) -> Dict[str, Any]:
    """Extract and visualize SINDy equations.
    
    Returns:
        info: Dictionary with equation information and sparsity metrics
    """
    print("\n" + "="*60)
    print("TASK 2: SINDY EQUATION EXTRACTION")
    print("="*60)
    
    eq_dir = output_dir / "sindy_equations"
    eq_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract coefficient matrix
    weights = model.sindy_model.SINDy_predict.weight.detach().cpu().numpy()  # [L, library_dim]
    latent_features = model.sindy_model.latent_features
    library_dim = model.sindy_model.library_dim
    poly_order = model.sindy_model.poly_order
    
    print(f"Coefficient matrix shape: {weights.shape}")
    print(f"Latent features: {latent_features}, Library dim: {library_dim}")
    
    # Get library term names
    term_names = get_library_term_names(latent_features, poly_order)
    
    if len(term_names) != library_dim:
        print(f"WARNING: Term name count ({len(term_names)}) != library_dim ({library_dim})")
        print("Using generic term names instead.")
        term_names = [f"term_{i}" for i in range(library_dim)]
    
    # Extract equations
    equations_full = []
    equations_sparse = []
    sparsity = []
    
    with open(eq_dir / "sindy_equations.txt", 'w') as f:
        f.write("SINDY EQUATIONS - DISCOVERED DYNAMICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Latent dimensions: {latent_features}\n")
        f.write(f"Library size: {library_dim}\n")
        f.write(f"Polynomial order: {poly_order}\n")
        f.write(f"Sparsity threshold: {threshold}\n\n")
        
        for i in range(latent_features):
            coefs = weights[i, :]
            
            # Full equation (all non-zero)
            nonzero_mask = coefs != 0
            num_nonzero = np.sum(nonzero_mask)
            
            # Sparse equation (above threshold)
            active_mask = np.abs(coefs) > threshold
            num_active = np.sum(active_mask)
            sparsity.append(num_active / library_dim)
            
            # Sort by absolute value
            sorted_indices = np.argsort(-np.abs(coefs))
            
            f.write(f"dz{i}/dt = ")
            eq_terms = []
            for j in sorted_indices:
                if np.abs(coefs[j]) > threshold:
                    sign = "+" if coefs[j] >= 0 else "-"
                    eq_terms.append(f"{sign}{abs(coefs[j]):.6f}·{term_names[j]}")
            
            if eq_terms:
                eq_str = " ".join(eq_terms)
                eq_str = eq_str.replace("+ -", "- ")  # Clean up double signs
                if eq_str.startswith("+"):
                    eq_str = eq_str[1:]  # Remove leading +
                f.write(eq_str + "\n")
                equations_sparse.append(eq_str)
            else:
                f.write("0 (all coefficients below threshold)\n")
                equations_sparse.append("0")
            
            f.write(f"  Active terms: {num_active}/{library_dim} ({100*sparsity[-1]:.1f}%)\n")
            f.write(f"  Non-zero terms: {num_nonzero}/{library_dim}\n\n")
            
            # Store full equation info
            equations_full.append({
                'dimension': i,
                'coefficients': coefs.tolist(),
                'active_terms': num_active,
                'sparsity': sparsity[-1],
            })
    
    print(f"Equations extracted and saved to {eq_dir / 'sindy_equations.txt'}")
    
    # Visualize coefficient matrix as heatmap
    fig, ax = plt.subplots(figsize=(max(12, library_dim * 0.15), 6))
    
    # Create heatmap
    max_abs = np.max(np.abs(weights))
    im = ax.imshow(weights, cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)
    
    # Set ticks and labels
    ax.set_yticks(range(latent_features))
    ax.set_yticklabels([f"dz{i}/dt" for i in range(latent_features)])
    ax.set_xticks(range(0, library_dim, max(1, library_dim // 20)))
    ax.set_xticklabels(
        [term_names[i] for i in range(0, library_dim, max(1, library_dim // 20))],
        rotation=90, ha='center', fontsize=8
    )
    
    # Mark active coefficients with border
    for i in range(latent_features):
        for j in range(library_dim):
            if np.abs(weights[i, j]) > threshold:
                rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                     edgecolor='black', linewidth=1.5)
                ax.add_patch(rect)
    
    ax.set_xlabel('Library Terms')
    ax.set_ylabel('Latent Dimensions')
    ax.set_title(f'SINDy Coefficient Matrix (threshold={threshold})', fontsize=14, weight='bold')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Coefficient Value')
    
    fig.tight_layout()
    fig.savefig(eq_dir / "sindy_coefficients_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Heatmap saved to {eq_dir / 'sindy_coefficients_heatmap.png'}")
    
    # Sparsity analysis plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Active terms per dimension
    axes[0].bar(range(latent_features), [eq['active_terms'] for eq in equations_full])
    axes[0].set_xlabel('Latent Dimension')
    axes[0].set_ylabel('Number of Active Terms')
    axes[0].set_title('Active Terms per Dimension')
    axes[0].set_xticks(range(latent_features))
    axes[0].set_xticklabels([f"z{i}" for i in range(latent_features)])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Coefficient magnitude distribution
    all_coefs = weights.flatten()
    nonzero_coefs = all_coefs[all_coefs != 0]
    axes[1].hist(np.abs(nonzero_coefs), bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    axes[1].set_xlabel('|Coefficient|')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Coefficient Magnitude Distribution')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(eq_dir / "sindy_sparsity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Sparsity analysis saved to {eq_dir / 'sindy_sparsity_analysis.png'}")
    
    # Summary statistics
    mean_sparsity = np.mean(sparsity)
    print(f"\nSparsity summary:")
    print(f"  Mean active terms per dimension: {mean_sparsity * library_dim:.1f} / {library_dim}")
    print(f"  Mean sparsity: {100 * mean_sparsity:.1f}%")
    
    info = {
        'equations_full': equations_full,
        'equations_sparse': equations_sparse,
        'mean_sparsity': float(mean_sparsity),
        'library_dim': library_dim,
        'threshold': threshold,
    }
    
    return info


def simulate_dynamics(
    model: SINDySz,
    valid_set: Subset,
    output_dir: Path,
    num_sequences: int = 3,
    device: str = "cuda",
    dt: float = 3.0,
) -> Dict[str, Any]:
    """Simulate latent dynamics using SINDy and compare to ground truth.
    
    Args:
        dt: Time step in seconds (3.0s between consecutive bicoherence maps)
    
    Returns:
        metrics: Dictionary with simulation metrics
    """
    print("\n" + "="*60)
    print("TASK 3: LATENT DYNAMICS SIMULATION")
    print("="*60)
    
    sim_dir = output_dir / "dynamics_simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    
    # Select evenly spaced sequences
    valid_size = len(valid_set)
    if num_sequences > valid_size:
        num_sequences = valid_size
    indices = np.linspace(0, valid_size - 1, num_sequences, dtype=int)
    
    T = 8  # Time steps
    L = model.sindy_model.latent_features
    
    all_latent_errors = []
    all_recon_errors = []
    latent_error_over_time = [[] for _ in range(T)]
    recon_error_over_time = [[] for _ in range(T)]
    
    with torch.no_grad():
        for seq_num, idx in enumerate(indices):
            print(f"\n  Simulating sequence {idx}...")
            
            # Get data
            maps, mask, label = valid_set[idx]
            maps = maps.unsqueeze(0).to(device)  # [1, T, 1, H, W]
            mask = mask.unsqueeze(0).to(device)  # [1, 1, H, W]
            
            # Encode to latent
            z_gt = model.encoder(maps)  # [1, T, L]
            
            # Simulate SINDy dynamics using Explicit Euler
            z_sim = torch.zeros(1, T, L, device=device, dtype=z_gt.dtype)
            z_sim[0, 0, :] = z_gt[0, 0, :]  # Initial condition
            
            unstable = False
            for t in range(T - 1):
                z_t = z_sim[:, t:t+1, :]  # [1, 1, L] - keep batch dimension
                
                # Compute library features (expects [B, T, L])
                theta_t = model.sindy_model.compute_library(z_t)  # [1, 1, library_dim]
                
                # Predict dz/dt
                dz_dt = model.sindy_model.SINDy_predict(theta_t)  # [1, 1, L]
                
                # Euler step
                z_sim[0, t+1, :] = z_sim[0, t, :] + dt * dz_dt[0, 0, :]
                
                # Check for instability
                if not torch.isfinite(z_sim[0, t+1, :]).all():
                    print(f"    WARNING: Simulation became unstable at t={t+1}")
                    unstable = True
                    break
            
            if unstable:
                print(f"    Skipping sequence {idx} due to instability")
                continue
            
            # Decode trajectories
            x_gt = model.decoder(z_gt)  # [1, T, 1, H, W]
            x_sim = model.decoder(z_sim)  # [1, T, 1, H, W]
            
            # Reshape for loss computation
            H, W = model._map_h, model._map_w
            x_gt = x_gt.reshape(1, T, H, W)
            x_sim = x_sim.reshape(1, T, H, W)
            maps_flat = maps.reshape(1, T, H, W)
            
            # Move to CPU
            z_gt_cpu = z_gt[0].cpu().numpy()  # [T, L]
            z_sim_cpu = z_sim[0].cpu().numpy()  # [T, L]
            x_gt_cpu = x_gt[0].cpu().numpy()  # [T, H, W]
            x_sim_cpu = x_sim[0].cpu().numpy()  # [T, H, W]
            maps_cpu = maps_flat[0].cpu().numpy()  # [T, H, W]
            mask_cpu = mask[0, 0].cpu().numpy()  # [H, W]
            
            # Compute errors
            latent_errors_t = np.linalg.norm(z_gt_cpu - z_sim_cpu, axis=1)  # [T]
            
            recon_errors_t = []
            for t in range(T):
                mask_t = mask_cpu > 0
                err_t = np.mean((maps_cpu[t, mask_t] - x_sim_cpu[t, mask_t]) ** 2)
                recon_errors_t.append(err_t)
            recon_errors_t = np.array(recon_errors_t)
            
            all_latent_errors.append(np.mean(latent_errors_t))
            all_recon_errors.append(np.mean(recon_errors_t))
            
            for t in range(T):
                latent_error_over_time[t].append(latent_errors_t[t])
                recon_error_over_time[t].append(recon_errors_t[t])
            
            # Visualize latent trajectories
            fig, axes = plt.subplots(L, 1, figsize=(10, 2*L), sharex=True)
            if L == 1:
                axes = [axes]
            
            for i in range(L):
                axes[i].plot(range(T), z_gt_cpu[:, i], 'o-', label='Ground Truth', linewidth=2)
                axes[i].plot(range(T), z_sim_cpu[:, i], 's--', label='SINDy Simulation', linewidth=2)
                axes[i].set_ylabel(f'z{i}', fontsize=11)
                axes[i].grid(alpha=0.3)
                if i == 0:
                    axes[i].legend(loc='best')
            
            axes[-1].set_xlabel('Time Step')
            fig.suptitle(f'Sequence {idx}: Latent Trajectories (Epoch ID: {label.item()})', 
                        fontsize=14, weight='bold')
            fig.tight_layout()
            fig.savefig(sim_dir / f"sequence_{idx:03d}_latent_trajectories.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Visualize decoded comparisons
            fig = plt.figure(figsize=(20, 8))
            gs = gridspec.GridSpec(3, T, figure=fig, hspace=0.3, wspace=0.2)
            
            vmin = np.min([maps_cpu.min(), x_gt_cpu.min(), x_sim_cpu.min()])
            vmax = np.max([maps_cpu.max(), x_gt_cpu.max(), x_sim_cpu.max()])
            
            for t in range(T):
                # Input
                ax0 = fig.add_subplot(gs[0, t])
                im = ax0.imshow(maps_cpu[t] * mask_cpu, cmap='viridis', vmin=vmin, vmax=vmax)
                ax0.set_title(f't={t}', fontsize=9)
                ax0.axis('off')
                
                # Ground-truth reconstruction
                ax1 = fig.add_subplot(gs[1, t])
                ax1.imshow(x_gt_cpu[t] * mask_cpu, cmap='viridis', vmin=vmin, vmax=vmax)
                mask_t = mask_cpu > 0
                mse_gt = np.mean((maps_cpu[t, mask_t] - x_gt_cpu[t, mask_t]) ** 2)
                ax1.set_title(f'MSE={mse_gt:.4f}', fontsize=8)
                ax1.axis('off')
                
                # SINDy-simulated reconstruction
                ax2 = fig.add_subplot(gs[2, t])
                ax2.imshow(x_sim_cpu[t] * mask_cpu, cmap='viridis', vmin=vmin, vmax=vmax)
                ax2.set_title(f'MSE={recon_errors_t[t]:.4f}', fontsize=8)
                ax2.axis('off')
            
            # Row labels
            fig.text(0.02, 0.83, 'Input', rotation=90, va='center', fontsize=12, weight='bold')
            fig.text(0.02, 0.5, 'GT Recon', rotation=90, va='center', fontsize=12, weight='bold')
            fig.text(0.02, 0.17, 'SINDy Recon', rotation=90, va='center', fontsize=12, weight='bold')
            
            # Colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            
            fig.suptitle(
                f'Sequence {idx}: Decoded Comparison (Mean Latent Error: {np.mean(latent_errors_t):.4f})',
                fontsize=14, weight='bold'
            )
            fig.savefig(sim_dir / f"sequence_{idx:03d}_decoded_comparison.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"    Mean latent error: {np.mean(latent_errors_t):.4f}")
            print(f"    Mean recon error: {np.mean(recon_errors_t):.6f}")
    
    if not all_latent_errors:
        print("WARNING: All simulations were unstable. No metrics to report.")
        return {}
    
    # Aggregate analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Latent error over time
    mean_latent = [np.mean(errs) for errs in latent_error_over_time]
    std_latent = [np.std(errs) for errs in latent_error_over_time]
    axes[0].errorbar(range(T), mean_latent, yerr=std_latent, marker='o', capsize=5)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('||z_sim - z_gt||')
    axes[0].set_title('Latent Space Error Over Time')
    axes[0].grid(alpha=0.3)
    
    # Reconstruction error over time
    mean_recon = [np.mean(errs) for errs in recon_error_over_time]
    std_recon = [np.std(errs) for errs in recon_error_over_time]
    axes[1].errorbar(range(T), mean_recon, yerr=std_recon, marker='o', capsize=5, color='orange')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('MSE(x_input, x_sim)')
    axes[1].set_title('Reconstruction Error Over Time')
    axes[1].grid(alpha=0.3)
    
    # Final frame error distribution
    final_latent_errors = [errs[-1] for errs in latent_error_over_time]
    axes[2].hist(final_latent_errors, bins=20, edgecolor='black', alpha=0.7)
    axes[2].set_xlabel('Final Frame Latent Error')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Final Frame (t=7) Error Distribution')
    axes[2].grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(sim_dir / "aggregate_simulation_analysis.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Compute metrics
    metrics = {
        'mean_latent_error': float(np.mean(all_latent_errors)),
        'std_latent_error': float(np.std(all_latent_errors)),
        'mean_recon_error': float(np.mean(all_recon_errors)),
        'std_recon_error': float(np.std(all_recon_errors)),
        'num_stable': len(all_latent_errors),
        'num_attempted': num_sequences,
    }
    
    # Save metrics
    with open(sim_dir / "dynamics_simulation_metrics.txt", 'w') as f:
        f.write("DYNAMICS SIMULATION METRICS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of sequences: {metrics['num_stable']}/{metrics['num_attempted']} stable\n")
        f.write(f"Time step (dt): {dt}s\n")
        f.write(f"Integration method: Explicit Euler\n\n")
        f.write("Latent Space Error (||z_sim - z_gt||):\n")
        f.write(f"  Mean:   {metrics['mean_latent_error']:.6f}\n")
        f.write(f"  Std:    {metrics['std_latent_error']:.6f}\n\n")
        f.write("Reconstruction Error (MSE):\n")
        f.write(f"  Mean:   {metrics['mean_recon_error']:.6f}\n")
        f.write(f"  Std:    {metrics['std_recon_error']:.6f}\n\n")
        f.write("Error Over Time (mean ± std):\n")
        f.write("  Latent Error:\n")
        for t, (m, s) in enumerate(zip(mean_latent, std_latent)):
            f.write(f"    t={t}: {m:.6f} ± {s:.6f}\n")
        f.write("  Reconstruction Error:\n")
        for t, (m, s) in enumerate(zip(mean_recon, std_recon)):
            f.write(f"    t={t}: {m:.6f} ± {s:.6f}\n")
    
    print(f"\nDynamics simulation complete. Results saved to {sim_dir}")
    print(f"  Stable simulations: {metrics['num_stable']}/{metrics['num_attempted']}")
    print(f"  Mean latent error: {metrics['mean_latent_error']:.6f}")
    print(f"  Mean recon error:  {metrics['mean_recon_error']:.6f}")
    
    return metrics


def generate_summary_report(
    output_dir: Path,
    checkpoint_path: Path,
    model: SINDySz,
    dataset: BicoherenceSequenceDataset,
    train_set: Subset,
    valid_set: Subset,
    test_set: Subset,
    recon_metrics: Dict,
    sindy_info: Dict,
    sim_metrics: Dict,
):
    """Generate comprehensive summary report."""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    report_path = output_dir / "validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SINDYSZ CONVOLUTIONAL AUTOENCODER - VALIDATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        
        # Model info
        f.write("MODEL INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Architecture: Convolutional SINDy Autoencoder\n")
        f.write(f"  Latent features: {model.sindy_model.latent_features}\n")
        f.write(f"  Time dimension: {model.sindy_model.time_dim}\n")
        f.write(f"  Polynomial order: {model.sindy_model.poly_order}\n")
        f.write(f"  Library size: {model.sindy_model.library_dim}\n")
        f.write(f"  Grid size: {model._map_h}×{model._map_w}\n")
        f.write(f"  Dual optimizers: {model.use_dual_optimizers}\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total sequences: {len(dataset)}\n")
        f.write(f"Train set: {len(train_set)}\n")
        f.write(f"Validation set: {len(valid_set)}\n")
        f.write(f"Test set: {len(test_set)}\n")
        f.write(f"Sequence length: {dataset.seq_len} (T=8 time steps)\n")
        f.write(f"Map time step: 3.0s\n")
        f.write(f"Grid size: {dataset.height}×{dataset.width}\n\n")
        
        # Reconstruction results
        f.write("RECONSTRUCTION VALIDATION\n")
        f.write("-"*70 + "\n")
        if recon_metrics:
            f.write(f"Mean MSE: {recon_metrics['mean_mse']:.6f} ± {recon_metrics['std_mse']:.6f}\n")
            f.write(f"Median MSE: {recon_metrics['median_mse']:.6f}\n")
            f.write(f"Mean R²: {recon_metrics['mean_r2']:.4f} ± {recon_metrics['std_r2']:.4f}\n")
            f.write(f"Median R²: {recon_metrics['median_r2']:.4f}\n")
        else:
            f.write("No reconstruction metrics available.\n")
        f.write("\n")
        
        # SINDy equations
        f.write("SINDY EQUATION EXTRACTION\n")
        f.write("-"*70 + "\n")
        if sindy_info:
            f.write(f"Library dimension: {sindy_info['library_dim']}\n")
            f.write(f"Sparsity threshold: {sindy_info['threshold']}\n")
            f.write(f"Mean sparsity: {100 * sindy_info['mean_sparsity']:.1f}%\n")
            f.write(f"Mean active terms per dimension: {sindy_info['mean_sparsity'] * sindy_info['library_dim']:.1f}\n")
            f.write("\nSparse Equations (above threshold):\n")
            for i, eq in enumerate(sindy_info['equations_sparse']):
                f.write(f"  dz{i}/dt = {eq}\n")
        else:
            f.write("No SINDy equation information available.\n")
        f.write("\n")
        
        # Dynamics simulation
        f.write("DYNAMICS SIMULATION\n")
        f.write("-"*70 + "\n")
        if sim_metrics:
            f.write(f"Stable simulations: {sim_metrics['num_stable']}/{sim_metrics['num_attempted']}\n")
            f.write(f"Mean latent error: {sim_metrics['mean_latent_error']:.6f} ± {sim_metrics['std_latent_error']:.6f}\n")
            f.write(f"Mean reconstruction error: {sim_metrics['mean_recon_error']:.6f} ± {sim_metrics['std_recon_error']:.6f}\n")
        else:
            f.write("No simulation metrics available.\n")
        f.write("\n")
        
        # File inventory
        f.write("OUTPUT FILES\n")
        f.write("-"*70 + "\n")
        f.write("reconstructions/\n")
        f.write("  - sequence_*_reconstruction.png: Individual sequence comparisons\n")
        f.write("  - reconstruction_summary.png: Aggregate metrics\n")
        f.write("  - reconstruction_metrics.txt: Detailed metrics\n\n")
        f.write("sindy_equations/\n")
        f.write("  - sindy_equations.txt: Extracted equations\n")
        f.write("  - sindy_coefficients_heatmap.png: Coefficient matrix visualization\n")
        f.write("  - sindy_sparsity_analysis.png: Sparsity analysis\n\n")
        f.write("dynamics_simulation/\n")
        f.write("  - sequence_*_latent_trajectories.png: Latent space trajectories\n")
        f.write("  - sequence_*_decoded_comparison.png: Decoded reconstructions\n")
        f.write("  - aggregate_simulation_analysis.png: Error over time\n")
        f.write("  - dynamics_simulation_metrics.txt: Detailed metrics\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    # Also create README
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write("# SINDySz Model Validation Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Overview\n\n")
        f.write("This directory contains validation results for a trained SINDySz convolutional autoencoder model.\n\n")
        f.write("## Directory Structure\n\n")
        f.write("- `reconstructions/`: Input vs reconstructed bicoherence maps\n")
        f.write("- `sindy_equations/`: Discovered dynamical system equations\n")
        f.write("- `dynamics_simulation/`: Latent dynamics simulation results\n")
        f.write("- `validation_report.txt`: Comprehensive summary report\n\n")
        f.write("## Quick Summary\n\n")
        if recon_metrics:
            f.write(f"- **Reconstruction MSE**: {recon_metrics['mean_mse']:.6f}\n")
            f.write(f"- **Reconstruction R²**: {recon_metrics['mean_r2']:.4f}\n")
        if sindy_info:
            f.write(f"- **Mean sparsity**: {100 * sindy_info['mean_sparsity']:.1f}%\n")
        if sim_metrics:
            f.write(f"- **Simulation stability**: {sim_metrics['num_stable']}/{sim_metrics['num_attempted']} sequences\n")
        f.write("\nSee `validation_report.txt` for detailed results.\n")
    
    print(f"\nSummary report saved to {report_path}")
    print(f"README saved to {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate trained SINDySz convolutional autoencoder model"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Automatically find and use the latest checkpoint"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="/app/Data/WR/WR5_Run4.hdf5",
        help="Path to HDF5 data file"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default="/app/Data/WR/Annotations/260218_annotations_a.pkl",
        help="Path to annotation pickle file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: validation_results/<timestamp>)"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=3,
        help="Number of sequences to visualize (default: 3)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Sparsity threshold for SINDy coefficients (default: 1e-3)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.checkpoint is None and not args.auto_discover:
        print("Error: Must specify either --checkpoint or --auto-discover", file=sys.stderr)
        return 1
    
    if args.checkpoint is not None and args.auto_discover:
        print("Error: Cannot specify both --checkpoint and --auto-discover", file=sys.stderr)
        return 1
    
    try:
        # Find checkpoint
        if args.auto_discover:
            checkpoint_path = find_latest_checkpoint()
            print(f"Auto-discovered checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.checkpoint
            if not checkpoint_path.exists():
                print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
                return 1
        
        # Create output directory
        if args.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("validation_results") / timestamp
        else:
            output_dir = args.output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # Load model and data
        model, dataset, train_set, valid_set, test_set = load_checkpoint_and_data(
            checkpoint_path, args.data_file, args.annotation_file, args.device
        )
        
        # Task 1: Reconstruction validation
        recon_metrics = validate_reconstruction(
            model, valid_set, output_dir, args.num_sequences, args.device
        )
        
        # Task 2: SINDy equation extraction
        sindy_info = extract_sindy_equations(
            model, output_dir, args.threshold
        )
        
        # Task 3: Latent dynamics simulation
        sim_metrics = simulate_dynamics(
            model, valid_set, output_dir, args.num_sequences, args.device
        )
        
        # Task 4: Generate summary report
        generate_summary_report(
            output_dir, checkpoint_path, model, dataset,
            train_set, valid_set, test_set,
            recon_metrics, sindy_info, sim_metrics
        )
        
        print("\n" + "="*60)
        print("VALIDATION COMPLETE")
        print("="*60)
        print(f"\nAll results saved to: {output_dir}")
        print("\nKey files:")
        print(f"  - {output_dir / 'validation_report.txt'}")
        print(f"  - {output_dir / 'README.md'}")
        print(f"  - {output_dir / 'reconstructions' / 'reconstruction_summary.png'}")
        print(f"  - {output_dir / 'sindy_equations' / 'sindy_coefficients_heatmap.png'}")
        print(f"  - {output_dir / 'dynamics_simulation' / 'aggregate_simulation_analysis.png'}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
