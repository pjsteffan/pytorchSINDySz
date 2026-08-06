# Model Validation Guide

This guide explains how to validate a trained SINDySz convolutional autoencoder model using `validate_model.py`.

## Quick Start

### Using the latest checkpoint automatically:
```bash
uv run python validate_model.py --auto-discover --num-sequences 5
```

### Using a specific checkpoint:
```bash
uv run python validate_model.py \
    --checkpoint lightning_logs/conv_masked_ae/lightning_logs/version_13/checkpoints/epoch=7-step=2352.ckpt \
    --num-sequences 5
```

### Custom output directory:
```bash
uv run python validate_model.py \
    --auto-discover \
    --num-sequences 5 \
    --output-dir my_validation_results
```

## What the Script Does

The validation script performs three comprehensive analyses:

### 1. Reconstruction Validation
- **Purpose**: Verify that the autoencoder can reconstruct input bicoherence maps
- **Outputs**:
  - Individual sequence visualizations (input vs reconstruction side-by-side)
  - MSE and R² metrics per sequence
  - Aggregate statistics and distributions
  - Per-time-step error analysis

**Key Metrics**:
- Mean MSE: ~0.031 (lower is better)
- Mean R²: ~0.20-0.22 (closer to 1.0 is better)

### 2. SINDy Equation Extraction
- **Purpose**: Extract and visualize the discovered dynamical system equations
- **Outputs**:
  - Human-readable equations showing discovered dynamics
  - Coefficient matrix heatmap
  - Sparsity analysis (number of active terms per dimension)

**Key Findings**:
- Mean sparsity: ~16.5% (only 6-7 terms active out of 40 possible)
- Dominant terms: `sin_φ(z_i)` terms (Hilbert phase features)
- Each latent dimension has a unique sparse equation

**Example Equation**:
```
dz3/dt = -0.087822·sin_φ(z3)
```

### 3. Latent Dynamics Simulation
- **Purpose**: Simulate dynamics forward in time using SINDy equations and compare to ground truth
- **Outputs**:
  - Latent space trajectories (SINDy vs ground truth)
  - Decoded reconstructions (both from ground-truth and simulated latents)
  - Error over time analysis
  - Stability metrics

**Key Metrics**:
- Stable simulations: 100% (all sequences stable)
- Mean latent error: ~0.93 (L2 distance in latent space)
- Mean reconstruction error: ~0.042 MSE (comparable to direct autoencoder)

**Integration Details**:
- Method: Explicit Euler
- Time step: 3.0 seconds (matches training data)
- Initial condition: First time step of ground-truth sequence

## Output Directory Structure

After running validation, you'll get:

```
validation_results/<timestamp>/
├── README.md                           # Quick summary
├── validation_report.txt               # Comprehensive report
├── reconstructions/
│   ├── sequence_*_reconstruction.png   # Individual comparisons
│   ├── reconstruction_summary.png      # Aggregate metrics
│   └── reconstruction_metrics.txt      # Detailed metrics
├── sindy_equations/
│   ├── sindy_equations.txt             # Extracted equations
│   ├── sindy_coefficients_heatmap.png  # Coefficient visualization
│   └── sindy_sparsity_analysis.png     # Sparsity analysis
└── dynamics_simulation/
    ├── sequence_*_latent_trajectories.png  # Latent space plots
    ├── sequence_*_decoded_comparison.png   # Decoded comparisons
    ├── aggregate_simulation_analysis.png   # Error over time
    └── dynamics_simulation_metrics.txt     # Detailed metrics
```

## Command-Line Options

```
--checkpoint CHECKPOINT       Path to checkpoint file (.ckpt)
--auto-discover              Automatically find latest checkpoint
--data-file DATA_FILE        Path to HDF5 data (default: /app/Data/WR/WR5_Run4.hdf5)
--annotation-file FILE       Path to annotations (default: /app/Data/WR/Annotations/260218_annotations_a.pkl)
--output-dir DIR             Output directory (default: validation_results/<timestamp>)
--num-sequences N            Number of sequences to visualize (default: 3)
--device DEVICE              Device to use (default: cuda if available)
--threshold THRESHOLD        SINDy sparsity threshold (default: 1e-3)
```

## Interpreting Results

### Reconstruction Quality
- **Good**: MSE < 0.05, R² > 0.15
- **Acceptable**: MSE < 0.10, R² > 0.05
- **Poor**: MSE > 0.10 or R² < 0

### SINDy Sparsity
- **Highly sparse**: < 20% active terms (excellent interpretability)
- **Moderately sparse**: 20-40% active terms (good interpretability)
- **Dense**: > 40% active terms (limited interpretability)

### Dynamics Simulation Stability
- **Stable**: All sequences complete without NaN/Inf
- **Mostly stable**: > 90% sequences stable
- **Unstable**: < 90% sequences stable (may need smaller dt or different integrator)

### Simulation Accuracy
- **Excellent**: Latent error < 0.5, recon error < 1.5× direct reconstruction MSE
- **Good**: Latent error < 1.0, recon error < 2× direct reconstruction MSE
- **Fair**: Latent error < 2.0, recon error < 3× direct reconstruction MSE
- **Poor**: Latent error > 2.0 or recon error >> direct reconstruction MSE

## Model Insights from Validation

Based on the validation results, the trained model exhibits:

1. **Reasonable Reconstruction**: R² ~0.20 indicates the autoencoder captures ~20% of the variance in masked bicoherence maps. This is reasonable given the high dimensionality (19×19 masked grids) compressed to 5 latent dimensions.

2. **Sparse Dynamics**: Only 16.5% of library terms are active, indicating the model discovered interpretable, parsimonious equations. The dominant `sin_φ(z_i)` terms suggest phase-driven dynamics.

3. **Stable Forward Simulation**: All simulations complete without numerical instabilities, suggesting the discovered equations are physically reasonable and numerically well-behaved.

4. **Modest Long-Term Prediction**: Latent error grows over 8 time steps (~24 seconds total), but remains bounded. This is expected for complex EEG dynamics and suggests the model captures short-to-medium term dynamics.

## Troubleshooting

### "Checkpoint not found"
Ensure you're in the correct directory or specify the full path to the checkpoint.

### "No module named 'torch'"
Run with `uv run python` instead of `python3` directly.

### "Simulation became unstable"
- Try reducing the number of sequences: `--num-sequences 1`
- Consider implementing RK4 integration for better stability (currently uses Euler)
- Check if dt=3.0s is too large for your dynamics

### GPU out of memory
```bash
uv run python validate_model.py --auto-discover --device cpu
```

## Tips

1. **First run**: Use `--num-sequences 3` for quick validation (~2 minutes)
2. **Full validation**: Use `--num-sequences 10-20` for comprehensive analysis (~10 minutes)
3. **Publication figures**: Increase DPI by editing the script (search for `dpi=150`)
4. **Compare checkpoints**: Run validation on multiple checkpoints and compare the generated reports
5. **Custom analysis**: The script structure is modular—add your own analysis functions following the existing patterns

## Citation

If you use this validation script in your research, please cite:
- The original SINDy paper: Brunton et al. (2016)
- The bicoherence analysis method: Your EEG analysis paper
- This implementation: [Your repository/paper]

## Support

For issues or questions:
1. Check this guide first
2. Review the generated `validation_report.txt`
3. Examine individual plot files for visual diagnostics
4. Check the detailed metrics `.txt` files in each subdirectory
