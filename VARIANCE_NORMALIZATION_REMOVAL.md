# Variance Normalization Removal - Summary

## Problem Identified (Version 12)

After fixing the time step issue (dt=3.0s instead of 0.0002s), a new problem emerged:

```
train_R2_zdot: -54,793 to -73,576 (catastrophic!)
train_z_dot_var: 1e-6 (clamped minimum)
train_zdot_mse_unnorm: ~0.05
Normalized zdot loss: 0.05 / 1e-6 = 50,000 (explosion!)
```

## Root Cause

**Variance normalization backfires when derivatives have near-zero variance.**

With the correct 3-second time step between bicoherence maps:
- Latent codes `z` change very slowly
- Therefore `z_dot ≈ 0` (nearly constant)
- Variance of z_dot ≈ 1e-6 (essentially zero)
- Dividing by this tiny variance causes loss explosion

The variance normalization was well-intentioned (make losses scale-invariant), but it assumes the variance is "reasonably large". When variance is genuinely tiny, it's telling us **there's no temporal dynamics to learn**, and we shouldn't penalize the model heavily for not fitting near-zero values.

## Solution: Remove Variance Normalization

Use **unnormalized MSE losses** with appropriately tuned lambda weights.

### Changes Made

#### 1. Loss Function Classes Updated

All three loss classes now use unnormalized MSE:

**SINDyPathLoss** (dual-optimizer mode, SINDy path):
```python
self.lambda2 = 50.0   # xdot weight (was 1)
self.lambda3 = 2.0    # zdot weight (was 1)
self.lambda4 = 0.01   # regularization

# Loss computation (no variance division)
sindy_loss_xdot = self.lambda2 * xdot_mse
sindy_loss_zdot = self.lambda3 * zdot_mse
```

**DecoderPathLoss** (dual-optimizer mode, decoder path):
```python
self.lambda1 = 1.0    # reconstruction weight

# Loss computation (no variance division)
recon_loss = self.lambda1 * recon_mse
```

**SINDyLoss** (single-optimizer mode):
```python
self.lambda1 = 1.0    # reconstruction weight
self.lambda2 = 50.0   # xdot weight (was 1)
self.lambda3 = 2.0    # zdot weight (was 1)
self.lambda4 = 0.01   # regularization

# All losses use unnormalized MSE
```

#### 2. Lambda Weight Rationale

Weights chosen based on empirical MSE scales from version 12:
- `recon_mse` ≈ 0.14
- `xdot_mse` ≈ 0.002 (67× smaller than recon)
- `zdot_mse` ≈ 0.05 (2.6× smaller than recon)

Lambda weights scale up the smaller terms to make all contributions comparable:
- `lambda1 = 1.0` (baseline)
- `lambda2 = 50.0` (boost xdot 50×)
- `lambda3 = 2.0` (boost zdot 2×)

#### 3. R² Computation Updated

R² is still computed for monitoring, but now uses the true variance (not clamped):

```python
"R2_xdot": 1.0 - xdot_mse / max(x_dot_var, 1e-9)
"R2_zdot": 1.0 - zdot_mse / max(z_dot_var, 1e-9)
```

When variance is near-zero, R² becomes meaningless (will be large negative), but that's okay - it's just a monitoring metric.

#### 4. Variance Still Logged

Variances are still computed and logged as `x_var`, `x_dot_var`, `z_dot_var` for diagnostics, but they're **not used in the loss computation**.

## Expected Behavior After Fix

### Immediately After Training Starts

**Version 12 (with variance normalization)**:
```
train_sindyzdot_loss: 54,795 (exploded)
train_R2_zdot: -54,793 (catastrophic)
Total loss dominated by exploded zdot term
```

**Version 13 (without variance normalization)**:
```
train_sindyzdot_loss: 2.0 × 0.05 = 0.1 (reasonable)
train_R2_zdot: ~0 initially (meaningful when variance is non-zero)
Total loss balanced across all terms
```

### As Training Progresses

All loss terms should decrease:
- `recon_loss`: 1.0 × recon_mse (starts ~0.14, should decrease)
- `sindy_loss_xdot`: 50.0 × xdot_mse (starts ~0.11, should decrease)
- `sindy_loss_zdot`: 2.0 × zdot_mse (starts ~0.1, should decrease)
- `sindy_regularization`: 0.01 × |weights|

Total loss should start around **0.14 + 0.11 + 0.1 + reg ≈ 0.35-0.4** and decrease.

### R² Metrics

- `R2_recon`: Should improve from ~0 to positive values
- `R2_xdot`: Should improve if x changes enough between maps
- `R2_zdot`: May remain near 0 if z truly changes slowly (this is fine!)

## Justification for This Approach

### Why Remove Variance Normalization?

1. **Principled**: When variance is genuinely small, there's no signal to fit
2. **Simpler**: Unnormalized MSE is more interpretable and predictable
3. **Robust**: No more division-by-near-zero pathologies
4. **Practical**: Lambda tuning is needed anyway; variance normalization doesn't eliminate it

### Why Not Use Adaptive Normalization?

Adaptive approaches (e.g., "use normalization when variance > threshold") would:
- Change loss behavior based on runtime statistics (unpredictable)
- Add complexity and hyperparameters
- Still require lambda tuning anyway

### Why These Specific Lambda Values?

Based on empirical MSE scales from version 12 data:
- Bring all terms to similar magnitude (~0.1-0.14)
- Allow all terms to contribute meaningfully to gradients
- Can be adjusted if one term dominates or is ignored during training

## Files Modified

- `model.py` lines 797-804: SINDyLoss.__init__
- `model.py` lines 894-919: SINDyLoss.forward (loss computation)
- `model.py` lines 930-955: SINDyLoss diagnostics
- `model.py` lines 989-999: SINDyPathLoss.__init__
- `model.py` lines 1044-1064: SINDyPathLoss.forward (loss computation)
- `model.py` lines 1076-1097: SINDyPathLoss diagnostics
- `model.py` lines 1129-1154: DecoderPathLoss.forward

## Verification Steps

After retraining with version 13:

1. ✓ Check that `train_sindyzdot_loss` is ~0.1-0.2 (not 50,000!)
2. ✓ Check that `train_R2_zdot` is reasonable (not -50,000!)
3. ✓ Check that `train_total_loss` starts around 0.35-0.5 and decreases
4. ✓ Check that all loss components contribute (none dominates)
5. ✓ Monitor R² metrics improve over training
6. ✓ Check final reconstruction quality is maintained or improved

---

**Date**: 2026-07-28  
**Issue**: Variance normalization explosion when z_dot has near-zero variance  
**Solution**: Remove variance normalization, use weighted unnormalized MSE  
**Key Change**: lambda2=50.0, lambda3=2.0 (instead of variance division)
