# Derivative Explosion Fix - Summary

## Problem Identified

The SINDy dynamics section was experiencing severe derivative explosion while the reconstruction section worked well. Analysis of `lightning_logs/conv_masked_ae/lightning_logs/version_11/metrics.csv` revealed:

### Symptoms
1. **Reconstruction R² improving nicely**: from -0.89 to +0.68 over 54 steps
2. **Dynamics R² stuck near zero or negative**: 
   - `train_R2_xdot`: oscillating around 0.000 to -0.006
   - `train_R2_zdot`: oscillating around -0.19 to +0.19
3. **Explosive derivative variances**:
   - `train_x_dot_var`: ~400,000-570,000 (massive)
   - `train_z_dot_var`: **exploding from 5 to 92,000** in first 15 steps
   - `train_zdot_mse_unnorm`: 6→18→73→96→106→320→706→1,576→4,344→22,444

### Root Cause

**CRITICAL ERROR**: The finite difference calculation was using the raw EEG sample rate (5000 Hz) instead of the actual time step between consecutive bicoherence maps.

- **Actual setup**: 
  - Each bicoherence map is computed over a 5-second window (`epoch_size=5.0`)
  - Consecutive maps in a sequence are separated by **3 seconds**
  - The SINDy model operates on sequences of 8 consecutive maps

- **What the code was doing**:
  ```python
  sample_rate = 5000  # Hz (EEG sampling rate)
  dt = 1.0 / sample_rate  # = 0.0002 seconds
  ```

- **What it should have been doing**:
  ```python
  map_time_step = 3.0  # seconds between consecutive bicoherence maps
  dt = map_time_step  # = 3.0 seconds
  ```

**Scale of error**: The derivatives were being computed with a time step that was **15,000× too small**, causing them to be **15,000× too large**!

## Secondary Issues

1. **Variance normalization clamp too small**: `clamp_min(1e-12)` caused extreme sensitivity when `z_dot_var` was small (~5-6 initially)
2. **No gradient clipping**: Allowed explosive gradients to propagate unchecked
3. **Potential loss weight imbalance**: All λ weights set to 1, but x_dot variance is 100,000× larger than z_dot initially

## Fixes Applied

### 1. Correct Time Step (CRITICAL FIX)

**File**: `main.py` lines 38-44, 107-111

```python
# Define the actual inter-map time step
map_time_step = 3.0  # seconds between consecutive bicoherence maps

# Pass it to SINDySz as a "sample rate" (frequency = 1/period)
sindy_sz = SINDySz(
    ...
    sample_rate=(1.0 / map_time_step),  # = 0.333 Hz
    ...
)
```

The loss functions compute `dt = 1/fs`, so:
- `fs = 1/3.0 = 0.333...` Hz
- `dt = 1/0.333 = 3.0` seconds ✓

### 2. Increased Variance Clamp

**File**: `model.py` lines 896-904, 1046-1053, 1136-1140

Changed all variance clamps from `1e-12` to `1e-6`:
```python
x_var = x.detach().var().clamp_min(1e-6)       # was 1e-12
x_dot_var = x_dot.detach().var().clamp_min(1e-6)  # was 1e-12
z_dot_var = z_dot.detach().var().clamp_min(1e-6)  # was 1e-12
```

This prevents extreme loss sensitivity when variances are small.

### 3. Gradient Clipping

**File**: `model.py` lines 1675-1678, 1688-1691

Added gradient norm clipping after each backward pass in dual-optimizer mode:

```python
self.manual_backward(sindy_loss)
# Prevent derivative explosion
torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(self.sindy_model.parameters(), max_norm=1.0)
opt_sindy.step()
```

```python
self.manual_backward(decoder_loss)
# Prevent explosion
torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
opt_decoder.step()
```

## Expected Impact

With `dt` corrected from 0.0002s to 3.0s, the finite differences will be divided by **15,000** instead of multiplied:

- **Before**: `derivative = Δvalue / 0.0002` → huge numbers
- **After**: `derivative = Δvalue / 3.0` → reasonable numbers

This should:
1. Bring `x_dot_var` down from ~500,000 to ~33 (÷15,000)
2. Bring `z_dot_var` down from exploding 5→92,000 to stable ~0.0003→6
3. Allow the SINDy dynamics to actually learn meaningful patterns
4. Improve `R2_xdot` and `R2_zdot` from ~0 to positive values

## Verification Steps

After retraining with these fixes:

1. Check that `train_x_dot_var` is in the range 10-100 (not 400,000-500,000)
2. Check that `train_z_dot_var` doesn't explode (should stay < 100)
3. Check that `train_R2_xdot` and `train_R2_zdot` improve over training
4. Check that `train_xdot_mse_unnorm` and `train_zdot_mse_unnorm` decrease
5. Monitor gradient norms don't exceed the clipping threshold consistently

## Additional Recommendations (Not Implemented)

If issues persist after the critical fix, consider:

1. **Loss weight rebalancing**: 
   ```python
   self.lambda2 = 0.1   # reduce xdot weight
   self.lambda3 = 2.0   # increase zdot weight
   ```

2. **Input normalization**: Ensure bicoherence maps are properly scaled (e.g., to [0,1] or standardized)

3. **Learning rate reduction**: If training is unstable, reduce `lr` from 0.001 to 0.0001

4. **Longer warmup**: Consider training decoder-only for a few epochs before enabling SINDy loss

---

**Date**: 2026-07-28  
**Analysis**: Based on metrics from `version_11`  
**Critical Fix**: dt = 3.0s instead of 0.0002s (15,000× correction)
