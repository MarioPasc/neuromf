# Velocity Magnitude Overshoot

## Observation

Training dashboard at epoch ~700 shows a systematic overshoot in compound velocity magnitude:

- `||V|| / ||v_c||` ratio: **~1.2-1.4x** (should be ~1.0)
- The ratio is stable (not growing), suggesting a consistent bias rather than divergence
- FID continues improving despite the overshoot, indicating the direction is correct but magnitude is inflated

## Possible Causes

### 1. JVP Correction Term Adding Magnitude

The compound velocity is `V = u + (t-r) * sg[du/dt]`. The JVP correction term `(t-r) * du/dt` adds magnitude on top of `u`. If the JVP term has a positive projection onto `u`, `||V|| > ||u||` systematically.

This is expected behavior for MF samples (where `r < t`), but the magnitude excess suggests the correction overshoots the target `v_c`.

### 2. Equal u/v Loss Weighting

The dual-head loss is `loss = loss_u + loss_v` with equal weighting (lambda=1 for both). If the v-head loss dominates gradients early on, the u-head may under-optimize, leaving residual magnitude error in `V`.

### 3. Adaptive Weighting Interaction

Adaptive weighting normalizes by `(raw_loss + eps)^norm_p`. If the magnitude error is correlated with the adaptive weight, samples with larger overshoot may receive less gradient signal, preserving the bias.

## Potential Mitigations (Future Work)

1. **Configurable `lambda_v`**: Weight the v-head loss relative to u-head loss. Try `lambda_v` in {0.1, 0.5, 1.0, 2.0} to find optimal balance.

2. **Velocity norm regularization**: Add `lambda_norm * (||V|| / ||v_c|| - 1)^2` to the loss. This directly penalizes magnitude mismatch without affecting direction.

3. **Separate norm_p for u/v heads**: Allow different adaptive weighting exponents for each head, e.g. `norm_p_u=1.0, norm_p_v=0.5`.

4. **Investigate per-channel overshoot**: Check if all 4 latent channels overshoot equally or if specific channels are responsible.

## Priority

Low. FID continues improving through epoch 700 despite the overshoot. The direction alignment (`cosine_sim_V_vc > 0.99`) is excellent. This is worth investigating after the current training run completes, especially if FID plateaus before reaching the target.
