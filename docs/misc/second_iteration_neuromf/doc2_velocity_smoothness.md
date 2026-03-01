# NeuroMF Enhancement: Velocity Field Smoothness Regularization

**Project:** NeuroMF — Improved MeanFlow for 3D Brain MRI Synthesis
**Author:** Mario Pascual-González
**Date:** 2026-03-01 (revised)
**Status:** SPECIFICATION READY — awaiting implementation
**Depends on:** v2 config changes (doc1) applied; Phase 5 gate OPEN
**Scope:** Auxiliary regularizer on the v-head velocity field, applied during training
**Proposal origin:** Ezequiel López-Rubio (PhD thesis advisor)
**Revision notes:** Corrected computational cost estimates, added FD-based implementation (preferred over exact JVP for memory), dimension-normalized loss, delayed activation, codebase-specific integration, data-driven calibration.

---

## 0. Preamble

This document specifies a Lipschitz smoothness regularizer on the instantaneous velocity field $v_\theta(z_t, t)$ produced by the auxiliary v-head. For a fixed time $t$, the velocity field should vary smoothly with respect to its spatial argument $z_t$. Nearby latent points should be assigned similar velocities.

**Why this matters for MeanFlow:** The v-head output serves as the JVP tangent vector in the compound velocity:

$$V_\theta = u_\theta + (t - r) \cdot \text{sg}\!\left[\frac{\partial u_\theta}{\partial z_t} v_\theta + \frac{\partial u_\theta}{\partial t}\right]$$

The spatial Jacobian term $\frac{\partial u_\theta}{\partial z_t} v_\theta$ is the component most affected by v-head noise. If $v_\theta$ has high-frequency oscillations, the JVP output has high variance, degrading the compound velocity $V_\theta$ and the MF self-consistency loss signal. From v1 diagnostics at epoch 689: $\cos(V, v_c) = 0.307$ while $\cos(\tilde{v}, v_c) = 0.436$ — the compound velocity has *worse* directional alignment than the raw v-head, suggesting the JVP amplifies noise.

**Scope:** The regularizer adds a single loss term $\lambda_s \mathcal{L}_{\text{smooth}}$ to the total loss. No architectural changes. No new modules. Approximately one extra UNet forward pass per training step.

---

## 1. Theoretical Justification

### 1.1 Velocity Field Regularity

In Flow Matching (Lipman et al., ICLR 2023), the marginal velocity field is:

$$v(z_t, t) \triangleq \mathbb{E}[\epsilon - x \mid z_t]$$

By properties of conditional expectations, if $p_{\text{data}}$ has smooth density, the mapping $z_t \mapsto v(z_t, t)$ is Lipschitz continuous:

$$\|v(z_t, t) - v(z_t', t)\|_2 \leq L(t) \|z_t - z_t'\|_2$$

The trained network $v_\theta$ should approximate this property. Enforcing it explicitly regularizes the v-head toward the true velocity field structure.

### 1.2 Jacobian Regularization for Neural ODEs

Finlay et al. ("How to Train Your Neural ODE", ICML 2020) demonstrate that penalizing the Frobenius norm of the velocity field Jacobian:

$$R_{\text{Jacobian}} = \mathbb{E}_{t, z_t}\!\left[\left\|\frac{\partial v_\theta(z_t, t)}{\partial z_t}\right\|_F^2\right]$$

leads to straighter ODE trajectories with lower kinetic energy. Their Theorem 1 shows this is equivalent to minimizing path curvature:

$$\kappa[z] = \int_0^1 \left\|\frac{d^2 z_t}{dt^2}\right\|^2 dt$$

Lower curvature means smaller Euler discretization error at any step size, improving all NFE levels.

### 1.3 Hutchinson Trace Estimator

Computing the full Frobenius norm $\|\partial_{z_t} v_\theta\|_F^2$ requires the Jacobian matrix $J \in \mathbb{R}^{d \times d}$ with $d = 4 \times 48^3 = 442{,}368$. This is intractable. The Hutchinson trace estimator (1989) provides an unbiased estimate:

$$\|J\|_F^2 = \text{Tr}(J^\top J) = \mathbb{E}_{\xi \sim \mathcal{N}(0, I)}\!\left[\|J \xi\|^2\right]$$

A single draw $\xi$ gives an unbiased estimator with relative variance $2/d$, which is negligible for $d = 442{,}368$ (Grathwohl et al., FFJORD, ICLR 2019).

### 1.4 Supporting Evidence from v1 Training

From the v1 main training run (x-pred, exact JVP, dual-head, 690 epochs):

| Metric | Epoch 0 | Epoch 46 | Epoch 689 | Interpretation |
|--------|---------|----------|-----------|----------------|
| $\cos(V, v_c)$ | 0.083 | 0.242 | 0.307 | Compound velocity alignment — poor |
| $\cos(\tilde{v}, v_c)$ | 0.171 | 0.390 | 0.436 | v-head alignment — modest |
| JVP temporal frac | 0.793 | 0.728 | 0.630 | Temporal component dominates JVP |
| Compound V norm | 855 | 1403 | 1092 | Norm ratio V/v_c: 0.91→1.49→1.16 |
| raw_loss_v | 859K | 740K | 709K | v-head converged early (by epoch 46) |

**Key observations:**

1. **cos(V, v_c) < cos(v_tilde, v_c) at epoch 689** (0.307 < 0.436): The JVP-based correction *degrades* directional alignment. This indicates the JVP tangent quality is the bottleneck.

2. **raw_loss_v plateaued by epoch 46** (740K → 709K over 644 epochs): The v-head converged rapidly to a moderate fit and barely improved afterward. With only 228K trainable parameters (0.6% of total), it may be capacity-limited. **Caution:** The smoothness regularizer further constrains the v-head, which could hurt fitting. Monitor `raw_loss_v` carefully.

3. **Temporal JVP fraction = 63-79%** (v1 main run): The temporal component $\partial u / \partial t$ dominates the JVP. The spatial component $(\partial u / \partial z_t) v_\theta$ is 21-37% of the total JVP norm. This means our regularizer targets a meaningful (though minority) component.

---

## 2. Mathematical Formulation

### 2.1 Loss Definition

$$\mathcal{L}_{\text{smooth}} = \frac{1}{d}\,\mathbb{E}_{t, z_t, \xi}\!\left[\|J_v \xi\|_2^2\right], \quad J_v \triangleq \frac{\partial v_\theta(z_t, t)}{\partial z_t}, \quad \xi \sim \mathcal{N}(0, I_d)$$

where $d = 4 \times 48^3 = 442{,}368$ is the latent dimensionality. The $1/d$ normalization makes $\mathcal{L}_{\text{smooth}}$ represent the **average squared Jacobian entry**, independent of dimensionality.

**Why normalize by d:** Without normalization, $\mathbb{E}[\|J\xi\|^2] = \|J\|_F^2 = \sum_{ij} J_{ij}^2$, which scales with $d$. With our latent size, this produces values in the millions — the same order as `raw_loss_mf`. The adaptive-weighted main losses are ~1.0 each (by construction). Without normalization, any non-trivial $\lambda_s$ would dominate. With $1/d$ normalization, $\mathcal{L}_{\text{smooth}}$ represents the per-element Jacobian magnitude, yielding values of order $O(1)$.

### 2.2 Finite-Difference Approximation (Recommended Implementation)

The Hutchinson estimate can be computed via exact JVP (`torch.func.jvp`) or finite differences. **We recommend finite differences** due to memory constraints (see §5):

$$\|J_v \xi\|^2 \approx \frac{\|v_\theta(z_t + h\xi, t) - v_\theta(z_t, t)\|^2}{h^2}$$

with step size $h = 10^{-3}$ (matching the FD step used for compound velocity JVP).

**Key advantage:** $v_\theta(z_t, t)$ is already computed as `v_tangent` in the dual-head pipeline (`meanflow_loss.py:347`). We can reuse this value (detached), requiring only ONE additional forward pass for $v_\theta(z_t + h\xi, t)$.

**Bias:** The FD approximation has $O(h^2)$ bias. For $h = 10^{-3}$, this is negligible for regularization (we don't need an exact Jacobian norm — we need a training signal that penalizes non-smoothness). Finlay et al. (2020) use FD in their experiments.

### 2.3 Integration with Existing Loss

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{MF}}^{\text{adaptive}} + \mathcal{L}_{\text{FM}}^{\text{adaptive}}}_{\text{existing dual loss, each $\sim$1.0}} + \lambda_s \cdot \underbrace{\mathcal{L}_{\text{smooth}}}_{\text{NOT adaptive-weighted}}$$

The smoothness loss is a regularizer, not a data-fitting term. It should maintain consistent regularization strength regardless of per-sample loss magnitude.

### 2.4 Gradient Flow

With the FD approach and detached $v_\theta(z_t, t)$:

$$\nabla_\phi \mathcal{L}_{\text{smooth}} = \frac{2}{d \cdot h^2}\,\mathbb{E}\!\left[(v_\theta(z_t{+}h\xi) - v_\theta^{\text{sg}}(z_t)) \cdot \nabla_\phi\, v_\theta(z_t{+}h\xi)\right]$$

where $v_\theta^{\text{sg}}$ denotes the detached (stop-gradient) reuse. Gradients flow through the forward pass of $v_\theta(z_t + h\xi, t)$, which traverses the **entire UNet backbone + v-head projection**. This means:

1. The regularizer trains both the v-head (228K params) AND the shared backbone (~38M params)
2. The shared backbone is indirectly regularized toward smoother features, benefiting the u-head too
3. The gradient from $\mathcal{L}_{\text{smooth}}$ competes with the MF/FM gradients at the backbone level

---

## 3. Implementation Specification

### 3.1 Pre-Implementation Checks

| ID | Check | Location | What to verify |
|----|-------|----------|----------------|
| C1 | v_tangent availability | `meanflow_loss.py:346-347` | `v_tangent = dual_fn(z_t, t, t)[1]` is computed under `no_grad`. Confirm it can be reused as the detached $v_\theta(z_t, t)$. |
| C2 | Dual-head forward API | `maisi_unet.py:421-476` | `model(z_t, r=t, t=t)` returns `(x_hat, v)`. Confirm `v` is the v-head output. |
| C3 | FD step size config | `configs/train_meanflow.yaml:93` | Existing `fd_step_size: 0.001` — reuse this value. |
| C4 | Training step injection point | `latent_meanflow.py:204` | `loss = result["loss"]` — this is where we add `+ lambda_s * L_smooth`. |
| C5 | Gradient checkpointing state | `maisi_unet.py:313-319` | Verify checkpointing is disabled (incompatible with existing JVP but irrelevant for FD smoothness). |
| C6 | v_tangent returned in result dict? | `meanflow_loss.py:370-405` | Check if v_tangent is in the result dict. If not, we need to pass it out or recompute. |

**C6 is critical:** The current `_forward_dual_head` does NOT return `v_tangent` in the result dict. Either:
- (a) Add `result["v_tangent"] = v_tangent` in `_forward_dual_head` (preferred — minimal change), or
- (b) Recompute with `model(z_t, r=t, t=t)` in `training_step` (expensive — full extra forward pass)

### 3.2 Core Implementation

**New file:** `src/neuromf/losses/smoothness_loss.py`

```python
"""Velocity field smoothness loss via finite-difference Hutchinson estimator.

Penalizes the Frobenius norm of the v-head Jacobian w.r.t. z_t, estimated
stochastically via a single Hutchinson probe with finite differences.
Encourages Lipschitz continuity of the instantaneous velocity field.

The FD approach is preferred over exact JVP for memory efficiency:
- Exact JVP: ~1.5x forward pass memory (forward-mode AD tape)
- FD: ~1x forward pass memory (no AD overhead)

Reference: Finlay et al., "How to Train Your Neural ODE", ICML 2020.
Reference: Hutchinson, "A stochastic estimator of the trace", 1989.
"""

import torch
from torch import Tensor
from collections.abc import Callable


def velocity_smoothness_loss(
    v_fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    z_t: Tensor,
    t: Tensor,
    v_original: Tensor,
    fd_step: float = 1e-3,
) -> dict[str, Tensor]:
    """Compute FD Hutchinson estimate of ||dv/dz_t||_F^2 / d.

    Reuses the already-computed v_original = v_theta(z_t, t) (detached)
    to avoid a redundant forward pass.

    Args:
        v_fn: The dual-head model callable (z_t, r, t) -> (x_hat, v).
            Used as v_fn(z_t_perturbed, t, t) to get v at the perturbed point.
            Note: r=t for instantaneous velocity (h=0).
        z_t: Noisy latent (B, C, D, H, W).
        t: Time values (B,).
        v_original: Detached v_theta(z_t, t) from the main loss computation.
            Shape (B, C, D, H, W). MUST be detached.
        fd_step: Finite-difference step size (default: 1e-3).

    Returns:
        Dict with keys:
            "loss": scalar smoothness loss (normalized by d),
            "jac_frob_norm_est": sqrt of unnormalized ||J xi||^2 mean (for logging).
    """
    d = z_t[0].numel()  # 4 * 48^3 = 442368

    # Hutchinson probe: fresh noise independent of training epsilon
    xi = torch.randn_like(z_t)

    # Perturbed forward pass (WITH grad — this is the differentiable path)
    z_t_perturbed = z_t.detach() + fd_step * xi
    _, v_perturbed = v_fn(z_t_perturbed, t, t)  # r=t for instantaneous

    # FD estimate of J @ xi
    jvp_est = (v_perturbed - v_original.detach()) / fd_step

    # ||J @ xi||^2 per sample, summed over (C, D, H, W), then mean over batch
    jac_norm_sq_per_sample = (jvp_est ** 2).sum(
        dim=tuple(range(1, jvp_est.ndim))
    )
    loss = jac_norm_sq_per_sample.mean() / d

    # Monitoring
    with torch.no_grad():
        jac_frob_norm_est = jac_norm_sq_per_sample.mean().sqrt()

    return {
        "loss": loss,
        "jac_frob_norm_est": jac_frob_norm_est,
    }
```

**Design decisions:**

1. **`v_fn` signature is `(z_t, r, t) -> (x_hat, v)`** — matches `model.forward()` directly. The caller passes `r=t` to get the instantaneous velocity. This avoids creating a wrapper function.

2. **`v_original` is passed in detached** — reused from `v_tangent` computed at `meanflow_loss.py:347`. Only one new forward pass needed.

3. **`z_t.detach()` in the perturbed input** — ensures we don't accidentally differentiate through the interpolation `z_t = (1-t)*z_0 + t*eps`. The smoothness loss should only affect the model parameters, not the data pipeline.

4. **Division by `d`** — normalizes the loss to per-element scale.

5. **Gradients flow through `v_perturbed`** only (not `v_original`). This is asymmetric but correct: it pushes the model toward producing similar outputs for nearby inputs, without requiring exact matching of the unperturbed output.

### 3.3 Integration into Training Step

**Modify:** `src/neuromf/models/latent_meanflow.py`, in `training_step()` after line 204.

```python
# --- Existing code (lines 196-204) ---
result = self.loss_pipeline(self.net, z_0, eps, t, r, ...)
loss = result["loss"]

# --- NEW: Smoothness regularizer ---
if self._smooth_enabled and self.global_step >= self._smooth_start_step:
    from neuromf.losses.smoothness_loss import velocity_smoothness_loss

    smooth_result = velocity_smoothness_loss(
        v_fn=self.net,                         # model(z_t, r, t) -> (x_hat, v)
        z_t=result["_z_t"],                    # interpolated latent (must be added to result dict)
        t=t,
        v_original=result["_v_tangent"],       # detached v(z_t, t) (must be added to result dict)
        fd_step=self._smooth_fd_step,
    )
    loss_smooth = smooth_result["loss"]
    loss = loss + self._lambda_smooth * loss_smooth

    self.log("train/loss_smooth", loss_smooth.detach(), sync_dist=True)
    self.log("train/loss_smooth_weighted", (self._lambda_smooth * loss_smooth).detach(), sync_dist=True)
    self.log("train/jac_frob_norm", smooth_result["jac_frob_norm_est"], sync_dist=True)
```

**Required plumbing changes:**

1. **In `_forward_dual_head` (meanflow_loss.py:370-405):** Add to result dict:
   ```python
   result["_v_tangent"] = v_tangent.detach()  # already detached (computed under no_grad)
   result["_z_t"] = z_t.detach()              # detached interpolated latent
   ```

2. **In `LatentMeanFlow.__init__` (latent_meanflow.py):** Parse smoothness config:
   ```python
   smooth_cfg = config.get("smoothness", {})
   self._smooth_enabled = bool(smooth_cfg.get("enabled", False))
   self._lambda_smooth = float(smooth_cfg.get("lambda_smooth", 0.01))
   self._smooth_fd_step = float(smooth_cfg.get("fd_step", 1e-3))
   self._smooth_start_step = int(smooth_cfg.get("start_step", 1000))
   ```

### 3.4 Config Changes

In `configs/train_meanflow.yaml`:

```yaml
smoothness:                          # NEW section
  enabled: false                     # Disabled by default; enable for v2+ experiments
  lambda_smooth: 0.01               # Weight relative to adaptive losses (~1.0 each)
  fd_step: 0.001                    # FD step size (matches existing fd_step_size)
  start_step: 1000                  # Delay activation until after LR warmup
  log_every_n_steps: 50             # Log smoothness loss to TensorBoard
```

### 3.5 Delayed Activation Rationale

The regularizer is disabled for the first 1000 steps (matching LR warmup). At step 0, gradient clipping fraction is 90.5% (from v1 epoch 0 diagnostics). Adding smoothness gradients during this chaotic phase would further destabilize training. By step 1000 (~24 epochs), the gradient clipping fraction drops to ~20% and the model has a stable gradient flow.

### 3.6 Burn-In Calibration Procedure

**Before the first v2 training run with smoothness enabled:**

1. Run 50 training steps with `enabled: true` but with the smoothness loss LOGGED but NOT added to total loss (set `lambda_smooth: 0.0` but still compute and log the value)
2. Observe the magnitude of `train/loss_smooth` (dimension-normalized)
3. Set $\lambda_s$ so that $\lambda_s \cdot \mathcal{L}_{\text{smooth}} \approx 0.1$

This ensures the regularizer contributes ~5% of the total loss ($\sim$0.1 out of $\sim$2.0), acting as a gentle constraint.

**Alternative (if burn-in is not practical):** Start with $\lambda_s = 0.01$. If `train/loss_smooth_weighted` exceeds 0.5 (i.e., >25% of total), reduce by 10x. If below 0.001, increase by 10x. Monitor during first 100 epochs.

---

## 4. Hyperparameter Analysis

### 4.1 Choice of $\lambda_s$

The smoothness loss $\mathcal{L}_{\text{smooth}} = \frac{1}{d}\mathbb{E}[\|J\xi\|^2]$ has units of (velocity / latent)$^2$ — it's the average squared Jacobian entry. The FM loss after adaptive weighting is ~1.0 (dimensionless). Therefore $\lambda_s$ controls the balance between fitting the velocity target and keeping the Jacobian small.

**Expected magnitude of $\mathcal{L}_{\text{smooth}}$:** For a randomly initialized network with moderate weights, each Jacobian entry $J_{ij} \sim O(0.01)$, giving $\mathcal{L}_{\text{smooth}} \sim 0.01^2 = 10^{-4}$. For a trained network with sharper features, $J_{ij}$ could be $O(0.1)$, giving $\mathcal{L}_{\text{smooth}} \sim 0.01$. With $\lambda_s = 0.01$, the contribution is $O(10^{-4})$ to $O(10^{-4})$ — likely negligible. **This will need calibration via the burn-in procedure.**

### 4.2 V-Head Capacity Risk

The v-head has 228K parameters (1 ResBlock + projection, branching after the shared backbone's output feature map `h`). The FM loss `raw_loss_v` plateaued at ~709K by epoch 46 and barely improved to epoch 689. This suggests the v-head is capacity-limited.

Adding a smoothness constraint reduces the effective capacity of the v-head to fit v_c. **If `raw_loss_v` increases after enabling smoothness**, consider:

| Mitigation | Config change | Effect |
|------------|--------------|--------|
| Increase v-head capacity | `v_head_num_res_blocks: 2` | +228K params, ~2x v-head capacity |
| Reduce smoothness weight | `lambda_smooth: 0.001` | 10x weaker constraint |
| Late activation | `start_step: 5000` | Let v-head fit first |

### 4.3 FD Step Size

The step $h = 10^{-3}$ matches the existing `fd_step_size` for the compound velocity FD-JVP. For the smoothness loss:

- $h = 10^{-3}$: FD bias $\sim O(h^2) = O(10^{-6})$ — negligible relative to the estimator variance
- $h = 10^{-2}$: Larger bias ($10^{-4}$) but more robust numerically. Use if bf16 mixed precision causes issues with the subtraction $v(z_t + h\xi) - v(z_t)$

The subtraction should be done in fp32 (matching `jvp_strategies.py:166`).

---

## 5. Memory Budget

### 5.1 Current Memory Footprint (A100 40GB, batch_size=2)

| Component | Estimate |
|-----------|----------|
| Model params (fp32) + optimizer (Adam) | ~12 GB |
| Forward + exact JVP activations | ~20 GB |
| Total (existing) | ~32 GB |
| **Headroom** | **~8 GB** |

### 5.2 Additional Cost of Smoothness Regularizer

| Approach | Additional Memory | Fits? |
|----------|-------------------|-------|
| **Exact JVP** (`torch.func.jvp` through backbone+v-head) | ~8-10 GB | **NO — OOM** |
| **FD with grad** (one forward pass, computation graph stored) | ~5-6 GB | Tight but likely OK |
| **FD with no_grad** (no backprop through regularizer) | ~2-3 GB | **YES — safe** |

**Recommended:** FD with grad. If OOM, fall back to FD with no_grad (the regularizer still provides a useful monitoring signal even without gradients, though it won't train the model).

**If combined with perturbation consistency (doc3):** Both need ~5-6 GB each. Combined ~10-12 GB exceeds the 8 GB headroom. See §5.3.

### 5.3 Options for Running Both Regularizers

1. **Reduce batch_size to 1**, increase `accumulate_grad_batches` from 11 to 22. This halves activation memory (~10 GB), giving ~18 GB headroom. Both regularizers fit easily.
2. **Alternate steps:** Apply smoothness on even steps, perturbation on odd steps. Peak memory = one regularizer only.
3. **Implement independently first,** measure actual memory, then decide on combination strategy.

**Recommendation:** Option 3. Implement and ablate each independently before attempting to combine.

---

## 6. Verification Tests

All tests go in `tests/test_smoothness_loss.py`. Test naming: `test_P6_T{N}_SM_{description}`.

| Test ID | Description | Pass Criterion | Priority |
|---------|-------------|----------------|----------|
| SM-T1 | `velocity_smoothness_loss` returns dict with `"loss"` and `"jac_frob_norm_est"` for mock inputs `(2, 4, 8, 8, 8)` | Finite positive values, no errors | CRITICAL |
| SM-T2 | Gradient flows: `loss.backward()` produces non-zero gradients on v-head parameters | ≥90% of v-head params have `grad.norm() > 0` | CRITICAL |
| SM-T3 | For a known linear function $v(z) = Az + b$, compare FD estimate to $\|A\|_F^2 / d$ over 1000 draws | Relative error < 10% (FD bias allowed) | CRITICAL |
| SM-T4 | `lambda_smooth=0.0` does not change the total loss (exact match given same seed) | Bit-exact loss value | CRITICAL |
| SM-T5 | `start_step > global_step` skips the smoothness computation entirely | Loss matches baseline; no extra memory allocated | CRITICAL |
| SM-T6 | FD subtraction in fp32: verify no catastrophic cancellation with bf16 model outputs | `jvp_est` has finite values with variance > 0 | CRITICAL |
| SM-T7 | Memory: training step completes without OOM at `batch_size=2` per GPU (mock A100 test with reduced spatial size) | No CUDA OOM | CRITICAL |

**Note:** Informational tests (convergence behavior, cosine improvement) require actual training runs and belong in the ablation protocol (Phase 6), not unit tests.

---

## 7. Expected Outcomes

### 7.1 Training Behavior

- $\mathcal{L}_{\text{smooth}}$ should decrease over training as the v-head learns smoother spatial mappings
- $\mathcal{L}_{\text{FM}}$ (`raw_loss_v`) may increase slightly (capacity trade-off). If it increases by >20% relative to baseline, reduce $\lambda_s$
- The JVP norm variance (panel (f) in training dashboard) should decrease, indicating more stable compound velocity computation
- The compound velocity norm ratio $\|V\|/\|v_c\|$ may converge faster toward 1.0

### 7.2 Generation Quality

- **NFE=1:** Primary expected beneficiary. Smoother v-head → more reliable JVP tangent → better compound velocity at $(r, t) = (0, 1)$. Target: FID-3D improvement of 5-20% (conservative estimate given the modest spatial JVP contribution of 21-37%)
- **NFE ≥ 10:** Maintained or slightly improved (lower Euler discretization error from straighter trajectories)
- **MS-SSIM:** May improve due to more coherent structural generation

### 7.3 Ablation Value

Clean ablation axis for the paper: "with vs. without smoothness regularization" at each NFE level. If the spatial JVP fraction is small (21-37%), the effect may be modest but still publishable as a principled regularization approach.

### 7.4 Risks

1. **V-head under-capacity:** If smoothness + FM supervision exceeds the v-head's 228K-parameter capacity, both objectives suffer. Mitigation: increase to 2 ResBlocks.
2. **Backbone interference:** Smoothness gradients through the shared backbone could interfere with MF loss convergence. Mitigation: reduce $\lambda_s$ or increase `start_step`.
3. **Negligible effect:** If the v-head is already approximately smooth (possible for a shallow network), the regularizer has no effect. Monitoring: if $\mathcal{L}_{\text{smooth}}$ starts small and doesn't decrease, it's already smooth and the regularizer is unnecessary.

---

## 8. References

1. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." ICLR 2023.
2. Finlay, C. et al. (2020). "How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization." ICML 2020.
3. Hutchinson, M. F. (1989). "A stochastic estimator of the trace of the influence matrix." Communications in Statistics, 19(2), 433–450.
4. Grathwohl, W. et al. (2019). "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." ICLR 2019.
5. Geng, Z. et al. (2025b). "Improved Mean Flows." arXiv:2512.02012.
6. Geng, Z. et al. (2025a). "Mean Flows for One-step Generative Modeling." NeurIPS 2025 Oral. arXiv:2505.13447.
