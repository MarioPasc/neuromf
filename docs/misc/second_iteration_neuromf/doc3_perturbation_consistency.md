# NeuroMF Enhancement: Perturbation Consistency Regularization

**Project:** NeuroMF — Improved MeanFlow for 3D Brain MRI Synthesis
**Author:** Mario Pascual-González
**Date:** 2026-03-01 (revised)
**Status:** SPECIFICATION READY — **blocked on VAE equivariance validation (§1)**
**Depends on:** v2 config changes (doc1) applied; Phase 5 gate OPEN; **VAE equivariance validated**
**Scope:** Auxiliary loss term enforcing trajectory consistency under small data perturbations
**Proposal origin:** Ezequiel López-Rubio (PhD thesis advisor)
**Revision notes:** Added blocking VAE equivariance validation protocol, corrected computational costs (full UNet, not just v-head), v_tangent reuse optimization, time-dependent signal weakness analysis, gradient checkpointing for extra forward pass, codebase-specific integration.

---

## 0. Preamble

This document specifies a perturbation consistency regularizer for the instantaneous velocity field $v_\theta(z_t, t)$. The core idea: given a clean latent $z_0$ and a slightly perturbed version $z_0' = \alpha z_0 + \beta$, both noised with the SAME $\epsilon$, the resulting trajectories $z_t$ and $z_t'$ are close and the velocity field should assign them similar velocities.

**Relationship to doc2 (smoothness):** Doc2 constrains $\|\partial v / \partial z_t\|_F$ along **random** directions (isotropic). This document constrains $\|v(z_t') - v(z_t)\|$ along **data-manifold tangent** directions (anisotropic). In high-dimensional latent space ($d = 442{,}368$), random probes almost never align with manifold tangent directions. The perturbation loss provides a complementary signal that the smoothness loss cannot.

**Key simplification:** Perturbations are applied directly in latent space, avoiding expensive online VAE encoding. This relies on the MAISI VAE's approximate equivariance under affine intensity transforms — **an assumption that must be validated before implementation** (see §1).

---

## 1. BLOCKING PRE-CONDITION: VAE Equivariance Validation

**This section describes a mandatory validation step. The perturbation consistency regularizer MUST NOT be implemented until this validation passes.**

### 1.1 The Assumption

The regularizer assumes that the MAISI VAE encoder satisfies approximate affine equivariance:

$$\mathcal{E}(\alpha x + \beta) \approx \alpha \cdot \mathcal{E}(x) + \beta'$$

for small perturbations $|\alpha - 1| \leq 0.05$ and $|\beta| \leq 0.05 \cdot \text{range}(x)$. This allows generating perturbed latents $z_0' = \alpha z_0 + \beta$ directly in latent space without running the VAE encoder.

### 1.2 Why This Might Not Hold

The MAISI VAE uses GroupNorm at every encoder level. GroupNorm computes per-group statistics:

$$\text{GN}(x) = \frac{x - \mu_G(x)}{\sigma_G(x)} \cdot \gamma + \beta_{\text{GN}}$$

where $\mu_G$ and $\sigma_G$ depend on the input. For $x' = \alpha x + \beta_{\text{in}}$:

$$\mu_G(x') = \alpha \mu_G(x) + \beta_{\text{in}}, \quad \sigma_G(x') = |\alpha| \sigma_G(x)$$

$$\text{GN}(x') = \frac{(\alpha x + \beta_{\text{in}}) - (\alpha \mu_G(x) + \beta_{\text{in}})}{|\alpha| \sigma_G(x)} \cdot \gamma + \beta_{\text{GN}} = \text{sign}(\alpha) \cdot \text{GN}(x)$$

For $\alpha > 0$ (always true in our case), $\text{GN}(x') = \text{GN}(x)$. **GroupNorm is exactly contrast-invariant for multiplicative scaling.** This means contrast perturbations ($\alpha \neq 1, \beta = 0$) produce **identical** GroupNorm outputs — the contrast perturbation is completely absorbed.

**Implication:** For pure contrast perturbations, $\mathcal{E}(\alpha x) = \mathcal{E}(x)$ (not $\alpha \mathcal{E}(x)$). The equivariance assumption is **wrong** for the multiplicative component. However, additive shifts ($\alpha = 1, \beta \neq 0$) may still produce meaningful perturbations because:

$$\mu_G(x + \beta_{\text{in}}) = \mu_G(x) + \beta_{\text{in}}, \quad \sigma_G(x + \beta_{\text{in}}) = \sigma_G(x)$$

$$\text{GN}(x + \beta_{\text{in}}) = \text{GN}(x) \quad \text{(only if } \beta_{\text{in}} \text{ is constant within each group)}$$

For a global additive shift (same value across all spatial positions), GroupNorm absorbs it entirely. For a spatially-varying shift, the effect depends on the within-group statistics.

**This analysis suggests that the original formulation ($z_0' = \alpha z_0 + \beta$) in latent space does NOT correspond to intensity transforms in pixel space.** Instead, latent-space perturbations should be treated as generic local perturbations on the data manifold, without claiming pixel-space correspondence.

### 1.3 Validation Protocol

**Script:** Create `experiments/validation/vae_equivariance_test.py` (run on Picasso with A100).

**Procedure:**
1. Select 20 volumes from the FOMO-60K latent cache
2. Load both the pixel-space volumes and pre-computed latents
3. For each volume, apply pixel-space perturbations:
   - Contrast: $x' = \alpha x$ with $\alpha \in \{0.95, 0.97, 1.03, 1.05\}$
   - Brightness: $x' = x + \beta$ with $\beta \in \{-0.05, -0.02, 0.02, 0.05\}$
   - Combined: $x' = \alpha x + \beta$ (9 combinations)
4. Encode each $x'$ through the frozen MAISI VAE
5. Compare $\mathcal{E}(x')$ to the latent-space prediction:
   - For contrast: $z_0^{\text{pred}} = \alpha z_0$
   - For brightness: $z_0^{\text{pred}} = z_0 + \beta'$ (with fitted $\beta'$)
   - For combined: $z_0^{\text{pred}} = \alpha z_0 + \beta'$
6. Compute relative error: $\|\mathcal{E}(x') - z_0^{\text{pred}}\| / \|z_0\|$

**Pass criterion:** Relative error < 5% for all perturbation levels.

**If validation FAILS (likely based on GroupNorm analysis):**

The regularizer can still be used with a modified interpretation: **latent-space perturbations as generic manifold perturbations.** Instead of claiming pixel-space correspondence, we directly perturb $z_0$ in latent space and enforce velocity consistency. The regularizer remains valid (it enforces local Lipschitz continuity along specific directions in latent space), but the "clinically plausible variation" narrative must be dropped.

In this case:
- Remove the $\alpha$ (contrast) component entirely — use only additive perturbation $z_0' = z_0 + \delta$
- Sample $\delta$ from a low-magnitude isotropic distribution: $\delta \sim \mathcal{N}(0, \sigma_\delta^2 I)$ with $\sigma_\delta = 0.03$ (3% of latent channel std, since $\sigma_c \approx 1.0$ from latent_stats.json)
- This makes the perturbation direction random (like doc2) but the penalty is finite-difference Lipschitz rather than Jacobian-Frobenius

**If validation PASSES:** Proceed with the affine formulation as specified below.

---

## 2. Theoretical Justification

### 2.1 Trajectory Proximity Under Shared Noise

Consider the flow matching interpolation with shared noise $\epsilon$:

$$z_t = (1-t) z_0 + t \epsilon, \quad z_t' = (1-t) z_0' + t \epsilon$$

The pointwise difference:

$$z_t' - z_t = (1-t)(z_0' - z_0) \triangleq (1-t) \Delta z_0$$

**Two key properties:**
1. **Bounded magnitude:** $\|z_t' - z_t\| = (1-t) \|\Delta z_0\|$, vanishing as $t \to 1$
2. **Direction:** $\Delta z_0 = z_0' - z_0$ lies along the tangent space of the data manifold (connects two valid/near-valid data points)

Since the true velocity field is Lipschitz (§1.1 in doc2):

$$\|v(z_t', t) - v(z_t, t)\| \leq L(t)(1-t) \|\Delta z_0\|$$

### 2.2 Complementarity with Smoothness Regularizer

By first-order Taylor expansion:

$$v_\theta(z_t', t) - v_\theta(z_t, t) \approx \frac{\partial v_\theta}{\partial z_t} \cdot (z_t' - z_t) = (1-t) \frac{\partial v_\theta}{\partial z_t} \Delta z_0$$

- **Smoothness (doc2):** $\mathbb{E}_\xi[\|J \xi\|^2]$ — probes Jacobian along random directions
- **Perturbation (this doc):** $\|J \cdot \Delta z_0\|^2$ — probes Jacobian along manifold tangent directions

In $d = 442{,}368$ dimensions, random $\xi$ has negligible projection onto the low-dimensional data manifold tangent space. The perturbation loss provides signal exactly where it matters for generation quality.

### 2.3 Critical Limitation: Time-Dependent Signal Strength

The perturbation magnitude in $z_t$ space scales as $(1-t)$:

| Time $t$ | Region | $\|z_t' - z_t\| / \|\Delta z_0\|$ | Regularizer signal |
|----------|--------|-----|----|
| 0.0 | Data | 1.0 | Maximum |
| 0.5 | Mid | 0.5 | Moderate |
| 0.9 | Near-noise | 0.1 | Weak |
| 0.99 | 1-NFE boundary | 0.01 | **Negligible** |

**At the 1-NFE operating point ($t \approx 1$), the perturbation signal is essentially zero.** This is the opposite of where we need the most regularization.

**Mitigating arguments:**
1. The v-head is a continuous neural network. Smoothness learned at mid-range $t$ propagates to $t \approx 1$ via weight sharing.
2. The boundary sampling (doc1, §2.1) provides 10% of training samples at $t \in [0.95, 1.0]$. Even with small perturbations, the sheer volume of boundary samples provides indirect benefit.
3. The perturbation loss provides the strongest signal at $t \in [0.2, 0.8]$, which is where the multi-step Euler solver operates (NFE ≥ 2).

**Optional time-weighted formulation** (deferred to ablation):

$$\mathcal{L}_{\text{perturb}}^{\text{weighted}} = \mathbb{E}\!\left[\frac{\|v(z_t') - v(z_t)\|^2}{\max((1-t)^2, \eta) \|\Delta z_0\|^2}\right]$$

This normalizes out both the $(1-t)$ scaling and the perturbation magnitude, measuring the empirical Lipschitz ratio directly. The $\max(\cdot, \eta)$ clamp prevents divergence at $t \approx 1$. Use $\eta = 0.01$.

---

## 3. Mathematical Formulation

### 3.1 Loss Definition (Simple Formulation — Recommended Start)

Given training sample $(z_0, \epsilon, t)$ and perturbation $\delta$:

$$z_0' = z_0 + \delta, \quad \delta \sim \mathcal{N}(0, \sigma_\delta^2 I_d)$$

$$z_t = (1-t) z_0 + t \epsilon, \quad z_t' = (1-t) z_0' + t \epsilon = z_t + (1-t) \delta$$

$$\mathcal{L}_{\text{perturb}} = \frac{1}{d} \mathbb{E}_{t, z_0, \epsilon, \delta}\!\left[\|v_\theta(z_t', t) - v_\theta(z_t, t)\|_2^2\right]$$

**Simplified perturbation model:** We use additive isotropic noise $\delta \sim \mathcal{N}(0, \sigma_\delta^2 I)$ instead of the original affine model ($\alpha z_0 + \beta$). Reasons:

1. The affine model's $\alpha$ component is absorbed by GroupNorm (§1.2), making it ineffective
2. Isotropic $\delta$ is simpler, has no unverified assumptions, and still provides manifold-tangent signal (for small $\sigma_\delta$, $z_0 + \delta$ remains near the data manifold)
3. $\sigma_\delta = 0.03$ gives $\|\delta\| / \|z_0\| \approx 3\%$ (since per-channel $\sigma_c \approx 1.0$ and $\|\delta\|_{\text{expected}} = \sigma_\delta \sqrt{d}$ while $\|z_0\|_{\text{expected}} \approx \sqrt{d}$)

**If VAE equivariance validation passes** (§1.3): Revert to the affine formulation $z_0' = \alpha z_0 + \beta$ with $\alpha \sim \mathcal{U}(1 - \sigma_\alpha, 1 + \sigma_\alpha)$ and $\beta \sim \mathcal{U}(-\sigma_\beta, \sigma_\beta)$.

**Dimension normalization by $1/d$:** Same rationale as doc2 — makes the loss scale-independent and comparable to the adaptive-weighted main losses (~1.0 each).

### 3.2 Integration with Total Loss

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{MF}}^{\text{adaptive}} + \mathcal{L}_{\text{FM}}^{\text{adaptive}}}_{\text{~2.0}} + \lambda_s \cdot \mathcal{L}_{\text{smooth}} + \lambda_p \cdot \mathcal{L}_{\text{perturb}}$$

Like smoothness, the perturbation loss is NOT adaptive-weighted.

### 3.3 Gradient Flow

With the simple formulation and detached $v(z_t, t)$ (reused from compound velocity computation):

$$\nabla_\phi \mathcal{L}_{\text{perturb}} = \frac{2}{d} \mathbb{E}\!\left[(v_\theta(z_t') - v_\theta^{\text{sg}}(z_t)) \cdot \nabla_\phi\, v_\theta(z_t')\right]$$

Gradients flow through the forward pass of $v_\theta(z_t', t)$, traversing the entire UNet backbone + v-head. **Same implications as doc2:** both the v-head (228K params) and shared backbone (~38M params) are regularized.

**Asymmetric gradient flow:** Only $v(z_t')$ is differentiable. This pushes $v(z_t')$ toward $v(z_t)$ (the "anchor"), without pulling $v(z_t)$ toward $v(z_t')$. During early training when $v(z_t)$ is poor, this anchor is noisy. However, the FM loss $\|v - v_c\|^p$ simultaneously supervises $v(z_t)$ toward the correct target, so the anchor quality improves over training.

---

## 4. Implementation Specification

### 4.1 Pre-Implementation Checks

| ID | Check | Location | What to verify |
|----|-------|----------|----------------|
| C1 | VAE equivariance | **§1.3 validation script** | **BLOCKING.** Must pass before implementing this regularizer. |
| C2 | v_tangent availability | `meanflow_loss.py:346-347` | `v_tangent` computed under `no_grad`. Can be reused as detached $v(z_t, t)$. Same check as doc2 C6. |
| C3 | z_t availability | `meanflow_loss.py:212` | `z_t = (1-t_broad) * z_0 + t_broad * eps`. Must be passed out in result dict. Same plumbing as doc2. |
| C4 | z_0 availability | `latent_meanflow.py:181` | `z_0 = batch["z"]` — available in `training_step`. |
| C5 | eps availability | `latent_meanflow.py:183` | `eps = torch.randn_like(z_0)` — available in `training_step`. |
| C6 | Memory estimate | See §6 | One extra forward pass (~5-6 GB with grad). Current usage ~32 GB on 40 GB. |

### 4.2 Core Implementation

**New file:** `src/neuromf/losses/perturbation_consistency.py`

```python
"""Perturbation consistency loss for the instantaneous velocity field.

Enforces that the v-head produces similar outputs for nearby points on the
data manifold. Perturbations are applied directly in latent space as small
additive noise, creating perturbed trajectories that share the same
training noise epsilon.

The regularizer measures local Lipschitz continuity along perturbation
directions, complementing the isotropic smoothness regularizer (doc2).
"""

import torch
from torch import Tensor
from collections.abc import Callable


def perturbation_consistency_loss(
    v_fn: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    z_0: Tensor,
    z_t: Tensor,
    t: Tensor,
    eps: Tensor,
    v_original: Tensor,
    sigma_delta: float = 0.03,
) -> dict[str, Tensor]:
    """Compute perturbation consistency loss on the velocity field.

    Generates a perturbed latent z_0' = z_0 + delta, constructs the
    corresponding noisy state z_t' with the SAME noise eps, and penalizes
    the velocity field difference ||v(z_t', t) - v(z_t, t)||^2 / d.

    Args:
        v_fn: The dual-head model callable (z_t, r, t) -> (x_hat, v).
            Called as v_fn(z_t_prime, t, t) with r=t for instantaneous velocity.
        z_0: Clean latent data (B, C, D, H, W).
        z_t: Noisy latent at time t (B, C, D, H, W). Detached.
        t: Time values (B,).
        eps: Noise used to construct z_t (B, C, D, H, W). MUST be the
            same noise — shared noise is essential for trajectory proximity.
        v_original: Detached v_theta(z_t, t) from the main loss computation.
            Shape (B, C, D, H, W). MUST be detached.
        sigma_delta: Std of the perturbation noise in latent space.
            Default 0.03 (~3% of channel std, since sigma_c ~ 1.0).

    Returns:
        Dict with keys:
            "loss": scalar perturbation consistency loss (normalized by d),
            "delta_z0_norm": mean ||delta|| for monitoring,
            "delta_v_norm": mean ||v(z_t') - v(z_t)|| for monitoring,
            "lipschitz_ratio": mean ||delta_v|| / ||delta_zt|| for monitoring.
    """
    B = z_0.shape[0]
    d = z_0[0].numel()  # 4 * 48^3 = 442368

    # --- Sample perturbation ---
    delta = sigma_delta * torch.randn_like(z_0)

    # --- Perturbed latent ---
    z_0_prime = z_0.detach() + delta

    # --- Construct perturbed noisy state with SHARED noise ---
    t_bcast = t.view(B, *([1] * (z_0.ndim - 1)))
    z_t_prime = (1.0 - t_bcast) * z_0_prime + t_bcast * eps.detach()

    # --- Velocity at perturbed point (WITH grad for backprop) ---
    _, v_perturbed = v_fn(z_t_prime, t, t)  # r=t for instantaneous

    # --- Consistency loss ---
    delta_v = v_perturbed - v_original.detach()
    loss_per_sample = (delta_v ** 2).sum(dim=tuple(range(1, delta_v.ndim)))
    loss = loss_per_sample.mean() / d

    # --- Monitoring quantities (detached) ---
    with torch.no_grad():
        delta_zt = z_t_prime - z_t.detach()
        delta_z0_norm = delta.flatten(1).norm(dim=1).mean()
        delta_v_norm = delta_v.flatten(1).norm(dim=1).mean()
        delta_zt_norm = delta_zt.flatten(1).norm(dim=1).mean()
        lipschitz_ratio = delta_v_norm / (delta_zt_norm + 1e-8)

    return {
        "loss": loss,
        "delta_z0_norm": delta_z0_norm,
        "delta_v_norm": delta_v_norm,
        "lipschitz_ratio": lipschitz_ratio,
    }
```

**Design decisions:**

1. **`v_fn` signature is `(z_t, r, t) -> (x_hat, v)`** — same as doc2, matches `model.forward()`. The caller passes `r=t` for instantaneous velocity.

2. **`v_original` reused from `v_tangent`** — computed at `meanflow_loss.py:347` under `no_grad`. Zero additional cost.

3. **Shared noise `eps`** is critical. Using independent noise would make $z_t$ and $z_t'$ unrelated, destroying the trajectory proximity property.

4. **`z_0.detach()` and `eps.detach()`** — ensures the perturbation loss only affects model parameters, not the data interpolation gradient flow.

5. **Additive isotropic perturbation** $\delta \sim \mathcal{N}(0, \sigma_\delta^2 I)$ — simpler and more robust than the affine formulation. See §1.2 for why.

6. **Division by $d$** — dimension normalization, same as doc2.

### 4.3 Integration into Training Step

**Modify:** `src/neuromf/models/latent_meanflow.py`, in `training_step()` after doc2's smoothness block.

```python
# --- NEW: Perturbation consistency regularizer ---
if self._perturb_enabled and self.global_step >= self._perturb_start_step:
    from neuromf.losses.perturbation_consistency import perturbation_consistency_loss

    perturb_result = perturbation_consistency_loss(
        v_fn=self.net,
        z_0=z_0,
        z_t=result["_z_t"],
        t=t,
        eps=eps,
        v_original=result["_v_tangent"],
        sigma_delta=self._perturb_sigma_delta,
    )
    loss_perturb = perturb_result["loss"]
    loss = loss + self._lambda_perturb * loss_perturb

    self.log("train/loss_perturb", loss_perturb.detach(), sync_dist=True)
    self.log("train/loss_perturb_weighted",
             (self._lambda_perturb * loss_perturb).detach(), sync_dist=True)
    self.log("train/perturb_delta_v_norm", perturb_result["delta_v_norm"], sync_dist=True)
    self.log("train/perturb_lipschitz_ratio", perturb_result["lipschitz_ratio"], sync_dist=True)
```

**Required plumbing (shared with doc2):**

1. In `_forward_dual_head`: Add `result["_v_tangent"]` and `result["_z_t"]` (same change as doc2 §3.3)
2. In `__init__`: Parse perturbation config:
   ```python
   perturb_cfg = config.get("perturbation_consistency", {})
   self._perturb_enabled = bool(perturb_cfg.get("enabled", False))
   self._lambda_perturb = float(perturb_cfg.get("lambda_perturb", 0.01))
   self._perturb_sigma_delta = float(perturb_cfg.get("sigma_delta", 0.03))
   self._perturb_start_step = int(perturb_cfg.get("start_step", 1000))
   ```

### 4.4 Config Changes

In `configs/train_meanflow.yaml`:

```yaml
perturbation_consistency:            # NEW section
  enabled: false                     # Disabled by default
  lambda_perturb: 0.01              # Weight relative to adaptive losses (~1.0 each)
  sigma_delta: 0.03                 # Perturbation std (3% of channel std)
  start_step: 1000                  # Delay activation until after LR warmup
  log_every_n_steps: 50             # Log to TensorBoard
```

### 4.5 Delayed Activation Rationale

Same as doc2 §3.5: avoid destabilizing the chaotic first 1000 steps (90% gradient clipping).

### 4.6 Shared Plumbing with Doc2

Both regularizers need:
1. `result["_v_tangent"]` from `_forward_dual_head` — one code change serves both
2. `result["_z_t"]` from `_forward_dual_head` — one code change serves both
3. Config parsing in `__init__` — independent config sections but same pattern
4. Integration in `training_step` — sequential (smoothness first, then perturbation)

**Engineering note:** Implement the plumbing changes (returning `_v_tangent` and `_z_t`) as a single shared prerequisite, regardless of which regularizer is enabled.

---

## 5. Hyperparameter Selection

### 5.1 Perturbation Magnitude $\sigma_\delta$

From `latent_stats.json`, per-channel statistics:

| Channel | Mean | Std ($\sigma_c$) |
|---------|------|-------------------|
| 0 | -0.054 | 0.970 |
| 1 | -0.185 | 1.019 |
| 2 | -0.051 | 0.971 |
| 3 | 0.001 | 1.011 |

All channels have $\sigma_c \approx 1.0$. With $\sigma_\delta = 0.03$:

$$\frac{\|\delta\|}{\|z_0\|} \approx \frac{\sigma_\delta \sqrt{d}}{\sqrt{d}} = \frac{\sigma_\delta}{1} = 3\%$$

This is a conservative perturbation that keeps $z_0'$ near the data manifold.

| $\sigma_\delta$ | Expected $\|\delta\| / \|z_0\|$ | Regime |
|-----------------|------|--------|
| 0.01 | ~1% | Very conservative (may be too small) |
| 0.03 | ~3% | **Recommended** |
| 0.05 | ~5% | Moderate |
| 0.10 | ~10% | Aggressive (risk of off-manifold) |

### 5.2 Choice of $\lambda_p$

Same calibration strategy as doc2: burn-in procedure (§3.6 in doc2) or monitor-and-adjust.

**Expected magnitude of $\mathcal{L}_{\text{perturb}}$:** With $\sigma_\delta = 0.03$ and a moderately smooth network, the velocity difference $\|v(z_t') - v(z_t)\|$ should be $O(\sigma_\delta)$. The per-element loss is $O(\sigma_\delta^2 / d) = O(10^{-3} \times 2 \times 10^{-6}) = O(10^{-9})$. Wait — this seems too small. Let me reconsider.

Actually: $\|v(z_t') - v(z_t)\|^2 \approx \|J \cdot (1-t)\delta\|^2$. For a single sample: $\sum_i (J_i \cdot (1-t)\delta)^2$. If each $J_{ij} \sim O(0.01)$ and $\delta_j \sim N(0, 0.03^2)$, then $|(J\delta)_i| \sim O(0.01 \times 0.03 \times \sqrt{d})$ by CLT... this gets complicated.

**Practical recommendation:** Use the burn-in calibration procedure. Log $\mathcal{L}_{\text{perturb}}$ for 50 steps without applying it, then set $\lambda_p$ to achieve ~5-10% contribution to total loss.

### 5.3 Interaction with Smoothness Loss (Doc2)

If both are active:

$$\mathcal{L}_{\text{total}} = \underbrace{\sim 2.0}_{\text{MF+FM}} + \lambda_s \mathcal{L}_{\text{smooth}} + \lambda_p \mathcal{L}_{\text{perturb}}$$

Combined effect: the v-head Jacobian is constrained in all directions (smoothness) AND specifically along manifold tangent directions (perturbation). Risk of over-regularization if both $\lambda_s$ and $\lambda_p$ are too large.

**Monitoring rule:** If `raw_loss_v` (FM v-head loss) increases by >20% relative to baseline when both are active, reduce both $\lambda$ values by 2x.

**Recommended ablation schedule:**

| Experiment | $\lambda_s$ | $\lambda_p$ | Purpose |
|------------|-------------|-------------|---------|
| A (baseline v2) | 0.0 | 0.0 | Reference |
| B (smooth only) | TBD | 0.0 | Isolate smoothness effect |
| C (perturb only) | 0.0 | TBD | Isolate perturbation effect |
| D (both) | TBD | TBD | Combined effect |

TBD values set via burn-in calibration.

---

## 6. Memory Budget

### 6.1 Additional Cost

The perturbation loss requires one extra WITH-grad forward pass for $v_\theta(z_t', t)$:

| Configuration | Additional Memory | Fits (8 GB headroom)? |
|---------------|-------------------|-----------------------|
| Perturbation only (with grad) | ~5-6 GB | Yes (tight) |
| Perturbation only (no grad) | ~2-3 GB | Yes (comfortable) |
| Perturbation + smoothness (both with grad) | ~10-12 GB | **NO — OOM** |
| Perturbation + smoothness (alternating steps) | ~5-6 GB peak | Yes |

### 6.2 Gradient Checkpointing Option

The perturbation forward pass is a normal forward pass (no `torch.func.jvp`). Unlike the compound velocity JVP, it IS compatible with gradient checkpointing. Enabling checkpointing for this specific pass could reduce memory from ~5-6 GB to ~3 GB.

**Implementation:** Wrap the perturbation forward pass:
```python
from torch.utils.checkpoint import checkpoint
_, v_perturbed = checkpoint(v_fn, z_t_prime, t, t, use_reentrant=False)
```

**Caveat:** This requires the model to support `use_reentrant=False` checkpointing in its forward method. The existing `_forward_with_dual_emb` already supports this (`maisi_unet.py:362-413`), so it should work.

### 6.3 Combined Strategy

If both regularizers are needed simultaneously, use:
1. Smoothness: FD approach with no_grad (~2-3 GB) — the smoothness loss provides a *monitoring signal* without training gradients
2. Perturbation: with grad (~5-6 GB) — the perturbation loss provides training gradients
3. Total additional: ~7-9 GB → fits in 8 GB headroom

Rationale for keeping smoothness as monitoring-only: the perturbation loss already constrains the Jacobian along manifold directions (the most important directions). The smoothness loss provides complementary isotropic constraint but is less critical if memory is tight.

---

## 7. Verification Tests

All tests go in `tests/test_perturbation_consistency.py`. Test naming: `test_P6_T{N}_PC_{description}`.

| Test ID | Description | Pass Criterion | Priority |
|---------|-------------|----------------|----------|
| PC-T1 | `perturbation_consistency_loss` returns dict with all keys for mock inputs `(2, 4, 8, 8, 8)` | Finite positive values, no errors | CRITICAL |
| PC-T2 | Gradient flows: `loss.backward()` produces non-zero gradients on v-head parameters | ≥90% of v-head params have `grad.norm() > 0` | CRITICAL |
| PC-T3 | Shared noise: verify $z_t' - z_t = (1-t) \delta$ exactly | Max absolute error < 1e-6 | CRITICAL |
| PC-T4 | Zero perturbation ($\sigma_\delta = 0$): loss = 0 | Loss < 1e-10 | CRITICAL |
| PC-T5 | `lambda_perturb=0.0` does not change total loss | Bit-exact match given same seed | CRITICAL |
| PC-T6 | `start_step > global_step` skips computation | Loss matches baseline; no extra memory | CRITICAL |
| PC-T7 | Perturbation magnitude: $\|\delta\| / \|z_0\| < 0.05$ with default $\sigma_\delta = 0.03$ | Ratio check over 100 random batches | CRITICAL |
| PC-T8 | Memory: training step completes at `batch_size=2` with perturbation enabled (mock test with reduced spatial size) | No CUDA OOM | CRITICAL |
| PC-T9 | Lipschitz ratio decreases for a trained-vs-random network | Random network has ratio > trained network | INFORMATIONAL |

---

## 8. Expected Outcomes

### 8.1 Training Behavior

- $\mathcal{L}_{\text{perturb}}$ should decrease as the v-head learns consistent velocities for nearby points
- The Lipschitz ratio (`perturb_lipschitz_ratio`) should decrease, indicating reduced sensitivity to perturbations
- `raw_loss_v` (FM loss) should not increase by more than 20%

### 8.2 Generation Quality

- **NFE=1:** Modest expected improvement. The perturbation signal is weakest at $t \approx 1$ (the 1-NFE operating point). Any benefit comes indirectly through weight sharing. Estimate: 0-10% FID improvement.
- **NFE=2-10:** Strongest expected improvement. The Euler solver traverses mid-range $t$ values where the perturbation signal is strongest. Estimate: 5-15% FID improvement.
- **NFE=50:** Maintained (already near convergence).

### 8.3 Scientific Novelty

Perturbation consistency regularization of velocity fields in flow matching / MeanFlow models is, to our knowledge, novel. Combined with the smoothness regularizer (doc2), this forms a two-pronged strategy:

1. **Isotropic** (doc2): constrains Jacobian norm in all directions
2. **Anisotropic** (this doc): constrains Jacobian along data-manifold tangent directions

For the TMI paper: clean 4-condition ablation table (baseline, smooth-only, perturb-only, both) at each NFE level.

### 8.4 Risks

1. **VAE equivariance failure:** If the validation (§1.3) shows the affine assumption is wrong, we fall back to isotropic perturbation ($z_0' = z_0 + \delta$). This is still valid but loses the "manifold-tangent" narrative — the perturbation direction is random, making this doc more similar to doc2.
2. **Negligible 1-NFE effect:** The time-dependent signal weakness (§2.3) means the primary target (1-NFE) may see minimal improvement.
3. **Backbone interference:** Perturbation gradients through the shared backbone could destabilize MF loss convergence.

---

## 9. References

1. Lipman, Y. et al. (2023). "Flow Matching for Generative Modeling." ICLR 2023.
2. Finlay, C. et al. (2020). "How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization." ICML 2020.
3. Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
4. Song, Y. et al. (2023). "Consistency Models." ICML 2023.
5. Geng, Z. et al. (2025b). "Improved Mean Flows." arXiv:2512.02012.
6. Geng, Z. et al. (2025a). "Mean Flows for One-step Generative Modeling." NeurIPS 2025 Oral. arXiv:2505.13447.
