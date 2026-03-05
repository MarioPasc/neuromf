# NeuroMF Scientific Audit: Diagnosis and Improvement Roadmap

**Author:** Claude (Opus 4.6) — requested by Mario Pascual-González  
**Date:** 2026-03-05  
**Scope:** Implementation correctness, hyperparameter analysis, 1-NFE quality improvement strategy

---

## Executive Summary

Your 1-NFE FID-3D of 73.9 versus 6.1 at NFE=50 reveals a **variance collapse** at the generation boundary. The PCA plot (Fig. 3, NFE=1 panel) is the smoking gun: generated samples cluster in a tight, low-variance region with near-zero Coverage and Density. The model has learned the *mean* of the data distribution but not its full support at the 1-NFE operating point.

After auditing the codebase, technical report, reference implementations (iMF, pMF, original MF), and three recent concurrent papers (α-Flow, Kim et al. "Accelerating MeanFlow", Re-MeanFlow), I identify **one critical implementation issue**, **several hyperparameter mismatches**, and **three theoretically-grounded improvement strategies** ranked by effort and expected impact.

**Bottom line:** The model's excellent NFE≥10 performance proves the architecture and core algorithm are correct. The 1-NFE failure is primarily a *training dynamics* problem, not an architectural one.

---

## Part 1: Implementation Audit — Is Anything Wrong?

### 1.1 Core Algorithm: CORRECT ✅

The MeanFlow Identity implementation, JVP computation, compound velocity construction, stop-gradient placement, and x-prediction reparameterisation are all verified correct against the iMF reference (Algorithm 1 in Geng et al., 2025b). Specifically:

- Interpolation: `z_t = (1-t)*z_0 + t*eps` ✅
- Target: `v_c = eps - z_0` ✅
- JVP tangent direction: `(v_tilde, 1, 0)` for args `(z_t, t, r)` ✅ (accounts for your arg order swap vs iMF's `(z, r, t)` with tangent `(v, 0, 1)`)
- Compound velocity: `V = u + (t-r) * sg[du/dt]` ✅
- x-to-u conversion: `u = (z_t - x_hat) / max(t, t_min)` ✅
- 1-NFE sampling: `z_0 = f_theta(eps, r=0, t=1)` directly outputs x_hat ✅

### 1.2 Adaptive Weighting: FUNCTIONALLY CORRECT but SUBOPTIMAL ⚠️

Your implementation uses `norm_eps=1.0, norm_p=1.0`, giving:

$$w^{(b)} = \text{sg}[\ell^{(b)}] + 1.0, \quad \ell^{(b)}_{\text{weighted}} = \frac{\ell^{(b)}}{\text{sg}[\ell^{(b)}] + 1.0}$$

The gradient is:

$$\nabla_\theta \ell_{\text{weighted}} = \frac{\nabla_\theta \ell^{(b)}}{\text{sg}[\ell^{(b)}] + 1.0}$$

With per-sample raw losses of order ~700K (FM) to ~6.3M (MF), the gradient magnitude is suppressed by a factor of ~10⁻⁶. This is **by design** — it normalises all samples to contribute roughly unit gradient — but it creates an extremely flat loss landscape where the optimiser takes tiny steps regardless of how wrong the predictions are.

**Comparison with iMF reference:** iMF uses `c ≈ 1e-3, p=1.0` in their JAX code. However, at these raw loss scales (>10⁵), both `eps=1.0` and `eps=1e-3` produce essentially the same normalisation since `eps << loss`. The real question is whether this aggressive normalisation is appropriate for your training scale.

**The original MF paper (Table 1e)** shows that `p=1.0` is optimal on ImageNet with ~400K+ gradient updates. With only ~29K updates, the normalisation may be *too aggressive*, suppressing the gradient signal needed for rapid convergence. This is not a bug, but a hyperparameter mismatch with your compute budget.

### 1.3 Time Sampling: CORRECT but INCOMPLETE ⚠️

The logit-normal(-0.4, 1.0) sampling matches iMF exactly. However, the critical diagnostic is:

$$P(t > 0.95 \text{ and } r < 0.05) \approx P(t > 0.95) \times P(r < 0.05) \approx 0.0004 \times 0.4 \approx 0.016\%$$

At a batch size of 2 per GPU (effective 132 with accumulation), the expected number of boundary samples per epoch is:

$$5471 \times 0.00016 / 2 \approx 0.44 \text{ samples/epoch}$$

**The model sees fewer than 1 boundary sample per epoch.** This is the primary reason boundary injection was proposed. But you report that even with 10% boundary injection, 1-NFE FID stays at 70-90. This tells us the problem is **not just sampling sparsity** — there's a deeper training dynamics issue.

### 1.4 JVP Tangent Source: POTENTIAL ISSUE ⚠️

In your training step (Algorithm 1, Step 6 of the report), the tangent vector for the JVP is:

```python
# Step 6: Compute tangent (no grad)
(_, v_tilde) = f_theta(z_t, r=t, t)  # v-head output at r=t
```

This runs the model with `r=t` under `torch.no_grad()` to get the v-head's prediction as the tangent. This is the iMF approach. However, there's a subtle issue:

**The v-head has only 228K parameters (0.13% of the model)**. The iMF reference uses an auxiliary head with **8 Transformer layers** sharing the backbone (effectively millions of parameters). Your v-head's `cos(v_hat, v_c) = 0.44` is decent but means the tangent vector has **56% angular error** relative to the true conditional velocity. For comparison, iMF's aux head on ImageNet achieves much higher alignment.

The JVP uses this tangent to compute `∂u/∂z · v_tilde + ∂u/∂t`. If `v_tilde` is inaccurate, the JVP correction `(t-r) * du/dt` will be wrong, making the compound velocity `V` a poor approximation to `v_c`. This error **compounds** for large `(t-r)` intervals — precisely the 1-NFE operating point where `t-r = 1`.

**This is the most likely proximate cause of the 1-NFE failure.** The model can correct for this at multi-step (small `dt`), but at 1-NFE the error is maximal.

### 1.5 Data Proportion: SUBOPTIMAL ⚠️

You use `data_proportion=0.5` (50% FM / 50% MF). The original MF paper (Table 1a) found **75% FM / 25% MF** to be optimal (FID 61.06 vs 63.14 for 50/50).

More importantly, α-Flow (Zhang et al., 2025) proves that the FM and MF gradients are **strongly negatively correlated** (cosine similarity approaching -1.0 in early training). This gradient conflict means that 50% of your gradient updates are actively fighting each other. The 75% FM ratio provides a better surrogate for trajectory flow matching that mitigates this conflict.

However, you already tried this change and it didn't help 1-NFE. This is consistent with the α-Flow finding: simply adjusting the ratio doesn't solve the fundamental curriculum problem.

### 1.6 EMA Decay: LIKELY MISMATCHED ⚠️

With `β = 0.9999`, the effective window is `N_eff = 1/(1-β) = 10,000` steps. You trained for 28,980 steps, so the EMA averages over ~34% of your training trajectory. This is reasonable but borderline.

For the pMF reference, they maintain **multiple EMA decay rates** (`{500, 1000, 2000}` M-images half-life) and select the best at inference. You use a single rate. This is not a bug but leaves performance on the table.

---

## Part 2: Low-Effort Improvements (Hyperparameter/Config Changes Only)

### 2.1 Switch to Constant LR Schedule (no cosine decay)

**Current:** Cosine decay from 1e-4 → 0.  
**Recommended:** Constant 1e-4 with linear warmup of 1000 steps.  
**Justification:** Both iMF and the original MF use **constant LR** (Table 4 in iMF). The cosine decay means that by epoch 388 (your best FID), the LR has already decayed to ~8.5e-5, and by epoch 689 it's at 5.6e-5. This premature decay robs the model of learning rate precisely when it needs to learn the harder MF objective. MeanFlow benefits from constant LR because the adaptive weighting already controls effective step size per-sample.

### 2.2 Multiple EMA Decay Rates

**Current:** Single `β = 0.9999`.  
**Recommended:** Track 3 EMA models with `β ∈ {0.999, 0.9995, 0.9999}`, select best at evaluation.  
**Justification:** pMF (Table 8) and EDM2 (Karras et al., 2024) both maintain multiple EMA rates. The optimal rate depends on training dynamics — shorter runs need faster EMA. With only ~29K steps, `β=0.999` (N_eff=1000) may outperform `β=0.9999` (N_eff=10000).  
**Implementation:** ~30 lines of code change in `ema.py` to maintain shadow parameters for each rate. Memory cost: 3× model size for EMA shadows (~534M params total), which is ~2GB in bf16 per GPU.

### 2.3 Reduce Adaptive Weighting Aggressiveness

**Current:** `norm_p=1.0, norm_eps=1.0`.  
**Recommended:** `norm_p=0.5, norm_eps=1.0`.  
**Justification:** With `p=0.5`, the weight becomes:

$$w = \sqrt{\text{sg}[\ell] + 1.0}, \quad \ell_{\text{weighted}} = \frac{\ell}{\sqrt{\text{sg}[\ell] + 1.0}}$$

The gradient is:

$$\nabla_\theta \ell_{\text{weighted}} = \frac{\nabla_\theta \ell}{\sqrt{\text{sg}[\ell] + 1.0}}$$

For `ℓ = 700K`: gradient scale ≈ 1/837 instead of 1/700K. This is **3 orders of magnitude** more gradient signal while still providing stabilisation. The MF paper (Table 1e) shows `p=0.5` achieves FID 63.98 vs 61.06 for `p=1.0` on ImageNet at 400K steps — only a 5% difference. But their training is 14× longer than yours. For shorter training, the faster convergence from `p=0.5` likely dominates.

**CAVEAT:** You report that `p_norm=0.5` caused "1000× gradient explosion" in Phase 4e. This was likely with `eps=1.0` and the smaller dataset/shorter training. With the expanded 5471-sample dataset and proper warmup, `p=0.5` should be stable. Run a 50-epoch pilot first.

### 2.4 Inference-Time Variance Rescaling

**Current:** Direct x_hat output with no post-processing.  
**Recommended:** After 1-NFE generation, apply per-channel variance rescaling:

$$\hat{z}_0^{\text{rescaled}} = \mu_{\text{data}} + \frac{\sigma_{\text{data}}}{\sigma_{\text{gen}}} \cdot (\hat{z}_0 - \mu_{\text{gen}})$$

where `σ_data ≈ 0.97-1.02` (from your latent stats) and `σ_gen ≈ 0.78` (from your generated stats, Table 17). This is a simple post-hoc rescaling that corrects the 20-25% variance underestimation.

**Mathematical justification:** If the model learns the correct mean but underestimates variance, this is equivalent to an affine transport map with incorrect scale. The rescaling corrects this without retraining.

**Expected impact:** Moderate improvement in FID (perhaps 10-20% relative), but will NOT fix the mode collapse visible in Coverage/Density since those are structure-dependent.

**Implementation:** ~5 lines in the sampling code. Zero training cost.

---

## Part 3: Medium-Effort Improvements (Code Changes Required)

### 3.1 ★★★ Progressive Gap Curriculum (DTD) — HIGHEST PRIORITY

**Source:** Kim et al., "Understanding, Accelerating, and Improving MeanFlow Training" (arXiv:2511.19065, Nov 2025).

**Core insight:** The MF objective has a *task dependency structure*: learning accurate instantaneous velocity `v` is a **prerequisite** for learning average velocity `u` over large gaps. Furthermore, learning small-gap `u` (small `t-r`) is a prerequisite for large-gap `u`. This creates a curriculum:

$$\text{Learn } v \text{ first} \to \text{small-gap } u \to \text{medium-gap } u \to \text{large-gap } u \text{ (1-NFE)}$$

The standard MF training samples all gap sizes simultaneously, violating this dependency. The result: gradient signal for large gaps is noisy and unconstructive in early training.

**The DTD (Dynamic Time Distribution) method:** Instead of fixed logit-normal sampling, use a time-varying distribution that starts narrow (small gaps only) and progressively widens:

At training step `k` out of total `K`:

$$\sigma_{\text{gap}}(k) = \sigma_{\min} + (\sigma_{\max} - \sigma_{\min}) \cdot \min\left(1, \frac{k}{k_{\text{transition}}}\right)$$

The gap `h = t - r` is sampled from a distribution that starts concentrated near 0 (FM-like) and gradually allows larger gaps.

**Concrete implementation:**

In the first phase (steps 0 to `k_transition ≈ 0.3 * K`):
- Sample `(t, r)` with gap constraint: `h = t - r ~ Beta(α, β)` where `α` increases over time
- Start with `data_proportion = 0.9` (90% FM, 10% MF with small gaps)
- Gradually reduce to `data_proportion = 0.5` and allow full gap range

In the second phase (steps `k_transition` to `K`):
- Standard logit-normal sampling with uniform gap distribution
- Full 50/50 FM/MF split

**Result from the paper:** On ImageNet with DiT-XL, DTD improves 1-NFE FID from 3.43 → 2.87 (16% reduction) with the **same architecture and compute budget**. The convergence is also **2.5× faster**.

**Why this addresses your specific problem:** Your cos(V, v_c) = 0.307 at epoch 689 is very low for 442K-dimensional space. The v-head's cos(v_hat, v_c) = 0.44 is also low. The progressive curriculum ensures the velocity field is well-established before asking the model to extrapolate across the full [0, 1] interval. Currently, the model is asked to predict `u(z_1, 0, 1)` from the very first epoch, when it has no concept of what the velocity field looks like.

**Implementation effort:** Modify `time_sampler.py` to accept a `progress` parameter (current_step / total_steps) and implement the progressive schedule. ~50-100 lines of code.

### 3.2 ★★★ Two-Stage Training (FM Pretraining → MF Fine-tuning)

**Source:** Decoupled MeanFlow (Yan et al., 2025), α-Flow (Zhang et al., 2025), Kim et al. (2025).

**Core insight from α-Flow:** The MF loss decomposes as:

$$\mathcal{L}_{\text{MF}} = \underbrace{\mathcal{L}_{\text{TFM}}}_{\text{trajectory flow matching}} + \underbrace{\mathcal{L}_{\text{TC}}}_{\text{trajectory consistency}}$$

These two gradients are **strongly negatively correlated** (cosine similarity ≈ -0.8 in early training). Training both simultaneously wastes computation on cancelling gradients.

**Strategy:**

**Stage 1 (60-70% of compute):** Train as **pure flow matching** (`data_proportion=1.0`, i.e., `r=t` always). This is identical to standard velocity-matching training. The model learns `v_θ(z_t, t)` accurately across all timesteps.

**Stage 2 (30-40% of compute):** Switch to MeanFlow objective with `data_proportion=0.5`. The pre-trained velocity field provides excellent JVP tangents from the start, and the model only needs to learn the *average* velocity correction.

**Why this is particularly effective for your case:** You already have the MAISI rflow transfer learning pipeline. The rflow checkpoint IS a pre-trained flow matching model in the same latent space. You can:

1. Load rflow weights (99% parameter compatibility)
2. Fine-tune with pure FM objective for ~50K steps to adapt to brain-specific data
3. Switch to MF objective for ~50K steps to learn 1-step generation

This combines transfer learning AND curriculum learning, addressing both the undertrained backbone and the 1-NFE gap simultaneously.

**Expected impact:** Based on Kim et al.'s results, this should yield **2-3× faster convergence** to the same multi-step quality, AND substantially better 1-NFE quality because the velocity field is pre-established.

**Implementation effort:** Mostly config changes. The rflow loading infrastructure already exists. Need to add a "stage switching" mechanism in the training loop (~100 lines).

### 3.3 ★★ Eliminate v-Head, Use Boundary Condition

**Current:** Separate v-head (228K params, 1 ResBlock) with auxiliary loss.  
**Recommended:** Use `v_θ(z_t, t) = u_θ(z_t, t, t)` (evaluate main network at `r=t`).

**Justification from iMF Table 1a (without CFG):**
- Boundary condition: FID = 29.42
- Aux head: FID = 30.76

The boundary condition is **better** in the unconditional case (which is your setting). It also:
- Eliminates 228K auxiliary parameters
- Removes the auxiliary loss term (simpler optimisation landscape)
- Guarantees the tangent uses the **full capacity** of the 178M backbone, not a 228K bottleneck

**The current v-head is 21-256× smaller than iMF's aux head**, creating a capacity mismatch that limits tangent quality. Rather than scaling up the v-head (which adds training cost), the boundary condition leverages the full backbone for free.

**However**, this requires an extra forward pass through the main network with `r=t` to compute the tangent, whereas the v-head computes it as a side output of the JVP pass. The computational cost is ~1.5× per step (one extra forward pass under `no_grad`). Given that the JVP pass is already expensive, this may be acceptable.

**Implementation:** ~20 lines: replace v-head tangent call with `u_fn(z_t, t, t)` under `no_grad`. Remove v-head architecture and aux loss.

### 3.4 ★★ Post-hoc Norm Correction with Grid Search

**Current:** `γ = 1.0` (no correction).  
**Recommended:** After training, calibrate γ on a held-out validation set:

$$z_0 = \varepsilon - \frac{u_\theta(\varepsilon, 0, 1)}{\gamma}$$

Search over `γ ∈ {1.0, 1.05, 1.10, 1.15, 1.20, 1.25}`.

Your compound velocity norm ratio is 1092/941 ≈ 1.16, suggesting `γ ≈ 1.16` might be optimal. This is analogous to classifier-free guidance rescale (Lin et al., 2024) and costs nothing at training time.

---

## Part 4: High-Effort Improvements (Significant Code/Methodology Changes)

### 4.1 ★★★ α-Flow Curriculum Objective

**Source:** Zhang et al., "AlphaFlow: Understanding and Improving MeanFlow Models" (arXiv:2510.20771, Oct 2025).

α-Flow introduces a parametric family of objectives:

$$\mathcal{L}_\alpha = \mathbb{E}_{t,r,x,\varepsilon}\left[\|V_\theta^{(\alpha)} - v_c\|^2\right]$$

where the compound velocity is modified by a parameter α ∈ [0, 1]:

- α = 0: Pure flow matching (velocity consistency disabled)
- α = 1: Full MeanFlow
- Intermediate: Smooth interpolation

The training curriculum anneals α from 0 to 1 over the training run, using a sigmoid schedule:

$$\alpha(k) = \sigma\left(\frac{k - k_s}{\tau}\right)$$

where `k_s` is the transition midpoint and `τ` controls sharpness.

**Results:** α-Flow-XL/2+ achieves **FID = 2.58 at 1-NFE** on ImageNet 256×256 (vs 3.43 for MF and 1.72 for iMF). This is the current SOTA for vanilla DiT backbones.

**Implementation complexity:** Moderate. Requires modifying the loss computation to scale the JVP correction by α, and implementing the schedule. The key equations are:

$$V_\theta^{(\alpha)} = u_\theta + \alpha \cdot (t-r) \cdot \text{sg}\left[\frac{du}{dt}\right]$$

When `α = 0`, `V = u = u_θ(z_t, t, t)` for FM samples, recovering pure FM. When `α = 1`, it's standard MF.

**Why this is better than naive boundary injection:** Instead of injecting a fixed fraction of boundary samples (which you've tried and doesn't work), α-Flow gradually teaches the model to make larger and larger "jumps" in the velocity space. The model first learns what the velocity field looks like locally, then learns to average it over progressively longer intervals.

**This directly addresses the core failure mode:** The 1-NFE operating point requires `V_θ(z_1, 0, 1) = v_c`, but the model has never seen accurate gradients for this because the JVP tangent (v_tilde) is poor at large gaps when the velocity field itself is poorly learned. α-Flow breaks this chicken-and-egg problem.

**Estimated implementation:** ~200 lines of code (modify `meanflow_loss.py` to accept α parameter, implement schedule in training loop, add config entries).

### 4.2 ★★ Knowledge Distillation from Multi-Step Teacher

**Concept:** Your model at NFE=50 achieves FID-3D = 6.1, which is excellent. Use this as a **teacher** to distill knowledge into the 1-NFE student.

**Approach (progressive distillation, Salimans & Ho 2022):**

1. Generate a large set of latents using the NFE=50 model (or even NFE=10, which has FID 7.3)
2. For each noise vector ε, compute the final clean latent z_0^{teacher} using multi-step Euler
3. Train the 1-NFE path to match: `||f_θ(ε, 0, 1) - z_0^{teacher}||²`

This provides direct supervision at the 1-NFE operating point, bypassing the MF identity entirely.

**Mathematical justification:** Let `Φ_K(ε)` be the K-step Euler mapping from noise to data. The teacher provides:

$$z_0^{\text{teacher}} = \Phi_{50}(\varepsilon)$$

The student learns:

$$\mathcal{L}_{\text{distill}} = \mathbb{E}_\varepsilon\left[\|\Phi_1(\varepsilon) - \text{sg}[\Phi_{50}(\varepsilon)]\|^2\right]$$

This is a standard L2 regression with a fixed (stop-gradient) target. The teacher signal is much lower-variance than the MF identity at large gaps.

**Implementation:** Generate ~10K teacher samples (NFE=50), store paired (noise, clean_latent). Add a distillation loss term to training. ~150 lines of code + generation time.

### 4.3 ★ pMF-Style Perceptual Loss in Decoded Space

**Source:** Lu et al., "One-step Latent-free Image Generation with Pixel Mean Flows" (arXiv:2601.22158, Jan 2026).

pMF achieves SOTA by adding **LPIPS perceptual loss** computed on decoded images. For your setting, this would require:

1. During training, periodically decode `x_hat` through the MAISI VAE decoder
2. Compute LPIPS(decoded_x_hat, decoded_z_0) as an auxiliary loss
3. Weight appropriately

**The problem:** The MAISI VAE decoder is frozen and expensive. Computing LPIPS on 192³ volumes requires either slice-based approximation or a 3D perceptual metric. The VAE decode + LPIPS backprop would roughly double training time.

**A lighter alternative:** Compute perceptual loss on the **latent** space using a pre-trained 3D feature extractor (Med3D ResNet-50). This avoids the VAE decode but provides semantic guidance beyond L2.

**Expected impact:** Moderate. pMF's LPIPS weight is only 0.4 relative to the main loss, so it's a refinement, not a game-changer. Worth trying after the curriculum-based improvements.

---

## Part 5: Ranked Improvement Roadmap

| Rank | Improvement | Effort | Expected FID-3D Impact (1-NFE) | Dependencies |
|------|------------|--------|-------------------------------|-------------|
| 1 | **Two-stage training** (FM pretrain → MF finetune with rflow init) | Medium | **40-60** (from 73.9) | rflow loading (done) |
| 2 | **Progressive gap curriculum (DTD)** | Medium | **35-55** | None |
| 3 | **α-Flow annealing** | Medium-High | **30-50** | None (subsumes DTD) |
| 4 | Constant LR (remove cosine decay) | Low | ~5-10% relative | None |
| 5 | `norm_p=0.5` adaptive weighting | Low | ~5-15% relative | Pilot run first |
| 6 | Eliminate v-head → boundary condition | Low-Medium | ~5-10% relative | None |
| 7 | Multiple EMA rates | Low | ~5-10% relative | None |
| 8 | Inference-time variance rescaling | Trivial | ~10-20% relative | None |
| 9 | Norm correction γ calibration | Trivial | ~5-10% relative | None |
| 10 | Knowledge distillation from NFE=50 teacher | Medium | **25-40** | NFE=50 generation |

---

## Part 6: Recommended Next Training Run Configuration

Based on the analysis above, here is my recommended configuration for the v3 training run, incorporating the highest-impact changes:

```yaml
# NeuroMF v3 Configuration — Two-Stage + Progressive Curriculum

# Stage 1: Flow Matching Pretraining (60% of compute)
stage_1:
  objective: flow_matching_only  # data_proportion = 1.0 (all r=t)
  epochs: 1800  # ~60% of 3000
  initialization: rflow_transfer  # Load MAISI rflow weights
  lr: 1.0e-4
  lr_schedule: constant  # NOT cosine
  warmup_steps: 1000
  use_v_head: false  # No v-head in FM stage
  
# Stage 2: MeanFlow Fine-tuning (40% of compute)  
stage_2:
  objective: meanflow_imf
  epochs: 1200  # ~40% of 3000
  data_proportion: 0.5  # 50% FM, 50% MF
  progressive_gap: true  # DTD-style curriculum within stage 2
  gap_warmup_fraction: 0.3  # First 30% of stage 2: small gaps only
  lr: 5.0e-5  # Lower LR for fine-tuning
  lr_schedule: constant
  use_v_head: false  # Use boundary condition v(z,t) = u(z,t,t)

# Shared settings
training:
  optimizer: adamw
  betas: [0.9, 0.95]
  weight_decay: 0.0
  gradient_clip: 1.0
  precision: bf16-mixed
  max_epochs: 3000
  effective_batch: 132

meanflow:
  p: 2.0
  norm_p: 0.5  # Less aggressive than 1.0
  norm_eps: 1.0
  t_min: 0.001

ema:
  decays: [0.999, 0.9995, 0.9999]  # Track 3 rates

time_sampling:
  distribution: logit_normal
  mu: -0.4
  sigma: 1.0
  # No boundary injection needed — progressive curriculum handles this
```

**Key changes from v1:**
1. Two-stage training with rflow initialisation
2. Progressive gap curriculum in stage 2
3. Boundary condition instead of v-head
4. Constant LR (no cosine decay)
5. `norm_p=0.5` for faster convergence
6. Multiple EMA decay rates
7. No boundary injection (subsumed by curriculum)

**Expected training cost:** ~3× v1 wall time (3000 epochs vs 690 early-stopped). With rflow init, the FM stage should converge much faster than training from scratch, so effective quality per compute hour should be significantly better.

---

## Part 7: Why Boundary Injection Alone Doesn't Work

You mentioned that boundary injection (10% of batch from `(t≈1, r≈0)`) hasn't helped. Here's the mathematical explanation:

At the 1-NFE operating point `(t=1, r=0)`, the compound velocity is:

$$V_\theta(z_1, 0, 1) = u_\theta(z_1, 0, 1) + 1 \cdot \text{sg}\left[\frac{\partial u_\theta}{\partial z_1} \cdot \tilde{v} + \frac{\partial u_\theta}{\partial t}\right]$$

The JVP correction scales linearly with `(t-r) = 1`. If the JVP tangent `v_tilde` has angular error `δ` (your v-head has `cos = 0.44`, so `δ ≈ 64°`), the error in V is:

$$\|V_\theta - v_c\| \geq (t-r) \cdot \left\|\frac{\partial u_\theta}{\partial z}\right\| \cdot \|v_c\| \cdot \sin(\delta)$$

For `sin(64°) ≈ 0.9`, this is nearly the full JVP magnitude. **The compound velocity at (t=1, r=0) is dominated by a JVP computed with a bad tangent.**

Simply injecting more boundary samples doesn't fix this because the loss at those samples is so large that the adaptive weighting suppresses the gradient to ~1.0 (same as all other samples). The model receives no *preferential* signal at the boundary.

**The curriculum approach fixes this** by ensuring the tangent `v_tilde` is accurate (from Stage 1 FM pretraining) before asking the model to compute JVPs at large gaps. By the time the model encounters `(t=1, r=0)` samples in Stage 2, the tangent quality is much higher, making the compound velocity informative rather than noisy.

---

## References

1. Geng, Z. et al. (2025a). "Mean Flows for One-step Generative Modeling." NeurIPS 2025, arXiv:2505.13447.
2. Geng, Z. et al. (2025b). "Improved Mean Flows." arXiv:2512.02012.
3. Lu, Y. et al. (2026). "One-step Latent-free Image Generation with Pixel Mean Flows." arXiv:2601.22158.
4. Kim, J.-Y. et al. (2025). "Understanding, Accelerating, and Improving MeanFlow Training." arXiv:2511.19065.
5. Zhang, H. et al. (2025). "AlphaFlow: Understanding and Improving MeanFlow Models." arXiv:2510.20771.
6. Yan et al. (2025). "Decoupled MeanFlow." (cited in iMF as concurrent work).
7. Salimans, T. & Ho, J. (2022). "Progressive Distillation for Fast Sampling of Diffusion Models." ICLR 2022.
8. Karras, T. et al. (2024). "Analyzing and Improving the Training Dynamics of Diffusion Models." NeurIPS 2024.
9. Lin, S. et al. (2024). "Common Diffusion Noise Schedules and Sample Steps are Flawed." WACV 2024.
