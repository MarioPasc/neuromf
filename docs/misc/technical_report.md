# NeuroMF: Latent MeanFlow for One-Step 3D Brain MRI Synthesis

## Technical Report — Detailed Scientific Summary

**Author:** Mario Pascual-Gonzalez
**Date:** February 2026
**Status:** Phases 0-5 complete (code); Phase 4 best model trained; Phase 5 awaiting Picasso evaluation
**Supporting Notes:** `docs/misc/technical_report_notes.md`

---

## Table of Contents

1. [Motivation and Problem Statement](#1-motivation-and-problem-statement)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [Architecture Design](#3-architecture-design)
4. [Loss Function and Training Objective](#4-loss-function-and-training-objective)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Protocol](#6-training-protocol)
7. [Sampling and Generation](#7-sampling-and-generation)
8. [Experimental Results and Ablations](#8-experimental-results-and-ablations)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Implementation Details](#10-implementation-details)
11. [Novel Contributions](#11-novel-contributions)
12. [Current Status and Next Steps](#12-current-status-and-next-steps)
13. [References](#13-references)

---

## 1. Motivation and Problem Statement

### 1.1 Clinical Need

Generative models for 3D brain MRI synthesis serve three clinical purposes: (i) data augmentation for rare pathologies with scarce training data, (ii) synthetic cohorts for privacy-preserving research, and (iii) counterfactual generation for explainability. The critical bottleneck in existing methods is **sampling cost**: state-of-the-art approaches (DDPM, flow matching, rectified flow) require 5-1000 network evaluations per volume, making large-scale synthesis prohibitively slow.

### 1.2 Gap in the Literature

No prior work has applied MeanFlow — or any 1-step flow-based model — to 3D medical image synthesis. The closest works are:

| Method | Space | Steps (NFE) | Paradigm | Domain |
|--------|-------|-------------|----------|--------|
| MAISI-v2 (Zhao et al., 2025) | Latent | 5-50 | Rectified Flow | 3D CT/MRI |
| MOTFM (Yazdani et al., 2025) | Pixel | 10-50 | OT Flow Matching | 3D Brain MRI |
| Med-DDPM (Dorjsembe et al., 2024) | Pixel | 1000 | DDPM | 3D Brain MRI |
| pMF (Lu et al., 2026) | Pixel | 1 | Progressive MeanFlow | 2D Natural Images |
| **NeuroMF (ours)** | **Latent** | **1** | **MeanFlow (iMF dual-head)** | **3D Brain MRI** |

Our work fills the intersection: **1-step + latent + 3D + medical**.

### 1.3 Approach Overview

NeuroMF trains a MeanFlow model in the latent space of a frozen MAISI 3D VAE. The core pipeline is:

```
Input MRI (1x192^3) --> Frozen MAISI VAE Encoder --> Latent (4x48^3)
                                                         |
                                                  Train MeanFlow
                                                         |
                                                         v
Synthetic MRI (1x192^3) <-- Frozen MAISI VAE Decoder <---'
```

At inference, a **single forward pass** through the MeanFlow network generates a complete 3D brain MRI volume. The network learns the *average velocity* of a probability flow ODE, which by construction encodes the entire transport from noise to data in one evaluation.

---

## 2. Theoretical Foundations

### 2.1 Flow Matching Preliminaries

Flow matching (Lipman et al., 2023; Liu et al., 2023) defines a probability path between data and noise via linear interpolation:

```
z_t = (1 - t) * z_0 + t * eps,    z_0 ~ p_data, eps ~ N(0, I),    t in [0, 1]
```

where `t=0` is data and `t=1` is noise. The conditional velocity field is `v_c(z_t, t) = eps - z_0`, and a neural network is trained to match this field: `L_FM = E[||v_theta(z_t, t) - v_c||^2]`.

**Limitation:** Sampling requires integrating the learned velocity field from t=1 to t=0 via numerical ODE solvers, requiring K >= 5 network evaluations even with trajectory straightening (rectified flow).

### 2.2 MeanFlow: Average Velocity for 1-Step Generation

MeanFlow (Geng et al., 2025a) replaces the instantaneous velocity `v(z_t, t)` with the **average velocity** over an interval [r, t]:

```
u(z_t, r, t) = (1 / (t - r)) * integral_r^t v(z_s, s) ds
```

The key insight: if the average velocity from t=0 to t=1 is known exactly, the entire flow can be computed in one step:

```
z_0 = z_1 - 1 * u_theta(z_1, 0, 1)     [1-NFE generation]
```

### 2.3 The MeanFlow Identity and Self-Consistency

The average velocity u must satisfy a self-consistency condition. Differentiating the integral definition and applying the chain rule yields the **MeanFlow Identity**:

```
v(z_t, t) = u(z_t, r, t) + (t - r) * [du/dz_t * v(z_t, t) + du/dt]
```

The right-hand side, evaluated with the neural approximation, gives the **compound velocity**:

```
V_theta = u_theta + (t - r) * sg[JVP(u_theta, (z_t,t,r), (v_tilde, 1, 0))]
```

where `sg[.]` denotes stop-gradient and the JVP (Jacobian-Vector Product) is:

```
JVP = du/dz_t * v_tilde + du/dt * 1 + du/dr * 0
```

This JVP is computed in O(d) time via forward-mode automatic differentiation (`torch.func.jvp`), avoiding the O(d^2) cost of constructing the full Jacobian. The tangent vector `v_tilde` is the model's own estimate of the instantaneous velocity at r=t: `v_tilde = u_theta(z_t, t, t)`.

### 2.4 Improved MeanFlow (iMF) and the Dual-Head Architecture

The original MeanFlow uses the ground-truth velocity `v_c = eps - z_0` as the JVP tangent. **Problem:** this creates a dependency on data at inference time (where we don't have z_0). The improved MeanFlow (iMF; Geng et al., 2025b) resolves this by using the model's own prediction as the tangent, making V a function of z_t alone.

**The dual-head extension** (our architecture, inspired by iMF) introduces two output heads on a shared UNet backbone:

- **u-head** (main): Predicts the average velocity u (or equivalently, x_hat in x-prediction mode). Used at inference.
- **v-head** (auxiliary): Predicts the instantaneous velocity, directly supervised against v_c. Provides the JVP tangent during training. **Disabled at inference** (zero cost).

The v-head solves a critical practical problem: without it, the tangent comes from the u-head's own prediction, which is poor in early training, creating a bootstrapping problem that causes loss divergence. The v-head receives direct supervision (`||v - v_c||^p`), providing high-quality tangents from the first epoch.

### 2.5 x-Prediction Reparameterisation

Instead of directly outputting the average velocity u, the network outputs a denoised data estimate x_hat (x-prediction, following pMF; Lu et al., 2026):

```
x_hat = net_theta(z_t, r, t)
u_theta = (z_t - x_hat) / max(t, t_min)
```

**Justification (manifold hypothesis):** The x-prediction target lies on the data manifold, which has low intrinsic dimensionality. The velocity u spans a higher-dimensional space. For architectures with a bottleneck (UNet encoder-decoder), predicting a low-dimensional target (x_hat) is easier than predicting a high-dimensional one (u).

**Quantitative criterion (pMF Table 2):** x-prediction dominates when `d_input / d_bottleneck > 1`. For our 3D UNet: `d_input = 4 x 48^3 = 442,368` vs `d_bottleneck = 512 x 6^3 = 110,592`, giving ratio ~4, firmly in the x-prediction regime.

**At inference (1-NFE):** With x-prediction, 1-step sampling simplifies to:

```
z_0 = model(noise, r=0, t=1)     [the output IS the denoised data]
```

No velocity-to-data conversion is needed — the model directly outputs the synthetic latent.

### 2.6 Latent Space Formulation

All of the above operates identically in latent space. The computational advantages are:

1. **JVP cost reduction:** Latent dimension d = 4 x 48^3 = 442,368 vs pixel space d = 192^3 = 7,077,888 — a **16x reduction** in JVP compute per iteration.
2. **Memory reduction:** UNet operates on 48^3 feature maps vs 192^3 — approximately **64x reduction** in activation memory per JVP pass.
3. **Training efficiency:** The latent space is pre-computed (Phase 1), so VAE encode/decode cost is amortised across epochs.

---

## 3. Architecture Design

### 3.1 Frozen MAISI VAE (Encoder-Decoder)

The MAISI VAE (Guo et al., 2024) is a 3D variational autoencoder with adversarial training, pre-trained on ~55K CT+MRI volumes. We use it as a **frozen foundation model** — all parameters are locked, and it is never fine-tuned.

| Property | Value |
|----------|-------|
| Parameters | 20,944,897 (~21M) |
| Encoder stages | 3 levels of 2x strided 3D convolution |
| Latent channels | 4 (with KL regularisation) |
| Spatial compression | 4x per axis: 192^3 -> 48^3 |
| Attention | None (all `attention_levels=false`) |
| Training losses | L1 + LPIPS perceptual + PatchGAN adversarial + KL |
| Checkpoint format | Wrapped in `"unet_state_dict"` key |
| scale_factor | 0.96240234375 (extracted from diffusion checkpoint) |
| Memory optimisation | `num_splits` parameter for chunk-based processing |

**Reconstruction quality (Phase 0, 20 IXI volumes):** Mean SSIM = 0.9213, Mean PSNR = 30.86 dB. This establishes the VAE as a faithful encoder-decoder for brain MRI, with acceptable smoothing in cortical boundary regions.

**Scale factor:** The decode operation divides by scale_factor before passing to the decoder: `x_hat = decoder(z / 0.9624)`. This calibration ensures the latent distribution matches the prior used during VAE training. The scale_factor was extracted from the MAISI diffusion checkpoint (`diff_unet_3d_rflow-mr.pt["scale_factor"]`), not the VAE checkpoint.

### 3.2 MeanFlow UNet (Generative Model)

The MeanFlow UNet (`MAISIUNetWrapper`) uses the **same architecture** as the MAISI diffusion UNet but with random initialisation and custom dual-time conditioning.

| Property | Value |
|----------|-------|
| Backbone | MONAI `DiffusionModelUNet` (3D) |
| Total parameters | ~178M |
| Channels per level | [64, 128, 256, 512] |
| Attention levels | [false, false, true, true] — only at 12^3 and 6^3 resolution |
| Attention heads | 32 channels per head at attention levels |
| ResBlocks per level | 2 |
| GroupNorm groups | 32 |
| Transformer layers | 1 per attention level |
| Flash attention | **Disabled** (required for `torch.func.jvp` compatibility) |
| Gradient checkpointing | **Disabled** (required for exact JVP forward-mode AD) |
| ResBlock downsampling | Strided convolution (not pooling) |
| Prediction type | x-prediction (network outputs denoised data estimate) |

**Flash attention incompatibility:** `torch.func.jvp` uses forward-mode automatic differentiation, which requires the computation graph to be fully traceable. Flash attention's fused CUDA kernels are opaque to PyTorch's AD system. Disabling flash attention adds ~10% latency per forward pass but enables exact JVP computation.

**Gradient checkpointing incompatibility:** Gradient checkpointing re-executes forward passes during backward, which conflicts with the forward-mode AD tape maintained by `torch.func.jvp`. Both cannot be active simultaneously.

### 3.3 Dual-Time Conditioning

MeanFlow requires conditioning on two time variables: t (current time) and r (interval lower bound). We implement three conditioning modes and select `t_h` based on ablation:

| Mode | Inputs | Embedding | Source |
|------|--------|-----------|--------|
| `dual` | (r, t) | sin(r) + sin(t) through separate MLPs | pMF convention |
| `h` | h = t-r | sin(h) through UNet's built-in MLP | iMF convention |
| **`t_h`** | **(t, h=t-r)** | **sin(t) through UNet MLP + sin(h) through new h_embed MLP** | **MF Table 1c optimal** |

The `t_h` mode conditions on both the absolute time t and the interval width h = t-r. This is strictly more informative than h-only conditioning (the model can distinguish between "near data at t=0.1" and "near noise at t=0.9" even when h is the same).

**Implementation:** Continuous time values in [0,1] are scaled by 1000 before computing sinusoidal embeddings (`_TIME_SCALE = 1000.0`). This prevents degenerate embeddings — with the standard `max_period=10000`, times in [0,1] would produce nearly constant embeddings. After sinusoidal encoding, each embedding passes through a 2-layer MLP (`Linear(64, 256) -> SiLU -> Linear(256, 256)`), and the two embeddings are summed: `emb = t_emb + h_emb`.

### 3.4 v-Head (Auxiliary Tangent Predictor)

The v-head is a lightweight auxiliary output path branching from the UNet's final feature map (before the main output convolution):

```
Shared Backbone Features (B, 64, 48, 48, 48)
    |
    +-- u-head: self.unet.out(h)  -->  (B, 4, 48, 48, 48)  [main output]
    |
    +-- v-head: ResBlock(64) -> GroupNorm(32,64) -> SiLU -> Conv3d(64,4,3,pad=1)  -->  (B, 4, 48, 48, 48)
```

| Property | Value |
|----------|-------|
| ResBlocks | 1 (configurable via `v_head_num_res_blocks`) |
| Parameters | ~228K (<0.13% of total model) |
| Initialisation | Final Conv3d is zero-initialised (v-head starts at zero) |
| Inference cost | Zero (v-head output is discarded at sampling time) |

**Zero initialisation rationale:** By initialising the v-head's final convolution to zero, the initial v-head output is identically zero. This means the u-head's training is completely unaffected by the v-head at initialisation. As training progresses, the v-head learns to predict the instantaneous velocity, providing an increasingly accurate tangent for the JVP computation.

---

## 4. Loss Function and Training Objective

### 4.1 Combined iMF Dual-Head Loss

The training loss has two independently-weighted components:

```
L = L_u_weighted + L_v_weighted
```

where:

- **L_u (compound velocity loss):** `||V_theta - v_c||_p^p` — enforces MeanFlow self-consistency
- **L_v (tangent supervision loss):** `||v_head - v_c||_p^p` — directly supervises the v-head

Both losses target the same conditional velocity `v_c = eps - z_0`. The compound velocity V incorporates the MeanFlow identity correction, while v_head is a direct prediction.

### 4.2 Adaptive Weighting

Each loss component is independently normalised by its own magnitude:

```
weight_u = (raw_loss_u.detach() + norm_eps) ^ norm_p
loss_u_weighted = raw_loss_u / weight_u

weight_v = (raw_loss_v.detach() + norm_eps) ^ norm_p
loss_v_weighted = raw_loss_v / weight_v
```

With `norm_eps=1.0` and `norm_p=1.0`:

```
weight = raw_loss + 1.0
loss_weighted = raw_loss / (raw_loss + 1.0)
```

This is a form of **logarithmic loss normalisation**: the effective loss is bounded in [0, 1) and varies slowly with raw_loss magnitude. It prevents large loss spikes from destabilising training and equalises the gradient contribution across timesteps (where raw loss naturally varies by orders of magnitude due to the signal-to-noise ratio at different t values).

**Why norm_eps=1.0 and not smaller?** Through Phase 4c debugging, we discovered that `norm_eps=0.01` caused catastrophic gradient amplification for samples with small raw loss (near-perfect predictions). The weight `1/(loss + 0.01)` can reach 100x, creating a positive feedback loop. Setting `norm_eps=1.0` caps the maximum amplification at a benign level.

**Why norm_p=1.0 and not 0.5?** `norm_p=0.5` causes `weight = sqrt(raw_loss + 1.0)`, which under-normalises large losses, leading to a 1000x gradient explosion observed in Phase 4e testing.

### 4.3 Per-Channel Lp Loss

The base loss function computes a per-channel, spatially-summed Lp norm:

```
lp_loss(pred, target, p) = sum_c sum_spatial |pred_c - target_c|^p
```

For the best model: `p=2.0` (standard L2 loss). The per-channel Lp framework supports ablation over p in {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0} and per-channel weight vectors, extending the SLIM-Diff per-channel loss (Pascual-Gonzalez et al., 2026) from pixel-space DDPM to latent-space MeanFlow.

**Key scientific question:** Does the optimal Lp exponent from pixel space (p=1.5 for images, p=2.0 for masks) transfer through the VAE nonlinearity to latent space? The VAE's encoder Jacobian mixes spatial and channel information, so the error distribution in latent space may have different statistical properties than in pixel space.

### 4.4 JVP Strategies

Two strategies are implemented:

**Exact JVP** (`torch.func.jvp`):
```python
u, du_dt = torch.func.jvp(u_fn, (z_t, t, r), (v_tangent, dt=1, dr=0))
V = u + (t-r) * sg[du_dt]
```
- Most accurate, O(d) cost
- Requires: no in-place ops, no flash attention, no gradient checkpointing
- Memory: ~20GB activation per sample at batch=2 on A100-40GB

**Finite Difference JVP** (FD-JVP):
```python
u = u_fn(z_t, t, r)
u_perturbed = u_fn(z_t + h*v_tangent, t + h, r)  # no_grad
du_dt = (u_perturbed.float() - u.detach().float()) / h
V = u + (t-r) * sg[du_dt]
```
- Step size h=1e-3
- FP32 subtraction to avoid bf16 catastrophic cancellation
- Lower memory (perturbed pass is no_grad)

**Critical incompatibility discovered (Phase 4f):** x-prediction + FD-JVP is **numerically unstable**. The x-to-u conversion `u = (z_t - x_hat) / t` has a 1/t singularity. When FD-JVP computes `(u(t+h) - u(t)) / h`, the result involves `O(1/t)` terms divided by h=0.001, yielding `du/dt ~ O(1/t^2)` which explodes as t approaches t_min. With exact JVP, the 1/t factor is analytically differentiated, yielding stable gradients.

**Rule:** `x-prediction + exact JVP = stable`. `u-prediction + FD-JVP = stable`. `x-prediction + FD-JVP = explosion`.

For dual-head models, exact JVP uses `has_aux=True` to capture the v-head output alongside the JVP computation, avoiding redundant forward passes:

```python
u, du_dt, v = torch.func.jvp(
    u_with_v_aux, (z_t, t, r), (v_tangent, dt, dr), has_aux=True
)
```

---

## 5. Data Pipeline

### 5.1 Dataset

We use a 3-dataset subset of FOMO-60K, a large-scale preprocessed brain MRI collection:

| Dataset | N subjects | Properties |
|---------|-----------|------------|
| PT001_OASIS1 | 436 | T1w, skull-stripped, RAS, co-registered |
| PT002_OASIS2 | 362 | T1w, skull-stripped, RAS, co-registered |
| PT005_IXI | 581 | T1w, skull-stripped, RAS, co-registered |
| **Total** | **1,379** | 85/10/5 split -> ~1,172 train / 138 val / 69 test |

The split is stratified by dataset to ensure proportional representation. Split seed is fixed at 42 for reproducibility.

### 5.2 Preprocessing Pipeline

```
NIfTI (.nii.gz)
  -> LoadImaged (MONAI)
  -> EnsureChannelFirstd
  -> Spacingd(pixdim=1.0mm isotropic, bilinear)
  -> ScaleIntensityRangePercentilesd(lower=0%, upper=99.5%, b_min=0, b_max=1, clip=False)
  -> CropForegroundd(source_key="image", margin=4)     [brain-centered crop]
  -> ResizeWithPadOrCropd(spatial_size=(192,192,192))   [pad to target]
  -> EnsureTyped(dtype=float32)
```

**Resolution choice (192^3 at 1.0mm isotropic):** This was selected after a quantitative analysis of brain extent across 30 FOMO-60K subjects (see `docs/data/resolution_analysis.md`). Key findings:

- Brain AP extent reaches 193mm for the largest subjects
- The brain is systematically 13mm anterior to the volume center
- 128^3 and 160^3 both clip frontal/occipital cortex
- `CropForegroundd` centers the crop on brain tissue (not volume center), preventing systematic tissue loss
- 192^3 achieves 100% brain coverage for all subjects

**Latent shape:** 192 / 4 = 48 per spatial axis, giving latents of shape (4, 48, 48, 48).

### 5.3 Latent Pre-Computation (Phase 1)

All training volumes are pre-encoded through the frozen MAISI VAE and stored in HDF5 shards:

| Property | Value |
|----------|-------|
| Storage format | Per-dataset HDF5 shards (e.g. `PT005_IXI.h5`) |
| Layout per shard | `/latents` (N,4,48,48,48), `/written` (bool mask), `/subject_id`, `/session_id` |
| Per-sample size | 4 x 48^3 x 4 bytes = 1.77 MB (float32) |
| Total storage | ~1,379 x 1.77 MB ~ 2.4 GB |
| Resumability | Written mask tracks completed slots |

**Latent statistics** are computed online during encoding via `LatentStatsAccumulator` (sum-of-powers method in float64):
- Per-channel mean, std, skewness, kurtosis, min, max
- 4x4 cross-channel Pearson correlation matrix

These statistics serve dual purposes: (i) per-channel normalisation for training, and (ii) quality assessment of the latent distribution.

### 5.4 Latent Augmentation

Three augmentation transforms operate directly on latents during training (enabled on Picasso):

| Transform | Probability | Parameters | Rationale |
|-----------|-------------|------------|-----------|
| Flip depth axis | 0.5 | Along dim=2 (L-R in RAS) | Brain is approximately bilaterally symmetric in this axis |
| Gaussian noise | 0.2 | std = 5% of per-channel std | Regularisation; simulates encoding noise |
| Intensity scale | 0.2 | +/- 5% scaling | Simulates scanner-dependent intensity variation |

Augmentation is critical for 1500-epoch training on ~1,172 samples (effective epochs-per-sample ~1,280).

---

## 6. Training Protocol

### 6.1 Algorithm (One Training Step)

```
Input: batch of pre-computed latents {z_0}, model with u-head and v-head

1.  Sample eps ~ N(0, I), same shape as z_0
2.  Sample t ~ LogitNormal(mu=-0.4, sigma=1.0), clamp to [0.001, 1.0]
3.  With probability 0.5: set r = t (FM sample)
    Otherwise: sample r independently, enforce r < t (MF sample)
4.  Compute z_t = (1-t) * z_0 + t * eps                       [interpolation]
5.  Compute v_c = eps - z_0                                     [target]
6.  v_tangent = v_head(z_t, t, t)  [no_grad]                   [tangent from v-head]
7.  (u, du/dt, v) = JVP(dual_fn, (z_t,t,r), (v_tangent,1,0))  [exact JVP with aux]
8.  V = u + (t-r) * sg[du/dt]                                   [compound velocity]
9.  raw_loss_u = ||V - v_c||_2^2                                [MF consistency loss]
10. raw_loss_v = ||v - v_c||_2^2                                [tangent supervision]
11. loss = adaptive(raw_loss_u) + adaptive(raw_loss_v)           [independent weighting]
12. Backward, clip gradients (max_norm=1.0), optimizer step
13. EMA update (decay=0.9999)
```

### 6.2 Time Sampling

Times are drawn from a logit-normal distribution: `t = sigmoid(N(mu=-0.4, sigma=1.0))`, clamped to `[0.001, 1.0]`. This distribution is heavy near t=0 (near data), which focuses training on the denoising regime. The negative mean (mu=-0.4) shifts the mass toward smaller t values compared to uniform sampling.

**data_proportion = 0.5:** Half the batch has r = t (FM samples where the JVP term vanishes, reducing to standard flow matching loss), and half has r < t (MF samples where the self-consistency constraint is enforced). This 50/50 split was found optimal empirically — pure MF samples (data_proportion=0) diverge, while too many FM samples (data_proportion=0.75) slow convergence of the 1-NFE capability.

### 6.3 Optimiser and Schedule

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimiser | AdamW | Standard for diffusion/flow models |
| Learning rate | 1e-4 | MeanFlow reference default |
| Weight decay | 0.0 | MeanFlow reference (no regularisation needed with adaptive loss) |
| Betas | (0.9, 0.95) | beta2=0.95 for faster momentum adaptation (default 0.999 caused stale gradients) |
| LR schedule | Cosine decay | From lr to 0 over total steps |
| Warmup | 0 steps | Warmup destabilised MF loss (Phase 4c finding) |
| Gradient clip | max_norm=1.0 | Prevents spike propagation |
| Mixed precision | bf16 | Standard for A100 training |

### 6.4 Compute Budget

| Property | Value |
|----------|-------|
| Hardware | 6x A100-SXM4-40GB (within one DGX node) |
| Strategy | DDP (DistributedDataParallel) |
| Per-GPU batch | 2 (limited by exact JVP activation memory ~20GB) |
| Gradient accumulation | 11 steps |
| Effective batch | 2 x 6 x 11 = 132 (~128) |
| Training steps | ~1,172 / 2 / 6 = ~98 batches/epoch x 1500 epochs / 11 accum = ~13,364 optimizer steps |
| Estimated wall time | ~7 days |

### 6.5 EMA (Exponential Moving Average)

An EMA model with decay 0.9999 is maintained throughout training. After each optimiser step, shadow parameters are updated:

```
shadow_param = 0.9999 * shadow_param + 0.0001 * current_param
```

At inference time, the EMA weights replace the model parameters. This smooths over training noise and typically improves generation quality by 10-20% in FID.

### 6.6 Divergence Monitoring

An EMA-smoothed raw loss monitor tracks training stability:
- EMA half-life: ~69 steps (decay=0.99)
- Warnings at 3x and 5x the minimum EMA loss
- Grace period of 1000 steps before monitoring begins
- Warning-only (no automatic stopping) — early stopping is FID-based via EvaluationCallback

---

## 7. Sampling and Generation

### 7.1 One-Step Sampling (1-NFE)

With x-prediction, 1-NFE sampling is a single forward pass:

```python
noise = torch.randn(B, 4, 48, 48, 48)    # Sample from prior
r = torch.zeros(B)                         # Full interval
t = torch.ones(B)
z_0_hat = model(noise, r, t)              # Direct x-prediction output
# Dual-head: output is (x_hat, v) — take first element
```

The v-head output is discarded at inference — only the u-head (main output) is used.

### 7.2 Multi-Step Euler Sampling

For comparison, multi-step Euler sampling divides [1, 0] into uniform steps:

```
For i in 0..n_steps-1:
    t_curr = 1 - i/n_steps
    t_next = 1 - (i+1)/n_steps
    dt = t_curr - t_next
    x_hat = model(z, r=t_next, t=t_curr)
    u = (z - x_hat) / max(t_curr, 0.05)     # x-to-u conversion
    z = z - dt * u
```

NFE levels tested: {1, 2, 5, 10}. MeanFlow models are trained for 1-step generation but can also be used multi-step by treating them as standard velocity fields.

### 7.3 Full Generation Pipeline (Phase 5)

The complete generation pipeline from noise to decoded MRI volume:

```
1. Pre-generate shared noise (for NFE-consistency analysis)
2. For each NFE level:
   a. Generate latents via one_step or euler sampling
   b. Store in HDF5 archive (latents, seeds, timing)
3. Denormalise: z_0 = z_hat * latent_std + latent_mean
4. Decode: x_hat = VAE.decode(z_0)  [internally divides by scale_factor]
5. Clamp to [0, 1]
6. Store decoded volumes in HDF5
```

**Shared noise protocol:** The same noise tensor is used across all NFE levels for a given sample index, enabling direct visual and quantitative comparison of how additional sampling steps refine the output.

---

## 8. Experimental Results and Ablations

### 8.1 Toy Validation (Phase 2): MeanFlow on Flat Torus

Before scaling to brain MRI, MeanFlow was validated on a tractable manifold — a flat torus in R^4:

| Metric | Result | Target |
|--------|--------|--------|
| 1-NFE torus distance | 0.0892 | < 0.1 |
| 5-step torus distance | 0.0271 | < 0.05 |
| Angular KS p-value (theta1) | 0.374 | > 0.01 |
| Angular KS p-value (theta2) | 0.575 | > 0.01 |
| x-pred vs u-pred loss ratio | < 1.2 | < 1.5 |

This confirmed: (i) the MeanFlow implementation correctly learns 1-step generation, (ii) the angular distribution of generated samples is statistically indistinguishable from the true distribution (KS test), and (iii) both x-prediction and u-prediction converge.

### 8.2 Main Training (Phase 4): Best Model

The best model uses x-prediction + exact JVP + (t,h) conditioning + v-head (1 ResBlock) + L2 loss:

| Property | Value |
|----------|-------|
| Best epoch | 589 |
| Configuration | `experiments/ablations/xpred_vs_upred/configs/xpred_exact_jvp.yaml` |
| Prediction type | x-prediction |
| JVP strategy | Exact (`torch.func.jvp`) |
| Conditioning | t_h |
| v-head | 1 ResBlock |
| Loss | L2 (p=2) with adaptive weighting |

### 8.3 Key Ablation Insights

**x-pred vs u-pred:** The x-prediction configuration with exact JVP trained stably to epoch 589 (best raw loss checkpoint). The u-prediction + FD-JVP baseline collapsed around epoch 150. This supports the pMF manifold hypothesis extending to latent space with 3D UNets, though a controlled comparison with matched JVP strategies is needed for a definitive conclusion.

**x-pred + FD-JVP instability:** This combination produces exponential growth of JVP norms (10,000x within 50 epochs). Root cause: the x-to-u conversion `u = (z_t - x_hat) / t` introduces a 1/t factor; finite difference of two O(1/t) terms divided by h=0.001 amplifies numerical error as `O(1/(t^2 * h))`. Exact JVP analytically differentiates through the 1/t factor, avoiding this issue.

**Conditioning mode:** `t_h` (condition on both t and h=t-r) was selected based on the original MeanFlow paper's Table 1c, which reports FID 61.06 for (t,h) vs 63.13 for h-only on ImageNet.

**Training stability fixes (Phase 4c-4e):**
- beta2: 0.999 -> 0.95 (faster momentum adaptation prevents stale gradient accumulation)
- warmup: 5000 -> 0 (linear warmup destabilises MF loss by producing inconsistent gradient magnitudes during ramp-up)
- norm_eps: 0.01 -> 1.0 (prevents runaway gradient amplification in adaptive weighting)
- norm_p: 0.5 -> 1.0 (sub-linear normalisation causes gradient explosion)
- data_proportion: stabilised at 0.5 (50/50 FM/MF split)

---

## 9. Evaluation Protocol

### 9.1 Image Quality Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| 2.5D FID | FID on central axial/coronal/sagittal slices using RadImageNet ResNet-50 | Custom; follows MOTFM protocol |
| 3D FID | FID on volumetric features from Med3D ResNet-50 | Custom; `src/neuromf/metrics/fid_3d.py` |
| MMD | Maximum Mean Discrepancy in Med3D feature space | Gaussian kernel, `src/neuromf/metrics/mmd.py` |
| Coverage | Fraction of real samples with a close synthetic neighbour | Feature-space NN, `src/neuromf/metrics/coverage_density.py` |
| Density | Fraction of synthetic samples near real manifold | Complementary to coverage |
| MS-SSIM | 3-level Multi-Scale SSIM (3D) | MONAI SSIMMetric, `src/neuromf/metrics/ms_ssim_3d.py` |
| PSNR | Peak Signal-to-Noise Ratio | Standard formula with actual data range |

### 9.2 Spectral Analysis

High-frequency energy ratio via 3D FFT:
```
rho = sum_{|k| > k0} |F(x)|^2 / sum_k |F(x)|^2
```

Reported for real, VAE-reconstructed, and generated volumes to disentangle VAE smoothing from generative model quality.

### 9.3 Morphological Assessment

SynthSeg (Billot et al., 2023) segmentation on both real and synthetic volumes, computing:
- Regional volume correlation (hippocampus, ventricles, cortex)
- Distribution of regional volumes (KL divergence real vs synthetic)
- SynthSeg Dice overlap (paired by nearest-neighbour in feature space)

### 9.4 NFE Consistency Analysis

Generated samples at NFE = {1, 2, 5, 10} using shared noise, compared via:
- Per-sample L2 distance between NFE=1 and NFE=K (convergence measure)
- FID at each NFE level (quality vs compute trade-off)

---

## 10. Implementation Details

### 10.1 Software Stack

| Component | Version/Tool |
|-----------|-------------|
| Python | 3.11.14 |
| PyTorch | 2.10+cu128 |
| MONAI | 1.5.2 |
| PyTorch Lightning | 2.6.1 |
| Config management | OmegaConf / Hydra |
| Data storage | HDF5 (h5py) |
| Logging | Python logging + Rich handler + TensorBoard |

### 10.2 Project Structure

```
neuromf/
  src/neuromf/
    callbacks/         # PL callbacks: diagnostics, evaluation, sample_collector, performance
    data/              # Datasets: latent_dataset, latent_hdf5, mri_preprocessing, fomo60k
    errors/            # Custom exceptions
    generation/        # Phase 5: H5Manager, LatentGenerator, VolumeDecoder
    losses/            # lp_loss, meanflow_jvp, combined_loss
    metrics/           # fid, fid_3d, mmd, ms_ssim_3d, spectral, coverage_density, ...
    models/            # latent_meanflow (PL module), toy_mlp, lora, rectflow_baseline
    sampling/          # one_step, multi_step (Euler)
    utils/             # ema, time_sampler, latent_stats, pretrained_loading, ...
    wrappers/          # maisi_vae, maisi_unet, jvp_strategies, meanflow_loss
  configs/             # YAML configs (base, train_meanflow, generate, picasso overlays)
  experiments/
    cli/               # Training/evaluation CLIs
    ablations/         # Ablation configs and results
    slurm/             # SLURM launcher + worker scripts per phase
  tests/               # Pytest suite (58+ tests, phase-gated)
  docs/                # Methodology, splits, explorations, papers
```

### 10.3 Gated Phase System

The project is implemented in 9 gated phases (0-8). Phase N+1 cannot begin until Phase N's critical tests all pass.

| Phase | Title | Status | Tests |
|-------|-------|--------|-------|
| 0 | VAE Validation | Complete | 7/7 PASS |
| 1 | Latent Pre-computation | Complete | 7/7 PASS |
| 2 | Toy Experiment (Torus) | Complete | 6/6 CRITICAL + 2 INFO |
| 3 | MeanFlow Loss + 3D UNet | Complete | 28/28 PASS |
| 4 | Training on Brain MRI | Complete | 12+ PASS |
| 5 | Generation + Evaluation | Code complete | 11/11 local PASS |
| 6 | Ablation Runs | Planned | — |
| 7 | LoRA Fine-Tuning (FCD) | Planned | — |
| 8 | Paper Figures | Planned | — |

### 10.4 Compute Infrastructure

| Resource | Local | Picasso |
|----------|-------|---------|
| GPU | RTX 4060 8GB | 4 nodes x 8x A100-40GB |
| Use case | Development, testing, analysis | Training, encoding, evaluation |
| SLURM | N/A | `--constraint=dgx` |
| Conda env | `neuromf` | `neuromf` |

---

## 11. Novel Contributions

### Contribution 1: First MeanFlow for 3D Medical Image Synthesis

We bring the MeanFlow framework — previously applied only to 2D natural images — to 3D volumetric medical imaging. The model achieves 1-NFE generation of 192^3 brain MRI volumes in the latent space of a frozen MAISI VAE. This represents a **50-1000x reduction in sampling cost** compared to existing 3D medical generative models (DDPM: 1000 steps; flow matching: 10-50 steps; rectified flow: 5-50 steps).

### Contribution 2: Per-Channel Lp Loss in Latent MeanFlow

We extend the SLIM-Diff per-channel Lp loss from pixel-space DDPM to latent-space MeanFlow. This provides a principled framework for channel-specific loss geometry that will be ablated over p in {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}, investigating whether the optimal exponent from pixel space transfers through the VAE nonlinearity.

### Contribution 3: x-Prediction vs u-Prediction Ablation in Latent 3D UNets

We provide the first investigation of the x-prediction vs u-prediction dichotomy (established by pMF for 2D ViTs) in latent space with 3D UNets. Initial results support x-prediction superiority, consistent with the manifold hypothesis, but the VAE's pre-compression may attenuate the gap. This is a novel empirical contribution regardless of outcome.

### Contribution 4: iMF Dual-Head in Medical Context

We adapt the iMF dual-head architecture (shared backbone + u-head + v-head) for 3D medical image synthesis. The v-head provides directly-supervised JVP tangents, solving the MeanFlow loss divergence problem that occurs in early training. Our implementation is the first application of this architecture outside the original iMF/pMF natural image setting.

### Contribution 5 (Planned): LoRA Fine-Tuning for Joint Image-Mask Synthesis

Phase 7 will demonstrate LoRA fine-tuning of the pre-trained MeanFlow model for joint synthesis of FLAIR MRI and FCD segmentation masks, combining 1-step generation with per-channel Lp loss in a clinically relevant data-scarce setting (~50-100 FCD cases).

---

## 12. Current Status and Next Steps

### 12.1 Current State (as of 2026-02-26)

- **Phases 0-4:** Complete. Best model trained (x-pred + exact JVP, epoch 589).
- **Phase 5:** Code complete, 11/11 local tests pass. Awaiting Picasso execution for:
  - Latent generation at NFE={1,2,5,10}
  - VAE decoding of generated latents
  - Quantitative evaluation (FID, MMD, MS-SSIM, spectral, morphological)
- **Phases 6-8:** Planned.

### 12.2 Immediate Next Steps

1. **Run Phase 5 on Picasso** — generate samples and compute metrics
2. **Phase 6 ablations:**
   - x-pred vs u-pred (controlled: same JVP strategy, multiple seeds)
   - Lp sweep: p in {1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}
   - NFE trade-off: quality vs steps at {1, 2, 5, 10, 25, 50}
3. **Phase 7:** LoRA fine-tuning for FCD joint synthesis
4. **Phase 8:** Publication-ready figures and tables

### 12.3 Open Questions

- Does the Lp exponent effect from pixel space transfer to latent space?
- How much does the VAE smoothing artefact degrade MeanFlow-generated volumes compared to multi-step methods?
- Can LoRA fine-tuning on ~50-100 FCD cases produce anatomically plausible lesion synthesis?

---

## 13. References

1. Geng, Z., Deng, M., Bai, X., Kolter, J.Z., He, K. "Mean Flows for One-step Generative Modeling." NeurIPS 2025 (Oral). arXiv:2505.13447.
2. Geng, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J.Z., He, K. "Improved Mean Flows: On the Challenges of Fastforward Generative Models." arXiv:2512.02012, 2025.
3. Lu, Y., Lu, S., Sun, Q., Zhao, H., et al. "One-step Latent-free Image Generation with Pixel Mean Flows." arXiv:2601.22158, 2026.
4. Guo, P., Zhao, C., Yang, D., et al. "MAISI: Medical AI for Synthetic Imaging." WACV 2025. arXiv:2409.11169, 2024.
5. Zhao, C., Guo, P., Yang, D., et al. "MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow." arXiv:2508.05772, 2025.
6. Pascual-Gonzalez, M., et al. "SLIM-Diff: Shared Latent Image-Mask Diffusion with Lp Loss for Data-Scarce Epilepsy FLAIR MRI." arXiv:2602.03372, 2026.
7. Yazdani, M., et al. "Flow Matching for Medical Image Synthesis." MICCAI 2025. arXiv:2503.00266.
8. Dorjsembe, Z., et al. "Semantic 3D Brain MRI Synthesis with Channel-Wise Conditioning." IEEE JBHI, 2024.
9. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M. "Flow Matching for Generative Modeling." ICLR, 2023.
10. Liu, X., Gong, C., Liu, Q. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR, 2023.
11. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR, 2022.
12. Hu, E., Shen, Y., Wallis, P., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.
13. Billot, B., et al. "SynthSeg: Segmentation of Brain MRI Scans of Any Contrast and Resolution." Medical Image Analysis, 2023.
14. Ho, J., Jain, A., Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020.
15. Puglisi, R., et al. "Brain Latent Progression (BrLP)." Medical Image Analysis, 2025.
