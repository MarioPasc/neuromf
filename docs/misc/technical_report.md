# NeuroMF: Latent MeanFlow for One-Step 3D Brain MRI Synthesis

## Technical Report — Detailed Scientific Summary

**Author:** Mario Pascual-Gonzalez
**Date:** February 2026
**Status:** Phases 0–5 complete (code); Phase 4 best model trained and evaluated; Phase 5 generation pipeline ready
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
8. [Experimental Results](#8-experimental-results)
9. [Evaluation Protocol](#9-evaluation-protocol)
10. [Implementation Details](#10-implementation-details)
11. [Novel Contributions](#11-novel-contributions)
12. [Current Status and Next Steps](#12-current-status-and-next-steps)
13. [References](#13-references)

---

## 1. Motivation and Problem Statement

### 1.1 Clinical Need

Generative models for 3D brain MRI synthesis serve three clinical purposes: (i) data augmentation for rare pathologies with scarce training data, (ii) synthetic cohorts for privacy-preserving research, and (iii) counterfactual generation for explainability. The critical bottleneck in existing methods is **sampling cost**: state-of-the-art approaches (DDPM, flow matching, rectified flow) require 5–1000 network evaluations per volume, making large-scale synthesis prohibitively slow.

### 1.2 Gap in the Literature

No prior work has applied MeanFlow — or any 1-step flow-based model — to 3D medical image synthesis. The closest works are:

| Method | Space | Steps (NFE) | Paradigm | Domain |
|--------|-------|-------------|----------|--------|
| MAISI-v2 (Zhao et al., 2025) | Latent | 5–50 | Rectified Flow | 3D CT/MRI |
| MOTFM (Yazdani et al., 2025) | Pixel | 10–50 | OT Flow Matching | 3D Brain MRI |
| Med-DDPM (Dorjsembe et al., 2024) | Pixel | 1000 | DDPM | 3D Brain MRI |
| pMF (Lu et al., 2026) | Pixel | 1 | Progressive MeanFlow | 2D Natural Images |
| **NeuroMF (ours)** | **Latent** | **1** | **MeanFlow (iMF dual-head)** | **3D Brain MRI** |

Our work fills the intersection: **1-step + latent + 3D + medical**.

### 1.3 Approach Overview

NeuroMF trains a MeanFlow model in the latent space of a frozen MAISI 3D VAE. The core pipeline is:

```
Input MRI (1×192³) ──► Frozen MAISI VAE Encoder ──► Latent (4×48³)
                                                         │
                                                  Train MeanFlow
                                                         │
                                                         ▼
Synthetic MRI (1×192³) ◄── Frozen MAISI VAE Decoder ◄───┘
```

At inference, a **single forward pass** through the MeanFlow network generates a complete 3D brain MRI volume. The network learns the *average velocity* of a probability flow ODE, which by construction encodes the entire transport from noise to data in one evaluation.

---

## 2. Theoretical Foundations

### 2.1 Flow Matching Preliminaries

Flow matching (Lipman et al., 2023; Liu et al., 2023) defines a probability path between data and noise via linear interpolation:

$$z_t = (1 - t)\, z_0 + t\, \varepsilon, \qquad z_0 \sim p_{\mathrm{data}},\quad \varepsilon \sim \mathcal{N}(0, I), \quad t \in [0, 1]$$

where $t = 0$ is data and $t = 1$ is noise. The conditional velocity field is $v_c(z_t, t) = \varepsilon - z_0$, and a neural network is trained to match this field:

$$\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{z_0, \varepsilon, t}\Big[\|v_\theta(z_t, t) - v_c\|^2\Big]$$

**Limitation:** Sampling requires integrating the learned velocity field from $t=1$ to $t=0$ via numerical ODE solvers, requiring $K \geq 5$ network evaluations even with trajectory straightening (rectified flow).

### 2.2 MeanFlow: Average Velocity for 1-Step Generation

MeanFlow (Geng et al., 2025a) replaces the instantaneous velocity $v(z_t, t)$ with the **average velocity** over an interval $[r, t]$:

$$u(z_t, r, t) = \frac{1}{t - r}\int_r^t v(z_s, s)\, ds$$

The key insight: if the average velocity from $t=0$ to $t=1$ is known exactly, the entire flow can be computed in one step:

$$z_0 = z_1 - u_\theta(z_1, 0, 1) \qquad \text{[1-NFE generation]}$$

### 2.3 The MeanFlow Identity and Self-Consistency

The average velocity $u$ must satisfy a self-consistency condition. Differentiating the integral definition and applying the chain rule yields the **MeanFlow Identity**:

$$v(z_t, t) = u(z_t, r, t) + (t - r)\left[\frac{\partial u}{\partial z_t}\, v(z_t, t) + \frac{\partial u}{\partial t}\right]$$

The right-hand side, evaluated with the neural approximation, gives the **compound velocity**:

$$V_\theta = u_\theta + (t - r) \cdot \mathrm{sg}\!\left[\mathrm{JVP}\!\left(u_\theta,\; (z_t, t, r),\; (\tilde{v}, 1, 0)\right)\right]$$

where $\mathrm{sg}[\cdot]$ denotes stop-gradient and the JVP (Jacobian-Vector Product) is:

$$\mathrm{JVP} = \frac{\partial u}{\partial z_t}\,\tilde{v} + \frac{\partial u}{\partial t}\cdot 1 + \frac{\partial u}{\partial r}\cdot 0$$

This JVP is computed in $O(d)$ time via forward-mode automatic differentiation (`torch.func.jvp`), avoiding the $O(d^2)$ cost of constructing the full Jacobian. The tangent vector $\tilde{v}$ is the model's own estimate of the instantaneous velocity at $r = t$: $\tilde{v} = u_\theta(z_t, t, t)$.

### 2.4 Improved MeanFlow (iMF) and the Dual-Head Architecture

The original MeanFlow uses the ground-truth velocity $v_c = \varepsilon - z_0$ as the JVP tangent. **Problem:** this creates a dependency on data at inference time (where $z_0$ is unavailable). The improved MeanFlow (iMF; Geng et al., 2025b) resolves this by using the model's own prediction as the tangent, making $V$ a function of $z_t$ alone.

**The dual-head extension** (our architecture, inspired by iMF) introduces two output heads on a shared UNet backbone:

- **u-head** (main): Predicts the average velocity $u$ (or equivalently, $\hat{x}$ in x-prediction mode). Used at inference.
- **v-head** (auxiliary): Predicts the instantaneous velocity, directly supervised against $v_c$. Provides the JVP tangent during training. **Disabled at inference** (zero cost).

The v-head solves a critical practical problem: without it, the tangent comes from the u-head's own prediction, which is poor in early training, creating a bootstrapping problem that causes loss divergence. The v-head receives direct supervision ($\|v - v_c\|^p$), providing high-quality tangents from the first epoch. In our training, the v-head cosine alignment $\cos(\hat{v}, v_c)$ reached 0.39 by epoch 50 — 85% of its final value of 0.44.

### 2.5 x-Prediction Reparameterisation

Instead of directly outputting the average velocity $u$, the network outputs a denoised data estimate $\hat{x}$ (x-prediction, following pMF; Lu et al., 2026):

$$\hat{x} = f_\theta(z_t, r, t), \qquad u_\theta = \frac{z_t - \hat{x}}{\max(t,\, t_{\min})}$$

**Justification (manifold hypothesis):** The x-prediction target lies on the data manifold, which has low intrinsic dimensionality. The velocity $u$ spans a higher-dimensional space. For architectures with a bottleneck (UNet encoder-decoder), predicting a low-dimensional target ($\hat{x}$) is easier than predicting a high-dimensional one ($u$).

**Quantitative criterion (pMF Table 2):** x-prediction dominates when $d_{\mathrm{input}} / d_{\mathrm{bottleneck}} > 1$. For our 3D UNet:

$$\frac{d_{\mathrm{input}}}{d_{\mathrm{bottleneck}}} = \frac{4 \times 48^3}{512 \times 6^3} = \frac{442{,}368}{110{,}592} \approx 4$$

This is firmly in the x-prediction regime.

**At inference (1-NFE):** With x-prediction, 1-step sampling simplifies to:

$$z_0 = f_\theta(\varepsilon,\; r{=}0,\; t{=}1)$$

No velocity-to-data conversion is needed — the model directly outputs the synthetic latent.

### 2.6 Latent Space Formulation

All of the above operates identically in latent space. The computational advantages are:

1. **JVP cost reduction:** Latent dimension $d = 4 \times 48^3 = 442{,}368$ vs pixel space $d = 192^3 = 7{,}077{,}888$ — a **16× reduction** in JVP compute per iteration.
2. **Memory reduction:** The UNet operates on $48^3$ feature maps vs $192^3$ — approximately **64× reduction** in activation memory per JVP pass.
3. **Training efficiency:** The latent space is pre-computed (Phase 1), so VAE encode/decode cost is amortised across all training epochs.

---

## 3. Architecture Design

### 3.1 Frozen MAISI VAE (Encoder-Decoder)

The MAISI VAE (Guo et al., 2024) is a 3D variational autoencoder with adversarial training, pre-trained on ~55K CT+MRI volumes. We use it as a **frozen foundation model** — all parameters are locked, and it is never fine-tuned.

| Property | Value |
|----------|-------|
| Parameters | 20,944,897 (~21M) |
| Encoder stages | 3 levels of 2× strided 3D convolution |
| Latent channels | 4 (with KL regularisation) |
| Spatial compression | 4× per axis: $(1, 192^3) \to (4, 48^3)$ |
| Attention | None (all `attention_levels=false`) |
| Training losses | $L_1$ + LPIPS perceptual + PatchGAN adversarial + KL |
| Checkpoint format | Wrapped in `"unet_state_dict"` key |
| scale_factor | 0.96240234375 (extracted from diffusion checkpoint) |
| Memory optimisation | `num_splits` parameter for chunk-based processing |

**Reconstruction quality (Phase 0, 20 IXI volumes):** Mean SSIM = 0.9213, Mean PSNR = 30.86 dB. This establishes the VAE as a faithful encoder-decoder for brain MRI, with acceptable smoothing in cortical boundary regions.

**Scale factor:** The decode operation divides by scale_factor before passing to the decoder: $\hat{x} = \mathrm{decoder}(z / 0.9624)$. This calibration ensures the latent distribution matches the prior used during VAE training. The scale_factor was extracted from the MAISI diffusion checkpoint (`diff_unet_3d_rflow-mr.pt["scale_factor"]`), not the VAE checkpoint.

**Latent distribution quality:** Encoding all 6,471 volumes (train + val + test) confirms the latent space is well-regularised:

| Channel | Mean | Std | Skewness | Kurtosis |
|---------|------|-----|----------|----------|
| 0 | $-0.053$ | $0.970$ | $+0.105$ | $-0.123$ |
| 1 | $-0.185$ | $1.019$ | $+0.002$ | $-0.019$ |
| 2 | $-0.051$ | $0.970$ | $+0.045$ | $+0.099$ |
| 3 | $+0.001$ | $1.011$ | $+0.069$ | $+0.144$ |

The distributions are near-Gaussian: skewness $|\gamma| < 0.11$, excess kurtosis $|\kappa| < 0.15$, and standard deviations cluster around $0.97$–$1.02$. Cross-channel correlations are negligible (max off-diagonal $|r| = 0.046$), confirming the KL regularisation produces approximately independent, unit-variance channels.

### 3.2 MeanFlow UNet (Generative Model)

The MeanFlow UNet (`MAISIUNetWrapper`) uses the **same architecture** as the MAISI diffusion UNet but with random initialisation and custom dual-time conditioning.

| Property | Value |
|----------|-------|
| Backbone | MONAI `DiffusionModelUNet` (3D) |
| Total parameters | ~178M |
| Channels per level | [64, 128, 256, 512] |
| Attention levels | [false, false, true, true] — only at $12^3$ and $6^3$ resolution |
| Attention heads | 32 channels per head at attention levels |
| ResBlocks per level | 2 |
| GroupNorm groups | 32 |
| Transformer layers | 1 per attention level |
| Flash attention | **Disabled** (required for `torch.func.jvp` compatibility) |
| Gradient checkpointing | **Disabled** (required for exact JVP forward-mode AD) |
| ResBlock downsampling | Strided convolution (not pooling) |
| Prediction type | x-prediction (network outputs denoised data $\hat{x}$) |

**Flash attention incompatibility:** `torch.func.jvp` uses forward-mode automatic differentiation, which requires the computation graph to be fully traceable. Flash attention's fused CUDA kernels are opaque to PyTorch's AD system. Disabling flash attention adds ~10% latency per forward pass but enables exact JVP computation.

**Gradient checkpointing incompatibility:** Gradient checkpointing re-executes forward passes during backward, which conflicts with the forward-mode AD tape maintained by `torch.func.jvp`. Both cannot be active simultaneously.

### 3.3 Dual-Time Conditioning

MeanFlow requires conditioning on two time variables: $t$ (current time) and $r$ (interval lower bound). We implement three conditioning modes and select `t_h` based on ablation:

| Mode | Inputs | Embedding | Source |
|------|--------|-----------|--------|
| `dual` | $(r, t)$ | $\mathrm{sin}(r) + \mathrm{sin}(t)$ through separate MLPs | pMF convention |
| `h` | $h = t - r$ | $\mathrm{sin}(h)$ through UNet's built-in MLP | iMF convention |
| **`t_h`** | **$(t,\; h = t - r)$** | **$\mathrm{sin}(t)$ through UNet MLP + $\mathrm{sin}(h)$ through new $h$-embed MLP** | **MF Table 1c optimal** |

The `t_h` mode conditions on both the absolute time $t$ and the interval width $h = t - r$. This is strictly more informative than $h$-only conditioning (the model can distinguish between "near data at $t = 0.1$" and "near noise at $t = 0.9$" even when $h$ is the same). The original MeanFlow paper (Table 1c) reports FID 61.06 for $(t, h)$ vs 63.13 for $h$-only on ImageNet.

**Implementation:** Continuous time values in $[0, 1]$ are scaled by 1000 before computing sinusoidal embeddings. This prevents degenerate embeddings — with the standard $\mathrm{max\_period} = 10000$, times in $[0, 1]$ would produce nearly constant embeddings. After sinusoidal encoding, each embedding passes through a 2-layer MLP ($\mathrm{Linear}(64, 256) \to \mathrm{SiLU} \to \mathrm{Linear}(256, 256)$), and the two embeddings are summed:

$$\mathrm{emb} = \mathrm{MLP}_t\!\left(\mathrm{sin}(t \cdot 1000)\right) + \mathrm{MLP}_h\!\left(\mathrm{sin}(h \cdot 1000)\right)$$

### 3.4 v-Head (Auxiliary Tangent Predictor)

The v-head is a lightweight auxiliary output path branching from the UNet's final feature map (before the main output convolution):

```
Shared Backbone Features (B, 64, 48, 48, 48)
    │
    ├── u-head: UNet output conv  ──►  (B, 4, 48, 48, 48)  [main output]
    │
    └── v-head: ResBlock(64) → GN(32,64) → SiLU → Conv3d(64→4)  ──►  (B, 4, 48, 48, 48)
```

| Property | Value |
|----------|-------|
| ResBlocks | 1 (configurable via `v_head_num_res_blocks`) |
| Parameters | ~228K (<0.13% of total 178M) |
| Initialisation | Final Conv3d is zero-initialised ($v$-head starts at zero) |
| Inference cost | Zero ($v$-head output is discarded at sampling time) |

**Zero initialisation rationale:** By initialising the v-head's final convolution to zero, the initial v-head output is identically zero. This means the u-head's training is completely unaffected by the v-head at initialisation. As training progresses, the v-head learns to predict the instantaneous velocity, providing an increasingly accurate tangent for the JVP computation. Our training data confirms this: $\cos(\hat{v}, v_c)$ rises from 0.17 (epoch 0) to 0.39 (epoch 50) to 0.44 (epoch 689).

---

## 4. Loss Function and Training Objective

### 4.1 Combined iMF Dual-Head Loss

The training loss has two independently-weighted components:

$$\mathcal{L} = \mathcal{L}_u^{\mathrm{weighted}} + \mathcal{L}_v^{\mathrm{weighted}}$$

where:

- $\mathcal{L}_u$ (**compound velocity loss**): $\|V_\theta - v_c\|_p^p$ — enforces MeanFlow self-consistency
- $\mathcal{L}_v$ (**tangent supervision loss**): $\|\hat{v} - v_c\|_p^p$ — directly supervises the v-head

Both losses target the same conditional velocity $v_c = \varepsilon - z_0$. The compound velocity $V$ incorporates the MeanFlow identity correction, while $\hat{v}$ is a direct prediction.

### 4.2 Adaptive Weighting

Each loss component is independently normalised by its own magnitude:

$$w_u = \left(\mathcal{L}_u^{\mathrm{raw}}\big|_{\mathrm{detach}} + \varepsilon_{\mathrm{norm}}\right)^{p_{\mathrm{norm}}}, \qquad \mathcal{L}_u^{\mathrm{weighted}} = \frac{\mathcal{L}_u^{\mathrm{raw}}}{w_u}$$

$$w_v = \left(\mathcal{L}_v^{\mathrm{raw}}\big|_{\mathrm{detach}} + \varepsilon_{\mathrm{norm}}\right)^{p_{\mathrm{norm}}}, \qquad \mathcal{L}_v^{\mathrm{weighted}} = \frac{\mathcal{L}_v^{\mathrm{raw}}}{w_v}$$

With $\varepsilon_{\mathrm{norm}} = 1.0$ and $p_{\mathrm{norm}} = 1.0$:

$$w = \mathcal{L}^{\mathrm{raw}} + 1.0, \qquad \mathcal{L}^{\mathrm{weighted}} = \frac{\mathcal{L}^{\mathrm{raw}}}{\mathcal{L}^{\mathrm{raw}} + 1.0}$$

This is a form of **logarithmic loss normalisation**: the effective loss is bounded in $[0, 1)$ and varies slowly with raw loss magnitude. It prevents large loss spikes from destabilising training and equalises the gradient contribution across timesteps (where raw loss naturally varies by orders of magnitude due to the signal-to-noise ratio at different $t$ values).

At convergence, the loss decomposition (epoch 388, best FID) shows this balancing in action:

| Component | Raw Loss | Share |
|-----------|----------|-------|
| FM loss ($r = t$, flow matching) | 715,432 | 10.3% |
| MF loss ($r < t$, self-consistency) | 6,254,997 | 89.7% |
| v-head loss | 714,646 | — |
| u-head loss (compound $V$) | 3,485,215 | — |

The MF loss is ~9× larger than the FM loss, which is expected: the compound velocity $V$ must also capture the JVP correction term, a harder target than the direct velocity at $r = t$. The adaptive weighting equalises their gradient contributions to ~1.0 each.

**Why $\varepsilon_{\mathrm{norm}} = 1.0$ and not smaller?** Through Phase 4c debugging, we discovered that $\varepsilon_{\mathrm{norm}} = 0.01$ caused catastrophic gradient amplification for samples with small raw loss. The weight $1 / (\mathcal{L} + 0.01)$ can reach 100×, creating a positive feedback loop. Setting $\varepsilon_{\mathrm{norm}} = 1.0$ caps the maximum amplification at a benign level.

**Why $p_{\mathrm{norm}} = 1.0$ and not 0.5?** With $p_{\mathrm{norm}} = 0.5$, the weight becomes $w = \sqrt{\mathcal{L} + 1.0}$, which under-normalises large losses, leading to a 1000× gradient explosion observed in Phase 4e testing.

### 4.3 Per-Channel $L_p$ Loss

The base loss function computes a per-channel, spatially-summed $L_p$ norm:

$$\ell_p(\hat{y}, y) = \sum_{c=1}^{C} \sum_{\mathbf{x}} \left|\hat{y}_{c,\mathbf{x}} - y_{c,\mathbf{x}}\right|^p$$

For the best model: $p = 2.0$ (standard $L_2$ loss). The per-channel $L_p$ framework supports ablation over $p \in \{1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0\}$ and per-channel weight vectors, extending the SLIM-Diff per-channel loss (Pascual-Gonzalez et al., 2026) from pixel-space DDPM to latent-space MeanFlow.

**Key scientific question:** Does the optimal $L_p$ exponent from pixel space ($p = 1.5$ for images, $p = 2.0$ for masks in SLIM-Diff) transfer through the VAE nonlinearity to latent space? The VAE's encoder Jacobian mixes spatial and channel information, so the error distribution in latent space may have different statistical properties.

### 4.4 JVP Strategies

Two strategies are implemented:

**Exact JVP** (`torch.func.jvp`):

$$u,\; \frac{du}{dt} = \texttt{jvp}\!\left(u_\theta,\; (z_t, t, r),\; (\tilde{v},\, 1,\, 0)\right), \qquad V = u + (t - r) \cdot \mathrm{sg}\!\left[\frac{du}{dt}\right]$$

- Most accurate, $O(d)$ cost
- Requires: no in-place ops, no flash attention, no gradient checkpointing
- Memory: ~20 GB activation per sample at batch=2 on A100-40GB

**Finite Difference JVP** (FD-JVP):

$$u_h = u_\theta(z_t + h\tilde{v},\; t + h,\; r), \qquad \frac{du}{dt} \approx \frac{u_h^{\mathrm{(fp32)}} - u^{\mathrm{(fp32)}}}{h}, \qquad V = u + (t - r) \cdot \mathrm{sg}\!\left[\frac{du}{dt}\right]$$

- Step size $h = 10^{-3}$
- FP32 subtraction to avoid bf16 catastrophic cancellation
- Lower memory (perturbed pass is no\_grad)

**Critical incompatibility discovered (Phase 4f):** x-prediction + FD-JVP is **numerically unstable**. The x-to-u conversion $u = (z_t - \hat{x}) / t$ has a $1/t$ singularity. When FD-JVP computes $(u(t{+}h) - u(t)) / h$, the result involves $O(1/t)$ terms divided by $h = 0.001$, yielding:

$$\frac{du}{dt} \sim O\!\left(\frac{1}{t^2 \cdot h}\right) \to \infty \quad \text{as } t \to t_{\min}$$

With exact JVP, the $1/t$ factor is analytically differentiated, yielding stable gradients.

**Rule:** $\textbf{x-prediction + exact JVP = stable}$. $\textbf{u-prediction + FD-JVP = stable}$. $\textbf{x-prediction + FD-JVP = explosion}$.

For dual-head models, exact JVP uses `has_aux=True` to capture the v-head output alongside the JVP computation, avoiding redundant forward passes:

$$(u,\; du/dt,\; \hat{v}) = \texttt{jvp}\!\left(u_{\mathrm{with\_v\_aux}},\; (z_t, t, r),\; (\tilde{v}, 1, 0),\; \texttt{has\_aux=True}\right)$$

---

## 5. Data Pipeline

### 5.1 Dataset

We use 8 datasets from FOMO-60K, a large-scale preprocessed brain MRI collection. The actual training corpus is substantially larger than originally planned:

| Dataset | Train Subjects | Train Scans | Val Scans | Test Scans |
|---------|---------------|-------------|-----------|------------|
| PT001\_OASIS1 | 352 | 1,414 | 170 | 96 |
| PT002\_OASIS2 | 71 | 682 | 82 | 37 |
| PT005\_IXI | 494 | 494 | 58 | 29 |
| PT007\_NIMH | 212 | 428 | 48 | 23 |
| PT008\_DLBS | 394 | 807 | 109 | 51 |
| PT011\_MBSR | 125 | 295 | 36 | 16 |
| PT012\_UCLA | 106 | 106 | 13 | 6 |
| PT015\_NKI | 722 | 1,245 | 158 | 68 |
| **Total** | **2,476** | **5,471** | **674** | **326** |

Many subjects have multiple sessions (longitudinal scans), yielding 5,471 training scans from 2,476 subjects — approximately 4× more data than the initially planned ~1,172 scans from 3 datasets. All volumes are T1-weighted, skull-stripped, RAS-oriented, and co-registered. The split is stratified by dataset with a fixed seed of 42 for reproducibility.

### 5.2 Preprocessing Pipeline

```
NIfTI (.nii.gz)
  → LoadImaged (MONAI)
  → EnsureChannelFirstd
  → Spacingd(pixdim=1.0mm isotropic, bilinear)
  → ScaleIntensityRangePercentilesd(lower=0%, upper=99.5%, b_min=0, b_max=1, clip=False)
  → CropForegroundd(source_key="image", margin=4)     [brain-centred crop]
  → ResizeWithPadOrCropd(spatial_size=(192, 192, 192)) [pad to target]
  → EnsureTyped(dtype=float32)
```

**Resolution choice ($192^3$ at 1.0 mm isotropic):** This was selected after a quantitative analysis of brain extent across 30 FOMO-60K subjects. Key findings:

- Brain AP extent reaches 193 mm for the largest subjects
- The brain is systematically ~13 mm anterior to the volume centre
- $128^3$ and $160^3$ both clip frontal/occipital cortex
- `CropForegroundd` centres the crop on brain tissue (not volume centre), preventing systematic tissue loss
- $192^3$ achieves 100% brain coverage for all subjects

**Latent shape:** $192 / 4 = 48$ per spatial axis, giving latents of shape $(4, 48, 48, 48)$.

### 5.3 Latent Pre-Computation (Phase 1)

All training volumes are pre-encoded through the frozen MAISI VAE and stored in HDF5 shards:

| Property | Value |
|----------|-------|
| Storage format | Per-dataset HDF5 shards (e.g. `PT005_IXI.h5`) |
| Layout per shard | `/latents` $(N, 4, 48, 48, 48)$, `/written`, `/subject_id`, `/session_id` |
| Per-sample size | $4 \times 48^3 \times 4$ bytes = 1.77 MB (float32) |
| Total encoded | 6,471 volumes (train + val + test) |
| Total storage | ~11.5 GB |

**Latent statistics** are computed online during encoding via `LatentStatsAccumulator` (sum-of-powers method in float64):
- Per-channel mean, std, skewness, kurtosis, min, max
- $4 \times 4$ cross-channel Pearson correlation matrix

These statistics serve dual purposes: (i) per-channel normalisation for training, and (ii) quality assessment of the latent distribution (see Section 3.1).

### 5.4 Latent Augmentation

Three augmentation transforms operate directly on latents during training:

| Transform | Probability | Parameters | Rationale |
|-----------|-------------|------------|-----------|
| Flip depth axis | 0.5 | Along dim=2 (L–R in RAS) | Brain is approximately bilaterally symmetric |
| Gaussian noise | 0.2 | std = 5% of per-channel std | Regularisation; simulates encoding noise |
| Intensity scale | 0.2 | ±5% scaling | Simulates scanner-dependent intensity variation |

Approximately 68% of training samples are augmented per epoch (~3,720 augmented out of 5,471).

---

## 6. Training Protocol

### 6.1 Algorithm (One Training Step)

Given a batch of pre-computed latents $\{z_0\}$ and the model with u-head and v-head:

1. Sample $\varepsilon \sim \mathcal{N}(0, I)$, same shape as $z_0$
2. Sample $t \sim \mathrm{LogitNormal}(\mu{=}{-0.4},\, \sigma{=}1.0)$, clamp to $[0.001, 1.0]$
3. With probability 0.5: set $r = t$ (FM sample); otherwise: sample $r < t$ (MF sample)
4. Interpolate: $z_t = (1 - t)\,z_0 + t\,\varepsilon$
5. Target: $v_c = \varepsilon - z_0$
6. Tangent: $\tilde{v} = v\text{-head}(z_t, t, t)$ [no\_grad]
7. JVP: $(u, \; du/dt, \; \hat{v}) = \texttt{jvp}(\text{dual\_fn}, (z_t, t, r), (\tilde{v}, 1, 0))$ [exact, with aux]
8. Compound velocity: $V = u + (t - r)\cdot\mathrm{sg}[du/dt]$
9. Losses: $\mathcal{L}_u^{\mathrm{raw}} = \|V - v_c\|_2^2$, $\;\mathcal{L}_v^{\mathrm{raw}} = \|\hat{v} - v_c\|_2^2$
10. Adaptive weighting: $\mathcal{L} = \mathcal{L}_u^{\mathrm{raw}} / (|\mathcal{L}_u^{\mathrm{raw}}| + 1) + \mathcal{L}_v^{\mathrm{raw}} / (|\mathcal{L}_v^{\mathrm{raw}}| + 1)$
11. Backward, clip gradients ($\|\nabla\|_{\max} = 1.0$), optimiser step
12. EMA update ($\beta = 0.9999$)

### 6.2 Time Sampling

Times are drawn from a logit-normal distribution:

$$t = \sigma\!\left(\mathcal{N}(\mu{=}{-0.4},\, \sigma{=}1.0)\right), \qquad t \in [0.001, 1.0]$$

where $\sigma(\cdot)$ is the sigmoid function. This distribution is heavy near $t = 0$ (near data), which focuses training on the denoising regime. The negative mean ($\mu = -0.4$) shifts the mass toward smaller $t$ values compared to uniform sampling.

**$\mathrm{data\_proportion} = 0.5$:** Half the batch has $r = t$ (FM samples where the JVP term vanishes, reducing to standard flow matching loss), and half has $r < t$ (MF samples where the self-consistency constraint is enforced). This 50/50 split was found optimal empirically — pure MF samples ($\mathrm{data\_proportion} = 0$) diverge, while too many FM samples ($\mathrm{data\_proportion} = 0.75$) slow convergence of the 1-NFE capability.

### 6.3 Optimiser and Schedule

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Optimiser | AdamW | Standard for diffusion/flow models |
| Learning rate | $10^{-4}$ | MeanFlow reference default |
| Weight decay | 0.0 | MeanFlow reference (no regularisation needed with adaptive loss) |
| Betas | $(0.9, 0.95)$ | $\beta_2 = 0.95$ for faster momentum adaptation (default 0.999 caused stale gradients) |
| LR schedule | Cosine decay | From $10^{-4}$ to $5.6 \times 10^{-5}$ at early stop |
| Warmup | 0 steps | Warmup destabilised MF loss (Phase 4c finding) |
| Gradient clip | $\|\nabla\|_{\max} = 1.0$ | Prevents spike propagation |
| Mixed precision | bf16 | Standard for A100 training |

### 6.4 Compute Budget

| Property | Value |
|----------|-------|
| Hardware | 6× A100-SXM4-40GB (within one DGX node) |
| Strategy | DDP (DistributedDataParallel) |
| Per-GPU batch | 2 (limited by exact JVP activation memory ~20 GB) |
| Gradient accumulation | 11 steps |
| Effective batch | $2 \times 6 \times 11 = 132$ |
| Optimiser steps/epoch | 41 |
| Planned optimiser steps | 61,500 (1,500 epochs × 41) |
| Actual optimiser steps | ~28,980 (early-stopped at epoch 690) |
| Mean epoch time | 268.1 s (~4.5 min) |
| **Total wall time** | **~51.4 hours (~2.1 days)** |

### 6.5 EMA (Exponential Moving Average)

An EMA model with decay $\beta = 0.9999$ is maintained throughout training. After each optimiser step, shadow parameters are updated:

$$\theta_{\mathrm{EMA}} \leftarrow \beta \cdot \theta_{\mathrm{EMA}} + (1 - \beta) \cdot \theta$$

At inference time, the EMA weights replace the model parameters. All FID evaluations during training use EMA weights. The best model checkpoint is selected by FID computed from EMA-weight samples.

### 6.6 Divergence Monitoring

An EMA-smoothed raw loss monitor tracks training stability:
- EMA half-life: ~69 steps (decay 0.99)
- Warnings at 3× and 5× the minimum EMA loss
- Grace period of 1000 steps before monitoring begins
- Warning-only (no automatic stopping) — early stopping is FID-based via EvaluationCallback with patience=10

---

## 7. Sampling and Generation

### 7.1 One-Step Sampling (1-NFE)

With x-prediction, 1-NFE sampling is a single forward pass:

$$\varepsilon \sim \mathcal{N}(0, I), \qquad z_0 = f_\theta(\varepsilon,\; r{=}0,\; t{=}1)$$

The v-head output is discarded at inference — only the u-head (main output) is used.

### 7.2 Multi-Step Euler Sampling

For comparison, multi-step Euler sampling divides $[1, 0]$ into $K$ uniform steps:

$$\text{For } i = 0, \ldots, K{-}1: \qquad t_i = 1 - \frac{i}{K}, \quad t_{i+1} = 1 - \frac{i+1}{K}, \quad \Delta t = t_i - t_{i+1}$$

$$\hat{x} = f_\theta(z,\; r{=}t_{i+1},\; t{=}t_i), \qquad u = \frac{z - \hat{x}}{\max(t_i,\, 0.05)}, \qquad z \leftarrow z - \Delta t \cdot u$$

NFE levels tested: $\{1, 2, 5, 10\}$. MeanFlow models are trained for 1-step generation but can also be used multi-step by treating them as standard velocity fields.

### 7.3 Full Generation Pipeline (Phase 5)

The complete generation pipeline from noise to decoded MRI volume:

1. Pre-generate shared noise (for NFE-consistency analysis)
2. For each NFE level:
   - Generate latents via `sample_one_step` or `sample_euler`
   - Store in HDF5 archive (latents, seeds, per-sample timing)
3. Denormalise: $z_0 = \hat{z} \cdot \sigma_{\mathrm{latent}} + \mu_{\mathrm{latent}}$
4. Decode: $\hat{x} = \mathrm{VAE.decode}(z_0)$ [internally divides by scale\_factor]
5. Clamp to $[0, 1]$
6. Store decoded volumes in HDF5

**Shared noise protocol:** The same noise tensor $\varepsilon$ is used across all NFE levels for a given sample index, enabling direct visual and quantitative comparison of how additional sampling steps refine the output.

---

## 8. Experimental Results

### 8.1 Toy Validation (Phase 2): MeanFlow on Flat Torus

Before scaling to brain MRI, MeanFlow was validated on a tractable manifold — a flat torus in $\mathbb{R}^4$:

| Metric | Result | Target |
|--------|--------|--------|
| 1-NFE torus distance | 0.0892 | < 0.1 |
| 5-step torus distance | 0.0271 | < 0.05 |
| Angular KS $p$-value ($\theta_1$) | 0.374 | > 0.01 |
| Angular KS $p$-value ($\theta_2$) | 0.575 | > 0.01 |
| x-pred vs u-pred loss ratio | < 1.2 | < 1.5 |

This confirmed: (i) the MeanFlow implementation correctly learns 1-step generation, (ii) the angular distribution of generated samples is statistically indistinguishable from the true distribution (KS test), and (iii) both x-prediction and u-prediction converge.

### 8.2 Main Training (Phase 4): Best Model

The best model uses x-prediction + exact JVP + $(t, h)$ conditioning + v-head (1 ResBlock) + $L_2$ loss.

**Configuration summary:**

| Property | Value |
|----------|-------|
| Config file | `experiments/ablations/xpred_vs_upred/configs/xpred_exact_jvp.yaml` |
| Prediction type | x-prediction |
| JVP strategy | Exact (`torch.func.jvp`) |
| Conditioning | $t\_h$ (condition on $t$ and $h = t - r$) |
| v-head | 1 ResBlock (~228K params, <0.13% of total) |
| Loss norm | $L_2$ ($p = 2$) with independent adaptive weighting |
| Best FID epoch | **388** (step 16,338) |
| Early-stopped at | Epoch 690 (patience=10, 46% of planned 1,500 epochs) |

### 8.3 FID Progression

FID is computed as a 2.5D metric using RadImageNet ResNet-50 features on central axial, coronal, and sagittal slices:

| Epoch | Step | $\mathrm{FID}_{\mathrm{avg}}$ | $\mathrm{FID}_{xy}$ | $\mathrm{FID}_{yz}$ | $\mathrm{FID}_{zx}$ | Notes |
|-------|------|------|------|------|------|-------|
| 9 | 420 | 46.76 | 45.12 | 46.68 | 48.47 | First evaluation |
| 28 | 1,218 | 28.40 | 17.25 | 43.63 | 24.33 | Rapid initial drop |
| 88 | 3,738 | 14.61 | 14.08 | 16.25 | 13.49 | Approaching plateau |
| 178 | 7,518 | 14.06 | 13.79 | 15.46 | 12.92 | |
| 268 | 11,298 | 12.94 | 12.77 | 14.06 | 11.98 | |
| **388** | **16,338** | **11.67** | **11.85** | **12.09** | **11.07** | **Best FID** |
| 628 | 26,418 | 11.92 | 12.03 | 12.33 | 11.40 | |
| 688 | 28,938 | 11.88 | 12.14 | 12.08 | 11.42 | Final evaluation |

The FID drops rapidly from 46.76 to ~14 in the first 88 epochs, then gradually improves to **11.67** at epoch 388, after which it plateaus. Training continued for 302 more epochs without improvement before early stopping triggered.

### 8.4 Training Dynamics

| Epoch | Raw Loss | $\cos(V, v_c)$ | $\cos(\hat{v}, v_c)$ | Rel. Error | $\|\nabla\|$ | LR |
|-------|----------|------|------|------|------|------|
| 0 | 2,788,299 | 0.083 | 0.170 | 1.353 | 1.384 | $1.0 \times 10^{-4}$ |
| 50 | 5,316,270 | 0.240 | 0.388 | 1.759 | 0.953 | ${\sim}9.9 \times 10^{-5}$ |
| 100 | 6,275,983 | 0.274 | 0.412 | 1.778 | 0.910 | ${\sim}9.7 \times 10^{-5}$ |
| 388 | 4,724,751 | 0.295 | 0.429 | 1.645 | 1.180 | $8.5 \times 10^{-5}$ |
| 689 | 4,327,238 | 0.307 | 0.436 | 1.516 | 0.610 | $5.6 \times 10^{-5}$ |

**Key observations:**

1. **Raw loss is not a good proxy for generation quality.** The raw loss is dominated by the MF term ($\mathcal{L}_{\mathrm{MF}} \approx 6.5 \times 10^6$ vs $\mathcal{L}_{\mathrm{FM}} \approx 7 \times 10^5$), and does not monotonically decrease. FID (computed from EMA-weight samples) is the correct early-stopping metric.

2. **Cosine alignment improves steadily.** The compound velocity alignment $\cos(V, v_c)$ increases from 0.083 to 0.307, indicating the model increasingly satisfies the MeanFlow identity. The v-head alignment converges faster (0.170 → 0.436), confirming the v-head provides good tangents from early training.

3. **Gradient norm decreases dramatically.** The gradient norm drops from 1.384 to 0.610, and the gradient clipping fraction drops from ~90% to ~5%, indicating stable training dynamics by epoch ~300.

### 8.5 Generated Sample Statistics

The model's output $\hat{x}$ statistics evolve during training:

| Epoch | $\mathbb{E}[\hat{x}]$ | $\mathrm{std}[\hat{x}]$ | $\min[\hat{x}]$ | $\max[\hat{x}]$ |
|-------|------|------|------|------|
| 0 | $+0.004$ | $0.243$ | $-2.06$ | $3.41$ |
| 100 | $-0.001$ | $0.737$ | $-5.88$ | $8.56$ |
| 388 | $-0.001$ | $0.753$ | $-5.84$ | $8.69$ |
| 689 | $-0.001$ | $0.781$ | $-6.22$ | $9.38$ |

The standard deviation stabilises around 0.75–0.78, approximately 75% of the true latent std (~0.97–1.02). This **variance under-estimation** is the primary quality bottleneck: generated samples have slightly less contrast than real ones. The means and ranges are consistent with the true latent distribution.

### 8.6 NFE Quality Analysis

Visual inspection of decoded samples at different NFE levels reveals a clear quality hierarchy:

| NFE | Visual Quality | Key Features |
|-----|----------------|--------------|
| 1 | Recognisable brain anatomy, blurry | Global structure correct, low contrast, sulci indistinct |
| 2 | Substantially sharper | Cortical sulci visible, improved grey-white contrast |
| 5 | Sharp, anatomically plausible | Clear grey-white matter contrast, realistic cortical folding |
| 10 | Marginally crisper than NFE=5 | Diminishing returns beyond 5 steps |

**NFE-consistency metrics** (shared noise, converged model):
- Cosine similarity: NFE=1 vs NFE=2: ~0.96; NFE=1 vs NFE=5: ~0.91; NFE=1 vs NFE=10: ~0.90
- MSE between 1-step and multi-step decreases over training for all NFE levels

The 1-NFE output captures global anatomy correctly but lacks high-frequency detail — a known limitation of MeanFlow. NFE=2 provides a substantial quality improvement at only 2× cost, suggesting a practical 2-step generation mode for quality-sensitive applications.

### 8.7 Spectral Analysis

The radially-averaged power spectrum of generated latents evolves from flat (noise-like at epoch ~100) to the characteristic $1/f$ brain spectrum by epoch 600. All 4 channels show proper low-frequency dominance with high-frequency rolloff, confirming the model learns the correct spectral structure of brain MRI.

### 8.8 SWD (Sliced Wasserstein Distance)

| Metric | Epoch 9 | Epoch 689 |
|--------|---------|-----------|
| SWD | 1.779 (best) | 2.555 |

Notably, SWD improves rapidly in early training but then **diverges from FID** — SWD increases monotonically after epoch ~50 while FID continues to decrease. This indicates SWD is **not a reliable proxy** for perceptual quality in this setting, likely because it measures distributional distances in latent space that do not correlate with perceptual quality in pixel space.

### 8.9 Key Ablation Insights

**x-pred vs u-pred:** The x-prediction configuration with exact JVP trained stably through 690 epochs (best FID at 388). This supports the pMF manifold hypothesis extending to latent space with 3D UNets: $d_{\mathrm{input}} / d_{\mathrm{bottleneck}} \approx 4$, firmly in the x-prediction regime.

**x-pred + FD-JVP instability:** This combination produces exponential growth of JVP norms (10,000× within 50 epochs). Root cause:

$$u = \frac{z_t - \hat{x}}{t} \implies \frac{du}{dt}\bigg|_{\mathrm{FD}} \approx \frac{u(t{+}h) - u(t)}{h} \sim \frac{O(1/t)}{h} = O\!\left(\frac{1}{t^2 h}\right)$$

**Conditioning mode:** $(t, h)$ conditioning was selected based on the MeanFlow paper Table 1c: FID 61.06 for $(t, h)$ vs 63.13 for $h$-only on ImageNet.

**Training stability fixes (Phase 4c–4e):**
- $\beta_2$: 0.999 → 0.95 (faster momentum adaptation prevents stale gradient accumulation)
- Warmup: 5000 → 0 (linear warmup destabilises MF loss by producing inconsistent gradient magnitudes)
- $\varepsilon_{\mathrm{norm}}$: 0.01 → 1.0 (prevents runaway gradient amplification in adaptive weighting)
- $p_{\mathrm{norm}}$: 0.5 → 1.0 (sub-linear normalisation causes gradient explosion)
- $\mathrm{data\_proportion}$: stabilised at 0.5 (50/50 FM/MF split)

---

## 9. Evaluation Protocol

### 9.1 Image Quality Metrics

| Metric | Description | Implementation |
|--------|-------------|----------------|
| 2.5D FID | FID on central axial/coronal/sagittal slices using RadImageNet ResNet-50 | Custom; follows MOTFM protocol |
| 3D FID | FID on volumetric features from Med3D ResNet-50 | Custom; `src/neuromf/metrics/fid_3d.py` |
| MMD | Maximum Mean Discrepancy in Med3D feature space | Gaussian kernel |
| Coverage | Fraction of real samples with a close synthetic neighbour | Feature-space NN |
| Density | Fraction of synthetic samples near real manifold | Complementary to coverage |
| MS-SSIM | 3-level Multi-Scale SSIM (3D) | MONAI `SSIMMetric` |
| PSNR | Peak Signal-to-Noise Ratio | Standard formula with actual data range |

### 9.2 Spectral Analysis

High-frequency energy ratio via 3D FFT:

$$\rho = \frac{\sum_{|\mathbf{k}| > k_0} |\mathcal{F}(x)(\mathbf{k})|^2}{\sum_{\mathbf{k}} |\mathcal{F}(x)(\mathbf{k})|^2}$$

Reported for real, VAE-reconstructed, and generated volumes to disentangle VAE smoothing from generative model quality.

### 9.3 Morphological Assessment

SynthSeg (Billot et al., 2023) segmentation on both real and synthetic volumes, computing:
- Regional volume correlation (hippocampus, ventricles, cortex)
- Distribution of regional volumes (KL divergence real vs synthetic)
- SynthSeg Dice overlap (paired by nearest-neighbour in feature space)

### 9.4 NFE Consistency Analysis

Generated samples at NFE $\in \{1, 2, 5, 10\}$ using shared noise, compared via:
- Per-sample $L_2$ distance between NFE=1 and NFE=$K$ (convergence measure)
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
    callbacks/         # PL callbacks: diagnostics, evaluation, sample_collector
    data/              # Datasets: latent_dataset, latent_hdf5, mri_preprocessing, fomo60k
    errors/            # Custom exceptions
    generation/        # H5Manager, LatentGenerator, VolumeDecoder
    losses/            # lp_loss, meanflow_jvp, combined_loss
    metrics/           # fid, fid_3d, mmd, ms_ssim_3d, spectral, coverage_density
    models/            # latent_meanflow (PL module), toy_mlp, lora, rectflow_baseline
    sampling/          # one_step, multi_step (Euler)
    utils/             # ema, time_sampler, latent_stats, pretrained_loading
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

The project is implemented in 9 gated phases (0–8). Phase $N{+}1$ cannot begin until Phase $N$'s critical tests all pass.

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
| GPU | RTX 4060 8 GB | 4 nodes × 8× A100-40 GB |
| Use case | Development, testing, analysis | Training, encoding, evaluation |
| SLURM | N/A | `--constraint=dgx` |
| Conda env | `neuromf` | `neuromf` |

---

## 11. Novel Contributions

### Contribution 1: First MeanFlow for 3D Medical Image Synthesis

We bring the MeanFlow framework — previously applied only to 2D natural images — to 3D volumetric medical imaging. The model achieves 1-NFE generation of $192^3$ brain MRI volumes in the latent space of a frozen MAISI VAE. This represents a **50–1000× reduction in sampling cost** compared to existing 3D medical generative models (DDPM: 1000 steps; flow matching: 10–50 steps; rectified flow: 5–50 steps). Our best model achieves a 2.5D FID of **11.67** using EMA weights, trained in ~51 hours on 6× A100 GPUs.

### Contribution 2: Per-Channel $L_p$ Loss in Latent MeanFlow

We extend the SLIM-Diff per-channel $L_p$ loss from pixel-space DDPM to latent-space MeanFlow. This provides a principled framework for channel-specific loss geometry that will be ablated over $p \in \{1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0\}$, investigating whether the optimal exponent from pixel space transfers through the VAE nonlinearity.

### Contribution 3: x-Prediction vs u-Prediction Ablation in Latent 3D UNets

We provide the first investigation of the x-prediction vs u-prediction dichotomy (established by pMF for 2D ViTs) in latent space with 3D UNets. Our results support x-prediction superiority, consistent with the manifold hypothesis ($d_{\mathrm{input}} / d_{\mathrm{bottleneck}} \approx 4$). We also document the critical incompatibility: **x-prediction + FD-JVP is numerically unstable** due to the $1/t$ singularity in the x-to-u conversion.

### Contribution 4: iMF Dual-Head in Medical Context

We adapt the iMF dual-head architecture (shared backbone + u-head + v-head) for 3D medical image synthesis. The v-head provides directly-supervised JVP tangents, solving the MeanFlow loss divergence problem. Our implementation is the first application of this architecture outside the original iMF/pMF natural image setting. The v-head adds only ~228K parameters (<0.13% of total) and is disabled at inference.

### Contribution 5 (Planned): LoRA Fine-Tuning for Joint Image-Mask Synthesis

Phase 7 will demonstrate LoRA fine-tuning of the pre-trained MeanFlow model for joint synthesis of FLAIR MRI and FCD segmentation masks, combining 1-step generation with per-channel $L_p$ loss in a clinically relevant data-scarce setting (~50–100 FCD cases).

---

## 12. Current Status and Next Steps

### 12.1 Current State (as of 2026-02-26)

- **Phases 0–4:** Complete. Best model trained (x-pred + exact JVP, best FID 11.67 at epoch 388, early-stopped at epoch 690).
- **Phase 5:** Code complete, 11/11 local tests pass. Awaiting Picasso execution for:
  - Latent generation at NFE $\in \{1, 2, 5, 10\}$
  - VAE decoding of generated latents
  - Full quantitative evaluation (FID, MMD, MS-SSIM, spectral, morphological)
- **Phases 6–8:** Planned.

### 12.2 Key Findings So Far

1. **Early stopping was effective:** Best FID at epoch 388, used only 46% of planned 1,500 epochs. Total training: 51.4 hours.
2. **1-NFE quality gap is significant:** NFE=1 outputs are blurry; NFE $\geq$ 2 is substantially sharper. This is a known MeanFlow limitation.
3. **Variance under-estimation:** Generated latent std ~0.78 vs true ~0.97–1.02. This 20–25% gap is the primary quality bottleneck.
4. **v-head converges fast:** $\cos(\hat{v}, v_c)$ reaches 0.39 by epoch 50 (85% of final 0.44), validating the dual-head design.
5. **SWD is anti-correlated with FID** beyond early training — not a reliable proxy for perceptual quality.
6. **Gradient clipping fraction drops 90% → 5%**, confirming stable training dynamics by epoch ~300.

### 12.3 Immediate Next Steps

1. **Run Phase 5 on Picasso** — generate samples and compute full metrics
2. **Phase 6 ablations:**
   - x-pred vs u-pred (controlled: same JVP strategy, multiple seeds)
   - $L_p$ sweep: $p \in \{1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0\}$
   - NFE trade-off: quality vs steps at $\{1, 2, 5, 10, 25, 50\}$
3. **Phase 7:** LoRA fine-tuning for FCD joint synthesis
4. **Phase 8:** Publication-ready figures and tables

### 12.4 Open Questions

- Does the $L_p$ exponent effect from pixel space transfer to latent space?
- How much does the VAE smoothing artefact degrade MeanFlow-generated volumes compared to multi-step methods?
- Can LoRA fine-tuning on ~50–100 FCD cases produce anatomically plausible lesion synthesis?
- Can the variance under-estimation (~75% of true std) be mitigated by post-hoc rescaling or loss modifications?

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
