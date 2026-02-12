# Latent MeanFlow for 3D Brain MRI Synthesis: Expanded Methodology

**Working Title:** *"MeanFlow for 3D Brain MRI Synthesis: One-Step Latent Generation with Per-Channel $L_p$ Loss for Rare Pathology"*

**Author:** Mario Pascual-González
**Date:** February 2026
**Version:** 2.0 — Expanded Methodology
**Status:** Methodology Definition — Pre-Implementation

---

## Table of Contents

1. [Theoretical Foundations](#1-theoretical-foundations)
2. [Latent MeanFlow: Formal Derivation in Compressed Space](#2-latent-meanflow-formal-derivation-in-compressed-space)
3. [Per-Channel $L_p$ Loss in Latent MeanFlow: Theory and Transfer](#3-per-channel-lp-loss-in-latent-meanflow-theory-and-transfer)
4. [x-Prediction vs. u-Prediction in Latent Space: Manifold-Theoretic Analysis](#4-x-prediction-vs-u-prediction-in-latent-space-manifold-theoretic-analysis)
5. [LoRA Fine-Tuning for Domain Adaptation under Data Scarcity](#5-lora-fine-tuning-for-domain-adaptation-under-data-scarcity)
6. [MAISI VAE: Foundation Encoder–Decoder for Medical Volumes](#6-maisi-vae-foundation-encoder-decoder-for-medical-volumes)
7. [Literature Framing and Positioning](#7-literature-framing-and-positioning)
8. [Novel Contributions Statement](#8-novel-contributions-statement)
9. [Ablation Design with Statistical Rigour](#9-ablation-design-with-statistical-rigour)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Data Strategy](#11-data-strategy)
12. [Proposed Paper Structure](#12-proposed-paper-structure)
13. [References](#13-references)

---

## 1. Theoretical Foundations

### 1.1 Flow Matching Preliminaries

Consider the task of learning a transport map from a prior distribution $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ to a data distribution $p_0$ over a space $\mathcal{X} \subseteq \mathbb{R}^d$. Flow Matching (FM; Lipman et al., 2023; Liu et al., 2023; Albergo & Vanden-Eijnden, 2023) defines a time-dependent probability path $p_t$ via the conditional linear interpolation:

$$
\mathbf{z}_t = (1 - t)\,\mathbf{x} + t\,\boldsymbol{\epsilon}, \quad \mathbf{x} \sim p_0,\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad t \in [0, 1]
\tag{1}
$$

where $t = 0$ corresponds to clean data and $t = 1$ to pure noise. The conditional instantaneous velocity field is:

$$
\mathbf{v}_c(\mathbf{z}_t, t) = \boldsymbol{\epsilon} - \mathbf{x}
\tag{2}
$$

and a neural network $\mathbf{v}_\theta(\mathbf{z}_t, t)$ is trained to approximate the marginal velocity $\mathbf{v}(\mathbf{z}_t, t) = \mathbb{E}[\mathbf{v}_c \mid \mathbf{z}_t]$ via:

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}} \left[ \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|^2 \right]
\tag{3}
$$

Sampling requires solving the ODE $\frac{d\mathbf{z}_t}{dt} = \mathbf{v}_\theta(\mathbf{z}_t, t)$ from $t=1$ to $t=0$ via a numerical solver (e.g., Euler, Heun), incurring $K$ network evaluations (NFE = $K$). Rectified Flow (Liu et al., 2023) further straightens the ODE trajectories via a "reflow" procedure to enable fewer steps, and MAISI-v2 (Zhao et al., 2025) applies this framework to 3D medical volumes in latent space.

**Critical limitation for our setting:** even with straightened trajectories, Rectified Flow requires $K \geq 5$ NFE for high-quality generation, and the quality degrades significantly at $K=1$.

### 1.2 MeanFlow: From Instantaneous to Average Velocity

MeanFlow (Geng et al., 2025a) introduces a fundamentally different quantity: the **average velocity** over an interval $[r, t]$ along the flow path:

$$
\mathbf{u}(\mathbf{z}_t, r, t) \triangleq \frac{1}{t - r} \int_r^t \mathbf{v}(\mathbf{z}_s, s)\, ds, \quad 0 \leq r < t \leq 1
\tag{4}
$$

where $\mathbf{z}_s$ follows the ODE trajectory passing through $\mathbf{z}_t$ at time $t$. The key insight is that if the average velocity is perfectly learned, the entire flow path can be reconstructed from a single evaluation:

$$
\mathbf{z}_0 = \mathbf{z}_1 - 1 \cdot \mathbf{u}_\theta(\mathbf{z}_1, 0, 1)
\tag{5}
$$

enabling **1-NFE generation**. This stands in contrast to FM, where the instantaneous velocity must be integrated over $[0, 1]$.

**The MeanFlow Identity.** Differentiating both sides of Eq. (4) with respect to $t$ yields:

$$
(t - r)\,\frac{\partial \mathbf{u}}{\partial t}(\mathbf{z}_t, r, t) + \mathbf{u}(\mathbf{z}_t, r, t) = \mathbf{v}(\mathbf{z}_t, t)
\tag{6}
$$

Since $\frac{d\mathbf{z}_t}{dt} = \mathbf{v}(\mathbf{z}_t, t)$, expanding the total derivative of $\mathbf{u}$ along the flow:

$$
\frac{d}{dt}\mathbf{u}(\mathbf{z}_t, r, t) = \underbrace{\frac{\partial \mathbf{u}}{\partial \mathbf{z}_t} \cdot \mathbf{v}(\mathbf{z}_t, t)}_{\text{spatial Jacobian}} + \underbrace{\frac{\partial \mathbf{u}}{\partial t}}_{\text{temporal derivative}}
\tag{7}
$$

Substituting Eq. (7) into Eq. (6) and rearranging gives the **MeanFlow Identity**:

$$
\mathbf{v}(\mathbf{z}_t, t) = \mathbf{u}(\mathbf{z}_t, r, t) + (t - r)\left[\frac{\partial \mathbf{u}}{\partial \mathbf{z}_t} \cdot \mathbf{v}(\mathbf{z}_t, t) + \frac{\partial \mathbf{u}}{\partial t}\right]
\tag{8}
$$

This identity is **exact** and holds for the ground-truth fields. It provides a self-consistency condition that $\mathbf{u}$ must satisfy — and it is this condition that the neural network is trained to enforce.

### 1.3 MeanFlow Training Objective

The neural network $\mathbf{u}_\theta(\mathbf{z}_t, r, t)$ is trained to satisfy Eq. (8). The right-hand side of Eq. (8), when evaluated with the neural approximation $\mathbf{u}_\theta$, yields the **compound prediction**:

$$
V_\theta(\mathbf{z}_t, r, t) = \mathbf{u}_\theta(\mathbf{z}_t, r, t) + (t - r) \cdot \text{sg}\!\left[\frac{\partial \mathbf{u}_\theta}{\partial \mathbf{z}_t} \cdot \tilde{\mathbf{v}}_\theta + \frac{\partial \mathbf{u}_\theta}{\partial t}\right]
\tag{9}
$$

where $\text{sg}[\cdot]$ denotes stop-gradient (the JVP term is treated as a fixed target, not differentiated through), and $\tilde{\mathbf{v}}_\theta$ is the model's own instantaneous velocity estimate, obtained by evaluating $\mathbf{u}_\theta$ at $r = t$:

$$
\tilde{\mathbf{v}}_\theta(\mathbf{z}_t, t) = \mathbf{u}_\theta(\mathbf{z}_t, t, t)
\tag{10}
$$

The bracketed quantity in Eq. (9) is computed via a **Jacobian-Vector Product (JVP)**:

$$
\text{JVP}\!\left(\mathbf{u}_\theta,\; (\mathbf{z}_t, r, t),\; (\tilde{\mathbf{v}}_\theta, 0, 1)\right) = \frac{\partial \mathbf{u}_\theta}{\partial \mathbf{z}_t} \cdot \tilde{\mathbf{v}}_\theta + \frac{\partial \mathbf{u}_\theta}{\partial t} \cdot 1
\tag{11}
$$

This is efficiently computed in $O(d)$ time and memory via forward-mode automatic differentiation (`torch.func.jvp` in PyTorch), without constructing the full $d \times d$ Jacobian.

The MeanFlow loss is:

$$
\mathcal{L}_{\text{MF}} = \mathbb{E}_{t, r, \mathbf{x}, \boldsymbol{\epsilon}} \left[ w(t) \cdot \|V_\theta(\mathbf{z}_t, r, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|_p^p \right]
\tag{12}
$$

where $\mathbf{v}_c = \boldsymbol{\epsilon} - \mathbf{x}$ is the conditional velocity, and $w(t)$ is an adaptive weight (see §1.4).

**Time sampling.** Following Geng et al. (2025a) and Lu et al. (2026):
- $t \sim \text{LogitNormal}(\mu=0.8, \sigma=0.8)$, clipped to $[t_{\min}, 1]$ with $t_{\min} = 0.05$
- With probability $0.5$, set $r = t$ (which reduces Eq. (9) to direct velocity regression); otherwise sample $r \sim \text{Uniform}(0, t)$

The $r = t$ case is critical: Geng et al. (2025a) empirically found that 50–75% $r=t$ samples stabilise training. When $r = t$, the JVP correction vanishes, providing a low-variance gradient signal.

### 1.4 Improved MeanFlow (iMF) Objective

The iMF reformulation (Geng et al., 2025b) addresses a key limitation: the original MF objective (Eq. 12) has a training target $V_\theta$ that depends on the network itself, creating a non-standard regression problem with high variance.

iMF recasts the objective as a loss on the **instantaneous velocity** $\mathbf{v}$, re-parameterised by a network that predicts the average velocity $\mathbf{u}$:

$$
\mathcal{L}_{\text{iMF}} = \mathbb{E}_{t, r, \mathbf{x}, \boldsymbol{\epsilon}} \left[ w(t) \cdot \|\tilde{\mathbf{v}}_\theta(\mathbf{z}_t, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|_p^p \right] + \lambda_{\text{MF}} \cdot \mathcal{L}_{\text{MF}}
\tag{13}
$$

where the first term is a direct velocity regression loss (standard FM, evaluated at $r = t$), and the second is the original MeanFlow consistency loss. The balance $\lambda_{\text{MF}}$ is adaptively scheduled. This yields a more standard regression problem and improves training stability.

**Decision for our work:** We adopt the iMF-style combined loss (Eq. 13) as our primary training objective. The auxiliary FM term stabilises early training; the MF term ensures one-step generation capability.

### 1.5 Adaptive Weighting

Following pMF (Lu et al., 2026, §A.1), the adaptive weight is:

$$
w(t) = \frac{1}{\text{sg}[\|\mathbf{e}\|_p^p] + c}
\tag{14}
$$

where $\mathbf{e} = V_\theta - \mathbf{v}_c$ is the error vector, and $c > 0$ is a small constant preventing division by zero. The actual gradient update uses $\text{sg}[w(t)] \cdot \|\mathbf{e}\|_p^p$, which is equivalent to normalising the loss by its own magnitude — a form of loss landscape normalisation that stabilises training across timesteps.

### 1.6 x-Prediction Reparameterisation

The pMF paper (Lu et al., 2026) introduces a crucial reparameterisation. Instead of having the network directly output $\mathbf{u}_\theta$, define an auxiliary field:

$$
\hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t) \triangleq \mathbf{z}_t - t \cdot \mathbf{u}_\theta(\mathbf{z}_t, r, t)
\tag{15}
$$

The network is trained to output $\hat{\mathbf{x}}_\theta$ (x-prediction), and the average velocity is derived:

$$
\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{\mathbf{z}_t - \hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t)}{t}
\tag{16}
$$

**Manifold hypothesis argument (Lu et al., 2026, §4):** When $r = t$, $\hat{\mathbf{x}}_\theta(\mathbf{z}_t, t, t) = \mathbf{z}_t - t \cdot \mathbf{v}_\theta(\mathbf{z}_t, t)$, which is the standard "denoised" estimate. The target $\mathbf{x} = \mathbf{z}_0$ lies on the data manifold $\mathcal{M} \subset \mathbb{R}^d$, which has intrinsic dimensionality $d_{\text{int}} \ll d$. A network predicting a point on $\mathcal{M}$ has an easier task (lower-dimensional target) than predicting the high-dimensional velocity field $\mathbf{v}$.

**Quantitative criterion (pMF Table 2):** x-prediction dominates u-prediction when $d_{\text{patch}} / d_{\text{hidden}} > 1$, where $d_{\text{patch}}$ is the input patch dimensionality and $d_{\text{hidden}}$ is the transformer hidden dimension. This ratio governs whether the model has sufficient capacity to represent the full velocity field.

---

## 2. Latent MeanFlow: Formal Derivation in Compressed Space

### 2.1 The Latent Diffusion / Latent Flow Matching Paradigm

Let $\mathcal{E}_\phi: \mathbb{R}^{1 \times H \times W \times D} \to \mathbb{R}^{C \times H' \times W' \times D'}$ and $\mathcal{D}_\phi: \mathbb{R}^{C \times H' \times W' \times D'} \to \mathbb{R}^{1 \times H \times W \times D}$ denote the frozen encoder and decoder of a pretrained VAE (MAISI; Guo et al., 2024), where $H' = H/f$, $W' = W/f$, $D' = D/f$ for spatial downsampling factor $f$. The encoder maps MRI volumes $\mathbf{x} \in \mathcal{X}$ to latent representations $\mathbf{z}_0 = \mathcal{E}_\phi(\mathbf{x}) \in \mathcal{Z}$.

**For MAISI at $128^3$ resolution:**
- $f = 4$, $C = 4$
- $\mathbf{z}_0 \in \mathbb{R}^{4 \times 32 \times 32 \times 32}$
- Compression ratio: $128^3 / (4 \times 32^3) = 16\times$ overall

The **Latent Flow Matching** paradigm (Rombach et al., 2022; Guo et al., 2024; Zhao et al., 2025) trains the generative model entirely in $\mathcal{Z}$:

$$
\mathbf{z}_t = (1 - t)\,\mathbf{z}_0 + t\,\boldsymbol{\epsilon}, \quad \mathbf{z}_0 = \mathcal{E}_\phi(\mathbf{x}),\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\tag{17}
$$

Synthesis: $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \xrightarrow{\text{MeanFlow}} \hat{\mathbf{z}}_0 \xrightarrow{\mathcal{D}_\phi} \hat{\mathbf{x}}$.

### 2.2 MeanFlow Identity in Latent Space

The MeanFlow identity (Eq. 8) is defined over arbitrary $\mathbb{R}^d$ and makes no assumptions about the representation space. When we instantiate it in $\mathcal{Z}$, every quantity simply operates on $\mathbf{z}_t \in \mathbb{R}^{4 \times 32 \times 32 \times 32}$ rather than $\mathbf{x}_t \in \mathbb{R}^{1 \times 128 \times 128 \times 128}$:

$$
\mathbf{v}(\mathbf{z}_t, t) = \mathbf{u}(\mathbf{z}_t, r, t) + (t - r)\left[\frac{\partial \mathbf{u}}{\partial \mathbf{z}_t} \cdot \mathbf{v}(\mathbf{z}_t, t) + \frac{\partial \mathbf{u}}{\partial t}\right]
\tag{18}
$$

The mathematical structure is identical; only the dimensionality changes. However, there are important implications:

**Proposition 1 (Computational tractability).** The JVP computation in Eq. (11) has complexity $O(d \cdot C_{\text{net}})$ where $d = |\mathbf{z}_t|$ is the latent dimensionality and $C_{\text{net}}$ is the cost of one forward pass. In latent space, $d = 4 \times 32^3 = 131{,}072$, compared to $d = 128^3 = 2{,}097{,}152$ in pixel space — a **$16\times$ reduction** in JVP cost per iteration.

**Proposition 2 (Memory reduction).** A single JVP evaluation requires storing the primal computation graph (for forward-mode AD). In latent space, the UNet operates on $32^3$ feature maps (vs. $128^3$), yielding approximately $64\times$ reduction in activation memory per JVP pass — the dominant memory bottleneck.

### 2.3 Latent x-Prediction

Adapting Eq. (15)–(16) to latent space:

$$
\hat{\mathbf{z}}_{0,\theta}(\mathbf{z}_t, r, t) = \text{net}_\theta(\mathbf{z}_t, r, t)
\tag{19}
$$

$$
\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{\mathbf{z}_t - \hat{\mathbf{z}}_{0,\theta}(\mathbf{z}_t, r, t)}{t}
\tag{20}
$$

The singularity at $t = 0$ in Eq. (20) is avoided by clipping $t \geq t_{\min} = 0.05$.

**One-step sampling:**

$$
\hat{\mathbf{z}}_0 = \boldsymbol{\epsilon} - 1 \cdot \mathbf{u}_\theta(\boldsymbol{\epsilon}, 0, 1) = \hat{\mathbf{z}}_{0,\theta}(\boldsymbol{\epsilon}, 0, 1)
\tag{21}
$$

$$
\hat{\mathbf{x}} = \mathcal{D}_\phi(\hat{\mathbf{z}}_0)
\tag{22}
$$

### 2.4 VAE Nonlinearity and Its Implications

The VAE encoder $\mathcal{E}_\phi$ is a **nonlinear**, **surjective** mapping from pixel space to latent space. This nonlinearity has non-trivial consequences for MeanFlow:

**2.4.1 Distribution shape in latent space.** The MAISI VAE is trained with a KL regulariser calibrated so that the marginal latent distribution $q(\mathbf{z}) = \mathbb{E}_{\mathbf{x} \sim p_0}[q_\phi(\mathbf{z} | \mathbf{x})]$ has per-channel statistics $\mu_c \approx 0$, $\sigma_c \in [0.9, 1.1]$ (Guo et al., 2024, §3.1). However, $q(\mathbf{z})$ is not exactly Gaussian — it is a mixture of Gaussians (one per data point), which can exhibit heavy tails, multimodality, and non-trivial channel correlations. The flow-matching prior $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ may not perfectly match the marginal latent distribution.

**Implication:** Latent normalisation is important. Following standard LDM practice (Rombach et al., 2022), we compute per-channel statistics $(\mu_c, \sigma_c)$ from the training latents and normalise: $\tilde{\mathbf{z}}_0 = (\mathbf{z}_0 - \boldsymbol{\mu}) / \boldsymbol{\sigma}$. This ensures the flow endpoints are well-matched.

**2.4.2 Manifold structure.** The VAE maps the data manifold $\mathcal{M}_x \subset \mathbb{R}^{128^3}$ to a latent manifold $\mathcal{M}_z = \mathcal{E}_\phi(\mathcal{M}_x) \subset \mathbb{R}^{4 \times 32^3}$. The intrinsic dimensionality of $\mathcal{M}_z$ is at most $\dim(\mathcal{M}_x)$ (by the rank theorem for smooth maps). In practice, the VAE has already performed dimensionality reduction, so $\mathcal{M}_z$ is a lower-dimensional manifold embedded in $\mathbb{R}^{131,072}$.

**2.4.3 Implications for the x-prediction vs. u-prediction dichotomy (see §4).**

### 2.5 The Complete Latent MeanFlow Training Algorithm

**Algorithm 1: Latent MeanFlow Training Step**

```
Input: batch of pre-computed latents {z_0^(i)}, network net_θ with EMA
1. Sample ε ~ N(0, I), same shape as z_0
2. Sample t ~ LogitNormal(μ=0.8, σ=0.8), clip to [0.05, 1.0]
3. With probability 0.5: set r = t; else: r ~ Uniform(0, t)
4. Compute z_t = (1 - t) * z_0 + t * ε
5. Forward pass: x̂_θ = net_θ(z_t, r, t)           [x-prediction]
6. Compute u_θ = (z_t - x̂_θ) / t                    [average velocity]
7. Compute ṽ_θ = u_θ evaluated at r = t              [instantaneous velocity estimate]
8. Compute JVP:
   tangent = (sg[ṽ_θ], 0, 1)
   jvp_val = JVP(u_θ, (z_t, r, t), tangent)
9. Compound prediction: V_θ = u_θ + (t - r) * sg[jvp_val]
10. Target: v_c = ε - z_0
11. Loss: L = w(t) * ||V_θ - v_c||_p^p   (+ auxiliary FM loss on ṽ_θ)
12. Backward pass, optimizer step, EMA update
```

---

## 3. Per-Channel $L_p$ Loss in Latent MeanFlow: Theory and Transfer

### 3.1 The $L_p$ Loss Landscape

The $L_p$ norm for a $d$-dimensional error vector $\mathbf{e} \in \mathbb{R}^d$ is:

$$
\|\mathbf{e}\|_p^p = \sum_{i=1}^d |e_i|^p
\tag{23}
$$

The gradient of the $L_p$ loss with respect to a single component $e_i$ is:

$$
\frac{\partial}{\partial e_i} |e_i|^p = p \cdot |e_i|^{p-1} \cdot \text{sign}(e_i)
\tag{24}
$$

**Key properties by regime:**

| $p$ | Gradient at $|e_i| \ll 1$ | Gradient at $|e_i| \gg 1$ | Effect |
|---|---|---|---|
| $p = 1$ | Constant | Constant | Robust to outliers; sparse gradients |
| $p = 1.5$ | $\propto |e_i|^{0.5}$ | $\propto |e_i|^{0.5}$ | Balanced; moderate outlier suppression |
| $p = 2$ | $\propto |e_i|$ | $\propto |e_i|$ | Standard; penalises large errors quadratically |
| $p = 3$ | $\propto |e_i|^2$ | $\propto |e_i|^2$ | Aggressive penalisation of large errors |

### 3.2 SLIM-Diff Insight in Pixel Space

SLIM-Diff (Pascual-González et al., 2026) established that for joint image-mask diffusion:

1. **Image channels benefit from $p < 2$** (specifically $p = 1.5$): the sub-quadratic penalisation reduces sensitivity to high-frequency noise in the diffusion loss, improving perceptual quality. The image is a continuous-valued signal where small errors across many voxels are preferable to large localised errors.

2. **Mask channels benefit from $p = 2$**: the binary mask has a bimodal distribution ($\{0, 1\}$), and the quadratic penalisation enforces crisp boundaries by strongly penalising intermediate values (soft predictions).

Formally, the SLIM-Diff per-channel loss is:

$$
\mathcal{L}_{\text{SLIM}} = \sum_{c \in \mathcal{C}_{\text{img}}} \lambda_c \cdot \frac{1}{|\Omega|} \sum_{\mathbf{p} \in \Omega} |e_{c,\mathbf{p}}|^{p_{\text{img}}} + \sum_{c \in \mathcal{C}_{\text{mask}}} \lambda_c \cdot \frac{1}{|\Omega|} \sum_{\mathbf{p} \in \Omega} |e_{c,\mathbf{p}}|^{p_{\text{mask}}}
\tag{25}
$$

with $p_{\text{img}} = 1.5$, $p_{\text{mask}} = 2.0$, and $\lambda_c$ are per-channel balancing weights.

### 3.3 Extension to Latent Space: The Transfer Problem

**The core scientific question:** Does the $L_p$ exponent effect established in pixel space transfer to latent space?

**Formal analysis.** Let $\mathcal{E}_\phi$ denote the VAE encoder. For a pixel-space error $\mathbf{e}_x = \hat{\mathbf{x}} - \mathbf{x}$, the corresponding latent-space error is:

$$
\mathbf{e}_z \approx \mathbf{J}_\phi(\mathbf{x}) \cdot \mathbf{e}_x
\tag{26}
$$

where $\mathbf{J}_\phi(\mathbf{x}) = \frac{\partial \mathcal{E}_\phi}{\partial \mathbf{x}}|_{\mathbf{x}}$ is the encoder Jacobian. This is a first-order approximation valid for small errors. The critical implication is:

$$
\|\mathbf{e}_z\|_p \neq \|\mathbf{J}_\phi \cdot \mathbf{e}_x\|_p \neq \|\mathbf{e}_x\|_p
\tag{27}
$$

The Jacobian $\mathbf{J}_\phi$ mixes spatial and channel information nonlinearly. A spatially localised pixel error can map to a distributed latent error across multiple channels and positions. Therefore, the $L_p$ exponent that is optimal in pixel space ($p = 1.5$ for images) may not be optimal in latent space, because the error distribution is transformed.

**Hypotheses:**

- **H1 (Transfer):** The $L_p$ effect transfers with similar optimal exponents, because the VAE approximately preserves the error distribution geometry (the VAE is trained with $L_1$ + LPIPS, which encourages perceptual fidelity).

- **H2 (Attenuation):** The effect is attenuated, with $p = 2$ being near-optimal for all channels, because the VAE's KL regularisation normalises the latent distribution, making all channels statistically similar.

- **H3 (Inversion):** The optimal exponent shifts, e.g., $p > 2$ for image channels, because the VAE compresses low-frequency information into dominant latent modes, and the $L_p$ exponent must adapt to the spectral characteristics of the latent error.

**Ablation required:** A sweep over $p \in \{1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0\}$ for the image-only (Stage 1, $C=4$) case, evaluated on FID and SSIM.

### 3.4 Per-Channel $L_p$ in the MeanFlow Objective

Substituting the per-channel $L_p$ norm into the MeanFlow loss (Eq. 12):

$$
\mathcal{L}_{\text{MF-}L_p} = \mathbb{E}_{t,r,\mathbf{z}_0,\boldsymbol{\epsilon}} \left[ w(t) \cdot \sum_{c=1}^{C} \lambda_c \left( \frac{1}{|H'W'D'|} \sum_{h,w,d} |V_{\theta,c}(\mathbf{z}_t) - v_{c,c}(\mathbf{z}_t)|^{p_c} \right) \right]
\tag{28}
$$

where $V_{\theta,c}$ and $v_{c,c}$ are the $c$-th channel of the compound prediction and conditional velocity, respectively.

**For Stage 1 (image-only, $C=4$):** uniform $p_c = p$ across all channels; $p$ is ablated.

**For Stage 2 (joint image+mask, $C=8$ or $C=5$):**

$$
p_c = \begin{cases} p_{\text{img}} & \text{if } c \in \{1,2,3,4\} \text{ (image latent channels)} \\ p_{\text{mask}} & \text{if } c \in \{5,6,7,8\} \text{ (mask latent channels)} \end{cases}
\tag{29}
$$

### 3.5 Interaction with Adaptive Weighting

The adaptive weight $w(t)$ (Eq. 14) normalises the loss magnitude. When combined with per-channel $L_p$, the effective gradient for channel $c$ at voxel $(h,w,d)$ is:

$$
g_{c,h,w,d} \propto \frac{p_c \cdot |e_{c,h,w,d}|^{p_c - 1} \cdot \text{sign}(e_{c,h,w,d})}{\sum_{c'} \lambda_{c'} \cdot \frac{1}{|H'W'D'|} \sum_{h',w',d'} |e_{c',h',w',d'}|^{p_{c'}} + c_0}
\tag{30}
$$

This creates an **implicit cross-channel normalisation**: channels with smaller errors receive relatively larger gradients, preventing any single channel from dominating the loss.

---

## 4. x-Prediction vs. u-Prediction in Latent Space: Manifold-Theoretic Analysis

### 4.1 The pMF Argument in Pixel Space

Lu et al. (2026, §4) establish that x-prediction (Eq. 15) is superior to u-prediction (i.e., directly outputting $\mathbf{u}_\theta$) when the target lies on a low-dimensional manifold. Their argument proceeds as follows:

**Definition (Generalised Manifold Hypothesis).** The x-prediction target $\hat{\mathbf{x}}(\mathbf{z}_t, r, t)$ lies approximately on the data manifold $\mathcal{M}$ for all $(r, t)$, because it is a denoised estimate of the data. The u-prediction target $\mathbf{u}(\mathbf{z}_t, r, t)$, being a velocity field, spans a higher-dimensional space.

**Theorem (Informal, pMF §4).** For a ViT with patch dimensionality $d_p$ and hidden dimension $d_h$:
- If $d_p / d_h > 1$: x-prediction is easier because the network can exploit the low-dimensional manifold structure of $\hat{\mathbf{x}}$ via the bottleneck.
- If $d_p / d_h \leq 1$: x-prediction and u-prediction have similar difficulty because the bottleneck is not binding.

**Empirical evidence (pMF Table 2):** On ImageNet 256×256, x-prediction achieves FID 3.46 vs. u-prediction's 4.30 (a 19.5% relative improvement).

### 4.2 Adapting the Argument to Latent Space with UNets

In our setting, the architecture is a 3D UNet (not a ViT), and the input is a latent representation $\mathbf{z}_t \in \mathbb{R}^{4 \times 32^3}$ (not a pixel image). The pMF manifold argument must be reinterpreted.

**4.2.1 UNet vs. ViT bottleneck.** A UNet processes the input through an encoder-decoder path with skip connections. The bottleneck is the deepest level of the U, where the spatial resolution is minimised (e.g., $4^3$ or $2^3$) and the channel count is maximised (e.g., 512 or 1024). The "effective bottleneck dimensionality" is:

$$
d_{\text{bottleneck}} = C_{\text{deep}} \times H_{\text{deep}} \times W_{\text{deep}} \times D_{\text{deep}}
\tag{31}
$$

For a typical 3D UNet with $C_{\text{deep}} = 512$ and spatial resolution $4^3$:

$$
d_{\text{bottleneck}} = 512 \times 4^3 = 32{,}768
$$

The input dimensionality is $d_{\text{input}} = 4 \times 32^3 = 131{,}072$, giving a ratio:

$$
\frac{d_{\text{input}}}{d_{\text{bottleneck}}} = \frac{131{,}072}{32{,}768} = 4
\tag{32}
$$

This ratio $> 1$ suggests the bottleneck is binding, and x-prediction should be advantageous — analogous to the pMF ViT setting.

**4.2.2 VAE pre-compression effect.** However, the VAE has already mapped the data to a lower-dimensional manifold $\mathcal{M}_z \subset \mathbb{R}^{131,072}$. The intrinsic dimensionality of $\mathcal{M}_z$ is potentially much less than 131,072 (it is bounded by the intrinsic dimensionality of the brain MRI manifold, estimated at $\sim 100$–$1{,}000$ for structural brain MRI; Tenenbaum et al., 2000). This means both $\hat{\mathbf{z}}_0$ (x-prediction target) and $\mathbf{u}$ (u-prediction target) may already live in a low-dimensional subspace.

**Hypothesis (diminished effect):** The VAE's dimensionality reduction narrows the gap between x-prediction and u-prediction, because the manifold advantage is partially "consumed" by the VAE.

**Formal argument.** Let $d_{\text{int}}^x$ and $d_{\text{int}}^u$ denote the intrinsic dimensionalities of the x-target and u-target manifolds, respectively. In pixel space, $d_{\text{int}}^x \ll d_{\text{int}}^u$ (the image manifold is much lower-dimensional than the velocity field). In latent space:

$$
d_{\text{int}}^{z,x} \leq d_{\text{int}}^x, \quad d_{\text{int}}^{z,u} \leq d_{\text{int}}^u
\tag{33}
$$

but the gap $d_{\text{int}}^{z,u} - d_{\text{int}}^{z,x}$ may be smaller than $d_{\text{int}}^u - d_{\text{int}}^x$ if the VAE compresses both targets similarly.

**Regardless of outcome, this ablation is novel and publishable.** See §9.1 for the formal ablation design.

### 4.3 v-Head Architecture for Tangent Vectors

When using x-prediction, the instantaneous velocity estimate $\tilde{\mathbf{v}}_\theta$ used as the JVP tangent vector can be obtained in two ways:

1. **Implicit v (from u at $r=t$):** $\tilde{\mathbf{v}}_\theta = \mathbf{u}_\theta(\mathbf{z}_t, t, t) = (\mathbf{z}_t - \hat{\mathbf{z}}_{0,\theta}(\mathbf{z}_t, t, t)) / t$. This requires no architectural change but couples the tangent vector to the x-prediction.

2. **Explicit v-head:** A separate output head on the UNet that directly predicts $\tilde{\mathbf{v}}_\theta$. This decouples the tangent vector from the x-prediction at the cost of additional parameters.

Following pMF (Lu et al., 2026, §5), we use option (1) for simplicity, as it avoids the extra v-head and Lu et al. found no significant benefit from the explicit v-head.

---

## 5. LoRA Fine-Tuning for Domain Adaptation under Data Scarcity

### 5.1 Low-Rank Adaptation (LoRA)

LoRA (Hu et al., 2022) adapts a pretrained weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ by injecting a low-rank update:

$$
\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r},\; \mathbf{A} \in \mathbb{R}^{r \times k}
\tag{34}
$$

where $r \ll \min(d, k)$ is the rank, and $\alpha$ is a scaling factor. Only $\mathbf{A}$ and $\mathbf{B}$ are trained; $\mathbf{W}_0$ is frozen.

**Parameter count.** For rank $r = 16$ applied to attention $q, k, v$ projections (each $d_{\text{model}} \times d_{\text{model}}$) across $L$ attention layers:

$$
|\theta_{\text{LoRA}}| = 3 \times L \times 2 \times r \times d_{\text{model}} = 6 L r d_{\text{model}}
\tag{35}
$$

For a typical 3D UNet with $L \approx 16$ attention layers and $d_{\text{model}} = 256$:

$$
|\theta_{\text{LoRA}}| = 6 \times 16 \times 16 \times 256 = 393{,}216 \approx 0.4\text{M}
\tag{36}
$$

This is $< 1\%$ of the full model parameters, critical for the data-scarce FCD setting ($N \approx 50$–$100$).

### 5.2 Domain Adaptation Strategy

**Stage 2 protocol:**

1. Freeze the base MeanFlow model (trained in Stage 1 on healthy brain MRI).
2. Modify the first convolutional layer to accept $C_{\text{in}} = 8$ (or 5) channels.
3. Apply LoRA to attention layers only.
4. Train with the per-channel $L_p$ loss (§3.4, Eq. 28–29).

### 5.3 Joint Image-Mask Synthesis

Two strategies for incorporating the segmentation mask:

**Strategy A (VAE-encoded mask):** Encode both the image and the binary mask through the MAISI VAE:

$$
\mathbf{z}_{\text{joint}} = [\mathcal{E}_\phi(\mathbf{x}_{\text{img}}),\; \mathcal{E}_\phi(\mathbf{m})] \in \mathbb{R}^{8 \times 32 \times 32 \times 32}
\tag{37}
$$

**Risk:** Binary masks ($\{0, 1\}$) are far from the continuous-valued data the VAE was trained on. The reconstructed mask $\mathcal{D}_\phi(\mathcal{E}_\phi(\mathbf{m}))$ may exhibit blurring or ringing.

**Strategy B (Downsampled mask):** Nearest-neighbour downsample the mask to match the latent spatial resolution:

$$
\mathbf{z}_{\text{joint}} = [\mathcal{E}_\phi(\mathbf{x}_{\text{img}}),\; \text{NN}_{\downarrow 4}(\mathbf{m})] \in \mathbb{R}^{5 \times 32 \times 32 \times 32}
\tag{38}
$$

**Advantage:** No VAE artefacts on the mask. **Disadvantage:** The mask occupies a single channel vs. four, creating a channel imbalance.

**Decision criterion:** We implement both strategies and select based on the mask reconstruction quality (measured by Dice score of $\text{round}(\mathcal{D}_\phi(\mathcal{E}_\phi(\mathbf{m})))$ vs. $\mathbf{m}$). If Strategy A's Dice $> 0.85$, we use Strategy A; otherwise Strategy B.

### 5.4 Theoretical Justification for LoRA in MeanFlow

LoRA's effectiveness in MeanFlow fine-tuning can be motivated by the observation that the velocity field for pathological brain MRI (with FCD lesions) differs from healthy brain MRI primarily in a low-dimensional subspace corresponding to the lesion region and its surroundings. The bulk of the brain anatomy is shared between healthy and pathological populations. LoRA's low-rank constraint naturally captures this structured deviation.

---

## 6. MAISI VAE: Foundation Encoder–Decoder for Medical Volumes

### 6.1 Architecture Summary

The MAISI VAE (Guo et al., 2024) is a 3D VAE-GAN (cf. VQGAN; Esser et al., 2021) with the following properties:

| Property | Value |
|---|---|
| Input | 1-channel 3D volume, multiples of 128 |
| Encoder stages | 3 stages of $2\times$ strided 3D convolution → $4\times$ spatial downsampling |
| Latent channels | 4 (with KL regularisation) |
| Decoder | Symmetric to encoder, with skip-less transposed convolutions |
| Training losses | $L_1$ reconstruction + LPIPS perceptual + PatchGAN adversarial + KL |
| KL calibration | Adaptive KL weight to maintain $\sigma_c \in [0.9, 1.1]$ |
| Training data | 37,243 CT + 17,887 MRI volumes (including brain MRI) |

### 6.2 Reconstruction Quality Validation Protocol

Before training any generative model, we must validate that the frozen MAISI VAE faithfully reconstructs brain MRI at our target resolution. The protocol:

1. Select a held-out set of $N_{\text{val}} = 100$ brain MRI volumes from IXI.
2. Preprocess: skull-strip, resample to $128^3$ at $1 \text{mm}^3$ isotropic, intensity normalise to $[0, 1]$.
3. Encode and decode: $\hat{\mathbf{x}} = \mathcal{D}_\phi(\mathcal{E}_\phi(\mathbf{x}))$.
4. Compute metrics:
   - **SSIM** (Structural Similarity Index): $> 0.90$ required
   - **PSNR** (Peak Signal-to-Noise Ratio): $> 30$ dB required
   - **LPIPS** (Learned Perceptual Image Patch Similarity): $< 0.10$ required (adapted for 3D slices)
5. Visual inspection of cortical boundaries (the known VAE smoothing region).

### 6.3 Latent Statistics Characterisation

After encoding the full training set, we compute and report:

- Per-channel mean $\mu_c = \mathbb{E}[z_{0,c}]$ and standard deviation $\sigma_c = \text{std}(z_{0,c})$
- Per-channel skewness $\gamma_c$ and kurtosis $\kappa_c$ (to quantify deviation from Gaussianity)
- Channel correlation matrix $\mathbf{R} \in \mathbb{R}^{4 \times 4}$ where $R_{ij} = \text{corr}(z_{0,i}, z_{0,j})$
- PCA spectrum of the flattened latents (to estimate effective dimensionality)

These statistics inform normalisation and may reveal channel-specific structure that guides the $L_p$ exponent selection.

---

## 7. Literature Framing and Positioning

### 7.1 Taxonomy of 3D Medical Image Synthesis Methods

We position our work within four axes:

| Axis | Options | Our Choice |
|---|---|---|
| **Representation space** | Pixel vs. Latent | Latent (MAISI VAE) |
| **Generative paradigm** | DDPM / FM / Rectified Flow / MeanFlow | MeanFlow (1-NFE) |
| **Architecture** | 3D UNet / ViT / Hybrid | 3D UNet (MAISI-style) |
| **Sampling steps** | 1 / few / many | 1 (primary), multi-step (ablation) |

### 7.2 Detailed Literature Comparison

**7.2.1 MAISI (Guo et al., 2024) and MAISI-v2 (Zhao et al., 2025).**

MAISI establishes the latent diffusion paradigm for 3D medical volumes with a pretrained 3D VAE. MAISI-v2 replaces the DDPM sampler with Rectified Flow, reducing sampling from 1000 steps to 5–50. However, neither achieves 1-step generation.

**Our advancement:** We replace Rectified Flow with MeanFlow, achieving 1-NFE sampling while using the same MAISI VAE. This is a direct, controlled comparison: same encoder–decoder, same latent space, different generative training objective.

**7.2.2 MOTFM (Yazdani et al., 2025).**

MOTFM applies optimal transport flow matching to 3D brain MRI synthesis in **pixel space**. Achieves comparable quality to DDPM with 10–50 steps. Operates at $128^3$ with a custom UNet.

**Our advancement:** (i) Latent-space operation reduces compute by ~$16\times$; (ii) MeanFlow enables 1-step vs. MOTFM's 10–50 steps; (iii) we add per-channel $L_p$ loss and joint image-mask synthesis.

**7.2.3 Med-DDPM (Dorjsembe et al., 2024).**

Med-DDPM generates 3D brain MRI with semantic conditioning in pixel space using 1000-step DDPM. Demonstrates brain segmentation conditioning.

**Our advancement:** Order-of-magnitude reduction in sampling cost (1 vs. 1000 steps), latent operation, and explicit rare-pathology support.

**7.2.4 BrLP (Puglisi et al., 2025).**

Brain Latent Progression uses a latent diffusion model for longitudinal brain MRI synthesis. Acknowledges VAE smoothing as a limitation. Uses 2D slice-by-slice generation.

**Our advancement:** Full 3D volumetric generation; MeanFlow's 1-step inference; explicit evaluation of VAE smoothing via SynthSeg.

**7.2.5 Flow Matching (Lipman et al., 2023) and Rectified Flow (Liu et al., 2023).**

These foundational works establish the flow-matching paradigm. MeanFlow (Geng et al., 2025a) is a direct successor that introduces the average velocity concept for 1-step generation. Our work brings this advancement to the medical imaging domain.

**7.2.6 SLIM-Diff (Pascual-González et al., 2026).**

SLIM-Diff introduces per-channel $L_p$ loss for joint image-mask diffusion in pixel-space DDPM. We extend this insight to latent-space MeanFlow, investigating the transfer across the VAE nonlinearity.

**7.2.7 Concurrent: AlphaFlow (Zhang et al., 2025), Re-MeanFlow, Decoupled MeanFlow.**

The MeanFlow family is expanding rapidly. AlphaFlow unifies FM, Shortcut Models, and MeanFlow via the $\alpha$-Flow objective. Re-MeanFlow combines Rectified Flow pre-straightening with MeanFlow fine-tuning. These are concurrent to our work and focus on 2D natural images; we are the first to bring any MeanFlow variant to 3D medical imaging.

### 7.3 Gap Statement

**No prior work has applied MeanFlow (or any 1-step flow-based model) to 3D medical image synthesis.** The closest works are:
- MAISI-v2: same VAE, but uses multi-step Rectified Flow
- MOTFM: flow matching for 3D brain MRI, but multi-step and pixel-space
- pMF: 1-step MeanFlow, but 2D natural images

Our work fills the intersection: **1-step + latent + 3D + medical**.

---

## 8. Novel Contributions Statement

### Contribution 1 (Primary): First Application of MeanFlow to 3D Medical Image Synthesis

We introduce a latent MeanFlow model trained on pre-computed latents from a frozen MAISI 3D VAE. The model achieves one-step (1-NFE) generation of $128^3$ brain MRI volumes. This is the first application of the MeanFlow framework to any 3D domain and the first 1-step generative model for 3D medical imaging.

### Contribution 2 (Methodological): Per-Channel $L_p$ Loss in Latent MeanFlow

We extend the SLIM-Diff per-channel $L_p$ loss from pixel-space DDPM to latent-space MeanFlow. We provide empirical investigation of whether the $L_p$ exponent effect transfers across the VAE nonlinearity, yielding insights into the interaction between loss geometry and latent representation structure.

### Contribution 3 (Analytical): x-Prediction vs. u-Prediction Ablation in Latent Space

We provide the first investigation of the x-prediction vs. u-prediction dichotomy (established by pMF in pixel space with ViTs) in latent space with 3D UNets. We test the hypothesis that the VAE's pre-compression attenuates the x-prediction advantage.

### Contribution 4 (Application): One-Step Joint Image-Mask Synthesis for Rare Epilepsy Pathology

We demonstrate LoRA fine-tuning of the latent MeanFlow model for joint synthesis of FLAIR MRI and FCD segmentation masks, combining one-step generation with per-channel $L_p$ loss in a clinically relevant data-scarce setting.

---

## 9. Ablation Design with Statistical Rigour

### 9.1 Ablation 1: x-Prediction vs. u-Prediction

| Setting | Description |
|---|---|
| **x-pred** | Network outputs $\hat{\mathbf{z}}_0$; $\mathbf{u}_\theta = (\mathbf{z}_t - \hat{\mathbf{z}}_0) / t$ |
| **u-pred** | Network directly outputs $\mathbf{u}_\theta$ |
| **Fixed variables** | Same architecture, same learning rate, same batch size, same training data, same $t$-sampler, same EMA schedule |
| **Metric** | FID (slice-wise, axial/coronal/sagittal), 3D-FID (Med3D features), SSIM, PSNR |
| **Trials** | 3 seeds per setting |
| **Statistical test** | Two-sample $t$-test (Welch's) on FID across seeds; report mean $\pm$ std |

**Hypothesis:** x-pred $\leq$ u-pred in FID (lower is better), with diminished gap relative to pMF's pixel-space result.

### 9.2 Ablation 2: $L_p$ Exponent Sweep

| $p$ | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| Training | Full | Full | Full | Full | Full | Full | Full |

**Evaluation:** FID, SSIM, PSNR at convergence (500 epochs). Report the Pareto frontier across FID-SSIM.

**Statistical protocol:** 2 seeds per $p$ value (14 runs total). We report mean $\pm$ std and perform ANOVA followed by Tukey's HSD for pairwise comparisons.

### 9.3 Ablation 3: Number of Sampling Steps

Evaluate the trained MeanFlow model at $K \in \{1, 2, 5, 10, 25, 50\}$ NFE using Euler sampling. The hypothesis: MeanFlow's 1-NFE quality is competitive with multi-step sampling at $K \leq 10$, and may even degrade at very high $K$ (as MeanFlow is trained for 1-step, not multi-step).

**Comparison:** Retrain a standard Rectified Flow baseline (same architecture, same data) and evaluate at the same $K$ values.

### 9.4 Ablation 4: Per-Channel $L_p$ in Joint Synthesis

For Stage 2 (joint image + mask), compare:

| Setting | Image $p$ | Mask $p$ |
|---|---|---|
| Uniform $L_2$ | 2.0 | 2.0 |
| SLIM-Diff transfer | 1.5 | 2.0 |
| Image-focused | 1.0 | 2.0 |
| Mask-focused | 2.0 | 3.0 |
| Latent-optimised (from §9.2 result) | $p^*$ | 2.0 |

**Metrics:** Image quality (FID, SSIM) + mask quality (Dice, Hausdorff distance).

### 9.5 Ablation 5: Mask Encoding Strategy

Compare Strategy A (VAE-encoded mask, $C=8$) vs. Strategy B (downsampled mask, $C=5$) on mask reconstruction Dice and joint synthesis quality.

---

## 10. Evaluation Protocol

### 10.1 Image Quality Metrics

| Metric | Description | Implementation |
|---|---|---|
| **FID** (Fréchet Inception Distance) | Distribution-level quality; computed slice-wise (axial) on 2D slices extracted from 3D volumes | `torch-fidelity` or custom with InceptionV3 |
| **3D-FID** | FID computed on 3D feature vectors extracted by Med3D (Chen et al., 2019) or SynthSeg encoder | Custom implementation |
| **SSIM** | Per-volume structural similarity; averaged over all volumes | `skimage.metrics.structural_similarity` |
| **PSNR** | Peak signal-to-noise ratio | Standard formula |

### 10.2 Morphological Metrics (via SynthSeg)

Following MOTFM (Yazdani et al., 2025), we apply **SynthSeg** (Billot et al., 2023) — a domain-agnostic brain segmentation tool — to both real and synthetic volumes, then compare:

| Metric | Description |
|---|---|
| **SynthSeg Dice** | Dice overlap between SynthSeg labels of real and synthetic volumes (paired by nearest neighbour in feature space) |
| **Volume correlation** | Pearson $r$ between regional volumes (hippocampus, ventricles, cortex) in real vs. synthetic |
| **Morphological realism** | Distribution of regional volumes: KL divergence between real and synthetic histograms |

**Why SynthSeg?** It provides a proxy for anatomical validity that is independent of the generative model. If synthetic volumes have correct regional volumes and cortical folding patterns, they are anatomically plausible.

### 10.3 VAE Smoothing Quantification

To explicitly quantify the VAE smoothing artefact:
1. Compute the high-frequency energy ratio $\rho = \sum_{|\mathbf{k}| > k_0} |F(\hat{\mathbf{x}})|^2 / \sum_{\mathbf{k}} |F(\hat{\mathbf{x}})|^2$ for real, VAE-reconstructed, and generated volumes, where $F$ denotes the 3D DFT and $k_0$ is a frequency cutoff.
2. Report $\rho_{\text{real}}$, $\rho_{\text{VAE-recon}}$, $\rho_{\text{generated}}$ to disentangle smoothing from the VAE vs. the generative model.

### 10.4 Sampling Speed

| Method | NFE | Time per volume (A100) |
|---|---|---|
| MAISI (DDPM) | 50 | ~50s |
| MAISI-v2 (Rectified Flow) | 5–50 | ~5–50s |
| MOTFM | 10–50 | ~10–50s |
| Med-DDPM | 1000 | ~1000s |
| **Ours (MeanFlow)** | **1** | **~1s** |

(Approximate; exact timing depends on UNet size and batch.)

---

## 11. Data Strategy

### 11.1 Datasets

| Dataset | $N$ | Modality | Resolution | Preprocessing |
|---|---|---|---|---|
| **IXI** | ~580 | T1W, T2W, PD, MRA, DTI | Variable → $128^3$ | Skull-strip (SynthStrip), resample $1\text{mm}^3$, intensity $[0,1]$ |
| **OASIS-3** | ~1,000 | T1W, FLAIR | Variable → $128^3$ | Same as IXI |
| **BraTS 2021** (MSD) | ~484 | T1, T1-Gd, T2, FLAIR | $240 \times 240 \times 155 \to 128^3$ | Centre-crop + resample |
| **Epilepsy FCD** | ~50–100 | FLAIR + mask | Variable → $128^3$ | Same as IXI; masks binary |

### 11.2 Data Splits

| Split | IXI+OASIS | BraTS | FCD |
|---|---|---|---|
| Train | 80% | 80% | 80% |
| Val | 10% | 10% | 10% |
| Test | 10% | 10% | 10% |

### 11.3 Preprocessing Pipeline

```
Raw NIfTI → SynthStrip skull-stripping → N4 bias correction →
Resample to 1mm³ isotropic → Centre-crop/pad to 128³ →
Intensity normalisation [0, 1] (percentile-based) → MAISI VAE encode → Latent z₀
```

---

## 12. Proposed Paper Structure

| Section | Content | Length |
|---|---|---|
| **Abstract** | Problem, method, key results (1-NFE, FID comparison), impact | 150 words |
| **1. Introduction** | Gap: no 1-step model for 3D medical synthesis; motivation (clinical pipelines); our approach (latent MeanFlow); contributions | 1 page |
| **2. Related Work** | §2.1 3D medical synthesis (MAISI, MAISI-v2, MOTFM, Med-DDPM, BrLP); §2.2 Flow matching & MeanFlow (FM, RF, MF, iMF, pMF, AlphaFlow); §2.3 $L_p$ loss geometry (SLIM-Diff) | 1.5 pages |
| **3. Method** | §3.1 Preliminaries (FM, MeanFlow identity); §3.2 Latent MeanFlow (VAE + MF in $\mathcal{Z}$); §3.3 x-prediction reparameterisation; §3.4 Per-channel $L_p$ loss; §3.5 LoRA for joint synthesis | 3 pages |
| **4. Experiments** | §4.1 Setup (data, metrics, baselines); §4.2 VAE validation; §4.3 Main results (unconditional 1-NFE); §4.4 Ablation: x-pred vs. u-pred; §4.5 Ablation: $L_p$ sweep; §4.6 Ablation: NFE steps; §4.7 Joint image-mask synthesis (FCD) | 3 pages |
| **5. Discussion** | 1-step quality analysis; VAE smoothing; latent $L_p$ transfer; limitations; clinical implications | 1 page |
| **6. Conclusion** | Summary; future work (higher resolution, multi-modal conditioning) | 0.5 pages |

**Target venues (ranked):**
1. *Medical Image Analysis* (Elsevier, IF ~10.9) — comprehensive method + clinical application
2. *IEEE Transactions on Medical Imaging* (IF ~10.6) — emphasis on methodology
3. *MICCAI 2026* (conference, 8-page format) — condensed method-only version

---

## 13. References

1. Geng, Z., Deng, M., Bai, X., Kolter, J.Z., He, K. "Mean Flows for One-step Generative Modeling." NeurIPS 2025 (Oral). arXiv:2505.13447.
2. Geng, Z., Lu, Y., Wu, Z., Shechtman, E., Kolter, J.Z., He, K. "Improved Mean Flows: On the Challenges of Fastforward Generative Models." arXiv:2512.02012, 2025.
3. Lu, Y., Lu, S., Sun, Q., Zhao, H., et al. "One-step Latent-free Image Generation with Pixel Mean Flows." arXiv:2601.22158, 2026.
4. Guo, P., Zhao, C., Yang, D., et al. "MAISI: Medical AI for Synthetic Imaging." WACV 2025. arXiv:2409.11169, 2024.
5. Zhao, C., Guo, P., Yang, D., et al. "MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss." arXiv:2508.05772, 2025.
6. Pascual-González, M., et al. "SLIM-Diff: Shared Latent Image-Mask Diffusion with Lp Loss for Data-Scarce Epilepsy FLAIR MRI." arXiv:2602.03372, 2026.
7. Yazdani, M., et al. "Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality." MICCAI 2025. arXiv:2503.00266.
8. Dorjsembe, Z., et al. "Semantic 3D Brain MRI Synthesis with Channel-Wise Conditioning." IEEE JBHI, 2024.
9. Puglisi, R., et al. "Brain Latent Progression (BrLP)." Medical Image Analysis, 2025.
10. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M. "Flow Matching for Generative Modeling." ICLR, 2023.
11. Liu, X., Gong, C., Liu, Q. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR, 2023.
12. Albergo, M.S., Vanden-Eijnden, E. "Building Normalizing Flows with Stochastic Interpolants." ICLR, 2023.
13. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR, 2022.
14. Esser, P., Rombach, R., Ommer, B. "Taming Transformers for High-Resolution Image Synthesis." CVPR, 2021.
15. Hu, E., Shen, Y., Wallis, P., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR, 2022.
16. Zhang, H., et al. "AlphaFlow: Understanding and Improving MeanFlow Models." arXiv:2510.20771, 2025.
17. Billot, B., et al. "SynthSeg: Segmentation of Brain MRI Scans of Any Contrast and Resolution without Retraining." Medical Image Analysis, 2023.
18. Tenenbaum, J.B., de Silva, V., Langford, J.C. "A Global Geometric Framework for Nonlinear Dimensionality Reduction." Science, 2000.
19. Chen, S., Ma, K., Zheng, Y. "Med3D: Transfer Learning for 3D Medical Image Analysis." arXiv:1904.00625, 2019.
20. Ho, J., Jain, A., Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS, 2020.
21. Song, Y., Sohl-Dickstein, J., Kingma, D.P., et al. "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR, 2021.
22. Song, Y., Dhariwal, P. "Improved Techniques for Training Consistency Models." ICLR, 2024.
23. Karras, T., Aittala, M., Aila, T., Laine, S. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS, 2022.
24. Frans, K., Hafner, D., Levine, S., Abbeel, P. "One Step Diffusion via Shortcut Models." ICLR, 2025.
25. Jordan, K., et al. "Muon: An Optimizer for Hidden Layers in Neural Networks." 2024.
26. Li, T. and He, K. "Back to Basics: Let Denoising Generative Models Denoise." arXiv:2511.13720, 2025.
