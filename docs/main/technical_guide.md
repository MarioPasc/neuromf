# NeuroMF: Technical Implementation Guide

**Project Codename:** NeuroMF (Latent MeanFlow for 3D Brain MRI)
**Author:** Mario Pascual-González
**Date:** February 2026
**Version:** 1.0
**Purpose:** Step-by-step implementation guide for an autonomous coding agent (Opus 4.6)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Phase 0: Environment Bootstrap and VAE Validation](#2-phase-0-environment-bootstrap-and-vae-validation)
3. [Phase 1: Latent Pre-computation Pipeline](#3-phase-1-latent-pre-computation-pipeline)
4. [Phase 2: Toy Experiment — MeanFlow on a Toroidal Manifold](#4-phase-2-toy-experiment-meanflow-on-a-toroidal-manifold)
5. [Phase 3: MeanFlow Loss Integration with 3D UNet](#5-phase-3-meanflow-loss-integration-with-3d-unet)
6. [Phase 4: Training on Brain MRI Latents](#6-phase-4-training-on-brain-mri-latents)
7. [Phase 5: Evaluation Suite](#7-phase-5-evaluation-suite)
8. [Phase 6: Ablation Runs](#8-phase-6-ablation-runs)
9. [Phase 7: LoRA Fine-Tuning for FCD Joint Synthesis](#9-phase-7-lora-fine-tuning-for-fcd-joint-synthesis)
10. [Phase 8: Paper Figures and Tables](#10-phase-8-paper-figures-and-tables)
11. [Agent Context Specification](#11-agent-context-specification)

---

## 1. Project Structure

### 1.1 Repository Layout

```
neuromf/
├── README.md
├── pyproject.toml                  # Project metadata, dependencies
├── setup.cfg
├── .gitignore
├── LICENSE
│
├── configs/                        # OmegaConf YAML configurations
│   ├── base.yaml                   # Shared defaults
│   ├── vae_validation.yaml
│   ├── encode_dataset.yaml
│   ├── toy_toroid.yaml
│   ├── train_meanflow.yaml
│   ├── train_rectflow_baseline.yaml
│   ├── evaluate.yaml
│   ├── ablation_xpred_upred.yaml
│   ├── ablation_lp_sweep.yaml
│   ├── ablation_nfe_steps.yaml
│   ├── lora_fcd.yaml
│   └── overrides/                  # Per-experiment override files
│
├── src/
│   ├── external/                   # Cloned external repos (git submodules or copies)
│   │   ├── MeanFlow/              # github.com/Gsunshine/meanflow
│   │   ├── MeanFlow-PyTorch/      # github.com/HaoyiZhu/MeanFlow-PyTorch
│   │   ├── NV-Generate-CTMR/      # github.com/NVIDIA-Medtech/NV-Generate-CTMR 
│   │   └── MOTFM/                 # github.com/milad1378yz/MOTFM (baseline)
│   │
│   └── neuromf/                   # Core project package
│       ├── __init__.py
│       ├── wrappers/              # Adapters around external code
│       │   ├── __init__.py
│       │   ├── maisi_vae.py       # Frozen MAISI VAE encoder/decoder wrapper
│       │   ├── maisi_unet.py      # MAISI 3D UNet adapted for MeanFlow
│       │   └── meanflow_loss.py   # MeanFlow JVP loss extracted from external repos
│       │
│       ├── models/                # Model definitions
│       │   ├── __init__.py
│       │   ├── latent_meanflow.py # PyTorch Lightning module for Latent MeanFlow
│       │   ├── rectflow_baseline.py # Rectified Flow baseline for comparison
│       │   └── lora.py            # LoRA injection utilities
│       │
│       ├── data/                  # Data loading and preprocessing
│       │   ├── __init__.py
│       │   ├── mri_preprocessing.py   # NIfTI → preprocessed 128³ volumes
│       │   ├── latent_dataset.py      # Dataset of pre-computed .pt latents
│       │   ├── toroid_dataset.py      # Synthetic toroidal manifold data
│       │   └── fcd_dataset.py         # FCD image+mask pairs
│       │
│       ├── losses/                # Loss functions
│       │   ├── __init__.py
│       │   ├── meanflow_jvp.py    # Core MeanFlow JVP loss
│       │   ├── lp_loss.py         # Per-channel Lp loss
│       │   └── combined_loss.py   # iMF-style combined FM + MF loss
│       │
│       ├── sampling/              # Inference / sampling
│       │   ├── __init__.py
│       │   ├── one_step.py        # 1-NFE MeanFlow sampling
│       │   └── multi_step.py      # Euler multi-step sampling (for ablation)
│       │
│       ├── metrics/               # Evaluation metrics
│       │   ├── __init__.py
│       │   ├── fid.py             # Slice-wise FID
│       │   ├── fid_3d.py          # 3D-FID via Med3D features
│       │   ├── ssim_psnr.py       # SSIM and PSNR
│       │   ├── synthseg_metrics.py # SynthSeg-based morphological metrics
│       │   └── spectral.py        # High-frequency energy analysis
│       │
│       ├── utils/                 # Shared utilities
│       │   ├── __init__.py
│       │   ├── logging_config.py  # Python logging setup
│       │   ├── ema.py             # Exponential Moving Average
│       │   ├── time_sampler.py    # Logit-normal time sampler
│       │   ├── latent_stats.py    # Latent normalisation utilities
│       │   ├── visualisation.py   # 3D volume slice plotting
│       │   └── checkpoint.py      # Checkpoint management
│       │
│       └── errors/                # Custom exceptions
│           ├── __init__.py
│           └── exceptions.py
│
├── experiments/                   # Experiment execution layer
│   ├── cli/                       # Click/argparse entry points
│   │   ├── validate_vae.py        # Phase 0
│   │   ├── encode_dataset.py      # Phase 1
│   │   ├── run_toy_toroid.py      # Phase 2
│   │   ├── train.py               # Phases 3–4 (general training CLI)
│   │   ├── generate.py            # Sampling CLI
│   │   ├── evaluate.py            # Phase 5
│   │   ├── run_ablation.py        # Phase 6
│   │   └── train_lora.py          # Phase 7
│   │
│   ├── toy_toroid/                # Toy experiment tools for the experiment (scripts)
│   │   └── README.md
│   ├── vae_validation/            # VAE validation tools for the experiment (scripts)
│   │   └── README.md
│   ├── stage1_healthy/            # Main training tools for the experiment (scripts)
│   │   └── README.md
│   ├── ablations/                 # Ablation tools for the experiment (scripts)
│   │   ├── xpred_vs_upred/
│   │   ├── lp_sweep/
│   │   └── nfe_steps/
│   ├── stage2_fcd/                # LoRA FCD tools for the experiment (scripts)
│   │   └── README.md
│   └── utils/                     # Experiment-level utilities
│       ├── sweep.py               # Hyperparameter sweep launcher
│       └── aggregate_results.py   # Result aggregation scripts
│
├── tests/                         # Unit and integration tests
│   ├── test_maisi_vae_wrapper.py
│   ├── test_meanflow_loss.py
│   ├── test_lp_loss.py
│   ├── test_time_sampler.py
│   ├── test_ema.py
│   ├── test_toroid_dataset.py
│   ├── test_latent_dataset.py
│   └── test_one_step_sampling.py
│
├── docs/
│   ├── methodology_expanded.md    # This document (companion)
│   └── technical_guide.md         # This document
│
└── scripts/                       # Shell scripts
    ├── setup_env.sh               # Conda env creation
    ├── download_weights.sh        # MAISI weight download
    ├── clone_externals.sh         # Clone external repos
    └── run_full_pipeline.sh       # End-to-end pipeline
```

### 1.2 Key Design Decisions

| Decision | Rationale |
|---|---|
| **PyTorch Lightning** for training | Logging, checkpointing, multi-GPU, mixed precision out-of-the-box |
| **OmegaConf** for configs | Hierarchical YAML with command-line override; reproducible experiments |
| **wrappers/** pattern | Clean separation between external code and our adaptations; avoids modifying external repos |
| **`torch.func.jvp`** (not custom) | Built-in, tested, correct; avoids subtle AD bugs |
| **Pre-computed latents** | Standard LDM practice; eliminates VAE from training loop |
| **Dataclass-based configs** | Type-safe, IDE-friendly configuration management |

---

## 2. Phase 0: Environment Bootstrap and VAE Validation

### 2.1 Objectives

1. Create the conda environment with all dependencies.
2. Download MAISI VAE weights.
3. Build the MAISI VAE wrapper (`src/neuromf/wrappers/maisi_vae.py`).
4. Validate reconstruction quality on IXI brain MRI.

### 2.2 Implementation Steps

**Step 0.1: Environment Setup**
```bash
# scripts/setup_env.sh
conda create -n neuromf python=3.11
conda activate neuromf
pip install torch>=2.1 torchvision monai>=1.3 pytorch-lightning>=2.0
pip install omegaconf einops nibabel  scikit-image
pip install torch-fidelity lpips
```

**Step 0.2: MAISI VAE Wrapper**

The wrapper must:
- Load MAISI VAE weights from the model zoo
- Expose `encode(x: Tensor) -> Tensor` and `decode(z: Tensor) -> Tensor`
- Handle input/output shape validation (assert 128³ input, 4×32³ output)
- Freeze all parameters (`.requires_grad_(False)`)
- Support bf16 inference

```python
# Pseudocode structure for maisi_vae.py
@dataclass
class MAISIVAEConfig:
    weights_path: str
    spatial_dims: int = 3
    in_channels: int = 1
    latent_channels: int = 4
    downsample_factor: int = 4

class MAISIVAEWrapper:
    """Frozen MAISI VAE encoder-decoder for 3D medical volumes.

    Loads pretrained weights and provides encode/decode methods.
    All parameters are frozen; no gradients flow through this module.
    """
    def __init__(self, config: MAISIVAEConfig) -> None: ...
    def encode(self, x: torch.Tensor) -> torch.Tensor: ...
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor: ...
```

**Step 0.3: Validation Script**

Download a subset of IXI (e.g., 20 T1W volumes), preprocess to 128³, encode → decode, compute metrics.

### 2.3 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P0-T1 | MAISI VAE weights load without error | No exceptions | `python -c "from neuromf.wrappers.maisi_vae import MAISIVAEWrapper; w = MAISIVAEWrapper(config)"` |
| P0-T2 | Encode produces correct shape | `z.shape == (B, 4, 32, 32, 32)` for `x.shape == (B, 1, 128, 128, 128)` | Unit test `tests/test_maisi_vae_wrapper.py::test_encode_shape` |
| P0-T3 | Decode produces correct shape | `x_hat.shape == (B, 1, 128, 128, 128)` for `z.shape == (B, 4, 32, 32, 32)` | Unit test `tests/test_maisi_vae_wrapper.py::test_decode_shape` |
| P0-T4 | Round-trip reconstruction SSIM > 0.90 | Mean SSIM over 20 IXI volumes > 0.90 | `experiments/cli/validate_vae.py --dataset ixi --n 20` → check `vae_validation/metrics.json` |
| P0-T5 | Round-trip PSNR > 30 dB | Mean PSNR > 30.0 | Same as P0-T4 |
| P0-T6 | VAE is frozen (no grads) | All `param.requires_grad == False` | Unit test asserting all VAE params are frozen |
| P0-T7 | bf16 inference works | No NaN/Inf in output | Test with `torch.autocast("cuda", dtype=torch.bfloat16)` |

**Phase 0 is PASSED when ALL of P0-T1 through P0-T7 are green.**

---

## 3. Phase 1: Latent Pre-computation Pipeline

### 3.1 Objectives

1. Build the MRI preprocessing pipeline (NIfTI → 128³ normalised tensor).
2. Encode all training volumes through frozen MAISI VAE.
3. Store latents as `.pt` files with metadata.
4. Compute and store latent statistics.

### 3.2 Implementation Steps

**Step 1.1: Preprocessing Module**

```python
# src/neuromf/data/mri_preprocessing.py
# Pipeline: NIfTI → SynthStrip → N4 → resample 1mm³ → crop/pad 128³ → normalise [0,1]
```

Uses MONAI transforms: `LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `CropForegroundd`, `Resized`, `ScaleIntensityd`.

**Step 1.2: Encoding Script**

```bash
python experiments/cli/encode_dataset.py \
    --config configs/encode_dataset.yaml \
    --dataset_root /path/to/IXI \
    --output_dir /path/to/latents/ixi \
    --batch_size 4
```

Each latent saved as `{subject_id}.pt` containing `{"z": tensor, "metadata": {...}}`.

**Step 1.3: Latent Statistics**

Compute after all encoding is complete:
```python
# src/neuromf/utils/latent_stats.py
# Per-channel: mean, std, skewness, kurtosis
# Cross-channel: correlation matrix
# PCA: explained variance ratio (top 50 components)
```

### 3.3 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P1-T1 | All volumes encode without error | 0 failures in batch encoding | Check logs for exceptions |
| P1-T2 | Latent shape correct | Every `.pt` file has `z.shape == (4, 32, 32, 32)` | Iterate and assert |
| P1-T3 | Per-channel mean ≈ 0 | $|\mu_c| < 0.5$ for all $c$ | `latent_stats.json` |
| P1-T4 | Per-channel std ∈ [0.5, 2.0] | $\sigma_c \in [0.5, 2.0]$ for all $c$ | `latent_stats.json` |
| P1-T5 | No NaN/Inf in latents | `torch.isfinite(z).all()` for all files | Scan all `.pt` files |
| P1-T6 | Latent dataset loads correctly | `LatentDataset.__getitem__` returns correct shape and type | Unit test `tests/test_latent_dataset.py` |
| P1-T7 | Round-trip: decode(load(.pt)) ≈ original | SSIM > 0.89 for 5 random volumes | Decode stored latents and compare to originals |

**Phase 1 is PASSED when ALL of P1-T1 through P1-T7 are green.**

---

## 4. Phase 2: Toy Experiment — MeanFlow on a Toroidal Manifold

### 4.1 Rationale

Before training on expensive brain MRI latents, we validate the entire MeanFlow pipeline on a **known manifold** where ground-truth properties can be verified analytically. We choose a **flat torus** $\mathbb{T}^2$ embedded in $\mathbb{R}^4$ (or a higher-dimensional ambient space matching the latent dimensionality), because:

1. **Known topology:** $\mathbb{T}^2$ has genus 1, Euler characteristic 0; generated samples must lie on the torus (verifiable).
2. **Known geometry:** The torus has constant curvature (for a flat torus) or known Gaussian curvature; we can check the Riemannian metric of generated samples.
3. **Computationally cheap:** We use a small 3D UNet (or even a simple MLP) on low-dimensional data.
4. **Tests the full pipeline:** time sampling, JVP computation, loss backward, 1-NFE sampling, multi-step sampling.

### 4.2 Toroidal Manifold Construction

**4.2.1 Flat Torus in $\mathbb{R}^4$**

The flat torus $\mathbb{T}^2$ can be isometrically embedded in $\mathbb{R}^4$ via:

$$
\phi(\theta_1, \theta_2) = \frac{1}{\sqrt{2}}\begin{pmatrix} \cos\theta_1 \\ \sin\theta_1 \\ \cos\theta_2 \\ \sin\theta_2 \end{pmatrix}, \quad \theta_1, \theta_2 \in [0, 2\pi)
$$

This is a 2-dimensional manifold embedded in $\mathbb{R}^4$ with uniform measure.

**4.2.2 Volumetric Toroidal Data (for 3D UNet testing)**

To test the full 3D UNet pipeline, we generate **synthetic 3D volumes** whose parameters are sampled from a toroidal manifold. Concretely:

1. Sample $(\theta_1, \theta_2) \sim \text{Uniform}([0, 2\pi)^2)$.
2. Map to volumetric parameters: $(\theta_1, \theta_2) \mapsto$ a 3D Gaussian blob in a $32^3$ volume, with:
   - Centre position: $(x_c, y_c, z_c)$ determined by the torus embedding (e.g., $x_c = R + r\cos\theta_1$, $y_c = r\sin\theta_1$, $z_c$ from $\theta_2$)
   - Width: controlled by $\theta_2$
   - Amplitude: controlled by $\theta_1$
3. The resulting 3D volumes have 4 channels (to match latent shape): each channel is a different function of $(\theta_1, \theta_2)$.

This creates a dataset where:
- The true data manifold is $\mathbb{T}^2$ (2D).
- The ambient space is $\mathbb{R}^{4 \times 32 \times 32 \times 32}$ (matching latent dimensions).
- We can verify generated samples lie on the torus by checking $\|\phi^{-1}(\text{generated})\|$ and the angular distribution.

**4.2.3 Alternative: Pure R⁴ Torus (for fast MLP testing)**

For an even faster validation (without the 3D UNet), train an MLP-based MeanFlow on 4D points sampled from the flat torus. This validates:
- JVP computation correctness
- Time sampling
- Loss convergence
- 1-NFE sample quality (should produce points on $\mathbb{T}^2$)

### 4.3 Implementation

```python
# src/neuromf/data/toroid_dataset.py
@dataclass
class ToroidConfig:
    major_radius: float = 3.0    # R: major radius
    minor_radius: float = 1.0    # r: minor radius
    n_samples: int = 10_000
    spatial_size: int = 32       # per axis
    n_channels: int = 4          # to match MAISI latent

class ToroidDataset(Dataset):
    """Synthetic dataset of 3D volumes parameterised by a 2-torus.

    Each sample is a 4-channel 32³ volume determined by two angular
    parameters (θ₁, θ₂) on a torus T². Used to validate the MeanFlow
    pipeline on a known manifold before brain MRI training.
    """
    ...
```

### 4.4 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P2-T1 | Toroid dataset generates valid samples | All samples finite, correct shape `(4, 32, 32, 32)` | Unit test |
| P2-T2 | MeanFlow loss computes without error on toroid batch | No NaN/Inf in loss; loss is finite and positive | Forward pass test |
| P2-T3 | JVP computation produces correct shape | JVP output shape == input shape | Unit test on `torch.func.jvp` |
| P2-T4 | Training loss decreases monotonically (after warmup) | Loss at epoch 100 < loss at epoch 10 < loss at epoch 1 | Training log |
| P2-T5 | 1-NFE samples lie approximately on the torus | Mean distance to torus surface $< \epsilon$ (e.g., 0.1) | Compute $\|r_{\text{sample}} - r_{\text{torus}}\|$ where $r_{\text{torus}}$ is the distance from the torus centre |
| P2-T6 | Angular distribution of generated samples is approximately uniform | KS-test p-value $> 0.01$ for $\hat{\theta}_1$ and $\hat{\theta}_2$ marginals vs. Uniform | Apply $\phi^{-1}$ to generated samples, extract angles, run KS test |
| P2-T7 | Multi-step sampling (5 steps) produces better torus-fidelity than 1-step | Mean torus distance at 5-step ≤ 1-step | Compare distances |
| P2-T8 | x-prediction and u-prediction both converge on toroid | Both reach loss $< \tau$ within 200 epochs | Train both, check final loss |

**Phase 2 is PASSED when P2-T1 through P2-T6 are ALL green. P2-T7 and P2-T8 are informational.**

### 4.5 Toroid Verification Mathematics

Given a generated sample $\hat{\mathbf{z}} \in \mathbb{R}^4$ from the pure $\mathbb{R}^4$ torus experiment, we can verify it lies on $\mathbb{T}^2$ by checking:

1. **Norm constraint:** For the flat torus embedding, $\|\hat{\mathbf{z}}\|_2 = 1$ (all points have unit norm). Compute $\delta_{\text{norm}} = |\|\hat{\mathbf{z}}\|_2 - 1|$.

2. **Angular extraction:** $\hat{\theta}_1 = \text{atan2}(\hat{z}_2, \hat{z}_1)$, $\hat{\theta}_2 = \text{atan2}(\hat{z}_4, \hat{z}_3)$.

3. **Pair-wise norm:** $\hat{z}_1^2 + \hat{z}_2^2 = 1/2$ and $\hat{z}_3^2 + \hat{z}_4^2 = 1/2$. Compute deviations.

For the volumetric torus (4-channel 32³ volumes), verification involves:
1. Extracting the underlying $(\theta_1, \theta_2)$ parameters from each generated volume (via regression on known features).
2. Checking the parameter distribution.
3. Checking that the generated volumes are visually consistent with the parametric family.

---

## 5. Phase 3: MeanFlow Loss Integration with 3D UNet

### 5.1 Objectives

1. Adapt MAISI's 3D UNet to accept MeanFlow's dual time conditioning $(r, t)$.
2. Implement the MeanFlow JVP loss (Eqs. 9–12 from the methodology).
3. Implement the iMF combined loss (Eq. 13).
4. Implement per-channel $L_p$ loss.
5. Implement the logit-normal time sampler.
6. Implement EMA.

### 5.2 Key Implementation Details

**5.2.1 Dual Time Conditioning**

The original MAISI UNet conditions on a single timestep $t$ via sinusoidal embedding → MLP → AdaGN/AdaLN. For MeanFlow, we need conditioning on $(r, t)$:

```python
# In maisi_unet.py wrapper
# Option A: Concatenate embeddings
emb_r = self.time_embed(r)   # sinusoidal embedding + MLP
emb_t = self.time_embed(t)
emb = emb_r + emb_t           # sum (following pMF)
# The UNet conditions on emb at each resolution level
```

Following pMF (Lu et al., 2026), we use **sum** of two separate sinusoidal embeddings, each processed by a 2-layer MLP, then summed. This is simpler than concatenation and empirically equivalent.

**5.2.2 JVP Computation**

The JVP must be computed with respect to $(\mathbf{z}_t, t)$ while holding $r$ fixed:

```python
import torch
from torch.func import jvp

def compute_meanflow_jvp(
    model_fn,     # u_θ(z_t, r, t) -> average velocity
    z_t,          # current noisy latent
    r,            # interval start
    t,            # interval end (current time)
    v_tangent,    # instantaneous velocity estimate (tangent for z_t)
) -> torch.Tensor:
    """Compute the JVP term in the MeanFlow identity.

    Returns: ∂u/∂z_t · v_tangent + ∂u/∂t · 1
    """
    # Primals: (z_t, t) — we differentiate w.r.t. these
    # r is held fixed (not differentiated)
    def fn(z, t_scalar):
        return model_fn(z, r, t_scalar)

    primals = (z_t, t)
    tangents = (v_tangent.detach(), torch.ones_like(t))
    _, jvp_value = jvp(fn, primals, tangents)
    return jvp_value
```

**Critical: `torch.func.jvp` requires functional-style code.** The model must be compatible with `torch.func`. This means:
- No in-place operations in the UNet forward pass
- No global state mutations
- All parameters must be passed explicitly (use `torch.func.functional_call` or `torch.func.grad` with `functorch`)

If the MAISI UNet uses in-place ops, we must refactor them. This is a known pain point.

**5.2.3 Workaround for In-Place Operations**

If the MAISI UNet uses in-place ReLU or in-place additions:
1. First attempt: use `torch.func.jvp` directly. If it errors, identify in-place ops.
2. Replace `F.relu(x, inplace=True)` with `F.relu(x, inplace=False)` in the wrapper.
3. If persistent issues: use **finite-difference approximation** of the JVP as a fallback:

$$
\text{JVP} \approx \frac{\mathbf{u}_\theta(\mathbf{z}_t + h \cdot \tilde{\mathbf{v}}_\theta, r, t + h) - \mathbf{u}_\theta(\mathbf{z}_t, r, t)}{h}
$$

with $h = 10^{-3}$. This is less efficient (2 forward passes vs. 1 JVP) but always works. **Use only as a debugging fallback.**

### 5.3 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P3-T1 | 3D UNet accepts $(r, t)$ conditioning | Forward pass without error on random input | Unit test |
| P3-T2 | UNet output shape matches input latent shape | `out.shape == (B, 4, 32, 32, 32)` | Unit test |
| P3-T3 | `torch.func.jvp` executes on UNet | No error; JVP output shape matches latent shape | Unit test `tests/test_meanflow_loss.py::test_jvp_shape` |
| P3-T4 | MeanFlow loss is finite and positive | `0 < loss < 1000` on random data | Unit test |
| P3-T5 | MeanFlow loss gradient flows to UNet params | `all(p.grad is not None for p in model.parameters() if p.requires_grad)` | Check after `loss.backward()` |
| P3-T6 | Per-channel $L_p$ loss computes correctly for $p \in \{1.0, 1.5, 2.0, 3.0\}$ | Matches hand-computed reference on small tensor | Unit test with known input |
| P3-T7 | Logit-normal time sampler produces correct distribution | KS-test vs. theoretical CDF, p-value $> 0.05$ on 10k samples | Statistical test |
| P3-T8 | EMA updates correctly | EMA params differ from model params after updates | Unit test |
| P3-T9 | Combined iMF loss (FM + MF) computes without error | Finite loss | Unit test |
| P3-T10 | JVP and gradients work with bf16 mixed precision | No NaN; loss finite | Test under `torch.autocast` |

**Phase 3 is PASSED when ALL of P3-T1 through P3-T10 are green.**

### 5.4 Common Failure Modes and Mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| In-place ops in UNet | `torch.func.jvp` raises `RuntimeError` | Replace in-place ops; see §5.2.3 |
| JVP NaN at small $t$ | Loss becomes NaN near $t=0$ | Clip $t \geq 0.05$; check $1/t$ division |
| Memory OOM on JVP | CUDA OOM during JVP forward | Reduce batch size; use gradient checkpointing in UNet |
| FlashAttention incompatibility | JVP fails at attention layers | Disable FlashAttention (`torch.backends.cuda.flash_sdp_enabled = False`) |

---

## 6. Phase 4: Training on Brain MRI Latents

### 6.1 Objectives

1. Train the latent MeanFlow model on IXI + OASIS-3 pre-computed latents.
2. Monitor training with : loss curves, EMA selection, sample quality.
3. Periodically generate 1-NFE samples, decode through VAE, and visualise.

### 6.2 Training Configuration

```yaml
# configs/train_meanflow.yaml
model:
  architecture: "maisi_3d_unet"
  in_channels: 4
  out_channels: 4
  prediction_type: "x"   # x-prediction (primary); "u" for ablation
  time_embed_dim: 256
  model_channels: 128
  num_res_blocks: 2
  channel_mult: [1, 2, 4, 8]
  attention_resolutions: [8, 4]

training:
  optimizer: "adamw"
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
  scheduler: "cosine"
  warmup_steps: 1000
  max_epochs: 800
  batch_size: 24
  precision: "bf16-mixed"

meanflow:
  t_sampler: "logit_normal"
  t_mu: 0.8
  t_sigma: 0.8
  t_clip_min: 0.05
  r_equals_t_prob: 0.5
  loss_norm: 2.0          # Lp exponent (swept in ablation)
  imf_auxiliary_weight: 1.0  # weight of FM auxiliary loss
  adaptive_weight: true
  adaptive_eps: 1.0e-4

ema:
  half_lives: [500, 1000, 2000]  # in M-images
  update_every: 1

data:
  latent_dir: "/path/to/latents"
  normalise: true
  num_workers: 8

logging:
  project: "neuromf"
  sample_every_n_epochs: 25
  n_samples_per_log: 8
```

### 6.3 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P4-T1 | Training starts without error | First 100 steps complete | Training log |
| P4-T2 | Loss decreases over first 50 epochs | `loss[epoch_50] < loss[epoch_1]` |  plot |
| P4-T3 | No NaN in loss or gradients | Zero NaN events in training |  alerts |
| P4-T4 | 1-NFE samples at epoch 50 show vaguely brain-like structure | Visual inspection of mid-sagittal slices |  images |
| P4-T5 | 1-NFE samples at epoch 200 show clear brain anatomy | Visual: identifiable ventricles, cortex, white matter |  images |
| P4-T6 | EMA-selected model produces better samples than online model | FID(EMA) < FID(online) on 500 samples | Compare at epoch 400 |
| P4-T7 | Latent normalisation correctly applied | Generated latents have statistics close to training latents | Histogram comparison |
| P4-T8 | Checkpoints save and load correctly | Resume training from checkpoint; loss continues from previous value | Test resume |

**Phase 4 is PASSED when P4-T1 through P4-T6 are ALL green.**

---

## 7. Phase 5: Evaluation Suite

### 7.1 Objectives

1. Generate $N = 2{,}000$ synthetic volumes at 1-NFE.
2. Compute all metrics: FID, 3D-FID, SSIM, PSNR, SynthSeg Dice.
3. Compare against baselines (if available: MOTFM, Rectified Flow).

### 7.2 Implementation

```bash
# Generate samples
python experiments/cli/generate.py \
    --config configs/evaluate.yaml \
    --checkpoint /path/to/best_ema.ckpt \
    --n_samples 2000 \
    --nfe 1 \
    --output_dir experiments/stage1_healthy/generated

# Evaluate
python experiments/cli/evaluate.py \
    --generated_dir experiments/stage1_healthy/generated \
    --real_dir /path/to/test_set \
    --output_dir experiments/stage1_healthy/metrics
```

### 7.3 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P5-T1 | 2000 samples generated without error | 2000 `.nii.gz` files in output dir | File count |
| P5-T2 | FID (slice-wise, axial) < 50 | Competitive with published 3D brain MRI methods | `metrics.json` |
| P5-T3 | SSIM (mean over volumes) > 0.70 | Generated volumes have structural similarity to real | `metrics.json` |
| P5-T4 | SynthSeg runs on generated volumes | No SynthSeg failures | Log |
| P5-T5 | SynthSeg regional volumes correlate with real | Pearson $r > 0.7$ for hippocampus, ventricles | `metrics.json` |
| P5-T6 | Sampling speed < 2 seconds per volume (A100) | 1-NFE is fast | Timing log |

**Phase 5 is PASSED when P5-T1, P5-T2, P5-T3, P5-T6 are ALL green. P5-T4 and P5-T5 are desirable.**

---

## 8. Phase 6: Ablation Runs

### 8.1 Ablation Matrix

| Ablation | Variable | Values | Seeds | Total runs |
|---|---|---|---|---|
| x-pred vs. u-pred | `prediction_type` | `x`, `u` | 3 | 6 |
| $L_p$ sweep | `loss_norm` | 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 | 2 | 14 |
| NFE steps | `nfe` at inference | 1, 2, 5, 10, 25, 50 | 1 (best model) | 6 (eval only) |
| Rectified Flow baseline | (retrain with FM loss) | — | 2 | 2 |
| **Total training runs** | | | | **22 + 2 = 24** |

### 8.2 Execution Strategy

Use a sweep launcher that generates all configs and submits jobs:

```bash
python experiments/utils/sweep.py \
    --sweep_config configs/ablation_xpred_upred.yaml \
    --base_config configs/train_meanflow.yaml \
    --output_dir experiments/ablations/xpred_vs_upred
```

### 8.3 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P6-T1 | All ablation runs complete without crash | 24 runs with final checkpoints | Check output dirs |
| P6-T2 | x-pred vs. u-pred: FID difference is statistically significant or reported as non-significant | $t$-test computed, p-value reported | `aggregate_results.py` |
| P6-T3 | $L_p$ sweep: at least one $p < 2$ achieves FID ≤ $p=2$ | Or report that $p=2$ is optimal in latent space | Results table |
| P6-T4 | NFE ablation: 1-NFE within 2× FID of 50-NFE | MeanFlow's 1-step is competitive | Results table |
| P6-T5 | Rectified Flow baseline at 1-NFE is substantially worse than MeanFlow at 1-NFE | FID(RF, 1-NFE) >> FID(MF, 1-NFE) | Direct comparison |

**Phase 6 is PASSED when P6-T1 and at least 2 of P6-T2–T5 are green.**

---

## 9. Phase 7: LoRA Fine-Tuning for FCD Joint Synthesis

### 9.1 Objectives

1. Test mask encoding strategy (VAE-encoded vs. downsampled).
2. Implement LoRA injection into the trained MeanFlow UNet.
3. Fine-tune with per-channel $L_p$ loss on FCD data.
4. Evaluate joint image-mask synthesis quality.

### 9.2 Phase Verification Tests

| Test ID | Description | Pass Criterion | How to Check |
|---|---|---|---|
| P7-T1 | Mask VAE reconstruction Dice (Strategy A test) | Record Dice; select strategy based on threshold 0.85 | `mask_vae_test.json` |
| P7-T2 | LoRA parameters inject without error | Model forward pass succeeds with LoRA | Unit test |
| P7-T3 | Only LoRA params have `requires_grad=True` | Count trainable params << total params | Check |
| P7-T4 | Training loss decreases on FCD data | Loss at epoch 200 < epoch 1 | Training log |
| P7-T5 | Generated FLAIR images show lesion-like features | Visual inspection |  |
| P7-T6 | Generated masks have non-trivial overlap with real FCD masks | Mean Dice > 0.3 (data-scarce setting) | `metrics.json` |
| P7-T7 | Per-channel $L_p$ (1.5/2.0) outperforms uniform $L_2$ on mask Dice | Or report null result | Ablation comparison |

**Phase 7 is PASSED when P7-T1 through P7-T5 are ALL green.**

---

## 10. Phase 8: Paper Figures and Tables

### 10.1 Required Figures

| Figure | Content | Source |
|---|---|---|
| Fig. 1 | Method overview: VAE → Latent MeanFlow → Decode | TikZ / draw.io |
| Fig. 2 | Toy toroid results: (a) training loss, (b) generated samples on torus, (c) angular distribution | Phase 2 |
| Fig. 3 | Sample brain MRI: real vs. generated (axial/coronal/sagittal slices) | Phase 5 |
| Fig. 4 | FID vs. NFE curve: MeanFlow vs. Rectified Flow vs. MAISI-v2 | Phase 6 |
| Fig. 5 | $L_p$ sweep: FID and SSIM vs. $p$ | Phase 6 |
| Fig. 6 | x-pred vs. u-pred comparison (FID distribution, sample quality) | Phase 6 |
| Fig. 7 | FCD joint synthesis: generated FLAIR + mask overlays | Phase 7 |
| Fig. 8 | SynthSeg regional volume distributions: real vs. generated | Phase 5 |

### 10.2 Required Tables

| Table | Content |
|---|---|
| Table 1 | Main comparison: FID, SSIM, NFE, time — ours vs. MAISI, MAISI-v2, MOTFM, Med-DDPM |
| Table 2 | x-pred vs. u-pred ablation (FID ± std, SSIM ± std) |
| Table 3 | $L_p$ sweep results |
| Table 4 | Joint synthesis: image quality + mask Dice for different $L_p$ settings |

---

## 11. Agent Context Specification

### 11.1 Papers the Agent Should Have Access To

The following papers should be placed in the agent's context (as PDFs or text) to enable it to correctly implement each component:

| Paper | Why | Priority |
|---|---|---|
| **MeanFlow** (Geng et al., 2025a), arXiv:2505.13447 | Core method. Agent needs the MeanFlow Identity, JVP computation details, Algorithm 1 (training), Algorithm 2 (sampling), and the time sampling strategy. | **Critical** |
| **iMF** (Geng et al., 2025b), arXiv:2512.02012 | The improved objective (Eq. 13). Agent needs the reformulated loss and CFG conditioning strategy. | **Critical** |
| **pMF** (Lu et al., 2026), arXiv:2601.22158 | x-prediction reparameterisation (Eqs. 15–16), manifold hypothesis argument, adaptive weighting formula (Eq. 14). | **Critical** |
| **MAISI** (Guo et al., 2024), arXiv:2409.11169 | VAE architecture, weight loading, preprocessing pipeline, latent statistics. | **Critical** |
| **MAISI-v2** (Zhao et al., 2025), arXiv:2508.05772 | Rectified Flow baseline details, evaluation protocol, UNet architecture specifics. | **High** |
| **SLIM-Diff** (Pascual-González et al., 2026), arXiv:2602.03372 | Per-channel $L_p$ loss formulation, joint image-mask strategy. | **High** |
| **MOTFM** (Yazdani et al., 2025), arXiv:2503.00266 | Evaluation metrics (FID, 3D-FID protocol), BraTS data pipeline, baseline numbers. | **Medium** |
| **LoRA** (Hu et al., 2022) | LoRA implementation details for the fine-tuning stage. | **Medium** |
| **Flow Matching** (Lipman et al., 2023) | Theoretical foundations. Agent needs this to understand the FM → MF progression. | **Medium** |
| **Rectified Flow** (Liu et al., 2023) | Understanding the multi-step predecessor for the baseline. | **Low** |

### 11.2 Code Repositories the Agent Needs

| Repository | Path in `src/external/` | What the Agent Extracts |
|---|---|---|
| `zhuyu-cs/MeanFlow` | `MeanFlow/` | MeanFlow loss computation in `train.py`; time sampling; EMA. The agent wraps the loss logic in `neuromf/losses/meanflow_jvp.py` |
| `HaoyiZhu/MeanFlow-PyTorch` | `MeanFlow-PyTorch/` | Alternative reference; note on FlashAttention incompatibility |
| `Lyy-iiis/pMF` (JAX) | `pMF/` | x-prediction implementation; adaptive weighting; authoritative JAX JVP for correctness verification |
| `Project-MONAI/model-zoo` | `model-zoo/` | MAISI VAE weights and architecture definitions |
| `Project-MONAI/tutorials` | `monai-tutorials/` | MAISI training tutorial notebooks; preprocessing pipeline code |
| `milad1378yz/MOTFM` | `MOTFM/` | Evaluation protocol; FID/3D-FID implementation; BraTS data loading |

### 11.3 SKILL Files the Agent Should Have

The following skill files would significantly accelerate the agent's work. I recommend creating them:

#### SKILL 1: `MeanFlow_JVP_Implementation.md`

**Purpose:** Step-by-step guide to implementing the MeanFlow JVP loss in PyTorch.

**Contents:**
- Exact `torch.func.jvp` usage pattern
- How to handle in-place operations in the UNet
- The stop-gradient pattern (`detach()` on JVP output)
- The time sampling implementation (logit-normal)
- The adaptive weighting formula
- The $r = t$ vs. $r \neq t$ branching logic
- Common failure modes (NaN at small $t$, OOM, FlashAttention)
- A complete, tested reference implementation for a toy 2D case

#### SKILL 2: `MAISI_VAE_Integration.md`

**Purpose:** How to load, use, and wrap the MAISI VAE.

**Contents:**
- Weight download procedure (from `large_files.yml`)
- Architecture class instantiation (MONAI's `AutoencoderKL`)
- Correct preprocessing for brain MRI (intensity normalisation, spacing)
- encode/decode API with shape assertions
- bf16 inference setup
- Known quirks (e.g., input must be multiples of 128)

#### SKILL 3: `3D_UNet_MeanFlow_Adaptation.md`

**Purpose:** How to adapt MAISI's 3D UNet for MeanFlow dual time conditioning.

**Contents:**
- Where to modify the time embedding (which layers, which modules)
- Adding the second time input $(r)$
- Ensuring `torch.func.jvp` compatibility (no in-place ops checklist)
- Output head: x-prediction output (same shape as input)
- Optional: v-head architecture

#### SKILL 4: `Medical_Image_Evaluation.md`

**Purpose:** How to evaluate 3D medical image synthesis.

**Contents:**
- FID computation on 3D volumes (slice-wise protocol)
- 3D-FID via Med3D features
- SynthSeg usage: installation, inference, metric extraction
- SSIM/PSNR computation for 3D volumes
- High-frequency energy analysis
- Statistical testing (Welch's $t$-test, ANOVA, Tukey HSD)

#### SKILL 5: `LoRA_for_UNets.md`

**Purpose:** How to inject LoRA into 3D UNet attention layers.

**Contents:**
- Identifying attention layers in MAISI UNet
- LoRA injection with `peft` or manual implementation
- Freezing base model + unfreezing LoRA params
- Modifying input channels (from 4 to 8 or 5) for joint synthesis
- Per-channel $L_p$ loss configuration

### 11.4 Additional Context Files

| File | Purpose |
|---|---|
| `methodology_expanded.md` | The companion methodology document (this project's §1–13) |
| `technical_guide.md` | This document |
| `configs/train_meanflow.yaml` | Reference training config |
| `configs/base.yaml` | Shared default config |

### 11.5 Environment the Agent Needs

| Resource | Specification |
|---|---|
| **GPU** | 1× A100 40GB (or equivalent). A100 40GB is marginal for batch size 24 at 32³. |
| **Conda env** | `neuromf` with all deps installed (§2.2) |
| **Data** | Pre-downloaded IXI + OASIS-3 NIfTI files (or at least a 50-volume subset for development) |
| **Weights** | MAISI VAE weights downloaded to a known path |
| **Disk** | ~100GB for latents + checkpoints + generated volumes |
| **Network** | Access to PyPI, GitHub (for dependency installation) |
| **Results folder** | A designated output directory the agent writes all results to |
| **** | A  project `neuromf` configured with API key (or local tensorboard fallback) |

### 11.6 Suggested Agent Workflow

```
Agent receives: this guide + methodology doc + all SKILL files + access to src/external/

Phase 0: Read SKILL 2 → implement maisi_vae.py → run P0 tests
Phase 1: implement preprocessing + encoding → run P1 tests
Phase 2: Read SKILL 1 → implement toroid dataset + MeanFlow loss → run P2 tests
Phase 3: Read SKILL 1 + SKILL 3 → implement full pipeline → run P3 tests
Phase 4: Read training config → train → monitor → run P4 tests
Phase 5: Read SKILL 4 → implement metrics → run P5 tests
Phase 6: Launch ablation sweep → run P6 tests
Phase 7: Read SKILL 5 → implement LoRA → run P7 tests
Phase 8: Generate figures and tables
```

Each phase gate prevents the agent from proceeding to the next phase if critical tests fail, ensuring errors are caught early.

---

## Appendix A: Dependency Versions

```
python==3.11
torch>=2.1.0
torchvision>=0.16.0
monai>=1.3.0
pytorch-lightning>=2.0.0
omegaconf>=2.3.0
einops>=0.7.0
nibabel>=5.0.0
>=0.16.0
scikit-image>=0.21.0
torch-fidelity>=0.3.0
lpips>=0.1.4
scipy>=1.11.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
click>=8.1.0
```

## Appendix B: GPU Memory Estimation

| Component | Memory (fp32) | Memory (bf16) |
|---|---|---|
| 3D UNet (~40M params) | ~160 MB | ~80 MB |
| Latent batch (B=24, 4×32³) | ~384 MB | ~192 MB |
| UNet activations (forward) | ~4 GB | ~2 GB |
| JVP primal tape (≈ 1× forward) | ~4 GB | ~2 GB |
| Gradients | ~160 MB | ~80 MB |
| Optimizer states (AdamW, 2× params) | ~320 MB | ~320 MB |
| **Total estimated** | **~9 GB** | **~5 GB** |
| **With headroom (1.5×)** | **~14 GB** | **~7.5 GB** |

Comfortably fits on A100 40GB. Batch size can be increased to 48–64 on A100 40GB.

## Appendix C: Risk Register

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| MAISI UNet incompatible with `torch.func.jvp` | Blocks Phase 3 | Medium | Refactor in-place ops; finite-difference fallback |
| MAISI VAE poorly reconstructs FLAIR | Blocks Phase 0 | Low | VAE was trained on MRI including FLAIR; fine-tune if needed |
| MeanFlow fails to converge on 3D latents | Blocks Phase 4 | Low | Toy experiment (Phase 2) validates pipeline first |
| $L_p$ effect does not transfer to latent space | Weakens Contribution 2 | Medium | Report as null result; still publishable |
| x-pred vs. u-pred shows no difference | Weakens Contribution 3 | Medium | Report as insight into VAE compression; still publishable |
| FCD data too scarce for LoRA | Weakens Contribution 4 | Medium | Reduce LoRA rank; increase augmentation |
| SynthSeg fails on synthetic volumes | Limits evaluation | Low | Use alternative: FreeSurfer or manual QC |
