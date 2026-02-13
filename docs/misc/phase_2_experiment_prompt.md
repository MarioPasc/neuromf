# Phase 2 Formal Experiment: MeanFlow Validation on Toroidal Manifold

**Purpose:** This prompt defines a formal experiment suite for the Phase 2 toy toroid that (a) fully validates the MeanFlow implementation, (b) produces publication-quality figures and tables for the paper, and (c) ensures all reusable components live in `src/neuromf/` so downstream phases only swap data and architecture.

**Results directory:** `/media/mpascual/Sandisk2TB/research/neuromf/results/phase_2/toroid`

---

## 0. Pre-Requisites

Before implementing, read and internalise:
1. `docs/main/methodology_expanded.md` §1–§4 (MeanFlow theory, x-prediction, Lp loss)
2. `docs/papers/meanflow_2025/pytorch_code_exploration.md` (PyTorch reference patterns)
3. `docs/papers/pmf_2026/code_exploration.md` (x-prediction, dimensionality scaling toy experiment)
4. `src/external/MeanFlow-PyTorch/meanflow.py` (primary PyTorch reference — study `__call__`, `sample_tr`, `solver_step`)
5. `src/external/pmf/pmf.py` (x-prediction algorithm, Algorithm 1)
6. The existing Phase 2 code in `src/neuromf/` (losses, sampling, data, utils modules)

**CRITICAL:** All experiment-agnostic code (losses, sampling, metrics, time sampling) MUST live in `src/neuromf/`. Only experiment-specific orchestration, configs, and plotting go in `experiments/`.

---

## 1. Scientific Goals

This experiment must answer five questions, each corresponding to a subsection in the final paper figure (Fig. 2):

| ID | Question | Why It Matters for NeuroMF |
|----|----------|---------------------------|
| Q1 | Does MeanFlow converge on a known manifold? | Basic algorithmic correctness |
| Q2 | Does x-prediction outperform u-prediction as ambient dimension D increases? | Justifies x-prediction for 4×48³ latent space (D=442,368) |
| Q3 | How does the Lp norm affect convergence and sample quality? | Informs the Lp sweep in Phase 6 |
| Q4 | What is the quality-vs-NFE tradeoff? | Justifies 1-NFE as the operating point |
| Q5 | How does the data_proportion (fraction of r=t samples) affect the iMF dual loss? | Validates the iMF combined loss mechanism |

---

## 2. Data Distribution: Flat Torus with Dimensionality Scaling

### 2.1 Base Distribution (D=4)

The flat torus $\mathbb{T}^2$ isometrically embedded in $\mathbb{R}^4$:

$$\phi(\theta_1, \theta_2) = (\cos\theta_1,\; \sin\theta_1,\; \cos\theta_2,\; \sin\theta_2), \quad \theta_1, \theta_2 \sim \mathrm{Uniform}[0, 2\pi)$$

**NOTE:** We use the **unnormalised** embedding (no $1/\sqrt{2}$ factor), yielding per-coordinate $\sigma \approx 0.707$. This is closer to the VAE latent per-channel std ($\sigma_c \approx 1.0$, from Phase 0 report) than the normalised embedding ($\sigma \approx 0.5$). The torus is a harder test than Gaussian latents because it has compact support and is far from the $\mathcal{N}(0,I)$ prior.

### 2.2 Dimensionality Scaling (D > 4)

Following pMF (Lu et al., 2026, §5), embed the 2D torus into $\mathbb{R}^D$ using a fixed, random $D \times 4$ column-orthogonal projection matrix $\mathbf{P}$:

$$\mathbf{x} = \mathbf{P} \cdot \phi(\theta_1, \theta_2) \in \mathbb{R}^D$$

The projection preserves pairwise distances (up to a scale) and maintains the 2-dimensional manifold structure in the $D$-dimensional ambient space. This directly parallels the manifold hypothesis: brain MRI latents lie on a low-dimensional manifold (dim $\ll$ 442K) embedded in the 442K-dimensional latent space.

**Fixed random seed** for $\mathbf{P}$: use seed=42 for reproducibility across all runs.

### 2.3 Implementation Location

```
src/neuromf/data/toroid_dataset.py  — ToroidDataset class with D parameter
```

Modify the existing `ToroidDataset` to accept a `ambient_dim: int = 4` parameter. When `ambient_dim > 4`, generate the orthogonal projection matrix $\mathbf{P}$ using `torch.linalg.qr` on a random matrix (seeded). The `mode` parameter is always `"r4"` for this experiment (no volumetric mode needed).

---

## 3. Ablation Matrix

### 3.1 Ablation A: Convergence Baseline (Q1)

Train a single model to convergence to establish baseline metrics.

| Parameter | Value |
|-----------|-------|
| Ambient dim D | 4 |
| Prediction type | u-prediction |
| Lp norm p | 2.0 |
| Epochs | 500 |
| data_proportion | 0.75 |
| Architecture | 7-layer MLP, 256 hidden, ReLU (inplace=False) |
| Learning rate | 1e-3 (Adam, β₁=0.9, β₂=0.95) |
| Batch size | 512 |
| n_samples | 50,000 |
| Time sampling | logit-normal(μ=-0.4, σ=1.0) |
| Adaptive weighting | On (norm_eps=0.01) |
| EMA decay | 0.999 |
| Evaluation every | 50 epochs |

Collect at each evaluation checkpoint:
- Training loss (running average over last 10 epochs)
- 1-NFE sample quality: mean torus distance, pair-wise norm errors
- Multi-step (10 steps) sample quality: same metrics
- 1000 generated samples for MMD computation
- KS test p-values for angular uniformity

### 3.2 Ablation B: x-Prediction vs u-Prediction × Dimensionality (Q2) — KEY RESULT

This is the most important ablation for the paper. It directly replicates pMF §5 and justifies our choice of x-prediction for the high-dimensional MRI latent space.

| Parameter | Values |
|-----------|--------|
| Ambient dim D | {4, 16, 64, 256} |
| Prediction type | {u-prediction, x-prediction} |
| Total runs | 4 × 2 = **8 runs** |
| Epochs | 500 each |
| All other params | Same as Ablation A |

**x-prediction implementation:** The model outputs $\hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t)$ directly. The average velocity is derived as:

$$\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{\mathbf{z}_t - \hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t)}{t}$$

The JVP closure wraps this conversion:

```python
def u_fn(z, r, t):
    x_pred = net(z, r, t)          # network outputs x-prediction
    return (z - x_pred) / t        # convert to u
```

**u-prediction implementation:** The model outputs $\mathbf{u}_\theta(\mathbf{z}_t, r, t)$ directly. The JVP closure is simply:

```python
def u_fn(z, r, t):
    return net(z, r, t)            # network outputs u directly
```

For both, the loss target is the conditional velocity $\mathbf{v}_c = \boldsymbol{\epsilon} - \mathbf{x}$ (v-loss), and the compound prediction $V_\theta = \mathbf{u}_\theta + (t - r) \cdot \text{sg}[\text{JVP}]$ is compared to $\mathbf{v}_c$.

### 3.3 Ablation C: Lp Norm Sweep (Q3)

| Parameter | Values |
|-----------|--------|
| Lp norm p | {1.0, 1.5, 2.0, 3.0} |
| Ambient dim D | 4 |
| Prediction type | u-prediction |
| Total runs | **4 runs** |
| Epochs | 500 each |
| All other params | Same as Ablation A |

### 3.4 Ablation D: NFE Sweep (Q4)

Train **one model** (the Ablation A baseline at epoch 500), then evaluate with varying NFE at inference:

| NFE | Method |
|-----|--------|
| 1 | 1-NFE MeanFlow: $\hat{\mathbf{x}} = \boldsymbol{\epsilon} - \mathbf{u}_\theta(\boldsymbol{\epsilon}, 0, 1)$ |
| 2 | Euler: schedule [1.0, 0.5, 0.0] |
| 5 | Euler: 5 uniform steps |
| 10 | Euler: 10 uniform steps |
| 25 | Euler: 25 uniform steps |
| 50 | Euler: 50 uniform steps |

Generate 2000 samples per NFE setting and compute all metrics.

### 3.5 Ablation E: data_proportion Sweep (Q5)

| Parameter | Values |
|-----------|--------|
| data_proportion | {0.0, 0.25, 0.5, 0.75, 1.0} |
| Ambient dim D | 4 |
| Prediction type | u-prediction |
| Total runs | **5 runs** |
| Epochs | 500 each |

When `data_proportion=1.0`, all samples have `r=t` and the MeanFlow JVP term vanishes — this is pure Flow Matching. When `data_proportion=0.0`, all samples have `r≠t` — this is pure MeanFlow. This ablation validates the iMF combined loss mechanism.

### 3.6 Total Compute Budget

| Ablation | Runs | Epochs/run | Total epochs |
|----------|------|-----------|-------------|
| A (baseline) | 1 | 500 | 500 |
| B (x-pred × dim) | 8 | 500 | 4,000 |
| C (Lp sweep) | 4 | 500 | 2,000 |
| D (NFE sweep) | 0 (inference only) | — | 0 |
| E (data_proportion) | 5 | 500 | 2,500 |
| **Total** | **18 training runs** | | **9,000 epochs** |

At ~0.3s/epoch on CPU (MLP on R⁴ with 50K samples, batch 512), this is ~45 minutes total. Feasible on the local machine.

---

## 4. Metrics

All metrics must be implemented as reusable functions in `src/neuromf/metrics/` for downstream phases.

### 4.1 Geometric Fidelity (toroid-specific, in `experiments/toy_toroid/`)

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| Mean torus distance | $\delta = \mathbb{E}\left[\left\|\|\hat{\mathbf{z}}\|_2 - 1\right\|\right]$ for D=4; generalised for D>4 via projection back to R⁴ | How close samples are to the manifold surface |
| Pair-wise norm error | $\|\hat{z}_1^2 + \hat{z}_2^2 - 1\|$ and $\|\hat{z}_3^2 + \hat{z}_4^2 - 1\|$ (in original 4D coordinates after projection) | Structural constraint violation |
| KS uniformity test | `scipy.stats.kstest(θ_i, 'uniform', args=(0, 2π))` for i∈{1,2} | Whether angular distribution matches the true uniform |

### 4.2 Distributional Metrics (reusable, in `src/neuromf/metrics/`)

| Metric | Implementation | What It Measures |
|--------|---------------|-----------------|
| **MMD (RBF kernel)** | $\text{MMD}^2 = \mathbb{E}[k(x,x')] + \mathbb{E}[k(y,y')] - 2\mathbb{E}[k(x,y)]$ with $k(x,y) = \exp(-\|x-y\|^2 / 2\sigma^2)$, median heuristic for $\sigma$ | Two-sample test between generated and real distributions |
| **Coverage** | Fraction of real samples whose nearest neighbour is a generated sample (within threshold) | Mode dropping detection |
| **Density** | Ratio of generated samples falling within the support of the real distribution | Hallucination detection |

**MMD implementation note:** Use multiple kernel bandwidths $\sigma \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ and report the maximum MMD (conservative). This is standard practice (Gretton et al., 2012).

### 4.3 Training Dynamics (reusable, in `src/neuromf/utils/`)

| Metric | What It Measures |
|--------|-----------------|
| Loss curve (per-epoch) | Convergence speed and stability |
| Gradient norm (per-epoch) | Training stability |
| EMA vs non-EMA loss gap | Whether EMA is helping |

---

## 5. Figures

All figures must be publication-quality: serif font (Times New Roman or Computer Modern), 10pt labels, 300 DPI, PDF+PNG output. Use `matplotlib` with `rcParams` configured via a shared `src/neuromf/utils/plot_style.py`.

### Fig 2a — Training Loss Convergence (Ablation A)

- **Type:** Line plot
- **X-axis:** Epoch
- **Y-axis:** MeanFlow loss (log scale)
- **Content:** Training loss curve for the baseline run, with EMA evaluation loss overlaid as a dashed line
- **Annotation:** Mark the epoch where P2-T4 criterion is satisfied

### Fig 2b — Dimensionality Scaling: x-Prediction vs u-Prediction (Ablation B) — KEY FIGURE

- **Type:** 2×4 grid of scatter plots (following pMF Fig. 2 layout)
- **Rows:** x-prediction (top), u-prediction (bottom)
- **Columns:** D=4, D=16, D=64, D=256
- **Content:** 1-NFE generated samples projected onto the first two principal components of the true data distribution. Overlay the ground-truth torus projection as a dashed red circle/curve.
- **Caption element:** FID-like metric (MMD) annotated in each panel corner

### Fig 2c — NFE vs Quality Tradeoff (Ablation D)

- **Type:** Dual-axis line plot
- **X-axis:** NFE (log scale: 1, 2, 5, 10, 25, 50)
- **Left Y-axis:** Mean torus distance (lower is better)
- **Right Y-axis:** MMD (lower is better)
- **Content:** Two lines showing how geometric fidelity and distributional match improve with more NFE steps
- **Key insight to highlight:** The gap between 1-NFE and 50-NFE is modest, justifying 1-NFE for MRI generation

### Fig 2d — Angular Distribution Recovery (Ablation A, baseline)

- **Type:** 2×2 subplot grid
- **Content:** θ₁ and θ₂ histograms for 1-NFE (top row) and 10-step (bottom row), each with Uniform(0, 2π) reference line
- **Annotation:** KS test p-value in each panel

### Fig 2e — Lp Norm Impact (Ablation C)

- **Type:** Grouped bar chart
- **X-axis:** p value (1.0, 1.5, 2.0, 3.0)
- **Y-axis (bars):** Final MMD (left group), final torus distance (right group)
- **Overlaid line:** Final training loss

### Fig 2f — data_proportion Effect (Ablation E)

- **Type:** Line plot with shaded confidence bands
- **X-axis:** data_proportion (0.0, 0.25, 0.5, 0.75, 1.0)
- **Y-axis:** Final 1-NFE MMD
- **Secondary line:** Multi-step (10) MMD for comparison
- **Key insight:** Shows optimal balance between Flow Matching (r=t, data_proportion=1.0) and MeanFlow (r≠t, data_proportion=0.0) terms

---

## 6. Tables

### Table S1 — Full Ablation Results (Supplementary)

| Ablation | D | Pred | p | data_prop | Final Loss | Torus Dist (1-NFE) | Torus Dist (10-step) | MMD (1-NFE) | KS-θ₁ p | KS-θ₂ p | Coverage | Density |
|----------|---|------|---|-----------|------------|-------------------|--------------------|-------------|---------|---------|----------|---------|
| A | 4 | u | 2.0 | 0.75 | ... | ... | ... | ... | ... | ... | ... | ... |
| B-1 | 4 | x | 2.0 | 0.75 | ... | ... | ... | ... | ... | ... | ... | ... |
| ... | | | | | | | | | | | | |

### Table 1 — Dimensionality Scaling Summary (Main paper)

| D | x-pred MMD (1-NFE) | u-pred MMD (1-NFE) | x-pred Loss | u-pred Loss | x-pred Torus Dist | u-pred Torus Dist |
|---|-------|-------|------|------|------|------|
| 4 | | | | | | |
| 16 | | | | | | |
| 64 | | | | | | |
| 256 | | | | | | |

---

## 7. HTML Report

Generate a self-contained HTML report at:
```
/media/mpascual/Sandisk2TB/research/neuromf/results/phase_2/toroid/report.html
```

### Report Structure

```
# Phase 2: MeanFlow Validation on Toroidal Manifold

## 1. Executive Summary
- One paragraph: does the MeanFlow implementation pass all tests?
- Table: pass/fail status for P2-T1 through P2-T8

## 2. Experimental Setup
- Data distribution description with equations
- Model architecture summary
- Hyperparameter table
- Total compute time

## 3. Ablation A: Convergence Baseline
- Fig 2a (loss curve) embedded
- Fig 2d (angular distributions) embedded
- Quantitative results table
- Interpretation

## 4. Ablation B: Dimensionality Scaling (Key Result)
- Fig 2b (2×4 grid) embedded
- Table 1 (dimensionality summary) embedded
- Statistical analysis: at what D does u-prediction start degrading?
- Connection to MRI latent space dimensionality

## 5. Ablation C: Lp Norm Impact
- Fig 2e (bar chart) embedded
- Discussion: is p=2 optimal, or does p<2 help on the torus?
- Implications for Phase 6 Lp sweep on MRI

## 6. Ablation D: NFE vs Quality
- Fig 2c (NFE tradeoff) embedded
- Quantitative results per NFE
- Justification for 1-NFE as default

## 7. Ablation E: data_proportion
- Fig 2f (proportion effect) embedded
- Optimal balance between FM and MF terms

## 8. Conclusions & Implications for Phase 3+
- Summary of validated components
- Hyperparameter recommendations carrying forward
- Known limitations of the toy experiment
```

---

## 8. Code Organisation

### 8.1 Reusable modules in `src/neuromf/` (these persist to downstream phases)

| Module | Key Functions/Classes | Used in Phases |
|--------|----------------------|----------------|
| `src/neuromf/losses/meanflow_jvp.py` | `compute_meanflow_jvp()`, `meanflow_loss()` | 2, 3, 4 |
| `src/neuromf/losses/lp_loss.py` | `lp_loss()` | 2, 3, 4, 6 |
| `src/neuromf/losses/combined_loss.py` | `imf_combined_loss()` | 3, 4 |
| `src/neuromf/sampling/one_step.py` | `sample_one_step()` | 2, 4, 5 |
| `src/neuromf/sampling/multi_step.py` | `sample_euler()` | 2, 4, 5, 6 |
| `src/neuromf/utils/time_sampler.py` | `sample_logit_normal()`, `sample_tr()` | 2, 3, 4 |
| `src/neuromf/utils/ema.py` | `EMAModel` class | 2, 4 |
| `src/neuromf/utils/plot_style.py` | `setup_publication_style()` | 2, 5, 6, 8 |
| `src/neuromf/metrics/mmd.py` | `compute_mmd()` | 2, 5 |
| `src/neuromf/metrics/coverage_density.py` | `compute_coverage()`, `compute_density()` | 2, 5 |
| `src/neuromf/data/toroid_dataset.py` | `ToroidDataset`, `ToroidConfig` | 2 only |

### 8.2 Experiment-specific code in `experiments/`

```
experiments/
├── cli/
│   └── run_toy_toroid.py          # CLI entry point with subcommands
├── toy_toroid/
│   ├── configs/                    # YAML configs per ablation
│   │   ├── base.yaml              # Shared defaults
│   │   ├── ablation_a.yaml        # Convergence baseline
│   │   ├── ablation_b.yaml        # x-pred vs u-pred × dim
│   │   ├── ablation_c.yaml        # Lp sweep
│   │   ├── ablation_d.yaml        # NFE sweep (inference only)
│   │   └── ablation_e.yaml        # data_proportion sweep
│   ├── train.py                   # Training loop (shared across ablations)
│   ├── evaluate.py                # Evaluation: metrics computation
│   ├── sweep.py                   # Sweep runner: iterates over ablation configs
│   ├── report.py                  # HTML report generator
│   ├── figures.py                 # Figure generation (calls src/neuromf/utils/plot_style)
│   └── models/
│       └── toy_mlp.py             # 7-layer MLP for toy experiment
```

### 8.3 CLI Interface

```bash
# Run a single training experiment
python experiments/cli/run_toy_toroid.py train --config experiments/toy_toroid/configs/ablation_a.yaml

# Run the full ablation sweep (all 18 runs)
python experiments/cli/run_toy_toroid.py sweep --ablations A B C D E

# Evaluate a trained model with different NFE settings
python experiments/cli/run_toy_toroid.py evaluate --checkpoint <path> --nfe 1 2 5 10 25 50

# Generate all figures from results
python experiments/cli/run_toy_toroid.py figures --results-dir /media/mpascual/Sandisk2TB/research/neuromf/results/phase_2/toroid

# Generate the HTML report
python experiments/cli/run_toy_toroid.py report --results-dir /media/mpascual/Sandisk2TB/research/neuromf/results/phase_2/toroid
```

### 8.4 Results Directory Structure

```
/media/mpascual/Sandisk2TB/research/neuromf/results/phase_2/toroid/
├── ablation_a/
│   └── baseline_D4_u-pred_p2.0_dp0.75/
│       ├── checkpoints/
│       │   ├── model_epoch_050.pt
│       │   ├── model_epoch_100.pt
│       │   ├── ...
│       │   └── model_epoch_500.pt
│       ├── metrics/
│       │   ├── epoch_050.json
│       │   ├── ...
│       │   └── epoch_500.json
│       └── training_log.json        # Per-epoch loss, grad norm
├── ablation_b/
│   ├── D4_x-pred/
│   ├── D4_u-pred/
│   ├── D16_x-pred/
│   ├── D16_u-pred/
│   ├── D64_x-pred/
│   ├── D64_u-pred/
│   ├── D256_x-pred/
│   └── D256_u-pred/
├── ablation_c/
│   ├── p1.0/
│   ├── p1.5/
│   ├── p2.0/
│   └── p3.0/
├── ablation_d/
│   └── nfe_sweep.json               # Metrics for each NFE setting
├── ablation_e/
│   ├── dp0.00/
│   ├── dp0.25/
│   ├── dp0.50/
│   ├── dp0.75/
│   └── dp1.00/
├── figures/
│   ├── fig2a_loss_convergence.pdf
│   ├── fig2a_loss_convergence.png
│   ├── fig2b_dim_scaling.pdf
│   ├── fig2b_dim_scaling.png
│   ├── fig2c_nfe_tradeoff.pdf
│   ├── fig2c_nfe_tradeoff.png
│   ├── fig2d_angular_distributions.pdf
│   ├── fig2d_angular_distributions.png
│   ├── fig2e_lp_impact.pdf
│   ├── fig2e_lp_impact.png
│   ├── fig2f_data_proportion.pdf
│   └── fig2f_data_proportion.png
├── tables/
│   ├── table_s1_full_results.csv
│   └── table_1_dim_scaling.csv
├── summary_metrics.json              # Aggregated results across all ablations
└── report.html
```

---

## 9. Implementation Notes

### 9.1 Toy MLP Architecture

```python
# experiments/toy_toroid/models/toy_mlp.py
class ToyMLP(nn.Module):
    """7-layer ReLU MLP for toy MeanFlow experiments.

    Input: concatenation of (z_t, r, t) -> dim = data_dim + 2
    Output: u-prediction or x-prediction of shape (B, data_dim)

    Args:
        data_dim: Ambient dimension D of the torus embedding.
        hidden_dim: Hidden layer width.
        n_layers: Number of hidden layers.
        prediction_type: "u" for u-prediction, "x" for x-prediction.
    """
```

- Use `inplace=False` for all ReLUs (required for `torch.func.jvp`)
- Input: `torch.cat([z_t, r.expand(B, 1), t.expand(B, 1)], dim=-1)` where r, t are scalars broadcast to batch
- Hidden layers: `Linear(hidden_dim, hidden_dim)` + `ReLU(inplace=False)`
- Output layer: `Linear(hidden_dim, data_dim)` (no activation)
- Initialisation: Xavier uniform (default PyTorch)

### 9.2 Time Convention

The model receives `(z_t, r, t)` as three separate arguments. Inside the JVP closure:

```python
def u_fn(z, r, t):
    if prediction_type == "x":
        x_pred = model(z, r, t)
        return (z - x_pred) / t.clamp(min=0.05)  # clamp avoids singularity at t=0
    else:
        return model(z, r, t)
```

The JVP is computed with tangents `(v_tilde, 0, 1)` where:
- `v_tilde = u_fn(z_t, t, t)` is the instantaneous velocity estimate (boundary condition: u at r=t equals v)
- `0` is the tangent for r (held fixed)
- `1` is the tangent for t (differentiating w.r.t. time)

### 9.3 Torus Distance for D > 4

When D > 4, samples $\hat{\mathbf{z}} \in \mathbb{R}^D$ must be projected back to $\mathbb{R}^4$ before computing torus metrics:

$$\hat{\mathbf{z}}_{4} = \mathbf{P}^{\top} \hat{\mathbf{z}}$$

where $\mathbf{P}$ is the $D \times 4$ projection matrix. Then apply the standard torus distance checks on $\hat{\mathbf{z}}_4$.

### 9.4 MMD Implementation

```python
# src/neuromf/metrics/mmd.py
def compute_mmd(x: torch.Tensor, y: torch.Tensor,
                bandwidths: list[float] | None = None) -> float:
    """Compute MMD² with RBF kernel using multiple bandwidths.

    Uses median heuristic if bandwidths not provided.
    Reports maximum MMD² over all bandwidths (conservative).
    
    Reference: Gretton et al. (2012), JMLR.
    """
```

### 9.5 Reproducibility

- Set `torch.manual_seed(seed)`, `np.random.seed(seed)`, `random.seed(seed)` at the start of each run
- Use `seed=42` as the default; ablation runs use seeds `{42, 43, 44}` for 3 repetitions if time permits (otherwise single seed)
- Save the full config YAML alongside each checkpoint
- Log the git commit hash and environment info in each run's metadata

---

## 10. Acceptance Criteria

The experiment is COMPLETE when:

1. **All 18 training runs** have finished without errors
2. **All 6 figures** are generated as PDF+PNG at 300 DPI
3. **Both tables** are generated as CSV
4. **The HTML report** is generated and all sections are populated with real data
5. **Key scientific findings are confirmed:**
   - (Q1) Loss decreases monotonically after warmup; 1-NFE torus distance < 0.1
   - (Q2) At D≥64, x-prediction achieves lower MMD than u-prediction (the main result)
   - (Q3) p=2.0 is competitive; p=1.0 may show different convergence dynamics
   - (Q4) 1-NFE achieves ≥70% of 50-NFE quality (measured by MMD)
   - (Q5) data_proportion ∈ [0.5, 0.75] outperforms the extremes (0.0 or 1.0)
6. **No hardcoded paths** in `src/neuromf/` — all paths come from configs or CLI args
7. **All reusable code** has docstrings, type hints, and passes existing test suite

If any scientific finding is NOT confirmed (e.g., x-prediction does NOT beat u-prediction at high D), that is still a valid result — document it honestly and analyse why. Do NOT cherry-pick or tune hyperparameters to force a desired outcome.

---

## 11. After Completion: What Carries Forward

After this experiment, the following components are validated and frozen for Phase 3+:

| Component | Status | Next Phase |
|-----------|--------|------------|
| `meanflow_jvp.py` | Validated on MLP | Phase 3: test on 3D UNet |
| `lp_loss.py` | Validated for p∈{1,1.5,2,3} | Phase 4: use on MRI latents |
| `one_step.py` / `multi_step.py` | Validated geometrically | Phase 5: generate MRI volumes |
| `time_sampler.py` | Validated: logit-normal + data_proportion | Phase 4: same sampler |
| `ema.py` | Validated on toy | Phase 4: same EMA |
| `mmd.py` | Validated on known distribution | Phase 5: use for evaluation |
| `plot_style.py` | Publication style established | Phase 8: all paper figures |

The ONLY things that change in Phase 3+ are:
1. Data: MRI latents instead of torus samples
2. Architecture: MAISI 3D UNet instead of toy MLP
3. Scale: 4×48³ instead of 4 or 256 dimensions
