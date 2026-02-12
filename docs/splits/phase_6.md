# Phase 6: Ablation Runs

**Depends on:** Phase 5 (gate must be OPEN)
**Modules touched:** `src/neuromf/models/`, `experiments/cli/`, `experiments/utils/`, `experiments/ablations/`, `configs/`
**Estimated effort:** 3–5 sessions (includes 24 training runs)

---

## 1. Objective

Run the four ablation studies that form the core experimental contributions of the paper: (1) x-prediction vs. u-prediction, (2) Lp exponent sweep, (3) NFE steps comparison, and (4) Rectified Flow baseline. These ablations produce Tables 2–3 and Figures 4–6 of the paper.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §9.1 Ablation 1: x-Prediction vs. u-Prediction

| Setting | Description |
|---|---|
| **x-pred** | Network outputs $\hat{\mathbf{z}}_0$; $\mathbf{u}_\theta = (\mathbf{z}_t - \hat{\mathbf{z}}_0) / t$ |
| **u-pred** | Network directly outputs $\mathbf{u}_\theta$ |
| **Fixed variables** | Same architecture, same learning rate, same batch size, same training data, same $t$-sampler, same EMA schedule |
| **Metric** | FID (slice-wise, axial/coronal/sagittal), 3D-FID (Med3D features), SSIM, PSNR |
| **Trials** | 3 seeds per setting |
| **Statistical test** | Two-sample $t$-test (Welch's) on FID across seeds; report mean ± std |

**Hypothesis:** x-pred ≤ u-pred in FID (lower is better), with diminished gap relative to pMF's pixel-space result.

### §9.2 Ablation 2: $L_p$ Exponent Sweep

| $p$ | 1.0 | 1.25 | 1.5 | 1.75 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|---|
| Training | Full | Full | Full | Full | Full | Full | Full |

**Evaluation:** FID, SSIM, PSNR at convergence (500 epochs). Report the Pareto frontier across FID-SSIM.

**Statistical protocol:** 2 seeds per $p$ value (14 runs total). ANOVA followed by Tukey's HSD for pairwise comparisons.

### §9.3 Ablation 3: Number of Sampling Steps

Evaluate the trained MeanFlow model at $K \in \{1, 2, 5, 10, 25, 50\}$ NFE using Euler sampling. Compare against a Rectified Flow baseline at the same steps.

### §4 Manifold-Theoretic Analysis (Eqs. 31–33)

**UNet bottleneck dimensionality:**
$$
d_{\text{bottleneck}} = C_{\text{deep}} \times H_{\text{deep}} \times W_{\text{deep}} \times D_{\text{deep}}
\tag{31}
$$

$$
\frac{d_{\text{input}}}{d_{\text{bottleneck}}} = \frac{131{,}072}{32{,}768} = 4
\tag{32}
$$

**Intrinsic dimensionality bounds:**
$$
d_{\text{int}}^{z,x} \leq d_{\text{int}}^x, \quad d_{\text{int}}^{z,u} \leq d_{\text{int}}^u
\tag{33}
$$

## 3. External Code to Leverage

No new external code. Uses Phase 3–5 components. Rectified Flow baseline is implemented fresh.

## 4. Implementation Specification

### `src/neuromf/models/rectflow_baseline.py`
- **Purpose:** Rectified Flow baseline for comparison (standard FM loss, Euler sampling).
- **Key class:**
  ```python
  class RectFlowBaseline(pl.LightningModule):
      def __init__(self, config: OmegaConf) -> None: ...
      def training_step(self, batch, batch_idx) -> torch.Tensor: ...
  ```

### `experiments/cli/run_ablation.py`
- **Purpose:** Launch ablation runs with config overrides.
- **Usage:**
  ```bash
  python experiments/cli/run_ablation.py --sweep_config configs/ablation_xpred_upred.yaml
  ```

### `experiments/utils/sweep.py`
- **Purpose:** Hyperparameter sweep launcher that generates configs and manages runs.
- **Key functions:**
  ```python
  def generate_sweep_configs(sweep_config: Path, base_config: Path) -> list[Path]: ...
  def launch_sweep(configs: list[Path], output_dir: Path, max_parallel: int = 1) -> None: ...
  ```

### `experiments/utils/aggregate_results.py`
- **Purpose:** Aggregate results from multiple ablation runs into tables.
- **Key functions:**
  ```python
  def aggregate_ablation_results(results_dir: Path) -> pd.DataFrame: ...
  def compute_statistical_tests(df: pd.DataFrame, test_type: str) -> dict: ...
  ```

### Config files
- `configs/ablation_xpred_upred.yaml` — sweep over prediction_type: [x, u], seeds: [0, 1, 2]
- `configs/ablation_lp_sweep.yaml` — sweep over loss_norm: [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], seeds: [0, 1]
- `configs/ablation_nfe_steps.yaml` — evaluation-only: nfe: [1, 2, 5, 10, 25, 50]
- `configs/train_rectflow_baseline.yaml` — Rectified Flow training config

### Ablation Matrix (from tech guide §8.1)

| Ablation | Variable | Values | Seeds | Total runs |
|---|---|---|---|---|
| x-pred vs. u-pred | `prediction_type` | `x`, `u` | 3 | 6 |
| $L_p$ sweep | `loss_norm` | 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 | 2 | 14 |
| NFE steps | `nfe` at inference | 1, 2, 5, 10, 25, 50 | 1 (best model) | 6 (eval only) |
| Rectified Flow baseline | (retrain with FM loss) | — | 2 | 2 |
| **Total training runs** | | | | **22 + 2 = 24** |

## 5. Data and I/O

- **Input:** Pre-computed latents (from Phase 1), base model checkpoint (from Phase 4)
- **Output:**
  - `experiments/ablations/xpred_vs_upred/` — 6 run directories with checkpoints and metrics
  - `experiments/ablations/lp_sweep/` — 14 run directories
  - `experiments/ablations/nfe_steps/` — evaluation results for 6 NFE values
  - `experiments/ablations/rectflow_baseline/` — 2 baseline runs
  - Aggregated results tables (CSV/JSON)

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P6-T1 | All ablation runs complete without crash | 24 runs with final checkpoints | CRITICAL | Check output dirs for all expected checkpoints |
| P6-T2 | x-pred vs. u-pred: FID difference reported with statistical test | $t$-test computed, p-value reported | PASS/REPORT | Run `aggregate_results.py` |
| P6-T3 | $L_p$ sweep: results for all 7 values | Or report that $p=2$ is optimal in latent space | PASS/REPORT | Results table |
| P6-T4 | NFE ablation: 1-NFE within 2× FID of 50-NFE | MeanFlow's 1-step is competitive | PASS/REPORT | Compare FID values |
| P6-T5 | Rectified Flow baseline at 1-NFE is substantially worse than MeanFlow at 1-NFE | FID(RF, 1-NFE) >> FID(MF, 1-NFE) | PASS/REPORT | Direct comparison |

**Phase 6 is PASSED when P6-T1 is green and at least 2 of P6-T2–T5 produce reportable results.**

## 7. Expected Outputs

- `src/neuromf/models/rectflow_baseline.py`
- `experiments/cli/run_ablation.py`
- `experiments/utils/sweep.py`
- `experiments/utils/aggregate_results.py`
- `configs/ablation_xpred_upred.yaml`
- `configs/ablation_lp_sweep.yaml`
- `configs/ablation_nfe_steps.yaml`
- `configs/train_rectflow_baseline.yaml`
- `experiments/ablations/` — all run results
- `experiments/phase_6/verification_report.md`

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| $L_p$ effect does not transfer to latent space | Weakens Contribution 2 | Medium | Report as null result; still publishable as insight |
| x-pred vs. u-pred shows no difference | Weakens Contribution 3 | Medium | Report as insight into VAE compression; still publishable |
| Some ablation runs crash mid-training | Blocks P6-T1 | Medium | Implement robust checkpointing; restart failed runs |
| Rectified Flow baseline too strong | Weakens MeanFlow narrative | Low | Still report honestly; may need more MeanFlow training |
