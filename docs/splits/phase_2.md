# Phase 2: Toy Experiment — MeanFlow on Toroidal Manifold

**Depends on:** Phase 1 (gate must be OPEN)
**Modules touched:** `src/neuromf/data/`, `src/neuromf/losses/`, `src/neuromf/sampling/`, `src/neuromf/utils/`, `tests/`, `configs/`, `experiments/cli/`, `experiments/toy_toroid/`
**Estimated effort:** 2–3 sessions

---

## 1. Objective

Validate the entire MeanFlow pipeline — time sampling, JVP computation, loss backward, 1-NFE sampling, multi-step sampling — on a **known manifold** (flat torus $\mathbb{T}^2$ embedded in $\mathbb{R}^4$) where ground-truth properties can be verified analytically. This catches bugs before training on expensive brain MRI latents.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §1.1 Flow Matching Preliminaries (Eqs. 1–3)

$$
\mathbf{z}_t = (1 - t)\,\mathbf{x} + t\,\boldsymbol{\epsilon}, \quad \mathbf{x} \sim p_0,\; \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \quad t \in [0, 1]
\tag{1}
$$

$$
\mathbf{v}_c(\mathbf{z}_t, t) = \boldsymbol{\epsilon} - \mathbf{x}
\tag{2}
$$

$$
\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}, \boldsymbol{\epsilon}} \left[ \|\mathbf{v}_\theta(\mathbf{z}_t, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|^2 \right]
\tag{3}
$$

### §1.2 MeanFlow: Average Velocity (Eqs. 4–8)

$$
\mathbf{u}(\mathbf{z}_t, r, t) \triangleq \frac{1}{t - r} \int_r^t \mathbf{v}(\mathbf{z}_s, s)\, ds, \quad 0 \leq r < t \leq 1
\tag{4}
$$

$$
\mathbf{z}_0 = \mathbf{z}_1 - 1 \cdot \mathbf{u}_\theta(\mathbf{z}_1, 0, 1)
\tag{5}
$$

$$
(t - r)\,\frac{\partial \mathbf{u}}{\partial t}(\mathbf{z}_t, r, t) + \mathbf{u}(\mathbf{z}_t, r, t) = \mathbf{v}(\mathbf{z}_t, t)
\tag{6}
$$

$$
\frac{d}{dt}\mathbf{u}(\mathbf{z}_t, r, t) = \frac{\partial \mathbf{u}}{\partial \mathbf{z}_t} \cdot \mathbf{v}(\mathbf{z}_t, t) + \frac{\partial \mathbf{u}}{\partial t}
\tag{7}
$$

**MeanFlow Identity:**
$$
\mathbf{v}(\mathbf{z}_t, t) = \mathbf{u}(\mathbf{z}_t, r, t) + (t - r)\left[\frac{\partial \mathbf{u}}{\partial \mathbf{z}_t} \cdot \mathbf{v}(\mathbf{z}_t, t) + \frac{\partial \mathbf{u}}{\partial t}\right]
\tag{8}
$$

### §1.3 MeanFlow Training Objective (Eqs. 9–12)

**Compound prediction:**
$$
V_\theta(\mathbf{z}_t, r, t) = \mathbf{u}_\theta(\mathbf{z}_t, r, t) + (t - r) \cdot \text{sg}\!\left[\frac{\partial \mathbf{u}_\theta}{\partial \mathbf{z}_t} \cdot \tilde{\mathbf{v}}_\theta + \frac{\partial \mathbf{u}_\theta}{\partial t}\right]
\tag{9}
$$

**Instantaneous velocity estimate:**
$$
\tilde{\mathbf{v}}_\theta(\mathbf{z}_t, t) = \mathbf{u}_\theta(\mathbf{z}_t, t, t)
\tag{10}
$$

**JVP:**
$$
\text{JVP}\!\left(\mathbf{u}_\theta,\; (\mathbf{z}_t, r, t),\; (\tilde{\mathbf{v}}_\theta, 0, 1)\right) = \frac{\partial \mathbf{u}_\theta}{\partial \mathbf{z}_t} \cdot \tilde{\mathbf{v}}_\theta + \frac{\partial \mathbf{u}_\theta}{\partial t} \cdot 1
\tag{11}
$$

**MeanFlow loss:**
$$
\mathcal{L}_{\text{MF}} = \mathbb{E}_{t, r, \mathbf{x}, \boldsymbol{\epsilon}} \left[ w(t) \cdot \|V_\theta(\mathbf{z}_t, r, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|_p^p \right]
\tag{12}
$$

### §1.4 iMF Combined Loss (Eq. 13)

$$
\mathcal{L}_{\text{iMF}} = \mathbb{E}_{t, r, \mathbf{x}, \boldsymbol{\epsilon}} \left[ w(t) \cdot \|\tilde{\mathbf{v}}_\theta(\mathbf{z}_t, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|_p^p \right] + \lambda_{\text{MF}} \cdot \mathcal{L}_{\text{MF}}
\tag{13}
$$

### §1.5 Adaptive Weighting (Eq. 14)

$$
w(t) = \frac{1}{\text{sg}[\|\mathbf{e}\|_p^p] + c}
\tag{14}
$$

## 3. External Code to Leverage

### `src/external/MeanFlow/` (JAX reference)
- **Insights doc:** `docs/papers/meanflow_2025/insights.md`
- **Specific files:** `meanflow.py` (lines 206–256 — JAX JVP loss computation)
- **What to extract:** Loss computation pattern, stop-gradient usage, time sampling
- **What to AVOID:** JAX-specific API (`jax.jvp`, `jax.lax.stop_gradient`); translate to PyTorch equivalents

### `src/external/MeanFlow-PyTorch/` (PyTorch reference)
- **Specific files:** `meanflow.py` (lines 125–188 — PyTorch JVP loss)
- **What to extract:** `torch.func.jvp` usage pattern, in-place ops handling
- **What to AVOID:** Any FlashAttention-related code (not needed for toy experiment)

### `src/external/pmf/` (x-prediction, adaptive weighting)
- **Insights doc:** `docs/papers/pmf_2026/insights.md`
- **Specific files:** `pmf.py` — x-prediction implementation, adaptive weighting formula
- **What to extract:** x-prediction reparameterisation (Eqs. 15–16), adaptive weight computation

## 4. Implementation Specification

### `src/neuromf/data/toroid_dataset.py`
- **Purpose:** Synthetic dataset of 4D points on a flat torus and 3D volumes parameterised by a 2-torus.
- **Key classes:**
  ```python
  @dataclass
  class ToroidConfig:
      major_radius: float = 3.0
      minor_radius: float = 1.0
      n_samples: int = 10_000
      spatial_size: int = 32
      n_channels: int = 4
      mode: str = "r4"  # "r4" for pure R^4 torus, "volumetric" for 3D volumes

  class ToroidDataset(Dataset):
      def __init__(self, config: ToroidConfig) -> None: ...
      def __getitem__(self, idx: int) -> torch.Tensor: ...
  ```
- **R^4 mode:** Points on flat torus via φ(θ₁, θ₂) = (1/√2)(cos θ₁, sin θ₁, cos θ₂, sin θ₂)
- **Volumetric mode:** 4-channel 32^3 volumes parameterised by (θ₁, θ₂)

### `src/neuromf/losses/meanflow_jvp.py`
- **Purpose:** Core MeanFlow JVP loss computation.
- **Key functions:**
  ```python
  def compute_meanflow_jvp(model_fn, z_t, r, t, v_tangent) -> torch.Tensor: ...
  def meanflow_loss(model, z_t, r, t, x_0, eps, p: float = 2.0, adaptive: bool = True) -> torch.Tensor: ...
  ```

### `src/neuromf/losses/lp_loss.py`
- **Purpose:** Per-channel Lp loss computation.
- **Key functions:**
  ```python
  def lp_loss(pred: torch.Tensor, target: torch.Tensor, p: float, channel_weights: Optional[torch.Tensor] = None, reduction: str = "mean") -> torch.Tensor: ...
  ```

### `src/neuromf/sampling/one_step.py`
- **Purpose:** 1-NFE MeanFlow sampling.
- **Key functions:**
  ```python
  def sample_one_step(model, noise: torch.Tensor, r: float = 0.0, t: float = 1.0) -> torch.Tensor: ...
  ```

### `src/neuromf/sampling/multi_step.py`
- **Purpose:** Multi-step Euler sampling for ablation.
- **Key functions:**
  ```python
  def sample_euler(model, noise: torch.Tensor, n_steps: int = 50) -> torch.Tensor: ...
  ```

### `src/neuromf/utils/time_sampler.py`
- **Purpose:** Logit-normal time sampling.
- **Key functions:**
  ```python
  def sample_logit_normal(batch_size: int, mu: float = 0.8, sigma: float = 0.8, t_min: float = 0.05) -> torch.Tensor: ...
  def sample_r_given_t(t: torch.Tensor, r_equals_t_prob: float = 0.5) -> torch.Tensor: ...
  ```

### `configs/toy_toroid.yaml`
- **Key fields:** toroid params, small MLP/UNet architecture, learning rate, epochs, loss norm p

### `experiments/cli/run_toy_toroid.py`
- **Purpose:** End-to-end toy experiment: train MeanFlow on toroid, evaluate, produce plots.

## 5. Data and I/O

- **Input:** Synthetically generated (no external data needed)
- **Output:**
  - Training loss curves → `experiments/toy_toroid/loss_curves.json`
  - Generated samples → `experiments/toy_toroid/generated_samples.pt`
  - Verification metrics → `experiments/toy_toroid/verification_metrics.json`
  - Plots → `experiments/toy_toroid/figures/`

## 6. Verification Tests

### Toroid Verification Mathematics

Given a generated sample $\hat{\mathbf{z}} \in \mathbb{R}^4$ from the R^4 torus experiment:
1. **Norm constraint:** For flat torus, $\|\hat{\mathbf{z}}\|_2 = 1$. Compute $\delta_{\text{norm}} = |\|\hat{\mathbf{z}}\|_2 - 1|$.
2. **Angular extraction:** $\hat{\theta}_1 = \text{atan2}(\hat{z}_2, \hat{z}_1)$, $\hat{\theta}_2 = \text{atan2}(\hat{z}_4, \hat{z}_3)$.
3. **Pair-wise norm:** $\hat{z}_1^2 + \hat{z}_2^2 = 1/2$ and $\hat{z}_3^2 + \hat{z}_4^2 = 1/2$.

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P2-T1 | Toroid dataset generates valid samples | All samples finite, correct shape `(4, 32, 32, 32)` or `(4,)` | CRITICAL | Unit test on dataset |
| P2-T2 | MeanFlow loss computes without error on toroid batch | No NaN/Inf in loss; loss is finite and positive | CRITICAL | Forward pass test |
| P2-T3 | JVP computation produces correct shape | JVP output shape == input shape | CRITICAL | Unit test on `torch.func.jvp` with small MLP |
| P2-T4 | Training loss decreases monotonically (after warmup) | Loss at epoch 100 < loss at epoch 10 < loss at epoch 1 | CRITICAL | Training log |
| P2-T5 | 1-NFE samples lie approximately on the torus | Mean distance to torus surface < 0.1 | CRITICAL | Compute norm deviations |
| P2-T6 | Angular distribution of generated samples is approximately uniform | KS-test p-value > 0.01 for θ₁ and θ₂ marginals vs. Uniform | CRITICAL | Apply φ⁻¹, extract angles, run scipy.stats.kstest |
| P2-T7 | Multi-step (5 steps) produces better torus-fidelity than 1-step | Mean torus distance at 5-step ≤ 1-step | INFORMATIONAL | Compare distances |
| P2-T8 | x-prediction and u-prediction both converge on toroid | Both reach loss < τ within 200 epochs | INFORMATIONAL | Train both, check final loss |

**Suggested test files:** `tests/test_toroid_dataset.py`, `tests/test_meanflow_loss.py`, `tests/test_lp_loss.py`, `tests/test_time_sampler.py`

**Phase 2 is PASSED when P2-T1 through P2-T6 are ALL green. P2-T7 and P2-T8 are informational.**

## 7. Expected Outputs

- `src/neuromf/data/toroid_dataset.py`
- `src/neuromf/losses/meanflow_jvp.py`
- `src/neuromf/losses/lp_loss.py`
- `src/neuromf/sampling/one_step.py`
- `src/neuromf/sampling/multi_step.py`
- `src/neuromf/utils/time_sampler.py`
- `configs/toy_toroid.yaml`
- `experiments/cli/run_toy_toroid.py`
- `tests/test_toroid_dataset.py`
- `tests/test_meanflow_loss.py`
- `tests/test_lp_loss.py`
- `tests/test_time_sampler.py`
- `experiments/toy_toroid/` — results, plots, metrics
- `experiments/phase_2/verification_report.md`

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| `torch.func.jvp` fails on MLP | Blocks P2-T3 | Low | MLPs rarely have in-place ops; if they do, replace ReLU inplace=True |
| Loss doesn't decrease | Blocks P2-T4 | Medium | Check learning rate, verify JVP correctness with finite differences, ensure time sampling is correct |
| Generated samples not on torus | Blocks P2-T5 | Medium | Check that 1-NFE formula is z_0 = eps - u_theta(eps, 0, 1), not z_0 = eps + u_theta |
| KS test fails (non-uniform angles) | Blocks P2-T6 | Medium | May need more training epochs; check for mode collapse in specific angular regions |
