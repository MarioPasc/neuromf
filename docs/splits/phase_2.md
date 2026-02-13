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

## 9. REVIEW NOTES ON PHASE 2 (READ!)

Now I have a thorough understanding of the project structure, the external codebases, and the theoretical foundations. Let me provide a detailed critical review.

---

## Critical Review of `phase_2.md`

### Overall Assessment

The document is **well-structured and nearly sufficient**, but has several gaps and one substantive issue that could cause the agent to either stall or produce incorrect code. I'll organize the feedback by severity.

---

### 1. HIGH PRIORITY: Missing or Ambiguous Specifications

#### 1.1 The MeanFlow-PyTorch `h = t - r` Convention Is Not Explained

This is the single most likely source of agent confusion. In the `MeanFlow-PyTorch` reference (which is the primary PyTorch codebase the agent should study), the model's forward signature is:

```python
def u_fn(self, model, x, t, h, y, train=True):
    return model(x, t.reshape(bz), h.reshape(bz), y, train=train)
```

where `h = t - r` (the **interval length**), not `r` directly. The JVP is then computed over `(z_t, t, r)` with tangents `(v_g, 1, 0)`, but internally the model receives `h = t - r`. This means the tangent vector `(v, 0, 1)` in your Eq. 11 is **not** directly how the PyTorch reference implements it — the PyTorch code computes `jvp(u_fn, (z_t, t, r), (v_g, dtdt, dtdr))` where `dtdt = ones`, `dtdr = zeros`, and inside `u_fn`, it passes `t - r` to the model.

The document should explicitly state:

- The model in Phase 2 should accept `(z_t, r, t)` or `(z_t, t, h)` — specify which convention you want.
- How the JVP tangent vectors map to the chosen parameterisation.
- A concrete note: "In `MeanFlow-PyTorch/meanflow.py`, `h = t - r` is passed to the model. Since $\partial h / \partial t = 1$ and $\partial h / \partial r = -1$, the JVP tangent `(v, 1, 0)` for `(z, t, r)` becomes `(v, 1)` for `(z, h)` when `t` is not an independent input. Verify which convention you adopt."

**Recommendation:** Add a subsection **§4.1 "Time Parameterisation Convention"** that explicitly locks the convention for the toy MLP and explains the chain rule transformation. Without this, the agent will likely implement one convention in the model and another in the loss, producing silently wrong gradients that still decrease the loss (because the MLP is flexible enough to partially compensate).

#### 1.2 No Model Architecture Specification for the Toy MLP

Section 4 provides function signatures for the loss, sampling, and dataset modules but **never specifies the toy model architecture**. The agent needs to know:

- How many hidden layers and units (pMF's toy uses a 7-layer ReLU MLP with 256 hidden units — is that what you want?).
- How the model receives `(z_t, r, t)`: concatenation? Separate embeddings? The external code uses a full DiT/MMDiT with sinusoidal embeddings and AdaLN, which is overkill for the toy. The agent needs explicit guidance.
- Whether the model outputs $\mathbf{u}$ directly (u-prediction) or $\hat{\mathbf{x}}$ (x-prediction) by default, and how the loss module should handle both.
- Whether the model should handle the `(B, 4)` R⁴ case and the `(B, 4, 32, 32, 32)` volumetric case with the same architecture or different ones.

**Recommendation:** Add a `src/neuromf/models/toy_mlp.py` specification with:

```python
class ToyMLP(nn.Module):
    """7-layer ReLU MLP for toy MeanFlow experiments.
    
    Accepts (z_t, r, t) via concatenation: input_dim = data_dim + 2.
    Outputs average velocity u of shape (B, data_dim).
    """
    def __init__(self, data_dim: int = 4, hidden_dim: int = 256, 
                 n_layers: int = 7, prediction_type: str = "u") -> None: ...
    def forward(self, z_t: torch.Tensor, r: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor: ...
```

Specify that `inplace=False` for all ReLUs to ensure `torch.func.jvp` compatibility.

#### 1.3 The `data_proportion` / `r = t` Sampling Logic Is Under-Specified

The `sample_r_given_t` function signature in §4 says:

```python
def sample_r_given_t(t: torch.Tensor, r_equals_t_prob: float = 0.5) -> torch.Tensor
```

But the MeanFlow-PyTorch reference uses a different pattern: it samples `t` and `r` independently from the same distribution, then enforces `t ≥ r` via `torch.maximum/minimum`, and sets `r = t` for a fraction `data_proportion` of the batch. This is a critical detail because when `r = t`, the JVP term vanishes (multiplied by `t - r = 0`), and the loss reduces to a standard flow matching loss on $\tilde{\mathbf{v}}_\theta$. This is the iMF dual-loss mechanism.

The document should clarify:
- Whether `sample_r_given_t` takes a pre-sampled `t` and produces `r` conditioned on it (as the name suggests), or whether both are sampled jointly (as the reference does).
- What the distribution of `r` is when `r ≠ t`: uniform on $[0, t]$? Logit-normal then clamped? The reference samples both from the same logit-normal and swaps so `t ≥ r`.
- That `data_proportion = 0.5` corresponds to the `r_equals_t_prob = 0.5` parameter — make this mapping explicit.

#### 1.4 Missing x-Prediction Equations

Section 3 says to extract "x-prediction reparameterisation (Eqs. 15–16)" from `src/external/pmf/`, but **Eqs. 15–16 are not included in the Theoretical Background** (§2 only goes up to Eq. 14). Since P2-T8 tests both x-prediction and u-prediction convergence, the agent needs these equations in the document:

$$
\hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t) = \mathbf{z}_t - t \cdot \mathbf{u}_\theta(\mathbf{z}_t, r, t) \tag{15}
$$

$$
\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{\mathbf{z}_t - \hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t)}{t} \tag{16}
$$

And the corresponding modification to the loss: when using x-prediction, `u_fn` inside the JVP closure is defined as `u = (z - net(z, r, t)) / t` (as shown in pMF Algorithm 1).

---

### 2. MEDIUM PRIORITY: Correctness Concerns

#### 2.1 The 1-NFE Sampling Formula Has a Sign Ambiguity

The document states (Eq. 5):

$$\mathbf{z}_0 = \mathbf{z}_1 - 1 \cdot \mathbf{u}_\theta(\mathbf{z}_1, 0, 1)$$

The failure modes table (§8) mentions checking "z_0 = eps - u_theta(eps, 0, 1), not z_0 = eps + u_theta". However, the `sample_one_step` function signature:

```python
def sample_one_step(model, noise: torch.Tensor, r: float = 0.0, t: float = 1.0) -> torch.Tensor
```

does not specify the formula in the docstring. For the MeanFlow-PyTorch reference, the solver step is:

```python
def solver_step(self, model, z_t, t, r, labels):
    u = self.u_fn(model, z_t, t=t, h=(t - r), y=labels)
    return z_t - (t - r).view(-1, 1, 1, 1) * u
```

So `z_r = z_t - (t - r) * u`. For 1-NFE with `t=1, r=0`: `z_0 = z_1 - 1 * u_theta(z_1, 1, 1)`. Note that the model receives `h = t - r = 1`, not `r = 0`. This is consistent with your Eq. 5 **but** the agent might confuse `u_theta(z_1, 0, 1)` (your notation: $r=0, t=1$) with the model forward call `model(z_1, t=1, h=1)`. Spell this out explicitly.

#### 2.2 Volumetric Mode Parameterisation Is Vague

The document says "4-channel 32³ volumes parameterised by (θ₁, θ₂)" but doesn't specify **how** two angles produce a 4×32×32×32 tensor. The agent needs to know the generative process. For instance:

- Is each voxel value a deterministic function of (θ₁, θ₂) plus its spatial position? 
- Is it a smooth function like $f(x, y, z) = \cos(\theta_1) \cdot g_1(x, y, z) + \sin(\theta_2) \cdot g_2(x, y, z)$?
- What are the 4 channels encoding?

Without this, the agent will have to invent the parameterisation, which may or may not be suitable for flow matching. Since the R⁴ mode is already well-defined and sufficient for all critical tests (P2-T1 through P2-T6), I would recommend either:

(a) Removing volumetric mode from Phase 2 entirely (deferring to Phase 3 where 3D volumes are needed), or
(b) Providing an explicit generative formula.

#### 2.3 Adaptive Weighting Detail

Eq. 14 gives $w(t) = 1/(\text{sg}[\|\mathbf{e}\|_p^p] + c)$, but it's not clear what $\mathbf{e}$ is in this context. Looking at the MeanFlow-PyTorch code:

```python
adp_wt = (denoising_loss + self.norm_eps) ** self.norm_p
denoising_loss = denoising_loss / adp_wt.detach()
```

Here, `denoising_loss` is the per-sample squared error (summed over dimensions), and `norm_p` and `norm_eps` control the adaptive weighting. This is **not** the same as Eq. 14 — the reference applies the weighting to the per-sample loss, not per-voxel. Clarify whether $\mathbf{e}$ in Eq. 14 is the per-sample error vector or the batch-level loss, and whether the adaptive weight is applied per-sample or per-voxel.

---

### 3. LOW PRIORITY: Improvements for Agent Autonomy

#### 3.1 Explicit Priority Ordering of External References

The document lists three external codebases equally, but the agent should be told:

1. **Primary reference: `src/external/MeanFlow-PyTorch/meanflow.py`** — This is PyTorch, directly translatable. Study the `__call__`, `sample_tr`, `solver_step`, and `sample_one_step` methods.
2. **Secondary reference: `src/external/pmf/`** — For x-prediction reparameterisation and Algorithm 1 pseudocode.
3. **Tertiary reference: `src/external/MeanFlow/`** — JAX, useful only for understanding the mathematical intent; do NOT translate JAX API calls.

Currently the JAX reference is listed first (§3), which may bias the agent toward studying it more closely.

#### 3.2 Missing `__init__.py` Exports

The expected outputs list files but don't mention updating `src/neuromf/losses/__init__.py`, `src/neuromf/sampling/__init__.py`, etc. The agent should be told to update package `__init__.py` files so that `from neuromf.losses import meanflow_loss` works.

#### 3.3 Config YAML Example

Section 4 says `configs/toy_toroid.yaml` should contain "toroid params, small MLP/UNet architecture, learning rate, epochs, loss norm p" but provides no example YAML. Give the agent a skeleton:

```yaml
data:
  mode: "r4"
  n_samples: 10000
model:
  type: "mlp"
  data_dim: 4
  hidden_dim: 256
  n_layers: 7
  prediction_type: "u"  # or "x"
training:
  epochs: 200
  batch_size: 256
  lr: 1.0e-3
  optimizer: "adam"
loss:
  p: 2.0
  adaptive: true
  norm_eps: 0.01
  data_proportion: 0.75
  lambda_mf: 1.0
sampling:
  time_dist: "logit_normal"
  mu: -0.4
  sigma: 1.0
```

#### 3.4 Finite Difference JVP Verification Test

The failure modes table mentions "verify JVP correctness with finite differences" but this isn't a formal test. Given how critical the JVP computation is, I would add a **P2-T3b** test:

| P2-T3b | JVP matches finite-difference approximation | Relative error < 1e-3 for random MLP inputs | CRITICAL |

This catches subtle bugs in tangent vector construction that shape-matching alone (P2-T3) would miss.

#### 3.5 Training Hyperparameter Defaults from the Reference

The iMF paper (Geng et al., 2025b, Table 4) uses specific defaults: logit-normal(-0.4, 1.0), data_proportion=0.5 (ratio of r≠t = 50%), Adam with (β₁, β₂) = (0.9, 0.95). Your `sample_logit_normal` defaults use μ=0.8, σ=0.8, which differ from the reference. Either justify the difference or align them.

---

### 4. Summary of Recommended Changes

| Priority | Change | Rationale |
|----------|--------|-----------|
| **HIGH** | Add §4.1 "Time Parameterisation Convention" clarifying `h = t - r` vs `(r, t)` | Prevents silently wrong JVP |
| **HIGH** | Add toy MLP architecture specification | Agent cannot implement without it |
| **HIGH** | Include Eqs. 15–16 (x-prediction) in §2 | Required for P2-T8 |
| **HIGH** | Fully specify `sample_tr` logic with data_proportion | Prevents incorrect time sampling |
| **MEDIUM** | Clarify 1-NFE formula in terms of model forward call convention | Prevents sign/argument bugs |
| **MEDIUM** | Either remove volumetric mode or provide explicit formula | Prevents agent guesswork |
| **MEDIUM** | Clarify adaptive weighting scope (per-sample vs per-voxel) | Prevents loss computation bugs |
| **LOW** | Reorder external refs (PyTorch first, JAX last) | Reduces agent confusion |
| **LOW** | Add JVP finite-difference verification test (P2-T3b) | Catches subtle gradient bugs |
| **LOW** | Provide skeleton YAML config | Accelerates implementation |
| **LOW** | Align logit-normal defaults with iMF reference (μ=-0.4, σ=1.0) | Consistency with literature |

The document is methodologically sound — the choice of flat torus in $\mathbb{R}^4$ is excellent for validation because the norm constraint $\|\hat{\mathbf{z}}\|_2 = 1$ and pair-wise norm $\hat{z}_1^2 + \hat{z}_2^2 = 1/2$ provide analytically verifiable properties. The KS test on angular uniformity (P2-T6) is a rigorous statistical check. The test hierarchy with critical vs. informational classification is well-designed for gated progression.

The main risk is that an autonomous agent will spend significant time reverse-engineering the model architecture and time convention from the external code, when these could be specified directly. Addressing the HIGH-priority items above would make this document fully self-contained for autonomous execution.