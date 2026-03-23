# NeuroMF v2: Complete Implementation Specification

**Purpose:** Actionable specification for a local coding agent to implement all changes needed to fix the 5 identified failure modes in NeuroMF v1 and retrain from scratch.

**Target:** Single training run on Picasso A100 that fixes gradient conflict, v-head capacity, adaptive weight saturation, SWD divergence, and 1-NFE variance collapse.

**Repository root:** The agent has access to the full `neuromf` repository at the project root.

**Key principle:** Every change addresses a specific, diagnosed failure mode. No change is speculative.

---

## Table of Contents

1. [Summary of Changes](#1-summary-of-changes)
2. [Change 1: α-Flow Curriculum Objective](#2-change-1-α-flow-curriculum-objective)
3. [Change 2: Boundary Condition v-Head (Replace Aux v-Head)](#3-change-2-boundary-condition-v-head)
4. [Change 3: Two-Stage Training with rflow Initialisation](#4-change-3-two-stage-training-with-rflow-initialisation)
5. [Change 4: Constant Learning Rate](#5-change-4-constant-learning-rate)
6. [Change 5: Adaptive Weighting Fix (norm_p=0.5, norm_eps=0.01)](#6-change-5-adaptive-weighting-fix)
7. [Change 6: Multiple EMA Decay Rates](#7-change-6-multiple-ema-decay-rates)
8. [Change 7: Post-hoc Norm Correction at Inference](#8-change-7-post-hoc-norm-correction)
9. [Training Configuration (v2 config)](#9-training-configuration)
10. [Verification Tests](#10-verification-tests)
11. [External Dependencies](#11-external-dependencies)
12. [File Change Summary](#12-file-change-summary)

---

## 1. Summary of Changes

| # | Change | Failure Mode Addressed | Effort | Files Modified |
|---|--------|----------------------|--------|----------------|
| 1 | α-Flow curriculum objective | FM/MF gradient conflict | HIGH | `meanflow_loss.py`, `latent_meanflow.py`, new `alpha_scheduler.py` |
| 2 | Boundary condition v-head | v-head capacity bottleneck | MEDIUM | `maisi_unet.py`, `meanflow_loss.py`, config |
| 3 | Two-stage training (FM pretrain → MF fine-tune) | Chicken-and-egg dynamics | HIGH | `latent_meanflow.py`, `train.py`, new config |
| 4 | Constant LR (no cosine decay) | Premature LR decay | LOW | `latent_meanflow.py`, config |
| 5 | norm_p=0.5, norm_eps=0.01 | Adaptive weight saturation | LOW | config |
| 6 | Multiple EMA decay rates | Suboptimal EMA selection | MEDIUM | `ema.py`, `latent_meanflow.py` |
| 7 | Post-hoc norm correction | Velocity overshoot at 1-NFE | LOW | inference script, config |

---

## 2. Change 1: α-Flow Curriculum Objective

### 2.1 Failure Mode

The MeanFlow objective decomposes into trajectory flow matching (TFM) and trajectory consistency (TC). α-Flow (Zhang et al., arXiv:2510.20771) proves these gradients are strongly negatively correlated (cos → -1.0), causing optimisation conflict. Our v1 training confirms this: `cos(V, v_c)` plateaus at 0.31 despite 710 epochs.

### 2.2 Mathematical Formulation

The α-Flow objective modifies the compound velocity by controlling the consistency step size. Instead of sampling `(r, t)` where the gap `h = t - r` can be arbitrarily large from epoch 0, α-Flow introduces a parameter α(k) that controls the maximum consistency gap at training step k:

**Standard MeanFlow sampling:**
For MF samples (r ≠ t), sample `t ~ logit-normal`, then `r ~ U[0, t]`. The gap `h = t - r` can be up to 1.0 from the start.

**α-Flow sampling:**
For MF samples, sample `t ~ logit-normal`, then set `s = t - α · (t - r_sampled)`, effectively clamping the gap to `α · h_max`. When α = 0, `s = t` (pure FM). When α = 1, `s = r` (full MF).

**The α schedule** uses a sigmoid function:

$$\alpha(k) = \eta \cdot \sigma\left(\gamma \cdot \frac{k - k_s}{k_e - k_s} - \frac{\gamma}{2}\right)$$

where:
- `k` = current training step
- `k_s` = step where annealing starts (after FM pretraining stage)
- `k_e` = step where annealing ends
- `γ` = temperature (default 25, controls sharpness of sigmoid)
- `η` = clamping value (optimal: `5e-3` per α-Flow Table 5c — but we may need to tune this for our setting; start with `η = 1.0` and ablate)
- `σ(x) = 1 / (1 + exp(-x))` = standard sigmoid

**Key insight from α-Flow:** The optimal **final** α is NOT 0 (pure FM) nor 1 (full MF), but a small value around `5e-3`. This means the model benefits from a tiny consistency gap even at convergence, not a full MeanFlow gap. However, this was found on ImageNet with DiT; for our 3D medical setting, we should start with `η = 1.0` (full MF at convergence) and ablate down.

**Practical implementation for our two-stage setting:**

- **Stage 1 (FM pretraining):** α = 0. All samples have r = t. Pure flow matching loss. This stage uses the MAISI rflow checkpoint for initialisation.
- **Stage 2 (α-Flow annealing):** α ramps from 0 → η via the sigmoid schedule. The model progressively encounters larger consistency gaps.

### 2.3 Implementation

**New file: `src/neuromf/utils/alpha_scheduler.py`**

```python
"""α-Flow curriculum scheduler for progressive MeanFlow training.

Implements the sigmoid-based annealing schedule from:
Zhang et al., "AlphaFlow: Understanding and Improving MeanFlow Models,"
arXiv:2510.20771, 2025. Section 5.2 and Algorithm 2.

The scheduler controls the consistency step ratio α(k) which determines
the maximum gap between r and t in MeanFlow sampling. α=0 corresponds
to pure flow matching (r=t), α=1 to full MeanFlow.
"""

import math
from dataclasses import dataclass


@dataclass
class AlphaSchedulerConfig:
    """Configuration for the α-Flow curriculum scheduler.

    Args:
        start_step: Training step where annealing begins (k_s).
            Set to 0 for no FM pretraining, or to the end of Stage 1.
        end_step: Training step where annealing ends (k_e).
        gamma: Temperature for the sigmoid transition (default 25.0).
            Higher = sharper transition.
        eta: Final clamping value for α (default 1.0).
            α-Flow paper finds optimal η=5e-3 on ImageNet.
            Start with 1.0 for our setting and ablate.
        mode: "sigmoid" (α-Flow) or "linear" (simpler alternative).
    """
    start_step: int = 0
    end_step: int = 100000
    gamma: float = 25.0
    eta: float = 1.0
    mode: str = "sigmoid"


class AlphaScheduler:
    """Computes α(k) at each training step.

    Args:
        config: Scheduler configuration.
    """

    def __init__(self, config: AlphaSchedulerConfig) -> None:
        self.config = config

    def get_alpha(self, step: int) -> float:
        """Compute α at the given training step.

        Args:
            step: Current global training step.

        Returns:
            α value in [0, η].
        """
        cfg = self.config

        if step < cfg.start_step:
            return 0.0  # Pure FM during Stage 1

        if step >= cfg.end_step:
            return cfg.eta  # Converged

        # Normalised progress in [0, 1]
        progress = (step - cfg.start_step) / max(cfg.end_step - cfg.start_step, 1)

        if cfg.mode == "sigmoid":
            # Sigmoid schedule: α = η * σ(γ * (progress - 0.5))
            x = cfg.gamma * (progress - 0.5)
            alpha = cfg.eta * (1.0 / (1.0 + math.exp(-x)))
        elif cfg.mode == "linear":
            alpha = cfg.eta * progress
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

        return alpha
```

**Modified file: `src/neuromf/utils/time_sampler.py`**

Add a method that uses α to control the consistency gap:

```python
def sample_alpha_flow(
    self,
    batch_size: int,
    alpha: float,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample (t, r) pairs using the α-Flow curriculum.

    When α=0, all samples have r=t (pure FM).
    When α>0, MF samples have gap h = α * h_sampled.

    Args:
        batch_size: Number of samples.
        alpha: Current curriculum parameter in [0, 1].
        device: Target device.

    Returns:
        Tuple (t, r) of shape (B,).
    """
    # Sample t from logit-normal
    t = self.sample_t(batch_size, device)  # existing method

    # Determine FM vs MF split
    n_fm = int(batch_size * self.data_proportion)
    n_mf = batch_size - n_fm

    r = torch.zeros(batch_size, device=device)

    # FM samples: r = t (no consistency gap)
    r[:n_fm] = t[:n_fm]

    # MF samples: sample base r, then scale gap by α
    if n_mf > 0 and alpha > 0:
        # Sample r_base uniformly in [0, t] for MF samples
        r_base = torch.rand(n_mf, device=device) * t[n_fm:]
        # Scale gap: r = t - α * (t - r_base)
        gap = t[n_fm:] - r_base
        r[n_fm:] = t[n_fm:] - alpha * gap
    else:
        # α=0: all samples are FM (r=t)
        r[n_fm:] = t[n_fm:]

    # Clamp for numerical safety
    r = r.clamp(min=0.0, max=1.0)

    return t, r
```

**Modified file: `src/neuromf/models/latent_meanflow.py`**

In `__init__`, instantiate the scheduler:

```python
# α-Flow curriculum scheduler
alpha_cfg = config.get("alpha_flow", {})
self.alpha_scheduler = AlphaScheduler(AlphaSchedulerConfig(
    start_step=int(alpha_cfg.get("start_step", 0)),
    end_step=int(alpha_cfg.get("end_step", 100000)),
    gamma=float(alpha_cfg.get("gamma", 25.0)),
    eta=float(alpha_cfg.get("eta", 1.0)),
    mode=str(alpha_cfg.get("mode", "sigmoid")),
))
```

In `training_step`, use the scheduler to get α and pass to time sampling:

```python
alpha = self.alpha_scheduler.get_alpha(self.global_step)
t, r = self.time_sampler.sample_alpha_flow(batch_size, alpha, device)
self.log("train/alpha", alpha, on_step=True, prog_bar=True)
```

### 2.4 Justification

- α-Flow achieves FID=2.58 at 1-NFE on ImageNet 256×256, a 15% improvement over MeanFlow (3.43) and competitive with iMF (1.72).
- The curriculum resolves the gradient conflict by ensuring the FM objective is well-optimised before introducing consistency constraints.
- Our v1 data shows `cos(V, v_c)` plateaus at 0.31 — the curriculum should enable this to continue improving.

### 2.5 Testable Criteria

1. **Unit test:** `AlphaScheduler` returns 0.0 before `start_step`, η after `end_step`, and monotonically increasing values between.
2. **Integration test:** When α=0, all returned `(t, r)` pairs have `r ≈ t` (within float precision).
3. **Training diagnostic:** `train/alpha` should follow the expected sigmoid curve in TensorBoard.
4. **Early training:** During Stage 1 (α=0), the FM loss should decrease steadily (no gradient conflict).
5. **Transition:** When α starts increasing, `cos(V, v_c)` should improve beyond the v1 plateau of 0.31.

### 2.6 External Dependencies

- **α-Flow reference code:** `https://github.com/snap-research/alphaflow` (PyTorch, MIT licence). Study `configs/` for schedule parameters and `src/training/` for the α-sampling logic.
- No new pip packages needed.

---

## 3. Change 2: Boundary Condition v-Head

### 3.1 Failure Mode

The current v-head has 228K parameters (single ResBlock + Conv3d), producing `cos(ṽ, v_c) ≈ 0.46` (63° angular error). The iMF paper uses L=8 Transformer layers for the v-head. Our v-head is 21–256× smaller.

### 3.2 Mathematical Formulation

The boundary condition approach uses:

$$v_\theta(\mathbf{z}_t, t) \equiv u_\theta(\mathbf{z}_t, t, t)$$

That is, the instantaneous velocity is obtained by evaluating the main network `u_θ` with `r = t`. This leverages the **full 178M parameter backbone** for tangent estimation, at the cost of one extra forward pass.

**iMF Table 1a (without CFG):**
- Boundary condition: FID = 29.42
- Auxiliary head (L=8): FID = 30.76
- Original MF: FID = 32.69

The boundary condition is **better** than the auxiliary head in the unconditional setting (our setting). The iMF paper explains: "when the model has more capacity, it can better leverage the capacity to learn v_θ by u_θ(z_t, t, t)."

### 3.3 Implementation

**Modified file: `src/neuromf/wrappers/maisi_unet.py`**

Remove the v-head construction entirely. The `MAISIUNetConfig` should set `use_v_head: false`. No v-head parameters, no v-head loss.

The tangent for JVP is computed by calling the main network with `r = t`:

```python
# In the forward method or in the loss pipeline:
# v_tilde = u_theta(z_t, t, t)  — boundary condition
with torch.no_grad():
    v_tilde = self.net(z_t, t, t)  # r=t gives instantaneous velocity
```

This call is under `no_grad()` because the tangent is used inside `stop_gradient` in the JVP computation.

**Modified file: `src/neuromf/wrappers/meanflow_loss.py`**

Update `_make_u_fn` and the forward method to always use boundary condition:

```python
def _compute_v_tilde(self, model, z_t, t):
    """Compute instantaneous velocity via boundary condition.

    v_theta(z_t, t) = u_theta(z_t, t, t)  (iMF Eq. 12, boundary variant)

    Args:
        model: The u_theta network.
        z_t: Noisy latent, shape (B, C, D, H, W).
        t: Time, shape (B,).

    Returns:
        v_tilde of same shape as z_t.
    """
    u_fn = self._make_u_fn(model)
    with torch.no_grad():
        v_tilde = u_fn(z_t, t, t)  # r = t → boundary condition
    return v_tilde
```

Remove all `use_v_head` branches from the forward pass. The v-head auxiliary loss is eliminated.

**Modified config: `configs/train_meanflow_v2.yaml`**

```yaml
unet:
  use_v_head: false  # CHANGED: use boundary condition instead
```

### 3.4 Computational Cost

The boundary condition requires one extra forward pass through the backbone per training step (to compute `v_tilde = u(z_t, t, t)`) in addition to the main forward pass (for `u(z_t, r, t)`) and the JVP pass. This is ~1.5× the cost of the v-head approach (which piggybacks on the main forward pass).

However, the v-head required an auxiliary loss computation that is now eliminated, partially offsetting the cost. Net increase: ~30–40% per step.

On A100 with current batch size, epoch time should go from ~547s to ~750s. With 1500 epochs target, total training time increases from ~9.5 days to ~13 days. Acceptable.

### 3.5 Testable Criteria

1. **Unit test:** `v_tilde = model(z_t, t, t)` should have the same shape as `model(z_t, r, t)`.
2. **Gradient check:** `v_tilde` should have `requires_grad = False` (it's computed under `no_grad`).
3. **Training diagnostic:** `cos(ṽ, v_c)` should start higher than v1's 0.11 (epoch 0) because the full backbone is used.
4. **Convergence criterion:** `cos(ṽ, v_c)` should exceed v1's plateau of 0.46 within 200 epochs.

### 3.6 External Dependencies

None. This is a simplification.

---

## 4. Change 3: Two-Stage Training with rflow Initialisation

### 4.1 Failure Mode

The chicken-and-egg problem: accurate v_tilde is needed for good JVP, but good JVP is needed to learn v_tilde. Starting MF training from scratch (or from MOTFM pixel-space weights) means both are bad initially.

### 4.2 Design

**Stage 1: FM Pretraining (α = 0)**
- Load MAISI rflow checkpoint (402/430 keys match)
- Train pure flow matching: `r = t` for all samples, so `V = u` (no JVP)
- Objective: `L_FM = ||u_θ(z_t, t, t) - v_c||^2`
- Duration: ~500 epochs (until FM loss stabilises)
- This stage produces a model that can accurately predict instantaneous velocity v(z_t, t) everywhere in the (z, t) space

**Stage 2: α-Flow MF Fine-tuning (α: 0 → η)**
- Continue from Stage 1 checkpoint
- Enable α-Flow curriculum: α ramps from 0 to η
- Now `v_tilde = u(z_t, t, t)` is accurate (from Stage 1), so JVP corrections are informative
- Duration: ~1000 epochs

### 4.3 Implementation

**Modified file: `experiments/cli/train.py`**

Add stage-aware training logic. The simplest approach: two config files.

```yaml
# configs/train_v2_stage1.yaml (FM pretraining)
alpha_flow:
  start_step: 999999999  # Never start annealing → α=0 always
  eta: 0.0
training:
  max_epochs: 500
unet:
  use_v_head: false
  initialization:
    method: rflow_transfer
    rflow_checkpoint_path: "/path/to/maisi_rflow_checkpoint.pt"
```

```yaml
# configs/train_v2_stage2.yaml (α-Flow fine-tuning)
alpha_flow:
  start_step: 0          # Start annealing immediately
  end_step: 42000        # ~1000 epochs × 42 steps/epoch
  gamma: 25.0
  eta: 1.0               # Full MF at convergence (ablate: try 0.005)
  mode: sigmoid
training:
  max_epochs: 1000
  resume_from: "/path/to/stage1_best_checkpoint.ckpt"
unet:
  use_v_head: false
  initialization:
    method: resume  # Load from Stage 1 checkpoint
```

**The agent should implement a `--stage` CLI flag** or use separate config files. The key is that Stage 2 **resumes** from Stage 1's best checkpoint.

### 4.4 Justification

- DTD (Kim et al., 2025) and Decoupled MeanFlow (Yan et al., 2025) independently propose two-stage training
- MAISI rflow checkpoint provides 402/430 matched keys — expected 2–5× faster FM convergence than Kaiming init
- α-Flow's curriculum ensures the consistency objective is introduced only after the velocity field is well-learned

### 4.5 Testable Criteria

1. **Stage 1 end:** FM loss should be < 100K (current v1 FM loss at convergence: ~690K; with rflow init, it should converge faster and lower)
2. **Stage 1 end:** `cos(v_tilde, v_c) > 0.7` (boundary condition v_tilde should be accurate after FM pretraining)
3. **Stage 2 start:** Verify checkpoint loading: all 178M parameters match between Stage 1 output and Stage 2 input
4. **Stage 2 early:** α should be visible in TensorBoard, following sigmoid curve
5. **Stage 2 mid:** `cos(V, v_c) > 0.5` (exceeding v1's plateau of 0.31)

---

## 5. Change 4: Constant Learning Rate

### 5.1 Failure Mode

v1 used cosine decay from 1e-4 → 0. By epoch 430 (best FID), LR had decayed to ~8.5e-5. By epoch 689, it was ~5.6e-5. The iMF and MF papers both use constant LR. MeanFlow's adaptive weighting already controls effective step size per-sample, making external LR scheduling redundant and potentially harmful.

### 5.2 Implementation

**Modified file: `src/neuromf/models/latent_meanflow.py` → `configure_optimizers()`**

Replace the cosine scheduler with constant LR + linear warmup:

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.net.parameters(),
        lr=float(self.cfg.training.lr),  # 1e-4
        weight_decay=0.0,
        betas=(0.9, 0.95),  # iMF default
    )
    # Linear warmup, then constant
    warmup_steps = int(self.cfg.training.get("warmup_steps", 1000))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
    }
```

**Config change:**

```yaml
training:
  lr: 1.0e-4
  lr_schedule: constant  # CHANGED from "cosine"
  warmup_steps: 1000
  betas: [0.9, 0.95]     # iMF default (was [0.9, 0.999])
  weight_decay: 0.0
```

### 5.3 Testable Criteria

1. **Unit test:** After warmup, LR should be exactly 1e-4 at step 2000, 10000, 50000.
2. **Training diagnostic:** `learning_rate` in TensorBoard should be flat after warmup.

---

## 6. Change 5: Adaptive Weighting Fix

### 6.1 Failure Mode

v1 uses `norm_p=1.0` and `norm_eps=0.01`. The weighted loss saturates at exactly 2.0. The adaptive weight `w = 1 / (sg[||error||^p] + eps)^norm_p` becomes so large that the effective loss is constant. With `norm_p=0.5`, the weighting is less aggressive, preserving more gradient signal.

### 6.2 Implementation

**Config change only:**

```yaml
meanflow:
  norm_p: 0.5   # CHANGED from 1.0 — less aggressive adaptive weighting
  norm_eps: 0.01 # Keep at 0.01 (iMF default)
```

The mathematical effect: with `norm_p=0.5`:

$$w(t) = \frac{1}{(\text{sg}[\|\text{error}\|^p] + \epsilon)^{0.5}}$$

Compared to `norm_p=1.0`:

$$w(t) = \frac{1}{\text{sg}[\|\text{error}\|^p] + \epsilon}$$

The square root reduces the dynamic range of the weights, preventing saturation. If the raw loss varies from 1K to 10M (as in v1), with `norm_p=1.0` the weight range is 10⁻⁷ to 10⁻³ (10⁴ ratio). With `norm_p=0.5`, the range is 10⁻³·⁵ to 10⁻¹·⁵ (10² ratio) — much less extreme.

### 6.3 Testable Criteria

1. **Training diagnostic:** `loss_mean` should NOT be constant at 2.0. It should vary meaningfully (expected range: 1.5–3.0).
2. **Training diagnostic:** The ratio `raw_loss_mean / loss_mean` should have lower variance than v1.

---

## 7. Change 6: Multiple EMA Decay Rates

### 7.1 Failure Mode

v1 uses a single EMA rate β=0.9999 (effective window: 10K steps). With ~30K steps, this averages over 33% of training. For our extended v2 training (~63K steps across both stages), we should track multiple rates and select the best.

### 7.2 Implementation

**Modified file: `src/neuromf/utils/ema.py`**

The current `EMATracker` maintains one set of shadow parameters. Extend it to maintain multiple:

```python
class MultiEMATracker:
    """Track multiple EMA decay rates simultaneously.

    Maintains N copies of shadow parameters, one per decay rate.
    At evaluation, the best-performing EMA is selected.

    Args:
        model: The model to track.
        decays: List of decay rates (e.g., [0.999, 0.9995, 0.9999]).
        active_index: Which EMA to use for inference by default.
    """

    def __init__(
        self,
        model: nn.Module,
        decays: list[float],
        active_index: int = -1,  # default: highest decay
    ) -> None:
        self.decays = decays
        self.active_index = active_index if active_index >= 0 else len(decays) - 1
        self.shadow_params: list[list[Tensor]] = []

        for _ in decays:
            self.shadow_params.append(
                [p.clone().detach() for p in model.parameters()]
            )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update all EMA shadows."""
        for i, decay in enumerate(self.decays):
            for shadow, param in zip(self.shadow_params[i], model.parameters()):
                shadow.lerp_(param.data, 1.0 - decay)

    def apply(self, model: nn.Module, index: int | None = None) -> list[Tensor]:
        """Copy EMA params into model, return originals for restore."""
        idx = index if index is not None else self.active_index
        originals = []
        for shadow, param in zip(self.shadow_params[idx], model.parameters()):
            originals.append(param.data.clone())
            param.data.copy_(shadow)
        return originals

    def restore(self, model: nn.Module, originals: list[Tensor]) -> None:
        """Restore original params after EMA evaluation."""
        for orig, param in zip(originals, model.parameters()):
            param.data.copy_(orig)
```

**Modified file: `src/neuromf/models/latent_meanflow.py`**

Replace single EMA with MultiEMA:

```python
# In __init__:
ema_decays = list(self.cfg.ema.get("decays", [0.999, 0.9995, 0.9999]))
self.multi_ema = MultiEMATracker(self.net, ema_decays)

# In training_step (after optimizer.step()):
self.multi_ema.update(self.net)

# In validation_step or sample generation:
originals = self.multi_ema.apply(self.net)  # Use active EMA
# ... generate samples, compute FID ...
self.multi_ema.restore(self.net, originals)
```

**Config:**

```yaml
ema:
  decays: [0.999, 0.9995, 0.9999]
  active_index: 2  # Use β=0.9999 by default; sweep at eval
```

### 7.3 Testable Criteria

1. **Unit test:** After N updates, shadow params differ from model params.
2. **Unit test:** Different decay rates produce different shadow params.
3. **Memory test:** 3 EMA copies × 178M params × 4 bytes = ~2.1 GB. This fits on A100 (37 GB reserved in v1).

---

## 8. Change 7: Post-hoc Norm Correction

### 8.1 Failure Mode

v1's compound velocity norm ratio converges to ~1.19 (19% overshoot). At 1-NFE inference, this systematic bias causes the model to overshoot the data manifold.

### 8.2 Implementation

**New file or modify: `src/neuromf/sampling/one_step.py`**

```python
def sample_one_step(
    model: nn.Module,
    noise: Tensor,
    gamma: float = 1.0,
) -> Tensor:
    """Generate samples via 1-NFE MeanFlow.

    z_0 = eps - u_theta(eps, 0, 1) / gamma

    Args:
        model: Trained MeanFlow model.
        noise: Gaussian noise, shape (B, C, D, H, W).
        gamma: Norm correction factor. Default 1.0 (no correction).
            Calibrate on validation set by searching γ ∈ {1.0, 1.05, ..., 1.25}.

    Returns:
        Predicted clean latent z_0.
    """
    u = model(noise, r=torch.zeros(noise.shape[0], device=noise.device),
              t=torch.ones(noise.shape[0], device=noise.device))
    return noise - u / gamma
```

**Post-training calibration script:**

The agent should create `experiments/cli/calibrate_gamma.py` that:
1. Loads the best checkpoint + EMA
2. Generates N=100 samples at γ ∈ {1.0, 1.05, 1.10, 1.15, 1.20, 1.25}
3. Computes per-channel std of the generated latents
4. Selects γ that makes std closest to 1.0 (the target)

### 8.3 Testable Criteria

1. **Post-training:** At optimal γ, per-channel std of generated latents should be in [0.95, 1.05].
2. **FID improvement:** 3D-FID at optimal γ should be lower than at γ=1.0.

---

## 9. Training Configuration

### 9.1 Complete v2 Config

**File: `configs/train_v2_stage1.yaml`**

```yaml
# NeuroMF v2 Stage 1: FM Pretraining
# Purpose: Learn accurate velocity field v(z_t, t) via pure flow matching
# Initialisation: MAISI rflow checkpoint

unet:
  spatial_dims: 3
  in_channels: 4
  out_channels: 4
  channels: [64, 128, 256, 512]
  attention_levels: [false, false, true, true]
  num_res_blocks: 2
  num_head_channels: [0, 0, 32, 32]
  norm_num_groups: 32
  norm_eps: 1.0e-6
  resblock_updown: true
  transformer_num_layers: 1
  use_flash_attention: false
  with_conditioning: false
  use_v_head: false                   # CHANGED: boundary condition instead
  conditioning_mode: dual
  prediction_type: x
  initialization:
    method: rflow_transfer
    rflow_checkpoint_path: "${paths.external_ssd}/maisi_checkpoints/rflow_checkpoint.pt"
    slice_time_emb_proj: true
    reinit_output_conv: false

training:
  max_epochs: 500
  lr: 1.0e-4
  lr_schedule: constant               # CHANGED from cosine
  warmup_steps: 1000
  betas: [0.9, 0.95]                  # CHANGED: iMF default
  weight_decay: 0.0
  batch_size: 3
  gradient_accumulation_steps: 3
  num_workers: 4
  gradient_clip_val: 1.0
  precision: bf16-mixed
  split_ratios: [0.85, 0.10, 0.05]

meanflow:
  p: 2.0
  adaptive: true
  norm_eps: 0.01
  norm_p: 0.5                         # CHANGED from 1.0
  prediction_type: x
  t_min: 0.05
  jvp_strategy: exact
  fd_step_size: 0.001

alpha_flow:
  start_step: 999999999               # Never start → α=0 always (pure FM)
  end_step: 999999999
  eta: 0.0
  mode: sigmoid

time_sampling:
  distribution: logit_normal
  mu: -0.4
  sigma: 1.0
  t_min: 0.001
  data_proportion: 1.0                # CHANGED: 100% FM in Stage 1

ema:
  decays: [0.999, 0.9995, 0.9999]
  active_index: 2

sample_collector:
  enabled: true
  collect_every_n_epochs: 100
  n_samples: 2
  nfe_steps: [1, 2, 5, 10, 50]
  seed: 42
```

**File: `configs/train_v2_stage2.yaml`**

```yaml
# NeuroMF v2 Stage 2: α-Flow MF Fine-tuning
# Purpose: Learn MeanFlow consistency via progressive curriculum
# Initialisation: Stage 1 best checkpoint

unet:
  # ... same architecture as Stage 1 ...
  use_v_head: false
  initialization:
    method: resume

training:
  max_epochs: 1000
  resume_from: "${paths.output}/v2_stage1/checkpoints/best.ckpt"
  lr: 1.0e-4
  lr_schedule: constant
  warmup_steps: 500                    # Short warmup for Stage 2
  betas: [0.9, 0.95]
  weight_decay: 0.0
  # ... rest same as Stage 1 ...

meanflow:
  p: 2.0
  adaptive: true
  norm_eps: 0.01
  norm_p: 0.5
  prediction_type: x
  t_min: 0.05
  jvp_strategy: exact

alpha_flow:
  start_step: 0                        # Start annealing immediately
  end_step: 42000                      # ~1000 epochs × 42 steps/epoch
  gamma: 25.0
  eta: 1.0                            # Full MF at convergence
  mode: sigmoid

time_sampling:
  distribution: logit_normal
  mu: -0.4
  sigma: 1.0
  t_min: 0.001
  data_proportion: 0.5                 # 50% FM, 50% MF (for MF samples)

ema:
  decays: [0.999, 0.9995, 0.9999]
  active_index: 2

sample_collector:
  enabled: true
  collect_every_n_epochs: 50
  n_samples: 2
  nfe_steps: [1, 2, 5, 10, 50]
  seed: 42
```

---

## 10. Verification Tests

The agent MUST implement and pass these tests before launching training.

### 10.1 Unit Tests

| Test ID | Description | Pass Criterion | File |
|---------|-------------|----------------|------|
| V2-T1 | AlphaScheduler sigmoid correctness | α(start) ≈ 0, α(end) ≈ η, monotonic | `tests/test_alpha_scheduler.py` |
| V2-T2 | AlphaScheduler at α=0, all r≈t | `max(|r - t|) < 1e-6` for B=1000 | `tests/test_alpha_scheduler.py` |
| V2-T3 | Boundary condition v_tilde shape | `v_tilde.shape == u.shape` | `tests/test_boundary_condition.py` |
| V2-T4 | Boundary condition under no_grad | `v_tilde.requires_grad == False` | `tests/test_boundary_condition.py` |
| V2-T5 | MultiEMA different shadows | After 100 updates, shadow[0] ≠ shadow[2] | `tests/test_multi_ema.py` |
| V2-T6 | MultiEMA memory footprint | Total EMA memory < 3 GB | `tests/test_multi_ema.py` |
| V2-T7 | Constant LR after warmup | `lr == 1e-4` at step 5000 | `tests/test_lr_schedule.py` |
| V2-T8 | norm_p=0.5 changes loss dynamics | `loss_mean ≠ 2.0` after 10 steps on random data | `tests/test_adaptive_weighting.py` |
| V2-T9 | Full forward-backward pass | No NaN, loss is finite | `tests/test_v2_smoke.py` |
| V2-T10 | rflow checkpoint loading | ≥ 402 keys loaded, no shape mismatches | `tests/test_rflow_loading.py` |

### 10.2 Integration Smoke Test

Before launching on Picasso, run a 5-epoch smoke test on the local RTX 4060 (8 GB) with `batch_size: 1`, `gradient_accumulation_steps: 1`, a subset of 100 latents:

```bash
~/.conda/envs/neuromf/bin/python experiments/cli/train.py \
    --config configs/train_v2_stage1.yaml \
    training.max_epochs=5 \
    training.batch_size=1 \
    training.gradient_accumulation_steps=1 \
    data.subset_size=100
```

**Pass criteria:**
- No OOM on 8 GB GPU
- Loss decreases over 5 epochs
- `train/alpha` logged as 0.0 (Stage 1)
- EMA update runs without error
- Sample collection at epoch 5 produces valid latents

---

## 11. External Dependencies

### 11.1 Existing (no changes)

- `torch >= 2.1` (for `torch.func.jvp`)
- `pytorch-lightning >= 2.0`
- `monai >= 1.3` (for `DiffusionModelUNet`)
- `omegaconf`

### 11.2 New

- **None.** All changes use standard PyTorch. No new packages needed.

### 11.3 Reference Codebases to Study

| Codebase | URL | What to Extract |
|----------|-----|-----------------|
| α-Flow | `https://github.com/snap-research/alphaflow` | Sigmoid schedule implementation, α-sampling logic |
| iMF | `https://github.com/Lyy-iiis/imeanflow` | Boundary condition implementation, Algorithm 1 |
| pMF | `src/external/pmf/` (already in repo) | Adaptive weighting with norm_p, x-prediction |

---

## 12. File Change Summary

### 12.1 New Files

| File | Purpose |
|------|---------|
| `src/neuromf/utils/alpha_scheduler.py` | α-Flow curriculum scheduler |
| `configs/train_v2_stage1.yaml` | Stage 1 config (FM pretraining) |
| `configs/train_v2_stage2.yaml` | Stage 2 config (α-Flow MF fine-tuning) |
| `experiments/cli/calibrate_gamma.py` | Post-training γ calibration |
| `tests/test_alpha_scheduler.py` | α scheduler unit tests |
| `tests/test_boundary_condition.py` | Boundary condition tests |
| `tests/test_multi_ema.py` | MultiEMA tests |
| `tests/test_v2_smoke.py` | Full integration smoke test |

### 12.2 Modified Files

| File | Changes |
|------|---------|
| `src/neuromf/utils/time_sampler.py` | Add `sample_alpha_flow()` method |
| `src/neuromf/utils/ema.py` | Add `MultiEMATracker` class |
| `src/neuromf/wrappers/maisi_unet.py` | Remove v-head construction when `use_v_head=false` |
| `src/neuromf/wrappers/meanflow_loss.py` | Use boundary condition for v_tilde; remove v-head loss branches |
| `src/neuromf/models/latent_meanflow.py` | Integrate α scheduler, MultiEMA, constant LR, boundary condition |
| `src/neuromf/sampling/one_step.py` | Add γ norm correction parameter |

### 12.3 Unchanged Files (verify still work)

| File | Reason |
|------|--------|
| `src/neuromf/losses/lp_loss.py` | norm_p change is config-only |
| `src/neuromf/losses/meanflow_jvp.py` | JVP math unchanged |
| `src/neuromf/wrappers/jvp_strategies.py` | Strategy abstraction unchanged |
| `src/neuromf/data/` | Data pipeline unchanged |
| `src/neuromf/callbacks/` | Callbacks unchanged |

---

## 13. Training Timeline

| Phase | Duration | Compute | Expected Outcome |
|-------|----------|---------|------------------|
| Implementation + testing | 2–3 days | Local RTX 4060 | All V2-T* tests pass |
| Stage 1 (FM pretraining) | ~5 days | 1× A100 40GB | FM loss < 100K, cos(v,vc) > 0.7 |
| Stage 2 (α-Flow MF) | ~10 days | 1× A100 40GB | cos(V,vc) > 0.5, FID-3D < 30 at 1-NFE |
| γ calibration + eval | 1 day | 1× A100 | Final metrics across NFE sweep |
| **Total** | **~18 days** | | |

---

## 14. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Stage 1 doesn't converge | rflow init should ensure fast convergence; if not, increase LR to 5e-4 |
| α-Flow annealing too fast/slow | Log `train/alpha` and `cos(V,vc)` every step; adjust γ mid-training if needed |
| OOM with boundary condition (extra forward pass) | Reduce batch_size from 3 to 2; increase gradient_accumulation_steps from 3 to 4 |
| Boundary condition worse than v-head | v1 v-head was 228K params; boundary condition uses 178M. Literature shows boundary > aux head. If somehow worse, revert to v-head with more ResBlocks (3-5) |
| Full MF (η=1) at convergence is wrong for our setting | α-Flow finds optimal η=5e-3 on ImageNet; add η as a sweep parameter in Stage 2 |
