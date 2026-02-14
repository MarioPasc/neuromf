# Phase 3: MeanFlow Loss Integration with 3D UNet

**Depends on:** Phase 2 (gate must be OPEN)
**Modules touched:** `src/neuromf/wrappers/`, `src/neuromf/losses/`, `src/neuromf/utils/`, `tests/`, `configs/`
**Estimated effort:** 3–4 sessions (hardest engineering phase)

---

## 1. Objective

Adapt MAISI's 3D UNet to accept MeanFlow's dual time conditioning $(r, t)$, ensure compatibility with `torch.func.jvp`, implement the full MeanFlow loss pipeline with per-channel $L_p$ loss, and verify gradient flow end-to-end. **This is the most challenging engineering phase** due to potential incompatibilities between the UNet's in-place operations and forward-mode AD.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §1.3 MeanFlow Training Objective (Eqs. 9–12)

**Compound prediction:**
$$
V_\theta(\mathbf{z}_t, r, t) = \mathbf{u}_\theta(\mathbf{z}_t, r, t) + (t - r) \cdot \text{sg}\!\left[\frac{\partial \mathbf{u}_\theta}{\partial \mathbf{z}_t} \cdot \tilde{\mathbf{v}}_\theta + \frac{\partial \mathbf{u}_\theta}{\partial t}\right]
\tag{9}
$$

$$
\tilde{\mathbf{v}}_\theta(\mathbf{z}_t, t) = \mathbf{u}_\theta(\mathbf{z}_t, t, t)
\tag{10}
$$

$$
\text{JVP}\!\left(\mathbf{u}_\theta,\; (\mathbf{z}_t, r, t),\; (\tilde{\mathbf{v}}_\theta, 0, 1)\right) = \frac{\partial \mathbf{u}_\theta}{\partial \mathbf{z}_t} \cdot \tilde{\mathbf{v}}_\theta + \frac{\partial \mathbf{u}_\theta}{\partial t} \cdot 1
\tag{11}
$$

$$
\mathcal{L}_{\text{MF}} = \mathbb{E}_{t, r, \mathbf{x}, \boldsymbol{\epsilon}} \left[ w(t) \cdot \|V_\theta(\mathbf{z}_t, r, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|_p^p \right]
\tag{12}
$$

### §1.4 iMF Combined Loss (Eq. 13)

$$
\mathcal{L}_{\text{iMF}} = \mathbb{E}_{t, r, \mathbf{x}, \boldsymbol{\epsilon}} \left[ w(t) \cdot \|\tilde{\mathbf{v}}_\theta(\mathbf{z}_t, t) - \mathbf{v}_c(\mathbf{z}_t, t)\|_p^p \right] + \lambda_{\text{MF}} \cdot \mathcal{L}_{\text{MF}}
\tag{13}
$$

### §1.6 x-Prediction Reparameterisation (Eqs. 15–16)

$$
\hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t) \triangleq \mathbf{z}_t - t \cdot \mathbf{u}_\theta(\mathbf{z}_t, r, t)
\tag{15}
$$

$$
\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{\mathbf{z}_t - \hat{\mathbf{x}}_\theta(\mathbf{z}_t, r, t)}{t}
\tag{16}
$$

### §2.2 MeanFlow Identity in Latent Space (Eq. 18, Props. 1–2)

$$
\mathbf{v}(\mathbf{z}_t, t) = \mathbf{u}(\mathbf{z}_t, r, t) + (t - r)\left[\frac{\partial \mathbf{u}}{\partial \mathbf{z}_t} \cdot \mathbf{v}(\mathbf{z}_t, t) + \frac{\partial \mathbf{u}}{\partial t}\right]
\tag{18}
$$

**Proposition 1 (Computational tractability).** JVP complexity $O(d \cdot C_{\text{net}})$ where $d = 4 \times 32^3 = 131{,}072$ in latent space — a $16\times$ reduction vs pixel space.

**Proposition 2 (Memory reduction).** UNet operates on $32^3$ feature maps (vs $128^3$), yielding ~$64\times$ reduction in activation memory per JVP pass.

### §2.3 Latent x-Prediction (Eqs. 19–22)

$$
\hat{\mathbf{z}}_{0,\theta}(\mathbf{z}_t, r, t) = \text{net}_\theta(\mathbf{z}_t, r, t)
\tag{19}
$$

$$
\mathbf{u}_\theta(\mathbf{z}_t, r, t) = \frac{\mathbf{z}_t - \hat{\mathbf{z}}_{0,\theta}(\mathbf{z}_t, r, t)}{t}
\tag{20}
$$

$$
\hat{\mathbf{z}}_0 = \boldsymbol{\epsilon} - 1 \cdot \mathbf{u}_\theta(\boldsymbol{\epsilon}, 0, 1) = \hat{\mathbf{z}}_{0,\theta}(\boldsymbol{\epsilon}, 0, 1)
\tag{21}
$$

$$
\hat{\mathbf{x}} = \mathcal{D}_\phi(\hat{\mathbf{z}}_0)
\tag{22}
$$

### §3.4 Per-Channel $L_p$ in MeanFlow Objective (Eq. 28)

$$
\mathcal{L}_{\text{MF-}L_p} = \mathbb{E}_{t,r,\mathbf{z}_0,\boldsymbol{\epsilon}} \left[ w(t) \cdot \sum_{c=1}^{C} \lambda_c \left( \frac{1}{|H'W'D'|} \sum_{h,w,d} |V_{\theta,c}(\mathbf{z}_t) - v_{c,c}(\mathbf{z}_t)|^{p_c} \right) \right]
\tag{28}
$$

## 3. External Code to Leverage

### `src/external/MeanFlow/` and `src/external/MeanFlow-PyTorch/`
- **Loss computation pattern:** How they compute JVP, apply stop-gradient, handle r=t case
- **What to extract:** JVP computation pattern, compound prediction formula

### `src/external/pmf/`
- **x-prediction implementation:** How they derive u from x-pred, adaptive weighting

### `src/external/NV-Generate-CTMR/`
- **UNet architecture:** How to load and configure the MAISI 3D UNet
- **Time conditioning:** Original single-timestep embedding approach
- **What to extract:** UNet class, weight loading, time embedding module
- **What to AVOID:** The diffusion training loop (we replace it with MeanFlow)

## 4. Implementation Specification

### `src/neuromf/wrappers/maisi_unet.py`
- **Purpose:** MAISI 3D UNet adapted for MeanFlow dual time conditioning.
- **Key modifications:**
  1. Accept $(r, t)$ instead of single $t$
  2. Two separate sinusoidal embeddings → sum (following pMF)
  3. Replace any in-place operations for `torch.func.jvp` compatibility
  4. x-prediction output head (same shape as input: 4 channels)
- **Key class:**
  ```python
  class MAISIUNetWrapper(nn.Module):
      def __init__(self, config: MAISIUNetConfig) -> None: ...
      def forward(self, z_t: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor: ...
  ```

### `src/neuromf/wrappers/meanflow_loss.py`
- **Purpose:** Complete MeanFlow loss computation wrapping JVP + compound prediction + iMF.
- **Key class:**
  ```python
  class MeanFlowLoss(nn.Module):
      def __init__(self, config: MeanFlowLossConfig) -> None: ...
      def forward(self, model, z_0, eps, t, r) -> dict[str, torch.Tensor]: ...
  ```

### `src/neuromf/losses/combined_loss.py`
- **Purpose:** iMF-style combined FM + MF loss (Eq. 13).
- **Key function:**
  ```python
  def combined_imf_loss(fm_loss, mf_loss, lambda_mf: float = 1.0) -> torch.Tensor: ...
  ```

### `src/neuromf/utils/ema.py`
- **Purpose:** Exponential Moving Average for model parameters.
- **Key class:**
  ```python
  class EMA:
      def __init__(self, model: nn.Module, half_lives: list[int], update_every: int = 1) -> None: ...
      def update(self) -> None: ...
      def get_model(self, idx: int = 0) -> nn.Module: ...
  ```

### `configs/train_meanflow.yaml`
- **Update** with full model, training, meanflow, and ema config sections (see tech guide §6.2)

## 5. Data and I/O

- **Input:** Random tensors for unit testing; pre-computed latents for integration testing
- **Tensor shapes:** `z_t: (B, 4, 32, 32, 32)`, `r: (B,)`, `t: (B,)`, output: `(B, 4, 32, 32, 32)`
- **Memory estimate (bf16):** ~7.5 GB for batch=24 (see tech guide Appendix B)

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P3-T1 | 3D UNet accepts $(r, t)$ conditioning | Forward pass without error on random input | CRITICAL | Unit test with random `(B, 4, 32, 32, 32)` |
| P3-T2 | UNet output shape matches input latent shape | `out.shape == (B, 4, 32, 32, 32)` | CRITICAL | Shape assertion |
| P3-T3 | `torch.func.jvp` executes on UNet | No error; JVP output shape matches latent shape | CRITICAL | Test `torch.func.jvp` on wrapped UNet |
| P3-T4 | MeanFlow loss is finite and positive | `0 < loss < 1000` on random data | CRITICAL | Forward pass test |
| P3-T5 | MeanFlow loss gradient flows to UNet params | `all(p.grad is not None for p in model.parameters() if p.requires_grad)` | CRITICAL | Check after `loss.backward()` |
| P3-T6 | Per-channel $L_p$ loss computes correctly for $p \in \{1.0, 1.5, 2.0, 3.0\}$ | Matches hand-computed reference on small tensor | CRITICAL | Compare with manual numpy computation |
| P3-T7 | Logit-normal time sampler produces correct distribution | KS-test vs. theoretical CDF, p-value > 0.05 on 10k samples | CRITICAL | `scipy.stats.kstest` against logit-normal CDF |
| P3-T8 | EMA updates correctly | EMA params differ from model params after updates | CRITICAL | Unit test: update EMA, check params diverge |
| P3-T9 | Combined iMF loss (FM + MF) computes without error | Finite loss | CRITICAL | Unit test with random data |
| P3-T10 | JVP and gradients work with bf16 mixed precision | No NaN; loss finite | CRITICAL | Test under `torch.autocast` |

**Suggested test files:** `tests/test_meanflow_loss.py`, `tests/test_ema.py` (plus updates to `tests/test_lp_loss.py`, `tests/test_time_sampler.py`)

**Phase 3 is PASSED when ALL of P3-T1 through P3-T10 are green.**

## 7. Expected Outputs

- `src/neuromf/wrappers/maisi_unet.py`
- `src/neuromf/wrappers/meanflow_loss.py`
- `src/neuromf/losses/combined_loss.py`
- `src/neuromf/utils/ema.py`
- `configs/train_meanflow.yaml` (updated)
- `tests/test_meanflow_loss.py` (updated with P3 tests)
- `tests/test_ema.py`
- `experiments/phase_3/verification_report.md`

## 8. Failure Modes and Mitigations

From tech guide §5.4 and Appendix C:

| Risk | Symptom | Likelihood | Mitigation |
|---|---|---|---|
| **In-place ops in UNet** | `torch.func.jvp` raises `RuntimeError` | Medium | Replace `F.relu(x, inplace=True)` with `F.relu(x, inplace=False)` in wrapper; replace `+=` with `x = x + y` |
| **JVP NaN at small $t$** | Loss becomes NaN near $t=0$ | Medium | Clip $t \geq 0.05$; check $1/t$ division in x-pred → u conversion |
| **Memory OOM on JVP** | CUDA OOM during JVP forward | Medium | Reduce batch size; use gradient checkpointing in UNet |
| **FlashAttention incompatibility** | JVP fails at attention layers | Medium | Disable FlashAttention: `torch.backends.cuda.flash_sdp_enabled = False` |
| **MAISI UNet incompatible with `torch.func.jvp`** | Persistent JVP errors | Medium | Use finite-difference approximation as fallback: JVP ≈ (u(z+h·v, r, t+h) - u(z, r, t))/h with h=10^{-3}. Less efficient (2 forward passes) but always works |

# Phase 3: MeanFlow Loss Integration with 3D UNet — Agent Prompt

**Depends on:** Phase 2 (PASSED — all ablations complete, x-prediction validated)
**GPU:** RTX 4060 8GB VRAM (engineering validation only — production training on A100)
**Results:** `/media/mpascual/Sandisk2TB/research/neuromf/results/phase_3/`

---

## ⚠️ CRITICAL CORRECTIONS TO PHASE 3 SPEC

Before starting, read these corrections to `docs/splits/phase_3.md`. The original spec contains errors that will cause failures.

### Correction 1: Latent Shape is 4×48×48×48, NOT 4×32×32×32

The resolution analysis (`docs/data/resolution_analysis.md`) decided on **192³ voxels at 1.0mm isotropic** with the MAISI VAE downsample factor of 4, producing latents of shape `(4, 48, 48, 48)`. Every reference to `4×32×32×32` in the Phase 3 spec is outdated. The correct shapes are:

| Tensor | Shape |
|--------|-------|
| Input latent z_0 | `(B, 4, 48, 48, 48)` |
| Noise epsilon | `(B, 4, 48, 48, 48)` |
| Interpolated z_t | `(B, 4, 48, 48, 48)` |
| UNet output | `(B, 4, 48, 48, 48)` |
| Time tensors r, t | `(B,)` |

The latent dimensionality is $D = 4 \times 48^3 = 442{,}368$.

### Correction 2: Memory Estimates are Wrong — 8GB VRAM is Insufficient for Full-Size Training

The Appendix B estimates were for `4×32³` latents. At `4×48³` (3.4× more voxels), the memory profile changes dramatically:

| Component | 4×32³ (bf16) | 4×48³ (bf16) | Factor |
|-----------|-------------|-------------|--------|
| UNet params (180M) | 360 MB | 360 MB | 1× |
| Latent batch (B=1) | 0.25 MB | 0.85 MB | 3.4× |
| UNet activations (fwd) | ~2 GB | ~6.8 GB | ~3.4× |
| JVP primal tape | ~2 GB | ~6.8 GB | ~3.4× |
| **Total (B=1, fwd+JVP)** | **~4.6 GB** | **~14 GB** | — |

**Conclusion:** Even batch=1 with `torch.func.jvp` on full 4×48³ latents DOES NOT FIT in 8GB VRAM.

### Correction 3: x-Prediction is the Default (Phase 2 Validated This)

Phase 2 conclusively demonstrated that **x-prediction is essential at high ambient dimensionality**. At D=256, u-prediction collapsed (MMD=0.77, coverage=0.0%) while x-prediction maintained quality (MMD=0.003, coverage=72.9%). Since our target is D=442,368, x-prediction is mandatory, not optional.

All implementations should use **x-prediction by default**:

```python
# The network outputs x-prediction: x_hat = net(z_t, r, t)
# The average velocity is derived as: u = (z_t - x_hat) / t
prediction_type: str = "x"  # DEFAULT — not "u"
```

### Correction 4: We DO NOT Use the MAISI Diffusion UNet Weights

From `docs/data/checkpoint_exploration.md`: the MAISI diffusion UNet (`diff_unet_3d_rflow-mr.pt`, 180M params) was trained with standard rectified flow on a different dataset (mixed CT/MR) with a single-timestep conditioning. We train the NeuroMF MeanFlow model from scratch because:

1. The time conditioning changes from single $t$ to dual $(r, t)$ — the time embedding MLP weights are incompatible
2. The training objective changes from FM to MeanFlow — learned velocity fields are different
3. The original model was trained on mixed CT/MR, not brain-specific data

We use the **same MONAI DiffusionModelUNet architecture** but with **random initialisation** and our custom dual-time conditioning wrapper.

---

## 0. Strategy: Two-Stage Testing

Given the 8GB VRAM constraint, Phase 3 uses a two-stage approach:

### Stage A: Engineering Validation (RTX 4060, 8GB)

Test ALL components with **reduced spatial dimensions** to fit in VRAM:

| Test Shape | Purpose | Estimated VRAM (bf16) |
|------------|---------|----------------------|
| `(2, 4, 12, 12, 12)` | Unit tests: forward, JVP, loss, gradients | ~1.5 GB |
| `(1, 4, 24, 24, 24)` | Integration test: full loss pipeline | ~3 GB |
| `(1, 4, 48, 48, 48)` | Smoke test: single forward pass only (no JVP) | ~7 GB |

All P3-T1 through P3-T10 tests run at reduced resolution. This validates algorithmic correctness.

### Stage B: Real-Data Smoke Test (RTX 4060, 8GB)

Load 5 real latent files from `/media/mpascual/Sandisk2TB/research/neuromf/results/latents/` and verify:
- They load correctly and have shape `(4, 48, 48, 48)`
- A single forward pass through the wrapped UNet produces correct output shape
- Latent statistics match the Phase 1 report (per-channel mean ≈ 0, std ≈ 1.0)

For the full-size forward+JVP+loss pipeline at `4×48³`, we use ONE of these strategies to fit in 8GB:

**Strategy 1 (Preferred): Finite-Difference JVP Approximation**

$$\text{JVP} \approx \frac{\mathbf{u}_\theta(\mathbf{z}_t + h \cdot \tilde{\mathbf{v}}_\theta, r, t + h) - \mathbf{u}_\theta(\mathbf{z}_t, r, t)}{h}, \quad h = 10^{-3}$$

This requires 2 forward passes instead of 1 JVP. Each forward pass uses ~6.8 GB at batch=1. The two passes are sequential (not simultaneous), so peak VRAM ≈ 7.2 GB — fits in 8GB.

**Strategy 2: Gradient Checkpointing + Reduced Batch**

Enable gradient checkpointing in the UNet (`use_checkpointing: true`) and try `torch.func.jvp` at batch=1. This trades compute for memory. May or may not fit.

**Strategy 3: Patch-Based JVP**

Split the 48³ volume into overlapping 24³ patches, compute JVP per patch, stitch. Complex but always fits. Only use as last resort.

**THE AGENT MUST TRY Strategy 1 (finite-difference) FIRST for full-size testing.** If `torch.func.jvp` works with gradient checkpointing, prefer that for final training on A100.

---

## 1. What Phase 2 Validated (Context for the Agent)

The following components from `src/neuromf/` are **already validated** by Phase 2 and should be reused directly:

| Module | Status | Phase 3 Action |
|--------|--------|----------------|
| `src/neuromf/losses/meanflow_jvp.py` | ✅ Validated on MLP | Adapt for 3D UNet (same math, different model) |
| `src/neuromf/losses/lp_loss.py` | ✅ Validated for p∈{1,1.5,2,3} | Add per-channel weighting (Eq. 28) |
| `src/neuromf/sampling/one_step.py` | ✅ Validated geometrically | Works as-is on any tensor shape |
| `src/neuromf/sampling/multi_step.py` | ✅ Validated | Works as-is |
| `src/neuromf/utils/time_sampler.py` | ✅ Validated | Works as-is |
| `src/neuromf/utils/ema.py` | ✅ Validated on toy | Works on any nn.Module |
| `src/neuromf/metrics/mmd.py` | ✅ Validated | Works as-is |

**Do NOT rewrite these modules.** Only extend them where needed (e.g., adding per-channel weighting to `lp_loss.py`).

### Key Phase 2 Findings to Carry Forward

1. **x-prediction is mandatory** at high D (see Ablation B results)
2. **data_proportion=0.25 was optimal** for 1-NFE quality on the torus (better than the iMF default of 0.75)
3. **Lp=2.0 is a safe default** (p=3.0 was marginally better on torus but may not transfer to MRI latents)
4. **Adaptive weighting works** — keep `norm_eps=0.01`
5. **EMA decay=0.999** was used and is standard
6. **logit-normal(μ=-0.4, σ=1.0)** is the time sampling distribution (matching iMF paper)

---

## 2. MAISI UNet Architecture Details

### 2.1 Source Code Location

The MAISI UNet is imported from MONAI:
```python
from monai.networks.nets import DiffusionModelUNet
```

The MOTFM external code (`src/external/MOTFM/`) provides a **directly relevant pattern** for wrapping MONAI's `DiffusionModelUNet` with flow matching. Study `src/external/MOTFM/utils/utils_fm.py` → `MergedModel` class, which shows:
- How to convert continuous $t \in [0, 1]$ to the UNet's expected integer timestep format
- How to handle conditioning
- How to forward through the UNet

### 2.2 Architecture Configuration (from NV-Generate-CTMR configs)

```python
# Based on the MAISI rflow-MR configuration
unet_config = {
    "spatial_dims": 3,
    "in_channels": 4,           # latent channels
    "out_channels": 4,          # x-prediction: same as input
    "num_channels": [64, 128, 256, 512],  # progressive widths
    "attention_levels": [False, False, True, True],
    "num_res_blocks": [2, 2, 2, 2],
    "num_head_channels": [64, 128, 256, 512],
    "norm_num_groups": 32,
    "resblock_updown": True,
    "transformer_num_layers": 1,
    "use_flash_attention": False,  # MUST be False for torch.func.jvp
    "with_conditioning": False,    # unconditional for now
}
```

**CRITICAL: `use_flash_attention` MUST be `False`** — FlashAttention kernels are incompatible with `torch.func.jvp` forward-mode AD.

### 2.3 Dual Time Conditioning Implementation

The MONAI `DiffusionModelUNet` accepts a `timesteps: torch.Tensor` argument. For MeanFlow's dual $(r, t)$ conditioning, the wrapper should:

1. Create two separate sinusoidal time embedding layers (sharing architecture but NOT weights)
2. Each processes its respective time input through: sinusoidal_embedding → Linear → SiLU → Linear
3. Sum the two embeddings: `emb = emb_r + emb_t`
4. Pass the summed embedding to the UNet's internal conditioning pathway

Following pMF (Lu et al., 2026), the sum operation is preferred over concatenation.

**Implementation approach:** Rather than modifying the UNet internals, create a wrapper that:
1. Has its own dual time embedding layers
2. Calls the inner UNet's forward with the summed embedding
3. Handles the x-prediction → u-prediction conversion in the JVP closure

```python
class MAISIUNetWrapper(nn.Module):
    """MAISI 3D UNet adapted for MeanFlow dual time conditioning.
    
    Args:
        unet_config: Configuration dict for DiffusionModelUNet.
        prediction_type: "x" for x-prediction (default), "u" for u-prediction.
        t_min: Minimum time clamp to avoid 1/t singularity in x-pred.
    """
    def __init__(self, unet_config: dict, prediction_type: str = "x",
                 t_min: float = 0.05) -> None: ...
    
    def forward(self, z_t: torch.Tensor, r: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
        """Forward pass returning x-prediction or u-prediction.
        
        Args:
            z_t: Noisy latent (B, 4, 48, 48, 48).
            r: Interval start time (B,).
            t: Interval end time (B,).
            
        Returns:
            If prediction_type="x": x_hat of shape (B, 4, 48, 48, 48)
            If prediction_type="u": u of shape (B, 4, 48, 48, 48)
        """
    
    def u_from_x(self, z_t: torch.Tensor, x_pred: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """Convert x-prediction to average velocity u.
        
        u = (z_t - x_pred) / max(t, t_min)
        """
```

### 2.4 In-Place Operation Remediation

The MONAI `DiffusionModelUNet` likely contains in-place operations. The agent must:

1. **First:** Try `torch.func.jvp` on the wrapped model with a small random input
2. **If RuntimeError about in-place ops:** Identify which layers cause it using the traceback
3. **Fix strategy:** Create a monkey-patching function that replaces in-place ops:

```python
def patch_inplace_ops(module: nn.Module) -> None:
    """Recursively replace in-place ReLU/SiLU/GELU with out-of-place variants."""
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        elif isinstance(child, nn.SiLU):
            setattr(module, name, nn.SiLU(inplace=False))  
        # Also check for += patterns in forward methods
        patch_inplace_ops(child)
```

4. **If persistent issues:** Some MONAI layers may use `x += residual` in forward(). These require subclassing and overriding forward to use `x = x + residual`.

---

## 3. Verification Tests (Updated for 8GB VRAM)

All tests use **reduced spatial dimensions** unless noted.

| Test ID | Description | Test Shape | Pass Criterion |
|---------|-------------|------------|----------------|
| P3-T1 | UNet accepts dual (r, t) conditioning | `(2, 4, 12, 12, 12)` | Forward pass without error |
| P3-T2 | Output shape matches input | `(2, 4, 12, 12, 12)` | `out.shape == in.shape` |
| P3-T3a | `torch.func.jvp` works on small input | `(1, 4, 12, 12, 12)` | JVP output correct shape, no error |
| P3-T3b | JVP matches finite-difference approx | `(1, 4, 12, 12, 12)` | Relative error < 0.01 (h=1e-3) |
| P3-T4 | MeanFlow loss finite and positive | `(2, 4, 12, 12, 12)` | `0 < loss < 1000` |
| P3-T5 | Gradients flow to all params | `(2, 4, 12, 12, 12)` | All `requires_grad` params have `.grad` |
| P3-T6 | Per-channel Lp loss correct | `(2, 4, 8, 8, 8)` | Matches manual numpy computation |
| P3-T7 | Time sampler distribution | N/A (CPU) | KS test p > 0.05 on 10k samples |
| P3-T8 | EMA updates correctly | `(2, 4, 12, 12, 12)` | EMA params differ after updates |
| P3-T9 | Combined iMF loss computes | `(2, 4, 12, 12, 12)` | Finite loss |
| P3-T10 | bf16 mixed precision works | `(1, 4, 12, 12, 12)` | No NaN under `torch.autocast` |
| P3-T11 | **Full-size forward pass** | `(1, 4, 48, 48, 48)` | Forward pass completes, output shape correct |
| P3-T12 | **Full-size finite-diff JVP** | `(1, 4, 48, 48, 48)` | FD-JVP completes within 8GB VRAM |
| P3-T13 | **Real latent smoke test** | 5 files from disk | Load, forward, correct shape, finite output |

**P3-T3b is new** (recommended in Phase 2 spec review): verifies the JVP implementation against finite differences, catching subtle tangent-vector construction bugs that shape-matching alone would miss.

**P3-T11, T12, T13 are new**: validate that the full-size pipeline works on the actual hardware.

---

## 4. HTML Report with Figures

The agent should generate an HTML report at:
```
/media/mpascual/Sandisk2TB/research/neuromf/results/phase_3/report.html
```

### Report Structure

```
# Phase 3: MeanFlow Loss Integration — Verification Report

## 1. Executive Summary
- Pass/fail table for P3-T1 through P3-T13
- GPU: RTX 4060 8GB, CUDA version, PyTorch version, MONAI version

## 2. UNet Architecture Summary
- Table: layer counts, parameter counts per block, total params
- Figure: model architecture diagram (text-based is fine)

## 3. JVP Compatibility
- Which in-place ops were found and patched (if any)
- torch.func.jvp success/failure at each spatial resolution tested
- Figure: JVP vs finite-difference relative error histogram (P3-T3b)
- VRAM usage table per test shape

## 4. Loss Pipeline Verification
- MeanFlow loss value on random data (should be ~2.0, matching Phase 2 baseline)
- Per-channel Lp loss verification table (computed vs expected)
- Combined iMF loss decomposition (FM term + MF term)

## 5. Memory Profiling
- Table: VRAM usage for each test shape and operation
- Recommendation: batch size and strategy for A100 80GB training
- Figure: VRAM usage bar chart per test

## 6. Real Latent Smoke Test
- Loaded N latent files, shapes confirmed
- Per-channel statistics of loaded latents (mean, std)
- Forward pass output statistics
- Figure: histogram of UNet output values (sanity check — should not be all zeros or NaN)

## 7. Conclusions & Phase 4 Readiness
- All critical tests passed? → Phase 3 gate OPEN/CLOSED
- Recommended training config for Phase 4 (batch size, precision, JVP strategy)
```

---

## 5. Code Organisation

### 5.1 New Modules

```
src/neuromf/wrappers/
├── __init__.py
├── maisi_unet.py              # MAISIUNetWrapper (dual time conditioning)
├── meanflow_pipeline.py       # Full MeanFlow loss pipeline (wraps JVP + compound pred)
└── jvp_strategies.py          # torch.func.jvp + finite-difference fallback

src/neuromf/losses/
├── combined_loss.py           # iMF combined FM + MF loss (Eq. 13) — NEW
└── lp_loss.py                 # EXTEND with per-channel weighting (Eq. 28)

tests/
├── test_maisi_unet_wrapper.py # P3-T1, T2, T11
├── test_jvp_compatibility.py  # P3-T3a, T3b, T12
├── test_meanflow_pipeline.py  # P3-T4, T5, T9, T10
├── test_lp_loss_perchannel.py # P3-T6
└── test_real_latents.py       # P3-T13

experiments/
├── cli/
│   └── run_phase3_verification.py  # CLI: runs all tests, generates report
└── phase_3/
    └── report_generator.py         # HTML report generation
```

### 5.2 JVP Strategy Abstraction

Create an abstraction that Phase 4+ can swap between JVP methods:

```python
# src/neuromf/wrappers/jvp_strategies.py
class JVPStrategy(Protocol):
    def compute(self, u_fn, z_t, r, t, v_tangent) -> torch.Tensor: ...

class ExactJVP(JVPStrategy):
    """Uses torch.func.jvp — requires JVP-compatible model."""
    ...

class FiniteDifferenceJVP(JVPStrategy):
    """Finite-difference approximation — always works, 2× forward cost."""
    def __init__(self, h: float = 1e-3): ...
    ...
```

This lets us use `FiniteDifferenceJVP` on the RTX 4060 and `ExactJVP` on A100 without changing any other code.

---

## 6. Paths and Resources

| Resource | Path |
|----------|------|
| Pre-computed latents | `/media/mpascual/Sandisk2TB/research/neuromf/results/latents/` |
| Latent statistics | `/media/mpascual/Sandisk2TB/research/neuromf/results/latents/latent_stats.json` |
| MAISI VAE weights | `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt` |
| MAISI diffusion UNet (ref only) | `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt` |
| Phase 3 results | `/media/mpascual/Sandisk2TB/research/neuromf/results/phase_3/` |
| Project source | `src/neuromf/` |
| External code | `src/external/` |
| Phase 2 validated code | `src/neuromf/losses/`, `src/neuromf/sampling/`, `src/neuromf/utils/`, `src/neuromf/metrics/` |

### MONAI UNet Reference

Study these files for the UNet wrapping pattern:
1. **PRIMARY:** `src/external/MOTFM/utils/utils_fm.py` — `MergedModel` class shows how to wrap MONAI's DiffusionModelUNet for flow matching (continuous t→discrete timestep conversion, conditioning handling)
2. **SECONDARY:** `src/external/NV-Generate-CTMR/scripts/` — MAISI inference patterns
3. **REFERENCE:** `src/external/MeanFlow-PyTorch/meanflow.py` — JVP computation pattern (study `__call__` and `u_fn` methods)

---

## 7. Acceptance Criteria

Phase 3 is **PASSED** when:

1. ✅ All P3-T1 through P3-T10 pass at reduced spatial dimensions
2. ✅ P3-T11: Full-size forward pass (1, 4, 48, 48, 48) completes successfully
3. ✅ P3-T12: Full-size finite-difference JVP completes within 8GB VRAM
4. ✅ P3-T13: Real latent smoke test passes (5 files loaded, forwarded, finite outputs)
5. ✅ HTML report generated with all sections populated
6. ✅ JVP strategy abstraction implemented (ExactJVP + FiniteDifferenceJVP)
7. ✅ No hardcoded spatial dimensions — code works for any `(B, 4, D, H, W)`

Phase 3 is **NOT** about training. It is purely engineering validation. Training happens in Phase 4.

---

## 8. Failure Modes and Mitigations (Updated)

| Risk | Likelihood | Symptom | Mitigation |
|------|-----------|---------|------------|
| In-place ops in MONAI UNet | HIGH | `torch.func.jvp` RuntimeError | Monkey-patch in-place ops; see §2.4 |
| FlashAttention + JVP | HIGH | JVP errors at attention layers | Set `use_flash_attention=False` in config |
| VRAM OOM at 48³ with JVP | CERTAIN | CUDA OOM | Use FiniteDifferenceJVP (Strategy 1) |
| VRAM OOM at 48³ with FD-JVP | LOW | CUDA OOM | Ensure `torch.no_grad()` for both forward passes; `del` intermediates explicitly |
| GroupNorm + JVP incompatibility | MEDIUM | Incorrect gradients or NaN | Test with LayerNorm alternative; verify with finite diff |
| MONAI version mismatch | LOW | Import errors | Check `monai.__version__`, ensure ≥1.3.0 |
| bf16 precision issues | MEDIUM | NaN in loss at small t | Clamp t ≥ 0.05; compute 1/t in fp32 |

---

## 9. Implementation Order

The agent should implement in this order:

1. **Read external code** (30 min): Study MOTFM's `MergedModel`, MeanFlow-PyTorch's `meanflow.py`, and the existing `src/neuromf/wrappers/maisi_vae.py` for patterns
2. **MAISIUNetWrapper** (1 hr): Dual time conditioning wrapper around `DiffusionModelUNet`
3. **P3-T1, T2** (15 min): Basic forward pass tests at reduced resolution
4. **In-place op patching** (30 min–2 hr): Run `torch.func.jvp`, fix errors iteratively
5. **P3-T3a, T3b** (30 min): JVP correctness verification
6. **JVP strategy abstraction** (30 min): `ExactJVP` + `FiniteDifferenceJVP`
7. **MeanFlow pipeline** (1 hr): Full loss computation wrapping JVP + compound prediction
8. **P3-T4, T5, T9, T10** (30 min): Loss and gradient tests
9. **Per-channel Lp extension** (30 min): Add channel weighting to existing `lp_loss.py`
10. **P3-T6** (15 min): Lp verification
11. **P3-T11, T12** (30 min): Full-size tests
12. **P3-T13** (30 min): Real latent smoke test
13. **HTML report** (30 min): Generate verification report

**Total estimated: 6–10 hours of agent time** (the in-place op debugging at step 4 is the wildcard — could be 30 min or 3 hours depending on how many ops MONAI uses).