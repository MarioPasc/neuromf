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
