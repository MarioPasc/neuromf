# Phase 7: LoRA Fine-Tuning for FCD Joint Synthesis

**Depends on:** Phase 6 (gate must be OPEN)
**Modules touched:** `src/neuromf/models/`, `src/neuromf/data/`, `experiments/cli/`, `configs/`, `experiments/stage2_fcd/`
**Estimated effort:** 2–3 sessions

---

## 1. Objective

Test mask encoding strategies (VAE-encoded vs. downsampled), implement LoRA injection into the trained MeanFlow UNet, fine-tune with per-channel $L_p$ loss on FCD data for joint FLAIR image and segmentation mask synthesis, and evaluate synthesis quality. This phase addresses Contribution 4 of the paper.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §5.1 Low-Rank Adaptation — LoRA (Eqs. 34–36)

$$
\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B}\mathbf{A}, \quad \mathbf{B} \in \mathbb{R}^{d \times r},\; \mathbf{A} \in \mathbb{R}^{r \times k}
\tag{34}
$$

where $r \ll \min(d, k)$ is the rank, and $\alpha$ is a scaling factor. Only $\mathbf{A}$ and $\mathbf{B}$ are trained; $\mathbf{W}_0$ is frozen.

**Parameter count** for rank $r = 16$ applied to attention $q, k, v$ projections:

$$
|\theta_{\text{LoRA}}| = 3 \times L \times 2 \times r \times d_{\text{model}} = 6 L r d_{\text{model}}
\tag{35}
$$

For a typical 3D UNet with $L \approx 16$ attention layers and $d_{\text{model}} = 256$:

$$
|\theta_{\text{LoRA}}| = 6 \times 16 \times 16 \times 256 = 393{,}216 \approx 0.4\text{M}
\tag{36}
$$

This is $< 1\%$ of the full model parameters.

### §5.2 Domain Adaptation Strategy

**Stage 2 protocol:**
1. Freeze the base MeanFlow model (trained in Stage 1 on healthy brain MRI).
2. Modify the first convolutional layer to accept $C_{\text{in}} = 8$ (or 5) channels.
3. Apply LoRA to attention layers only.
4. Train with the per-channel $L_p$ loss (§3.4, Eq. 28–29).

### §5.3 Joint Image-Mask Synthesis (Eqs. 37–38)

**Strategy A (VAE-encoded mask):**

$$
\mathbf{z}_{\text{joint}} = [\mathcal{E}_\phi(\mathbf{x}_{\text{img}}),\; \mathcal{E}_\phi(\mathbf{m})] \in \mathbb{R}^{8 \times 32 \times 32 \times 32}
\tag{37}
$$

**Strategy B (Downsampled mask):**

$$
\mathbf{z}_{\text{joint}} = [\mathcal{E}_\phi(\mathbf{x}_{\text{img}}),\; \text{NN}_{\downarrow 4}(\mathbf{m})] \in \mathbb{R}^{5 \times 32 \times 32 \times 32}
\tag{38}
$$

**Decision criterion:** If Strategy A's Dice > 0.85, use Strategy A; otherwise Strategy B.

### §3.4 Per-Channel $L_p$ in Joint Synthesis (Eqs. 28–29)

$$
\mathcal{L}_{\text{MF-}L_p} = \mathbb{E}_{t,r,\mathbf{z}_0,\boldsymbol{\epsilon}} \left[ w(t) \cdot \sum_{c=1}^{C} \lambda_c \left( \frac{1}{|H'W'D'|} \sum_{h,w,d} |V_{\theta,c}(\mathbf{z}_t) - v_{c,c}(\mathbf{z}_t)|^{p_c} \right) \right]
\tag{28}
$$

$$
p_c = \begin{cases} p_{\text{img}} & \text{if } c \in \{1,2,3,4\} \text{ (image latent channels)} \\ p_{\text{mask}} & \text{if } c \in \{5,6,7,8\} \text{ (mask latent channels)} \end{cases}
\tag{29}
$$

### §5.4 Theoretical Justification

LoRA's effectiveness in MeanFlow fine-tuning is motivated by the observation that the velocity field for pathological brain MRI (with FCD lesions) differs from healthy brain MRI primarily in a low-dimensional subspace corresponding to the lesion region. The bulk of brain anatomy is shared between healthy and pathological populations.

## 3. External Code to Leverage

No new external code. LoRA implementation via the `peft` library (installed via pip).

## 4. Implementation Specification

### `src/neuromf/models/lora.py`
- **Purpose:** LoRA injection utilities for the MeanFlow UNet.
- **Key functions:**
  ```python
  def inject_lora(model: nn.Module, rank: int = 16, alpha: float = 16.0, target_modules: list[str] = None) -> nn.Module: ...
  def freeze_base_unfreeze_lora(model: nn.Module) -> None: ...
  def count_trainable_params(model: nn.Module) -> tuple[int, int]: ...
  ```
- **Dependencies:** `peft` library

### `src/neuromf/data/fcd_dataset.py`
- **Purpose:** FCD image+mask pair loading and encoding.
- **Key classes:**
  ```python
  class FCDDataset(Dataset):
      def __init__(self, data_dir: Path, mask_strategy: str = "vae", vae: MAISIVAEWrapper = None) -> None: ...
      def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: ...
  ```
- **Supports:** Both Strategy A (VAE-encoded, C=8) and Strategy B (downsampled, C=5)

### `experiments/cli/train_lora.py`
- **Purpose:** LoRA fine-tuning CLI.
- **Usage:**
  ```bash
  python experiments/cli/train_lora.py --config configs/lora_fcd.yaml --base_checkpoint best_ema.ckpt
  ```

### `configs/lora_fcd.yaml`
- **Key fields:**
  - `lora.rank`: 16
  - `lora.alpha`: 16.0
  - `lora.target_modules`: attention q, k, v projections
  - `data.mask_strategy`: "vae" or "downsample"
  - `loss.p_img`: 1.5 (or best from Phase 6 Lp sweep)
  - `loss.p_mask`: 2.0
  - `training.max_epochs`: 500
  - `training.lr`: 5.0e-5

## 5. Data and I/O

- **Input:** FCD FLAIR MRI volumes + binary segmentation masks, base MeanFlow checkpoint
- **Output:**
  - LoRA fine-tuned checkpoint
  - Mask strategy evaluation: `experiments/stage2_fcd/mask_vae_test.json`
  - Generated joint samples: `experiments/stage2_fcd/generated/`
  - Metrics: `experiments/stage2_fcd/metrics/metrics.json`
- **Tensor shapes:**
  - Strategy A: `(B, 8, 32, 32, 32)` — 4 image channels + 4 mask channels
  - Strategy B: `(B, 5, 32, 32, 32)` — 4 image channels + 1 mask channel

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P7-T1 | Mask VAE reconstruction Dice (Strategy A test) | Record Dice; select strategy based on threshold 0.85 | CRITICAL | Encode+decode binary masks, compute Dice |
| P7-T2 | LoRA parameters inject without error | Model forward pass succeeds with LoRA | CRITICAL | Unit test: inject LoRA, forward random input |
| P7-T3 | Only LoRA params have `requires_grad=True` | Count trainable params << total params | CRITICAL | Assert trainable < 1% of total |
| P7-T4 | Training loss decreases on FCD data | Loss at epoch 200 < epoch 1 | CRITICAL | Training log |
| P7-T5 | Generated FLAIR images show lesion-like features | Visual inspection | CRITICAL | Log sample images to wandb |
| P7-T6 | Generated masks have non-trivial overlap with real FCD masks | Mean Dice > 0.3 (data-scarce setting) | INFORMATIONAL | Compute Dice on generated masks |
| P7-T7 | Per-channel $L_p$ (1.5/2.0) outperforms uniform $L_2$ on mask Dice | Or report null result | INFORMATIONAL | Ablation comparison |

**Phase 7 is PASSED when P7-T1 through P7-T5 are ALL green.**

## 7. Expected Outputs

- `src/neuromf/models/lora.py`
- `src/neuromf/data/fcd_dataset.py`
- `experiments/cli/train_lora.py`
- `configs/lora_fcd.yaml`
- `experiments/stage2_fcd/mask_vae_test.json`
- `experiments/stage2_fcd/generated/` — joint synthetic volumes
- `experiments/stage2_fcd/metrics/metrics.json`
- `experiments/phase_7/verification_report.md`

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| FCD data too scarce for LoRA | Weakens Contribution 4 | Medium | Reduce LoRA rank; increase augmentation; use data from multiple epilepsy centers |
| Mask VAE reconstruction Dice < 0.85 | Switches to Strategy B | Medium | Strategy B is simpler and may work well enough |
| LoRA + JVP incompatibility | Blocks P7-T2 | Low | LoRA only modifies attention layers; should be compatible. If not, use manual LoRA instead of `peft` |
| Generated masks too blurry | Weakens P7-T6 | Medium | Increase p_mask to 3.0; post-process with threshold |
