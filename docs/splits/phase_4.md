# Phase 4: Training on Brain MRI Latents

**Depends on:** Phase 3 (gate must be OPEN)
**Modules touched:** `src/neuromf/models/`, `experiments/cli/`, `configs/`
**Estimated effort:** 2–3 sessions (includes GPU training time)

---

## 1. Objective

Train the latent MeanFlow model on FOMO-60K pre-computed latents (healthy controls from OASIS-1, OASIS-2, IXI), monitor training with loss curves and periodic sample generation, and select the best EMA checkpoint. This phase produces the core trained model that Phases 5–8 all depend on.

## 2. Theoretical Background

From `docs/main/methodology_expanded.md`:

### §2.5 The Complete Latent MeanFlow Training Algorithm

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

## 3. External Code to Leverage

No new external code. This phase uses the components built in Phases 2–3.

## 4. Implementation Specification

### `src/neuromf/models/latent_meanflow.py`
- **Purpose:** PyTorch Lightning module for Latent MeanFlow training.
- **Key class:**
  ```python
  class LatentMeanFlow(pl.LightningModule):
      def __init__(self, config: OmegaConf) -> None: ...
      def training_step(self, batch, batch_idx) -> torch.Tensor: ...
      def validation_step(self, batch, batch_idx) -> None: ...
      def configure_optimizers(self) -> dict: ...
      def on_train_epoch_end(self) -> None: ...  # periodic sampling
  ```
- **Dependencies:** `MAISIUNetWrapper`, `MeanFlowLoss`, `EMA`, `LatentDataset`, `one_step.sample_one_step`
- **Implements:** Algorithm 1 from methodology §2.5

### `experiments/cli/train.py`
- **Purpose:** General training CLI for Phases 3–4.
- **Usage:**
  ```bash
  python experiments/cli/train.py --config configs/train_meanflow.yaml
  ```
- **Features:** Resume from checkpoint, config overrides via CLI

### `configs/train_meanflow.yaml` (full specification)
From tech guide §6.2:
```yaml
model:
  architecture: "maisi_3d_unet"
  in_channels: 4
  out_channels: 4
  prediction_type: "x"        # x-prediction (primary); "u" for ablation
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
  loss_norm: 2.0
  imf_auxiliary_weight: 1.0
  adaptive_weight: true
  adaptive_eps: 1.0e-4

ema:
  half_lives: [500, 1000, 2000]
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

## 5. Data and I/O

- **Input:** Pre-computed `.pt` latents from Phase 1 (loaded via `LatentDataset`)
- **Output:**
  - Model checkpoints → `{checkpoint_dir}/`
  - EMA checkpoints → `{checkpoint_dir}/ema/`
  - Training logs → wandb or tensorboard
  - Periodic 1-NFE samples (decoded through VAE) → logged to wandb
  - `experiments/stage1_healthy/training_log.json`

## 6. Verification Tests

| Test ID | Description | Pass Criterion | Critical? | Implementation Hint |
|---|---|---|---|---|
| P4-T1 | Training starts without error | First 100 steps complete | CRITICAL | Check training log for exceptions |
| P4-T2 | Loss decreases over first 50 epochs | `loss[epoch_50] < loss[epoch_1]` | CRITICAL | Parse wandb/tensorboard logs |
| P4-T3 | No NaN in loss or gradients | Zero NaN events in training | CRITICAL | Check wandb alerts or log grep |
| P4-T4 | 1-NFE samples at epoch 50 show vaguely brain-like structure | Visual inspection of mid-sagittal slices | CRITICAL | Log sample images to wandb |
| P4-T5 | 1-NFE samples at epoch 200 show clear brain anatomy | Visual: identifiable ventricles, cortex, white matter | CRITICAL | Log sample images to wandb |
| P4-T6 | EMA-selected model produces better samples than online model | FID(EMA) < FID(online) on 500 samples | CRITICAL | Compare at epoch 400 |
| P4-T7 | Latent normalisation correctly applied | Generated latents have statistics close to training latents | INFORMATIONAL | Histogram comparison |
| P4-T8 | Checkpoints save and load correctly | Resume training from checkpoint; loss continues from previous value | INFORMATIONAL | Test resume |

**Phase 4 is PASSED when P4-T1 through P4-T6 are ALL green.**

## 7. Expected Outputs

- `src/neuromf/models/latent_meanflow.py` — Lightning training module
- `experiments/cli/train.py` — training CLI (updated)
- `configs/train_meanflow.yaml` — full training config
- Model checkpoints (best EMA)
- Training logs (wandb/tensorboard)
- `experiments/phase_4/verification_report.md`

## 8. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|
| MeanFlow fails to converge on 3D latents | Blocks Phase 4 | Low | Toy experiment (Phase 2) validates pipeline first; check LR, batch size |
| NaN in loss during training | Blocks P4-T3 | Medium | Check gradient clipping, t_min clipping, adaptive weight epsilon |
| OOM during training | Blocks P4-T1 | Low | Reduce batch size from 24; use gradient checkpointing |
| EMA not improving over online | Blocks P4-T6 | Low | Check EMA half-lives; may need longer training before EMA diverges |
| Samples don't improve visually | Blocks P4-T4/T5 | Medium | Check that VAE decode is correct; verify latent normalisation |
