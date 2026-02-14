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

---

## 9. CRITICAL: Configuration Discrepancies — Read Before Implementing

The YAML block in Section 4 above (copied from `docs/main/technical_guide.md §6.2`) contains **stale hyperparameters** from an early draft. The **ground truth** is the actual `configs/train_meanflow.yaml` file that was prepared in Phase 3 and validated against the Phase 2 ablation results and the MeanFlow-PyTorch reference implementation. The table below lists every discrepancy; **always use the "Correct (actual config)" column**.

| Parameter | Stale (Section 4 YAML) | Correct (actual config) | Rationale |
|---|---|---|---|
| `time_sampling.mu` | 0.8 | **-0.4** | Matches iMF paper (Geng et al., 2025b) and MeanFlow-PyTorch `P_mean=-0.4`. Phase 2 validated this. |
| `time_sampling.sigma` | 0.8 | **1.0** | Matches iMF `P_std=1.0`. Phase 2 validated this. |
| `time_sampling.data_proportion` | 0.5 (`r_equals_t_prob`) | **0.25** | Phase 2 ablation E found `data_proportion=0.25` optimal for 1-NFE on the torus. We start with this and sweep in Phase 6. |
| `training.weight_decay` | 0.01 | **0.0** | MeanFlow-PyTorch uses `weight_decay=0`. Non-zero WD interacts poorly with EMA. |
| `training.max_epochs` | 800 | **500** | A100 throughput allows 500 epochs to be sufficient; more can be added if loss hasn't plateaued. |
| `training.warmup_steps` | 1000 | **5000** | Longer warmup for stability with the JVP loss on high-dimensional latents. |
| `meanflow.norm_eps` | 1e-4 (`adaptive_eps`) | **0.01** | Phase 2 used `norm_eps=0.01` successfully. 1e-4 can cause division instability early in training when per-channel norms are near zero. |
| `ema.decay` | multi-half-life `[500, 1000, 2000]` | **0.999** (simple exponential) | Phase 2 used simple `decay=0.999`. Multi-half-life EMA is an optional enhancement for Phase 6 ablations but not the Phase 4 default. The existing `src/neuromf/utils/ema.py` supports both — use simple mode. |
| `time_sampling.t_min` | 0.05 (`t_clip_min`) | **0.001** (sampling) / **0.05** (loss `t_min` for $1/t$ division) | Two distinct clamp values: `time_sampling.t_min=0.001` controls the logit-normal sampler floor, while `meanflow.t_min=0.05` controls the safe denominator in $u_\theta = (\mathbf{z}_t - \hat{\mathbf{z}}_{0,\theta}) / \max(t, 0.05)$. Do not confuse them. |

**The Algorithm 1 pseudocode in Section 2 is also stale** on steps 2–3. The correct sampling is:
- Step 2: Sample $t, r \sim \text{LogitNormal}(\mu=-0.4, \sigma=1.0)$, then swap so $t \geq r$ (following MeanFlow-PyTorch convention — NOT uniform $r$).
- Step 3: With probability `data_proportion=0.25`: set $r = t$.

This is already implemented correctly in `src/neuromf/utils/time_sampler.py::sample_t_and_r()`.

**DO NOT modify `configs/train_meanflow.yaml` to match the stale Section 4 YAML.** The config file is correct as-is.

---

## 10. Scope Clarification

This Phase 4 task covers **code implementation only**: the `LatentMeanFlow` Lightning module, the training CLI, the SLURM launcher/worker scripts, and local lightweight tests. Specifically:

**IN scope:**
- `src/neuromf/models/latent_meanflow.py` — the PyTorch Lightning training module
- `experiments/cli/train.py` — the training CLI with config loading and resume support
- `configs/picasso/train_meanflow.yaml` — Picasso-specific config overlay
- `experiments/slurm/phase_4/train.sh` — SLURM launcher
- `experiments/slurm/phase_4/train_worker.sh` — SLURM worker
- `tests/test_latent_meanflow.py` — lightweight local tests (see Section 14)

**NOT in scope (deferred to a separate pipeline refinement task):**
- Per-epoch metric logging beyond loss (FID, SSIM, etc.)
- Fancy plotting or dashboard setup
- Verification report generation (P4-T1 through P4-T8 are evaluated manually after training)
- Wandb integration beyond basic scalar logging

The agent should focus on getting a correct, clean, and robust training loop that can run on Picasso via SLURM.

---

## 11. Execution Environment: Picasso Supercomputer

All GPU training runs for Phase 4 will execute on the **Picasso supercomputer**. The agent must read `configs/picasso/base.yaml` for all hardware specifications and absolute paths. Key facts:

- **GPU:** NVIDIA A100 40GB
- **Constraint:** `--constraint=dgx` in SLURM
- **Conda env:** `neuromf` (activated via `module load` + `conda activate`)
- **Project root:** `/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/neuromf`
- **Results root:** `/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results`
- **Latents:** `/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results/latents`
- **Training checkpoints:** `/mnt/home/users/tic_163_uma/mpascual/execs/neuromf/results/training_checkpoints`
- **MAISI VAE weights (for periodic sample decoding):** `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/NV-Generate-MR/models/autoencoder_v2.pt`

### 11.1 SLURM Scripts

Create the following files, following the established pattern from `experiments/slurm/phase_0/` and `experiments/slurm/phase_1/`:

**`experiments/slurm/phase_4/train.sh`** (launcher, run from login node):
- Export `REPO_SRC`, `CONFIGS_DIR`, `RESULTS_DST`, `CONDA_ENV_NAME`
- Create output directories: `${RESULTS_DST}/training_checkpoints`, `${RESULTS_DST}/phase_4/logs`, `${RESULTS_DST}/phase_4/samples`
- Submit the worker via `sbatch` with:
  - `--job-name="neuromf_p4_train"`
  - `--time=2-00:00:00` (2 days wall time — 500 epochs at ~45 steps/epoch on A100)
  - `--ntasks=1`, `--cpus-per-task=16`, `--mem=64G`
  - `--constraint=dgx`, `--gres=gpu:1`
  - Output/error logs to `${RESULTS_DST}/phase_4/`

**`experiments/slurm/phase_4/train_worker.sh`** (submitted by launcher):
- Environment setup: `module load` conda, `conda activate neuromf`
- Pre-flight checks: verify Python, PyTorch, CUDA, GPU, config files, latent directory
- Run: `python experiments/cli/train.py --config configs/picasso/train_meanflow.yaml`
- Support `--resume` flag for checkpoint resumption

### 11.2 Picasso-Specific Config Overlay

Create `configs/picasso/train_meanflow.yaml` that is merged at runtime with `configs/picasso/base.yaml` + `configs/train_meanflow.yaml`:

```yaml
# Picasso-specific overrides for Phase 4 training
# Merged: base.yaml + train_meanflow.yaml + this file

paths:
  latents_dir: "${paths.results_root}/latents"
  checkpoints_dir: "${paths.results_root}/training_checkpoints"
  logs_dir: "${paths.results_root}/phase_4/logs"
  samples_dir: "${paths.results_root}/phase_4/samples"
  maisi_vae_weights: "${paths.checkpoints_root}/NV-Generate-MR/models/autoencoder_v2.pt"

training:
  num_workers: 16    # 128-core Picasso nodes
  prefetch_factor: 4
```

---

## 12. Train/Validation Split

The Phase 4 spec does not mention a train/val split, but one is needed for:
1. Monitoring validation loss to detect overfitting (important with only ~1100 latents).
2. Selecting the best EMA checkpoint (P4-T6).

**Strategy:** Use a deterministic 90/10 split (seed=42) of the pre-computed latent files. With ~1100 latents: ~990 train, ~110 val. The validation set is used for:
- Computing validation loss every `val_every_n_epochs=10` epochs.
- Generating 1-NFE samples for visual inspection (P4-T4, P4-T5).

The `LatentDataset` should accept a `split` argument (`"train"` or `"val"`) and perform the split based on sorted filenames + seed for reproducibility. This ensures the split is deterministic across runs without requiring a separate split file.

---

## 13. Periodic Sample Generation: VAE Loading

Generating 1-NFE samples (P4-T4, P4-T5) requires decoding latents through the frozen MAISI VAE. This is a non-trivial engineering concern:

1. **VAE memory:** The MAISI VAE adds ~500MB VRAM. On A100 40GB this is not an issue, but the agent must ensure the VAE is only loaded when needed (in `on_train_epoch_end` or a callback), not kept in GPU memory during the entire training loop.
2. **Denormalisation:** If latents are normalised during training (mean=0, std=1), the generated latents must be denormalised back to the original latent statistics before VAE decoding. Use the `latent_stats.json` from Phase 1 for this.
3. **Scale factor:** The MAISI VAE uses `scale_factor=0.96240234375`. The decode call must undo this: `decoded = vae.decode(z_hat / scale_factor)`.
4. **Sample saving:** Save both the raw latent tensors (`.pt`) and the decoded mid-sagittal/axial/coronal slices (`.png`) to `${RESULTS_DST}/phase_4/samples/epoch_{N}/`.

The recommended pattern:

```python
@torch.no_grad()
def _generate_samples(self, n_samples: int = 8) -> None:
    """Generate 1-NFE samples and decode through frozen VAE."""
    # 1. Sample noise
    z_1 = torch.randn(n_samples, 4, 48, 48, 48, device=self.device)
    
    # 2. One-step generation: x-prediction at (r=0, t=1)
    z_0_hat = self.model(z_1, r=torch.zeros(n_samples, device=self.device),
                          t=torch.ones(n_samples, device=self.device))
    
    # 3. Denormalise if training with normalised latents
    z_0_hat = z_0_hat * self.latent_std + self.latent_mean
    
    # 4. Load VAE, decode, unload
    vae = self._load_vae()  # lazy-load, cached
    decoded = vae.decode(z_0_hat / self.scale_factor)
    
    # 5. Log slices
    ...
```

---

## 14. Local Lightweight Tests

Local tests run on the RTX 4060 Laptop (8GB VRAM) and must be lightweight. They verify component correctness without running full training. Create `tests/test_latent_meanflow.py`:

| Test | Description | Strategy |
|---|---|---|
| `test_lightning_module_init` | `LatentMeanFlow` instantiates without error | Use tiny config: `channels=[8,16,32,64]`, spatial_size=16 |
| `test_training_step_runs` | `training_step` returns finite loss | Fake batch of shape `(2, 4, 16, 16, 16)`, one step |
| `test_training_step_gradients_flow` | All trainable params receive gradients after one step | Check `param.grad is not None` |
| `test_ema_updates` | EMA shadow params differ from model params after N steps | Run 5 steps, compare |
| `test_checkpoint_save_load` | Save checkpoint, reload, verify state matches | Use `tmpdir`, save/load/compare |
| `test_resume_loss_continuity` | Loss after resume is close to loss before save | Run 3 steps, save, load, run 1 more, compare loss magnitude |
| `test_sample_generation_shape` | `_generate_samples` produces correct output shape | Mock VAE with identity, verify shapes |
| `test_cli_dry_run` | CLI parses config without crashing | Call with `--dry-run` flag |

All tests must use a **tiny UNet** (`channels=[8,16,32,64]`, input shape `(B, 4, 16, 16, 16)`) so they run in <10s on CPU or <5s on the laptop GPU.

---

## 15. Implementation Details the Agent Must Get Right

### 15.1 The `h = t - r` Convention in MAISIUNetWrapper

The MAISI UNet wrapper from Phase 3 accepts dual time conditioning `(r, t)` via a modified embedding. The MeanFlow-PyTorch reference uses `h = t - r` as the second conditioning input (not `r` directly). Verify that the existing `MAISIUNetWrapper` follows this convention. The forward signature should be:

```python
def forward(self, x: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Noisy latent, shape (B, 4, 48, 48, 48).
        t: Diffusion time, shape (B,).
        h: Interval width h = t - r, shape (B,). h=0 means r=t (FM mode).
    """
```

When `r = t`, `h = 0`, and the model reduces to standard flow matching. This is already implemented in Phase 3 — do not change it.

### 15.2 x-Prediction → 1-NFE Inference Shortcut

With x-prediction, 1-NFE sampling is a single forward pass:

$$\hat{\mathbf{z}}_0 = \text{net}_\theta(\boldsymbol{\epsilon},\; t=1,\; h=1-0=1) \tag{direct}$$

The network directly outputs $\hat{\mathbf{z}}_0$ — **no** $u \to x$ conversion is needed at inference. This is because at $(r, t) = (0, 1)$:

$$u_\theta = \frac{\boldsymbol{\epsilon} - \hat{\mathbf{z}}_{0,\theta}}{1} = \boldsymbol{\epsilon} - \hat{\mathbf{z}}_{0,\theta} \implies \hat{\mathbf{z}}_{0,\theta} = \boldsymbol{\epsilon} - u_\theta$$

But since the network is parameterised in x-prediction, it directly outputs $\hat{\mathbf{z}}_{0,\theta}$. The agent must use this shortcut and NOT apply the $u \to x$ conversion during inference — that would apply the conversion twice.

### 15.3 Gradient Checkpointing + JVP Interaction

`torch.func.jvp` requires functionalised models. The Phase 3 wrapper already handles this via `torch.func.functional_call`. When `gradient_checkpointing=true` (config default), the UNet uses MONAI's `use_checkpointing` flag, which adds `torch.utils.checkpoint.checkpoint` inside residual blocks. This is compatible with `torch.func.jvp` as validated in P3-T3a, but the agent should NOT additionally wrap the JVP computation in gradient checkpointing — the checkpointing happens inside the UNet forward pass, not around the JVP.

### 15.4 Loss Logging Granularity

Log the following scalars at every `log_every_n_steps`:
- `train/loss_total`: the combined iMF loss $\mathcal{L}_\text{FM} + \lambda_\text{MF} \cdot \mathcal{L}_\text{MF}$
- `train/loss_fm`: the FM component
- `train/loss_mf`: the MF component  
- `train/grad_norm`: global gradient norm (before clipping)
- `train/lr`: current learning rate

At every `val_every_n_epochs`:
- `val/loss_total`, `val/loss_fm`, `val/loss_mf`: same decomposition on validation set

### 15.5 Checkpoint Format

Every checkpoint saved to disk must contain:

```python
checkpoint = {
    "epoch": current_epoch,
    "global_step": global_step,
    "model_state_dict": model.state_dict(),
    "ema_state_dict": ema.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),  # if using LR scheduler
    "config": OmegaConf.to_container(config),
    "loss_history": {
        "train": list_of_epoch_losses,
        "val": list_of_val_losses,
    },
}
```

Save checkpoints every `save_every_n_epochs=50` epochs AND keep the best checkpoint by validation loss. Resume must restore ALL of these fields.

---

## 16. Updated Expected Outputs

In addition to Section 7, this phase must also produce:

- `configs/picasso/train_meanflow.yaml` — Picasso config overlay
- `experiments/slurm/phase_4/train.sh` — SLURM launcher
- `experiments/slurm/phase_4/train_worker.sh` — SLURM worker
- `tests/test_latent_meanflow.py` — local lightweight tests