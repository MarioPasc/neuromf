# Phase 4b: Training Diagnostics and Structured Logging

**Depends on:** Phase 4 (training loop implemented)
**Modules touched:** `src/neuromf/wrappers/meanflow_loss.py`, `src/neuromf/callbacks/`, `src/neuromf/models/latent_meanflow.py`, `experiments/cli/train.py`, `configs/`
**Scope:** Enhance training with publication-quality diagnostic logging. NO changes to the core training loop logic, optimiser, loss formulation, or model architecture. This is instrumentation only.

---

## 1. Objective

Instrument the training pipeline with comprehensive, structured logging that serves three purposes:

1. **Stability diagnosis:** detect training pathologies (gradient explosion, NaN propagation, mode collapse, loss divergence) early enough to intervene.
2. **Performance tracking:** provide quantitative evidence that the model is learning (loss decreases, velocity fields converge, generated latent statistics approach data statistics).
3. **Throughput monitoring:** track wall-clock efficiency to estimate time-to-completion and detect I/O bottlenecks.

The system must be **zero-overhead when disabled** and **<2% wall-clock overhead when enabled** at the default logging frequencies.

---

## 2. Scientific Rationale for Each Metric Category

### 2.1 Loss Decomposition (what the model is learning)

The iMF loss decomposes as:

$$\mathcal{L}_\text{iMF} = \mathcal{L}_\text{FM} + \lambda_\text{MF} \cdot \mathcal{L}_\text{MF}$$

Monitoring the ratio $\mathcal{L}_\text{FM} / \mathcal{L}_\text{total}$ over time reveals whether the FM term (low-variance, stabilising) dominates early training and the MF term (one-step consistency) takes over later — this is the expected healthy trajectory. If $\mathcal{L}_\text{MF}$ never decreases, the JVP signal is not propagating correctly. If $\mathcal{L}_\text{FM}$ diverges while $\mathcal{L}_\text{MF}$ decreases, the velocity field is degenerate.

Per-channel loss ($\|\cdot\|_p^p$ computed separately per latent channel $c \in \{0,1,2,3\}$) detects channel imbalance — the MAISI VAE latent channels encode different frequency bands, and if one channel dominates the loss, the model may ignore the others.

The within-epoch loss standard deviation quantifies gradient noise. High variance relative to mean suggests the batch size is too small or the time sampling is introducing excessive variance through the JVP term.

### 2.2 MeanFlow-Specific Diagnostics (is the MF identity being enforced?)

The compound velocity $V_\theta = u_\theta + (t-r) \cdot \text{sg}[\text{JVP}(u_\theta; \tilde{v}_\theta)]$ must satisfy $V_\theta \approx v_c$ at convergence. Tracking:

- $\|V_\theta\|_2$ vs $\|v_c\|_2$: these should converge to similar magnitudes. A growing gap indicates the compound velocity is diverging.
- $\|\text{JVP}\|_2$: the JVP magnitude should be moderate. If $\|\text{JVP}\| \gg \|u_\theta\|$, the derivative term dominates, causing instability.
- $\|u_\theta\|_2$ and $\|\tilde{v}_\theta\|_2$: the average and instantaneous velocity norms track the overall scale of the learned velocity field.
- Adaptive weight statistics $w(t) = 1/(\|\text{error}\|_p^p + \epsilon)$: if these are highly variable, the adaptive normalisation may be oscillating.
- FM/MF sample counts per batch: verifies `data_proportion` is working correctly (should be ~25% FM, ~75% MF with `data_proportion=0.25`).

### 2.3 Gradient Flow (is the architecture healthy?)

The 178M-parameter UNet has a deep encoder-decoder structure where gradients must traverse many blocks. Monitoring per-block gradient norms ($\|g\|_2$ for `conv_in`, `down_blocks.{0-3}`, `middle_block`, `up_blocks.{0-3}`, `out`, `r_embed`, `time_embed`) detects:

- Vanishing gradients: if any block has $\|g\| < 10^{-8}$, the block is not being trained.
- Exploding gradients: if $\|g\| > 10^3$, training will become unstable.
- The gradient clip fraction (what fraction of steps exceed `gradient_clip_val=1.0`) measures how often the optimiser is being constrained. A clip fraction >50% persistently suggests the learning rate is too high.

The relative update magnitude $\|\Delta\theta\| / \|\theta\|$ per block measures the effective learning rate after clipping and adaptive optimisation. Values $<10^{-7}$ indicate the block is frozen; values $>10^{-2}$ indicate instability.

### 2.4 EMA Diagnostics (is the shadow model diverging appropriately?)

The EMA shadow should gradually diverge from the online model: $\|\theta_\text{EMA} - \theta_\text{online}\| / \|\theta_\text{online}\|$ should be small initially and grow to a stable plateau. If it collapses back to zero, the EMA decay is too small (shadow tracks online too closely). If it grows without bound, decay is too large.

### 2.5 Time Sampling Verification (is the distribution correct?)

Logging the per-batch statistics of $t$ (mean, std, min, max) and $h = t - r$ (mean, std, fraction with $h = 0$) provides a continuous sanity check that the logit-normal sampler and `data_proportion` are functioning correctly.

### 2.6 Sample Quality (periodic visual assessment)

Beyond the existing slice images, logging per-channel statistics (mean, std, min, max) of generated latents $\hat{\mathbf{z}}_0$ and comparing them to the training data statistics provides a quantitative proxy for sample quality before we have FID. At convergence, generated latent statistics should approximate training latent statistics.

### 2.7 Performance Counters (throughput and efficiency)

Training throughput (samples/sec, steps/sec), GPU memory (peak allocated, peak reserved), epoch wall-clock time, and data loading fraction (time spent in DataLoader vs compute) are essential for estimating time-to-completion and identifying bottlenecks.

---

## 3. Output Structure

```
results/phase_4/
├── logs/
│   └── meanflow/
│       └── version_N/
│           ├── events.out.tfevents.*     # TensorBoard (scalars + images + histograms)
│           └── hparams.yaml
│
├── diagnostics/
│   ├── epoch_001/
│   │   └── summary.json                  # Full epoch diagnostic snapshot
│   ├── epoch_002/
│   │   └── summary.json
│   ├── ...
│   └── training_summary.json             # Running summary: appended each epoch
│
├── samples/
│   └── epoch_025/
│       ├── grid_axial.png                # N-sample grid, mid-axial slice
│       ├── grid_coronal.png
│       ├── grid_sagittal.png
│       ├── latent_stats.json             # Per-channel mean/std of generated latents
│       └── latents/
│           └── sample_{K}.pt             # Raw generated latent tensors
│
└── checkpoints/                          # (already exists from Phase 4)
```

### 3.1 TensorBoard Scalar Hierarchy

Organised for TensorBoard's folder grouping (the `/` separator creates collapsible sections):

```
train/loss_total                    # Combined iMF loss (prog_bar)
train/loss_fm                       # Flow matching term
train/loss_mf                       # MeanFlow consistency term
train/loss_ratio_fm                 # L_FM / L_total
train/loss_std                      # Within-epoch loss std (logged per epoch)

train/channel/loss_ch0              # Per-channel Lp loss (4 channels)
train/channel/loss_ch1
train/channel/loss_ch2
train/channel/loss_ch3

train/gradients/global_norm         # Pre-clip gradient norm
train/gradients/clip_fraction       # Running fraction of clipped steps (per epoch)
train/gradients/conv_in             # Per-block gradient norms
train/gradients/down_block_0
train/gradients/down_block_1
train/gradients/down_block_2
train/gradients/down_block_3
train/gradients/middle_block
train/gradients/up_block_0
train/gradients/up_block_1
train/gradients/up_block_2
train/gradients/up_block_3
train/gradients/out
train/gradients/r_embed
train/gradients/time_embed

train/updates/relative_update_norm  # ||Δθ|| / ||θ|| (global, per epoch)

train/meanflow/jvp_norm             # ||JVP(u; v)|| batch mean
train/meanflow/u_norm               # ||u_θ|| batch mean
train/meanflow/v_tilde_norm         # ||ṽ_θ|| batch mean
train/meanflow/compound_v_norm      # ||V_θ|| batch mean
train/meanflow/target_v_norm        # ||v_c|| batch mean
train/meanflow/adaptive_weight_mean # Mean adaptive weight
train/meanflow/adaptive_weight_std  # Std of adaptive weights

train/sampling/t_mean               # Batch mean of t
train/sampling/t_std                # Batch std of t
train/sampling/h_mean               # Batch mean of h = t - r
train/sampling/h_zero_frac          # Fraction of batch with h ≈ 0 (FM samples)

train/lr                            # Current learning rate

val/loss_total                      # Validation losses
val/loss_fm
val/loss_mf

ema/param_divergence                # ||θ_EMA - θ_online|| / ||θ_online|| (per epoch)

perf/samples_per_sec                # Training throughput
perf/steps_per_sec
perf/gpu_mem_allocated_gb           # Peak GPU memory
perf/gpu_mem_reserved_gb
perf/epoch_time_sec                 # Wall clock per epoch
perf/data_loading_frac              # Fraction of time in DataLoader (estimated)
```

### 3.2 TensorBoard Images and Histograms (periodic, every `sample_every_n_epochs`)

```
samples/generated_grid              # N×3 grid (axial, coronal, sagittal)
samples/real_vs_generated           # Side-by-side comparison grid

histograms/t_distribution           # Histogram of sampled t values
histograms/h_distribution           # Histogram of h = t - r values
histograms/generated_latent_ch0     # Per-channel histogram of ẑ_0
histograms/generated_latent_ch1
histograms/generated_latent_ch2
histograms/generated_latent_ch3
```

### 3.3 JSON Epoch Summary Schema

Each `diagnostics/epoch_{NNN}/summary.json` contains:

```json
{
  "epoch": 25,
  "global_step": 1025,
  "wall_time_sec": 142.3,
  "train": {
    "loss_total": 1.423,
    "loss_fm": 0.712,
    "loss_mf": 0.711,
    "loss_std": 0.083,
    "loss_per_channel": [0.38, 0.35, 0.34, 0.36],
    "grad_norm_mean": 0.42,
    "grad_norm_max": 2.1,
    "clip_fraction": 0.05,
    "lr": 4.2e-5
  },
  "val": {
    "loss_total": 1.51,
    "loss_fm": 0.76,
    "loss_mf": 0.75
  },
  "meanflow": {
    "jvp_norm": 0.82,
    "u_norm": 1.23,
    "v_tilde_norm": 1.18,
    "compound_v_norm": 1.35,
    "target_v_norm": 1.41,
    "adaptive_weight_mean": 0.97,
    "adaptive_weight_std": 0.12
  },
  "ema": {
    "param_divergence": 0.0023
  },
  "performance": {
    "samples_per_sec": 48.2,
    "steps_per_sec": 2.01,
    "gpu_mem_allocated_gb": 34.5,
    "gpu_mem_reserved_gb": 42.1,
    "epoch_time_sec": 142.3
  }
}
```

The `training_summary.json` at the diagnostics root is an array of these objects (one per epoch), enabling quick offline analysis without loading TensorBoard.

---

## 4. Architecture

### 4.1 Extend `MeanFlowPipeline.forward()` to Return Diagnostics

The existing `MeanFlowPipeline.forward()` returns `{"loss", "loss_fm", "loss_mf"}`. Add a `return_diagnostics: bool = False` parameter. When `True`, the same forward pass additionally returns all intermediate tensors needed for diagnostics. This avoids any recomputation.

Extended return dict (when `return_diagnostics=True`):
```python
{
    "loss": Tensor,                  # (existing)
    "loss_fm": Tensor,               # (existing)
    "loss_mf": Tensor,               # (existing)
    # --- New: batch-level diagnostics (detached, scalar) ---
    "diag_jvp_norm": Tensor,         # ||JVP||_2 batch mean
    "diag_u_norm": Tensor,           # ||u_θ||_2 batch mean
    "diag_v_tilde_norm": Tensor,     # ||ṽ_θ||_2 batch mean
    "diag_compound_v_norm": Tensor,  # ||V_θ||_2 batch mean
    "diag_target_v_norm": Tensor,    # ||v_c||_2 batch mean
    "diag_adaptive_weight_mean": Tensor,   # mean of adaptive weights
    "diag_adaptive_weight_std": Tensor,    # std of adaptive weights
    "diag_loss_per_channel": Tensor, # (4,) per-channel loss
}
```

All `diag_*` tensors must be **`.detach()`ed** — they must NOT participate in the backward pass or the computational graph. They are read-only telemetry extracted from the existing computation with zero gradient overhead.

**Critical:** This must not change behaviour when `return_diagnostics=False` (the default). The existing return dict must be identical. All existing tests must pass unchanged.

### 4.2 Implement a Lightning Callback: `TrainingDiagnosticsCallback`

Create `src/neuromf/callbacks/diagnostics.py` containing a single Lightning Callback that handles ALL diagnostic logging. This keeps the `LatentMeanFlow` module clean — the module's only responsibility is to expose the diagnostic dict; the callback consumes it.

The callback reads diagnostics from `trainer.lightning_module._step_diagnostics` (a dict stored on the module by `training_step`). The LatentMeanFlow module stores the extended result dict on `self._step_diagnostics` each step.

**Logging frequency tiers** (configurable via a `DiagnosticsConfig`):

| Tier | Frequency | What |
|---|---|---|
| **step** | Every `log_every_n_steps` (50) | Loss components, LR, global grad norm, throughput, GPU mem |
| **epoch** | Every epoch | Epoch-averaged losses, loss std, loss ratios, per-block grad norms, clip fraction, relative update norm, EMA divergence, MeanFlow norms, time sampling stats, JSON summary |
| **periodic** | Every `diag_every_n_epochs` (25) | Per-channel losses, generated latent statistics, TensorBoard histograms, sample images |

### 4.3 Implement `src/neuromf/callbacks/performance.py`

A small utility callback that tracks:
- Step start/end timestamps for throughput
- Epoch start/end timestamps for wall clock
- GPU memory queries via `torch.cuda.max_memory_allocated()`

### 4.4 Minimal Changes to `LatentMeanFlow`

The only changes to `latent_meanflow.py`:

1. In `training_step`: call `self.loss_pipeline(model, z_0, eps, t, r, return_diagnostics=self._diag_enabled)` and store the result as `self._step_diagnostics`.
2. In `training_step`: also store `t` and `r` tensors on `self._step_diagnostics["t"]` and `self._step_diagnostics["r"]` for time sampling stats.
3. Add a `self._diag_enabled: bool` attribute controlled by config.
4. Store a snapshot of parameters at epoch start for relative update norm computation.

Do NOT move any logging logic into the Lightning module. All `self.log()` calls beyond the basic three (loss_total, loss_fm, loss_mf on prog_bar) belong in the callback.

### 4.5 Register Callbacks in `train.py`

Add the diagnostics callback to the Trainer callback list:

```python
from neuromf.callbacks.diagnostics import TrainingDiagnosticsCallback
from neuromf.callbacks.performance import PerformanceCallback

diag_cb = TrainingDiagnosticsCallback(
    log_every_n_steps=config.training.log_every_n_steps,
    diag_every_n_epochs=config.get("diagnostics", {}).get("every_n_epochs", 25),
    diagnostics_dir=diagnostics_dir,
    block_names=["conv_in", "down_blocks.0", "down_blocks.1", "down_blocks.2",
                 "down_blocks.3", "middle_block", "up_blocks.0", "up_blocks.1",
                 "up_blocks.2", "up_blocks.3", "out", "r_embed", "time_embed"],
)
perf_cb = PerformanceCallback()

trainer = pl.Trainer(
    ...,
    callbacks=[checkpoint_cb, lr_monitor, diag_cb, perf_cb],
)
```

---

## 5. Configuration

Add to `configs/train_meanflow.yaml`:

```yaml
diagnostics:
  enabled: true
  every_n_epochs: 25       # periodic diagnostics (histograms, per-channel, etc.)
  json_every_n_epochs: 1   # JSON summary frequency (every epoch by default)
  grad_block_norms: true   # per-block gradient norms
  meanflow_norms: true     # velocity/JVP magnitude tracking
  time_sampling_stats: true
  ema_divergence: true
  performance_counters: true
  latent_histograms: true  # TensorBoard histograms of generated latents
```

---

## 6. Implementation Order

| Order | File | What |
|---|---|---|
| 1 | `src/neuromf/wrappers/meanflow_loss.py` | Add `return_diagnostics` parameter to `forward()` |
| 2 | `src/neuromf/callbacks/__init__.py` | Create callbacks subpackage |
| 3 | `src/neuromf/callbacks/performance.py` | PerformanceCallback: throughput + GPU memory |
| 4 | `src/neuromf/callbacks/diagnostics.py` | TrainingDiagnosticsCallback: all diagnostic logging |
| 5 | `src/neuromf/models/latent_meanflow.py` | Wire diagnostics: store `_step_diagnostics`, add `_diag_enabled` |
| 6 | `experiments/cli/train.py` | Register callbacks, create diagnostics dirs |
| 7 | `configs/train_meanflow.yaml` | Add `diagnostics:` section |
| 8 | `tests/test_diagnostics.py` | Tests (see Section 7) |
| 9 | Run full test suite | Verify no regressions |

---

## 7. Tests

Create `tests/test_diagnostics.py`:

| Test | What it verifies |
|---|---|
| `test_pipeline_return_diagnostics_false` | When `return_diagnostics=False`, output dict has exactly 3 keys (loss, loss_fm, loss_mf) — no regressions |
| `test_pipeline_return_diagnostics_true` | When `return_diagnostics=True`, output dict has all `diag_*` keys, all finite, all detached |
| `test_pipeline_diagnostics_no_grad_leak` | With `return_diagnostics=True`, calling `loss.backward()` works identically — `diag_*` tensors do not affect gradients |
| `test_per_channel_loss_shape` | `diag_loss_per_channel` has shape `(4,)` for 4-channel latents |
| `test_diagnostics_callback_logs_scalars` | Instantiate callback, call its hooks with mock trainer/module, verify it calls `trainer.logger.log_metrics` |
| `test_json_summary_written` | After one epoch, `diagnostics/epoch_001/summary.json` exists and is valid JSON matching schema |
| `test_performance_callback_throughput` | PerformanceCallback produces finite `samples_per_sec` after a few mock steps |

All tests use tiny UNet (`channels=[8,16,32,64]`, spatial=16) and `jvp_strategy="finite_difference"` for speed.

---

## 8. Critical Constraints

1. **No new dependencies.** Use only PyTorch, Lightning, TensorBoard, and stdlib (`json`, `time`, `pathlib`).
2. **No changes to loss mathematics.** The `return_diagnostics=True` path computes the SAME loss. All diagnostic tensors are extracted from existing intermediate variables and detached.
3. **No changes to checkpoint format.** Diagnostics state is ephemeral — it is NOT saved in checkpoints.
4. **All existing tests must pass.** Run the full `pytest tests/` suite after implementation.
5. **Overhead budget:** The diagnostic computation (`detach`, `norm`, `mean`) adds at most ~10 microseconds per step on A100. The JSON writing and TensorBoard histogram logging happen only at epoch/periodic boundaries and are negligible relative to training time.
6. **The `_step_diagnostics` dict on the Lightning module is cleared at the start of each `training_step`.** It must not accumulate memory across steps.

---

## 9. How to Verify

After implementation, the agent should:

1. Run `pytest tests/test_diagnostics.py -v --tb=short` — all 7 tests pass.
2. Run `pytest tests/ -v --tb=short` — full suite, no regressions.
3. Run a 2-epoch dry training on CPU with the tiny UNet config:
   ```bash
   python experiments/cli/train.py --config configs/train_meanflow.yaml \
       --override training.max_epochs=2 training.batch_size=2 \
       diagnostics.every_n_epochs=1
   ```
   Verify that:
   - `diagnostics/epoch_001/summary.json` exists and is valid.
   - `diagnostics/training_summary.json` has 2 entries.
   - TensorBoard logs contain the expected scalar groups.

