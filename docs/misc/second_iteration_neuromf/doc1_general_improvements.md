# NeuroMF v2: Training Improvements Specification

**Project:** NeuroMF — Improved MeanFlow for 3D Brain MRI Synthesis
**Author:** Mario Pascual
**Date:** 2026-02-28
**Status:** SPECIFICATION READY — awaiting implementation
**Depends on:** Phase 5 evaluation pipeline completed (gate OPEN)
**Scope:** Config-level and minor code-level changes to the existing training pipeline

---

## 0. Context and Motivation

NeuroMF v1 trained a 178M-parameter dual-head UNet (x-prediction + exact JVP, (t,h) conditioning) for 690 epochs on 5,471 latent brain MRI volumes (effective batch size 132, 6× A100-SXM4-40GB DDP). Training completed 28,980 optimizer steps (~42 steps/epoch) before early stopping triggered (patience=10 on 3D-FID). The best intra-training 3D-FID was 11.67 at epoch 388; the final evaluation FID was 11.88 at epoch 688 — still competitive with the best, indicating the model had not converged.

**Note on intra-training FID:** The v1 run's intra-training FID callback had a denormalization bug (now fixed). The FID values reported here are from the corrected post-hoc evaluation pipeline (Phase 5), not from the training logs.

### Post-hoc evaluation results (n_gen=2,000, n_real=326)

| Metric              | NFE=1   | NFE=10  | NFE=50  | MOTFM (NFE=50) |
|---------------------|---------|---------|---------|-----------------|
| FID-3D ↓            | 73.85   | 7.34    | 6.14    | 7.93            |
| MMD ↓               | 0.993   | 0.229   | 0.169   | 0.22            |
| MS-SSIM ↑           | 0.331   | 0.662   | 0.655   | 0.77            |
| Coverage (k=5) ↑    | 0.000   | 0.166   | 0.258   | —               |
| Density (k=5) ↑     | 0.000   | 0.069   | 0.160   | —               |
| HF Energy Ratio     | 0.0003  | 0.0010  | 0.0011  | —               |

**Summary:** NeuroMF v1 matches or outperforms MOTFM at NFE ≥ 10 (FID-3D 7.34 vs 7.93, MMD 0.229 vs 0.22) but fails catastrophically at NFE=1 — the primary MeanFlow claim. This document specifies targeted improvements that address three identified failure modes through hyperparameter tuning and one minor code change (boundary sampling). No new components or architectural changes are proposed.

---

## 1. Identified Failure Modes

### 1.1 1-NFE Generation Deficit

**Observation.** At NFE=1, FID-3D = 73.85 (vs 7.34 at NFE=10 — a 10× gap). Coverage = 0.000 and Density = 0.000 indicate complete mode collapse: the generated distribution collapses to a narrow cluster that does not overlap with the real distribution. HF Energy Ratio drops from 0.0010 (NFE=10) to 0.0003 (NFE=1), confirming over-smoothed outputs.

**Root cause: insufficient training signal at the 1-NFE operating point.** At 1-NFE, the model performs $\hat{z}_0 = z_1 - u_\theta(z_1, r\!=\!0, t\!=\!1)$. This requires accurate compound velocity at the boundary $(r, t) = (0, 1)$. The logit-normal sampler with $\mu = -0.4$, $\sigma = 1.0$ assigns negligible probability mass to this region:

$$P(t > 0.95) = 1 - \Phi\!\left(\frac{\text{logit}(0.95) - (-0.4)}{1.0}\right) = 1 - \Phi(3.35) \approx 0.04\%$$

Less than 1 in 2,500 training samples probe the 1-NFE operating point.

**Supporting diagnostic evidence.** The training diagnostics reveal that the compound velocity (MeanFlow self-consistency) is systematically harder than direct velocity prediction:

| Loss component       | Epoch 0   | Epoch 46  | Epoch 689 | Trend                  |
|----------------------|-----------|-----------|-----------|------------------------|
| raw_loss_u (u-head)  | 1.93M     | 5.04M     | 3.62M     | Peaks then slowly falls |
| raw_loss_v (v-head)  | 859K      | 740K      | 709K      | Converges by epoch 46  |
| raw_loss_mf (MF)     | 1.89M     | 9.33M     | 6.53M     | Still decreasing       |
| raw_loss_fm (FM)     | 1.96M     | 747K      | 710K      | Converges by epoch 46  |
| MF/FM loss ratio     | 1.0       | 12.5      | 9.2       | MF dominates           |

The v-head converges quickly (loss_v stable at ~710K from epoch 46 onward), confirming that direct velocity prediction is well-learned. The u-head loss (3.62M at epoch 689) remains 5.1× higher than the v-head loss — the compound velocity self-consistency is the bottleneck. This is expected: the compound velocity $V = u + (t-r) \cdot \text{sg}[\partial_t u]$ requires accurate JVP estimates, which are hardest at the boundary where the $(t-r)$ coefficient is maximal.

### 1.2 Compound Velocity Norm Overshoot

**Observation.** The norm ratio $\|V_\theta\| / \|v_c\|$ evolves through training:

| Metric               | Epoch 0 | Epoch 46 | Epoch 100 | Epoch 689 |
|----------------------|---------|----------|-----------|-----------|
| compound_v_norm      | 855     | 1,403    | 1,339     | 1,092     |
| target_v_norm        | 941     | 941      | 941       | 941       |
| Norm ratio (V/target)| 0.91    | 1.49     | 1.42      | 1.16      |
| jvp_norm             | 3,888   | 11,214   | 9,662     | 6,523     |
| temporal_frac (JVP)  | 0.793   | 0.728    | 0.700     | 0.630     |

This is not a static bias — it is a transient overshoot that peaks around epoch 46 (ratio 1.49) and is converging toward 1.0, reaching 1.16 by epoch 689. The convergence is driven by improving x̂ predictions: as x̂ becomes more accurate, the JVP magnitude decreases (from 11,214 to 6,523) because the $-1/t^2 \cdot \partial_t x_\theta$ term shrinks.

**JVP decomposition.** The temporal component (from the $1/t$ factor in x-prediction) dominates the JVP: 79% at epoch 0, declining to 63% by epoch 689 as the spatial component (from improved x̂ gradients) grows. This temporal dominance is inherent to x-prediction and cannot be eliminated, but it attenuates with training as $x_\theta$ improves.

**Implication for 1-NFE.** At inference, $\hat{z}_0 = z_1 - u_\theta(z_1, 0, 1)$. A 16% norm overshoot means the step overshoots the data manifold. The overshoot is still converging, so extended training (Section 2.2) will reduce it further. A post-training norm correction (Section 2.4) can compensate for any residual bias.

### 1.3 Insufficient Training Duration

**Observation.** The v1 run completed 28,980 optimizer steps (690 epochs × 42 steps/epoch). For reference, the iMF paper trains for ~800K optimizer steps on ImageNet — our budget is ~3.6% of that. Even for medical imaging (smaller dataset, simpler distribution), this is likely insufficient for the compound velocity to converge.

**Evidence of continuing improvement.** The last 5 FID evaluations show the model was still competitive with its best:

| Train epoch | FID (2.5D avg) | Patience counter |
|-------------|----------------|------------------|
| 568         | 12.90          | 6                |
| 598         | 13.09          | 7                |
| 628         | 11.92          | 8                |
| 658         | 12.08          | 9                |
| 688         | 11.88          | 10 (stopped)     |

The FID oscillates between 11.67–13.09 in the last 300 epochs — the model is still fluctuating around a plateau but has not diverged. The patience=10 (= 10 FID evaluation epochs = 300 training epochs) was too aggressive: the model recovered to 11.88 at epoch 688 after a patience spike, but had already exhausted its budget. A patience of 30 would have allowed continued training.

Additional indicators of non-convergence:
- **grad_clip_fraction** fell from 90.5% to 4.8% — gradient norms are still settling.
- **cos(V, v_c)** rose from 0.083 to 0.307 — compound velocity alignment still improving.
- **ema_divergence** fell from 0.036 to 0.043 — EMA is tracking the online model.
- **relative_update_norm** fell from 0.037 to 0.014 — parameter updates are shrinking but non-zero.
- **lr at epoch 689** was 5.63e-5 (cosine decay from 1e-4) — still providing meaningful gradients.

---

## 2. Improvement Specifications

### 2.1 Boundary-Aware Time Sampling

**Priority:** HIGH (minor code change + config, directly addresses 1-NFE deficit)

**Current config:**
```yaml
time_sampling:
  distribution: logit_normal
  mu: -0.4
  sigma: 1.0
  t_min: 0.001
  data_proportion: 0.5
```

**Proposed change.** Implement a mixture distribution for $(t, r)$ sampling:

$$p(t, r) = (1 - \alpha)\,\text{LogitNormal}(\mu, \sigma) + \alpha\,\text{Uniform}([1 - \delta, 1] \times [0, \delta])$$

where $\alpha = 0.1$ is the boundary injection fraction and $\delta = 0.05$ controls the boundary width.

**New config keys:**
```yaml
time_sampling:
  distribution: logit_normal
  mu: -0.4
  sigma: 1.0
  t_min: 0.001
  data_proportion: 0.5
  boundary_fraction: 0.1      # NEW: fraction of batch from boundary region
  boundary_delta: 0.05         # NEW: boundary width around (t=1, r=0)
```

**Implementation.** Modify `src/neuromf/utils/time_sampler.py::sample_t_and_r` to accept `boundary_fraction` and `boundary_delta` parameters. When `boundary_fraction > 0`, sample a fraction $\alpha$ of the batch from uniform $t \in [1 - \delta, 1]$ and $r \in [0, \delta]$, concatenate with standard logit-normal samples, and shuffle.

```python
def sample_t_and_r(
    batch_size: int,
    mu: float = -0.4,
    sigma: float = 1.0,
    t_min: float = 0.001,
    data_proportion: float = 0.5,
    boundary_fraction: float = 0.0,
    boundary_delta: float = 0.05,
    device: torch.device = "cpu",
) -> tuple[Tensor, Tensor]:
    n_boundary = int(batch_size * boundary_fraction)
    n_standard = batch_size - n_boundary

    # Standard logit-normal sampling (existing logic)
    t_std, r_std = _sample_logit_normal(n_standard, mu, sigma, t_min, data_proportion, device)

    # Boundary sampling: (t, r) near (1, 0)
    t_bnd = torch.empty(n_boundary, device=device).uniform_(1.0 - boundary_delta, 1.0)
    r_bnd = torch.empty(n_boundary, device=device).uniform_(t_min, boundary_delta)

    # Concatenate and shuffle
    t = torch.cat([t_std, t_bnd])
    r = torch.cat([r_std, r_bnd])
    perm = torch.randperm(batch_size, device=device)
    return t[perm], r[perm]
```

**Interaction with adaptive weighting.** The adaptive weight $w = \text{loss} / (\text{loss} + \epsilon)$ effectively down-weights high-loss samples (gradient $\propto 1/\text{loss}$ for large losses). Boundary samples will initially have very high raw loss (since the model is untrained at $(t \approx 1, r \approx 0)$), so the adaptive term partially counteracts the boundary injection. This is acceptable: boundary injection provides the signal *exists* at all, and the adaptive weighting prevents it from destabilizing training. The 10% injection fraction is deliberately generous to overcome this damping.

**Verification tests:**

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| BI-T1 | At least 8% of samples have $t > 0.95$ and $r < 0.05$ | Count check over 100 batches |
| BI-T2 | Non-boundary samples follow original logit-normal | KS test $p > 0.05$ on $t$ marginal |
| BI-T3 | `boundary_fraction=0.0` reproduces original sampler | Exact match given same seed |

---

### 2.2 Extended Training + FID-Based Checkpointing

**Priority:** HIGH (config-only changes)

**Current config (v1):**
```yaml
training:
  max_epochs: 1500              # v1 stopped at epoch 690
  save_every_n_epochs: 50

evaluation:
  early_stop_patience: 10       # 10 FID eval epochs = 300 training epochs
  fid_every_n_val_epochs: 3     # FID every 3 val epochs = every 30 training epochs
```

**Proposed changes:**
```yaml
training:
  max_epochs: 3000              # CHANGED: 4.3× v1 budget → ~126K optimizer steps

evaluation:
  early_stop_patience: 30       # CHANGED: 30 FID evals = 300 training epochs (at fid_every=1)
  fid_every_n_val_epochs: 1     # CHANGED: FID every val epoch = every 10 training epochs
```

**Checkpointing strategy.** The existing `train.py` already implements a `ModelCheckpoint` monitoring FID with `save_top_k=1`. Change to `save_top_k=3` to retain the top-3 best 3D-FID checkpoints. This provides rollback options and enables post-hoc analysis across different training phases.

```python
# In experiments/cli/train.py, modify the FID checkpoint callback:
fid_ckpt_cb = ModelCheckpoint(
    dirpath=str(ckpt_dir),
    monitor=fid_monitor,        # "val/fid_3d"
    mode="min",
    save_top_k=3,               # CHANGED from 1 to 3
    filename="best_fid_{epoch:03d}_{" + fid_monitor + ":.2f}",
    save_last=False,
)
```

**Justification.** The v1 run completed 28,980 steps. Scaling to 3,000 epochs yields ~126,000 steps — a 4.3× increase. The diagnostic evidence (Section 1.3) shows the model was still improving: cos(V, v_c) rising, norm ratio converging, FID oscillating rather than diverging. The patience increase from 10 to 30 FID evaluations prevents premature termination during the oscillatory plateau regime observed in v1 (FID bounced between 11.67–13.09 for the final 300 epochs).

**Note:** With `fid_every_n_val_epochs: 1` and `val_every_n_epochs: 10`, the effective patience is 30 × 10 = 300 training epochs — the same as v1 (10 × 30 = 300) but with more frequent monitoring.

---

### 2.3 Learning Rate Warmup

**Priority:** HIGH (config-only change, directly addresses early-training instability)

**Current config:** `warmup_steps: 0` (no warmup)

**Proposed change:**
```yaml
training:
  warmup_steps: 1000            # NEW: ~24 epochs, ~3.4% of v1 training
```

**Justification from diagnostics.** The grad_clip_fraction trajectory reveals severe early-training instability:

| Epoch | Global step | grad_clip_fraction | grad_norm_mean |
|-------|-------------|-------------------|----------------|
| 0     | 42          | 90.5%             | 1.384          |
| 46    | 1,974       | 21.4%             | 0.838          |
| 100   | 4,242       | 31.0%             | 0.910          |
| 689   | 28,980      | 4.8%              | 0.610          |

At epoch 0, **90.5% of gradient updates are clipped** — the optimizer is saturating the gradient norm threshold (1.0) on nearly every step. By epoch 46 (~2,000 steps), this drops to 21.4%. A linear warmup over 1,000 steps would cover this unstable regime, allowing the model to find a reasonable loss basin before applying the full learning rate.

The warmup_steps config key already exists in the training config (currently set to 0). Implementation requires instantiating a warmup scheduler in `train.py` — this needs to be verified, but the config plumbing is already in place.

**Literature support.** Goyal et al. (2017), "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," established that gradual warmup is essential for large effective batch sizes. Our effective batch of 132 qualifies. The iMF and pMF references also use warmup schedules.

---

### 2.4 Inference-Time Norm Correction

**Priority:** MEDIUM (post-hoc calibration, zero retraining cost)

**Observation.** At epoch 689, the compound velocity norm ratio is 1,092/941 = 1.16 — a 16% overshoot. This is down from a peak of 1.49 at epoch 46, confirming convergence, but the residual 16% bias translates to a systematic overshoot at 1-NFE inference: $\hat{z}_0 = z_1 - u_\theta(z_1, 0, 1)$ steps too far from the noise distribution.

**Proposed correction.** Introduce a scalar correction factor $\gamma$ at inference:

$$\hat{z}_0 = z_1 - \frac{u_\theta(z_1, 0, 1)}{\gamma}$$

For multi-step Euler, apply per-step: $z_{t_{i-1}} = z_{t_i} - (t_i - t_{i-1}) \cdot v_\theta(z_{t_i}, t_i) / \gamma$.

**Calibration procedure.** After v2 training completes:
1. Generate 200 samples at each $\gamma \in \{1.0, 1.05, 1.10, 1.15, 1.20\}$ at NFE=1.
2. Compute FID-3D for each $\gamma$.
3. Select the $\gamma$ minimizing FID-3D.
4. Verify the optimal $\gamma$ does not degrade NFE=50 results (FID-3D within ±0.5).

**Config addition:**
```yaml
sampling:
  norm_correction: 1.0          # Default: no correction. Calibrate post-training.
```

**Note:** The v2 model (with extended training + boundary sampling) will likely have a smaller norm ratio than v1's 1.16, since the norm ratio was still converging at epoch 689. The calibration grid is centered around 1.0–1.2 accordingly.

**Verification tests:**

| Test ID | Description | Pass criterion |
|---------|-------------|----------------|
| NC-T1 | $\gamma = 1.0$ reproduces current results exactly | FID-3D matches baseline ± 0.1 |
| NC-T2 | Optimal $\gamma$ reduces FID-3D at NFE=1 by > 5% | Grid search over $\gamma$ values |
| NC-T3 | $\gamma > 1$ does not degrade NFE=50 results | FID-3D at NFE=50 within ± 0.5 |

---

### 2.5 VAE Reconstruction Spectral Baseline

**Priority:** LOW (evaluation-only, no training change)

**Motivation.** The spectral analysis shows HF Energy Ratio ≈ 0.0003 (NFE=1) to 0.0011 (NFE=50) for generated volumes, but without a VAE reconstruction baseline, it is impossible to determine whether this smoothing originates from the generative model or the frozen MAISI decoder. This disentanglement matters for the paper.

**Specification.** Encode 200 real test volumes through the frozen MAISI VAE (encode → decode) and compute the same spectral metrics. If $\rho_\text{VAE-recon} \approx \rho_\text{generated}$, the smoothing is a decoder ceiling. If $\rho_\text{VAE-recon} > \rho_\text{generated}$, the generator introduces additional smoothing.

**Implementation note:** Run on Picasso (A100) since the VAE pass requires GPU memory. Batch at `max_batch_size_vae: 4`.

---

### 2.6 Cosine Alignment Diagnostic Enhancement

**Priority:** LOW (logging-only, no training change)

**Observation.** The aggregate cosine similarities plateau early: cos(V, v_c) ≈ 0.307, cos(ṽ, v_c) ≈ 0.436 at epoch 689. These averages over all $t$ values obscure whether alignment is uniformly poor or concentrated at specific $t$ regimes.

**Specification.** Add per-time-bin logging to the diagnostics callback:

| Bin | $t$ range | Expected behavior |
|-----|-----------|-------------------|
| Near-data | [0.05, 0.2] | Higher cosine (low noise) |
| Mid | [0.2, 0.8] | Moderate cosine |
| Near-noise | [0.8, 1.0] | Lower cosine (high noise) |

Log as: `train/cos_V_vc_bin0`, `train/cos_V_vc_bin1`, `train/cos_V_vc_bin2`. This is a logging-only change — no training behavior modification.

---

## 3. Implementation Order

1. **Config changes** (trivial): `warmup_steps`, `max_epochs`, `early_stop_patience`, `fid_every_n_val_epochs`, `save_top_k` — pure YAML edits + one-line change in `train.py` for save_top_k=3.
2. **Boundary sampling** (minor code change): Modify `time_sampler.py::sample_t_and_r`, add config keys, run verification tests BI-T1..T3.
3. **Warmup scheduler** (verify): Confirm `warmup_steps` is wired to an actual LR scheduler in `train.py`. If not, implement a linear warmup wrapper.
4. **Launch v2 training** on Picasso with all changes enabled.
5. **Post-training — norm correction calibration**: Grid search $\gamma$ on the best checkpoint.
6. **Post-training — VAE spectral baseline**: Encode/decode real test volumes, compare spectra.
7. **Cosine diagnostic enhancement**: Add before or during v2 training.

---

## 4. Config Diff Summary

All proposed changes to the training configuration:

```yaml
# --- CHANGES ---
training:
  max_epochs: 3000              # CHANGED from 1500 (4.3× v1 budget)
  warmup_steps: 1000            # CHANGED from 0 (addresses 90% grad clipping at epoch 0)

time_sampling:
  boundary_fraction: 0.1        # NEW: 10% of batch from boundary region
  boundary_delta: 0.05          # NEW: boundary width around (t=1, r=0)

evaluation:
  early_stop_patience: 30       # CHANGED from 10 (prevents premature termination)
  fid_every_n_val_epochs: 1     # CHANGED from 3 (more frequent FID monitoring)

# --- NEW SECTION ---
sampling:
  norm_correction: 1.0          # Post-training calibration (default: no correction)

# --- CODE CHANGE ---
# In train.py: fid_ckpt_cb save_top_k changed from 1 to 3
```

---

## 5. References

- Geng, Z. et al. (2025). "MeanFlow: One-Step Generative Models via MeanFlow Training." arXiv:2504.18024.
- Geng, Z. et al. (2025). "Improved Mean Flows: On the Challenges of Fastforward Generative Models." arXiv:2512.02012.
- Li, Y. & He, K. (2025). "Pixel MeanFlow: One-step Latent-free Image Generation." arXiv:2506.xxxxx.
- Goyal, P. et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv:1706.02677.
- Lin, S. et al. (2024). "Common Diffusion Noise Schedules and Sample Steps are Flawed." WACV 2024.
- Yazdani, M. et al. (2025). "MOTFM: Multi-Objective Training with Flow Matching." MICCAI 2025.
