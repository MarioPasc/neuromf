# Technical Report — Research Notes & Key Facts

**Created:** 2026-02-26
**Updated:** 2026-02-26 (with actual training data, latent stats, FID results)
**Purpose:** Tracking document for the thesis tutor technical report. Records verified facts from exhaustive codebase + results review.

---

## 1. Architecture Summary (verified from source code)

### 1.1 UNet (MAISIUNetWrapper)
- **Source:** `src/neuromf/wrappers/maisi_unet.py`
- **Backbone:** MONAI `DiffusionModelUNet` (3D)
- **Channels:** [64, 128, 256, 512] — 4 resolution levels
- **Attention:** [false, false, true, true] — only at 12³ and 6³ resolution levels
- **Heads:** `num_head_channels=[0, 0, 32, 32]`
- **ResBlocks per level:** 2
- **GroupNorm groups:** 32
- **Total params:** ~178M (from Picasso overlay comment)
- **ResBlock updown:** true (strided conv for downsample)
- **Transformer layers:** 1 per attention level
- **Flash attention:** disabled (required for `torch.func.jvp` compatibility)
- **Gradient checkpointing:** disabled (required for exact JVP)

### 1.2 Dual-Head Architecture (iMF)
- **u-head:** Main UNet output conv (from `self.unet.out(h)`)
- **v-head:** `_VHeadResBlock(64, 32)` + `GroupNorm(32,64)` + `SiLU()` + `Conv3d(64, 4, 3, pad=1)`
  - v-head ResBlock: ~221K params; total v-head: ~228K params (<0.13% of total)
  - v-head final conv is zero-initialized
  - Disabled at inference — only u-head output used for generation
- **v-head tangent quality:** cos(v, v_c) improves from 0.17 (epoch 0) → 0.44 (epoch 689)

### 1.3 Time Conditioning
- **Mode:** `t_h` — condition on both t and h=t-r
- **Implementation:** `emb = unet.time_embed(sin(t*1000)) + h_embed(sin(h*1000))`
- **Justification:** MeanFlow paper Table 1c: (t,h) FID=61.06 vs h-only FID=63.13

### 1.4 VAE (MAISIVAEWrapper)
- **Model:** MONAI `AutoencoderKlMaisi` (21M params)
- **Compression:** 4× per axis, 1→4 channels. (1,192,192,192) → (4,48,48,48)
- **scale_factor:** 0.96240234375
- **Reconstruction quality (Phase 0):** SSIM=0.9213, PSNR=30.86 dB

## 2. Actual Dataset (from `dataset_summary.json`)

### 2.1 Dataset Composition — 8 datasets, NOT 3

| Dataset | Train Subjects | Train Scans | Val Scans | Test Scans |
|---------|---------------|-------------|-----------|------------|
| PT001_OASIS1 | 352 | 1,414 | 170 | 96 |
| PT002_OASIS2 | 71 | 682 | 82 | 37 |
| PT005_IXI | 494 | 494 | 58 | 29 |
| PT007_NIMH | 212 | 428 | 48 | 23 |
| PT008_DLBS | 394 | 807 | 109 | 51 |
| PT011_MBSR | 125 | 295 | 36 | 16 |
| PT012_UCLA | 106 | 106 | 13 | 6 |
| PT015_NKI | 722 | 1,245 | 158 | 68 |
| **Total** | **2,476** | **5,471** | **674** | **326** |

**Key insight:** The training used 5,471 scans (many subjects have multiple sessions), not ~1,172 as previously estimated from subjects alone. This is 4× more data than originally planned.

### 2.2 Effective Training Throughput
- Effective batch size: 132
- Optimizer steps/epoch: 41
- Total optimizer steps: 61,500 (planned), actual: ~28,980 (early-stopped at epoch 690)
- Augmentation: 68% of samples augmented per epoch (~3,720 augmented)

## 3. Latent Statistics (from `latent_stats.json`, n=6,471 encoded volumes)

| Channel | Mean | Std | Skewness | Kurtosis | Min | Max |
|---------|------|-----|----------|----------|-----|-----|
| Ch 0 | -0.053 | 0.970 | +0.105 | -0.123 | -5.82 | 8.46 |
| Ch 1 | -0.185 | 1.019 | +0.002 | -0.019 | -6.05 | 5.89 |
| Ch 2 | -0.051 | 0.970 | +0.045 | +0.099 | -6.76 | 8.31 |
| Ch 3 | +0.001 | 1.011 | +0.069 | +0.144 | -7.16 | 10.56 |

**Cross-channel correlation:** Near-diagonal (max off-diagonal |r| = 0.046). Channels are effectively uncorrelated.

**Distribution shape:** Near-Gaussian — low skewness (|γ| < 0.11) and near-zero excess kurtosis (|κ| < 0.15). The MAISI VAE KL regularisation successfully normalises the latent distribution. Standard deviations cluster around 0.97-1.02, close to the unit Gaussian target.

## 4. Training Results (from `training_summary.json` and `eval_summary.json`)

### 4.1 FID Progression (2.5D, RadImageNet features)

| Epoch | Step | FID_avg | FID_xy | FID_yz | FID_zx | Notes |
|-------|------|---------|--------|--------|--------|-------|
| 9 | 420 | 46.76 | 45.12 | 46.68 | 48.47 | First eval |
| 28 | 1,218 | 28.40 | 17.25 | 43.63 | 24.33 | Rapid initial improvement |
| 88 | 3,738 | 14.61 | 14.08 | 16.25 | 13.49 | Approaching plateau |
| 178 | 7,518 | 14.06 | 13.79 | 15.46 | 12.92 | |
| 268 | 11,298 | 12.94 | 12.77 | 14.06 | 11.98 | |
| **388** | **16,338** | **11.67** | **11.85** | **12.09** | **11.07** | **Best FID** |
| 628 | 26,418 | 11.92 | 12.03 | 12.33 | 11.40 | |
| 688 | 28,938 | 11.88 | 12.14 | 12.08 | 11.42 | Final eval |

**Best 2.5D FID:** 11.67 at epoch 388 (step 16,338). Training early-stopped at epoch 690 after patience=10.

### 4.2 Training Dynamics

| Epoch | Raw Loss | cos(V,v_c) | cos(v,v_c) | Rel Error | Grad Norm | LR |
|-------|----------|-----------|-----------|-----------|-----------|-----|
| 0 | 2,788,299 | 0.083 | 0.170 | 1.353 | 1.384 | 1.0e-4 |
| 50 | 5,316,270 | 0.240 | 0.388 | 1.759 | 0.953 | ~9.9e-5 |
| 100 | 6,275,983 | 0.274 | 0.412 | 1.778 | 0.910 | ~9.7e-5 |
| 388 | 4,724,751 | 0.295 | 0.429 | 1.645 | 1.180 | 8.5e-5 |
| 689 | 4,327,238 | 0.307 | 0.436 | 1.516 | 0.610 | 5.6e-5 |

**Key observation:** Raw loss is NOT the right metric for early stopping. Raw loss is dominated by the MF term (loss_mf ~6.5M vs loss_fm ~710K at convergence), and the adaptive weighting normalises both to ~1.0. FID (computed from EMA weights) is the correct early-stopping metric.

### 4.3 Loss Decomposition at Best FID (epoch 388)

| Component | Raw Loss | Share |
|-----------|----------|-------|
| FM loss (r=t, flow matching) | 715,432 | 10.3% |
| MF loss (r<t, self-consistency) | 6,254,997 | 89.7% |
| v-head loss | 714,646 | — |
| u-head loss (compound V) | 3,485,215 | — |

**Interpretation:** The MF loss is ~9× larger than the FM loss, which is expected: the compound velocity V must also capture the JVP correction term, which is a harder target than the direct velocity at r=t. The adaptive weighting equalises their gradient contributions.

### 4.4 x-hat Statistics

| Epoch | x̂ mean | x̂ std | x̂ min | x̂ max |
|-------|--------|-------|--------|--------|
| 0 | 0.004 | 0.243 | -2.06 | 3.41 |
| 100 | -0.001 | 0.737 | -5.88 | 8.56 |
| 388 | -0.001 | 0.753 | -5.84 | 8.69 |
| 689 | -0.001 | 0.781 | -6.22 | 9.38 |

x̂ std stabilises around 0.75-0.78 (vs true latent std ~0.97-1.02). The ~25% underestimation of variance suggests the model has some residual smoothing, consistent with the known VAE bottleneck.

### 4.5 Timing
- Mean epoch time: 268.1s (~4.5 min)
- Total training time: **~51.4 hours** (~2.1 days)
- Early-stopped at epoch 690 (of planned 1500)
- Hardware: 6× A100-SXM4-40GB

### 4.6 SWD (Sliced Wasserstein Distance)
- First eval (epoch 9): SWD = 1.779
- Best SWD: 1.779 (epoch 9 — SWD improves early, then diverges from FID)
- Final (epoch 689): SWD = 2.555
- **SWD is anti-correlated with FID beyond early training** — not a reliable proxy

## 5. Figure Descriptions (from visual inspection)

### 5.1 `training_dashboard.png` (3×3 grid)
- **(a) FID:** Drops from ~47 → ~12, plateau around epoch 200-400
- **(b) SWD:** Increases monotonically after epoch 50 (diverges from FID)
- **(c) Raw Loss:** FM (blue) ~7e5 stable, MF (red) ~5-8e6 noisy. Total (green) follows MF
- **(d) cos(V, v_c):** Gradually increases from 0.08 → 0.30
- **(e) cos(v̂, v_c):** v-head alignment rises rapidly to ~0.43 by epoch 50, then plateaus
- **(f) JVP Norm:** Noisy, ~5000-8000 range
- **(g) Learning Rate:** Cosine decay from 1e-4 to ~5.6e-5
- **(h) Gradient Norm:** Drops from 1.4 → 0.6, clip fraction drops from 90% → 5%
- **(i) Compound V / Target norm ratio:** ~1.2-1.5, noisy

### 5.2 `decoded_nfe_grid.png` (4×3: NFE rows × view columns)
- **NFE=1:** Recognisable brain anatomy but blurry, low contrast
- **NFE=2:** Substantially sharper, cortical sulci visible
- **NFE=5:** Sharp, clear grey-white matter contrast, anatomically plausible
- **NFE=10:** Similar to NFE=5, marginally crisper

### 5.3 `spectral_evolution.png` (2×2 per channel)
- Radially-averaged power spectrum evolves from flat (noise-like at epoch ~100) to characteristic 1/f brain spectrum
- By epoch 600, all 4 channels show proper low-frequency dominance with high-freq rolloff
- Low-frequency peak sharpens with training, high-frequency content increases slowly

### 5.4 `channel_stats_evolution.png`
- **Mean:** Stable near 0 across all channels and epochs — model preserves mean correctly
- **Std:** Increases from ~0.35 (epoch 100) → ~0.55 (epoch 700), still below true ~0.97-1.02
- **Skewness:** Positive (~0.2-0.4), mildly higher than true distribution (~0.05-0.10)
- **Kurtosis:** Positive (~0.3-0.9), higher than true (~0.1) — model samples are slightly leptokurtic

### 5.5 `nfe_consistency_evolution.png`
- **MSE (1-NFE vs multi-step):** Decreases over training for all NFE levels
- **Cosine similarity:** 1 vs 2 ~0.96, 1 vs 5 ~0.91, 1 vs 10 ~0.90 at convergence
- More Euler steps = more deviation from 1-NFE (expected: model trained for 1-step)

### 5.6 `inter_epoch_delta.png`
- L2 distance between consecutive snapshots decreases over training (model stabilising)
- Cosine similarity between consecutive snapshots increases from 0.92 → 0.99

### 5.7 `sample_evolution_grid.png` / `nfe_comparison_grid.png`
- Latent-space evolution visible: structure emerges by epoch ~100, refines through 400+
- NFE comparison shows clear sharpening: NFE=1 blurry → NFE=10 crisp

## 6. Critical Observations for Report

1. **Dataset is larger than previously documented:** 8 datasets, 6,471 latents (not 1,379 from 3 datasets). The MEMORY.md was outdated.
2. **Early stopping was effective:** Best FID at epoch 388, early-stopped at 690. Only used ~47% of planned 1500 epochs.
3. **1-NFE quality gap is significant:** Visual inspection shows NFE=1 is blurry; NFE>=2 is substantially better. This is a known MeanFlow limitation — the 1-step output captures global structure but lacks high-frequency detail.
4. **v-head converges fast:** cos(v, v_c) reaches ~0.39 by epoch 50 (85% of final 0.44), confirming it provides good tangents from early training.
5. **Generated std is ~55% of true:** Latent std ~0.55 vs true ~1.0. This variance collapse is the primary quality bottleneck.
6. **Gradient clipping rate drops 90% → 5%:** Training becomes very stable by epoch ~300.
