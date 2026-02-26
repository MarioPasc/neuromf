# Technical Report — Research Notes & Key Facts

**Created:** 2026-02-26
**Purpose:** Tracking document for the thesis tutor technical report. Records verified facts, architectural decisions, and insights gathered from exhaustive codebase review.

---

## 1. Architecture Summary (verified from source code)

### 1.1 UNet (MAISIUNetWrapper)
- **Source:** `src/neuromf/wrappers/maisi_unet.py`
- **Backbone:** MONAI `DiffusionModelUNet` (3D)
- **Channels:** [64, 128, 256, 512] — 4 resolution levels
- **Attention:** [false, false, true, true] — only at 2 deepest levels
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
  - v-head branches from the last feature map `h` (same as u-head input)
  - v-head ResBlock: ~221K params; total v-head: ~228K params (<0.01% of total)
  - v-head final conv is zero-initialized (starts at zero, u-head unaffected initially)
  - v-head is disabled at inference — only u-head output used for generation
- **v-head purpose:** Provides directly-supervised JVP tangent vector. Trained with `||v - v_c||^p`.
  - Solves early-training divergence: without v-head, tangent comes from u-head's own (poor) prediction

### 1.3 Time Conditioning
- **Mode:** `t_h` — condition on both t and h=t-r
- **Implementation:** Two separate sinusoidal embedding paths:
  - `t_emb = unet.time_embed(sin(t * 1000))` — reuses MONAI's built-in time embedding MLP
  - `h_emb = self.h_embed(sin(h * 1000))` — new MLP: `Linear(64, 256) -> SiLU -> Linear(256, 256)`
  - `emb = t_emb + h_emb` (additive combination)
- **Scale factor:** TIME_SCALE = 1000.0 (maps [0,1] to [0,1000] for sinusoidal resolution)
- **Justification:** MeanFlow paper Table 1c: (t,h) FID=61.06 vs h-only FID=63.13

### 1.4 VAE (MAISIVAEWrapper)
- **Source:** `src/neuromf/wrappers/maisi_vae.py`
- **Model:** MONAI `AutoencoderKlMaisi` (21M params)
- **Architecture:** 3 encoder levels [64,128,256], 2 ResBlocks per level, no attention
- **Compression:** 4x spatial per axis, 1→4 channels. Input (1,192,192,192) → Latent (4,48,48,48)
- **scale_factor:** 0.96240234375 (applied in decode: z_scaled = z / scale_factor before decoder)
- **Frozen:** All params frozen, eval mode at construction
- **Memory splits:** num_splits=6 (local), num_splits=1 (Picasso)

## 2. Loss Pipeline (verified from source code)

### 2.1 MeanFlow Pipeline (`meanflow_loss.py`)
- **Flow:** interpolate → get tangent → JVP → compound velocity → loss
- **Interpolation:** `z_t = (1-t)*z_0 + t*eps` (t=0 is data, t=1 is noise)
- **Target:** `v_c = eps - z_0` (conditional velocity)
- **Tangent (dual-head):** `v_tangent = v_head(z_t, t, t)` under `torch.no_grad()`
- **JVP:** Exact via `torch.func.jvp` with tangents `(v_tangent, dt=1, dr=0)`
  - Uses `has_aux=True` for dual-head to capture v-head output alongside JVP
- **Compound velocity:** `V = u + (t-r) * sg[du/dt]` where sg = stop_gradient
- **Loss_u:** `||V - v_c||^p` with adaptive weighting: `loss_u = raw_loss_u / (raw_loss_u.detach() + eps)^norm_p`
- **Loss_v:** `||v - v_c||^p` with independent adaptive weighting
- **Total:** `loss = (loss_u + loss_v).mean()`

### 2.2 Lp Loss (`lp_loss.py`)
- Computes `sum_over_spatial(|pred - target|^p)` per sample
- Optional per-channel weighting (not used in best config)
- Reduction modes: "mean", "sum", "none"

### 2.3 x-Prediction Conversion
- Model outputs x_hat directly
- Average velocity: `u = (z_t - x_hat) / t.clamp(min=t_min)` where t_min=0.05
- At inference (1-NFE): `z_0 = model(noise, r=0, t=1)` — x_hat IS the output directly

### 2.4 Adaptive Weighting
- Formula: `weight = (raw_loss.detach() + norm_eps) ^ norm_p`
- Config: `norm_eps=1.0`, `norm_p=1.0` → weight = raw_loss + 1.0
- Effect: normalizes loss magnitude, stabilizes training across timesteps
- Applied independently to loss_u and loss_v

## 3. Training Configuration (best model: xpred_exact_jvp)

### 3.1 Optimizer
- AdamW, lr=1e-4, weight_decay=0, betas=(0.9, 0.95)
- Cosine LR schedule (no warmup)
- Gradient clip norm: 1.0

### 3.2 Batch & Compute
- Picasso: 6 A100-40GB GPUs, DDP
- Per-GPU batch: 2 (exact JVP limits due to ~20GB activation memory)
- Gradient accumulation: 11 steps
- Effective batch: 2 × 6 × 11 = 132 (~128)
- Max epochs: 1500 → ~61,500 optimizer steps

### 3.3 Data
- ~1,379 subjects → 85/10/5 split → ~1,172 train / 138 val / 69 test
- HDF5 shards per dataset: PT001_OASIS1.h5, PT002_OASIS2.h5, PT005_IXI.h5
- Latent shape: (4, 48, 48, 48) per sample
- Augmentation (Picasso): flip_d (prob=0.5), gaussian_noise (prob=0.2, std=5%), intensity_scale (prob=0.2, ±5%)

### 3.4 Time Sampling
- Distribution: logit-normal(mu=-0.4, sigma=1.0)
- t_min: 0.001
- data_proportion: 0.5 (50% FM samples where r=t, 50% MF samples where r<t)

### 3.5 EMA
- Decay: 0.9999

### 3.6 MeanFlow Loss
- p=2.0 (L2 loss)
- Adaptive: true
- norm_eps: 1.0
- norm_p: 1.0
- jvp_strategy: exact
- prediction_type: x

## 4. Key Experimental Findings

### 4.1 x-pred + exact JVP vs u-pred + FD-JVP
- **Best model:** x-pred + exact JVP (best epoch 589)
- **Critical rule:** x-pred + FD-JVP = EXPLOSION (1/t factor in x→u conversion + finite difference = O(1/t²) du/dt)
- **u-pred + FD-JVP:** Works but u-pred baseline collapsed at epoch ~150

### 4.2 Conditioning Mode
- t_h (condition on both t and h=t-r) chosen based on MF Table 1c results

### 4.3 Training Stability Fixes (Phase 4c)
- warmup_steps: 5000→0 (warmup destabilized MF loss)
- beta2: 0.999→0.95 (faster momentum adaptation)
- data_proportion: 0.25→0.75→0.5 (50/50 FM/MF split)
- norm_eps: 0.01→1.0 (prevent gradient explosion in adaptive weighting)
- norm_p: 0.5→1.0 (0.5 caused 1000x gradient explosion)

### 4.4 Resolution Decision
- 128³ → 192³ at 1.0mm isotropic
- Rationale: brain AP extent up to 193mm, brain systematically -13mm off-center
- Solution: CropForegroundd (brain-centered) before ResizeWithPadOrCrop(192³)

## 5. Sampling

### 5.1 One-Step (1-NFE)
- `z_0 = model(noise, r=0, t=1)` for x-prediction (direct output)
- For u-prediction: `z_0 = noise - u_theta(noise, r=0, t=1)`

### 5.2 Multi-Step (Euler)
- Uniform time steps from t=1 to t=0
- At each step: get u (convert from x if needed), then z_{t-dt} = z_t - dt * u
- NFE levels tested: [1, 2, 5, 10]

### 5.3 Generation Pipeline (Phase 5)
- Shared noise protocol: same z_1 across NFE levels for fair comparison
- Latent generation → denormalization (z_hat * std + mean) → VAE decode → clamp [0,1]

## 6. Evaluation Metrics

- **FID:** 2.5D (slice-wise with RadImageNet) and 3D (Med3D features)
- **MMD:** Maximum Mean Discrepancy in feature space
- **Coverage/Density:** From feature extractors
- **MS-SSIM:** 3-level 3D MS-SSIM using MONAI SSIMMetric
- **PSNR:** Peak SNR with actual data range
- **Spectral:** High-frequency energy ratio via 3D FFT
- **SynthSeg:** Morphological realism (regional volumes, Dice)

## 7. Novel Contributions (verified against methodology + code)

1. **First MeanFlow for 3D medical imaging** — MeanFlow applied to 192³ brain MRI in latent space with frozen MAISI VAE
2. **Per-channel Lp loss in latent space** — extends SLIM-Diff from pixel-space DDPM to latent MeanFlow
3. **x-pred vs u-pred ablation in latent 3D UNet** — first investigation of pMF manifold hypothesis in latent space with UNets
4. **iMF dual-head in medical context** — v-head for tangent supervision, critical stability fix for x-pred + exact JVP
5. **LoRA for MeanFlow fine-tuning** — planned for joint FCD image-mask synthesis (Phase 7)
