# Phase 5: Evaluation Pipeline — Formal Specification

**Project:** NeuroMF — Improved MeanFlow for 3D Brain MRI Synthesis  
**Author:** Mario (auto-generated specification)  
**Date:** 2026-02-24  
**Status:** SPECIFICATION READY — awaiting implementation  
**Depends on:** Phase 4 trained checkpoint (gate OPEN), Phase 1 latent statistics

---

## 0. Preamble and Motivation

We have achieved a 2.5D-FID of 11.19 at 1-NFE generation in latent space, surpassing MOTFM's 1-NFE 3D-FID of 32.10 (Yazdani et al., 2025, Table 2). This phase formalises the full evaluation pipeline required for a Q1 journal submission, defining: (i) the generation pipeline from latent sampling through VAE decoding to volume storage, (ii) a multi-axis evaluation framework enabling direct comparison with MOTFM and other baselines, and (iii) the storage and I/O design for efficient large-scale generation.

### Primary Baselines for Comparison

| Method | Reference | NFE Range | Key Metrics Reported |
|--------|-----------|-----------|---------------------|
| **MOTFM** | Yazdani et al., MICCAI 2025 | {1, 10, 50} | 3D-FID, MS-SSIM, MMD |
| **DDPM** | Ho et al., NeurIPS 2020 (via MOTFM) | {1, 10, 50} | 3D-FID, MS-SSIM, MMD |
| **MAISI** | Guo et al., 2024 | {50} | VAE reconstruction baseline |
| **Med-DDPM** | Dorjsembe et al., IEEE JBHI 2024 | {1000} | 3D-FID, MS-SSIM |
| **HA-GAN** | Sun et al., IEEE JBHI 2022 | {1} | 3D-FID, MMD |

---

## 1. Evaluation Axes

The evaluation is organised along five orthogonal axes. Each axis answers a distinct scientific question.

### Axis 1 — Distributional Fidelity (Image Quality)

**Question:** Does the generated distribution match the real data distribution?

| Metric | Notation | Direction | Scope | Feature Extractor |
|--------|----------|-----------|-------|-------------------|
| 3D-FID | $\text{FID}_{\text{3D}}$ | $\downarrow$ | Distribution | Med3D ResNet-50 (2048-d) |
| 2.5D-FID | $\text{FID}_{\text{2.5D}}$ | $\downarrow$ | Distribution | RadImageNet ResNet-50 (2048-d) |
| MMD | $\text{MMD}^2$ | $\downarrow$ | Distribution | RBF kernel, median heuristic |
| MS-SSIM | $\overline{\text{MS-SSIM}}$ | $\uparrow$ | Per-volume (mean) | Multi-scale windows |
| Coverage | $C_k$ | $\uparrow$ | Distribution | k-NN in feature space |
| Density | $D_k$ | $\uparrow$ | Distribution | k-NN in feature space |

**Mathematical definitions:**

**3D-FID.** Let $\boldsymbol{\mu}_r, \boldsymbol{\Sigma}_r$ and $\boldsymbol{\mu}_g, \boldsymbol{\Sigma}_g$ be the mean and covariance of Med3D features for real and generated volumes respectively. Then:

$$
\text{FID}_{\text{3D}} = \|\boldsymbol{\mu}_r - \boldsymbol{\mu}_g\|_2^2 + \text{Tr}\!\bigl(\boldsymbol{\Sigma}_r + \boldsymbol{\Sigma}_g - 2(\boldsymbol{\Sigma}_r \boldsymbol{\Sigma}_g)^{1/2}\bigr)
$$

Reference: Heusel et al., "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017.

**MMD.** With RBF kernel $k_\sigma(\mathbf{x}, \mathbf{y}) = \exp\!\bigl(-\|\mathbf{x}-\mathbf{y}\|^2 / (2\sigma^2)\bigr)$:

$$
\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)}\sum_{i \neq j} k(x_i, x_j) + \frac{1}{m(m-1)}\sum_{i \neq j} k(y_i, y_j) - \frac{2}{nm}\sum_{i,j} k(x_i, y_j)
$$

We use the conservative maximum over 5 bandwidths $\sigma \in \{0.2\sigma_m, 0.5\sigma_m, \sigma_m, 2\sigma_m, 5\sigma_m\}$ where $\sigma_m$ is the median pairwise distance (Gretton et al., JMLR 2012).

**MS-SSIM.** Multi-scale structural similarity (Wang et al., Asilomar 2003):

$$
\text{MS-SSIM}(\mathbf{x}, \hat{\mathbf{x}}) = [l_M(\mathbf{x}, \hat{\mathbf{x}})]^{\alpha_M} \prod_{j=1}^{M} [c_j(\mathbf{x}, \hat{\mathbf{x}})]^{\beta_j} [s_j(\mathbf{x}, \hat{\mathbf{x}})]^{\gamma_j}
$$

Computed per-volume on 3D data, then averaged across all $N$ generated volumes. Paired with nearest-neighbour real volumes in Med3D feature space.

**Coverage and Density** (Naeem et al., ICML 2020):

$$
C_k = \frac{1}{|\mathcal{R}|}\sum_{\mathbf{r} \in \mathcal{R}} \mathbb{1}\!\left[\exists\, \mathbf{g} \in \mathcal{G} : \mathbf{g} \in B_k(\mathbf{r})\right], \qquad
D_k = \frac{1}{k|\mathcal{G}|}\sum_{\mathbf{g} \in \mathcal{G}} \sum_{\mathbf{r} \in \mathcal{R}} \mathbb{1}\!\left[\mathbf{g} \in B_k(\mathbf{r})\right]
$$

where $B_k(\mathbf{r})$ is the hypersphere centred at $\mathbf{r}$ with radius equal to the distance to $\mathbf{r}$'s $k$-th nearest real neighbour. We use $k=5$.


### Axis 2 — NFE Scaling

**Question:** How does generation quality degrade as NFE decreases?

| NFE | Sampling Method | Rationale |
|-----|----------------|-----------|
| 1 | `sample_one_step` | Primary claim (MeanFlow 1-NFE) |
| 2 | `sample_euler(n_steps=2)` | MeanFlow 2-NFE (cf. Geng et al., 2025: FID 2.20 on ImageNet) |
| 5 | `sample_euler(n_steps=5)` | Practical low-NFE regime |
| 10 | `sample_euler(n_steps=10)` | MOTFM's fast regime |
| 25 | `sample_euler(n_steps=25)` | Intermediate |
| 50 | `sample_euler(n_steps=50)` | MOTFM/DDPM best regime |

All Axis 1 metrics are computed at each NFE level. This produces the core results table:

```
Table: Evaluation of 3D Brain MRI Unconditional Generation.

              3D-FID ↓         2.5D-FID ↓       MS-SSIM ↑        MMD (×10³) ↓     Coverage ↑       Density ↑
          1   10   50      1   10   50      1   10   50      1   10   50      1   10   50      1   10   50
DDPM     ...  ...  ...    ...  ...  ...    ...  ...  ...    ...  ...  ...    ...  ...  ...    ...  ...  ...
MOTFM    32.1 9.27 7.93   --   --   --    .66  .77  .77    .51  .25  .22    --   --   --     --   --   --
Ours     xxx  xxx  xxx    xxx  xxx  xxx    xxx  xxx  xxx    xxx  xxx  xxx    xxx  xxx  xxx    xxx  xxx  xxx
```

### Axis 3 — Generation Speed

**Question:** What is the wall-clock time per volume at each NFE?

Measurements include three stages, timed independently:

| Stage | Description | Timing Method |
|-------|-------------|---------------|
| T_latent | Forward pass through UNet (latent generation) | `torch.cuda.Event` start/end |
| T_decode | VAE decode from latent to pixel space | `torch.cuda.Event` start/end |
| T_total | T_latent + T_decode | Sum |

Protocol: Generate 100 volumes, discard first 10 (warmup), report mean ± std of remaining 90. Report in milliseconds. GPU: A100 40GB. Batch size: 1 (to measure per-volume latency).

Target comparison table:

```
Table: Sampling Speed Comparison (A100, per volume).

Method          NFE    T_latent (ms)    T_decode (ms)    T_total (ms)
Med-DDPM        1000   --               --               ~1,000,000
MAISI (DDPM)    50     --               --               ~50,000
MOTFM           50     --               --               ~50,000
MOTFM           10     --               --               ~10,000
Ours            50     xxx ± xxx        xxx ± xxx        xxx ± xxx
Ours            10     xxx ± xxx        xxx ± xxx        xxx ± xxx
Ours            1      xxx ± xxx        xxx ± xxx        xxx ± xxx
```

### Axis 4 — Morphological Realism (SynthSeg)

**Question:** Do generated volumes contain anatomically plausible brain structures?

| Metric | Description | Implementation |
|--------|-------------|----------------|
| SynthSeg Success Rate | Fraction of generated volumes where SynthSeg produces labels without error | SynthSeg CLI |
| Regional Volume Correlation | Pearson $r$ between real and generated regional volumes (hippocampus, ventricles, cortex, thalamus, caudate, putamen, cerebellum) | Paired by NN in feature space |
| Regional Volume KL | $D_\text{KL}(p_\text{real} \| p_\text{gen})$ for each regional volume histogram | KDE-based estimation |
| SynthSeg Dice | Mean Dice overlap between SynthSeg labels of matched real-generated pairs | Standard Dice formula |

SynthSeg reference: Billot et al., "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining", Medical Image Analysis, 2023.

### Axis 5 — VAE Smoothing Quantification (Spectral Analysis)

**Question:** How much of the observed smoothing is attributable to the VAE versus the generative model?

Compute the high-frequency energy ratio:

$$
\rho(k_0) = \frac{\sum_{|\mathbf{k}| > k_0} |F(\hat{\mathbf{x}})(\mathbf{k})|^2}{\sum_{\mathbf{k}} |F(\hat{\mathbf{x}})(\mathbf{k})|^2}
$$

where $F$ denotes the 3D DFT, $\mathbf{k}$ is the frequency vector, and $k_0$ is the cutoff frequency (set to $0.5 k_\text{max}$, i.e., half the Nyquist frequency).

Report three conditions:

| Condition | Notation | Source |
|-----------|----------|--------|
| Real volumes | $\rho_\text{real}$ | Test set |
| VAE-reconstructed | $\rho_\text{VAE}$ | $\mathcal{D}_\phi(\mathcal{E}_\phi(\mathbf{x}))$ |
| Generated (1-NFE) | $\rho_\text{gen}$ | Full pipeline |

If $\rho_\text{gen} \approx \rho_\text{VAE} < \rho_\text{real}$, smoothing is attributable to the VAE. If $\rho_\text{gen} < \rho_\text{VAE}$, the generative model introduces additional smoothing.

---

## 2. Generation Pipeline

The generation pipeline is a three-stage process: latent sampling, latent storage, and VAE decoding. The stages are decoupled so that latent generation (GPU-intensive UNet forward pass) can run independently from VAE decoding (GPU-intensive but different memory profile).

### 2.1 Stage A — Latent Generation

```
Input:  Gaussian noise z_1 ~ N(0, I), shape (B, 4, 48, 48, 48)
Model:  Trained MeanFlow UNet (EMA weights)
Output: Normalised latent z_0_hat, shape (B, 4, 48, 48, 48)
```

**For each NFE level N in {1, 2, 5, 10, 25, 50}:**

1. Fix the random seed per NFE level: `seed_nfe = base_seed + nfe * 1000`. This ensures the same noise is used across NFE levels for fair comparison.
2. Generate `N_total = 2000` latent tensors in batches of `B_gen = 8`.
3. Store all latents in an HDF5 archive (Section 3).
4. Record per-batch wall-clock time with CUDA events.

**Shared noise protocol:** For NFE > 1, we use the same initial noise as NFE = 1. This enables the NFE-consistency analysis (how does the same noise produce different outputs under different NFE regimes?).

### 2.2 Stage B — Latent Storage (HDF5)

See Section 3 for the complete storage specification.

### 2.3 Stage C — VAE Decoding

```
Input:  Normalised latent z_0_hat from HDF5
Step 1: Denormalise: z_0 = z_0_hat * sigma + mu  (from latent_stats.json)
Step 2: Undo scale factor: z_vae = z_0 / scale_factor  (scale_factor = 0.96240234375)
Step 3: Decode: x_hat = D_phi(z_vae), shape (1, 1, 192, 192, 192)
Step 4: Clamp to [0, 1] and store as float32
Output: Decoded volume x_hat, shape (1, 192, 192, 192)
```

**Memory management:** Decode one volume at a time. Load VAE once, keep in GPU. Each decoded volume is approximately $192^3 \times 4 \text{ bytes} \approx 28 \text{ MB}$.

**Total storage estimate for 2000 decoded volumes:**
- Latent (float16): $2000 \times 4 \times 48^3 \times 2 \text{ bytes} \approx 1.7 \text{ GB}$ per NFE level
- Decoded (float32): $2000 \times 192^3 \times 4 \text{ bytes} \approx 56 \text{ GB}$ per NFE level

Given the storage cost, we decode only the NFE levels needed for the paper table (typically 1, 10, 50) and defer others.

### 2.4 Real Data Reference Set

The test set for metric computation consists of the held-out real volumes not used for training. These must undergo identical preprocessing:

1. Load from FOMO-60K test split
2. Skull-stripped, RAS-oriented (already in FOMO-60K)
3. Resample to 1mm³ isotropic
4. Intensity normalise to [0, 1] via percentile clipping (1st, 99th)
5. Crop/pad to 192³ (bounding box)
6. Store as float32 in matching HDF5 format

**Real feature caching:** Extract Med3D and RadImageNet features from the real test set once, cache to disk. Reuse across all NFE evaluations.

---

## 3. HDF5 Storage Format Specification

### 3.1 Rationale

HDF5 (Hierarchical Data Format v5) is the optimal format for this workload due to:
- Chunked I/O: random access to individual volumes without loading the entire file
- Compression: gzip/lz4 reduces storage by 2-3× for brain MRI
- Metadata: embed generation parameters, seeds, timing directly alongside data
- Parallel read: multiple metric computation processes can read concurrently

### 3.2 File Layout

```
experiments/stage1_healthy/
├── generation/
│   ├── latents/
│   │   ├── nfe_001.h5        # 2000 latents at 1-NFE
│   │   ├── nfe_002.h5        # 2000 latents at 2-NFE
│   │   ├── nfe_005.h5
│   │   ├── nfe_010.h5
│   │   ├── nfe_025.h5
│   │   └── nfe_050.h5
│   ├── volumes/
│   │   ├── nfe_001.h5        # 2000 decoded volumes at 1-NFE
│   │   ├── nfe_010.h5
│   │   └── nfe_050.h5
│   ├── real_test.h5          # Real test volumes (preprocessed)
│   └── generation_manifest.json
├── features/
│   ├── real_med3d.h5         # Cached Med3D features for real test set
│   ├── real_radimagenet.h5   # Cached RadImageNet features for real test set
│   ├── gen_med3d_nfe001.h5   # Med3D features for generated NFE=1
│   └── ...
├── metrics/
│   ├── metrics_nfe001.json   # All metrics for NFE=1
│   ├── metrics_nfe010.json
│   ├── metrics_nfe050.json
│   ├── metrics_summary.json  # Aggregated table across all NFE
│   ├── timing_log.json       # Detailed timing measurements
│   └── synthseg/
│       ├── real_labels/      # SynthSeg outputs for real volumes
│       ├── gen_labels_nfe001/
│       └── ...
└── verification_report.md
```

### 3.3 HDF5 Schema — Latent Archive

```python
# nfe_001.h5 schema
/latents          Dataset: (2000, 4, 48, 48, 48), dtype=float16, chunks=(1, 4, 48, 48, 48), compression="gzip", compression_opts=4
/noise_seeds      Dataset: (2000,), dtype=int64          # Per-sample seed for reproducibility
/timing_ms        Dataset: (2000,), dtype=float32         # Per-sample generation time (ms)

# Attributes (root group)
attrs["nfe"]              = 1
attrs["n_samples"]        = 2000
attrs["batch_size"]       = 8
attrs["base_seed"]        = 42
attrs["checkpoint_path"]  = "path/to/best_ema.ckpt"
attrs["checkpoint_epoch"] = 500
attrs["prediction_type"]  = "x"
attrs["model_params"]     = 186000000
attrs["timestamp"]        = "2026-02-24T12:00:00Z"
attrs["gpu"]              = "NVIDIA A100 40GB"
attrs["latent_mean"]      = [mu_0, mu_1, mu_2, mu_3]     # From latent_stats.json
attrs["latent_std"]       = [sigma_0, sigma_1, sigma_2, sigma_3]
attrs["scale_factor"]     = 0.96240234375
```

### 3.4 HDF5 Schema — Volume Archive

```python
# nfe_001.h5 (volumes) schema
/volumes          Dataset: (2000, 192, 192, 192), dtype=float32, chunks=(1, 192, 192, 192), compression="gzip", compression_opts=4
/decode_timing_ms Dataset: (2000,), dtype=float32

# Attributes (root group)
attrs["nfe"]              = 1
attrs["n_samples"]        = 2000
attrs["voxel_size_mm"]    = [1.0, 1.0, 1.0]
attrs["volume_shape"]     = [192, 192, 192]
attrs["intensity_range"]  = [0.0, 1.0]
attrs["source_latent_h5"] = "latents/nfe_001.h5"
```

### 3.5 HDF5 Schema — Feature Cache

```python
# real_med3d.h5 schema
/features         Dataset: (N_test, 2048), dtype=float32
/volume_indices   Dataset: (N_test,), dtype=int64         # Index into real_test.h5

# gen_med3d_nfe001.h5 schema
/features         Dataset: (2000, 2048), dtype=float32
/source_indices   Dataset: (2000,), dtype=int64           # Index into nfe_001.h5
```

### 3.6 Generation Manifest

```json
{
  "experiment": "stage1_healthy",
  "model": "NeuroMF",
  "checkpoint": "path/to/best_ema.ckpt",
  "checkpoint_epoch": 500,
  "n_samples_per_nfe": 2000,
  "nfe_levels": [1, 2, 5, 10, 25, 50],
  "decoded_nfe_levels": [1, 10, 50],
  "base_seed": 42,
  "prediction_type": "x",
  "latent_shape": [4, 48, 48, 48],
  "volume_shape": [1, 192, 192, 192],
  "scale_factor": 0.96240234375,
  "real_test_n": null,
  "generation_start": null,
  "generation_end": null,
  "gpu": "NVIDIA A100 40GB",
  "notes": ""
}
```

---

## 4. Metric Computation Pipeline

### 4.1 Execution Order

The metrics pipeline must follow a dependency graph. Some metrics require decoded volumes; others operate on features extracted from decoded volumes.

```
Stage A: Generate latents (all NFE levels)
   │
   ├──► Stage B.1: Compute latent-space metrics (SWD, latent MMD) — FAST
   │
   └──► Stage C: Decode volumes (NFE = {1, 10, 50})
          │
          ├──► Stage D.1: Extract Med3D features
          │      └──► 3D-FID, Coverage, Density
          │
          ├──► Stage D.2: Extract RadImageNet 2.5D features
          │      └──► 2.5D-FID
          │
          ├──► Stage D.3: Compute per-volume metrics
          │      └──► MS-SSIM (paired), PSNR (paired)
          │
          ├──► Stage D.4: Compute distributional metrics on raw features
          │      └──► MMD (in Med3D feature space)
          │
          ├──► Stage D.5: Run SynthSeg
          │      └──► Dice, Volume Correlations, KL
          │
          └──► Stage D.6: Spectral analysis
                 └──► HF energy ratios
```

### 4.2 Pairing Strategy for Per-Volume Metrics

MS-SSIM and PSNR require paired real-generated comparisons. Since unconditional generation has no natural pairing, we adopt the nearest-neighbour pairing protocol:

1. Extract Med3D features for all real test volumes: $\{\mathbf{f}_r^{(i)}\}_{i=1}^{N_\text{test}}$
2. Extract Med3D features for all generated volumes: $\{\mathbf{f}_g^{(j)}\}_{j=1}^{2000}$
3. For each generated volume $j$, find its nearest real neighbour: $\text{NN}(j) = \arg\min_i \|\mathbf{f}_g^{(j)} - \mathbf{f}_r^{(i)}\|_2$
4. Compute MS-SSIM and PSNR between generated volume $j$ and real volume $\text{NN}(j)$
5. Report mean ± std across all 2000 pairs

This is consistent with the MOTFM protocol.

### 4.3 Feature Extraction Details

**Med3D ResNet-50:**
- Weights: `resnet_50_23dataset.pth` (Chen et al., 2019)
- Input: single-channel 3D volume, min-max normalised to [0, 1]
- Output: 2048-d feature vector (global average pooling of last conv block)
- Process one volume at a time to avoid OOM

**RadImageNet ResNet-50 (2.5D):**
- Weights: RadImageNet-pretrained ResNet-50
- Protocol: extract 3 orthogonal central slices (axial, coronal, sagittal), each at 3 neighbouring positions (centre ± 2 slices), yielding 9 slices per volume
- Input per slice: resize to 224×224, replicate to 3 channels
- Output per slice: 2048-d vector; concatenate or average across slices per volume
- Strategy: average the 9 feature vectors to obtain a single 2048-d descriptor per volume

### 4.4 Metrics JSON Schema

```json
{
  "experiment": "stage1_healthy",
  "nfe": 1,
  "n_generated": 2000,
  "n_real": null,
  "timestamp": "2026-02-24T15:00:00Z",
  "distributional": {
    "fid_3d": {
      "value": null,
      "feature_extractor": "Med3D ResNet-50",
      "feature_dim": 2048
    },
    "fid_2d5": {
      "value": null,
      "feature_extractor": "RadImageNet ResNet-50",
      "feature_dim": 2048,
      "slicing": "3_axis_9_slices"
    },
    "mmd": {
      "value": null,
      "kernel": "RBF",
      "n_bandwidths": 5,
      "bandwidth_strategy": "median_heuristic"
    },
    "coverage_k5": {
      "value": null,
      "k": 5,
      "feature_space": "Med3D"
    },
    "density_k5": {
      "value": null,
      "k": 5,
      "feature_space": "Med3D"
    }
  },
  "per_volume": {
    "ms_ssim": {
      "mean": null,
      "std": null,
      "pairing": "nearest_neighbour_med3d"
    },
    "psnr_db": {
      "mean": null,
      "std": null,
      "pairing": "nearest_neighbour_med3d"
    }
  },
  "morphological": {
    "synthseg_success_rate": null,
    "regional_volume_correlation": {
      "hippocampus_L": {"pearson_r": null, "p_value": null},
      "hippocampus_R": {"pearson_r": null, "p_value": null},
      "lateral_ventricle_L": {"pearson_r": null, "p_value": null},
      "lateral_ventricle_R": {"pearson_r": null, "p_value": null},
      "cerebral_cortex": {"pearson_r": null, "p_value": null},
      "thalamus_L": {"pearson_r": null, "p_value": null},
      "caudate_L": {"pearson_r": null, "p_value": null},
      "cerebellum": {"pearson_r": null, "p_value": null}
    },
    "synthseg_dice": {
      "mean": null,
      "std": null
    }
  },
  "spectral": {
    "hf_energy_ratio": {
      "real": null,
      "vae_recon": null,
      "generated": null,
      "cutoff_k0": "0.5_nyquist"
    }
  },
  "timing": {
    "latent_generation_ms": {"mean": null, "std": null},
    "vae_decode_ms": {"mean": null, "std": null},
    "total_ms": {"mean": null, "std": null},
    "gpu": "NVIDIA A100 40GB",
    "batch_size": 1,
    "n_warmup": 10,
    "n_measured": 90
  }
}
```

---

## 5. Statistical Reporting Requirements

Every metric in the paper must be accompanied by appropriate statistical quantification. The following are computed but reported in the supplementary material or paper as context demands.

### 5.1 Confidence Intervals (95% CI)

**For distribution-level metrics (FID, MMD):** Use bootstrap resampling.

1. From the $N = 2000$ generated features, draw $B = 1000$ bootstrap samples of size $N$ with replacement.
2. For each bootstrap sample, compute the metric against the full real feature set.
3. Report the 2.5th and 97.5th percentiles as the 95% CI.

**For per-volume metrics (MS-SSIM, PSNR):** Standard CI from sample mean:

$$
\text{CI}_{95\%} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}
$$

where $\bar{x}$ is the sample mean, $s$ the sample standard deviation, and $n = 2000$.

### 5.2 Cohen's $d$ (Effect Size)

For comparing Ours vs. MOTFM on per-volume metrics:

$$
d = \frac{\bar{x}_\text{ours} - \bar{x}_\text{MOTFM}}{s_\text{pooled}}, \qquad s_\text{pooled} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}
$$

Interpretation: $|d| < 0.2$ (negligible), $0.2 \leq |d| < 0.5$ (small), $0.5 \leq |d| < 0.8$ (medium), $|d| \geq 0.8$ (large).

For distribution-level metrics where we only have a point estimate from each method, Cohen's $d$ is computed over bootstrap samples.

### 5.3 Statistical Tests

**For downstream tasks** (if included): Paired $t$-test or Wilcoxon signed-rank test, with Bonferroni correction for multiple comparisons. Report $p$-value.

**For distributional metrics:** Two-sample permutation test. Under $H_0$: no difference between methods, permute the method labels across bootstrap replicates. Report $p$-value.

### 5.4 Implementation Note

Statistical testing and visualisation modules are **not** implemented in this phase specification. They will be specified separately. However, all metric computation must store the raw per-sample values (not just aggregates) to enable downstream statistical analysis:

- Store per-volume MS-SSIM, PSNR arrays in the HDF5 features archive
- Store bootstrap FID/MMD replicates in metrics JSON
- Store per-region SynthSeg volumes for correlation analysis

---

## 6. Implementation Specification

### 6.1 Module Structure

```
src/neuromf/
├── generation/
│   ├── __init__.py
│   ├── latent_generator.py       # Stage A: batch latent generation
│   ├── volume_decoder.py         # Stage C: batch VAE decoding
│   └── h5_manager.py             # HDF5 read/write utilities
├── metrics/
│   ├── __init__.py               # (exists — extend exports)
│   ├── fid.py                    # (exists — 2.5D-FID)
│   ├── fid_3d.py                 # (exists — 3D-FID)
│   ├── mmd.py                    # (exists — MMD)
│   ├── swd.py                    # (exists — SWD)
│   ├── coverage_density.py       # (exists — C/D)
│   ├── ms_ssim.py                # NEW: 3D MS-SSIM
│   ├── psnr.py                   # NEW: 3D PSNR
│   ├── spectral.py               # NEW: HF energy analysis
│   ├── synthseg_runner.py        # NEW: SynthSeg wrapper
│   ├── feature_extractor.py      # NEW: unified feature extraction
│   ├── pairing.py                # NEW: NN pairing for per-volume metrics
│   └── bootstrap.py              # NEW: bootstrap CI computation
├── sampling/
│   ├── one_step.py               # (exists)
│   └── multi_step.py             # (exists)
└── ...

experiments/cli/
├── generate_latents.py           # NEW: CLI for Stage A
├── decode_volumes.py             # NEW: CLI for Stage C (different from existing decode_samples.py)
├── extract_features.py           # NEW: CLI for feature extraction
├── compute_metrics.py            # NEW: CLI for metric computation
├── run_synthseg.py               # NEW: CLI for SynthSeg
├── measure_timing.py             # NEW: CLI for speed benchmarks
└── run_full_evaluation.py        # NEW: orchestrator CLI

configs/
└── evaluate.yaml                 # NEW: evaluation configuration
```

### 6.2 Key Class — `LatentGenerator`

```python
class LatentGenerator:
    """Batch generation of latent tensors at specified NFE levels.
    
    Generates normalised latents from Gaussian noise via the trained
    MeanFlow model, storing results in chunked HDF5 archives with
    per-sample timing and seed metadata.
    
    Args:
        model: Trained MeanFlow UNet (EMA weights loaded).
        prediction_type: "u" or "x" prediction parameterisation.
        device: CUDA device for generation.
    """
    
    def generate(
        self,
        n_samples: int,
        nfe: int,
        output_path: Path,
        batch_size: int = 8,
        base_seed: int = 42,
        latent_stats: dict | None = None,
    ) -> None:
        """Generate n_samples latents and write to HDF5.
        
        Args:
            n_samples: Total number of latents to generate.
            nfe: Number of function evaluations (1 = one-step).
            output_path: Path to output HDF5 file.
            batch_size: Generation batch size (limited by VRAM).
            base_seed: Random seed (actual seed = base_seed + nfe * 1000).
            latent_stats: Dict with 'mean' and 'std' arrays for denorm metadata.
        """
        ...
```

### 6.3 Key Class — `VolumeDecoder`

```python
class VolumeDecoder:
    """Decode latents from HDF5 through frozen MAISI VAE.
    
    Reads normalised latents from a latent HDF5 archive, denormalises,
    decodes through the frozen MAISI VAE, and writes decoded volumes
    to a volume HDF5 archive.
    
    Args:
        vae_config: MAISIVAEConfig for VAE instantiation.
        latent_mean: Per-channel latent mean (4,).
        latent_std: Per-channel latent std (4,).
        scale_factor: MAISI VAE scale factor (0.96240234375).
        device: CUDA device for decoding.
    """
    
    def decode(
        self,
        latent_h5_path: Path,
        output_h5_path: Path,
        batch_size: int = 1,
    ) -> None:
        """Decode all latents in the archive and write volumes.
        
        Args:
            latent_h5_path: Path to latent HDF5 file.
            output_h5_path: Path to output volume HDF5 file.
            batch_size: Decode batch size (1 recommended for memory).
        """
        ...
```

### 6.4 Key Class — `H5Manager`

```python
@dataclass
class H5LatentConfig:
    """Configuration for latent HDF5 archives."""
    n_samples: int
    latent_shape: tuple[int, ...] = (4, 48, 48, 48)
    dtype: str = "float16"
    compression: str = "gzip"
    compression_opts: int = 4
    chunk_shape: tuple[int, ...] | None = None  # Default: (1, 4, 48, 48, 48)

@dataclass
class H5VolumeConfig:
    """Configuration for volume HDF5 archives."""
    n_samples: int
    volume_shape: tuple[int, ...] = (192, 192, 192)
    dtype: str = "float32"
    compression: str = "gzip"
    compression_opts: int = 4
    chunk_shape: tuple[int, ...] | None = None  # Default: (1, 192, 192, 192)


class H5Manager:
    """Unified HDF5 read/write manager for latents and volumes.
    
    Provides context-managed creation, random-access reading, and
    metadata attachment for both latent and volume archives.
    """
    
    @staticmethod
    def create_latent_archive(path: Path, config: H5LatentConfig, metadata: dict) -> h5py.File: ...
    
    @staticmethod
    def create_volume_archive(path: Path, config: H5VolumeConfig, metadata: dict) -> h5py.File: ...
    
    @staticmethod
    def read_latent(path: Path, index: int | slice) -> torch.Tensor: ...
    
    @staticmethod
    def read_volume(path: Path, index: int | slice) -> torch.Tensor: ...
    
    @staticmethod
    def read_metadata(path: Path) -> dict: ...
```

### 6.5 Key Class — `FeatureExtractor`

```python
class FeatureExtractor:
    """Unified feature extraction from 3D volumes.
    
    Supports multiple feature extraction backends (Med3D, RadImageNet)
    and caches features to HDF5 for reuse across evaluation runs.
    
    Args:
        backend: "med3d" or "radimagenet".
        weights_path: Path to pretrained weights.
        device: CUDA device.
    """
    
    def extract_and_cache(
        self,
        volume_h5_path: Path,
        output_h5_path: Path,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Extract features from all volumes and cache to HDF5.
        
        Returns the full feature matrix (N, D) as a tensor.
        """
        ...
    
    def load_cached(self, feature_h5_path: Path) -> torch.Tensor:
        """Load previously cached features."""
        ...
```

### 6.6 Orchestrator CLI

The `run_full_evaluation.py` script orchestrates the entire pipeline:

```bash
python experiments/cli/run_full_evaluation.py \
    --config configs/evaluate.yaml \
    --checkpoint path/to/best_ema.ckpt \
    --output-dir experiments/stage1_healthy/ \
    --nfe 1 2 5 10 25 50 \
    --decode-nfe 1 10 50 \
    --n-samples 2000 \
    --skip-synthseg          # Optional: skip SynthSeg for faster iteration
```

**Execution order:**
1. Generate latents for all NFE levels (Stage A)
2. Decode volumes for specified NFE levels (Stage C)
3. Extract features (Med3D + RadImageNet) for real and generated
4. Compute all distributional metrics
5. Compute per-volume metrics with NN pairing
6. Run SynthSeg (if not skipped)
7. Compute spectral analysis
8. Measure timing (separate warmup run)
9. Aggregate and write metrics JSON files
10. Write `metrics_summary.json` with the paper table

---

## 7. Configuration Schema

```yaml
# configs/evaluate.yaml

# --- Paths ---
paths:
  checkpoint: ${RESULTS_DST}/phase_4/checkpoints/best_ema.ckpt
  latent_stats: ${RESULTS_DST}/phase_1/latent_stats.json
  real_test_dir: ${DATA_ROOT}/fomo60k/test/
  output_dir: ${RESULTS_DST}/phase_5/stage1_healthy/
  med3d_weights: ${WEIGHTS_DIR}/resnet_50_23dataset.pth
  radimagenet_weights: ${WEIGHTS_DIR}/RadImageNet-ResNet50_notop.pth
  maisi_vae_weights: ${WEIGHTS_DIR}/maisi_vae.pt

# --- VAE ---
vae:
  scale_factor: 0.96240234375
  spatial_dims: 3
  in_channels: 4
  out_channels: 1
  latent_channels: 4
  num_channels: [256, 512, 512]
  num_res_blocks: [2, 2, 2]
  norm_num_groups: 32

# --- Generation ---
generation:
  n_samples: 2000
  nfe_levels: [1, 2, 5, 10, 25, 50]
  decode_nfe_levels: [1, 10, 50]
  batch_size_generate: 8
  batch_size_decode: 1
  base_seed: 42
  prediction_type: "x"
  latent_shape: [4, 48, 48, 48]
  volume_shape: [1, 192, 192, 192]

# --- Feature Extraction ---
features:
  med3d:
    enabled: true
    batch_size: 1
  radimagenet:
    enabled: true
    n_slices_per_axis: 3
    center_fraction: 0.1
    batch_size: 16

# --- Metrics ---
metrics:
  fid_3d: true
  fid_2d5: true
  mmd: true
  ms_ssim: true
  psnr: true
  coverage_density: true
  spectral: true
  synthseg: false          # Enable when SynthSeg is installed

  bootstrap:
    enabled: true
    n_replicates: 1000
    seed: 123

  pairing:
    method: "nearest_neighbour"
    feature_space: "med3d"

# --- Timing ---
timing:
  n_warmup: 10
  n_measured: 90
  batch_size: 1

# --- SynthSeg ---
synthseg:
  binary_path: "mri_synthseg"     # Assumes SynthSeg in PATH
  regions_of_interest:
    - "Left-Hippocampus"
    - "Right-Hippocampus"
    - "Left-Lateral-Ventricle"
    - "Right-Lateral-Ventricle"
    - "Left-Cerebral-Cortex"
    - "Right-Cerebral-Cortex"
    - "Left-Thalamus"
    - "Left-Caudate"
    - "Left-Putamen"
    - "Left-Cerebellum-Cortex"
```

---

## 8. Verification Tests

| Test ID | Description | Pass Criterion | Critical? |
|---------|-------------|----------------|-----------|
| P5-T1 | 2000 latents per NFE level generated without error | HDF5 files exist, shape correct | CRITICAL |
| P5-T2 | Decoded volumes pass sanity checks | Intensity in [0, 1], no NaN/Inf, shape (192, 192, 192) | CRITICAL |
| P5-T3 | 2.5D-FID (1-NFE) < 20 | Consistent with latent-space monitoring (11.19 observed) | CRITICAL |
| P5-T4 | 3D-FID (1-NFE) < 50 | Competitive with MOTFM 1-NFE (32.10) | CRITICAL |
| P5-T5 | MS-SSIM (1-NFE) > 0.60 | Matches or exceeds MOTFM 1-NFE (0.66) | CRITICAL |
| P5-T6 | 1-NFE latent generation < 500 ms per volume | Significantly faster than multi-step baselines | CRITICAL |
| P5-T7 | SynthSeg runs on ≥ 95% of generated volumes | Anatomical plausibility | DESIRABLE |
| P5-T8 | Regional volume Pearson $r > 0.5$ for major structures | Morphological realism | DESIRABLE |
| P5-T9 | Spectral $\rho_\text{gen} \geq 0.9 \times \rho_\text{VAE}$ | No additional smoothing from generative model | DESIRABLE |
| P5-T10 | All HDF5 archives readable and metadata consistent | Integrity check | CRITICAL |

**Phase 5 PASSES when P5-T1 through P5-T6 and P5-T10 are ALL green.**

---

## 9. Failure Modes and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 3D-FID > 50 at 1-NFE | Weakens primary claim | Low (given 2.5D-FID 11.19) | Compare with VAE reconstruction FID as upper bound; visual inspection |
| OOM during VAE decode of 192³ | Blocks volume generation | Medium | Decode batch_size=1; use amp autocast |
| Disk space exceeded (56 GB per NFE) | Blocks storage | Medium | Decode only {1, 10, 50}; use gzip compression |
| Med3D weights unavailable | Blocks 3D-FID | Low | Fall back to 2.5D-FID only; download from original repo |
| SynthSeg fails on synthetic volumes | Limits morphological analysis | Medium | Report success rate; use FreeSurfer as fallback |
| MS-SSIM paired comparison biased | Misleading per-volume metrics | Low | Also report unpaired distributional metrics; verify NN pairing quality |
| Bootstrap CI too wide | Unconvincing statistical claims | Low | Increase to N=2000+ bootstrap replicates |

---

## 10. Paper Table Templates

### Table 1 (Main Results — 3D Brain MRI Unconditional Generation)

The central results table mirrors and extends MOTFM Table 2:

```
Method          NFE    3D-FID ↓    2.5D-FID ↓    MS-SSIM ↑    MMD (×10³) ↓    Coverage ↑    Density ↑
────────────────────────────────────────────────────────────────────────────────────────────────────────
DDPM (*)         1     146.47       --             0.06          39.80           --            --
                10      51.68       --             0.51          26.10           --            --
                50      29.67       --             0.59           4.28           --            --

MOTFM (*)        1      32.10       --             0.66           0.51           --            --
                10       9.27       --             0.77           0.25           --            --
                50       7.93       --             0.77           0.22           --            --

Ours             1       xxx        xxx            xxx           xxx            xxx           xxx
                 2       xxx        xxx            xxx           xxx            xxx           xxx
                10       xxx        xxx            xxx           xxx            xxx           xxx
                50       xxx        xxx            xxx           xxx            xxx           xxx

VAE Recon.       --      xxx        xxx            xxx            --            --            --

(*) Values from Yazdani et al., MICCAI 2025.
```

### Table 2 (Timing Comparison)

```
Method          NFE    Time/Volume (ms)    Speedup vs MOTFM-50
──────────────────────────────────────────────────────────────
Med-DDPM        1000   ~1,000,000           --
MAISI (DDPM)    50     ~50,000              1×
MOTFM           50     ~50,000              1×
MOTFM           10     ~10,000              5×
Ours            50     xxx ± xxx            xxx×
Ours            10     xxx ± xxx            xxx×
Ours             1     xxx ± xxx            xxx×
```

### Table 3 (SynthSeg Morphological Analysis)

```
Region                  Pearson r    p-value    KL(real || gen)
─────────────────────────────────────────────────────────────
Left Hippocampus        xxx          xxx        xxx
Right Hippocampus       xxx          xxx        xxx
Left Lat. Ventricle     xxx          xxx        xxx
Right Lat. Ventricle    xxx          xxx        xxx
Cerebral Cortex         xxx          xxx        xxx
Thalamus                xxx          xxx        xxx
Mean Dice               xxx ± xxx    --         --
```

---

## 11. Dependencies and External Resources

| Resource | Source | Required For |
|----------|--------|-------------|
| Med3D ResNet-50 weights | [github.com/Tencent/MedicalNet](https://github.com/Tencent/MedicalNet) | 3D-FID |
| RadImageNet ResNet-50 weights | [github.com/BMEII-AI/RadImageNet](https://github.com/BMEII-AI/RadImageNet) | 2.5D-FID |
| SynthSeg | [github.com/BBillot/SynthSeg](https://github.com/BBillot/SynthSeg) | Morphological metrics |
| h5py | PyPI | HDF5 I/O |
| scikit-image | PyPI | MS-SSIM, PSNR |
| scipy | PyPI | Bootstrap, statistical tests |
| MONAI | PyPI | VAE wrapper, transforms |
| torch-fidelity | PyPI (optional) | FID validation |

---

## 12. Execution Checklist for Implementation Agent

This section provides the ordered task list for a local agent implementing this specification.

1. **Create module structure** — set up `src/neuromf/generation/` with `__init__.py`, `h5_manager.py`, `latent_generator.py`, `volume_decoder.py`
2. **Implement `H5Manager`** — HDF5 create/read/write with chunking, compression, metadata
3. **Implement `LatentGenerator`** — batch latent generation with timing, seed management, HDF5 output
4. **Implement `VolumeDecoder`** — batch VAE decoding from latent HDF5 to volume HDF5
5. **Implement `ms_ssim.py`** — 3D MS-SSIM metric (leverage `skimage` or `torchmetrics`)
6. **Implement `psnr.py`** — 3D PSNR metric
7. **Implement `spectral.py`** — HF energy ratio via 3D FFT
8. **Implement `feature_extractor.py`** — unified Med3D/RadImageNet extraction with HDF5 caching
9. **Implement `pairing.py`** — NN pairing in feature space
10. **Implement `bootstrap.py`** — bootstrap CI for distributional metrics
11. **Implement `synthseg_runner.py`** — SynthSeg CLI wrapper with volume I/O
12. **Create `configs/evaluate.yaml`** — full configuration
13. **Create CLIs** — `generate_latents.py`, `decode_volumes.py`, `extract_features.py`, `compute_metrics.py`, `run_synthseg.py`, `measure_timing.py`
14. **Create `run_full_evaluation.py`** — orchestrator
15. **Write unit tests** — for H5Manager, each metric, LatentGenerator (with tiny model)
16. **Run on small subset** — 10 samples, verify all metrics compute correctly
17. **Run full generation** — 2000 samples at all NFE levels
18. **Write `verification_report.md`** — pass/fail for all P5-T* tests

---

## 13. References

1. Yazdani, M., et al. "Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality." MICCAI 2025.
2. Geng, Z., et al. "Mean Flows for One-step Generative Modeling." arXiv:2505.13447, 2025.
3. Lu, Y., et al. "One-step Latent-free Image Generation with Pixel Mean Flows." arXiv:2601.22158, 2026.
4. Zheng, M., et al. "Improved Mean Flows for One-step Generative Modeling." arXiv, 2026.
5. Heusel, M., et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS 2017.
6. Wang, Z., Simoncelli, E.P., and Bovik, A.C. "Multiscale Structural Similarity for Image Quality Assessment." Asilomar 2003.
7. Gretton, A., et al. "A Kernel Two-Sample Test." JMLR 13(1), 2012.
8. Naeem, M.F., et al. "Reliable Fidelity and Diversity Metrics for Generative Models." ICML 2020.
9. Billot, B., et al. "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining." Medical Image Analysis, 2023.
10. Chen, S., et al. "Med3D: Transfer Learning for 3D Medical Image Analysis." arXiv:1904.00625, 2019.
11. Sun, L., et al. "Hierarchical Amortized GAN for 3D High Resolution Medical Image Synthesis." IEEE JBHI, 2022.
12. Dorjsembe, Z., et al. "Conditional Diffusion Models for Semantic 3D Brain MRI Synthesis." IEEE JBHI, 2024.
13. Guo, P., et al. "MAISI: Medical AI for Synthetic Imaging." arXiv, 2024.
14. Lipman, Y., et al. "Flow Matching for Generative Modeling." arXiv:2210.02747, 2022.
15. Ho, J., Jain, A., and Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
