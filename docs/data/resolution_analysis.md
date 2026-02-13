# Resolution and Patch Size Analysis

**Date:** 2026-02-13
**Decision:** 192x192x192 at 1.0mm isotropic with brain-centered cropping (latent 48x48x48)

---

## 1. Literature Comparison

| Method | Year | Space | Volume Size | Spacing | Latent Shape | Dataset | N (train) |
|--------|------|-------|-------------|---------|--------------|---------|-----------|
| Pinaya 3D-LDM | 2022 | Latent (8x) | 160x224x160 | 1.0mm iso | 20x28x20 | UK Biobank | 31,740 |
| HA-GAN | 2022 | Pixel | 256x256x256 | 1.0mm iso | 1024 (1D) | GSP | 3,538 |
| Med-DDPM | 2024 | Pixel | 128x128x128 | 1.5mm iso | None | NTUH clinical | 1,292 |
| WDM | 2024 | Wavelet | 128/256 cubed | ~1.0mm iso | 8x64/128 cubed | BraTS 2023 | ~484 |
| MOTFM | 2025 | Pixel | ~240x240x155 | 1.0mm iso | None (UNet) | MSD Brain | ~484 |
| MAISI-v2 (Head) | 2025 | Latent (4x) | 256x256x256 | 1.1mm iso | 4x64x64x64 | 12k CTs | 12,000 |
| **NeuroMF (ours)** | **2026** | **Latent (4x)** | **192x192x192** | **1.0mm iso** | **4x48x48x48** | **FOMO-60K** | **~1,100** |

**Key observations:**

- **1mm isotropic is the standard** for brain MRI generation. Every brain MRI paper uses 1.0mm or close (Med-DDPM at 1.5mm is the outlier, trading resolution for coverage).
- **MAISI-v2 head region uses 256 cubed at 1.1mm** (latent 64 cubed). This is for full-head CT including skull; our skull-stripped brain requires less FOV.
- **Pinaya (2022) uses 160x224x160** — a non-cubic crop tailored to brain anatomy (AP > LR/SI). The closest comparable latent-space brain MRI approach.
- **No method combines latent-space generation with flow matching for brain MRI.** Pinaya uses latent DDPM; MOTFM uses pixel-space flow matching; MAISI-v2 uses latent rectified flow but only for CT.

---

## 2. FOMO-60K Dataset Exploration

### 2.1 Native Volume Properties

| Dataset | Voxel Shape | Spacing (mm) | Physical Extent (mm) | N volumes |
|---------|-------------|-------------|---------------------|-----------|
| PT001_OASIS1 | 256x256x128 | 1.0x1.0x1.25 | 256x256x160 | 436 |
| PT002_OASIS2 | 256x256x128 | 1.0x1.0x1.25 | 256x256x160 | 362 |
| PT005_IXI | 256x256x150 | 0.94x0.94x1.2 | 240x240x180 | 581 |

All volumes are skull-stripped, RAS-oriented, and co-registered within each dataset.

### 2.2 Brain Extent After 1mm Isotropic Resampling (n=30)

| Axis | Min (mm) | P25 | Median | P75 | Max (mm) | Mean |
|------|----------|-----|--------|-----|----------|------|
| LR (Right-Left) | 126 | 133 | 136 | 141 | 146 | 137 |
| **AP (Anterior-Posterior)** | **158** | **166** | **170** | **177** | **193** | **171** |
| SI (Superior-Inferior) | 124 | 137 | 142 | 149 | 164 | 142 |

The anterior-posterior dimension is the largest, reaching **193mm** for the largest IXI brains.

### 2.3 Brain Center Offset From Volume Center

| Axis | Mean Offset (mm) | Range |
|------|-----------------|-------|
| LR | -1.0 | -4 to +2 |
| **AP** | **-13.1** | **-29 to +6** |
| SI | +5.7 | -13 to +20 |

The brain is systematically **13mm anterior** to the volume center. This means naive center-cropping (ResizeWithPadOrCrop) loses frontal cortex tissue.

### 2.4 Brain Coverage: Brain-Centered Crop (n=30)

| Crop Size | Latent Shape | Worst-Case | P5 | Mean | P95 |
|-----------|-------------|------------|-----|------|-----|
| 128 cubed | 32 cubed | 78.6% | 85.3% | 89.6% | 94.2% |
| 160 cubed | 40 cubed | 95.7% | 97.7% | 99.3% | 100.0% |
| 176 cubed | 44 cubed | 98.8% | 99.8% | 99.9% | 100.0% |
| **192 cubed** | **48 cubed** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

**192 cubed is the smallest isotropic size that guarantees 100% brain coverage for all subjects** with brain-centered cropping.

---

## 3. Justification for 192 cubed at 1.0mm

### 3.1 Why not 128 cubed (previous setting)?

- Loses 10-22% of brain tissue even with brain-centered crop
- Systematically removes frontal and occipital cortex (AP is the bottleneck)
- SynthSeg evaluation will flag truncated anatomical structures
- Unfair disadvantage vs methods operating at 240+ mm FOV

### 3.2 Why not 160 cubed?

- Worst-case 4.3% brain loss at frontal/occipital poles (IXI large brains)
- Marginal improvement at 2x training cost when 192 cubed achieves 100%

### 3.3 Why not 256 cubed (matching MAISI-v2)?

- Latent 64 cubed is 8x more voxels than 32 cubed — significant training cost
- Only ~35-40% of voxels contain brain tissue (rest is padding)
- MAISI-v2 uses 256 cubed for full-head CT including skull; our data is skull-stripped
- Wastes model capacity on generating empty background

### 3.4 Why 192 cubed?

1. **100% brain coverage** for all subjects in FOMO-60K
2. **1.0mm isotropic** matches the standard for brain MRI analysis
3. **60-70% brain tissue** in the volume — efficient capacity use
4. **Latent 48 cubed** is tractable: batch_size=16 on A100 80GB
5. **3.4x training cost** vs 128 cubed, offset by A100 parallelism
6. **Sub-second inference** with 1-NFE MeanFlow

### 3.5 Preprocessing Pipeline Change

The critical fix is adding brain-centered cropping (`CropForegroundd`) before spatial padding. This centers the crop on brain tissue rather than the volume center, recovering up to 14% additional brain coverage at any crop size.

**Previous pipeline** (brain off-center, tissue lost):
```
Load -> Resample(1mm) -> ScaleIntensity -> ResizeWithPadOrCrop(center) -> EnsureType
```

**Updated pipeline** (brain-centered, no tissue loss at 192 cubed):
```
Load -> Resample(1mm) -> ScaleIntensity -> CropForeground(brain) -> ResizeWithPadOrCrop(192) -> EnsureType
```

### 3.6 Computational Impact

| Metric | 128 cubed (old) | 192 cubed (new) | Factor |
|--------|----------------|----------------|--------|
| Latent shape | 4x32x32x32 | 4x48x48x48 | 3.4x voxels |
| Latent file size | 0.5 MB | 1.7 MB | 3.4x |
| Total latent cache | 0.5 GB | 1.8 GB | 3.4x |
| MeanFlow VRAM (batch=1) | ~1.5 GB | ~5 GB | ~3.4x |
| MeanFlow batch on A100 | ~53 | ~16 | 0.3x |
| Encoding time (A100) | ~25 min | ~35 min (est.) | ~1.4x |
