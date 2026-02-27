# 3D-FID Evaluation: Med3D Feature Collapse and R3D-18 Migration

## Summary

The original 3D-FID implementation used a **Med3D ResNet-50** feature extractor
(pretrained on 23 medical segmentation datasets). Investigation revealed that
this model produces **collapsed features** — cosine similarity ≈ 1.0 for all
inputs regardless of content — making 3D-FID scores meaningless.

We replaced it with **torchvision R3D-18** (Kinetics-400 pretrained), matching
the protocol released by the MOTFM authors (Feb 2026). R3D-18 produces
discriminative 512-d features with no collapse.

## The Med3D Collapse Problem

### Symptoms

- 3D-FID between real brain MRI and clearly blurry 1-NFE generations: **0.012**
  (suspiciously good)
- 3D-FID between real brain MRI and random Gaussian noise: **< 1.0**
- Cosine similarity between features of any two inputs: **≈ 1.0**

### Root Cause

Med3D ResNet-50 uses **dilated convolutions** in layers 3–4 (dilation 2 and 4,
stride 1) followed by **AdaptiveAvgPool3d(1,1,1)**. This architectural combination
causes feature collapse:

1. Dilated convolutions with stride=1 maintain spatial resolution at the cost of
   an effectively smoothed receptive field
2. Global average pooling then collapses the spatial dimensions
3. The result: all inputs produce near-identical feature vectors

This affects **both available Med3D weight files**:

| Weight file | Non-zero features (of 2048) | Cosine sim between random inputs |
|-------------|---------------------------|----------------------------------|
| `resnet_50.pth` | 545 | 1.0000 |
| `resnet_50_23dataset.pth` | 1290 | 1.0000 |

### Verification Experiment

```
Distribution A: 8 random noise volumes (48³)
Distribution B: 8 heavily smoothed volumes (48³, box kernel k=11)

Med3D ResNet-50:  FID = 0.41   (collapsed — should be >> 0)
R3D-18:           FID = 109.19 (discriminative — correctly separates distributions)
```

### HA-GAN Comparison

HA-GAN (Hierarchical Amortised GAN) uses the same Med3D ResNet-50 architecture
for their 3D-FID evaluation (`fid_score.py` + `resnet3D.py`). Their
`get_feature_extractor()` replaces `conv_seg` with `AdaptiveAvgPool3d + Flatten`
→ nominally 2048-d output, but subject to the same collapse issue. Their
implementation loads `resnet_50.pth` (not the 23-dataset version).

**Both HA-GAN and our original implementation suffer from the same Med3D
feature collapse**, meaning published 3D-FID values computed with Med3D may not
be meaningful distributional metrics.

## R3D-18 Protocol (MOTFM)

The replacement protocol matches MOTFM's `evaluation_3d/evaluate_3d.py` exactly:

| Aspect | Detail |
|--------|--------|
| **Backbone** | `torchvision.models.video.r3d_18` |
| **Pretraining** | Kinetics-400 (video classification) |
| **FC head** | Replaced with `nn.Identity()` |
| **Feature dim** | 512 |
| **Normalisation** | Per-set min-max to [0, 1] (global across all volumes, NOT per-volume) |
| **Channel handling** | Single-channel MRI replicated to 3 channels via `repeat` |
| **FID computation** | MONAI `FIDMetric` (standard Fréchet distance) |

### Domain Mismatch Note

R3D-18 is pretrained on natural video (Kinetics-400), not medical images. This
is analogous to using InceptionV3 (ImageNet) for 2D-FID — the domain mismatch
is accepted by the community because what matters is:

1. **Consistent feature space** for comparing methods
2. **Discriminative features** (no collapse)
3. **Reproducibility** (standard pretrained weights, no custom training)

For our evaluation, we report both:
- **3D-FID** (R3D-18, MOTFM protocol) — comparable with MOTFM published numbers
- **2.5D-FID** (RadImageNet ResNet-50) — domain-appropriate primary metric

## File Changes

| File | Change |
|------|--------|
| `src/neuromf/metrics/fid_3d.py` | Rewritten: Med3D → R3D-18 |
| `src/neuromf/metrics/feature_extractor.py` | Updated `med3d` backend to use R3D-18 |
| `src/neuromf/callbacks/evaluation.py` | Updated training-time FID callback |
| `experiments/cli/compute_metrics.py` | Updated feature extractor description |
| `tests/test_fid_3d.py` | Rewritten: 9 tests for R3D-18 protocol |
| `configs/generate.yaml` | Updated features section |
| `configs/picasso/generate.yaml` | Updated weights path for Picasso |

## Weights Location

| Environment | Path |
|-------------|------|
| Local | `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/fid_weights/r3d_18_fid3d/r3d_18-b3b3357e.pth` |
| Picasso | `/mnt/home/users/tic_163_uma/mpascual/fscratch/checkpoints/r3d_18_fid3d/r3d_18-b3b3357e.pth` |
| Auto-download | `~/.cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth` (torchvision default) |

Note: torchvision downloads R3D-18 weights automatically on first use. The
explicit paths above ensure offline evaluation on compute nodes without internet.

## Test Results

All 9 tests pass (`tests/test_fid_3d.py`):

| Test | Description | Status |
|------|-------------|--------|
| T1 | R3D-18 loads, eval mode, ~33M params, dim=512 | PASS |
| T2 | Backward-compat `load_med3d_resnet50()` alias | PASS |
| T3 | Forward pass shape (B, 512) at 32³ and 48³ | PASS |
| T4 | Feature extraction deterministic | PASS |
| T5 | Per-set min-max normalisation (global) | PASS |
| T6 | FID = 0 for identical features | PASS |
| T7 | FID > 1 for different distributions (no collapse) | PASS |
| T8 | Channel replication (1ch→3ch, 3ch pass, 4ch crop) | PASS |
| T9 | Input shape flexibility (3D, 4D, 5D) | PASS |
