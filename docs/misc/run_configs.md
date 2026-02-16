# Run Configurations — Picasso Overrides

All runs inherit from the three-layer merge: `picasso/base.yaml` → `configs/train_meanflow.yaml` → Picasso overlay below.

The base `train_meanflow.yaml` already contains the v2 Phase A defaults (cosine LR, weight_decay=1e-4, norm_p=0.5, norm_eps=0.01, EMA decay=0.9999, save_every_n_epochs=10).

---

## Run v2 — Phase A only (current)

Uses `configs/picasso/train_meanflow.yaml` as-is. Augmentation OFF, masking OFF.

| Setting | Value |
|---------|-------|
| `training.augmentation.enabled` | `false` |
| `meanflow.spatial_mask_ratio` | `0.0` (inherited) |

---

## Run v3 — Phase A + B (augmentation)

Picasso overlay diff from v2:

```yaml
# configs/picasso/train_meanflow_v3_aug.yaml
# Inherits everything from v2 overlay, only enable augmentation

training:
  augmentation:
    enabled: true
    flip_prob: 0.5
    flip_axes: [0, 1, 2]
    rotate90_prob: 0.5
    rotate90_axes: [[1, 2]]
    gaussian_noise_prob: 0.3
    gaussian_noise_std_fraction: 0.05
    intensity_scale_prob: 0.3
    intensity_scale_factors: 0.05
```

| Setting | Value |
|---------|-------|
| `training.augmentation.enabled` | **`true`** |
| `meanflow.spatial_mask_ratio` | `0.0` (inherited) |

**Rationale:** With ~990 unique training samples, augmentation provides virtual dataset expansion. Flips + 90-degree rotations exploit brain bilateral symmetry and scanner orientation invariance. Per-channel Gaussian noise (calibrated to 5% of each channel's std) acts as latent-space regularization. Intensity scaling adds robustness to encoding variance.

---

## Run v4 — Phase A + C (spatial masking)

Picasso overlay diff from v2:

```yaml
# configs/picasso/train_meanflow_v4_mask.yaml
# Inherits everything from v2 overlay, only enable spatial masking

meanflow:
  spatial_mask_ratio: 0.5
```

| Setting | Value |
|---------|-------|
| `training.augmentation.enabled` | `false` (inherited) |
| `meanflow.spatial_mask_ratio` | **`0.5`** |

**Rationale:** Masking 50% of spatial voxels during loss computation forces the model to learn globally coherent structure rather than memorizing local patterns. Inspired by MAE-style masking — particularly relevant for our small dataset where overfitting to voxel-level patterns is a risk. Loss is normalized by keep fraction so gradient magnitude stays consistent.

---

## Run v5 — Phase A + B + C (augmentation + masking)

Picasso overlay diff from v2:

```yaml
# configs/picasso/train_meanflow_v5_aug_mask.yaml
# Inherits everything from v2 overlay, enable both augmentation and masking

training:
  augmentation:
    enabled: true
    flip_prob: 0.5
    flip_axes: [0, 1, 2]
    rotate90_prob: 0.5
    rotate90_axes: [[1, 2]]
    gaussian_noise_prob: 0.3
    gaussian_noise_std_fraction: 0.05
    intensity_scale_prob: 0.3
    intensity_scale_factors: 0.05

meanflow:
  spatial_mask_ratio: 0.5
```

| Setting | Value |
|---------|-------|
| `training.augmentation.enabled` | **`true`** |
| `meanflow.spatial_mask_ratio` | **`0.5`** |

**Rationale:** Combined regularization — augmentation provides input-space diversity while spatial masking forces structural coherence at the loss level. These are complementary mechanisms: augmentation changes *what* the model sees, masking changes *how* the model is evaluated.

---

## Summary Table

| Run | Phase A | Phase B (augmentation) | Phase C (masking) | Key overlay diff |
|-----|---------|----------------------|-------------------|-----------------|
| v2 | ON | OFF | OFF | — (baseline) |
| v3 | ON | **ON** | OFF | `augmentation.enabled: true` |
| v4 | ON | OFF | **ON** | `spatial_mask_ratio: 0.5` |
| v5 | ON | **ON** | **ON** | Both enabled |

---

## How to Launch

```bash
# On Picasso login node:
# v2 (current):
bash experiments/slurm/phase_4/train.sh

# v3/v4/v5: create the overlay file, then update the launcher's --config path:
# export TRAIN_CONFIG=configs/picasso/train_meanflow_v3_aug.yaml
# Then modify train.sh or pass --config directly in train_worker.sh
```
