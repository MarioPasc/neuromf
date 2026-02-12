# IXI Dataset Exploration

**Location:** `/media/mpascual/Sandisk2TB/research/neuromf/datasets/IXI/IXI-T1/`
**Explored:** 2026-02-12

## Dataset Summary

| Property | Value |
|----------|-------|
| File count | 581 |
| Modality | T1-weighted MRI only (no T2, PD, FLAIR) |
| Format | NIfTI compressed (.nii.gz) |
| Total size | ~4.6 GB |
| Sites | Guys, HH, IOP |
| Naming pattern | `IXI{ID}-{Site}-{Number}-T1.nii.gz` |
| Metadata | `IXI.xls` spreadsheet (demographics, site info) |

## Per-Volume Properties

All checked volumes show consistent properties:

| Property | Value |
|----------|-------|
| Shape | (256, 256, 150) |
| Voxel spacing | (0.94, 0.94, 1.2) mm |
| Dtype | float32 (via nibabel) |
| Intensity range | [0, ~1000-3000] (varies per volume) |

### Sample Volumes

| File | Shape | Spacing (mm) | Range | Mean |
|------|-------|-------------|-------|------|
| IXI002-Guys-0828-T1 | (256, 256, 150) | (0.94, 0.94, 1.2) | [0, 1068] | 105.9 |
| IXI115-Guys-0738-T1 | (256, 256, 150) | (0.94, 0.94, 1.2) | [0, 3006] | 388.6 |
| IXI338-HH-1971-T1 | (256, 256, 150) | (0.94, 0.94, 1.2) | [0, 1474] | 136.7 |
| IXI559-HH-2394-T1 | (256, 256, 150) | (0.94, 0.94, 1.2) | [0, 1744] | 116.6 |

## Preprocessing Pipeline (to reach 128^3 at ~1mm isotropic)

Based on MAISI reference code (`src/external/NV-Generate-CTMR/scripts/transforms.py`):

```python
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, \
    ScaleIntensityRangePercentilesd, ResizeWithPadOrCropd, EnsureTyped

preprocessing = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=[1.0, 1.0, 1.0], mode="bilinear"),
    ScaleIntensityRangePercentilesd(
        keys=["image"],
        lower=0.0,
        upper=99.5,
        b_min=0.0,
        b_max=1.0,
        clip=False,
    ),
    ResizeWithPadOrCropd(keys=["image"], spatial_size=[128, 128, 128]),
    EnsureTyped(keys=["image"], dtype="float32"),
])
```

### Step-by-step:
1. **Orientationd(axcodes="RAS")** — Reorient to Right-Anterior-Superior standard
2. **Spacingd(pixdim=[1.0, 1.0, 1.0])** — Resample from (0.94, 0.94, 1.2) to 1mm isotropic
3. **ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0)** — Percentile-based normalization to [0, 1]
4. **ResizeWithPadOrCropd(spatial_size=[128, 128, 128])** — Center crop/pad to target size

### Notes:
- After resampling to 1mm isotropic, approximate shape: ~(241, 241, 180)
- Center cropping to 128^3 loses peripheral anatomy but preserves brain core
- 8GB VRAM constraint: batch_size=1 for VAE encoding at 128^3
- Consider `CropForegroundd` or brain extraction for more focused cropping
