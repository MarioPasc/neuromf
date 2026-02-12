# FOMO-60K Dataset Exploration

**Location:** `/media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K/`
**Explored:** 2026-02-12

## Dataset Summary

FOMO-60K is a large-scale preprocessed brain MRI dataset. We use a 3-dataset
subset: PT001_OASIS1, PT002_OASIS2, PT005_IXI.

| Property | Value |
|----------|-------|
| Total T1 sessions (subset) | ~1,379 |
| Format | NIfTI compressed (.nii.gz) |
| Preprocessing done | Skull-stripped, RAS-oriented, co-registered |
| Structure | `{dataset}/sub_{id}/ses_{n}/{sequence}.nii.gz` |

## Directory Structure

```
FOMO60K/
  participants.tsv         # demographics + group labels
  mapping.tsv              # filename mapping to original datasets
  mri_info.tsv             # MRI acquisition parameters
  PT001_OASIS1/
    sub_1/ses_1/t1.nii.gz
    ...
  PT002_OASIS2/
    sub_XXX/ses_1/t1.nii.gz
    ...
  PT005_IXI/
    sub_XXX/ses_1/t1.nii.gz
    ...
```

## Metadata Files

### participants.tsv
Columns: `dataset`, `participant_id`, `session_id`, `sex`, `age`, `handedness`, `group`

### mapping.tsv
Columns: `dataset`, `participant_id`, `session_id`, `filename`, `original_path`

### mri_info.tsv
Columns: `dataset`, `participant_id`, `session_id`, `filename`, `Modality`,
`MagneticFieldStrength`, `Manufacturer`, `ManufacturersModelName`, `SoftwareVersions`,
`MRAcquisitionType`, `SeriesDescription`, `ProtocolName`, `ScanningSequence`,
`SequenceVariant`, `ScanOptions`, `SequenceName`, `EchoTime`, `SliceThickness`,
`RepetitionTime`, `InversionTime`, `FlipAngle`

## Per-Dataset Properties

### PT001_OASIS1 (436 primary T1 sessions)

| Property | Value |
|----------|-------|
| Shape | 256 x 256 x 128 |
| Spacing | 1.0 x 1.0 x 1.25 mm |
| Group labels | `nondemented` (135), NaN/empty (201), dementia groups (100) |
| Field strength | 1.5T |
| Scanner | Siemens Vision |

### PT002_OASIS2 (362 primary T1 sessions)

| Property | Value |
|----------|-------|
| Shape | 256 x 256 x 128 |
| Spacing | 1.0 x 1.0 x 1.25 mm |
| Group labels | `Nondemented` (183), `Demented` (142), `Converted` (37) |

### PT005_IXI (581 primary T1 sessions)

| Property | Value |
|----------|-------|
| Shape | 256 x 256 x 150 |
| Spacing | 0.94 x 0.94 x 1.2 mm |
| Group labels | `Control` (584, includes multiple scans) |
| Sites | Guys, HH, IOP |

## Filtering for Healthy Controls

Default config (`configs/fomo60k.yaml`) filters to:
- PT001_OASIS1: `nondemented` + NaN/empty group (336 subjects)
- PT002_OASIS2: `Nondemented` (183 subjects)
- PT005_IXI: `Control` (581 subjects)

**Total healthy-control T1 volumes: ~1,100** (primary scans only)

## Preprocessing Pipeline

FOMO-60K data is already skull-stripped and RAS-oriented. Our pipeline
(`mri_preprocessing.py`) does:

1. **LoadImaged** - Load NIfTI
2. **EnsureChannelFirstd** - Add channel dim
3. **Spacingd(1mm iso)** - Resample to isotropic (datasets have different native spacings)
4. **ScaleIntensityRangePercentilesd(0-99.5 -> 0-1)** - Intensity normalisation
5. **ResizeWithPadOrCropd(128^3)** - Center crop/pad to target
6. **EnsureTyped(float32)** - Cast to float32

Note: `Orientationd` is NOT needed (already RAS in FOMO-60K).

## Primary Scan Filtering

`mapping.tsv` contains multiple scans per session (e.g., `t1.nii.gz`, `t1_2.nii.gz`,
`t1_3.nii.gz`). Setting `primary_scan_only: true` in config selects only `t1.nii.gz`,
avoiding duplicate/repeat scans of the same subject.
