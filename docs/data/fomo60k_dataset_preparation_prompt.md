# Agent Task: FOMO60K Dataset Preparation for NeuroMF

## Goal

Prepare ~8,400 T1w brain MRI volumes from FOMO60K for latent encoding and
MeanFlow training. When you finish, every `.nii.gz` T1 file across all eight
datasets must be skull-stripped (brain-only, zero background), and the latent
encoding + train/val splitting code must handle multi-scan subjects correctly.

---

## 0. Environment and Paths

```
CONDA_ENV   = neuromf          # Python 3.10+, PyTorch 2.x, CUDA
PROJECT     = ~/research/code/neuromf/
FOMO60K     = /media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K/
RESULTS     = /media/mpascual/Sandisk2TB/research/neuromf/results/
GPU         = RTX 4060 Laptop, 8 GB VRAM
```

Key source files (read these before writing any code):

| File | Purpose |
|------|---------|
| `src/neuromf/data/fomo60k.py` | `FOMO60KConfig`, `get_fomo60k_file_list()` — metadata filtering |
| `src/neuromf/data/latent_dataset.py` | `LatentDataset` — loads `.pt` files, train/val split |
| `experiments/cli/encode_dataset.py` | Encodes NIfTI→latent `.pt` via frozen MAISI VAE |
| `FOMO60K/mapping.tsv` | Columns: `dataset`, `participant_id`, `session_id`, `filename` |
| `FOMO60K/participants.tsv` | Columns: `dataset`, `participant_id`, `session_id`, `sex`, `age`, `group` |

### Datasets

| Code | Name | Subjects | Skull-stripped? | Action |
|------|------|----------|-----------------|--------|
| PT001 | OASIS1 | 436 | **Yes** | None |
| PT002 | OASIS2 | 362 | **Yes** | None |
| PT005 | IXI | 584 | **Yes** | None |
| PT007 | NIMH | 252 | **Yes** | None |
| PT008 | DLBS | 957 | **Yes** | None |
| PT011 | MBSR | 348 | **Defaced** | **Skull-strip** |
| PT012 | UCLA | 265 | **Defaced** | **Skull-strip** |
| PT015 | NKI | 2,280 | **Defaced** | **Skull-strip** |

---

## TASK 1 — Skull-Strip the Three Defaced Datasets

### 1.1 How HD-BET Works (reference code from another project)

The `brainles_preprocessing` package provides `HDBetExtractor`. Here is the
**exact calling pattern** that is known to work. Do NOT invent a different API;
use this one:

```python
from brainles_preprocessing.brain_extraction.brain_extractor import HDBetExtractor
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label
import torch

def skull_strip_volume(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    mode: str = "fast",       # "fast" for batch, "accurate" for validation
    device: int = 0,          # GPU index
    do_tta: bool = False,     # test-time augmentation (True = slower, better)
) -> dict:
    """Skull-strip one NIfTI volume via HD-BET.

    Writes the brain-extracted image to output_path and the binary brain
    mask to mask_path. Post-processes the mask to keep only the largest
    connected component (removes meningeal/dura fragments).

    Args:
        input_path:  Defaced .nii.gz to process.
        output_path: Where to write skull-stripped .nii.gz.
        mask_path:   Where to write binary brain mask .nii.gz.
        mode:        'fast' (~4 s/vol, ~2 GB VRAM) or 'accurate' (~15 s/vol).
        device:      CUDA device index.
        do_tta:      Enable test-time augmentation for robustness.

    Returns:
        dict with keys 'brain_volume_mm3', 'brain_coverage_percent'.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    # HD-BET writes to temp, we post-process
    temp_out = output_path.parent / f"_temp_hdbet_{output_path.name}"

    extractor = HDBetExtractor()
    extractor.extract(
        input_image_path=input_path,       # Path or str
        masked_image_path=temp_out,        # Path or str
        brain_mask_path=mask_path,         # Path or str
        mode=mode,
        device=device,
        do_tta=do_tta,
    )

    # --- Post-process: largest connected component only ---
    input_img = nib.load(str(input_path))
    mask_img  = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()

    binary = (mask_data > 0).astype(np.int32)
    labeled, n_comp = cc_label(binary)
    if n_comp > 1:
        sizes = np.bincount(labeled.ravel())[1:]   # skip background
        keep  = np.argmax(sizes) + 1
        mask_data = (labeled == keep).astype(mask_data.dtype)
        nib.save(
            nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header),
            str(mask_path),
        )

    # --- Apply cleaned mask to ORIGINAL intensities (not HD-BET output) ---
    input_data = input_img.get_fdata()
    final_data = input_data * mask_data
    nib.save(
        nib.Nifti1Image(final_data, input_img.affine, input_img.header),
        str(output_path),
    )

    # Clean up temp
    if temp_out.exists():
        temp_out.unlink()

    # --- Stats ---
    voxel_vol    = float(np.prod(input_img.header.get_zooms()))
    brain_voxels = int(np.sum(mask_data > 0))
    orig_nonzero = int(np.sum(input_data > 0))

    return {
        "brain_volume_mm3":      brain_voxels * voxel_vol,
        "brain_coverage_percent": brain_voxels / max(orig_nonzero, 1) * 100,
    }
```

### 1.2 Phase A — Validation (run FIRST; stop for human review)

Create `scripts/skull_strip_defaced.py` with a `--phase A` mode that:

1. **Selects 3 test subjects per dataset** (first 3 alphabetically from each
   of PT011_MBSR, PT012_UCLA, PT015_NKI → 9 subjects total).
2. For each, skull-strips the **primary T1** (`t1.nii.gz`) using
   `mode="accurate", do_tta=True` (highest quality for validation).
3. Saves results to a temporary directory:
   ```
   FOMO60K/_skull_strip_validation/{dataset}/{participant_id}/
       skull_stripped.nii.gz
       brain_mask.nii.gz
       visualization.png
   ```
4. **Generates a 3-row × 3-column visualization** per subject (matplotlib):
   - Rows: axial, sagittal, coronal (middle slice)
   - Col 0: original defaced (gray colormap)
   - Col 1: original with mask overlay (red, alpha=0.3)
   - Col 2: skull-stripped result (gray colormap)
   - Title: `"{dataset} / {participant_id}  |  vol={brain_volume_mm3:.0f} mm³  cov={brain_coverage_percent:.1f}%"`
5. **Prints a summary table** to stdout:
   ```
   dataset       | subject  | brain_vol_mm3 | coverage_% | status
   PT011_MBSR    | sub_1    |     1234567   |     45.2   | OK
   ...
   ```
6. **Sanity-check bounds**: brain volume ∈ [800 000, 1 900 000] mm³ and
   coverage ∈ [25%, 75%]. Flag any subject outside these as WARNING.
7. **STOP.** Print: `"Phase A complete. Inspect visualizations at FOMO60K/_skull_strip_validation/ before running Phase B."`

### 1.3 Phase B — Batch Processing (only after I approve Phase A)

Same script with `--phase B`:

1. **Find all T1w files** in the three defaced datasets using `mapping.tsv`:
   filter `dataset ∈ {PT011_MBSR, PT012_UCLA, PT015_NKI}` and `filename`
   matching regex `^t1(_\d+)?\.nii\.gz$`.
2. **Build the work list** of `(dataset, participant_id, session_id, filename)`
   tuples. Print the total count (expect ~2,882).
3. **Check disk space**: verify ≥50 GB free on the target drive.
4. **For each file**:
   a. Construct the full path:
      `FOMO60K/{dataset}/{participant_id}/{session_id}/{filename}`
   b. **Skip if already processed**: check if a companion mask file exists
      at `.../{stem}_brainmask.nii.gz` (e.g., `t1_brainmask.nii.gz`).
      This makes the script **resumable**.
   c. **Back up** the original: copy to `{filename}.bak` in the same
      directory. If `.bak` exists, skip backup.
   d. **Skull-strip** with `mode="fast", do_tta=False, device=0`.
   e. **Overwrite** the original `.nii.gz` with the skull-stripped result.
   f. **Save the brain mask** alongside: `{stem}_brainmask.nii.gz`.
   g. **Log** to `FOMO60K/_skull_strip_log.json` (append):
      `{dataset, participant_id, session_id, filename, brain_volume_mm3, brain_coverage_percent, status, error}`.
5. **Error handling**: catch exceptions per-volume, log `status="FAILED"` with
   the traceback, and **continue** to the next volume.
6. **GPU hygiene**: call `torch.cuda.empty_cache()` every 20 volumes.
7. **Progress**: print every 50 volumes: `"[50/2882] PT015_NKI sub_123 — OK (1.2M mm³, 48%)"`.
8. At the end, print summary: total processed, total skipped, total failed.

**Estimated runtime**: ~12 h at ~15 s/volume in fast mode on RTX 4060.

---

## TASK 2 — Analyse and Fix Multi-Timepoint Data Handling

### 2.1 The Multi-Scan Problem

Several datasets contain **multiple T1 scans per subject per session**. For
example, OASIS-1 has `t1.nii.gz`, `t1_2.nii.gz`, `t1_3.nii.gz`, `t1_4.nii.gz`
for a single subject. With `primary_scan_only: false`, all are included.

This causes **two bugs** in the current codebase:

### 2.2 BUG 1 — Latent File Naming Collision (CRITICAL)

In `experiments/cli/encode_dataset.py`, the function `_build_latent_filename`
produces:

```python
def _build_latent_filename(filepath: Path) -> str:
    session_id = filepath.parent.name            # e.g., "ses_1"
    participant_id = filepath.parent.parent.name  # e.g., "sub_1"
    dataset_name = filepath.parent.parent.parent.name  # e.g., "PT001_OASIS1"
    return f"{dataset_name}_{participant_id}_{session_id}.pt"
```

This means `t1.nii.gz` and `t1_2.nii.gz` from the same subject/session produce
the **same** `.pt` filename: `PT001_OASIS1_sub_1_ses_1.pt`. The second encode
**silently overwrites** the first. Multi-scan data is lost.

**FIX**: include the filename stem:
```python
def _build_latent_filename(filepath: Path) -> str:
    """Build unique .pt name: {dataset}_{participant}_{session}_{stem}.pt

    Examples:
        .../PT001_OASIS1/sub_1/ses_1/t1.nii.gz    -> PT001_OASIS1_sub_1_ses_1_t1.pt
        .../PT001_OASIS1/sub_1/ses_1/t1_2.nii.gz  -> PT001_OASIS1_sub_1_ses_1_t1_2.pt
    """
    stem = filepath.name.replace(".nii.gz", "")   # "t1" or "t1_2"
    session_id = filepath.parent.name
    participant_id = filepath.parent.parent.name
    dataset_name = filepath.parent.parent.parent.name
    return f"{dataset_name}_{participant_id}_{session_id}_{stem}.pt"
```

Also add a **duplicate-detection guard** in the encoding loop:
```python
seen_filenames: set[str] = set()
# ... inside loop:
if latent_filename in seen_filenames:
    logger.error("DUPLICATE latent filename: %s (from %s)", latent_filename, filepath)
    raise RuntimeError(f"Duplicate latent filename collision: {latent_filename}")
seen_filenames.add(latent_filename)
```

### 2.3 BUG 2 — Train/Val Data Leakage

The current `LatentDataset` splits by **shuffling file indices**:
```python
indices = list(range(len(all_files)))
random.Random(split_seed).shuffle(indices)
n_train = int(len(indices) * split_ratio)
```

If subject sub_1 has 4 `.pt` files, some may land in train and others in val.
Since those scans share the same gross anatomy (brain shape, ventricle size,
cortical folding), this inflates validation metrics and masks overfitting.

**FIX**: split by **subject**, not by file. After the naming fix in 2.2, the
subject key is extractable from the `.pt` filename:

```python
def _extract_subject_key(filename: str) -> str:
    """Extract subject identifier from latent filename.

    'PT001_OASIS1_sub_1_ses_1_t1.pt' -> 'PT001_OASIS1_sub_1'

    Convention: {dataset}_{participant_id}_{session_id}_{stem}.pt
    Subject key: everything before the session part, i.e. {dataset}_{participant_id}.
    """
    # Split on '_ses_' to separate subject from session
    parts = filename.replace(".pt", "").split("_ses_")
    if len(parts) >= 2:
        return parts[0]     # e.g., "PT001_OASIS1_sub_1"
    # Fallback: use full filename minus extension (no grouping)
    return filename.replace(".pt", "")
```

Then modify the split logic:
```python
if split is not None:
    # Group files by subject
    subject_to_files: dict[str, list[int]] = {}
    for i, fp in enumerate(all_files):
        key = _extract_subject_key(fp.name)
        subject_to_files.setdefault(key, []).append(i)

    subjects = sorted(subject_to_files.keys())
    random.Random(split_seed).shuffle(subjects)
    n_train_subj = int(len(subjects) * split_ratio)

    if split == "train":
        chosen_subjects = set(subjects[:n_train_subj])
    else:
        chosen_subjects = set(subjects[n_train_subj:])

    selected = sorted(
        i for subj in chosen_subjects for i in subject_to_files[subj]
    )
    self.file_paths = [all_files[i] for i in selected]
    logger.info(
        "LatentDataset: %d files from %d subjects (%s split)",
        len(self.file_paths), len(chosen_subjects), split,
    )
```

Keep the old file-level split as a **fallback** only if `_extract_subject_key`
produces all unique keys (meaning no grouping is possible), with a logged
warning.

### 2.4 Analysis Script

Create `scripts/analyze_multi_timepoint.py` that:

1. Reads `mapping.tsv`.
2. Filters to T1 scans only: `filename` matching `^t1(_\d+)?\.nii\.gz$`.
3. For each dataset, counts:
   - Total subjects (unique `participant_id`)
   - Total T1 scans
   - Subjects with >1 T1 scan ("multi-T1 subjects")
   - Max T1 scans per subject
4. Prints a table:
   ```
   Dataset          | Subjects | T1 scans | Multi-T1 subj | Max T1/subj
   PT001_OASIS1     |      436 |    1,688 |           436 |           6
   PT005_IXI        |      584 |      581 |             0 |           1
   ...
   TOTAL            |    5,484 |    8,xxx |           xxx |           -
   ```
5. **Simulates leakage**: with the old file-level split (seed=42, ratio=0.9),
   count how many subjects have files in BOTH train and val. Print the number.
6. **Verifies fix**: with subject-level split, confirm 0 leaking subjects.

---

## TASK 3 — Tests and End-to-End Verification

### 3.1 New Tests to Add

Add to `tests/test_latent_dataset.py`:

```python
def test_subject_level_split_no_leakage(tmp_path):
    """All files from one subject land in the same split."""
    # Create mock .pt files simulating multi-scan subjects:
    #   PT001_sub_1_ses_1_t1.pt, PT001_sub_1_ses_1_t1_2.pt  (same subject, 2 scans)
    #   PT001_sub_1_ses_1_t1_3.pt                            (same subject, 3rd scan)
    #   PT001_sub_2_ses_1_t1.pt                              (different subject)
    #   PT001_sub_3_ses_1_t1.pt                              (different subject)
    #   PT005_sub_10_ses_1_t1.pt                             (different dataset)
    # Each .pt contains {"z": torch.randn(4, 48, 48, 48), "metadata": {}}
    # Create a mock latent_stats.json with 4-channel stats.
    #
    # Then:
    train_ds = LatentDataset(tmp_path, normalise=False, split="train", split_ratio=0.6, split_seed=42)
    val_ds   = LatentDataset(tmp_path, normalise=False, split="val",   split_ratio=0.6, split_seed=42)

    train_subj = {_extract_subject_key(f.name) for f in train_ds.file_paths}
    val_subj   = {_extract_subject_key(f.name) for f in val_ds.file_paths}

    assert len(train_subj & val_subj) == 0, f"Leaked: {train_subj & val_subj}"
    assert len(train_ds) + len(val_ds) == 6  # all files accounted for


def test_build_latent_filename_includes_stem():
    """Verify the fixed naming includes the filename stem to prevent collisions."""
    from experiments.cli.encode_dataset import _build_latent_filename

    p1 = Path("/data/PT001_OASIS1/sub_1/ses_1/t1.nii.gz")
    p2 = Path("/data/PT001_OASIS1/sub_1/ses_1/t1_2.nii.gz")

    assert _build_latent_filename(p1) == "PT001_OASIS1_sub_1_ses_1_t1.pt"
    assert _build_latent_filename(p2) == "PT001_OASIS1_sub_1_ses_1_t1_2.pt"
    assert _build_latent_filename(p1) != _build_latent_filename(p2)
```

### 3.2 Verify File List After Skull-Stripping (post-Task 1B)

```python
# Run this after Task 1 Phase B completes:
from neuromf.data.fomo60k import FOMO60KConfig, FOMO60KDatasetFilter, get_fomo60k_file_list
from pathlib import Path
from collections import Counter

config = FOMO60KConfig(
    root=Path("/media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K"),
    datasets=[
        FOMO60KDatasetFilter(name="PT001_OASIS1", groups=["nondemented", "very_mild_dementia", "mild_dementia", ""]),
        FOMO60KDatasetFilter(name="PT002_OASIS2", groups=["Nondemented", "Converted"]),
        FOMO60KDatasetFilter(name="PT005_IXI", groups=["Control"]),
        FOMO60KDatasetFilter(name="PT007_NIMH", groups=["Control"]),
        FOMO60KDatasetFilter(name="PT008_DLBS", groups=["Control"]),
        FOMO60KDatasetFilter(name="PT011_MBSR", groups=[""]),
        FOMO60KDatasetFilter(name="PT012_UCLA", groups=["CONTROL"]),
        FOMO60KDatasetFilter(name="PT015_NKI", groups=[""]),
    ],
    sequences=["t1"],
    primary_scan_only=False,
)
files = get_fomo60k_file_list(config)
print(f"Total volumes: {len(files)}")

# Per-dataset breakdown
counts = Counter()
for f in files:
    ds = Path(f["image"]).relative_to(config.root).parts[0]
    counts[ds] += 1
for ds, n in sorted(counts.items()):
    print(f"  {ds}: {n}")

# Expected total: ~7,000-8,500
```

### 3.3 Spot-Check Skull-Stripped Files

For 5 random files from EACH newly processed dataset (PT011, PT012, PT015):
1. `nib.load(path)` succeeds.
2. `>30%` of voxels are exactly 0.0 (background was zeroed).
3. No NaN or Inf values.
4. Companion `{stem}_brainmask.nii.gz` exists.

### 3.4 Run Full Test Suite

```bash
cd ~/research/code/neuromf
python -m pytest tests/test_latent_dataset.py -v --tb=short
python -m pytest tests/ -v --tb=short -k "not picasso"
```

All tests must pass. Zero regressions.

---

## Execution Order

```
1. Read source files:
   - src/neuromf/data/fomo60k.py
   - src/neuromf/data/latent_dataset.py
   - experiments/cli/encode_dataset.py
   - FOMO60K/mapping.tsv (head -20)

2. Task 2.4  →  Run multi-timepoint analysis (understand the data first)
3. Task 2.2  →  Fix _build_latent_filename in encode_dataset.py
4. Task 2.3  →  Fix subject-level split in latent_dataset.py
5. Task 3.1  →  Add new tests, run them

6. Task 1A   →  Skull-strip 9 validation subjects → STOP for human review
7. (after approval) Task 1B → Batch skull-strip ~2,882 volumes

8. Task 3.2  →  Verify file list counts
9. Task 3.3  →  Spot-check skull-stripped files
10. Task 3.4 →  Full test suite
```

**Why this order?** Tasks 2–3 are code changes that take minutes and unblock
everything else. Task 1B is a 12-hour GPU job that should run last (ideally
overnight). Doing the analysis first (Task 2.4) also informs whether the
skull-stripping counts match expectations.

---

## Constraints

- **Never delete** original files without `.bak` backup.
- **Never modify** already skull-stripped datasets (PT001, PT002, PT005, PT007, PT008).
- **Conda env**: `conda activate neuromf` before every command.
- **GPU**: single RTX 4060 8 GB. HD-BET fast mode ≈ 2 GB. Do not parallelise.
- **Disk**: verify ≥50 GB free before batch processing (`.bak` files roughly double storage).
- **Resumable**: check for `{stem}_brainmask.nii.gz` to skip processed files.
- **Iterate until tests pass**: do not consider a task done until its tests are green.
