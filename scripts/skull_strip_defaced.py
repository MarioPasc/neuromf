#!/usr/bin/env python3
"""Skull-strip defaced FOMO-60K datasets using HD-BET.

Phase A: Validate on 9 subjects (3 per dataset), stop for human review.
Phase B: Batch-process all T1 volumes in defaced datasets.

Usage:
    python scripts/skull_strip_defaced.py --phase A
    python scripts/skull_strip_defaced.py --phase B
    python scripts/skull_strip_defaced.py --phase A --fomo60k-root /path/to/FOMO60K
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import traceback
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import label as cc_label

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from brainles_preprocessing.brain_extraction.brain_extractor import HDBetExtractor  # noqa: E402

logger = logging.getLogger(__name__)

DEFACED_DATASETS = ["PT011_MBSR", "PT012_UCLA", "PT015_NKI"]
T1_PATTERN = re.compile(r"^t1(_\d+)?\.nii\.gz$")

# Sanity-check bounds
MIN_BRAIN_VOL_MM3 = 800_000
MAX_BRAIN_VOL_MM3 = 1_900_000
MIN_COVERAGE_PCT = 25.0
MAX_COVERAGE_PCT = 75.0


def skull_strip_volume(
    input_path: Path,
    output_path: Path,
    mask_path: Path,
    mode: str = "fast",
    device: int = 0,
    do_tta: bool = False,
) -> dict:
    """Skull-strip one NIfTI volume via HD-BET.

    Writes the brain-extracted image to output_path and the binary brain
    mask to mask_path. Post-processes the mask to keep only the largest
    connected component (removes meningeal/dura fragments).

    Args:
        input_path: Defaced .nii.gz to process.
        output_path: Where to write skull-stripped .nii.gz.
        mask_path: Where to write binary brain mask .nii.gz.
        mode: 'fast' (~4 s/vol, ~2 GB VRAM) or 'accurate' (~15 s/vol).
        device: CUDA device index.
        do_tta: Enable test-time augmentation for robustness.

    Returns:
        Dict with keys 'brain_volume_mm3', 'brain_coverage_percent'.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    # HD-BET writes to temp, we post-process
    temp_out = output_path.parent / f"_temp_hdbet_{output_path.name}"

    extractor = HDBetExtractor()
    extractor.extract(
        input_image_path=input_path,
        masked_image_path=temp_out,
        brain_mask_path=mask_path,
        mode=mode,
        device=device,
        do_tta=do_tta,
    )

    # Post-process: largest connected component only
    input_img = nib.load(str(input_path))
    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata()

    binary = (mask_data > 0).astype(np.int32)
    labeled, n_comp = cc_label(binary)
    if n_comp > 1:
        sizes = np.bincount(labeled.ravel())[1:]  # skip background
        keep = np.argmax(sizes) + 1
        mask_data = (labeled == keep).astype(mask_data.dtype)
        nib.save(
            nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header),
            str(mask_path),
        )

    # Apply cleaned mask to ORIGINAL intensities (not HD-BET output)
    input_data = input_img.get_fdata()
    final_data = input_data * mask_data
    nib.save(
        nib.Nifti1Image(final_data, input_img.affine, input_img.header),
        str(output_path),
    )

    # Clean up temp
    if temp_out.exists():
        temp_out.unlink()

    # Stats
    voxel_vol = float(np.prod(input_img.header.get_zooms()))
    brain_voxels = int(np.sum(mask_data > 0))
    orig_nonzero = int(np.sum(input_data > 0))

    return {
        "brain_volume_mm3": brain_voxels * voxel_vol,
        "brain_coverage_percent": brain_voxels / max(orig_nonzero, 1) * 100,
    }


def create_visualization(
    input_path: Path,
    mask_path: Path,
    stripped_path: Path,
    output_png: Path,
    title: str,
) -> None:
    """Create 3x3 visualization grid for quality review.

    Rows: axial, sagittal, coronal (middle slice).
    Cols: original, original+mask overlay, skull-stripped.

    Args:
        input_path: Original defaced NIfTI.
        mask_path: Binary brain mask NIfTI.
        stripped_path: Skull-stripped NIfTI.
        output_png: Where to save the visualization.
        title: Figure title.
    """
    orig_data = nib.load(str(input_path)).get_fdata()
    mask_data = nib.load(str(mask_path)).get_fdata()
    strip_data = nib.load(str(stripped_path)).get_fdata()

    # Middle slices
    slices = [
        ("Axial", lambda d: d[:, :, d.shape[2] // 2].T),
        ("Sagittal", lambda d: d[d.shape[0] // 2, :, :].T),
        ("Coronal", lambda d: d[:, d.shape[1] // 2, :].T),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=14)

    col_titles = ["Original (defaced)", "Original + Mask", "Skull-stripped"]

    for row_idx, (view_name, slicer) in enumerate(slices):
        orig_slice = slicer(orig_data)
        mask_slice = slicer(mask_data)
        strip_slice = slicer(strip_data)

        # Col 0: original
        axes[row_idx, 0].imshow(orig_slice, cmap="gray", origin="lower")
        axes[row_idx, 0].set_ylabel(view_name, fontsize=12)

        # Col 1: original + mask overlay
        axes[row_idx, 1].imshow(orig_slice, cmap="gray", origin="lower")
        mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        axes[row_idx, 1].imshow(mask_overlay, cmap="Reds", alpha=0.3, origin="lower")

        # Col 2: skull-stripped
        axes[row_idx, 2].imshow(strip_slice, cmap="gray", origin="lower")

        for col in range(3):
            axes[row_idx, col].axis("off")
            if row_idx == 0:
                axes[row_idx, col].set_title(col_titles[col], fontsize=11)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_summary_visualization(
    fomo60k_root: Path,
    log_path: Path,
    output_png: Path,
    n_per_dataset: int = 3,
) -> None:
    """Create a 3-row × 9-col summary of skull-stripped results.

    Rows: Axial, Coronal, Sagittal (middle slice of skull-stripped volume).
    Columns: 3 subjects per defaced dataset (9 total).

    Picks subjects from the processing log, selecting 3 representative
    OK subjects per dataset (evenly spaced through the alphabetical list).

    Args:
        fomo60k_root: FOMO-60K root directory.
        log_path: Path to ``_skull_strip_log.json`` (JSONL format).
        output_png: Where to save the summary figure.
        n_per_dataset: Number of subjects to show per dataset.
    """
    # Parse log to find OK subjects per dataset
    ok_by_dataset: dict[str, list[dict]] = {ds: [] for ds in DEFACED_DATASETS}
    seen_subjects: dict[str, set[str]] = {ds: set() for ds in DEFACED_DATASETS}

    with open(str(log_path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ds = entry["dataset"]
            pid = entry["participant_id"]
            if ds in ok_by_dataset and entry["status"] == "OK" and pid not in seen_subjects[ds]:
                seen_subjects[ds].add(pid)
                ok_by_dataset[ds].append(entry)

    # Pick n_per_dataset evenly spaced subjects per dataset
    samples: list[tuple[str, str, Path]] = []  # (dataset, participant_id, t1_path)
    for ds in DEFACED_DATASETS:
        entries = sorted(ok_by_dataset[ds], key=lambda e: e["participant_id"])
        if not entries:
            print(f"WARNING: No OK entries for {ds} in log, skipping")
            continue
        n = min(n_per_dataset, len(entries))
        indices = np.linspace(0, len(entries) - 1, n, dtype=int)
        for idx in indices:
            e = entries[idx]
            t1_path = (
                fomo60k_root / e["dataset"] / e["participant_id"] / e["session_id"] / e["filename"]
            )
            samples.append((e["dataset"], e["participant_id"], t1_path))

    if not samples:
        print("WARNING: No samples available for summary visualization.")
        return

    n_cols = len(samples)
    view_names = ["Axial", "Coronal", "Sagittal"]
    slicers = [
        lambda d: d[:, :, d.shape[2] // 2].T,
        lambda d: d[:, d.shape[1] // 2, :].T,
        lambda d: d[d.shape[0] // 2, :, :].T,
    ]

    fig, axes = plt.subplots(3, n_cols, figsize=(3 * n_cols, 9))
    fig.suptitle("Skull-Strip Results — Representative Samples", fontsize=14, y=1.02)

    # Handle case where axes might be 1D if n_cols==1
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col_idx, (ds, pid, t1_path) in enumerate(samples):
        if not t1_path.exists():
            for row_idx in range(3):
                axes[row_idx, col_idx].text(0.5, 0.5, "Missing", ha="center", va="center")
                axes[row_idx, col_idx].axis("off")
            axes[0, col_idx].set_title(f"{ds}\n{pid}", fontsize=8)
            continue

        data = nib.load(str(t1_path)).get_fdata()

        for row_idx, (view_name, slicer) in enumerate(zip(view_names, slicers)):
            sl = slicer(data)
            axes[row_idx, col_idx].imshow(sl, cmap="gray", origin="lower")
            axes[row_idx, col_idx].axis("off")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(view_name, fontsize=11)

        axes[0, col_idx].set_title(f"{ds}\n{pid}", fontsize=8)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary visualization saved to {output_png}")


def get_t1_files_for_dataset(
    mapping: pd.DataFrame,
    dataset: str,
    fomo60k_root: Path,
) -> list[dict]:
    """Get all T1 files for a dataset from mapping.tsv.

    Args:
        mapping: Full mapping DataFrame.
        dataset: Dataset name.
        fomo60k_root: FOMO-60K root directory.

    Returns:
        List of dicts with keys: dataset, participant_id, session_id,
        filename, path.
    """
    dm = mapping[mapping["dataset"] == dataset].copy()
    t1_mask = dm["filename"].apply(lambda fn: bool(T1_PATTERN.match(str(fn))))
    dm = dm[t1_mask]

    results = []
    for _, row in dm.iterrows():
        path = (
            fomo60k_root
            / row["dataset"]
            / row["participant_id"]
            / row["session_id"]
            / row["filename"]
        )
        results.append(
            {
                "dataset": row["dataset"],
                "participant_id": row["participant_id"],
                "session_id": row["session_id"],
                "filename": row["filename"],
                "path": path,
            }
        )
    return results


def run_phase_a(fomo60k_root: Path, device: int) -> None:
    """Phase A: Validate skull-stripping on 9 subjects (3 per defaced dataset).

    Args:
        fomo60k_root: FOMO-60K root directory.
        device: CUDA device index.
    """
    participants = pd.read_csv(fomo60k_root / "participants.tsv", sep="\t")
    mapping = pd.read_csv(fomo60k_root / "mapping.tsv", sep="\t")

    val_dir = fomo60k_root / "_skull_strip_validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for ds in DEFACED_DATASETS:
        # Get first 3 subjects alphabetically
        ds_parts = participants[participants["dataset"] == ds]
        subjects = sorted(ds_parts["participant_id"].unique())[:3]

        if not subjects:
            print(f"WARNING: No subjects found for {ds}")
            continue

        for subj in subjects:
            # Find primary T1 path
            dm = mapping[
                (mapping["dataset"] == ds)
                & (mapping["participant_id"] == subj)
                & (mapping["filename"] == "t1.nii.gz")
            ]
            if dm.empty:
                print(f"WARNING: No primary T1 for {ds}/{subj}")
                continue

            row = dm.iloc[0]
            input_path = fomo60k_root / ds / subj / row["session_id"] / "t1.nii.gz"
            if not input_path.exists():
                print(f"WARNING: File not found: {input_path}")
                continue

            out_dir = val_dir / ds / subj
            out_dir.mkdir(parents=True, exist_ok=True)

            stripped_path = out_dir / "skull_stripped.nii.gz"
            mask_path = out_dir / "brain_mask.nii.gz"
            viz_path = out_dir / "visualization.png"

            print(f"Processing {ds}/{subj}...", flush=True)
            stats = skull_strip_volume(
                input_path=input_path,
                output_path=stripped_path,
                mask_path=mask_path,
                mode="accurate",
                device=device,
                do_tta=True,
            )

            # Create visualization
            create_visualization(
                input_path=input_path,
                mask_path=mask_path,
                stripped_path=stripped_path,
                output_png=viz_path,
                title=(
                    f"{ds} / {subj}  |  "
                    f"vol={stats['brain_volume_mm3']:.0f} mm³  "
                    f"cov={stats['brain_coverage_percent']:.1f}%"
                ),
            )

            # Sanity check
            vol = stats["brain_volume_mm3"]
            cov = stats["brain_coverage_percent"]
            status = "OK"
            if vol < MIN_BRAIN_VOL_MM3 or vol > MAX_BRAIN_VOL_MM3:
                status = "WARNING (volume)"
            if cov < MIN_COVERAGE_PCT or cov > MAX_COVERAGE_PCT:
                status = "WARNING (coverage)"

            results.append(
                {
                    "dataset": ds,
                    "subject": subj,
                    "brain_vol_mm3": vol,
                    "coverage_pct": cov,
                    "status": status,
                }
            )

            torch.cuda.empty_cache()

    # Print summary table
    print()
    print("=" * 80)
    print("Phase A — Skull-Strip Validation Summary")
    print("=" * 80)
    header = (
        f"{'dataset':<16} {'subject':<12} {'brain_vol_mm3':>14} {'coverage_%':>11} {'status':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<16} {r['subject']:<12} "
            f"{r['brain_vol_mm3']:>14.0f} {r['coverage_pct']:>11.1f} "
            f"{r['status']:>10}"
        )
    print()
    print(f"Phase A complete. Inspect visualizations at {val_dir}/ before running Phase B.")


def run_phase_b(
    fomo60k_root: Path,
    device: int,
    worker_id: int = 0,
    num_workers: int = 1,
) -> None:
    """Phase B: Batch skull-strip all T1 volumes in defaced datasets.

    Supports multi-GPU parallelism via ``worker_id`` / ``num_workers``.
    Each worker processes a round-robin slice of the full work list
    (worker *k* handles items *k, k+N, k+2N, ...*). All workers append
    to the same JSONL log file.

    Args:
        fomo60k_root: FOMO-60K root directory.
        device: CUDA device index.
        worker_id: This worker's index (0-based).
        num_workers: Total number of parallel workers.
    """
    mapping = pd.read_csv(fomo60k_root / "mapping.tsv", sep="\t")

    # Build full work list (deterministic order)
    full_work_list = []
    for ds in DEFACED_DATASETS:
        files = get_t1_files_for_dataset(mapping, ds, fomo60k_root)
        full_work_list.extend(files)

    total_all = len(full_work_list)

    # Round-robin slice for this worker
    work_list = full_work_list[worker_id::num_workers]
    total = len(work_list)

    print(f"Worker {worker_id}/{num_workers}: {total} files (of {total_all} total)")

    # Check disk space (only worker 0 to avoid race)
    if worker_id == 0:
        disk_usage = shutil.disk_usage(str(fomo60k_root))
        free_gb = disk_usage.free / (1024**3)
        print(f"Disk free: {free_gb:.1f} GB")
        if free_gb < 50:
            print(f"ERROR: Only {free_gb:.1f} GB free, need ≥50 GB. Aborting.")
            return

    log_path = fomo60k_root / "_skull_strip_log.json"
    n_processed = 0
    n_skipped = 0
    n_failed = 0

    for i, item in enumerate(work_list):
        path = item["path"]
        stem = path.name.replace(".nii.gz", "")
        mask_companion = path.parent / f"{stem}_brainmask.nii.gz"

        # Skip if already processed
        if mask_companion.exists():
            n_skipped += 1
            continue

        if not path.exists():
            n_skipped += 1
            continue

        try:
            # Backup original
            bak_path = path.parent / f"{path.name}.bak"
            if not bak_path.exists():
                shutil.copy2(str(path), str(bak_path))

            # Skull-strip — overwrite original, save mask alongside
            stats = skull_strip_volume(
                input_path=bak_path,  # read from backup
                output_path=path,  # overwrite original
                mask_path=mask_companion,
                mode="fast",
                device=device,
                do_tta=False,
            )

            n_processed += 1

            # Log
            log_entry = {
                "dataset": item["dataset"],
                "participant_id": item["participant_id"],
                "session_id": item["session_id"],
                "filename": item["filename"],
                "brain_volume_mm3": stats["brain_volume_mm3"],
                "brain_coverage_percent": stats["brain_coverage_percent"],
                "status": "OK",
                "error": None,
            }

        except Exception as e:
            n_failed += 1
            log_entry = {
                "dataset": item["dataset"],
                "participant_id": item["participant_id"],
                "session_id": item["session_id"],
                "filename": item["filename"],
                "brain_volume_mm3": None,
                "brain_coverage_percent": None,
                "status": "FAILED",
                "error": traceback.format_exc(),
            }
            print(f"FAILED: {item['dataset']}/{item['participant_id']}/{item['filename']}: {e}")

        # Append to log (atomic for short lines on POSIX)
        with open(str(log_path), "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # GPU hygiene
        if (n_processed + n_failed) % 20 == 0:
            torch.cuda.empty_cache()

        # Progress
        total_done = n_processed + n_skipped + n_failed
        if total_done % 50 == 0 or total_done == total:
            print(
                f"[W{worker_id}] [{total_done}/{total}] {item['dataset']} {item['participant_id']} — "
                f"processed={n_processed} skipped={n_skipped} failed={n_failed}",
                flush=True,
            )

    # Final summary
    print()
    print("=" * 60)
    print(f"Phase B — Worker {worker_id}/{num_workers} Summary")
    print("=" * 60)
    print(f"  Assigned:     {total}")
    print(f"  Processed:    {n_processed}")
    print(f"  Skipped:      {n_skipped}")
    print(f"  Failed:       {n_failed}")
    print(f"  Log file:     {log_path}")


def run_visualize(fomo60k_root: Path) -> None:
    """Generate summary visualization from the skull-strip log.

    Intended to run after all Phase B workers have finished (e.g. as a
    SLURM dependency job or manually).

    Args:
        fomo60k_root: FOMO-60K root directory.
    """
    log_path = fomo60k_root / "_skull_strip_log.json"
    if not log_path.exists():
        print(f"ERROR: Log file not found at {log_path}")
        return

    viz_path = fomo60k_root / "_skull_strip_summary.png"
    print("Generating summary visualization...")
    create_summary_visualization(fomo60k_root, log_path, viz_path)

    # Also print aggregate stats from the log
    n_ok = 0
    n_fail = 0
    ds_counts: dict[str, int] = {}
    with open(str(log_path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry["status"] == "OK":
                n_ok += 1
                ds_counts[entry["dataset"]] = ds_counts.get(entry["dataset"], 0) + 1
            else:
                n_fail += 1

    print()
    print("=" * 60)
    print("Phase B — Aggregate Summary (all workers)")
    print("=" * 60)
    print(f"  Total OK:     {n_ok}")
    print(f"  Total FAILED: {n_fail}")
    for ds in DEFACED_DATASETS:
        print(f"    {ds}: {ds_counts.get(ds, 0)} OK")
    print(f"  Visualization: {viz_path}")


def main() -> None:
    """Entry point for skull-stripping CLI."""
    parser = argparse.ArgumentParser(
        description="Skull-strip defaced FOMO-60K datasets using HD-BET"
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["A", "B", "visualize"],
        help=(
            "Phase A (validation, 9 subjects), "
            "Phase B (batch processing), or "
            "visualize (generate summary plot from log)"
        ),
    )
    parser.add_argument(
        "--fomo60k-root",
        type=Path,
        default=Path("/media/mpascual/Sandisk2TB/research/neuromf/datasets/FOMO60K"),
        help="Path to FOMO-60K root directory",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index (default: 0)",
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="Worker index for multi-GPU parallelism (default: 0)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel workers (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.phase == "A":
        run_phase_a(args.fomo60k_root, args.device)
    elif args.phase == "B":
        run_phase_b(args.fomo60k_root, args.device, args.worker_id, args.num_workers)
    else:
        run_visualize(args.fomo60k_root)


if __name__ == "__main__":
    main()
