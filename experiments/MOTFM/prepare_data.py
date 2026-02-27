#!/usr/bin/env python
"""Convert FOMO-60K NIfTI volumes to MOTFM pickle format.

Uses the SAME train/val/test split as Phase 1 (split_manifest.json, seed=42,
85/10/5%) and the SAME preprocessing pipeline (build_mri_preprocessing_from_config).

Two-phase processing to avoid OOM on large datasets:
  Phase 1: Process NIfTI volumes one-by-one → write to temporary HDF5 (constant memory)
  Phase 2: Load from HDF5 → build pickle (peak memory = all volumes in float32)

Output pickle format (expected by MOTFM's FlowMatchingDataModule):
    {
        "train": [{"image": Tensor[1, 192, 192, 192], "name": str}, ...],
        "valid": [{"image": Tensor[1, 192, 192, 192], "name": str}, ...],
    }

Usage:
    python experiments/MOTFM/prepare_data.py \
        --config configs/picasso/generate.yaml \
        --configs-dir configs/picasso \
        --output-path /path/to/motfm/data/fomo60k_3d.pkl \
        --split-manifest /path/to/latents/split_manifest.json
"""

from __future__ import annotations

SCRIPT_VERSION = "2.0-h5backed"

import argparse
import gc
import json
import logging
import os
import pickle
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Memory diagnostics ──────────────────────────────────────────────────

def _mem_rss_gb() -> float:
    """Return current RSS in GB (Linux /proc/self/status)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB → GB
    except OSError:
        pass
    # Fallback: resource module (ru_maxrss is in KB on Linux)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _mem_available_gb() -> float:
    """Return available system memory in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024  # kB → GB
    except OSError:
        pass
    return -1.0


def _log_memory(label: str) -> None:
    """Log current RSS and available memory."""
    rss = _mem_rss_gb()
    avail = _mem_available_gb()
    logger.info(
        "[MEM] %-30s RSS=%.1f GB  Available=%.1f GB",
        label,
        rss,
        avail,
    )


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert FOMO-60K NIfTI to MOTFM pickle format.",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Config YAML paths (merged left-to-right).",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output .pkl path for MOTFM dataset.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        required=True,
        help="Path to split_manifest.json from Phase 1.",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Limit total volumes (for testing).",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> OmegaConf:
    """Load and merge config layers."""
    config_paths = [Path(p) for p in args.config]
    configs_dir = Path(args.configs_dir) if args.configs_dir else config_paths[0].parent

    base_path = configs_dir / "base.yaml"
    project_root = Path(__file__).resolve().parent.parent.parent
    main_gen_path = project_root / "configs" / "generate.yaml"
    main_train_path = project_root / "configs" / "train_meanflow.yaml"

    layers = []
    if base_path.exists():
        layers.append(OmegaConf.load(base_path))
    if main_train_path.exists():
        layers.append(OmegaConf.load(main_train_path))
    if main_gen_path.exists() and main_gen_path.resolve() not in [
        p.resolve() for p in config_paths
    ]:
        layers.append(OmegaConf.load(main_gen_path))
    for cp in config_paths:
        layers.append(OmegaConf.load(cp))

    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)
    return config


# ── Phase 1: NIfTI → HDF5 (constant memory) ─────────────────────────────

def _process_split_to_h5(
    entries: list[dict],
    transform: object,
    h5_path: Path,
    split_name: str,
    max_volumes: int | None = None,
) -> list[str]:
    """Process a split's NIfTI volumes and write to HDF5 (constant memory).

    Args:
        entries: Split manifest entries with "source_path" and "subject_key".
        transform: MONAI preprocessing transform.
        h5_path: Path to temporary HDF5 file.
        split_name: Split name for logging.
        max_volumes: Optional volume limit.

    Returns:
        List of subject names (parallel to HDF5 dataset rows).
    """
    if max_volumes is not None:
        entries = entries[:max_volumes]

    n = len(entries)
    logger.info("Phase 1 — Processing %s split: %d volumes → %s", split_name, n, h5_path)
    _log_memory(f"phase1_{split_name}_start")

    names: list[str] = []
    n_written = 0
    n_skipped = 0
    t0 = time.time()

    with h5py.File(str(h5_path), "a") as hf:
        dset = None

        for i, entry in enumerate(entries):
            nifti_path = entry["source_path"]
            subject_key = entry.get("subject_key", f"unknown_{i}")

            if not Path(nifti_path).exists():
                logger.warning("Missing NIfTI: %s — skipping", nifti_path)
                n_skipped += 1
                continue

            # Apply same preprocessing as Phase 1
            data = transform({"image": nifti_path})
            x = data["image"].clamp(0.0, 1.0).float().numpy()

            # Explicitly free MONAI transform intermediates BEFORE HDF5 write
            del data
            gc.collect()

            # Create dataset on first successful volume
            if dset is None:
                shape = x.shape  # (1, D, H, W)
                dset = hf.create_dataset(
                    split_name,
                    shape=(n, *shape),
                    maxshape=(n, *shape),
                    dtype="float32",
                    chunks=(1, *shape),
                    compression="lzf",
                )
                logger.info(
                    "Created HDF5 dataset '%s': shape=%s, chunks=%s",
                    split_name,
                    (n, *shape),
                    (1, *shape),
                )

            dset[n_written] = x
            names.append(subject_key)
            n_written += 1

            del x

            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "  %s: %d/%d processed (%d written, %d skipped, "
                    "%.1f s, %.2f vol/s)",
                    split_name,
                    i + 1,
                    n,
                    n_written,
                    n_skipped,
                    elapsed,
                    rate,
                )
                _log_memory(f"phase1_{split_name}_{n_written}")

        # Resize dataset to actual number written (some may have been skipped)
        if dset is not None and n_written < n:
            dset.resize(n_written, axis=0)

    elapsed = time.time() - t0
    logger.info(
        "%s split: %d/%d written (%d skipped) to HDF5 in %.1f s",
        split_name,
        n_written,
        n,
        n_skipped,
        elapsed,
    )
    _log_memory(f"phase1_{split_name}_done")
    gc.collect()
    return names


# ── Phase 2: HDF5 → Pickle ──────────────────────────────────────────────

def _build_pickle_from_h5(
    h5_path: Path,
    split_name: str,
    names: list[str],
) -> list[dict]:
    """Load volumes from HDF5 and build MOTFM-format sample list.

    Args:
        h5_path: Path to temporary HDF5 file.
        split_name: Dataset name in HDF5 file.
        names: Parallel list of subject names.

    Returns:
        List of dicts with "image" (Tensor) and "name" (str).
    """
    samples: list[dict] = []
    with h5py.File(str(h5_path), "r") as hf:
        dset = hf[split_name]
        n = dset.shape[0]
        vol_mb = np.prod(dset.shape[1:]) * 4 / 1024 / 1024
        total_gb = n * vol_mb / 1024
        logger.info(
            "Phase 2 — Loading %s: %d volumes (%.1f MB each, %.1f GB total)",
            split_name,
            n,
            vol_mb,
            total_gb,
        )
        _log_memory(f"phase2_{split_name}_start")

        for i in range(n):
            x = torch.from_numpy(np.array(dset[i], dtype=np.float32))
            name = names[i] if i < len(names) else f"unknown_{i}"
            samples.append({"image": x, "name": name})

            if (i + 1) % 500 == 0 or i == n - 1:
                logger.info("  %s: %d/%d loaded", split_name, i + 1, n)
                _log_memory(f"phase2_{split_name}_{i+1}")

    logger.info("  %s: %d samples in memory", split_name, len(samples))
    _log_memory(f"phase2_{split_name}_done")
    return samples


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Convert FOMO-60K NIfTI to MOTFM pickle format."""
    logger.info("prepare_data.py version: %s", SCRIPT_VERSION)
    logger.info("Python: %s", sys.version)
    logger.info("PID: %d", os.getpid())
    _log_memory("startup")

    args = parse_args()
    config = _load_config(args)

    # Load split manifest
    manifest_path = Path(args.split_manifest)
    if not manifest_path.exists():
        logger.error("Split manifest not found: %s", manifest_path)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    splits = manifest.get("splits", {})
    logger.info(
        "Split manifest: %s",
        {k: v.get("n_scans", len(v.get("entries", []))) for k, v in splits.items()},
    )

    # Build preprocessing transform (same as Phase 1)
    transform = build_mri_preprocessing_from_config(config)
    _log_memory("after_transform_build")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_split_limit = args.max_volumes

    # Use temp HDF5 in same directory as output (avoid cross-filesystem issues)
    temp_h5_path = output_path.parent / f".{output_path.stem}_temp.h5"

    # Clean up any previous temp file
    if temp_h5_path.exists():
        temp_h5_path.unlink()
        logger.info("Removed stale temp file: %s", temp_h5_path)

    split_names_map: dict[str, list[str]] = {}

    # ── Phase 1: Process NIfTI → HDF5 (constant memory) ──
    logger.info("=" * 60)
    logger.info("PHASE 1: NIfTI → HDF5 (disk-backed, constant memory)")
    logger.info("=" * 60)

    for manifest_key, motfm_key in [("train", "train"), ("val", "valid")]:
        split_data = splits.get(manifest_key, {})
        entries = split_data.get("entries", [])

        if not entries:
            logger.warning("No entries for split '%s' — skipping", manifest_key)
            continue

        names = _process_split_to_h5(
            entries=entries,
            transform=transform,
            h5_path=temp_h5_path,
            split_name=motfm_key,
            max_volumes=per_split_limit,
        )
        split_names_map[motfm_key] = names

    # Free transform to reclaim memory before Phase 2
    del transform
    gc.collect()

    # Log temp HDF5 size
    if temp_h5_path.exists():
        h5_gb = temp_h5_path.stat().st_size / 1e9
        logger.info("Temp HDF5 size: %.2f GB (%s)", h5_gb, temp_h5_path)

    _log_memory("between_phases")

    # ── Phase 2: HDF5 → Pickle (peak memory = all volumes as float32) ──
    logger.info("=" * 60)
    logger.info("PHASE 2: HDF5 → Pickle")
    logger.info("=" * 60)

    dataset: dict[str, list[dict]] = {}
    for motfm_key, names in split_names_map.items():
        dataset[motfm_key] = _build_pickle_from_h5(temp_h5_path, motfm_key, names)

    _log_memory("before_pickle_dump")

    # Save pickle
    logger.info("Saving MOTFM dataset to: %s", output_path)
    t0 = time.time()
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = output_path.stat().st_size / 1e9
    logger.info(
        "Dataset saved: %.2f GB, %d train + %d val volumes in %.1f s",
        size_gb,
        len(dataset.get("train", [])),
        len(dataset.get("valid", [])),
        time.time() - t0,
    )
    _log_memory("after_pickle_dump")

    # Free dataset before cleanup
    del dataset
    gc.collect()
    _log_memory("after_free_dataset")

    # Clean up temp HDF5
    if temp_h5_path.exists():
        temp_h5_path.unlink()
        logger.info("Cleaned up temp HDF5: %s", temp_h5_path)

    # Verify pickle (lightweight — just check keys and first sample shape)
    logger.info("Verifying pickle structure (header only)...")
    _log_memory("before_verify")
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)

    for key in ["train", "valid"]:
        if key in loaded:
            n = len(loaded[key])
            if n > 0:
                sample = loaded[key][0]
                shape = sample["image"].shape if hasattr(sample["image"], "shape") else "?"
                dtype = sample["image"].dtype if hasattr(sample["image"], "dtype") else "?"
                logger.info("  %s: %d samples, shape=%s, dtype=%s", key, n, shape, dtype)

    del loaded
    gc.collect()
    _log_memory("final")
    logger.info("Done. version=%s", SCRIPT_VERSION)


if __name__ == "__main__":
    main()
