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

import argparse
import gc
import json
import logging
import pickle
import sys
import tempfile
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

    names: list[str] = []
    n_written = 0
    t0 = time.time()

    with h5py.File(str(h5_path), "a") as hf:
        dset = None

        for i, entry in enumerate(entries):
            nifti_path = entry["source_path"]
            subject_key = entry.get("subject_key", f"unknown_{i}")

            if not Path(nifti_path).exists():
                logger.warning("Missing NIfTI: %s — skipping", nifti_path)
                continue

            # Apply same preprocessing as Phase 1
            data = transform({"image": nifti_path})
            x = data["image"].clamp(0.0, 1.0).float().numpy()

            # Create dataset on first successful volume (shape may vary across datasets)
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

            dset[n_written] = x
            names.append(subject_key)
            n_written += 1

            # Explicitly free MONAI transform intermediates
            del data, x
            if (n_written) % 100 == 0:
                gc.collect()

            if (i + 1) % 50 == 0 or i == n - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "  %s: %d/%d processed (%d written, %.1f s, %.2f vol/s)",
                    split_name,
                    i + 1,
                    n,
                    n_written,
                    elapsed,
                    rate,
                )

        # Resize dataset to actual number written (some may have been skipped)
        if dset is not None and n_written < n:
            dset.resize(n_written, axis=0)

    logger.info(
        "%s split: %d/%d volumes written to HDF5 in %.1f s",
        split_name,
        n_written,
        n,
        time.time() - t0,
    )
    gc.collect()
    return names


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
        logger.info("Phase 2 — Loading %s split: %d volumes from HDF5", split_name, n)

        for i in range(n):
            x = torch.from_numpy(dset[i].astype(np.float32))
            name = names[i] if i < len(names) else f"unknown_{i}"
            samples.append({"image": x, "name": name})

            if (i + 1) % 500 == 0:
                logger.info("  %s: %d/%d loaded", split_name, i + 1, n)

    logger.info("  %s: %d samples loaded into memory", split_name, len(samples))
    return samples


def main() -> None:
    """Convert FOMO-60K NIfTI to MOTFM pickle format."""
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

    # ── Phase 2: HDF5 → Pickle (peak memory = all volumes as float32) ──
    logger.info("=" * 60)
    logger.info("PHASE 2: HDF5 → Pickle")
    logger.info("=" * 60)

    dataset: dict[str, list[dict]] = {}
    for motfm_key, names in split_names_map.items():
        dataset[motfm_key] = _build_pickle_from_h5(temp_h5_path, motfm_key, names)

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

    # Clean up temp HDF5
    del dataset
    gc.collect()

    if temp_h5_path.exists():
        temp_h5_path.unlink()
        logger.info("Cleaned up temp HDF5: %s", temp_h5_path)

    # Verify pickle (load just metadata, not full data)
    logger.info("Verifying pickle structure...")
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)

    for key in ["train", "valid"]:
        if key in loaded:
            n = len(loaded[key])
            if n > 0:
                sample = loaded[key][0]
                shape = sample["image"].shape if hasattr(sample["image"], "shape") else "?"
                logger.info("  %s: %d samples, first shape=%s", key, n, shape)

    logger.info("Done.")


if __name__ == "__main__":
    main()
