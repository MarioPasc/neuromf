#!/usr/bin/env python
"""Convert FOMO-60K NIfTI volumes to MOTFM pickle format.

Uses the SAME train/val/test split as Phase 1 (split_manifest.json, seed=42,
85/10/5%) and the SAME preprocessing pipeline (build_mri_preprocessing_from_config).

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
import json
import logging
import pickle
import sys
import time
from pathlib import Path

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


def _process_split(
    entries: list[dict],
    transform: object,
    split_name: str,
    max_volumes: int | None = None,
) -> list[dict]:
    """Process a split's entries into MOTFM format.

    Args:
        entries: Split manifest entries with "source_path" and "subject_key".
        transform: MONAI preprocessing transform.
        split_name: Split name for logging.
        max_volumes: Optional volume limit.

    Returns:
        List of dicts with "image" (Tensor[1, 192, 192, 192]) and "name" (str).
    """
    if max_volumes is not None:
        entries = entries[:max_volumes]

    n = len(entries)
    logger.info("Processing %s split: %d volumes", split_name, n)

    samples = []
    t0 = time.time()

    for i, entry in enumerate(entries):
        nifti_path = entry["source_path"]
        subject_key = entry.get("subject_key", f"unknown_{i}")

        if not Path(nifti_path).exists():
            logger.warning("Missing NIfTI: %s — skipping", nifti_path)
            continue

        # Apply same preprocessing as Phase 1
        data = transform({"image": nifti_path})
        x = data["image"]  # Tensor[1, H, W, D] from MONAI

        # Clamp to [0, 1]
        x = x.clamp(0.0, 1.0).float()

        samples.append({
            "image": x,  # Tensor[1, 192, 192, 192]
            "name": subject_key,
        })

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "  %s: %d/%d processed (%.1f s, %.2f vol/s)",
                split_name,
                i + 1,
                n,
                elapsed,
                rate,
            )

    logger.info(
        "%s split: %d/%d volumes processed in %.1f s",
        split_name,
        len(samples),
        n,
        time.time() - t0,
    )
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

    # Process train and val splits
    # MOTFM expects keys "train" and "valid"
    dataset = {}

    per_split_limit = None
    if args.max_volumes is not None:
        total_entries = sum(len(s.get("entries", [])) for s in splits.values())
        # Distribute limit proportionally
        per_split_limit = args.max_volumes

    for manifest_key, motfm_key in [("train", "train"), ("val", "valid")]:
        split_data = splits.get(manifest_key, {})
        entries = split_data.get("entries", [])

        if not entries:
            logger.warning("No entries for split '%s' — skipping", manifest_key)
            continue

        samples = _process_split(
            entries=entries,
            transform=transform,
            split_name=manifest_key,
            max_volumes=per_split_limit,
        )
        dataset[motfm_key] = samples

    # Save pickle
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    # Verify
    logger.info("Verifying pickle...")
    with open(output_path, "rb") as f:
        loaded = pickle.load(f)

    for key in ["train", "valid"]:
        if key in loaded:
            n = len(loaded[key])
            if n > 0:
                sample = loaded[key][0]
                shape = sample["image"].shape if hasattr(sample["image"], "shape") else "?"
                logger.info("  %s: %d samples, first shape=%s", key, n, shape)


if __name__ == "__main__":
    main()
