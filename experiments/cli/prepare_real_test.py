"""CLI script to prepare real test set volumes from original NIfTI files.

Loads the split manifest (or generates it) to identify test-split subjects,
reads the original preprocessed NIfTI volumes, and stores them in an HDF5
archive. No VAE encode/decode round-trip — the reference set is the actual
MRI data, not a VAE reconstruction.

This ensures that pixel-space metrics (FID, MS-SSIM, PSNR) compare
generated images against the true data distribution, not against
VAE-reconstructed approximations.

Usage:
    python experiments/cli/prepare_real_test.py \
        --config configs/generate.yaml \
        --output-dir /path/to/phase_5/generation/

    # With Picasso overlay:
    python experiments/cli/prepare_real_test.py \
        --config configs/picasso/generate.yaml \
        --configs-dir configs/picasso \
        --output-dir /path/to/phase_5/generation/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf
from rich.logging import RichHandler

from neuromf.data.latent_dataset import export_split_manifest
from neuromf.data.mri_preprocessing import build_mri_preprocessing_from_config
from neuromf.generation.h5_manager import H5Manager, H5VolumeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare real test set volumes from original NIfTI files."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Config YAML paths.",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for real test volumes.",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Limit number of test volumes (for debugging).",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> OmegaConf:
    """Load and merge config layers."""
    from neuromf.config import load_merged_config

    config_paths = [Path(p) for p in args.config]
    configs_dir = Path(args.configs_dir) if args.configs_dir else None

    return load_merged_config(
        config_paths,
        configs_dir=configs_dir,
        include_generate=True,
        require_base=False,
    )


def _load_or_create_manifest(
    latent_dir: Path,
    split_ratios: list[float],
    split_seed: int,
) -> dict:
    """Load existing split manifest or create one.

    Args:
        latent_dir: Directory containing HDF5 shard files.
        split_ratios: Fractions for N-way split.
        split_seed: Random seed for deterministic splitting.

    Returns:
        The split manifest dict.
    """
    manifest_path = latent_dir / "split_manifest.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Verify the manifest was generated with the same config
        cfg = manifest.get("config", {})
        if cfg.get("split_ratios") == split_ratios and cfg.get("split_seed") == split_seed:
            logger.info("Loaded existing split manifest: %s", manifest_path)
            return manifest
        logger.warning(
            "Manifest config mismatch (ratios=%s seed=%s vs expected %s/%d). Regenerating.",
            cfg.get("split_ratios"),
            cfg.get("split_seed"),
            split_ratios,
            split_seed,
        )

    logger.info("Generating split manifest...")
    return export_split_manifest(
        latent_dir=latent_dir,
        output_path=manifest_path,
        split_ratios=split_ratios,
        split_seed=split_seed,
    )


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = _load_config(args)

    latent_dir = Path(config.paths.latents_dir)
    raw_output = args.output_dir or str(config.paths.get("generation_dir", ""))
    if not raw_output:
        logger.error("No output directory specified (use --output-dir or set paths.generation_dir)")
        sys.exit(1)
    output_dir = Path(raw_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create split manifest
    split_ratios = list(config.training.get("split_ratios", [0.85, 0.10, 0.05]))
    split_seed = int(config.training.get("split_seed", 42))
    manifest = _load_or_create_manifest(latent_dir, split_ratios, split_seed)

    # Get test split entries
    test_split = manifest["splits"].get("test", {})
    test_entries = test_split.get("entries", [])
    if not test_entries:
        logger.error("No test entries found in split manifest.")
        sys.exit(1)

    n_test = len(test_entries)
    if args.max_volumes:
        n_test = min(n_test, args.max_volumes)
        test_entries = test_entries[:n_test]

    logger.info(
        "Test set: %d volumes (%d subjects)",
        n_test,
        test_split.get("n_subjects", "?"),
    )

    # Verify source files exist
    missing = [e for e in test_entries if not Path(e["source_path"]).exists()]
    if missing:
        logger.error(
            "%d of %d source NIfTI files not found. First missing: %s",
            len(missing),
            n_test,
            missing[0]["source_path"],
        )
        sys.exit(1)
    logger.info("All %d source NIfTI files verified.", n_test)

    # Build preprocessing transform (same pipeline used for VAE encoding)
    transform = build_mri_preprocessing_from_config(config)

    # Check if archive already complete
    vol_path = output_dir / "real_test.h5"
    if vol_path.exists() and H5Manager.is_complete(vol_path):
        logger.info("Real test volumes already complete: %s", vol_path)
        logger.info("Real test preparation complete.")
        return

    if vol_path.exists():
        logger.warning("Incomplete archive found, recreating: %s", vol_path)
        vol_path.unlink()

    # Infer volume shape from config
    volume_shape = tuple(config.data.target_shape)
    vol_config = H5VolumeConfig(n_samples=n_test, volume_shape=volume_shape)
    vol_meta = {
        "source": "original_nifti",
        "split": "test",
        "split_seed": split_seed,
        "split_ratios": split_ratios,
        "n_samples": n_test,
        "volume_shape": list(volume_shape),
        "note": "Preprocessed original NIfTI volumes (no VAE round-trip).",
    }
    vol_h5 = H5Manager.create_volume_archive(vol_path, vol_config, vol_meta)

    try:
        for i, entry in enumerate(test_entries):
            nifti_path = entry["source_path"]

            # Apply the same preprocessing pipeline used for VAE encoding
            data = transform({"image": nifti_path})
            x = data["image"]  # (1, H, W, D)

            # Squeeze channel dim and convert to (D, H, W) for HDF5 storage
            vol = x.squeeze(0)  # (H, W, D)
            # MONAI loads as (C, H, W, D); after squeeze channel we have (H, W, D)
            # HDF5 volumes dataset expects (D, H, W) — already correct if
            # MONAI channel-first convention was applied

            # Clamp to [0, 1] — ScaleIntensityRangePercentilesd with clip=False
            # may produce values slightly outside this range
            vol = vol.clamp(0.0, 1.0)

            H5Manager.write_volume_batch(
                vol_h5,
                i,
                vol.unsqueeze(0),  # (1, D, H, W) — batch dim for write_volume_batch
                [0.0],  # timing_ms placeholder
            )

            if (i + 1) % 10 == 0 or i == n_test - 1:
                logger.info(
                    "  Preprocessed %d / %d test volumes [%s]",
                    i + 1,
                    n_test,
                    entry["subject_key"],
                )
    finally:
        vol_h5.close()

    logger.info("Real test volumes saved: %s", vol_path)
    logger.info("Real test preparation complete (no VAE round-trip).")


if __name__ == "__main__":
    main()
