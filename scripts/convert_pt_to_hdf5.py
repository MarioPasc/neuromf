"""One-time migration script: convert individual .pt latent files to HDF5 shards.

Groups .pt files by dataset name (prefix before first subject id), creates one
.h5 shard per dataset, writes each latent, and verifies read-back integrity.

Usage:
    ~/.conda/envs/neuromf/bin/python scripts/convert_pt_to_hdf5.py \
        --input-dir /path/to/latents \
        --output-dir /path/to/latents   # can be same dir
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from rich.logging import RichHandler
from tqdm import tqdm

from neuromf.data.latent_hdf5 import create_shard, read_sample, write_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def _extract_dataset_name(pt_filename: str) -> str:
    """Extract dataset name from .pt filename.

    Convention: ``{dataset}_{participant_id}_{session_id}.pt``
    Dataset names contain exactly one underscore separator before the
    participant prefix (``sub_``). We split on ``_sub_`` to get the
    dataset part.

    Args:
        pt_filename: Filename like ``PT005_IXI_sub_002_ses_1.pt``.

    Returns:
        Dataset name like ``PT005_IXI``.
    """
    stem = pt_filename.replace(".pt", "")
    # Split on first occurrence of "_sub_"
    idx = stem.find("_sub_")
    if idx == -1:
        # Fallback: use everything before the last two underscore-separated parts
        parts = stem.rsplit("_", 2)
        return parts[0] if len(parts) > 1 else stem
    return stem[:idx]


def main() -> None:
    """Convert .pt latent files to per-dataset HDF5 shards."""
    parser = argparse.ArgumentParser(description="Convert .pt latent files to HDF5 shards")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with .pt files")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for .h5 shards (default: same as input-dir)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify read-back integrity (default: True)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    # Discover .pt files
    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        logger.error("No .pt files found in %s", input_dir)
        return

    logger.info("Found %d .pt files in %s", len(pt_files), input_dir)

    # Group by dataset name
    groups: dict[str, list[Path]] = defaultdict(list)
    for pt_file in pt_files:
        ds_name = _extract_dataset_name(pt_file.name)
        groups[ds_name].append(pt_file)

    logger.info("Datasets: %s", {ds: len(files) for ds, files in groups.items()})

    # Create shards and write
    output_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    total_verified = 0

    for ds_name, files in groups.items():
        shard_path = output_dir / f"{ds_name}.h5"
        n_vols = len(files)

        # Detect latent shape from first file
        first_data = torch.load(files[0], map_location="cpu", weights_only=True)
        z0 = first_data["z"] if isinstance(first_data, dict) else first_data
        latent_shape = tuple(z0.shape)

        logger.info(
            "Creating shard %s: %d volumes, shape=%s", shard_path.name, n_vols, latent_shape
        )
        h5f = create_shard(shard_path, ds_name, n_vols, latent_shape=latent_shape)

        for idx, pt_file in enumerate(tqdm(files, desc=ds_name, unit="vol")):
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            z = data["z"] if isinstance(data, dict) else data
            metadata = data.get("metadata", {})

            write_sample(
                h5f,
                idx,
                z,
                subject_id=str(metadata.get("subject_id", "")),
                session_id=str(metadata.get("session_id", "")),
                source_path=str(metadata.get("source_path", "")),
            )
            total_written += 1

        h5f.attrs["n_written"] = n_vols
        h5f.close()

        # Verify
        if args.verify:
            import h5py

            with h5py.File(str(shard_path), "r") as rf:
                for idx, pt_file in enumerate(files):
                    data = torch.load(pt_file, map_location="cpu", weights_only=True)
                    z_orig = data["z"] if isinstance(data, dict) else data
                    z_read, _ = read_sample(rf, idx)
                    if not torch.allclose(z_orig.float(), z_read.float(), atol=1e-6):
                        logger.error(
                            "VERIFICATION FAILED: %s[%d] (%s)",
                            shard_path.name,
                            idx,
                            pt_file.name,
                        )
                    else:
                        total_verified += 1

    # Copy latent_stats.json if it exists
    stats_src = input_dir / "latent_stats.json"
    if stats_src.exists() and output_dir != input_dir:
        shutil.copy2(stats_src, output_dir / "latent_stats.json")
        logger.info("Copied latent_stats.json to %s", output_dir)

    logger.info(
        "Migration complete: %d written, %d verified across %d shards",
        total_written,
        total_verified,
        len(groups),
    )


if __name__ == "__main__":
    main()
