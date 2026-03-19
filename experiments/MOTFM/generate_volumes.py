#!/usr/bin/env python
"""Generate MOTFM volumes and save as HDF5 (same format as NeuroMF).

Usage:
    python experiments/MOTFM/generate_volumes.py \
        --config configs/picasso/motfm/fomo60k_unconditional_3d.yaml \
        --checkpoint /path/to/last.ckpt \
        --nfe 1 10 50 \
        --n-samples 2000 \
        --output-dir /path/to/volumes/motfm/ \
        --seed 42
"""

from __future__ import annotations

import argparse
import datetime
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate MOTFM volumes and save as HDF5.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to MOTFM YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to MOTFM Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="NFE levels to generate (default: 1 10 50).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of volumes to generate per NFE (default: 2000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generation batch size (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for HDF5 files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--volume-size",
        type=int,
        default=None,
        help="Spatial size per axis (overrides config). "
        "96 for pretrained MSD Brain, 192 for FOMO-60K.",
    )
    return parser.parse_args()


def write_volumes_h5(
    output_path: Path,
    volumes: torch.Tensor,
    nfe: int,
    checkpoint_path: str,
    seed: int,
) -> None:
    """Write generated volumes to HDF5 in the same format as NeuroMF.

    Args:
        output_path: Path to write the .h5 file.
        volumes: Tensor of shape (N, 192, 192, 192) float32.
        nfe: Number of function evaluations used.
        checkpoint_path: Source checkpoint path for metadata.
        seed: Random seed used for generation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = volumes.shape[0]
    vol_shape = volumes.shape[1:]  # (D, H, W)

    with h5py.File(str(output_path), "w") as f:
        f.create_dataset(
            "volumes",
            shape=(n_samples, *vol_shape),
            dtype=np.float32,
            chunks=(1, *vol_shape),
            compression="gzip",
            compression_opts=4,
        )

        f.create_dataset(
            "decode_timing_ms",
            shape=(n_samples,),
            dtype=np.float32,
        )

        # Write data
        vols_np = volumes.numpy()
        f["volumes"][:] = vols_np
        f["decode_timing_ms"][:] = 0.0  # No separate decode step for MOTFM

        # Metadata (matches NeuroMF convention)
        f.attrs["n_samples"] = n_samples
        f.attrs["n_written"] = n_samples
        f.attrs["nfe"] = nfe
        f.attrs["model"] = "motfm"
        f.attrs["checkpoint_path"] = str(checkpoint_path)
        f.attrs["seed"] = seed
        f.attrs["timestamp"] = datetime.datetime.now().isoformat()

    size_gb = output_path.stat().st_size / 1e9
    logger.info("Saved %s: %d volumes, %.2f GB", output_path.name, n_samples, size_gb)


def main() -> None:
    """Generate MOTFM volumes at multiple NFE levels."""
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load MOTFM generator
    from motfm_wrapper import MOTFMGenerator

    generator = MOTFMGenerator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        volume_size=args.volume_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nfe in args.nfe:
        output_path = output_dir / f"nfe_{nfe:03d}.h5"

        # Skip if already complete
        if output_path.exists():
            try:
                with h5py.File(str(output_path), "r") as f:
                    n_written = int(f.attrs.get("n_written", 0))
                    n_expected = f["volumes"].shape[0]
                if n_written >= n_expected and n_expected == args.n_samples:
                    logger.info(
                        "Skipping NFE=%d — already complete (%d/%d)",
                        nfe,
                        n_written,
                        n_expected,
                    )
                    continue
            except Exception:
                pass

        logger.info("=" * 60)
        logger.info("Generating NFE=%d (%d samples)", nfe, args.n_samples)
        logger.info("=" * 60)

        t0 = time.time()
        volumes = generator.generate(
            n_samples=args.n_samples,
            nfe=nfe,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        dt = time.time() - t0

        logger.info(
            "NFE=%d: generated %d volumes in %.1f s (%.2f s/vol)",
            nfe,
            volumes.shape[0],
            dt,
            dt / volumes.shape[0],
        )

        write_volumes_h5(
            output_path=output_path,
            volumes=volumes,
            nfe=nfe,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
        )

    logger.info("All NFE levels complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
