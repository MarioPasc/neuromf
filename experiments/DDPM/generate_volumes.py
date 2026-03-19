#!/usr/bin/env python
"""Generate DDPM volumes via DDIM sampling and save as HDF5.

Same output format as MOTFM's generate_volumes.py for downstream evaluation.

Usage:
    export PYTHONPATH="${REPO}/src/external/MOTFM:${REPO}/experiments/DDPM:${REPO}/experiments/MOTFM:${PYTHONPATH}"
    python experiments/DDPM/generate_volumes.py \
        --config configs/picasso/ddpm/fomo60k_unconditional_3d.yaml \
        --checkpoint /path/to/best.ckpt \
        --nfe 1 10 50 \
        --n-samples 500 \
        --output-dir /path/to/volumes/ddpm/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _ensure_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for sub in ["src/external/MOTFM", "experiments/DDPM", "experiments/MOTFM"]:
        p = str(repo_root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DDPM volumes via DDIM.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 10, 50])
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--volume-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_paths()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    logger.info("Device: %s", device)

    from ddpm_wrapper import DDPMGenerator

    # Reuse MOTFM's write_volumes_h5 for identical output format
    from generate_volumes import write_volumes_h5

    generator = DDPMGenerator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        volume_size=args.volume_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nfe in args.nfe:
        output_path = output_dir / f"nfe_{nfe:03d}.h5"

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
            "NFE=%d: %d volumes in %.1f s (%.2f s/vol)",
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
