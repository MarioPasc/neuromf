#!/usr/bin/env python
"""Unified generation CLI for pixel-space competitor models.

Uses the competitor registry to instantiate the correct generator by name.
Replaces per-model ``generate_volumes.py`` scripts with a single entry point.

Usage::

    # List available models:
    python experiments/cli/generate_competitor.py --list-models

    # Generate MOTFM volumes:
    python experiments/cli/generate_competitor.py \\
        --model motfm \\
        --config configs/picasso/motfm/fomo60k_unconditional_3d.yaml \\
        --checkpoint /path/to/last.ckpt \\
        --nfe 1 10 50 \\
        --n-samples 500 \\
        --output-dir /path/to/volumes/

    # Generate DDPM volumes:
    python experiments/cli/generate_competitor.py \\
        --model ddpm \\
        --config configs/picasso/ddpm/fomo60k_unconditional_3d.yaml \\
        --checkpoint /path/to/best.ckpt \\
        --nfe 1 10 50 \\
        --n-samples 500 \\
        --output-dir /path/to/volumes/
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import torch
from rich.logging import RichHandler

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from neuromf.competitors import CompetitorRegistry, write_volumes_h5

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate volumes from a registered competitor model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help=f"Model name from registry: {CompetitorRegistry.list_names()}",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print registered models and exit.",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model-specific YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint.",
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
        default=500,
        help="Number of volumes per NFE level (default: 500).",
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
        help="Output directory for HDF5 volume archives.",
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
        help="Override spatial size per axis (e.g. 192 for FOMO-60K).",
    )
    return parser.parse_args()


def _resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main() -> None:
    """Generate volumes using a registered competitor model."""
    args = parse_args()

    # List mode
    if args.list_models:
        names = CompetitorRegistry.list_names()
        print(f"Registered competitor models ({len(names)}):")
        for name in names:
            cls = CompetitorRegistry.get(name)
            print(f"  {name:15s}  ->  {cls.__name__} ({cls.model_name})")
        return

    # Validate required args
    if not args.model:
        logger.error("--model is required (use --list-models to see options)")
        sys.exit(1)
    if not args.config or not args.checkpoint:
        logger.error("--config and --checkpoint are required")
        sys.exit(1)

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    # Instantiate generator from registry
    GeneratorCls = CompetitorRegistry.get(args.model)
    logger.info(
        "Using %s (%s)", GeneratorCls.__name__, GeneratorCls.model_name,
    )

    generator = GeneratorCls(
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
                        nfe, n_written, n_expected,
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
            "NFE=%d: %d volumes in %.1f s (%.2f s/vol)",
            nfe, volumes.shape[0], dt, dt / max(volumes.shape[0], 1),
        )

        write_volumes_h5(
            output_path=output_path,
            volumes=volumes,
            nfe=nfe,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            model_name=generator.model_name,
        )

    logger.info("All NFE levels complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
