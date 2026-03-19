#!/usr/bin/env python
"""Verify MOTFM pretrained checkpoint by generating volumes at multiple NFE levels.

Generates a small number of volumes, saves mid-slice PNGs, a summary grid figure,
and logs per-NFE volume statistics. Useful for confirming that the author-provided
checkpoint produces reasonable outputs and for visual comparison across NFE levels.

Usage:
    # Ensure MOTFM is on PYTHONPATH:
    export PYTHONPATH="${REPO}/src/external/MOTFM:${REPO}/experiments/MOTFM:${PYTHONPATH}"

    python experiments/MOTFM/verify_pretrained.py \
        --config configs/motfm_pretrained/config_3D.yaml \
        --checkpoint /path/to/epoch119-valloss0.040524.ckpt \
        --nfe 1 2 5 10 50 \
        --n-samples 4 \
        --output-dir results/motfm_pretrained_verification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        description="Verify MOTFM pretrained checkpoint quality.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to MOTFM YAML config matching the checkpoint architecture.",
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
        default=[1, 2, 5, 10, 50],
        help="NFE levels to generate (default: 1 2 5 10 50).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4,
        help="Number of volumes to generate per NFE (default: 4).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/motfm_pretrained_verification",
        help="Output directory for PNGs and stats.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', 'cpu' (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def save_slice_png(arr: np.ndarray, path: Path, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """Save a 2D slice as a grayscale PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(arr, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0.02, dpi=100)
    plt.close(fig)


def main() -> None:
    """Generate and inspect volumes from pretrained MOTFM checkpoint."""
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    from motfm_wrapper import MOTFMGenerator

    generator = MOTFMGenerator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all slices for the summary grid: nfe_slices[nfe][sample_idx] = (axial, coronal, sagittal)
    nfe_slices: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    stats_lines: list[str] = []

    for nfe in args.nfe:
        logger.info("=" * 60)
        logger.info("Generating NFE=%d (%d samples)", nfe, args.n_samples)
        logger.info("=" * 60)

        volumes = generator.generate(
            n_samples=args.n_samples,
            nfe=nfe,
            batch_size=1,
            seed=args.seed,
        )

        # Volume statistics
        stat_line = (
            f"NFE={nfe:3d}: mean={volumes.mean():.4f}, std={volumes.std():.4f}, "
            f"min={volumes.min():.4f}, max={volumes.max():.4f}, "
            f"pct_in_01={((volumes >= 0) & (volumes <= 1)).float().mean():.1%}"
        )
        logger.info(stat_line)
        stats_lines.append(stat_line)

        nfe_dir = output_dir / f"nfe_{nfe:03d}"
        nfe_dir.mkdir(parents=True, exist_ok=True)
        nfe_slices[nfe] = []

        for s_idx in range(args.n_samples):
            vol = volumes[s_idx].numpy()  # (D, H, W)
            d, h, w = vol.shape

            axial = vol[d // 2, :, :]
            coronal = vol[:, h // 2, :]
            sagittal = vol[:, :, w // 2]

            save_slice_png(axial, nfe_dir / f"sample_{s_idx}_axial.png")
            save_slice_png(coronal, nfe_dir / f"sample_{s_idx}_coronal.png")
            save_slice_png(sagittal, nfe_dir / f"sample_{s_idx}_sagittal.png")

            nfe_slices[nfe].append((axial, coronal, sagittal))

    # Save stats summary
    stats_path = output_dir / "volume_stats.txt"
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines) + "\n")
    logger.info("Stats saved to %s", stats_path)

    # Summary grid figure: rows = samples, columns = NFE levels (axial mid-slice)
    n_nfe = len(args.nfe)
    n_samples = args.n_samples

    fig, axes = plt.subplots(
        n_samples,
        n_nfe,
        figsize=(3 * n_nfe, 3 * n_samples),
        squeeze=False,
    )
    fig.suptitle("MOTFM Pretrained Verification: Axial Mid-Slices", fontsize=14, y=1.01)

    for col, nfe in enumerate(args.nfe):
        axes[0, col].set_title(f"NFE={nfe}", fontsize=11)
        for row in range(n_samples):
            axial_slice = nfe_slices[nfe][row][0]
            axes[row, col].imshow(axial_slice, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(f"Sample {row}", fontsize=9)

    fig.tight_layout()
    grid_path = output_dir / "summary_grid.png"
    fig.savefig(str(grid_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary grid saved to %s", grid_path)

    logger.info("Verification complete. Output: %s", output_dir)


if __name__ == "__main__":
    main()
