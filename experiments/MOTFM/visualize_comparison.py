#!/usr/bin/env python
"""MOTFM vs NeuroMF vs Real comparison grid.

Produces a publication-quality comparison figure showing:
- Rows: subjects × 2 views (axial, coronal)
- Columns: Real | NeuroMF-NFE1 | MOTFM-NFE1 | NeuroMF-NFE10 | MOTFM-NFE10 | ...

CPU-only. Reads directly from HDF5 archives.

Usage:
    python experiments/MOTFM/visualize_comparison.py \
        --generation-dir /path/to/phase_5/generation/ \
        --output-dir /path/to/figures/ \
        --n-subjects 3 --nfe 1 10 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

matplotlib.use("Agg")

# ---- project imports -------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from neuromf.generation.h5_manager import H5Manager  # noqa: E402
from neuromf.utils.visualisation import (  # noqa: E402
    COLORBLIND_PALETTE,
    _apply_style,
    _get_center_slices,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MOTFM vs NeuroMF vs Real comparison grid.",
    )
    parser.add_argument(
        "--generation-dir",
        type=str,
        required=True,
        help="Path to phase_5/generation/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="Number of subjects to show (default: 3).",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="NFE levels to compare (default: 1 10 50).",
    )
    parser.add_argument(
        "--crop-frac",
        type=float,
        default=0.8,
        help="Center crop fraction for 2D slices (default: 0.8).",
    )
    return parser.parse_args()


def _load_volume(h5_path: Path, idx: int) -> np.ndarray:
    """Load a volume from HDF5 as numpy array."""
    vol = H5Manager.read_volume(h5_path, idx)
    return vol.numpy()


def main() -> None:
    """Generate comparison grid figure."""
    args = parse_args()
    _apply_style()

    gen_dir = Path(args.generation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_subjects = args.n_subjects
    nfe_levels = args.nfe
    crop_frac = args.crop_frac

    # Check available data
    real_path = gen_dir / "real_test.h5"
    if not real_path.exists():
        logger.error("Real test volumes not found: %s", real_path)
        sys.exit(1)

    # Build column spec: Real | (NeuroMF-nfe, MOTFM-nfe) for each NFE
    columns = [("Real", None, None)]
    for nfe in nfe_levels:
        neuromf_path = gen_dir / "volumes" / f"nfe_{nfe:03d}.h5"
        motfm_path = gen_dir / "volumes" / "motfm" / f"nfe_{nfe:03d}.h5"

        if neuromf_path.exists():
            columns.append((f"NeuroMF\nNFE={nfe}", neuromf_path, nfe))
        else:
            logger.warning("NeuroMF NFE=%d not found: %s", nfe, neuromf_path)

        if motfm_path.exists():
            columns.append((f"MOTFM\nNFE={nfe}", motfm_path, nfe))
        else:
            logger.warning("MOTFM NFE=%d not found: %s", nfe, motfm_path)

    n_cols = len(columns)
    n_views = 2  # axial, coronal
    n_rows = n_subjects * n_views

    logger.info(
        "Grid: %d subjects × %d views × %d columns",
        n_subjects,
        n_views,
        n_cols,
    )

    # Create figure
    fig_width = 2.0 * n_cols
    fig_height = 2.0 * n_subjects * n_views
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        wspace=0.02,
        hspace=0.05,
    )

    # Subject colors for row grouping
    subject_colors = COLORBLIND_PALETTE[: n_subjects]

    for subj_idx in range(n_subjects):
        for col_idx, (col_label, h5_path, nfe) in enumerate(columns):
            if h5_path is None:
                # Real column
                vol = _load_volume(real_path, subj_idx)
            else:
                vol = _load_volume(h5_path, subj_idx)

            axial, sagittal, coronal = _get_center_slices(vol, crop_frac)

            for view_idx, (slice_2d, view_name) in enumerate(
                [(axial, "Ax"), (coronal, "Cor")]
            ):
                row = subj_idx * n_views + view_idx
                ax = fig.add_subplot(gs[row, col_idx])
                ax.imshow(slice_2d.T, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])

                # Column headers on first row
                if row == 0:
                    ax.set_title(col_label, fontsize=8, fontweight="bold")

                # Row labels on first column
                if col_idx == 0:
                    ax.set_ylabel(
                        f"S{subj_idx + 1} {view_name}",
                        fontsize=7,
                        rotation=90,
                        labelpad=5,
                    )

                # Subject color border
                for spine in ax.spines.values():
                    spine.set_edgecolor(subject_colors[subj_idx % len(subject_colors)])
                    spine.set_linewidth(1.5)

    fig.suptitle(
        "NeuroMF vs MOTFM — Visual Comparison",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    save_figure(fig, output_dir / "comparison_grid")
    logger.info("Comparison grid saved to %s", output_dir / "comparison_grid")


if __name__ == "__main__":
    main()
