"""CLI script for quick visual sanity-checks of Phase 5 generation outputs.

Produces two publication-quality figures from the HDF5 archives:

1. **generation_comparison_grid** — Multi-panel MRI comparison:
   4 rows (Real, NFE=1, NFE=10, NFE=50) × (N subjects × 3 views).
   Colored spine borders group per-subject columns.

2. **latent_psd_by_nfe** — 2×2 radially-averaged power spectra:
   one panel per latent channel, lines colored by NFE level,
   dashed gray line for pure-noise reference.

CPU-only. No VAE or model needed — reads HDF5 archives directly.

Usage:
    python experiments/cli/visualize_generation.py \
        --generation-dir /path/to/phase_5/generation/ \
        --output-dir /path/to/phase_5/generation/figures/ \
        --n-subjects 3 \
        --nfe 1 10 50
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
from neuromf.utils.sample_plots import _radial_average_3d  # noqa: E402
from neuromf.utils.visualisation import (  # noqa: E402
    COLORBLIND_PALETTE,
    _apply_style,
    _get_center_slices,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# NFE label colormap (viridis endpoints + midpoint)
_NFE_CMAP = plt.cm.viridis


# ---------------------------------------------------------------------------
# Figure 1: Generation Comparison Grid
# ---------------------------------------------------------------------------


def _load_volumes_for_grid(
    generation_dir: Path,
    nfe_levels: list[int],
    n_subjects: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Load real + generated volumes for the comparison grid.

    Args:
        generation_dir: Root generation directory containing ``real_test.h5``
            and ``volumes/nfe_*.h5``.
        nfe_levels: NFE levels to include (e.g. ``[1, 10, 50]``).
        n_subjects: Number of subjects per row.

    Returns:
        Tuple of (row_volumes, subject_ids) where row_volumes maps
        row labels to arrays of shape ``(n_subjects, D, H, W)``
        and subject_ids is a list of subject identifiers.
    """
    row_volumes: dict[str, list[np.ndarray]] = {}
    subject_ids: list[str] = []

    # Real volumes
    real_path = generation_dir / "real_test.h5"
    if not real_path.exists():
        logger.warning("real_test.h5 not found at %s — skipping Real row.", real_path)
    else:
        real_vols: list[np.ndarray] = []
        for i in range(n_subjects):
            try:
                vol = H5Manager.read_volume(real_path, i).numpy()
                real_vols.append(vol)
                subject_ids.append(f"Subj {i}")
            except Exception as e:
                logger.warning("Cannot read real volume %d: %s", i, e)
        if real_vols:
            row_volumes["Real"] = real_vols

    # If no real volumes loaded, create subject IDs from generated
    if not subject_ids:
        subject_ids = [f"Subj {i}" for i in range(n_subjects)]

    # Generated volumes at each NFE level
    for nfe in nfe_levels:
        vol_path = generation_dir / "volumes" / f"nfe_{nfe:03d}.h5"
        if not vol_path.exists():
            logger.warning("Volume archive not found: %s — skipping NFE=%d row.", vol_path, nfe)
            continue
        gen_vols: list[np.ndarray] = []
        for i in range(n_subjects):
            try:
                vol = H5Manager.read_volume(vol_path, i).numpy()
                gen_vols.append(vol)
            except Exception as e:
                logger.warning("Cannot read NFE=%d volume %d: %s", nfe, i, e)
        if gen_vols:
            row_volumes[f"NFE={nfe}"] = gen_vols

    return row_volumes, subject_ids


def plot_generation_comparison_grid(
    generation_dir: Path,
    output_dir: Path,
    nfe_levels: list[int],
    n_subjects: int = 3,
    crop_frac: float = 0.8,
) -> None:
    """Plot multi-panel MRI comparison grid.

    Layout: N_rows × (N_subjects × 3) columns. Rows = Real + one per NFE level.
    Column groups = N subjects × (Axial, Sagittal, Coronal).

    Args:
        generation_dir: Path to ``phase_5/generation/``.
        output_dir: Output directory for figures.
        nfe_levels: NFE levels to show (e.g. ``[1, 10, 50]``).
        n_subjects: Number of subjects per row.
        crop_frac: Center crop fraction for 2D slices.
    """
    _apply_style()

    row_volumes, subject_ids = _load_volumes_for_grid(generation_dir, nfe_levels, n_subjects)

    if not row_volumes:
        logger.error("No volumes loaded — cannot create comparison grid.")
        return

    # Build row order: Real first (if present), then NFE levels in order
    row_order: list[str] = []
    if "Real" in row_volumes:
        row_order.append("Real")
    for nfe in nfe_levels:
        key = f"NFE={nfe}"
        if key in row_volumes:
            row_order.append(key)

    actual_n_subjects = min(
        n_subjects,
        min(len(vols) for vols in row_volumes.values()),
    )
    n_rows = len(row_order)
    n_cols = actual_n_subjects * 3  # 3 views per subject

    # Compute shared vmin/vmax (1st/99th percentile across all volumes)
    all_vals: list[float] = []
    for row_key in row_order:
        for vol in row_volumes[row_key][:actual_n_subjects]:
            # HDF5 stores (D, H, W), transpose to (H, W, D) for _get_center_slices
            vol_hwp = vol.transpose(1, 2, 0)
            for sl in _get_center_slices(vol_hwp, crop_frac):
                all_vals.append(float(np.percentile(sl, 1)))
                all_vals.append(float(np.percentile(sl, 99)))
    vmin = np.percentile(all_vals[::2], 1) if all_vals else 0.0
    vmax = np.percentile(all_vals[1::2], 99) if all_vals else 1.0

    # Figure layout
    col_width = 1.6
    row_height = 1.6
    fig_width = n_cols * col_width + 1.5  # extra for labels
    fig_height = n_rows * row_height + 1.5  # extra for titles

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        wspace=0.03,
        hspace=0.08,
    )

    view_names = ["Axial", "Sagittal", "Coronal"]

    for row_idx, row_key in enumerate(row_order):
        for subj_idx in range(actual_n_subjects):
            vol = row_volumes[row_key][subj_idx]
            # (D, H, W) -> (H, W, D) for _get_center_slices
            vol_hwp = vol.transpose(1, 2, 0)
            slices = _get_center_slices(vol_hwp, crop_frac)

            for view_idx, sl in enumerate(slices):
                col_idx = subj_idx * 3 + view_idx
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.imshow(
                    sl.T,
                    cmap="gray",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )
                ax.set_xticks([])
                ax.set_yticks([])

                # Colored spine border per subject group
                color = COLORBLIND_PALETTE[subj_idx % len(COLORBLIND_PALETTE)]
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)

                # View labels on top row
                if row_idx == 0:
                    ax.set_title(view_names[view_idx], fontsize=8, pad=3)

                # Row label on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(row_key, fontsize=9, rotation=90, labelpad=8)

    # Subject ID super-titles above column groups
    for subj_idx in range(actual_n_subjects):
        center_col = subj_idx * 3 + 1  # middle of the 3-column group
        x_pos = (center_col + 0.5) / n_cols
        color = COLORBLIND_PALETTE[subj_idx % len(COLORBLIND_PALETTE)]
        fig.text(
            x_pos,
            0.98,
            subject_ids[subj_idx],
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    fig.suptitle(
        "Generation Comparison: Real vs Synthetic",
        fontsize=12,
        y=1.02,
    )
    save_figure(fig, output_dir / "generation_comparison_grid")


# ---------------------------------------------------------------------------
# Figure 2: Latent PSD by NFE
# ---------------------------------------------------------------------------


def plot_latent_psd_by_nfe(
    generation_dir: Path,
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Plot 2×2 panel of radially-averaged latent power spectra by NFE level.

    Args:
        generation_dir: Path to ``phase_5/generation/``.
        output_dir: Output directory for figures.
        nfe_levels: NFE levels to compare.
    """
    _apply_style()

    # Collect PSD per channel per NFE from sample #0
    spectra: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {ch: [] for ch in range(4)}

    nfe_min = min(nfe_levels) if nfe_levels else 1
    nfe_max = max(nfe_levels) if nfe_levels else 50
    norm = plt.Normalize(vmin=nfe_min, vmax=nfe_max)

    for nfe in nfe_levels:
        latent_path = generation_dir / "latents" / f"nfe_{nfe:03d}.h5"
        if not latent_path.exists():
            logger.warning("Latent archive not found: %s — skipping NFE=%d.", latent_path, nfe)
            continue
        latent = H5Manager.read_latent(latent_path, 0).numpy()  # (4, D, H, W)
        for ch in range(4):
            freqs, psd = _radial_average_3d(latent[ch])
            spectra[ch].append((freqs, psd, nfe))

    if not any(spectra.values()):
        logger.error("No latent archives loaded — cannot create PSD plot.")
        return

    # Generate noise reference (same shape as latent channel)
    sample_shape = None
    for ch_spectra in spectra.values():
        if ch_spectra:
            sample_shape = ch_spectra[0][0].shape  # just need freq bins
            break

    # Compute noise PSD from standard normal
    rng = np.random.default_rng(42)
    # Infer channel spatial dims from the first loaded latent
    first_nfe = nfe_levels[0]
    first_path = generation_dir / "latents" / f"nfe_{first_nfe:03d}.h5"
    noise_vol = rng.standard_normal((48, 48, 48)).astype(np.float32)
    if first_path.exists():
        first_latent = H5Manager.read_latent(first_path, 0).numpy()
        spatial = first_latent.shape[1:]
        noise_vol = rng.standard_normal(spatial).astype(np.float32)
    noise_freqs, noise_psd = _radial_average_3d(noise_vol)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ch in range(4):
        ax = axes_flat[ch]

        for freqs, psd, nfe in spectra[ch]:
            psd_safe = np.clip(psd, 1e-30, None)
            color = _NFE_CMAP(norm(nfe))
            ax.semilogy(
                freqs,
                psd_safe,
                color=color,
                linewidth=1.2,
                label=f"NFE={nfe}",
            )

        # Noise reference
        noise_psd_safe = np.clip(noise_psd, 1e-30, None)
        ax.semilogy(
            noise_freqs,
            noise_psd_safe,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="Noise",
        )

        ax.set_title(f"Channel {ch}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes_flat[2].set_xlabel("Spatial frequency (cycles/voxel)")
    axes_flat[3].set_xlabel("Spatial frequency (cycles/voxel)")
    axes_flat[0].set_ylabel("PSD")
    axes_flat[2].set_ylabel("PSD")

    # Colorbar for NFE mapping
    sm = plt.cm.ScalarMappable(cmap=_NFE_CMAP, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat.tolist(), label="NFE", shrink=0.8, pad=0.02)

    fig.suptitle(
        "Latent Power Spectrum by NFE Level (sample #0)",
        fontsize=11,
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.10, hspace=0.25)
    save_figure(fig, output_dir / "latent_psd_by_nfe")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Phase 5 generation outputs (CPU-only).")
    parser.add_argument(
        "--generation-dir",
        type=str,
        required=True,
        help="Path to phase_5/generation/ directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures (default: generation-dir/figures/).",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="Number of subjects per row in comparison grid.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="NFE levels to include.",
    )
    parser.add_argument(
        "--crop-frac",
        type=float,
        default=0.8,
        help="Center crop fraction for 2D slices.",
    )
    return parser.parse_args()


def main() -> None:
    """Run visualization pipeline."""
    args = parse_args()
    generation_dir = Path(args.generation_dir)
    output_dir = Path(args.output_dir) if args.output_dir else generation_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generation dir: %s", generation_dir)
    logger.info("Output dir:     %s", output_dir)
    logger.info("NFE levels:     %s", args.nfe)
    logger.info("N subjects:     %d", args.n_subjects)
    logger.info("Crop fraction:  %.2f", args.crop_frac)

    # Figure 1: Generation comparison grid (volumes)
    has_volumes = (generation_dir / "real_test.h5").exists() or any(
        (generation_dir / "volumes" / f"nfe_{nfe:03d}.h5").exists() for nfe in args.nfe
    )
    if has_volumes:
        logger.info("Creating generation comparison grid...")
        plot_generation_comparison_grid(
            generation_dir=generation_dir,
            output_dir=output_dir,
            nfe_levels=args.nfe,
            n_subjects=args.n_subjects,
            crop_frac=args.crop_frac,
        )
    else:
        logger.warning("No volume archives found — skipping comparison grid.")

    # Figure 2: Latent PSD by NFE
    has_latents = any(
        (generation_dir / "latents" / f"nfe_{nfe:03d}.h5").exists() for nfe in args.nfe
    )
    if has_latents:
        logger.info("Creating latent PSD comparison...")
        plot_latent_psd_by_nfe(
            generation_dir=generation_dir,
            output_dir=output_dir,
            nfe_levels=args.nfe,
        )
    else:
        logger.warning("No latent archives found — skipping PSD plot.")

    logger.info("Visualization complete. Figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
