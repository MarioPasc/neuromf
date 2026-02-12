"""Publication-quality plotting for 3D medical image validation.

Paper-style constants: serif font, colorblind-safe palette, 300 DPI.
All functions accept numpy arrays and manage their own figure lifecycle.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper style constants
# ---------------------------------------------------------------------------
FONT_FAMILY = "serif"
DPI = 300
COLORBLIND_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
]


def _apply_style() -> None:
    """Set matplotlib rcParams for publication figures."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure as both PDF and PNG, then close.

    Args:
        fig: Matplotlib figure to save.
        path: Output path (without extension). Both ``.pdf`` and ``.png``
            are written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    fig.savefig(str(stem.with_suffix(".pdf")))
    fig.savefig(str(stem.with_suffix(".png")))
    plt.close(fig)
    logger.info("Saved figure: %s.{pdf,png}", stem)


def _get_center_slices(
    vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract center axial, sagittal, and coronal slices from a 3D volume.

    Args:
        vol: 3D array of shape ``(H, W, D)``.

    Returns:
        Tuple of (axial, sagittal, coronal) 2D slices.
    """
    h, w, d = vol.shape
    axial = vol[:, :, d // 2]
    sagittal = vol[h // 2, :, :]
    coronal = vol[:, w // 2, :]
    return axial, sagittal, coronal


def plot_reconstruction_comparison(
    original: np.ndarray,
    recon: np.ndarray,
    name: str,
    path: Path,
) -> None:
    """Plot 3x3 grid: rows=axial/sag/cor, cols=original/recon/abs error.

    Args:
        original: Original volume as 3D numpy array ``(H, W, D)``.
        recon: Reconstructed volume as 3D numpy array ``(H, W, D)``.
        name: Volume identifier for the title.
        path: Output path (extension ignored; PDF+PNG saved).
    """
    _apply_style()
    error = np.abs(original - recon)

    orig_slices = _get_center_slices(original)
    recon_slices = _get_center_slices(recon)
    error_slices = _get_center_slices(error)

    view_labels = ["Axial", "Sagittal", "Coronal"]
    col_labels = ["Original", "Reconstruction", "|Error|"]

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for row in range(3):
        vmin = min(orig_slices[row].min(), recon_slices[row].min())
        vmax = max(orig_slices[row].max(), recon_slices[row].max())

        axes[row, 0].imshow(orig_slices[row].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[row, 1].imshow(recon_slices[row].T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        im = axes[row, 2].imshow(error_slices[row].T, cmap="hot", origin="lower")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

        axes[row, 0].set_ylabel(view_labels[row])

    for col in range(3):
        axes[0, col].set_title(col_labels[col])

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"VAE Reconstruction — {name}", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, path)


def plot_metrics_distribution(
    values: list[float] | np.ndarray,
    name: str,
    threshold: float,
    path: Path,
) -> None:
    """Violin + strip plot of per-volume metric with threshold line.

    Args:
        values: Per-volume metric values.
        name: Metric name (e.g. "SSIM", "PSNR (dB)").
        threshold: Minimum acceptable value (drawn as horizontal line).
        path: Output path (extension ignored; PDF+PNG saved).
    """
    _apply_style()
    values = np.asarray(values)

    fig, ax = plt.subplots(figsize=(3.5, 4))
    parts = ax.violinplot(values, positions=[0], showmeans=True, showmedians=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORBLIND_PALETTE[0])
        pc.set_alpha(0.3)

    # Strip plot (individual points)
    jitter = np.random.default_rng(42).uniform(-0.05, 0.05, len(values))
    ax.scatter(jitter, values, c=COLORBLIND_PALETTE[0], s=20, alpha=0.7, zorder=3)

    ax.axhline(
        threshold,
        color=COLORBLIND_PALETTE[1],
        linestyle="--",
        linewidth=1,
        label=f"Threshold = {threshold}",
    )
    ax.legend(fontsize=8)
    ax.set_ylabel(name)
    ax.set_xticks([])
    ax.set_title(f"{name} Distribution (n={len(values)})")
    fig.tight_layout()
    save_figure(fig, path)


def plot_latent_histograms(
    channel_samples: dict[int, np.ndarray],
    channel_stats: dict[int, dict[str, float]],
    path: Path,
) -> None:
    """2x2 per-channel histograms with Gaussian overlay and mu/sigma annotation.

    Args:
        channel_samples: Dict mapping channel index to 1D array of sampled values.
        channel_stats: Dict mapping channel index to ``{"mean": ..., "std": ...}``.
        path: Output path (extension ignored; PDF+PNG saved).
    """
    _apply_style()
    n_channels = len(channel_samples)
    ncols = 2
    nrows = (n_channels + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for ch in range(n_channels):
        ax = axes[ch // ncols, ch % ncols]
        samples = channel_samples[ch]
        stats = channel_stats[ch]
        mu, sigma = stats["mean"], stats["std"]

        ax.hist(
            samples,
            bins=80,
            density=True,
            alpha=0.6,
            color=COLORBLIND_PALETTE[ch % len(COLORBLIND_PALETTE)],
        )

        # Gaussian overlay
        x_range = np.linspace(samples.min(), samples.max(), 200)
        gaussian = np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x_range, gaussian, "k--", linewidth=1, label=f"N({mu:.2f}, {sigma:.2f})")

        ax.set_title(f"Channel {ch}")
        ax.set_xlabel("Latent value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.annotate(
            f"\u03bc={mu:.3f}\n\u03c3={sigma:.3f}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    # Hide unused axes
    for idx in range(n_channels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle("Latent Space Per-Channel Distribution", fontsize=11, y=1.0)
    fig.tight_layout()
    save_figure(fig, path)


def plot_error_heatmap(
    mean_error_3d: np.ndarray,
    path: Path,
) -> None:
    """1x3 axial/sagittal/coronal center slices of mean absolute error.

    Args:
        mean_error_3d: 3D array of mean absolute error ``(H, W, D)``.
        path: Output path (extension ignored; PDF+PNG saved).
    """
    _apply_style()
    slices = _get_center_slices(mean_error_3d)
    view_labels = ["Axial", "Sagittal", "Coronal"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    for i, (sl, label) in enumerate(zip(slices, view_labels)):
        im = axes[i].imshow(sl.T, cmap="hot", origin="lower")
        axes[i].set_title(label)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    fig.suptitle("Mean Absolute Reconstruction Error", fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, path)
