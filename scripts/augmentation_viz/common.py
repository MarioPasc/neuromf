"""Shared utilities for the augmentation visualisation suite.

Provides config loading, VAE/dataset helpers, decode-to-slices, transform
builders, subject selection, intensity range computation, and plotting.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from monai.transforms import Flip, Rotate90, ScaleIntensity
from omegaconf import DictConfig, OmegaConf

from neuromf.data.latent_augmentation import PerChannelGaussianNoise
from neuromf.data.latent_dataset import LatentDataset
from neuromf.utils.latent_stats import load_latent_stats
from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

logger = logging.getLogger(__name__)

# Anatomical plane order used throughout the suite
PLANES = ("axial", "coronal", "sagittal")


# ======================================================================
# Config loading
# ======================================================================


def load_merged_config(configs_dir: str | None = None) -> DictConfig:
    """Load and merge configs following the train.py pattern.

    Merge chain: base.yaml -> configs/train_meanflow.yaml -> overlay (if any).

    Args:
        configs_dir: Optional overlay directory (e.g. ``configs/picasso``).
            If None, uses ``configs/`` as the sole config source.

    Returns:
        Fully resolved OmegaConf config.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    root_configs = project_root / "configs"

    if configs_dir is not None:
        overlay_dir = Path(configs_dir)
    else:
        overlay_dir = root_configs

    # Layer 1: base.yaml (from overlay dir or root)
    base_path = overlay_dir / "base.yaml"
    if not base_path.exists():
        base_path = root_configs / "base.yaml"

    layers = [OmegaConf.load(base_path)]

    # Layer 2: root train_meanflow.yaml (full augmentation settings)
    main_train = root_configs / "train_meanflow.yaml"
    if main_train.exists():
        layers.append(OmegaConf.load(main_train))

    # Layer 3: overlay train_meanflow.yaml (hardware settings)
    if configs_dir is not None:
        overlay_train = overlay_dir / "train_meanflow.yaml"
        if overlay_train.exists() and overlay_train.resolve() != main_train.resolve():
            layers.append(OmegaConf.load(overlay_train))

    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)
    return config


# ======================================================================
# VAE and normalisation loading
# ======================================================================


def load_vae(config: DictConfig, device: torch.device) -> MAISIVAEWrapper:
    """Load the frozen MAISI VAE.

    Args:
        config: Merged config with ``vae`` and ``paths`` sections.
        device: Target device.

    Returns:
        Loaded VAE wrapper in eval mode.
    """
    vae_config = MAISIVAEConfig.from_omegaconf(config)
    return MAISIVAEWrapper(vae_config, device=device)


def load_normalisation_params(
    config: DictConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Load per-channel normalisation stats from latent_stats.json.

    Args:
        config: Merged config with ``paths.latents_dir``.

    Returns:
        Tuple of ``(norm_mean, norm_std, channel_stds)`` where mean/std are
        ``(C, 1, 1, 1)`` tensors and channel_stds is a plain list.
    """
    stats_path = Path(config.paths.latents_dir) / "latent_stats.json"
    stats = load_latent_stats(stats_path)
    per_ch = stats["per_channel"]
    n_ch = len(per_ch)

    means = [per_ch[f"channel_{c}"]["mean"] for c in range(n_ch)]
    stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_ch)]

    norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1, 1)
    norm_std = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1, 1)

    return norm_mean, norm_std, stds


# ======================================================================
# Decode helper
# ======================================================================


@torch.no_grad()
def decode_latent_to_slices(
    vae: MAISIVAEWrapper,
    z_norm: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Denormalise, decode through VAE, extract 3 mid-slices.

    Frees the full volume immediately to keep peak memory low.

    Args:
        vae: Loaded VAE wrapper.
        z_norm: Normalised latent ``(C, D, H, W)``.
        norm_mean: Per-channel mean ``(C, 1, 1, 1)``.
        norm_std: Per-channel std ``(C, 1, 1, 1)``.
        device: Target device.

    Returns:
        Dict with ``"axial"``, ``"coronal"``, ``"sagittal"`` 2D numpy arrays.
    """
    z_raw = z_norm * norm_std + norm_mean
    z_batch = z_raw.unsqueeze(0).to(device)
    x_hat = vae.decode(z_batch)
    vol = x_hat.squeeze().cpu().float().clamp(0.0, 1.0).numpy()

    # RAS convention: dim0=R-L (sagittal), dim1=A-P (coronal), dim2=I-S (axial)
    d, h, w = vol.shape
    slices = {
        "sagittal": vol[d // 2, :, :].copy(),
        "coronal": vol[:, h // 2, :].copy(),
        "axial": vol[:, :, w // 2].copy(),
    }

    del z_raw, z_batch, x_hat, vol
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return slices


# ======================================================================
# Transform builders
# ======================================================================


def build_standard_transforms(
    config: DictConfig,
    channel_stds: list[float],
    noise_seed: int = 12345,
) -> list[tuple[str, Callable]]:
    """Build the 7 standard augmentation transforms for visualisation.

    Uses the exact same classes as the training pipeline with ``prob=1.0``.

    Args:
        config: Merged config with ``training.augmentation`` section.
        channel_stds: Per-channel standard deviations from latent stats.
        noise_seed: Fixed seed for reproducible Gaussian noise.

    Returns:
        List of ``(label, transform_fn)`` tuples.
    """
    aug_cfg = config.training.augmentation

    noise_transform = PerChannelGaussianNoise(
        prob=1.0,
        std_fraction=float(aug_cfg.gaussian_noise_std_fraction),
        channel_stds=channel_stds,
    )

    def _seeded_noise(z: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(noise_seed)
        return noise_transform(z)

    return [
        ("Original", lambda z: z),
        ("Flip D", Flip(spatial_axis=0)),
        ("Flip H", Flip(spatial_axis=1)),
        ("Flip W", Flip(spatial_axis=2)),
        ("Rot90 sag", Rotate90(k=1, spatial_axes=tuple(aug_cfg.rotate90_axes[0]))),
        ("Gauss noise", _seeded_noise),
        (
            "Scale \u00d71.05",
            ScaleIntensity(minv=None, maxv=None, factor=float(aug_cfg.intensity_scale_factors)),
        ),
    ]


# ======================================================================
# Subject selection
# ======================================================================


def select_diverse_subjects(
    ds: LatentDataset,
    n: int = 5,
    seed: int = 42,
) -> list[int]:
    """Pick diverse subjects spanning different FOMO-60K datasets.

    Scans metadata to group by dataset, picks one per dataset, then fills
    remainder from the largest dataset.

    Args:
        ds: Latent dataset instance.
        n: Number of subjects to select.
        seed: Random seed.

    Returns:
        List of dataset indices.
    """
    rng = np.random.default_rng(seed)

    # Sample a subset of indices to read metadata (avoid scanning full dataset)
    total = len(ds)
    scan_indices = rng.choice(total, size=min(200, total), replace=False)

    dataset_groups: dict[str, list[int]] = {}
    for idx in scan_indices:
        sample = ds[int(idx)]
        dataset_name = sample["metadata"].get("dataset", "unknown")
        dataset_groups.setdefault(dataset_name, []).append(int(idx))

    selected: list[int] = []
    # Pick one from each dataset
    for name in sorted(dataset_groups.keys()):
        if len(selected) >= n:
            break
        choice = int(rng.choice(dataset_groups[name]))
        selected.append(choice)

    # Fill remainder from the largest dataset
    if len(selected) < n:
        largest = max(dataset_groups.keys(), key=lambda k: len(dataset_groups[k]))
        pool = [i for i in dataset_groups[largest] if i not in selected]
        remaining = min(n - len(selected), len(pool))
        if remaining > 0:
            extras = rng.choice(pool, size=remaining, replace=False)
            selected.extend(int(e) for e in extras)

    return selected


# ======================================================================
# Intensity range
# ======================================================================


def compute_global_intensity_range(
    all_slices: list[dict[str, np.ndarray]],
) -> tuple[float, float]:
    """Compute 1st/99th percentile intensity across all slice dicts.

    Args:
        all_slices: List of slice dicts (each has ``"axial"``, ``"coronal"``,
            ``"sagittal"`` keys).

    Returns:
        ``(vmin, vmax)`` tuple.
    """
    vals: list[np.ndarray] = []
    for s in all_slices:
        for plane in PLANES:
            vals.append(s[plane].ravel())
    flat = np.concatenate(vals)
    vmin = float(np.percentile(flat, 1))
    vmax = float(np.percentile(flat, 99))
    return vmin, vmax


# ======================================================================
# Plotting
# ======================================================================


def plot_strip(
    slices_1d: list[dict[str, np.ndarray]],
    labels: list[str],
    title: str,
    output_dir: Path,
    filename_prefix: str,
    vmin: float,
    vmax: float,
) -> None:
    """Plot a single row of images for each anatomical plane.

    First image gets a green border (original), rest get a thin grey border.

    Args:
        slices_1d: List of slice dicts (one per column).
        labels: Column labels.
        title: Figure title (plane name appended automatically).
        output_dir: Directory to save PNGs.
        filename_prefix: Filename prefix (e.g. ``"noise_sweep"``).
        vmin: Colormap minimum.
        vmax: Colormap maximum.
    """
    n_cols = len(slices_1d)
    for plane in PLANES:
        fig, axes = plt.subplots(1, n_cols, figsize=(2.8 * n_cols, 3.2))
        if n_cols == 1:
            axes = [axes]

        for i, (sl_dict, label) in enumerate(zip(slices_1d, labels)):
            sl = sl_dict[plane]
            axes[i].imshow(
                sl.T if plane == "sagittal" else sl,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="equal",
            )
            axes[i].set_title(label, fontsize=8, fontweight="bold")
            colour = "#2ecc71" if i == 0 else "#aaaaaa"
            for spine in axes[i].spines.values():
                spine.set_edgecolor(colour)
                spine.set_linewidth(2 if i == 0 else 0.5)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        fig.suptitle(
            f"{title} \u2014 {plane.capitalize()} View",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        out_path = output_dir / f"{filename_prefix}_{plane}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out_path)


def plot_grid(
    slices_2d: list[list[dict[str, np.ndarray]]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    output_dir: Path,
    filename_prefix: str,
    vmin: float,
    vmax: float,
    highlight_diagonal: bool = False,
) -> None:
    """Plot an N x M grid of images for each anatomical plane.

    Args:
        slices_2d: 2D list of slice dicts ``[row][col]``.
        row_labels: Row header labels.
        col_labels: Column header labels.
        title: Figure title (plane name appended automatically).
        output_dir: Directory to save PNGs.
        filename_prefix: Filename prefix.
        vmin: Colormap minimum.
        vmax: Colormap maximum.
        highlight_diagonal: If True, highlight diagonal cells (red) and
            origin cell (green).
    """
    n_rows = len(slices_2d)
    n_cols = len(slices_2d[0])

    for plane in PLANES:
        fig = plt.figure(figsize=(2.4 * n_cols + 1.5, 2.4 * n_rows + 1.5))
        gs = GridSpec(
            n_rows + 1,
            n_cols + 1,
            figure=fig,
            wspace=0.05,
            hspace=0.05,
            left=0.06,
            right=0.97,
            top=0.93,
            bottom=0.06,
        )

        # Column headers
        for j, label in enumerate(col_labels):
            ax = fig.add_subplot(gs[0, j + 1])
            ax.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                rotation=45 if n_cols > 5 else 0,
                transform=ax.transAxes,
            )
            ax.axis("off")

        # Row headers
        for i, label in enumerate(row_labels):
            ax = fig.add_subplot(gs[i + 1, 0])
            ax.text(
                0.5,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                transform=ax.transAxes,
            )
            ax.axis("off")

        # Corner label
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_corner.text(
            0.5,
            0.5,
            "row\u2192col",
            ha="center",
            va="center",
            fontsize=7,
            fontstyle="italic",
            transform=ax_corner.transAxes,
        )
        ax_corner.axis("off")

        # Image cells
        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(gs[i + 1, j + 1])
                sl = slices_2d[i][j][plane]
                ax.imshow(
                    sl.T if plane == "sagittal" else sl,
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                    aspect="equal",
                )
                if highlight_diagonal:
                    if i == 0 and j == 0:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#2ecc71")
                            spine.set_linewidth(2)
                    elif i == j and i > 0:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#e74c3c")
                            spine.set_linewidth(2)
                ax.set_xticks([])
                ax.set_yticks([])

        fig.suptitle(
            f"{title} \u2014 {plane.capitalize()} View",
            fontsize=11,
            fontweight="bold",
        )
        out_path = output_dir / f"{filename_prefix}_{plane}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out_path)
