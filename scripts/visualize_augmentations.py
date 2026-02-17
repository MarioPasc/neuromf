"""Visualise the effect of latent augmentations on decoded brain MRI.

Picks a random latent from the HDF5 shards, applies each augmentation
individually and all pairwise combinations, decodes through the frozen
MAISI VAE, and produces three image matrices (axial, sagittal, coronal).

Rows and columns represent augmentation transforms. Cell (i, j) shows
transform_j applied on top of transform_i (row-first, then column).
The diagonal shows single transforms. Cell (0, 0) is the original.

All transforms use the EXACT same classes as the training pipeline
(monai.transforms and neuromf.data.latent_augmentation), configured
with prob=1.0 for deterministic visualisation:

  - Identity (no transform)
  - Flip D (depth)  — monai.transforms.Flip(spatial_axis=0)
  - Flip H (height) — monai.transforms.Flip(spatial_axis=1)
  - Flip W (width)  — monai.transforms.Flip(spatial_axis=2)
  - Rotate90 sag     — monai.transforms.Rotate90(k=1, spatial_axes=(1,2))
  - Gaussian noise   — neuromf PerChannelGaussianNoise(prob=1.0)
  - Intensity scale   — monai.transforms.ScaleIntensity(minv=None, maxv=None, factor=0.05)

Usage:
    python scripts/visualize_augmentations.py
    python scripts/visualize_augmentations.py --seed 123
    python scripts/visualize_augmentations.py --single-only
    python scripts/visualize_augmentations.py --output-dir /tmp/aug_vis
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from monai.transforms import Flip, Rotate90, ScaleIntensity
from omegaconf import OmegaConf
from rich.logging import RichHandler

# ---- project imports (same modules as training pipeline) ----
from neuromf.data.latent_augmentation import PerChannelGaussianNoise
from neuromf.data.latent_dataset import LatentDataset
from neuromf.utils.latent_stats import load_latent_stats
from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


# ======================================================================
# Decode helper
# ======================================================================


@torch.no_grad()
def decode_latent_slices(
    vae: MAISIVAEWrapper,
    z_norm: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Denormalise, decode through VAE, extract mid-slices, free memory.

    Only keeps three tiny 2D slices instead of the full 192^3 volume,
    making this feasible on 8 GB VRAM GPUs.

    Args:
        vae: Loaded VAE wrapper.
        z_norm: Normalised latent (C, D, H, W).
        norm_mean: Per-channel mean (C, 1, 1, 1).
        norm_std: Per-channel std (C, 1, 1, 1).
        device: Target device.

    Returns:
        Dict with keys ``"axial"``, ``"coronal"``, ``"sagittal"`` mapping
        to 2D float32 numpy arrays.
    """
    z_raw = z_norm * norm_std + norm_mean
    z_batch = z_raw.unsqueeze(0).to(device)
    x_hat = vae.decode(z_batch)
    vol = x_hat.squeeze().cpu().float().clamp(0.0, 1.0).numpy()

    # RAS convention: dim0=R-L, dim1=A-P, dim2=I-S
    # Slicing along R-L → sagittal, A-P → coronal, I-S → axial
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
# Main
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise latent augmentation effects")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for latent selection")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/phase_4/augmentation_viz/)",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Config directory (default: configs/)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU decoding (slower but avoids VRAM limits).",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="Only decode single transforms (no pairwise matrix). 7 decodes instead of 49.",
    )
    args = parser.parse_args()

    # ---- Config ----
    project_root = Path(__file__).resolve().parent.parent
    configs_dir = Path(args.configs_dir) if args.configs_dir else project_root / "configs"
    base_cfg = OmegaConf.load(configs_dir / "base.yaml")
    train_cfg = OmegaConf.load(configs_dir / "train_meanflow.yaml")
    config = OmegaConf.merge(base_cfg, train_cfg)
    OmegaConf.resolve(config)

    latent_dir = Path(config.paths.latents_dir)
    stats_path = latent_dir / "latent_stats.json"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.paths.results_root) / "phase_4" / "augmentation_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # ---- Load a random latent ----
    logger.info("Loading latent dataset from %s", latent_dir)
    ds = LatentDataset(latent_dir, normalise=True, stats_path=stats_path)

    rng = np.random.default_rng(args.seed)
    idx = int(rng.integers(0, len(ds)))
    sample = ds[idx]
    z_norm = sample["z"]  # (C, D, H, W), normalised
    meta = sample["metadata"]
    logger.info(
        "Selected latent %d: %s / %s / %s  shape=%s",
        idx,
        meta.get("dataset", "?"),
        meta.get("subject_id", "?"),
        meta.get("session_id", "?"),
        list(z_norm.shape),
    )

    # ---- Normalisation stats for denormalisation ----
    stats = load_latent_stats(stats_path)
    per_ch = stats["per_channel"]
    n_ch = len(per_ch)
    means = [per_ch[f"channel_{c}"]["mean"] for c in range(n_ch)]
    stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_ch)]
    norm_mean = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1, 1)
    norm_std = torch.tensor(stds, dtype=torch.float32).view(-1, 1, 1, 1)

    # ---- Load VAE ----
    logger.info("Loading VAE from %s", config.paths.maisi_vae_weights)
    vae_config = MAISIVAEConfig.from_omegaconf(config)
    vae = MAISIVAEWrapper(vae_config, device=device)

    # ---- Define transforms ----
    # Uses the EXACT same transform classes as the training pipeline in
    # latent_augmentation.py, but with prob=1.0 for deterministic output.
    #
    # Training builds: [RandFlip×3, RandRotate90, PerChannelGaussianNoise, RandScaleIntensity]
    # Here we use:     [Flip×3,     Rotate90,     PerChannelGaussianNoise, ScaleIntensity    ]

    aug_cfg = config.training.augmentation
    noise_seed = 12345

    # Per-channel stds from latent stats (same source as build_latent_augmentation)
    channel_stds = [per_ch[f"channel_{c}"]["std"] for c in range(n_ch)]

    noise_transform = PerChannelGaussianNoise(
        prob=1.0,
        std_fraction=float(aug_cfg.gaussian_noise_std_fraction),
        channel_stds=channel_stds,
    )

    def _seeded_noise(z: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(noise_seed)
        return noise_transform(z)

    transforms: list[tuple[str, callable]] = [
        ("Original", lambda z: z),
        ("Flip D", Flip(spatial_axis=0)),
        ("Flip H", Flip(spatial_axis=1)),
        ("Flip W", Flip(spatial_axis=2)),
        ("Rot90 sag", Rotate90(k=1, spatial_axes=tuple(aug_cfg.rotate90_axes[0]))),
        ("Gauss noise", _seeded_noise),
        (
            "Scale ×1.05",
            ScaleIntensity(minv=None, maxv=None, factor=float(aug_cfg.intensity_scale_factors)),
        ),
    ]

    n_transforms = len(transforms)
    logger.info("Transforms (%d): %s", n_transforms, [t[0] for t in transforms])

    # ---- Build slice data ----
    if args.single_only:
        logger.info("Decoding %d single transforms (--single-only)...", n_transforms)
        single_slices: list[dict[str, np.ndarray]] = []
        for i, (name, fn) in enumerate(transforms):
            z_aug = fn(z_norm.clone())
            slices = decode_latent_slices(vae, z_aug, norm_mean, norm_std, device)
            single_slices.append(slices)
            logger.info("  [%d/%d] %s", i + 1, n_transforms, name)
    else:
        n_total = n_transforms**2
        logger.info("Decoding %d × %d = %d combinations...", n_transforms, n_transforms, n_total)
        slice_matrix: list[list[dict[str, np.ndarray]]] = []
        count = 0
        for i, (name_i, fn_i) in enumerate(transforms):
            row: list[dict[str, np.ndarray]] = []
            for j, (name_j, fn_j) in enumerate(transforms):
                z_aug = fn_j(fn_i(z_norm.clone()))
                slices = decode_latent_slices(vae, z_aug, norm_mean, norm_std, device)
                row.append(slices)
                count += 1
                logger.info("  [%d/%d] %s → %s", count, n_total, name_i, name_j)
            slice_matrix.append(row)

    # ---- Global intensity range for consistent colormap ----
    all_vals: list[np.ndarray] = []
    if args.single_only:
        for s in single_slices:
            for plane in ("axial", "coronal", "sagittal"):
                all_vals.append(s[plane].ravel())
    else:
        for row in slice_matrix:
            for cell in row:
                for plane in ("axial", "coronal", "sagittal"):
                    all_vals.append(cell[plane].ravel())
    all_flat = np.concatenate(all_vals)
    vmin_p, vmax_p = float(np.percentile(all_flat, 1)), float(np.percentile(all_flat, 99))
    del all_vals, all_flat

    # ---- Plot ----
    planes = ["axial", "coronal", "sagittal"]
    subject_label = f"{meta.get('dataset', '?')} / {meta.get('subject_id', '?')}"

    if args.single_only:
        for plane in planes:
            logger.info("Plotting %s view (single-only)...", plane)
            fig, axes = plt.subplots(1, n_transforms, figsize=(2.8 * n_transforms, 3.2))
            for i, (name, _) in enumerate(transforms):
                sl = single_slices[i][plane]
                axes[i].imshow(
                    sl.T if plane == "sagittal" else sl,
                    cmap="gray",
                    vmin=vmin_p,
                    vmax=vmax_p,
                    origin="lower",
                    aspect="equal",
                )
                axes[i].set_title(name, fontsize=8, fontweight="bold")
                colour = "#2ecc71" if i == 0 else "#e74c3c"
                for spine in axes[i].spines.values():
                    spine.set_edgecolor(colour)
                    spine.set_linewidth(2)
                axes[i].set_xticks([])
                axes[i].set_yticks([])

            fig.suptitle(
                f"Latent Augmentations — {plane.capitalize()} View\n{subject_label}",
                fontsize=11,
                fontweight="bold",
            )
            fig.tight_layout()
            out_path = output_dir / f"augmentation_single_{plane}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", out_path)
    else:
        for plane in planes:
            logger.info("Plotting %s view...", plane)
            fig = plt.figure(figsize=(2.4 * n_transforms + 1.5, 2.4 * n_transforms + 1.5))
            gs = GridSpec(
                n_transforms + 1,
                n_transforms + 1,
                figure=fig,
                wspace=0.05,
                hspace=0.05,
                left=0.06,
                right=0.97,
                top=0.93,
                bottom=0.06,
            )

            # Column headers
            for j, (name_j, _) in enumerate(transforms):
                ax = fig.add_subplot(gs[0, j + 1])
                ax.text(
                    0.5,
                    0.5,
                    name_j,
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    rotation=45,
                    transform=ax.transAxes,
                )
                ax.axis("off")

            # Row headers
            for i, (name_i, _) in enumerate(transforms):
                ax = fig.add_subplot(gs[i + 1, 0])
                ax.text(
                    0.5,
                    0.5,
                    name_i,
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

            # Image matrix
            for i in range(n_transforms):
                for j in range(n_transforms):
                    ax = fig.add_subplot(gs[i + 1, j + 1])
                    sl = slice_matrix[i][j][plane]
                    ax.imshow(
                        sl.T if plane == "sagittal" else sl,
                        cmap="gray",
                        vmin=vmin_p,
                        vmax=vmax_p,
                        origin="lower",
                        aspect="equal",
                    )
                    if i == j and i > 0:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#e74c3c")
                            spine.set_linewidth(2)
                    elif i == 0 and j == 0:
                        for spine in ax.spines.values():
                            spine.set_edgecolor("#2ecc71")
                            spine.set_linewidth(2)
                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.suptitle(
                f"Latent Augmentation Matrix — {plane.capitalize()} View\n"
                f"{subject_label}  |  Diagonal = single transform  |  "
                f"Off-diagonal = row then column",
                fontsize=11,
                fontweight="bold",
            )
            out_path = output_dir / f"augmentation_matrix_{plane}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", out_path)

    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
