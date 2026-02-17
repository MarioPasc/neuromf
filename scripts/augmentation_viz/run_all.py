"""Augmentation Visualisation Suite — CLI entrypoint.

Produces 6 visualisations (18 PNGs total) showing how latent-space
augmentations affect decoded brain MRI. Loads the VAE and dataset once,
then runs each visualisation sequentially.

Usage:
    python scripts/augmentation_viz/run_all.py
    python scripts/augmentation_viz/run_all.py --configs-dir configs/picasso
    python scripts/augmentation_viz/run_all.py --skip pairwise,multi_subject
    python scripts/augmentation_viz/run_all.py --cpu --seed 123
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Rotate90, ScaleIntensity
from rich.logging import RichHandler

# Allow imports from sibling module
script_dir = str(Path(__file__).resolve().parent)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from common import (
    build_standard_transforms,
    compute_global_intensity_range,
    decode_latent_to_slices,
    load_merged_config,
    load_normalisation_params,
    load_vae,
    plot_grid,
    plot_strip,
    select_diverse_subjects,
)

from neuromf.data.latent_augmentation import PerChannelGaussianNoise
from neuromf.data.latent_dataset import LatentDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


# ======================================================================
# Visualisation functions
# ======================================================================


def _make_decode_fn(
    vae: object,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    device: torch.device,
):
    """Return a closure that decodes a normalised latent to slices."""

    def _decode(z_norm: torch.Tensor) -> dict[str, np.ndarray]:
        return decode_latent_to_slices(vae, z_norm, norm_mean, norm_std, device)

    return _decode


def _run_pairwise(
    z_norm: torch.Tensor,
    transforms: list[tuple[str, object]],
    decode_fn: object,
    output_dir: Path,
    subject_label: str,
) -> int:
    """7x7 pairwise augmentation matrix.

    Returns:
        Number of decode calls.
    """
    n = len(transforms)
    labels = [t[0] for t in transforms]
    logger.info("Pairwise matrix: %d x %d = %d decodes", n, n, n * n)

    slice_matrix: list[list[dict[str, np.ndarray]]] = []
    count = 0
    for i, (name_i, fn_i) in enumerate(transforms):
        row: list[dict[str, np.ndarray]] = []
        for j, (name_j, fn_j) in enumerate(transforms):
            z_aug = fn_j(fn_i(z_norm.clone()))
            row.append(decode_fn(z_aug))
            count += 1
            logger.info("  [%d/%d] %s -> %s", count, n * n, name_i, name_j)
        slice_matrix.append(row)

    vmin, vmax = compute_global_intensity_range([cell for row in slice_matrix for cell in row])

    plot_grid(
        slice_matrix,
        row_labels=labels,
        col_labels=labels,
        title=f"Pairwise Augmentation Matrix\n{subject_label}  |  Diagonal = single  |  Off-diag = row then col",
        output_dir=output_dir,
        filename_prefix="pairwise_matrix",
        vmin=vmin,
        vmax=vmax,
        highlight_diagonal=True,
    )
    return count


def _run_noise_sweep(
    z_norm: torch.Tensor,
    channel_stds: list[float],
    decode_fn: object,
    output_dir: Path,
    subject_label: str,
    noise_seed: int = 12345,
) -> int:
    """Sweep Gaussian noise std_fraction from 0.01 to 0.5.

    Returns:
        Number of decode calls.
    """
    fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    labels = ["Original"] + [f"std={f}" for f in fractions]
    logger.info(
        "Noise sweep: %d levels + original = %d decodes", len(fractions), len(fractions) + 1
    )

    slices_list: list[dict[str, np.ndarray]] = []

    # Original
    slices_list.append(decode_fn(z_norm.clone()))

    # Noise levels
    for f in fractions:
        noise_t = PerChannelGaussianNoise(prob=1.0, std_fraction=f, channel_stds=channel_stds)
        torch.manual_seed(noise_seed)
        z_aug = noise_t(z_norm.clone())
        slices_list.append(decode_fn(z_aug))

    vmin, vmax = compute_global_intensity_range(slices_list)
    plot_strip(
        slices_list,
        labels,
        f"Gaussian Noise Sweep\n{subject_label}",
        output_dir,
        "noise_sweep",
        vmin,
        vmax,
    )
    return len(slices_list)


def _run_scale_sweep(
    z_norm: torch.Tensor,
    decode_fn: object,
    output_dir: Path,
    subject_label: str,
) -> int:
    """Sweep intensity scale factors from -0.2 to +0.2.

    Returns:
        Number of decode calls.
    """
    factors = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    labels = [
        "\u00d70.80",
        "\u00d70.90",
        "\u00d70.95",
        "Original",
        "\u00d71.05",
        "\u00d71.10",
        "\u00d71.20",
    ]
    logger.info("Scale sweep: %d factors = %d decodes", len(factors), len(factors))

    slices_list: list[dict[str, np.ndarray]] = []
    for f in factors:
        if f == 0.0:
            z_aug = z_norm.clone()
        else:
            scale_t = ScaleIntensity(minv=None, maxv=None, factor=f)
            z_aug = scale_t(z_norm.clone())
        slices_list.append(decode_fn(z_aug))

    vmin, vmax = compute_global_intensity_range(slices_list)
    plot_strip(
        slices_list,
        labels,
        f"Intensity Scale Sweep\n{subject_label}",
        output_dir,
        "scale_sweep",
        vmin,
        vmax,
    )
    return len(slices_list)


def _run_rotation_planes(
    z_norm: torch.Tensor,
    decode_fn: object,
    output_dir: Path,
    subject_label: str,
) -> int:
    """3 rotation planes x 3 k values = 3x4 grid (with original column).

    Returns:
        Number of decode calls.
    """
    plane_specs = [
        ("Axial (D-H)", (0, 1)),
        ("Coronal (D-W)", (0, 2)),
        ("Sagittal (H-W)", (1, 2)),
    ]
    k_values = [1, 2, 3]
    col_labels = ["Original"] + [f"k={k}" for k in k_values]
    row_labels = [name for name, _ in plane_specs]
    logger.info("Rotation planes: 3 planes x 4 cols = ~%d decodes", 3 * 4)

    # Decode original once, reuse for all rows
    orig_slices = decode_fn(z_norm.clone())
    total_decodes = 1

    slice_matrix: list[list[dict[str, np.ndarray]]] = []
    for _, axes in plane_specs:
        row: list[dict[str, np.ndarray]] = [orig_slices]
        for k in k_values:
            rot = Rotate90(k=k, spatial_axes=axes)
            z_aug = rot(z_norm.clone())
            row.append(decode_fn(z_aug))
            total_decodes += 1
        slice_matrix.append(row)

    all_slices = [cell for row in slice_matrix for cell in row]
    vmin, vmax = compute_global_intensity_range(all_slices)

    plot_grid(
        slice_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=f"Rotation Planes\n{subject_label}",
        output_dir=output_dir,
        filename_prefix="rotation_planes",
        vmin=vmin,
        vmax=vmax,
    )
    return total_decodes


def _run_multi_subject(
    ds: LatentDataset,
    transforms: list[tuple[str, object]],
    decode_fn: object,
    output_dir: Path,
    seed: int = 42,
) -> int:
    """N subjects x 7 transforms grid.

    Returns:
        Number of decode calls.
    """
    indices = select_diverse_subjects(ds, n=5, seed=seed)
    n_subj = len(indices)
    n_transforms = len(transforms)
    logger.info(
        "Multi-subject: %d subjects x %d transforms = %d decodes",
        n_subj,
        n_transforms,
        n_subj * n_transforms,
    )

    col_labels = [t[0] for t in transforms]
    row_labels: list[str] = []
    slice_matrix: list[list[dict[str, np.ndarray]]] = []
    count = 0

    for idx in indices:
        sample = ds[idx]
        z_norm = sample["z"]
        meta = sample["metadata"]
        row_label = f"{meta.get('dataset', '?')} / {meta.get('subject_id', '?')}"
        row_labels.append(row_label)

        row: list[dict[str, np.ndarray]] = []
        for name, fn in transforms:
            z_aug = fn(z_norm.clone())
            row.append(decode_fn(z_aug))
            count += 1
            logger.info("  [%d/%d] %s: %s", count, n_subj * n_transforms, row_label, name)
        slice_matrix.append(row)

    all_slices = [cell for row in slice_matrix for cell in row]
    vmin, vmax = compute_global_intensity_range(all_slices)

    plot_grid(
        slice_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Multi-Subject Augmentation Comparison",
        output_dir=output_dir,
        filename_prefix="multi_subject",
        vmin=vmin,
        vmax=vmax,
    )
    return count


def _run_channel_ablation(
    z_norm: torch.Tensor,
    decode_fn: object,
    output_dir: Path,
    subject_label: str,
) -> int:
    """Zero out each latent channel individually (5 images total).

    Zeroing in normalised space = setting to the per-channel mean in raw space.

    Returns:
        Number of decode calls.
    """
    n_ch = z_norm.shape[0]
    labels = ["Original"] + [f"Zero ch{c}" for c in range(n_ch)]
    logger.info("Channel ablation: %d images", 1 + n_ch)

    slices_list: list[dict[str, np.ndarray]] = []

    # Original
    slices_list.append(decode_fn(z_norm.clone()))

    # Zero each channel
    for c in range(n_ch):
        z_abl = z_norm.clone()
        z_abl[c] = 0.0  # In normalised space, 0 = channel mean
        slices_list.append(decode_fn(z_abl))

    vmin, vmax = compute_global_intensity_range(slices_list)
    plot_strip(
        slices_list,
        labels,
        f"Channel Ablation\n{subject_label}",
        output_dir,
        "channel_ablation",
        vmin,
        vmax,
    )
    return len(slices_list)


# ======================================================================
# Main
# ======================================================================


ALL_VIZ_NAMES = [
    "pairwise",
    "noise_sweep",
    "scale_sweep",
    "rotation_planes",
    "multi_subject",
    "channel_ablation",
]


def main() -> None:
    """CLI entry point for the augmentation visualisation suite."""
    parser = argparse.ArgumentParser(
        description="Augmentation Visualisation Suite — 6 visualisations, 18 PNGs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject selection")
    parser.add_argument("--configs-dir", type=str, default=None, help="Config overlay directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for PNGs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU decoding")
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated viz names to skip: " + ", ".join(ALL_VIZ_NAMES),
    )
    args = parser.parse_args()

    t_start = time.time()
    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())
    for s in skip_set:
        if s not in ALL_VIZ_NAMES:
            logger.warning("Unknown viz name to skip: '%s'. Valid: %s", s, ALL_VIZ_NAMES)

    # ---- Config ----
    config = load_merged_config(args.configs_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.paths.results_root) / "phase_4" / "augmentation_viz"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s", device)

    # ---- Load shared resources ----
    logger.info("Loading normalisation params...")
    norm_mean, norm_std, channel_stds = load_normalisation_params(config)

    logger.info("Loading dataset...")
    latent_dir = Path(config.paths.latents_dir)
    stats_path = latent_dir / "latent_stats.json"
    ds = LatentDataset(latent_dir, normalise=True, stats_path=stats_path)

    logger.info("Loading VAE...")
    vae = load_vae(config, device)

    decode_fn = _make_decode_fn(vae, norm_mean, norm_std, device)

    # ---- Select primary subject ----
    rng = np.random.default_rng(args.seed)
    idx = int(rng.integers(0, len(ds)))
    sample = ds[idx]
    z_norm = sample["z"]
    meta = sample["metadata"]
    subject_label = f"{meta.get('dataset', '?')} / {meta.get('subject_id', '?')}"
    logger.info("Primary subject [%d]: %s  shape=%s", idx, subject_label, list(z_norm.shape))

    # ---- Build transforms ----
    transforms = build_standard_transforms(config, channel_stds)

    # ---- Run visualisations ----
    total_decodes = 0

    if "pairwise" not in skip_set:
        logger.info("=" * 50)
        logger.info("1/6: Pairwise Matrix")
        total_decodes += _run_pairwise(z_norm, transforms, decode_fn, output_dir, subject_label)

    if "noise_sweep" not in skip_set:
        logger.info("=" * 50)
        logger.info("2/6: Noise Sweep")
        total_decodes += _run_noise_sweep(
            z_norm, channel_stds, decode_fn, output_dir, subject_label
        )

    if "scale_sweep" not in skip_set:
        logger.info("=" * 50)
        logger.info("3/6: Scale Sweep")
        total_decodes += _run_scale_sweep(z_norm, decode_fn, output_dir, subject_label)

    if "rotation_planes" not in skip_set:
        logger.info("=" * 50)
        logger.info("4/6: Rotation Planes")
        total_decodes += _run_rotation_planes(z_norm, decode_fn, output_dir, subject_label)

    if "multi_subject" not in skip_set:
        logger.info("=" * 50)
        logger.info("5/6: Multi-Subject")
        total_decodes += _run_multi_subject(ds, transforms, decode_fn, output_dir, seed=args.seed)

    if "channel_ablation" not in skip_set:
        logger.info("=" * 50)
        logger.info("6/6: Channel Ablation")
        total_decodes += _run_channel_ablation(z_norm, decode_fn, output_dir, subject_label)

    # ---- Summary ----
    elapsed = time.time() - t_start
    n_pngs = len(list(output_dir.glob("*.png")))
    logger.info("=" * 50)
    logger.info("DONE: %d decodes, %d PNGs, %.1f min", total_decodes, n_pngs, elapsed / 60)
    logger.info("Output: %s", output_dir)


if __name__ == "__main__":
    main()
