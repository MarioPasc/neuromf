"""Latent space visualization for NeuroMF.

NeuroMF operates in the 4x48x48x48 latent space of a frozen MAISI VAE,
unlike pixel-space methods (MOTFM, DDPM). This module provides publication-
quality visualizations of the latent space, including t-SNE embeddings,
per-channel distribution histograms, and generation trajectory plots.

All figures follow IEEE TMI style and save both PDF and PNG outputs alongside
.npz source data for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from experiments.utils.settings import (
    PAUL_TOL_BRIGHT,
    PLOT_SETTINGS,
    apply_ieee_style,
    get_figure_size,
)
from neuromf.utils.visualisation import save_figure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color scheme for NFE levels (consistent with figures.py)
# ---------------------------------------------------------------------------
_NFE_COLORS: dict[int, str] = {
    1: PAUL_TOL_BRIGHT["red"],
    10: PAUL_TOL_BRIGHT["yellow"],
    50: PAUL_TOL_BRIGHT["green"],
}

_REAL_COLOR: str = PAUL_TOL_BRIGHT["blue"]

# Latent space constants
_LATENT_CHANNELS = 4
_LATENT_SPATIAL = 48
_LATENT_FLAT_DIM = _LATENT_CHANNELS * _LATENT_SPATIAL ** 3  # 442368

# Random projection target dimensions
_RANDOM_PROJ_DIM = 128
_PCA_DIM = 50


# =========================================================================
# HDF5 shard loading helpers
# =========================================================================


def _collect_latents_from_shards(
    latent_dir: Path,
    n_subsample: int,
    seed: int = 42,
) -> np.ndarray:
    """Load up to ``n_subsample`` latents from HDF5 shards.

    Iterates over ``shard_*.h5`` files in ``latent_dir``, collecting
    latents until the budget is reached. Shards are processed in
    sorted order for reproducibility.

    Args:
        latent_dir: Directory containing ``shard_000.h5``, etc. Each
            shard has dataset ``/latents`` of shape ``(N, 4, 48, 48, 48)``.
        n_subsample: Maximum number of latents to collect.
        seed: Random seed for subsampling within the last shard.

    Returns:
        Array of shape ``(n_collected, 4, 48, 48, 48)`` in float32.

    Raises:
        FileNotFoundError: If ``latent_dir`` does not exist.
        RuntimeError: If no shard files are found.
    """
    latent_dir = Path(latent_dir)
    if not latent_dir.exists():
        raise FileNotFoundError(f"Latent directory not found: {latent_dir}")

    shard_paths = sorted(latent_dir.glob("shard_*.h5"))
    if not shard_paths:
        raise RuntimeError(f"No shard_*.h5 files found in {latent_dir}")

    rng = np.random.RandomState(seed)
    collected: list[np.ndarray] = []
    remaining = n_subsample

    for shard_path in shard_paths:
        if remaining <= 0:
            break
        with h5py.File(shard_path, "r") as f:
            data = f["latents"]
            n_in_shard = data.shape[0]
            if n_in_shard <= remaining:
                collected.append(np.array(data[:], dtype=np.float32))
                remaining -= n_in_shard
            else:
                # Subsample from this shard
                indices = rng.choice(n_in_shard, size=remaining, replace=False)
                indices.sort()
                collected.append(np.array(data[indices], dtype=np.float32))
                remaining = 0

    if not collected:
        raise RuntimeError(f"Collected 0 latents from {latent_dir}")

    result = np.concatenate(collected, axis=0)
    logger.info(
        "Collected %d real latents from %d shards in %s",
        result.shape[0],
        len(shard_paths),
        latent_dir,
    )
    return result


def _load_generated_latents(
    h5_path: Path,
    n_subsample: int,
    seed: int = 42,
) -> np.ndarray:
    """Load up to ``n_subsample`` generated latents from a single HDF5 file.

    Args:
        h5_path: Path to HDF5 file with dataset ``/latents`` of shape
            ``(N, 4, 48, 48, 48)``.
        n_subsample: Maximum number of latents to load.
        seed: Random seed for subsampling.

    Returns:
        Array of shape ``(n_loaded, 4, 48, 48, 48)`` in float32.

    Raises:
        FileNotFoundError: If ``h5_path`` does not exist.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Generated latent file not found: {h5_path}")

    rng = np.random.RandomState(seed)
    with h5py.File(h5_path, "r") as f:
        data = f["latents"]
        n_total = data.shape[0]
        if n_total <= n_subsample:
            result = np.array(data[:], dtype=np.float32)
        else:
            indices = rng.choice(n_total, size=n_subsample, replace=False)
            indices.sort()
            result = np.array(data[indices], dtype=np.float32)

    logger.info("Loaded %d generated latents from %s", result.shape[0], h5_path)
    return result


# =========================================================================
# 1. Latent t-SNE Visualization
# =========================================================================


def plot_latent_tsne(
    real_latent_dir: Path,
    gen_latent_paths: dict[int, Path],
    output_dir: Path,
    n_subsample: int = 500,
) -> None:
    """t-SNE visualization of real vs generated latent distributions.

    Reduces 442,368-dimensional latent vectors to 2D via random projection
    (to 128D), PCA (to 50D), then t-SNE. Uses a fixed-seed random
    projection matrix for reproducibility.

    The scatter plot shows Real latents in blue alongside generated latents
    at each NFE level in distinct colors, revealing distributional overlap
    and mode coverage in the latent space.

    Args:
        real_latent_dir: Directory containing real latent HDF5 shards
            (``shard_000.h5``, etc.), each with ``/latents`` dataset.
        gen_latent_paths: Mapping from NFE value to the path of the
            generated latent HDF5 file (e.g.
            ``{1: Path(".../nfe_001.h5"), 10: Path(".../nfe_010.h5")}``).
        output_dir: Directory where figures and data are saved. Creates
            ``figures/`` and ``data/`` subdirectories.
        n_subsample: Number of latents to subsample per distribution.
            Controls memory usage and t-SNE runtime.
    """
    apply_ieee_style()
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load latents
    # ------------------------------------------------------------------
    logger.info("Loading real latents for t-SNE (n=%d)...", n_subsample)
    try:
        real_latents = _collect_latents_from_shards(real_latent_dir, n_subsample)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Cannot load real latents: %s", e)
        return

    gen_latents: dict[int, np.ndarray] = {}
    for nfe, path in sorted(gen_latent_paths.items()):
        try:
            gen_latents[nfe] = _load_generated_latents(path, n_subsample)
        except FileNotFoundError as e:
            logger.warning("Skipping NFE=%d: %s", nfe, e)

    if not gen_latents:
        logger.error("No generated latents loaded — aborting t-SNE plot")
        return

    # ------------------------------------------------------------------
    # Flatten and combine
    # ------------------------------------------------------------------
    n_real = real_latents.shape[0]
    real_flat = real_latents.reshape(n_real, -1)  # (N, 442368)

    gen_flat_by_nfe: dict[int, np.ndarray] = {}
    gen_counts: dict[int, int] = {}
    all_flat = [real_flat]
    for nfe in sorted(gen_latents.keys()):
        g = gen_latents[nfe]
        n_gen = g.shape[0]
        flat = g.reshape(n_gen, -1)
        gen_flat_by_nfe[nfe] = flat
        gen_counts[nfe] = n_gen
        all_flat.append(flat)

    combined = np.concatenate(all_flat, axis=0)  # (N_total, 442368)
    logger.info(
        "Combined %d samples (real=%d, gen=%s) for dimensionality reduction",
        combined.shape[0],
        n_real,
        {nfe: c for nfe, c in gen_counts.items()},
    )

    # ------------------------------------------------------------------
    # Dimensionality reduction: Random projection -> PCA -> t-SNE
    # ------------------------------------------------------------------
    logger.info("Random projection to %dD...", _RANDOM_PROJ_DIM)
    rng = np.random.RandomState(42)
    proj_matrix = rng.randn(_LATENT_FLAT_DIM, _RANDOM_PROJ_DIM).astype(np.float32)
    proj_matrix /= np.sqrt(_LATENT_FLAT_DIM)  # Normalize for JL lemma
    projected = combined @ proj_matrix  # (N_total, 128)

    logger.info("PCA to %dD...", _PCA_DIM)
    pca = PCA(n_components=_PCA_DIM, random_state=42)
    pca_result = pca.fit_transform(projected)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    logger.info("PCA variance explained: %.1f%%", var_explained)

    logger.info("t-SNE to 2D...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, combined.shape[0] // 4),
        random_state=42,
        n_iter=1000,
    )
    embedded = tsne.fit_transform(pca_result)  # (N_total, 2)
    logger.info("t-SNE complete (KL divergence: %.4f)", tsne.kl_divergence_)

    # ------------------------------------------------------------------
    # Split embeddings back
    # ------------------------------------------------------------------
    real_emb = embedded[:n_real]
    offset = n_real
    gen_emb: dict[int, np.ndarray] = {}
    for nfe in sorted(gen_counts.keys()):
        count = gen_counts[nfe]
        gen_emb[nfe] = embedded[offset : offset + count]
        offset += count

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=get_figure_size("single", height_ratio=1.0))

    # Real points (background, smaller)
    ax.scatter(
        real_emb[:, 0],
        real_emb[:, 1],
        c=_REAL_COLOR,
        s=PLOT_SETTINGS["scatter_size"],
        alpha=0.5,
        edgecolors="none",
        label=f"Real (n={n_real})",
        zorder=2,
        rasterized=True,
    )

    # Generated points (foreground, slightly larger)
    for nfe in sorted(gen_emb.keys()):
        color = _NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["grey"])
        count = gen_emb[nfe].shape[0]
        ax.scatter(
            gen_emb[nfe][:, 0],
            gen_emb[nfe][:, 1],
            c=color,
            s=PLOT_SETTINGS["scatter_size"] * 1.2,
            alpha=0.5,
            edgecolors="none",
            label=f"NFE={nfe} (n={count})",
            zorder=3,
            rasterized=True,
        )

    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title("Latent Space t-SNE")
    ax.legend(
        fontsize=PLOT_SETTINGS["legend_fontsize"],
        loc="best",
        markerscale=2,
        framealpha=0.8,
    )
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    npz_data: dict[str, np.ndarray] = {
        "real_tsne": real_emb,
        "pca_variance_explained_pct": np.array([var_explained]),
        "tsne_kl_divergence": np.array([tsne.kl_divergence_]),
    }
    for nfe, emb in gen_emb.items():
        npz_data[f"gen_nfe{nfe}_tsne"] = emb

    np.savez(output_dir / "data" / "latent_tsne.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "latent_tsne")
    logger.info("Saved latent_tsne figure and source data")


# =========================================================================
# 2. Per-Channel Latent Distribution Histograms
# =========================================================================


def plot_latent_channel_histograms(
    real_latent_dir: Path,
    gen_latent_paths: dict[int, Path],
    output_dir: Path,
    n_subsample: int = 500,
    n_bins: int = 80,
) -> None:
    """Per-channel histogram comparison of real vs generated latent means.

    Creates a 2x2 grid (one subplot per latent channel). For each channel,
    computes the spatial mean across the 48x48x48 volume to obtain a
    scalar per sample, then plots the distribution of these per-sample
    channel means.

    Shows whether generated latents match the per-channel distributional
    statistics of real data, which is critical for VAE decoding quality.

    Args:
        real_latent_dir: Directory containing real latent HDF5 shards.
        gen_latent_paths: Mapping from NFE value to generated latent HDF5 path.
        output_dir: Directory where figures and data are saved.
        n_subsample: Number of latents to subsample per distribution.
        n_bins: Number of histogram bins.
    """
    apply_ieee_style()
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load latents
    # ------------------------------------------------------------------
    logger.info("Loading real latents for channel histograms (n=%d)...", n_subsample)
    try:
        real_latents = _collect_latents_from_shards(real_latent_dir, n_subsample)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error("Cannot load real latents: %s", e)
        return

    gen_latents: dict[int, np.ndarray] = {}
    for nfe, path in sorted(gen_latent_paths.items()):
        try:
            gen_latents[nfe] = _load_generated_latents(path, n_subsample)
        except FileNotFoundError as e:
            logger.warning("Skipping NFE=%d: %s", nfe, e)

    if not gen_latents:
        logger.error("No generated latents loaded — aborting histogram plot")
        return

    # ------------------------------------------------------------------
    # Compute per-sample channel means: (N, 4, 48, 48, 48) -> (N, 4)
    # ------------------------------------------------------------------
    real_ch_means = real_latents.mean(axis=(2, 3, 4))  # (N, 4)
    gen_ch_means: dict[int, np.ndarray] = {}
    for nfe, g in gen_latents.items():
        gen_ch_means[nfe] = g.mean(axis=(2, 3, 4))  # (N, 4)

    # ------------------------------------------------------------------
    # Plot: 2x2 grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=get_figure_size("double", height_ratio=0.7))
    axes_flat = axes.flatten()

    npz_data: dict[str, np.ndarray] = {}

    for ch in range(_LATENT_CHANNELS):
        ax = axes_flat[ch]

        # Real distribution (filled histogram)
        real_vals = real_ch_means[:, ch]
        npz_data[f"real_ch{ch}_means"] = real_vals

        ax.hist(
            real_vals,
            bins=n_bins,
            density=True,
            alpha=0.4,
            color=_REAL_COLOR,
            edgecolor=_REAL_COLOR,
            linewidth=0.5,
            label=f"Real (n={len(real_vals)})",
            zorder=2,
        )

        # Generated distributions (line histograms for clarity)
        for nfe in sorted(gen_ch_means.keys()):
            gen_vals = gen_ch_means[nfe][:, ch]
            npz_data[f"gen_nfe{nfe}_ch{ch}_means"] = gen_vals
            color = _NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["grey"])

            counts, bin_edges = np.histogram(gen_vals, bins=n_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.plot(
                bin_centers,
                counts,
                color=color,
                linewidth=PLOT_SETTINGS["line_width"],
                label=f"NFE={nfe} (n={len(gen_vals)})",
                zorder=3,
            )

        # Annotation: real mean +/- std
        mu_real = real_vals.mean()
        std_real = real_vals.std()
        ax.annotate(
            f"Real: $\\mu$={mu_real:.3f}, $\\sigma$={std_real:.3f}",
            xy=(0.97, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(f"Channel {ch} mean value")
        ax.set_ylabel("Density")
        ax.set_title(f"({chr(97 + ch)}) Channel {ch}")
        ax.grid(True, alpha=0.3)

        if ch == 0:
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Latent Per-Channel Distribution: Real vs Generated",
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        y=1.01,
    )
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.savez(output_dir / "data" / "latent_channel_histograms.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "latent_channel_histograms")
    logger.info("Saved latent_channel_histograms figure and source data")


# =========================================================================
# 3. Generation Trajectory Visualization
# =========================================================================


def plot_generation_trajectory(
    trajectory_latents: list[torch.Tensor | np.ndarray],
    trajectory_times: list[float],
    output_dir: Path,
    vae_decode_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    """Visualize the generation trajectory from noise to data.

    Shows how the latent (or decoded pixel) representation evolves across
    time steps during multi-step generation. For 1-NFE, only two columns
    are shown (noise at t=1 and output at t=0).

    If ``vae_decode_fn`` is provided, decodes each latent to pixel space
    and displays axial, coronal, and sagittal center slices (3 rows).
    Otherwise, displays the raw latent channels directly (4 rows, one per
    channel) as 2D center slices from the 48^3 spatial volume.

    Args:
        trajectory_latents: List of latent tensors, each of shape
            ``(4, 48, 48, 48)``. Ordered from noise (t=1) to data (t=0).
        trajectory_times: List of float time values corresponding to each
            latent (e.g. ``[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]``).
        output_dir: Directory where figures and data are saved.
        vae_decode_fn: Optional callable that takes a latent tensor of
            shape ``(1, 4, 48, 48, 48)`` and returns a volume tensor of
            shape ``(1, 1, 192, 192, 192)``. If None, raw latent channels
            are displayed.
    """
    apply_ieee_style()
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    n_steps = len(trajectory_latents)
    if n_steps == 0:
        logger.error("Empty trajectory — nothing to plot")
        return

    if len(trajectory_times) != n_steps:
        logger.error(
            "Mismatch: %d latents but %d time values",
            n_steps,
            len(trajectory_times),
        )
        return

    npz_data: dict[str, np.ndarray] = {
        "times": np.array(trajectory_times, dtype=np.float32),
    }

    decoded_mode = vae_decode_fn is not None

    if decoded_mode:
        # ----- Decoded pixel-space visualization (3 rows: axial/coronal/sagittal) -----
        n_rows = 3
        row_labels = ["Axial", "Coronal", "Sagittal"]

        fig, axes = plt.subplots(
            n_rows,
            n_steps,
            figsize=(min(n_steps * 1.8, get_figure_size("double")[0]), 4.5),
        )
        if n_steps == 1:
            axes = axes[:, np.newaxis]

        for step_idx, (latent, t_val) in enumerate(
            zip(trajectory_latents, trajectory_times)
        ):
            # Ensure tensor format
            if isinstance(latent, np.ndarray):
                latent = torch.from_numpy(latent)
            latent = latent.float()

            # Decode: (4, 48, 48, 48) -> (1, 1, 192, 192, 192) -> (192, 192, 192)
            logger.info("Decoding latent at t=%.2f...", t_val)
            with torch.no_grad():
                volume = vae_decode_fn(latent.unsqueeze(0))
            vol_np = volume.squeeze().cpu().numpy()  # (192, 192, 192)
            npz_data[f"decoded_t{t_val:.2f}"] = vol_np

            # Center slices
            d, h, w = vol_np.shape
            slices = [
                vol_np[d // 2, :, :],  # Axial
                vol_np[:, h // 2, :],  # Coronal
                vol_np[:, :, w // 2],  # Sagittal
            ]

            for row_idx, sl in enumerate(slices):
                ax = axes[row_idx, step_idx]
                # Robust vmin/vmax from non-zero region
                fg = sl[sl > 0]
                if fg.size > 0:
                    vmin, vmax = np.percentile(fg, [1, 99])
                else:
                    vmin, vmax = sl.min(), sl.max()

                ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
                ax.axis("off")

                if step_idx == 0:
                    ax.set_ylabel(
                        row_labels[row_idx],
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        rotation=90,
                        labelpad=10,
                    )

            # Column title
            if t_val >= 0.99:
                title_text = f"$t = {t_val:.1f}$ (noise)"
            elif t_val <= 0.01:
                title_text = f"$t = {t_val:.1f}$ (data)"
            else:
                title_text = f"$t = {t_val:.1f}$"
            axes[0, step_idx].set_title(title_text, fontsize=PLOT_SETTINGS["tick_labelsize"])

    else:
        # ----- Raw latent channel visualization (4 rows: Ch 0-3) -----
        n_rows = _LATENT_CHANNELS
        row_labels = [f"Ch {ch}" for ch in range(_LATENT_CHANNELS)]

        fig, axes = plt.subplots(
            n_rows,
            n_steps,
            figsize=(min(n_steps * 1.8, get_figure_size("double")[0]), 5.0),
        )
        if n_steps == 1:
            axes = axes[:, np.newaxis]

        # Compute global vmin/vmax per channel across all time steps for
        # consistent colormaps
        channel_ranges: list[tuple[float, float]] = []
        all_latent_np: list[np.ndarray] = []
        for latent in trajectory_latents:
            if isinstance(latent, torch.Tensor):
                lat_np = latent.cpu().float().numpy()
            else:
                lat_np = np.asarray(latent, dtype=np.float32)
            all_latent_np.append(lat_np)

        for ch in range(_LATENT_CHANNELS):
            all_ch_vals = np.concatenate(
                [lat[ch].ravel() for lat in all_latent_np]
            )
            vmin = float(np.percentile(all_ch_vals, 1))
            vmax = float(np.percentile(all_ch_vals, 99))
            channel_ranges.append((vmin, vmax))

        for step_idx, (lat_np, t_val) in enumerate(
            zip(all_latent_np, trajectory_times)
        ):
            npz_data[f"latent_t{t_val:.2f}"] = lat_np

            for ch in range(_LATENT_CHANNELS):
                ax = axes[ch, step_idx]
                # Show center axial slice of this channel: (48, 48, 48) -> (48, 48)
                ch_vol = lat_np[ch]
                center_slice = ch_vol[ch_vol.shape[0] // 2, :, :]

                vmin, vmax = channel_ranges[ch]
                ax.imshow(
                    center_slice.T,
                    cmap="RdBu_r",
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )
                ax.axis("off")

                if step_idx == 0:
                    ax.set_ylabel(
                        row_labels[ch],
                        fontsize=PLOT_SETTINGS["tick_labelsize"],
                        rotation=90,
                        labelpad=10,
                    )

            # Column title
            if t_val >= 0.99:
                title_text = f"$t = {t_val:.1f}$ (noise)"
            elif t_val <= 0.01:
                title_text = f"$t = {t_val:.1f}$ (data)"
            else:
                title_text = f"$t = {t_val:.1f}$"
            axes[0, step_idx].set_title(title_text, fontsize=PLOT_SETTINGS["tick_labelsize"])

    fig.suptitle(
        "Generation Trajectory" + (" (Decoded)" if decoded_mode else " (Latent Space)"),
        fontsize=PLOT_SETTINGS["axes_titlesize"],
        y=1.01,
    )
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    np.savez(output_dir / "data" / "generation_trajectory.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "generation_trajectory")
    logger.info("Saved generation_trajectory figure and source data")
