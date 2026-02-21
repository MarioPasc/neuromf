"""Publication-quality sample evolution plots from training archives.

Three diagnostic figures for analysing latent MeanFlow training dynamics
without requiring VAE decoding:

1. **Sample & Channel Evolution Grid** — axial mid-slice visualisation
   across training epochs, showing spatial structure emergence.
2. **Per-Channel Statistics Panel** — mean/std/skewness/kurtosis trajectories
   with healthy-range bands.
3. **Radially-Averaged Power Spectrum** — 3D FFT spectral analysis revealing
   coarse-to-fine learning progression.

All functions accept a ``sample_archive.pt`` dict and output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from numpy.typing import NDArray

from neuromf.utils.latent_diagnostics import compute_latent_stats
from neuromf.utils.visualisation import (
    COLORBLIND_PALETTE,
    _apply_style,
    save_figure,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Maximum columns in the evolution grid before subsampling
_MAX_GRID_COLS = 12


def _subsample_epochs(epochs: list[int], max_cols: int) -> list[int]:
    """Evenly subsample epoch list to at most ``max_cols`` entries.

    Always includes the first and last epoch.

    Args:
        epochs: Sorted list of epoch numbers.
        max_cols: Maximum number of epochs to keep.

    Returns:
        Subsampled sorted list of epochs.
    """
    if len(epochs) <= max_cols:
        return list(epochs)
    indices = np.linspace(0, len(epochs) - 1, max_cols, dtype=int)
    return [epochs[i] for i in np.unique(indices)]


def _radial_average_3d(volume: NDArray) -> tuple[NDArray, NDArray]:
    """Compute radially-averaged 3D power spectral density.

    Takes a real-valued 3D volume, computes ``|FFT|^2``, then bins voxels
    by their Euclidean distance from the DC component in frequency space.

    Args:
        volume: Real-valued 3D array of shape ``(D, H, W)``.

    Returns:
        Tuple of ``(frequencies, psd)`` where ``frequencies`` are in
        cycles/voxel (0 to Nyquist) and ``psd`` is the radially-averaged
        power spectral density at each frequency bin.
    """
    D, H, W = volume.shape
    fft_vol = np.fft.fftn(volume)
    power = np.abs(fft_vol) ** 2

    # Frequency coordinates (centred so DC is at origin)
    kd = np.fft.fftfreq(D)
    kh = np.fft.fftfreq(H)
    kw = np.fft.fftfreq(W)
    kd_grid, kh_grid, kw_grid = np.meshgrid(kd, kh, kw, indexing="ij")
    radius = np.sqrt(kd_grid**2 + kh_grid**2 + kw_grid**2)

    # Bin by radius
    nyquist = 0.5
    n_bins = min(D, H, W) // 2
    bin_edges = np.linspace(0, nyquist, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    psd = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.any():
            psd[i] = power[mask].mean()

    return bin_centres, psd


def plot_sample_evolution_grid(
    archive: dict,
    output_dir: Path,
) -> None:
    """Axial mid-slice grid: sample energy + per-channel evolution.

    5 rows (energy + 4 channels) x N columns (one per logged epoch).
    Energy row uses grayscale; channel rows use diverging RdBu_r with a
    shared full-width colorbar at the bottom.

    Args:
        archive: Loaded ``sample_archive.pt`` dict.
        output_dir: Directory to save the figure.
    """
    import torch

    _apply_style()

    epochs = _subsample_epochs(archive["epochs"], _MAX_GRID_COLS)
    n_cols = len(epochs)
    if n_cols == 0:
        logger.warning("No epochs in archive; skipping evolution grid.")
        return

    # Collect mid-slice data: (n_epochs, 4, 48, 48) for channels
    slices_ch: list[NDArray] = []  # per-epoch list of (4, H, W)
    slices_energy: list[NDArray] = []  # per-epoch list of (H, W)

    for ep in epochs:
        epoch_key = f"epoch_{ep:04d}"
        z = archive[epoch_key]["nfe_1"]  # (N, 4, D, H, W)
        if isinstance(z, torch.Tensor):
            z = z.numpy()
        z0 = z[0]  # sample #0: (4, D, H, W)
        mid = z0.shape[1] // 2  # axial mid-slice index
        ch_slices = z0[:, mid, :, :]  # (4, H, W)
        slices_ch.append(ch_slices)

        energy = np.sqrt((z0[:, mid, :, :] ** 2).sum(axis=0))  # (H, W)
        slices_energy.append(energy)

    # Compute shared vmin/vmax for channel rows (symmetric diverging)
    all_ch = np.stack(slices_ch)  # (n_epochs, 4, H, W)
    ch_absmax = float(np.percentile(np.abs(all_ch), 99.5))
    ch_vmin, ch_vmax = -ch_absmax, ch_absmax

    # Layout: 5 rows + colorbar row
    n_rows = 5
    fig_width = max(1.3 * n_cols + 1.2, 4.0)
    fig_height = 1.3 * n_rows + 0.6

    fig = plt.figure(figsize=(fig_width, fig_height))
    # Height ratios: 5 image rows + thin colorbar row
    gs = gridspec.GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1] * n_rows + [0.05],
        hspace=0.15,
        wspace=0.05,
    )

    row_labels = ["Sample", "Ch 0", "Ch 1", "Ch 2", "Ch 3"]

    for col_idx, ep in enumerate(epochs):
        # Row 0: energy (grayscale)
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(
            slices_energy[col_idx].T,
            cmap="gray",
            origin="lower",
            aspect="equal",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Ep {ep}", fontsize=7, pad=2)
        if col_idx == 0:
            ax.set_ylabel(row_labels[0], fontsize=8)

        # Rows 1-4: per-channel (RdBu_r, shared normalization)
        for ch in range(4):
            ax = fig.add_subplot(gs[ch + 1, col_idx])
            im = ax.imshow(
                slices_ch[col_idx][ch].T,
                cmap="RdBu_r",
                origin="lower",
                vmin=ch_vmin,
                vmax=ch_vmax,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_labels[ch + 1], fontsize=8)

    # Full-width colorbar at bottom (spans all columns)
    cbar_ax = fig.add_subplot(gs[n_rows, :])
    fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        label="Latent value",
    )

    save_figure(fig, output_dir / "sample_evolution_grid")
    logger.info("Saved sample evolution grid (%d epochs)", n_cols)


def plot_channel_stats_evolution(
    archive: dict,
    output_dir: Path,
) -> None:
    """2x2 panel of per-channel mean/std/skewness/kurtosis over epochs.

    Each subplot shows 4 lines (one per channel) with a dashed reference
    at the ideal value and a shaded healthy band.

    Args:
        archive: Loaded ``sample_archive.pt`` dict.
        output_dir: Directory to save the figure.
    """
    import torch

    _apply_style()

    epochs = sorted(archive["epochs"])
    if not epochs:
        logger.warning("No epochs in archive; skipping stats evolution.")
        return

    # Compute stats from raw tensors for each epoch
    n_epochs = len(epochs)
    means = np.zeros((n_epochs, 4))
    stds = np.zeros((n_epochs, 4))
    skews = np.zeros((n_epochs, 4))
    kurts = np.zeros((n_epochs, 4))

    for i, ep in enumerate(epochs):
        epoch_key = f"epoch_{ep:04d}"
        z = archive[epoch_key]["nfe_1"]
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        stats = compute_latent_stats(z)
        means[i] = stats["mean"].numpy()
        stds[i] = stats["std"].numpy()
        skews[i] = stats["skewness"].numpy()
        kurts[i] = stats["kurtosis"].numpy()

    # Panel config: (data, ylabel, ideal, band_lo, band_hi)
    panels = [
        (means, "Mean", 0.0, -0.1, 0.1),
        (stds, "Std", 1.0, 0.8, 1.2),
        (skews, "Skewness", 0.0, -0.3, 0.3),
        (kurts, "Excess Kurtosis", 0.0, -0.5, 0.5),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes_flat = axes.flatten()

    for ax, (data, ylabel, ideal, band_lo, band_hi) in zip(axes_flat, panels):
        # Healthy band
        ax.axhspan(band_lo, band_hi, color="lightgray", alpha=0.4, zorder=0)
        # Reference line
        ax.axhline(ideal, color="gray", linestyle="--", linewidth=0.8, zorder=1)

        for ch in range(4):
            ax.plot(
                epochs,
                data[:, ch],
                color=COLORBLIND_PALETTE[ch],
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=f"Ch {ch}",
            )

        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")

    axes_flat[2].set_xlabel("Epoch")
    axes_flat[3].set_xlabel("Epoch")

    fig.suptitle("Per-Channel Latent Statistics (1-NFE)", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_dir / "channel_stats_evolution")
    logger.info("Saved channel stats evolution (%d epochs)", n_epochs)


def plot_spectral_evolution(
    archive: dict,
    output_dir: Path,
) -> None:
    """2x2 panel of radially-averaged 3D power spectra over epochs.

    One subplot per latent channel. Lines coloured by epoch using a
    sequential colormap, with a dashed white-noise reference.

    Args:
        archive: Loaded ``sample_archive.pt`` dict.
        output_dir: Directory to save the figure.
    """
    import torch

    _apply_style()

    epochs = _subsample_epochs(archive["epochs"], _MAX_GRID_COLS)
    if not epochs:
        logger.warning("No epochs in archive; skipping spectral evolution.")
        return

    # Compute noise reference spectrum (flat)
    noise = archive["noise"]
    if isinstance(noise, torch.Tensor):
        noise = noise.numpy()
    noise_0 = noise[0]  # (4, D, H, W) — sample #0

    # Collect per-channel PSD for each epoch
    # Shape: spectra[ch] = list of (freqs, psd) per epoch
    spectra: dict[int, list[tuple[NDArray, NDArray]]] = {ch: [] for ch in range(4)}
    noise_spectra: dict[int, tuple[NDArray, NDArray]] = {}

    for ch in range(4):
        noise_spectra[ch] = _radial_average_3d(noise_0[ch])

    for ep in epochs:
        epoch_key = f"epoch_{ep:04d}"
        z = archive[epoch_key]["nfe_1"]
        if isinstance(z, torch.Tensor):
            z = z.numpy()
        z0 = z[0]  # sample #0: (4, D, H, W)
        for ch in range(4):
            spectra[ch].append(_radial_average_3d(z0[ch]))

    # Epoch-to-colour mapping
    cmap = plt.cm.viridis
    if len(epochs) > 1:
        epoch_min, epoch_max = epochs[0], epochs[-1]
        norm = plt.Normalize(vmin=epoch_min, vmax=epoch_max)
    else:
        norm = plt.Normalize(vmin=epochs[0], vmax=epochs[0] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for ch in range(4):
        ax = axes_flat[ch]

        # Epoch lines
        for idx, ep in enumerate(epochs):
            freqs, psd = spectra[ch][idx]
            # Avoid log10(0) by clamping
            psd_safe = np.clip(psd, 1e-30, None)
            ax.plot(
                freqs,
                np.log10(psd_safe),
                color=cmap(norm(ep)),
                linewidth=0.8,
                alpha=0.85,
            )

        # Noise reference
        n_freqs, n_psd = noise_spectra[ch]
        n_psd_safe = np.clip(n_psd, 1e-30, None)
        ax.plot(
            n_freqs,
            np.log10(n_psd_safe),
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="Noise",
        )

        ax.set_title(f"Ch {ch}", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")

    axes_flat[2].set_xlabel("Spatial frequency (cycles/voxel)")
    axes_flat[3].set_xlabel("Spatial frequency (cycles/voxel)")
    axes_flat[0].set_ylabel("log$_{10}$ PSD")
    axes_flat[2].set_ylabel("log$_{10}$ PSD")

    # Colorbar for epoch mapping
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat.tolist(), label="Epoch", shrink=0.8, pad=0.02)

    fig.suptitle("Radially-Averaged Power Spectrum (1-NFE, sample #0)", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    save_figure(fig, output_dir / "spectral_evolution")
    logger.info("Saved spectral evolution (%d epochs, 4 channels)", len(epochs))


def plot_nfe_comparison_grid(
    archive: dict,
    output_dir: Path,
    decoded_volumes: dict[int, NDArray] | None = None,
) -> None:
    """NFE comparison grid: rows=NFE levels, columns=training epochs.

    Each cell shows the axial mid-slice of sample #0, channel 0.
    Visualises convergence of 1-NFE toward multi-step generation.

    Args:
        archive: Loaded ``sample_archive.pt`` dict.
        output_dir: Directory to save the figure.
        decoded_volumes: Optional mapping from epoch number to decoded
            pixel-space volume ``(D, H, W)`` from 1-NFE sample #0.
            When provided, adds a "Decoded" row at the bottom with
            grayscale MRI axial mid-slices.
    """
    import torch

    _apply_style()

    nfe_levels = [1, 2, 5, 10]
    nfe_keys = [f"nfe_{n}" for n in nfe_levels]
    epochs = _subsample_epochs(archive["epochs"], _MAX_GRID_COLS)
    if not epochs:
        logger.warning("No epochs in archive; skipping NFE comparison grid.")
        return

    # Check which NFE keys actually exist in the first epoch
    first_key = f"epoch_{epochs[0]:04d}"
    available_nfe = [k for k in nfe_keys if k in archive[first_key]]
    available_levels = [int(k.split("_")[1]) for k in available_nfe]
    if not available_nfe:
        logger.warning("No NFE data found; skipping NFE comparison grid.")
        return

    has_decoded = decoded_volumes is not None and len(decoded_volumes) > 0
    n_latent_rows = len(available_nfe)
    n_rows = n_latent_rows + (1 if has_decoded else 0)
    n_cols = len(epochs)

    # Collect all slices for shared colormap
    all_slices: list[NDArray] = []
    slice_grid: list[list[NDArray]] = []  # [row][col]

    for nfe_key in available_nfe:
        row_slices: list[NDArray] = []
        for ep in epochs:
            epoch_key = f"epoch_{ep:04d}"
            z = archive[epoch_key][nfe_key]
            if isinstance(z, torch.Tensor):
                z = z.numpy()
            sl = z[0, 0, z.shape[2] // 2, :, :]  # sample #0, ch 0, axial mid
            row_slices.append(sl)
            all_slices.append(sl)
        slice_grid.append(row_slices)

    # Shared symmetric colormap for latent rows
    absmax = float(np.percentile(np.abs(np.stack(all_slices)), 99.5))
    vmin, vmax = -absmax, absmax

    fig_width = max(1.3 * n_cols + 1.2, 4.0)
    fig_height = 1.3 * n_rows + 0.6

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        n_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1] * n_rows + [0.05],
        hspace=0.15,
        wspace=0.05,
    )

    im = None
    for row_idx, (nfe_level, nfe_key) in enumerate(zip(available_levels, available_nfe)):
        for col_idx, ep in enumerate(epochs):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            im = ax.imshow(
                slice_grid[row_idx][col_idx].T,
                cmap="RdBu_r",
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"Ep {ep}", fontsize=7, pad=2)
            if col_idx == 0:
                ax.set_ylabel(f"NFE={nfe_level}", fontsize=8)

    # Decoded row (grayscale MRI mid-slices)
    if has_decoded:
        assert decoded_volumes is not None
        decoded_row = n_latent_rows
        for col_idx, ep in enumerate(epochs):
            ax = fig.add_subplot(gs[decoded_row, col_idx])
            if ep in decoded_volumes:
                vol = decoded_volumes[ep]
                mid_slice = vol[vol.shape[0] // 2, :, :]  # axial mid
                ax.imshow(mid_slice.T, cmap="gray", origin="lower", aspect="equal")
            else:
                ax.set_facecolor("white")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel("Decoded", fontsize=8)

    if im is not None:
        cbar_ax = fig.add_subplot(gs[n_rows, :])
        fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Latent value (ch 0)")

    save_figure(fig, output_dir / "nfe_comparison_grid")
    logger.info("Saved NFE comparison grid (%d rows x %d epochs)", n_rows, n_cols)


def plot_decoded_nfe_grid(
    decoded_samples: dict[int, NDArray],
    output_dir: Path,
) -> None:
    """Decoded MRI comparison across NFE levels (last epoch).

    Layout: N rows (one per NFE level) x 3 columns (Axial, Coronal, Sagittal).
    Each cell shows a grayscale MRI mid-slice from sample #0.

    Args:
        decoded_samples: Mapping from NFE level (e.g. 1, 2, 5, 10) to
            decoded pixel-space volume ``(D, H, W)``.
        output_dir: Directory to save the figure.
    """
    _apply_style()

    if not decoded_samples:
        logger.warning("No decoded samples; skipping decoded NFE grid.")
        return

    nfe_levels = sorted(decoded_samples.keys())
    n_rows = len(nfe_levels)
    view_names = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(9, 3 * n_rows),
        squeeze=False,
    )

    for row_idx, nfe in enumerate(nfe_levels):
        vol = decoded_samples[nfe]  # (D, H, W)
        mid = [s // 2 for s in vol.shape]
        slices = [
            vol[mid[0], :, :],  # Axial
            vol[:, mid[1], :],  # Coronal
            vol[:, :, mid[2]],  # Sagittal
        ]

        # Shared intensity range per row
        vmin_row = float(vol.min())
        vmax_row = float(vol.max())

        for col_idx, (sl, name) in enumerate(zip(slices, view_names)):
            ax = axes[row_idx, col_idx]
            ax.imshow(
                sl.T,
                cmap="gray",
                origin="lower",
                vmin=vmin_row,
                vmax=vmax_row,
                aspect="equal",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(name, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"NFE={nfe}", fontsize=9)

    fig.suptitle("Decoded MRI Samples by NFE Level", fontsize=12, y=1.01)
    fig.tight_layout()
    save_figure(fig, output_dir / "decoded_nfe_grid")
    logger.info("Saved decoded NFE grid (%d NFE levels)", n_rows)


def plot_nfe_consistency_evolution(
    archive: dict,
    output_dir: Path,
) -> None:
    """1x2 panel of NFE consistency metrics over training epochs.

    (a) MSE between 1-NFE and multi-step (log scale).
    (b) Cosine similarity between 1-NFE and multi-step.

    Args:
        archive: Loaded ``sample_archive.pt`` dict.
        output_dir: Directory to save the figure.
    """
    _apply_style()

    epochs = sorted(archive["epochs"])
    if not epochs:
        logger.warning("No epochs in archive; skipping NFE consistency evolution.")
        return

    # Collect consistency data per epoch
    epoch_list: list[int] = []
    mse_data: dict[str, list[float]] = {}
    cosine_data: dict[str, list[float]] = {}

    for ep in epochs:
        epoch_key = f"epoch_{ep:04d}"
        nfe_con = archive[epoch_key].get("nfe_consistency", {})
        if not nfe_con:
            continue
        epoch_list.append(ep)
        for key, val in nfe_con.items():
            if key.startswith("mse_"):
                mse_data.setdefault(key, []).append(float(val))
            elif key.startswith("cosine_"):
                cosine_data.setdefault(key, []).append(float(val))

    if not epoch_list:
        logger.warning("No NFE consistency data; skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) MSE (log scale)
    for i, (key, vals) in enumerate(sorted(mse_data.items())):
        label = key.replace("mse_", "").replace("vs", " vs ")
        axes[0].plot(
            epoch_list,
            vals,
            color=COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (log)")
    axes[0].set_title("1-NFE vs Multi-Step MSE")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # (b) Cosine similarity
    for i, (key, vals) in enumerate(sorted(cosine_data.items())):
        label = key.replace("cosine_", "").replace("vs", " vs ")
        axes[1].plot(
            epoch_list,
            vals,
            color=COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=label,
        )
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("1-NFE vs Multi-Step Cosine Similarity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / "nfe_consistency_evolution")
    logger.info("Saved NFE consistency evolution (%d epochs)", len(epoch_list))


def plot_inter_epoch_delta(
    archive: dict,
    output_dir: Path,
) -> None:
    """1x2 panel of inter-epoch sample deltas (stability indicator).

    Compares consecutive-epoch 1-NFE samples generated from the same noise.
    (a) L2 distance between consecutive epochs.
    (b) Cosine similarity between consecutive epochs.

    Args:
        archive: Loaded ``sample_archive.pt`` dict.
        output_dir: Directory to save the figure.
    """
    import torch

    _apply_style()

    epochs = sorted(archive["epochs"])
    if len(epochs) < 2:
        logger.warning("Need >= 2 epochs for inter-epoch delta; skipping.")
        return

    mid_epochs: list[int] = []
    l2_distances: list[float] = []
    cosine_sims: list[float] = []

    prev_z = None
    for ep in epochs:
        epoch_key = f"epoch_{ep:04d}"
        z = archive[epoch_key]["nfe_1"]
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z_flat = z.float().reshape(z.shape[0], -1)  # (N, D)

        if prev_z is not None:
            # Per-sample L2 distance, averaged
            l2 = (z_flat - prev_z).norm(dim=1).mean().item()
            l2_distances.append(l2)

            # Per-sample cosine similarity, averaged
            cos = torch.nn.functional.cosine_similarity(z_flat, prev_z, dim=1).mean().item()
            cosine_sims.append(cos)

            mid_epochs.append(ep)

        prev_z = z_flat.clone()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) L2 distance
    axes[0].plot(
        mid_epochs,
        l2_distances,
        color=COLORBLIND_PALETTE[0],
        marker="o",
        markersize=4,
        linewidth=1.2,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L2 Distance")
    axes[0].set_title("Inter-Epoch Sample Delta (L2)")
    axes[0].grid(True, alpha=0.3)

    # (b) Cosine similarity
    axes[1].plot(
        mid_epochs,
        cosine_sims,
        color=COLORBLIND_PALETTE[2],
        marker="o",
        markersize=4,
        linewidth=1.2,
    )
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Inter-Epoch Sample Stability (Cosine)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir / "inter_epoch_delta")
    logger.info("Saved inter-epoch delta (%d transitions)", len(mid_epochs))
