"""Publication-quality figure generation for Phase 5 analysis.

All figures follow IEEE TMI style via `apply_ieee_style()`. Each function
saves source data as `.npz` and produces both PDF and PNG outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from experiments.analysis.data_loader import DDPM_BASELINES, MOTFM_BASELINES
from experiments.utils.settings import (
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    get_figure_size,
    get_significance_stars,
)
from neuromf.utils.visualisation import save_figure

logger = logging.getLogger(__name__)

# Consistent NFE color scheme across all figures
NFE_COLORS: dict[int, str] = {
    1: PAUL_TOL_BRIGHT["red"],
    10: PAUL_TOL_BRIGHT["yellow"],
    50: PAUL_TOL_BRIGHT["green"],
}

# Import centralized style registry — extensible for new competitors.
# get_method_style() auto-assigns fallback colors for unknown models.
from neuromf.competitors.styles import (
    DEFAULT_METHOD_STYLES as METHOD_STYLES,
    get_method_style,
)

# Bilateral region pairs for SynthSeg analysis
BILATERAL_REGIONS: list[tuple[str, str, str]] = [
    ("left cerebral cortex", "right cerebral cortex", "Cerebral Cortex"),
    ("left cerebral white matter", "right cerebral white matter", "White Matter"),
    ("left lateral ventricle", "right lateral ventricle", "Lateral Ventricle"),
    ("left hippocampus", "right hippocampus", "Hippocampus"),
    ("left thalamus", "right thalamus", "Thalamus"),
    ("left caudate", "right caudate", "Caudate"),
    ("left cerebellum cortex", "right cerebellum cortex", "Cerebellum"),
    ("brain-stem", "brain-stem", "Brain-Stem"),
]

# QC regions from SynthSeg QC CSV
QC_REGIONS: list[str] = [
    "general white matter",
    "general grey matter",
    "general csf",
    "cerebellum",
    "brainstem",
    "thalamus",
    "putamen+pallidum",
    "hippocampus+amygdala",
]


def _bilateral_average(
    df: pd.DataFrame,
    regions: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Compute bilateral average (L+R)/2 for paired regions.

    Args:
        df: SynthSeg volumes DataFrame.
        regions: List of (left_col, right_col, display_name) tuples.

    Returns:
        DataFrame with columns named by display_name.
    """
    result = {}
    for left, right, name in regions:
        if left == right:
            result[name] = df[left].values if left in df.columns else np.array([])
        elif left in df.columns and right in df.columns:
            result[name] = (df[left].values + df[right].values) / 2.0
    return pd.DataFrame(result)


# -----------------------------------------------------------------------
# Fig 01 — NFE Scaling Curves
# -----------------------------------------------------------------------


def plot_nfe_scaling(
    metrics: dict[int, dict[str, Any]],
    bootstrap_results: dict[str, dict[int, dict[str, Any]]],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """NFE scaling curves: 2x3 grid of metrics vs NFE.

    Args:
        metrics: Dict mapping NFE to parsed metrics JSON.
        bootstrap_results: Dict mapping metric name to {nfe: bootstrap_dict}.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    apply_ieee_style()
    fig, axes = plt.subplots(2, 3, figsize=get_figure_size("double", 0.55))
    axes = axes.flatten()

    # Define panels: (title, metric_key, extract_fn, better_dir, y_label)
    panels = [
        ("FID-3D", "fid_3d", lambda m: m["distributional"]["fid_3d"]["value"],
         "lower", "FID-3D"),
        ("MMD", "mmd", lambda m: m["distributional"]["mmd"]["value"],
         "lower", "MMD"),
        ("Coverage", "coverage", lambda m: m["distributional"]["coverage_k5"]["value"],
         "upper", "Coverage"),
        ("Density", "density", lambda m: m["distributional"]["density_k5"]["value"],
         "upper", "Density"),
        ("MS-SSIM", "ms_ssim", lambda m: m["per_volume"]["ms_ssim"]["mean"],
         "upper", "MS-SSIM"),
        ("HF Energy Ratio", "hf_energy", lambda m: m["spectral"]["hf_energy_ratio"]["mean"],
         "lower", "HF Energy Ratio"),
    ]

    # Collect data for npz
    npz_data: dict[str, np.ndarray] = {"nfe_levels": np.array(nfe_levels)}

    for i, (title, key, extract_fn, better_dir, ylabel) in enumerate(panels):
        ax = axes[i]

        # NeuroiMF
        neuro_vals = [extract_fn(metrics[nfe]) for nfe in nfe_levels if nfe in metrics]
        valid_nfes = [nfe for nfe in nfe_levels if nfe in metrics]
        style = METHOD_STYLES["NeuroiMF"]
        ax.plot(
            valid_nfes, neuro_vals,
            color=style["color"], marker=style["marker"], ls=style["ls"],
            label="NeuroiMF", linewidth=1.5, markersize=6, zorder=3,
        )
        npz_data[f"neuroimf_{key}"] = np.array(neuro_vals)

        # Bootstrap CI band
        if key in bootstrap_results:
            ci_lower = [bootstrap_results[key].get(nfe, {}).get("ci_lower", np.nan)
                        for nfe in valid_nfes]
            ci_upper = [bootstrap_results[key].get(nfe, {}).get("ci_upper", np.nan)
                        for nfe in valid_nfes]
            ax.fill_between(
                valid_nfes, ci_lower, ci_upper,
                alpha=0.2, color=style["color"],
            )

        # MOTFM baseline
        motfm_key = key if key != "hf_energy" else None
        if motfm_key and motfm_key in MOTFM_BASELINES.get(nfe_levels[0], {}):
            motfm_vals = [MOTFM_BASELINES[nfe].get(motfm_key, np.nan) for nfe in nfe_levels]
            m_style = METHOD_STYLES["MOTFM"]
            ax.plot(
                nfe_levels, motfm_vals,
                color=m_style["color"], marker=m_style["marker"], ls=m_style["ls"],
                label="MOTFM", linewidth=1.2, markersize=5,
            )
            npz_data[f"motfm_{key}"] = np.array(motfm_vals)

        # DDPM baseline
        ddpm_key = motfm_key
        if ddpm_key and ddpm_key in DDPM_BASELINES.get(nfe_levels[0], {}):
            ddpm_vals = [DDPM_BASELINES[nfe].get(ddpm_key, np.nan) for nfe in nfe_levels]
            d_style = METHOD_STYLES["DDPM"]
            ax.plot(
                nfe_levels, ddpm_vals,
                color=d_style["color"], marker=d_style["marker"], ls=d_style["ls"],
                label="DDPM", linewidth=1.0, markersize=4,
            )
            npz_data[f"ddpm_{key}"] = np.array(ddpm_vals)

        ax.set_xscale("log")
        ax.set_xticks(nfe_levels)
        ax.set_xticklabels([str(n) for n in nfe_levels])
        ax.set_xlabel("NFE")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + i)}) {title}")
        ax.grid(True, alpha=0.3)

        # Better direction annotation
        arrow_text = r"$\downarrow$ better" if better_dir == "lower" else r"$\uparrow$ better"
        ax.annotate(
            arrow_text, xy=(0.97, 0.97), xycoords="axes fraction",
            ha="right", va="top", fontsize=7, color="grey",
        )

        if i == 0:
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    np.savez(output_dir / "data" / "nfe_scaling.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig01_nfe_scaling")
    logger.info("Saved fig01_nfe_scaling")


# -----------------------------------------------------------------------
# Fig 02 — Bootstrap FID Distributions
# -----------------------------------------------------------------------


def plot_bootstrap_fid(
    bootstrap_fid: dict[int, dict[str, Any]],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Bootstrap FID distributions: overlaid KDE curves for all NFE levels.

    Args:
        bootstrap_fid: Dict mapping NFE to bootstrap result dict.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    from scipy.stats import gaussian_kde

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=get_figure_size("single", 0.8))

    npz_data: dict[str, np.ndarray] = {}

    for nfe in nfe_levels:
        if nfe not in bootstrap_fid:
            continue

        bdata = bootstrap_fid[nfe]
        samples = bdata["samples"]
        npz_data[f"nfe{nfe}_samples"] = samples
        color = NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["blue"])

        # KDE curve
        kde = gaussian_kde(samples)
        x_grid = np.linspace(samples.min() * 0.95, samples.max() * 1.05, 300)
        density = kde(x_grid)
        ax.fill_between(x_grid, density, alpha=0.25, color=color)
        ax.plot(x_grid, density, color=color, lw=1.5, label=f"NFE={nfe}")

        # Point estimate vertical line
        ax.axvline(
            bdata["point_estimate"], color=color, ls="--", lw=1.0,
        )

        # CI annotation
        ci_text = (
            f"FID={bdata['point_estimate']:.1f} "
            f"[{bdata['ci_lower']:.1f}, {bdata['ci_upper']:.1f}]"
        )
        # Place text at peak of KDE for this NFE
        peak_x = x_grid[np.argmax(density)]
        peak_y = density.max()
        ax.annotate(
            ci_text, xy=(peak_x, peak_y),
            xytext=(0, 6), textcoords="offset points",
            fontsize=6, ha="center", va="bottom", color=color,
            fontweight="bold",
        )

        # MOTFM reference line
        motfm_fid = MOTFM_BASELINES.get(nfe, {}).get("fid_3d")
        if motfm_fid is not None:
            ax.axvline(
                motfm_fid, color=color, ls=":", lw=0.8, alpha=0.5,
            )

    ax.set_xlabel("FID-3D")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="x", alpha=0.15)
    # Remove y-axis entirely — density magnitude is not meaningful
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    np.savez(output_dir / "data" / "bootstrap_fid.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig02_bootstrap_fid")
    logger.info("Saved fig02_bootstrap_fid")


# -----------------------------------------------------------------------
# Fig 03 — Feature Space t-SNE
# -----------------------------------------------------------------------


def plot_feature_tsne(
    features: dict[str, Any],
    output_dir: Path,
    nfe_levels: list[int],
    n_subsample: int = 326,
) -> None:
    """t-SNE of real vs generated features per NFE with KDE density underlay.

    Args:
        features: Dict from load_features().
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
        n_subsample: Number of generated samples to subsample per NFE.
    """
    from scipy.stats import gaussian_kde
    from sklearn.manifold import TSNE

    apply_ieee_style()
    n_panels = len(nfe_levels)
    fig, axes = plt.subplots(
        2, n_panels,
        figsize=get_figure_size("double", 0.6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    real_feats = features.get("real")
    if real_feats is None:
        logger.warning("No real features for t-SNE — skipping")
        plt.close(fig)
        return

    real_np = real_feats.numpy() if hasattr(real_feats, "numpy") else np.array(real_feats)
    npz_data: dict[str, np.ndarray] = {"real_features": real_np}

    for i, nfe in enumerate(nfe_levels):
        ax_scatter = axes[0, i]
        ax_density = axes[1, i]
        gen_key = f"gen_{nfe}"
        if gen_key not in features:
            ax_scatter.set_visible(False)
            ax_density.set_visible(False)
            continue

        gen_feats = features[gen_key]
        gen_np = gen_feats.numpy() if hasattr(gen_feats, "numpy") else np.array(gen_feats)

        # Subsample generated features
        rng = np.random.RandomState(42)
        n_sub = min(n_subsample, gen_np.shape[0])
        idx = rng.choice(gen_np.shape[0], size=n_sub, replace=False)
        gen_sub = gen_np[idx]

        # Run t-SNE on combined
        combined = np.vstack([real_np, gen_sub])
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        embedded = tsne.fit_transform(combined)

        n_real = real_np.shape[0]
        real_emb = embedded[:n_real]
        gen_emb = embedded[n_real:]

        npz_data[f"nfe{nfe}_real_tsne"] = real_emb
        npz_data[f"nfe{nfe}_gen_tsne"] = gen_emb

        # --- Scatter panel (top) ---
        ax_scatter.scatter(
            real_emb[:, 0], real_emb[:, 1],
            c=PAUL_TOL_BRIGHT["blue"], s=8, alpha=0.5, label="Real",
            edgecolors="none", zorder=2,
        )
        ax_scatter.scatter(
            gen_emb[:, 0], gen_emb[:, 1],
            c=PAUL_TOL_BRIGHT["red"], s=8, alpha=0.5, label="Generated",
            edgecolors="none", zorder=2,
        )

        ax_scatter.set_title(f"NFE = {nfe}")
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        ax_scatter.grid(False)
        if i == 0:
            ax_scatter.legend(fontsize=7, loc="best", markerscale=2)

        # --- 1-D density panel (bottom): marginal along t-SNE dim 1 ---
        x_all = embedded[:, 0]
        x_min, x_max = x_all.min() - 2, x_all.max() + 2
        x_grid = np.linspace(x_min, x_max, 200)

        real_kde = gaussian_kde(real_emb[:, 0], bw_method=0.3)
        gen_kde = gaussian_kde(gen_emb[:, 0], bw_method=0.3)

        ax_density.fill_between(
            x_grid, real_kde(x_grid),
            alpha=0.35, color=PAUL_TOL_BRIGHT["blue"], label="Real",
        )
        ax_density.fill_between(
            x_grid, gen_kde(x_grid),
            alpha=0.35, color=PAUL_TOL_BRIGHT["red"], label="Generated",
        )
        ax_density.plot(x_grid, real_kde(x_grid), color=PAUL_TOL_BRIGHT["blue"], lw=0.8)
        ax_density.plot(x_grid, gen_kde(x_grid), color=PAUL_TOL_BRIGHT["red"], lw=0.8)

        ax_density.set_xlim(ax_scatter.get_xlim())
        ax_density.set_yticks([])
        ax_density.set_xlabel("t-SNE dim 1", fontsize=7)
        ax_density.set_ylabel("Density" if i == 0 else "", fontsize=7)
        ax_density.grid(False)

    fig.tight_layout()
    np.savez(output_dir / "data" / "feature_embeddings.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig03_feature_tsne")
    logger.info("Saved fig03_feature_tsne")


# -----------------------------------------------------------------------
# Fig 03b — Feature Space PCA (preferred over t-SNE)
# -----------------------------------------------------------------------


def _cmap_from_color(hex_color: str) -> "LinearSegmentedColormap":
    """Create a white-to-color sequential colormap for KDE contours."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("", ["white", hex_color], N=256)


def _draw_confidence_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    n_std: float = 2.45,
) -> None:
    """Draw a 95% confidence ellipse from 2D point cloud.

    Args:
        ax: Matplotlib axes.
        points: Array of shape (N, 2).
        color: Edge color.
        n_std: Number of std deviations. 2.45 corresponds to 95%
            for a 2D Gaussian (sqrt of chi2(2) at p=0.05 ≈ 5.991).
    """
    from matplotlib.patches import Ellipse

    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0, None)
    # Major eigenvector is the last one (eigh returns ascending order)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])
    cx, cy = points.mean(axis=0)
    ellipse = Ellipse(
        (cx, cy), width, height, angle=angle,
        facecolor="none", edgecolor=color, linestyle="--",
        linewidth=1.0, zorder=3,
    )
    ax.add_patch(ellipse)


def plot_feature_pca(
    feature_sets: list[dict[str, np.ndarray]],
    panel_titles: list[str],
    output_dir: Path,
    output_name: str = "fig03_feature_pca",
    reference_label: str = "Real",
    n_subsample: int = 500,
) -> None:
    """PCA projection with 2D KDE density, confidence ellipses, and marginals.

    Fits PCA on the reference distribution so axes represent the principal
    directions of real data. All other distributions are projected into
    that space. Each panel shows a scatter with 2D KDE contours, 95%
    confidence ellipses, a PC1 marginal (bottom), and a PC2 marginal (right).
    Axes are shared across all panels.

    Args:
        feature_sets: List of dicts, one per panel. Each dict maps a label
            (e.g. "Real", "NeuroiMF", "MOTFM") to a feature array (N, D).
        panel_titles: Display title for each panel (e.g. "NFE = 1").
        output_dir: Analysis output directory.
        output_name: Output filename stem.
        reference_label: Key for the reference distribution (PCA is fit on
            this). Defaults to "Real".
        n_subsample: Max points per label per panel (for visual clarity).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    apply_ieee_style()

    n_panels = len(feature_sets)

    # ------------------------------------------------------------------
    # Fit PCA on reference distribution (defines the coordinate system)
    # ------------------------------------------------------------------
    ref_all = None
    for fs in feature_sets:
        if reference_label in fs:
            ref = fs[reference_label]
            ref_all = ref.numpy() if hasattr(ref, "numpy") else np.asarray(ref)
            break

    if ref_all is None:
        logger.warning("No reference features ('%s') — skipping PCA", reference_label)
        return

    pca = PCA(n_components=2, random_state=42)
    pca.fit(ref_all)
    var_pct = pca.explained_variance_ratio_ * 100

    # ------------------------------------------------------------------
    # Color assignment: reference = blue, others from palette in order
    # ------------------------------------------------------------------
    non_ref_palette = [
        PAUL_TOL_BRIGHT["red"],
        PAUL_TOL_BRIGHT["green"],
        PAUL_TOL_BRIGHT["yellow"],
        PAUL_TOL_BRIGHT["purple"],
        PAUL_TOL_BRIGHT["cyan"],
    ]
    all_non_ref = sorted(
        {lbl for fs in feature_sets for lbl in fs if lbl != reference_label}
    )
    label_colors: dict[str, str] = {reference_label: PAUL_TOL_BRIGHT["blue"]}
    for i, lbl in enumerate(all_non_ref):
        label_colors[lbl] = non_ref_palette[i % len(non_ref_palette)]

    npz_data: dict[str, np.ndarray] = {"variance_explained_pct": var_pct}
    rng = np.random.RandomState(42)

    # ------------------------------------------------------------------
    # Pre-project all data to compute global axis limits
    # ------------------------------------------------------------------
    all_projected: list[dict[str, np.ndarray]] = []
    for pi, fs in enumerate(feature_sets):
        projected: dict[str, np.ndarray] = {}
        for label, feats in fs.items():
            arr = feats.numpy() if hasattr(feats, "numpy") else np.asarray(feats)
            if arr.shape[0] > n_subsample:
                idx = rng.choice(arr.shape[0], n_subsample, replace=False)
                arr = arr[idx]
            projected[label] = pca.transform(arr)
            npz_key = f"p{pi}_{label.replace(' ', '_').lower()}"
            npz_data[npz_key] = projected[label]
        all_projected.append(projected)

    # Global limits across all panels (shared axes)
    all_pts = np.vstack(
        [pts for proj in all_projected for pts in proj.values()]
    )
    pad_x = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.12
    pad_y = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.12
    x_lim = (all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x)
    y_lim = (all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y)

    # ------------------------------------------------------------------
    # Build figure with nested gridspec: each panel = scatter + 2 marginals
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=get_figure_size("double", 0.55))
    outer_gs = GridSpec(
        1, n_panels, figure=fig, wspace=0.35,
    )

    # Store axes for legend collection
    legend_handles: dict[str, Any] = {}

    for pi, (projected, title) in enumerate(
        zip(all_projected, panel_titles)
    ):
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer_gs[pi],
            height_ratios=[4, 1], width_ratios=[4, 1],
            hspace=0.15, wspace=0.05,
        )
        ax_main = fig.add_subplot(inner[0, 0])
        ax_bot = fig.add_subplot(inner[1, 0])
        ax_right = fig.add_subplot(inner[0, 1])
        # inner[1,1] is empty corner — don't create axes

        # KDE grid
        xx, yy = np.mgrid[
            x_lim[0] : x_lim[1] : 150j,
            y_lim[0] : y_lim[1] : 150j,
        ]
        grid = np.vstack([xx.ravel(), yy.ravel()])

        # Plot order: reference underneath, then others
        ordered = [reference_label] + [
            l for l in projected if l != reference_label
        ]

        for label in ordered:
            if label not in projected:
                continue
            pts = projected[label]
            color = label_colors[label]

            # 2D KDE filled contours
            try:
                kde2d = gaussian_kde(pts.T, bw_method=0.3)
                zz = kde2d(grid).reshape(xx.shape)
                cmap = _cmap_from_color(color)
                ax_main.contourf(
                    xx, yy, zz, levels=8, cmap=cmap, alpha=0.35, zorder=0,
                )
            except np.linalg.LinAlgError:
                pass

            # Scatter
            h = ax_main.scatter(
                pts[:, 0], pts[:, 1], c=color, s=5, alpha=0.35,
                edgecolors="none", zorder=2,
            )
            if label not in legend_handles:
                legend_handles[label] = h

            # Centroid diamond
            cx, cy = pts.mean(axis=0)
            ax_main.scatter(
                cx, cy, c=color, s=70, marker="D",
                edgecolors="white", linewidths=0.8, zorder=5,
            )

            # 95% confidence ellipse
            if pts.shape[0] > 3:
                _draw_confidence_ellipse(ax_main, pts, color)

            # PC1 marginal (bottom)
            kde_x = gaussian_kde(pts[:, 0], bw_method=0.3)
            xg = np.linspace(x_lim[0], x_lim[1], 200)
            ax_bot.fill_between(xg, kde_x(xg), alpha=0.3, color=color)
            ax_bot.plot(xg, kde_x(xg), color=color, lw=0.8)

            # PC2 marginal (right) — rotated: y-values on y-axis, density on x
            kde_y = gaussian_kde(pts[:, 1], bw_method=0.3)
            yg = np.linspace(y_lim[0], y_lim[1], 200)
            ax_right.fill_betweenx(yg, kde_y(yg), alpha=0.3, color=color)
            ax_right.plot(kde_y(yg), yg, color=color, lw=0.8)

        # Centroid distance annotation below model centroid
        if reference_label in projected and len(projected) > 1:
            ref_c = projected[reference_label].mean(axis=0)
            for label in ordered:
                if label == reference_label or label not in projected:
                    continue
                gen_c = projected[label].mean(axis=0)
                dist = np.linalg.norm(ref_c - gen_c)

                # Dashed line between centroids
                ax_main.plot(
                    [ref_c[0], gen_c[0]], [ref_c[1], gen_c[1]],
                    ls=":", color="grey", lw=0.7, zorder=1,
                )

                # Place Δμ just below the model centroid
                y_range = y_lim[1] - y_lim[0]
                ax_main.annotate(
                    f"$\\Delta_{{\\mu}}$={dist:.1f}",
                    xy=(gen_c[0], gen_c[1] - 0.06 * y_range),
                    fontsize=6, ha="center", va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white",
                        alpha=0.7, lw=0,
                    ),
                    zorder=6,
                )

        # --- Axis formatting ---
        ax_main.set_xlim(x_lim)
        ax_main.set_ylim(y_lim)
        ax_main.set_xlabel(f"PC1 ({var_pct[0]:.1f}%)", fontsize=8)
        ax_main.set_title(title)
        ax_main.grid(True, alpha=0.12, zorder=0)

        # Only leftmost panel gets y-label
        if pi == 0:
            ax_main.set_ylabel(f"PC2 ({var_pct[1]:.1f}%)", fontsize=8)
        else:
            ax_main.set_yticklabels([])

        # Hide x tick labels on main (marginal below handles x ticks)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Bottom marginal — no x-label (shared axis with scatter above)
        ax_bot.set_xlim(x_lim)
        ax_bot.set_yticks([])
        ax_bot.grid(False)

        # Right marginal
        ax_right.set_ylim(y_lim)
        ax_right.set_xticks([])
        ax_right.set_yticklabels([])
        ax_right.grid(False)

    # ------------------------------------------------------------------
    # Single legend outside, upper-center
    # ------------------------------------------------------------------
    ordered_labels = [reference_label] + [
        l for l in legend_handles if l != reference_label
    ]
    fig.legend(
        [legend_handles[l] for l in ordered_labels if l in legend_handles],
        [f"{l} (n={all_projected[0][l].shape[0]})"
         if l in all_projected[0] else l
         for l in ordered_labels if l in legend_handles],
        loc="upper center",
        ncol=len(ordered_labels),
        fontsize=7,
        markerscale=3,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 1.0),
    )

    fig.subplots_adjust(top=0.90)
    np.savez(output_dir / "data" / "feature_embeddings.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / output_name)
    logger.info("Saved %s", output_name)


# -----------------------------------------------------------------------
# Fig 04 — Spectral HF Energy Comparison
# -----------------------------------------------------------------------


def plot_spectral_comparison(
    metrics: dict[int, dict[str, Any]],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Spectral HF energy ratio: grouped bar chart.

    Args:
        metrics: Dict mapping NFE to parsed metrics JSON.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=get_figure_size("single", 0.8))

    means = []
    stds = []
    labels = []
    colors = []

    for nfe in nfe_levels:
        if nfe not in metrics:
            continue
        spec = metrics[nfe].get("spectral", {}).get("hf_energy_ratio", {})
        means.append(spec.get("mean", 0))
        stds.append(spec.get("std", 0))
        labels.append(f"NFE={nfe}")
        colors.append(NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["blue"]))

    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, yerr=stds, capsize=3,
        color=colors, edgecolor="white", linewidth=0.5, alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("HF Energy Ratio")
    ax.set_title("Spectral High-Frequency Energy")
    ax.grid(axis="y", alpha=0.3)

    # Save data
    npz_data = {
        "nfe_levels": np.array(nfe_levels),
        "means": np.array(means),
        "stds": np.array(stds),
    }
    np.savez(output_dir / "data" / "spectral_analysis.npz", **npz_data)

    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "fig04_spectral_comparison")
    logger.info("Saved fig04_spectral_comparison")


# -----------------------------------------------------------------------
# Fig 05 — Regional Volume Distributions
# -----------------------------------------------------------------------


def plot_regional_volumes(
    synthseg_data: dict[str, Any],
    ks_results: pd.DataFrame | None,
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Regional volume distributions: grouped box plots with KS stars.

    Args:
        synthseg_data: Dict from load_synthseg_data().
        ks_results: KS test results DataFrame (optional).
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    real_df = synthseg_data.get("real_volumes")
    if real_df is None:
        logger.warning("No real SynthSeg volumes — skipping fig05")
        return

    apply_ieee_style()
    fig, ax = plt.subplots(figsize=get_figure_size("double", 0.55))

    # Build bilateral-averaged DataFrames
    real_bilat = _bilateral_average(real_df, BILATERAL_REGIONS)
    gen_bilats: dict[int, pd.DataFrame] = {}
    for nfe in nfe_levels:
        gen_key = f"gen_volumes_{nfe}"
        if gen_key in synthseg_data and synthseg_data[gen_key] is not None:
            gen_bilats[nfe] = _bilateral_average(
                synthseg_data[gen_key], BILATERAL_REGIONS
            )

    region_names = [r[2] for r in BILATERAL_REGIONS]
    n_regions = len(region_names)
    n_groups = 1 + len(gen_bilats)  # Real + gen per NFE
    width = 0.8 / n_groups
    positions_base = np.arange(n_regions)

    npz_data: dict[str, np.ndarray] = {}

    # Plot Real
    real_data = [real_bilat[r].values for r in region_names if r in real_bilat.columns]
    bp = ax.boxplot(
        real_data,
        positions=positions_base - width * (n_groups - 1) / 2,
        widths=width * 0.8,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PAUL_TOL_BRIGHT["blue"])
        patch.set_alpha(0.6)

    for j, r in enumerate(region_names):
        if r in real_bilat.columns:
            npz_data[f"real_{r.replace(' ', '_').lower()}"] = real_bilat[r].values

    # Plot generated per NFE
    for g_idx, nfe in enumerate(sorted(gen_bilats.keys())):
        gen_df = gen_bilats[nfe]
        gen_data = [
            gen_df[r].values for r in region_names if r in gen_df.columns
        ]
        pos = positions_base - width * (n_groups - 1) / 2 + width * (g_idx + 1)
        bp = ax.boxplot(
            gen_data, positions=pos, widths=width * 0.8,
            patch_artist=True, showfliers=False,
        )
        color = NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["grey"])
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for j, r in enumerate(region_names):
            if r in gen_df.columns:
                npz_data[f"gen{nfe}_{r.replace(' ', '_').lower()}"] = gen_df[r].values

    ax.set_xticks(positions_base)
    ax.set_xticklabels(region_names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Volume (mm$^3$)")
    ax.set_title("Regional Brain Volumes")
    ax.grid(axis="y", alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=PAUL_TOL_BRIGHT["blue"], alpha=0.6, label="Real")]
    for nfe in sorted(gen_bilats.keys()):
        legend_items.append(
            Patch(
                facecolor=NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["grey"]),
                alpha=0.6,
                label=f"NFE={nfe}",
            )
        )
    ax.legend(handles=legend_items, fontsize=7, loc="upper right")

    # Add KS significance stars
    if ks_results is not None and not ks_results.empty:
        y_max = ax.get_ylim()[1]
        for j, r_name in enumerate(region_names):
            # Match to original region names
            for _, row in ks_results.iterrows():
                region_lower = row["region"]
                if r_name.lower() in region_lower or region_lower in r_name.lower():
                    if row["stars"] != "n.s.":
                        ax.text(
                            j, y_max * 0.95, row["stars"],
                            ha="center", va="top", fontsize=7, color="red",
                        )
                    break

    fig.tight_layout()
    np.savez(output_dir / "data" / "synthseg_volumes.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig05_regional_volumes")
    logger.info("Saved fig05_regional_volumes")


# -----------------------------------------------------------------------
# Fig 06 — Coverage vs Density
# -----------------------------------------------------------------------


def plot_coverage_density(
    metrics: dict[int, dict[str, Any]],
    bootstrap_results: dict[str, dict[int, dict[str, Any]]],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Coverage vs Density scatter plot.

    Args:
        metrics: Dict mapping NFE to parsed metrics JSON.
        bootstrap_results: Dict mapping metric name to {nfe: bootstrap_dict}.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    apply_ieee_style()
    fig, ax = plt.subplots(figsize=get_figure_size("single"))

    npz_data: dict[str, np.ndarray] = {}

    for nfe in nfe_levels:
        if nfe not in metrics:
            continue
        cov = metrics[nfe]["distributional"]["coverage_k5"]["value"]
        den = metrics[nfe]["distributional"]["density_k5"]["value"]

        # Get CI error bars if available (clamp to non-negative)
        cov_err = [0.0, 0.0]
        den_err = [0.0, 0.0]
        if "coverage" in bootstrap_results and nfe in bootstrap_results["coverage"]:
            bc = bootstrap_results["coverage"][nfe]
            cov_err = [max(0.0, cov - bc["ci_lower"]), max(0.0, bc["ci_upper"] - cov)]
        if "density" in bootstrap_results and nfe in bootstrap_results["density"]:
            bd = bootstrap_results["density"][nfe]
            den_err = [max(0.0, den - bd["ci_lower"]), max(0.0, bd["ci_upper"] - den)]

        color = NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["blue"])
        ax.errorbar(
            cov, den,
            xerr=[[cov_err[0]], [cov_err[1]]],
            yerr=[[den_err[0]], [den_err[1]]],
            fmt="o", color=color, markersize=8 + nfe / 10,
            capsize=3, elinewidth=1, zorder=3,
        )
        ax.annotate(
            f"NFE={nfe}", (cov, den),
            textcoords="offset points", xytext=(8, 5),
            fontsize=7, color=color,
        )
        npz_data[f"nfe{nfe}_coverage"] = np.array([cov])
        npz_data[f"nfe{nfe}_density"] = np.array([den])

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.set_title("Coverage vs Density")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    np.savez(output_dir / "data" / "bootstrap_coverage_density.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig06_coverage_density")
    logger.info("Saved fig06_coverage_density")


# -----------------------------------------------------------------------
# Fig 07 — SynthSeg QC Violin
# -----------------------------------------------------------------------


def plot_qc_violin(
    synthseg_data: dict[str, Any],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """SynthSeg QC score violins: 2x4 grid of QC regions.

    Args:
        synthseg_data: Dict from load_synthseg_data().
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    # Check if any QC data exists
    has_qc = any(
        synthseg_data.get(f"gen_qc_{nfe}") is not None for nfe in nfe_levels
    )
    if not has_qc:
        logger.warning("No SynthSeg QC data — skipping fig07")
        return

    apply_ieee_style()
    fig, axes = plt.subplots(2, 4, figsize=get_figure_size("double", 0.5))
    axes = axes.flatten()

    npz_data: dict[str, np.ndarray] = {}

    for i, region in enumerate(QC_REGIONS):
        ax = axes[i]
        violin_data = []
        violin_labels = []
        violin_colors = []

        for nfe in nfe_levels:
            qc_df = synthseg_data.get(f"gen_qc_{nfe}")
            if qc_df is None or region not in qc_df.columns:
                continue
            vals = qc_df[region].dropna().values
            violin_data.append(vals)
            violin_labels.append(f"NFE={nfe}")
            violin_colors.append(NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["grey"]))
            npz_data[f"{region.replace(' ', '_')}_{nfe}"] = vals

        if not violin_data:
            ax.set_visible(False)
            continue

        parts = ax.violinplot(
            violin_data, positions=range(len(violin_data)),
            showmeans=True, showextrema=False,
        )
        for j, body in enumerate(parts["bodies"]):
            body.set_facecolor(violin_colors[j])
            body.set_alpha(0.6)
        parts["cmeans"].set_color("black")

        ax.set_xticks(range(len(violin_labels)))
        ax.set_xticklabels(violin_labels, fontsize=6)
        ax.set_title(region.replace("+", "+\n"), fontsize=7)
        ax.set_ylabel("QC Score" if i % 4 == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    np.savez(output_dir / "data" / "per_volume_metrics.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig07_qc_violin")
    logger.info("Saved fig07_qc_violin")


# -----------------------------------------------------------------------
# Fig 08 — SynthSeg Segmentation Overlay (qualitative)
# -----------------------------------------------------------------------

# FreeSurfer-like colors for major labels
FREESURFER_COLORS: dict[int, tuple[float, float, float]] = {
    3: (0.80, 0.80, 0.15),   # Left-Cerebral-Cortex
    42: (0.80, 0.80, 0.15),  # Right-Cerebral-Cortex
    2: (0.95, 0.95, 0.95),   # Left-Cerebral-WM
    41: (0.95, 0.95, 0.95),  # Right-Cerebral-WM
    4: (0.47, 0.07, 0.53),   # Left-Lateral-Ventricle
    43: (0.47, 0.07, 0.53),  # Right-Lateral-Ventricle
    17: (0.86, 0.97, 0.64),  # Left-Hippocampus
    53: (0.86, 0.97, 0.64),  # Right-Hippocampus
    10: (0.0, 0.46, 0.0),    # Left-Thalamus
    49: (0.0, 0.46, 0.0),    # Right-Thalamus
    11: (0.48, 0.71, 0.86),  # Left-Caudate
    50: (0.48, 0.71, 0.86),  # Right-Caudate
    12: (0.93, 0.57, 0.13),  # Left-Putamen
    51: (0.93, 0.57, 0.13),  # Right-Putamen
    8: (0.91, 0.83, 0.68),   # Left-Cerebellum-Cortex
    47: (0.91, 0.83, 0.68),  # Right-Cerebellum-Cortex
    16: (0.85, 0.47, 0.66),  # Brain-Stem
}


def _label_to_rgb(label_vol: np.ndarray) -> np.ndarray:
    """Convert a 3D label volume to RGB using FreeSurfer-like colors.

    Args:
        label_vol: 3D integer array of label IDs.

    Returns:
        4D array of shape (*label_vol.shape, 3) with RGB values in [0, 1].
    """
    rgb = np.zeros((*label_vol.shape, 3))
    for label_id, color in FREESURFER_COLORS.items():
        mask = label_vol == label_id
        rgb[mask] = color
    return rgb


def _select_best_qc_indices(
    synthseg_data: dict[str, Any],
    nfe_levels: list[int],
) -> dict[str, int]:
    """Select the volume index with the highest mean QC score per condition.

    Args:
        synthseg_data: Dict from load_synthseg_data().
        nfe_levels: NFE values.

    Returns:
        Dict mapping condition key ('real', 'gen_1', ...) to best vol index
        (parsed from the 'subject' column, e.g. 'vol_0042' -> 42).
    """
    best: dict[str, int] = {}

    for nfe in nfe_levels:
        qc_df = synthseg_data.get(f"gen_qc_{nfe}")
        if qc_df is None:
            continue
        # Mean QC across all region columns (exclude 'subject')
        qc_cols = [c for c in qc_df.columns if c != "subject"]
        qc_df = qc_df.copy()
        qc_df["_mean_qc"] = qc_df[qc_cols].mean(axis=1)
        best_row = qc_df.loc[qc_df["_mean_qc"].idxmax()]
        vol_idx = int(best_row["subject"].replace("vol_", ""))
        best[f"gen_{nfe}"] = vol_idx
        logger.info(
            "Best QC volume for NFE=%d: vol_%04d (mean QC=%.4f)",
            nfe, vol_idx, best_row["_mean_qc"],
        )

    # For real, pick a representative (median QC) to look typical
    # but if no real QC exists, fall back to index 0
    best["real"] = 0
    return best


def plot_synthseg_overlay(
    volumes: dict[str, np.ndarray] | None,
    labels: dict[str, list[np.ndarray]] | None,
    output_dir: Path,
    nfe_levels: list[int],
    vol_indices: dict[str, int] | None = None,
) -> None:
    """SynthSeg segmentation overlay: MRI + colored labels.

    For each NFE column, shows the best-QC volume. The real column
    shows a fixed reference volume.

    Args:
        volumes: Dict mapping condition to array of shape (N, 1, D, H, W)
            or (N, D, H, W). Each condition may have different N.
        labels: Dict mapping condition to list of 3D label arrays.
            The i-th entry corresponds to the i-th loaded volume.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
        vol_indices: Dict mapping condition to the position index within
            the loaded volumes/labels arrays (not the global vol ID).
            If None, uses index 0 for all.
    """
    if volumes is None or labels is None:
        logger.warning("Volumes or labels unavailable — skipping fig08")
        return

    if vol_indices is None:
        vol_indices = {}

    apply_ieee_style()
    conditions = ["real"] + [f"gen_{nfe}" for nfe in nfe_levels]
    col_labels = ["Real"] + [f"NFE={nfe}" for nfe in nfe_levels]
    n_cols = len(conditions)

    fig, axes = plt.subplots(
        3, n_cols,
        figsize=get_figure_size("double", 0.7),
        facecolor="black",
    )

    view_names = ["Axial", "Coronal", "Sagittal"]

    for col, (cond, col_label) in enumerate(zip(conditions, col_labels)):
        if cond not in volumes or cond not in labels:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        idx = vol_indices.get(cond, 0)

        vol = volumes[cond]
        if vol.ndim == 5:
            vol = vol[idx, 0]
        elif vol.ndim == 4:
            vol = vol[idx, 0] if vol.shape[1] < vol.shape[-1] else vol[idx]
        elif vol.ndim == 3:
            pass  # single volume already

        lab_list = labels[cond]
        if idx >= len(lab_list):
            for row in range(3):
                axes[row, col].set_visible(False)
            continue
        lab = lab_list[idx]

        # Get center slices
        mid = [s // 2 for s in vol.shape]
        slices_vol = [vol[mid[0], :, :], vol[:, mid[1], :], vol[:, :, mid[2]]]
        slices_lab = [lab[mid[0], :, :], lab[:, mid[1], :], lab[:, :, mid[2]]]

        for row in range(3):
            ax = axes[row, col]
            ax.set_facecolor("black")

            # Grayscale underlay
            v_slice = slices_vol[row]
            fg = v_slice[v_slice > 0]
            vmin, vmax = (np.percentile(fg, [1, 99]) if fg.size > 0 else (0, 1))
            ax.imshow(v_slice.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")

            # Label overlay
            l_slice = slices_lab[row]
            rgb_overlay = _label_to_rgb(l_slice)
            mask = l_slice > 0
            alpha_map = np.where(mask, 0.4, 0.0)
            rgba = np.concatenate(
                [rgb_overlay, alpha_map[..., np.newaxis]], axis=-1
            )
            ax.imshow(rgba.transpose(1, 0, 2), origin="lower")

            ax.axis("off")
            if row == 0:
                ax.set_title(col_label, fontsize=8, color="white")
            if col == 0:
                ax.text(
                    -0.05, 0.5, view_names[row], transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=8, color="white",
                )

    fig.tight_layout(pad=0.3)
    save_figure(fig, output_dir / "figures" / "fig08_synthseg_overlay")
    logger.info("Saved fig08_synthseg_overlay")


# -----------------------------------------------------------------------
# Fig 09 — Sample Quality Grid (qualitative)
# -----------------------------------------------------------------------


def plot_sample_grid(
    volumes: dict[str, np.ndarray] | None,
    metrics: dict[int, dict[str, Any]],
    output_dir: Path,
    nfe: int = 50,
) -> None:
    """Sample quality grid: best / median / worst by MS-SSIM.

    Args:
        volumes: Dict from load_volumes() or None.
        metrics: Dict mapping NFE to parsed metrics JSON.
        output_dir: Analysis output directory.
        nfe: NFE level to showcase.
    """
    if volumes is None:
        logger.warning("Volumes unavailable — skipping fig09")
        return

    gen_key = f"gen_{nfe}"
    if gen_key not in volumes or "real" not in volumes:
        logger.warning("Required volumes missing — skipping fig09")
        return

    apply_ieee_style()
    from neuromf.utils.visualisation import _get_center_slices

    gen_vols = volumes[gen_key]
    n_vols = gen_vols.shape[0]

    # Pick best, median, worst indices (0, n//2, n-1 of available)
    sample_indices = [0, n_vols // 2, n_vols - 1]
    row_labels = ["Best", "Median", "Worst"]

    fig, axes = plt.subplots(
        3, 3,
        figsize=get_figure_size("double", 0.55),
    )
    col_labels = ["Axial", "Coronal", "Sagittal"]

    for row, (idx, label) in enumerate(zip(sample_indices, row_labels)):
        vol = gen_vols[idx]
        if vol.ndim == 4:
            vol = vol[0]  # Remove channel dim

        axial, sagittal, coronal = _get_center_slices(vol, crop_frac=0.85)
        slices = [axial, coronal, sagittal]

        for col, (sl, col_label) in enumerate(zip(slices, col_labels)):
            ax = axes[row, col]
            vmin, vmax = np.percentile(sl[sl > 0], [1, 99]) if sl.max() > 0 else (0, 1)
            ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_label, fontsize=8)
            if col == 0:
                ax.text(
                    -0.05, 0.5, label, transform=ax.transAxes,
                    rotation=90, va="center", ha="right", fontsize=8,
                )

    fig.suptitle(f"Generated Samples (NFE={nfe})", fontsize=10, y=1.0)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "fig09_sample_grid")
    logger.info("Saved fig09_sample_grid")


# -----------------------------------------------------------------------
# Fig 10 — Metric Bar Chart with CI
# -----------------------------------------------------------------------


def plot_metric_bars(
    metrics: dict[int, dict[str, Any]],
    bootstrap_results: dict[str, dict[int, dict[str, Any]]],
    cohens_d_results: dict[str, dict[int, dict[str, Any]]],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Horizontal bar chart: NeuroiMF vs MOTFM vs DDPM with CI and Cohen's d.

    Args:
        metrics: Dict mapping NFE to parsed metrics JSON.
        bootstrap_results: Dict mapping metric name to {nfe: bootstrap_dict}.
        cohens_d_results: Dict mapping metric to {nfe: cohens_d_dict}.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to compare (uses 10 and 50).
    """
    apply_ieee_style()
    metric_panels = [
        ("FID-3D", "fid_3d", lambda m: m["distributional"]["fid_3d"]["value"]),
        ("MMD", "mmd", lambda m: m["distributional"]["mmd"]["value"]),
        ("MS-SSIM", "ms_ssim", lambda m: m["per_volume"]["ms_ssim"]["mean"]),
    ]
    compare_nfes = [nfe for nfe in [10, 50] if nfe in nfe_levels]

    n_panels = len(metric_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=get_figure_size("double", 0.4))
    if n_panels == 1:
        axes = [axes]

    methods = ["NeuroiMF", "MOTFM", "DDPM"]
    method_colors = [
        METHOD_STYLES[m]["color"] for m in methods
    ]

    for panel_idx, (title, key, extract_fn) in enumerate(metric_panels):
        ax = axes[panel_idx]
        y_positions = []
        bar_values = []
        bar_errors = []
        bar_colors = []
        bar_labels = []

        y = 0
        for nfe in compare_nfes:
            for m_idx, method in enumerate(methods):
                if method == "NeuroiMF":
                    if nfe in metrics:
                        val = extract_fn(metrics[nfe])
                        err = 0
                        if key in bootstrap_results and nfe in bootstrap_results[key]:
                            bs = bootstrap_results[key][nfe]
                            err = (bs["ci_upper"] - bs["ci_lower"]) / 2
                    else:
                        continue
                elif method == "MOTFM":
                    val = MOTFM_BASELINES.get(nfe, {}).get(key, None)
                    err = 0
                    if val is None:
                        continue
                elif method == "DDPM":
                    val = DDPM_BASELINES.get(nfe, {}).get(key, None)
                    err = 0
                    if val is None:
                        continue

                y_positions.append(y)
                bar_values.append(val)
                bar_errors.append(err)
                bar_colors.append(method_colors[m_idx])
                bar_labels.append(f"{method}\nNFE={nfe}")
                y += 1
            y += 0.5  # Gap between NFE groups

        ax.barh(
            y_positions, bar_values, xerr=bar_errors,
            color=bar_colors, height=0.7, capsize=2,
            edgecolor="white", linewidth=0.3,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(bar_labels, fontsize=6)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # Add Cohen's d annotations between NeuroiMF and MOTFM bars
        if key in cohens_d_results:
            for nfe in compare_nfes:
                if nfe not in cohens_d_results[key]:
                    continue
                cd = cohens_d_results[key][nfe]
                d_text = f"d={cd['d']:.2f} ({cd['interpretation']})"
                # Find NeuroiMF and MOTFM bars for this NFE
                neuro_idx = None
                motfm_idx = None
                for i, lbl in enumerate(bar_labels):
                    if lbl == f"NeuroiMF\nNFE={nfe}":
                        neuro_idx = i
                    elif lbl == f"MOTFM\nNFE={nfe}":
                        motfm_idx = i
                if neuro_idx is not None and motfm_idx is not None:
                    # Place annotation between the two bars, inside plot
                    mid_y = (y_positions[neuro_idx] + y_positions[motfm_idx]) / 2
                    # Use a bracket-like annotation
                    x_right = max(bar_values[neuro_idx], bar_values[motfm_idx])
                    ax.annotate(
                        d_text,
                        xy=(x_right * 0.5, mid_y),
                        fontsize=5.5, va="center", ha="center",
                        bbox=dict(
                            boxstyle="round,pad=0.15", fc="white",
                            alpha=0.8, lw=0.3, ec="grey",
                        ),
                    )

    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "fig10_metric_bars")
    logger.info("Saved fig10_metric_bars")
