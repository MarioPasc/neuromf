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

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "NeuroiMF": {"color": PAUL_TOL_BRIGHT["blue"], "marker": "o", "ls": "-"},
    "MOTFM": {"color": PAUL_TOL_BRIGHT["red"], "marker": "^", "ls": "--"},
    "DDPM": {"color": PAUL_TOL_BRIGHT["grey"], "marker": "s", "ls": ":"},
}

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
    """Bootstrap FID distributions: histogram per NFE.

    Args:
        bootstrap_fid: Dict mapping NFE to bootstrap result dict.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
    """
    apply_ieee_style()
    n_panels = len(nfe_levels)
    fig, axes = plt.subplots(1, n_panels, figsize=get_figure_size("double", 0.3))
    if n_panels == 1:
        axes = [axes]

    npz_data: dict[str, np.ndarray] = {}

    for i, nfe in enumerate(nfe_levels):
        ax = axes[i]
        if nfe not in bootstrap_fid:
            ax.set_visible(False)
            continue

        bdata = bootstrap_fid[nfe]
        samples = bdata["samples"]
        npz_data[f"nfe{nfe}_samples"] = samples

        ax.hist(
            samples, bins=40, density=True,
            color=NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["blue"]),
            alpha=0.7, edgecolor="white", linewidth=0.3,
        )
        ax.axvline(
            bdata["point_estimate"], color="black", ls="--", lw=1.2,
            label=f"FID = {bdata['point_estimate']:.2f}",
        )
        ax.axvspan(
            bdata["ci_lower"], bdata["ci_upper"],
            alpha=0.15, color="grey", label="95% CI",
        )

        # MOTFM reference
        motfm_fid = MOTFM_BASELINES.get(nfe, {}).get("fid_3d")
        if motfm_fid is not None:
            ax.axvline(
                motfm_fid, color=METHOD_STYLES["MOTFM"]["color"],
                ls="--", lw=1.0, label=f"MOTFM = {motfm_fid:.1f}",
            )

        ax.set_xlabel("FID-3D")
        ax.set_ylabel("Density" if i == 0 else "")
        ax.set_title(f"NFE = {nfe}")
        ax.legend(fontsize=6, loc="upper right")

        # Text annotation
        ci_text = f"[{bdata['ci_lower']:.1f}, {bdata['ci_upper']:.1f}]"
        ax.text(
            0.03, 0.85, ci_text, transform=ax.transAxes, fontsize=6,
            verticalalignment="top",
        )

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
    """t-SNE of real vs generated features per NFE.

    Args:
        features: Dict from load_features().
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
        n_subsample: Number of generated samples to subsample per NFE.
    """
    from sklearn.manifold import TSNE

    apply_ieee_style()
    n_panels = len(nfe_levels)
    fig, axes = plt.subplots(1, n_panels, figsize=get_figure_size("double", 0.38))
    if n_panels == 1:
        axes = [axes]

    real_feats = features.get("real")
    if real_feats is None:
        logger.warning("No real features for t-SNE — skipping")
        plt.close(fig)
        return

    real_np = real_feats.numpy() if hasattr(real_feats, "numpy") else np.array(real_feats)
    npz_data: dict[str, np.ndarray] = {"real_features": real_np}

    for i, nfe in enumerate(nfe_levels):
        ax = axes[i]
        gen_key = f"gen_{nfe}"
        if gen_key not in features:
            ax.set_visible(False)
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

        ax.scatter(
            real_emb[:, 0], real_emb[:, 1],
            c=PAUL_TOL_BRIGHT["blue"], s=8, alpha=0.5, label="Real",
            edgecolors="none",
        )
        ax.scatter(
            gen_emb[:, 0], gen_emb[:, 1],
            c=PAUL_TOL_BRIGHT["red"], s=8, alpha=0.5, label="Generated",
            edgecolors="none",
        )

        ax.set_title(f"NFE = {nfe}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        if i == 0:
            ax.legend(fontsize=7, loc="best", markerscale=2)

    fig.tight_layout()
    np.savez(output_dir / "data" / "feature_embeddings.npz", **npz_data)
    save_figure(fig, output_dir / "figures" / "fig03_feature_tsne")
    logger.info("Saved fig03_feature_tsne")


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

        # Get CI error bars if available
        cov_err = [0, 0]
        den_err = [0, 0]
        if "coverage" in bootstrap_results and nfe in bootstrap_results["coverage"]:
            bc = bootstrap_results["coverage"][nfe]
            cov_err = [cov - bc["ci_lower"], bc["ci_upper"] - cov]
        if "density" in bootstrap_results and nfe in bootstrap_results["density"]:
            bd = bootstrap_results["density"][nfe]
            den_err = [den - bd["ci_lower"], bd["ci_upper"] - den]

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


def plot_synthseg_overlay(
    volumes: dict[str, np.ndarray] | None,
    labels: dict[str, list[np.ndarray]] | None,
    output_dir: Path,
    nfe_levels: list[int],
    vol_idx: int = 0,
) -> None:
    """SynthSeg segmentation overlay: MRI + colored labels.

    Args:
        volumes: Dict from load_volumes() or None.
        labels: Dict from load_synthseg_labels() or None.
        output_dir: Analysis output directory.
        nfe_levels: NFE values to plot.
        vol_idx: Index within the loaded volumes/labels lists.
    """
    if volumes is None or labels is None:
        logger.warning("Volumes or labels unavailable — skipping fig08")
        return

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

        vol = volumes[cond]
        if vol.ndim == 4:
            vol = vol[vol_idx, 0]
        elif vol.ndim == 5:
            vol = vol[vol_idx, 0]
        elif vol.ndim == 3:
            pass  # already 3D

        lab_list = labels[cond]
        if vol_idx >= len(lab_list):
            for row in range(3):
                axes[row, col].set_visible(False)
            continue
        lab = lab_list[vol_idx]

        # Get center slices
        mid = [s // 2 for s in vol.shape]
        slices_vol = [vol[mid[0], :, :], vol[:, mid[1], :], vol[:, :, mid[2]]]
        slices_lab = [lab[mid[0], :, :], lab[:, mid[1], :], lab[:, :, mid[2]]]

        for row in range(3):
            ax = axes[row, col]
            ax.set_facecolor("black")

            # Grayscale underlay
            v_slice = slices_vol[row]
            vmin, vmax = np.percentile(v_slice[v_slice > 0], [1, 99]) if v_slice.max() > 0 else (0, 1)
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

        # Add Cohen's d annotations
        if key in cohens_d_results:
            for nfe in compare_nfes:
                if nfe in cohens_d_results[key]:
                    cd = cohens_d_results[key][nfe]
                    d_text = f"d={cd['d']:.2f} ({cd['interpretation']})"
                    # Find NeuroiMF bar for this NFE
                    for i, lbl in enumerate(bar_labels):
                        if f"NeuroiMF\nNFE={nfe}" == lbl:
                            ax.text(
                                bar_values[i] + bar_errors[i] + 0.5,
                                y_positions[i],
                                d_text, fontsize=5, va="center",
                            )
                            break

    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / "fig10_metric_bars")
    logger.info("Saved fig10_metric_bars")
