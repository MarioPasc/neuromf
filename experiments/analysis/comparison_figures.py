"""Multi-model comparison figures for IEEE TMI publication.

Generates publication-quality comparison plots across NeuroiMF, MOTFM, and
DDPM models. All figures follow IEEE TMI style via ``apply_ieee_style()``
and save both PDF and PNG outputs alongside source data as ``.npz``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from experiments.analysis.data_loader import ModelResults
from experiments.analysis.figures import (
    BILATERAL_REGIONS,
    METHOD_STYLES,
    NFE_COLORS,
)
from experiments.utils.settings import (
    PAUL_TOL_BRIGHT,
    apply_ieee_style,
    get_figure_size,
    get_significance_stars,
)
from neuromf.utils.visualisation import save_figure

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _extract_metric(
    metrics: dict[int, dict[str, Any]],
    nfe: int,
    key: str,
) -> float | None:
    """Safely extract a metric value from the nested JSON structure.

    Args:
        metrics: Mapping of NFE to parsed metrics JSON.
        nfe: NFE level.
        key: Metric key (e.g. ``"fid_3d"``, ``"ms_ssim"``).

    Returns:
        Metric value, or ``None`` if unavailable.
    """
    if nfe not in metrics:
        return None
    m = metrics[nfe]
    extractors: dict[str, Any] = {
        "fid_3d": lambda m: m["distributional"]["fid_3d"]["value"],
        "mmd": lambda m: m["distributional"]["mmd"]["value"],
        "coverage": lambda m: m["distributional"]["coverage_k5"]["value"],
        "density": lambda m: m["distributional"]["density_k5"]["value"],
        "ms_ssim": lambda m: m["per_volume"]["ms_ssim"]["mean"],
        "hf_energy": lambda m: m["spectral"]["hf_energy_ratio"]["mean"],
    }
    try:
        return extractors[key](m)
    except (KeyError, TypeError):
        return None


def _bilateral_average(
    df: pd.DataFrame,
    regions: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Compute bilateral average (L+R)/2 for paired regions.

    Args:
        df: SynthSeg volumes DataFrame.
        regions: List of ``(left_col, right_col, display_name)`` tuples.

    Returns:
        DataFrame with columns named by display_name.
    """
    result: dict[str, np.ndarray] = {}
    for left, right, name in regions:
        if left == right:
            if left in df.columns:
                result[name] = df[left].values
        elif left in df.columns and right in df.columns:
            result[name] = (df[left].values + df[right].values) / 2.0
    return pd.DataFrame(result)


def _ensure_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Create figures/ and data/ subdirectories.

    Args:
        output_dir: Root output directory.

    Returns:
        Tuple of (figures_dir, data_dir).
    """
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, data_dir


# -----------------------------------------------------------------------
# Fig — NFE Scaling Comparison (2x3 grid)
# -----------------------------------------------------------------------


def plot_nfe_scaling_comparison(
    all_results: dict[str, ModelResults],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """NFE scaling curves comparing all models on a 2x3 metric grid.

    Each panel plots one metric vs NFE (log scale) with one line per model
    using ``METHOD_STYLES``. Bootstrap CI bands are added when available.

    Args:
        all_results: Mapping of model name to ``ModelResults``.
        output_dir: Root analysis output directory.
        nfe_levels: NFE values to plot on the x-axis.
    """
    apply_ieee_style()
    fig_dir, data_dir = _ensure_dirs(output_dir)

    fig, axes = plt.subplots(2, 3, figsize=get_figure_size("double", 0.55))
    axes_flat = axes.flatten()

    panels: list[tuple[str, str, str, str]] = [
        ("FID-3D $\\downarrow$", "fid_3d", "lower", "FID-3D"),
        ("MMD $\\downarrow$", "mmd", "lower", "MMD"),
        ("Coverage $\\uparrow$", "coverage", "upper", "Coverage"),
        ("Density $\\uparrow$", "density", "upper", "Density"),
        ("MS-SSIM $\\uparrow$", "ms_ssim", "upper", "MS-SSIM"),
        ("HF-Ratio $\\downarrow$", "hf_energy", "lower", "HF Energy Ratio"),
    ]

    npz_data: dict[str, np.ndarray] = {"nfe_levels": np.array(nfe_levels)}

    for i, (title, metric_key, better_dir, ylabel) in enumerate(panels):
        ax = axes_flat[i]

        for model_name, mr in all_results.items():
            style = METHOD_STYLES.get(model_name, {
                "color": PAUL_TOL_BRIGHT["grey"],
                "marker": "x",
                "ls": "-.",
            })

            vals: list[float] = []
            valid_nfes: list[int] = []
            for nfe in nfe_levels:
                v = _extract_metric(mr.metrics, nfe, metric_key)
                if v is not None:
                    vals.append(v)
                    valid_nfes.append(nfe)

            if not vals:
                continue

            ax.plot(
                valid_nfes, vals,
                color=style["color"],
                marker=style["marker"],
                ls=style["ls"],
                label=model_name,
                linewidth=1.5,
                markersize=6,
                zorder=3,
            )
            safe_name = model_name.lower().replace(" ", "_")
            npz_data[f"{safe_name}_{metric_key}"] = np.array(vals)
            npz_data[f"{safe_name}_{metric_key}_nfes"] = np.array(valid_nfes)

            # Bootstrap CI band
            if metric_key in mr.bootstrap_results:
                ci_lower = [
                    mr.bootstrap_results[metric_key]
                    .get(nfe, {})
                    .get("ci_lower", np.nan)
                    for nfe in valid_nfes
                ]
                ci_upper = [
                    mr.bootstrap_results[metric_key]
                    .get(nfe, {})
                    .get("ci_upper", np.nan)
                    for nfe in valid_nfes
                ]
                if not all(np.isnan(ci_lower)):
                    ax.fill_between(
                        valid_nfes, ci_lower, ci_upper,
                        alpha=0.15, color=style["color"],
                    )

        ax.set_xscale("log")
        ax.set_xticks(nfe_levels)
        ax.set_xticklabels([str(n) for n in nfe_levels])
        ax.set_xlabel("NFE")
        ax.set_ylabel(ylabel)
        ax.set_title(f"({chr(97 + i)}) {title}")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    np.savez(data_dir / "nfe_scaling_comparison.npz", **npz_data)
    save_figure(fig, fig_dir / "nfe_scaling_comparison")
    logger.info("Saved nfe_scaling_comparison")


# -----------------------------------------------------------------------
# Fig — Grouped Horizontal Bar Chart
# -----------------------------------------------------------------------


def plot_metric_bars_comparison(
    all_results: dict[str, ModelResults],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Grouped horizontal bar chart comparing models across metrics and NFEs.

    For each metric + NFE combination, draws one bar per model. Error bars
    are derived from bootstrap CIs when available. The best value per group
    is rendered in bold, and the second best is underlined.

    Args:
        all_results: Mapping of model name to ``ModelResults``.
        output_dir: Root analysis output directory.
        nfe_levels: NFE values to include.
    """
    apply_ieee_style()
    fig_dir, data_dir = _ensure_dirs(output_dir)

    metric_defs: list[tuple[str, str, str]] = [
        ("FID-3D", "fid_3d", "lower"),
        ("MMD", "mmd", "lower"),
        ("Coverage", "coverage", "upper"),
        ("Density", "density", "upper"),
        ("MS-SSIM", "ms_ssim", "upper"),
        ("HF-Ratio", "hf_energy", "lower"),
    ]

    model_names = list(all_results.keys())
    n_models = len(model_names)
    bar_height = 0.7 / max(n_models, 1)
    npz_data: dict[str, np.ndarray] = {}

    fig, axes = plt.subplots(
        2, 3,
        figsize=get_figure_size("double", 0.6),
    )
    axes_flat = axes.flatten()

    for panel_idx, (display_name, metric_key, better_dir) in enumerate(metric_defs):
        ax = axes_flat[panel_idx]

        y_labels: list[str] = []
        y_pos = 0.0
        positions: list[float] = []
        group_centers: list[tuple[float, str]] = []

        for nfe in nfe_levels:
            group_start = y_pos
            group_vals: dict[str, float | None] = {}

            for m_idx, model_name in enumerate(model_names):
                mr = all_results[model_name]
                val = _extract_metric(mr.metrics, nfe, metric_key)
                group_vals[model_name] = val

                # Error from bootstrap CI
                err = 0.0
                if (
                    metric_key in mr.bootstrap_results
                    and nfe in mr.bootstrap_results[metric_key]
                ):
                    bs = mr.bootstrap_results[metric_key][nfe]
                    err = (bs.get("ci_upper", val) - bs.get("ci_lower", val)) / 2

                style = METHOD_STYLES.get(model_name, {
                    "color": PAUL_TOL_BRIGHT["grey"],
                })

                if val is not None:
                    ax.barh(
                        y_pos, val,
                        height=bar_height,
                        xerr=err if err > 0 else None,
                        color=style["color"],
                        capsize=2,
                        edgecolor="white",
                        linewidth=0.3,
                        alpha=0.85,
                    )

                y_labels.append(model_name)
                positions.append(y_pos)
                y_pos += 1.0

            # Determine best and second best for annotation
            valid = {k: v for k, v in group_vals.items() if v is not None}
            if valid:
                if better_dir == "lower":
                    ranked = sorted(valid.items(), key=lambda x: x[1])
                else:
                    ranked = sorted(valid.items(), key=lambda x: -x[1])

                best_name = ranked[0][0] if len(ranked) > 0 else None
                second_name = ranked[1][0] if len(ranked) > 1 else None

                for m_idx, model_name in enumerate(model_names):
                    val = group_vals.get(model_name)
                    if val is None:
                        continue
                    bar_y = group_start + m_idx
                    fontweight = "bold" if model_name == best_name else "normal"
                    fontstyle = "italic" if model_name == second_name else "normal"
                    ax.text(
                        val * 1.02, bar_y,
                        f"{val:.2f}",
                        va="center", ha="left",
                        fontsize=5.5,
                        fontweight=fontweight,
                        fontstyle=fontstyle,
                    )

            group_centers.append(
                ((group_start + y_pos - 1.0) / 2, f"NFE={nfe}")
            )
            y_pos += 0.5  # gap between NFE groups

        ax.set_yticks(positions)
        ax.set_yticklabels(y_labels, fontsize=6)
        ax.set_xlabel(display_name)
        ax.set_title(f"({chr(97 + panel_idx)}) {display_name}")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

        # NFE group labels on right spine
        for center_y, nfe_label in group_centers:
            ax.annotate(
                nfe_label,
                xy=(1.0, center_y),
                xycoords=("axes fraction", "data"),
                fontsize=6, ha="left", va="center",
                color="grey",
            )

    # Shared legend at top
    legend_patches = [
        Patch(facecolor=METHOD_STYLES[m]["color"], label=m, alpha=0.85)
        for m in model_names
        if m in METHOD_STYLES
    ]
    if legend_patches:
        fig.legend(
            handles=legend_patches,
            loc="upper center",
            ncol=len(legend_patches),
            fontsize=7,
            framealpha=0.8,
            bbox_to_anchor=(0.5, 1.0),
        )
        fig.subplots_adjust(top=0.90)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    np.savez(data_dir / "metric_bars_comparison.npz", **npz_data)
    save_figure(fig, fig_dir / "metric_bars_comparison")
    logger.info("Saved metric_bars_comparison")


# -----------------------------------------------------------------------
# Fig — Regional Volume Box Plots (Real + 3 models)
# -----------------------------------------------------------------------


def plot_regional_volumes_comparison(
    all_results: dict[str, ModelResults],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Box plots of bilateral regional brain volumes across models.

    For each region in ``BILATERAL_REGIONS``, shows boxes for Real data plus
    each model at the best available NFE (highest in ``nfe_levels``). KS
    test significance stars are added above each model's box when the
    distribution differs significantly from real.

    Args:
        all_results: Mapping of model name to ``ModelResults``.
        output_dir: Root analysis output directory.
        nfe_levels: NFE values; the highest available is used.
    """
    apply_ieee_style()
    fig_dir, data_dir = _ensure_dirs(output_dir)

    # Find real volumes from first model that has them
    real_df: pd.DataFrame | None = None
    for mr in all_results.values():
        candidate = mr.synthseg_data.get("real_volumes")
        if candidate is not None:
            real_df = candidate
            break

    if real_df is None:
        logger.warning("No real SynthSeg volumes available -- skipping "
                       "regional volumes comparison")
        return

    real_bilat = _bilateral_average(real_df, BILATERAL_REGIONS)
    region_names = [r[2] for r in BILATERAL_REGIONS]

    # Collect per-model bilateral-averaged DataFrames at best NFE
    best_nfe = max(nfe_levels)
    model_bilats: dict[str, pd.DataFrame] = {}
    for model_name, mr in all_results.items():
        # Try best NFE, fall back to next available
        gen_df: pd.DataFrame | None = None
        for nfe_candidate in sorted(nfe_levels, reverse=True):
            gen_df = mr.synthseg_data.get(f"gen_volumes_{nfe_candidate}")
            if gen_df is not None:
                break
        if gen_df is not None:
            model_bilats[model_name] = _bilateral_average(
                gen_df, BILATERAL_REGIONS
            )

    if not model_bilats:
        logger.warning("No generated SynthSeg volumes found -- skipping "
                       "regional volumes comparison")
        return

    n_regions = len(region_names)
    n_groups = 1 + len(model_bilats)  # Real + models
    width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=get_figure_size("double", 0.5))

    positions_base = np.arange(n_regions, dtype=float)
    npz_data: dict[str, np.ndarray] = {}

    # Plot Real
    offset = -width * (n_groups - 1) / 2
    real_data = [
        real_bilat[r].values for r in region_names
        if r in real_bilat.columns
    ]
    if real_data:
        bp_real = ax.boxplot(
            real_data,
            positions=positions_base + offset,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp_real["boxes"]:
            patch.set_facecolor(PAUL_TOL_BRIGHT["blue"])
            patch.set_alpha(0.6)
        for median_line in bp_real["medians"]:
            median_line.set_color("black")
            median_line.set_linewidth(1.0)

        for j, r in enumerate(region_names):
            if r in real_bilat.columns:
                npz_data[f"real_{r.replace(' ', '_').lower()}"] = (
                    real_bilat[r].values
                )

    # Plot each model
    for g_idx, (model_name, gen_bilat) in enumerate(model_bilats.items()):
        style = METHOD_STYLES.get(model_name, {
            "color": PAUL_TOL_BRIGHT["grey"],
        })
        pos = positions_base + offset + width * (g_idx + 1)

        gen_data = [
            gen_bilat[r].values for r in region_names
            if r in gen_bilat.columns
        ]
        if not gen_data:
            continue

        bp_gen = ax.boxplot(
            gen_data,
            positions=pos,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
        )
        for patch in bp_gen["boxes"]:
            patch.set_facecolor(style["color"])
            patch.set_alpha(0.6)
        for median_line in bp_gen["medians"]:
            median_line.set_color("black")
            median_line.set_linewidth(1.0)

        safe_name = model_name.lower().replace(" ", "_")
        for j, r in enumerate(region_names):
            if r in gen_bilat.columns:
                npz_data[f"{safe_name}_{r.replace(' ', '_').lower()}"] = (
                    gen_bilat[r].values
                )

        # KS test significance stars
        y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
        for j, r in enumerate(region_names):
            if r not in real_bilat.columns or r not in gen_bilat.columns:
                continue
            real_vals = real_bilat[r].dropna().values
            gen_vals = gen_bilat[r].dropna().values
            if len(real_vals) < 3 or len(gen_vals) < 3:
                continue
            _, p_val = stats.ks_2samp(real_vals, gen_vals)
            stars = get_significance_stars(p_val)
            if stars != "n.s.":
                # Bonferroni correction
                p_corrected = min(p_val * n_regions, 1.0)
                stars_corr = get_significance_stars(p_corrected)
                if stars_corr != "n.s.":
                    ax.text(
                        pos[j], y_max * 0.97,
                        stars_corr,
                        ha="center", va="top",
                        fontsize=6,
                        color=style["color"],
                    )

    ax.set_xticks(positions_base)
    ax.set_xticklabels(region_names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Volume (mm$^3$)")
    ax.set_title("Regional Brain Volumes: Real vs Models")
    ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_items = [
        Patch(facecolor=PAUL_TOL_BRIGHT["blue"], alpha=0.6, label="Real")
    ]
    for model_name in model_bilats:
        style = METHOD_STYLES.get(model_name, {
            "color": PAUL_TOL_BRIGHT["grey"],
        })
        legend_items.append(
            Patch(facecolor=style["color"], alpha=0.6, label=model_name)
        )
    ax.legend(handles=legend_items, fontsize=7, loc="upper right")

    fig.tight_layout()
    np.savez(data_dir / "regional_volumes_comparison.npz", **npz_data)
    save_figure(fig, fig_dir / "regional_volumes_comparison")
    logger.info("Saved regional_volumes_comparison")


# -----------------------------------------------------------------------
# Fig — Sample Quality Grid (Real + 3 models)
# -----------------------------------------------------------------------


def plot_sample_grid_comparison(
    all_results: dict[str, ModelResults],
    output_dir: Path,
    nfe: int = 50,
    vol_indices: list[int] | None = None,
) -> None:
    """Qualitative sample comparison grid across models.

    Produces a 4-column (Real, NeuroiMF, MOTFM, DDPM) by 3-row
    (axial, coronal, sagittal) grid of center slices. Volumes are loaded
    from each model's results directory via ``load_volumes()``.

    Args:
        all_results: Mapping of model name to ``ModelResults``.
        output_dir: Root analysis output directory.
        nfe: NFE level to showcase.
        vol_indices: Volume indices to load; the first index is displayed.
            Defaults to ``[0]``.
    """
    from experiments.analysis.data_loader import load_volumes

    apply_ieee_style()
    fig_dir, data_dir = _ensure_dirs(output_dir)

    if vol_indices is None:
        vol_indices = [0]

    # Load volumes per model
    model_vols: dict[str, np.ndarray | None] = {}
    real_vol: np.ndarray | None = None

    for model_name, mr in all_results.items():
        vols = load_volumes(mr.results_dir, [nfe], vol_indices)
        if vols is None:
            model_vols[model_name] = None
            continue

        # Cache real volume from first model that has it
        if real_vol is None and "real" in vols:
            real_vol = vols["real"]

        gen_key = f"gen_{nfe}"
        model_vols[model_name] = vols.get(gen_key)

    if real_vol is None:
        logger.warning("No real volumes available -- skipping sample grid")
        return

    # Build column order: Real + models in METHOD_STYLES order
    col_order = ["Real"]
    col_volumes: dict[str, np.ndarray | None] = {"Real": real_vol}

    for model_name in METHOD_STYLES:
        if model_name in all_results:
            col_order.append(model_name)
            col_volumes[model_name] = model_vols.get(model_name)

    # Add any models not in METHOD_STYLES
    for model_name in all_results:
        if model_name not in col_order:
            col_order.append(model_name)
            col_volumes[model_name] = model_vols.get(model_name)

    n_cols = len(col_order)
    view_names = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(
        3, n_cols,
        figsize=get_figure_size("double", 0.55),
        facecolor="black",
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col, col_name in enumerate(col_order):
        vol_arr = col_volumes.get(col_name)
        if vol_arr is None:
            for row in range(3):
                axes[row, col].set_visible(False)
            continue

        # Extract single volume (first index)
        vol = vol_arr[0]
        # Remove channel dim if present
        while vol.ndim > 3:
            vol = vol[0]

        mid = [s // 2 for s in vol.shape]
        slices = [
            vol[mid[0], :, :],  # axial
            vol[:, mid[1], :],  # coronal
            vol[:, :, mid[2]],  # sagittal
        ]

        for row in range(3):
            ax = axes[row, col]
            ax.set_facecolor("black")
            sl = slices[row]
            fg = sl[sl > 0]
            vmin, vmax = (
                (np.percentile(fg, [1, 99])) if fg.size > 0 else (0, 1)
            )
            ax.imshow(sl.T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            ax.axis("off")

            if row == 0:
                ax.set_title(col_name, fontsize=8, color="white")
            if col == 0:
                ax.text(
                    -0.05, 0.5, view_names[row],
                    transform=ax.transAxes,
                    rotation=90, va="center", ha="right",
                    fontsize=8, color="white",
                )

    fig.suptitle(
        f"Sample Comparison (NFE={nfe})", fontsize=10, y=1.0, color="white"
    )
    fig.tight_layout(pad=0.3)
    save_figure(fig, fig_dir / "sample_grid_comparison")
    logger.info("Saved sample_grid_comparison")


# -----------------------------------------------------------------------
# Fig — Pairwise Bootstrap Deltas
# -----------------------------------------------------------------------


def plot_pairwise_bootstrap_deltas(
    paired_results: dict[
        tuple[str, str], dict[str, dict[int, dict[str, Any]]]
    ],
    output_dir: Path,
    nfe_levels: list[int],
) -> None:
    """Bootstrap delta distributions for pairwise model comparisons.

    For each model pair and metric, shows the distribution of bootstrap
    deltas (model_A - model_B) as a violin or KDE with a vertical zero line.
    Confidence intervals are shown as horizontal error bars.

    Args:
        paired_results: Nested dict structured as
            ``{(model_a, model_b): {metric_key: {nfe: delta_dict}}}``.
            Each ``delta_dict`` has keys ``"delta"``, ``"samples"``,
            ``"ci_lower"``, ``"ci_upper"``.
        output_dir: Root analysis output directory.
        nfe_levels: NFE values to include.
    """
    from scipy.stats import gaussian_kde

    apply_ieee_style()
    fig_dir, data_dir = _ensure_dirs(output_dir)

    pairs = list(paired_results.keys())
    metric_keys = sorted({
        mk
        for pair_data in paired_results.values()
        for mk in pair_data
    })

    if not pairs or not metric_keys:
        logger.warning("No paired bootstrap data -- skipping delta plot")
        return

    n_pairs = len(pairs)
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(
        n_pairs, n_metrics,
        figsize=get_figure_size("double", 0.35 * n_pairs),
        squeeze=False,
    )

    npz_data: dict[str, np.ndarray] = {}

    for row, (model_a, model_b) in enumerate(pairs):
        pair_data = paired_results[(model_a, model_b)]

        for col, metric_key in enumerate(metric_keys):
            ax = axes[row, col]

            if metric_key not in pair_data:
                ax.set_visible(False)
                continue

            nfe_data = pair_data[metric_key]

            for nfe in nfe_levels:
                if nfe not in nfe_data:
                    continue

                d = nfe_data[nfe]
                delta = d.get("delta", 0.0)
                samples = d.get("samples")
                ci_lower = d.get("ci_lower", delta)
                ci_upper = d.get("ci_upper", delta)
                color = NFE_COLORS.get(nfe, PAUL_TOL_BRIGHT["grey"])

                if samples is not None and len(samples) > 10:
                    samples = np.asarray(samples)
                    safe_pair = (
                        f"{model_a}_{model_b}".lower().replace(" ", "_")
                    )
                    npz_data[f"{safe_pair}_{metric_key}_nfe{nfe}"] = samples

                    # KDE
                    try:
                        kde = gaussian_kde(samples)
                        x_grid = np.linspace(
                            samples.min() * 1.1,
                            samples.max() * 1.1,
                            300,
                        )
                        density = kde(x_grid)
                        ax.fill_between(
                            x_grid, density,
                            alpha=0.2, color=color,
                        )
                        ax.plot(
                            x_grid, density,
                            color=color, lw=1.2,
                            label=f"NFE={nfe}",
                        )
                    except np.linalg.LinAlgError:
                        ax.axvline(
                            delta, color=color, lw=1.5,
                            label=f"NFE={nfe}",
                        )

                    # CI line at bottom
                    y_base = ax.get_ylim()[0]
                    ax.plot(
                        [ci_lower, ci_upper],
                        [y_base, y_base],
                        color=color, lw=2.5, solid_capstyle="round",
                    )
                    ax.plot(
                        delta, y_base, "o",
                        color=color, markersize=4, zorder=5,
                    )
                else:
                    # No samples: show point + CI
                    ax.errorbar(
                        delta, 0.5,
                        xerr=[[delta - ci_lower], [ci_upper - delta]],
                        fmt="o", color=color, capsize=3,
                        label=f"NFE={nfe}",
                    )

            # Zero reference line
            ax.axvline(0, color="grey", ls="--", lw=0.8, zorder=0)

            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(axis="x", alpha=0.15)

            if row == 0:
                ax.set_title(metric_key.replace("_", " ").upper(), fontsize=8)
            if row == n_pairs - 1:
                ax.set_xlabel(f"$\\Delta$ ({model_a} $-$ {model_b})",
                              fontsize=7)
            if col == 0:
                ax.set_ylabel(
                    f"{model_a} vs\n{model_b}",
                    fontsize=7, rotation=0, ha="right", va="center",
                    labelpad=40,
                )
            if row == 0 and col == n_metrics - 1:
                ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    np.savez(data_dir / "pairwise_bootstrap_deltas.npz", **npz_data)
    save_figure(fig, fig_dir / "pairwise_bootstrap_deltas")
    logger.info("Saved pairwise_bootstrap_deltas")
