"""LaTeX table generators for Phase 5 analysis.

Produces IEEE TMI-quality tables comparing NeuroiMF against MOTFM and DDPM
baselines, SynthSeg regional volume statistics, and statistical tests.

Includes both single-model tables (``generate_main_results_table``, using
hardcoded baselines) and multi-model comparison tables
(``generate_comparison_results_table``) that use actual computed metrics
from all models.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from experiments.analysis.data_loader import DDPM_BASELINES, MOTFM_BASELINES

logger = logging.getLogger(__name__)


def _bold(text: str) -> str:
    """Wrap text in LaTeX bold."""
    return f"\\textbf{{{text}}}"


def _underline(text: str) -> str:
    """Wrap text in LaTeX underline."""
    return f"\\underline{{{text}}}"


def _format_metric(
    val: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    precision: int = 2,
) -> str:
    """Format a metric value with optional 95% CI.

    Args:
        val: Metric value.
        ci_lower: Lower CI bound (optional).
        ci_upper: Upper CI bound (optional).
        precision: Decimal precision.

    Returns:
        Formatted string like '6.14 [5.82, 6.51]'.
    """
    if ci_lower is not None and ci_upper is not None:
        return f"{val:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
    return f"{val:.{precision}f}"


def _format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """Format as mean +/- std."""
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def generate_main_results_table(
    metrics: dict[int, dict[str, Any]],
    bootstrap_results: dict[str, dict[int, dict[str, Any]]],
    nfe_levels: list[int],
) -> str:
    """Generate Table 1: main quantitative results.

    Compares NeuroiMF, MOTFM, and DDPM across all NFE levels.

    Args:
        metrics: Dict mapping NFE to parsed metrics JSON.
        bootstrap_results: Dict mapping metric name to {nfe: bootstrap_dict}.
        nfe_levels: NFE values to include.

    Returns:
        LaTeX table string.
    """
    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Quantitative evaluation of 3D brain MRI synthesis. "
        "Best results in \\textbf{bold}, second-best \\underline{underlined}. "
        "$\\downarrow$: lower is better, $\\uparrow$: higher is better.}\n"
        "\\label{tab:main_results}\n"
        "\\begin{tabular}{llccccccc}\n"
        "\\toprule\n"
        "Method & NFE & FID-3D $\\downarrow$ & MMD $\\downarrow$ & "
        "Coverage $\\uparrow$ & Density $\\uparrow$ & "
        "MS-SSIM $\\uparrow$ & PSNR $\\uparrow$ & HF Ratio $\\downarrow$ \\\\\n"
        "\\midrule\n"
    )

    rows: list[str] = []

    for nfe in nfe_levels:
        # Collect values for ranking: {method: {metric: val}}
        method_vals: dict[str, dict[str, float]] = {}

        # DDPM
        ddpm = DDPM_BASELINES.get(nfe, {})
        if ddpm:
            method_vals["DDPM"] = {
                "fid": ddpm.get("fid_3d", np.nan),
                "mmd": ddpm.get("mmd", np.nan),
                "ms_ssim": ddpm.get("ms_ssim", np.nan),
            }

        # MOTFM
        motfm = MOTFM_BASELINES.get(nfe, {})
        if motfm:
            method_vals["MOTFM"] = {
                "fid": motfm.get("fid_3d", np.nan),
                "mmd": motfm.get("mmd", np.nan),
                "ms_ssim": motfm.get("ms_ssim", np.nan),
            }

        # NeuroiMF
        if nfe in metrics:
            m = metrics[nfe]
            method_vals["NeuroiMF"] = {
                "fid": m["distributional"]["fid_3d"]["value"],
                "mmd": m["distributional"]["mmd"]["value"],
                "coverage": m["distributional"]["coverage_k5"]["value"],
                "density": m["distributional"]["density_k5"]["value"],
                "ms_ssim": m["per_volume"]["ms_ssim"]["mean"],
                "ms_ssim_std": m["per_volume"]["ms_ssim"]["std"],
                "psnr": m["per_volume"]["psnr_db"]["mean"],
                "psnr_std": m["per_volume"]["psnr_db"]["std"],
                "hf_ratio": m["spectral"]["hf_energy_ratio"]["mean"],
            }

        # Ranking for bold/underline (lower is better: fid, mmd, hf_ratio)
        def _rank(metric_key: str, lower_better: bool = True) -> dict[str, int]:
            vals = {
                k: v.get(metric_key, np.nan)
                for k, v in method_vals.items()
                if not np.isnan(v.get(metric_key, np.nan))
            }
            sorted_methods = sorted(vals, key=lambda x: vals[x], reverse=not lower_better)
            return {m: i for i, m in enumerate(sorted_methods)}

        fid_rank = _rank("fid", lower_better=True)
        mmd_rank = _rank("mmd", lower_better=True)
        ssim_rank = _rank("ms_ssim", lower_better=False)

        # Build rows
        for method in ["DDPM", "MOTFM", "NeuroiMF"]:
            if method not in method_vals:
                continue
            v = method_vals[method]

            # FID
            fid_val = v.get("fid", np.nan)
            if not np.isnan(fid_val):
                fid_str = f"{fid_val:.2f}"
                # Add CI for NeuroiMF
                if method == "NeuroiMF" and "fid_3d" in bootstrap_results:
                    bs = bootstrap_results["fid_3d"].get(nfe, {})
                    if bs:
                        fid_str = _format_metric(fid_val, bs.get("ci_lower"), bs.get("ci_upper"))
                if fid_rank.get(method) == 0:
                    fid_str = _bold(fid_str)
                elif fid_rank.get(method) == 1:
                    fid_str = _underline(fid_str)
            else:
                fid_str = "---"

            # MMD
            mmd_val = v.get("mmd", np.nan)
            if not np.isnan(mmd_val):
                mmd_str = f"{mmd_val:.2f}"
                if method == "NeuroiMF" and "mmd" in bootstrap_results:
                    bs = bootstrap_results["mmd"].get(nfe, {})
                    if bs:
                        mmd_str = _format_metric(mmd_val, bs.get("ci_lower"), bs.get("ci_upper"))
                if mmd_rank.get(method) == 0:
                    mmd_str = _bold(mmd_str)
                elif mmd_rank.get(method) == 1:
                    mmd_str = _underline(mmd_str)
            else:
                mmd_str = "---"

            # Coverage / Density (only NeuroiMF)
            cov_str = f"{v['coverage']:.2f}" if "coverage" in v else "---"
            den_str = f"{v['density']:.2f}" if "density" in v else "---"

            # MS-SSIM
            ssim_val = v.get("ms_ssim", np.nan)
            if not np.isnan(ssim_val):
                if "ms_ssim_std" in v:
                    ssim_str = _format_mean_std(ssim_val, v["ms_ssim_std"])
                else:
                    ssim_str = f"{ssim_val:.2f}"
                if ssim_rank.get(method) == 0:
                    ssim_str = _bold(ssim_str)
                elif ssim_rank.get(method) == 1:
                    ssim_str = _underline(ssim_str)
            else:
                ssim_str = "---"

            # PSNR
            psnr_str = (
                _format_mean_std(v["psnr"], v["psnr_std"])
                if "psnr" in v
                else "---"
            )

            # HF Ratio
            hf_str = f"{v['hf_ratio']:.4f}" if "hf_ratio" in v else "---"

            rows.append(
                f"{method} & {nfe} & {fid_str} & {mmd_str} & "
                f"{cov_str} & {den_str} & {ssim_str} & "
                f"{psnr_str} & {hf_str} \\\\"
            )

        rows.append("\\midrule")

    # Remove trailing midrule
    if rows and rows[-1] == "\\midrule":
        rows[-1] = "\\bottomrule"

    footer = "\\end{tabular}\n\\end{table*}"

    table = header + "\n".join(rows) + "\n" + footer
    return table


def generate_synthseg_table(
    synthseg_stats: pd.DataFrame,
    ks_results: pd.DataFrame,
    nfe: int = 50,
) -> str:
    """Generate Table 2: SynthSeg regional volume comparison.

    Args:
        synthseg_stats: Regional statistics DataFrame.
        ks_results: KS test results DataFrame.
        nfe: NFE level for generated data.

    Returns:
        LaTeX table string.
    """
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{SynthSeg regional volume comparison (NFE={nfe}). "
        "Volumes in mm$^3$.}}\n"
        "\\label{tab:synthseg}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "Region & Real & Generated & Pearson $r$ & KL & KS & Sig. \\\\\n"
        "\\midrule\n"
    )

    rows: list[str] = []
    for _, stat_row in synthseg_stats.iterrows():
        region = stat_row["region"]
        real_str = _format_mean_std(stat_row["real_mean"], stat_row["real_std"], 0)
        gen_str = _format_mean_std(stat_row["gen_mean"], stat_row["gen_std"], 0)
        r_str = f"{stat_row['pearson_r']:.3f}"
        kl_str = f"{stat_row['kl_divergence']:.3f}"

        # Find matching KS result
        ks_row = ks_results[ks_results["region"] == region]
        if not ks_row.empty:
            ks_stat = f"{ks_row.iloc[0]['statistic']:.3f}"
            stars = ks_row.iloc[0]["stars"]
        else:
            ks_stat = "---"
            stars = ""

        # Truncate long region names
        display_name = region.replace("left ", "L-").replace("right ", "R-")
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."

        rows.append(
            f"{display_name} & {real_str} & {gen_str} & "
            f"{r_str} & {kl_str} & {ks_stat} & {stars} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer


def generate_statistical_table(
    bootstrap_results: dict[str, dict[int, dict[str, Any]]],
    per_volume_stats: dict[int, dict[str, Any]],
    cohens_d_results: dict[str, dict[int, dict[str, Any]]],
    nfe_levels: list[int],
) -> str:
    """Generate Table 3: statistical analysis summary.

    Args:
        bootstrap_results: Dict mapping metric to {nfe: bootstrap_dict}.
        per_volume_stats: Dict mapping NFE to per-volume CI info.
        cohens_d_results: Dict mapping metric to {nfe: cohens_d_dict}.
        nfe_levels: NFE values to include.

    Returns:
        LaTeX table string.
    """
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Statistical analysis: NeuroiMF vs MOTFM.}\n"
        "\\label{tab:statistical}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{llccccl}\n"
        "\\toprule\n"
        "NFE & Metric & NeuroiMF & MOTFM & $\\Delta$ & Cohen's $d$ & Effect \\\\\n"
        "\\midrule\n"
    )

    rows: list[str] = []
    metric_labels = {
        "fid_3d": "FID-3D",
        "mmd": "MMD",
        "ms_ssim": "MS-SSIM",
    }

    for nfe in nfe_levels:
        for metric_key, metric_label in metric_labels.items():
            # NeuroiMF value with CI
            neuro_str = "---"
            neuro_val = np.nan
            if metric_key in bootstrap_results and nfe in bootstrap_results[metric_key]:
                bs = bootstrap_results[metric_key][nfe]
                neuro_val = bs["point_estimate"]
                neuro_str = _format_metric(
                    neuro_val, bs.get("ci_lower"), bs.get("ci_upper")
                )

            # MOTFM value
            motfm_val = MOTFM_BASELINES.get(nfe, {}).get(metric_key, np.nan)
            motfm_str = f"{motfm_val:.2f}" if not np.isnan(motfm_val) else "---"

            # Delta
            if not np.isnan(neuro_val) and not np.isnan(motfm_val):
                delta = neuro_val - motfm_val
                delta_str = f"{delta:+.2f}"
            else:
                delta_str = "---"

            # Cohen's d
            d_str = "---"
            effect_str = "---"
            if metric_key in cohens_d_results and nfe in cohens_d_results[metric_key]:
                cd = cohens_d_results[metric_key][nfe]
                d_str = f"{cd['d']:.2f}"
                effect_str = cd["interpretation"]

            rows.append(
                f"{nfe} & {metric_label} & {neuro_str} & {motfm_str} & "
                f"{delta_str} & {d_str} & {effect_str} \\\\"
            )

        if nfe != nfe_levels[-1]:
            rows.append("\\midrule")

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer


# ---------------------------------------------------------------------------
# Multi-model comparison tables (use actual computed data, not baselines)
# ---------------------------------------------------------------------------


def _extract_metric_value(
    metrics_json: dict[str, Any],
    section: str,
    key: str,
    subkey: str = "value",
) -> float:
    """Safely extract a metric from the nested JSON structure.

    Args:
        metrics_json: Parsed metrics JSON for one NFE level.
        section: Top-level section (``"distributional"``, ``"per_volume"``).
        key: Metric key (``"fid_3d"``, ``"mmd"``).
        subkey: Value key (``"value"``, ``"mean"``).

    Returns:
        Extracted float, or ``nan`` if path doesn't exist.
    """
    try:
        return float(metrics_json[section][key][subkey])
    except (KeyError, TypeError):
        return float("nan")


def generate_comparison_results_table(
    all_results: dict[str, Any],
    nfe_levels: list[int],
) -> str:
    """Generate main results table comparing all models with actual data.

    Unlike ``generate_main_results_table`` which uses hardcoded baselines
    for MOTFM/DDPM, this function reads computed metrics from all models'
    ``ModelResults.metrics`` dicts.

    Args:
        all_results: ``{model_name: ModelResults}`` with actual metrics.
        nfe_levels: NFE values to include.

    Returns:
        LaTeX ``table*`` string with bold best / underlined second-best.
    """
    model_names = list(all_results.keys())

    # Metric definitions: (section, key, subkey, label, lower_is_better, precision)
    metric_defs = [
        ("distributional", "fid_3d", "value", "FID-3D $\\downarrow$", True, 2),
        ("distributional", "mmd", "value", "MMD $\\downarrow$", True, 2),
        ("distributional", "coverage_k5", "value", "Cov $\\uparrow$", False, 2),
        ("distributional", "density_k5", "value", "Den $\\uparrow$", False, 2),
        ("per_volume", "ms_ssim", "mean", "MS-SSIM $\\uparrow$", False, 2),
        ("per_volume", "psnr_db", "mean", "PSNR $\\uparrow$", False, 1),
        ("spectral", "hf_energy_ratio", "mean", "HF $\\downarrow$", True, 4),
    ]

    col_labels = " & ".join(md[3] for md in metric_defs)
    n_cols = len(metric_defs)

    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Quantitative comparison of 3D brain MRI synthesis methods. "
        "Best in \\textbf{bold}, second-best \\underline{underlined}.}\n"
        "\\label{tab:comparison_results}\n"
        f"\\begin{{tabular}}{{ll{'c' * n_cols}}}\n"
        "\\toprule\n"
        f"Method & NFE & {col_labels} \\\\\n"
        "\\midrule\n"
    )

    rows: list[str] = []

    for nfe in nfe_levels:
        # Collect values for ranking
        all_vals: dict[int, dict[str, float]] = {}
        for mi in range(n_cols):
            all_vals[mi] = {}
            for model_name, mr in all_results.items():
                if nfe in mr.metrics:
                    sec, key, subkey = metric_defs[mi][:3]
                    val = _extract_metric_value(mr.metrics[nfe], sec, key, subkey)
                    if not np.isnan(val):
                        all_vals[mi][model_name] = val

        # Rank per metric
        ranks: dict[int, dict[str, int]] = {}
        for mi in range(n_cols):
            lower_better = metric_defs[mi][4]
            vals = all_vals[mi]
            sorted_models = sorted(
                vals, key=lambda m: vals[m], reverse=not lower_better
            )
            ranks[mi] = {m: i for i, m in enumerate(sorted_models)}

        for model_name in model_names:
            mr = all_results[model_name]
            if nfe not in mr.metrics:
                continue

            cells: list[str] = []
            for mi in range(n_cols):
                sec, key, subkey, _, _, prec = metric_defs[mi]
                val = _extract_metric_value(mr.metrics[nfe], sec, key, subkey)

                if np.isnan(val):
                    cells.append("---")
                    continue

                # Format with CI from bootstrap if available
                bs = mr.bootstrap_results.get(key, {}).get(nfe, {})
                if bs:
                    cell = _format_metric(
                        val, bs.get("ci_lower"), bs.get("ci_upper"), prec
                    )
                else:
                    cell = f"{val:.{prec}f}"

                # Bold/underline ranking
                rank = ranks[mi].get(model_name, 99)
                if rank == 0:
                    cell = _bold(cell)
                elif rank == 1:
                    cell = _underline(cell)

                cells.append(cell)

            row = f"{model_name} & {nfe} & " + " & ".join(cells) + " \\\\"
            rows.append(row)

        if nfe != nfe_levels[-1]:
            rows.append("\\midrule")

    footer = (
        "\\bottomrule\n"
        f"\\end{{tabular}}\n"
        "\\end{table*}"
    )

    return header + "\n".join(rows) + "\n" + footer


def generate_pairwise_statistical_table(
    paired_results: dict[tuple[str, str], dict[str, dict[int, dict[str, Any]]]],
    nfe_levels: list[int],
) -> str:
    """Generate table of pairwise statistical tests between models.

    Args:
        paired_results: ``{(model_a, model_b): {metric: {nfe: result_dict}}}``.
            Each ``result_dict`` has keys: ``delta``, ``ci_lower``,
            ``ci_upper``, ``p_value``.
        nfe_levels: NFE values to include.

    Returns:
        LaTeX table string with delta, CI, p-value, and significance stars.
    """
    from experiments.utils.settings import get_significance_stars

    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Pairwise statistical comparison (paired bootstrap, "
        "$n=1000$). Holm-Bonferroni corrected $p$-values.}\n"
        "\\label{tab:pairwise_stats}\n"
        "\\begin{tabular}{llccccl}\n"
        "\\toprule\n"
        "Comparison & NFE & Metric & $\\Delta$ & 95\\% CI & "
        "$p$-value & Sig. \\\\\n"
        "\\midrule\n"
    )

    metric_labels = {
        "fid_3d": "FID-3D",
        "mmd": "MMD",
        "coverage": "Coverage",
        "density": "Density",
    }

    rows: list[str] = []
    for (model_a, model_b), metric_results in paired_results.items():
        comparison = f"{model_a} vs {model_b}"
        first_row = True

        for nfe in nfe_levels:
            for metric_key, metric_label in metric_labels.items():
                res = metric_results.get(metric_key, {}).get(nfe)
                if res is None:
                    continue

                comp_str = comparison if first_row else ""
                first_row = False

                delta_str = f"{res['delta']:+.2f}"
                ci_str = f"[{res['ci_lower']:.2f}, {res['ci_upper']:.2f}]"
                p_str = f"{res['p_value']:.4f}"
                stars = get_significance_stars(res["p_value"])

                rows.append(
                    f"{comp_str} & {nfe} & {metric_label} & "
                    f"{delta_str} & {ci_str} & {p_str} & {stars} \\\\"
                )

        rows.append("\\midrule")

    # Remove trailing midrule
    if rows and rows[-1] == "\\midrule":
        rows[-1] = "\\bottomrule"

    footer = "\\end{tabular}\n\\end{table*}"

    return header + "\n".join(rows) + "\n" + footer


def generate_comparison_synthseg_table(
    all_results: dict[str, Any],
    nfe: int = 50,
) -> str:
    """Generate SynthSeg table comparing regional volumes across all models.

    Args:
        all_results: ``{model_name: ModelResults}`` with SynthSeg data.
        nfe: NFE level for generated data.

    Returns:
        LaTeX table string.
    """
    model_names = list(all_results.keys())

    # Build column headers
    model_cols = " & ".join(f"{m}" for m in model_names)
    n_model_cols = len(model_names)

    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{SynthSeg regional volumes (NFE={nfe}). "
        "Volumes in mm$^3$ (mean $\\pm$ std).}}\n"
        "\\label{tab:comparison_synthseg}\n"
        f"\\begin{{tabular}}{{l{'c' * (n_model_cols + 1)}}}\n"
        "\\toprule\n"
        f"Region & Real & {model_cols} \\\\\n"
        "\\midrule\n"
    )

    # Find real volumes from first model that has them
    real_df = None
    gen_dfs: dict[str, pd.DataFrame | None] = {}
    for model_name, mr in all_results.items():
        if real_df is None and mr.synthseg_data.get("real_volumes") is not None:
            real_df = mr.synthseg_data["real_volumes"]
        gen_dfs[model_name] = mr.synthseg_data.get(f"gen_volumes_{nfe}")

    if real_df is None:
        return "% No SynthSeg data available\n"

    # Get volume columns
    skip_cols = {"subject", "total intracranial"}
    vol_columns = [c for c in real_df.columns if c.lower() not in skip_cols]

    rows: list[str] = []
    for region in vol_columns[:15]:  # Limit to avoid overly long table
        real_vals = real_df[region].dropna()
        real_str = _format_mean_std(
            float(real_vals.mean()), float(real_vals.std()), precision=0
        )

        model_strs: list[str] = []
        for model_name in model_names:
            gdf = gen_dfs.get(model_name)
            if gdf is not None and region in gdf.columns:
                gen_vals = gdf[region].dropna()
                model_strs.append(
                    _format_mean_std(
                        float(gen_vals.mean()),
                        float(gen_vals.std()),
                        precision=0,
                    )
                )
            else:
                model_strs.append("---")

        display_name = region.replace("left ", "L-").replace("right ", "R-")
        if len(display_name) > 22:
            display_name = display_name[:19] + "..."

        row_cells = " & ".join(model_strs)
        rows.append(f"{display_name} & {real_str} & {row_cells} \\\\")

    footer = (
        "\\bottomrule\n"
        f"\\end{{tabular}}\n"
        "\\end{table*}"
    )

    return header + "\n".join(rows) + "\n" + footer
