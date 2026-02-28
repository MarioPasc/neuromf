"""CLI entry point for Phase 5 post-hoc analysis.

Produces publication-quality figures, LaTeX tables, and statistical tests
from Phase 5 evaluation results. All source data is saved as .npz files
to enable future cross-run comparison.

Usage::

    python experiments/cli/run_analysis.py \\
        --results-dir /path/to/NeuroiMF_version1_28022026/ \\
        --nfe 1 10 50 \\
        --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for experiments.* imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.analysis.data_loader import (
    DDPM_BASELINES,
    MOTFM_BASELINES,
    load_features,
    load_metrics_jsons,
    load_synthseg_data,
    load_synthseg_labels,
    load_volumes,
)
from experiments.analysis.figures import (
    BILATERAL_REGIONS,
    plot_bootstrap_fid,
    plot_coverage_density,
    plot_feature_tsne,
    plot_metric_bars,
    plot_nfe_scaling,
    plot_qc_violin,
    plot_regional_volumes,
    plot_sample_grid,
    plot_spectral_comparison,
    plot_synthseg_overlay,
)
from experiments.analysis.manifest import generate_manifest
from experiments.analysis.statistics import (
    bootstrap_metric,
    compute_cohens_d,
    compute_ks_tests,
    compute_per_volume_ci,
    compute_regional_stats,
    compute_wilcoxon_across_nfe,
)
from experiments.analysis.tables import (
    generate_main_results_table,
    generate_statistical_table,
    generate_synthseg_table,
)

logger = logging.getLogger(__name__)


def _setup_logging(log_dir: Path) -> None:
    """Configure rich console + file logging.

    Args:
        log_dir: Directory for log file output.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / "analysis.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    root.addHandler(fh)

    # Console handler — use rich if available
    try:
        from rich.logging import RichHandler
        ch = RichHandler(level=logging.INFO, show_time=True, show_path=False)
    except ImportError:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)


def _run_bootstrap(
    features: dict,
    nfe_levels: list[int],
    n_bootstrap: int,
    output_dir: Path,
) -> dict[str, dict[int, dict]]:
    """Run bootstrap CIs for distributional metrics.

    Args:
        features: Dict from load_features().
        nfe_levels: NFE values.
        n_bootstrap: Number of bootstrap samples.
        output_dir: Analysis output directory.

    Returns:
        Dict mapping metric name to {nfe: bootstrap_result}.
    """
    from neuromf.metrics.fid_3d import compute_fid_3d
    from neuromf.metrics.mmd import compute_mmd
    from neuromf.metrics.coverage_density import compute_coverage, compute_density

    real_feats = features.get("real")
    if real_feats is None:
        logger.warning("No real features — skipping bootstrap")
        return {}

    metric_fns = {
        "fid_3d": compute_fid_3d,
        "mmd": compute_mmd,
        "coverage": compute_coverage,
        "density": compute_density,
    }

    bootstrap_results: dict[str, dict[int, dict]] = {}

    for metric_name, metric_fn in metric_fns.items():
        bootstrap_results[metric_name] = {}
        for nfe in nfe_levels:
            gen_key = f"gen_{nfe}"
            if gen_key not in features:
                continue
            logger.info(
                "Bootstrap %s NFE=%d (n=%d)...", metric_name, nfe, n_bootstrap
            )
            t0 = time.time()
            result = bootstrap_metric(
                real_feats, features[gen_key], metric_fn,
                n_bootstrap=n_bootstrap,
            )
            elapsed = time.time() - t0
            logger.info(
                "  %s NFE=%d: %.3f [%.3f, %.3f] (%.1fs)",
                metric_name, nfe,
                result["point_estimate"],
                result["ci_lower"],
                result["ci_upper"],
                elapsed,
            )
            bootstrap_results[metric_name][nfe] = result

        # Save bootstrap samples to npz
        npz_data = {}
        for nfe, res in bootstrap_results[metric_name].items():
            npz_data[f"nfe{nfe}_samples"] = res["samples"]
            npz_data[f"nfe{nfe}_point"] = np.array([res["point_estimate"]])
            npz_data[f"nfe{nfe}_ci"] = np.array([res["ci_lower"], res["ci_upper"]])
        if npz_data:
            np.savez(output_dir / "data" / f"bootstrap_{metric_name}.npz", **npz_data)

    return bootstrap_results


def _compute_cohens_d_vs_motfm(
    metrics: dict[int, dict],
    bootstrap_results: dict[str, dict[int, dict]],
    nfe_levels: list[int],
) -> dict[str, dict[int, dict]]:
    """Compute Cohen's d for NeuroiMF vs MOTFM.

    Args:
        metrics: Dict mapping NFE to parsed metrics JSON.
        bootstrap_results: Bootstrap results dict.
        nfe_levels: NFE values to compare.

    Returns:
        Dict mapping metric name to {nfe: cohens_d_result}.
    """
    results: dict[str, dict[int, dict]] = {}

    metric_map = {
        "fid_3d": ("distributional", "fid_3d", "value"),
        "mmd": ("distributional", "mmd", "value"),
        "ms_ssim": ("per_volume", "ms_ssim", "mean"),
    }

    for metric_key, path in metric_map.items():
        results[metric_key] = {}
        for nfe in nfe_levels:
            if nfe not in metrics:
                continue
            motfm_val = MOTFM_BASELINES.get(nfe, {}).get(metric_key, None)
            if motfm_val is None:
                continue

            # NeuroiMF value and std
            m = metrics[nfe]
            neuro_val = m[path[0]][path[1]][path[2]]

            # Use bootstrap std if available, else per-volume std
            if metric_key in bootstrap_results and nfe in bootstrap_results[metric_key]:
                neuro_std = bootstrap_results[metric_key][nfe]["std"]
                n1 = 1000  # bootstrap samples
            elif "std" in m.get(path[0], {}).get(path[1], {}):
                neuro_std = m[path[0]][path[1]]["std"]
                n1 = m.get("n_generated", 2000)
            else:
                neuro_std = 0.1  # fallback
                n1 = 2000

            # MOTFM: no std available, assume comparable
            motfm_std = neuro_std  # conservative assumption
            n2 = n1

            results[metric_key][nfe] = compute_cohens_d(
                neuro_val, neuro_std, n1, motfm_val, motfm_std, n2
            )

    return results


def main() -> None:
    """Run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 5 post-hoc analysis pipeline",
    )
    parser.add_argument(
        "--results-dir", type=Path, required=True,
        help="Path to evaluation results directory",
    )
    parser.add_argument(
        "--nfe", type=int, nargs="+", default=[1, 10, 50],
        help="NFE levels to analyse (default: 1 10 50)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Number of bootstrap samples (default: 1000)",
    )
    parser.add_argument(
        "--skip-qualitative", action="store_true",
        help="Skip volume-dependent figures (fig08, fig09)",
    )
    parser.add_argument(
        "--skip-bootstrap", action="store_true",
        help="Skip slow bootstrap computation",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    nfe_levels = sorted(args.nfe)
    n_bootstrap = args.n_bootstrap

    # Create output directories
    analysis_dir = results_dir / "analysis"
    for subdir in ["data", "figures", "tables", "logs"]:
        (analysis_dir / subdir).mkdir(parents=True, exist_ok=True)

    _setup_logging(analysis_dir / "logs")
    logger.info("=" * 60)
    logger.info("Phase 5 Post-Hoc Analysis Pipeline")
    logger.info("Results dir: %s", results_dir)
    logger.info("NFE levels: %s", nfe_levels)
    logger.info("Bootstrap samples: %d", n_bootstrap)
    logger.info("=" * 60)

    skipped: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading data...")
    features = load_features(results_dir, nfe_levels)
    metrics = load_metrics_jsons(results_dir, nfe_levels)
    synthseg_data = load_synthseg_data(results_dir, nfe_levels)

    if not metrics:
        logger.error("No metrics JSONs found — cannot proceed")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Bootstrap CIs
    # ------------------------------------------------------------------
    bootstrap_results: dict[str, dict[int, dict]] = {}
    if args.skip_bootstrap:
        logger.info("Step 2: Skipping bootstrap (--skip-bootstrap)")
        skipped.append("bootstrap")
    elif features:
        logger.info("Step 2: Computing bootstrap CIs...")
        bootstrap_results = _run_bootstrap(
            features, nfe_levels, n_bootstrap, analysis_dir
        )
    else:
        logger.warning("Step 2: No features loaded — skipping bootstrap")
        skipped.append("bootstrap")

    # ------------------------------------------------------------------
    # 3. Per-volume stats and CIs
    # ------------------------------------------------------------------
    logger.info("Step 3: Computing per-volume stats...")
    per_volume_stats: dict[int, dict] = {}
    for nfe in nfe_levels:
        if nfe not in metrics:
            continue
        m = metrics[nfe]
        n_gen = m.get("n_generated", 2000)
        pvs = {}
        for metric_name in ["ms_ssim", "psnr_db"]:
            pv = m.get("per_volume", {}).get(metric_name, {})
            if "mean" in pv and "std" in pv:
                ci = compute_per_volume_ci(pv["mean"], pv["std"], n_gen)
                pvs[metric_name] = {
                    "mean": pv["mean"],
                    "std": pv["std"],
                    "ci_lower": ci[0],
                    "ci_upper": ci[1],
                    "n": n_gen,
                }
        per_volume_stats[nfe] = pvs

    # ------------------------------------------------------------------
    # 4. SynthSeg statistics
    # ------------------------------------------------------------------
    logger.info("Step 4: Computing SynthSeg statistics...")
    real_vol_df = synthseg_data.get("real_volumes")
    ks_results = None
    regional_stats = None
    wilcoxon_results = None

    if real_vol_df is not None:
        # Get all volume columns (exclude 'subject' and 'total intracranial')
        vol_columns = [
            c for c in real_vol_df.columns
            if c not in ("subject", "total intracranial")
        ]

        # KS tests against best NFE (50)
        best_nfe = max(nfe_levels)
        gen_vol_df = synthseg_data.get(f"gen_volumes_{best_nfe}")
        if gen_vol_df is not None:
            ks_results = compute_ks_tests(real_vol_df, gen_vol_df, vol_columns)
            regional_stats = compute_regional_stats(
                real_vol_df, gen_vol_df, vol_columns
            )
            logger.info("KS tests: %d regions tested", len(ks_results))

        # Wilcoxon across NFE
        gen_dfs = {}
        for nfe in nfe_levels:
            gdf = synthseg_data.get(f"gen_volumes_{nfe}")
            if gdf is not None:
                gen_dfs[nfe] = gdf

        if len(gen_dfs) >= 2:
            nfe_pairs = [(nfe_levels[0], nfe_levels[-1])]
            if len(nfe_levels) >= 3:
                nfe_pairs.append((nfe_levels[1], nfe_levels[-1]))
            wilcoxon_results = compute_wilcoxon_across_nfe(
                gen_dfs, nfe_pairs, vol_columns[:10]
            )

    # ------------------------------------------------------------------
    # 5. Cohen's d for MOTFM comparisons
    # ------------------------------------------------------------------
    logger.info("Step 5: Computing Cohen's d...")
    cohens_d_results = _compute_cohens_d_vs_motfm(
        metrics, bootstrap_results, nfe_levels
    )

    # ------------------------------------------------------------------
    # 6. Generate quantitative figures
    # ------------------------------------------------------------------
    logger.info("Step 6: Generating quantitative figures...")

    # Fig 01 — NFE Scaling
    plot_nfe_scaling(metrics, bootstrap_results, analysis_dir, nfe_levels)

    # Fig 02 — Bootstrap FID
    fid_bootstrap = bootstrap_results.get("fid_3d", {})
    if fid_bootstrap:
        plot_bootstrap_fid(fid_bootstrap, analysis_dir, nfe_levels)
    else:
        logger.warning("No FID bootstrap data — skipping fig02")
        skipped.append("fig02_bootstrap_fid")

    # Fig 03 — Feature t-SNE
    if features:
        plot_feature_tsne(features, analysis_dir, nfe_levels)
    else:
        logger.warning("No features — skipping fig03")
        skipped.append("fig03_feature_tsne")

    # Fig 04 — Spectral Comparison
    plot_spectral_comparison(metrics, analysis_dir, nfe_levels)

    # Fig 05 — Regional Volumes
    if real_vol_df is not None:
        plot_regional_volumes(synthseg_data, ks_results, analysis_dir, nfe_levels)
    else:
        logger.warning("No SynthSeg data — skipping fig05")
        skipped.append("fig05_regional_volumes")

    # Fig 06 — Coverage vs Density
    plot_coverage_density(metrics, bootstrap_results, analysis_dir, nfe_levels)

    # Fig 07 — QC Violin
    plot_qc_violin(synthseg_data, analysis_dir, nfe_levels)

    # Fig 10 — Metric Bars
    plot_metric_bars(
        metrics, bootstrap_results, cohens_d_results, analysis_dir, nfe_levels
    )

    # ------------------------------------------------------------------
    # 7. Qualitative figures (require decoded volumes)
    # ------------------------------------------------------------------
    if args.skip_qualitative:
        logger.info("Step 7: Skipping qualitative figures (--skip-qualitative)")
        skipped.extend(["fig08_synthseg_overlay", "fig09_sample_grid"])
    else:
        logger.info("Step 7: Attempting qualitative figures...")
        # Find representative volume indices for overlay
        sample_indices = [0, 10, 50]
        volumes = load_volumes(results_dir, nfe_levels, sample_indices)
        labels = load_synthseg_labels(results_dir, nfe_levels, sample_indices)

        if volumes is not None and labels is not None:
            plot_synthseg_overlay(volumes, labels, analysis_dir, nfe_levels)
        else:
            logger.warning("Volumes or labels unavailable — skipping fig08")
            skipped.append("fig08_synthseg_overlay")

        if volumes is not None:
            plot_sample_grid(volumes, metrics, analysis_dir, nfe=max(nfe_levels))
        else:
            logger.warning("Volumes unavailable — skipping fig09")
            skipped.append("fig09_sample_grid")

    # ------------------------------------------------------------------
    # 8. Generate LaTeX tables
    # ------------------------------------------------------------------
    logger.info("Step 8: Generating LaTeX tables...")
    table_dir = analysis_dir / "tables"

    # Table 1 — Main results
    table1 = generate_main_results_table(metrics, bootstrap_results, nfe_levels)
    (table_dir / "table1_main_results.tex").write_text(table1)
    logger.info("Wrote table1_main_results.tex")

    # Table 2 — SynthSeg
    if regional_stats is not None and ks_results is not None:
        table2 = generate_synthseg_table(
            regional_stats, ks_results, nfe=max(nfe_levels)
        )
        (table_dir / "table2_synthseg.tex").write_text(table2)
        logger.info("Wrote table2_synthseg.tex")
    else:
        skipped.append("table2_synthseg")

    # Table 3 — Statistical
    table3 = generate_statistical_table(
        bootstrap_results, per_volume_stats, cohens_d_results, nfe_levels
    )
    (table_dir / "table3_statistical.tex").write_text(table3)
    logger.info("Wrote table3_statistical.tex")

    # ------------------------------------------------------------------
    # 9. Write manifest
    # ------------------------------------------------------------------
    logger.info("Step 9: Writing manifest...")
    manifest = generate_manifest(
        results_dir, analysis_dir, nfe_levels, n_bootstrap, skipped
    )

    # ------------------------------------------------------------------
    # 10. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("Output directory: %s", analysis_dir)
    logger.info("Figures: %d generated", len(list((analysis_dir / "figures").glob("*.pdf"))))
    logger.info("Tables: %d generated", len(list((analysis_dir / "tables").glob("*.tex"))))
    logger.info("Data files: %d saved", len(list((analysis_dir / "data").glob("*.npz"))))
    if skipped:
        logger.info("Skipped: %s", ", ".join(skipped))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
