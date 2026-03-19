"""Multi-model comparison entry point for IEEE TMI publication.

Orchestrates a unified comparison of NeuroiMF against any number of
pixel-space competitor models.  Produces publication-quality figures,
LaTeX tables, bootstrap confidence intervals, paired statistical tests,
and a manifest JSON cataloguing every artifact.

The pipeline is model-agnostic: competitors are specified via repeatable
``--competitor NAME:PATH`` flags.  NeuroiMF is special-cased (it always
appears and uses a dedicated flag) because it operates in latent space.

Usage::

    # Standard 3-model comparison:
    python experiments/cli/run_comparison.py \\
        --neuroimf-dir /path/to/NeuroiMF/ \\
        --competitor MOTFM:/path/to/MOTFM/ \\
        --competitor DDPM:/path/to/DDPM/ \\
        --output-dir /path/to/comparison/ \\
        --nfe 1 10 50

    # Adding a 4th model — no code changes required:
    python experiments/cli/run_comparison.py \\
        --neuroimf-dir /path/to/NeuroiMF/ \\
        --competitor MOTFM:/path/to/MOTFM/ \\
        --competitor DDPM:/path/to/DDPM/ \\
        --competitor ScoreSDE:/path/to/ScoreSDE/ \\
        --output-dir /path/to/comparison/

    # Backward-compatible aliases still work:
    python experiments/cli/run_comparison.py \\
        --neuroimf-dir /path/to/NeuroiMF/ \\
        --motfm-dir /path/to/MOTFM/ \\
        --ddpm-dir /path/to/DDPM/ \\
        --output-dir /path/to/comparison/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path for experiments.* imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.analysis.data_loader import (
    ModelResults,
    load_multi_model_results,
    load_volumes,
)
from experiments.analysis.statistics import (
    bootstrap_metric,
    paired_bootstrap_test,
    holm_bonferroni_correction,
    compute_friedman_test,
    compute_nemenyi_posthoc,
    compute_ks_tests,
    compute_regional_stats,
    compute_per_volume_ci,
)
from experiments.analysis.comparison_figures import (
    plot_nfe_scaling_comparison,
    plot_metric_bars_comparison,
    plot_regional_volumes_comparison,
    plot_sample_grid_comparison,
    plot_pairwise_bootstrap_deltas,
)
from experiments.analysis.tables import (
    generate_comparison_results_table,
    generate_pairwise_statistical_table,
    generate_comparison_synthseg_table,
)
from neuromf.metrics.fid_3d import compute_fid_3d
from neuromf.metrics.mmd import compute_mmd
from neuromf.metrics.coverage_density import compute_coverage, compute_density

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


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

    # Console handler -- use rich if available
    try:
        from rich.logging import RichHandler
        ch = RichHandler(level=logging.INFO, show_time=True, show_path=False)
    except ImportError:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Model-pair helper
# ---------------------------------------------------------------------------


def _ordered_model_pairs(
    model_names: list[str],
) -> list[tuple[str, str]]:
    """Generate all unique pairwise combinations of loaded models.

    NeuroiMF is placed first when present, so its pairwise comparisons
    appear before competitor-vs-competitor pairs.

    Args:
        model_names: Available model names (order preserved).

    Returns:
        List of ``(model_a, model_b)`` tuples.
    """
    # Ensure NeuroiMF comes first if present
    ordered = sorted(
        model_names,
        key=lambda n: (0 if n == "NeuroiMF" else 1, n),
    )
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Step 2: Validate real features consistency
# ---------------------------------------------------------------------------


def _validate_real_features(
    all_results: dict[str, ModelResults],
) -> bool:
    """Check that all models used the same real test set.

    Compares real feature tensor shapes across models. Logs a warning if
    shapes differ (which would indicate evaluation set mismatch).

    Args:
        all_results: Loaded model results.

    Returns:
        True if all real feature shapes match (or only one model has them).
    """
    real_shapes: dict[str, tuple[int, ...]] = {}
    for model_name, mr in all_results.items():
        real = mr.features.get("real")
        if real is not None:
            real_shapes[model_name] = tuple(real.shape)

    if len(real_shapes) <= 1:
        return True

    shapes = list(real_shapes.values())
    if all(s == shapes[0] for s in shapes):
        logger.info(
            "Real features consistent across all models: %s", shapes[0]
        )
        return True

    logger.warning(
        "Real feature shapes DIFFER across models -- results may not be "
        "comparable: %s",
        real_shapes,
    )
    return False


# ---------------------------------------------------------------------------
# Step 3: Bootstrap CIs for all models
# ---------------------------------------------------------------------------

_METRIC_FNS: dict[str, Any] = {
    "fid_3d": compute_fid_3d,
    "mmd": compute_mmd,
    "coverage": compute_coverage,
    "density": compute_density,
}


def _run_bootstrap_all_models(
    all_results: dict[str, ModelResults],
    nfe_levels: list[int],
    n_bootstrap: int,
    data_dir: Path,
) -> None:
    """Compute bootstrap CIs for every model and save .npz artefacts.

    Populates ``ModelResults.bootstrap_results`` in-place.

    Args:
        all_results: Loaded model results (mutated in-place).
        nfe_levels: NFE values to bootstrap.
        n_bootstrap: Number of bootstrap samples.
        data_dir: Directory for .npz output.
    """
    for model_name, mr in all_results.items():
        real_feats = mr.features.get("real")
        if real_feats is None:
            logger.warning(
                "No real features for %s -- skipping bootstrap", model_name
            )
            continue

        for metric_name, metric_fn in _METRIC_FNS.items():
            mr.bootstrap_results.setdefault(metric_name, {})
            for nfe in nfe_levels:
                gen_key = f"gen_{nfe}"
                if gen_key not in mr.features:
                    continue

                logger.info(
                    "Bootstrap %s / %s NFE=%d (n=%d)...",
                    model_name, metric_name, nfe, n_bootstrap,
                )
                t0 = time.time()
                result = bootstrap_metric(
                    real_feats,
                    mr.features[gen_key],
                    metric_fn,
                    n_bootstrap=n_bootstrap,
                )
                elapsed = time.time() - t0
                logger.info(
                    "  %s / %s NFE=%d: %.3f [%.3f, %.3f] (%.1fs)",
                    model_name, metric_name, nfe,
                    result["point_estimate"],
                    result["ci_lower"],
                    result["ci_upper"],
                    elapsed,
                )
                mr.bootstrap_results[metric_name][nfe] = result

            # Save per-model per-metric .npz
            bs = mr.bootstrap_results[metric_name]
            npz_data: dict[str, np.ndarray] = {}
            for nfe, res in bs.items():
                npz_data[f"nfe{nfe}_samples"] = res["samples"]
                npz_data[f"nfe{nfe}_point"] = np.array(
                    [res["point_estimate"]]
                )
                npz_data[f"nfe{nfe}_ci"] = np.array(
                    [res["ci_lower"], res["ci_upper"]]
                )
            if npz_data:
                safe_name = model_name.lower().replace(" ", "_")
                np.savez(
                    data_dir / f"bootstrap_{safe_name}_{metric_name}.npz",
                    **npz_data,
                )


# ---------------------------------------------------------------------------
# Step 4: Paired bootstrap tests
# ---------------------------------------------------------------------------


def _run_paired_bootstrap_tests(
    all_results: dict[str, ModelResults],
    nfe_levels: list[int],
    n_bootstrap: int,
    data_dir: Path,
) -> dict[tuple[str, str], dict[str, dict[int, dict[str, Any]]]]:
    """Paired bootstrap tests for every model pair, metric, and NFE.

    After collecting all raw p-values, applies Holm-Bonferroni correction
    to control FWER.

    Args:
        all_results: Loaded model results (must have features).
        nfe_levels: NFE values to test.
        n_bootstrap: Number of bootstrap samples.
        data_dir: Directory for .npz output.

    Returns:
        Nested dict ``{(model_a, model_b): {metric: {nfe: result_dict}}}``.
    """
    model_names = list(all_results.keys())
    pairs = _ordered_model_pairs(model_names)

    paired_results: dict[
        tuple[str, str], dict[str, dict[int, dict[str, Any]]]
    ] = {}

    # Accumulate raw p-values for correction
    raw_p_values: list[float] = []
    p_value_refs: list[tuple[tuple[str, str], str, int]] = []

    for model_a, model_b in pairs:
        mr_a = all_results[model_a]
        mr_b = all_results[model_b]

        # Need real features and gen features from both
        real_feats = mr_a.features.get("real")
        if real_feats is None:
            real_feats = mr_b.features.get("real")
        if real_feats is None:
            logger.warning(
                "No real features for %s vs %s -- skipping paired test",
                model_a, model_b,
            )
            continue

        pair_key = (model_a, model_b)
        paired_results[pair_key] = {}

        for metric_name, metric_fn in _METRIC_FNS.items():
            paired_results[pair_key][metric_name] = {}

            for nfe in nfe_levels:
                gen_a = mr_a.features.get(f"gen_{nfe}")
                gen_b = mr_b.features.get(f"gen_{nfe}")
                if gen_a is None or gen_b is None:
                    continue

                logger.info(
                    "Paired bootstrap %s vs %s / %s NFE=%d...",
                    model_a, model_b, metric_name, nfe,
                )
                t0 = time.time()
                result = paired_bootstrap_test(
                    real_feats, gen_a, gen_b, metric_fn,
                    n_bootstrap=n_bootstrap,
                )
                elapsed = time.time() - t0
                logger.info(
                    "  delta=%.3f [%.3f, %.3f] p=%.4f (%.1fs)",
                    result["delta"],
                    result["ci_lower"],
                    result["ci_upper"],
                    result["p_value"],
                    elapsed,
                )

                paired_results[pair_key][metric_name][nfe] = result
                raw_p_values.append(result["p_value"])
                p_value_refs.append((pair_key, metric_name, nfe))

    # Apply Holm-Bonferroni correction across all tests
    if raw_p_values:
        corrected = holm_bonferroni_correction(raw_p_values)
        for (pair_key, metric_name, nfe), p_corrected in zip(
            p_value_refs, corrected
        ):
            paired_results[pair_key][metric_name][nfe][
                "p_value_corrected"
            ] = p_corrected

        logger.info(
            "Holm-Bonferroni correction applied to %d tests", len(corrected)
        )

    # Save paired bootstrap samples
    for pair_key, metric_results in paired_results.items():
        safe_pair = f"{pair_key[0]}_{pair_key[1]}".lower().replace(" ", "_")
        npz_data: dict[str, np.ndarray] = {}
        for metric_name, nfe_results in metric_results.items():
            for nfe, res in nfe_results.items():
                if "samples" in res:
                    npz_data[f"{metric_name}_nfe{nfe}_deltas"] = res["samples"]
                    npz_data[f"{metric_name}_nfe{nfe}_delta"] = np.array(
                        [res["delta"]]
                    )
                    npz_data[f"{metric_name}_nfe{nfe}_ci"] = np.array(
                        [res["ci_lower"], res["ci_upper"]]
                    )
        if npz_data:
            np.savez(
                data_dir / f"paired_bootstrap_{safe_pair}.npz", **npz_data
            )

    return paired_results


# ---------------------------------------------------------------------------
# Step 5: Per-volume CIs and SynthSeg stats
# ---------------------------------------------------------------------------


def _compute_per_volume_stats(
    all_results: dict[str, ModelResults],
    nfe_levels: list[int],
) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
    """Compute per-volume CIs from metrics JSONs for all models.

    Args:
        all_results: Loaded model results.
        nfe_levels: NFE values.

    Returns:
        ``{model_name: {nfe: {metric: {mean, std, ci_lower, ci_upper, n}}}}``.
    """
    all_stats: dict[str, dict[int, dict[str, dict[str, float]]]] = {}

    for model_name, mr in all_results.items():
        model_stats: dict[int, dict[str, dict[str, float]]] = {}
        for nfe in nfe_levels:
            if nfe not in mr.metrics:
                continue
            m = mr.metrics[nfe]
            n_gen = m.get("n_generated", 2000)
            pvs: dict[str, dict[str, float]] = {}
            for metric_name in ["ms_ssim", "psnr_db"]:
                pv = m.get("per_volume", {}).get(metric_name, {})
                if "mean" in pv and "std" in pv:
                    ci = compute_per_volume_ci(pv["mean"], pv["std"], n_gen)
                    pvs[metric_name] = {
                        "mean": pv["mean"],
                        "std": pv["std"],
                        "ci_lower": ci[0],
                        "ci_upper": ci[1],
                        "n": float(n_gen),
                    }
            model_stats[nfe] = pvs
        all_stats[model_name] = model_stats

    return all_stats


# ---------------------------------------------------------------------------
# Step 6: Friedman test + Nemenyi
# ---------------------------------------------------------------------------


def _run_friedman_nemenyi(
    all_results: dict[str, ModelResults],
    nfe_levels: list[int],
) -> dict[str, Any]:
    """Run Friedman test + Nemenyi post-hoc on per-volume MS-SSIM.

    Requires per-volume metrics from all 3 models at a common NFE. The test
    is only meaningful when volumes are paired (same NN seed). Skips
    gracefully if data is missing.

    Args:
        all_results: Loaded model results.
        nfe_levels: NFE values to check.

    Returns:
        Dict with ``friedman`` and ``nemenyi`` results, or empty if skipped.
    """
    # Try the highest NFE first (most likely to have good metrics)
    for nfe in sorted(nfe_levels, reverse=True):
        per_vol: dict[str, np.ndarray] = {}
        for model_name, mr in all_results.items():
            m = mr.metrics.get(nfe, {})
            pv = m.get("per_volume", {}).get("ms_ssim", {})
            if "values" in pv:
                per_vol[model_name] = np.array(pv["values"])
            elif "mean" in pv and "std" in pv:
                # Cannot run Friedman without individual values
                continue

        if len(per_vol) >= 3:
            logger.info(
                "Running Friedman test with %d models at NFE=%d",
                len(per_vol), nfe,
            )
            friedman = compute_friedman_test(per_vol)
            logger.info(
                "Friedman: chi2=%.3f, p=%.4f (n=%d volumes)",
                friedman["statistic"],
                friedman["p_value"],
                friedman["n_volumes"],
            )

            nemenyi_df = None
            if friedman["p_value"] < 0.05:
                nemenyi_df = compute_nemenyi_posthoc(per_vol)
                logger.info(
                    "Nemenyi post-hoc:\n%s", nemenyi_df.to_string()
                )

            return {
                "nfe": nfe,
                "friedman": friedman,
                "nemenyi": nemenyi_df,
                "n_models": len(per_vol),
            }

    logger.info(
        "Friedman test skipped -- per-volume metric arrays not available "
        "from all models"
    )
    return {}


# ---------------------------------------------------------------------------
# Step 9: Manifest
# ---------------------------------------------------------------------------


def _write_comparison_manifest(
    output_dir: Path,
    model_dirs: dict[str, Path],
    nfe_levels: list[int],
    n_bootstrap: int,
    skipped: list[str],
    elapsed_total: float,
) -> dict[str, Any]:
    """Write comparison_manifest.json cataloguing all outputs.

    Args:
        output_dir: Root output directory.
        model_dirs: ``{model_name: results_dir}`` as provided by the user.
        nfe_levels: NFE levels analysed.
        n_bootstrap: Number of bootstrap samples.
        skipped: List of skipped steps/figures.
        elapsed_total: Total wall-clock time in seconds.

    Returns:
        Manifest dict (also written to disk).
    """
    manifest: dict[str, Any] = {
        "version": "2.0",
        "type": "multi_model_comparison",
        "created": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "models": {
            name: str(path) for name, path in model_dirs.items()
        },
        "nfe_levels": nfe_levels,
        "n_bootstrap": n_bootstrap,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed_total, 1),
    }

    # Data files
    data_dir = output_dir / "data"
    if data_dir.exists():
        manifest["data_files"] = {
            p.stem: str(p) for p in sorted(data_dir.glob("*.npz"))
        }
    else:
        manifest["data_files"] = {}

    # Figures
    fig_dir = output_dir / "figures"
    if fig_dir.exists():
        manifest["figures"] = {}
        for pdf in sorted(fig_dir.glob("*.pdf")):
            manifest["figures"][pdf.stem] = {
                "pdf": str(pdf),
                "png": str(pdf.with_suffix(".png")),
            }
    else:
        manifest["figures"] = {}

    # Tables
    table_dir = output_dir / "tables"
    if table_dir.exists():
        manifest["tables"] = {
            p.stem: str(p) for p in sorted(table_dir.glob("*.tex"))
        }
    else:
        manifest["tables"] = {}

    manifest_path = output_dir / "comparison_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full multi-model comparison pipeline."""
    parser = argparse.ArgumentParser(
        description="Multi-model comparison pipeline for IEEE TMI",
    )
    parser.add_argument(
        "--neuroimf-dir", type=Path, required=True,
        help="Path to NeuroiMF evaluation results directory",
    )
    parser.add_argument(
        "--competitor", type=str, action="append", default=[],
        metavar="NAME:PATH",
        help="Competitor model as NAME:PATH (repeatable). "
        "E.g. --competitor MOTFM:/path/to/MOTFM",
    )
    # Backward-compatible aliases
    parser.add_argument(
        "--motfm-dir", type=Path, default=None,
        help="(Legacy alias) Path to MOTFM results. Equivalent to "
        "--competitor MOTFM:PATH.",
    )
    parser.add_argument(
        "--ddpm-dir", type=Path, default=None,
        help="(Legacy alias) Path to DDPM results. Equivalent to "
        "--competitor DDPM:PATH.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Path to comparison output directory",
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
        "--skip-bootstrap", action="store_true",
        help="Skip all bootstrap computations (CIs and paired tests)",
    )
    parser.add_argument(
        "--skip-qualitative", action="store_true",
        help="Skip volume-dependent qualitative figures",
    )
    args = parser.parse_args()

    nfe_levels = sorted(args.nfe)
    n_bootstrap = args.n_bootstrap
    output_dir = args.output_dir.resolve()

    # Build model_dirs: NeuroiMF always first, then competitors
    model_dirs: dict[str, Path] = {
        "NeuroiMF": args.neuroimf_dir.resolve(),
    }

    # Legacy aliases
    if args.motfm_dir is not None:
        model_dirs["MOTFM"] = args.motfm_dir.resolve()
    if args.ddpm_dir is not None:
        model_dirs["DDPM"] = args.ddpm_dir.resolve()

    # New --competitor NAME:PATH entries
    for entry in args.competitor:
        if ":" not in entry:
            logger.error(
                "Invalid --competitor format '%s'. Expected NAME:PATH.", entry
            )
            sys.exit(1)
        name, path_str = entry.split(":", 1)
        model_dirs[name] = Path(path_str).resolve()

    if len(model_dirs) < 2:
        logger.error(
            "At least one competitor is required. Use --competitor NAME:PATH "
            "or --motfm-dir / --ddpm-dir."
        )
        sys.exit(1)

    # Create output directories
    for subdir in ["data", "figures", "tables", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    _setup_logging(output_dir / "logs")

    logger.info("=" * 70)
    logger.info("Multi-Model Comparison Pipeline (IEEE TMI)")
    logger.info("=" * 70)
    for name, path in model_dirs.items():
        logger.info("  %s: %s", name, path)
    logger.info("Output dir: %s", output_dir)
    logger.info("NFE levels: %s", nfe_levels)
    logger.info("Bootstrap samples: %d", n_bootstrap)
    logger.info("Skip bootstrap: %s", args.skip_bootstrap)
    logger.info("Skip qualitative: %s", args.skip_qualitative)
    logger.info("=" * 70)

    t_start = time.time()
    skipped: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Load data from all 3 model directories
    # ------------------------------------------------------------------
    logger.info("Step 1: Loading data from all models...")
    t0 = time.time()
    all_results = load_multi_model_results(model_dirs, nfe_levels)
    logger.info("Step 1 complete (%.1fs)", time.time() - t0)

    n_with_metrics = sum(1 for mr in all_results.values() if mr.metrics)
    if n_with_metrics == 0:
        logger.error("No models have metrics JSONs -- cannot proceed")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Validate real features consistency
    # ------------------------------------------------------------------
    logger.info("Step 2: Validating real features consistency...")
    _validate_real_features(all_results)

    # ------------------------------------------------------------------
    # Step 3: Bootstrap CIs for all 3 models
    # ------------------------------------------------------------------
    if args.skip_bootstrap:
        logger.info("Step 3: Skipping bootstrap CIs (--skip-bootstrap)")
        skipped.append("bootstrap_cis")
    else:
        has_features = any(mr.features for mr in all_results.values())
        if has_features:
            logger.info("Step 3: Computing bootstrap CIs for all models...")
            t0 = time.time()
            _run_bootstrap_all_models(
                all_results, nfe_levels, n_bootstrap, output_dir / "data"
            )
            logger.info("Step 3 complete (%.1fs)", time.time() - t0)
        else:
            logger.warning(
                "Step 3: No features loaded for any model -- skipping "
                "bootstrap"
            )
            skipped.append("bootstrap_cis")

    # ------------------------------------------------------------------
    # Step 4: Paired bootstrap tests
    # ------------------------------------------------------------------
    paired_results: dict[
        tuple[str, str], dict[str, dict[int, dict[str, Any]]]
    ] = {}

    if args.skip_bootstrap:
        logger.info(
            "Step 4: Skipping paired bootstrap tests (--skip-bootstrap)"
        )
        skipped.append("paired_bootstrap")
    else:
        has_gen_features = any(
            any(f"gen_{nfe}" in mr.features for nfe in nfe_levels)
            for mr in all_results.values()
        )
        if has_gen_features:
            logger.info(
                "Step 4: Running paired bootstrap tests..."
            )
            t0 = time.time()
            paired_results = _run_paired_bootstrap_tests(
                all_results, nfe_levels, n_bootstrap, output_dir / "data"
            )
            logger.info("Step 4 complete (%.1fs)", time.time() - t0)
        else:
            logger.warning(
                "Step 4: No generated features available -- skipping "
                "paired tests"
            )
            skipped.append("paired_bootstrap")

    # ------------------------------------------------------------------
    # Step 5: Per-volume CIs and SynthSeg stats
    # ------------------------------------------------------------------
    logger.info("Step 5: Computing per-volume CIs and SynthSeg stats...")
    per_volume_stats = _compute_per_volume_stats(all_results, nfe_levels)

    # SynthSeg KS tests and regional stats per model (at best NFE)
    best_nfe = max(nfe_levels)
    synthseg_ks: dict[str, Any] = {}
    synthseg_regional: dict[str, Any] = {}

    for model_name, mr in all_results.items():
        real_vol_df = mr.synthseg_data.get("real_volumes")
        gen_vol_df = mr.synthseg_data.get(f"gen_volumes_{best_nfe}")
        if real_vol_df is not None and gen_vol_df is not None:
            vol_columns = [
                c for c in real_vol_df.columns
                if c not in ("subject", "total intracranial")
            ]
            synthseg_ks[model_name] = compute_ks_tests(
                real_vol_df, gen_vol_df, vol_columns
            )
            synthseg_regional[model_name] = compute_regional_stats(
                real_vol_df, gen_vol_df, vol_columns
            )
            logger.info(
                "SynthSeg KS tests for %s: %d regions",
                model_name, len(synthseg_ks[model_name]),
            )

    # ------------------------------------------------------------------
    # Step 6: Friedman test + Nemenyi
    # ------------------------------------------------------------------
    logger.info("Step 6: Attempting Friedman + Nemenyi tests...")
    friedman_results = _run_friedman_nemenyi(all_results, nfe_levels)
    if not friedman_results:
        skipped.append("friedman_nemenyi")

    # Save Friedman/Nemenyi results if computed
    if friedman_results:
        friedman_out: dict[str, Any] = {
            "nfe": friedman_results["nfe"],
            "friedman_statistic": friedman_results["friedman"]["statistic"],
            "friedman_p_value": friedman_results["friedman"]["p_value"],
            "n_volumes": friedman_results["friedman"]["n_volumes"],
        }
        nemenyi_df = friedman_results.get("nemenyi")
        if nemenyi_df is not None:
            friedman_out["nemenyi"] = nemenyi_df.to_dict(orient="records")
        (output_dir / "data" / "friedman_nemenyi.json").write_text(
            json.dumps(friedman_out, indent=2)
        )

    # ------------------------------------------------------------------
    # Step 7: Generate comparison figures
    # ------------------------------------------------------------------
    logger.info("Step 7: Generating comparison figures...")
    t0 = time.time()

    # Fig -- NFE scaling comparison
    try:
        plot_nfe_scaling_comparison(all_results, output_dir, nfe_levels)
    except Exception:
        logger.exception("Failed to generate nfe_scaling_comparison")
        skipped.append("fig_nfe_scaling_comparison")

    # Fig -- Metric bars comparison
    try:
        plot_metric_bars_comparison(all_results, output_dir, nfe_levels)
    except Exception:
        logger.exception("Failed to generate metric_bars_comparison")
        skipped.append("fig_metric_bars_comparison")

    # Fig -- Regional volumes comparison (needs SynthSeg data)
    try:
        plot_regional_volumes_comparison(all_results, output_dir, nfe_levels)
    except Exception:
        logger.exception("Failed to generate regional_volumes_comparison")
        skipped.append("fig_regional_volumes_comparison")

    # Fig -- Pairwise bootstrap deltas (needs paired results)
    if paired_results:
        try:
            plot_pairwise_bootstrap_deltas(
                paired_results, output_dir, nfe_levels
            )
        except Exception:
            logger.exception(
                "Failed to generate pairwise_bootstrap_deltas"
            )
            skipped.append("fig_pairwise_bootstrap_deltas")
    else:
        logger.info(
            "No paired bootstrap results -- skipping delta figure"
        )
        skipped.append("fig_pairwise_bootstrap_deltas")

    # Fig -- Sample grid comparison (qualitative)
    if args.skip_qualitative:
        logger.info("Skipping qualitative sample grid (--skip-qualitative)")
        skipped.append("fig_sample_grid_comparison")
    else:
        try:
            plot_sample_grid_comparison(
                all_results, output_dir, nfe=best_nfe
            )
        except Exception:
            logger.exception("Failed to generate sample_grid_comparison")
            skipped.append("fig_sample_grid_comparison")

    logger.info("Step 7 complete (%.1fs)", time.time() - t0)

    # ------------------------------------------------------------------
    # Step 8: Generate comparison tables
    # ------------------------------------------------------------------
    logger.info("Step 8: Generating LaTeX tables...")
    table_dir = output_dir / "tables"

    # Table -- Main comparison results
    try:
        table_comparison = generate_comparison_results_table(
            all_results, nfe_levels
        )
        (table_dir / "comparison_results.tex").write_text(table_comparison)
        logger.info("Wrote comparison_results.tex")
    except Exception:
        logger.exception("Failed to generate comparison_results table")
        skipped.append("table_comparison_results")

    # Table -- Pairwise statistical tests
    if paired_results:
        try:
            table_pairwise = generate_pairwise_statistical_table(
                paired_results, nfe_levels
            )
            (table_dir / "pairwise_statistical.tex").write_text(
                table_pairwise
            )
            logger.info("Wrote pairwise_statistical.tex")
        except Exception:
            logger.exception(
                "Failed to generate pairwise_statistical table"
            )
            skipped.append("table_pairwise_statistical")
    else:
        logger.info(
            "No paired results -- skipping pairwise statistical table"
        )
        skipped.append("table_pairwise_statistical")

    # Table -- SynthSeg comparison
    try:
        table_synthseg = generate_comparison_synthseg_table(
            all_results, nfe=best_nfe
        )
        (table_dir / "comparison_synthseg.tex").write_text(table_synthseg)
        logger.info("Wrote comparison_synthseg.tex")
    except Exception:
        logger.exception("Failed to generate comparison_synthseg table")
        skipped.append("table_comparison_synthseg")

    # ------------------------------------------------------------------
    # Step 9: Write manifest
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    logger.info("Step 9: Writing manifest...")
    _write_comparison_manifest(
        output_dir, model_dirs, nfe_levels, n_bootstrap, skipped,
        elapsed_total,
    )

    # ------------------------------------------------------------------
    # Step 10: Summary log
    # ------------------------------------------------------------------
    n_figures = len(list((output_dir / "figures").glob("*.pdf")))
    n_tables = len(list((output_dir / "tables").glob("*.tex")))
    n_data = len(list((output_dir / "data").glob("*.npz")))

    logger.info("=" * 70)
    logger.info("Multi-model comparison complete!")
    logger.info("Output directory: %s", output_dir)
    logger.info("Figures: %d generated", n_figures)
    logger.info("Tables: %d generated", n_tables)
    logger.info("Data files: %d saved", n_data)
    logger.info("Total time: %.1fs (%.1f min)", elapsed_total, elapsed_total / 60)
    if skipped:
        logger.info("Skipped: %s", ", ".join(skipped))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
