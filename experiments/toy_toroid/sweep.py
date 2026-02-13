#!/usr/bin/env python
"""Ablation sweep runner for Phase 2 toroid experiment.

Runs all 18 training runs + NFE inference sweep sequentially (~45 min on CPU).
Generates figures and HTML report after all runs complete.

Usage:
    python experiments/toy_toroid/sweep.py --ablation all
    python experiments/toy_toroid/sweep.py --ablation a b
    python experiments/toy_toroid/sweep.py --ablation d  # inference only
"""

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for experiments.* imports
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from omegaconf import OmegaConf

log = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"


def _load_config(ablation: str | None = None) -> OmegaConf:
    """Load base config merged with ablation overrides."""
    base = OmegaConf.load(CONFIGS_DIR / "base.yaml")
    if ablation:
        ablation_file = CONFIGS_DIR / f"ablation_{ablation}.yaml"
        if ablation_file.exists():
            override = OmegaConf.load(ablation_file)
            base = OmegaConf.merge(base, override)
    return base


def run_ablation_a(results_dir: Path) -> None:
    """Ablation A: Convergence baseline."""
    from experiments.toy_toroid.train import train_run

    cfg = _load_config("a")
    cfg.model.data_dim = cfg.data.ambient_dim
    run_dir = results_dir / "ablation_a" / cfg.run_name
    log.info("=== Ablation A: Convergence Baseline ===")
    train_run(cfg, run_dir)


def run_ablation_b(results_dir: Path) -> None:
    """Ablation B: x-pred vs u-pred x dimensionality."""
    from experiments.toy_toroid.train import train_run

    base = _load_config("b")
    dims = base.sweep.ambient_dim
    pred_types = base.sweep.prediction_type

    for dim, pred in itertools.product(dims, pred_types):
        cfg = _load_config()  # fresh base
        cfg.data.ambient_dim = dim
        cfg.model.data_dim = dim
        cfg.model.prediction_type = pred
        run_name = f"D{dim}_{pred}-pred"
        run_dir = results_dir / "ablation_b" / run_name
        log.info("=== Ablation B: %s ===", run_name)
        train_run(cfg, run_dir)


def run_ablation_c(results_dir: Path) -> None:
    """Ablation C: Lp norm sweep."""
    from experiments.toy_toroid.train import train_run

    base = _load_config("c")
    p_values = base.sweep.p

    for p in p_values:
        cfg = _load_config()  # fresh base
        cfg.loss.p = p
        cfg.model.data_dim = cfg.data.ambient_dim
        run_name = f"p{p:.1f}"
        run_dir = results_dir / "ablation_c" / run_name
        log.info("=== Ablation C: p=%.1f ===", p)
        train_run(cfg, run_dir)


def run_ablation_d(results_dir: Path) -> None:
    """Ablation D: NFE sweep (inference only)."""
    from experiments.toy_toroid.evaluate import evaluate_nfe_sweep

    cfg = _load_config("d")
    log.info("=== Ablation D: NFE Sweep ===")
    evaluate_nfe_sweep(cfg, results_dir)


def run_ablation_e(results_dir: Path) -> None:
    """Ablation E: data_proportion sweep."""
    from experiments.toy_toroid.train import train_run

    base = _load_config("e")
    dp_values = base.sweep.data_proportion

    for dp in dp_values:
        cfg = _load_config()  # fresh base
        cfg.sampling.data_proportion = dp
        cfg.model.data_dim = cfg.data.ambient_dim
        run_name = f"dp{dp:.2f}"
        run_dir = results_dir / "ablation_e" / run_name
        log.info("=== Ablation E: data_proportion=%.2f ===", dp)
        train_run(cfg, run_dir)


def generate_summary(results_dir: Path) -> None:
    """Aggregate results across all ablations into summary_metrics.json."""
    summary = {}

    # Collect all metrics.json files
    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        rel = metrics_file.relative_to(results_dir)
        key = str(rel.parent)
        with open(metrics_file) as f:
            summary[key] = json.load(f)

    # Collect NFE sweep if exists
    nfe_file = results_dir / "ablation_d" / "nfe_sweep.json"
    if nfe_file.exists():
        with open(nfe_file) as f:
            summary["ablation_d/nfe_sweep"] = json.load(f)

    with open(results_dir / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("Summary saved to %s", results_dir / "summary_metrics.json")


def generate_tables(results_dir: Path) -> None:
    """Generate CSV tables from results."""
    import csv

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_dir / "summary_metrics.json"
    if not summary_file.exists():
        log.warning("No summary_metrics.json found — skipping tables")
        return

    with open(summary_file) as f:
        summary = json.load(f)

    # Table S1: Full ablation results
    rows = []
    for key, metrics in summary.items():
        if key.startswith("ablation_d"):
            continue  # NFE sweep handled separately
        parts = key.split("/")
        ablation = parts[0].replace("ablation_", "").upper()
        run_name = parts[1] if len(parts) > 1 else ""

        one = metrics.get("one_step", {})
        multi = metrics.get("multi_step", {})
        rows.append(
            {
                "ablation": ablation,
                "run": run_name,
                "final_loss": metrics.get("final_loss", ""),
                "torus_dist_1nfe": one.get("mean_torus_distance", ""),
                "torus_dist_multi": multi.get("mean_torus_distance", ""),
                "mmd_1nfe": one.get("mmd", ""),
                "ks_theta1_p": one.get("theta1_ks_pvalue", ""),
                "ks_theta2_p": one.get("theta2_ks_pvalue", ""),
                "coverage_1nfe": one.get("coverage", ""),
                "density_1nfe": one.get("density", ""),
            }
        )

    if rows:
        with open(tables_dir / "table_s1_full_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # Table 1: Dimensionality scaling
    dim_rows = []
    for dim in [4, 16, 64, 256]:
        row = {"D": dim}
        for pred in ["u", "x"]:
            key = f"ablation_b/D{dim}_{pred}-pred"
            m = summary.get(key, {})
            one = m.get("one_step", {})
            row[f"{pred}_mmd"] = one.get("mmd", "")
            row[f"{pred}_loss"] = m.get("final_loss", "")
            row[f"{pred}_torus_dist"] = one.get("mean_torus_distance", "")
        dim_rows.append(row)

    if dim_rows:
        with open(tables_dir / "table_1_dim_scaling.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dim_rows[0].keys())
            writer.writeheader()
            writer.writerows(dim_rows)

    log.info("Tables saved to %s", tables_dir)


def main() -> None:
    """Main entry point for the sweep runner."""
    parser = argparse.ArgumentParser(description="Phase 2 toroid ablation sweep")
    parser.add_argument(
        "--ablation",
        nargs="+",
        choices=["a", "b", "c", "d", "e", "all"],
        default=["all"],
        help="Which ablations to run",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip training, only generate figures and report",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    base_cfg = _load_config()
    results_dir = Path(args.results_dir) if args.results_dir else Path(base_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ablations = args.ablation
    if "all" in ablations:
        ablations = ["a", "b", "c", "d", "e"]

    if not args.figures_only:
        t_start = time.time()

        runners = {
            "a": run_ablation_a,
            "b": run_ablation_b,
            "c": run_ablation_c,
            "d": run_ablation_d,
            "e": run_ablation_e,
        }

        # Run training ablations first, then D (inference), then figures
        training = [a for a in ablations if a != "d"]
        inference = [a for a in ablations if a == "d"]

        for abl in training:
            runners[abl](results_dir)

        for abl in inference:
            runners[abl](results_dir)

        elapsed = time.time() - t_start
        log.info("All training runs complete in %.1f seconds", elapsed)

    # Generate summary, tables, figures, report
    generate_summary(results_dir)
    generate_tables(results_dir)

    log.info("Generating figures...")
    from experiments.toy_toroid.figures import generate_all_figures

    generate_all_figures(results_dir)

    log.info("Generating HTML report...")
    from experiments.toy_toroid.report import generate_report

    generate_report(results_dir)

    log.info("=== Sweep complete ===")
    log.info("Results: %s", results_dir)
    log.info("Report: %s", results_dir / "report.html")


if __name__ == "__main__":
    main()
