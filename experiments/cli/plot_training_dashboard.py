"""Standalone CLI for generating the training performance dashboard.

Discovers ``training_summary.json`` in the given results directory and
produces a 3x3 panel figure.

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/plot_training_dashboard.py \
        --results-dir /path/to/ablation/xpred_exact_jvp

    # Custom output directory:
    python experiments/cli/plot_training_dashboard.py \
        --results-dir /path/to/results --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate training performance dashboard from diagnostics JSON."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root results directory (contains diagnostics/ subdirectory).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for dashboard figure. Defaults to {results-dir}/figures/.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    results_dir = Path(args.results_dir)

    # Discover training_summary.json
    summary_path = results_dir / "diagnostics" / "aggregate_results" / "training_summary.json"
    if not summary_path.exists():
        logger.error("training_summary.json not found at %s", summary_path)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"

    from neuromf.utils.training_dashboard import plot_training_dashboard

    out_path = plot_training_dashboard(summary_path, output_dir)
    logger.info("Dashboard saved to %s", out_path)


if __name__ == "__main__":
    main()
