"""Analysis manifest generation for cross-run comparison.

Creates a JSON manifest cataloguing all outputs (data, figures, tables)
along with experiment metadata, enabling automated comparison between
different training runs.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_manifest(
    results_dir: Path,
    analysis_dir: Path,
    nfe_levels: list[int],
    n_bootstrap: int,
    skipped: list[str] | None = None,
) -> dict[str, Any]:
    """Create analysis_manifest.json cataloguing all outputs.

    Args:
        results_dir: Root results directory for the evaluation run.
        analysis_dir: Output directory for analysis artifacts.
        nfe_levels: NFE levels analysed.
        n_bootstrap: Number of bootstrap samples used.
        skipped: List of skipped items (e.g., 'fig08', 'fig09').

    Returns:
        Manifest dict (also written to analysis_dir/analysis_manifest.json).
    """
    manifest: dict[str, Any] = {
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "analysis_dir": str(analysis_dir),
        "nfe_levels": nfe_levels,
        "n_bootstrap": n_bootstrap,
        "skipped": skipped or [],
    }

    # Data files
    data_dir = analysis_dir / "data"
    if data_dir.exists():
        manifest["data_files"] = {
            p.stem: str(p) for p in sorted(data_dir.glob("*.npz"))
        }
    else:
        manifest["data_files"] = {}

    # Figures
    fig_dir = analysis_dir / "figures"
    if fig_dir.exists():
        manifest["figures"] = {}
        for pdf in sorted(fig_dir.glob("*.pdf")):
            name = pdf.stem
            manifest["figures"][name] = {
                "pdf": str(pdf),
                "png": str(pdf.with_suffix(".png")),
            }
    else:
        manifest["figures"] = {}

    # Tables
    table_dir = analysis_dir / "tables"
    if table_dir.exists():
        manifest["tables"] = {
            p.stem: str(p) for p in sorted(table_dir.glob("*.tex"))
        }
    else:
        manifest["tables"] = {}

    # Write manifest
    manifest_path = analysis_dir / "analysis_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest: %s", manifest_path)

    return manifest
