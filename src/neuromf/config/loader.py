"""Unified config loading with hierarchical YAML merging.

All CLI scripts share the same merge chain::

    base.yaml → train_meanflow.yaml → [generate.yaml] → user overlays

This module extracts the common loading logic so each script doesn't
duplicate it with subtly different behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Resolved once per import; CLI scripts live at experiments/cli/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_merged_config(
    config_paths: list[Path] | Path,
    configs_dir: Path | None = None,
    include_generate: bool = False,
    require_base: bool = True,
    project_root: Path | None = None,
) -> DictConfig:
    """Load and merge config layers in standard order.

    Merge chain (left-to-right, later overrides earlier):
    1. ``base.yaml`` from *configs_dir*
    2. ``train_meanflow.yaml`` from project root (if not already in *config_paths*)
    3. ``generate.yaml`` from project root (if *include_generate* and not in *config_paths*)
    4. Each path in *config_paths* (user overlays, ablation diffs, etc.)

    Args:
        config_paths: One or more YAML config files. The first path's parent
            is used as *configs_dir* default.
        configs_dir: Directory containing ``base.yaml``. If ``None``, inferred
            from ``config_paths[0].parent``.
        include_generate: Whether to include ``generate.yaml`` in the merge
            chain (needed for generation/evaluation scripts).
        require_base: If ``True``, exit with error when ``base.yaml`` is
            missing. If ``False``, silently skip it.
        project_root: Project root directory. Auto-detected if ``None``.

    Returns:
        Fully merged and resolved OmegaConf config.
    """
    if isinstance(config_paths, Path):
        config_paths = [config_paths]

    if not config_paths:
        logger.error("No config paths provided")
        sys.exit(1)

    if configs_dir is None:
        configs_dir = config_paths[0].parent

    root = project_root or _PROJECT_ROOT
    base_path = configs_dir / "base.yaml"
    main_train_path = root / "configs" / "train_meanflow.yaml"
    main_gen_path = root / "configs" / "generate.yaml"

    resolved_user = {p.resolve() for p in config_paths}

    layers: list[DictConfig] = []

    # 1. base.yaml
    if base_path.exists():
        layers.append(OmegaConf.load(base_path))  # type: ignore[arg-type]
    elif require_base:
        logger.error("base.yaml not found at %s", base_path)
        sys.exit(1)

    # 2. train_meanflow.yaml (skip if already in user overlays)
    if main_train_path.exists() and main_train_path.resolve() not in resolved_user:
        layers.append(OmegaConf.load(main_train_path))  # type: ignore[arg-type]

    # 3. generate.yaml (optional)
    if include_generate and main_gen_path.exists() and main_gen_path.resolve() not in resolved_user:
        layers.append(OmegaConf.load(main_gen_path))  # type: ignore[arg-type]

    # 4. User overlays (left-to-right)
    for cp in config_paths:
        layers.append(OmegaConf.load(cp))  # type: ignore[arg-type]

    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)

    loaded = " + ".join(str(p) for p in [base_path, main_train_path, *config_paths] if p.exists())
    logger.info("Config loaded: %s", loaded)

    return config  # type: ignore[return-value]
