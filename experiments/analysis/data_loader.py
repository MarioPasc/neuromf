"""Data loading utilities for Phase 5 post-hoc analysis.

Loads feature caches, metrics JSONs, SynthSeg CSVs, and decoded volumes
from the Phase 5 evaluation results directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from neuromf.metrics.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Baseline metrics from Yazdani et al. (MOTFM, MICCAI 2025)
# ---------------------------------------------------------------------------

MOTFM_BASELINES: dict[int, dict[str, float]] = {
    1: {"fid_3d": 32.10, "ms_ssim": 0.66, "mmd": 0.51},
    10: {"fid_3d": 9.27, "ms_ssim": 0.77, "mmd": 0.25},
    50: {"fid_3d": 7.93, "ms_ssim": 0.77, "mmd": 0.22},
}

DDPM_BASELINES: dict[int, dict[str, float]] = {
    1: {"fid_3d": 146.47, "ms_ssim": 0.06, "mmd": 39.80},
    10: {"fid_3d": 51.68, "ms_ssim": 0.51, "mmd": 26.10},
    50: {"fid_3d": 29.67, "ms_ssim": 0.59, "mmd": 4.28},
}


def load_features(
    results_dir: Path,
    nfe_levels: list[int],
) -> dict[str, Tensor]:
    """Load cached R3D-18 features from HDF5 files.

    Args:
        results_dir: Root results directory for the evaluation run.
        nfe_levels: List of NFE values to load (e.g. [1, 10, 50]).

    Returns:
        Dict mapping 'real' and 'gen_{nfe}' to feature tensors of shape
        (N, 512).
    """
    features: dict[str, Tensor] = {}
    feat_dir = results_dir / "features"

    real_path = feat_dir / "real_med3d.h5"
    if real_path.exists():
        features["real"] = FeatureExtractor.load_cached(real_path)
        logger.info("Loaded real features: %s", features["real"].shape)
    else:
        logger.warning("Real features not found: %s", real_path)

    for nfe in nfe_levels:
        gen_path = feat_dir / f"gen_med3d_nfe{nfe:03d}.h5"
        if gen_path.exists():
            features[f"gen_{nfe}"] = FeatureExtractor.load_cached(gen_path)
            logger.info(
                "Loaded gen NFE=%d features: %s",
                nfe,
                features[f"gen_{nfe}"].shape,
            )
        else:
            logger.warning("Gen features not found for NFE=%d: %s", nfe, gen_path)

    return features


def load_metrics_jsons(
    results_dir: Path,
    nfe_levels: list[int],
) -> dict[int, dict[str, Any]]:
    """Load metrics JSON files for each NFE level.

    Args:
        results_dir: Root results directory for the evaluation run.
        nfe_levels: List of NFE values to load.

    Returns:
        Dict mapping NFE int to the parsed JSON contents.
    """
    metrics: dict[int, dict[str, Any]] = {}
    for nfe in nfe_levels:
        path = results_dir / "metrics" / f"metrics_nfe{nfe:03d}.json"
        if path.exists():
            with open(path) as f:
                metrics[nfe] = json.load(f)
            logger.info("Loaded metrics JSON for NFE=%d", nfe)
        else:
            logger.warning("Metrics JSON not found for NFE=%d: %s", nfe, path)
    return metrics


def load_synthseg_data(
    results_dir: Path,
    nfe_levels: list[int],
) -> dict[str, Any]:
    """Load SynthSeg volume and QC CSVs.

    Args:
        results_dir: Root results directory for the evaluation run.
        nfe_levels: List of NFE values to load.

    Returns:
        Dict with keys:
          - 'real_volumes': DataFrame (326 rows) or None
          - 'gen_volumes_{nfe}': DataFrame (268 rows) per NFE, or None
          - 'gen_qc_{nfe}': DataFrame per NFE, or None
    """
    synthseg_dir = results_dir / "metrics" / "synthseg"
    data: dict[str, Any] = {}

    real_path = synthseg_dir / "real_volumes.csv"
    if real_path.exists():
        data["real_volumes"] = pd.read_csv(real_path)
        logger.info("Loaded real SynthSeg volumes: %d rows", len(data["real_volumes"]))
    else:
        data["real_volumes"] = None
        logger.warning("Real SynthSeg volumes not found: %s", real_path)

    for nfe in nfe_levels:
        vol_path = synthseg_dir / f"gen_volumes_nfe{nfe:03d}.csv"
        qc_path = synthseg_dir / f"gen_qc_nfe{nfe:03d}.csv"

        if vol_path.exists():
            data[f"gen_volumes_{nfe}"] = pd.read_csv(vol_path)
            logger.info(
                "Loaded gen SynthSeg volumes NFE=%d: %d rows",
                nfe,
                len(data[f"gen_volumes_{nfe}"]),
            )
        else:
            data[f"gen_volumes_{nfe}"] = None
            logger.warning("Gen SynthSeg volumes not found NFE=%d", nfe)

        if qc_path.exists():
            data[f"gen_qc_{nfe}"] = pd.read_csv(qc_path)
            logger.info(
                "Loaded gen SynthSeg QC NFE=%d: %d rows",
                nfe,
                len(data[f"gen_qc_{nfe}"]),
            )
        else:
            data[f"gen_qc_{nfe}"] = None
            logger.warning("Gen SynthSeg QC not found NFE=%d", nfe)

    return data


def load_volumes(
    results_dir: Path,
    nfe_levels: list[int],
    indices: list[int],
) -> dict[str, np.ndarray] | None:
    """Load specific volume indices from HDF5 archives.

    Args:
        results_dir: Root results directory for the evaluation run.
        nfe_levels: List of NFE values to load.
        indices: Volume indices to load (0-based).

    Returns:
        Dict mapping 'real' and 'gen_{nfe}' to arrays of shape
        (len(indices), 1, 192, 192, 192), or None if files missing.
    """
    gen_dir = results_dir / "generation"
    volumes: dict[str, np.ndarray] = {}

    real_path = gen_dir / "real_test.h5"
    if real_path.exists():
        with h5py.File(real_path, "r") as f:
            n_total = f["volumes"].shape[0]
            valid_idx = [i for i in indices if i < n_total]
            volumes["real"] = np.array([f["volumes"][i] for i in valid_idx])
            logger.info("Loaded %d real volumes", len(valid_idx))
    else:
        logger.warning("Real volumes HDF5 not found: %s", real_path)
        return None

    for nfe in nfe_levels:
        vol_dir = gen_dir / "volumes"
        h5_path = vol_dir / f"nfe_{nfe:03d}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                n_total = f["volumes"].shape[0]
                valid_idx = [i for i in indices if i < n_total]
                volumes[f"gen_{nfe}"] = np.array(
                    [f["volumes"][i] for i in valid_idx]
                )
                logger.info("Loaded %d gen volumes NFE=%d", len(valid_idx), nfe)
        else:
            logger.warning("Gen volumes HDF5 not found NFE=%d: %s", nfe, h5_path)

    return volumes if volumes else None


def load_synthseg_labels(
    results_dir: Path,
    nfe_levels: list[int],
    indices: list[int],
) -> dict[str, list[np.ndarray]] | None:
    """Load NIfTI SynthSeg label maps for specific volume indices.

    Args:
        results_dir: Root results directory.
        nfe_levels: NFE values to load labels for.
        indices: Volume indices.

    Returns:
        Dict mapping 'real' and 'gen_{nfe}' to lists of 3D label arrays,
        or None if nibabel unavailable or files missing.
    """
    try:
        import nibabel as nib
    except ImportError:
        logger.warning("nibabel not installed — skipping SynthSeg label loading")
        return None

    synthseg_dir = results_dir / "metrics" / "synthseg"
    labels: dict[str, list[np.ndarray]] = {}

    # Real labels
    real_dir = synthseg_dir / "real_labels"
    if real_dir.exists():
        real_list: list[np.ndarray] = []
        for idx in indices:
            path = real_dir / f"vol_{idx:04d}_synthseg.nii.gz"
            if path.exists():
                real_list.append(np.asarray(nib.load(str(path)).dataobj))
            else:
                logger.debug("Real label not found: %s", path)
        if real_list:
            labels["real"] = real_list
    else:
        logger.warning("Real labels dir not found: %s", real_dir)

    # Gen labels per NFE
    for nfe in nfe_levels:
        gen_dir = synthseg_dir / f"gen_labels_nfe{nfe:03d}"
        if not gen_dir.exists():
            logger.warning("Gen labels dir not found NFE=%d: %s", nfe, gen_dir)
            continue
        gen_list: list[np.ndarray] = []
        for idx in indices:
            path = gen_dir / f"vol_{idx:04d}_synthseg.nii.gz"
            if path.exists():
                gen_list.append(np.asarray(nib.load(str(path)).dataobj))
            else:
                logger.debug("Gen label not found NFE=%d idx=%d", nfe, idx)
        if gen_list:
            labels[f"gen_{nfe}"] = gen_list

    return labels if labels else None
