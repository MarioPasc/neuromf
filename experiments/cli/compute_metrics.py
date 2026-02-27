"""CLI script for computing evaluation metrics on generated volumes.

Computes distributional metrics (3D-FID, 2.5D-FID, MMD, Coverage, Density),
per-volume metrics (MS-SSIM, PSNR) with NN pairing in feature space,
spectral analysis, and SynthSeg morphological evaluation.

Usage:
    python experiments/cli/compute_metrics.py \
        --config configs/generate.yaml \
        --volumes-dir /path/to/generation/volumes/ \
        --real-features-dir /path/to/features/ \
        --nfe 1 10 50

    # Skip SynthSeg for faster iteration:
    python experiments/cli/compute_metrics.py \
        --config configs/generate.yaml \
        --volumes-dir /path/to/volumes/ \
        --real-features-dir /path/to/features/ \
        --nfe 1 10 50 \
        --skip-synthseg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler

from neuromf.metrics.coverage_density import compute_coverage, compute_density
from neuromf.metrics.fid_3d import compute_fid_3d
from neuromf.metrics.mmd import compute_mmd
from neuromf.metrics.ms_ssim_3d import compute_ms_ssim_3d
from neuromf.metrics.pairing import compute_nn_pairs
from neuromf.metrics.spectral import compute_hf_energy_ratio
from neuromf.metrics.ssim_psnr import compute_psnr

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compute evaluation metrics on generated volumes.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Config YAML paths.",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml.",
    )
    parser.add_argument(
        "--volumes-dir",
        type=str,
        required=True,
        help="Directory containing volume HDF5 files (nfe_XXX.h5).",
    )
    parser.add_argument(
        "--real-features-dir",
        type=str,
        required=True,
        help="Directory containing real feature HDF5 caches.",
    )
    parser.add_argument(
        "--real-volumes-h5",
        type=str,
        default=None,
        help="HDF5 file with real test volumes (for per-volume metrics).",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=[1, 10, 50],
        help="NFE levels to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for metrics JSON files.",
    )
    parser.add_argument(
        "--skip-synthseg",
        action="store_true",
        help="Skip SynthSeg morphological evaluation.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> OmegaConf:
    """Load and merge config layers."""
    config_paths = [Path(p) for p in args.config]
    configs_dir = Path(args.configs_dir) if args.configs_dir else config_paths[0].parent

    base_path = configs_dir / "base.yaml"
    project_root = Path(__file__).resolve().parent.parent.parent
    main_train_path = project_root / "configs" / "train_meanflow.yaml"
    main_gen_path = project_root / "configs" / "generate.yaml"

    layers = []
    if base_path.exists():
        layers.append(OmegaConf.load(base_path))
    if main_train_path.exists():
        layers.append(OmegaConf.load(main_train_path))
    if main_gen_path.exists() and main_gen_path.resolve() not in [
        p.resolve() for p in config_paths
    ]:
        layers.append(OmegaConf.load(main_gen_path))
    for cp in config_paths:
        layers.append(OmegaConf.load(cp))

    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)
    return config


def _load_features(h5_path: Path) -> torch.Tensor:
    """Load features from HDF5 cache."""
    with h5py.File(str(h5_path), "r") as f:
        return torch.from_numpy(f["features"][:].astype(np.float32))


def _compute_distributional_metrics(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    metrics_cfg: OmegaConf,
) -> dict:
    """Compute distributional metrics."""
    results: dict = {}

    if metrics_cfg.get("fid_3d", True):
        fid_val = compute_fid_3d(real_feats, gen_feats)
        results["fid_3d"] = {
            "value": fid_val,
            "feature_extractor": "R3D-18 (Kinetics-400, MOTFM protocol)",
            "feature_dim": 512,
        }
        logger.info("  3D-FID: %.4f", fid_val)

    if metrics_cfg.get("mmd", True):
        mmd_val = compute_mmd(real_feats, gen_feats)
        results["mmd"] = {
            "value": mmd_val,
            "kernel": "RBF",
            "n_bandwidths": 5,
        }
        logger.info("  MMD: %.6f", mmd_val)

    if metrics_cfg.get("coverage_density", True):
        k = metrics_cfg.get("pairing", {}).get("k", 5)
        cov = compute_coverage(real_feats, gen_feats, k=k)
        dens = compute_density(real_feats, gen_feats, k=k)
        results["coverage_k5"] = {"value": cov, "k": k}
        results["density_k5"] = {"value": dens, "k": k}
        logger.info("  Coverage: %.4f, Density: %.4f", cov, dens)

    return results


def _compute_per_volume_metrics(
    gen_volumes_path: Path,
    real_volumes_path: Path,
    nn_indices: torch.Tensor,
    n_pairs: int,
) -> dict:
    """Compute per-volume metrics with NN pairing."""
    ms_ssim_vals: list[float] = []
    psnr_vals: list[float] = []

    with h5py.File(str(gen_volumes_path), "r") as gf, h5py.File(str(real_volumes_path), "r") as rf:
        for i in range(min(n_pairs, gf["volumes"].shape[0])):
            real_idx = int(nn_indices[i].item())
            gen_vol = torch.from_numpy(gf["volumes"][i].astype(np.float32))
            real_vol = torch.from_numpy(rf["volumes"][real_idx].astype(np.float32))

            # Shape: (D,H,W) -> (1,1,D,H,W)
            gen_5d = gen_vol.unsqueeze(0).unsqueeze(0)
            real_5d = real_vol.unsqueeze(0).unsqueeze(0)

            ms_ssim_vals.append(compute_ms_ssim_3d(gen_5d, real_5d))
            psnr_vals.append(compute_psnr(real_5d, gen_5d))

            if (i + 1) % 100 == 0:
                logger.info("  Paired metrics: %d / %d", i + 1, n_pairs)

    ms_ssim_arr = np.array(ms_ssim_vals)
    psnr_arr = np.array(psnr_vals)

    results = {
        "ms_ssim": {
            "mean": float(ms_ssim_arr.mean()),
            "std": float(ms_ssim_arr.std()),
            "pairing": "nearest_neighbour_med3d",
        },
        "psnr_db": {
            "mean": float(psnr_arr.mean()),
            "std": float(psnr_arr.std()),
            "pairing": "nearest_neighbour_med3d",
        },
    }
    logger.info(
        "  MS-SSIM: %.4f +/- %.4f, PSNR: %.2f +/- %.2f dB",
        results["ms_ssim"]["mean"],
        results["ms_ssim"]["std"],
        results["psnr_db"]["mean"],
        results["psnr_db"]["std"],
    )
    return results


def _compute_spectral_metrics(
    gen_volumes_path: Path,
    n_samples: int = 100,
) -> dict:
    """Compute spectral HF energy ratio on a subset of generated volumes."""
    hf_ratios: list[float] = []
    with h5py.File(str(gen_volumes_path), "r") as gf:
        total = min(n_samples, gf["volumes"].shape[0])
        for i in range(total):
            vol = torch.from_numpy(gf["volumes"][i].astype(np.float32))
            hf_ratios.append(compute_hf_energy_ratio(vol))

    arr = np.array(hf_ratios)
    return {
        "hf_energy_ratio": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "n_samples": len(hf_ratios),
            "cutoff": "0.5_nyquist",
        }
    }


def main() -> None:
    """Main entry point for metrics computation."""
    args = parse_args()
    config = _load_config(args)
    metrics_cfg = config.metrics

    volumes_dir = Path(args.volumes_dir)
    features_dir = Path(args.real_features_dir)
    output_dir = Path(
        args.output_dir or str(config.paths.get("metrics_dir", volumes_dir.parent / "metrics"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real features
    real_med3d_path = features_dir / "real_med3d.h5"
    if not real_med3d_path.exists():
        logger.error("Real Med3D features not found: %s", real_med3d_path)
        sys.exit(1)
    real_feats = _load_features(real_med3d_path)
    logger.info("Loaded real features: %s", real_feats.shape)

    for nfe in args.nfe:
        logger.info("=== Computing metrics for NFE=%d ===", nfe)

        gen_vol_path = volumes_dir / f"nfe_{nfe:03d}.h5"
        gen_feat_path = features_dir / f"gen_med3d_nfe{nfe:03d}.h5"

        if not gen_feat_path.exists():
            logger.warning("Generated features not found: %s (skipping)", gen_feat_path)
            continue

        gen_feats = _load_features(gen_feat_path)
        logger.info("Loaded generated features: %s", gen_feats.shape)

        # Distributional metrics
        dist_results = _compute_distributional_metrics(real_feats, gen_feats, metrics_cfg)

        # NN pairing (used by per-volume metrics and SynthSeg)
        nn_indices: torch.Tensor | None = None
        per_vol_results: dict = {}
        spectral_results: dict = {}
        morphological_results: dict = {}

        has_real_vols = args.real_volumes_h5 and Path(args.real_volumes_h5).exists()

        if has_real_vols and gen_vol_path.exists():
            nn_indices = compute_nn_pairs(real_feats, gen_feats)

        # Per-volume metrics
        if nn_indices is not None and gen_vol_path.exists():
            real_vol_path = Path(args.real_volumes_h5)
            per_vol_results = _compute_per_volume_metrics(
                gen_vol_path, real_vol_path, nn_indices, gen_feats.shape[0]
            )

        if gen_vol_path.exists() and metrics_cfg.get("spectral", True):
            spectral_results = _compute_spectral_metrics(gen_vol_path)

        # SynthSeg morphological evaluation
        if (
            nn_indices is not None
            and gen_vol_path.exists()
            and metrics_cfg.get("synthseg", False)
            and not args.skip_synthseg
        ):
            from neuromf.metrics.synthseg_metrics import (
                SynthSegConfig,
                run_synthseg_evaluation,
            )

            synthseg_cfg_raw = config.get("synthseg", {})
            synthseg_config = SynthSegConfig(
                command=list(synthseg_cfg_raw.get("command", ["mri_synthseg"])),
                regions_of_interest=list(synthseg_cfg_raw.get("regions_of_interest", [])) or None,
                robust=bool(synthseg_cfg_raw.get("robust", True)),
                parc=bool(synthseg_cfg_raw.get("parc", False)),
                voxel_size_mm=float(synthseg_cfg_raw.get("voxel_size_mm", 1.0)),
                threads=int(synthseg_cfg_raw.get("threads", 0)),
                crop=int(synthseg_cfg_raw.get("crop", 0)),
            )

            synthseg_result = run_synthseg_evaluation(
                gen_volumes_h5=gen_vol_path,
                real_volumes_h5=Path(args.real_volumes_h5),
                output_dir=output_dir,
                nn_indices=nn_indices,
                config=synthseg_config,
                nfe=nfe,
            )
            if synthseg_result is not None:
                morphological_results = synthseg_result

        # Assemble final metrics
        metrics_result = {
            "experiment": "stage1_healthy",
            "nfe": nfe,
            "n_generated": int(gen_feats.shape[0]),
            "n_real": int(real_feats.shape[0]),
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "distributional": dist_results,
            "per_volume": per_vol_results,
            "spectral": spectral_results,
            "morphological": morphological_results,
        }

        out_path = output_dir / f"metrics_nfe{nfe:03d}.json"
        out_path.write_text(json.dumps(metrics_result, indent=2))
        logger.info("Metrics saved: %s", out_path)

    logger.info("All metrics computed.")


if __name__ == "__main__":
    main()
