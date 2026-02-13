"""Evaluation utilities for Phase 2 toroid experiment.

Provides NFE sweep evaluation (Ablation D) and reusable metric computation.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from scipy import stats

from neuromf.data.toroid_dataset import ToroidConfig, ToroidDataset
from neuromf.metrics.coverage_density import compute_coverage, compute_density
from neuromf.metrics.mmd import compute_mmd
from neuromf.models.toy_mlp import ToyMLP
from neuromf.sampling.multi_step import sample_euler
from neuromf.sampling.one_step import sample_one_step
from neuromf.utils.ema import EMAModel

log = logging.getLogger(__name__)


def compute_torus_metrics(
    samples: torch.Tensor,
    projection_matrix: torch.Tensor | None = None,
) -> dict:
    """Compute torus-specific geometric fidelity metrics.

    Args:
        samples: Generated samples of shape (N, D).
        projection_matrix: If D > 4, the (D, 4) projection matrix.

    Returns:
        Dict with torus metrics.
    """
    if projection_matrix is not None:
        samples_4d = samples @ projection_matrix
    else:
        samples_4d = samples

    norms = samples_4d.norm(dim=1)
    target_norm = math.sqrt(2.0)
    torus_distance = (norms - target_norm).abs()

    pair1 = samples_4d[:, 0] ** 2 + samples_4d[:, 1] ** 2
    pair2 = samples_4d[:, 2] ** 2 + samples_4d[:, 3] ** 2

    theta1 = torch.atan2(samples_4d[:, 1], samples_4d[:, 0]).numpy()
    theta2 = torch.atan2(samples_4d[:, 3], samples_4d[:, 2]).numpy()

    theta1_norm = (theta1 + np.pi) / (2 * np.pi)
    theta2_norm = (theta2 + np.pi) / (2 * np.pi)

    ks1 = stats.kstest(theta1_norm, "uniform")
    ks2 = stats.kstest(theta2_norm, "uniform")

    return {
        "mean_torus_distance": float(torus_distance.mean()),
        "std_torus_distance": float(torus_distance.std()),
        "mean_pair1_error": float((pair1 - 1.0).abs().mean()),
        "mean_pair2_error": float((pair2 - 1.0).abs().mean()),
        "theta1_ks_pvalue": float(ks1.pvalue),
        "theta2_ks_pvalue": float(ks2.pvalue),
    }


def evaluate_nfe_sweep(
    cfg: DictConfig,
    results_dir: Path,
) -> dict:
    """Run Ablation D: NFE sweep on Ablation A's trained model.

    Args:
        cfg: Merged config with ablation_d settings.
        results_dir: Root results directory.

    Returns:
        Dict mapping NFE -> metrics.
    """
    # Load source checkpoint (Ablation A baseline)
    source_dir = results_dir / cfg.source_checkpoint
    ema_path = source_dir / "ema_model.pt"
    if not ema_path.exists():
        raise FileNotFoundError(f"Source EMA model not found: {ema_path}")

    # Read model config from source run's training log
    log_path = source_dir / "training_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Source training log not found: {log_path}")

    with open(log_path) as f:
        train_log = json.load(f)

    run_cfg = train_log["config"]
    data_dim = run_cfg["ambient_dim"]
    pred_type = run_cfg["prediction_type"]

    # Recreate model with matching architecture
    torch.manual_seed(42)

    # Infer hidden_dim and n_layers from EMA state dict
    ema_state = torch.load(ema_path, weights_only=False)
    shadow = ema_state["shadow"]
    # First layer weight shape: (hidden_dim, data_dim + 2)
    first_weight_key = next(k for k in shadow if k.endswith(".0.weight"))
    hidden_dim = shadow[first_weight_key].shape[0]
    # Count linear layers (weight keys)
    n_linear = sum(1 for k in shadow if k.endswith(".weight"))
    n_layers = n_linear - 1  # subtract output layer

    model = ToyMLP(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        prediction_type=pred_type,
    )

    # Load EMA weights
    ema = EMAModel(model, decay=ema_state.get("decay", 0.999))
    ema.load_state_dict(ema_state)
    ema.apply_shadow(model)
    model.eval()

    # Dataset for MMD reference
    ds_cfg = ToroidConfig(n_samples=cfg.n_eval_samples, mode="r4", ambient_dim=data_dim)
    dataset = ToroidDataset(ds_cfg, seed=42)
    real = dataset.data

    nfe_results = {}
    n = cfg.n_eval_samples

    for nfe in cfg.nfe_values:
        noise = torch.randn(n, data_dim)

        if nfe == 1:
            samples = sample_one_step(model, noise, prediction_type=pred_type).cpu()
        else:
            samples = sample_euler(model, noise, n_steps=nfe, prediction_type=pred_type).cpu()

        torus = compute_torus_metrics(samples, dataset.projection_matrix)
        mmd = compute_mmd(real, samples)
        cov = compute_coverage(real, samples, k=5)
        den = compute_density(real, samples, k=5)

        nfe_results[str(nfe)] = {
            **torus,
            "mmd": mmd,
            "coverage": cov,
            "density": den,
        }

        log.info(
            "NFE=%d: torus_dist=%.4f, MMD=%.6f, coverage=%.3f, density=%.3f",
            nfe,
            torus["mean_torus_distance"],
            mmd,
            cov,
            den,
        )

    # Save results
    out_dir = results_dir / "ablation_d"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nfe_sweep.json", "w") as f:
        json.dump(nfe_results, f, indent=2)

    log.info("NFE sweep saved to %s", out_dir / "nfe_sweep.json")
    return nfe_results
