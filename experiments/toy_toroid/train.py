"""Training loop for Phase 2 toroid experiment.

Single-run training function used by sweep.py. Supports EMA, periodic
evaluation, and checkpoint saving.
"""

import json
import logging
import math
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from neuromf.data.toroid_dataset import ToroidConfig, ToroidDataset
from neuromf.losses.meanflow_jvp import meanflow_loss
from neuromf.metrics.coverage_density import compute_coverage, compute_density
from neuromf.metrics.mmd import compute_mmd
from neuromf.models.toy_mlp import ToyMLP
from neuromf.sampling.multi_step import sample_euler
from neuromf.sampling.one_step import sample_one_step
from neuromf.utils.ema import EMAModel
from neuromf.utils.time_sampler import sample_t_and_r

log = logging.getLogger(__name__)


def _compute_torus_metrics(
    samples: torch.Tensor,
    projection_matrix: torch.Tensor | None = None,
) -> dict:
    """Compute torus-specific geometric fidelity metrics.

    Args:
        samples: Generated samples of shape (N, D).
        projection_matrix: If D > 4, the (D, 4) projection matrix for
            back-projection to R^4.

    Returns:
        Dict with torus distance, pair errors, KS p-values.
    """
    import numpy as np
    from scipy import stats

    # Project back to R^4 if needed
    if projection_matrix is not None:
        samples_4d = samples @ projection_matrix
    else:
        samples_4d = samples

    # Torus distance: ||z|| should be sqrt(2) for unnormalised torus
    norms = samples_4d.norm(dim=1)
    target_norm = math.sqrt(2.0)
    torus_distance = (norms - target_norm).abs()

    # Pair norms should be 1.0 each
    pair1 = samples_4d[:, 0] ** 2 + samples_4d[:, 1] ** 2
    pair2 = samples_4d[:, 2] ** 2 + samples_4d[:, 3] ** 2

    # Angular distributions
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
        "theta1_ks_statistic": float(ks1.statistic),
        "theta2_ks_statistic": float(ks2.statistic),
    }


def _quick_eval(
    model: ToyMLP,
    cfg: DictConfig,
    dataset: ToroidDataset,
) -> dict:
    """Quick evaluation: 1-NFE torus distance only."""
    model.eval()
    device = next(model.parameters()).device
    n = cfg.eval.n_samples_quick
    noise = torch.randn(n, cfg.model.data_dim, device=device)

    pred_type = cfg.model.prediction_type
    samples = sample_one_step(model, noise, prediction_type=pred_type).cpu()
    metrics = _compute_torus_metrics(samples, dataset.projection_matrix)
    model.train()
    return {"torus_distance_1nfe": metrics["mean_torus_distance"]}


def _full_eval(
    model: ToyMLP,
    cfg: DictConfig,
    dataset: ToroidDataset,
) -> dict:
    """Full evaluation: torus + distributional metrics for 1-NFE and multi-step."""
    model.eval()
    device = next(model.parameters()).device
    n = cfg.eval.n_samples_full
    pred_type = cfg.model.prediction_type
    noise = torch.randn(n, cfg.model.data_dim, device=device)

    # 1-NFE
    samples_1nfe = sample_one_step(model, noise, prediction_type=pred_type).cpu()
    torus_1nfe = _compute_torus_metrics(samples_1nfe, dataset.projection_matrix)

    # Multi-step
    nfe = cfg.eval.multi_step_nfe
    samples_multi = sample_euler(model, noise, n_steps=nfe, prediction_type=pred_type).cpu()
    torus_multi = _compute_torus_metrics(samples_multi, dataset.projection_matrix)

    # Distributional metrics: compare to real data
    real = dataset.data[:n].cpu()
    mmd_1nfe = compute_mmd(real, samples_1nfe)
    mmd_multi = compute_mmd(real, samples_multi)
    cov_1nfe = compute_coverage(real, samples_1nfe, k=5)
    cov_multi = compute_coverage(real, samples_multi, k=5)
    den_1nfe = compute_density(real, samples_1nfe, k=5)
    den_multi = compute_density(real, samples_multi, k=5)

    model.train()
    return {
        "one_step": {**torus_1nfe, "mmd": mmd_1nfe, "coverage": cov_1nfe, "density": den_1nfe},
        "multi_step": {
            **torus_multi,
            "mmd": mmd_multi,
            "coverage": cov_multi,
            "density": den_multi,
        },
        "samples_1nfe": samples_1nfe,
        "samples_multi": samples_multi,
    }


def train_run(cfg: DictConfig, run_dir: Path) -> dict:
    """Execute a single training run.

    Args:
        cfg: Full merged config (base + ablation overrides).
        run_dir: Directory to save checkpoints and logs.

    Returns:
        Dict with training log, final metrics, and model state.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(cfg.training.seed)
    device = torch.device("cpu")  # Toy experiment runs on CPU

    # Dataset
    ds_cfg = ToroidConfig(
        n_samples=cfg.data.n_samples,
        mode=cfg.data.mode,
        ambient_dim=cfg.data.ambient_dim,
    )
    dataset = ToroidDataset(ds_cfg, seed=cfg.training.seed)
    loader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)

    # Model
    data_dim = cfg.data.ambient_dim
    model = ToyMLP(
        data_dim=data_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        prediction_type=cfg.model.prediction_type,
    ).to(device)

    # EMA
    ema = EMAModel(model, decay=cfg.training.ema_decay)

    # Optimiser
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, betas=tuple(cfg.training.betas)
    )

    epoch_losses: list[float] = []
    epoch_losses_fm: list[float] = []
    epoch_losses_mf: list[float] = []
    eval_log: list[dict] = []

    t_start = time.time()

    for epoch in range(cfg.training.epochs):
        model.train()
        batch_losses = []
        batch_fm = []
        batch_mf = []

        for x_0 in loader:
            x_0 = x_0.to(device)
            B = x_0.shape[0]
            eps = torch.randn_like(x_0)

            t, r = sample_t_and_r(
                B,
                mu=cfg.sampling.mu,
                sigma=cfg.sampling.sigma,
                t_min=cfg.sampling.t_min,
                data_proportion=cfg.sampling.data_proportion,
                device=device,
            )

            result = meanflow_loss(
                model,
                x_0,
                eps,
                t,
                r,
                p=cfg.loss.p,
                adaptive=cfg.loss.adaptive,
                norm_eps=cfg.loss.norm_eps,
                lambda_mf=cfg.loss.lambda_mf,
                prediction_type=cfg.model.prediction_type,
            )

            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()
            ema.update(model)

            batch_losses.append(result["loss"].item())
            batch_fm.append(result["loss_fm"].item())
            batch_mf.append(result["loss_mf"].item())

        mean_loss = sum(batch_losses) / len(batch_losses)
        mean_fm = sum(batch_fm) / len(batch_fm)
        mean_mf = sum(batch_mf) / len(batch_mf)
        epoch_losses.append(mean_loss)
        epoch_losses_fm.append(mean_fm)
        epoch_losses_mf.append(mean_mf)

        # Periodic evaluation and checkpoint
        if (epoch + 1) % cfg.training.eval_every == 0 or epoch == 0:
            # Evaluate with EMA weights
            ema.apply_shadow(model)
            quick = _quick_eval(model, cfg, dataset)
            ema.restore(model)

            eval_entry = {
                "epoch": epoch + 1,
                "loss": mean_loss,
                "loss_fm": mean_fm,
                "loss_mf": mean_mf,
                **quick,
            }
            eval_log.append(eval_entry)

            log.info(
                "Epoch %d/%d — loss: %.4f (fm: %.4f, mf: %.4f) | torus_dist: %.4f",
                epoch + 1,
                cfg.training.epochs,
                mean_loss,
                mean_fm,
                mean_mf,
                quick["torus_distance_1nfe"],
            )

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": mean_loss,
                },
                ckpt_dir / f"model_epoch_{epoch + 1:03d}.pt",
            )

    elapsed = time.time() - t_start

    # Final full evaluation with EMA
    ema.apply_shadow(model)
    final_metrics = _full_eval(model, cfg, dataset)
    ema.restore(model)

    # Extract samples for saving
    samples_1nfe = final_metrics.pop("samples_1nfe")
    samples_multi = final_metrics.pop("samples_multi")

    # Save final EMA model
    torch.save(ema.state_dict(), run_dir / "ema_model.pt")

    # Save training log
    training_log = {
        "epoch_losses": epoch_losses,
        "epoch_losses_fm": epoch_losses_fm,
        "epoch_losses_mf": epoch_losses_mf,
        "eval_log": eval_log,
        "elapsed_seconds": elapsed,
        "config": {
            "ambient_dim": cfg.data.ambient_dim,
            "prediction_type": cfg.model.prediction_type,
            "p": cfg.loss.p,
            "data_proportion": cfg.sampling.data_proportion,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "ema_decay": cfg.training.ema_decay,
        },
    }
    with open(run_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    # Save final metrics
    final_metrics["final_loss"] = epoch_losses[-1]
    final_metrics["final_loss_fm"] = epoch_losses_fm[-1]
    final_metrics["final_loss_mf"] = epoch_losses_mf[-1]
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save samples
    torch.save(
        {"one_step": samples_1nfe, "multi_step": samples_multi},
        run_dir / "samples.pt",
    )

    log.info("Run complete in %.1fs — saved to %s", elapsed, run_dir)
    return final_metrics
