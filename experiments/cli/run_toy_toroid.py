#!/usr/bin/env python
"""End-to-end toy toroid experiment: train MeanFlow on flat torus, evaluate, produce plots.

Usage:
    python experiments/cli/run_toy_toroid.py --config configs/toy_toroid.yaml

Outputs (to config.output.dir):
    - loss_curves.json: per-epoch loss values
    - generated_samples.pt: 1-NFE and multi-step samples
    - verification_metrics.json: torus distance, angular KS test
    - figures/: loss curve, projections, angular histograms
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from scipy import stats
from torch.utils.data import DataLoader

from neuromf.data.toroid_dataset import ToroidConfig, ToroidDataset
from neuromf.losses.meanflow_jvp import meanflow_loss
from neuromf.models.toy_mlp import ToyMLP
from neuromf.sampling.multi_step import sample_euler
from neuromf.sampling.one_step import sample_one_step
from neuromf.utils.time_sampler import sample_t_and_r

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def train(cfg: OmegaConf) -> tuple[ToyMLP, list[float]]:
    """Train MeanFlow MLP on toroid data.

    Returns:
        Tuple of (trained model, list of per-epoch mean losses).
    """
    torch.manual_seed(cfg.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Dataset
    ds_cfg = ToroidConfig(n_samples=cfg.data.n_samples, mode=cfg.data.mode)
    dataset = ToroidDataset(ds_cfg, seed=cfg.training.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Model
    model = ToyMLP(
        data_dim=cfg.model.data_dim,
        hidden_dim=cfg.model.hidden_dim,
        n_layers=cfg.model.n_layers,
        prediction_type=cfg.model.prediction_type,
    ).to(device)
    log.info("Model: %d parameters", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    epoch_losses: list[float] = []

    for epoch in range(cfg.training.epochs):
        model.train()
        batch_losses: list[float] = []

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

            batch_losses.append(result["loss"].item())

        mean_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(mean_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            log.info(
                "Epoch %d/%d — loss: %.6f (fm: %.4f, mf: %.4f)",
                epoch + 1,
                cfg.training.epochs,
                mean_loss,
                result["loss_fm"].item(),
                result["loss_mf"].item(),
            )

    return model, epoch_losses


def compute_torus_metrics(samples: torch.Tensor) -> dict:
    """Compute verification metrics for R^4 torus samples.

    Args:
        samples: Tensor of shape (N, 4).

    Returns:
        Dict with mean_torus_distance, mean_pair1_error, mean_pair2_error,
        theta1_ks_pvalue, theta2_ks_pvalue.
    """
    norms = samples.norm(dim=1)
    torus_distance = (norms - 1.0).abs()

    pair1 = samples[:, 0] ** 2 + samples[:, 1] ** 2
    pair2 = samples[:, 2] ** 2 + samples[:, 3] ** 2

    theta1 = torch.atan2(samples[:, 1], samples[:, 0]).numpy()
    theta2 = torch.atan2(samples[:, 3], samples[:, 2]).numpy()

    # Normalise to [0, 1] for KS test against Uniform
    theta1_norm = (theta1 + np.pi) / (2 * np.pi)
    theta2_norm = (theta2 + np.pi) / (2 * np.pi)

    ks1 = stats.kstest(theta1_norm, "uniform")
    ks2 = stats.kstest(theta2_norm, "uniform")

    return {
        "mean_torus_distance": float(torus_distance.mean()),
        "std_torus_distance": float(torus_distance.std()),
        "mean_pair1_error": float((pair1 - 0.5).abs().mean()),
        "mean_pair2_error": float((pair2 - 0.5).abs().mean()),
        "theta1_ks_pvalue": float(ks1.pvalue),
        "theta2_ks_pvalue": float(ks2.pvalue),
        "theta1_ks_statistic": float(ks1.statistic),
        "theta2_ks_statistic": float(ks2.statistic),
    }


def generate_and_evaluate(model: ToyMLP, cfg: OmegaConf, output_dir: Path) -> dict:
    """Generate samples and compute metrics.

    Returns:
        Full metrics dict.
    """
    device = next(model.parameters()).device
    model.eval()

    n_samples = 1000
    noise = torch.randn(n_samples, cfg.model.data_dim, device=device)

    # 1-NFE sampling
    samples_1nfe = sample_one_step(model, noise).cpu()
    log.info("1-NFE: generated %d samples", n_samples)

    # Multi-step (5 steps) sampling
    samples_5step = sample_euler(model, noise, n_steps=5).cpu()
    log.info("5-step Euler: generated %d samples", n_samples)

    # Save samples
    torch.save(
        {"one_step": samples_1nfe, "multi_step": samples_5step},
        output_dir / "generated_samples.pt",
    )

    # Compute metrics
    metrics_1nfe = compute_torus_metrics(samples_1nfe)
    metrics_5step = compute_torus_metrics(samples_5step)

    log.info(
        "1-NFE metrics: torus_dist=%.4f, KS1_p=%.4f, KS2_p=%.4f",
        metrics_1nfe["mean_torus_distance"],
        metrics_1nfe["theta1_ks_pvalue"],
        metrics_1nfe["theta2_ks_pvalue"],
    )
    log.info(
        "5-step metrics: torus_dist=%.4f, KS1_p=%.4f, KS2_p=%.4f",
        metrics_5step["mean_torus_distance"],
        metrics_5step["theta1_ks_pvalue"],
        metrics_5step["theta2_ks_pvalue"],
    )

    return {"one_step": metrics_1nfe, "multi_step": metrics_5step}


def plot_results(
    epoch_losses: list[float],
    output_dir: Path,
) -> None:
    """Generate loss curve and sample projection figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MeanFlow Training Loss (Flat Torus)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # Sample projections (if samples exist)
    samples_file = output_dir / "generated_samples.pt"
    if samples_file.exists():
        data = torch.load(samples_file, weights_only=True)
        for name, samples in data.items():
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # z1 vs z2
            axes[0].scatter(
                samples[:, 0].numpy(),
                samples[:, 1].numpy(),
                s=2,
                alpha=0.5,
            )
            theta = np.linspace(-np.pi, np.pi, 200)
            r = 1.0 / np.sqrt(2.0)
            axes[0].plot(r * np.cos(theta), r * np.sin(theta), "r--", alpha=0.5)
            axes[0].set_xlabel("z1")
            axes[0].set_ylabel("z2")
            axes[0].set_title(f"{name}: (z1, z2) projection")
            axes[0].set_aspect("equal")

            # z3 vs z4
            axes[1].scatter(
                samples[:, 2].numpy(),
                samples[:, 3].numpy(),
                s=2,
                alpha=0.5,
            )
            axes[1].plot(r * np.cos(theta), r * np.sin(theta), "r--", alpha=0.5)
            axes[1].set_xlabel("z3")
            axes[1].set_ylabel("z4")
            axes[1].set_title(f"{name}: (z3, z4) projection")
            axes[1].set_aspect("equal")

            fig.tight_layout()
            fig.savefig(fig_dir / f"{name}_projections.png", dpi=150)
            plt.close(fig)

        # Angular histograms
        for name, samples in data.items():
            theta1 = torch.atan2(samples[:, 1], samples[:, 0]).numpy()
            theta2 = torch.atan2(samples[:, 3], samples[:, 2]).numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].hist(theta1, bins=50, density=True, alpha=0.7)
            axes[0].axhline(1.0 / (2 * np.pi), color="r", linestyle="--", label="Uniform")
            axes[0].set_xlabel("theta1")
            axes[0].set_ylabel("Density")
            axes[0].set_title(f"{name}: theta1 distribution")
            axes[0].legend()

            axes[1].hist(theta2, bins=50, density=True, alpha=0.7)
            axes[1].axhline(1.0 / (2 * np.pi), color="r", linestyle="--", label="Uniform")
            axes[1].set_xlabel("theta2")
            axes[1].set_ylabel("Density")
            axes[1].set_title(f"{name}: theta2 distribution")
            axes[1].legend()

            fig.tight_layout()
            fig.savefig(fig_dir / f"{name}_angular_hist.png", dpi=150)
            plt.close(fig)

    log.info("Figures saved to %s", fig_dir)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Toy toroid MeanFlow experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/toy_toroid.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Config: %s", OmegaConf.to_yaml(cfg))

    # Train
    model, epoch_losses = train(cfg)

    # Save loss curves
    with open(output_dir / "loss_curves.json", "w") as f:
        json.dump({"epoch_losses": epoch_losses}, f, indent=2)
    log.info("Saved loss curves (%d epochs)", len(epoch_losses))

    # Generate and evaluate
    metrics = generate_and_evaluate(model, cfg, output_dir)

    # Add training metrics
    metrics["u_pred_final_loss"] = epoch_losses[-1]

    with open(output_dir / "verification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved verification metrics")

    # Plot
    plot_results(epoch_losses, output_dir)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info("Final loss: %.6f", epoch_losses[-1])
    log.info(
        "1-NFE torus distance: %.4f (target < 0.1)",
        metrics["one_step"]["mean_torus_distance"],
    )
    log.info(
        "5-step torus distance: %.4f",
        metrics["multi_step"]["mean_torus_distance"],
    )
    log.info(
        "Angular KS p-values: theta1=%.4f, theta2=%.4f (target > 0.01)",
        metrics["one_step"]["theta1_ks_pvalue"],
        metrics["one_step"]["theta2_ks_pvalue"],
    )


if __name__ == "__main__":
    main()
