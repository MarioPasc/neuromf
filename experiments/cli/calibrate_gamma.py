"""Post-training γ calibration for 1-NFE norm correction.

Loads a trained checkpoint + EMA, generates samples at multiple γ values,
and selects the γ that makes per-channel std closest to 1.0.

The optimal γ compensates for compound velocity norm overshoot observed
in MeanFlow models (v1 converged to ~1.19 overshoot).

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/calibrate_gamma.py \
        --config configs/train_v2_stage2.yaml \
        --checkpoint /path/to/best.ckpt \
        --n-samples 100 \
        --gammas 1.0 1.05 1.10 1.15 1.20 1.25
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler

from neuromf.models.latent_meanflow import LatentMeanFlow
from neuromf.sampling.one_step import sample_one_step

logger = logging.getLogger(__name__)


def calibrate_gamma(
    model: LatentMeanFlow,
    n_samples: int,
    gammas: list[float],
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict[float, dict[str, float]]:
    """Generate samples at each gamma and compute per-channel std.

    Args:
        model: Trained LatentMeanFlow model (EMA applied).
        n_samples: Number of samples to generate per gamma.
        gammas: List of gamma values to test.
        seed: Random seed for noise generation.
        device: Device for generation.

    Returns:
        Dict mapping gamma -> {channel_stds, mean_std, std_error}.
    """
    model.eval()
    model.to(device)

    # Get latent shape from model config
    in_channels = model._in_channels
    spatial = model._latent_spatial
    noise_shape = (n_samples, in_channels, spatial, spatial, spatial)

    # Generate fixed noise
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(*noise_shape, generator=gen, device=device)

    prediction_type = str(model.cfg.meanflow.prediction_type)

    results: dict[float, dict[str, float]] = {}

    for gamma in gammas:
        with torch.no_grad():
            z_0 = sample_one_step(
                model.net,
                noise,
                prediction_type=prediction_type,
                norm_correction=gamma,
            )

        # Per-channel std across samples
        # z_0 shape: (N, C, D, H, W), compute std over (N, D, H, W)
        per_ch_std = z_0.std(dim=(0, 2, 3, 4)).cpu().tolist()
        mean_std = sum(per_ch_std) / len(per_ch_std)
        std_error = abs(mean_std - 1.0)

        results[gamma] = {
            "channel_stds": per_ch_std,
            "mean_std": mean_std,
            "std_error_from_1": std_error,
        }
        logger.info(
            "γ=%.3f: per-channel std=%s, mean=%.4f, |mean-1|=%.4f",
            gamma,
            [f"{s:.4f}" for s in per_ch_std],
            mean_std,
            std_error,
        )

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    parser = argparse.ArgumentParser(description="Calibrate γ for 1-NFE norm correction")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Config file(s) (merged left-to-right)",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default="configs",
        help="Base config directory (for base.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per gamma (default: 100)",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[1.0, 1.05, 1.10, 1.15, 1.20, 1.25],
        help="Gamma values to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noise",
    )
    args = parser.parse_args()

    # Load config
    configs_dir = Path(args.configs_dir)
    base_path = configs_dir / "base.yaml"
    if base_path.exists():
        base = OmegaConf.load(base_path)
    else:
        base = OmegaConf.create()

    merged = base
    for cfg_path in args.config:
        merged = OmegaConf.merge(merged, OmegaConf.load(cfg_path))

    # Load model from checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model = LatentMeanFlow.load_from_checkpoint(
        args.checkpoint,
        config=merged,
        map_location="cpu",
    )

    # Apply EMA weights
    logger.info("Applying EMA weights (active index)")
    model.ema.apply_shadow(model.net)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Run calibration
    results = calibrate_gamma(
        model=model,
        n_samples=args.n_samples,
        gammas=args.gammas,
        seed=args.seed,
        device=device,
    )

    # Find optimal gamma
    best_gamma = min(results, key=lambda g: results[g]["std_error_from_1"])
    logger.info(
        "Optimal γ=%.3f (mean_std=%.4f, error=%.4f)",
        best_gamma,
        results[best_gamma]["mean_std"],
        results[best_gamma]["std_error_from_1"],
    )

    # Serialize results
    output = {
        "gammas_tested": args.gammas,
        "optimal_gamma": best_gamma,
        "results": {str(k): v for k, v in results.items()},
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Results saved to %s", out_path)
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
