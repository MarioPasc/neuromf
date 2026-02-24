"""CLI script for generating latent samples at multiple NFE levels.

Loads a trained MeanFlow checkpoint (with EMA weights), generates latents
via 1-step or multi-step sampling, and stores results in HDF5 archives.

Usage:
    python experiments/cli/generate_latents.py \
        --config configs/generate.yaml \
        --checkpoint /path/to/best.ckpt \
        --nfe 1 2 5 10 25 50 \
        --n-samples 2000

    # With Picasso overlay:
    python experiments/cli/generate_latents.py \
        --config configs/picasso/generate.yaml \
        --configs-dir configs/picasso \
        --use-ema
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler

from neuromf.generation.latent_generator import LatentGenerator
from neuromf.utils.ema import EMAModel
from neuromf.wrappers.maisi_unet import MAISIUNetWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate latent samples at multiple NFE levels.")
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help="Config YAML paths, merged left-to-right on top of base.yaml.",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Lightning checkpoint. Overrides config.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=None,
        help="NFE levels (default: from config).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples per NFE level (default: from config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Generation batch size (default: from config).",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Apply EMA shadow weights to model (recommended).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for latent archives. Overrides config.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> OmegaConf:
    """Load and merge config layers."""
    config_paths = [Path(p) for p in args.config]
    configs_dir = Path(args.configs_dir) if args.configs_dir else config_paths[0].parent

    base_path = configs_dir / "base.yaml"
    if not base_path.exists():
        logger.error("base.yaml not found at %s", base_path)
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent.parent
    main_train_path = project_root / "configs" / "train_meanflow.yaml"
    main_gen_path = project_root / "configs" / "generate.yaml"

    layers = [OmegaConf.load(base_path)]
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


def _build_model(config: OmegaConf, device: torch.device) -> MAISIUNetWrapper:
    """Build the UNet model from config."""
    from neuromf.wrappers.maisi_unet import MAISIUNetConfig

    unet_cfg = MAISIUNetConfig.from_omegaconf(config)
    model = MAISIUNetWrapper(unet_cfg)
    model = model.to(device)
    model.eval()
    return model


def _load_checkpoint_and_apply_ema(
    model: MAISIUNetWrapper,
    checkpoint_path: str,
    use_ema: bool,
) -> int:
    """Load checkpoint and optionally apply EMA weights.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to Lightning ``.ckpt`` file.
        use_ema: If True, apply EMA shadow weights.

    Returns:
        Checkpoint epoch number.
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract model state dict from Lightning checkpoint
    state_dict = ckpt.get("state_dict", ckpt)
    # Lightning prefixes keys with "model." — strip it
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model.") :]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded model weights (%d keys)", len(cleaned))

    epoch = ckpt.get("epoch", 0)

    if use_ema and "ema_state_dict" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow(model)
        logger.info("Applied EMA shadow weights (decay=%.4f)", ema.decay)
    elif use_ema:
        logger.warning("--use-ema specified but no ema_state_dict in checkpoint")

    return epoch


def main() -> None:
    """Main entry point for latent generation."""
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    config = _load_config(args)

    # Resolve parameters (CLI overrides config)
    gen_cfg = config.generation
    nfe_levels = args.nfe or list(gen_cfg.nfe_levels)
    n_samples = args.n_samples or int(gen_cfg.n_samples)
    batch_size = args.batch_size or int(gen_cfg.batch_size_generate)
    prediction_type = str(gen_cfg.prediction_type)
    base_seed = int(gen_cfg.base_seed)
    latent_shape = tuple(gen_cfg.latent_shape)

    checkpoint_path = args.checkpoint or str(config.paths.get("checkpoint", ""))
    if not checkpoint_path or not Path(checkpoint_path).exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    output_dir = Path(args.output_dir or config.paths.generation_dir) / "latents"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load latent stats
    stats_path = Path(config.paths.latent_stats)
    latent_stats = None
    if stats_path.exists():
        with open(stats_path) as f:
            latent_stats = json.load(f)
        logger.info("Loaded latent stats from %s", stats_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Build model and load weights
    model = _build_model(config, device)
    epoch = _load_checkpoint_and_apply_ema(model, checkpoint_path, args.use_ema)

    n_params = sum(p.numel() for p in model.parameters())
    extra_meta = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": epoch,
        "model_params": n_params,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "scale_factor": float(config.vae.scale_factor),
    }

    # Pre-generate shared noise (same z_1 for all NFE levels)
    generator = LatentGenerator(model, prediction_type, device)
    shared_noise = generator._pre_generate_noise(n_samples, latent_shape, base_seed)
    logger.info("Pre-generated shared noise: %s (seed=%d)", shared_noise.shape, base_seed)

    # Generate for each NFE level with the same noise
    for nfe in nfe_levels:
        out_path = output_dir / f"nfe_{nfe:03d}.h5"
        if out_path.exists():
            logger.info("Skipping NFE=%d (already exists: %s)", nfe, out_path)
            continue

        generator.generate(
            n_samples=n_samples,
            nfe=nfe,
            output_path=out_path,
            batch_size=batch_size,
            base_seed=base_seed,
            latent_stats=latent_stats,
            metadata=extra_meta,
            latent_shape=latent_shape,
            shared_noise=shared_noise,
        )

    # Write generation manifest
    manifest = {
        "experiment": "stage1_healthy",
        "model": "NeuroMF",
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": epoch,
        "n_samples_per_nfe": n_samples,
        "nfe_levels": nfe_levels,
        "base_seed": base_seed,
        "prediction_type": prediction_type,
        "latent_shape": list(latent_shape),
        "scale_factor": float(config.vae.scale_factor),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "gpu": extra_meta["gpu"],
    }
    manifest_path = output_dir.parent / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest saved: %s", manifest_path)


if __name__ == "__main__":
    main()
