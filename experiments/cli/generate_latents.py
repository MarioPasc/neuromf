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
    # Enhancement flags
    parser.add_argument(
        "--norm-correction",
        type=float,
        default=None,
        help="Norm correction gamma (divides velocity). Default: from config or 1.0.",
    )
    parser.add_argument(
        "--sigma-inject",
        type=float,
        nargs=4,
        default=None,
        metavar=("S0", "S1", "S2", "S3"),
        help="Per-channel noise injection std (4 values). Overrides auto-calibration.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Auto-compute sigma_inject from NFE=1 vs NFE=50 baseline samples.",
    )
    parser.add_argument(
        "--variance-rescale",
        action="store_true",
        help="Enable per-channel variance rescaling to match training data stats.",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Generate both baseline (no enhancements) and enhanced with same noise.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> OmegaConf:
    """Load and merge config layers."""
    from neuromf.config import load_merged_config

    config_paths = [Path(p) for p in args.config]
    configs_dir = Path(args.configs_dir) if args.configs_dir else None

    return load_merged_config(
        config_paths,
        configs_dir=configs_dir,
        include_generate=True,
        require_base=True,
    )


def _is_archive_complete(path: Path) -> bool:
    """Check if an HDF5 archive has all expected samples written."""
    from neuromf.generation.h5_manager import H5Manager

    return H5Manager.is_complete(path)


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
    # Lightning prefixes keys with "model." or "net." — strip to match MAISIUNetWrapper
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model.") :]] = v
        elif k.startswith("net."):
            cleaned[k[len("net.") :]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=False)
    logger.info("Loaded model weights (%d keys)", len(cleaned))

    epoch = ckpt.get("epoch", 0)

    if use_ema and "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]
        # Auto-detect multi-EMA vs single-EMA format
        if "emas" in ema_state:
            from neuromf.utils.ema import MultiEMAModel

            ema = MultiEMAModel(
                model, decays=ema_state["decays"], active_index=ema_state.get("active_index", -1)
            )
            ema.load_state_dict(ema_state)
        else:
            ema = EMAModel(model)
            ema.load_state_dict(ema_state)
        ema.apply_shadow(model)
        logger.info("Applied EMA shadow weights (decay=%.4f)", ema.decay)
    elif use_ema:
        logger.warning("--use-ema specified but no ema_state_dict in checkpoint")

    return epoch


def _auto_calibrate_sigma_inject(
    model: MAISIUNetWrapper,
    prediction_type: str,
    device: torch.device,
    latent_shape: tuple[int, ...],
    n_cal: int = 50,
    seed: int = 42,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute per-channel sigma_inject from NFE=1 vs NFE=50 baseline variance.

    sigma_inject_c = sqrt(max(0, sigma_50_c^2 - sigma_1_c^2))

    Args:
        model: Trained MeanFlow model.
        prediction_type: ``"u"`` or ``"x"``.
        device: Compute device.
        latent_shape: Per-latent shape (e.g. ``(4, 48, 48, 48)``).
        n_cal: Number of calibration samples.
        seed: Random seed for calibration noise.
        batch_size: Batch size for calibration forward passes.

    Returns:
        Per-channel sigma_inject tensor of shape ``(C,)``.
    """
    from neuromf.sampling.multi_step import sample_euler
    from neuromf.sampling.one_step import sample_one_step

    logger.info("Auto-calibrating sigma_inject with %d samples...", n_cal)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(n_cal, *latent_shape, generator=gen)

    # NFE=1: deterministic 1-step (collapsed variance)
    samples_1 = []
    for i in range(0, n_cal, batch_size):
        b = min(batch_size, n_cal - i)
        with torch.no_grad():
            s = sample_one_step(model, noise[i : i + b].to(device), prediction_type)
        samples_1.append(s.cpu())
    samples_1 = torch.cat(samples_1)

    # NFE=50: multi-step (full variance)
    samples_50 = []
    for i in range(0, n_cal, batch_size):
        b = min(batch_size, n_cal - i)
        with torch.no_grad():
            s = sample_euler(model, noise[i : i + b].to(device), 50, prediction_type)
        samples_50.append(s.cpu())
    samples_50 = torch.cat(samples_50)

    # Per-channel std across batch and spatial dims
    std_1 = samples_1.float().std(dim=(0, 2, 3, 4))
    std_50 = samples_50.float().std(dim=(0, 2, 3, 4))
    sigma_inject = torch.sqrt(torch.clamp(std_50**2 - std_1**2, min=0.0))

    logger.info("  NFE=1  per-channel std: %s", [f"{s:.4f}" for s in std_1.tolist()])
    logger.info("  NFE=50 per-channel std: %s", [f"{s:.4f}" for s in std_50.tolist()])
    logger.info("  sigma_inject:           %s", [f"{s:.4f}" for s in sigma_inject.tolist()])
    return sigma_inject


def _make_post_process_fn(
    sigma_inject: torch.Tensor | None,
    mu_data: torch.Tensor | None,
    sigma_data: torch.Tensor | None,
    do_variance_rescale: bool,
) -> callable | None:
    """Build a composable post-processing function for enhancements.

    Returns None if no enhancements are active.
    """
    from neuromf.sampling.variance_rescaling import stochastic_perturb, variance_rescale

    if sigma_inject is None and not do_variance_rescale:
        return None

    def post_process(z: torch.Tensor) -> torch.Tensor:
        if sigma_inject is not None:
            z = stochastic_perturb(z, sigma_inject.to(z.device))
        if do_variance_rescale and mu_data is not None and sigma_data is not None:
            z = variance_rescale(z, mu_data.to(z.device), sigma_data.to(z.device))
        return z

    return post_process


def _generate_for_nfe_levels(
    generator: LatentGenerator,
    nfe_levels: list[int],
    output_dir: Path,
    n_samples: int,
    batch_size: int,
    base_seed: int,
    latent_stats: dict | None,
    metadata: dict | None,
    latent_shape: tuple[int, ...],
    shared_noise: torch.Tensor,
    post_process_fn: callable | None = None,
) -> None:
    """Generate latents for all NFE levels into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for nfe in nfe_levels:
        out_path = output_dir / f"nfe_{nfe:03d}.h5"
        if out_path.exists() and _is_archive_complete(out_path):
            logger.info("Skipping NFE=%d (already complete: %s)", nfe, out_path)
            continue
        elif out_path.exists():
            logger.warning("Incomplete archive for NFE=%d, recreating: %s", nfe, out_path)
            out_path.unlink()

        generator.generate(
            n_samples=n_samples,
            nfe=nfe,
            output_path=out_path,
            batch_size=batch_size,
            base_seed=base_seed,
            latent_stats=latent_stats,
            metadata=metadata,
            latent_shape=latent_shape,
            shared_noise=shared_noise,
            post_process_fn=post_process_fn,
        )


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

    gen_root = Path(args.output_dir or config.paths.generation_dir)

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

    # =========================================================================
    # Resolve enhancement parameters (CLI > config)
    # =========================================================================
    enh_raw = gen_cfg.get("enhancements", None)
    if enh_raw is not None and hasattr(enh_raw, "_metadata"):
        enh_cfg = OmegaConf.to_container(enh_raw, resolve=True) or {}
    else:
        enh_cfg = dict(enh_raw) if enh_raw else {}

    gamma = args.norm_correction
    if gamma is None:
        gamma = float(enh_cfg.get("norm_correction", 1.0))

    do_variance_rescale = args.variance_rescale or bool(
        enh_cfg.get("variance_rescale", {}).get("enabled", False)
    )

    comparison_mode = args.comparison or bool(enh_cfg.get("comparison_mode", False))

    # sigma_inject: CLI > config > auto-calibrate
    sigma_inject = None
    sp_cfg = enh_cfg.get("stochastic_perturbation", {})
    do_auto_calibrate = args.auto_calibrate or (
        bool(sp_cfg.get("enabled", False)) and bool(sp_cfg.get("auto_calibrate", False))
    )

    if args.sigma_inject is not None:
        sigma_inject = torch.tensor(args.sigma_inject, dtype=torch.float32)
        logger.info("Using CLI sigma_inject: %s", sigma_inject.tolist())
    elif sp_cfg.get("sigma_inject") is not None:
        sigma_inject = torch.tensor(sp_cfg["sigma_inject"], dtype=torch.float32)
        logger.info("Using config sigma_inject: %s", sigma_inject.tolist())
    elif do_auto_calibrate:
        cal_n = int(sp_cfg.get("calibration_n_samples", 50))
        sigma_inject = _auto_calibrate_sigma_inject(
            model,
            prediction_type,
            device,
            latent_shape,
            n_cal=cal_n,
            seed=base_seed,
            batch_size=batch_size,
        )

    # Determine if any enhancements are active
    has_enhancements = (gamma != 1.0) or (sigma_inject is not None) or do_variance_rescale

    if has_enhancements:
        logger.info("=== ENHANCEMENTS ACTIVE ===")
        logger.info("  norm_correction (gamma): %.4f", gamma)
        logger.info(
            "  sigma_inject: %s", sigma_inject.tolist() if sigma_inject is not None else "None"
        )
        logger.info("  variance_rescale: %s", do_variance_rescale)
        logger.info("  comparison_mode: %s", comparison_mode)

    # Extract per-channel data stats for variance rescaling
    mu_data = None
    sigma_data = None
    if do_variance_rescale and latent_stats is not None:
        per_ch = latent_stats.get("per_channel", {})
        n_ch = len(per_ch)
        if n_ch > 0:
            mu_data = torch.tensor(
                [per_ch[f"channel_{c}"]["mean"] for c in range(n_ch)],
                dtype=torch.float32,
            ).view(1, n_ch, 1, 1, 1)
            sigma_data = torch.tensor(
                [per_ch[f"channel_{c}"]["std"] for c in range(n_ch)],
                dtype=torch.float32,
            ).view(1, n_ch, 1, 1, 1)
            logger.info("  mu_data:    %s", [f"{m:.4f}" for m in mu_data.squeeze().tolist()])
            logger.info("  sigma_data: %s", [f"{s:.4f}" for s in sigma_data.squeeze().tolist()])

    # Build post-process function
    post_fn = _make_post_process_fn(sigma_inject, mu_data, sigma_data, do_variance_rescale)

    # =========================================================================
    # Pre-generate shared noise (same z_1 for all NFE levels and modes)
    # =========================================================================
    tmp_gen = LatentGenerator(model, prediction_type, device)
    shared_noise = tmp_gen._pre_generate_noise(n_samples, latent_shape, base_seed)
    logger.info("Pre-generated shared noise: %s (seed=%d)", shared_noise.shape, base_seed)

    # =========================================================================
    # Generate latents
    # =========================================================================
    if comparison_mode and has_enhancements:
        # --- Baseline: no enhancements ---
        baseline_dir = gen_root / "latents_baseline"
        logger.info("=== BASELINE GENERATION (no enhancements) ===")
        baseline_gen = LatentGenerator(model, prediction_type, device, norm_correction=1.0)
        _generate_for_nfe_levels(
            baseline_gen,
            nfe_levels,
            baseline_dir,
            n_samples,
            batch_size,
            base_seed,
            latent_stats,
            extra_meta,
            latent_shape,
            shared_noise,
            post_process_fn=None,
        )

        # --- Enhanced: all enhancements active ---
        enhanced_dir = gen_root / "latents_enhanced"
        logger.info("=== ENHANCED GENERATION (gamma=%.4f) ===", gamma)
        enhanced_gen = LatentGenerator(model, prediction_type, device, norm_correction=gamma)
        _generate_for_nfe_levels(
            enhanced_gen,
            nfe_levels,
            enhanced_dir,
            n_samples,
            batch_size,
            base_seed,
            latent_stats,
            extra_meta,
            latent_shape,
            shared_noise,
            post_process_fn=post_fn,
        )
    else:
        # Single mode: either baseline or enhanced
        output_dir = gen_root / "latents"
        gen = LatentGenerator(model, prediction_type, device, norm_correction=gamma)
        _generate_for_nfe_levels(
            gen,
            nfe_levels,
            output_dir,
            n_samples,
            batch_size,
            base_seed,
            latent_stats,
            extra_meta,
            latent_shape,
            shared_noise,
            post_process_fn=post_fn if has_enhancements else None,
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
        "enhancements": {
            "norm_correction": gamma,
            "sigma_inject": sigma_inject.tolist() if sigma_inject is not None else None,
            "sigma_inject_auto_calibrated": do_auto_calibrate and args.sigma_inject is None,
            "variance_rescale": do_variance_rescale,
            "comparison_mode": comparison_mode,
        },
    }
    manifest_path = gen_root / "generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest saved: %s", manifest_path)


if __name__ == "__main__":
    main()
