"""Post-training figure generation from sample archives.

Loads a sample_archive.pt, optionally decodes samples through the MAISI VAE,
and generates all evolution plots + decoded NFE comparison figures. Use this
when on_fit_end figure generation failed or was skipped.

Usage:
    # Full run (decode + plots) — requires GPU for VAE decode:
    python experiments/cli/generate_figures.py \
        --samples-dir /path/to/results/ablations/xpred_exact_jvp/samples \
        --config configs/picasso/train_meanflow.yaml

    # Latent-only plots (no VAE decode, CPU-safe):
    python experiments/cli/generate_figures.py \
        --samples-dir /path/to/results/ablations/xpred_exact_jvp/samples \
        --skip-decode

    # Custom output directory:
    python experiments/cli/generate_figures.py \
        --samples-dir /path/to/samples \
        --config configs/picasso/train_meanflow.yaml \
        --output-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate figures from a sample archive (post-training)."
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        required=True,
        help="Path to the samples directory containing sample_archive.pt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (for VAE settings). Required unless --skip-decode.",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml. Defaults to parent of --config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for figures. Defaults to {samples-dir}/../figures/.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip VAE decoding; only generate latent-space plots.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=None,
        help="NFE levels to decode (e.g. 1 2 5 10). Defaults to all in archive.",
    )
    return parser.parse_args()


def _load_config(config_path: Path, configs_dir: Path | None) -> OmegaConf:
    """Load merged config for VAE settings.

    Args:
        config_path: Path to training config YAML.
        configs_dir: Directory containing base.yaml.

    Returns:
        Merged OmegaConf config.
    """
    from neuromf.config import load_merged_config

    return load_merged_config(
        config_path,
        configs_dir=configs_dir,
        require_base=False,
    )


@torch.no_grad()
def _decode_samples(
    archive: dict,
    vae: object,
    nfe_steps: list[int],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Decode samples from the archive through the MAISI VAE.

    Args:
        archive: Loaded sample_archive.pt.
        vae: MAISIVAEWrapper instance.
        nfe_steps: NFE levels to decode.
        device: Torch device for decode.

    Returns:
        Tuple of (decoded_nfe, decoded_1nfe_by_epoch).
    """
    latent_mean = archive["latent_mean"]
    latent_std = archive["latent_std"]
    if latent_mean.ndim == 1:
        latent_mean = latent_mean.view(1, -1, 1, 1, 1)
        latent_std = latent_std.view(1, -1, 1, 1, 1)

    epochs = sorted(archive["epochs"])
    last_epoch = epochs[-1]
    last_key = f"epoch_{last_epoch:04d}"

    # 1. Decode all NFE levels for last epoch (sample #0)
    decoded_nfe: dict[int, np.ndarray] = {}
    for nfe in nfe_steps:
        nfe_key = f"nfe_{nfe}"
        if nfe_key not in archive[last_key]:
            logger.warning("NFE key %s not in last epoch; skipping", nfe_key)
            continue
        z_norm = archive[last_key][nfe_key]
        z_denorm = (z_norm[0:1] * latent_std + latent_mean).float().to(device)
        decoded = vae.decode(z_denorm).cpu().float()
        decoded_nfe[nfe] = decoded[0, 0].numpy()
        del z_denorm, decoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Decoded epoch %d NFE=%d", last_epoch, nfe)

    # 2. Decode 1-NFE sample #0 for all epochs
    decoded_1nfe_by_epoch: dict[int, np.ndarray] = {}
    for ep in epochs:
        epoch_key = f"epoch_{ep:04d}"
        if "nfe_1" not in archive[epoch_key]:
            continue
        z_norm = archive[epoch_key]["nfe_1"]
        z_denorm = (z_norm[0:1] * latent_std + latent_mean).float().to(device)
        decoded = vae.decode(z_denorm).cpu().float()
        decoded_1nfe_by_epoch[ep] = decoded[0, 0].numpy()
        del z_denorm, decoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "Decoded %d NFE levels (last epoch) + %d epochs (1-NFE)",
        len(decoded_nfe),
        len(decoded_1nfe_by_epoch),
    )
    return decoded_nfe, decoded_1nfe_by_epoch


def main() -> None:
    """Main entry point."""
    args = parse_args()

    samples_dir = Path(args.samples_dir)
    archive_path = samples_dir / "sample_archive.pt"
    if not archive_path.exists():
        logger.error("sample_archive.pt not found at %s", archive_path)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else samples_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load archive
    logger.info("Loading archive: %s", archive_path)
    archive = torch.load(archive_path, map_location="cpu", weights_only=False)
    n_epochs = len(archive.get("epochs", []))
    logger.info(
        "Archive: %d epochs, keys per epoch: %s", n_epochs, list(archive.get("metadata", {}).keys())
    )

    # Determine NFE steps
    if args.nfe:
        nfe_steps = args.nfe
    else:
        nfe_steps = archive.get("metadata", {}).get("nfe_steps", [1, 2, 5, 10])

    # VAE decode
    decoded_nfe: dict[int, np.ndarray] = {}
    decoded_1nfe_by_epoch: dict[int, np.ndarray] = {}

    if not args.skip_decode:
        if not args.config:
            logger.error(
                "--config is required for VAE decode. Use --skip-decode for latent-only plots."
            )
            sys.exit(1)

        config_path = Path(args.config)
        configs_dir = Path(args.configs_dir) if args.configs_dir else None
        config = _load_config(config_path, configs_dir)

        vae_weights = str(config.paths.get("maisi_vae_weights", ""))
        if not vae_weights or not Path(vae_weights).exists():
            logger.error("VAE weights not found: %s", vae_weights)
            sys.exit(1)

        from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

        vae_cfg = config.vae
        vae_config = MAISIVAEConfig(
            weights_path=vae_weights,
            scale_factor=float(vae_cfg.scale_factor),
            spatial_dims=int(vae_cfg.spatial_dims),
            in_channels=int(vae_cfg.in_channels),
            out_channels=int(vae_cfg.out_channels),
            latent_channels=int(vae_cfg.latent_channels),
            num_channels=list(vae_cfg.num_channels),
            num_res_blocks=list(vae_cfg.num_res_blocks),
            norm_num_groups=int(vae_cfg.norm_num_groups),
            norm_eps=float(vae_cfg.norm_eps),
            attention_levels=list(vae_cfg.attention_levels),
            with_encoder_nonlocal_attn=bool(vae_cfg.with_encoder_nonlocal_attn),
            with_decoder_nonlocal_attn=bool(vae_cfg.with_decoder_nonlocal_attn),
            use_checkpointing=bool(vae_cfg.use_checkpointing),
            use_convtranspose=bool(vae_cfg.use_convtranspose),
            norm_float16=bool(vae_cfg.norm_float16),
            num_splits=int(vae_cfg.num_splits),
            dim_split=int(vae_cfg.dim_split),
            downsample_factor=int(vae_cfg.downsample_factor),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading MAISI VAE on %s...", device)
        vae = MAISIVAEWrapper(vae_config, device=device)

        decoded_nfe, decoded_1nfe_by_epoch = _decode_samples(
            archive,
            vae,
            nfe_steps,
            device,
        )

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("Skipping VAE decode (--skip-decode)")

    # Generate all figures
    from neuromf.utils.sample_plots import (
        plot_channel_stats_evolution,
        plot_decoded_nfe_grid,
        plot_inter_epoch_delta,
        plot_nfe_comparison_grid,
        plot_nfe_consistency_evolution,
        plot_sample_evolution_grid,
        plot_spectral_evolution,
    )

    logger.info("Generating figures to %s...", output_dir)

    plot_sample_evolution_grid(archive, output_dir)
    plot_channel_stats_evolution(archive, output_dir)
    plot_spectral_evolution(archive, output_dir)
    plot_nfe_comparison_grid(
        archive,
        output_dir,
        decoded_volumes=decoded_1nfe_by_epoch if decoded_1nfe_by_epoch else None,
    )
    plot_nfe_consistency_evolution(archive, output_dir)
    plot_inter_epoch_delta(archive, output_dir)

    if decoded_nfe:
        plot_decoded_nfe_grid(decoded_nfe, output_dir)

    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
