"""CLI script for decoding latent HDF5 archives through the MAISI VAE.

Reads normalised latents, denormalises, decodes through the frozen VAE,
and writes decoded volumes to HDF5 archives.

Usage:
    python experiments/cli/decode_volumes.py \
        --config configs/generate.yaml \
        --latent-dir /path/to/generation/latents/ \
        --nfe 1 10 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from rich.logging import RichHandler

from neuromf.generation.volume_decoder import VolumeDecoder
from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Decode latent archives through MAISI VAE.")
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
        "--latent-dir",
        type=str,
        default=None,
        help="Directory containing latent HDF5 files.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=None,
        help="NFE levels to decode (default: from config decode_nfe_levels).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for volume archives.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Decode batch size (default: 1).",
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


def main() -> None:
    """Main entry point for volume decoding."""
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    config = _load_config(args)
    gen_cfg = config.generation

    nfe_levels = args.nfe or list(gen_cfg.decode_nfe_levels)
    batch_size = args.batch_size or int(gen_cfg.batch_size_decode)
    latent_dir = Path(args.latent_dir or str(config.paths.generation_dir) + "/latents")
    output_dir = Path(args.output_dir or str(config.paths.generation_dir) + "/volumes")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load latent stats for denormalisation
    stats_path = Path(config.paths.latent_stats)
    if not stats_path.exists():
        logger.error("Latent stats not found: %s", stats_path)
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    per_ch = stats["per_channel"]
    n_channels = len(per_ch)
    mean_list = [per_ch[f"channel_{c}"]["mean"] for c in range(n_channels)]
    std_list = [per_ch[f"channel_{c}"]["std"] for c in range(n_channels)]

    latent_mean = torch.tensor(mean_list, dtype=torch.float32)
    latent_std = torch.tensor(std_list, dtype=torch.float32)
    logger.info("Latent stats: mean=%s, std=%s", latent_mean.tolist(), latent_std.tolist())

    # Build VAE
    vae_cfg = MAISIVAEConfig.from_omegaconf(config)
    vae = MAISIVAEWrapper(vae_cfg, device=device)

    decoder = VolumeDecoder(
        vae=vae,
        latent_mean=latent_mean,
        latent_std=latent_std,
        device=device,
    )

    for nfe in nfe_levels:
        latent_path = latent_dir / f"nfe_{nfe:03d}.h5"
        if not latent_path.exists():
            logger.warning("Latent archive not found for NFE=%d: %s", nfe, latent_path)
            continue

        vol_path = output_dir / f"nfe_{nfe:03d}.h5"
        if vol_path.exists():
            logger.info("Skipping NFE=%d (already exists: %s)", nfe, vol_path)
            continue

        decoder.decode(latent_path, vol_path, batch_size=batch_size)

    logger.info("Decoding complete. Volumes in: %s", output_dir)


if __name__ == "__main__":
    main()
