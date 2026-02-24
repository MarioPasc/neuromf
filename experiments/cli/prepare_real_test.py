"""CLI script to prepare real test set volumes and extract features.

Loads test-split latents from Phase 1 HDF5 shards, denormalises,
decodes through the frozen MAISI VAE, stores decoded volumes in HDF5,
and extracts features (Med3D / RadImageNet) for metric computation.

Usage:
    python experiments/cli/prepare_real_test.py \
        --config configs/generate.yaml \
        --output-dir /path/to/phase_5/
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

from neuromf.data.latent_dataset import LatentDataset
from neuromf.generation.h5_manager import H5Manager, H5VolumeConfig
from neuromf.wrappers.maisi_vae import MAISIVAEConfig, MAISIVAEWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare real test set volumes and features.")
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for real test volumes and features.",
    )
    parser.add_argument(
        "--max-volumes",
        type=int,
        default=None,
        help="Limit number of test volumes (for debugging).",
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


@torch.no_grad()
def main() -> None:
    """Main entry point."""
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    config = _load_config(args)

    latent_dir = Path(config.paths.latents_dir)
    output_dir = Path(args.output_dir or str(config.paths.get("generation_dir", "")))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load latent stats for denormalisation
    stats_path = latent_dir / "latent_stats.json"
    if not stats_path.exists():
        logger.error("Latent stats not found: %s", stats_path)
        sys.exit(1)

    with open(stats_path) as f:
        stats = json.load(f)

    per_ch = stats["per_channel"]
    n_channels = len(per_ch)
    mean_list = [per_ch[f"channel_{c}"]["mean"] for c in range(n_channels)]
    std_list = [per_ch[f"channel_{c}"]["std"] for c in range(n_channels)]

    latent_mean = torch.tensor(mean_list, dtype=torch.float32, device=device).reshape(
        1, -1, 1, 1, 1
    )
    latent_std = torch.tensor(std_list, dtype=torch.float32, device=device).reshape(1, -1, 1, 1, 1)

    # Build test dataset
    split_ratios = list(config.training.get("split_ratios", [0.85, 0.10, 0.05]))
    split_seed = int(config.training.get("split_seed", 42))

    test_ds = LatentDataset(
        latent_dir,
        normalise=True,
        split="test",
        split_ratios=split_ratios,
        split_seed=split_seed,
    )
    n_test = len(test_ds)
    if args.max_volumes:
        n_test = min(n_test, args.max_volumes)
    logger.info("Test set: %d volumes", n_test)

    # Build VAE
    vae_cfg = MAISIVAEConfig.from_omegaconf(config)
    vae = MAISIVAEWrapper(vae_cfg, device=device)

    # Decode test latents -> volumes
    vol_path = output_dir / "real_test.h5"
    if vol_path.exists():
        logger.info("Real test volumes already exist: %s", vol_path)
    else:
        vol_config = H5VolumeConfig(n_samples=n_test)
        vol_meta = {
            "source": "test_split",
            "n_samples": n_test,
            "volume_shape": list(vol_config.volume_shape),
        }
        vol_h5 = H5Manager.create_volume_archive(vol_path, vol_config, vol_meta)

        try:
            for i in range(n_test):
                z_hat, _ = test_ds[i]  # normalised latent

                # Denormalise
                z_0 = z_hat.unsqueeze(0).to(device) * latent_std + latent_mean

                # VAE decode (divides by scale_factor internally)
                x_hat = vae.decode(z_0).clamp(0.0, 1.0)

                H5Manager.write_volume_batch(vol_h5, i, x_hat.cpu(), [0.0])

                if (i + 1) % 10 == 0 or i == n_test - 1:
                    logger.info("  Decoded %d / %d test volumes", i + 1, n_test)
        finally:
            vol_h5.close()

        logger.info("Real test volumes saved: %s", vol_path)

    logger.info("Real test preparation complete.")


if __name__ == "__main__":
    main()
