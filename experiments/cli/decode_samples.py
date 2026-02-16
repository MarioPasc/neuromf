"""Post-training CLI for decoding latent archives through the MAISI VAE.

Loads all epoch archives from ``generated_samples/``, decodes selected NFE
levels and sample indices through the frozen VAE, and generates visualizations
(evolution grids, NFE comparisons, latent stats plots).

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/decode_samples.py \
        --samples-dir /path/to/results/phase_4/samples \
        --config configs/train_meanflow.yaml

    # Decode only specific NFEs and samples:
    python experiments/cli/decode_samples.py \
        --samples-dir /path/to/results/phase_4/samples \
        --config configs/train_meanflow.yaml \
        --nfe 1 5 10 --sample-indices 0 1 2

    # Skip VAE decode (latent stats and visualizations only):
    python experiments/cli/decode_samples.py \
        --samples-dir /path/to/results/phase_4/samples \
        --config configs/train_meanflow.yaml \
        --skip-decode
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
        description="Decode latent sample archives through the MAISI VAE."
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        required=True,
        help="Path to the samples directory containing generated_samples/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config YAML (for VAE settings).",
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
        help="Output directory for decoded images. Defaults to {samples-dir}/decoded/.",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=None,
        help="NFE levels to decode (e.g. 1 5 10). Defaults to all in archive.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=None,
        help="Sample indices to decode (e.g. 0 1 2). Defaults to first 3.",
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Skip VAE decoding; only generate latent stats and visualizations.",
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
    if configs_dir is None:
        configs_dir = config_path.parent

    base_path = configs_dir / "base.yaml"
    layers = []
    if base_path.exists():
        layers.append(OmegaConf.load(base_path))

    project_root = Path(__file__).resolve().parent.parent.parent
    main_train_path = project_root / "configs" / "train_meanflow.yaml"
    if main_train_path.exists() and main_train_path.resolve() != config_path.resolve():
        layers.append(OmegaConf.load(main_train_path))

    layers.append(OmegaConf.load(config_path))
    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)
    return config


def _load_archives(samples_dir: Path) -> list[dict]:
    """Load all epoch archives sorted by epoch.

    Args:
        samples_dir: Root samples directory.

    Returns:
        List of archive dicts sorted by epoch.
    """
    archive_dir = samples_dir / "generated_samples"
    if not archive_dir.exists():
        logger.error("No generated_samples/ directory found at %s", samples_dir)
        return []

    files = sorted(archive_dir.glob("epoch_*.pt"))
    if not files:
        logger.error("No epoch_*.pt archives found in %s", archive_dir)
        return []

    archives = []
    for f in files:
        try:
            archive = torch.load(f, map_location="cpu", weights_only=False)
            archives.append(archive)
            logger.info("Loaded %s (epoch %d)", f.name, archive["epoch"])
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)

    return archives


def _decode_and_save(
    archive: dict,
    vae: object,
    nfe_keys: list[str],
    sample_indices: list[int],
    output_dir: Path,
    device: torch.device,
) -> None:
    """Decode selected samples from an archive and save as PNG.

    Args:
        archive: Loaded .pt archive dict.
        vae: MAISIVAEWrapper instance.
        nfe_keys: NFE keys to decode (e.g. ``["nfe_1", "nfe_5"]``).
        sample_indices: Which samples to decode.
        output_dir: Output directory for PNGs.
        device: Torch device for VAE decode.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epoch = archive["epoch"]
    latent_mean = archive["latent_mean"]
    latent_std = archive["latent_std"]

    # Reshape for broadcasting: (C,) -> (1, C, 1, 1, 1)
    if latent_mean.ndim == 1:
        latent_mean = latent_mean.view(1, -1, 1, 1, 1)
        latent_std = latent_std.view(1, -1, 1, 1, 1)

    vol_dir = output_dir / "decoded_volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    for nfe_key in nfe_keys:
        if nfe_key not in archive:
            logger.warning("NFE key %s not in epoch %d archive", nfe_key, epoch)
            continue

        z_normalized = archive[nfe_key]
        for idx in sample_indices:
            if idx >= z_normalized.shape[0]:
                continue

            # Denormalize
            z_denorm = z_normalized[idx : idx + 1] * latent_std + latent_mean
            z_denorm = z_denorm.to(device)

            # Decode through VAE
            decoded = vae.decode(z_denorm).cpu().float()

            # Save 3-view PNG
            vol = decoded[0, 0]  # (D, H, W)
            mid = [s // 2 for s in vol.shape]
            view_names = ["Sagittal", "Coronal", "Axial"]
            slices = [
                vol[mid[0], :, :],
                vol[:, mid[1], :],
                vol[:, :, mid[2]],
            ]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            vmin, vmax = vol.min().item(), vol.max().item()
            for j, (sl, name) in enumerate(zip(slices, view_names)):
                axes[j].imshow(sl.numpy().T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                axes[j].set_title(name, fontsize=11)
                axes[j].set_axis_off()

            nfe_n = nfe_key.replace("nfe_", "")
            fig.suptitle(
                f"Epoch {epoch}, NFE={nfe_n}, Sample #{idx}",
                fontsize=12,
            )
            fig.tight_layout()

            fname = f"epoch_{epoch:04d}_{nfe_key}_sample{idx}.png"
            fig.savefig(vol_dir / fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

            del z_denorm, decoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Decoded epoch %d samples to %s", epoch, vol_dir)


def _plot_latent_stats_evolution(
    archives: list[dict],
    output_dir: Path,
) -> None:
    """Plot per-channel latent stats evolution across epochs.

    Args:
        archives: List of archive dicts (sorted by epoch).
        output_dir: Output directory for PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    epochs = [a["epoch"] for a in archives]

    # Extract stats for NFE-1 (primary interest)
    nfe_key = "nfe_1"
    means_per_epoch = []
    stds_per_epoch = []

    for archive in archives:
        s = archive.get("stats", {}).get(nfe_key, {})
        means_per_epoch.append(s.get("mean", [0] * 4))
        stds_per_epoch.append(s.get("std", [1] * 4))

    means = np.array(means_per_epoch)  # (E, C)
    stds = np.array(stds_per_epoch)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ch in range(means.shape[1]):
        axes[0].plot(epochs, means[:, ch], label=f"Ch {ch}", marker="o", markersize=3)
        axes[1].plot(epochs, stds[:, ch], label=f"Ch {ch}", marker="o", markersize=3)

    axes[0].set_title("Per-Channel Mean (1-NFE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Per-Channel Std (1-NFE)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Std")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "latent_stats_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved latent stats evolution plot")


def _plot_nfe_consistency(
    archives: list[dict],
    output_dir: Path,
) -> None:
    """Plot NFE consistency (1-NFE vs multi-step MSE) across epochs.

    Args:
        archives: List of archive dicts (sorted by epoch).
        output_dir: Output directory for PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = []
    consistency_data: dict[str, list[float]] = {}

    for archive in archives:
        nfe_con = archive.get("nfe_consistency", {})
        if not nfe_con:
            continue
        epochs.append(archive["epoch"])
        for key, val in nfe_con.items():
            if key not in consistency_data:
                consistency_data[key] = []
            consistency_data[key].append(val)

    if not epochs:
        logger.info("No NFE consistency data to plot")
        return

    mse_keys = sorted(k for k in consistency_data if k.startswith("mse_"))
    cosine_keys = sorted(k for k in consistency_data if k.startswith("cosine_"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for key in mse_keys:
        label = key.replace("mse_", "MSE ").replace("vs", " vs ")
        axes[0].plot(epochs, consistency_data[key], label=label, marker="o", markersize=3)
    axes[0].set_title("NFE Consistency: MSE (lower = 1-NFE converging)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    for key in cosine_keys:
        label = key.replace("cosine_", "Cos ").replace("vs", " vs ")
        axes[1].plot(epochs, consistency_data[key], label=label, marker="o", markersize=3)
    axes[1].set_title("NFE Consistency: Cosine Similarity")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "nfe_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved NFE consistency plot")


def main() -> None:
    """Main entry point for decode_samples CLI."""
    args = parse_args()

    samples_dir = Path(args.samples_dir)
    if not samples_dir.exists():
        logger.error("Samples directory not found: %s", samples_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else samples_dir / "decoded"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load archives
    archives = _load_archives(samples_dir)
    if not archives:
        logger.error("No archives found; exiting")
        sys.exit(1)
    logger.info(
        "Loaded %d archives (epochs %d-%d)",
        len(archives),
        archives[0]["epoch"],
        archives[-1]["epoch"],
    )

    # Generate latent stats and NFE consistency plots (no VAE needed)
    _plot_latent_stats_evolution(archives, output_dir)
    _plot_nfe_consistency(archives, output_dir)

    # Determine NFE keys and sample indices
    sample_nfe_keys: list[str] = []
    if args.nfe:
        sample_nfe_keys = [f"nfe_{n}" for n in args.nfe]
    else:
        # Auto-detect from first archive
        sample_nfe_keys = sorted(k for k in archives[0] if k.startswith("nfe_"))

    sample_indices = args.sample_indices if args.sample_indices else [0, 1, 2]

    # VAE decoding
    if not args.skip_decode:
        config_path = Path(args.config)
        configs_dir = Path(args.configs_dir) if args.configs_dir else None
        config = _load_config(config_path, configs_dir)

        vae_weights = str(config.paths.get("maisi_vae_weights", ""))
        if not vae_weights or not Path(vae_weights).exists():
            logger.error("VAE weights not found: %s", vae_weights)
            logger.info("Use --skip-decode to generate plots without VAE decoding")
            sys.exit(1)

        # Load VAE once
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
        vae = MAISIVAEWrapper(vae_config, device=device)
        logger.info("Loaded MAISI VAE on %s", device)

        for archive in archives:
            _decode_and_save(archive, vae, sample_nfe_keys, sample_indices, output_dir, device)

        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logger.info("Skipping VAE decode (--skip-decode)")

    # Save summary JSON
    summary = {
        "n_archives": len(archives),
        "epochs": [a["epoch"] for a in archives],
        "nfe_keys": sample_nfe_keys,
        "sample_indices": sample_indices,
        "skip_decode": args.skip_decode,
        "stats_per_epoch": {},
    }
    for archive in archives:
        epoch = archive["epoch"]
        summary["stats_per_epoch"][str(epoch)] = {
            "stats": archive.get("stats", {}),
            "nfe_consistency": archive.get("nfe_consistency", {}),
        }

    summary_path = output_dir / "decode_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary written to %s", summary_path)
    logger.info("Done. Output at %s", output_dir)


if __name__ == "__main__":
    main()
