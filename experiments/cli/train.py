"""CLI script for training the Latent MeanFlow model.

Loads config from base.yaml + train_meanflow.yaml + optional overlay configs,
builds datasets and Lightning Trainer, and launches training. Supports resume
from checkpoint, dry-run mode, and multi-config layering for ablations.

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/train.py \
        --config configs/train_meanflow.yaml

    # With Picasso configs:
    python experiments/cli/train.py \
        --config configs/picasso/train_meanflow.yaml \
        --configs-dir configs/picasso

    # Multi-config for ablations (layers are merged left-to-right):
    python experiments/cli/train.py \
        --config configs/picasso/train_meanflow.yaml \
                 experiments/ablations/phase_4/configs/v3_aug.yaml \
        --configs-dir configs/picasso

    # Resume from checkpoint:
    python experiments/cli/train.py \
        --config configs/train_meanflow.yaml \
        --resume /path/to/checkpoint.ckpt

    # Dry run (config validation only):
    python experiments/cli/train.py \
        --config configs/train_meanflow.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from neuromf.data.latent_dataset import LatentDataset, latent_collate_fn
from neuromf.models.latent_meanflow import LatentMeanFlow

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
        description="Train the Latent MeanFlow model on pre-computed latents."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more config YAML paths, merged left-to-right on top of "
            "base.yaml + train_meanflow.yaml. For ablations, pass the Picasso "
            "overlay first, then the ablation diff."
        ),
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Directory containing base.yaml. Defaults to parent of --config.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a Lightning checkpoint to resume from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load configs and model, print summary, and exit.",
    )
    return parser.parse_args()


def _save_config_snapshot(
    merged_config: OmegaConf,
    config_paths: list[Path],
    base_path: Path,
    main_train_path: Path,
) -> None:
    """Save a snapshot of all config files to the results folder.

    Creates a timestamped directory under ``logs_dir/config_snapshots/``
    containing the merged resolved config and copies of each input YAML.

    Args:
        merged_config: Fully merged and resolved OmegaConf config.
        config_paths: Ordered list of overlay config paths (--config arguments).
        base_path: Path to base.yaml.
        main_train_path: Path to the main train_meanflow.yaml.
    """
    logs_dir = Path(merged_config.paths.logs_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = logs_dir / "config_snapshots" / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Save the fully merged + resolved config
    resolved_path = snapshot_dir / "resolved_config.yaml"
    resolved_path.write_text(OmegaConf.to_yaml(merged_config, resolve=True))

    # Copy each input config layer
    all_sources = [base_path, main_train_path] + config_paths
    seen_names: set[str] = set()
    for idx, src in enumerate(all_sources):
        if not src.exists():
            continue
        dst_name = src.name
        # Prefix with parent dir name for Picasso overlays / ablation configs
        if src.parent.name not in ("configs", ""):
            dst_name = f"{src.parent.name}_{dst_name}"
        # Deduplicate: append index if name already seen
        if dst_name in seen_names:
            stem, suffix = dst_name.rsplit(".", 1)
            dst_name = f"{stem}_{idx:02d}.{suffix}"
        seen_names.add(dst_name)
        shutil.copy2(src, snapshot_dir / dst_name)

    logger.info("Config snapshot saved to %s", snapshot_dir)


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Enable TF32 on Ampere+ GPUs (A100/H100) for faster matmuls
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # Config loading — supports N config layers merged left-to-right
    # ------------------------------------------------------------------
    config_paths = [Path(p) for p in args.config]
    configs_dir = Path(args.configs_dir) if args.configs_dir else config_paths[0].parent

    base_path = configs_dir / "base.yaml"
    if not base_path.exists():
        logger.error("base.yaml not found at %s", base_path)
        sys.exit(1)

    # Merge chain: base.yaml → main train config → config_1 → config_2 → ...
    # The main train_meanflow.yaml provides defaults; additional configs
    # (Picasso overlay, ablation diff) override only the keys they specify.
    project_root = Path(__file__).resolve().parent.parent.parent
    main_train_path = project_root / "configs" / "train_meanflow.yaml"

    layers = [OmegaConf.load(base_path)]
    # Add main config if it isn't the first --config arg (avoids double-loading)
    if main_train_path.exists() and main_train_path.resolve() != config_paths[0].resolve():
        layers.append(OmegaConf.load(main_train_path))
    for cp in config_paths:
        layers.append(OmegaConf.load(cp))

    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)

    loaded = " + ".join(str(p) for p in [base_path, main_train_path, *config_paths] if p.exists())
    logger.info("Config loaded: %s", loaded)
    logger.info("Latents dir: %s", config.paths.latents_dir)
    logger.info("Checkpoints dir: %s", config.paths.checkpoints_dir)

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Trainer infrastructure (devices, strategy, accelerator)
    # ------------------------------------------------------------------
    trainer_cfg = config.get("trainer", {})
    devices = int(trainer_cfg.get("devices", 1))
    strategy_name = str(trainer_cfg.get("strategy", "auto"))
    accelerator = str(trainer_cfg.get("accelerator", "auto"))

    # For DDP, disable buffer broadcasting — we use GroupNorm (not
    # BatchNorm), so there are no running stats to sync.  Our only
    # registered buffers (latent_mean/std) are deterministic constants.
    if strategy_name == "ddp":
        from pytorch_lightning.strategies import DDPStrategy

        strategy: str | DDPStrategy = DDPStrategy(broadcast_buffers=False)
    else:
        strategy = strategy_name
    accum_steps = int(config.training.get("accumulate_grad_batches", 1))
    effective_batch = int(config.training.batch_size) * devices * accum_steps

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("UNet channels: %s", list(config.unet.channels))
        logger.info("Prediction type: %s", config.unet.prediction_type)
        logger.info("Batch size: %d (per-GPU)", config.training.batch_size)
        logger.info("Devices: %d, Strategy: %s", devices, strategy)
        logger.info("Effective batch size: %d", effective_batch)
        logger.info("Max epochs: %d", config.training.max_epochs)
        logger.info("LR: %.1e", config.training.lr)
        logger.info("Betas: %s", list(config.training.betas))
        logger.info("Warmup steps: %d", config.training.warmup_steps)
        logger.info("LR schedule: %s", config.training.get("lr_schedule", "cosine"))
        logger.info("Norm eps: %.2f", config.meanflow.norm_eps)
        logger.info(
            "MeanFlow p=%.1f, adaptive=%s, norm_p=%.2f",
            config.meanflow.p,
            config.meanflow.adaptive,
            config.meanflow.get("norm_p", 1.0),
        )
        logger.info(
            "Time sampling: mu=%.2f, sigma=%.2f, data_proportion=%.2f",
            config.time_sampling.mu,
            config.time_sampling.sigma,
            config.time_sampling.data_proportion,
        )
        logger.info("EMA decay: %.4f", config.ema.decay)
        logger.info("Weight decay: %.1e", config.training.weight_decay)
        logger.info(
            "Divergence guard: threshold=%.1f, grace_steps=%d",
            config.training.get("divergence_threshold", 0.0),
            config.training.get("divergence_grace_steps", 100),
        )
        logger.info(
            "Spatial mask ratio: %.2f",
            config.meanflow.get("spatial_mask_ratio", 0.0),
        )
        aug_enabled = config.training.get("augmentation", {}).get("enabled", False)
        logger.info("Augmentation: %s", "enabled" if aug_enabled else "disabled")
        logger.info("Config OK — dry run complete.")
        return

    # ------------------------------------------------------------------
    # Save config snapshot to results folder
    # ------------------------------------------------------------------
    _save_config_snapshot(config, config_paths, base_path, main_train_path)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    latent_dir = Path(config.paths.latents_dir)
    if not latent_dir.exists():
        logger.error("Latent directory not found: %s", latent_dir)
        sys.exit(1)

    # Build augmentation pipeline (Phase B, toggleable)
    aug_cfg = OmegaConf.to_container(config.training.get("augmentation", {}), resolve=True)
    aug_transform = None
    if aug_cfg and aug_cfg.get("enabled", False):
        from neuromf.data.latent_augmentation import build_latent_augmentation

        stats_path = Path(config.paths.latents_dir) / "latent_stats.json"
        aug_transform = build_latent_augmentation(aug_cfg, latent_stats_path=stats_path)
        logger.info("Latent augmentation enabled")

    split_ratio = float(config.training.get("split_ratio", 0.9))
    split_seed = int(config.training.get("split_seed", 42))
    train_ds = LatentDataset(
        latent_dir,
        normalise=True,
        split="train",
        split_ratio=split_ratio,
        split_seed=split_seed,
        transform=aug_transform,
    )
    val_ds = LatentDataset(
        latent_dir,
        normalise=True,
        split="val",
        split_ratio=split_ratio,
        split_seed=split_seed,
    )
    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    batch_size = int(config.training.batch_size)
    num_workers = int(config.training.get("num_workers", 0))
    prefetch = config.training.get("prefetch_factor", None)
    prefetch = int(prefetch) if prefetch is not None else None

    # prefetch_factor requires num_workers > 0
    dl_kwargs: dict = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": latent_collate_fn,
    }
    if num_workers > 0 and prefetch is not None:
        dl_kwargs["prefetch_factor"] = prefetch

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **dl_kwargs
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **dl_kwargs)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = LatentMeanFlow(config)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    ckpt_dir = Path(config.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        every_n_epochs=int(config.training.get("save_every_n_epochs", 50)),
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        filename="epoch_{epoch:03d}_vloss_{val/loss:.4f}",
        save_last=True,
    )
    best_raw_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="train/raw_loss",
        mode="min",
        save_top_k=1,
        filename="best_raw_{epoch:03d}_{train/raw_loss:.4f}",
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_cb, best_raw_cb, lr_monitor]

    diag_cfg = config.get("diagnostics", {})
    if diag_cfg.get("enabled", False):
        from neuromf.callbacks.diagnostics import TrainingDiagnosticsCallback
        from neuromf.callbacks.performance import PerformanceCallback

        diag_dir = Path(config.paths.get("diagnostics_dir", ""))
        if diag_dir:
            diag_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            TrainingDiagnosticsCallback(
                diag_every_n_epochs=int(diag_cfg.get("every_n_epochs", 25)),
                diagnostics_dir=str(diag_dir) if diag_dir else "",
                gradient_clip_val=float(config.training.gradient_clip_norm),
            )
        )
        callbacks.append(
            PerformanceCallback(
                log_every_n_steps=int(config.training.log_every_n_steps),
            )
        )
        logger.info("Diagnostics enabled (every %d epochs)", diag_cfg.get("every_n_epochs", 25))

    # Sample collector callback (replaces inline sample generation)
    sample_cfg = config.get("sample_collector", {})
    if sample_cfg.get("enabled", True):
        from neuromf.callbacks.sample_collector import SampleCollectorCallback

        samples_dir = str(config.paths.get("samples_dir", ""))
        if samples_dir:
            callbacks.append(
                SampleCollectorCallback(
                    samples_dir=samples_dir,
                    collect_every_n_epochs=int(
                        sample_cfg.get(
                            "collect_every_n_epochs",
                            config.get("sample_every_n_epochs", 25),
                        )
                    ),
                    n_samples=int(sample_cfg.get("n_samples", config.get("n_samples_per_log", 8))),
                    nfe_steps=list(sample_cfg.get("nfe_steps", [1, 2, 5, 10])),
                    seed=int(sample_cfg.get("seed", 42)),
                    prediction_type=str(config.unet.prediction_type),
                )
            )
            logger.info(
                "SampleCollector enabled (every %d epochs, NFE=%s)",
                sample_cfg.get(
                    "collect_every_n_epochs",
                    config.get("sample_every_n_epochs", 25),
                ),
                list(sample_cfg.get("nfe_steps", [1, 2, 5, 10])),
            )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logs_dir = Path(config.paths.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tb_logger = TensorBoardLogger(save_dir=str(logs_dir), name="meanflow")

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------
    precision_map = {
        "bf16": "bf16-mixed",
        "bf16-mixed": "bf16-mixed",
        "fp16": "16-mixed",
        "fp16-mixed": "16-mixed",
        "fp32": "32-true",
        "32": "32-true",
    }
    precision = precision_map.get(str(config.training.mixed_precision), "bf16-mixed")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    logger.info(
        "Trainer: devices=%d, strategy=%s, accelerator=%s",
        devices,
        strategy,
        accelerator,
    )
    logger.info(
        "Effective batch size: %d (per-GPU=%d × %d devices × %d accum)",
        effective_batch,
        batch_size,
        devices,
        accum_steps,
    )

    trainer = pl.Trainer(
        devices=devices,
        strategy=strategy,
        accelerator=accelerator,
        max_epochs=int(config.training.max_epochs),
        precision=precision,
        gradient_clip_val=float(config.training.gradient_clip_norm),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=accum_steps,
        callbacks=callbacks,
        logger=tb_logger,
        check_val_every_n_epoch=int(config.training.val_every_n_epochs),
        log_every_n_steps=int(config.training.log_every_n_steps),
        enable_progress_bar=True,
        deterministic=False,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    resume_path = args.resume
    if resume_path and not Path(resume_path).exists():
        logger.error("Resume checkpoint not found: %s", resume_path)
        sys.exit(1)

    logger.info("Starting training...")
    trainer.fit(model, train_dl, val_dl, ckpt_path=resume_path)
    logger.info("Training complete. Best model: %s", checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
