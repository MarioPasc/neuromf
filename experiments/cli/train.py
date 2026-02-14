"""CLI script for training the Latent MeanFlow model.

Loads config from base.yaml + train_meanflow.yaml, builds datasets and
Lightning Trainer, and launches training. Supports resume from checkpoint
and dry-run mode.

Usage:
    ~/.conda/envs/neuromf/bin/python experiments/cli/train.py \
        --config configs/train_meanflow.yaml

    # With Picasso configs:
    python experiments/cli/train.py \
        --config configs/picasso/train_meanflow.yaml \
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
import sys
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
        required=True,
        help="Path to the training config YAML (e.g. configs/train_meanflow.yaml).",
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


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Enable TF32 on Ampere+ GPUs (A100/H100) for faster matmuls
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    configs_dir = Path(args.configs_dir) if args.configs_dir else config_path.parent

    base_path = configs_dir / "base.yaml"
    if not base_path.exists():
        logger.error("base.yaml not found at %s", base_path)
        sys.exit(1)

    # Three-layer config: base.yaml + main train config + overlay (--config)
    # When --config IS the main config (local), the middle layer is redundant.
    # When --config is a Picasso overlay, the middle layer provides defaults.
    project_root = Path(__file__).resolve().parent.parent.parent
    main_train_path = project_root / "configs" / "train_meanflow.yaml"

    layers = [OmegaConf.load(base_path)]
    if main_train_path.exists() and main_train_path.resolve() != config_path.resolve():
        layers.append(OmegaConf.load(main_train_path))
    layers.append(OmegaConf.load(config_path))

    config = OmegaConf.merge(*layers)
    OmegaConf.resolve(config)

    loaded = " + ".join(str(p) for p in [base_path, main_train_path, config_path] if p.exists())
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
    strategy = str(trainer_cfg.get("strategy", "auto"))
    accelerator = str(trainer_cfg.get("accelerator", "auto"))
    effective_batch = int(config.training.batch_size) * devices

    if args.dry_run:
        logger.info("=== DRY RUN ===")
        logger.info("UNet channels: %s", list(config.unet.channels))
        logger.info("Prediction type: %s", config.unet.prediction_type)
        logger.info("Batch size: %d (per-GPU)", config.training.batch_size)
        logger.info("Devices: %d, Strategy: %s", devices, strategy)
        logger.info("Effective batch size: %d", effective_batch)
        logger.info("Max epochs: %d", config.training.max_epochs)
        logger.info("LR: %.1e", config.training.lr)
        logger.info("Warmup steps: %d", config.training.warmup_steps)
        logger.info(
            "MeanFlow p=%.1f, adaptive=%s",
            config.meanflow.p,
            config.meanflow.adaptive,
        )
        logger.info(
            "Time sampling: mu=%.2f, sigma=%.2f",
            config.time_sampling.mu,
            config.time_sampling.sigma,
        )
        logger.info("EMA decay: %.4f", config.ema.decay)
        logger.info("Config OK — dry run complete.")
        return

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    latent_dir = Path(config.paths.latents_dir)
    if not latent_dir.exists():
        logger.error("Latent directory not found: %s", latent_dir)
        sys.exit(1)

    train_ds = LatentDataset(
        latent_dir, normalise=True, split="train", split_ratio=0.9, split_seed=42
    )
    val_ds = LatentDataset(latent_dir, normalise=True, split="val", split_ratio=0.9, split_seed=42)
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
        monitor="val/loss_total",
        mode="min",
        filename="epoch_{epoch:03d}_vloss_{val/loss_total:.4f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_cb, lr_monitor]

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
        "Effective batch size: %d (per-GPU=%d × %d devices)",
        effective_batch,
        batch_size,
        devices,
    )

    trainer = pl.Trainer(
        devices=devices,
        strategy=strategy,
        accelerator=accelerator,
        max_epochs=int(config.training.max_epochs),
        precision=precision,
        gradient_clip_val=float(config.training.gradient_clip_norm),
        gradient_clip_algorithm="norm",
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
