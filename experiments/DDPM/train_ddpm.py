#!/usr/bin/env python
"""Train DDPM baseline with HDF5-backed lazy data loading.

Uses the same UNet architecture as MOTFM (132.5M params), the same FOMO-60K
dataset, and the same training budget (step-matched ~150K optimizer steps).
Only the loss function (noise prediction MSE) and sampler (DDIM) differ.

Usage:
    export PYTHONPATH="${REPO}/src/external/MOTFM:${REPO}/experiments/DDPM:${REPO}/experiments/MOTFM:${PYTHONPATH}"
    python experiments/DDPM/train_ddpm.py --config_path configs/picasso/ddpm/fomo60k_unconditional_3d.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_VERSION = "1.0.0"


def _ensure_paths_on_sys() -> None:
    """Add MOTFM external code and experiment dirs to sys.path."""
    repo_root = Path(__file__).resolve().parents[2]
    for sub in ["src/external/MOTFM", "experiments/DDPM", "experiments/MOTFM"]:
        p = str(repo_root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)


def main() -> None:
    """Train DDPM with HDF5-backed lazy data loading."""
    parser = argparse.ArgumentParser(description="Train DDPM baseline.")
    parser.add_argument("--config_path", type=str, required=True, help="DDPM YAML config.")
    args = parser.parse_args()

    logger.info("train_ddpm.py version: %s", SCRIPT_VERSION)
    _ensure_paths_on_sys()

    from ddpm_module import DDPMLightningModule

    # Reuse MOTFM infrastructure
    from train_motfm import (
        H5DataModule,
        _enable_gradient_checkpointing,
        _resolve_resume_checkpoint,
        _resolve_strategy,
    )
    from utils.general_utils import load_config

    config = load_config(args.config_path)
    run_name = os.path.splitext(os.path.basename(args.config_path))[0]
    tr = config["train_args"]
    root_ckpt_dir = tr["checkpoint_dir"]

    logger.info("Config: %s", args.config_path)
    logger.info("Run name: %s", run_name)

    # Resolve HDF5 path
    data_config = config["data_args"]
    h5_path = data_config.get("h5_path")
    if not h5_path:
        h5_path = str(Path(data_config["pickle_path"]).with_suffix(".h5"))
    if not Path(h5_path).exists():
        logger.error("HDF5 file not found: %s", h5_path)
        sys.exit(1)
    logger.info("HDF5 data: %s (%.2f GB)", h5_path, Path(h5_path).stat().st_size / 1e9)

    # Seed
    seed = tr.get("seed")
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)

    # Model
    model = DDPMLightningModule(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("DDPM model: %d trainable params (%.1f MB)", n_params, n_params * 4 / 1e6)

    # Transfer learning: load pretrained weights only (no optimizer/scheduler)
    pretrained_path = tr.get("pretrained_checkpoint_path")
    if pretrained_path and Path(pretrained_path).is_file():
        logger.info("Loading pretrained weights from: %s", pretrained_path)
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Pretrained weights loaded: %d matched, %d missing, %d unexpected",
            len(state_dict) - len(unexpected),
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.warning("Missing keys: %s", missing[:10])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:10])
        del ckpt, state_dict
    elif pretrained_path:
        logger.warning("Pretrained checkpoint not found: %s", pretrained_path)

    # Gradient checkpointing
    n_wrapped = _enable_gradient_checkpointing(model)
    logger.info("Gradient checkpointing: %d blocks", n_wrapped)

    torch.set_float32_matmul_precision("medium")

    # Data
    datamodule = H5DataModule(config, h5_path=h5_path)

    # Callbacks
    ckpt_every = max(1, int(tr.get("val_freq", 10)))
    tb_logger = TensorBoardLogger(save_dir=root_ckpt_dir, name=run_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(root_ckpt_dir, run_name),
        filename="epoch{epoch:03d}-valloss{val/loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        every_n_epochs=ckpt_every,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    callbacks_list: list = [ckpt_cb, lr_cb]

    # Early stopping
    early_stop_patience = int(tr.get("early_stop_patience", 0))
    if early_stop_patience > 0:
        from pytorch_lightning.callbacks import EarlyStopping

        callbacks_list.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=early_stop_patience,
                verbose=True,
            )
        )
        logger.info("Early stopping: patience=%d val epochs", early_stop_patience)

    # Training monitor
    from ddpm_callbacks import DDPMTrainingMonitor

    monitor_cb = DDPMTrainingMonitor(
        output_dir=os.path.join(root_ckpt_dir, run_name, "monitor"),
        nfe_list=[1, 10, 50],
        n_samples=4,
        seed=42,
        val_freq=ckpt_every,
        config=config,
    )

    # Precision
    _bf16 = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    precision = tr.get("precision", "bf16-mixed" if _bf16 else "32-true")

    # Resume (full Lightning resume, not transfer learning)
    resume_ckpt = _resolve_resume_checkpoint(tr.get("ckpt_path"), root_ckpt_dir, run_name)
    if resume_ckpt:
        logger.info("Resuming from: %s", resume_ckpt)

    # Strategy + SLURM fix
    accelerator = tr.get("accelerator", "auto")
    devices = tr.get("devices", "auto")
    strategy = _resolve_strategy(accelerator, devices)

    from pytorch_lightning.plugins.environments import LightningEnvironment

    plugins: list = []
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_visible > 1 and os.environ.get("SLURM_JOB_ID"):
        plugins.append(LightningEnvironment())
        logger.info("SLURM override: LightningEnvironment for %d GPUs", n_visible)

    trainer = pl.Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=tr["num_epochs"],
        precision=precision,
        accumulate_grad_batches=tr.get("gradient_accumulation_steps", 1),
        gradient_clip_val=tr.get("grad_clip_norm", 0.0) or None,
        check_val_every_n_epoch=ckpt_every,
        enable_progress_bar=True,
        logger=tb_logger,
        callbacks=callbacks_list + [monitor_cb],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=tr.get("log_every_n_steps", 50),
        num_sanity_val_steps=tr.get("num_sanity_val_steps", 0),
        plugins=plugins if plugins else None,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    logger.info("DDPM training complete.")


if __name__ == "__main__":
    main()
