#!/usr/bin/env python
"""DDP smoke test — verifies multi-GPU training with PyTorch Lightning.

Catches the SLURMEnvironment bug where ntasks=1 forces world_size=1,
ignoring the devices parameter. Uses the same LightningEnvironment fix
applied to train.py and train_motfm.py.

Usage (on loginexa or any multi-GPU node):
    python scripts/test_ddp.py              # auto-detect GPUs
    python scripts/test_ddp.py --gpus 3     # use exactly 3 GPUs
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


class TinyModel(pl.LightningModule):
    """Minimal model: linear layer on random 4-channel 8^3 tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(4, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(8, 4, 3, padding=1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        (x,) = batch
        loss = self.loss_fn(self(x), x)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        # Log which GPU this rank is using
        if batch_idx == 0:
            rank = self.global_rank
            device = self.device
            logger.info(f"[Rank {rank}] training_step on device={device}")
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP smoke test")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs (0 = auto-detect)")
    parser.add_argument("--steps", type=int, default=10, help="Training steps per epoch")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    args = parser.parse_args()

    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_gpus = args.gpus if args.gpus > 0 else n_visible

    logger.info("=" * 60)
    logger.info("DDP SMOKE TEST")
    logger.info("=" * 60)
    logger.info(f"PyTorch Lightning version: {pl.__version__}")
    logger.info(f"PyTorch version:           {torch.__version__}")
    logger.info(f"CUDA available:            {torch.cuda.is_available()}")
    logger.info(f"Visible GPUs:              {n_visible}")
    logger.info(f"Requested GPUs:            {n_gpus}")
    logger.info(f"SLURM_JOB_ID:              {os.environ.get('SLURM_JOB_ID', 'not set')}")
    logger.info(f"SLURM_NTASKS:              {os.environ.get('SLURM_NTASKS', 'not set')}")
    logger.info(f"CUDA_VISIBLE_DEVICES:       {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    if n_gpus < 2:
        logger.error(f"Need at least 2 GPUs for DDP test. Found {n_visible} visible GPUs.")
        raise SystemExit(1)

    # Fake dataset: 64 random 4-channel 8^3 volumes
    n_samples = max(64, args.steps * n_gpus)
    dataset = TensorDataset(torch.randn(n_samples, 4, 8, 8, 8))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = TinyModel()

    # Apply the same fix we use in train.py / train_motfm.py
    plugins: list = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(LightningEnvironment())
        logger.info(
            "SLURM detected (ntasks=%s): using LightningEnvironment override",
            os.environ.get("SLURM_NTASKS", "?"),
        )
    else:
        logger.info("Not inside SLURM — no environment override needed")

    strategy = "ddp" if n_gpus > 1 else "auto"
    logger.info(f"Strategy: {strategy}, devices: {n_gpus}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=n_gpus,
        strategy=strategy,
        max_epochs=args.epochs,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        plugins=plugins if plugins else None,
    )

    logger.info("Starting training...")
    t0 = time.time()
    trainer.fit(model, loader)
    elapsed = time.time() - t0

    # Verification
    world_size = trainer.world_size
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"World size (actual):  {world_size}")
    logger.info(f"Expected:             {n_gpus}")
    logger.info(f"Training time:        {elapsed:.1f}s")

    if world_size == n_gpus:
        logger.info("PASS — All %d GPUs were used for DDP training.", n_gpus)
    else:
        logger.error(
            "FAIL — Expected world_size=%d but got %d. DDP is NOT working correctly!",
            n_gpus,
            world_size,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
