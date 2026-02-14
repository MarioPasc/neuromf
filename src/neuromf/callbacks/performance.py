"""Lightweight callback for throughput and GPU memory monitoring."""

from __future__ import annotations

import logging
import time

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)


class PerformanceCallback(pl.Callback):
    """Tracks throughput (samples/sec) and GPU memory usage.

    Args:
        log_every_n_steps: How often to log step-level metrics.
    """

    def __init__(self, log_every_n_steps: int = 50) -> None:
        super().__init__()
        self._log_every = log_every_n_steps
        self._step_start: float = 0.0
        self._epoch_start: float = 0.0

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Record step start time."""
        self._step_start = time.monotonic()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """Log throughput and GPU memory."""
        if trainer.global_step % self._log_every != 0:
            return

        elapsed = time.monotonic() - self._step_start
        if elapsed <= 0:
            return

        batch_size = batch["z"].shape[0] if isinstance(batch, dict) and "z" in batch else 1
        world_size = trainer.world_size
        pl_module.log("perf/samples_per_sec", batch_size / elapsed)
        pl_module.log("perf/total_samples_per_sec", batch_size * world_size / elapsed)
        pl_module.log("perf/steps_per_sec", 1.0 / elapsed)

        if torch.cuda.is_available():
            device = pl_module.device
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                pl_module.log("perf/gpu_mem_allocated_gb", allocated)
                pl_module.log("perf/gpu_mem_reserved_gb", reserved)

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Record epoch start time."""
        self._epoch_start = time.monotonic()

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log epoch wall time."""
        elapsed = time.monotonic() - self._epoch_start
        pl_module.log("perf/epoch_time_sec", elapsed)
