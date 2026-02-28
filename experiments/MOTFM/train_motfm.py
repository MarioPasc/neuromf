#!/usr/bin/env python
"""Memory-efficient MOTFM training wrapper.

Drop-in replacement for ``src/external/MOTFM/trainer.py`` that uses an
HDF5-backed lazy-loading dataset instead of MOTFM's pickle-based
``load_and_prepare_data()`` (which loads the entire dataset into RAM and
causes OOM on large 3D datasets).

The model, loss function, optimiser, and Lightning callbacks are
**identical** to MOTFM's ``trainer.py``.  Only the data path changes:
we read volumes on-demand from an HDF5 file produced by
``experiments/MOTFM/prepare_data.py``.

Usage::

    python experiments/MOTFM/train_motfm.py --config_path <yaml>

The config YAML must have an additional key under ``data_args``::

    data_args:
      h5_path: "/path/to/fomo60k_3d.h5"   # <-- HDF5 file
      pickle_path: "..."                    # still needed for inference, ignored here
"""

from __future__ import annotations

import argparse
import logging
import os
import resource
import sys
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

SCRIPT_VERSION = "1.0-h5lazy"


# ── Memory diagnostics ──────────────────────────────────────────────────

def _mem_rss_gb() -> float:
    """Return current RSS in GB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def _mem_available_gb() -> float:
    """Return available system memory in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return -1.0


def _log_memory(label: str) -> None:
    """Log current RSS and available memory."""
    logger.info(
        "[MEM] %-30s RSS=%.1f GB  Available=%.1f GB",
        label,
        _mem_rss_gb(),
        _mem_available_gb(),
    )


# ── Ensure MOTFM is importable ──────────────────────────────────────────

def _ensure_motfm_on_path() -> None:
    """Add MOTFM vendored code to sys.path if needed."""
    project_root = Path(__file__).resolve().parent.parent.parent
    motfm_path = str(project_root / "src" / "external" / "MOTFM")
    if motfm_path not in sys.path:
        sys.path.insert(0, motfm_path)

    # Also add experiments/MOTFM for h5_dataset import
    exp_path = str(project_root / "experiments" / "MOTFM")
    if exp_path not in sys.path:
        sys.path.insert(0, exp_path)


# ── HDF5 Data Module ────────────────────────────────────────────────────

class H5DataModule(pl.LightningDataModule):
    """Lightning DataModule backed by HDF5 lazy loading.

    Replaces MOTFM's ``FlowMatchingDataModule`` (pickle-based, loads all
    data into RAM).  Memory footprint: ~1 volume per worker instead of
    the entire dataset.

    Args:
        config: Full training config dict.
        h5_path: Path to the HDF5 file with ``/train`` and ``/valid`` datasets.
    """

    def __init__(self, config: dict, h5_path: str) -> None:
        super().__init__()
        self.config = config
        self.h5_path = h5_path
        self.train_dataset: Optional[object] = None
        self.val_dataset: Optional[object] = None

    def setup(self, stage: Optional[str] = None) -> None:
        from h5_dataset import H5VolumeDataset

        data_config = self.config["data_args"]
        split_train = data_config.get("split_train", "train")
        split_val = data_config.get("split_val", "valid")

        logger.info("Setting up H5DataModule (lazy loading) from: %s", self.h5_path)
        _log_memory("h5_datamodule_setup_start")

        if stage in (None, "fit"):
            self.train_dataset = H5VolumeDataset(
                h5_path=self.h5_path,
                split=split_train,
            )
            self.val_dataset = H5VolumeDataset(
                h5_path=self.h5_path,
                split=split_val,
            )
            logger.info(
                "H5DataModule ready: train=%d, val=%d",
                len(self.train_dataset),
                len(self.val_dataset),
            )
        elif stage == "validate":
            self.val_dataset = H5VolumeDataset(
                h5_path=self.h5_path,
                split=split_val,
            )
            logger.info("H5DataModule ready: val=%d", len(self.val_dataset))

        _log_memory("h5_datamodule_setup_done")

    def train_dataloader(self) -> DataLoader:
        tr_args = self.config["train_args"]
        num_workers = int(tr_args.get("num_workers", 4))
        return DataLoader(
            self.train_dataset,
            batch_size=tr_args["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=bool(tr_args.get("drop_last", False)),
        )

    def val_dataloader(self) -> DataLoader:
        tr_args = self.config["train_args"]
        num_workers = int(tr_args.get("num_workers", 4))
        return DataLoader(
            self.val_dataset,
            batch_size=tr_args["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )


# ── Strategy helper (copied from MOTFM trainer.py) ──────────────────────

def _resolve_strategy(
    accelerator: Union[str, int, list, tuple],
    devices: Union[str, int, list, tuple],
) -> object:
    """Use DDP only for true multi-GPU execution."""
    if str(accelerator).lower() not in {"gpu", "cuda"}:
        return "auto"
    if isinstance(devices, int):
        return DDPStrategy(find_unused_parameters=True) if devices > 1 else "auto"
    if isinstance(devices, (list, tuple)):
        return DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else "auto"
    if isinstance(devices, str):
        d = devices.strip().lower()
        if d in {"auto", "1"}:
            return "auto"
        if "," in d:
            ids = [x for x in d.split(",") if x.strip() != ""]
            return DDPStrategy(find_unused_parameters=True) if len(ids) > 1 else "auto"
    return "auto"


# ── Resume checkpoint helper (copied from MOTFM trainer.py) ─────────────

def _resolve_resume_checkpoint(
    explicit_ckpt_path: Optional[str], root_ckpt_dir: str, run_name: str
) -> Optional[str]:
    """Return the checkpoint path to resume from, if any."""
    if explicit_ckpt_path:
        return explicit_ckpt_path
    ckpt_dir = os.path.join(root_ckpt_dir, run_name)
    if not os.path.isdir(ckpt_dir):
        return None
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt
    candidates = [
        os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt") and os.path.isfile(os.path.join(ckpt_dir, f))
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Train MOTFM with HDF5-backed lazy data loading."""
    parser = argparse.ArgumentParser(
        description="Train MOTFM with memory-efficient HDF5 data loading.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the MOTFM YAML config file.",
    )
    args = parser.parse_args()

    logger.info("train_motfm.py version: %s", SCRIPT_VERSION)
    logger.info("Python: %s", sys.version)
    logger.info("PID: %d", os.getpid())
    _log_memory("startup")

    # Ensure MOTFM is importable
    _ensure_motfm_on_path()

    from utils.general_utils import load_config
    from trainer import FlowMatchingLightningModule

    config = load_config(args.config_path)
    run_name = os.path.splitext(os.path.basename(args.config_path))[0]
    tr = config["train_args"]
    root_ckpt_dir = tr["checkpoint_dir"]

    logger.info("Config: %s", args.config_path)
    logger.info("Run name: %s", run_name)
    logger.info("Checkpoint dir: %s", root_ckpt_dir)

    # Resolve HDF5 path from config
    data_config = config["data_args"]
    h5_path = data_config.get("h5_path")
    if not h5_path:
        # Fallback: derive from pickle_path by changing extension
        pickle_path = data_config["pickle_path"]
        h5_path = str(Path(pickle_path).with_suffix(".h5"))
        logger.info("No h5_path in config; derived from pickle_path: %s", h5_path)

    if not Path(h5_path).exists():
        logger.error("HDF5 file not found: %s", h5_path)
        logger.error("Run prepare_data.py with --keep-h5 first.")
        sys.exit(1)

    logger.info("HDF5 data: %s (%.2f GB)", h5_path, Path(h5_path).stat().st_size / 1e9)

    # Seed
    seed = tr.get("seed")
    if seed is not None:
        seed = int(seed)
        pl.seed_everything(seed, workers=True)
        logger.info("Seed: %d", seed)

    # Model (identical to MOTFM's trainer.py)
    model = FlowMatchingLightningModule(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable params (%.1f MB)", n_params, n_params * 4 / 1e6)
    _log_memory("after_model_build")

    # Data module (HDF5-backed, NOT pickle-based)
    datamodule = H5DataModule(config, h5_path=h5_path)
    _log_memory("after_datamodule_init")

    # Lightning callbacks (identical to MOTFM's trainer.py)
    tb_logger = TensorBoardLogger(save_dir=root_ckpt_dir, name=run_name)
    ckpt_every = max(1, int(tr.get("val_freq", 5)))
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

    # Precision
    _bf16_supported = (
        torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    _fp16_supported = torch.cuda.is_available()
    default_precision = (
        "bf16-mixed" if _bf16_supported else ("16-mixed" if _fp16_supported else "32-true")
    )
    precision = tr.get("precision", default_precision)

    # Resume
    resume_ckpt = _resolve_resume_checkpoint(tr.get("ckpt_path"), root_ckpt_dir, run_name)
    if resume_ckpt:
        logger.info("Resuming from: %s", resume_ckpt)
    else:
        logger.info("No checkpoint found. Training from scratch.")

    # Strategy
    accelerator = tr.get("accelerator", "auto")
    devices = tr.get("devices", "auto")
    strategy = _resolve_strategy(accelerator, devices)
    deterministic = bool(tr.get("deterministic", False))

    logger.info(
        "Trainer: accelerator=%s, devices=%s, strategy=%s, precision=%s",
        accelerator,
        devices,
        strategy,
        precision,
    )
    _log_memory("before_trainer_fit")

    trainer = pl.Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=tr["num_epochs"],
        precision=precision,
        accumulate_grad_batches=tr.get("gradient_accumulation_steps", 8),
        gradient_clip_val=tr.get("grad_clip_norm", 0.0) or None,
        check_val_every_n_epoch=ckpt_every,
        enable_progress_bar=True,
        logger=tb_logger,
        callbacks=[ckpt_cb, lr_cb],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        deterministic=deterministic,
        log_every_n_steps=tr.get("log_every_n_steps", 50),
        num_sanity_val_steps=tr.get("num_sanity_val_steps", 0),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    _log_memory("after_training")
    logger.info("Training complete. version=%s", SCRIPT_VERSION)


if __name__ == "__main__":
    main()
