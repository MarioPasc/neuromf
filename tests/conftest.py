"""Shared pytest fixtures for NeuroMF test suite."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture(scope="session")
def base_config() -> OmegaConf:
    """Load the base configuration from configs/base.yaml."""
    config_path = Path(__file__).parent.parent / "configs" / "base.yaml"
    return OmegaConf.load(config_path)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def results_root(base_config: OmegaConf) -> Path:
    """Return the results root directory."""
    return Path(base_config.paths.results_root)
