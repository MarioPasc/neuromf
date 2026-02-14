"""Adapter modules wrapping external code from vendored repositories.

Provides clean interfaces to the frozen MAISI VAE, MAISI 3D UNet, and MeanFlow
loss computation without modifying the original external code.
"""

from neuromf.wrappers.jvp_strategies import ExactJVP, FiniteDifferenceJVP
from neuromf.wrappers.maisi_unet import MAISIUNetConfig, MAISIUNetWrapper
from neuromf.wrappers.meanflow_loss import MeanFlowPipeline, MeanFlowPipelineConfig

__all__ = [
    "ExactJVP",
    "FiniteDifferenceJVP",
    "MAISIUNetConfig",
    "MAISIUNetWrapper",
    "MeanFlowPipeline",
    "MeanFlowPipelineConfig",
]
