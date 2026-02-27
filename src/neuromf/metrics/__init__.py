"""Evaluation: FID, 3D-FID, SSIM, PSNR, SynthSeg, spectral, MMD, coverage/density.

Implements slice-wise FID, volumetric 3D-FID via Med3D features, structural
similarity metrics, SynthSeg-based morphological evaluation, high-frequency
energy analysis for VAE smoothing quantification, MMD with RBF kernel, and
k-NN based coverage/density metrics.
"""

from neuromf.metrics.coverage_density import compute_coverage, compute_density
from neuromf.metrics.fid import (
    compute_fid_2d5,
    extract_2d5_features,
    load_radimagenet_resnet50,
)
from neuromf.metrics.fid_3d import (
    compute_fid_3d,
    extract_3d_features,
    load_fid3d_feature_net,
    load_med3d_resnet50,
)
from neuromf.metrics.mmd import compute_mmd
from neuromf.metrics.ms_ssim_3d import compute_ms_ssim_3d
from neuromf.metrics.pairing import compute_nn_pairs
from neuromf.metrics.spectral import compute_hf_energy_ratio
from neuromf.metrics.swd import compute_swd
from neuromf.metrics.synthseg_metrics import (
    SynthSegConfig,
    check_synthseg_available,
    run_synthseg_evaluation,
)

__all__ = [
    "compute_mmd",
    "compute_coverage",
    "compute_density",
    "compute_swd",
    "compute_fid_2d5",
    "extract_2d5_features",
    "load_radimagenet_resnet50",
    "compute_fid_3d",
    "extract_3d_features",
    "load_fid3d_feature_net",
    "load_med3d_resnet50",
    "compute_ms_ssim_3d",
    "compute_nn_pairs",
    "compute_hf_energy_ratio",
    "SynthSegConfig",
    "check_synthseg_available",
    "run_synthseg_evaluation",
]
