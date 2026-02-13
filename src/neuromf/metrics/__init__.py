"""Evaluation: FID, 3D-FID, SSIM, PSNR, SynthSeg, spectral, MMD, coverage/density.

Implements slice-wise FID, volumetric 3D-FID via Med3D features, structural
similarity metrics, SynthSeg-based morphological evaluation, high-frequency
energy analysis for VAE smoothing quantification, MMD with RBF kernel, and
k-NN based coverage/density metrics.
"""

from neuromf.metrics.coverage_density import compute_coverage, compute_density
from neuromf.metrics.mmd import compute_mmd

__all__ = ["compute_mmd", "compute_coverage", "compute_density"]
