"""Inference: 1-NFE one-step and multi-step Euler sampling.

Provides one-step MeanFlow sampling (z_0 = eps - u_theta(eps, 0, 1)) and
multi-step Euler integration for NFE ablation studies.
"""

from neuromf.sampling.multi_step import sample_euler
from neuromf.sampling.one_step import sample_one_step

__all__ = ["sample_one_step", "sample_euler"]
