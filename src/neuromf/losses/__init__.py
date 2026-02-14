"""Loss functions: MeanFlow JVP, per-channel Lp, and iMF combined loss.

Implements the core MeanFlow JVP loss (Eq. 12), per-channel Lp norm (Eq. 28),
and the iMF combined objective (Eq. 13) with adaptive weighting (Eq. 14).
"""

from neuromf.losses.combined_loss import combined_imf_loss
from neuromf.losses.lp_loss import lp_loss
from neuromf.losses.meanflow_jvp import compute_compound_velocity, meanflow_loss

__all__ = [
    "combined_imf_loss",
    "compute_compound_velocity",
    "lp_loss",
    "meanflow_loss",
]
