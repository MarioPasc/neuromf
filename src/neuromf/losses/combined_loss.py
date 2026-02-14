"""iMF combined FM + MF loss (Eq. 13).

Thin utility for combining flow-matching and MeanFlow loss terms
with configurable weighting.
"""

from torch import Tensor


def combined_imf_loss(
    fm_loss: Tensor,
    mf_loss: Tensor,
    lambda_mf: float = 1.0,
) -> Tensor:
    """Combine flow-matching and MeanFlow losses (Eq. 13).

    ``L_iMF = L_FM + lambda_MF * L_MF``

    Args:
        fm_loss: Flow-matching loss (scalar).
        mf_loss: MeanFlow loss (scalar).
        lambda_mf: Weight for MeanFlow term.

    Returns:
        Combined scalar loss.
    """
    return fm_loss + lambda_mf * mf_loss
