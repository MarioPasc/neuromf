"""α-Flow curriculum scheduler for progressive MeanFlow training.

Implements the sigmoid-based annealing schedule from:
Zhang et al., "AlphaFlow: Understanding and Improving MeanFlow Models,"
arXiv:2510.20771, 2025. Section 5.2 and Algorithm 2.

The scheduler controls the consistency step ratio α(k) which determines
the maximum gap between r and t in MeanFlow sampling. α=0 corresponds
to pure flow matching (r=t), α=1 to full MeanFlow.
"""

import math
from dataclasses import dataclass


@dataclass
class AlphaSchedulerConfig:
    """Configuration for the α-Flow curriculum scheduler.

    Args:
        start_step: Training step where annealing begins (k_s).
            Set to 0 for no FM pretraining, or to the end of Stage 1.
        end_step: Training step where annealing ends (k_e).
        gamma: Temperature for the sigmoid transition (default 25.0).
            Higher = sharper transition.
        eta: Final clamping value for α (default 1.0).
            α-Flow paper finds optimal η=5e-3 on ImageNet.
            Start with 1.0 for our setting and ablate.
        mode: "sigmoid" (α-Flow) or "linear" (simpler alternative).
    """

    start_step: int = 0
    end_step: int = 100_000
    gamma: float = 25.0
    eta: float = 1.0
    mode: str = "sigmoid"

    def __post_init__(self) -> None:
        if self.mode not in ("sigmoid", "linear"):
            raise ValueError(f"Unknown mode: {self.mode!r}. Must be 'sigmoid' or 'linear'.")
        if self.eta < 0.0:
            raise ValueError(f"eta must be >= 0, got {self.eta}")
        if self.end_step < self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must be >= start_step ({self.start_step})"
            )


class AlphaScheduler:
    """Computes α(k) at each training step.

    The schedule transitions from α=0 (pure flow matching) to α=η
    (MeanFlow with controlled consistency gap) over the interval
    [start_step, end_step].

    Args:
        config: Scheduler configuration.
    """

    def __init__(self, config: AlphaSchedulerConfig) -> None:
        self.config = config

    def get_alpha(self, step: int) -> float:
        """Compute α at the given training step.

        Args:
            step: Current global training step.

        Returns:
            α value in [0, η].
        """
        cfg = self.config

        if cfg.eta == 0.0:
            return 0.0

        if step < cfg.start_step:
            return 0.0  # Pure FM during Stage 1

        if step >= cfg.end_step:
            return cfg.eta  # Converged

        # Normalised progress in [0, 1]
        progress = (step - cfg.start_step) / max(cfg.end_step - cfg.start_step, 1)

        if cfg.mode == "sigmoid":
            # Sigmoid schedule: α = η * σ(γ * (progress - 0.5))
            x = cfg.gamma * (progress - 0.5)
            alpha = cfg.eta * (1.0 / (1.0 + math.exp(-x)))
        elif cfg.mode == "linear":
            alpha = cfg.eta * progress
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

        return alpha
