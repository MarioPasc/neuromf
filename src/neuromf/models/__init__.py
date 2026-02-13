"""Model definitions: Lightning module, baseline, and LoRA fine-tuning.

Contains the main Latent MeanFlow Lightning module, Rectified Flow baseline
for ablation comparison, and LoRA injection utilities for domain adaptation.
"""

from neuromf.models.toy_mlp import ToyMLP

__all__ = ["ToyMLP"]
