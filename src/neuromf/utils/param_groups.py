"""Differential learning rate parameter groups for transfer learning.

Splits model parameters into two optimizer groups: pretrained (backbone) params
get a reduced learning rate, while newly initialised params (embeddings, v-head)
get the full learning rate.
"""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def build_transfer_param_groups(
    model: nn.Module,
    loaded_keys: set[str],
    base_lr: float,
    backbone_lr_factor: float = 0.1,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Build optimizer parameter groups with differential learning rates.

    Parameters whose names (from ``model.named_parameters()``) appear in
    ``loaded_keys`` are assigned ``base_lr * backbone_lr_factor``. All others
    get ``base_lr``.

    Args:
        model: The full wrapper model (e.g. ``MAISIUNetWrapper``).
        loaded_keys: Set of parameter names (as returned by
            ``model.named_parameters()``) that were loaded from the pretrained
            checkpoint. For inner UNet keys, these should include the ``unet.``
            prefix.
        base_lr: Base learning rate for new parameters.
        backbone_lr_factor: Multiplier for pretrained parameters
            (e.g. 0.1 = 10x slower).
        weight_decay: Weight decay for all groups.

    Returns:
        List of two dicts, each with ``params``, ``lr``, and ``weight_decay``.
    """
    pretrained_params: list[nn.Parameter] = []
    new_params: list[nn.Parameter] = []
    pretrained_numel = 0
    new_numel = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in loaded_keys:
            pretrained_params.append(param)
            pretrained_numel += param.numel()
        else:
            new_params.append(param)
            new_numel += param.numel()

    backbone_lr = base_lr * backbone_lr_factor

    groups: list[dict[str, Any]] = []
    if pretrained_params:
        groups.append(
            {
                "params": pretrained_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            }
        )
    if new_params:
        groups.append(
            {
                "params": new_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )

    logger.info(
        "Parameter groups: %d pretrained (%.2fM, lr=%.2e) + %d new (%.2fM, lr=%.2e)",
        len(pretrained_params),
        pretrained_numel / 1e6,
        backbone_lr,
        len(new_params),
        new_numel / 1e6,
        base_lr,
    )

    return groups
