"""Load pretrained MAISI rflow UNet weights into MAISIUNetWrapper.

The MAISI-v2 rectified flow checkpoint (`diff_unet_3d_rflow-mr.pt`) shares the
same DiffusionModelUNet architecture but uses 512-dim time embeddings (time +
class + spacing) vs our 256-dim (time only). 402/430 inner keys transfer with
exact shape match; 28 ``time_emb_proj.weight`` keys require slicing from
``(C, 512)`` to ``(C, 256)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Keys in rflow checkpoint that have no counterpart in our wrapper's inner UNet
_RFLOW_ONLY_KEYS = frozenset(
    {
        "class_embedding.weight",
        "spacing_layer.0.weight",
        "spacing_layer.0.bias",
        "spacing_layer.2.weight",
        "spacing_layer.2.bias",
    }
)


def load_rflow_pretrained(
    wrapper: torch.nn.Module,
    checkpoint_path: str | Path,
    slice_time_emb_proj: bool = True,
    reinit_output_conv: bool = False,
    log_details: bool = True,
) -> dict[str, Any]:
    """Load pretrained MAISI rflow weights into a MAISIUNetWrapper.

    Args:
        wrapper: ``MAISIUNetWrapper`` instance. Weights are loaded into
            ``wrapper.unet``.
        checkpoint_path: Path to ``diff_unet_3d_rflow-mr.pt``.
        slice_time_emb_proj: If ``True``, slice rflow ``time_emb_proj.weight``
            from ``(C, 512)`` to ``(C, 256)`` (first 256 cols = time portion).
            If ``False``, leave mismatched keys at their current init.
        reinit_output_conv: If ``True``, keep Kaiming/zero init for the output
            conv (``out.2.conv.*``). Useful since rflow predicts velocity but
            we use x-prediction.
        log_details: If ``True``, log per-key loading details at DEBUG level.

    Returns:
        Stats dict with keys: ``loaded_exact``, ``loaded_sliced``,
        ``skipped_reinit``, ``skipped_rflow_only``, ``not_in_rflow``,
        ``loaded_keys`` (set of inner UNet key names that were loaded).

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        KeyError: If checkpoint lacks ``"unet_state_dict"`` key.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "unet_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint missing 'unet_state_dict' key. "
            f"Available keys: {list(ckpt.keys())}"
        )

    rflow_sd = ckpt["unet_state_dict"]
    target_sd = wrapper.unet.state_dict()

    # Build the new state dict starting from current (Kaiming) init
    new_sd = dict(target_sd)

    loaded_exact = 0
    loaded_sliced = 0
    skipped_reinit = 0
    not_in_rflow = 0
    loaded_keys: set[str] = set()

    for key in target_sd:
        if key not in rflow_sd:
            not_in_rflow += 1
            if log_details:
                logger.debug("NOT_IN_RFLOW: %s", key)
            continue

        target_shape = target_sd[key].shape
        rflow_shape = rflow_sd[key].shape

        # Output conv: optionally skip (keep Kaiming/zero init)
        if reinit_output_conv and key.startswith("out."):
            skipped_reinit += 1
            if log_details:
                logger.debug("REINIT (output conv): %s", key)
            continue

        if target_shape == rflow_shape:
            # Exact shape match — copy directly
            new_sd[key] = rflow_sd[key].clone()
            loaded_exact += 1
            loaded_keys.add(key)
            if log_details:
                logger.debug("EXACT: %s %s", key, list(target_shape))
        elif (
            key.endswith("time_emb_proj.weight")
            and len(target_shape) == 2
            and len(rflow_shape) == 2
            and target_shape[0] == rflow_shape[0]
            and rflow_shape[1] > target_shape[1]
        ):
            # time_emb_proj.weight: rflow (C, 512) -> ours (C, 256)
            if slice_time_emb_proj:
                target_dim = target_shape[1]
                new_sd[key] = rflow_sd[key][:, :target_dim].clone()
                loaded_sliced += 1
                loaded_keys.add(key)
                if log_details:
                    logger.debug(
                        "SLICED: %s rflow%s -> %s",
                        key,
                        list(rflow_shape),
                        list(target_shape),
                    )
            else:
                not_in_rflow += 1  # treat as "not loaded"
                if log_details:
                    logger.debug(
                        "SKIP_SLICE (disabled): %s rflow%s vs %s",
                        key,
                        list(rflow_shape),
                        list(target_shape),
                    )
        else:
            # Unexpected shape mismatch
            logger.warning(
                "Shape mismatch (skipping): %s rflow%s vs target%s",
                key,
                list(rflow_shape),
                list(target_shape),
            )
            not_in_rflow += 1

    # Count rflow-only keys (class_embedding, spacing_layer, etc.)
    skipped_rflow_only = sum(1 for k in rflow_sd if k in _RFLOW_ONLY_KEYS)

    # Load the assembled state dict
    wrapper.unet.load_state_dict(new_sd, strict=True)

    total_target = len(target_sd)
    total_loaded = loaded_exact + loaded_sliced
    coverage = total_loaded / max(total_target, 1)

    if coverage < 0.95:
        logger.warning(
            "Low transfer coverage: %d/%d (%.1f%%) target keys loaded",
            total_loaded,
            total_target,
            coverage * 100,
        )

    logger.info(
        "rflow transfer: %d exact + %d sliced = %d/%d loaded (%.1f%%), "
        "%d reinit, %d rflow-only skipped, %d not in rflow",
        loaded_exact,
        loaded_sliced,
        total_loaded,
        total_target,
        coverage * 100,
        skipped_reinit,
        skipped_rflow_only,
        not_in_rflow,
    )

    return {
        "loaded_exact": loaded_exact,
        "loaded_sliced": loaded_sliced,
        "skipped_reinit": skipped_reinit,
        "skipped_rflow_only": skipped_rflow_only,
        "not_in_rflow": not_in_rflow,
        "loaded_keys": loaded_keys,
    }
