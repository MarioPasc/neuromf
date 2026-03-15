---
description: General coding standards for the neuromf project
---

# Coding Standards

## Code Quality
1. **Type hints** on ALL function signatures and return types.
2. **Google-style docstrings** on all public functions and classes. No usage examples needed.
3. **Brief inline comments** on non-obvious code only. Do not comment obvious lines.
4. **Logging:** Python `logging` module with `rich` handler. INFO for training events, DEBUG for shapes/values.
5. **No magic numbers.** All hyperparameters from YAML configs via OmegaConf.
6. **Prefer library functions.** MONAI transforms over custom preprocessing. `einops.rearrange` over manual reshapes. `F.scaled_dot_product_attention` over manual QKV matmuls.
7. **Keep functions atomic.** One conceptual task per function.
8. **Leverage reference codebases.** Start from PyTorch MeanFlow reference (`src/external/MeanFlow-PyTorch/`), do not reimplement tested patterns.

## Configuration
9. **Config loading:** Use `from neuromf.config import load_merged_config` for all CLI scripts. Never duplicate the OmegaConf merge chain.
10. **Config merge order:** `base.yaml` -> `train_meanflow.yaml` -> `[generate.yaml]` -> user overlays.

## Testing
11. **Tests use pytest.** Each test file runnable independently: `pytest tests/test_xxx.py -v`.
12. **Test markers:** All tests must have `@pytest.mark.phaseN` and `@pytest.mark.critical` or `@pytest.mark.informational`. Tests constructing UNet/LatentMeanFlow must also have `@pytest.mark.slow`.
13. **Fast suite:** `pytest -m "not slow"` must pass in <1 min after any code change.

## Critical Safety Rules
14. **FORBIDDEN combo:** `prediction_type="x"` + `jvp_strategy="finite_difference"` causes 1/t singularity explosion. **Safe combos:** `x + exact`, `u + FD`, `u + exact`. This applies to both source code AND test fixtures.
15. **DDP safety:** All `on_fit_end` callbacks with rank-0-only work must include a `dist.barrier()` so non-zero ranks wait. Without this, NCCL timeout kills training after 30 minutes.
16. **VAE decode in callbacks:** Use CPU device to avoid bf16/autocast conflicts from the training precision context.
