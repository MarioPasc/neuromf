# Agent Task: Implement MAISI rflow → NeuroMF Pretrained Weight Loading

**Objective:** Implement a transfer learning pipeline that loads MAISI-v2 rectified flow UNet weights as initialisation for NeuroMF's improved-MeanFlow (iMF) training, replacing Kaiming initialisation. This includes a weight-loading utility, config extension, differential learning rate support, EMA initialisation, comprehensive tests, and diagnostic logging.

**Priority:** HIGH — This is expected to provide 2-5× faster convergence by initialising the backbone with 28,851 epochs of pre-trained 3D medical latent features.

---

## 1. Context

### 1.1 What We Are Doing

NeuroMF trains a 3D UNet (MONAI's `DiffusionModelUNet`) with an improved MeanFlow (iMF) objective to perform 1-NFE diffusion on brain MRI latents (shape `4×48×48×48`) from MAISI's frozen VAE-GAN. Currently, the UNet is initialised with Kaiming weights and trained from scratch.

MAISI-v2 provides a pre-trained rectified flow UNet (`diff_unet_3d_rflow-mr.pt`) that was trained for 28,851 epochs on mixed MR data in the **same MAISI latent space** using a standard flow matching objective. The architecture is nearly identical (same base class), with 99.0% of parameters having exact shape matches.

### 1.2 Why This Works

- **Same latent space:** Both models operate on `4×48³` latents from MAISI's VAE-GAN
- **Same base architecture:** `DiffusionModelUNetMaisi` inherits from `DiffusionModelUNet`
- **Same spatial domain:** Both process 3D medical MR volumes
- **99.0% parameter compatibility:** 402/430 keys match shapes exactly
- The backbone (convolutions, attention, normalisation) encodes **spatial structure** of the latent manifold, which is objective-agnostic

### 1.3 Known Mismatches (All Manageable)

1. **`time_emb_proj` shape mismatch (28 keys):** rflow has `(C, 512)` due to spacing concatenation; ours has `(C, 256)`. The first 256 columns are time-conditioning — slice `[:, :256]`.
2. **Time convention reversal:** rflow uses `t_rflow = 1 - t_ours`. Affects `time_embed` MLP and `time_emb_proj` semantics but <1% of parameters. The spatial backbone is convention-agnostic.
3. **Prediction type:** rflow predicts velocity `v = x_0 - eps`; we use x-prediction. The output conv will adapt quickly.
4. **5 extra keys in rflow:** `class_embedding` (128→256) and `spacing_layer` (3→256→256) — dropped, we are unconditional.
5. **New keys in our model not in rflow:** `r_embed`/`h_embed` (conditioning), `v_out` (v-head) — initialised fresh.

---

## 2. Files to Create

### 2.1 `src/neuromf/utils/pretrained_loading.py` (NEW — ~150 lines)

Weight-loading utility with explicit key mapping, slicing, and diagnostics.

```python
"""Utilities for loading MAISI rflow pretrained weights into NeuroMF.

Handles the key mapping between DiffusionModelUNetMaisi (rflow) and
our DiffusionModelUNet wrapper, including time_emb_proj slicing and
conditioning module exclusion.
"""
```

**Required function:**

```python
def load_rflow_pretrained(
    wrapper: MAISIUNetWrapper,
    checkpoint_path: str | Path,
    slice_time_emb_proj: bool = True,
    reinit_output_conv: bool = False,
    log_details: bool = True,
) -> dict[str, Any]:
    """Load MAISI rflow weights into our MAISIUNetWrapper.

    Args:
        wrapper: Target MAISIUNetWrapper instance.
        checkpoint_path: Path to ``diff_unet_3d_rflow-mr.pt``.
        slice_time_emb_proj: If True, slice rflow's time_emb_proj[:, :256]
            to match our 256-dim time embedding. If False, skip these keys
            and let them remain at Kaiming init.
        reinit_output_conv: If True, re-initialise the output conv
            (``out.2.conv.weight/bias``) with Kaiming/zeros instead of
            loading from rflow. Consider this since rflow predicts velocity
            while we use x-prediction.
        log_details: If True, log detailed key-by-key loading information.

    Returns:
        Dictionary with loading statistics:
        - ``loaded_exact``: int — keys loaded with exact shape match
        - ``loaded_sliced``: int — keys loaded via slicing
        - ``skipped_rflow_only``: list[str] — rflow keys not in our model
        - ``skipped_shape_mismatch``: list[str] — keys skipped due to shape
        - ``reinitialised``: list[str] — keys explicitly re-initialised
        - ``new_in_wrapper``: list[str] — wrapper keys not in rflow (kept at init)
    """
```

**Implementation requirements:**

1. Load checkpoint: `ckpt = torch.load(checkpoint_path, map_location="cpu")`
2. Extract `rflow_sd = ckpt["unet_state_dict"]`
3. Get target state dict: `target_sd = wrapper.unet.state_dict()` (no `unet.` prefix — we load into the inner UNet)
4. Iterate target keys and classify each:
   - **Exact match:** `rflow_sd[key].shape == target_sd[key].shape` → copy
   - **`time_emb_proj` mismatch:** key contains `time_emb_proj` and rflow shape is `(C, 512)` vs target `(C, 256)` → if `slice_time_emb_proj`, copy `rflow_sd[key][:, :256]`; else skip
   - **Output conv reinit:** if `reinit_output_conv` and key starts with `out.2.conv` → skip (keep Kaiming init)
   - **Not in rflow:** keep at current init (these are the conditioning layers in the wrapper, not the inner UNet)
5. Call `wrapper.unet.load_state_dict(new_sd, strict=True)` — since we build a complete state dict, this should be strict
6. Log summary at INFO level:
   ```
   [NeuroMF] Loaded rflow pretrained weights:
     - Exact match: 402 keys (176,731,268 params)
     - Sliced time_emb_proj: 28 keys (1,835,008 params)
     - Skipped (rflow-only): 5 keys [class_embedding.weight, spacing_layer.0.weight, ...]
     - New (not in rflow): r_embed.0.weight, r_embed.0.bias, ...  (kept at init)
   ```
7. Return the statistics dict for programmatic verification

**Critical correctness checks inside the function:**
- Assert `ckpt` contains key `"unet_state_dict"`
- Assert the number of exact-match keys is exactly 402 (hardcode as a sanity check, log warning if different)
- Assert all `time_emb_proj` mismatches have rflow dim 512 and target dim 256
- Assert the loaded state dict covers ALL keys in `target_sd`

### 2.2 `src/neuromf/utils/param_groups.py` (NEW — ~80 lines)

Differential learning rate parameter group construction.

```python
"""Parameter group construction for transfer learning.

Splits model parameters into pretrained (lower LR) and new (full LR)
groups based on which keys were loaded from the rflow checkpoint.
"""
```

**Required function:**

```python
def build_transfer_param_groups(
    model: nn.Module,
    loaded_keys: set[str],
    base_lr: float,
    backbone_lr_factor: float = 0.1,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Build optimizer parameter groups for transfer learning.

    Args:
        model: The full model (MAISIUNetWrapper with v-head etc.).
        loaded_keys: Set of parameter names that were loaded from
            pretrained checkpoint (as they appear in model.named_parameters(),
            i.e., with the ``unet.`` prefix from the wrapper).
        base_lr: Base learning rate for new parameters.
        backbone_lr_factor: Multiplicative factor for pretrained params.
            E.g., 0.1 means pretrained params get lr = base_lr * 0.1.
        weight_decay: Weight decay for all groups.

    Returns:
        List of parameter group dicts suitable for torch.optim.AdamW.
    """
```

**Implementation notes:**
- The `loaded_keys` from `load_rflow_pretrained` are inner UNet keys (no prefix). The wrapper's `named_parameters()` uses `unet.` prefix. Map accordingly: a key `k` was loaded if `k.replace("unet.", "")` is in the loaded keys set.
- Wrapper-only parameters (`r_embed.*`, `h_embed.*`, `v_out.*`) are always in the "new" group.
- Log the count and total params in each group.

### 2.3 Modifications to `src/neuromf/models/latent_meanflow.py`

**Location:** `__init__()` method

Add pretrained weight loading **before** EMA construction. The ordering is critical: if pretrained weights are loaded after EMA init, the EMA shadows will be at Kaiming init while the online model is at pretrained init.

```python
# === EXISTING CODE (around line 60-70) ===
# self.net = MAISIUNetWrapper(unet_config)
# self.ema = EMAModel(self.net, decay=float(config.ema.decay))

# === NEW CODE ===
self.net = MAISIUNetWrapper(unet_config)

# Load pretrained weights if configured
self._pretrained_load_stats = None
if hasattr(self.config, "transfer_learning") and self.config.transfer_learning.get("pretrained_ckpt_path"):
    from neuromf.utils.pretrained_loading import load_rflow_pretrained
    tl_cfg = self.config.transfer_learning
    self._pretrained_load_stats = load_rflow_pretrained(
        wrapper=self.net,
        checkpoint_path=tl_cfg.pretrained_ckpt_path,
        slice_time_emb_proj=tl_cfg.get("slice_time_emb_proj", True),
        reinit_output_conv=tl_cfg.get("reinit_output_conv", False),
    )

# EMA copies from model — MUST come after pretrained loading
self.ema = EMAModel(self.net, decay=float(config.ema.decay))
```

**Location:** `configure_optimizers()` method

Replace single param group with differential LR when transfer learning is active:

```python
# === NEW CODE (replaces existing single-group optimizer) ===
if self._pretrained_load_stats is not None:
    from neuromf.utils.param_groups import build_transfer_param_groups
    tl_cfg = self.config.transfer_learning
    # Build loaded_keys set with unet. prefix for wrapper-level matching
    loaded_inner_keys = (
        set(f"unet.{k}" for k in self._pretrained_load_stats.get("loaded_keys", set()))
    )
    param_groups = build_transfer_param_groups(
        model=self.net,
        loaded_keys=loaded_inner_keys,
        base_lr=float(tr.lr),
        backbone_lr_factor=float(tl_cfg.get("backbone_lr_factor", 0.1)),
        weight_decay=float(tr.weight_decay),
    )
    optimizer = torch.optim.AdamW(param_groups, betas=tuple(tr.betas))
else:
    # Existing single-group optimizer (unchanged)
    optimizer = torch.optim.AdamW(
        self.net.parameters(),
        lr=float(tr.lr),
        weight_decay=float(tr.weight_decay),
        betas=tuple(tr.betas),
    )
```

**IMPORTANT:** The `load_rflow_pretrained` function must also return the set of loaded keys (both exact and sliced) in its return dict under key `"loaded_keys"` — a `set[str]` of inner UNet key names that were populated from the checkpoint.

### 2.4 Config Extension to `configs/train_meanflow.yaml`

Add a new `transfer_learning` section:

```yaml
# Transfer learning from MAISI rflow pretrained weights
transfer_learning:
  # Path to MAISI rflow checkpoint. Set to null to disable (Kaiming init).
  pretrained_ckpt_path: null  # or: /path/to/diff_unet_3d_rflow-mr.pt
  # LR multiplier for pretrained backbone params (new params get full LR)
  backbone_lr_factor: 0.1
  # Slice rflow time_emb_proj[:, :256] to match our 256-dim embedding
  slice_time_emb_proj: true
  # Re-initialise output conv (rflow predicts velocity, we use x-prediction)
  reinit_output_conv: false
```

**Default:** `pretrained_ckpt_path: null` — this means the existing Kaiming-init behaviour is the default, and transfer learning is opt-in.

### 2.5 `tests/test_transfer_loading.py` (NEW — ~250 lines)

Comprehensive test suite for the transfer learning pipeline.

**Test infrastructure requirements:**
- Use tiny configs: `channels=[8, 16, 32, 64]`, `num_res_blocks=1`, `latent_spatial_size=16`
- Create a mock rflow checkpoint that mimics the real structure (435 keys with correct shapes, including the 512-dim `time_emb_proj` and 5 conditioning-only keys)
- All tests must run on CPU

**Required tests:**

```python
# Fixtures
def _tiny_rflow_config() -> dict:
    """Config dict matching DiffusionModelUNetMaisi structure (tiny)."""
    # channels=[8, 16, 32, 64], num_head_channels=[0,0,32,32], etc.
    # include_spacing_input=True, num_class_embeds=128
    # This produces time_emb_proj with dim 64 (= 8*4*2 from spacing concat)
    # Actually: time_embed_dim = channels[0]*4 = 32
    # With spacing: new_time_embed_dim = 64
    # So time_emb_proj layers are (C, 64) in rflow vs (C, 32) in ours
    ...

def _tiny_target_config() -> MAISIUNetConfig:
    """Config for our MAISIUNetWrapper (tiny)."""
    # Same spatial structure but DiffusionModelUNet, no spacing
    # time_emb_proj layers are (C, 32)
    ...

def _create_mock_rflow_ckpt(tmp_path: Path) -> Path:
    """Create a mock rflow-like checkpoint file."""
    # Instantiate DiffusionModelUNetMaisi with _tiny_rflow_config()
    # Save as {"unet_state_dict": model.state_dict(), "epoch": 100, ...}
    ...


# Tests
def test_key_coverage():
    """Verify expected number of keys load correctly."""
    # Load mock checkpoint into tiny wrapper
    # Assert stats["loaded_exact"] == expected count
    # Assert stats["loaded_sliced"] == expected count (time_emb_proj keys)
    # Assert len(stats["skipped_rflow_only"]) == 5 (class_embedding + spacing_layer)
    ...

def test_shape_match_after_loading():
    """Verify no shape mismatches remain after loading."""
    # Load weights, then iterate all params and assert shapes match config
    ...

def test_time_emb_proj_slicing():
    """Verify time_emb_proj[:, :256] slicing preserves correct values."""
    # Create rflow checkpoint with known values in time_emb_proj
    # Load with slice_time_emb_proj=True
    # Assert loaded weights equal rflow_weight[:, :first_half]
    ...

def test_time_emb_proj_skip():
    """Verify time_emb_proj stays at Kaiming init when slicing disabled."""
    # Load with slice_time_emb_proj=False
    # Assert time_emb_proj weights differ from rflow values
    ...

def test_output_sanity():
    """Forward pass after loading produces finite output."""
    # Load pretrained, run forward with random input, assert all finite
    ...

def test_gradient_flow():
    """Gradients flow through all loaded weights."""
    # Load pretrained, run forward + backward, assert >95% params have grad
    ...

def test_ema_initialised_from_pretrained():
    """EMA shadows match pretrained state after init."""
    # Simulate the init order: load pretrained → create EMA
    # Assert EMA shadow params equal model params (both pretrained)
    ...

def test_reinit_output_conv():
    """Output conv is re-initialised when reinit_output_conv=True."""
    # Load with reinit_output_conv=True
    # Assert out.2.conv.weight differs from rflow values
    # Assert out.2.conv.weight is zero-initialised (MONAI convention)
    ...

def test_param_groups():
    """Differential LR param groups are correctly constructed."""
    # Build param groups, assert two groups exist
    # Assert pretrained group has lr = base_lr * factor
    # Assert new group has lr = base_lr
    # Assert total params across groups == total model params
    ...

def test_no_transfer_when_disabled():
    """Model uses Kaiming init when pretrained_ckpt_path is null."""
    # Construct LatentMeanFlow without transfer_learning config
    # Assert weights differ from rflow checkpoint
    ...
```

---

## 3. Key Implementation Details

### 3.1 Checkpoint Loading

```python
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
rflow_sd = ckpt["unet_state_dict"]
```

The checkpoint top-level keys are: `epoch`, `loss`, `num_train_timesteps`, `scale_factor`, `unet_state_dict`, `epoch_finished`, `optimizer_state_dict`, `scheduler_state_dict`. We only need `unet_state_dict`.

### 3.2 Key Classification Logic

```python
RFLOW_ONLY_PREFIXES = ("class_embedding", "spacing_layer")
TIME_EMB_PROJ_SUFFIX = "time_emb_proj.weight"

for key in target_sd:
    if key in rflow_sd:
        if rflow_sd[key].shape == target_sd[key].shape:
            # Exact match — load directly
            new_sd[key] = rflow_sd[key]
        elif key.endswith(TIME_EMB_PROJ_SUFFIX) and slice_time_emb_proj:
            # Slice the time-conditioning portion
            target_dim = target_sd[key].shape[1]  # 256
            new_sd[key] = rflow_sd[key][:, :target_dim].clone()
        else:
            # Shape mismatch, cannot load — keep init
            new_sd[key] = target_sd[key]
    else:
        # Not in rflow — keep init
        new_sd[key] = target_sd[key]
```

### 3.3 Biases for `time_emb_proj`

Note: the `time_emb_proj` has both `.weight` and `.bias`. The **bias** has shape `(C_out,)` which is the same in both models — it transfers directly as an exact match. Only the weight has the 512→256 mismatch.

### 3.4 Output Conv Reinitialisation

If `reinit_output_conv=True`:
```python
OUTPUT_CONV_KEYS = ("out.2.conv.weight", "out.2.conv.bias")
# Re-init with zeros (matching MONAI's zero-init convention for final conv)
if key in OUTPUT_CONV_KEYS:
    new_sd[key] = torch.zeros_like(target_sd[key])
```

### 3.5 Real Checkpoint Path

```
/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt
```

For tests, always use mock checkpoints created via `_create_mock_rflow_ckpt()`. Never depend on the real checkpoint in tests (it is 2GB).

### 3.6 `DiffusionModelUNetMaisi` for Mock Checkpoints

To create a realistic mock checkpoint, instantiate the MAISI variant:

```python
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import DiffusionModelUNetMaisi

mock_rflow = DiffusionModelUNetMaisi(
    spatial_dims=3,
    in_channels=4,
    out_channels=4,
    num_channels=[8, 16, 32, 64],  # Tiny
    attention_levels=[False, False, True, True],
    num_head_channels=[0, 0, 32, 32],
    num_res_blocks=1,  # Tiny
    use_flash_attention=False,
    include_spacing_input=True,
    num_class_embeds=128,
    resblock_updown=True,
    include_fc=True,
)
```

If `DiffusionModelUNetMaisi` is unavailable in the installed MONAI version, manually construct a mock state dict by taking the tiny `DiffusionModelUNet` state dict and:
- Adding `class_embedding.weight` with shape `(128, tiny_time_embed_dim)`
- Adding `spacing_layer.{0,2}.{weight,bias}` with appropriate shapes
- Expanding all `time_emb_proj.weight` tensors from `(C, tiny_time_embed_dim)` to `(C, 2*tiny_time_embed_dim)` by concatenating random data

### 3.7 `include_fc` Verification

Both rflow and our model use `include_fc=True`. This is the default in `DiffusionModelUNet`. Verify this is set in our `MAISIUNetConfig` — if the field doesn't exist, it defaults to True. If our config explicitly sets it to False, this would cause attention key mismatches. Check and fix if needed.

---

## 4. Verification Criteria

### 4.1 Automated (Tests Must Pass)

- [ ] All keys in inner UNet state dict are populated (no missing keys)
- [ ] 402 keys loaded with exact shape match (for full-size config)
- [ ] 28 `time_emb_proj.weight` keys loaded via slicing (for full-size config)
- [ ] 5 rflow-only keys correctly skipped
- [ ] Forward pass produces finite output after loading
- [ ] Gradients flow to >95% of parameters after loading
- [ ] EMA shadows == model params immediately after construction
- [ ] Param groups correctly split pretrained vs. new params
- [ ] Total params across groups == total model params
- [ ] `reinit_output_conv=True` produces zero-initialised output conv
- [ ] `slice_time_emb_proj=False` leaves time_emb_proj at Kaiming init
- [ ] No transfer when `pretrained_ckpt_path` is null/missing

### 4.2 Manual (Verify by Inspection)

- [ ] Loading logs are clear and informative at INFO level
- [ ] Config YAML extension is backward-compatible (null default)
- [ ] No changes to existing test behaviour

---

## 5. Files Summary

| File | Action | Lines (est.) |
|------|--------|------|
| `src/neuromf/utils/pretrained_loading.py` | CREATE | ~150 |
| `src/neuromf/utils/param_groups.py` | CREATE | ~80 |
| `src/neuromf/models/latent_meanflow.py` | MODIFY | ~40 lines changed |
| `configs/train_meanflow.yaml` | MODIFY | ~8 lines added |
| `tests/test_transfer_loading.py` | CREATE | ~250 |
| **Total** | | **~530** |

---

## 6. What NOT to Do

- **Do NOT modify `MAISIUNetWrapper`** — the wrapper architecture is correct as-is. We are only loading weights into it differently.
- **Do NOT change `DiffusionModelUNet` to `DiffusionModelUNetMaisi`** — Option C from the exploration was rejected. We keep our simpler architecture and handle the key mapping externally.
- **Do NOT load optimizer state** from the rflow checkpoint — the optimizer state is for a different objective/LR schedule and would cause instability.
- **Do NOT modify the EMA class** — just ensure the construction order (load pretrained → create EMA) is correct.
- **Do NOT make transfer learning the default** — `pretrained_ckpt_path: null` means the existing Kaiming-init path is unchanged.
- **Do NOT use the real 2GB checkpoint in tests** — always use mock checkpoints.

---

## 7. Execution Order

1. Create `src/neuromf/utils/pretrained_loading.py` with `load_rflow_pretrained()`
2. Create `src/neuromf/utils/param_groups.py` with `build_transfer_param_groups()`
3. Create `tests/test_transfer_loading.py` with mock checkpoint fixtures and all tests
4. Run tests to verify loading utility works correctly on mock data
5. Modify `src/neuromf/models/latent_meanflow.py` — add pretrained loading in `__init__()` and param groups in `configure_optimizers()`
6. Modify `configs/train_meanflow.yaml` — add `transfer_learning` section
7. Run full test suite to verify no regressions
8. Manual smoke test: load real checkpoint, print loading stats, verify forward pass
