# MAISI rflow Checkpoint → NeuroMF Transfer Learning: Exploration Report

**Date:** 2026-02-22
**Purpose:** Evaluate feasibility and specify implementation for loading MAISI-v2 rectified flow UNet weights as initialisation for our improved-MeanFlow training, replacing Kaiming init.

---

## 1. MAISI rflow Checkpoint Structure

**Path:** `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/diff_unet_3d_rflow-mr.pt`
**Size:** 2,066.3 MB

### 1.1 Top-Level Keys

| Key | Type | Value |
|-----|------|-------|
| `epoch` | int | 28,851 |
| `loss` | float | 1.198 |
| `num_train_timesteps` | int | 1,000 |
| `scale_factor` | float16 tensor | 0.96240234375 |
| `unet_state_dict` | dict | 435 keys (the model weights) |
| `epoch_finished` | bool | True |
| `optimizer_state_dict` | dict | 2 keys (Adam state) |
| `scheduler_state_dict` | dict | 2 keys |

The checkpoint contains full training state (optimizer, scheduler), not just model weights. We only need `unet_state_dict`.

### 1.2 Architecture Config

**Config path:** `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/configs/config_network_rflow.json`

```json
{
  "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
  "spatial_dims": 3,
  "in_channels": 4,
  "out_channels": 4,
  "num_channels": [64, 128, 256, 512],
  "attention_levels": [false, false, true, true],
  "num_head_channels": [0, 0, 32, 32],
  "num_res_blocks": 2,
  "use_flash_attention": true,
  "include_spacing_input": true,
  "num_class_embeds": 128,
  "resblock_updown": true,
  "include_fc": true
}
```

### 1.3 State Dict Key Analysis

**Total:** 435 keys, all `torch.float32`, **180,500,868 parameters** (~180.5M)

**Top-level modules:**

| Module | Description |
|--------|-------------|
| `conv_in` | 4 → 64 channels (3×3×3 conv) |
| `time_embed` | Sinusoidal 64 → 256 → 256 MLP |
| `class_embedding` | 128 classes → 256 dims (nn.Embedding) |
| `spacing_layer` | 3 → 256 → 256 MLP (voxel spacing) |
| `down_blocks.0–3` | Encoder with resnets + optional attention |
| `middle_block` | Bottleneck with attention |
| `up_blocks.0–3` | Decoder with resnets + optional attention |
| `out` | GroupNorm → SiLU → Conv3d (64 → 4) |

### 1.4 Attention Layers

**Attention IS present** (unlike the VAE which has none):

| Block | Attention? | Head Channels | Channels |
|-------|-----------|---------------|----------|
| `down_blocks.0` | No | 0 | 64 |
| `down_blocks.1` | No | 0 | 128 |
| `down_blocks.2` | **Yes** | **32** | 256 |
| `down_blocks.3` | **Yes** | **32** | 512 |
| `middle_block` | **Yes** | **32** | 512 |
| `up_blocks.0` | **Yes** | **32** | 512 |
| `up_blocks.1` | **Yes** | **32** | 256 |
| `up_blocks.2` | No | 0 | 128 |
| `up_blocks.3` | No | 0 | 64 |

### 1.5 Time Embedding Dimension Issue

The critical architectural difference is in the **time embedding projection** (`time_emb_proj`) layers:

- **MAISI rflow:** `time_embed_dim = 64 * 4 = 256`, but then `include_spacing_input=True` **concatenates** a 256-dim spacing embedding, making `new_time_embed_dim = 512`. All ResBlock `time_emb_proj` layers have input dim 512.
- **Our wrapper:** Uses MONAI's `DiffusionModelUNet` (not `DiffusionModelUNetMaisi`), which has no spacing input. `time_embed_dim = 256`, and all `time_emb_proj` layers have input dim 256.

**Concatenation order** (verified from source):
```python
# In DiffusionModelUNetMaisi.forward():
emb = self.time_embed(t_emb)         # Shape: (B, 256) — positions [0:256]
if num_class_embeds:
    emb += class_embedding            # ADDED to positions [0:256]
emb = torch.cat((emb, spacing_emb), dim=1)  # Spacing at positions [256:512]
```

So `time_emb_proj.weight[:, 0:256]` corresponds to **time + class conditioning**, and `time_emb_proj.weight[:, 256:512]` corresponds to **spacing conditioning**. Slicing `[:, :256]` preserves all time-relevant projection knowledge.

---

## 2. Our MAISIUNetWrapper Internals

### 2.1 Model Construction

```python
# In MAISIUNetWrapper.__init__():
self.unet = DiffusionModelUNet(
    spatial_dims=3, in_channels=4, out_channels=4,
    channels=[64, 128, 256, 512],
    attention_levels=[False, False, True, True],
    num_res_blocks=2, num_head_channels=[0, 0, 32, 32],
    norm_num_groups=32, resblock_updown=True,
    use_flash_attention=False,   # Required for torch.func.jvp
    with_conditioning=False,     # Unconditional
)
```

**Inner UNet:** 430 keys, 178,566,276 params (~178.6M)

### 2.2 State Dict Key Mapping

Our wrapper's `state_dict()` uses the prefix `unet.` for all inner UNet keys:
- Inner UNet key: `down_blocks.0.resnets.0.conv1.weight`
- Wrapper key: `unet.down_blocks.0.resnets.0.conv1.weight`

Additionally, the wrapper has **4 extra keys** for the r-embedding (in "dual" mode) or h-embedding (in "t_h" mode):
- `r_embed.0.weight`, `r_embed.0.bias`, `r_embed.2.weight`, `r_embed.2.bias` (dual mode)
- `h_embed.0.weight`, `h_embed.0.bias`, `h_embed.2.weight`, `h_embed.2.bias` (t_h mode)

### 2.3 v-head Keys

When `use_v_head=True` with 1 ResBlock, the v-head adds these keys:
```
v_out.0.norm1.{weight,bias}    # GroupNorm
v_out.0.conv1.{weight,bias}    # Conv3d
v_out.0.norm2.{weight,bias}    # GroupNorm
v_out.0.conv2.{weight,bias}    # Conv3d (zero-init)
v_out.1.{weight,bias}          # GroupNorm
v_out.3.{weight,bias}          # Conv3d (final projection, zero-init)
```

Total v-head params: ~228K (1 ResBlock)

### 2.4 Output Conv Layer Name

The output layer in MONAI's `DiffusionModelUNet`:
```
out.0.weight  → GroupNorm weight  (64,)
out.0.bias    → GroupNorm bias    (64,)
out.2.conv.weight → Conv3d weight (4, 64, 3, 3, 3)
out.2.conv.bias   → Conv3d bias  (4,)
```

---

## 3. Key-by-Key Comparison: rflow ↔ Our Inner UNet

### 3.1 Summary

| Category | Count | Params |
|----------|-------|--------|
| Common keys with **matching shapes** | 402 | 176,731,268 (97.9% of rflow, 99.0% of ours) |
| Common keys with **mismatched shapes** | 28 | — |
| Keys only in rflow | 5 | 98,560 |
| Keys only in ours | 0 | — |

### 3.2 Shape-Matched Keys (402 / 402 — 100% Directly Transferable)

All conv weights, norm weights/biases, attention Q/K/V/O projections, skip connections, conv_in, and output layer match exactly. These account for **99.0% of our model's parameters**.

### 3.3 Shape-Mismatched Keys (28 — All `time_emb_proj`)

All 28 mismatched keys are `time_emb_proj.weight` layers in ResBlocks/downsamplers/upsamplers:
- **rflow:** shape `(C_out, 512)`
- **ours:** shape `(C_out, 256)`

The first 256 columns of the rflow weights correspond to time conditioning; the last 256 correspond to spacing. **Slicing `rflow_weight[:, :256]` is a valid partial transfer.**

### 3.4 Keys Only in rflow (5 — Conditioning Modules)

| Key | Shape | Purpose |
|-----|-------|---------|
| `class_embedding.weight` | (128, 256) | Class conditioning |
| `spacing_layer.0.weight` | (256, 3) | Spacing MLP layer 1 |
| `spacing_layer.0.bias` | (256,) | Spacing MLP bias 1 |
| `spacing_layer.2.weight` | (256, 256) | Spacing MLP layer 2 |
| `spacing_layer.2.bias` | (256,) | Spacing MLP bias 2 |

These are dropped during transfer (our model is unconditional).

### 3.5 Keys Only in Ours (0)

Every key in our inner UNet exists in the rflow checkpoint. Perfect subset.

---

## 4. Time Convention Mismatch

### 4.1 MAISI rflow Convention

From `monai.networks.schedulers.rectified_flow.RFlowScheduler`:

```python
# add_noise():
timepoints = timesteps.float() / self.num_train_timesteps
timepoints = 1 - timepoints  # Inverted!
noisy_samples = timepoints * original_samples + (1 - timepoints) * noise
```

**Interpretation:** When `timesteps` is high (near 1000), `timepoints` is near 0, so the sample is mostly noise. When `timesteps` is low (near 0), `timepoints` is near 1, so the sample is mostly data.

**Velocity target:**
```python
loss = loss_l1(predicted_velocity, (inputs - noise))  # v = x_0 - eps
```

### 4.2 Our MeanFlow Convention

```python
z_t = (1 - t) * z_0 + t * eps  # t=0 is data, t=1 is noise
v_c = eps - z_0                 # Velocity target
```

### 4.3 Comparison

| Aspect | MAISI rflow | Our MeanFlow |
|--------|------------|--------------|
| Interpolation | `t_rflow * x_0 + (1-t_rflow) * eps` | `(1-t) * x_0 + t * eps` |
| t=0 | Near pure noise | Pure data |
| t=1 | Near pure data | Pure noise |
| Velocity target | `x_0 - eps` | `eps - x_0` |
| Relationship | `t_rflow = 1 - t_ours` | `t_ours = 1 - t_rflow` |

**The velocity is exactly negated**: `v_rflow = -v_ours`.

### 4.4 Impact on Transfer Learning

The time convention affects:
1. **`time_embed` MLP weights** — trained to produce useful embeddings for the rflow convention. When we pass our times (reversed), the sinusoidal inputs will differ, but the MLP can learn to adapt.
2. **`time_emb_proj` weights** — project the time embedding into each ResBlock. These encode time-dependent scaling learned under rflow conventions.
3. **`out` layer weights** — trained to output `x_0 - eps`; we want `eps - x_0` (or `x_hat` in x-prediction mode).

**Key insight for x-prediction:** We use **x-prediction** (`prediction_type="x"`), meaning our model outputs `x_hat` directly, and the conversion to `u` happens outside the network: `u = (z_t - x_hat) / t`. The rflow model also predicts velocity, but:
- rflow velocity = `x_0 - eps`
- Our conversion: from `v_rflow = x_0 - eps`, we can derive `x_hat` via the flow ODE

However, the model's internal representations (conv/attention features) learn **spatial structure** (edges, textures, anatomy) that is **time-convention-agnostic**. The bulk of the parameters (convolutions, attention, normalization) encode data-driven features, not time conditioning.

**Practical recommendation:** Load all 402 shape-matched keys directly. The time_embed MLP and output conv will be retrained early in training. The spatial feature extraction (>99% of params) provides a strong initialization.

---

## 5. Compatibility Constraints

### 5.1 Flash Attention

| Aspect | rflow | Ours |
|--------|-------|------|
| `use_flash_attention` | `true` | `false` (required for `torch.func.jvp`) |

**Impact on weight transfer: NONE.** Flash attention is a runtime optimization — it changes the computation path but not the parameter shapes. The Q/K/V/O projection weights are identical regardless of flash attention setting. Verified: all attention keys have matching shapes.

### 5.2 `include_fc`

| Aspect | rflow | Ours |
|--------|-------|------|
| `include_fc` | `true` | `true` (DiffusionModelUNet default) |

The `include_fc` parameter affects **attention block internals** (adds a fully-connected layer after attention). Both use `include_fc=True`, so attention weights are compatible. Note: `DiffusionModelUNet` defaults to `include_fc=True` (verified from signature).

### 5.3 `with_conditioning` / `cross_attention_dim`

The rflow model does NOT use cross-attention conditioning (`with_conditioning` is not set in the config). The `class_embedding` and `spacing_layer` are additive/concatenative to the time embedding, not cross-attention. Our model also has `with_conditioning=False`. No cross-attention weight mismatch.

### 5.4 Model Class Difference

| Aspect | rflow | Ours |
|--------|-------|------|
| Class | `DiffusionModelUNetMaisi` | `DiffusionModelUNet` |
| Source | `monai.apps.generation.maisi.networks.diffusion_model_unet_maisi` | `monai.networks.nets.diffusion_model_unet` |

`DiffusionModelUNetMaisi` **inherits from** `DiffusionModelUNet` and adds:
- `class_embedding` (nn.Embedding)
- `spacing_layer` (MLP)
- `top/bottom_region_index_layer` (optional MLPs)
- Modified `forward()` that concatenates these embeddings

The core UNet architecture (conv_in, down_blocks, middle_block, up_blocks, out, time_embed) is identical. This is why 402/430 keys match directly.

---

## 6. Training Infrastructure

### 6.1 Optimizer Construction

**Location:** `src/neuromf/models/latent_meanflow.py`, `configure_optimizers()` (lines 316–371)

```python
optimizer = torch.optim.AdamW(
    self.net.parameters(),  # SINGLE param group, all params
    lr=float(tr.lr),        # 1e-4
    weight_decay=float(tr.weight_decay),  # 0
    betas=tuple(tr.betas),  # (0.9, 0.95)
)
```

**Currently:** Single param group with one LR. No differential learning rates.

**For transfer learning:** Need to split into param groups:
1. **Pretrained backbone** — lower LR (e.g., `lr * 0.1`)
2. **New/reinitialised layers** (time_emb_proj, r/h_embed, v_head, output conv) — full LR

### 6.2 EMA Initialisation

**Location:** `src/neuromf/models/latent_meanflow.py`, `__init__()` (line 70)

```python
self.ema = EMAModel(self.net, decay=float(config.ema.decay))
```

EMA **copies from model at `__init__`**: each parameter is cloned (`param.data.clone()`). So if we load pretrained weights before EMA init, the EMA shadows will start from the pretrained state. If loading after, we need to manually reinitialize the EMA.

### 6.3 Checkpoint Resume

**Location:** `experiments/cli/train.py` (line 628)

```python
trainer.fit(model, train_dl, val_dl, ckpt_path=resume_path)
```

This is Lightning's native checkpoint resume. For pretrained weight loading (not resume), we need a different mechanism — loading partial weights at model construction time, before training starts.

### 6.4 Config Structure

**Location:** `configs/train_meanflow.yaml`

Relevant sections for transfer learning extension:
```yaml
unet:
  # ... existing architecture params ...

training:
  lr: 1.0e-4
  # ... existing training params ...

# NEW (to be added):
# transfer_learning:
#   pretrained_ckpt_path: null  # or path to rflow checkpoint
#   backbone_lr_factor: 0.1    # backbone LR = lr * factor
#   reinit_output: true        # reinit output conv
#   reinit_time_emb_proj: true # reinit time_emb_proj (mismatched dims)
#   slice_time_emb_proj: true  # slice rflow [:, :256] into ours
```

---

## 7. Test Patterns

### 7.1 Existing Test Infrastructure

**Test directory:** `/home/mpascual/research/code/neuromf/tests/`

Relevant patterns for transfer learning tests:

- **Tiny config factory:** Used in `test_diagnostics.py`, `test_latent_meanflow.py`, `test_spatial_masking.py`
  ```python
  channels=[8, 16, 32, 64]  # Reduced from [64, 128, 256, 512]
  num_res_blocks=1           # Reduced from 2
  latent_spatial_size=16     # Reduced from 48
  jvp_strategy="finite_difference"  # Avoid torch.func on CPU
  ```

- **Checkpoint roundtrip test:** `test_latent_meanflow.py::test_P4_T5_checkpoint_save_load` — creates model, simulates training, saves/loads, verifies EMA integrity.

- **Weight sharing test:** `test_maisi_unet_wrapper.py` — copies `model.unet.state_dict()` between wrappers, verifies outputs.

### 7.2 Recommended New Test File

```
tests/test_transfer_loading.py
├── _tiny_source_config()      # Config matching rflow structure (with spacing)
├── _tiny_target_config()      # Config matching our wrapper
├── _create_mock_rflow_ckpt()  # Create fake rflow-like checkpoint
├── test_P4_T9_key_coverage    # Verify 402/430 keys load correctly
├── test_P4_T10_shape_match    # Verify no shape mismatches after loading
├── test_P4_T11_time_emb_slice # Verify time_emb_proj[:, :256] slicing
├── test_P4_T12_output_sanity  # Forward pass after loading produces finite output
├── test_P4_T13_gradient_flow  # Gradients flow through loaded weights
├── test_P4_T14_ema_init       # EMA initialised from pretrained state
├── test_P4_T15_strict_false   # strict=False loading logs skipped keys
```

### 7.3 Conftest Fixtures Available

```python
base_config     # Session-scoped, loads configs/base.yaml
device          # Session-scoped, CPU or CUDA
results_root    # Session-scoped, results directory path
tmp_path        # Function-scoped (pytest builtin), temporary directory
```

---

## 8. Transfer Strategy Recommendations

### 8.1 Option A: Partial Load with Slicing (Recommended)

1. Load rflow `unet_state_dict` (435 keys)
2. Drop 5 conditioning-only keys (`class_embedding`, `spacing_layer`)
3. For 402 shape-matched keys: load directly into `wrapper.unet`
4. For 28 `time_emb_proj` keys: **slice** `rflow_weight[:, :256]` → preserves time conditioning knowledge
5. Reinitialise: v-head (zero-init per existing convention), r/h_embed (Kaiming)
6. Optionally reinitialise output conv `out.2.conv` (the velocity prediction head)

**Pros:** Maximum knowledge transfer (99%+ params), includes time conditioning projections
**Cons:** Assumes time-embedding portion occupies first 256 dims (verified from source)

### 8.2 Option B: Shape-Match Only (Conservative)

1. Load rflow `unet_state_dict` (435 keys)
2. Only load 402 shape-matched keys into `wrapper.unet` via `strict=False`
3. Kaiming-reinitialise all 28 `time_emb_proj` layers + v-head + r/h_embed

**Pros:** No assumptions about embedding order, simple
**Cons:** Loses time-conditioning projections (but these will retrain quickly)

### 8.3 Option C: Architecture Match (Maximum Transfer)

1. Modify `MAISIUNetWrapper` to use `DiffusionModelUNetMaisi` instead of `DiffusionModelUNet`
2. Set `include_spacing_input=True`, `num_class_embeds=128` to match rflow exactly
3. Load all 435 keys with `strict=True`
4. At forward time, pass dummy spacing/class inputs (zeros)

**Pros:** Full weight transfer, no shape mismatches
**Cons:** Increases model size, adds unused conditioning pathways, may interfere with JVP

---

## 9. Implementation Considerations

### 9.1 Output Layer Handling

The rflow model outputs velocity `v = x_0 - eps`. Our model uses x-prediction (`x_hat`). The output conv's learned weights encode this mapping. Options:

1. **Reinitialise output conv** — safest, lets model learn fresh x-prediction mapping
2. **Keep and negate** — since `v_rflow = -(eps - x_0)`, negating the output conv weights would give the correct sign. But we use x-prediction, not u-prediction, so the relationship is more complex
3. **Keep as-is** — the output conv will rapidly adapt during training; the feature extraction backbone is the valuable part

**Recommendation:** Keep output conv weights as-is. The backbone features are the primary value; the output projection will adapt within a few hundred steps.

### 9.2 Differential Learning Rates

```python
# Pseudo-code for param group construction:
pretrained_params = []
new_params = []
for name, param in model.net.named_parameters():
    if name in loaded_keys:
        pretrained_params.append(param)
    else:
        new_params.append(param)

optimizer = AdamW([
    {"params": pretrained_params, "lr": lr * backbone_lr_factor},
    {"params": new_params, "lr": lr},
], weight_decay=weight_decay, betas=betas)
```

### 9.3 EMA Initialisation from Pretrained

Load pretrained weights **before** EMA construction:
```python
# In LatentMeanFlow.__init__():
if pretrained_ckpt_path:
    load_pretrained_weights(self.net, pretrained_ckpt_path)
self.ema = EMAModel(self.net, decay=ema_decay)  # Copies pretrained state
```

### 9.4 Parameter Budget

| Component | Params | Source |
|-----------|--------|--------|
| Shape-matched (direct load) | 176,731,268 | rflow checkpoint |
| time_emb_proj (slice or reinit) | 1,835,008 | 28 layers |
| r/h_embed (new, Kaiming) | ~131K | 4 keys |
| v-head (new, zero-init) | ~228K | 12 keys |
| **Total our model** | **178,566,276 + ~359K** | — |
| **Directly transferable** | **176,731,268** | **99.0% of inner UNet** |

---

## 10. Noise Scheduler Details (for Reference)

The rflow model was trained with:
```json
{
  "_target_": "monai.networks.schedulers.rectified_flow.RFlowScheduler",
  "num_train_timesteps": 1000,
  "use_discrete_timesteps": false,
  "use_timestep_transform": true,
  "sample_method": "uniform",
  "scale": 1.4
}
```

The `use_timestep_transform=True` with `scale=1.4` applies a resolution-dependent timestep transformation that shifts the sampling distribution based on spatial resolution. This is different from our logit-normal sampling but only affects training dynamics, not the learned weights' utility.

---

## 11. Summary: What Can Be Transferred

| What | Status | Action |
|------|--------|--------|
| `conv_in` (4→64) | Exact match | Load directly |
| `time_embed` MLP (64→256→256) | Exact match | Load directly (will adapt to our time convention) |
| Down/Up blocks: conv, norm, skip | Exact match (374 keys) | Load directly |
| Attention Q/K/V/O projections | Exact match (28 keys) | Load directly |
| `out` layer (GroupNorm + Conv3d) | Exact match | Load directly (or optionally reinit) |
| `time_emb_proj` (28 layers) | Shape mismatch (512→256) | Slice `[:, :256]` or reinit |
| `class_embedding` | Not in our model | Drop |
| `spacing_layer` | Not in our model | Drop |
| `r_embed` / `h_embed` | Not in rflow | Kaiming init (new) |
| `v_out` (v-head) | Not in rflow | Zero init (existing convention) |

**Bottom line:** 402 of 430 inner UNet keys (99.0% of params) transfer directly with exact shape match. The remaining 28 keys can be handled by slicing or reinitialisation.
