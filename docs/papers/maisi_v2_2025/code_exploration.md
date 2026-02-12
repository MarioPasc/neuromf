# MAISI VAE Code Exploration

**Source:** `src/external/NV-Generate-CTMR/`

## 1. VAE Class

- **Import:** `monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi`
- **Instantiation:** Via `define_instance(args, "autoencoder_def")` using MONAI's `ConfigParser`

**Constructor args** (from `configs/config_network_rflow.json`):

```json
{
    "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
    "spatial_dims": 3,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "num_channels": [64, 128, 256],
    "num_res_blocks": [2, 2, 2],
    "norm_num_groups": 32,
    "norm_eps": 1e-06,
    "attention_levels": [false, false, false],
    "with_encoder_nonlocal_attn": false,
    "with_decoder_nonlocal_attn": false,
    "use_checkpointing": false,
    "use_convtranspose": false,
    "norm_float16": true,
    "num_splits": 4,
    "dim_split": 1
}
```

Key: **No attention layers** (`attention_levels` all false, no nonlocal attention). This is why the checkpoint is only 80MB.

## 2. Weight Loading

**Location:** `scripts/inference.py:164-167`, `scripts/diff_model_create_training_data.py:219-222`

```python
autoencoder = define_instance(args, "autoencoder_def").to(device)
checkpoint_autoencoder = torch.load(args.trained_autoencoder_path)
if "unet_state_dict" in checkpoint_autoencoder.keys():
    checkpoint_autoencoder = checkpoint_autoencoder["unet_state_dict"]
autoencoder.load_state_dict(checkpoint_autoencoder)
```

The checkpoint may contain a raw state dict OR be wrapped in a `"unet_state_dict"` key.

## 3. scale_factor

**CRITICAL:** The scale_factor is stored in the **diffusion UNet checkpoint**, NOT the VAE checkpoint.

**Location:** `scripts/inference.py:170-172`

```python
checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
diffusion_unet.load_state_dict(checkpoint_diffusion_unet["unet_state_dict"], strict=False)
scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)
```

**Calculation** (`scripts/diff_model_train.py:182-203`):

```python
def calculate_scale_factor(train_loader, device, logger):
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    # Averaged across ranks in distributed training
    return scale_factor
```

Formula: `scale_factor = 1 / std(latents)` computed over the training set.

Example value from config: `1.0055984258651733` (mask generation).

## 4. Encode API

**Method:** `autoencoder.encode_stage_2_inputs(x)`

- **Input:** `x` shape `[B, 1, H, W, D]` (float32)
- **Output:** `z` shape `[B, 4, H/4, W/4, D/4]` — direct latent tensor (no posterior object)
- Example: `[1, 1, 128, 128, 128]` -> `[1, 4, 32, 32, 32]`

Usage with SlidingWindowInferer for large volumes (`scripts/diff_model_create_training_data.py:160-189`):

```python
with torch.amp.autocast("cuda"):
    inferer = SlidingWindowInferer(roi_size=[320, 320, 160], sw_batch_size=1, overlap=0.4)
    z = dynamic_infer(inferer, autoencoder.encode_stage_2_inputs, pt_nda)
```

For 128^3 volumes on 8GB VRAM, direct encoding without SlidingWindowInferer should work (with `num_splits=4`).

## 5. Decode API

**Method:** `autoencoder.decode_stage_2_outputs(z / scale_factor)`

**IMPORTANT:** Divide by `scale_factor` before decoding.

From `scripts/sample.py:58-69`:

```python
class ReconModel(torch.nn.Module):
    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
```

- **Input:** `z_scaled` shape `[B, 4, H/4, W/4, D/4]`
- **Output:** reconstructed image shape `[B, 1, H, W, D]`

## 6. Preprocessing

**MRI-specific** (`scripts/transforms.py:64-67`):

```python
ScaleIntensityRangePercentilesd(
    keys=image_keys,
    lower=0.0,
    upper=99.5,
    b_min=0.0,
    b_max=1.0,
    clip=False
)
```

Full VAE transform pipeline:
1. `LoadImaged` — load NIfTI
2. `EnsureChannelFirstd` — `[C, H, W, D]` format
3. `Orientationd(axcodes="RAS")`
4. `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0)`
5. `Spacingd` (optional) — resample to fixed spacing
6. `SpatialPadd` / `RandSpatialCropd` — crop/pad to target size
7. `EnsureTyped(dtype=float32)`

## 7. Memory Optimization: num_splits

```json
"num_splits": 4,
"dim_split": 1
```

- Splits the volume into 4 sub-patches along dimension 1 during encode/decode
- Processes sequentially to reduce peak VRAM usage
- Example: `[1, 1, 128, 128, 128]` processed as 4x `[1, 1, 128, 128, 32]`
- Override at inference time via `autoencoder_tp_num_splits` config key

## 8. 2.5D FID Computation

**Source:** `scripts/compute_fid_2-5d_ct.py`

**Function:** `get_features_2p5d` (lines 253-352)

Extracts 2D slices along three orthogonal planes:
- **XY-plane:** `torch.unbind(image, dim=-1)` — slices along D axis
- **YZ-plane:** `torch.unbind(image, dim=2)` — slices along H axis
- **ZX-plane:** `torch.unbind(image, dim=3)` — slices along W axis

Processing pipeline per plane:
1. Repeat single channel to 3 channels: `image.repeat(1, 3, 1, 1, 1)`
2. RGB->BGR: `image[:, [2, 1, 0], ...]`
3. Concatenate slices: `torch.cat(slices, dim=0)` -> `[N_slices, 3, H, W]`
4. Radimagenet intensity normalization (min-max to [0,1], subtract ImageNet means)
5. Feature extraction with `radimagenet_resnet50`
6. Spatial averaging: `mean([2, 3])` -> feature vector per slice

FID computation:

```python
from monai.metrics.fid import FIDMetric
fid = FIDMetric()
fid_xy = fid(synth_xy, real_xy)
fid_yz = fid(synth_yz, real_yz)
fid_zx = fid(synth_zx, real_zx)
fid_avg = (fid_xy + fid_yz + fid_zx) / 3.0
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/inference.py` | Main inference; loads autoencoder, diffusion UNet, ControlNet |
| `scripts/diff_model_train.py` | Training; calculates and saves `scale_factor` |
| `scripts/diff_model_create_training_data.py` | Encodes dataset with `encode_stage_2_inputs` |
| `scripts/sample.py` | `ReconModel` wrapper with `z / scale_factor` decode |
| `scripts/compute_fid_2-5d_ct.py` | 2.5D FID: XY/YZ/ZX slices, radimagenet features |
| `scripts/transforms.py` | MONAI transforms; MRI percentile normalization |
| `configs/config_network_rflow.json` | VAE architecture config |

## Key Parameters Summary

| Parameter | Value |
|-----------|-------|
| Spatial compression | 4x per axis (128^3 -> 32^3) |
| Channel expansion | 1 -> 4 latent channels |
| Total compression | 64x |
| Attention | None (no attention layers) |
| Memory splits | num_splits=4, dim_split=1 |
| Preprocessing | Percentile 0-99.5% -> [0, 1] |
| scale_factor location | Diffusion checkpoint, NOT VAE checkpoint |
| Latent shape | [B, 4, 32, 32, 32] for 128^3 input |
