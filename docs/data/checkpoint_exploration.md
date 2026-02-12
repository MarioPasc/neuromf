# MAISI Checkpoint Exploration

**Location:** `/media/mpascual/Sandisk2TB/research/neuromf/checkpoints/NV-Generate-MR/models/`
**Explored:** 2026-02-12

## autoencoder_v2.pt (VAE)

| Property | Value |
|----------|-------|
| File size | 80 MB |
| Type | dict |
| Top-level keys | `epoch`, `unet_state_dict`, `epoch_finished` |
| State dict wrapper | YES — must extract `ckpt["unet_state_dict"]` |
| State dict entries | 130 |
| Total parameters | 20,944,897 (~21M) |

### Key Structure

```
encoder.blocks.{N}.conv.conv.{weight,bias}
encoder.blocks.{N}.norm{1,2}.{weight,bias}
encoder.blocks.{N}.conv{1,2}.conv.conv.{weight,bias}
...
decoder.blocks.{N}...
quant_conv_mu.conv.{weight,bias}
quant_conv_log_sigma.conv.{weight,bias}
post_quant_conv.conv.{weight,bias}
```

### Loading Pattern

```python
checkpoint = torch.load("autoencoder_v2.pt", map_location=device, weights_only=False)
state_dict = checkpoint["unet_state_dict"]  # Must unwrap!
autoencoder.load_state_dict(state_dict)
```

## diff_unet_3d_rflow-mr.pt (Diffusion UNet)

| Property | Value |
|----------|-------|
| File size | 2.1 GB |
| Type | dict |
| Top-level keys | `epoch`, `loss`, `num_train_timesteps`, **`scale_factor`**, `unet_state_dict`, `epoch_finished`, `optimizer_state_dict`, `scheduler_state_dict` |
| UNet state dict entries | 435 |
| UNet total parameters | 180,500,868 (~180M) |

### scale_factor

| Property | Value |
|----------|-------|
| **Value** | **0.96240234375** |
| Type | `monai.data.meta_tensor.MetaTensor` (scalar) |
| Location | Top-level key in diffusion checkpoint |

**CRITICAL for Phase 0:** This value must be extracted and used in the VAE wrapper for correct encode/decode. Formula: `z_decoded = autoencoder.decode_stage_2_outputs(z / scale_factor)`.

### Extracting scale_factor

```python
diff_ckpt = torch.load("diff_unet_3d_rflow-mr.pt", map_location="cpu", weights_only=False)
scale_factor = diff_ckpt["scale_factor"]  # tensor(0.9624)
```

Note: We only need the `scale_factor` from this checkpoint. The UNet itself (180M params) is NOT used in NeuroMF — we train our own MeanFlow model.
