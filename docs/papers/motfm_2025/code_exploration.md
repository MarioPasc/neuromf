# MOTFM Code Exploration

**Source:** `src/external/MOTFM/`

## 1. Inference/Sampling (inferer.py)

**ODE Solver** (line 95-103):
- Default method: `"midpoint"` (also supports `"rk4"`, `"euler"`)
- Time grid: `torch.linspace(0, 1, time_points)` (configurable via `solver_args`)
- Uses `flow_matching.solver.ODESolver` from Meta's flow_matching package

**Sampling pipeline** (lines 219-226):
```python
final_imgs = sample_batch(
    model=model,
    solver_config=solver_config,  # method, time_points, step_size
    batch=batch,
    device=device,
    class_conditioning=class_conditioning,
    mask_conditioning=mask_conditioning,
)
```

Default: `--num_inference_steps 5` (line 135).

## 2. Model Wrapper (utils/utils_fm.py)

**`MergedModel` class** (lines 16-80):
```python
class MergedModel(nn.Module):
    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        self.unet = unet           # MONAI generative DiffusionModelUNet
        self.controlnet = controlnet  # Optional ControlNet for mask conditioning
        self.max_timestep = max_timestep
```

**Time handling** (lines 49-55):
```python
# Continuous [0, 1] -> Discrete [0, max_timestep-1]
t = t * (self.max_timestep - 1)
t = t.floor().long()
```

**ControlNet conditioning** (lines 62-75):
- ControlNet receives: `x`, `timesteps`, `controlnet_cond=masks`, `context=cond`
- Returns: `down_block_res_samples`, `mid_block_res_sample`
- UNet receives: cross-attention context + residual connections from ControlNet

**`build_model()`** (lines 83-127):
- Instantiates `DiffusionModelUNet` from `generative.networks.nets`
- Conditionally creates `ControlNet` initialized from UNet weights

## 3. Training (trainer.py)

**Loss** (lines 178-208):
```python
def _compute_loss(self, batch):
    sample_info = self.path.sample(t=t, x_0=x_0, x_1=im_batch)  # Affine OT path
    v_pred = self.model(x=sample_info.x_t, t=sample_info.t, masks=mask_batch, cond=class_batch)
    return F.mse_loss(v_pred, sample_info.dx_t)
```

- **Path:** `AffineProbPath(scheduler=CondOTScheduler())` from `flow_matching` package
- **Loss:** Simple MSE on velocity predictions vs OT-scheduled velocity targets
- **Framework:** PyTorch Lightning

**Training config:**
- Optimizer: Adam with configurable LR (default 0.0001)
- Precision: auto-detects bf16/fp16 (lines 322-329)
- Checkpointing: top-3 + latest checkpoint (lines 308-317)
- Class balancing: optional WeightedRandomSampler (lines 108-137)

## 4. File Map

| File | Lines | Purpose |
|------|-------|---------|
| `trainer.py` | 361 | PyTorch Lightning module: training, loss, data |
| `inferer.py` | 301 | Inference: checkpoint loading, batch sampling |
| `utils/utils_fm.py` | 335 | MergedModel wrapper, solver, validation sampling |
| `utils/general_utils.py` | 481 | Data loading, normalization, visualization |
| `configs/default.yaml` | 89 | 2D medical imaging baseline |
| `configs/config_3D.yaml` | 89 | 3D medical imaging config |
| `configs/mask_class_conditioning.yaml` | 49 | Joint mask + class conditioning |
| `configs/class_conditioning.yaml` | 48 | Class-only conditioning |
| `configs/unconditional.yaml` | 45 | Unconditional generation |

## 5. Config Structure

```yaml
model_args:
  spatial_dims: 2  # or 3
  in_channels: 1
  out_channels: 1
  num_res_blocks: [2, 2, 2, 2, 2]
  num_channels: [32, 64, 128, 256, 512]
  attention_levels: [False, False, ...]
  norm_num_groups: 32
  use_flash_attention: true
  with_conditioning: True/False
  mask_conditioning: True/False

solver_args:
  method: "midpoint"  # midpoint / rk4 / euler
  step_size: 0.1
  time_points: 10

train_args:
  num_epochs: 200
  batch_size: 1
  lr: 0.0001
  gradient_accumulation_steps: 8
  precision: "bf16-mixed"
```

## 6. FID Computation

**MOTFM does NOT implement FID internally.** Inference outputs raw `.pkl` files with generated samples. FID must be computed externally — use NV-Generate-CTMR's `compute_fid_2-5d_ct.py` (see `docs/papers/maisi_2024/code_exploration.md`).

## Key Dependencies

- `flow_matching`: ODE solver, `AffineProbPath`, `CondOTScheduler`
- `monai_generative==0.2.3`: `DiffusionModelUNet`, `ControlNet`
- `pytorch_lightning`: Training framework

## Key Algorithms

1. **Velocity matching:** Train UNet to predict `v_θ(x_t, t) ≈ dx_t` where `(x_t, dx_t)` from affine OT path
2. **Inference:** ODE solver with `x_init ~ N(0, I)`, integrated from t=0 to t=1
3. **ControlNet:** Mask encoder produces residuals added to UNet features at multiple scales
