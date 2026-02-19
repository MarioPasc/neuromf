# iMeanFlow (iMF) Code Exploration

> Source: `src/external/imeanflow/` — JAX/Flax reference implementation
> Paper: "Improved Mean Flows for One-Step Generation" (2025)
> Explored: 2026-02-19

## 1. Architecture: Dual-Head MiT (MeanFlow improved Transformer)

**File:** `models/mit.py` — `MiT(nn.Module)`

The MiT has a shared backbone and two separate heads (u-head and v-head):

```
Input patches → Shared blocks (depth - aux_head_depth) → Fork
                                                          ├─ u-head blocks (aux_head_depth) → u_final_layer → u output
                                                          └─ v-head blocks (aux_head_depth) → v_final_layer → v output
```

**Key architectural details:**
- **Shared backbone:** `depth - aux_head_depth` TransformerBlocks (e.g., 12 - 8 = 4 shared blocks for MiT-B)
- **Dual heads:** 8 TransformerBlocks each for u-head and v-head (`aux_head_depth=8`)
- **v-head disabled at eval:** `v_heads` list is empty when `eval=True` — zero inference overhead
- **RoPE attention** with QK-norm (RMSNorm per head) instead of sinusoidal position embeddings
- **SwiGLU MLP** with zero-initialized vector gates on residuals (both attn_scale and mlp_scale)
- **In-context conditioning:** all conditioning prepended as learnable tokens in the sequence

**Conditioning (all via prepended tokens, no AdaGN):**
- `h = t - r` via `h_embedder` (TimestepEmbedder) — NOT separate t and r
- `omega` (CFG scale) via `omega_embedder`
- `t_min`, `t_max` (CFG interval) via separate embedders
- `y` (class label) via `y_embedder`
- **Critical:** `t` is NOT explicitly conditioned on — only `h = t - r` (line 343-344: "We don't explicitly condition on time t, only on h = t - r")

**Model configurations:**
| Config | Depth | Hidden | Heads | Shared | Head depth | Params |
|--------|-------|--------|-------|--------|------------|--------|
| MiT-B  | 12    | 768    | 12    | 4      | 8          | ~130M  |
| MiT-M  | 24    | 768    | 12    | 16     | 8          | ~260M  |
| MiT-L  | 32    | 1024   | 16    | 24     | 8          | ~500M  |
| MiT-XL | 48    | 1024   | 16    | 40     | 8          | ~750M  |

## 2. Loss Formulation

**File:** `imf.py` — `iMeanFlow.forward()`

### 2.1 u-prediction (NOT x-prediction)

The model directly predicts `u` (average velocity) and `v` (instantaneous velocity) — there is **no x-prediction reparameterization** and **no 1/t division** anywhere in the code.

```python
u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
```

The `u_fn` returns `(u, v)` where `v` is the auxiliary output (`has_aux=True`). The JVP is computed with respect to the u-output only.

### 2.2 Tangent direction: v_c (conditioned velocity from v-head)

Unlike the original MeanFlow (which uses ground truth `e - x` as tangent) and our iMF formulation (which uses `v_tilde = u(z_t, t, t)`), the iMF reference uses `v_c` — the **conditioned** instantaneous velocity from the v-head with CFG applied:

```python
v_g, v_c = self.guidance_fn(v_t, z_t, t, r, y, fm_mask, w, t_min, t_max)
# ...
u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
```

This is a key difference: the tangent comes from the model's v-head (with CFG), not from `u(z_t, t, t)`.

### 2.3 Dual loss: loss_u + loss_v

```python
# u-loss (compound velocity vs guided target)
loss_u = sum((V - v_g)^2, axis=(1,2,3))
loss_u = adp_wt_fn(loss_u)

# v-loss (auxiliary v-head vs guided target)
loss_v = sum((v - v_g)^2, axis=(1,2,3))
loss_v = adp_wt_fn(loss_v)

loss = loss_u + loss_v  # equal weight, both adaptively normalised
```

**Note:** `v_g` (the CFG-guided velocity) is stop-gradiented before the loss — it's a target, not a prediction.

### 2.4 Adaptive weighting

```python
def adp_wt_fn(loss):
    adp_wt = (loss + norm_eps) ** norm_p
    return loss / stop_gradient(adp_wt)
```

Applied independently to both `loss_u` and `loss_v`. With default `norm_eps=0.01, norm_p=1.0`, this normalises each loss term to approximately 1.0.

## 3. Time Sampling

**File:** `imf.py` — `iMeanFlow.sample_tr()`

```python
t = logit_normal(P_mean=-0.4, P_std=1.0)
r = logit_normal(P_mean=-0.4, P_std=1.0)
t, r = max(t, r), min(t, r)

# FM samples: set r = t for data_proportion fraction
fm_mask = arange(bz) < int(bz * data_proportion)
r = where(fm_mask, t, r)
```

**Key:** `data_proportion=0.5` means 50% FM samples (r=t) and 50% MF samples (r<t). This is different from PyTorch MeanFlow's default of 0.75.

## 4. CFG-Aware Training

The iMF implementation trains with classifier-free guidance (CFG) built into the loss:

1. **CFG scale sampling:** `omega` drawn from power distribution (configurable via `cfg_beta`)
2. **CFG interval sampling:** `[t_min, t_max]` uniform random; FM samples get `[0, 1]` (full guidance)
3. **Class dropout:** 10% of samples get null class label (standard CFG training)
4. **Guided velocity:** `v_g = v_t + (1 - 1/omega) * (v_c - v_u)` where v_c/v_u are conditioned/unconditioned

This means the model learns to be guided at training time, not just at inference — which is important for the iMF paper's results but not relevant for our unconditional brain MRI generation.

## 5. Sampling

**File:** `imf.py` — `iMeanFlow.sample_one_step()`

```python
u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]  # [0] = u output only
z_next = z_t - (t - r) * u
```

Multi-step: Euler integration from t=1 to t=0 via `jax.lax.fori_loop`. Only the u-head is used — v-head is completely disabled at eval time.

## 6. Default Hyperparameters

**File:** `configs/default.py`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `learning_rate` | 1e-4 | Same as ours |
| `batch_size` | 256 | Global (ImageNet); ours is 128 effective |
| `num_epochs` | 1000 | Much longer; we use 300 |
| `adam_b2` | 0.95 | Matches our setting |
| `ema_val` | 0.9999 | Matches our setting |
| `lr_schedule` | "warmup_const" | Warmup then constant; ours is cosine |
| `warmup_epochs` | 0 | No warmup; matches our setting |
| `data_proportion` | 0.5 | 50% FM; we had 0.75, now changed to 0.5 |
| `norm_p` | 1.0 | Matches our setting |
| `norm_eps` | 0.01 | We use 1.0 (PyTorch CLI default) |
| `P_mean` | -0.4 | Logit-normal mean; matches ours |
| `P_std` | 1.0 | Logit-normal std; matches ours |
| `weight_decay` | 0 | Not in config (AdamW not used); we changed to 0 |
| `class_dropout_prob` | 0.1 | CFG training; N/A for us (unconditional) |

## 7. Key Differences from Our Implementation

| Aspect | Our NeuroMF | iMF Reference |
|--------|-------------|---------------|
| **Prediction type** | u-prediction (changed from x) | u-prediction (always) |
| **JVP method** | FD (h=1e-3) with grad ckpt | Exact (`jax.jvp`) |
| **Architecture** | MONAI DiffusionModelUNet (3D) | MiT Transformer (2D) |
| **v-head** | None | Dual-head (8 blocks) with aux loss |
| **Tangent** | `v_tilde = u(z_t, t, t)` | `v_c` from v-head with CFG |
| **Loss terms** | `loss_u` only | `loss_u + loss_v` |
| **Time conditioning** | Separate (r, t) | `h = t - r` only |
| **CFG** | None (unconditional) | Full CFG-aware training |
| **data_proportion** | 0.5 (now aligned) | 0.5 |
| **norm_eps** | 1.0 | 0.01 |
| **Spatial dims** | 3D (48^3 latents) | 2D (32x32 latents) |

## 8. Lessons for NeuroMF

### 8.1 Critical: u-prediction for FD-JVP compatibility

The iMF reference uses u-prediction throughout. With x-prediction, `u = (z_t - x̂)/t` introduces a 1/t factor that causes FD-JVP to explode (the v2_baseline failure). Since we must use FD-JVP (MONAI UNet + gradient checkpointing + flash attention are incompatible with `torch.func.jvp`), u-prediction is required.

### 8.2 Future: Auxiliary v-head

The v-head provides a better tangent for JVP and adds a supervision signal. The iMF paper reports +3.27 FID improvement from the v-head. For Phase 6 ablations, consider adding a simple v-head (double output channels to 8, split into u[0:4] and v[4:8]).

### 8.3 Future: h-conditioning

Conditioning on `h = t - r` instead of separate `(r, t)` is simpler and reduces the embedding dimensionality. The iMF paper shows this works well. Could be tested as a Phase 6 ablation.

### 8.4 norm_eps sweep

The iMF default is 0.01, while PyTorch MeanFlow CLI overrides to 1.0. These produce very different adaptive weighting behavior. Worth ablating in Phase 6.
