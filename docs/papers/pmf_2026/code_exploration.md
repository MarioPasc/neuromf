# pMF (Progressive MeanFlow) Code Exploration

**Source:** `src/external/pmf/pmf.py`

## 1. x-Prediction (line 423)

```python
pred_x = z_t - t * u
```

- Reparameterization: predicted clean image from noisy state and average velocity
- Used only for perceptual loss computation (LPIPS + ConvNeXt), not as main loss target
- `z_t` is noisy image at time `t`, `u` is the predicted average velocity

## 2. Compound V (line 403)

```python
# JVP computation (lines 392-401)
def u_fn(z_t, t, r):
    return self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)

dtdt = jnp.ones_like(t)
dtdr = jnp.zeros_like(t)
u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)

# Compound velocity
V = u + (t - r) * jax.lax.stop_gradient(du_dt)
```

**Key difference from original MeanFlow:** Uses predicted `v` from auxiliary head as tangent vector in JVP (comment at line 399: "Different from original MeanFlow, we use predicted v in the jvp"). The JVP returns `has_aux=True` with `v` as auxiliary output.

`stop_gradient(du_dt)` prevents gradient flow through the derivative term.

## 3. Adaptive Weighting (lines 407-409)

```python
def adp_wt_fn(loss):
    adp_wt = (loss + self.norm_eps) ** self.norm_p
    return loss / jax.lax.stop_gradient(adp_wt)
```

Default: `norm_p=1.0`, `norm_eps=0.01` (from `configs/default.py:64-65`).

Applied to all loss components:
```python
loss_u = adp_wt_fn(jnp.sum((V - v_g) ** 2, axis=(1, 2, 3)))    # Line 412-413
loss_v = adp_wt_fn(jnp.sum((v - v_g) ** 2, axis=(1, 2, 3)))    # Line 416-417
aux_loss = adp_wt_fn(aux_loss_lpips) * lpips_lambda + \
           adp_wt_fn(aux_loss_convnext) * convnext_lambda        # Line 432
```

## 4. Perceptual Losses

**Initialization** (`utils/auxloss_util.py:81-96`):
- **LPIPS:** Pre-trained JAX port from `lpips_j` library, operates on 224x224 crops
- **ConvNeXt:** ConvNeXtV2-Base (facebook/convnextv2-base-22k-224), L2 distance on features

**Loss computation** (`utils/auxloss_util.py:99-124`):
```python
def auxloss_fn(model_images, gt_images, rng=None):
    # Paired random resized crop to 224x224 (both images cropped identically)
    model_images, gt_images = paired_random_resized_crop(rng, model_images, gt_images, out_size=224)

    lpips_dist = lpips_model.apply(lpips_params, model_images, gt_images)
    convnext_features_pred = convnext_head_model.apply(convnext_head_params, model_images)
    convnext_features_gt = convnext_head_model.apply(convnext_head_params, gt_images)
    class_dist = jnp.sum((convnext_features_pred - convnext_features_gt) ** 2, axis=-1)

    return lpips_dist, class_dist
```

**Time gating** (lines 425-432):
```python
pred_x = z_t - t * u
aux_loss_lpips, aux_loss_convnext = aux_fn(pred_x, x, rng)
mask = t.flatten() < self.perceptual_max_t    # Default 1.0 (always enabled)
aux_loss_lpips = jnp.where(mask, aux_loss_lpips, 0.0)
aux_loss_convnext = jnp.where(mask, aux_loss_convnext, 0.0)
```

When `perceptual_max_t < 1.0`, perceptual losses only apply for small `t` (less noisy samples).

## 5. Combined Loss (lines 354-448)

```python
def forward(self, images, labels, aux_fn=None):
    x = images.astype(self.dtype)

    # 1. Sample t, r with data_proportion constraint (default 0.5)
    t, r, fm_mask = self.sample_tr(bz)

    # 2. Interpolation and velocity
    e = jax.random.normal(rng, x.shape) * self.noise_scale
    z_t = (1 - t) * x + t * e
    v_t = (z_t - x) / jnp.clip(t, 0.05, 1.0)

    # 3. CFG-guided velocity
    v_g, v_c = self.guidance_fn(v_t, z_t, t, r, labels, fm_mask, omega, t_min, t_max)

    # 4. JVP with compound V
    u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
    V = u + (t - r) * jax.lax.stop_gradient(du_dt)

    # 5. Main losses
    loss_u = adp_wt_fn(jnp.sum((V - v_g) ** 2, axis=(1, 2, 3)))
    loss_v = adp_wt_fn(jnp.sum((v - v_g) ** 2, axis=(1, 2, 3)))

    # 6. Perceptual losses on predicted x_0
    pred_x = z_t - t * u
    aux_loss = adp_wt_fn(lpips) * lpips_lambda + adp_wt_fn(convnext) * convnext_lambda

    # 7. Total loss
    loss = (loss_u + loss_v + aux_loss).mean()
    return loss, dict_losses
```

## 6. Novel Techniques (not in original MeanFlow)

**A. Dual-head architecture** (`models/mit.py:264-270, 365-366`):
- Shared backbone with separate u-head and v-head
- Learns both average velocity `u` and instantaneous velocity `v` simultaneously
- Enables x-prediction without extra forward passes

**B. JVP with predicted v** (`pmf.py:399-403`):
- Uses predicted `v` (from auxiliary head) as tangent vector instead of ground-truth velocity
- Improves coupling between u and v heads

**C. Perceptual auxiliary losses on x_0** (`pmf.py:419-432`):
- LPIPS + ConvNeXt feature distance on `pred_x = z_t - t * u`
- Time-gated via `perceptual_max_t` threshold
- Per-element adaptive weighting

**D. In-context conditioning tokens** (`models/mit.py:152-224`):
- Learnable tokens: `time_tokens` (4), `class_tokens` (8), `omega_tokens` (4), `t_min/t_max_tokens` (2 each)
- Unified representation for multiple conditioning signals

**E. h-only time conditioning** (`models/mit.py:354-356`):
- Conditions on `h = t - r` only (not absolute `t`)
- "We don't explicitly condition on time t, only on h = t - r following https://arxiv.org/abs/2502.13129"

**F. CFG with learned interval** (`pmf.py:318-348`):
- CFG scale `w` sampled from power distribution during training
- CFG interval `[t_min, t_max]` sampled during training for test-time control

## Hyperparameters (configs/default.py)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| norm_p | 1.0 | Adaptive weighting power |
| norm_eps | 0.01 | Adaptive weighting epsilon |
| lpips_lambda | 1.0 | LPIPS loss weight |
| convnext_lambda | 0.0 | ConvNeXt loss weight (disabled by default) |
| perceptual_max_t | 1.0 | Time threshold for perceptual loss |
| data_proportion | 0.5 | Fraction for flow matching (t=r) |
| cfg_max | 7.0 | Maximum CFG scale during training |
| P_mean, P_std | -0.4, 1.0 | Logit-normal distribution |

## File Structure

| File | Purpose |
|------|---------|
| `pmf.py` | Core pixelMeanFlow class (452 lines) |
| `train.py` | Training loop with EMA & FID (344 lines) |
| `main.py` | CLI entry point |
| `models/mit.py` | MiT architecture with dual heads (474 lines) |
| `models/convnext.py` | ConvNeXt-Base feature extractor (279 lines) |
| `utils/auxloss_util.py` | LPIPS + ConvNeXt loss (127 lines) |
| `configs/default.py` | Hyperparameters (93 lines) |

# Official pMF Implementation Analysis
## Critical Findings for vMF Project

**Source:** https://github.com/Lyy-iiis/pMF (Yiyang Lu, first author)
**Legitimacy:** Confirmed official. Author "Lyy-iiis" matches Yiyang Lu.
Repository links arXiv:2601.22158, provides pre-trained checkpoints with
reproducible FID (3.11 vs paper's 3.12 for B/16), MIT license, credits
Kaiming He's group. Repo created 2026-02-04, 116 stars.

---

## FINDING 1: The Model Outputs u Directly (x→u Conversion is INSIDE the Network)

This is the single most consequential architectural detail. In `models/mit.py:371-377`:

```python
u = self.unpatchify(self.u_final_layer(u_tokens))
v = self.unpatchify(self.v_final_layer(v_tokens))

t = t.reshape((-1, 1, 1, 1))

u = (x - u) / jnp.clip(t, 0.05, 1.0)   # <-- x→u conversion INSIDE model
v = (x - v) / jnp.clip(t, 0.05, 1.0)   # <-- same for v-head
```

**What this means:**
- The final layers (`u_final_layer`, `v_final_layer`) output x-predictions
  (denoised images).
- The conversion u = (z_t - x_pred) / t happens INSIDE the model's own
  forward pass.
- The model returns `(u, v)` to the caller, NOT `(x_pred, v_pred)`.
- Therefore, the JVP in `pmf.py` differentiates through a function that
  already returns u, and the 1/t factor is inside the differentiated
  function (as it should be for correctness).

**Critical detail:** The division uses `jnp.clip(t, 0.05, 1.0)`, not
`t.clamp(min=1e-5)`. This hard-clips t at 0.05, which:
- Completely eliminates the 1/t gradient explosion for t < 0.05
- At t=0.05, the maximum amplification is 20× (vs 10,000× at t=0.001
  with our 1e-5 clamp)
- With logit-normal(0.8, 0.8), P(t < 0.05) is negligible anyway, but
  the clip provides a hard safety floor

**Impact on vMF:** Your wrapper's `t_safe = t.clamp(min=1e-5)` is far too
permissive. Change to `t.clamp(min=0.05)` to match. This alone may fix
a significant portion of the x-prediction instability.

---

## FINDING 2: Dedicated v-Head for JVP Tangent (NOT model velocity at r=t)

This is the biggest surprise relative to what the paper describes and
what your wrapper implements. In `pmf.py:392-400`:

```python
# Warped u-function for jvp computation
def u_fn(z_t, t, r):
    return self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)

dtdt = jnp.ones_like(t)
dtdr = jnp.zeros_like(t)

# Different from original MeanFlow, we use predicted v in the jvp
u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
```

**The tangent for z_t is `v_c`** — the **predicted conditioned velocity
from a dedicated auxiliary v-head**, not:
- NOT the true velocity v_t = ε - x (as in the reference MeanFlow code)
- NOT u_fn(z_t, t, t) (model's own u evaluated at r=t, as per Algorithm 1)

The v-head is a separate branch of 8 transformer layers (same depth as
the u-head) that shares the first `depth - 8` layers with the u-head.
It is trained with its own loss:

```python
loss_v = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
loss_v = adp_wt_fn(loss_v)
```

The comment at line 399 is explicit: **"Different from original MeanFlow,
we use predicted v in the jvp"**

**Why this matters:**
1. The v-head provides a clean, gradually-improving velocity estimate
   for the JVP tangent, without requiring an extra forward pass through
   the u-head.
2. The v-head is trained against v_g (guided velocity), so it learns to
   approximate the target velocity field directly.
3. `has_aux=True` in `jax.jvp` means the JVP only differentiates through
   the u-output; v is returned as an auxiliary (not differentiated).
4. This avoids the fundamental chicken-and-egg problem: using the model's
   own u at r=t means the tangent is garbage early in training. Using a
   dedicated v-head that is trained in parallel means the tangent improves
   steadily alongside the u-head.

For the toy problem, option (b) is the pragmatic choice since there is no
auxiliary head. For the production ViT3D, you should implement option (a)
to match the official code.

---

## FINDING 3: Adaptive Weighting is ALWAYS On (norm_p=1.0)

In `pmf.py:407-409`:

```python
def adp_wt_fn(loss):
    adp_wt = (loss + self.norm_eps) ** self.norm_p
    return loss / jax.lax.stop_gradient(adp_wt)
```

Default config: `norm_p=1.0, norm_eps=0.01`. With p=1:

    adp_wt_fn(L) = L / (L + 0.01)

For L >> 0.01, this → 1.0 (saturates).
For L ≈ 0.01, this → 0.5 (downweighted).
For L << 0.01, this → L/0.01 (strongly downweighted).

This is NOT optional — it's always active. It normalizes per-sample
loss to approximately 1, making every sample contribute roughly equally
regardless of loss magnitude. This automatically handles the 1/t
gradient amplification: high-loss small-t samples are downweighted.

**Impact on vMF:** Your wrapper lacks adaptive weighting entirely. The
agent's plan to add it (Finding 2b in the Module 0C v2 doc) is correct,
but the formula is slightly different from the reference MeanFlow code:
- Official pMF: `L / (L + ε)^p` (divisive, always active)
- Reference MeanFlow: `(1 / (L + ε)^p) * L` (multiplicative, optional)
Numerically equivalent, but the official pMF always uses p=1.0, while
the reference defaults to p=1.0 with "adaptive" weighting mode.

---

## FINDING 4: Loss Reduction is SUM-then-MEAN (Confirmed)

In `pmf.py:412-438`:

```python
loss_u = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))   # SUM over spatial
loss_u = adp_wt_fn(loss_u)
...
loss = loss.mean()                                     # MEAN over batch
```

Your wrapper defaults to `dim_reduction="mean"`, which is incorrect.
The agent's Finding 2a is confirmed TRUE.

---

## FINDING 5: Time Sampling — Independent t and r

In `pmf.py:137-162`, the sampling is different from the reference:

```python
t = self.logit_normal_dist(bz)   # independent sample
r = self.logit_normal_dist(bz)   # independent sample
...
data_size = int(bz * self.data_proportion)  # 50% by default
fm_mask = jnp.arange(bz) < data_size
r = jnp.where(fm_mask, t, r)               # first 50%: r=t
t, r = jnp.maximum(t, r), jnp.minimum(t, r) # ensure t >= r
```

**Key differences from your wrapper's SILoss delegation:**
- t and r are sampled independently (not sorted pairs from 2 samples)
- 50% of samples have r=t (flow matching), not 25%
- No sorting of paired samples — just independent draws + max/min swap

**Production config (B/16):** P_mean=0.8, P_std=0.8 (not -0.4, 1.0)

---

## FINDING 6: Architecture Has a Bottleneck (128 PCA channels for B/L)

In `models/mit.py:175-180`:

```python
self.x_embedder = BottleneckPatchEmbedder(
    self.input_size,
    self.patch_size,
    128 if self.hidden_size <= 1024 else 256,  # PCA bottleneck
    self.in_channels,
    self.hidden_size,
)
```

Bottleneck=128 for B (768) and L (1024), bottleneck=256 for H (1280).
This confirms the PCA bottleneck from the paper's Table 8.

---

## FINDING 7: No adaLN-Zero — Uses In-Context Conditioning with Vec Gates

The MiT architecture does NOT use adaLN-Zero (modulate-shift-scale per
block). Instead:
- Uses learnable tokens prepended to the sequence for conditioning
  (class tokens, time tokens, cfg tokens, interval tokens)
- Uses RoPE attention (not absolute position embeddings)
- Uses SwiGLU MLP (not GELU)
- Uses vector gates (attn_scale, mlp_scale) per block
- References arxiv:2502.13129 for the h-only time conditioning
- FinalLayer uses RMSNorm + zero-init linear (not adaLN)

**Impact on vMF Module 1:** Your ViT3D architecture plan assumes adaLN-Zero
conditioning. The official pMF uses a fundamentally different conditioning
mechanism. For scientific validity, you may want to replicate the official
conditioning or explicitly acknowledge the architectural difference.

---

## FINDING 8: CFG Is Trained Into the Model

The official implementation trains with CFG built into the loss:
- CFG omega sampled from a power distribution at training time
- CFG interval [t_min, t_max] sampled at training time
- v_g (guided velocity target) incorporates CFG
- Labels are randomly dropped for classifier-free guidance

This is NOT a separate inference trick — CFG is integral to training.
B/16 eval config: omega=8.5, t_min=0.1, t_max=0.7.

---

## FINDING 9: Noise Scale = 1.0 for 256×256

From B/16 config: `noise_scale: 1.0`. The noise is scaled before
interpolation:

```python
e = jax.random.normal(...) * self.noise_scale
z_t = (1 - t) * x + t * e
```

This confirms that noise_scale = image_size / 256 from the paper's
Table 8 (i.e., 1.0 for 256, 2.0 for 512).

---

