# MeanFlow (PyTorch) Code Exploration

**Source:** `src/external/MeanFlow-PyTorch/meanflow.py`

## 1. PyTorch JVP (lines 58-62)

```python
assert jvp_fn in ['func', 'autograd'], "jvp_fn must be 'func' or 'autograd'"
if jvp_fn == 'func':
    self.jvp_fn = torch.func.jvp       # Preferred: functional API
elif jvp_fn == 'autograd':
    self.jvp_fn = partial(torch.autograd.functional.jvp, create_graph=True)  # Fallback
```

- **Default:** `torch.func.jvp` (set via `--jvp-fn func` in train_meanflow.py:451)
- **CRITICAL:** JVP is incompatible with FlashAttention — must use standard attention (README.md:88)

Usage at line 158:
```python
u, du_dt = self.jvp_fn(u_fn, (z_t, t, r), (v_g, dtdt, dtdr))
```

## 2. Class Design

- **Standalone class** — NOT `nn.Module`
- **Model passed as argument** to `__call__` and methods (not stored in constructor)
- Acts as a **loss calculator**: instantiated once, called with different models

```python
# Instantiation (train_meanflow.py:188-204)
meanflow_fn = MeanFlow(noise_dist='logit_normal', P_mean=-0.4, ...)

# Usage (train_meanflow.py:315)
loss, proj_loss, v_loss = meanflow_fn(model, x, labels=labels, zs=zs)
```

Model interface (line 87-89):
```python
def u_fn(self, model, x, t, h, y, train=True):
    bz = x.shape[0]
    return model(x, t.reshape(bz), h.reshape(bz), y, train=train)
```

Model receives both `t` (current time) and `h = t - r` (time difference) separately.

## 3. Key Differences from JAX Version

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| Gradient stopping | `.detach()` (lines 163, 170, 175) | `jax.lax.stop_gradient()` |
| Clamping | `torch.clamp(val, min=a, max=b)` | `jnp.clip(val, a, b)` |
| Conditionals | `torch.where(mask, a, b)` | `jnp.where(mask, a, b)` |
| Min/Max | `torch.maximum()` / `torch.minimum()` | `jnp.maximum()` / `jnp.minimum()` |
| Reduction | `torch.sum(x, dim=(1,2,3))` | `jnp.sum(x, axis=(1,2,3))` |
| JVP | `torch.func.jvp` or `torch.autograd.functional.jvp` | `jax.jvp` |

## 4. File Map

| File | Purpose |
|------|---------|
| `meanflow.py` | Core MeanFlow class (256 lines) — loss + sampling |
| `train_meanflow.py` | Training loop with Accelerate (478 lines) |
| `generate_meanflow.py` | Distributed sampling (219 lines) |
| `utils.py` | Encoder loading helpers (215 lines) |
| `dataset.py` | CustomDataset class |
| `evaluator.py` | FID/precision-recall metrics |
| `models/sit_meanflow.py` | SiT architecture with adaLN-Zero (399 lines) |
| `models/mmdit.py` | Alternative multimodal DiT |

## 5. Loss Computation (lines 125-188)

```python
def __call__(self, model, imgs, labels, zs=None, train=True):
    bz = imgs.shape[0]
    x = imgs.to(dtype=self.dtype)

    # Noise and interpolation
    t, r = self.sample_tr(bz, device)
    e = torch.randn_like(x)
    z_t = (1 - t) * x + t * e
    v = e - x

    # Guided velocity (CFG)
    v_g = self.guidance_fn(model, v, z_t, t, labels, train=False) if self.guidance_eq == "cfg" else v

    # Conditional dropout
    y_inp, v_g = self.cond_drop(v, v_g, labels)

    # JVP
    def u_fn(z_t, t, r):
        return self.u_fn(model, z_t, t, t - r, y=y_inp, train=train)

    dtdt = torch.ones_like(t)
    dtdr = torch.zeros_like(r)
    u, du_dt = self.jvp_fn(u_fn, (z_t, t, r), (v_g, dtdt, dtdr))

    # Loss
    u_tgt = v_g - torch.clamp(t - r, min=0.0, max=1.0) * du_dt
    u_tgt = u_tgt.detach()

    denoising_loss = (u - u_tgt) ** 2
    denoising_loss = torch.sum(denoising_loss, dim=(1, 2, 3))

    # Adaptive weighting
    adp_wt = (denoising_loss + self.norm_eps) ** self.norm_p
    denoising_loss = denoising_loss / adp_wt.detach()
    denoising_loss = denoising_loss.mean()

    v_loss = torch.sum((u - v) ** 2, dim=(1, 2, 3)).mean().detach()

    return denoising_loss, proj_loss, v_loss
```

## 6. Sampling Code

**1-NFE** (lines 196-203):
```python
def solver_step(self, model, z_t, t, r, labels):
    u = self.u_fn(model, z_t, t=t, h=(t - r), y=labels)
    return z_t - (t - r).view(-1, 1, 1, 1) * u

def sampling_schedule(self):
    return torch.tensor([1.0, 0.0])  # noise -> data
```

**Multi-step** (generate_meanflow.py:240-256):
```python
@torch.no_grad()
def mean_flow_sampler(model, mean_flow, latents, labels, num_steps=1):
    model.eval()
    x_next = latents
    t_steps = mean_flow.sampling_schedule().to(device)
    for i in range(num_steps):
        x_next = mean_flow.sample_one_step(model, x_next, labels, i, t_steps)
    return x_next
```

## Key Implementation Notes

1. **JVP is central** — entire training depends on `du/dt`. FlashAttention disabled.
2. **Dual time parameterization:** Model takes `(t, h=t-r)`, not just `t`.
3. **Three `.detach()` calls:** target (line 163), adaptive weight (line 170), v_loss (line 175).
4. **Not nn.Module:** Plug-and-play loss calculator, model passed as argument.
5. **Sampling is deterministic:** `@torch.no_grad()`, pure ODE (no stochasticity).

## Tensor Shapes (batch_size=256, 2D latent)

| Tensor | Shape |
|--------|-------|
| imgs (input) | `(256, 4, 32, 32)` |
| e (noise) | `(256, 4, 32, 32)` |
| z_t (interpolated) | `(256, 4, 32, 32)` |
| t, r (times) | `(256, 1, 1, 1)` |
| u (prediction) | `(256, 4, 32, 32)` |
| du_dt (JVP) | `(256, 4, 32, 32)` |
| denoising_loss | `(256,)` -> scalar |
