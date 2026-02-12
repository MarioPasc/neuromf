# MeanFlow (JAX) Code Exploration

**Source:** `src/external/MeanFlow/meanflow.py`

## 1. JVP Loss Computation (lines 226-236)

```python
def u_fn(z_t, t, r):
    return self.u_fn(z_t, t, t - r, y=y_inp, train=train)

dt_dt = jnp.ones_like(t)
dr_dt = jnp.zeros_like(t)
u, du_dt = jax.jvp(u_fn, (z_t, t, r), (v_g, dt_dt, dr_dt))

# Target with clipping
u_tgt = v_g - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
u_tgt = jax.lax.stop_gradient(u_tgt)
```

- **Primals:** `(z_t, t, r)` — noisy state, time, reference time
- **Tangents:** `(v_g, 1.0, 0.0)` — guided velocity, dt/dt=1, dr/dt=0
- **Output:** `u` = model prediction, `du_dt` = JVP derivative w.r.t. time
- **Target:** `u_tgt = v_g - clip(t-r, [0,1]) * du_dt` (MeanFlow identity, Eq. 8)
- **Gradient stop** on target prevents backprop through JVP target

## 2. t,r Sampling (lines 147-157)

```python
def sample_tr(self, bz):
    t = self.noise_distribution()(bz)  # logit-normal or uniform
    r = self.noise_distribution()(bz)
    t, r = jnp.maximum(t, r), jnp.minimum(t, r)  # enforce t >= r

    data_size = int(bz * self.data_proportion)  # default 0.75
    zero_mask = jnp.arange(bz) < data_size
    zero_mask = zero_mask.reshape(bz, 1, 1, 1)
    r = jnp.where(zero_mask, t, r)  # 75% of batch: r = t (pure data, h=0)

    return t, r
```

- **Logit-normal** (lines 138-142): `sigmoid(normal * P_std + P_mean)` with P_mean=-0.4, P_std=1.0
- **t >= r enforcement:** `jnp.maximum(t, r)` / `jnp.minimum(t, r)`
- **data_proportion=0.75:** For 75% of the batch, `r = t` (flow matching regime, h=0)

## 3. 1-NFE Sampling (lines 111-113)

```python
def solver_step(self, z_t, t, r, labels):
    u = self.u_fn(z_t, t=t, h=(t - r), y=labels, train=False)
    return z_t - jnp.einsum('n,n...->n...', t - r, u)
```

Formula: `z_0 = z_t - (t - r) * u(z_t, t, t-r, y)`

For 1-NFE with schedule `[1.0, 0.0]`: `z_0 = z_1 - 1.0 * u(z_1, 1.0, 1.0, y)`

## 4. Prediction Type: u-Prediction

The network directly predicts the average velocity `u`. NOT x-prediction.

```python
def u_fn(self, x, t, h, y, train=True):
    bz = x.shape[0]
    return self.net(x, t.reshape(bz), h.reshape(bz), y, train=train, key=self.make_rng('gen'))
```

Loss is MSE on `u` (lines 238-239):
```python
loss = (u - u_tgt) ** 2
loss = jnp.sum(loss, axis=(1, 2, 3))
```

## 5. Algorithm 1: Full Training Flow (lines 206-256)

```python
def forward(self, imgs, labels, train=True):
    x = imgs.astype(self.dtype)
    bz = imgs.shape[0]

    # 1. Sample t, r with data_proportion constraint
    t, r = self.sample_tr(bz)

    # 2. Create noisy interpolation
    e = jax.random.normal(self.make_rng('gen'), x.shape, dtype=self.dtype)
    z_t = (1 - t) * x + t * e        # Linear interpolation
    v = e - x                          # Instantaneous velocity

    # 3. Guided velocity (CFG)
    v_g = self.guidance_fn(v, z_t, t, labels, train=False) if self.guidance_eq else v

    # 4. Conditional dropout
    y_inp, v_g = self.cond_drop(v, v_g, labels)

    # 5. JVP: compute u and du/dt
    u, du_dt = jax.jvp(u_fn, (z_t, t, r), (v_g, dt_dt, dr_dt))

    # 6. Target with clipping
    u_tgt = v_g - jnp.clip(t - r, a_min=0.0, a_max=1.0) * du_dt
    u_tgt = jax.lax.stop_gradient(u_tgt)

    # 7. MSE loss with adaptive weighting
    loss = (u - u_tgt) ** 2
    loss = jnp.sum(loss, axis=(1, 2, 3))
    adp_wt = (loss + self.norm_eps) ** self.norm_p
    loss = loss / jax.lax.stop_gradient(adp_wt)
    loss = loss.mean()

    return loss, {'loss': loss, 'v_loss': v_loss}
```

## 6. Numerical Stability Tricks

1. **Gradient stopping** (2 places):
   - Line 236: `u_tgt = jax.lax.stop_gradient(u_tgt)` — target is constant
   - Line 243: `loss / jax.lax.stop_gradient(adp_wt)` — weight denominator
2. **Clipping:** `jnp.clip(t-r, 0.0, 1.0)` — bounds weighting coefficient
3. **Adaptive weighting:** `(loss + 0.01)^1.0` — prevents gradient explosion on high-loss samples
4. **Einsum broadcasting:** `einsum('n,n...->n...')` for safe scalar-tensor multiply

## Default Hyperparameters (configs/default.py)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| noise_dist | logit_normal | t,r sampling distribution |
| P_mean | -0.4 | Logit-normal mean |
| P_std | 1.0 | Logit-normal std |
| data_proportion | 0.75 | Fraction with r=t |
| norm_p | 1.0 | Adaptive weighting power |
| norm_eps | 0.01 | Adaptive weighting epsilon |
| num_steps | 1 | 1-NFE sampling |
| guidance_eq | cfg | Classifier-free guidance |
| omega | 1.0 | CFG scale |

## Key Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `meanflow.py` | Core MeanFlow class | 56-256 |
| `train.py` | Training loop with EMA | 96-139 |
| `configs/default.py` | Hyperparameters | 1-112 |
