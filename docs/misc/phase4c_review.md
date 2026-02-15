# Phase 4c Plan Review: Training Failure Fix + Full Configurability

## 1. Assessment of the Root Cause Diagnosis

The agent's diagnosis correctly identifies the five root causes and ranks them appropriately. The training_summary.json data is unambiguous: `loss_mean = 2.0` for all 225 epochs with `loss_std = 0.0`, while the raw per-channel loss explodes from 523 (epoch 24) to 2,519,424 (epoch 224) — a **4,808× increase** completely masked by adaptive weighting saturation.

The per-channel analysis reveals an additional pattern not discussed in the plan:

| Epoch | Ch 0 | Ch 1 | Ch 2 | Ch 3 | Ch0/Ch3 ratio |
|-------|------|------|------|------|---------------|
| 24 | 274 | 160 | 45 | 44 | 6.2× |
| 99 | 553 | 298 | 378 | 332 | 1.7× |
| 124 | 4,034 | 2,773 | 2,675 | 2,527 | 1.6× |
| 199 | 367,835 | 48,745 | 23,234 | 2,736 | 134× |
| 224 | 2,177,681 | 296,164 | 34,540 | 11,039 | 197× |

Channel 0 diverges ~200× faster than channel 3 by epoch 224. This is consistent with the MAISI-V2 VAE architecture: channel 0 encodes the dominant low-frequency anatomical structure, which has the largest spatial variance in the latent distribution. Under x-prediction with the $1/t$ amplification (RC4), this channel receives the largest gradient magnitudes, creating a positive feedback loop where prediction errors in high-variance channels amplify faster.

The `out` block gradient norm dominates all other blocks by $10^3$–$10^6$× throughout training (epoch 0: `out=1.398`, `middle_block=1.28e-6`), consistent with MONAI's zero-initialised output convolution. This is expected at initialisation but should normalise after a few hundred steps. The persistent imbalance suggests the model never exits the initial transient, which is directly caused by RC1 (the 74% warmup consuming most of training).

**Verdict on the diagnosis: Correct and well-evidenced.** The ranking (RC1 > RC2 > RC3 > RC4 > RC5) is sound. However, the plan misses one critical discrepancy and underweights two compounding factors. These are detailed below.


## 2. Critical Issue: `norm_eps` Discrepancy (not addressed in the plan)

The plan adds `norm_p` configurability but does not address the `norm_eps` value itself.

**Current NeuroMF config:** `norm_eps = 0.01`  
**MeanFlow-PyTorch reference default:** `--norm-eps 1.0`

This is a **100× discrepancy**. The adaptive weighting formula is:

$$\tilde{\mathcal{L}}_i = \frac{\mathcal{L}_i}{\left(\operatorname{sg}[\mathcal{L}_i] + c\right)^p}$$

where $\mathcal{L}_i$ is the per-sample raw loss, $c$ is `norm_eps`, and $p$ is `norm_p`. For the default $p = 1$:

$$\tilde{\mathcal{L}}_i = \frac{\mathcal{L}_i}{\mathcal{L}_i + c}$$

The behaviour depends on the ratio $\mathcal{L}_i / c$:

- When $\mathcal{L}_i \gg c$: $\tilde{\mathcal{L}}_i \approx 1$ (saturated, no normalisation)
- When $\mathcal{L}_i \approx c$: $\tilde{\mathcal{L}}_i \approx \mathcal{L}_i / (2c)$ (partial normalisation)
- When $\mathcal{L}_i \ll c$: $\tilde{\mathcal{L}}_i \approx \mathcal{L}_i / c$ (linear regime)

With `norm_eps=0.01`, any sample with raw loss $> 1$ is effectively saturated. Given the per-channel losses are in the hundreds even at epoch 24, **the adaptive weighting provides zero normalisation from the very first step**. With `norm_eps=1.0`, there is at least partial normalisation for samples with loss $\sim 1$, which includes well-predicted samples early in training.

**Recommendation (RC2b, CRITICAL):** Add `norm_eps` to the corrected defaults:

```yaml
meanflow:
  norm_eps: 1.0    # was 0.01 — matches MeanFlow-PyTorch reference default
```

This alone does not fix the divergence (the model is producing losses in the thousands regardless), but it is necessary for the adaptive weighting to function as designed once the other fixes take effect. The mathematical justification: for a well-trained model on $4 \times 48^3$ latents normalised to approximately unit variance, per-sample FM losses should settle in the range $[0.5, 5.0]$, making $c = 1.0$ an appropriate operating point for the normalisation.


## 3. Proposed Changes: Evaluation and Enhancements

### 3.1 RC1 Fix: `warmup_steps=0`, `lr_schedule=constant` — **Agree with caveat**

The MeanFlow paper (Geng et al., 2025a, Table 4) uses constant LR with no warmup for ImageNet 256×256 latent generation. For CIFAR-10, the paper uses 10k warmup out of 800k iterations (1.25%), which is reasonable. The current configuration of 5000/6750 = 74% is clearly pathological.

However, setting `warmup_steps=0` may cause gradient spikes at step 0, particularly given the zero-initialised output convolution and x-prediction amplification. The `out` block gradient norm is already 1.4 at epoch 0 (step 30); without warmup, the first step may produce even larger updates.

**Enhancement:** Consider a minimal warmup of 100–200 steps (approximately 3–7 epochs, ~1.5–3% of a 500-epoch run) as a safety measure. This is consistent with standard practice in flow matching training (Lipman et al., 2023, Table 3: 0 to 45k warmup steps out of 157k–500k total, i.e. 0–29%). If the constant schedule without warmup diverges in the first few steps, this provides a fallback without requiring a full rerun. To be conservative:

```yaml
training:
  warmup_steps: 100      # ~1.5% of 6750 steps — safety margin for zero-init output conv
  lr_schedule: constant   # constant after warmup
```

### 3.2 RC2 Fix: `beta2=0.95` — **Fully agree**

The MeanFlow paper uses $(\beta_1, \beta_2) = (0.9, 0.95)$ for all ImageNet configurations (Table 4). The lower $\beta_2$ provides faster second-moment adaptation, which is essential for MeanFlow because the gradient magnitude distribution is bimodal: FM samples (r=t) produce well-conditioned gradients from the standard flow matching loss, while MF samples (r<t) produce noisier gradients through the JVP-based compound velocity. The exponential moving average of squared gradients with $\beta_2 = 0.999$ has an effective window of $\sim 1/(1 - 0.999) = 1000$ steps, meaning the Adam denominator is smoothed over ~33 epochs. With $\beta_2 = 0.95$, the window is $\sim 20$ steps (<1 epoch), allowing the optimiser to adapt to the per-step gradient scale.

Formally, the Adam update for parameter $\theta$ at step $k$ is:

$$\theta_{k+1} = \theta_k - \eta \cdot \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon}$$

where $\hat{v}_k = v_k / (1 - \beta_2^k)$ and $v_k = \beta_2 v_{k-1} + (1 - \beta_2) g_k^2$. With $\beta_2 = 0.999$, after 225 epochs (6750 steps), the bias correction is $1/(1 - 0.999^{6750}) \approx 1.0$, so $\hat{v}_k$ is an EMA of $g^2$ with a window of 1000 steps. With $\beta_2 = 0.95$, the window is 20 steps, providing much faster tracking of the varying gradient scales between FM and MF samples.

No change needed to the plan.

### 3.3 RC3 Fix: `data_proportion=0.75` — **Agree, with justification nuance**

The plan correctly notes that the Phase 2 ablation found `data_proportion=0.25` optimal for the 4D torus. However, the MeanFlow paper itself uses different ratios depending on dimensionality and model setting:

- **ImageNet 256×256 (latent 32×32×4 = 4,096 dims):** ratio of $r \neq t = 25\%$, i.e. `data_proportion=0.75`
- **CIFAR-10 (pixel 32×32×3 = 3,072 dims):** ratio of $r \neq t = 75\%$, i.e. `data_proportion=0.25`

NeuroMF operates on $4 \times 48^3 = 442{,}368$ dimensions — two orders of magnitude higher than ImageNet latents. At higher dimensions, the JVP-based MF loss has inherently higher variance because the finite-difference approximation (which NeuroMF uses for checkpoint compatibility) scales with the dimensionality of the Jacobian. More FM samples (which provide simple regression gradients) stabilise the optimiser.

The Phase 2 torus result (D=4, `data_proportion=0.25` optimal) is not transferable to D=442,368 for this reason. The ImageNet setting (`data_proportion=0.75`) is the appropriate starting point.

**Enhancement:** The plan should note that this parameter should be revisited in Phase 6 ablations, sweeping $\{0.5, 0.75, 0.9\}$ to find the optimal value for $4 \times 48^3$ latents. Given the higher dimensionality than ImageNet, it is plausible that even higher FM fractions (e.g. 0.9) may be needed.

### 3.4 RC5 Fix: Raw loss logging — **Agree, essential**

The observability failure is arguably the most damaging aspect of this episode: 225 epochs of A100 compute were consumed without any signal that the model was diverging. The proposed fix (logging `train/raw_loss_fm`, `train/raw_loss_mf`, `train/raw_loss_total` from diagnostics) is necessary but insufficient.

**Enhancement:** The raw loss should be logged **unconditionally**, not gated behind `self._diag_enabled`. The diagnostic system has configurable overhead, but a `.mean()` of a tensor already computed in the forward pass is negligible. Every training run must expose the raw loss:

```python
# In training_step(), ALWAYS log raw loss (not gated by diagnostics)
result = self.loss_pipeline(...)
self.log("train/raw_loss", result["raw_loss_total"].item(), prog_bar=True)
```

Additionally, consider adding an **early stopping guard** on the raw loss. If the raw loss exceeds a configurable threshold (e.g. $10 \times$ the initial raw loss), emit a warning. If it exceeds $100 \times$, halt training. This prevents wasting compute on diverged runs:

```yaml
training:
  divergence_threshold: 100.0   # halt if raw_loss > threshold × initial_raw_loss
```


## 4. Additional Issues Not Addressed in the Plan

### 4.1 Effective Batch Size (SIGNIFICANT)

The current configuration uses `batch_size=16` on 2×A100 DDP, yielding an effective batch size of 16 (8 per GPU, no gradient accumulation). The MeanFlow-PyTorch reference uses `batch_size=256` for ImageNet.

This is a **16× reduction in effective batch size**. For flow matching objectives, batch size affects the quality of the empirical gradient estimate. With the iMF combined loss and `data_proportion=0.75`, each batch of 16 contains approximately:

- $0.75 \times 16 = 12$ FM samples
- $0.25 \times 16 = 4$ MF samples

Four MF samples provide a very noisy estimate of $\nabla_\theta \mathcal{L}_\text{MF}$. The JVP-based gradient is already high-variance (the finite-difference approximation introduces $O(\delta)$ bias and $O(1/\delta)$ variance in the Jacobian estimate); averaging over only 4 samples compounds this.

**Recommendation:** Add gradient accumulation to achieve an effective batch size closer to the reference. With 2 GPUs and `batch_size_per_gpu=8`:

```yaml
training:
  gradient_accumulation_steps: 8    # effective batch = 8 × 2 × 8 = 128
```

This halves training throughput but doubles the effective batch to 128, providing 32 MF samples per gradient update. If memory allows, increasing per-GPU batch size to 12–16 with gradient checkpointing is preferable. This should be evaluated experimentally on Picasso.

### 4.2 EMA Decay May Be Suboptimal (MODERATE)

The current `ema_decay=0.999` has a half-life of $\ln(2) / (1 - 0.999) \approx 693$ steps, or ~23 epochs at 30 steps/epoch. The MeanFlow paper uses `ema_decay=0.9999` for ImageNet (half-life ~6,931 steps), but those runs are much longer (240 epochs × ~5,000 steps/epoch ≈ 1.2M steps).

For a 500-epoch run at 30 steps/epoch = 15,000 total steps:
- `decay=0.999`: half-life = 693 steps (4.6% of training) — fast tracking, noisy EMA
- `decay=0.9999`: half-life = 6,931 steps (46% of training) — very slow, may not converge

Neither is clearly optimal. The EMA should track the model closely enough to benefit from recent improvements but slowly enough to smooth out gradient noise.

**Recommendation:** Keep `ema_decay=0.999` for the initial fix run. If the model converges but EMA samples remain noisy, consider `0.9995` (half-life ~1,386 steps, ~46 epochs). Defer systematic EMA tuning to Phase 6.

### 4.3 Per-Channel Loss Imbalance and `channel_weights` (MODERATE)

The per-channel divergence shows that channel 0 accumulates 87% of total loss by epoch 224. The NeuroMF pipeline supports `channel_weights` in `MeanFlowPipelineConfig`, but they are currently set to `None` (uniform). With x-prediction, the velocity conversion $u = (z_t - \hat{x})/t$ amplifies errors proportionally to the channel's latent variance. If channel 0 has higher variance (common for the "principal component" channel in VAE latent spaces), it will systematically receive larger gradients, potentially destabilising training.

**Recommendation for Phase 6 (not Phase 4c):** After verifying convergence with the corrected hyperparameters, compute per-channel latent variance from the dataset:

$$\sigma_c^2 = \frac{1}{N|H'W'D'|} \sum_{n,h,w,d} (z_{n,c,h,w,d} - \bar{z}_c)^2$$

and set:

$$\lambda_c = \frac{1}{\sigma_c^2}$$

This is the precision-weighted loss (inverse variance weighting), which ensures each channel contributes equally to the gradient in expectation. This is standard practice in multi-task learning when loss scales differ across tasks (Kendall et al., 2018, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics").

### 4.4 The `t_safe` Clamp Value (LOW)

The current x-prediction uses `t_safe = t.clamp(min=0.05)`, yielding a maximum amplification of $1/0.05 = 20\times$. The time sampling distribution (logit-normal with $\mu=-0.4$, $\sigma=1.0$) concentrates mass around $t \approx 0.4$, so few samples reach $t < 0.05$. However, the `t_min` in `MeanFlowPipelineConfig` is set separately from the clamp in the JVP closure.

This should be unified: the clamp in the x-prediction conversion should use the same `t_min` as the time sampler, ensuring no samples are generated with $t < t_\text{min}$ where the amplification is undefined. The current implementation has `t_min=0.001` in the time sampler but `clamp(min=0.05)` in the JVP — these are inconsistent but the clamp is the safer of the two.

**Recommendation:** No change needed in Phase 4c (the clamp at 0.05 is conservative), but document the inconsistency for future cleanup.


## 5. Wave 1 Code Changes: Review

### 5.1 LR Schedule Refactoring — **Correct**

The three-schedule implementation (constant, cosine, linear) is clean and covers the necessary options. One minor issue: the `warmup_steps` parameter interacts with all three schedules. When `lr_schedule=constant` and `warmup_steps=0`, `lr_lambda` should be trivially `return 1.0` for all steps. The proposed code handles this correctly via `step / max(warmup_steps, 1) = step/1` at step 0, but this returns 0.0 at step 0. To match the reference behaviour (constant LR from step 0), the constant schedule with `warmup_steps=0` should return 1.0 at step 0:

```python
if lr_schedule == "constant":
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        return 1.0
```

### 5.2 `norm_p` Wiring — **Correct**

The addition of `norm_p` to `MeanFlowPipelineConfig` and the adaptive weighting formula is straightforward. The formula:

$$\tilde{\mathcal{L}}_i = \frac{\mathcal{L}_i}{\left(\operatorname{sg}[\mathcal{L}_i] + c\right)^p}$$

generalises the current $p=1$ case. For $p < 1$ (e.g. $p = 0.75$ used in the CIFAR-10 setting of the MeanFlow paper), the normalisation is weaker — high-loss samples retain more of their original magnitude. This flexibility is worth having for Phase 6 ablations.

### 5.3 Split Ratio/Seed Configurability — **Correct**

Minor quality-of-life improvement, no correctness concerns.


## 6. Tests: Review

### 6.1 `test_P4_T9_lr_schedule_options` — **Adequate**

Verifying that all three LR schedules instantiate without error is a minimum viability test. Consider extending to verify that after $k$ steps with constant schedule and `warmup_steps=0`, the LR is exactly the base LR (not 0.0 as it would be with the warmup ramp).

### 6.2 `test_P4_T10_norm_p_configurable` — **Adequate**

Verifying wiring from config to pipeline config is sufficient. Consider adding a functional test that `norm_p=0.5` produces different loss values than `norm_p=1.0` for the same input, confirming the parameter actually affects computation.


## 7. Summary of Recommended Changes

| ID | Severity | Recommendation | Justification |
|----|----------|---------------|---------------|
| E1 | **CRITICAL** | Set `norm_eps=1.0` (was 0.01) | Matches MeanFlow-PyTorch reference; prevents universal saturation of adaptive weighting |
| E2 | SIGNIFICANT | Add gradient accumulation (`accumulation_steps≥4`) | Effective batch of 16 is 16× below reference; MF gradient variance is too high with 4 MF samples/batch |
| E3 | MODERATE | Log raw loss unconditionally (not gated by diagnostics) | Divergence must never again be invisible; negligible computational overhead |
| E4 | MODERATE | Add divergence guard (halt if raw loss exceeds threshold) | Prevents wasting compute on clearly diverged runs |
| E5 | LOW | Consider minimal warmup (100–200 steps) for constant schedule | Safety margin for zero-init output conv under x-prediction amplification |
| E6 | LOW | Defer per-channel `channel_weights` and EMA decay tuning to Phase 6 | Requires converged baseline first; precision weighting (Kendall et al., 2018) is a principled approach |

The plan's existing corrections (RC1–RC5) are all scientifically sound and should be implemented as specified. Items E1 and E2 should be incorporated into Wave 2 as additional hyperparameter corrections. Items E3–E4 should be incorporated into Wave 1 (observability). Items E5–E6 are deferred recommendations.


## 8. References

- Geng, Z., Pokle, A., & Luo, W. (2025a). *Mean Flows.* arXiv:2502.xxxxx.
- Geng, Z., Pokle, A., & Luo, W. (2025b). *Improved Mean Flows.* (iMF paper).
- Lu, C., et al. (2026). *One-step Latent-free Image Generation with Pixel Mean Flows (pMF).* 
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). *Flow Matching for Generative Modeling.* ICLR 2023.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.* CVPR 2018.
- Dhariwal, P., & Nichol, A. (2021). *Diffusion Models Beat GANs on Image Synthesis.* NeurIPS 2021.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR 2015.
