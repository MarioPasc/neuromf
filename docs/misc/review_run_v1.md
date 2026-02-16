# Scientific Review: Run v1 Agent Analysis & Recommendations for Run v2

## 1. Validation of the Agent's Analysis

### 1.1 What the Agent Got Right

The agent's core diagnosis is **correct and well-supported by the data**. I verified the JSON directly and confirm the following quantitative claims:

The raw loss trajectory shows the three-phase pattern as described. The actual per-channel sums from the JSON are: epoch 24 → 142.9, epoch 74 → 1078.3, epoch 149 → 292.9, epoch 174 → 3025.2, epoch 249 → 16463.8, epoch 399 → 86127.5, epoch 499 → 331607.7. These match the agent's report closely (the agent stated epoch 149 as 293 and epoch 499 as 331,608, which are consistent with the raw data).

The adaptive weighting saturation analysis is mathematically sound. With `norm_eps=1.0` and `norm_p=1.0`, the adaptive weight in the codebase computes as:

$$w = (\text{raw\_loss} + \varepsilon)^{p} = (L + 1)^{1}$$

giving weighted loss:

$$\mathcal{L}_{\text{weighted}} = \frac{L}{L + 1} \xrightarrow{L \to \infty} 1$$

The gradient with respect to parameters scales as:

$$\frac{\partial \mathcal{L}_{\text{weighted}}}{\partial \theta} = \frac{1}{(L+1)^2} \cdot \frac{\partial L}{\partial \theta}$$

At epoch 499 with $L \approx 331{,}608$, the effective gradient magnitude is suppressed by a factor of $\sim 10^{11}$ relative to the raw gradient — the optimizer is indeed blind. This analysis is correct and represents the **primary failure mode**.

The epoch 274 latent channels (Image 3) confirm healthy latent structure: $\sigma \in [0.26, 0.41]$, ranges within $[-1.6, 1.75]$, visible bilateral symmetry in axial views. By epoch 499 (Image 1), $\sigma$ has grown to 1.47 for channel 0 with extreme values $[-12.99, 9.95]$, confirming variance explosion.

The generated samples at epoch 274 (Image 4) show blurry but anatomically recognisable brain MRI with correct bilateral symmetry, ventricle structure, and grey-white matter contrast. Epoch 499 samples (Image 2) show amorphous blobs with lost anatomical structure.

### 1.2 Corrections and Nuances

**Channel 0 dominance claim requires nuancing.** The agent states channel 0 grows from 28% to 77% of total loss. The JSON confirms this trajectory: 28.4% at epoch 24 → 27.2% at epoch 74 → 39.3% at epoch 149 → 77.4% at epoch 274. However, by epoch 499 channel 0 has actually decreased to **52.9%** of total loss (175,469 / 331,607), as channels 2 and 3 have also begun diverging substantially (90,637 and 64,248 respectively). Channel 1 remains relatively stable at 1,252. This suggests the instability propagated from channel 0 to channels 2 and 3 but spared channel 1, which is consistent with channel 1 encoding high-frequency texture information with lower spatial coherence (visible in Image 3 as noise-like patterns).

**The best-model epoch may be earlier than stated.** The agent identifies epochs 125–175 as the best region. Looking at the actual per-channel loss data, the minimum recorded raw loss is 142.9 at epoch 24 (the earliest available per-channel measurement). The true minimum almost certainly occurs between epochs 100–149, since epoch 149 has a raw loss of 292.9 (already higher than epoch 24). However, we only have per-channel logs every 25 epochs, so the exact optimum lies somewhere in the gap. The agent's recommendation for early stopping with patience ~50 epochs is sound, but the target should be saving checkpoints more frequently (every 5–10 epochs) during the first 200 epochs.

**The `loss_mean` column is misleading in the JSON.** All 500 entries show `loss_mean ≈ 0.999999`, confirming the adaptive weighting saturation claim. The `loss_std` values are $\mathcal{O}(10^{-7})$, which would be indistinguishable from numerical noise in fp32 computation ($\epsilon_{\text{fp32}} \approx 6 \times 10^{-8}$). This is a critical observation the agent made correctly.

**The EMA divergence analysis is directionally correct but the values are modest.** EMA divergence grows from 0.012 → 0.054 over 500 epochs. While monotonically increasing, these values are not catastrophic in absolute terms. The more alarming diagnostic is the raw loss explosion, not the EMA divergence.

### 1.3 Missing Analyses

The agent did not examine:

**h_zero_frac = 0.75 throughout all epochs.** This means 75% of samples have $r = t$ (pure FM samples) and only 25% have $r < t$ (MF samples contributing the JVP self-consistency term). This is governed by `data_proportion` in `time_sampling`. While the MeanFlow paper recommends 50–75% FM fraction, the agent should verify this is intentional and not suppressing the MF learning signal.

**grad_clip_fraction dynamics.** The fraction starts at 0.875 (most gradients being clipped), drops to 0.0 by epoch 149, and remains low throughout. This means gradient clipping was active primarily during early training and became ineffective later — consistent with the 1/(L+1)² gradient suppression making raw gradients vanishingly small.

**The `relative_update_norm` remains stable at ~0.005–0.008 throughout.** This is suspicious: even as the raw loss explodes 2,320×, the parameter update norms barely change. This confirms the optimizer is performing random-walk updates of constant magnitude rather than responding to the diverging loss.


## 2. Enhanced Recommendations for Run v2

I concur with the agent's recommendations and propose the following modifications and additions:

### 2.1 Critical Priority: Fix the Adaptive Weighting

The agent's suggestion to lower `norm_p` to 0.5 is reasonable but I recommend a more principled approach. The MeanFlow paper (Gat et al., 2025, Eq. 22 in Appendix B.2) defines the adaptive weight as:

$$w = \frac{1}{(\|\Delta\|_2^2 + c)^p}$$

where $p = 1 - \gamma$ and $c$ is the stabilisation constant. With the current settings $p=1.0, c=1.0$, the weight decays as $1/L$, which is excessively aggressive at our scale.

**Recommendation:** Set `norm_p=0.5, norm_eps=0.01`. This gives $w = 1/\sqrt{L + 0.01}$, which still normalises the loss but preserves gradient signal proportional to $\sqrt{L}$ rather than $1$. The smaller $\varepsilon = 0.01$ prevents the constant additive bias from dominating when losses are small. This configuration is closer to the Pseudo-Huber loss (Karras et al., "Analyzing and Improving the Training Dynamics of Diffusion Models", NeurIPS 2024), which has been shown to work well for small-to-medium scale diffusion training.

Alternatively, for the very first diagnostic run, **temporarily disable adaptive weighting entirely** (`adaptive: false`) to observe the true loss landscape and determine when the model naturally overfits.

### 2.2 LR Schedule and Regularisation

Cosine LR schedule is appropriate. However, the specific formulation matters. With only $T = 4000$ steps and cosine decay from $\eta_0 = 10^{-4}$:

$$\eta(t) = \frac{\eta_0}{2}\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

At step 2000 (epoch ~250), $\eta = 5 \times 10^{-5}$. At step 3600 (epoch ~450), $\eta \approx 1 \times 10^{-5}$. This provides a natural ~10× reduction in step size during the memorisation danger zone.

**Weight decay at $10^{-4}$** is appropriate. With AdamW, this effectively constrains the $\ell_2$ norm of weights and prevents the unbounded growth visible in late epochs.

### 2.3 Channel Weighting

The agent's suggestion of `channel_weights: [0.5, 1.0, 1.0, 1.0]` is reasonable but somewhat arbitrary. A better principled approach is **inverse-variance weighting**:

$$w_c = \frac{1/\sigma_c^2}{\sum_{c'} 1/\sigma_{c'}^2}$$

where $\sigma_c$ is the per-channel latent standard deviation from the pre-computed statistics. Since channel 0 encodes low-frequency structure with higher variance in the MAISI VAE, this naturally down-weights it.

Alternatively, switching to **L1 loss** (`p=1.0`) is an elegant solution that the agent correctly identified: L1 is more robust to outlier errors and prevents the quadratic penalty on large errors that accelerates the divergence cascade.

### 2.4 EMA Decay

The agent recommends `ema.decay=0.9999`. The half-life in steps is:

$$t_{1/2} = \frac{\ln 2}{1 - \beta} = \frac{0.693}{1 - 0.9999} \approx 6{,}931 \text{ steps}$$

With only 4,000 total steps, this EMA would barely begin incorporating the model — the EMA at step 4000 would still be dominated by early-training weights:

$$\text{EMA contribution from step 0} = 0.9999^{4000} = e^{-0.4} \approx 0.67$$

This is actually desirable: it makes the EMA act as a strong average over the entire training trajectory, smoothing out late-training instability. However, **if training is extended to more steps with the larger dataset, decay should be adjusted accordingly** (target half-life of ~10% of total steps).

### 2.5 Fixed Divergence Threshold

The agent is correct that EMA-relative thresholds drift with the diverging loss. A fixed threshold of `raw_loss > 5000` would have caught the divergence at epoch 174 ($L = 3025$, still below) or epoch 224 ($L = 8918$, above). A more robust strategy: **halt if the raw loss exceeds 10× the minimum observed raw loss** (tracked as a running minimum, not an EMA). This is scale-adaptive without drifting.


## 3. Addressing Specific Concerns

### 3.1 Concern 1: Latent-Space Data Augmentation

This is a critical intervention for our regime. In latent space, the standard voxel-space augmentation taxonomy translates to a more restricted set of valid operations, because the VAE's learned representation has specific statistical properties that arbitrary transformations may violate.

**Tier 1 — Safe, nearly free augmentations (recommended for immediate use):**

These are geometric operations that commute with equivariant convolutions and respect the latent distribution:

**Random 3D flips** along each spatial axis $(d, h, w)$ independently. Because the brain is approximately bilaterally symmetric across the sagittal plane, flipping the left-right axis (axis 2 in the 48³ representation) is biologically valid and effectively doubles the dataset. Axial and coronal flips are less anatomically justified but still produce valid latent representations because the VAE is translation-equivariant. Implementation: `monai.transforms.RandFlip` operates on tensors of shape $(C, D, H, W)$ and can be applied directly to the 4×48³ latents. Each axis-flip is an independent Bernoulli($p$) trial.

**Random 90° rotations** around the inferior-superior axis. Brain MRI is rotationally symmetric modulo 90° in the axial plane (after skull stripping). Implementation: `monai.transforms.RandRotate90` with `spatial_axes=(1, 2)` for axial-plane rotation.

The theoretical justification: if $z = \text{Enc}(x)$ and $T$ is a geometric transformation such that $\text{Enc}(T(x)) \approx T'(z)$ (equivariance), then augmenting in latent space with $T'$ is equivalent to augmenting in voxel space with $T$. For 3D convolutional VAEs like MAISI, flips and 90° rotations satisfy this equivariance property exactly (Bronstein et al., "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges", 2021).

With 3 independent axis flips ($2^3 = 8$ combinations) and 4 axial rotations, the theoretical augmentation factor is **up to 32×**, though in practice many combinations are equivalent due to brain symmetry.

**Tier 2 — Moderate augmentations (recommended after validating Tier 1):**

**Small Gaussian noise injection.** Adding $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma \ll \sigma_{\text{latent}}$ to the latent vectors. This acts as implicit regularisation by smoothing the empirical data distribution. The key constraint is that $\sigma$ must be small enough not to push the augmented latents outside the VAE's valid decoding range. A safe starting point: $\sigma = 0.05 \times \sigma_c$ per channel, where $\sigma_c$ is the per-channel standard deviation from `latent_stats.json`. This is related to the Gaussian data augmentation analysed by Bishop (1995, "Training with Noise is Equivalent to Tikhonov Regularisation", Neural Computation) and more recently by Daras et al. in the Ambient Diffusion framework (NeurIPS 2023).

**Intensity scaling per channel.** Random scaling of each latent channel by $s_c \sim \text{Uniform}(1 - \alpha, 1 + \alpha)$ with $\alpha \approx 0.05$. This is analogous to intensity augmentation in voxel space and accounts for scanner-to-scanner intensity variability.

**Tier 3 — Advanced augmentations (use with caution):**

**Latent-space interpolation (mixup).** Given two latent vectors $z_1, z_2$, generate $z_\lambda = \lambda z_1 + (1 - \lambda) z_2$ with $\lambda \sim \text{Beta}(\alpha, \alpha)$. Zhang et al. ("mixup: Beyond Empirical Risk Minimisation", ICLR 2018) showed this provides strong regularisation for overparameterised models. In latent diffusion, this has been validated by Pinaya et al. ("Brain Imaging Generation with Latent Diffusion Models", 2022) though care must be taken that interpolated latents remain within the VAE's valid decoding manifold.

**Available libraries:** MONAI provides `RandFlip`, `RandRotate90`, `RandGaussianNoise`, `RandScaleIntensity`, and `RandAffine` (all supporting 3D tensors with shape $(C, D, H, W)$). These are directly applicable to the pre-computed latents without custom code. For mixup, `torch` native operations suffice. There is no need to code custom augmentation transforms.

### 3.2 Concern 2: Sparsely Supervised Diffusion and Anti-Memorisation

The paper by Zhao et al. ("Sparsely Supervised Diffusion", arXiv 2602.02699, 2026) proposes **masking a fraction of spatial locations** during the denoising loss computation. The key claim is that training diffusion models with up to 98% of pixels masked still produces competitive FID scores while significantly reducing memorisation and improving stability on small datasets.

This connects to a broader family of anti-memorisation techniques:

**Ambient Diffusion** (Daras et al., NeurIPS 2023) demonstrated that training diffusion models on corrupted data (randomly masked or noisy) provably reduces memorisation. The framework modifies the score matching objective from $\mathbb{E}[\|\epsilon_\theta(x_t, t) - \epsilon\|^2]$ to operate on a corrupted observation $y = M \odot x + (1-M) \odot \xi$, where $M$ is a random binary mask and $\xi$ is fill noise. The theoretical result shows that the learned score converges to the true data score as training proceeds, despite never seeing clean data.

**MaskDiT** (Zheng et al., TMLR 2024) showed that masking 50% of patches during diffusion training in latent space with a DiT architecture achieves comparable FID to full training while reducing per-iteration cost by 2×. They add an auxiliary MAE reconstruction loss on masked patches to maintain global coherence.

**Application to NeuroMF:** The Sparsely Supervised Diffusion approach is directly applicable to our latent-space MeanFlow formulation. For the flow matching loss $\|\hat{v}(z_t, t) - v_c\|^p$, we can apply a spatial mask $M \in \{0, 1\}^{D \times H \times W}$ to compute only on unmasked locations:

$$\mathcal{L}_{\text{masked}} = \frac{1}{|M|} \sum_{i \in M} |\hat{v}_i - (v_c)_i|^p$$

where $|M|$ is the number of unmasked spatial locations (shared across all 4 channels to maintain cross-channel consistency). This reduces the effective information the model receives per training step, acting as an implicit regulariser that forces the model to learn global structure rather than memorising per-voxel correspondences.

**For our regime (990 samples, 178M parameters), I recommend a mask ratio of 50–75%** (not 98%, as that is designed for much larger datasets). The mask should be applied at the spatial level (same mask across all 4 channels) and should be a random block mask (not pixel-level) for computational efficiency and to better match the local receptive field of the UNet.

Implementation is straightforward: generate a random binary mask $M$ per batch element, multiply both the prediction and target by $M$ before the loss computation, and normalise by $|M|$. This requires ~5 lines of code in the `MeanFlowPipeline.forward()` method.

**Regarding adding noise to latents:** This is conceptually distinct from the sparse supervision approach. Adding Gaussian noise to latents before training acts as data augmentation (Tier 2 above), while the sparse supervision approach masks the loss computation. Both can be applied simultaneously and address different aspects of the memorisation problem. The noise augmentation smooths the empirical distribution in input space, while the sparse loss masks reduce the information bottleneck in gradient space.

### 3.3 Concern 3: Dataset Expansion (FOMO-60K and UK Biobank)

The single most impactful intervention for Run v2 is increasing the training set. The memorisation literature (Gu et al., "Why Diffusion Models Don't Memorize", 2025) establishes that the memorisation onset time $\tau_{\text{mem}}$ scales linearly with dataset size $n$:

$$\tau_{\text{mem}} \propto n$$

Currently with $n = 990$ and $p = 178\text{M}$, the parameter-to-sample ratio is $\sim 180{,}000:1$. Even modest dataset expansion changes the regime dramatically:

| Dataset size $n$ | Param/sample ratio | Projected $\tau_{\text{mem}}$ (relative) | Expected regime |
|---|---|---|---|
| 990 | 180,000:1 | 1× | Severe memorisation |
| 5,000 | 35,600:1 | ~5× | Moderate memorisation |
| 10,000 | 17,800:1 | ~10× | Manageable with regularisation |
| 50,000 | 3,560:1 | ~50× | Likely sufficient |

Since T1W brain MRI is available from both FOMO-60K (subset matching our inclusion criteria) and UK Biobank (~40,000+ T1W scans), reaching $n \geq 10{,}000$ is realistic and would fundamentally change the training dynamics.

**Important note for the agent:** The latent encoding pipeline (Phase 1) must be re-run for any new data. The latent statistics (`latent_stats.json`) should be recomputed on the expanded dataset to ensure the per-channel normalisation remains valid.

### 3.4 Concern 4: Augmentation Toggle and Effective Sample Logging

The augmentation system should be designed with the following interface:

1. A `LatentAugmentation` class wrapping MONAI transforms with a `Compose` pipeline.
2. An `enabled: bool` flag in the training config (`config.training.augmentation.enabled`).
3. Per-epoch logging of `effective_samples = n_raw × augmentation_factor`, where `augmentation_factor` is estimated from the probability-weighted combination of enabled transforms.
4. A deterministic mode for reproducibility (seeded random state per epoch).

MONAI transforms are the correct choice for the geometric augmentations (flips, rotations) since they natively support $(C, D, H, W)$ tensors and integrate with PyTorch datasets. For the noise injection and masking, simple `torch` operations suffice.


## 4. Proposed Prioritised Action Plan for Run v2

**Phase A — Immediate fixes (before retraining):**
1. Fix adaptive weighting: `norm_p=0.5, norm_eps=0.01`
2. Cosine LR schedule over total training steps
3. Weight decay: `1e-4`
4. Log raw loss to TensorBoard alongside weighted loss
5. Fixed divergence threshold: halt if `raw_loss > 10 × min_raw_loss`
6. Checkpoint every 10 epochs; track best model by raw loss

**Phase B — Augmentation (implement in parallel):**
1. Latent-space augmentation pipeline with MONAI transforms (flips + 90° rotations)
2. Configurable enable/disable flag
3. Effective sample count logging per epoch
4. Optional: Gaussian noise injection with per-channel calibrated $\sigma$

**Phase C — Anti-memorisation (implement after Phase B):**
1. Spatial loss masking (50–75% mask ratio)
2. Configurable mask ratio and mask type (random block vs. random voxel)
3. Log unmasked-only loss as the monitoring metric

**Phase D — Dataset expansion (in parallel, Mario's responsibility):**
1. Download and encode additional FOMO-60K T1W samples
2. Evaluate UK Biobank T1W feasibility
3. Re-compute latent statistics on expanded dataset
4. Re-run training with expanded data


## 5. Agent Prompt

```
## Task: Implement Run v2 Training Improvements for NeuroMF

You are implementing Phase A and Phase B improvements for the NeuroMF latent-space MeanFlow training pipeline based on the Run v1 failure analysis. Work in the existing project structure under `src/neuromf/`.

### Context
Run v1 failed due to adaptive weighting saturation masking divergence after ~150 epochs of useful learning. The 178M-parameter model on 990 samples has a 180,000:1 param/sample ratio. The best samples were produced around epoch 125-175 before variance explosion. See the Run v1 analysis document for full details.

### Phase A: Critical Training Fixes

1. **Adaptive weighting configuration**: Change defaults in `configs/train_meanflow.yaml` to:
   - `meanflow.norm_p: 0.5`
   - `meanflow.norm_eps: 0.01`
   - Verify these propagate correctly through `MeanFlowPipelineConfig` → `MeanFlowPipeline.forward()`
   - The weighted loss should now behave as L/sqrt(L + 0.01), preserving gradient signal

2. **LR schedule**: Ensure `training.lr_schedule: cosine` is the default. Verify the cosine schedule implementation in `LatentMeanFlow.configure_optimizers()` computes the correct decay over `training.max_epochs × steps_per_epoch`.

3. **Weight decay**: Set `training.weight_decay: 1e-4` in config. Verify it is passed to AdamW.

4. **Raw loss TensorBoard logging**: In `LatentMeanFlow.training_step()`, log `raw_loss` as a separate scalar (`self.log("train/raw_loss", ...)`) in addition to the existing weighted `loss_mean`. This is the true training signal. Also log per-channel raw losses if available from diagnostics.

5. **Fixed divergence guard**: Replace the EMA-relative divergence threshold with a fixed threshold system:
   - Track `min_raw_loss` as a running minimum (updated each logged epoch)
   - Halt training if `current_raw_loss > divergence_multiplier × min_raw_loss` where `divergence_multiplier` defaults to 10.0 (configurable)
   - Log warnings when raw_loss exceeds 3× and 5× the minimum

6. **Checkpoint strategy**: 
   - Save checkpoints every 10 epochs (not just at end)
   - Track best model by raw_loss (not weighted loss)
   - Add `ModelCheckpoint` callback monitoring `train/raw_loss` with `mode="min"`

### Phase B: Latent-Space Data Augmentation

Create a new module `src/neuromf/data/latent_augmentation.py` that provides a configurable augmentation pipeline for pre-computed 4×48³ latent tensors.

**Requirements:**
- Use MONAI transforms wherever possible (`monai.transforms.RandFlip`, `monai.transforms.RandRotate90`, `monai.transforms.RandGaussianNoise`, `monai.transforms.RandScaleIntensity`)
- All transforms operate on tensors of shape (4, 48, 48, 48) — i.e., (C, D, H, W)
- The augmentation must be toggleable via config: `training.augmentation.enabled: true/false`
- Individual transforms should be independently toggleable with their own probability parameters

**Augmentation pipeline (in order):**
1. `RandFlip` on each spatial axis independently (prob=0.5 each). Axes: 1 (depth/sagittal), 2 (height/coronal), 3 (width/axial)
2. `RandRotate90` in the axial plane (spatial_axes=(1, 2), prob=0.25, max_k=3)
3. `RandGaussianNoise` with std calibrated to 0.05 × per-channel latent std (prob=0.3). Load per-channel std from latent_stats.json
4. `RandScaleIntensity` with factors=0.05 (prob=0.2) — applies per-channel random scaling ∈ [0.95, 1.05]

**Integration:**
- Apply augmentation inside `LatentDataset.__getitem__()` when `augmentation_enabled=True`
- Pass the composed MONAI transform pipeline as an argument to LatentDataset
- In the training script (`experiments/cli/train.py`), build the augmentation pipeline from config and pass it to the dataset

**Logging:**
- Log `effective_samples_per_epoch` to TensorBoard: this is `len(dataset)` × the expected augmentation factor
- The augmentation factor = product of (1 + p_i × (k_i - 1)) for each transform, where p_i is the probability and k_i is the number of distinct outcomes
- For the default config: flips give 2^3=8 (but prob-weighted: 1 + 0.5×7 = 4.5), rotations give ~1.75, noise gives ~1.0, scaling gives ~1.0. Approximate effective factor ≈ 4.5 × 1.75 ≈ 7.9×
- Log this as a scalar at epoch 0 for reference

**Config structure in `configs/train_meanflow.yaml`:**
```yaml
training:
  augmentation:
    enabled: true
    flip_prob: 0.5          # per-axis flip probability
    rotate90_prob: 0.25     # axial-plane 90° rotation
    gaussian_noise_prob: 0.3
    gaussian_noise_std_fraction: 0.05  # fraction of per-channel latent std
    scale_intensity_prob: 0.2
    scale_intensity_factors: 0.05
```

**Testing:**
- Write a test in `tests/test_latent_augmentation.py` that:
  1. Creates a dummy 4×48³ tensor
  2. Applies the augmentation pipeline
  3. Verifies output shape is preserved
  4. Verifies that with augmentation enabled, repeated calls produce different outputs
  5. Verifies that with augmentation disabled, output equals input
  6. Verifies flipped tensors match expected torch.flip() output

### General Guidelines
- Follow the existing code style: type hints, docstrings, logging via `logging.getLogger(__name__)`
- Use `from __future__ import annotations` at top of new files
- Keep changes minimal and focused — do not refactor unrelated code
- All new config keys must have sensible defaults so existing configs remain valid
- Run existing tests (`pytest tests/ -x`) after changes to verify nothing breaks
```


## References

- Gat, I. et al., "Flow Matching Guide and Code" (2024). arXiv:2412.06264.
- Gat, I. et al., "Improved Mean Flows" (2025). Improved MeanFlow (iMF) formulation.
- Karras, T. et al., "Analyzing and Improving the Training Dynamics of Diffusion Models" (2024). NeurIPS.
- Daras, G. et al., "Ambient Diffusion: Learning Clean Distributions from Corrupted Data" (2023). NeurIPS.
- Zheng, H. et al., "Fast Training of Diffusion Models with Masked Transformers (MaskDiT)" (2024). TMLR.
- Zhao, W. et al., "Sparsely Supervised Diffusion" (2026). arXiv:2602.02699.
- Gu, R. et al., "Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training" (2025). arXiv:2505.17638.
- Fontanella, A. et al., "Bigger Isn't Always Memorizing: Early Stopping Overparameterized Diffusion Models" (2025). arXiv:2505.16959.
- Zhang, H. et al., "mixup: Beyond Empirical Risk Minimization" (2018). ICLR.
- Bishop, C.M., "Training with Noise is Equivalent to Tikhonov Regularization" (1995). Neural Computation.
- Bronstein, M. et al., "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (2021). arXiv:2104.13478.
- Khader, F. et al., "Denoising Diffusion Probabilistic Models for 3D Medical Image Generation" (2023). Scientific Reports.
- Pinaya, W. et al., "Brain Imaging Generation with Latent Diffusion Models" (2022). MICCAI Workshop.
- NVIDIA DLMED, "MAISI: Medical AI for Synthetic Imaging" (2024). NVIDIA Technical Blog.
