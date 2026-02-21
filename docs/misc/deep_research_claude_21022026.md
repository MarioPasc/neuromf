# NeuroMF x-prediction plateau: diagnosis and path forward

**The x-prediction arm's quality plateau stems primarily from a ~600× gradient update deficit, not from fundamental algorithmic failure.** With only ~1,920 gradient updates versus iMF's ~1.125 million on ImageNet, the 178M-parameter model has barely begun converging. Secondary factors—an overly aggressive adaptive weighting at p=1.0, a dramatically undersized v-head, using only 15% of available training data, and a suboptimal FM/MF sample ratio—compound this undertrained state. The good news: cos(V, v_c) ≈ 0.25 in 442K-dimensional space is non-trivial and confirms the algorithm is working. The model needs orders of magnitude more optimization steps, combined with several configuration fixes identified through careful auditing against the iMF and pMF reference implementations.

---

## 1. Implementation audit against iMF and pMF

The table below systematically compares NeuroMF's implementation against the iMF and pMF references across every critical component. Items marked ⚠️ represent discrepancies that likely affect training quality.

| Component | NeuroMF (current) | iMF reference | pMF reference | Status |
|---|---|---|---|---|
| **Interpolation** | z_t = (1−t)·z_0 + t·ε | Same | Same | ✅ Correct |
| **Parameterization** | x-prediction: u = (z − x̂)/t | u-prediction (direct) | x-prediction: u = (z − x̂)/t | ✅ Matches pMF |
| **JVP tangent** | u_fn(z, t, r), tangent (v, 1, 0) | fn(z, r, t), tangent (v, 0, 1) | Same structure | ✅ Correct — arg order swapped, tangent matches |
| **Compound V** | V = u + (t−r)·sg(du/dt) | Same (Eq. 12) | Same | ✅ Correct |
| **Loss target** | v_c = ε − z_0 | Same | Same | ✅ Correct |
| **Adaptive weight** | norm_eps=0.01, norm_p=1.0, sg on weight | c ≈ 1e-3, p=1.0, sg on weight | Inherited from iMF | ⚠️ norm_eps differs (0.01 vs ~1e-3); minor |
| **v-head** | 1 ResBlock, 221K params (~0.12% of model) | 8 Transformer layers, ~3–57M params (shared backbone) | 8 layers (aux head) | ⚠️ **21–256× under-capacity** |
| **Conditioning** | h = t−r only | Separate r, t tokens (4 each) | Not specified | ⚠️ **Missing absolute time t**; original MF shows (t, t−r) is optimal |
| **Time sampling** | logit-normal(−0.4, 1.0) | logit-normal(−0.4, 1.0) | logit-normal(0.0, 0.8) | ✅ Matches iMF |
| **data_proportion** | 50% FM / 50% MF | Original MF: 75% FM / 25% MF (best) | Not specified | ⚠️ **50% MF is suboptimal**; 25% MF is recommended |
| **EMA decay** | 0.9999 | 0.9999 | Standard | ⚠️ **Mismatched to update count** (see §3) |
| **LR / schedule** | 1e-4 constant, 10-epoch warmup | 1e-4 constant, 10-epoch warmup | Muon optimizer | ✅ Matches iMF |
| **1-NFE sampling** | z_0 = z_1 − u_θ(z_1, 0, 1) | Same | Same | ✅ Correct |
| **t_min clamp** | 0.05 (in u = (z−x̂)/t) | N/A (direct u-pred) | Not specified | ⚠️ May truncate learning at small t |

**Three critical discrepancies** emerge from this audit. First, the v-head is dramatically undersized. iMF's 8-layer auxiliary head comprises 33–67% of backbone depth depending on model size and adds millions of parameters for tangent estimation, while NeuroMF's single ResBlock at 221K parameters is negligible. Second, h-only conditioning discards absolute time information that the original MF paper showed matters (FID 63.13 for h-only versus **61.06 for (t, h)** conditioning). Third, the FM/MF ratio at 50/50 diverges from the empirically optimal 75/25 split that AlphaFlow's gradient conflict analysis explains as necessary to stabilize competing trajectory objectives.

---

## 2. What the diagnostics reveal about where the model fails

Each diagnostic figure provides a window into a specific failure mode. The picture that emerges is a model learning the right direction but lacking the optimization budget to refine magnitudes and alignment.

**Loss dynamics (Fig 1) show adaptive weighting saturation, not convergence.** The weighted loss flatlines at ~2.0 while raw loss sits at ~1e7. This ratio is mathematically expected: with norm_p=1.0, the weighted loss equals `raw_loss / (raw_loss + 0.01) ≈ 1.0` for large raw_loss, with the ~2.0 value reflecting the dual-head sum of two such terms. The FM component's healthy decline shows the model learns standard flow matching well. The MF component's stagnation at ~1e7 reveals that the compound velocity V has not converged—consistent with the JVP tangent quality being limited by the undersized v-head and insufficient training.

**The 2× norm ratio (Fig 2) signals systematic velocity overshoot.** The compound velocity ||V|| ≈ 2×||v_c|| reveals the model predicts velocity magnitudes that are systematically too large. For x-prediction, u = (z_t − x̂)/t. Early in training when x̂ is inaccurate, u ≈ z_t/t = ((1−t)/t)·z_0 + ε. At the modal sampling time t ≈ 0.4 (from logit-normal(−0.4, 1.0)), the (1−t)/t ≈ 1.5 amplification of z_0 directly inflates ||u|| beyond ||v_c|| = ||ε − z_0||. The 2× ratio reflects **insufficient convergence of x̂ toward z_0**, not a fundamental parameterization error. With more training, x̂ → z_0 and u → v_c, collapsing the ratio toward 1.0. The stable JVP norm at ~1e4 and convergent v_tilde norms (~400) confirm numerical stability is not the issue.

**Cosine similarity metrics (Fig 3) reveal partial but real learning.** cos(V, v_c) ≈ 0.25 in **D = 442,368** dimensional space is meaningfully above the random baseline of ~0 (which has standard deviation ~1/√D ≈ 0.0015). The directional alignment exists but is weak. Critically, **cos(v_tilde, v_c) ≈ 0.42 exceeds cos(V, v_c) ≈ 0.25**, meaning the v-head's tangent estimate is better aligned than the compound velocity. This gap indicates the JVP computation (and the (t−r)·du/dt correction term) introduces error rather than reducing it—a signature of the model being too early in training for the compound velocity construction to help. The relative error near 0 for x-prediction confirms the computational pipeline (JVP, stop-gradient) is functioning correctly without numerical drift.

**Training health indicators (Fig 4) show no pathology.** Declining gradient norms (~1.0), vanishing clip fraction, and smoothly declining EMA divergence all indicate healthy optimization dynamics. The model is learning—it simply has not had enough gradient steps to converge.

**Sample quality (Fig 9) confirms learning but plateauing.** The 1-NFE MSE plateau at ~0.7 with stable latent statistics (mean ~0, std ~0.65) shows the model generates brain-like structures but cannot refine details. The clear anatomical structures visible by epoch 240 in all 4 channels are encouraging—the model has learned the coarse latent manifold but lacks the fine-grained velocity field accuracy that would require substantially more optimization.

---

## 3. Root cause analysis: five factors behind the plateau

### The gradient update deficit is catastrophic

This is the dominant factor. The arithmetic is stark:

- **iMF on ImageNet**: 1,200,000 images ÷ 256 batch × 240 epochs = **~1,125,000 gradient updates**
- **NeuroMF**: 990 images ÷ 128 batch × 240 epochs = **~1,856 gradient updates**
- **Deficit**: ~600× fewer updates

A 178M-parameter model with 1,856 gradient updates has seen each parameter adjusted an average of 1,856 times—far below the >100K updates typically needed for deep networks to find good minima. The cosine similarity plateau at 0.25, the 2× norm ratio, and the MF loss stagnation are all consistent with a model in the earliest phase of convergence that has been evaluated as if it were trained to completion. Even perfect hyperparameters cannot compensate for a three-order-of-magnitude shortfall in optimization budget.

### EMA decay is mismatched to training length

With β = 0.9999, the EMA's effective window is 1/(1−β) = **10,000 updates**. Since total training is only ~1,920 updates, the EMA model is a heavily smoothed average over the *entire* training trajectory—including early random-initialization weights. For iMF with ~1.125M updates, β = 0.9999 gives an effective window of 10,000/1,125,000 ≈ 0.9% of training—reasonable. For NeuroMF, the window exceeds total training by 5×, meaning the EMA model **has not converged** and substantially lags the online model. Setting β = 0.999 (window = 1,000 updates) or β = 0.9995 (window = 2,000) would be appropriate for the current update count.

### Adaptive weighting may over-suppress gradients in the undertrained regime

With stop-gradient on the weight, the effective gradient is d(raw_loss)/dθ ÷ (raw_loss + c). For p = 1.0, this is equivalent to optimizing log(raw_loss)—a scale-invariant objective that bounds gradient magnitudes regardless of loss magnitude. While this is intentional and works well for iMF's million-update regime (where the model transitions from large to small losses), it means that in the early phase where raw_loss ≈ 1e7, every gradient step produces the same small relative improvement. The original MF paper found **p = 0.75 optimal for CIFAR-10** (a smaller-scale setting), while p = 1.0 was best for ImageNet's large-scale regime. NeuroMF's small-data setting is closer to CIFAR-10's optimization landscape than ImageNet's, suggesting p = 0.5–0.75 may accelerate convergence.

### The FM/MF sample ratio amplifies gradient conflict

AlphaFlow's analysis reveals that MeanFlow's loss decomposes into trajectory flow matching (TFM) and trajectory consistency (TC) terms with **strongly negatively correlated gradients**. The original MF paper's ablation shows 25% MF samples is optimal (FID 61.06 versus 63.14 for 50% MF). NeuroMF's 50/50 split doubles the proportion of conflicting-gradient MF samples relative to the reference, exacerbating the slow convergence that AlphaFlow identifies as the core MeanFlow challenge. The 75% FM ratio isn't wasted computation—it provides the flow matching supervision that stabilizes the conflicting optimization landscape.

### Using 990 of 6,471 available samples wastes data

The current configuration uses **only 15% of available FOMO60K data**. Expanding to 6,471 samples would increase gradient updates per epoch from ~8 to ~50, meaning 240 epochs would yield ~12,000 updates instead of ~1,920—a 6.5× improvement. Combined with data augmentation (left-right flipping for brain MRI doubles effective data, small intensity perturbations add further diversity), the effective dataset could reach ~13,000+ samples, yielding ~25,000 updates in 240 epochs.

---

## 4. Ranked improvement recommendations

### Rank 1: Massively increase gradient updates (expected impact: transformative)

The single most impactful change is extending training by 50–100× in gradient update count. Use all 6,471 FOMO60K samples (50 updates/epoch at batch 128), train for 2,000–4,000 epochs (100K–200K total updates). At ~8 updates/epoch currently, reaching 100K updates requires either more data, more epochs, or both. **Mathematical justification**: iMF's convergence occurs over ~1M updates; 100K updates would be 10× closer to this target while remaining computationally tractable. The compound velocity V requires accurate du/dt, which depends on the model having learned a smooth velocity field—this only emerges with sufficient optimization. **Expected effect**: cos(V, v_c) should increase from 0.25 toward 0.5–0.7; the norm ratio should decrease from 2× toward 1.2–1.5×; 1-NFE MSE should drop below 0.5. **Computational cost**: ~10–20× current wall-clock time (mitigated by per-epoch cost being the same). **Implementation**: Change `max_epochs` and `data_proportion` in config; no code changes.

### Rank 2: Fix data_proportion to 75% FM / 25% MF (expected impact: high)

Change `data_proportion` from 0.5 to 0.25 (if this parameter represents the MF fraction) or 0.75 (if it represents the FM fraction). **Mathematical justification**: AlphaFlow proves that TFM and TC gradients are strongly negatively correlated. The 75% FM ratio provides a surrogate loss for trajectory flow matching that mitigates this conflict, acting as a critical stabilizer. Original MF ablation shows this ratio achieves the best FID. **Expected effect**: Faster loss decrease for the MF component; more stable gradient directions; improved compound velocity alignment. **Implementation**: Single config change, zero computational cost.

### Rank 3: Reduce EMA decay to match update count (expected impact: high for evaluation)

Set β = 0.999 for runs with <10K total updates, or β = 0.9995 for runs with 10K–50K updates. Keep β = 0.9999 only if training reaches 100K+ updates. **Mathematical justification**: The EMA's effective window N_eff = 1/(1−β) should be 1–10% of total training updates. Currently N_eff = 10,000 >> 1,920 total updates, meaning the EMA model averages over the entire trajectory including initialization. **Expected effect**: EMA model quality will jump immediately—current EMA evaluations likely substantially underestimate the online model's true capability. **Implementation**: Single config change.

### Rank 4: Reduce adaptive weighting aggressiveness (expected impact: medium-high)

Reduce norm_p from 1.0 to **0.5–0.75**. The CIFAR-10 setting (p = 0.75) is a better analogue for NeuroMF's small-data regime than ImageNet's p = 1.0. At p = 0.5 (Pseudo-Huber-like), the gradient scaling is dL/dθ / √(L + c) rather than dL/dθ / (L + c), providing stronger gradient signal when the model is far from convergence. Alternatively, increase norm_eps from 0.01 to **1.0** to add a constant floor to gradient magnitude. **Expected effect**: Faster initial convergence, particularly for the MF loss component. May require gradient clipping adjustment. **Implementation**: Config change only.

### Rank 5: Add conditioning on absolute time t (expected impact: medium)

Switch from h-only conditioning to (t, h) dual conditioning. The original MF paper shows this improves FID from 63.13 to **61.06** (3.3% relative improvement). The network needs absolute time t to know the noise level of z_t (since z_t = (1−t)·z_0 + t·ε varies with t). With h-only conditioning, the model cannot distinguish between u(z_t, r=0, t=0.5) and u(z_t, r=0.3, t=0.8) despite z_t having very different noise levels. **Implementation**: Modify the timestep embedding to accept two inputs. For MONAI's UNet, embed t and h separately through two timestep MLP heads and sum the embeddings before injection into residual blocks. Moderate code change (~50 lines).

### Rank 6: Enable data augmentation (expected impact: medium)

Enable left-right flipping (brain MRI has approximate bilateral symmetry), which doubles effective dataset size to ~13K samples. Add small random intensity scaling (±5%) and minor affine perturbations (±2° rotation, ±2% translation). **Expected effect**: ~2× effective data, improved generalization, reduced overfitting risk with longer training. **Implementation**: Add transforms to the data pipeline; standard in MONAI.

### Rank 7: Increase v-head capacity or use boundary condition (expected impact: medium)

Two options exist. **Option A (recommended)**: Use the boundary condition variant v_θ = u_θ(z_t, t, t), eliminating the v-head entirely. iMF Table 1a shows this achieves FID 29.42 versus 30.76 for the aux head without CFG—the boundary condition is actually *better* in the simpler regime. This removes 221K parameters and the auxiliary loss term, simplifying training. **Option B**: Scale the v-head to 3–4 ResBlocks (~1–2M params). While still far below iMF's aux head, this provides better tangent estimation for the JVP. **Implementation**: Option A requires modifying the forward pass to evaluate the main network with r=t for tangent computation (~20 lines). Option B requires adding ResBlocks to the v-head module.

### Rank 8: Consider two-stage training (expected impact: medium, longer-term)

The Decoupled MeanFlow (DMF) approach trains a standard flow matching model first, then converts to MeanFlow and fine-tunes. Pre-trained flow models adapt rapidly to flow maps because the FM objective establishes a good velocity field that the MF objective can then refine for one-step generation. This is especially effective for limited compute budgets and would likely produce better results than training MeanFlow from scratch with only ~2K gradient updates. Similarly, the "Understanding, Accelerating, and Improving MeanFlow Training" paper shows that prioritizing small temporal gaps early and gradually increasing gap size achieves equivalent quality with **2.5× fewer iterations**.

### Rank 9: Adjust learning rate for long training (expected impact: low-medium)

For extended training runs (2,000+ epochs), add cosine decay from 1e-4 to 1e-6 over the full schedule. The constant LR matches iMF's ablation setting but was designed for 240 epochs at ~1M updates. With 100K+ updates over thousands of epochs, cosine decay prevents the model from oscillating around the loss minimum in later training. **Implementation**: Config change.

---

## 5. Concrete parameter recommendations for the next training run

The following configuration incorporates all high-impact changes identified above, ordered by priority. Bold values indicate changes from the current configuration.

| Parameter | Current value | Recommended value | Rationale |
|---|---|---|---|
| Dataset samples | 990 | **6,471** (full FOMO60K) | 6.5× more data and gradient updates |
| Augmentation | Disabled | **Left-right flip + intensity jitter (±5%)** | ~2× effective data |
| Total epochs | 240 | **3,000–5,000** | Target ~150K–250K gradient updates |
| data_proportion (MF fraction) | 0.50 | **0.25** | Match original MF optimal; reduce gradient conflict |
| norm_p | 1.0 | **0.5** | Less aggressive gradient suppression for small-data regime |
| norm_eps | 0.01 | **1.0** | Floor on gradient magnitude; prevents over-suppression |
| EMA decay | 0.9999 | **0.9995** (or 0.999 for shorter runs) | Match N_eff to actual update count |
| v-head | 1 ResBlock (221K params) | **Boundary condition: v = u(z, t, t)** | Zero extra params; competitive with aux head; simpler |
| Conditioning | h = t−r only | **(t, t−r)** dual embedding | Best setting per MF Table 1c; preserves noise-level info |
| LR schedule | Constant 1e-4 | **Cosine decay 1e-4 → 1e-6** over full run | Prevents late-training oscillation |
| Optimizer | AdamW(0.9, 0.95), wd=0 | AdamW(0.9, 0.95), **wd=0.01** | Mild regularization for longer training |
| Time sampling | logit-normal(−0.4, 1.0) | logit-normal(−0.4, 1.0) | Keep; matches iMF |
| t_min clamp | 0.05 | **0.01** | Expand learnable time range; monitor for instability |
| Batch size | 128 effective | 128 effective | Keep; matched to memory constraints |
| Gradient clipping | Current value | **Monitor and adjust** if changing norm_p | Lower p may increase gradient variance |

**Estimated gradient updates**: 6,471 samples ÷ 128 batch ≈ 50 updates/epoch × 4,000 epochs = **~200,000 updates**. This is ~100× more than current (~1,920) and ~18% of iMF's ImageNet budget (~1.125M). Given the substantially lower-dimensional generation task (442K latent dims versus ImageNet's 4×32×32 = 4,096 in latent space, but targeting a narrower data distribution of brain anatomy versus 1,000 ImageNet classes), 200K updates should be sufficient for meaningful convergence.

**Implementation priority order**: Apply changes in this sequence to isolate effects—(1) expand to full dataset, (2) fix data_proportion to 0.25, (3) increase epochs to 3,000+, (4) adjust EMA to 0.9995, (5) switch to boundary condition v-head, (6) add (t, h) conditioning, (7) tune norm_p/norm_eps, (8) enable augmentation. Changes 1–4 are pure config modifications requiring zero code changes. Changes 5–6 require moderate code modifications. Changes 7–8 are config-only but benefit from ablation.

---

## Conclusion

The x-prediction arm is not failing—it is severely undertrained. The core algorithm is correctly implemented and producing the expected directional learning signal, as evidenced by cos(V, v_c) = 0.25 in a 442K-dimensional space and visible brain anatomy in generated samples. The primary bottleneck is a **~600× gradient update deficit** relative to the reference iMF training, compounded by a suboptimal FM/MF ratio that amplifies the gradient conflict AlphaFlow identified, an EMA decay that has not converged, and several minor configuration mismatches.

The single most important insight from this analysis is that the adaptive weighting is not causing catastrophic gradient vanishing (the stop-gradient prevents the eps/L² scaling the user hypothesized), but the combination of p = 1.0 with very few gradient updates does create unnecessarily slow convergence. Reducing p to 0.5 and training for 100–200× longer should unlock substantially better performance.

A secondary insight: **the boundary condition variant** v_θ = u_θ(z_t, t, t) is likely preferable to the current undersized v-head. It adds zero parameters, matches iMF's ablation results, and avoids the capacity mismatch between a 221K-parameter auxiliary head and a 178M-parameter backbone. The v-head's cos(v_tilde, v_c) = 0.42 already exceeds the compound velocity's 0.25, suggesting the tangent quality is not the primary bottleneck—but eliminating the auxiliary loss term simplifies the training objective and directs all gradient signal toward the main network.

The path to publication-quality results requires, at minimum, expanding to the full 6,471-sample dataset with augmentation, training for 3,000+ epochs (~200K gradient updates), and correcting the FM/MF ratio to 75/25. These changes alone should yield a substantial improvement over the current plateau. Further gains from conditioning, adaptive weighting, and curriculum-based training strategies can be explored as subsequent ablations.