# Ablation: x-Prediction vs. u-Prediction (with JVP/Checkpointing Tradeoff)

**Phase:** Pre-Phase 6 — Critical Design Decision
**Hardware:** Picasso A100 40GB, single GPU per arm
**Max wall time:** 7 days per arm (Picasso limit)

---

## Scientific Question

Does the x-prediction manifold advantage (Lu et al., 2026; pMF) transfer to
MAISI VAE-GAN latent space ($\mathbf{z} \in \mathbb{R}^{4 \times 48^3}$, $D = 442{,}368$)?

The pMF paper demonstrates that x-prediction is superior to u-prediction when
the target lives on a low-dimensional manifold and the network bottleneck ratio
$d_{\text{input}} / d_{\text{bottleneck}} > 1$. For our 3D UNet:

$$\frac{d_{\text{input}}}{d_{\text{bottleneck}}} = \frac{4 \times 48^3}{512 \times 6^3} = \frac{442{,}368}{110{,}592} \approx 4.0$$

This ratio $> 1$ suggests x-prediction should be advantageous. However, the MAISI
VAE has already compressed brain MRI onto a lower-dimensional latent manifold
$\mathcal{M}_z$, which may attenuate the gap (see `docs/main/methodology_expanded.md` §4.2.2).

## Technical Constraint

In PyTorch, `torch.func.jvp` (exact forward-mode AD) is incompatible with
`torch.utils.checkpoint.checkpoint` (activation checkpointing). This forces a
practical tradeoff:

| Arm | Prediction | JVP method | Grad checkpoint | Flash attention | Batch/GPU |
|-----|-----------|------------|----------------|----------------|-----------|
| **A** | x-pred | Exact (`torch.func.jvp`) | OFF | OFF | 4 |
| **B** | u-pred | FD ($h=10^{-3}$, fp32 sub) | ON | ON | 16 |

Both arms use `accumulate_grad_batches` to match the same effective batch size
of 64. Both arms train on a single A100 GPU.

## Confounds and Interpretation

This ablation conflates prediction type with memory strategy. If Arm A wins,
we know x-prediction provides sufficient quality gain to justify the memory
cost. If Arm B wins, we adopt u-pred + FD-JVP as the production configuration,
consistent with our current Phase 4 setup.

A clean disentanglement (x-pred + FD-JVP or u-pred + exact JVP) is possible
but would require additional runs and is deferred to Phase 6.

## Expected Outcome

Based on pMF Table 2 and our bottleneck ratio analysis, we expect Arm A
(x-pred) to produce lower validation loss and better 1-NFE sample quality,
at the cost of ~4× lower throughput (steps/sec) due to larger batch
accumulation and no checkpointing.

## References

- Lu et al., "One-step Latent-free Image Generation with Pixel Mean Flows" (2026), §4, §6.1
- Geng et al., "Improved Techniques for Training MeanFlow" (2025), iMF formulation
- Zhu & He, "On the Mean Field Theory of MeanFlow" (2025), original MeanFlow