# Ablation: x-Prediction vs. u-Prediction (with JVP/Checkpointing Tradeoff)

**Phase:** Pre-Phase 6 — Critical Design Decision
**Hardware:** Picasso A100 40GB, multi-GPU DDP per arm
**Max wall time:** 5 days (x-pred) / 3 days (u-pred)

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

| Arm | Prediction | JVP method | Grad checkpoint | Flash attention | v-head | h-cond | Batch/GPU | GPUs | Accum | Eff. batch |
|-----|-----------|------------|----------------|----------------|--------|--------|-----------|------|-------|------------|
| **A** | x-pred | Exact (`torch.func.jvp`) | OFF | OFF | ON | ON | 2 | 4 | 16 | 128 |
| **B** | u-pred | FD ($h=10^{-3}$, fp32 sub) | ON | ON | ON | ON | 16 | 2 | 4 | 128 |

Both arms use the iMF dual-head architecture (`use_v_head=true`, `conditioning_mode=h`,
`v_head_num_res_blocks=1`) inherited from the Picasso overlay. The v-head provides the
JVP tangent in both arms — this isolates the prediction type + JVP method as the primary
variable while keeping tangent quality consistent.

Both arms use `accumulate_grad_batches` to match the same effective batch size
of 128. Arm A requires 4 GPUs because `batch_size=2` is needed on A100 40GB without
gradient checkpointing (exact JVP doubles activation memory).

## Confounds and Interpretation

This ablation conflates prediction type with JVP method and memory strategy:
- Arm A: x-pred + exact JVP + no grad checkpointing + no flash attention
- Arm B: u-pred + FD-JVP + grad checkpointing + flash attention

The v-head and h-conditioning are held constant across both arms. If Arm A
wins, we know x-prediction provides sufficient quality gain to justify the
memory cost. If Arm B wins, we adopt u-pred + FD-JVP as the production
configuration.

Note: x-pred + FD-JVP is known to be numerically unstable (1/t amplification
in finite differences, see Phase 4f). A clean disentanglement is deferred to
Phase 6.

## Expected Outcome

Based on pMF Table 2 and our bottleneck ratio analysis, we expect Arm A
(x-pred) to produce lower validation loss and better 1-NFE sample quality,
at the cost of ~4× lower throughput (steps/sec) due to larger batch
accumulation and no checkpointing.

## References

- Lu et al., "One-step Latent-free Image Generation with Pixel Mean Flows" (2026), §4, §6.1
- Geng et al., "Improved Techniques for Training MeanFlow" (2025), iMF formulation
- Zhu & He, "On the Mean Field Theory of MeanFlow" (2025), original MeanFlow