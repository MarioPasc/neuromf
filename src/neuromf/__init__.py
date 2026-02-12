"""NeuroMF: Latent MeanFlow for 3D Brain MRI Synthesis.

Trains a MeanFlow model in the latent space of a frozen MAISI 3D VAE to achieve
1-step (1-NFE) generation of 128^3 brain MRI volumes, with per-channel Lp loss
and LoRA fine-tuning for rare epilepsy pathology (FCD).
"""
