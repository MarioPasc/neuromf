"""DDPM inference wrapper — backward-compatible re-export.

The canonical implementation now lives in
:mod:`neuromf.competitors.ddpm_gen`.  This file preserves the
``DDPMGenerator`` import name so that existing scripts and SLURM
workers continue to work unchanged.

Usage (unchanged)::

    from ddpm_wrapper import DDPMGenerator
    gen = DDPMGenerator(config_path, checkpoint_path, device)
"""

from neuromf.competitors.ddpm_gen import DDPMCompetitorGenerator as DDPMGenerator

__all__ = ["DDPMGenerator"]
