"""MOTFM inference wrapper — backward-compatible re-export.

The canonical implementation now lives in
:mod:`neuromf.competitors.motfm_gen`.  This file preserves the
``MOTFMGenerator`` import name so that existing scripts and SLURM
workers continue to work unchanged.

Usage (unchanged)::

    from motfm_wrapper import MOTFMGenerator
    gen = MOTFMGenerator(config_path, checkpoint_path, device)
"""

from neuromf.competitors.motfm_gen import MOTFMCompetitorGenerator as MOTFMGenerator

__all__ = ["MOTFMGenerator"]
