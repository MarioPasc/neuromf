"""Extensible competitor model infrastructure for evaluation.

Provides an abstract base class, a model registry, shared I/O, and a
style system so that adding a new competitor model requires only:

1. Subclass :class:`BaseCompetitorGenerator` in a new file.
2. Decorate with ``@register_competitor("name")``.
3. Import the new module here to trigger registration.

See ``docs/adding_competitors.md`` (generated) for the full walkthrough.
"""

from neuromf.competitors.base import BaseCompetitorGenerator
from neuromf.competitors.io import write_volumes_h5
from neuromf.competitors.registry import (
    CompetitorRegistry,
    register_competitor,
)
from neuromf.competitors.styles import (
    DEFAULT_METHOD_STYLES,
    get_method_style,
    register_style,
)

# Import concrete implementations to trigger auto-registration.
# When adding a new competitor, add its import here.
import neuromf.competitors.motfm_gen  # noqa: F401
import neuromf.competitors.ddpm_gen  # noqa: F401

__all__ = [
    "BaseCompetitorGenerator",
    "CompetitorRegistry",
    "register_competitor",
    "write_volumes_h5",
    "get_method_style",
    "register_style",
    "DEFAULT_METHOD_STYLES",
]
