"""Visual style registry for method comparison figures.

Provides a consistent color/marker/linestyle for each model name across
all figures.  Known models get fixed styles; unknown models are assigned
from a colorblind-safe fallback palette.

To register a style for a new competitor::

    from neuromf.competitors.styles import register_style
    register_style("ScoreSDE", {"color": "#228833", "marker": "D", "ls": "-."})

Or rely on the automatic fallback (next unused color from Paul Tol Muted).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Paul Tol Bright palette (colorblind-safe)
_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
}

_MARKERS = ["D", "v", "P", "X", "h", "*", "p"]
_LINESTYLES = ["-.", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]

# Fixed styles for the core models
DEFAULT_METHOD_STYLES: dict[str, dict[str, Any]] = {
    "NeuroiMF": {"color": _BRIGHT["blue"], "marker": "o", "ls": "-"},
    "MOTFM": {"color": _BRIGHT["red"], "marker": "^", "ls": "--"},
    "DDPM": {"color": _BRIGHT["grey"], "marker": "s", "ls": ":"},
}

# Fallback palette for unknown models (Paul Tol Muted, excluding already used)
_FALLBACK_COLORS = [
    _BRIGHT["green"],
    _BRIGHT["yellow"],
    _BRIGHT["purple"],
    _BRIGHT["cyan"],
    "#CC6677",  # rose (Muted)
    "#332288",  # indigo (Muted)
    "#DDCC77",  # sand (Muted)
    "#44AA99",  # teal (Muted)
]

_fallback_idx: int = 0


def get_method_style(name: str) -> dict[str, Any]:
    """Return the visual style dict for a model name.

    Known models return their fixed style.  Unknown models are assigned
    a color/marker/linestyle from the fallback palette (persisted for the
    session so the same model always gets the same style).

    Args:
        name: Model display name (e.g. ``"ScoreSDE"``).

    Returns:
        Dict with ``color``, ``marker``, ``ls`` keys.
    """
    if name in DEFAULT_METHOD_STYLES:
        return DEFAULT_METHOD_STYLES[name]

    # Auto-assign and persist for the session
    global _fallback_idx
    idx = _fallback_idx
    style = {
        "color": _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)],
        "marker": _MARKERS[idx % len(_MARKERS)],
        "ls": _LINESTYLES[idx % len(_LINESTYLES)],
    }
    DEFAULT_METHOD_STYLES[name] = style
    _fallback_idx += 1
    logger.debug("Auto-assigned style for '%s': %s", name, style)
    return style


def register_style(name: str, style: dict[str, Any]) -> None:
    """Explicitly register a visual style for a model.

    Args:
        name: Model display name.
        style: Dict with ``color``, ``marker``, ``ls`` keys.
    """
    DEFAULT_METHOD_STYLES[name] = style
    logger.debug("Registered style for '%s': %s", name, style)
