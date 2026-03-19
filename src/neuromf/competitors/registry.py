"""Competitor model registry.

Provides a simple decorator-based registry so that new competitor models
are discoverable by name at runtime (e.g. ``--model motfm``).

Example::

    @register_competitor("scoresde")
    class ScoreSDEGenerator(BaseCompetitorGenerator):
        ...

    # Later:
    cls = CompetitorRegistry.get("scoresde")
    gen = cls(config_path=..., checkpoint_path=..., device=...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuromf.competitors.base import BaseCompetitorGenerator

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[BaseCompetitorGenerator]] = {}


class CompetitorRegistry:
    """Central lookup for registered competitor generators."""

    @staticmethod
    def get(name: str) -> type[BaseCompetitorGenerator]:
        """Look up a generator class by its registered name.

        Args:
            name: Case-insensitive model name (e.g. ``"motfm"``).

        Returns:
            The generator class.

        Raises:
            KeyError: If the name is not registered.
        """
        key = name.lower()
        if key not in _REGISTRY:
            available = ", ".join(sorted(_REGISTRY.keys()))
            raise KeyError(
                f"Unknown competitor '{name}'. "
                f"Registered: [{available}]. "
                f"Register new models with @register_competitor('{name}')."
            )
        return _REGISTRY[key]

    @staticmethod
    def list_names() -> list[str]:
        """Return sorted list of registered competitor names."""
        return sorted(_REGISTRY.keys())

    @staticmethod
    def register(name: str, cls: type[BaseCompetitorGenerator]) -> None:
        """Register a generator class under the given name.

        Args:
            name: Case-insensitive model name.
            cls: Generator class (must subclass ``BaseCompetitorGenerator``).
        """
        key = name.lower()
        if key in _REGISTRY:
            logger.warning(
                "Overwriting competitor '%s' (%s -> %s)",
                key,
                _REGISTRY[key].__name__,
                cls.__name__,
            )
        _REGISTRY[key] = cls
        logger.debug("Registered competitor '%s' -> %s", key, cls.__name__)


def register_competitor(name: str):
    """Class decorator to register a competitor generator.

    Args:
        name: Case-insensitive model name used in CLI ``--model`` flag.

    Returns:
        The original class, unchanged.

    Example::

        @register_competitor("motfm")
        class MOTFMCompetitorGenerator(BaseCompetitorGenerator):
            ...
    """

    def decorator(cls: type[BaseCompetitorGenerator]) -> type[BaseCompetitorGenerator]:
        CompetitorRegistry.register(name, cls)
        return cls

    return decorator
