"""
Architecture adapters for tern-core conversion pipeline.

Each adapter maps a model architecture's HuggingFace weight names,
layer structure, and protection requirements to tern-core's internal
conversion schema.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terncore.adapters.base import ArchitectureAdapter

_REGISTRY: dict[str, type["ArchitectureAdapter"]] = {}


def register(name: str):
    """Decorator to register an adapter class by name."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_adapter(name: str) -> "ArchitectureAdapter":
    """Instantiate a registered adapter by name.

    Triggers lazy imports so that adapter modules are only loaded
    when requested.
    """
    key = name.lower()
    if key not in _REGISTRY:
        # Trigger lazy registration via import
        if key == "gemma4":
            import terncore.adapters.gemma4  # noqa: F401
        elif key == "llama":
            import terncore.adapters.llama  # noqa: F401
        elif key == "gemma3":
            import terncore.adapters.gemma3  # noqa: F401
        else:
            raise ValueError(
                f"Unknown adapter '{name}'. "
                f"Available: {sorted(_REGISTRY.keys()) or ['gemma4', 'llama']}"
            )
    cls = _REGISTRY[key]
    return cls()


__all__ = ["register", "get_adapter"]
