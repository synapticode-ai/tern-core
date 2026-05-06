"""
Architecture adapters for tern-core conversion pipeline.

Each adapter maps a model architecture's HuggingFace weight names,
layer structure, and protection requirements to tern-core's internal
conversion schema.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from terncore.adapters.base import ArchitectureAdapter

_REGISTRY: dict[str, type["ArchitectureAdapter"]] = {}

_KNOWN_ADAPTERS = ["llama", "gemma3", "gemma4", "phi3", "qwen3_moe"]


def register(name: str):
    """Decorator to register an adapter class by name."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_adapter(name: str) -> "ArchitectureAdapter":
    """Instantiate a registered adapter by name.

    Triggers lazy imports so that adapter modules are only loaded
    when requested. Adapter names are matched case-insensitively
    against ``_KNOWN_ADAPTERS``; on first request, the matching
    module is imported and self-registers via the ``@register``
    decorator.
    """
    key = name.lower()
    if key not in _REGISTRY:
        if key not in _KNOWN_ADAPTERS:
            raise ValueError(
                f"Unknown adapter: {name}. Known: {_KNOWN_ADAPTERS}"
            )
        importlib.import_module(f"terncore.adapters.{key}")
    cls = _REGISTRY[key]
    return cls()


__all__ = ["register", "get_adapter", "_KNOWN_ADAPTERS"]
