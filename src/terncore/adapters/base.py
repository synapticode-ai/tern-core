"""
Base class for architecture adapters.

Adapters translate between a HuggingFace model's weight naming
conventions and tern-core's internal conversion schema.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class WeightClassification:
    """Classification of a single weight tensor for conversion."""

    name: str
    canonical_name: str  # name after adapter normalization
    category: str  # "ternary_eligible", "fp16_retain", "skip"
    reason: str  # why this classification was chosen
    component: str  # "language", "vision", "audio", "projector"


@dataclass
class AdapterInfo:
    """Metadata about an architecture adapter."""

    name: str
    architecture: str  # HF architecture class name
    model_type: str  # HF model_type field
    description: str
    block_pattern: re.Pattern
    projection_priority: list[str]
    protection_patterns: list[str]
    multimodal: bool = False
    multimodal_components: list[str] = field(default_factory=list)


class ArchitectureAdapter:
    """Base class for architecture-specific conversion adapters.

    Subclasses must implement:
    - info() -> AdapterInfo
    - classify_weight(name, shape) -> WeightClassification
    - normalize_name(name) -> str
    """

    def info(self) -> AdapterInfo:
        """Return adapter metadata."""
        raise NotImplementedError

    def classify_weight(
        self,
        name: str,
        shape: Optional[list[int]] = None,
    ) -> WeightClassification:
        """Classify a weight tensor for conversion.

        Args:
            name: HuggingFace weight name.
            shape: Tensor shape (if known).

        Returns:
            WeightClassification with category and reason.
        """
        raise NotImplementedError

    def normalize_name(self, name: str) -> str:
        """Strip architecture-specific prefixes to get a canonical name.

        For multimodal models, this strips the ``language_model.`` prefix
        from language weights so the downstream pipeline sees the same
        names as a text-only model.
        """
        raise NotImplementedError

    def is_block_weight(self, name: str) -> bool:
        """Check if a weight belongs to a transformer block."""
        return self.info().block_pattern.search(name) is not None

    def block_index(self, name: str) -> Optional[int]:
        """Extract the transformer block index from a weight name."""
        m = self.info().block_pattern.search(name)
        return int(m.group(1)) if m else None

    def projection_priority(self) -> list[str]:
        """Return projection types ordered by ternary tolerance."""
        return self.info().projection_priority
