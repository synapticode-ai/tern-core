"""
Base class for architecture adapters.

Adapters translate between a HuggingFace model's weight naming
conventions and tern-core's internal conversion schema.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class WeightClassification:
    """Classification of a single weight tensor for conversion."""

    name: str
    canonical_name: str  # name after adapter normalization
    category: str  # "ternary_eligible", "fp16_retain", "skip"
    reason: str  # why this classification was chosen
    component: str  # "language", "vision", "audio", "projector"
    expert_idx: Optional[int] = None
    """Integer expert index for MoE expert weights (e.g., 0–7 for
    Mixtral's 8 experts). ``None`` = not an expert weight."""
    attention_type: Optional[Literal["full", "linear"]] = None
    """Attention layer type for hybrid architectures. ``None`` =
    adapter does not distinguish (default for non-hybrid models)."""


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
    expert_pattern: Optional[re.Pattern] = None
    """Regex with named group ``"expert_idx"`` capturing the expert
    integer for MoE models. ``None`` = non-MoE adapter. When set,
    the base :meth:`ArchitectureAdapter._extract_expert_idx`
    helper uses this to populate
    :attr:`WeightClassification.expert_idx`."""
    attention_type_pattern: Optional[re.Pattern] = None
    """Regex matching weight names that belong to linear-attention
    layers (DeltaNet, Mamba, RWKV) in hybrid architectures.
    ``None`` = standard attention only. When set, the base
    :meth:`ArchitectureAdapter._detect_attention_type` helper
    tags weights with ``attention_type="linear"`` if they match,
    ``"full"`` otherwise."""


class ArchitectureAdapter:
    """Base class for architecture-specific conversion adapters.

    Subclasses must implement:
    - info() -> AdapterInfo
    - classify_weight(name, shape) -> WeightClassification
    - normalize_name(name) -> str

    Subclasses may override the ``_VISION_PATTERNS`` /
    ``_AUDIO_PATTERNS`` / ``_PROJECTOR_PATTERNS`` class attributes
    to drive the default :meth:`_detect_component` behaviour.
    Empty defaults make text-only adapters trivial.
    """

    _VISION_PATTERNS: list[str] = []
    _AUDIO_PATTERNS: list[str] = []
    _PROJECTOR_PATTERNS: list[str] = []

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

    def _detect_component(self, name: str) -> str:
        """Return component bucket for a weight name.

        Default 4-bucket vocab: ``"vision"`` | ``"audio"`` |
        ``"projector"`` | ``"language"``. Subclasses override the
        ``_VISION_PATTERNS`` / ``_AUDIO_PATTERNS`` /
        ``_PROJECTOR_PATTERNS`` class attributes; the method body
        stays shared.

        Pattern scan priority: vision → audio → projector. If a
        name matches patterns in multiple buckets, the first
        match wins. Returns ``"language"`` if no pattern matches
        (text-only fallback).
        """
        for pattern in self._VISION_PATTERNS:
            if re.search(pattern, name):
                return "vision"
        for pattern in self._AUDIO_PATTERNS:
            if re.search(pattern, name):
                return "audio"
        for pattern in self._PROJECTOR_PATTERNS:
            if re.search(pattern, name):
                return "projector"
        return "language"

    def classify_all(
        self,
        weight_shapes: dict[str, list[int]],
    ) -> dict[str, WeightClassification]:
        """Classify every weight in the dict.

        Concrete default — calls :meth:`classify_weight` on each
        name paired with its shape. Not expected to be overridden;
        ``classify_weight`` is the per-weight policy hook.
        """
        return {
            name: self.classify_weight(name, shape)
            for name, shape in weight_shapes.items()
        }

    def get_ternary_eligible(
        self,
        weight_shapes: dict[str, list[int]],
    ) -> list[str]:
        """Return weight names that classify as ``"ternary_eligible"``.

        Concrete default — filters :meth:`classify_all` results by
        category. Not expected to be overridden.
        """
        classifications = self.classify_all(weight_shapes)
        return [
            name for name, cls in classifications.items()
            if cls.category == "ternary_eligible"
        ]

    def _extract_expert_idx(self, name: str) -> Optional[int]:
        """Extract MoE expert index from a weight name.

        Uses ``self.info().expert_pattern``. Returns ``None`` if no
        pattern is declared on this adapter (non-MoE) or if the
        pattern does not match this name. The pattern must declare
        a named group ``"expert_idx"`` capturing the integer.

        Concrete default — not expected to be overridden. MoE
        adapters declare ``expert_pattern`` in ``info()`` and
        inherit this helper.
        """
        pattern = self.info().expert_pattern
        if pattern is None:
            return None
        match = pattern.search(name)
        if match is None:
            return None
        try:
            return int(match.group("expert_idx"))
        except (IndexError, ValueError):
            return None

    def _detect_attention_type(
        self,
        name: str,
    ) -> Optional[Literal["full", "linear"]]:
        """Detect attention layer type for hybrid architectures.

        Uses ``self.info().attention_type_pattern``. Returns
        ``None`` if no pattern is declared (adapter does not
        distinguish), ``"linear"`` if the name matches the
        linear-attention pattern, ``"full"`` otherwise.

        Concrete default — not expected to be overridden. Hybrid
        adapters declare ``attention_type_pattern`` in ``info()``
        and inherit this helper.
        """
        pattern = self.info().attention_type_pattern
        if pattern is None:
            return None
        if pattern.search(name):
            return "linear"
        return "full"
