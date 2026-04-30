"""
Gemma 3 architecture adapter for tern-core conversion pipeline.

Maps Google Gemma 3 (``Gemma3ForConditionalGeneration``) HuggingFace
weight names to tern-core's internal weight schema. Handles the
multimodal variant (vision encoder) by flagging encoder weights for
FP16 retention.

Architecture notes:
- Multimodal: vision encoder present, text-only language backbone
- Sliding window attention: pattern of N (sliding_window_pattern=6
  means every 6th layer is global attention, rest are sliding)
- Activation: gelu_pytorch_tanh (not SiLU like Llama)
- RMSNorm with +1.0 shift (unlike Gemma 4 which uses +0.0)
- Language weights prefixed with ``language_model.`` in multimodal
- No MoE, no per-layer embeddings, no layer_scalar (unlike Gemma 4)

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import re
from typing import Optional

from terncore.adapters import register
from terncore.adapters.base import (
    AdapterInfo,
    ArchitectureAdapter,
    WeightClassification,
)

_BLOCK_PATTERN = re.compile(r"\.layers\.(\d+)\.")

_PROJ_PRIORITY = [
    "v_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_ALWAYS_PROTECTED = (
    "embed_tokens",
    "lm_head",
    "norm",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "classifier",
)


@register("gemma3")
class Gemma3Adapter(ArchitectureAdapter):
    """Architecture adapter for Gemma 3 (text + vision multimodal).

    Weight classification:
    1. Vision encoder weights → FP16-retain.
    2. Multi-modal projector weights → FP16-retain.
    3. Embeddings, norms, LM head → FP16-retain.
    4. 1-D weights (biases, scalars) → FP16-retain.
    5. All 2-D weights in transformer blocks → ternary-eligible.
    """

    _VISION_PATTERNS: list[str] = [
        "vision_tower",
        "vision_model",
    ]
    _AUDIO_PATTERNS: list[str] = []
    _PROJECTOR_PATTERNS: list[str] = [
        "multi_modal_projector",
        "multimodal_projector",
    ]

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="gemma3",
            architecture="Gemma3ForConditionalGeneration",
            model_type="gemma3",
            description=(
                "Google Gemma 3 adapter — supports text + vision. "
                "Sliding window attention, gelu_pytorch_tanh activation. "
                "Flags vision encoder for FP16 retention."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=True,
            multimodal_components=["vision", "projector"],
        )

    def normalize_name(self, name: str) -> str:
        if "language_model." in name:
            name = name.replace("language_model.", "", 1)
        return name

    def classify_weight(
        self,
        name: str,
        shape: Optional[list[int]] = None,
    ) -> WeightClassification:
        canonical = self.normalize_name(name)
        component = self._detect_component(name)

        if component == "vision":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Vision encoder — modality encoders are FP16-retain",
                component=component,
            )

        if component == "projector":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Multi-modal projector — cross-modal alignment is precision-sensitive",
                component=component,
            )

        canonical_lower = canonical.lower()
        for pattern in _ALWAYS_PROTECTED:
            if pattern in canonical_lower:
                return WeightClassification(
                    name=name,
                    canonical_name=canonical,
                    category="fp16_retain",
                    reason=f"Protected pattern: '{pattern}'",
                    component=component,
                )

        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (bias or scalar parameter)",
                component=component,
            )

        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in transformer block",
            component=component,
        )

    def classify_all(
        self,
        weight_names: dict[str, list[int]],
    ) -> dict[str, WeightClassification]:
        return {
            name: self.classify_weight(name, shape)
            for name, shape in weight_names.items()
        }

    def get_protection_list(
        self,
        weight_names: dict[str, list[int]],
    ) -> list[str]:
        return [
            name for name, cls in self.classify_all(weight_names).items()
            if cls.category == "fp16_retain"
        ]

    def get_ternary_eligible(
        self,
        weight_names: dict[str, list[int]],
    ) -> list[str]:
        return [
            name for name, cls in self.classify_all(weight_names).items()
            if cls.category == "ternary_eligible"
        ]
