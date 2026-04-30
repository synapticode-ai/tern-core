"""
Gemma 4 architecture adapter for tern-core conversion pipeline.

Maps Google Gemma 4 (``Gemma4ForConditionalGeneration``) HuggingFace
weight names to tern-core's internal weight schema.  Handles the
multimodal E4B variant (vision + audio encoders) by flagging encoder
weights for FP16 retention.

Architecture reference: llama.cpp ``convert_hf_to_gguf.py`` Gemma4Model.

Key differences from Llama:
- Multimodal weights prefixed with ``model.language_model.``
- Vision tower: ``model.vision_tower.*`` → FP16-retain
- Audio tower: ``model.audio_tower.*`` → FP16-retain
- Multi-modal projector: ``multi_modal_projector.*`` → FP16-retain
- Shared KV layers (``num_kv_shared_layers``)
- Mixed attention: sliding_attention vs full_attention per layer
- MoE support: ``experts.gate_up_proj``, ``experts.down_proj``,
  ``router.proj``
- Extra norms: ``pre_feedforward_layernorm_2``,
  ``post_feedforward_layernorm_1/2``, ``layer_scalar``
- RMSNorm does NOT add +1.0 shift (unlike Gemma 3)

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

# Block pattern matches both bare and language_model-prefixed names.
# Examples:
#   model.layers.0.self_attn.q_proj.weight
#   language_model.model.layers.31.mlp.gate_proj.weight
_BLOCK_PATTERN = re.compile(r"\.layers\.(\d+)\.")

# Projection types ordered by empirical ternary tolerance (most → least).
# Gemma 4 uses the same projection names as Llama for dense layers,
# plus MoE expert projections.
_PROJ_PRIORITY = [
    "v_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # MoE expert projections (treat as less tolerant — larger tensors)
    "gate_up_proj",  # fused gate+up for experts
    "down_proj",     # expert down projection (same name, inside .experts.)
]

# Patterns for weights that must always stay in FP16.
_ALWAYS_PROTECTED = (
    "embed_tokens",
    "lm_head",
    "norm",           # catches model.norm, input_layernorm, etc.
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "classifier",
    "output",         # HF output projection (sometimes used instead of lm_head)
    "layer_scalar",   # Gemma 4 per-layer scalar
    "per_dim_scale",  # Gemma 4 per-dimension attention scaling
    "router",         # MoE router weights — keep precise
    "embed_tokens_per_layer",  # Gemma per-layer token embeddings
    "per_layer_model_projection",
    "per_layer_projection_norm",
    "per_layer_input_gate",
    "per_layer_projection",
)


@register("gemma4")
class Gemma4Adapter(ArchitectureAdapter):
    """Architecture adapter for Gemma 4 (text-only and E4B multimodal).

    Weight classification rules:
    1. Vision/audio encoder weights → FP16-retain (never ternary-compress
       modality encoders — information-dense perceptual features).
    2. Multi-modal projector weights → FP16-retain (cross-modal alignment
       is precision-sensitive).
    3. Embeddings, norms, LM head, router → FP16-retain (standard tern-core
       protection policy).
    4. 1-D weights (biases, scalar params) → FP16-retain.
    5. All remaining 2-D weights in transformer blocks → ternary-eligible.
    """

    _VISION_PATTERNS: list[str] = [
        "vision_tower",
        "vision_model",
        "embed_vision",
    ]
    _AUDIO_PATTERNS: list[str] = [
        "audio_tower",
        "embed_audio",
    ]
    _PROJECTOR_PATTERNS: list[str] = [
        "multi_modal_projector",
        "multimodal_projector",
    ]

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="gemma4",
            architecture="Gemma4ForConditionalGeneration",
            model_type="gemma4",
            description=(
                "Google Gemma 4 adapter — supports text-only and E4B "
                "multimodal (vision + audio). Maps HF weight names, "
                "handles MoE expert layers, flags modality encoders "
                "for FP16 retention."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=True,
            multimodal_components=["vision", "audio", "projector"],
        )

    def normalize_name(self, name: str) -> str:
        """Strip the ``language_model.`` prefix from multimodal weight names.

        In Gemma 4 multimodal (E4B), language model weights are stored as
        ``model.language_model.model.layers.N...``.  We strip to
        ``model.layers.N...`` so the downstream pipeline matches
        the same patterns as a text-only model.

        Non-language weights (vision/audio/projector) are returned as-is
        since they go through separate FP16-retain handling.
        """
        # Strip nested language_model prefix
        if "language_model." in name:
            name = name.replace("language_model.", "", 1)
        return name

    def classify_weight(
        self,
        name: str,
        shape: Optional[list[int]] = None,
    ) -> WeightClassification:
        """Classify a weight tensor for conversion."""
        canonical = self.normalize_name(name)
        component = self._detect_component(name)

        # Rule 1–2: Non-language components → FP16-retain
        if component == "vision":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Vision encoder — modality encoders are FP16-retain",
                component=component,
            )
        if component == "audio":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Audio encoder — modality encoders are FP16-retain",
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

        # Rule 3: Always-protected language weights
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

        # Rule 4: 1-D weights (biases, scalars) → FP16-retain
        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (bias or scalar parameter)",
                component=component,
            )

        # Rule 5: Remaining 2-D weights → ternary-eligible
        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in transformer block — eligible for ternary conversion",
            component=component,
        )

    def classify_all(
        self,
        weight_names: dict[str, list[int]],
    ) -> dict[str, WeightClassification]:
        """Classify all weights in a model.

        Args:
            weight_names: Mapping of weight name → shape.

        Returns:
            Mapping of weight name → WeightClassification.
        """
        return {
            name: self.classify_weight(name, shape)
            for name, shape in weight_names.items()
        }

    def get_protection_list(
        self,
        weight_names: dict[str, list[int]],
    ) -> list[str]:
        """Return list of weight names that should be FP16-retained."""
        classifications = self.classify_all(weight_names)
        return [
            name for name, cls in classifications.items()
            if cls.category == "fp16_retain"
        ]

    def get_ternary_eligible(
        self,
        weight_names: dict[str, list[int]],
    ) -> list[str]:
        """Return list of weight names eligible for ternary conversion."""
        classifications = self.classify_all(weight_names)
        return [
            name for name, cls in classifications.items()
            if cls.category == "ternary_eligible"
        ]
