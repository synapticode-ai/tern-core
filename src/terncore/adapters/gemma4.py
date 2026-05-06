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
    StackedSlice,
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

# Stacked-experts pattern. Matches the bare safetensors entry name for
# Gemma 4 MoE expert tensors, which pack all N experts into a single
# 3-D safetensors entry along axis 0:
#   model.language_model.layers.<N>.experts.gate_up_proj  [128, 1408, 2816]
#   model.language_model.layers.<N>.experts.down_proj     [128, 2816, 704]
# The end-anchor ``$`` distinguishes the parent stacked entry from
# synthesised per-expert names which carry an expert index plus
# trailing ``.weight`` (e.g., ``...experts.5.gate_up_proj.weight``).
_STACKED_EXPERTS_RE = re.compile(r"\.experts\.(gate_up_proj|down_proj)$")

# Expert index pattern. Matches synthesised per-expert names produced by
# ``expand_stacked`` and populates ``WeightClassification.expert_idx``
# via the base helper :meth:`ArchitectureAdapter._extract_expert_idx`.
_EXPERT_IDX_RE = re.compile(r"\.experts\.(?P<expert_idx>\d+)\.")


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
            architectures=["Gemma4ForConditionalGeneration"],
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
            expert_pattern=_EXPERT_IDX_RE,
        )

    def expand_stacked(
        self,
        name: str,
        shape: list[int],
    ) -> Optional[list[StackedSlice]]:
        """Fan stacked MoE expert tensors out into per-expert slices.

        Recognises Gemma 4's two stacked-experts patterns
        (``experts.gate_up_proj`` and ``experts.down_proj``) by the
        bare safetensors name (no expert index, no ``.weight`` suffix).
        Returns one :class:`StackedSlice` per expert slot along axis 0,
        with synthesised names of the form
        ``...experts.<K>.<projection>.weight``. Returns ``None`` for
        non-stacked weights so dense / attention / multimodal tensors
        flow through the converter unchanged.
        """
        match = _STACKED_EXPERTS_RE.search(name)
        if match is None:
            return None
        if len(shape) != 3:
            raise ValueError(
                f"Gemma4Adapter recognised '{name}' as a stacked-experts "
                f"pattern but shape {shape} is not 3-D. Expected "
                f"[num_experts, ..., ...] — refusing to fan out with "
                f"unknown stacking layout."
            )
        projection = match.group(1)  # "gate_up_proj" or "down_proj"
        num_experts = shape[0]
        # Strip trailing "<projection>" (last segment) so the prefix ends
        # in ".experts." — synthesised per-expert names slot the index in.
        prefix = name[: -len(projection)]
        return [
            StackedSlice(
                synthesised_name=f"{prefix}{k}.{projection}.weight",
                slice_axis=0,
                slice_index=k,
                expert_idx=k,
            )
            for k in range(num_experts)
        ]

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
        expert_idx = self._extract_expert_idx(name)

        # Rule 1–2: Non-language components → FP16-retain
        if component == "vision":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Vision encoder — modality encoders are FP16-retain",
                component=component,
                expert_idx=expert_idx,
            )
        if component == "audio":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Audio encoder — modality encoders are FP16-retain",
                component=component,
                expert_idx=expert_idx,
            )
        if component == "projector":
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="Multi-modal projector — cross-modal alignment is precision-sensitive",
                component=component,
                expert_idx=expert_idx,
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
                    expert_idx=expert_idx,
                )

        # Rule 4: 1-D weights (biases, scalars) → FP16-retain
        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (bias or scalar parameter)",
                component=component,
                expert_idx=expert_idx,
            )

        # Rule 5: Remaining 2-D weights → ternary-eligible
        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in transformer block — eligible for ternary conversion",
            component=component,
            expert_idx=expert_idx,
        )
