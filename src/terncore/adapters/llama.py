"""
Llama architecture adapter for tern-core conversion pipeline.

Maps standard Llama/Llama-2/Llama-3.x (``LlamaForCausalLM``) HuggingFace
weight names to tern-core's internal weight schema.  Covers all Llama
family models including Mistral, CodeLlama, and TinyLlama which share
the same architecture.

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


@register("llama")
class LlamaAdapter(ArchitectureAdapter):
    """Architecture adapter for LlamaForCausalLM family.

    Weight classification:
    1. Embeddings, norms, LM head → FP16-retain.
    2. 1-D weights (biases, scalars) → FP16-retain.
    3. All 2-D weights in transformer blocks → ternary-eligible.
    """

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="llama",
            architectures=["LlamaForCausalLM"],
            model_type="llama",
            description=(
                "Standard Llama adapter — covers Llama 1/2/3.x, "
                "Mistral, CodeLlama, TinyLlama. Text-only."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=False,
        )

    def normalize_name(self, name: str) -> str:
        return name

    def classify_weight(
        self,
        name: str,
        shape: Optional[list[int]] = None,
    ) -> WeightClassification:
        canonical = self.normalize_name(name)
        name_lower = canonical.lower()

        for pattern in _ALWAYS_PROTECTED:
            if pattern in name_lower:
                return WeightClassification(
                    name=name,
                    canonical_name=canonical,
                    category="fp16_retain",
                    reason=f"Protected pattern: '{pattern}'",
                    component="language",
                )

        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (bias or scalar parameter)",
                component="language",
            )

        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in transformer block",
            component="language",
        )
