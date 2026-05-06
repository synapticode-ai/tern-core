"""
Phi-3 / Phi-4 architecture adapter for tern-core conversion pipeline.

Maps Microsoft Phi-3 / Phi-4 (``Phi3ForCausalLM``) HuggingFace weight
names to tern-core's internal weight schema. Note that Phi-4 retains
the Phi3 architecture class name; the adapter covers both generations.

Phi-3/4 quirks vs Llama:
- Fused QKV projection: ``self_attn.qkv_proj.weight`` (single tensor
  instead of separate q_proj/k_proj/v_proj)
- Fused gate+up projection: ``mlp.gate_up_proj.weight`` (single tensor
  instead of separate gate_proj/up_proj)
- ``self_attn.o_proj.weight`` and ``mlp.down_proj.weight`` remain
  separate as in Llama

Fused-projection handling decision: treat as single tensors. Each
fused tensor receives one threshold-derived ternary cutoff and one
sparsity record. The per-component sparsity question (does Q have
different sparsity than V? does gate vs up?) is deferred until
cross-architecture analysis surfaces a per-component signal worth
investigating; the fused-as-single approach matches Phi-4's actual
forward-pass shape (one nn.Linear) and matches April 2026 prior
production practice (LlamaAdapter route, 160 ternary entries on
``microsoft/phi-4`` — verified).

Classification rules mirror LlamaAdapter (proven on Phi-4 in prior
compression); this adapter exists primarily to declare the
architecture allow-list for ``validate_architecture`` per the
post-Group-A allow-list discipline (description strings are
aspirational, allow-lists are gates).

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
    "qkv_proj",       # fused Q/K/V — single tensor per layer
    "o_proj",         # separate output projection
    "gate_up_proj",   # fused gate+up — single tensor per layer
    "down_proj",      # separate down projection
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


@register("phi3")
class Phi3Adapter(ArchitectureAdapter):
    """Architecture adapter for Phi3ForCausalLM (Phi-3 + Phi-4).

    Weight classification (mirrors LlamaAdapter, validated on Phi-4 in
    April 2026 production compression):
    1. Embeddings, norms, LM head → FP16-retain.
    2. 1-D weights (biases, scalars) → FP16-retain.
    3. All 2-D weights in transformer blocks → ternary-eligible
       (including fused qkv_proj and gate_up_proj as single tensors).
    """

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="phi3",
            architectures=["Phi3ForCausalLM"],
            model_type="phi3",
            description=(
                "Microsoft Phi-3 / Phi-4 adapter — fused QKV + fused "
                "gate_up projections treated as single ternary tensors. "
                "Text-only."
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
            reason="2-D weight in transformer block — eligible for ternary conversion",
            component="language",
        )
