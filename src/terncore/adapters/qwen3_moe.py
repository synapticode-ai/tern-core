"""
Qwen3 MoE architecture adapter for tern-core conversion pipeline.

Maps Alibaba Qwen3 MoE (``Qwen3MoeForCausalLM``) HuggingFace weight
names to tern-core's internal weight schema. Targets the
Qwen3-30B-A3B variant (128 experts, top-k 8, ~3B active) and any
future Qwen3MoE family members sharing the same architecture class.

Qwen3MoE quirks vs Gemma 4 26B-A4B:
- **Per-expert tensors are 2-D indexed in the safetensors index**
  (``mlp.experts.K.gate_proj.weight`` for each K in 0..127), NOT
  packed as 3-D stacked tensors like Gemma 4's
  ``experts.gate_up_proj``. No ``expand_stacked`` logic needed —
  per-expert sparsity falls out naturally during quantisation.
- **No parallel dense MLP path.** Qwen3 has only the expert FFN
  branch; Gemma 4 has both per-expert weights AND parallel dense
  MLP per layer. Cross-architecture analysis must account for this
  structural difference (cf. Thursday 2026-05-07 banked finding in
  ``project_gemopus_26b_moe_compression_v1`` memory).
- **Router weight is ``mlp.gate.weight``** (not ``router.*`` like
  Gemma 4). Discriminating from expert gate projections
  (``mlp.experts.K.gate_proj.weight``) requires the protected
  substring ``"mlp.gate."`` with trailing period — the underscore
  in ``gate_proj`` breaks the substring match. Without the trailing
  period, expert gate weights would silently misclassify as
  FP16-protected (cf. ``pattern_substring_pattern_discrimination_v1``).
- **Separate gate/up/down per expert.** Gemma 4 fuses gate+up in
  ``experts.gate_up_proj``; Qwen3 keeps them separate. Means more
  per-layer entries (3 per expert instead of 2) but uniform
  per-component sparsity measurement.

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

# Protection patterns. The trailing period on ``"mlp.gate."`` is
# load-bearing — it discriminates the router (``mlp.gate.weight``)
# from expert gate projections (``mlp.experts.K.gate_proj.weight``)
# via substring matching. Removing the trailing period would cause
# all expert gate weights to silently misclassify as FP16-protected
# (the underscore in ``gate_proj`` is what breaks the substring
# match cleanly). Cf. ``pattern_substring_pattern_discrimination_v1``.
_ALWAYS_PROTECTED = (
    "embed_tokens",
    "lm_head",
    "norm",
    "layernorm",
    "layer_norm",
    "rmsnorm",
    "classifier",
    "mlp.gate.",  # router weight; trailing period is load-bearing
)

# Expert index pattern. Matches the Qwen3 per-expert names produced
# directly in the safetensors index (no synthesis needed — Qwen3
# stores experts as 2-D indexed tensors). The inherited
# ``_extract_expert_idx`` helper uses this to populate
# ``WeightClassification.expert_idx`` from the matched name.
_EXPERT_IDX_RE = re.compile(r"\.mlp\.experts\.(?P<expert_idx>\d+)\.")


@register("qwen3_moe")
class Qwen3MoeAdapter(ArchitectureAdapter):
    """Architecture adapter for Qwen3MoeForCausalLM family.

    Weight classification:
    1. Embeddings, norms, LM head, **router** (``mlp.gate.weight``)
       → FP16-retain.
    2. 1-D weights (biases, scalars) → FP16-retain.
    3. All 2-D weights in transformer blocks → ternary-eligible.
       Per-expert weights (``mlp.experts.K.{gate,up,down}_proj.weight``)
       receive their own per-tensor threshold during compression
       (per-expert IP measurement granularity); the inherited
       ``_extract_expert_idx`` populates ``expert_idx`` from the
       matched index.
    """

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="qwen3_moe",
            architectures=["Qwen3MoeForCausalLM"],
            model_type="qwen3_moe",
            description=(
                "Alibaba Qwen3 MoE adapter — per-expert 2-D indexed "
                "tensors (no expand_stacked needed), router as "
                "mlp.gate.weight discriminated from expert gate "
                "projections via trailing-period substring match. "
                "Pure MoE FFN (no parallel dense MLP path). Text-only."
            ),
            block_pattern=_BLOCK_PATTERN,
            projection_priority=list(_PROJ_PRIORITY),
            protection_patterns=list(_ALWAYS_PROTECTED),
            multimodal=False,
            expert_pattern=_EXPERT_IDX_RE,
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
        expert_idx = self._extract_expert_idx(name)

        for pattern in _ALWAYS_PROTECTED:
            if pattern in name_lower:
                return WeightClassification(
                    name=name,
                    canonical_name=canonical,
                    category="fp16_retain",
                    reason=f"Protected pattern: '{pattern}'",
                    component="language",
                    expert_idx=expert_idx,
                )

        if shape is not None and len(shape) < 2:
            return WeightClassification(
                name=name,
                canonical_name=canonical,
                category="fp16_retain",
                reason="1-D tensor (bias or scalar parameter)",
                component="language",
                expert_idx=expert_idx,
            )

        return WeightClassification(
            name=name,
            canonical_name=canonical,
            category="ternary_eligible",
            reason="2-D weight in transformer block — eligible for ternary conversion",
            component="language",
            expert_idx=expert_idx,
        )
