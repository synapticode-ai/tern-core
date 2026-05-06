"""
Tests for Phi3Adapter — Microsoft Phi-3 / Phi-4 architecture support.

PR #16 cross-architecture sprint cluster (2026-05-07): Phi3Adapter
declares the architecture allow-list for ``Phi3ForCausalLM`` and
mirrors LlamaAdapter's classification logic, validated against
April 2026 prior production compression of Phi-4 via the
``--adapter llama`` route.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters.base import ArchitectureMismatch
from terncore.adapters.phi3 import Phi3Adapter


# ── Architecture allow-list ─────────────────────────────────────────


def test_phi3_adapter_accepts_phi3forcausallm():
    Phi3Adapter().validate_architecture("Phi3ForCausalLM")


def test_phi3_adapter_rejects_other_architectures():
    adapter = Phi3Adapter()
    for arch in ("LlamaForCausalLM", "Gemma4ForConditionalGeneration",
                 "Qwen3MoeForCausalLM", "MistralForCausalLM"):
        with pytest.raises(ArchitectureMismatch):
            adapter.validate_architecture(arch)


# ── Fused-projection classification (the Phi quirk) ─────────────────


def test_fused_qkv_proj_is_ternary_eligible():
    """Phi's fused QKV is a single 2-D Linear weight; treat as one
    ternary entity (single threshold, single sparsity record).
    """
    cls = Phi3Adapter().classify_weight(
        "model.layers.0.self_attn.qkv_proj.weight",
        shape=[15360, 5120],
    )
    assert cls.category == "ternary_eligible"


def test_fused_gate_up_proj_is_ternary_eligible():
    """Phi's fused gate+up is a single 2-D Linear weight; treat as one
    ternary entity.
    """
    cls = Phi3Adapter().classify_weight(
        "model.layers.0.mlp.gate_up_proj.weight",
        shape=[35840, 5120],
    )
    assert cls.category == "ternary_eligible"


def test_separate_o_proj_and_down_proj_are_ternary_eligible():
    adapter = Phi3Adapter()
    for name in (
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    ):
        cls = adapter.classify_weight(name, shape=[5120, 5120])
        assert cls.category == "ternary_eligible", (
            f"{name} should be ternary_eligible, got {cls.category}"
        )


# ── Standard protection rules ───────────────────────────────────────


@pytest.mark.parametrize(
    "name, shape, expected",
    [
        ("model.layers.0.input_layernorm.weight", [5120], "fp16_retain"),
        ("model.layers.0.post_attention_layernorm.weight", [5120], "fp16_retain"),
        ("model.embed_tokens.weight", [100352, 5120], "fp16_retain"),
        ("model.norm.weight", [5120], "fp16_retain"),
        ("lm_head.weight", [100352, 5120], "fp16_retain"),
    ],
)
def test_phi3_protected_patterns(name, shape, expected):
    cls = Phi3Adapter().classify_weight(name, shape)
    assert cls.category == expected


def test_phi3_one_d_weight_is_protected():
    """1-D tensors (biases, scalars) get FP16-retained even without
    matching a protected substring."""
    cls = Phi3Adapter().classify_weight(
        "model.layers.0.something.bias",
        shape=[5120],
    )
    assert cls.category == "fp16_retain"


# ── No MoE, no expand_stacked ───────────────────────────────────────


def test_phi3_expand_stacked_returns_none():
    """Phi-4 is dense; expand_stacked is a no-op (inherited base default)."""
    adapter = Phi3Adapter()
    for name, shape in (
        ("model.layers.0.self_attn.qkv_proj.weight", [15360, 5120]),
        ("model.layers.0.mlp.gate_up_proj.weight", [35840, 5120]),
    ):
        assert adapter.expand_stacked(name, shape) is None
