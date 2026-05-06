"""
Tests for Qwen3MoeAdapter — Alibaba Qwen3 MoE architecture support.

PR #16 cross-architecture sprint cluster (2026-05-07): Qwen3MoeAdapter
handles per-expert 2-D indexed tensors directly (no expand_stacked
needed since each expert is already its own safetensors entry), with
the substantive quirk being the router/expert-gate discrimination
via the trailing-period substring pattern ``"mlp.gate."`` (cf.
``pattern_substring_pattern_discrimination_v1``).

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters.base import ArchitectureMismatch
from terncore.adapters.qwen3_moe import Qwen3MoeAdapter


# ── Architecture allow-list ─────────────────────────────────────────


def test_qwen3_moe_adapter_accepts_qwen3moeforcausallm():
    Qwen3MoeAdapter().validate_architecture("Qwen3MoeForCausalLM")


def test_qwen3_moe_adapter_rejects_other_architectures():
    adapter = Qwen3MoeAdapter()
    for arch in ("LlamaForCausalLM", "Gemma4ForConditionalGeneration",
                 "Phi3ForCausalLM", "Qwen2ForCausalLM"):
        with pytest.raises(ArchitectureMismatch):
            adapter.validate_architecture(arch)


# ── Critical: router vs expert-gate discrimination ──────────────────


def test_router_classification_discriminates_from_expert_gate_proj():
    """The router (``mlp.gate.weight``) is FP16-protected while the
    expert gate projection (``mlp.experts.K.gate_proj.weight``) is
    ternary-eligible.

    This test exists because the protected-substring ``"mlp.gate."``
    (with trailing period) is the discriminator — the underscore in
    ``gate_proj`` breaks the substring match. Removing the trailing
    period would silently misclassify all expert gate weights as
    FP16-protected (cf. ``pattern_substring_pattern_discrimination_v1``).
    """
    adapter = Qwen3MoeAdapter()

    router_cls = adapter.classify_weight(
        "model.layers.0.mlp.gate.weight",
        shape=[128, 2048],
    )
    assert router_cls.category == "fp16_retain", (
        f"Router should be FP16-protected, got {router_cls.category}"
    )
    assert router_cls.expert_idx is None

    expert_gate_cls = adapter.classify_weight(
        "model.layers.0.mlp.experts.5.gate_proj.weight",
        shape=[768, 2048],
    )
    assert expert_gate_cls.category == "ternary_eligible", (
        f"Expert gate should be ternary, got {expert_gate_cls.category}"
    )
    assert expert_gate_cls.expert_idx == 5, (
        f"Expected expert_idx=5, got {expert_gate_cls.expert_idx}"
    )


# ── Per-expert classification + expert_idx wiring ───────────────────


@pytest.mark.parametrize(
    "name, expected_idx",
    [
        ("model.layers.0.mlp.experts.0.gate_proj.weight", 0),
        ("model.layers.0.mlp.experts.0.up_proj.weight", 0),
        ("model.layers.0.mlp.experts.0.down_proj.weight", 0),
        ("model.layers.0.mlp.experts.5.gate_proj.weight", 5),
        ("model.layers.47.mlp.experts.127.down_proj.weight", 127),
    ],
)
def test_per_expert_weights_classify_with_expert_idx(name, expected_idx):
    cls = Qwen3MoeAdapter().classify_weight(name, shape=[768, 2048])
    assert cls.category == "ternary_eligible"
    assert cls.expert_idx == expected_idx


# ── Attention projections ───────────────────────────────────────────


@pytest.mark.parametrize(
    "name, shape",
    [
        ("model.layers.0.self_attn.q_proj.weight", [4096, 2048]),
        ("model.layers.0.self_attn.k_proj.weight", [512, 2048]),
        ("model.layers.0.self_attn.v_proj.weight", [512, 2048]),
        ("model.layers.0.self_attn.o_proj.weight", [2048, 4096]),
    ],
)
def test_attention_projections_are_ternary_eligible(name, shape):
    cls = Qwen3MoeAdapter().classify_weight(name, shape)
    assert cls.category == "ternary_eligible"
    assert cls.expert_idx is None


# ── Norms (including Qwen3-specific q_norm / k_norm) ────────────────


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.self_attn.k_norm.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ],
)
def test_qwen3_protected_patterns(name):
    cls = Qwen3MoeAdapter().classify_weight(name, shape=[2048])
    assert cls.category == "fp16_retain"
    assert cls.expert_idx is None


# ── No expand_stacked needed (Qwen3 is per-expert 2-D in safetensors) ──


def test_qwen3_expand_stacked_returns_none_for_all_qwen3_patterns():
    """Qwen3 stores experts as 2-D indexed tensors directly in the
    safetensors index — no stacked-tensor expansion needed."""
    adapter = Qwen3MoeAdapter()
    for name, shape in (
        ("model.layers.0.mlp.experts.0.gate_proj.weight", [768, 2048]),
        ("model.layers.0.mlp.experts.0.down_proj.weight", [2048, 768]),
        ("model.layers.0.mlp.gate.weight", [128, 2048]),
        ("model.layers.0.self_attn.q_proj.weight", [4096, 2048]),
    ):
        assert adapter.expand_stacked(name, shape) is None, (
            f"expand_stacked should return None for Qwen3 (no stacked "
            f"tensors); got non-None for {name}"
        )


# ── expert_pattern declared on info() ──────────────────────────────


def test_qwen3_info_declares_expert_pattern():
    """Qwen3MoeAdapter declares expert_pattern on info() so the
    inherited _extract_expert_idx helper works without subclass
    overrides."""
    info = Qwen3MoeAdapter().info()
    assert info.expert_pattern is not None
    # The pattern should match Qwen3's per-expert name structure
    match = info.expert_pattern.search(
        "model.layers.0.mlp.experts.5.gate_proj.weight"
    )
    assert match is not None
    assert match.group("expert_idx") == "5"
