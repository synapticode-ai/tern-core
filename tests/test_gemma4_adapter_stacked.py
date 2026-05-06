"""
Tests for Gemma4Adapter stacked-experts expansion + expert_idx wiring.

Session 3 per-expert slicing rework adds:
- ``Gemma4Adapter.expand_stacked`` — fans stacked MoE expert tensors
  (``experts.gate_up_proj`` / ``experts.down_proj``) out into per-expert
  StackedSlice records along axis 0.
- ``expert_pattern`` declared in ``info()`` so the inherited
  ``_extract_expert_idx`` helper populates ``WeightClassification.expert_idx``
  from synthesised per-expert names.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters.base import StackedSlice
from terncore.adapters.gemma4 import Gemma4Adapter


# ── expand_stacked: stacked-pattern recognition ─────────────────────


def test_expand_stacked_recognises_experts_gate_up_proj():
    """Parent ``experts.gate_up_proj`` fans out into 128 per-expert slices."""
    adapter = Gemma4Adapter()
    parent = "model.language_model.layers.0.experts.gate_up_proj"
    plan = adapter.expand_stacked(parent, [128, 1408, 2816])

    assert plan is not None
    assert len(plan) == 128
    assert all(isinstance(s, StackedSlice) for s in plan)

    first = plan[0]
    assert first.synthesised_name == (
        "model.language_model.layers.0.experts.0.gate_up_proj.weight"
    )
    assert first.slice_axis == 0
    assert first.slice_index == 0
    assert first.expert_idx == 0

    last = plan[-1]
    assert last.synthesised_name == (
        "model.language_model.layers.0.experts.127.gate_up_proj.weight"
    )
    assert last.slice_index == 127
    assert last.expert_idx == 127


def test_expand_stacked_recognises_experts_down_proj():
    """Parent ``experts.down_proj`` fans out with the right projection token."""
    adapter = Gemma4Adapter()
    parent = "model.language_model.layers.7.experts.down_proj"
    plan = adapter.expand_stacked(parent, [128, 2816, 704])

    assert plan is not None
    assert len(plan) == 128
    assert plan[5].synthesised_name == (
        "model.language_model.layers.7.experts.5.down_proj.weight"
    )
    assert plan[5].expert_idx == 5


# ── expand_stacked: non-stacked patterns return None ────────────────


def test_expand_stacked_returns_none_for_dense_mlp():
    adapter = Gemma4Adapter()
    name = "model.language_model.layers.0.mlp.gate_proj.weight"
    assert adapter.expand_stacked(name, [2816, 2112]) is None


def test_expand_stacked_returns_none_for_attention():
    adapter = Gemma4Adapter()
    name = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert adapter.expand_stacked(name, [4096, 2816]) is None


def test_expand_stacked_returns_none_for_router_proj():
    """``router.proj.weight`` is 2-D with leading 128 dim — must not expand.

    The leading 128 matches num_experts but the name lacks the ``experts.``
    substring, so the stacked-experts regex correctly declines.
    """
    adapter = Gemma4Adapter()
    name = "model.language_model.layers.0.router.proj.weight"
    assert adapter.expand_stacked(name, [128, 2816]) is None


def test_expand_stacked_does_not_re_expand_synthesised_names():
    """Synthesised per-expert names must NOT trigger re-expansion.

    Exercises the end-anchor ``$`` asymmetry in ``_STACKED_EXPERTS_RE``.
    A synthesised name carries an expert index plus trailing ``.weight``
    and so must not match the parent-only regex.
    """
    adapter = Gemma4Adapter()
    synthesised = "model.language_model.layers.0.experts.5.gate_up_proj.weight"
    assert adapter.expand_stacked(synthesised, [1408, 2816]) is None


# ── expand_stacked: halt-and-surface guard on wrong ndim ────────────


def test_expand_stacked_raises_on_wrong_ndim():
    """Pattern matched but shape isn't 3-D → ValueError naming the input."""
    adapter = Gemma4Adapter()
    name = "model.language_model.layers.0.experts.gate_up_proj"
    with pytest.raises(ValueError) as exc_info:
        adapter.expand_stacked(name, [1408, 2816])
    msg = str(exc_info.value)
    assert "experts.gate_up_proj" in msg
    assert "[1408, 2816]" in msg or "shape" in msg.lower()


# ── classify_weight: expert_idx wiring ──────────────────────────────


def test_synthesised_name_classifies_with_expert_idx():
    """Synthesised per-expert names get ``expert_idx`` populated."""
    adapter = Gemma4Adapter()
    name = "model.language_model.layers.0.experts.5.gate_up_proj.weight"
    cls = adapter.classify_weight(name, [1408, 2816])
    assert cls.category == "ternary_eligible"
    assert cls.expert_idx == 5


@pytest.mark.parametrize(
    "name, shape, expected_category",
    [
        # Rule 5 — dense MLP, ternary_eligible
        ("model.language_model.layers.0.mlp.gate_proj.weight", [2816, 2112], "ternary_eligible"),
        # Rule 5 — attention, ternary_eligible
        ("model.language_model.layers.0.self_attn.q_proj.weight", [4096, 2816], "ternary_eligible"),
        # Rule 3 — norm, fp16_retain
        ("model.language_model.layers.0.input_layernorm.weight", [2816], "fp16_retain"),
        # Rule 3 — embed, fp16_retain
        ("model.language_model.embed_tokens.weight", [262144, 2816], "fp16_retain"),
        # Rule 1 — vision encoder, fp16_retain
        ("model.vision_tower.encoder.layer.0.attention.q.weight", [1024, 1024], "fp16_retain"),
    ],
)
def test_classify_weight_expert_idx_none_for_non_expert_names(
    name, shape, expected_category,
):
    """Non-expert names get ``expert_idx=None`` across every classify return path.

    Covers Rules 1, 3, and 5 (vision, protected pattern, ternary_eligible).
    The wiring change touched all 5 return paths; this parametrise hits each.
    """
    adapter = Gemma4Adapter()
    cls = adapter.classify_weight(name, shape)
    assert cls.category == expected_category
    assert cls.expert_idx is None


# ── name uniqueness ─────────────────────────────────────────────────


def test_expand_stacked_synthesised_names_are_unique():
    """All 128 synthesised names for a parent must be distinct.

    Catches off-by-one or template-string regressions in the
    name-construction logic.
    """
    adapter = Gemma4Adapter()
    parent = "model.language_model.layers.0.experts.gate_up_proj"
    plan = adapter.expand_stacked(parent, [128, 1408, 2816])
    names = [s.synthesised_name for s in plan]
    assert len(set(names)) == 128
    indices = [s.slice_index for s in plan]
    assert sorted(indices) == list(range(128))
