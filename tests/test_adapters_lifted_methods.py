"""
Tests for the lifted classify_all / get_ternary_eligible contract
in ``terncore.adapters.base``.

Group A item A2 lifts these two de-facto methods from per-adapter
copy-paste to base.py concrete defaults. The third originally-scoped
method, ``get_protection_list``, is dead code in the codebase and is
removed from all adapters with no replacement; a regression-prevention
test guards against accidental re-introduction.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters.base import ArchitectureAdapter, WeightClassification
from terncore.adapters.gemma3 import Gemma3Adapter
from terncore.adapters.gemma4 import Gemma4Adapter
from terncore.adapters.llama import LlamaAdapter


# ── Base default behaviour ──────────────────────────────────────────


def test_base_classify_all_returns_dict_keyed_by_name():
    """classify_all returns a dict mapping every input name to a
    WeightClassification, preserving the input keys."""
    adapter = LlamaAdapter()
    weight_shapes = {
        "model.embed_tokens.weight": [32000, 4096],
        "model.layers.0.self_attn.q_proj.weight": [4096, 4096],
        "model.layers.0.input_layernorm.weight": [4096],
    }
    out = adapter.classify_all(weight_shapes)
    assert set(out.keys()) == set(weight_shapes.keys())
    assert all(isinstance(v, WeightClassification) for v in out.values())


def test_base_get_ternary_eligible_filters_by_category():
    """get_ternary_eligible returns only the names whose
    classification is category == 'ternary_eligible'."""
    adapter = LlamaAdapter()
    weight_shapes = {
        "model.embed_tokens.weight": [32000, 4096],            # fp16_retain (protected)
        "model.layers.0.self_attn.q_proj.weight": [4096, 4096],  # ternary_eligible
        "model.layers.0.input_layernorm.weight": [4096],       # fp16_retain (1-D)
        "model.layers.0.mlp.gate_proj.weight": [11008, 4096],  # ternary_eligible
    }
    eligible = adapter.get_ternary_eligible(weight_shapes)
    assert set(eligible) == {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    }


# ── Inheritance contract ────────────────────────────────────────────


@pytest.mark.parametrize(
    "adapter_cls",
    [LlamaAdapter, Gemma3Adapter, Gemma4Adapter],
)
def test_classify_all_is_inherited_by_each_adapter(adapter_cls):
    """No adapter overrides classify_all — all inherit from base.

    If a future refactor accidentally re-introduces a per-adapter
    override (e.g., reverting a commit), this assertion catches it.
    """
    assert adapter_cls.classify_all is ArchitectureAdapter.classify_all


@pytest.mark.parametrize(
    "adapter_cls",
    [LlamaAdapter, Gemma3Adapter, Gemma4Adapter],
)
def test_get_ternary_eligible_is_inherited_by_each_adapter(adapter_cls):
    """No adapter overrides get_ternary_eligible — all inherit from base."""
    assert adapter_cls.get_ternary_eligible is ArchitectureAdapter.get_ternary_eligible


# ── get_protection_list deletion regression guard ───────────────────


@pytest.mark.parametrize(
    "adapter_cls",
    [LlamaAdapter, Gemma3Adapter, Gemma4Adapter],
)
def test_get_protection_list_no_longer_exists_on_adapters(adapter_cls):
    """Pre-A2, every adapter carried a get_protection_list method
    that filtered classify_all by fp16_retain. The method was dead
    code (zero callers in src/, tests/, or convert.py) and was
    removed entirely in the A2 lift rather than being preserved
    as unused infrastructure.

    This regression guard ensures no adapter — including the base
    class — silently re-acquires the method via a future refactor.
    Future callers needing per-model fp16_retain names should
    filter classify_all inline; future callers needing static
    protection-pattern introspection should request a purpose-built
    method (e.g., get_protection_patterns()) at point of need.
    """
    assert not hasattr(adapter_cls, "get_protection_list")
    assert not hasattr(adapter_cls(), "get_protection_list")
