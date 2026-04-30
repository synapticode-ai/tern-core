"""
Tests for the schema widening that prepares
``terncore.adapters.base`` for MoE and hybrid-attention adapters.

Group A item A3 adds optional fields to ``AdapterInfo`` and
``WeightClassification`` plus two concrete helpers
(``_extract_expert_idx``, ``_detect_attention_type``) to
``ArchitectureAdapter``. Existing adapters (llama, gemma3, gemma4)
inherit the ``None`` defaults unchanged — no behaviour change for
non-MoE, non-hybrid models.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import re

import pytest

from terncore.adapters.base import (
    AdapterInfo,
    ArchitectureAdapter,
    WeightClassification,
)
from terncore.adapters.gemma3 import Gemma3Adapter
from terncore.adapters.gemma4 import Gemma4Adapter
from terncore.adapters.llama import LlamaAdapter


_BLOCK_PATTERN = re.compile(r"\.layers\.(\d+)\.")


class _StubAdapter(ArchitectureAdapter):
    """Minimal adapter for exercising the new base helpers.

    Lets each test declare ``expert_pattern`` and
    ``attention_type_pattern`` independently without dragging in
    any concrete adapter's full classification logic.
    """

    def __init__(
        self,
        *,
        expert_pattern: re.Pattern | None = None,
        attention_type_pattern: re.Pattern | None = None,
    ):
        self._expert_pattern = expert_pattern
        self._attention_type_pattern = attention_type_pattern

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="stub",
            architecture="StubForCausalLM",
            model_type="stub",
            description="test stub",
            block_pattern=_BLOCK_PATTERN,
            projection_priority=[],
            protection_patterns=[],
            expert_pattern=self._expert_pattern,
            attention_type_pattern=self._attention_type_pattern,
        )


# ── AdapterInfo schema ──────────────────────────────────────────────


def test_adapter_info_has_expert_pattern_field_with_none_default():
    info = AdapterInfo(
        name="x",
        architecture="X",
        model_type="x",
        description="",
        block_pattern=_BLOCK_PATTERN,
        projection_priority=[],
        protection_patterns=[],
    )
    assert info.expert_pattern is None


def test_adapter_info_has_attention_type_pattern_field_with_none_default():
    info = AdapterInfo(
        name="x",
        architecture="X",
        model_type="x",
        description="",
        block_pattern=_BLOCK_PATTERN,
        projection_priority=[],
        protection_patterns=[],
    )
    assert info.attention_type_pattern is None


# ── WeightClassification schema ─────────────────────────────────────


def test_weight_classification_has_expert_idx_field_with_none_default():
    cls = WeightClassification(
        name="w",
        canonical_name="w",
        category="ternary_eligible",
        reason="test",
        component="language",
    )
    assert cls.expert_idx is None


def test_weight_classification_has_attention_type_field_with_none_default():
    cls = WeightClassification(
        name="w",
        canonical_name="w",
        category="ternary_eligible",
        reason="test",
        component="language",
    )
    assert cls.attention_type is None


# ── _extract_expert_idx helper ──────────────────────────────────────


def test_extract_expert_idx_returns_none_when_no_pattern():
    adapter = _StubAdapter(expert_pattern=None)
    assert adapter._extract_expert_idx(
        "model.layers.0.block_sparse_moe.experts.3.w1.weight"
    ) is None


def test_extract_expert_idx_returns_int_when_pattern_matches():
    adapter = _StubAdapter(
        expert_pattern=re.compile(r"experts\.(?P<expert_idx>\d+)\."),
    )
    assert adapter._extract_expert_idx(
        "model.layers.0.block_sparse_moe.experts.3.w1.weight"
    ) == 3
    assert adapter._extract_expert_idx(
        "model.layers.5.block_sparse_moe.experts.7.w2.weight"
    ) == 7


def test_extract_expert_idx_returns_none_when_pattern_misses():
    adapter = _StubAdapter(
        expert_pattern=re.compile(r"experts\.(?P<expert_idx>\d+)\."),
    )
    assert adapter._extract_expert_idx(
        "model.layers.0.self_attn.q_proj.weight"
    ) is None


# ── _detect_attention_type helper ───────────────────────────────────


def test_detect_attention_type_returns_none_when_no_pattern():
    adapter = _StubAdapter(attention_type_pattern=None)
    assert adapter._detect_attention_type(
        "model.layers.0.linear_attn.in_proj_qkvz.weight"
    ) is None


def test_detect_attention_type_returns_linear_when_pattern_matches():
    adapter = _StubAdapter(
        attention_type_pattern=re.compile(r"linear_attn"),
    )
    assert adapter._detect_attention_type(
        "model.layers.0.linear_attn.in_proj_qkvz.weight"
    ) == "linear"


def test_detect_attention_type_returns_full_when_pattern_misses():
    adapter = _StubAdapter(
        attention_type_pattern=re.compile(r"linear_attn"),
    )
    assert adapter._detect_attention_type(
        "model.layers.0.self_attn.q_proj.weight"
    ) == "full"


# ── Existing adapters inherit None defaults ─────────────────────────


@pytest.mark.parametrize(
    "adapter_cls",
    [LlamaAdapter, Gemma3Adapter, Gemma4Adapter],
)
def test_existing_adapters_have_none_for_new_info_fields(adapter_cls):
    """Smoke positive: llama/gemma3/gemma4 declare neither
    expert_pattern nor attention_type_pattern. The schema widening
    must be a pure-additive no-op for non-MoE, non-hybrid adapters.
    """
    info = adapter_cls().info()
    assert info.expert_pattern is None
    assert info.attention_type_pattern is None
