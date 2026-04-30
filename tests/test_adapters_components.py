"""
Tests for the lifted ``_detect_component`` contract in
``terncore.adapters.base`` and the gemma3 4-bucket vocab migration.

Group A item A4 lifts ``_detect_component`` to ``ArchitectureAdapter``
as a concrete default driven by class-attribute pattern lists. The
gemma3 adapter migrates from a 2-bucket (vision/language) emission
to the 4-bucket vocab (vision/audio/projector/language) by splitting
multimodal projector patterns out of its vision pattern list.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters.base import ArchitectureAdapter
from terncore.adapters.gemma3 import Gemma3Adapter
from terncore.adapters.gemma4 import Gemma4Adapter


# ── Base default behaviour ──────────────────────────────────────────


class _BareAdapter(ArchitectureAdapter):
    """Minimal subclass for testing the inherited default behaviour."""


def test_base_detect_component_returns_language_for_text_only():
    adapter = _BareAdapter()
    assert adapter._detect_component(
        "model.layers.0.self_attn.q_proj.weight"
    ) == "language"
    assert adapter._detect_component("vision_tower.something") == "language"
    assert adapter._detect_component("anything") == "language"


def test_base_detect_component_pattern_priority_vision_first():
    """If a name matches patterns in multiple buckets, vision wins
    (per documented scan order: vision → audio → projector)."""

    class _MultiPatternAdapter(ArchitectureAdapter):
        _VISION_PATTERNS = ["shared_token"]
        _AUDIO_PATTERNS = ["shared_token"]
        _PROJECTOR_PATTERNS = ["shared_token"]

    adapter = _MultiPatternAdapter()
    assert adapter._detect_component("layer.shared_token.weight") == "vision"


def test_base_detect_component_pattern_priority_audio_before_projector():
    """Audio is scanned before projector when vision misses."""

    class _AudioProjectorAdapter(ArchitectureAdapter):
        _VISION_PATTERNS = []
        _AUDIO_PATTERNS = ["shared_token"]
        _PROJECTOR_PATTERNS = ["shared_token"]

    adapter = _AudioProjectorAdapter()
    assert adapter._detect_component("layer.shared_token.weight") == "audio"


# ── gemma3 vocab migration ──────────────────────────────────────────


def test_gemma3_emits_projector_for_multimodal_projector():
    adapter = Gemma3Adapter()
    cls = adapter.classify_weight(
        "multi_modal_projector.linear_1.weight",
        shape=[1024, 2048],
    )
    assert cls.component == "projector"


def test_gemma3_projector_weights_get_fp16_retain_category():
    """Regression guard: projector weights must NOT silently fall
    through to ternary_eligible after the vocab split.

    Pre-A4, gemma3 lumped multi_modal_projector under
    _VISION_PATTERNS, so its vision branch caught these weights and
    assigned fp16_retain. A4 splits projector into its own bucket;
    without an explicit projector branch in classify_weight,
    projector weights would fall through to the ternary_eligible
    default. This test locks the fp16_retain invariant.
    """
    adapter = Gemma3Adapter()
    cls = adapter.classify_weight(
        "multi_modal_projector.linear_1.weight",
        shape=[1024, 2048],
    )
    assert cls.category == "fp16_retain"


def test_gemma3_projector_reason_matches_gemma4_wording():
    """Cross-adapter consistency: gemma3 and gemma4 projector
    weights emit the same reason string so that fp16_reasons
    aggregation in dry_run_convert remains coherent across
    adapters."""
    g3 = Gemma3Adapter()
    g4 = Gemma4Adapter()
    g3_cls = g3.classify_weight(
        "multi_modal_projector.linear_1.weight",
        shape=[1024, 2048],
    )
    g4_cls = g4.classify_weight(
        "multi_modal_projector.linear_1.weight",
        shape=[1024, 2048],
    )
    assert g3_cls.reason == g4_cls.reason
    assert "Multi-modal projector" in g3_cls.reason


def test_gemma3_vision_tower_still_emits_vision():
    """Regression guard: the pattern split must not break the
    vision-encoder path. Vision tower weights stay component=vision
    and category=fp16_retain."""
    adapter = Gemma3Adapter()
    cls = adapter.classify_weight(
        "vision_tower.encoder.layers.0.attn.q_proj.weight",
        shape=[1024, 1024],
    )
    assert cls.component == "vision"
    assert cls.category == "fp16_retain"


# ── gemma4 deduplication ────────────────────────────────────────────


def test_gemma4_inherits_base_detect_component():
    """gemma4 dedupes by inheriting the base method; no override."""
    assert Gemma4Adapter._detect_component is ArchitectureAdapter._detect_component


def test_gemma4_class_attribute_patterns_are_populated():
    """gemma4's pattern lists live on the class, not at module
    level — the inheritance mechanism reads ``self._VISION_PATTERNS``
    and friends via class-attribute lookup."""
    assert "vision_tower" in Gemma4Adapter._VISION_PATTERNS
    assert "embed_vision" in Gemma4Adapter._VISION_PATTERNS
    assert "audio_tower" in Gemma4Adapter._AUDIO_PATTERNS
    assert "multi_modal_projector" in Gemma4Adapter._PROJECTOR_PATTERNS


@pytest.mark.parametrize(
    "name, expected",
    [
        ("vision_tower.encoder.weight", "vision"),
        ("audio_tower.encoder.weight", "audio"),
        ("multi_modal_projector.linear_1.weight", "projector"),
        ("model.layers.0.self_attn.q_proj.weight", "language"),
    ],
)
def test_gemma4_detect_component_via_inherited_method(name, expected):
    adapter = Gemma4Adapter()
    assert adapter._detect_component(name) == expected


# ── multimodal_components declaration accuracy ──────────────────────


def test_multimodal_components_declares_projector_for_gemma3():
    """gemma3.info().multimodal_components must declare 'projector'
    to accurately advertise what classify_weight emits."""
    adapter = Gemma3Adapter()
    components = adapter.info().multimodal_components
    assert "vision" in components
    assert "projector" in components


def test_multimodal_components_unchanged_for_gemma4():
    """gemma4 already declared 4-bucket components pre-A4; the
    deduplication commit doesn't touch the declaration."""
    adapter = Gemma4Adapter()
    components = adapter.info().multimodal_components
    assert components == ["vision", "audio", "projector"]
