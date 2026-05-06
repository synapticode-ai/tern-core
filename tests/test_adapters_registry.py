"""
Tests for the adapter registry in ``terncore.adapters``.

Covers the import-table refactor (A1): ``get_adapter`` resolves known
names via ``importlib.import_module`` against ``_KNOWN_ADAPTERS`` and
raises ``ValueError`` on unknown names with a message that lists the
known set.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import pytest

from terncore.adapters import _KNOWN_ADAPTERS, get_adapter
from terncore.adapters.base import ArchitectureAdapter


def test_get_adapter_raises_on_unknown_name():
    with pytest.raises(ValueError) as exc:
        get_adapter("nonesuch")
    msg = str(exc.value)
    assert "nonesuch" in msg
    assert "Known" in msg
    for known in _KNOWN_ADAPTERS:
        assert known in msg


def test_get_adapter_raises_on_unknown_name_case_insensitive():
    with pytest.raises(ValueError):
        get_adapter("LLAMA_FAKE")


@pytest.mark.parametrize("name", ["llama", "gemma3", "gemma4", "phi3", "qwen3_moe"])
def test_get_adapter_returns_instance_for_each_known_name(name):
    adapter = get_adapter(name)
    assert isinstance(adapter, ArchitectureAdapter)
    assert adapter.info().name == name


@pytest.mark.parametrize("name", ["LLAMA", "Gemma3", "GEMMA4", "PHI3", "Qwen3_MoE"])
def test_get_adapter_accepts_mixed_case_known_names(name):
    adapter = get_adapter(name)
    assert isinstance(adapter, ArchitectureAdapter)
    assert adapter.info().name == name.lower()


def test_known_adapters_is_canonical_source():
    assert set(_KNOWN_ADAPTERS) == {"llama", "gemma3", "gemma4", "phi3", "qwen3_moe"}
