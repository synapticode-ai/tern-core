"""
Tests for the validate_architecture allow-list guard and the
config.json reader helper.

Group A item A5 migrates ``AdapterInfo.architecture: str`` to
``AdapterInfo.architectures: list[str]``, introduces
:class:`ArchitectureMismatch`, adds a concrete
``validate_architecture`` method to :class:`ArchitectureAdapter`,
and inserts the validation call at the entry of
``full_convert`` and ``dry_run_convert``.

The guard prevents silent misrouting — a model whose HF config
architecture is not in the adapter's allow-list raises
:class:`ArchitectureMismatch` at convert-time rather than after
compression has run to completion.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terncore.adapters.base import ArchitectureMismatch
from terncore.adapters.gemma3 import Gemma3Adapter
from terncore.adapters.gemma4 import Gemma4Adapter
from terncore.adapters.llama import LlamaAdapter
from terncore.convert import _read_hf_arch_from_config


# ── validate_architecture: allow-list semantics ─────────────────────


@pytest.mark.parametrize(
    "adapter_cls, hf_arch",
    [
        (LlamaAdapter, "LlamaForCausalLM"),
        (Gemma3Adapter, "Gemma3ForConditionalGeneration"),
        (Gemma4Adapter, "Gemma4ForConditionalGeneration"),
    ],
)
def test_validate_architecture_accepts_each_listed_arch(adapter_cls, hf_arch):
    adapter = adapter_cls()
    adapter.validate_architecture(hf_arch)  # must not raise


def test_validate_architecture_raises_on_mismatch():
    adapter = LlamaAdapter()
    with pytest.raises(ArchitectureMismatch):
        adapter.validate_architecture("MistralForCausalLM")


def test_validate_architecture_error_message_names_both_sides():
    adapter = LlamaAdapter()
    with pytest.raises(ArchitectureMismatch) as exc:
        adapter.validate_architecture("Qwen3DeltaNetForCausalLM")
    msg = str(exc.value)
    assert "llama" in msg
    assert "Qwen3DeltaNetForCausalLM" in msg
    assert "LlamaForCausalLM" in msg


def test_architecture_mismatch_is_exception_subclass():
    assert issubclass(ArchitectureMismatch, Exception)


# ── architectures field migration ───────────────────────────────────


@pytest.mark.parametrize(
    "adapter_cls",
    [LlamaAdapter, Gemma3Adapter, Gemma4Adapter],
)
def test_architectures_field_is_list_not_string(adapter_cls):
    """Regression guard: the migration from architecture: str to
    architectures: list[str] must not regress to a string. A bare
    string is iterable character-by-character; ``"L" in "LlamaForCausalLM"``
    is True for any 1-char input, which would defeat the allow-list
    semantics."""
    info = adapter_cls().info()
    assert isinstance(info.architectures, list)
    assert not isinstance(info.architectures, str)


@pytest.mark.parametrize(
    "adapter_cls",
    [LlamaAdapter, Gemma3Adapter, Gemma4Adapter],
)
def test_each_adapter_declares_at_least_one_architecture(adapter_cls):
    info = adapter_cls().info()
    assert len(info.architectures) >= 1
    for arch in info.architectures:
        assert isinstance(arch, str)
        assert arch  # non-empty


# ── _read_hf_arch_from_config helper ────────────────────────────────


def test_read_hf_arch_returns_first_element(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "architectures": ["LlamaForCausalLM", "OtherArch"],
    }))
    assert _read_hf_arch_from_config(tmp_path) == "LlamaForCausalLM"


def test_read_hf_arch_raises_on_missing_config(tmp_path: Path):
    with pytest.raises(ArchitectureMismatch) as exc:
        _read_hf_arch_from_config(tmp_path)
    assert "config.json not found" in str(exc.value)


def test_read_hf_arch_raises_on_empty_architectures_field(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"architectures": []}))
    with pytest.raises(ArchitectureMismatch) as exc:
        _read_hf_arch_from_config(tmp_path)
    assert "architectures" in str(exc.value)


def test_read_hf_arch_raises_on_missing_architectures_field(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"foo": "bar"}))
    with pytest.raises(ArchitectureMismatch) as exc:
        _read_hf_arch_from_config(tmp_path)
    assert "architectures" in str(exc.value)


def test_read_hf_arch_raises_on_corrupt_config(tmp_path: Path):
    """Stdlib json failure path: malformed JSON should surface as
    ArchitectureMismatch with a readable message, not a raw
    JSONDecodeError that the operator has to interpret."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not valid json {{{")
    with pytest.raises(ArchitectureMismatch) as exc:
        _read_hf_arch_from_config(tmp_path)
    assert "Failed to read" in str(exc.value)
