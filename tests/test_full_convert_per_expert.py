"""
End-to-end tests for per-expert slicing in ``convert.py:full_convert``.

Session 3 per-expert slicing rework: ``full_convert`` calls
``adapter.expand_stacked`` during shape collection, then in the per-tensor
processing loop slices the parent stacked tensor along ``stack_axis``
and quantises each slice independently. The resulting manifest holds
one ternary entry per expert with stacking metadata (``stacked_parent``,
``stack_axis``, ``stack_index``, ``stack_total``).

These tests build a tiny synthetic Gemma 4 fixture under ``tmp_path``
(no HuggingFace downloads), run it through ``full_convert``, and inspect
the resulting ``.tern-model`` manifest.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from terncore.adapters.gemma4 import Gemma4Adapter
from terncore.convert import full_convert
from terncore.tern_model import TernModelReader


# ── Synthetic fixture builder ───────────────────────────────────────


def _write_synthetic_gemma4_model(
    model_dir: Path,
    *,
    include_stacked_experts: bool,
    num_experts: int = 4,
    expert_out_dim: int = 8,
    expert_in_dim: int = 16,
    seed: int = 1234,
) -> dict[str, list[int]]:
    """Build a minimal Gemma 4-shaped safetensors model under ``model_dir``.

    Writes ``config.json`` (with the architecture name + model_type +
    minimal text_config so adapter.validate_architecture passes cleanly)
    plus a single-shard ``model.safetensors`` containing a small set of
    representative tensors. Returns the dict of ``name -> shape`` so
    tests can cross-check against the manifest.

    The stacked-expert tensor (when included) is constructed with
    distinct content per expert slot (``slot_k = 0.5*k + randn``) so
    per-expert sparsity values come out distinct after quantisation —
    this is what lets test 2 distinguish per-expert quantisation from a
    silent shared-threshold degradation.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    tensors: dict[str, torch.Tensor] = {}

    if include_stacked_experts:
        # Stacked-experts parent: [num_experts, out, in]. Each slot
        # offset by a distinct mean so per-expert sparsity differs.
        slots = []
        for k in range(num_experts):
            slot = 0.5 * k + torch.randn(expert_out_dim, expert_in_dim)
            slots.append(slot)
        tensors["model.language_model.layers.0.experts.gate_up_proj"] = (
            torch.stack(slots, dim=0)
        )

    # Dense MLP weight (ternary_eligible, non-stacked)
    tensors["model.language_model.layers.0.mlp.gate_proj.weight"] = torch.randn(
        expert_in_dim, expert_in_dim
    )
    # Attention weight (ternary_eligible, non-stacked)
    tensors["model.language_model.layers.0.self_attn.q_proj.weight"] = torch.randn(
        expert_in_dim, expert_in_dim
    )
    # Norm (fp16_retain via Rule 3 protected pattern)
    tensors["model.language_model.layers.0.input_layernorm.weight"] = torch.randn(
        expert_in_dim
    )
    # Embed (fp16_retain via Rule 3 protected pattern)
    tensors["model.language_model.embed_tokens.weight"] = torch.randn(
        32, expert_in_dim
    )

    save_file(tensors, str(model_dir / "model.safetensors"))

    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "num_hidden_layers": 1,
            "hidden_size": expert_in_dim,
            "num_experts": num_experts,
            "moe_intermediate_size": expert_out_dim // 2,
        },
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Defensive: validate that the synthetic config will pass the adapter's
    # architecture check before the test exercises any conversion logic.
    # Surfaces fixture errors as clear AssertionError rather than confusing
    # downstream failures.
    Gemma4Adapter().validate_architecture(config["architectures"][0])

    return {name: list(t.shape) for name, t in tensors.items()}


def _read_manifest(tern_model_path: Path) -> list[dict]:
    """Return the manifest entries from a written .tern-model artefact."""
    return TernModelReader(str(tern_model_path)).manifest["layers"]


# ── Tests ───────────────────────────────────────────────────────────


def test_full_convert_expands_stacked_experts(tmp_path: Path):
    """4 stacked-slice entries with stacking metadata + 4 non-stacked entries."""
    model_dir = tmp_path / "synthetic_model"
    out_dir = tmp_path / "out"
    _write_synthetic_gemma4_model(
        model_dir, include_stacked_experts=True, num_experts=4
    )

    full_convert(
        model_id=str(model_dir),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    entries = _read_manifest(out_dir / "model.tern-model")

    stacked_entries = [e for e in entries if e.get("stacked_parent")]
    non_stacked = [e for e in entries if not e.get("stacked_parent")]

    assert len(stacked_entries) == 4, (
        f"Expected 4 per-expert slices, got {len(stacked_entries)}"
    )

    # Per-slice metadata is correct, indices are 0..3 contiguous
    assert all(e["stack_total"] == 4 for e in stacked_entries)
    assert all(e["stack_axis"] == 0 for e in stacked_entries)
    indices = sorted(e["stack_index"] for e in stacked_entries)
    assert indices == [0, 1, 2, 3]

    # Synthesised names follow the expected template
    for entry in stacked_entries:
        k = entry["stack_index"]
        assert entry["name"] == (
            f"model.layers.0.experts.{k}.gate_up_proj.weight"
        )
        # stacked_parent is the post-normalize_name canonical form
        assert entry["stacked_parent"] == "model.layers.0.experts.gate_up_proj"

    # Non-stacked: 2 ternary (mlp.gate_proj, self_attn.q_proj) + 2 fp16
    # (input_layernorm, embed_tokens). Norm + embed → float16 dtype.
    ternary_non_stacked = [e for e in non_stacked if e["dtype"] == "ternary2"]
    fp16_non_stacked = [e for e in non_stacked if e["dtype"] == "float16"]
    assert len(ternary_non_stacked) == 2
    assert len(fp16_non_stacked) == 2


def test_full_convert_per_expert_sparsity_distinct(tmp_path: Path):
    """Per-expert sparsity values are distinct (IP measurement validation).

    Distinguishes per-expert quantisation (each slice gets its own
    threshold-derived cutoff) from a silent shared-threshold degradation
    that would produce identical sparsity values across slices.
    """
    model_dir = tmp_path / "synthetic_model"
    out_dir = tmp_path / "out"
    _write_synthetic_gemma4_model(
        model_dir, include_stacked_experts=True, num_experts=4
    )

    full_convert(
        model_id=str(model_dir),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    entries = _read_manifest(out_dir / "model.tern-model")
    stacked_entries = [e for e in entries if e.get("stacked_parent")]
    sparsity_values = [entry["sparsity"] for entry in stacked_entries]

    assert len(set(round(s, 6) for s in sparsity_values)) == len(sparsity_values), (
        f"Per-expert sparsity values not distinct: {sparsity_values}"
    )


def test_full_convert_dense_path_unchanged_for_non_stacked(tmp_path: Path):
    """Non-stacked tensors carry no stacking metadata in the manifest."""
    model_dir = tmp_path / "synthetic_model"
    out_dir = tmp_path / "out"
    _write_synthetic_gemma4_model(
        model_dir, include_stacked_experts=True, num_experts=4
    )

    full_convert(
        model_id=str(model_dir),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    entries = _read_manifest(out_dir / "model.tern-model")
    for entry in entries:
        if entry["name"] in (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ):
            assert "stacked_parent" not in entry, (
                f"Non-stacked entry '{entry['name']}' carries stacking metadata"
            )


def test_full_convert_no_stacking_metadata_for_dense_only_model(tmp_path: Path):
    """E4B regression: dense-only model produces no stacking metadata.

    When the source has no ``experts.*`` tensors at all, every manifest
    entry must lack ``stacked_parent``. This is the most important
    regression case — confirms the rework adds zero overhead and zero
    behaviour change for dense models.
    """
    model_dir = tmp_path / "dense_model"
    out_dir = tmp_path / "out"
    shapes = _write_synthetic_gemma4_model(
        model_dir, include_stacked_experts=False
    )
    assert "model.language_model.layers.0.experts.gate_up_proj" not in shapes

    full_convert(
        model_id=str(model_dir),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    entries = _read_manifest(out_dir / "model.tern-model")
    for entry in entries:
        assert "stacked_parent" not in entry, (
            f"Dense-only model produced stacking metadata on entry "
            f"'{entry['name']}' — regression in non-MoE path."
        )

    # And the entry count equals what we wrote (no expansion)
    assert len(entries) == len(shapes), (
        f"Dense-only entry count drifted: wrote {len(shapes)}, "
        f"manifest has {len(entries)}."
    )


def test_full_convert_round_trip_restacking(tmp_path: Path):
    """End-to-end: write → reconstruct_all returns the parent under the
    canonical key with the expected restacked shape."""
    model_dir = tmp_path / "synthetic_model"
    out_dir = tmp_path / "out"
    _write_synthetic_gemma4_model(
        model_dir, include_stacked_experts=True, num_experts=4,
        expert_out_dim=8, expert_in_dim=16,
    )

    full_convert(
        model_id=str(model_dir),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    reader = TernModelReader(str(out_dir / "model.tern-model"))
    state_dict = reader.reconstruct_all()

    parent_key = "model.layers.0.experts.gate_up_proj"
    assert parent_key in state_dict, (
        f"Restacked parent missing from state_dict; keys: {sorted(state_dict)}"
    )
    restacked = state_dict[parent_key]
    assert restacked.shape == torch.Size([4, 8, 16]), (
        f"Restacked shape wrong: expected [4, 8, 16], got {list(restacked.shape)}"
    )

    # And the per-slice synthesised names should NOT appear in state_dict —
    # they were consumed during restacking and emitted under the parent name only.
    for k in range(4):
        synth = f"model.layers.0.experts.{k}.gate_up_proj.weight"
        assert synth not in state_dict, (
            f"Synthesised slice name '{synth}' leaked into state_dict alongside "
            f"the restacked parent — restacking should consume the slices."
        )


def test_full_convert_stacked_parent_uses_normalized_name(tmp_path: Path):
    """``stacked_parent`` field is the post-normalize_name canonical form.

    Both the per-slice ``name`` and the ``stacked_parent`` reference must
    travel through ``adapter.normalize_name`` so they share the same
    canonicalisation. Catches future drift where one is normalised but
    the other isn't.
    """
    model_dir = tmp_path / "synthetic_model"
    out_dir = tmp_path / "out"
    _write_synthetic_gemma4_model(
        model_dir, include_stacked_experts=True, num_experts=4
    )

    full_convert(
        model_id=str(model_dir),
        adapter_name="gemma4",
        output_dir=str(out_dir),
        threshold=0.7,
        verbose=False,
    )

    entries = _read_manifest(out_dir / "model.tern-model")
    stacked_entries = [e for e in entries if e.get("stacked_parent")]

    # Every stacked entry's name AND stacked_parent must be canonical
    # (no ``language_model.`` prefix — Gemma4Adapter strips it).
    for entry in stacked_entries:
        assert "language_model." not in entry["name"], (
            f"Slice name not normalised: {entry['name']}"
        )
        assert "language_model." not in entry["stacked_parent"], (
            f"stacked_parent not normalised: {entry['stacked_parent']}"
        )
        # And the slice name shares the parent's prefix (after normalisation).
        # Slice: model.layers.0.experts.K.gate_up_proj.weight
        # Parent: model.layers.0.experts.gate_up_proj
        # Strip the index + projection + .weight suffix off the slice name:
        # the result up to and including ".experts." matches parent up to ".experts."
        parent_prefix = entry["stacked_parent"].rsplit(".", 1)[0]  # drops "gate_up_proj"
        assert entry["name"].startswith(parent_prefix + "."), (
            f"Slice name '{entry['name']}' doesn't share parent prefix "
            f"'{parent_prefix}'"
        )
