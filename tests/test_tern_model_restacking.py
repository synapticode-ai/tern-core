"""
Reader-side tests for stacked-tensor restacking in
``TernModelReader.reconstruct_all``.

Session 3 per-expert slicing rework adds an in-flight per-parent
accumulation pattern: when a manifest entry carries ``stacked_parent``
metadata, slices are gathered until the per-parent count reaches
``stack_total`` then restacked via ``torch.stack`` and emitted under
the parent name. Three validation checks fire incrementally per parent
(axes/totals consistency, slice count match, contiguous indices).

These tests bypass ``convert.py:full_convert`` and exercise the reader
directly via ``TernModelWriter`` calls — failures here localise to the
reader's restacking logic without integration noise.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch

from terncore.sparse import unpack_ternary_weights
from terncore.tern_model import TernModelReader, TernModelWriter


# ── Helpers ─────────────────────────────────────────────────────────


def _write_temp_manifest(writer: TernModelWriter) -> Path:
    """Write a manifest to a NamedTemporaryFile and return its Path.

    Caller is responsible for ``os.unlink(path)`` after read.
    """
    with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
        path = Path(f.name)
    writer.write(path)
    return path


def _add_stacked_slice(
    writer: TernModelWriter,
    parent: str,
    slice_idx: int,
    stack_total: int,
    tensor: torch.Tensor,
    *,
    stack_axis: int = 0,
    threshold: float = 0.7,
) -> None:
    """Pack a slice and add it under the synthesised per-expert name."""
    packed, alpha, bitmap, sparsity = TernModelWriter.pack_ternary(
        tensor, threshold
    )
    slice_name = parent.replace(
        ".gate_up_proj", f".{slice_idx}.gate_up_proj"
    ) + ".weight"
    writer.add_ternary_layer(
        name=slice_name,
        packed_weights=packed,
        alpha=alpha,
        shape=list(tensor.shape),
        sparsity_bitmap=bitmap,
        threshold=threshold,
        sparsity=sparsity,
        stacked_parent=parent,
        stack_axis=stack_axis,
        stack_index=slice_idx,
        stack_total=stack_total,
    )


# ── Round-trip + ordering ───────────────────────────────────────────


def test_restacking_round_trip_preserves_slice_content_and_order():
    """Each slice K from the original ends up at restacked[K] with the
    quantised content preserved exactly (bitwise equal to direct
    dequantisation of the same packed bytes)."""
    torch.manual_seed(42)
    parent = "model.layers.0.experts.gate_up_proj"
    NUM = 4
    slices_orig = [torch.randn(8, 16) for _ in range(NUM)]

    # Capture the per-slice dequantised reference outside the writer
    # loop so we can compare bitwise against the restacked tensor.
    dequant_refs = []
    writer = TernModelWriter()
    for k, slice_t in enumerate(slices_orig):
        packed, alpha, bitmap, sparsity = TernModelWriter.pack_ternary(slice_t, 0.7)
        # Reference dequantisation via the same packing path
        ternary_tensor = unpack_ternary_weights(
            torch.frombuffer(bytearray(packed), dtype=torch.uint8),
            torch.Size(slice_t.shape),
        )
        dequant_refs.append(ternary_tensor.float() * alpha)
        writer.add_ternary_layer(
            name=f"model.layers.0.experts.{k}.gate_up_proj.weight",
            packed_weights=packed,
            alpha=alpha,
            shape=[8, 16],
            sparsity_bitmap=bitmap,
            threshold=0.7,
            sparsity=sparsity,
            stacked_parent=parent,
            stack_axis=0,
            stack_index=k,
            stack_total=NUM,
        )

    path = _write_temp_manifest(writer)
    try:
        state_dict = TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)

    assert parent in state_dict
    restacked = state_dict[parent]
    assert restacked.shape == torch.Size([NUM, 8, 16])

    for k in range(NUM):
        assert torch.equal(restacked[k], dequant_refs[k]), (
            f"Slice {k} content not preserved in restack — slice ended up "
            f"at wrong index or restacking introduced distortion."
        )


# ── Backwards compatibility ─────────────────────────────────────────


def test_reconstruct_all_unchanged_for_pre_rework_manifest():
    """Manifest with no stacking metadata behaves identically to pre-rework.

    Verifies the rework added zero behaviour change for manifests written
    without stacking kwargs (i.e., the entire installed-base of pre-Session-3
    .tern-model artefacts).
    """
    torch.manual_seed(7)
    writer = TernModelWriter()
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.input_layernorm.weight",  # production-suffix path
    ]
    for name in names:
        weights = torch.randn(8, 16) if not name.endswith("layernorm.weight") else torch.randn(16)
        if weights.ndim == 1:
            writer.add_layer(name=name, weights=weights, dtype="float16")
        else:
            packed, alpha, bitmap, sparsity = TernModelWriter.pack_ternary(weights, 0.7)
            writer.add_ternary_layer(
                name=name,
                packed_weights=packed,
                alpha=alpha,
                shape=list(weights.shape),
                sparsity_bitmap=bitmap,
                threshold=0.7,
                sparsity=sparsity,
            )

    path = _write_temp_manifest(writer)
    try:
        state_dict = TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)

    # Production-suffix names appear as-is. No new keys, no parent grouping.
    expected_keys = set(names)
    assert set(state_dict.keys()) == expected_keys, (
        f"Pre-rework manifest produced unexpected keys: "
        f"{set(state_dict.keys()) ^ expected_keys}"
    )


# ── Validation failure modes ────────────────────────────────────────


def test_reconstruct_all_raises_on_incomplete_group():
    """Truncated manifest: 2 of 4 slices written → end-of-manifest leftover guard."""
    torch.manual_seed(1)
    parent = "model.layers.0.experts.gate_up_proj"
    writer = TernModelWriter()
    for k in range(2):  # only 2 of 4 slices written
        _add_stacked_slice(writer, parent, k, stack_total=4, tensor=torch.randn(8, 16))

    path = _write_temp_manifest(writer)
    try:
        with pytest.raises(ValueError) as exc_info:
            TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)
    msg = str(exc_info.value)
    assert parent in msg
    assert "2/4" in msg


def test_reconstruct_all_raises_on_inconsistent_axes():
    """Slice index 2 declares a different stack_axis than its siblings."""
    torch.manual_seed(2)
    parent = "model.layers.0.experts.gate_up_proj"
    writer = TernModelWriter()
    for k in range(4):
        # Slice 2 declares axis=1; rest declare axis=0
        axis = 1 if k == 2 else 0
        _add_stacked_slice(
            writer, parent, k, stack_total=4,
            tensor=torch.randn(8, 16), stack_axis=axis,
        )

    path = _write_temp_manifest(writer)
    try:
        with pytest.raises(ValueError) as exc_info:
            TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)
    msg = str(exc_info.value)
    assert parent in msg
    assert "axes" in msg.lower()


def test_reconstruct_all_raises_on_inconsistent_totals():
    """Slice index 3 declares a different stack_total than its siblings."""
    torch.manual_seed(3)
    parent = "model.layers.0.experts.gate_up_proj"
    writer = TernModelWriter()
    for k in range(4):
        # Slice 3 claims total=8, rest claim total=4
        total = 8 if k == 3 else 4
        _add_stacked_slice(
            writer, parent, k, stack_total=total, tensor=torch.randn(8, 16),
        )

    path = _write_temp_manifest(writer)
    try:
        with pytest.raises(ValueError) as exc_info:
            TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)
    msg = str(exc_info.value)
    assert parent in msg
    assert "total" in msg.lower()


@pytest.mark.parametrize(
    "indices, case_label",
    [
        ([0, 1, 3, 4], "gap"),         # missing index 2
        ([0, 1, 1, 2], "duplicate"),   # index 1 appears twice
    ],
)
def test_reconstruct_all_raises_on_invalid_indices(indices, case_label):
    """Both gap and duplicate index patterns trigger the contiguity check."""
    torch.manual_seed(4)
    parent = "model.layers.0.experts.gate_up_proj"
    writer = TernModelWriter()
    for k in indices:
        _add_stacked_slice(
            writer, parent, k, stack_total=4, tensor=torch.randn(8, 16),
        )

    path = _write_temp_manifest(writer)
    try:
        with pytest.raises(ValueError) as exc_info:
            TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)
    msg = str(exc_info.value)
    assert parent in msg, f"[{case_label}] parent not in error message"
    # Either "non-contiguous" or "expected" should appear in the index-error message
    assert ("non-contiguous" in msg or "expected" in msg.lower()), (
        f"[{case_label}] error message doesn't reference index validation: {msg}"
    )


# ── Multi-parent independence ───────────────────────────────────────


def test_reconstruct_all_handles_multiple_independent_stacks():
    """Two parents interleaved in manifest order both restack correctly.

    Bounded-memory property is verified-by-design: the per-parent
    accumulation dict frees each parent's slice list immediately on
    count match. Explicit memory profiling lives in integration tests
    if needed; this test confirms the per-parent accumulation tracks
    parents independently and emits both under their canonical keys.
    """
    torch.manual_seed(5)
    parent_a = "model.layers.0.experts.gate_up_proj"
    parent_b = "model.layers.1.experts.gate_up_proj"
    NUM = 4
    a_slices = [torch.randn(8, 16) for _ in range(NUM)]
    b_slices = [torch.randn(8, 16) for _ in range(NUM)]

    writer = TernModelWriter()
    # Interleave: A0, B0, A1, B1, A2, B2, A3, B3
    for k in range(NUM):
        _add_stacked_slice(writer, parent_a, k, NUM, a_slices[k])
        _add_stacked_slice(writer, parent_b, k, NUM, b_slices[k])

    path = _write_temp_manifest(writer)
    try:
        state_dict = TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)

    assert parent_a in state_dict
    assert parent_b in state_dict
    assert state_dict[parent_a].shape == torch.Size([NUM, 8, 16])
    assert state_dict[parent_b].shape == torch.Size([NUM, 8, 16])
    # And the two restacked tensors are distinct (different content)
    assert not torch.equal(state_dict[parent_a], state_dict[parent_b])


# ── Edge case: writer emits more slices than stack_total declares ───


def test_reconstruct_all_raises_on_duplicate_parent_emission():
    """Writer accidentally emits a parent's slices more than once.

    The 4th slice triggers count match → restacks and frees the group.
    The 5th slice re-creates the group with count=1, never reaches 4
    again, end-of-manifest leftover guard fires reporting 1/4. This
    documents expected behaviour for a writer-side bug; production
    convert.py iterates each safetensors key exactly once so this case
    cannot arise via the standard write path, but the test pins the
    failure mode for diagnostic clarity if it ever does.
    """
    torch.manual_seed(6)
    parent = "model.layers.0.experts.gate_up_proj"
    writer = TernModelWriter()
    # 5 slices for a 4-slice parent — the 5th re-opens the group post-restack.
    for k in [0, 1, 2, 3, 0]:
        _add_stacked_slice(
            writer, parent, k, stack_total=4, tensor=torch.randn(8, 16),
        )

    path = _write_temp_manifest(writer)
    try:
        with pytest.raises(ValueError) as exc_info:
            TernModelReader(str(path)).reconstruct_all()
    finally:
        os.unlink(path)
    msg = str(exc_info.value)
    assert parent in msg
    assert "1/4" in msg, (
        f"Expected leftover-guard message reporting 1/4 (5th slice reopened "
        f"the group), got: {msg}"
    )
