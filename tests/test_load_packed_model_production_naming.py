"""
Diagnostic regression tests for ``TernModelReader.load_packed_model``
on production manifest naming convention.

Production manifests (written by ``convert.py:full_convert`` for all
adapter routes) use parameter-path naming for entries —
``model.layers.0.q_proj.weight``, ``model.layers.0.norm.weight``, etc.
The current ``load_packed_model`` implementation walks ``parts[:-1]``
and dispatches on ``isinstance(getattr(parent, parts[-1]), nn.Linear)``,
producing two distinct failure modes on parameter-path manifest entries:

1. **FP16 silent skip**: when ``parts[-1] == "weight"`` (or ``.bias``)
   and the target is a non-Linear module (LayerNorm, RMSNorm, Embedding)
   or even a Linear's ``.weight`` Parameter, the ``isinstance`` check
   is False and the entry is silently skipped. Parameter retains its
   random init from model construction.

2. **Ternary load-time TypeError** (originally framed as "silent
   corruption" during 2026-05-07 Commit 1 design probe; empirically
   corrected when this file's tests surfaced PyTorch's
   ``nn.Module.__setattr__`` guard): when ``parts[-1] == "weight"`` for
   a ternary entry, ``setattr(parent_linear, "weight", PackedTernaryLinear_instance)``
   tries to replace a registered ``nn.Parameter`` with an ``nn.Module``
   instance. PyTorch raises ``TypeError: cannot assign 'PackedTernaryLinear'
   as parameter 'weight'`` immediately. Loud failure rather than silent
   corruption — better than the originally-framed mechanism but still a
   bug because production manifests fail to load.

Both failure modes share a root cause: the loader doesn't detect
parameter-path naming (suffix ``.weight`` / ``.bias``) versus
module-path naming (bare module identifier, the test convention
existing tests in ``test_packed_linear.py`` use).

These tests use a synthetic 4-layer model to isolate each bug
independently. The two manifest fixtures (FP16-only vs
ternary-included) ensure each bug is demonstrated as a distinct
failure mechanism rather than masking each other. Production-manifest
integration testing across all 5 compressed `.tern-model` artefacts
on disk is in a separate ``@pytest.mark.slow`` test (Commit 5 of this
PR ladder).

Both tests are EXPECTED to fail on the current ``load_packed_model``
and will pass after Commit 2 lands the parameter-path-aware fix.

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from terncore.tern_model import TernModelReader, TernModelWriter


# ── Synthetic fixture ──────────────────────────────────────────────


class TinyModel(nn.Module):
    """Small synthetic model that exercises both bug paths.

    - ``embedding`` (Embedding): non-Linear module → triggers FP16
      silent-skip when manifest names ``embedding.weight`` (parameter-
      path) is loaded against the current isinstance(Linear) check
    - ``norm`` (LayerNorm): two parameters (weight + bias), non-Linear
      module → triggers FP16 silent-skip on both
    - ``linear1``, ``linear2`` (Linear): triggers ternary load-time
      TypeError when manifest names ``linear1.weight`` / ``linear2.weight``
      (parameter-path) are processed by the ternary path's
      ``setattr(parent, "weight", PackedTernaryLinear_instance)``
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.norm = nn.LayerNorm(4)
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 4)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        x = self.norm(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def _build_fp16_only_manifest(
    out_path: Path,
    *,
    sentinel_norm_weight: torch.Tensor,
    sentinel_norm_bias: torch.Tensor,
    sentinel_embedding_weight: torch.Tensor,
    sentinel_linear1_bias: torch.Tensor,
) -> int:
    """Manifest with FP16 entries only — isolates the FP16 silent-skip bug.

    No ternary entries → loader doesn't hit the ternary TypeError → FP16
    silent-skip behaviour visible directly. Linear weights are NOT in this
    manifest (the manifest is incomplete by design, just for FP16 isolation).

    Returns the count of manifest entries written.
    """
    writer = TernModelWriter({"source": "synthetic-fixture-fp16-only"})

    # All FP16 entries with parameter-path naming. Each targets a
    # non-Linear module's parameter or a Linear's bias — none of these
    # match the loader's isinstance(parts[-1], nn.Linear) check.
    writer.add_layer(
        name="embedding.weight",
        weights=sentinel_embedding_weight,
        dtype="float16",
    )
    writer.add_layer(
        name="norm.weight",
        weights=sentinel_norm_weight,
        dtype="float16",
    )
    writer.add_layer(
        name="norm.bias",
        weights=sentinel_norm_bias,
        dtype="float16",
    )
    writer.add_layer(
        name="linear1.bias",
        weights=sentinel_linear1_bias,
        dtype="float16",
    )

    writer.write(out_path)
    return 4


def _build_ternary_included_manifest(
    out_path: Path,
    *,
    linear1_weight: torch.Tensor,
    linear2_weight: torch.Tensor,
) -> int:
    """Manifest with ternary entries — isolates the ternary TypeError bug.

    First entries written are ternary so the TypeError fires before any
    FP16 entries can be reached. This matches what production manifests
    look like in iteration order (ternary entries dominate by count for
    MoE manifests).

    Returns the count of manifest entries written.
    """
    writer = TernModelWriter({"source": "synthetic-fixture-ternary-included"})

    # Ternary entries first (parameter-path names linear1.weight etc.)
    packed1, alpha1, bitmap1, _ = TernModelWriter.pack_ternary(
        linear1_weight, 0.7
    )
    writer.add_ternary_layer(
        name="linear1.weight",
        packed_weights=packed1,
        alpha=alpha1,
        shape=list(linear1_weight.shape),
        sparsity_bitmap=bitmap1,
    )
    packed2, alpha2, bitmap2, _ = TernModelWriter.pack_ternary(
        linear2_weight, 0.7
    )
    writer.add_ternary_layer(
        name="linear2.weight",
        packed_weights=packed2,
        alpha=alpha2,
        shape=list(linear2_weight.shape),
        sparsity_bitmap=bitmap2,
    )
    # Plus a few FP16 entries to round out the model (the bug means
    # these will never be reached during load — TypeError fires first).
    writer.add_layer(
        name="norm.weight",
        weights=torch.ones(4),
        dtype="float16",
    )
    writer.add_layer(
        name="norm.bias",
        weights=torch.zeros(4),
        dtype="float16",
    )

    writer.write(out_path)
    return 4


# ── Tests ──────────────────────────────────────────────────────────


def test_fp16_entries_with_parameter_path_naming_load_correctly(tmp_path: Path):
    """Reproduces backlog item bug 1: FP16 silent skip on parameter-path entries.

    Production manifests name FP16 entries like ``norm.weight``. Current
    ``load_packed_model`` walks ``parts[:-1]`` to the LayerNorm, then
    ``getattr(LayerNorm, "weight")`` returns a Parameter. The loader's
    ``isinstance(Parameter, nn.Linear)`` check is False → entry silently
    skipped. The norm.weight Parameter retains its random init from
    ``TinyModel.__init__``.

    Uses FP16-only fixture to isolate this bug — no ternary entries
    means no TypeError to mask the silent-skip behaviour.

    Test sets sentinel values in the manifest; after load, parameters
    must have the sentinel values, not their random init. Currently
    fails because the entries are silently skipped (load completes with
    no exception, but parameters are unchanged from random init).
    """
    torch.manual_seed(42)
    model = TinyModel()
    initial_norm_weight = model.norm.weight.data.clone()
    initial_norm_bias = model.norm.bias.data.clone()
    initial_embedding_weight = model.embedding.weight.data.clone()
    initial_linear1_bias = model.linear1.bias.data.clone()

    sentinel_norm_weight = torch.full_like(initial_norm_weight, 7.0)
    sentinel_norm_bias = torch.full_like(initial_norm_bias, -3.0)
    sentinel_embedding_weight = torch.full_like(initial_embedding_weight, 5.0)
    sentinel_linear1_bias = torch.full_like(initial_linear1_bias, 11.0)

    out_path = tmp_path / "fp16_only.tern-model"
    _build_fp16_only_manifest(
        out_path,
        sentinel_norm_weight=sentinel_norm_weight,
        sentinel_norm_bias=sentinel_norm_bias,
        sentinel_embedding_weight=sentinel_embedding_weight,
        sentinel_linear1_bias=sentinel_linear1_bias,
    )

    reader = TernModelReader(str(out_path))
    # Load should succeed without TypeError (FP16-only manifest has no
    # ternary entries that would trigger PyTorch's __setattr__ guard).
    reader.load_packed_model(model)

    # Assert each FP16 entry landed at its target Parameter, not silently
    # skipped. If silently skipped, the parameter retains its random init.
    assert torch.allclose(model.norm.weight.data, sentinel_norm_weight), (
        f"norm.weight not loaded — current value {model.norm.weight.data} "
        f"does not match sentinel {sentinel_norm_weight}. Silent-skip bug: "
        f"loader's isinstance(Parameter, nn.Linear) check rejects non-Linear "
        f"targets, leaving the Parameter at its random init from __init__."
    )
    assert torch.allclose(model.norm.bias.data, sentinel_norm_bias), (
        f"norm.bias not loaded — same silent-skip bug as norm.weight."
    )
    assert torch.allclose(model.embedding.weight.data, sentinel_embedding_weight), (
        f"embedding.weight not loaded — Embedding module is non-Linear, "
        f"same silent-skip bug as norm."
    )
    assert torch.allclose(model.linear1.bias.data, sentinel_linear1_bias), (
        f"linear1.bias not loaded — even though linear1 IS a Linear, "
        f"parts[-1]='bias' resolves to the bias Parameter, which fails "
        f"the isinstance(nn.Linear) check; silently skipped."
    )


def test_ternary_entries_with_parameter_path_naming_load_without_typeerror(
    tmp_path: Path,
):
    """Reproduces bug 2: ternary load-time TypeError on parameter-path entries.

    Originally framed as "silent corruption" during 2026-05-07 Commit 1
    design probe — analysis predicted ``setattr(linear, "weight",
    PackedTernaryLinear_instance)`` would silently replace the Parameter
    with a Module, surfacing as ``__matmul__`` error at first forward pass.

    Empirical correction (surfaced by this file's first run): PyTorch's
    ``nn.Module.__setattr__`` raises ``TypeError: cannot assign
    'PackedTernaryLinear' as parameter 'weight' (torch.nn.Parameter or
    None expected)`` immediately at the setattr call. Loud failure
    rather than silent corruption — strictly better than the originally-
    framed mechanism (debuggable at load time rather than as confusing
    forward-pass errors), but still a bug because production manifests
    fail to load via ``load_packed_model``.

    Test asserts the EXPECTED post-fix behaviour: ``load_packed_model``
    succeeds without TypeError, ``linear1`` and ``linear2`` are replaced
    by ``PackedTernaryLinear`` instances (not their ``.weight``
    Parameters mutated), and forward pass works.

    Currently FAILS: ``reader.load_packed_model(model)`` raises
    TypeError at ``tern_model.py:1293`` (the ternary path's
    ``setattr(parent, parts[-1], packed_layer)``). After Commit 2 lands
    the parameter-path-aware fix, this test passes.
    """
    from terncore.packed_linear import PackedTernaryLinear

    torch.manual_seed(42)
    model = TinyModel()

    out_path = tmp_path / "ternary_included.tern-model"
    _build_ternary_included_manifest(
        out_path,
        linear1_weight=torch.randn(6, 4),
        linear2_weight=torch.randn(4, 6),
    )

    reader = TernModelReader(str(out_path))

    # Currently raises TypeError due to bug 2 (ternary setattr hits
    # PyTorch's __setattr__ guard). After fix, this should succeed.
    reader.load_packed_model(model)

    # Post-fix structural expectation: ternary entries replace the parent
    # Linear with a PackedTernaryLinear instance — not corrupt the
    # original Linear's .weight Parameter.
    assert isinstance(model.linear1, PackedTernaryLinear), (
        f"linear1 should be replaced by PackedTernaryLinear after load; "
        f"got type {type(model.linear1).__name__}"
    )
    assert isinstance(model.linear2, PackedTernaryLinear), (
        f"linear2 should be replaced by PackedTernaryLinear after load; "
        f"got type {type(model.linear2).__name__}"
    )

    # Forward pass smoke check — confirms the loaded model is callable
    # end-to-end. If ternary replacement happened correctly,
    # PackedTernaryLinear's forward handles the matmul.
    token_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    output = model(token_ids)

    assert torch.is_tensor(output), (
        f"Forward pass returned non-tensor: {type(output)}"
    )
    assert not torch.isnan(output).any(), "Forward pass produced NaN output"
    assert output.shape == (1, 3, 4), (
        f"Output shape {output.shape} unexpected (model defines 4-dim output)"
    )
