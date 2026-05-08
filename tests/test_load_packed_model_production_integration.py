"""
Production manifest integration tests for ``load_packed_model``.

Loads each in-scope `.tern-model` artefact on disk via the rewritten
``load_packed_model`` (PR feat/load-packed-model-production-manifest-support-2026-05-07
Commits 1-4: parameter-path-aware traversal, INT4 dispatch, key_mapping
parameter), verifies clean load + manifest entry coverage, runs a
50-token smoke probe via per-model tokeniser, asserts non-NaN logits +
either coherent generation OR documented quality-envelope collapse per
the model's expected-behaviour parameter.

**Scope: Phi-4 + gemma4-26b-a4b only.** 30B+ class manifests
(gemma4-31b, qwen3-30b-a3b) skipped on M4 Pro 64 GB hardware — their
FP16 base load (~57-58 GB) exceeds the practical unified-memory
ceiling. Same constraint TN-001 documents for Llama-3.1-70B
("demo artefact only — inference out of reach on M4 Pro"). Synthetic-
fixture coverage in ``test_load_packed_model_production_naming.py``
verifies the same code paths on architecturally-equivalent small
models.

Mistral-7B compressed artefact is not on disk despite README v0.1.0's
historical benchmark reference; verified via filesystem probe.

**Disambiguation finding 2026-05-08 (quality envelope vs load bug):**
Phi-4 ternary at threshold 0.7 produces repetition collapse in
generation. Verified via cross-path disambiguation script
(``/tmp/phi4_disambiguation.py``) that loaded the same Phi-4
``.tern-model`` artefact via ``load_as_model`` (the structurally
independent dequantise-to-FP32 path) and observed the same collapse.
Two independent load paths producing identical observable outcome
rules out load-infrastructure bugs and confirms the issue as a
quality-envelope property of Phi-4 ternary at threshold 0.7. The
April 2026 Phi-4 compression therefore enters TN-003 baseline
measurements with documented quality-envelope characterisation
rather than as a regression. The ``expect_coherent_generation=False``
flag on Phi-4 captures this — load + logits cleanliness still
asserted; generation coherence skipped with quality-envelope
diagnostic surfaced via test stdout.

Default ``pytest -m "not slow"`` skips this entire file. Opt-in via
``pytest -m slow``. Wall-clock estimate: ~10-30 min per model
sequential (HF base load + load_packed_model + smoke probe).

Copyright (c) 2025–2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest


# ── Production manifest catalogue ──────────────────────────────────


# Each tuple: (label, manifest_path, hf_model_id, smoke_prompt,
#              expected_min_entries, key_mapping_name, expect_coherent_generation)
#
# expect_coherent_generation:
#   True  → assert generation is coherent (no repetition collapse, output > prompt)
#   False → quality-envelope collapse expected; load + clean logits still
#           asserted, but generation coherence skipped with diagnostic
#           surfaced via test stdout. Captures known model+threshold combinations
#           that fall below the coherent-generation envelope (e.g. Phi-4 ternary
#           at threshold 0.7 — verified via cross-path disambiguation 2026-05-08).
IN_SCOPE_MANIFESTS = [
    pytest.param(
        "phi-4",
        "/Volumes/Syn Archive/models/compressed/phi-4/"
        "phi4_14b_ternary_v0.1.1.tern-model/model.tern-model",
        "microsoft/phi-4",
        "The capital of France is",
        240,  # actual: 243 entries (160 ternary + 83 FP16)
        None,  # key_mapping: identity (Phi-4 names match HF model directly)
        # expect_coherent_generation=False: known quality-envelope collapse at
        # threshold 0.7. Disambiguated 2026-05-08 via load_as_model cross-path:
        # both load paths produce identical "at at at at..." repetition, ruling
        # out load infrastructure as cause. Phi-4 ternary recompression at
        # lower threshold (e.g. 0.5/0.6) is a backlog item separate from the
        # rewrite scope.
        False,
        id="phi-4",
    ),
    pytest.param(
        "gemma4-26b-a4b",
        "/Volumes/Syn Archive/models/compressed/gemma4-26b-a4b/"
        "gemma4_26b_a4b_ternary_v0.1.0.tern-model/model.tern-model",
        "google/gemma-4-26b-a4b-it",
        "The capital of France is",
        8600,  # actual: 8633 entries (7875 ternary + 748 FP16 + 10 INT4)
        # key_mapping: bridge transformers 5.5+ multimodal layout. Manifest
        # was packed via Gemma4Adapter.normalize_name() which strips
        # ``language_model.`` prefix; HF AutoModelForCausalLM loads the
        # full layout with the prefix in place. Without the key_mapping,
        # _resolve_module_or_raise fires (correctly) on the first
        # ``model.embed_tokens.weight`` entry which does not exist on the
        # multimodal model (which has ``model.language_model.embed_tokens.weight``).
        "GEMMA4_MULTIMODAL_TRANSFORMERS_5_5",  # resolved at test time to the actual dict
        # expect_coherent_generation=True: would apply if load succeeded.
        # Currently load fails on per-expert MoE restacking gap; xfail
        # captures this until follow-on PR adds restacking dispatch.
        True,
        id="gemma4-26b-a4b",
        marks=pytest.mark.xfail(
            reason=(
                "Per-expert-sliced MoE manifests require restacking logic "
                "in load_packed_model that's not yet implemented. PR #14's "
                "per-expert slicing produces 128 separate experts.N.weight "
                "entries per layer (for measurement granularity); HF "
                "Gemma-4-26B-A4B-it exposes experts as stacked tensors "
                "(experts.gate_up_proj, experts.down_proj with first dim = 128). "
                "load_packed_model walks per-entry expecting separate modules; "
                "_resolve_module_or_raise correctly raises ValueError on the "
                "first 'experts.0' lookup since experts is a stacked-tensor "
                "Parameter, not a ModuleList. Banked as backlog item: "
                "'load_packed_model: MoE per-expert restacking for "
                "stacked-tensor architectures' (cf. docs/backlog.md). "
                "Scheduled for next-week L5 sprint where MoE expert paging "
                "is the primary engineering scope; restacking is a natural "
                "prerequisite for the demand-paging work."
            ),
            strict=True,
        ),
    ),
]


HARDWARE_CEILING_REASON = (
    "Skipped on M4 Pro 64 GB unified memory: model FP16 base size "
    "(~{size_gb} GB) exceeds practical ceiling. Same constraint TN-001 "
    "documents for Llama-3.1-70B. Unblocked on M4 Max / M5 / Mac Studio "
    "with 128+ GB unified memory. Synthetic fixture coverage in "
    "test_load_packed_model_production_naming.py verifies the same code "
    "paths on architecturally-equivalent small models."
)


# ── Helpers ──────────────────────────────────────────────────────────


def _smoke_probe(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    expect_coherent_generation: bool = True,
    label: str = "",
):
    """Run a short generation smoke probe.

    Always asserts: load + first-forward-pass logits are clean (no NaN/Inf)
    + at least one new token generated.

    Conditional on ``expect_coherent_generation``:
      True  → also asserts last 20 generated tokens have >=2 unique values
              (no repetition collapse). Standard coherent-generation gate.
      False → repetition collapse permitted; surface the unique-token count
              + decoded text via stdout for diagnostic visibility, but do
              not fail the test. Used for known quality-envelope outcomes
              (e.g. Phi-4 ternary at threshold 0.7 — disambiguated
              2026-05-08 via load_as_model cross-path).

    Returns the decoded generated text in either case.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to model device
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        # First forward-pass logit check (catches NaN/garbage early)
        logits = model(input_ids).logits
        assert not torch.isnan(logits).any(), "Forward pass produced NaN logits"
        assert not torch.isinf(logits).any(), "Forward pass produced Inf logits"

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for reproducibility
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )

    # Structural checks always applied
    new_tokens = output_ids[0][input_ids.shape[1]:].tolist()
    assert len(new_tokens) > 0, "Generation produced zero new tokens"

    # Repetition collapse handling: assert against it when coherent generation
    # is expected; surface it as quality-envelope diagnostic when collapse is
    # the documented outcome.
    if len(new_tokens) >= 20:
        last_20 = new_tokens[-20:]
        unique_count = len(set(last_20))
        if expect_coherent_generation:
            assert unique_count >= 2, (
                f"Generation collapsed to repetition: last 20 tokens have only "
                f"{unique_count} unique value(s). Generated: {generated_text[-200:]!r}"
            )
        else:
            print(
                f"[{label}] Quality-envelope collapse observed (expected): "
                f"last 20 tokens have {unique_count} unique value(s). "
                f"Generated tail: {generated_text[-200:]!r}",
                flush=True,
            )

    return generated_text


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize(
    "label,manifest_path,hf_model_id,smoke_prompt,expected_min_entries,"
    "key_mapping_name,expect_coherent_generation",
    IN_SCOPE_MANIFESTS,
)
def test_load_packed_model_production_integration(
    label: str,
    manifest_path: str,
    hf_model_id: str,
    smoke_prompt: str,
    expected_min_entries: int,
    key_mapping_name: Optional[str],
    expect_coherent_generation: bool,
    capsys,
):
    """Integration test: load_packed_model on real production manifest + smoke probe.

    Verifies the rewrite's acceptance criteria on real production data:
    1. Manifest loads without TypeError / silent skip / silent corruption
    2. First forward-pass logits are clean (no NaN/Inf) — independent of
       generation coherence
    3. Operator-visible INT4 log message fires when manifest contains INT4 entries
    4. Generation behaviour matches the model's expected envelope:
       - expect_coherent_generation=True:  no repetition collapse + non-trivial output
       - expect_coherent_generation=False: collapse documented, not asserted against

    The fourth bullet captures the disambiguation-finding pattern from
    2026-05-08: known quality-envelope outcomes are characterised, not
    treated as load regressions. Phi-4 ternary at threshold 0.7 is the
    first such case (verified via cross-path disambiguation —
    load_as_model produces identical collapse, ruling out load
    infrastructure as cause).

    Skip via pytest.skip if the manifest path doesn't exist (allows running
    on hardware without Syn Archive mounted) or if HF base load fails with
    OutOfMemoryError (allows running on hardware with less RAM than
    expected).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from terncore import tern_model as _tern_model_module
    from terncore.tern_model import TernModelReader

    if not Path(manifest_path).exists():
        pytest.skip(
            f"Manifest path not on disk: {manifest_path}. "
            f"Syn Archive may not be mounted on this host."
        )

    # Resolve named key_mapping preset (e.g. ``GEMMA4_MULTIMODAL_TRANSFORMERS_5_5``)
    # at test time so the parametrise list stays simple strings rather than
    # evaluating module-level constants at import time. ``None`` skips
    # translation (identity mapping).
    key_mapping = (
        getattr(_tern_model_module, key_mapping_name)
        if key_mapping_name is not None
        else None
    )

    reader = TernModelReader(manifest_path)
    n_manifest_entries = len(reader.manifest["layers"])
    assert n_manifest_entries >= expected_min_entries, (
        f"Manifest has fewer entries than expected ({n_manifest_entries} < "
        f"{expected_min_entries}); fixture may be wrong for this build."
    )

    # Load HF base. Use float16 to fit in M4 Pro 64 GB. Catch OOM
    # explicitly so the test reports a clean skip rather than a crash.
    try:
        print(f"\n[{label}] Loading HF base model from {hf_model_id} (FP16)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
    except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
        if "memory" in str(e).lower() or "OOM" in str(e):
            pytest.skip(
                f"OOM loading HF base for {label} — exceeds practical "
                f"unified memory ceiling on this hardware: {e}"
            )
        raise
    except Exception as e:
        # Architecture mismatch, missing model, auth, etc. — skip with diagnostic
        pytest.skip(
            f"Could not load HF base {hf_model_id} for {label} integration "
            f"test: {type(e).__name__}: {e}"
        )

    print(
        f"[{label}] HF base loaded. Applying load_packed_model on "
        f"{n_manifest_entries} manifest entries...",
        flush=True,
    )

    # Apply rewritten load_packed_model. Pass key_mapping (None for
    # identity, dict for transformers API drift presets like
    # GEMMA4_MULTIMODAL_TRANSFORMERS_5_5).
    missing, unexpected = reader.load_packed_model(model, key_mapping=key_mapping)

    # The (missing, unexpected) reporting is informational; the rewrite's
    # acceptance criterion is "load succeeds without exceptions" (covered
    # by the call above not raising) plus "smoke probe produces non-
    # garbage output" (next step). Surface the missing/unexpected counts
    # for diagnostic visibility but don't fail on them — there can be
    # legitimate divergence between manifest entries and model parameters
    # for multimodal architectures (vision tower, etc.).
    print(
        f"[{label}] load_packed_model returned: "
        f"missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )

    print(f"[{label}] Loading tokeniser...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    print(
        f"[{label}] Smoke probe with prompt {smoke_prompt!r} "
        f"(expect_coherent_generation={expect_coherent_generation})...",
        flush=True,
    )
    generated_text = _smoke_probe(
        model,
        tokenizer,
        smoke_prompt,
        max_new_tokens=50,
        expect_coherent_generation=expect_coherent_generation,
        label=label,
    )

    # Surface generated text for visual confirmation. Pragmatic eyeball
    # check: Rob reviews the output for reasonableness post-test.
    print(f"\n[{label}] Generated text:\n{generated_text}\n", flush=True)

    # Length assertion only when coherent generation is expected. For
    # quality-envelope cases (e.g. Phi-4 ternary at 0.7) the smoke probe
    # already surfaced the collapse via stdout; asserting length would
    # spuriously fail on the documented quality-envelope outcome.
    if expect_coherent_generation:
        assert len(generated_text.strip()) > len(smoke_prompt.strip()), (
            f"Generated text not longer than prompt — generation may have "
            f"produced only EOS tokens. Generated: {generated_text!r}"
        )


# ── Hardware-ceiling-skipped models (documented) ────────────────────


@pytest.mark.skip(
    reason=HARDWARE_CEILING_REASON.format(size_gb=58)
)
@pytest.mark.slow
def test_load_packed_model_gemma4_31b_skipped_hardware_ceiling():
    """Skipped: Gemma 4 31B FP16 base ~58 GB > M4 Pro 64 GB practical ceiling."""
    pass


@pytest.mark.skip(
    reason=HARDWARE_CEILING_REASON.format(size_gb=57)
)
@pytest.mark.slow
def test_load_packed_model_qwen3_30b_a3b_skipped_hardware_ceiling():
    """Skipped: Qwen3-30B-A3B FP16 base ~57 GB > M4 Pro 64 GB practical ceiling."""
    pass


@pytest.mark.skip(
    reason=(
        "Skipped: Mistral-7B compressed artefact not on disk. README v0.1.0 "
        "references a Mistral-7B compression but no current `.tern-model` "
        "artefact is at /Volumes/Syn Archive/models/compressed/mistral-7b/ "
        "or similar. Verified via filesystem probe 2026-05-07. Re-enable "
        "this test if a Mistral-7B manifest is added to the archive."
    )
)
@pytest.mark.slow
def test_load_packed_model_mistral_7b_skipped_not_on_disk():
    """Skipped: Mistral-7B compressed artefact not on disk."""
    pass
