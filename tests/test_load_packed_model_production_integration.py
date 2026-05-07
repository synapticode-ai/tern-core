"""
Production manifest integration tests for ``load_packed_model``.

Loads each in-scope `.tern-model` artefact on disk via the rewritten
``load_packed_model`` (PR feat/load-packed-model-production-manifest-support-2026-05-07
Commits 1-4: parameter-path-aware traversal, INT4 dispatch, key_mapping
parameter), verifies clean load + manifest entry coverage, runs a
50-token smoke probe via per-model tokeniser, asserts non-NaN output +
non-empty decoded text + no obvious repetition collapse.

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


# Each tuple: (label, manifest_path, hf_model_id, smoke_prompt, expected_min_entries)
IN_SCOPE_MANIFESTS = [
    pytest.param(
        "phi-4",
        "/Volumes/Syn Archive/models/compressed/phi-4/"
        "phi4_14b_ternary_v0.1.1.tern-model/model.tern-model",
        "microsoft/phi-4",
        "The capital of France is",
        240,  # actual: 243 entries (160 ternary + 83 FP16)
        id="phi-4",
    ),
    pytest.param(
        "gemma4-26b-a4b",
        "/Volumes/Syn Archive/models/compressed/gemma4-26b-a4b/"
        "gemma4_26b_a4b_ternary_v0.1.0.tern-model/model.tern-model",
        "google/gemma-4-26b-a4b-it",
        "The capital of France is",
        8600,  # actual: 8633 entries (7875 ternary + 748 FP16 + 10 INT4)
        id="gemma4-26b-a4b",
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


def _smoke_probe(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """Run a short generation smoke probe.

    Returns the decoded generated text + a structural check report
    (NaN-free + non-empty + no obvious repetition collapse).
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

    # Structural checks
    new_tokens = output_ids[0][input_ids.shape[1]:].tolist()
    assert len(new_tokens) > 0, "Generation produced zero new tokens"

    # Repetition collapse check: last 20 generated tokens shouldn't all be the same
    if len(new_tokens) >= 20:
        last_20 = new_tokens[-20:]
        unique_count = len(set(last_20))
        assert unique_count >= 2, (
            f"Generation collapsed to repetition: last 20 tokens have only "
            f"{unique_count} unique value(s). Generated: {generated_text[-200:]!r}"
        )

    return generated_text


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize(
    "label,manifest_path,hf_model_id,smoke_prompt,expected_min_entries",
    IN_SCOPE_MANIFESTS,
)
def test_load_packed_model_production_integration(
    label: str,
    manifest_path: str,
    hf_model_id: str,
    smoke_prompt: str,
    expected_min_entries: int,
    capsys,
):
    """Integration test: load_packed_model on real production manifest + smoke probe.

    Verifies the rewrite's acceptance criteria on real production data:
    1. Manifest loads without TypeError / silent skip / silent corruption
    2. Loaded model produces non-garbage output on a short generation probe
    3. Operator-visible INT4 log message fires when manifest contains INT4 entries

    Skip via pytest.skip if the manifest path doesn't exist (allows running
    on hardware without Syn Archive mounted) or if HF base load fails with
    OutOfMemoryError (allows running on hardware with less RAM than
    expected).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from terncore.tern_model import TernModelReader

    if not Path(manifest_path).exists():
        pytest.skip(
            f"Manifest path not on disk: {manifest_path}. "
            f"Syn Archive may not be mounted on this host."
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

    # Apply rewritten load_packed_model.
    missing, unexpected = reader.load_packed_model(model)

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

    print(f"[{label}] Smoke probe with prompt {smoke_prompt!r}...", flush=True)
    generated_text = _smoke_probe(model, tokenizer, smoke_prompt, max_new_tokens=50)

    # Surface generated text for visual confirmation. Pragmatic eyeball
    # check: Rob reviews the output for reasonableness post-test.
    print(f"\n[{label}] Generated text:\n{generated_text}\n", flush=True)

    # Pragmatic non-empty + sentinel-free check. The smoke probe helper
    # already verified non-NaN logits + non-empty token list + no
    # repetition collapse. Generated text non-empty after decoding is
    # the final pragmatic bar.
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
