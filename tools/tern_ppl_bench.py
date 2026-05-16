"""
WikiText-2 perplexity benchmark — R7-A v1.0 conformant.

Implements ``docs/wikitext2_ppl_methodology.md`` v1.0 sliding-window
evaluation protocol. The §5 pseudocode is the canonical reference;
this module reproduces its semantics verbatim. Cross-references to
spec sections (§N) appear inline at the relevant code site.

Two code paths:

* **FP16 baseline** — ``--model-id <HF id>``. Loads via
  ``AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)``.
* **Ternary variant** — ``--tern-model-path <path>``. Loads via the
  R4-C ``TernModelReader.load_packed_model`` zero-copy path. A
  HuggingFace skeleton model is materialised first (via the source
  ``--model-id`` or the manifest-resolved id) and the ternary weights
  overlay it in-place.

Output: one JSON record per run conforming to ``wikitext2_ppl/1.0``
schema (R7-A §8) under ``results/wikitext2_ppl/``.

Copyright (c) 2025-2026 Gamma Seeds Pte Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

# Repo-relative src layout — mirror the pattern from benchmarks/eval_perplexity.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import terncore  # noqa: E402
from terncore.tern_model import TernModelReader  # noqa: E402

# ── R7-A §8 schema constants ───────────────────────────────────────────

SCHEMA_VERSION = "wikitext2_ppl/1.0"
SPEC_VERSION = "wikitext2_ppl_methodology v1.0"

WIKITEXT_HF_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"

DEFAULT_SEQ_LEN = 2048
DEFAULT_STRIDE = 2048
DEFAULT_ROLLING_STRIDE = 1024  # R7-A §5 diagnostic rolling-window variant

# ── R7-B v1.1 §7 schema constants ──────────────────────────────────────

SCHEMA_VERSION_AR = "wikitext2_ppl_autoregressive/1.0"
SPEC_VERSION_AR = "wikitext2_ppl_methodology_autoregressive v1.1"
DEFAULT_NUM_SEQUENCES = 16   # R7-B v1.0 §4 canonical N
DEFAULT_AR_SEQ_LEN = 2048    # R7-B v1.0 §4 canonical L (matches R7-A seq_len)


# ── R7-A §7 ppl_headroom band classification ───────────────────────────


def classify_ppl_headroom_band(ppl_headroom: float) -> str:
    """Map a ppl_headroom value to its R7-A §7 band label."""
    if ppl_headroom < 0.02:
        return "Excellent"
    if ppl_headroom < 0.10:
        return "Acceptable"
    if ppl_headroom < 0.25:
        return "Marginal"
    return "Fail"


# ── Dataset + tokenisation (R7-A §2, §3) ───────────────────────────────


def load_wikitext2_test_text() -> tuple[str, Optional[str]]:
    """
    Load the WikiText-2 test split and concatenate per R7-A §3.

    Returns:
        (test_text, huggingface_revision_sha). The sha is the dataset
        repo commit at load time per R7-A §2. None if the lookup fails.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "HuggingFace datasets is required. Install with: "
            "pip install datasets\n"
            f"Original error: {e}"
        ) from e

    dataset = load_dataset(WIKITEXT_HF_REPO, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT)
    # R7-A §3: concatenate non-empty entries with \n\n
    test_text = "\n\n".join(s for s in dataset["text"] if s.strip())

    # R7-A §2: capture dataset revision sha. Best-effort via HF Hub API.
    revision: Optional[str] = None
    try:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(WIKITEXT_HF_REPO).sha
    except Exception:
        revision = None

    return test_text, revision


def prepare_tokens(
    test_text: str,
    tokenizer: Any,
    bos_token_id: Optional[int],
) -> torch.Tensor:
    """
    Tokenise + prepend BOS once per R7-A §3 + §4.

    R7-A §3: ``add_special_tokens=False`` — special-token insertion is
    explicit, not delegated to tokeniser defaults. R7-A §4: BOS is
    prepended ONCE at position 0 (not per window). If
    ``bos_token_id is None``, no BOS is prepended (Phi-4 convention).

    Returns:
        1D LongTensor of token ids.
    """
    encoded = tokenizer(test_text, return_tensors="pt", add_special_tokens=False)
    tokens = encoded.input_ids[0]
    if bos_token_id is not None:
        bos = torch.tensor([bos_token_id], dtype=tokens.dtype, device=tokens.device)
        tokens = torch.cat([bos, tokens], dim=0)
    return tokens


# ── Sequence construction (R7-B §4) ────────────────────────────────────


def build_sequences_autoregressive(
    tokens: torch.Tensor,
    num_sequences: int,
    seq_len: int,
    bos_token_id: Optional[int],
) -> list[list[int]]:
    """Build N sequences of length L from tokens per R7-B v1.0 §4.

    Sequences are taken sequentially from the stream (first N×L tokens),
    NOT randomly sampled — bit-reproducible per §4 determinism note.
    BOS is prepended to EACH sequence when bos_token_id is not None;
    this contrasts with R7-A's once-at-corpus-start handling because each
    R7-B sequence is an independent autoregressive context with its own
    KV-cache lifecycle.

    Returns:
        List of N inner lists. Each inner list is bos_token_id + L raw
        tokens when BOS prepended, or L raw tokens otherwise. Total scored
        positions per sequence = (len(inner) - 1) per §5 final-PPL block.
    """
    needed = num_sequences * seq_len
    if int(tokens.shape[0]) < needed:
        raise ValueError(
            f"Insufficient tokens for R7-B §4 sequence construction: have "
            f"{tokens.shape[0]}, need {needed} (N={num_sequences}, L={seq_len})."
        )
    sequences: list[list[int]] = []
    for i in range(num_sequences):
        chunk = tokens[i * seq_len : (i + 1) * seq_len].tolist()
        if bos_token_id is not None:
            chunk = [bos_token_id] + chunk
        sequences.append(chunk)
    return sequences


# ── Sliding-window evaluation (R7-A §5) ────────────────────────────────


@dataclass
class PplEvalResult:
    """Result of a single sliding-window PPL pass."""

    mean_loss: float
    ppl: float
    windows_evaluated: int
    tokens_scored: int
    tokens_discarded: int
    per_window_losses: list[float]


def evaluate_ppl(
    model: Any,
    tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    device: str,
) -> PplEvalResult:
    """
    Sliding-window PPL eval per R7-A §5 pseudocode.

    Loss is un-meaned per window (``loss * (seq_len - 1)``), summed in
    float32 (R7-A §6), and divided by total scored positions.
    """
    # R7-A §6: loss accumulator MUST be float32. We use Python float
    # which on CPython is C double (float64), strictly stronger than
    # float32 — adequate for the spec's accumulation requirement.
    total_loss_sum: float = 0.0
    total_tokens_scored: int = 0
    per_window_losses: list[float] = []
    windows_evaluated = 0

    model.eval()

    # R7-A §5 pseudocode loop bounds — last partial window discarded.
    n_tokens = int(tokens.shape[0])
    for window_start in range(0, n_tokens - seq_len + 1, stride):
        window_end = window_start + seq_len
        input_ids = tokens[window_start:window_end].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        # outputs.loss = mean cross-entropy over (seq_len - 1) scored positions
        loss_val = float(outputs.loss.item())
        # Un-mean per R7-A §5 to enable correct aggregate across equal-size windows
        window_loss_sum = loss_val * (seq_len - 1)
        total_loss_sum += window_loss_sum
        total_tokens_scored += seq_len - 1
        per_window_losses.append(loss_val)
        windows_evaluated += 1

    if total_tokens_scored == 0:
        raise ValueError(
            f"No full windows fit in {n_tokens} tokens at seq_len={seq_len}, "
            f"stride={stride}. R7-A §5 requires at least one complete window."
        )

    mean_loss = total_loss_sum / total_tokens_scored
    ppl = math.exp(mean_loss)

    # R7-A §5: tokens_discarded is the final partial window length under
    # non-overlapping stride. Define generally as n_tokens minus the last
    # window end. For rolling stride < seq_len, the same formula remains
    # interpretable: tokens beyond the last fully-fitting window.
    last_window_end = ((n_tokens - seq_len) // stride) * stride + seq_len
    tokens_discarded = n_tokens - last_window_end

    return PplEvalResult(
        mean_loss=mean_loss,
        ppl=ppl,
        windows_evaluated=windows_evaluated,
        tokens_scored=total_tokens_scored,
        tokens_discarded=tokens_discarded,
        per_window_losses=per_window_losses,
    )


# ── Autoregressive evaluation (R7-B v1.1 §5) ───────────────────────────


@dataclass
class ARPplResult:
    """Result of an R7-B autoregressive PPL pass."""

    mean_loss: float
    ppl: float
    sequence_count: int
    tokens_scored: int
    total_loss_float64: float
    eval_wall_time_seconds: float
    per_sequence_losses: list[float]


def evaluate_ppl_autoregressive(
    model: Any,
    sequences: list[list[int]],
    kv_cache_hook: Optional[Any] = None,
    device: str = "mps",
) -> ARPplResult:
    """R7-B v1.1 §5 canonical autoregressive PPL.

    Per-token forward (input = single token, past_key_values carry context);
    ``kv_cache_hook`` applied between forwards if provided; loss summed in
    Python float (CPython double = float64) per §5 accumulator-type note;
    final PPL = exp(total_loss / total_scored) across all N sequences per
    §5 final-PPL block.

    Args:
        model:          HF causal LM with use_cache support.
        sequences:      List of N sequences from build_sequences_autoregressive.
        kv_cache_hook:  Optional callable(past_kv) -> past_kv applied between
                        forward calls. Closure lifetime is the caller's
                        concern (one factory per measurement per §5.4).
        device:         Device string for tensor placement.
    """
    import torch.nn.functional as F

    if kv_cache_hook is None:
        kv_cache_hook = lambda kv: kv

    model.eval()

    total_loss: float = 0.0
    total_scored: int = 0
    per_sequence_losses: list[float] = []

    t_start = time.perf_counter()
    with torch.no_grad():
        for seq in sequences:
            past_kv = None
            seq_loss: float = 0.0
            L_eff = len(seq)
            for t in range(L_eff - 1):
                input_ids = torch.tensor([[seq[t]]], device=device)
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = outputs.past_key_values
                # Per R7-B v1.1 §5.2: kv_cache_hook returns DynamicCache
                # directly. The legacy-tuple wrapping bridge from PR #30 is
                # removed; the hook factories (PR #26, PR #27) now produce
                # Cache-shaped objects natively.
                past_kv = kv_cache_hook(past_kv)

                logits = outputs.logits[0, -1]
                target = torch.tensor(seq[t + 1], device=device)
                # reduction='sum' on a single-position pair gives the
                # per-token CE; .item() casts to Python float (float64).
                loss = F.cross_entropy(
                    logits.unsqueeze(0),
                    target.unsqueeze(0),
                    reduction="sum",
                ).item()
                seq_loss += loss
                total_loss += loss
                total_scored += 1
            per_sequence_losses.append(seq_loss)

    eval_wall_time = time.perf_counter() - t_start

    if total_scored == 0:
        raise ValueError(
            "R7-B §5 autoregressive eval scored zero tokens — empty or "
            "single-token sequences cannot produce a PPL value."
        )

    mean_loss = total_loss / total_scored
    ppl = math.exp(mean_loss)

    return ARPplResult(
        mean_loss=mean_loss,
        ppl=ppl,
        sequence_count=len(sequences),
        tokens_scored=total_scored,
        total_loss_float64=total_loss,
        eval_wall_time_seconds=eval_wall_time,
        per_sequence_losses=per_sequence_losses,
    )


# ── Model loading (R7-A §7) ────────────────────────────────────────────


def load_fp16_baseline(model_id: str, device: str) -> tuple[Any, Any]:
    """
    Load FP16 baseline per R7-A §7: ``AutoModelForCausalLM.from_pretrained
    (model_id, torch_dtype=torch.float16)`` + matching tokenizer.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers required: pip install terncore[transformers]"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def load_ternary_variant(
    tern_model_path: Path,
    source_model_id: Optional[str],
    tokenizer_id: Optional[str],
    device: str,
    key_mapping: Optional[dict] = None,
) -> tuple[Any, Any, TernModelReader]:
    """
    Load ternary variant via R4-C ``load_packed_model`` path.

    The HF skeleton must be materialised first so ``load_packed_model``
    can overlay the packed weights in-place. Tokeniser resolves from the
    explicit ``--tokenizer`` override, else from ``--model-id`` (which
    must match the source the artefact was compressed from).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers required: pip install terncore[transformers]"
        ) from e

    reader = TernModelReader(tern_model_path)
    manifest_meta = reader.manifest.get("model_metadata", {})
    resolved_source = (
        source_model_id
        or manifest_meta.get("source")
        or manifest_meta.get("model_id")
    )
    if resolved_source is None:
        raise ValueError(
            "Could not resolve source model id. Pass --model-id explicitly "
            "or ensure the .tern-model manifest has model_metadata.source."
        )

    tokenizer_source = tokenizer_id or resolved_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    # FP32 skeleton load; load_packed_model overlays ternary layers in-place.
    # Mirrors tools/tern_loader.py cmd_infer().
    model = AutoModelForCausalLM.from_pretrained(
        resolved_source,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    reader.load_packed_model(model, key_mapping=key_mapping)
    model = model.to(device)
    model.eval()
    return model, tokenizer, reader


def manifest_sha256(reader: TernModelReader) -> str:
    """SHA-256 of the manifest JSON bytes as stored in the .tern-model file."""
    with open(reader.path, "rb") as f:
        f.seek(reader.header["manifest_offset"])
        manifest_bytes = f.read(reader.header["manifest_size"])
    return hashlib.sha256(manifest_bytes).hexdigest()


# ── Provenance helpers ─────────────────────────────────────────────────


def git_commit_short() -> str:
    """Return 10-char git sha of HEAD; 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=10", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now_compact() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ── Results JSON assembly (R7-A §8) ────────────────────────────────────


def build_results_json(
    *,
    variant: str,  # "fp16" | "ternary"
    model_id: str,
    source_path: str,
    tern_model_manifest_sha256: Optional[str],
    tokenizer_source: str,
    bos_token_id: Optional[int],
    bos_prepended: bool,
    huggingface_revision: Optional[str],
    total_tokens: int,
    seq_len: int,
    stride: int,
    rolling_variant_included: bool,
    device: str,
    dtype_activation: str,
    batch_size: int,
    eval_result: PplEvalResult,
    ppl_rolling: Optional[float],
    comparison_baseline_run_id: Optional[str] = None,
    comparison_baseline_ppl: Optional[float] = None,
    notes: str = "",
    store_per_window_losses: bool = True,
    run_id: Optional[str] = None,
) -> dict:
    """Build a dict conformant to wikitext2_ppl/1.0 schema (R7-A §8)."""

    if comparison_baseline_ppl is not None and comparison_baseline_ppl > 0:
        ppl_headroom = (eval_result.ppl - comparison_baseline_ppl) / comparison_baseline_ppl
        ppl_headroom_band: Optional[str] = classify_ppl_headroom_band(ppl_headroom)
    else:
        ppl_headroom = None
        ppl_headroom_band = None

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id or utc_now_compact(),
        "timestamp_utc": utc_now_iso(),
        "tern_core_version": terncore.__version__,
        "tern_core_git_commit": git_commit_short(),
        "model": {
            "model_id": model_id,
            "variant": variant,
            "source_path": source_path,
            "tern_model_manifest_sha256": tern_model_manifest_sha256,
        },
        "tokeniser": {
            "source": tokenizer_source,
            "bos_token_id": bos_token_id,
            "bos_prepended": bos_prepended,
        },
        "dataset": {
            "name": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "huggingface_revision": huggingface_revision,
            "total_tokens": total_tokens,
            "tokens_discarded": eval_result.tokens_discarded,
        },
        "methodology": {
            "spec_version": SPEC_VERSION,
            "seq_len": seq_len,
            "stride": stride,
            "rolling_variant_included": rolling_variant_included,
        },
        "hardware": {
            "device": device,
            "dtype_activation": dtype_activation,
            "dtype_loss": "float32",
            "batch_size": batch_size,
        },
        "results": {
            "windows_evaluated": eval_result.windows_evaluated,
            "tokens_scored": eval_result.tokens_scored,
            "mean_loss": round(eval_result.mean_loss, 6),
            "ppl": round(eval_result.ppl, 4),
            "ppl_rolling": (round(ppl_rolling, 4) if ppl_rolling is not None else None),
            "per_window_losses": (
                [round(x, 6) for x in eval_result.per_window_losses]
                if store_per_window_losses
                else None
            ),
        },
        "comparison": {
            "baseline_run_id": comparison_baseline_run_id,
            "baseline_ppl": (
                round(comparison_baseline_ppl, 4)
                if comparison_baseline_ppl is not None
                else None
            ),
            "ppl_headroom": (
                round(ppl_headroom, 4) if ppl_headroom is not None else None
            ),
            "ppl_headroom_band": ppl_headroom_band,
        },
        "notes": notes,
    }


def build_results_json_autoregressive(
    *,
    model_id: str,
    dtype: str,
    device: str,
    tokenizer_source: str,
    bos_token_id: Optional[int],
    vocab_size: int,
    dataset_revision: Optional[str],
    sequence_length: int,
    sequence_count: int,
    kv_cache_hook_enabled: bool,
    kv_cache_hook_spec: str,
    kv_cache_hook_parameters: dict,
    factory_build_seconds: float,
    model_load_seconds: float,
    eval_result: ARPplResult,
    comparison_baseline_run_id: Optional[str] = None,
    comparison_baseline_ppl: Optional[float] = None,
    comparison_baseline_methodology: Optional[str] = None,
    notes: str = "",
    run_id: Optional[str] = None,
) -> dict:
    """Build a dict conformant to wikitext2_ppl_autoregressive/1.0 (R7-B §7)."""
    if comparison_baseline_ppl is not None and comparison_baseline_ppl > 0:
        ppl_headroom = (
            eval_result.ppl - comparison_baseline_ppl
        ) / comparison_baseline_ppl
        ppl_headroom_band: Optional[str] = classify_ppl_headroom_band(ppl_headroom)
    else:
        ppl_headroom = None
        ppl_headroom_band = None

    try:
        import transformers as _tx

        transformers_version = _tx.__version__
    except ImportError:
        transformers_version = "unknown"

    tokens_per_second = (
        eval_result.tokens_scored / eval_result.eval_wall_time_seconds
        if eval_result.eval_wall_time_seconds > 0
        else 0.0
    )

    return {
        "schema_version": SCHEMA_VERSION_AR,
        "run_id": run_id or utc_now_compact(),
        "tern_core_version": terncore.__version__,
        "tern_core_git_commit": git_commit_short(),
        "spec_version": SPEC_VERSION_AR,
        "model": {
            "model_id": model_id,
            "huggingface_revision": None,  # populated by caller if resolvable
            "dtype": dtype,
            "device": device,
        },
        "tokeniser": {
            "tokenizer_id": tokenizer_source,
            "bos_token_id": bos_token_id,
            "vocab_size": vocab_size,
        },
        "dataset": {
            "name": WIKITEXT_CONFIG,
            "split": WIKITEXT_SPLIT,
            "dataset_revision": dataset_revision,
        },
        "config": {
            "sequence_length": sequence_length,
            "sequence_count": sequence_count,
            "stride_between_sequences": sequence_length,  # R7-B §4: non-overlapping
            "bos_handling": (
                "prepend_per_sequence" if bos_token_id is not None else "none"
            ),
            "eos_insertion": "none",
            "kv_cache_compression": {
                "enabled": kv_cache_hook_enabled,
                "hook_spec": kv_cache_hook_spec,
                "parameters": kv_cache_hook_parameters,
            },
        },
        "results": {
            "total_loss_float64": eval_result.total_loss_float64,
            "tokens_scored": eval_result.tokens_scored,
            "mean_loss": round(eval_result.mean_loss, 6),
            "ppl_autoregressive": round(eval_result.ppl, 4),
            "per_sequence_losses": [
                round(x, 6) for x in eval_result.per_sequence_losses
            ],
        },
        "comparison": {
            "baseline_run_id": comparison_baseline_run_id,
            "baseline_ppl": (
                round(comparison_baseline_ppl, 4)
                if comparison_baseline_ppl is not None
                else None
            ),
            "baseline_methodology": comparison_baseline_methodology,
            "ppl_headroom": (
                round(ppl_headroom, 4) if ppl_headroom is not None else None
            ),
            "ppl_headroom_band": ppl_headroom_band,
        },
        "timing": {
            "model_load_seconds": round(model_load_seconds, 2),
            "factory_build_seconds": round(factory_build_seconds, 3),
            "eval_wall_time_seconds": round(eval_result.eval_wall_time_seconds, 2),
            "tokens_per_second": round(tokens_per_second, 2),
        },
        "hardware": {
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
        },
        "notes": notes,
    }


def model_short_label(variant: str, model_id: str, tern_path: Optional[Path]) -> str:
    """Derive the <model_short> portion of the canonical output filename."""
    if variant == "ternary" and tern_path is not None:
        stem = tern_path.stem.replace(".tern-model", "")
        # If parent dir name carries the canonical short label, prefer it.
        return tern_path.parent.name if tern_path.parent.name else stem
    return model_id.rsplit("/", 1)[-1].lower()


# ── KV-cache hook factory selection (R12 v1.1 §8.2 / R7-B v1.1 §5.4) ───


def build_kv_cache_hook(
    hook_spec: str,
    b_mse: int,
    model: Any,
    device: str,
) -> tuple[Optional[Any], dict, float]:
    """Construct an R12 kv_cache_hook + capture factory build time (Q6).

    Args:
        hook_spec: ``"none"`` | ``"mixed"`` | ``"uniform"``.
        b_mse:     bits-of-MSE parameter when hook_spec != ``"none"``.
        model:     loaded HF causal LM (for ``model.config`` reflection).
        device:    target device for the hook's TurboQuant operations.

    Returns:
        (hook, parameters_dict, factory_build_seconds). When hook_spec is
        ``"none"``, returns (None, {}, 0.0). For ``"mixed"`` / ``"uniform"``,
        imports the factory from ``tools.tern_infer`` and times the call.

    Note: ``"mixed"`` factory has a lazy outlier-detection cold cost on
    first hook invocation (~10 min on TinyLlama per R12 v1.1 §8.2.1), not
    captured in factory_build_seconds. ``"uniform"`` (β1a) has near-zero
    cold cost because ``mixed_precision=False`` short-circuits the lazy path.
    """
    if hook_spec == "none":
        return None, {}, 0.0

    # Reflect cache geometry from model.config — see R12 v1.1 §8.2.4.
    cfg = model.config
    n_layers = int(cfg.num_hidden_layers)
    n_heads = int(getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads)
    head_dim = int(cfg.hidden_size // cfg.num_attention_heads)

    # Late import to avoid pulling tern_infer's deps on R7-A paths.
    _tools_dir = str(Path(__file__).resolve().parent)
    if _tools_dir not in sys.path:
        sys.path.insert(0, _tools_dir)
    from tern_infer import make_b_mse_hook, make_b_mse_hook_uniform  # noqa: E402

    if hook_spec == "mixed":
        factory = make_b_mse_hook
    elif hook_spec == "uniform":
        factory = make_b_mse_hook_uniform
    else:
        raise ValueError(
            f"Unknown hook_spec={hook_spec!r}; expected one of "
            f"'none', 'mixed', 'uniform'."
        )

    t0 = time.perf_counter()
    hook = factory(
        b_mse=b_mse,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
        device=device,
    )
    factory_build_seconds = time.perf_counter() - t0

    parameters = {
        "b_mse": b_mse,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "device": device,
        "factory_name": factory.__name__,
    }
    return hook, parameters, factory_build_seconds


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WikiText-2 PPL benchmark — R7-A v1.0 conformant",
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--model-id",
        type=str,
        help="HuggingFace model id for FP16 baseline path",
    )
    src_group.add_argument(
        "--tern-model-path",
        type=Path,
        help="Path to .tern-model artefact for ternary-variant path",
    )

    parser.add_argument(
        "--source-model-id",
        type=str,
        default=None,
        help=(
            "When using --tern-model-path: HF source id the artefact was "
            "compressed from (overrides manifest.model_metadata.source)"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Override tokenizer source (default: resolves from model id)",
    )
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--rolling-variant",
        action="store_true",
        help=(
            "Also compute rolling-window PPL (stride=1024) and report under "
            "results.ppl_rolling (R7-A §5 diagnostic variant)"
        ),
    )
    parser.add_argument(
        "--rolling-stride",
        type=int,
        default=DEFAULT_ROLLING_STRIDE,
        help="Stride used for the rolling-variant pass (default 1024)",
    )
    parser.add_argument(
        "--force-no-bos",
        action="store_true",
        help="Disable BOS prepend even if tokenizer.bos_token_id is set",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "wikitext2_ppl",
    )
    parser.add_argument(
        "--baseline-run-id",
        type=str,
        default=None,
        help="If set, populate comparison.baseline_run_id in output",
    )
    parser.add_argument(
        "--baseline-ppl",
        type=float,
        default=None,
        help=(
            "If set, populate comparison.baseline_ppl and compute "
            "comparison.ppl_headroom"
        ),
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form notes appended to results.notes",
    )
    parser.add_argument(
        "--no-per-window-losses",
        action="store_true",
        help="Omit results.per_window_losses array (smaller JSON)",
    )

    # ── R7-B v1.1 §5 / R12 v1.1 §8.2 — autoregressive methodology flags ─
    parser.add_argument(
        "--methodology",
        choices=["sliding", "autoregressive"],
        default="sliding",
        help="R7-A sliding-window (default) or R7-B v1.1 autoregressive",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=DEFAULT_NUM_SEQUENCES,
        help="R7-B v1.0 §4 N (default 16)",
    )
    parser.add_argument(
        "--ar-seq-len",
        type=int,
        default=DEFAULT_AR_SEQ_LEN,
        help="R7-B v1.0 §4 L (default 2048)",
    )
    parser.add_argument(
        "--kv-cache-hook",
        choices=["none", "mixed", "uniform"],
        default="none",
        help=(
            "R12 KV-cache compression hook: none (baseline), mixed "
            "(make_b_mse_hook reference factory, PR #26), uniform "
            "(make_b_mse_hook_uniform β1a factory, PR #27)"
        ),
    )
    parser.add_argument(
        "--b-mse",
        type=int,
        default=4,
        help="b_mse parameter when --kv-cache-hook != none (default 4)",
    )

    args = parser.parse_args()

    is_ternary = args.tern_model_path is not None
    variant = "ternary" if is_ternary else "fp16"
    methodology = args.methodology

    # Resolve output-dir default per methodology if user did not override.
    _r7a_default_dir = (
        Path(__file__).resolve().parent.parent / "results" / "wikitext2_ppl"
    )
    if methodology == "autoregressive" and args.output_dir == _r7a_default_dir:
        args.output_dir = (
            Path(__file__).resolve().parent.parent
            / "results"
            / "wikitext2_ppl_autoregressive"
        )

    methodology_tag = "R7-A v1.0" if methodology == "sliding" else "R7-B v1.1"
    print(f"[tern_ppl_bench] {methodology_tag} — variant={variant}")
    t_load = time.perf_counter()

    if is_ternary:
        model, tokenizer, reader = load_ternary_variant(
            tern_model_path=args.tern_model_path,
            source_model_id=args.source_model_id,
            tokenizer_id=args.tokenizer,
            device=args.device,
        )
        manifest_sha = manifest_sha256(reader)
        model_id = (
            args.source_model_id
            or reader.manifest.get("model_metadata", {}).get("source")
            or "unknown"
        )
        source_path = str(args.tern_model_path)
        dtype_activation = "mixed"
    else:
        model, tokenizer = load_fp16_baseline(args.model_id, args.device)
        manifest_sha = None
        model_id = args.model_id
        source_path = args.model_id
        dtype_activation = "float16"

    tokenizer_source = args.tokenizer or model_id
    model_load_seconds = time.perf_counter() - t_load
    print(f"[tern_ppl_bench]   model loaded in {model_load_seconds:.1f}s")

    # ── Dataset + tokens ────────────────────────────────────────────
    print("[tern_ppl_bench]   loading WikiText-2 test split...")
    test_text, hf_revision = load_wikitext2_test_text()

    bos_token_id = None if args.force_no_bos else tokenizer.bos_token_id
    tokens = prepare_tokens(test_text, tokenizer, bos_token_id)
    bos_prepended = bos_token_id is not None
    print(
        f"[tern_ppl_bench]   tokens: {tokens.shape[0]:,} "
        f"(bos_prepended={bos_prepended})"
    )

    if methodology == "sliding":
        # ── R7-A v1.0 sliding-window headline eval ────────────────────
        print(
            f"[tern_ppl_bench]   eval: seq_len={args.seq_len}, stride={args.stride}"
        )
        t_eval = time.perf_counter()
        result = evaluate_ppl(
            model=model,
            tokens=tokens,
            seq_len=args.seq_len,
            stride=args.stride,
            device=args.device,
        )
        print(
            f"[tern_ppl_bench]   PPL = {result.ppl:.4f} "
            f"(windows={result.windows_evaluated}, "
            f"scored={result.tokens_scored:,}, "
            f"discarded={result.tokens_discarded}, "
            f"{time.perf_counter() - t_eval:.1f}s)"
        )

        # Optional rolling-window diagnostic
        ppl_rolling: Optional[float] = None
        if args.rolling_variant:
            print(
                f"[tern_ppl_bench]   rolling-variant pass: stride={args.rolling_stride}"
            )
            rolling = evaluate_ppl(
                model=model,
                tokens=tokens,
                seq_len=args.seq_len,
                stride=args.rolling_stride,
                device=args.device,
            )
            ppl_rolling = rolling.ppl
            print(f"[tern_ppl_bench]   PPL (rolling) = {rolling.ppl:.4f}")

        # Assemble + persist R7-A JSON
        record = build_results_json(
            variant=variant,
            model_id=model_id,
            source_path=source_path,
            tern_model_manifest_sha256=manifest_sha,
            tokenizer_source=tokenizer_source,
            bos_token_id=bos_token_id,
            bos_prepended=bos_prepended,
            huggingface_revision=hf_revision,
            total_tokens=int(tokens.shape[0]),
            seq_len=args.seq_len,
            stride=args.stride,
            rolling_variant_included=args.rolling_variant,
            device=args.device,
            dtype_activation=dtype_activation,
            batch_size=1,
            eval_result=result,
            ppl_rolling=ppl_rolling,
            comparison_baseline_run_id=args.baseline_run_id,
            comparison_baseline_ppl=args.baseline_ppl,
            notes=args.notes,
            store_per_window_losses=not args.no_per_window_losses,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        short = model_short_label(variant, model_id, args.tern_model_path)
        out_path = (
            args.output_dir / f"ppl_{short}_{variant}_{record['run_id']}.json"
        )
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        print(f"[tern_ppl_bench]   wrote {out_path}")
        print(f"[tern_ppl_bench]   run_id = {record['run_id']}")
        print(f"[tern_ppl_bench]   ppl    = {record['results']['ppl']}")
        return

    # ── R7-B v1.1 §5 autoregressive eval ────────────────────────────
    print(
        f"[tern_ppl_bench]   eval: N={args.num_sequences}, L={args.ar_seq_len} "
        f"(R7-B v1.1 §4 — non-overlapping, BOS-per-sequence={bos_prepended})"
    )
    sequences = build_sequences_autoregressive(
        tokens=tokens,
        num_sequences=args.num_sequences,
        seq_len=args.ar_seq_len,
        bos_token_id=bos_token_id,
    )

    # Construct kv_cache_hook (factory build time captured for Q6 visibility).
    hook, hook_parameters, factory_build_seconds = build_kv_cache_hook(
        hook_spec=args.kv_cache_hook,
        b_mse=args.b_mse,
        model=model,
        device=args.device,
    )
    hook_enabled = hook is not None
    if hook_enabled:
        print(
            f"[tern_ppl_bench]   factory build: {factory_build_seconds:.2f}s "
            f"({hook_parameters['factory_name']}, b_mse={args.b_mse})"
        )
        if args.kv_cache_hook == "mixed":
            print(
                "[tern_ppl_bench]     note: first-call may include lazy cold "
                "warmup for mixed; see R12 v1.1 §8.2.1"
            )

    ar_result = evaluate_ppl_autoregressive(
        model=model,
        sequences=sequences,
        kv_cache_hook=hook,
        device=args.device,
    )
    tokens_per_second = (
        ar_result.tokens_scored / ar_result.eval_wall_time_seconds
        if ar_result.eval_wall_time_seconds > 0
        else 0.0
    )
    print(
        f"[tern_ppl_bench]   PPL (autoregressive) = {ar_result.ppl:.4f} "
        f"(sequences={ar_result.sequence_count}, "
        f"scored={ar_result.tokens_scored:,}, "
        f"{ar_result.eval_wall_time_seconds:.1f}s, "
        f"{tokens_per_second:.1f} tok/s)"
    )

    # Vocab size best-effort from tokenizer
    vocab_size = int(getattr(tokenizer, "vocab_size", 0)) or len(
        getattr(tokenizer, "get_vocab", lambda: {})() or {}
    )

    record = build_results_json_autoregressive(
        model_id=model_id,
        dtype="float16" if not is_ternary else "mixed",
        device=args.device,
        tokenizer_source=tokenizer_source,
        bos_token_id=bos_token_id,
        vocab_size=vocab_size,
        dataset_revision=hf_revision,
        sequence_length=args.ar_seq_len,
        sequence_count=args.num_sequences,
        kv_cache_hook_enabled=hook_enabled,
        kv_cache_hook_spec=args.kv_cache_hook,
        kv_cache_hook_parameters=hook_parameters,
        factory_build_seconds=factory_build_seconds,
        model_load_seconds=model_load_seconds,
        eval_result=ar_result,
        comparison_baseline_run_id=args.baseline_run_id,
        comparison_baseline_ppl=args.baseline_ppl,
        comparison_baseline_methodology=None,
        notes=args.notes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    short = model_short_label(variant, model_id, args.tern_model_path)
    dtype_tag = "ternary" if is_ternary else "fp16"
    out_path = args.output_dir / f"ppl_ar_{short}_{dtype_tag}_{record['run_id']}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[tern_ppl_bench]   wrote {out_path}")
    print(f"[tern_ppl_bench]   run_id = {record['run_id']}")
    print(f"[tern_ppl_bench]   ppl_ar = {record['results']['ppl_autoregressive']}")


if __name__ == "__main__":
    main()
