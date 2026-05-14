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


def model_short_label(variant: str, model_id: str, tern_path: Optional[Path]) -> str:
    """Derive the <model_short> portion of the canonical output filename."""
    if variant == "ternary" and tern_path is not None:
        stem = tern_path.stem.replace(".tern-model", "")
        # If parent dir name carries the canonical short label, prefer it.
        return tern_path.parent.name if tern_path.parent.name else stem
    return model_id.rsplit("/", 1)[-1].lower()


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

    args = parser.parse_args()

    is_ternary = args.tern_model_path is not None
    variant = "ternary" if is_ternary else "fp16"

    print(f"[tern_ppl_bench] R7-A v1.0 — variant={variant}")
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
    print(f"[tern_ppl_bench]   model loaded in {time.perf_counter() - t_load:.1f}s")

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

    # ── Headline eval ──────────────────────────────────────────────
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

    # ── Optional rolling-window diagnostic ──────────────────────────
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

    # ── Assemble + persist JSON ────────────────────────────────────
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
    out_path = args.output_dir / f"ppl_{short}_{variant}_{record['run_id']}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"[tern_ppl_bench]   wrote {out_path}")
    print(f"[tern_ppl_bench]   run_id = {record['run_id']}")
    print(f"[tern_ppl_bench]   ppl    = {record['results']['ppl']}")


if __name__ == "__main__":
    main()
