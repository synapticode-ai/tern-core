"""
Perplexity evaluation: FP32 baseline vs ternary on WikiText-2.

Downloads a HuggingFace causal LM (default: TinyLlama-1.1B-Chat-v1.0),
evaluates FP32 baseline perplexity on WikiText-2 test set using the
standard sliding-window approach, converts to ternary via
TernaryInferenceEngine, and re-evaluates ternary perplexity.

Reports: FP baseline PPL, ternary PPL, percentage gap, sparsity ratio,
and compression ratio.

Usage:
    python benchmarks/eval_perplexity.py
    python benchmarks/eval_perplexity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python benchmarks/eval_perplexity.py --threshold 0.5 --stride 256
    python benchmarks/eval_perplexity.py --json-only

Patent 12: Auto binary-to-ternary conversion.
Patent 36: Deterministic execution guarantee.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Ensure tern-core is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.accel import get_acceleration_info, is_accelerated
from terncore.engine.inference import TernaryInferenceEngine
from terncore.memory import profile_model_memory


# ═══════════════════════════════════════════════════════════════
# Dependency guards
# ═══════════════════════════════════════════════════════════════

_HF_AVAILABLE = False
_HF_IMPORT_ERROR: Optional[str] = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HF_AVAILABLE = True
except ImportError as e:
    _HF_IMPORT_ERROR = str(e)

_DATASETS_AVAILABLE = False
_DATASETS_IMPORT_ERROR: Optional[str] = None

try:
    from datasets import load_dataset

    _DATASETS_AVAILABLE = True
except ImportError as e:
    _DATASETS_IMPORT_ERROR = str(e)


def _require_dependencies() -> None:
    """Raise ImportError with install instructions if dependencies are missing."""
    if not _HF_AVAILABLE:
        raise ImportError(
            "HuggingFace transformers is required for perplexity evaluation. "
            "Install with: pip install terncore[transformers]\n"
            f"Original error: {_HF_IMPORT_ERROR}"
        )
    if not _DATASETS_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets is required for WikiText-2 loading. "
            "Install with: pip install datasets\n"
            f"Original error: {_DATASETS_IMPORT_ERROR}"
        )


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_THRESHOLD = 0.7
DEFAULT_STRIDE = 512
DEFAULT_MAX_LENGTH = 0  # 0 = use model's max_position_embeddings
SEED = 0  # Determinism (Patent 36)
WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"
RESULTS_MD_PATH = Path(__file__).resolve().parent / "RESULTS.md"


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class PerplexityResult:
    """Result from a single perplexity evaluation pass."""

    phase: str  # "fp32_baseline", "ternary_pytorch", "ternary_accel"
    perplexity: float
    avg_nll: float  # average negative log-likelihood per token
    num_tokens_scored: int
    num_windows: int
    eval_time_s: float


@dataclass
class EvalReport:
    """Full perplexity evaluation report."""

    timestamp: str
    model_id: str
    platform: dict
    acceleration: dict
    config: dict
    model_info: dict
    conversion_time_s: float
    sparsity_ratio: float
    compression_ratio: float
    results: list[PerplexityResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════


def _load_wikitext2(tokenizer: Any) -> torch.Tensor:
    """
    Load WikiText-2 test set and tokenize as a single concatenated sequence.

    Returns:
        input_ids: LongTensor of shape (1, seq_len).
    """
    dataset = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT)

    # Concatenate non-empty text entries (standard HF perplexity approach)
    text = "\n\n".join(
        entry["text"] for entry in dataset if entry["text"].strip()
    )

    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids


# ═══════════════════════════════════════════════════════════════
# Perplexity calculation — sliding window
# ═══════════════════════════════════════════════════════════════


def evaluate_perplexity(
    model: nn.Module,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
    phase_name: str,
) -> PerplexityResult:
    """
    Compute perplexity using the standard sliding-window approach.

    Per HuggingFace perplexity docs: slide a window of max_length tokens
    with stride step size.  For each window, compute cross-entropy loss
    only on the new (non-overlapping) tokens.

    Patent 36: Deterministic execution — fixed seed, eval mode.

    Args:
        model:       The model (FP32 or ternary-converted).
        input_ids:   Token IDs of shape (1, seq_len).
        stride:      Number of new tokens per window.
        max_length:  Context window size.
        phase_name:  Label for this evaluation phase.

    Returns:
        PerplexityResult with perplexity and evaluation metadata.
    """
    model.eval()
    torch.manual_seed(SEED)

    seq_len = input_ids.size(1)
    nlls: list[torch.Tensor] = []
    num_windows = 0
    prev_end = 0

    t0 = time.perf_counter()

    for i in range(0, seq_len, stride):
        begin = max(i + stride - max_length, 0)
        end = min(i + stride, seq_len)
        target_len = end - prev_end

        trg_input = input_ids[:, begin:end]
        target_ids = trg_input.clone()
        # Mask overlap tokens so loss ignores them
        target_ids[:, :-target_len] = -100

        with torch.no_grad():
            outputs = model(trg_input, labels=target_ids)
            neg_log_likelihood = outputs.loss * target_len

        nlls.append(neg_log_likelihood)
        num_windows += 1
        prev_end = end

        # Progress reporting
        pct = end / seq_len * 100
        if num_windows % 10 == 0 or end >= seq_len:
            current_ppl = torch.exp(torch.stack(nlls).sum() / end).item()
            print(
                f"\r    [{phase_name}] "
                f"{end:,}/{seq_len:,} tokens "
                f"({pct:.0f}%) — running PPL: {current_ppl:.2f}",
                end="",
                flush=True,
            )

        if end >= seq_len:
            break

    eval_time = time.perf_counter() - t0
    total_scored = prev_end

    total_nll = torch.stack(nlls).sum()
    avg_nll = (total_nll / total_scored).item()

    try:
        ppl = math.exp(avg_nll)
    except OverflowError:
        ppl = float("inf")

    print()  # newline after progress

    return PerplexityResult(
        phase=phase_name,
        perplexity=ppl,
        avg_nll=avg_nll,
        num_tokens_scored=total_scored,
        num_windows=num_windows,
        eval_time_s=eval_time,
    )


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _calculate_sparsity(model: nn.Module) -> float:
    """Calculate overall sparsity ratio across ternary layers."""
    from terncore.arithmetic.linear import TernaryLinear

    total_weights = 0
    zero_weights = 0

    for module in model.modules():
        if isinstance(module, TernaryLinear):
            if module._cached_ternary is None:
                module._cache_ternary_weights()
            tw = module._cached_ternary
            total_weights += tw.numel()
            zero_weights += (tw == 0).sum().item()

    if total_weights == 0:
        return 0.0
    return zero_weights / total_weights


def _gather_model_info(model: nn.Module, model_id: str) -> dict:
    """Extract architecture information from a loaded model."""
    total_params = sum(p.numel() for p in model.parameters())
    config = getattr(model, "config", None)

    linear_count = sum(
        1 for m in model.modules() if isinstance(m, nn.Linear)
    )

    return {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "total_params": total_params,
        "num_linear_layers": linear_count,
        "vocab_size": getattr(config, "vocab_size", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "num_hidden_layers": getattr(config, "num_hidden_layers", 0),
    }


# ═══════════════════════════════════════════════════════════════
# Main evaluation
# ═══════════════════════════════════════════════════════════════


def run_evaluation(
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_THRESHOLD,
    stride: int = DEFAULT_STRIDE,
    max_length: int = DEFAULT_MAX_LENGTH,
    skip_accel: bool = False,
) -> EvalReport:
    """Run the full perplexity evaluation."""
    import platform as plat

    _require_dependencies()

    torch.manual_seed(SEED)
    accel_info = get_acceleration_info()

    report = EvalReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        model_id=model_id,
        platform={
            "machine": plat.machine(),
            "processor": plat.processor(),
            "system": plat.system(),
            "python": plat.python_version(),
            "torch": torch.__version__,
        },
        acceleration=accel_info,
        config={
            "threshold": threshold,
            "stride": stride,
            "max_length": max_length,
            "seed": SEED,
            "dataset": f"{WIKITEXT_DATASET}/{WIKITEXT_CONFIG}",
        },
        model_info={},
        conversion_time_s=0.0,
        sparsity_ratio=0.0,
        compression_ratio=0.0,
    )

    # ─── Phase 1: Load model and tokenizer ─────────────────
    print(f"\n{'=' * 70}")
    print("  Perplexity Evaluation: WikiText-2")
    print(f"{'=' * 70}")
    print(f"\n  Model: {model_id}")
    print(f"  Threshold: {threshold}")
    print(f"  Stride: {stride}")
    print()

    print("[1/5] Loading model from HuggingFace...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    load_time = time.perf_counter() - t0
    model_info = _gather_model_info(model, model_id)
    report.model_info = model_info

    # Resolve max_length
    if max_length <= 0:
        config = getattr(model, "config", None)
        max_length = getattr(config, "max_position_embeddings", 2048)
    report.config["max_length"] = max_length

    print(f"       Loaded in {load_time:.1f}s")
    print(f"       Parameters: {model_info['total_params']:,}")
    print(f"       Context length: {max_length}")
    print()

    # ─── Phase 2: Load and tokenize WikiText-2 ────────────
    print("[2/5] Loading WikiText-2 test set...")
    input_ids = _load_wikitext2(tokenizer)
    seq_len = input_ids.size(1)
    print(f"       Tokens: {seq_len:,}")
    num_windows = math.ceil(seq_len / stride)
    print(f"       Windows: ~{num_windows} (stride={stride})")
    print()

    # ─── Phase 3: FP32 baseline perplexity ─────────────────
    print("[3/5] Evaluating FP32 baseline perplexity...")
    result_fp32 = evaluate_perplexity(
        model, input_ids, stride, max_length, "fp32_baseline"
    )
    report.results.append(result_fp32)
    print(f"       FP32 PPL: {result_fp32.perplexity:.2f}")
    print(f"       Time: {result_fp32.eval_time_s:.1f}s")
    print()

    # ─── Phase 4: Convert to ternary ──────────────────────
    print("[4/5] Converting to ternary...")
    engine = TernaryInferenceEngine(threshold=threshold)
    t0 = time.perf_counter()
    conversion_report = engine.convert(model, sensitivity_analysis=False)
    conversion_time = time.perf_counter() - t0
    report.conversion_time_s = conversion_time

    model.eval()
    sparsity = _calculate_sparsity(model)
    report.sparsity_ratio = sparsity

    mem_profile = profile_model_memory(model)
    report.compression_ratio = mem_profile.compression_ratio

    print(
        f"       Converted {conversion_report.converted_layers}/"
        f"{conversion_report.total_layers} layers in {conversion_time:.1f}s"
    )
    print(f"       Sparsity: {sparsity:.1%}")
    print(f"       Compression: {mem_profile.compression_ratio:.1f}x")
    print()

    # ─── Phase 5: Ternary perplexity ─────────────────────
    print("[5/5] Evaluating ternary perplexity...")
    result_ternary = evaluate_perplexity(
        model, input_ids, stride, max_length, "ternary_pytorch"
    )
    report.results.append(result_ternary)

    if result_fp32.perplexity > 0 and math.isfinite(result_ternary.perplexity):
        ppl_gap = (
            (result_ternary.perplexity - result_fp32.perplexity)
            / result_fp32.perplexity
            * 100
        )
    else:
        ppl_gap = float("inf")

    print(f"       Ternary PPL: {result_ternary.perplexity:.2f}")
    print(f"       Time: {result_ternary.eval_time_s:.1f}s")
    gap_str = f"{ppl_gap:+.1f}%" if math.isfinite(ppl_gap) else "inf"
    print(f"       Gap vs FP32: {gap_str}")
    print()

    # ─── Optional: Accelerated ternary perplexity ─────────
    if not skip_accel and is_accelerated():
        print("[5b] Upgrading to C+SIMD and re-evaluating...")
        from terncore.hf_loader import HFTernaryLoader

        HFTernaryLoader._replace_with_accel(model)
        model.eval()

        result_accel = evaluate_perplexity(
            model, input_ids, stride, max_length, "ternary_accel"
        )
        report.results.append(result_accel)
        print(f"       Accel PPL: {result_accel.perplexity:.2f}")
        print(f"       Time: {result_accel.eval_time_s:.1f}s")
        print()

    return report


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════


def print_summary(report: EvalReport) -> None:
    """Print a summary table of the perplexity results."""
    print(f"\n{'=' * 70}")
    print(f"  Perplexity Summary: {report.model_id}")
    print(f"{'=' * 70}\n")

    mi = report.model_info
    print(f"  Architecture: {mi.get('model_class', 'N/A')}")
    print(f"  Parameters: {mi.get('total_params', 0):,}")
    print(f"  Dataset: WikiText-2 (test)")
    print(
        f"  Stride: {report.config['stride']}, "
        f"Max length: {report.config['max_length']}"
    )
    print(f"  Threshold: {report.config['threshold']}")
    print(f"  Conversion time: {report.conversion_time_s:.1f}s")
    print(f"  Sparsity: {report.sparsity_ratio:.1%}")
    print(f"  Compression: {report.compression_ratio:.1f}x")
    print()

    # Results table
    print(
        f"  {'Phase':<22s}  {'PPL':>10s}  {'Avg NLL':>10s}  "
        f"{'Tokens':>10s}  {'Windows':>8s}  {'Time (s)':>10s}"
    )
    print(f"  {'-' * 76}")

    fp32_ppl = None
    for r in report.results:
        if r.phase == "fp32_baseline":
            fp32_ppl = r.perplexity

        ppl_str = f"{r.perplexity:.2f}" if math.isfinite(r.perplexity) else "inf"

        gap_str = ""
        if fp32_ppl is not None and r.phase != "fp32_baseline":
            if math.isfinite(r.perplexity) and fp32_ppl > 0:
                gap = (r.perplexity - fp32_ppl) / fp32_ppl * 100
                gap_str = f" ({gap:+.1f}%)"
            else:
                gap_str = " (inf)"

        print(
            f"  {r.phase:<22s}  {ppl_str:>10s}{gap_str:>8s}  "
            f"{r.avg_nll:>10.4f}  {r.num_tokens_scored:>10,}  "
            f"{r.num_windows:>8,}  {r.eval_time_s:>10.1f}"
        )

    print()
    print("  Notes:")
    print("  - Sliding-window perplexity per HuggingFace standard method")
    print("  - Deterministic: fixed seed, eval mode (Patent 36)")
    print("  - Lower PPL = better language modelling quality")
    print()


def report_to_dict(report: EvalReport) -> dict:
    """Convert report to a JSON-serialisable dict."""
    return {
        "timestamp": report.timestamp,
        "model_id": report.model_id,
        "platform": report.platform,
        "acceleration": report.acceleration,
        "config": report.config,
        "model_info": report.model_info,
        "conversion_time_s": round(report.conversion_time_s, 3),
        "sparsity_ratio": round(report.sparsity_ratio, 4),
        "compression_ratio": round(report.compression_ratio, 2),
        "results": [
            {
                "phase": r.phase,
                "perplexity": (
                    round(r.perplexity, 4)
                    if math.isfinite(r.perplexity)
                    else None
                ),
                "avg_nll": round(r.avg_nll, 6),
                "num_tokens_scored": r.num_tokens_scored,
                "num_windows": r.num_windows,
                "eval_time_s": round(r.eval_time_s, 3),
            }
            for r in report.results
        ],
    }


def print_json(report: EvalReport) -> None:
    """Print results as formatted JSON."""
    print(json.dumps(report_to_dict(report), indent=2))


# ═══════════════════════════════════════════════════════════════
# RESULTS.md update
# ═══════════════════════════════════════════════════════════════


def update_results_md(report: EvalReport) -> None:
    """
    Append perplexity results to RESULTS.md.

    Inserts a new 'Perplexity Evaluation' section before the
    'Further Optimisation Path' section.
    """
    if not RESULTS_MD_PATH.exists():
        print(f"  Warning: {RESULTS_MD_PATH} not found, skipping update.")
        return

    content = RESULTS_MD_PATH.read_text(encoding="utf-8")

    fp32_result = next(
        (r for r in report.results if r.phase == "fp32_baseline"), None
    )
    ternary_result = next(
        (r for r in report.results if r.phase == "ternary_pytorch"), None
    )
    accel_result = next(
        (r for r in report.results if r.phase == "ternary_accel"), None
    )

    if fp32_result is None or ternary_result is None:
        print("  Warning: incomplete results, skipping RESULTS.md update.")
        return

    if fp32_result.perplexity > 0 and math.isfinite(ternary_result.perplexity):
        gap = (
            (ternary_result.perplexity - fp32_result.perplexity)
            / fp32_result.perplexity
            * 100
        )
        gap_str = f"{gap:+.1f}%"
    else:
        gap_str = "inf"

    tern_ppl = (
        f"{ternary_result.perplexity:.2f}"
        if math.isfinite(ternary_result.perplexity)
        else "inf"
    )
    model_short = report.model_id.split("/")[-1]

    section = f"""## Perplexity Evaluation (WikiText-2)

Automated perplexity evaluation comparing FP32 baseline against ternary
conversion on the WikiText-2 test set using the standard sliding-window
approach (stride={report.config['stride']}, context={report.config['max_length']}).

### Configuration

- **Model**: {report.model_id}
- **Dataset**: WikiText-2 (test split, {fp32_result.num_tokens_scored:,} tokens)
- **Stride**: {report.config['stride']}
- **Context length**: {report.config['max_length']}
- **Quantisation threshold**: {report.config['threshold']}
- **Seed**: {SEED} (deterministic, Patent 36)

### Results

| Model | FP32 PPL | Ternary PPL | Gap | Sparsity | Compression |
|-------|----------|-------------|-----|----------|-------------|
| {model_short} | {fp32_result.perplexity:.2f} | {tern_ppl} | {gap_str} | {report.sparsity_ratio:.1%} | {report.compression_ratio:.1f}x |
"""

    if accel_result is not None:
        accel_ppl = (
            f"{accel_result.perplexity:.2f}"
            if math.isfinite(accel_result.perplexity)
            else "inf"
        )
        section += (
            f"| {model_short} (C+SIMD) | {fp32_result.perplexity:.2f} "
            f"| {accel_ppl} | {gap_str} | {report.sparsity_ratio:.1%} "
            f"| {report.compression_ratio:.1f}x |\n"
        )

    section += f"""
### Evaluation Time

| Phase | Time (s) | Tokens Scored | Windows |
|-------|----------|---------------|---------|
| FP32 baseline | {fp32_result.eval_time_s:.1f} | {fp32_result.num_tokens_scored:,} | {fp32_result.num_windows:,} |
| Ternary PyTorch | {ternary_result.eval_time_s:.1f} | {ternary_result.num_tokens_scored:,} | {ternary_result.num_windows:,} |
"""

    if accel_result is not None:
        section += (
            f"| Ternary C+SIMD | {accel_result.eval_time_s:.1f} "
            f"| {accel_result.num_tokens_scored:,} "
            f"| {accel_result.num_windows:,} |\n"
        )

    section += f"""
### Interpretation

Perplexity measures how well the model predicts the next token in the
WikiText-2 test corpus.  Lower is better.  The gap shows the quality
degradation introduced by ternary quantisation at threshold
{report.config['threshold']}.

"""

    # Insert before "## Further Optimisation Path"
    marker = "## Further Optimisation Path"
    if marker in content:
        content = content.replace(marker, section + marker)
    else:
        content = content.rstrip() + "\n\n" + section

    RESULTS_MD_PATH.write_text(content, encoding="utf-8")
    print(f"  Updated {RESULTS_MD_PATH}")


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate perplexity: FP32 baseline vs ternary on WikiText-2"
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Quantisation threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"Sliding window stride (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Max context length, 0 = auto from model (default: 0)",
    )
    parser.add_argument(
        "--skip-accel",
        action="store_true",
        help="Skip C+SIMD accelerated evaluation",
    )
    parser.add_argument(
        "--no-update-results",
        action="store_true",
        help="Do not update RESULTS.md",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only (no table)",
    )
    args = parser.parse_args()

    report = run_evaluation(
        model_id=args.model,
        threshold=args.threshold,
        stride=args.stride,
        max_length=args.max_length,
        skip_accel=args.skip_accel,
    )

    if args.json_only:
        print_json(report)
    else:
        print_summary(report)

        if not args.no_update_results:
            update_results_md(report)

        print("JSON output:\n")
        print_json(report)


if __name__ == "__main__":
    main()
