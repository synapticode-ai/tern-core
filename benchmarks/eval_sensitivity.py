"""
Per-layer sensitivity analysis: quantise one layer at a time, measure
perplexity impact on WikiText-2.

For each Linear layer in the model, replaces its weights with ternary
{-1, 0, +1} * alpha (while keeping all other layers in FP32), measures
perplexity, restores the original weights, and moves to the next layer.
Produces a ranked list of layers by sensitivity.

Usage:
    # Quick validation (512 tokens, ~5 min)
    python benchmarks/eval_sensitivity.py --eval-tokens 512 --baseline-ppl 7.19

    # Standard run (4096 tokens, ~3-4 hours)
    python benchmarks/eval_sensitivity.py --baseline-ppl 7.19

    # Full dataset (overnight)
    python benchmarks/eval_sensitivity.py --full-eval --baseline-ppl 7.19

Patent 4: Progressive Compression — identifies per-layer sensitivity
          for mixed-precision ternary/FP16 deployment.
Patent 36: Deterministic execution guarantee.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

# Ensure tern-core and benchmarks are importable
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))
sys.path.insert(0, str(_BENCH_DIR))

from eval_perplexity import (
    SEED,
    PerplexityResult,
    _load_wikitext2,
    _require_dependencies,
    evaluate_perplexity,
)
from terncore.arithmetic.quantizer import TernaryQuantizer


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_THRESHOLD = 0.7
DEFAULT_STRIDE = 512
DEFAULT_MAX_LENGTH = 0  # 0 = auto
DEFAULT_EVAL_TOKENS = 4096
DEFAULT_MIN_PARAMS = 1000
RESULTS_MD_PATH = _BENCH_DIR / "RESULTS.md"


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class LayerSensitivity:
    """Sensitivity result for a single layer."""

    rank: int
    layer_name: str
    ppl: float
    delta: float  # ppl - baseline_ppl
    ratio: float  # ppl / baseline_ppl
    num_params: int
    sparsity: float  # fraction of zeros after quantisation


@dataclass
class SensitivityReport:
    """Full sensitivity analysis report."""

    model_id: str
    threshold: float
    eval_tokens: int
    baseline_ppl: float
    stride: int
    max_length: int
    num_layers_tested: int
    num_layers_skipped: int
    total_time_s: float
    layers: list[LayerSensitivity]


# ═══════════════════════════════════════════════════════════════
# Core: single-layer quantise → measure → restore
# ═══════════════════════════════════════════════════════════════


def _quantise_weight(weight: torch.Tensor, threshold: float) -> tuple[torch.Tensor, float]:
    """
    Quantise a weight tensor to ternary and return (scaled_ternary, sparsity).

    Uses the same TernaryQuantizer as the inference engine to ensure
    identical quantisation behaviour.
    """
    quantizer = TernaryQuantizer(threshold=threshold)
    ternary, alpha = quantizer.quantize(weight)
    sparsity = (ternary == 0).float().mean().item()
    scaled = ternary * alpha
    return scaled, sparsity


def measure_layer_sensitivity(
    model: nn.Module,
    layer_name: str,
    module: nn.Linear,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
    threshold: float,
    baseline_ppl: float,
) -> LayerSensitivity:
    """
    Quantise a single layer, measure perplexity, restore original weights.

    This is the core of the sensitivity analysis. It:
    1. Clones the original weight (deep copy)
    2. Replaces with ternary-quantised weight
    3. Measures perplexity on the evaluation subset
    4. Restores the original weight exactly
    5. Verifies restoration

    Returns:
        LayerSensitivity with perplexity impact metrics.
    """
    num_params = module.weight.numel()

    # 1. Clone original weight (critical: must be deep copy)
    original_weight = module.weight.data.clone()

    # 2. Quantise this layer only
    scaled_ternary, sparsity = _quantise_weight(module.weight.data, threshold)
    module.weight.data = scaled_ternary

    # 3. Measure perplexity
    result = evaluate_perplexity(
        model, input_ids, stride, max_length,
        phase_name=layer_name, quiet=True,
    )
    ppl = result.perplexity

    # 4. Restore original weight
    module.weight.data = original_weight

    # 5. Verify restoration (critical correctness check)
    assert torch.equal(module.weight.data, original_weight), (
        f"Weight restore FAILED for {layer_name}! "
        "Subsequent measurements would be corrupted."
    )

    # Compute impact metrics
    delta = ppl - baseline_ppl
    ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

    return LayerSensitivity(
        rank=0,  # assigned later after sorting
        layer_name=layer_name,
        ppl=ppl,
        delta=delta,
        ratio=ratio,
        num_params=num_params,
        sparsity=sparsity,
    )


# ═══════════════════════════════════════════════════════════════
# Enumeration and orchestration
# ═══════════════════════════════════════════════════════════════


def enumerate_linear_layers(
    model: nn.Module, min_params: int,
) -> list[tuple[str, nn.Linear]]:
    """
    Find all nn.Linear layers with at least min_params weight parameters.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight.numel() >= min_params:
                layers.append((name, module))
    return layers


def run_sensitivity_analysis(
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_THRESHOLD,
    stride: int = DEFAULT_STRIDE,
    max_length: int = DEFAULT_MAX_LENGTH,
    eval_tokens: int = DEFAULT_EVAL_TOKENS,
    full_eval: bool = False,
    baseline_ppl: Optional[float] = None,
    min_params: int = DEFAULT_MIN_PARAMS,
) -> SensitivityReport:
    """Run the full per-layer sensitivity analysis."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _require_dependencies()
    torch.manual_seed(SEED)

    # ─── Load model ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Layer Sensitivity Analysis")
    print(f"{'=' * 70}")
    print(f"\n  Model: {model_id}")
    print(f"  Threshold: {threshold}")
    print(f"  Eval tokens: {'full dataset' if full_eval else eval_tokens}")
    print()

    print("[1/4] Loading model from HuggingFace...")
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

    # Resolve max_length
    if max_length <= 0:
        config = getattr(model, "config", None)
        max_length = getattr(config, "max_position_embeddings", 2048)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"       Loaded in {load_time:.1f}s")
    print(f"       Parameters: {total_params:,}")
    print()

    # ─── Load dataset ───────────────────────────────────────
    print("[2/4] Loading WikiText-2 test set...")
    input_ids = _load_wikitext2(tokenizer)
    full_seq_len = input_ids.size(1)

    if not full_eval:
        eval_len = min(eval_tokens, full_seq_len)
        input_ids = input_ids[:, :eval_len]
    else:
        eval_len = full_seq_len

    print(f"       Full dataset: {full_seq_len:,} tokens")
    print(f"       Evaluating on: {eval_len:,} tokens")
    num_windows = math.ceil(eval_len / stride)
    print(f"       Windows: ~{num_windows} (stride={stride})")
    print()

    # ─── Baseline perplexity ────────────────────────────────
    if baseline_ppl is not None:
        print(f"[3/4] Using provided baseline PPL: {baseline_ppl:.2f}")
    else:
        print("[3/4] Measuring FP32 baseline perplexity...")
        result_fp32 = evaluate_perplexity(
            model, input_ids, stride, max_length, "fp32_baseline"
        )
        baseline_ppl = result_fp32.perplexity
        print(f"       Baseline PPL: {baseline_ppl:.2f}")
        print(f"       Time: {result_fp32.eval_time_s:.1f}s")
    print()

    # ─── Enumerate layers ───────────────────────────────────
    layers_to_test = enumerate_linear_layers(model, min_params)
    all_linear = list(
        (n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)
    )
    skipped = len(all_linear) - len(layers_to_test)

    print(f"[4/4] Testing {len(layers_to_test)} layers "
          f"(skipped {skipped} with <{min_params} params)...")
    print()

    # ─── Per-layer sensitivity measurement ──────────────────
    results: list[LayerSensitivity] = []
    t_total = time.perf_counter()

    for idx, (name, module) in enumerate(layers_to_test):
        t_layer = time.perf_counter()
        sens = measure_layer_sensitivity(
            model, name, module, input_ids,
            stride, max_length, threshold, baseline_ppl,
        )
        layer_time = time.perf_counter() - t_layer
        results.append(sens)

        # Progress
        ppl_str = f"{sens.ppl:.2f}" if math.isfinite(sens.ppl) else "inf"
        print(
            f"  [{idx + 1:3d}/{len(layers_to_test)}] "
            f"{name:<50s} PPL={ppl_str:>12s}  "
            f"delta={sens.delta:>+10.2f}  "
            f"ratio={sens.ratio:>6.2f}x  "
            f"({layer_time:.1f}s)",
            flush=True,
        )

    total_time = time.perf_counter() - t_total

    # ─── Sort by sensitivity (most impactful first) ─────────
    results.sort(key=lambda s: s.delta, reverse=True)
    for rank, r in enumerate(results, 1):
        r.rank = rank

    print(f"\n  Total analysis time: {total_time:.1f}s")
    print()

    return SensitivityReport(
        model_id=model_id,
        threshold=threshold,
        eval_tokens=eval_len,
        baseline_ppl=baseline_ppl,
        stride=stride,
        max_length=max_length,
        num_layers_tested=len(results),
        num_layers_skipped=skipped,
        total_time_s=total_time,
        layers=results,
    )


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════


def _format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_summary(report: SensitivityReport) -> None:
    """Print formatted sensitivity ranking."""
    print(f"{'=' * 100}")
    print(
        f"  Layer Sensitivity Analysis: "
        f"{report.model_id.split('/')[-1]}"
    )
    print(
        f"  Threshold: {report.threshold} | "
        f"Eval tokens: {report.eval_tokens:,} | "
        f"Baseline PPL: {report.baseline_ppl:.2f}"
    )
    print(f"{'=' * 100}")
    print()

    # Full table
    header = (
        f"{'Rank':>4s} | {'Layer Name':<50s} | {'PPL':>10s} | "
        f"{'Delta':>10s} | {'Ratio':>7s} | {'Params':>8s} | {'Sparsity':>8s}"
    )
    print(header)
    print("-" * len(header))

    for r in report.layers:
        ppl_str = f"{r.ppl:.2f}" if math.isfinite(r.ppl) else "inf"
        delta_str = f"{r.delta:+.2f}" if math.isfinite(r.delta) else "+inf"
        ratio_str = f"{r.ratio:.1f}x" if math.isfinite(r.ratio) else "inf"
        print(
            f"{r.rank:4d} | {r.layer_name:<50s} | {ppl_str:>10s} | "
            f"{delta_str:>10s} | {ratio_str:>7s} | "
            f"{_format_params(r.num_params):>8s} | {r.sparsity:>7.1%}"
        )

    # Top 5 / Bottom 5
    print()
    print("TOP 5 MOST SENSITIVE (keep in FP16):")
    for r in report.layers[:5]:
        ratio_str = f"{r.ratio:.1f}x" if math.isfinite(r.ratio) else "inf"
        print(f"  {r.rank}. {r.layer_name} ({ratio_str} baseline)")

    print()
    print("BOTTOM 5 LEAST SENSITIVE (safe to ternarise):")
    for r in report.layers[-5:]:
        ratio_str = f"{r.ratio:.3f}x" if math.isfinite(r.ratio) else "inf"
        print(f"  {r.rank}. {r.layer_name} ({ratio_str} baseline)")

    # Summary statistics
    print()
    above_2x = sum(1 for r in report.layers if r.ratio >= 2.0)
    above_1_5x = sum(1 for r in report.layers if r.ratio >= 1.5)
    above_1_1x = sum(1 for r in report.layers if r.ratio >= 1.1)
    below_1_1x = sum(1 for r in report.layers if r.ratio < 1.1)
    total = len(report.layers)

    print(f"Layers above 2.0x baseline: {above_2x} ({above_2x / total:.1%} of layers)")
    print(f"Layers above 1.5x baseline: {above_1_5x} ({above_1_5x / total:.1%} of layers)")
    print(f"Layers above 1.1x baseline: {above_1_1x} ({above_1_1x / total:.1%} of layers)")
    print(f"Layers below 1.1x baseline: {below_1_1x} ({below_1_1x / total:.1%} of layers)")

    # Mixed-precision estimate
    protect_2x = [r for r in report.layers if r.ratio >= 2.0]
    ternary_2x = [r for r in report.layers if r.ratio < 2.0]
    if total > 0:
        print()
        print(f"Estimated mixed-precision compression (protecting 2.0x+ layers):")
        print(f"  Ternary layers: {len(ternary_2x)}/{total} ({len(ternary_2x) / total:.1%})")
        print(f"  Protected layers: {len(protect_2x)}/{total} ({len(protect_2x) / total:.1%})")

        # Estimate compression: protected layers stay FP32 (32 bits/param),
        # ternary layers get 3 bits/param (2-bit packed + 1-bit bitmap)
        ternary_params = sum(r.num_params for r in ternary_2x)
        protected_params = sum(r.num_params for r in protect_2x)
        total_params = ternary_params + protected_params
        if total_params > 0:
            fp32_bits = total_params * 32
            mixed_bits = ternary_params * 3 + protected_params * 32
            compression = fp32_bits / mixed_bits if mixed_bits > 0 else 0
            print(f"  Estimated weight compression: ~{compression:.1f}x")

    print()


def print_json(report: SensitivityReport) -> None:
    """Print results as JSON."""
    data = {
        "model_id": report.model_id,
        "threshold": report.threshold,
        "eval_tokens": report.eval_tokens,
        "baseline_ppl": round(report.baseline_ppl, 4),
        "stride": report.stride,
        "max_length": report.max_length,
        "num_layers_tested": report.num_layers_tested,
        "num_layers_skipped": report.num_layers_skipped,
        "total_time_s": round(report.total_time_s, 1),
        "layers": [
            {
                "rank": r.rank,
                "layer_name": r.layer_name,
                "ppl": round(r.ppl, 4) if math.isfinite(r.ppl) else None,
                "delta": round(r.delta, 4) if math.isfinite(r.delta) else None,
                "ratio": round(r.ratio, 4) if math.isfinite(r.ratio) else None,
                "num_params": r.num_params,
                "sparsity": round(r.sparsity, 4),
            }
            for r in report.layers
        ],
    }
    print(json.dumps(data, indent=2))


def write_csv(report: SensitivityReport, path: str) -> None:
    """Write results to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "layer_name", "ppl", "delta", "ratio",
            "num_params", "sparsity",
        ])
        for r in report.layers:
            writer.writerow([
                r.rank, r.layer_name,
                f"{r.ppl:.4f}" if math.isfinite(r.ppl) else "inf",
                f"{r.delta:.4f}" if math.isfinite(r.delta) else "inf",
                f"{r.ratio:.4f}" if math.isfinite(r.ratio) else "inf",
                r.num_params,
                f"{r.sparsity:.4f}",
            ])
    print(f"  CSV written to {path}")


# ═══════════════════════════════════════════════════════════════
# RESULTS.md update
# ═══════════════════════════════════════════════════════════════


def update_results_md(report: SensitivityReport) -> None:
    """Append layer sensitivity results to RESULTS.md."""
    if not RESULTS_MD_PATH.exists():
        print(f"  Warning: {RESULTS_MD_PATH} not found, skipping update.")
        return

    content = RESULTS_MD_PATH.read_text(encoding="utf-8")

    # Build section
    model_short = report.model_id.split("/")[-1]

    section = f"""## Layer Sensitivity Analysis

Per-layer sensitivity ranking for **{model_short}** at threshold
{report.threshold}.  Each layer is quantised individually to ternary while
all other layers remain in FP32.  Perplexity is measured on the first
{report.eval_tokens:,} tokens of WikiText-2 (stride={report.stride},
context={report.max_length}).

### Top 10 Most Sensitive Layers (keep in FP16)

| Rank | Layer | PPL | Delta | Ratio | Params | Sparsity |
|------|-------|-----|-------|-------|--------|----------|
"""

    for r in report.layers[:10]:
        ppl_str = f"{r.ppl:.2f}" if math.isfinite(r.ppl) else "inf"
        delta_str = f"{r.delta:+.2f}" if math.isfinite(r.delta) else "+inf"
        ratio_str = f"{r.ratio:.1f}x" if math.isfinite(r.ratio) else "inf"
        section += (
            f"| {r.rank} | {r.layer_name} | {ppl_str} | "
            f"{delta_str} | {ratio_str} | {_format_params(r.num_params)} | "
            f"{r.sparsity:.1%} |\n"
        )

    section += """
### Bottom 10 Least Sensitive Layers (safe to ternarise)

| Rank | Layer | PPL | Delta | Ratio | Params | Sparsity |
|------|-------|-----|-------|-------|--------|----------|
"""

    for r in report.layers[-10:]:
        ppl_str = f"{r.ppl:.2f}" if math.isfinite(r.ppl) else "inf"
        delta_str = f"{r.delta:+.2f}" if math.isfinite(r.delta) else "+inf"
        ratio_str = f"{r.ratio:.3f}x" if math.isfinite(r.ratio) else "inf"
        section += (
            f"| {r.rank} | {r.layer_name} | {ppl_str} | "
            f"{delta_str} | {ratio_str} | {_format_params(r.num_params)} | "
            f"{r.sparsity:.1%} |\n"
        )

    # Summary stats
    total = len(report.layers)
    above_2x = sum(1 for r in report.layers if r.ratio >= 2.0)
    above_1_5x = sum(1 for r in report.layers if r.ratio >= 1.5)
    below_1_1x = sum(1 for r in report.layers if r.ratio < 1.1)

    section += f"""
### Summary Statistics

- **Layers tested**: {report.num_layers_tested} (skipped {report.num_layers_skipped} with <{DEFAULT_MIN_PARAMS} params)
- **Baseline PPL**: {report.baseline_ppl:.2f} (FP32)
- **Layers above 2.0x baseline**: {above_2x} ({above_2x / total:.1%})
- **Layers above 1.5x baseline**: {above_1_5x} ({above_1_5x / total:.1%})
- **Layers below 1.1x baseline**: {below_1_1x} ({below_1_1x / total:.1%})
- **Evaluation**: {report.eval_tokens:,} tokens, {report.total_time_s:.0f}s total

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
        description="Per-layer sensitivity analysis: quantise one layer at a time"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Quantisation threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--baseline-ppl", type=float, default=None,
        help="Baseline FP32 perplexity (skip re-measurement if provided)",
    )
    parser.add_argument(
        "--eval-tokens", type=int, default=DEFAULT_EVAL_TOKENS,
        help=f"Number of tokens for quick evaluation (default: {DEFAULT_EVAL_TOKENS})",
    )
    parser.add_argument(
        "--full-eval", action="store_true",
        help="Use complete WikiText-2 test set (overrides --eval-tokens)",
    )
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE,
        help=f"Sliding window stride (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--max-length", type=int, default=DEFAULT_MAX_LENGTH,
        help="Max context length, 0 = auto (default: 0)",
    )
    parser.add_argument(
        "--min-params", type=int, default=DEFAULT_MIN_PARAMS,
        help=f"Skip layers below this param count (default: {DEFAULT_MIN_PARAMS})",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output JSON only",
    )
    parser.add_argument(
        "--no-update-results", action="store_true",
        help="Do not modify RESULTS.md",
    )
    parser.add_argument(
        "--output-csv", type=str, default=None,
        help="Save full results as CSV to this path",
    )
    args = parser.parse_args()

    report = run_sensitivity_analysis(
        model_id=args.model,
        threshold=args.threshold,
        stride=args.stride,
        max_length=args.max_length,
        eval_tokens=args.eval_tokens,
        full_eval=args.full_eval,
        baseline_ppl=args.baseline_ppl,
        min_params=args.min_params,
    )

    if args.json:
        print_json(report)
    else:
        print_summary(report)

        if not args.no_update_results:
            update_results_md(report)

        if args.output_csv:
            write_csv(report, args.output_csv)

        print("\nJSON output:\n")
        print_json(report)


if __name__ == "__main__":
    main()
