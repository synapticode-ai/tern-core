"""
Mixed-precision evaluation: iterative protection search.

Tests multiple protection configurations to find the optimal
mixed-precision ternary config that minimises perplexity gap
while maximising compression.

Five phases:
  1. Recon  -- Smoke test with 512 tokens, protecting only catastrophic layer
  2. Probe  -- 8 configs at 2048 tokens to identify the knee of the curve
  3. Target -- Full WikiText-2 validation of the best config (338K tokens)
  4. Report -- Print summary, update RESULTS.md
  5. Save   -- Write optimal config to JSON

Usage:
    # Full pipeline: recon -> probe -> target -> save
    python benchmarks/eval_mixed_precision.py

    # Recon only (quick smoke test)
    python benchmarks/eval_mixed_precision.py --recon-only

    # Probe only (skip recon, no target validation)
    python benchmarks/eval_mixed_precision.py --skip-recon --probe-only

    # Custom PPL target
    python benchmarks/eval_mixed_precision.py --target-ppl-gap 0.10

Patent 4: Progressive Compression -- iterative config search.
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

# Ensure imports work from repo root
_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))
sys.path.insert(0, str(_BENCH_DIR))

from eval_perplexity import (
    SEED,
    _load_wikitext2,
    _require_dependencies,
    evaluate_perplexity,
)
from terncore.mixed_precision import MixedPrecisionConverter


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_THRESHOLD = 0.7
DEFAULT_STRIDE = 512
DEFAULT_MAX_LENGTH = 0  # 0 = auto from model
DEFAULT_BASELINE_PPL = 7.19  # From Day 1 full-dataset evaluation
DEFAULT_TARGET_PPL_GAP = 0.05  # 5% gap target
RESULTS_MD_PATH = _BENCH_DIR / "RESULTS.md"
CONFIGS_DIR = _BENCH_DIR.parent / "configs"


# ═══════════════════════════════════════════════════════════════
# Sensitivity data from Day 2 analysis (4096 tokens, threshold 0.7)
# ═══════════════════════════════════════════════════════════════

# Ordered by sensitivity (most sensitive first).
# lm_head excluded — auto-protected by MixedPrecisionConverter.
TINYLLAMA_SENSITIVITY = [
    ("model.layers.2.mlp.down_proj", 9609.3),  # catastrophic
    ("model.layers.5.self_attn.q_proj", 2.61),
    ("model.layers.5.self_attn.k_proj", 2.47),
    ("model.layers.4.self_attn.k_proj", 2.32),
    ("model.layers.4.self_attn.q_proj", 2.06),
    ("model.layers.6.self_attn.k_proj", 1.86),
    ("model.layers.8.self_attn.k_proj", 1.57),
    ("model.layers.6.self_attn.q_proj", 1.49),
    ("model.layers.8.self_attn.q_proj", 1.43),
]

# Probe config approach: type-based progressive ternarisation.
#
# Day 2 per-layer sensitivity analysis showed individual layers have
# low sensitivity (87% below 1.1x baseline).  However, protecting by
# sensitivity rank failed catastrophically (compound errors dominate):
# even protecting 46 layers gave PPL 41,405 vs FP32 7.19.
#
# Type-based ternarisation works better.  TinyLlama v_proj layers are
# consistently the least sensitive across all transformer blocks.
# Progressive by-type analysis found:
#   - v_proj only (22 layers):  PPL +40%   (compound error)
#   - v_proj + o_proj (44):     PPL +6600% (catastrophic)
#   - v_proj late (18-21, 4):   PPL +5.0%  (meets target)
#   - v_proj late (19-21, 3):   PPL +2.8%  (safely under)

# Layer type order: least to most sensitive
TERNARY_TYPE_ORDER = ["v_proj", "o_proj", "gate_proj", "up_proj",
                      "down_proj", "q_proj", "k_proj"]

# 8 probe configurations: mix of type-based and sensitivity-based.
# Type-based configs specify ternary layers directly via "ternary_spec".
# Sensitivity-based configs use "top_n" / "pattern_layers".
PROBE_CONFIGS = [
    {
        "name": "v_proj_late3",
        "desc": "v_proj layers 19-21 (3 ternary, conservative)",
        "ternary_spec": {"types": ["v_proj"], "layer_range": [19, 22]},
    },
    {
        "name": "v_proj_late4",
        "desc": "v_proj layers 18-21 (4 ternary, target boundary)",
        "ternary_spec": {"types": ["v_proj"], "layer_range": [18, 22]},
    },
    {
        "name": "v_proj_late6",
        "desc": "v_proj layers 16-21 (6 ternary)",
        "ternary_spec": {"types": ["v_proj"], "layer_range": [16, 22]},
    },
    {
        "name": "v_proj_late11",
        "desc": "v_proj layers 11-21 (11 ternary)",
        "ternary_spec": {"types": ["v_proj"], "layer_range": [11, 22]},
    },
    {
        "name": "v_proj_all",
        "desc": "All v_proj layers (22 ternary)",
        "ternary_spec": {"types": ["v_proj"], "layer_range": [0, 22]},
    },
    {
        "name": "v_proj_o_proj",
        "desc": "All v_proj + o_proj (44 ternary)",
        "ternary_spec": {"types": ["v_proj", "o_proj"], "layer_range": [0, 22]},
    },
    {
        "name": "protect_top9",
        "desc": "Sensitivity-based: protect top-9 (145 ternary)",
        "top_n": 9,
    },
    {
        "name": "all_ternary",
        "desc": "No extra protection (154 ternary, baseline)",
        "top_n": 0,
    },
]


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class ConfigResult:
    """Result from evaluating a single protection configuration."""

    name: str
    desc: str
    protection_list: list[str]
    num_protected: int
    num_ternary: int
    compression_ratio: float
    ppl: float
    ppl_gap_pct: float  # vs FP32 baseline
    eval_time_s: float
    sparsity: float


@dataclass
class MixedPrecisionReport:
    """Full mixed-precision evaluation report."""

    model_id: str
    threshold: float
    baseline_ppl: float
    target_ppl_gap: float
    recon_result: Optional[ConfigResult] = None
    probe_results: list[ConfigResult] = field(default_factory=list)
    best_config: Optional[ConfigResult] = None
    target_result: Optional[ConfigResult] = None
    total_time_s: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def _build_protection_list(
    config: dict, all_linears: Optional[list[str]] = None,
) -> list[str]:
    """
    Build the protection list for a given config.

    Supports two approaches:
    - "ternary_spec": specify which layers TO ternarise (by type + range).
      Protection list is all_linears minus the ternary set.
    - "top_n": protect the top-N sensitive layers (sensitivity-based).
    """
    ternary_spec = config.get("ternary_spec")
    if ternary_spec and all_linears:
        # Type-based: compute ternary set, protect everything else
        types = ternary_spec["types"]
        lo, hi = ternary_spec["layer_range"]
        ternary_set = set()
        for name in all_linears:
            for t in types:
                if name.endswith(t):
                    # Check layer index is in range
                    for i in range(lo, hi):
                        if f".layers.{i}." in name or f".{i}." in name:
                            ternary_set.add(name)
                            break
        return [l for l in all_linears if l not in ternary_set]

    # Sensitivity-based: protect top-N from the ranked list
    top_n = config.get("top_n", 0)
    layers = [name for name, _ in TINYLLAMA_SENSITIVITY[:top_n]]

    # Add pattern-based layers if specified
    pattern = config.get("pattern_layers")
    if pattern:
        lo, hi = pattern["range"]
        types = pattern["types"]
        for i in range(lo, hi):
            for t in types:
                layer_name = f"model.layers.{i}.self_attn.{t}"
                if layer_name not in layers:
                    layers.append(layer_name)

    return layers


def _load_model(model_id: str) -> tuple:
    """Load a fresh model and tokenizer from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    return model, tokenizer


def _resolve_max_length(model: nn.Module, max_length: int) -> int:
    """Resolve max_length from model config if set to 0."""
    if max_length <= 0:
        cfg = getattr(model, "config", None)
        max_length = getattr(cfg, "max_position_embeddings", 2048)
    return max_length


def _calculate_sparsity(model: nn.Module) -> float:
    """Calculate overall sparsity across ternary layers."""
    from terncore.arithmetic.linear import TernaryLinear

    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            if module._cached_ternary is None:
                module._cache_ternary_weights()
            tw = module._cached_ternary
            total += tw.numel()
            zeros += (tw == 0).sum().item()

    return zeros / total if total > 0 else 0.0


def _ppl_gap(ppl: float, baseline: float) -> float:
    """Compute perplexity gap as percentage."""
    if baseline > 0 and math.isfinite(ppl):
        return (ppl - baseline) / baseline * 100
    return float("inf")


def _fmt_ppl(ppl: float) -> str:
    return f"{ppl:.2f}" if math.isfinite(ppl) else "inf"


def _fmt_gap(gap: float) -> str:
    return f"{gap:+.1f}%" if math.isfinite(gap) else "inf"


def _evaluate_config(
    model_id: str,
    config: dict,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
    threshold: float,
    baseline_ppl: float,
) -> ConfigResult:
    """Load fresh model, apply config, evaluate, return result."""
    # Load fresh model
    t0 = time.perf_counter()
    model, _ = _load_model(model_id)
    max_length = _resolve_max_length(model, max_length)

    # Build protection list (needs all_linears for ternary_spec configs)
    all_linears = [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    protection_list = _build_protection_list(config, all_linears)

    # Convert with mixed precision
    converter = MixedPrecisionConverter(
        threshold=threshold,
        protection_list=protection_list,
    )
    report = converter.convert(model)
    model.eval()

    sparsity = _calculate_sparsity(model)

    # Evaluate perplexity
    result = evaluate_perplexity(
        model, input_ids, stride, max_length,
        phase_name=config["name"], quiet=True,
    )

    eval_time = time.perf_counter() - t0
    gap = _ppl_gap(result.perplexity, baseline_ppl)

    return ConfigResult(
        name=config["name"],
        desc=config.get("desc", ""),
        protection_list=protection_list,
        num_protected=report.skipped_layers,
        num_ternary=report.converted_layers,
        compression_ratio=report.compression_ratio,
        ppl=result.perplexity,
        ppl_gap_pct=gap,
        eval_time_s=eval_time,
        sparsity=sparsity,
    )


# ═══════════════════════════════════════════════════════════════
# Main phases
# ═══════════════════════════════════════════════════════════════


def run_recon(
    model_id: str,
    threshold: float,
    baseline_ppl: float,
    stride: int,
    max_length: int,
    eval_tokens: int = 512,
) -> ConfigResult:
    """Phase 1: Recon -- smoke test protecting only catastrophic layer."""
    _require_dependencies()
    torch.manual_seed(SEED)

    print(f"\n{'=' * 70}")
    print("  Phase 1: RECON -- Smoke Test (512 tokens)")
    print(f"{'=' * 70}\n")

    print("  Loading model...")
    model, tokenizer = _load_model(model_id)
    max_length = _resolve_max_length(model, max_length)

    print("  Loading WikiText-2...")
    input_ids = _load_wikitext2(tokenizer)
    input_ids = input_ids[:, :eval_tokens]

    # Recon config: protect only catastrophic layer (sensitivity-based)
    protection_list = _build_protection_list({"top_n": 1})

    print(f"  Protecting: {protection_list}")
    converter = MixedPrecisionConverter(
        threshold=threshold,
        protection_list=protection_list,
    )
    report = converter.convert(model)
    model.eval()

    sparsity = _calculate_sparsity(model)

    print(f"  Converted: {report.converted_layers}/{report.total_layers} layers")
    print(f"  Compression: {report.compression_ratio:.1f}x")
    print(f"  Evaluating at {eval_tokens} tokens...")

    result = evaluate_perplexity(
        model, input_ids, stride, max_length,
        phase_name="recon", quiet=True,
    )

    gap = _ppl_gap(result.perplexity, baseline_ppl)

    print(f"\n  Recon PPL: {_fmt_ppl(result.perplexity)} (gap: {_fmt_gap(gap)})")
    print(f"  Time: {result.eval_time_s:.1f}s")

    return ConfigResult(
        name="recon_top1",
        desc="Recon: protect layers.2.mlp.down_proj only",
        protection_list=protection_list,
        num_protected=report.skipped_layers,
        num_ternary=report.converted_layers,
        compression_ratio=report.compression_ratio,
        ppl=result.perplexity,
        ppl_gap_pct=gap,
        eval_time_s=result.eval_time_s,
        sparsity=sparsity,
    )


def run_probe(
    model_id: str,
    threshold: float,
    baseline_ppl: float,
    stride: int,
    max_length: int,
    eval_tokens: int = 2048,
) -> list[ConfigResult]:
    """Phase 2: Probe -- test 8 configs at 2048 tokens."""
    _require_dependencies()
    torch.manual_seed(SEED)

    num_configs = len(PROBE_CONFIGS)

    print(f"\n{'=' * 70}")
    print(f"  Phase 2: PROBE -- {num_configs} Configs at {eval_tokens} tokens")
    print(f"{'=' * 70}\n")

    # Load tokenizer and dataset once
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    input_ids = _load_wikitext2(tokenizer)
    input_ids = input_ids[:, :eval_tokens]

    results = []
    for idx, config in enumerate(PROBE_CONFIGS):
        print(f"  [{idx + 1}/{num_configs}] {config['name']}: {config['desc']}")

        result = _evaluate_config(
            model_id=model_id,
            config=config,
            input_ids=input_ids,
            stride=stride,
            max_length=max_length,
            threshold=threshold,
            baseline_ppl=baseline_ppl,
        )
        results.append(result)

        print(
            f"         PPL={_fmt_ppl(result.ppl)} gap={_fmt_gap(result.ppl_gap_pct)} "
            f"protected={result.num_protected} "
            f"compression={result.compression_ratio:.1f}x "
            f"({result.eval_time_s:.0f}s)\n"
        )

    return results


def run_target(
    model_id: str,
    threshold: float,
    baseline_ppl: float,
    stride: int,
    max_length: int,
    best_config: dict,
) -> ConfigResult:
    """Phase 3: Target -- full WikiText-2 validation of best config."""
    _require_dependencies()
    torch.manual_seed(SEED)

    print(f"\n{'=' * 70}")
    print("  Phase 3: TARGET -- Full Dataset Validation")
    print(f"{'=' * 70}\n")

    # Load fresh model first (needed for ternary_spec configs)
    print("  Loading model...")
    model, tokenizer = _load_model(model_id)

    all_linears = [
        n for n, m in model.named_modules() if isinstance(m, nn.Linear)
    ]
    protection_list = _build_protection_list(best_config, all_linears)

    print(f"  Config: {best_config['name']}")
    num_ternary = len(all_linears) - len(
        [l for l in all_linears if l in set(protection_list)]
    )
    print(f"  Ternary layers: {num_ternary}")
    print(f"  Protected layers: {len(protection_list)} (explicit + auto)")
    max_length = _resolve_max_length(model, max_length)

    # Load full dataset
    print("  Loading full WikiText-2 test set...")
    input_ids = _load_wikitext2(tokenizer)
    seq_len = input_ids.size(1)
    print(f"  Tokens: {seq_len:,}")

    # Convert
    print("  Converting to mixed-precision ternary...")
    converter = MixedPrecisionConverter(
        threshold=threshold,
        protection_list=protection_list,
    )
    report = converter.convert(model)
    model.eval()

    sparsity = _calculate_sparsity(model)

    print(f"  Converted: {report.converted_layers}/{report.total_layers}")
    print(f"  Compression: {report.compression_ratio:.1f}x")
    print(f"  Evaluating full dataset ({seq_len:,} tokens)...")

    result = evaluate_perplexity(
        model, input_ids, stride, max_length,
        phase_name="target_mixed_precision",
    )

    gap = _ppl_gap(result.perplexity, baseline_ppl)

    print(f"\n  Target PPL: {_fmt_ppl(result.perplexity)}")
    print(f"  Gap vs FP32: {_fmt_gap(gap)}")
    print(f"  Time: {result.eval_time_s:.1f}s")

    return ConfigResult(
        name=best_config["name"],
        desc=best_config.get("desc", ""),
        protection_list=protection_list,
        num_protected=report.skipped_layers,
        num_ternary=report.converted_layers,
        compression_ratio=report.compression_ratio,
        ppl=result.perplexity,
        ppl_gap_pct=gap,
        eval_time_s=result.eval_time_s,
        sparsity=sparsity,
    )


# ═══════════════════════════════════════════════════════════════
# Output and saving
# ═══════════════════════════════════════════════════════════════


def print_probe_summary(
    results: list[ConfigResult], baseline_ppl: float,
) -> None:
    """Print probe results table."""
    print(f"\n{'=' * 90}")
    print("  Probe Results: Mixed-Precision Configurations")
    print(f"{'=' * 90}\n")

    header = (
        f"  {'Config':<30s} | {'PPL':>10s} | {'Gap':>8s} | "
        f"{'Protected':>9s} | {'Ternary':>7s} | {'Compress':>8s}"
    )
    print(header)
    print(f"  {'-' * 84}")

    for r in results:
        print(
            f"  {r.name:<30s} | {_fmt_ppl(r.ppl):>10s} | "
            f"{_fmt_gap(r.ppl_gap_pct):>8s} | "
            f"{r.num_protected:>9d} | {r.num_ternary:>7d} | "
            f"{r.compression_ratio:>7.1f}x"
        )

    print()


def find_best_config(
    results: list[ConfigResult], target_gap: float,
) -> tuple[dict, ConfigResult]:
    """
    Find the config with best compression that meets PPL target.

    Among configs where ppl_gap_pct <= target_gap * 100, pick the one
    with the highest compression ratio.  If none meet the target, pick
    the config with the lowest gap.
    """
    target_pct = target_gap * 100

    # Filter configs meeting the target
    meeting = [
        r for r in results
        if math.isfinite(r.ppl_gap_pct) and r.ppl_gap_pct <= target_pct
    ]

    if meeting:
        best = max(meeting, key=lambda r: r.compression_ratio)
        print(f"  Best config meeting <{target_gap:.0%} gap: {best.name}")
        print(
            f"    PPL gap: {_fmt_gap(best.ppl_gap_pct)}, "
            f"Compression: {best.compression_ratio:.1f}x"
        )
    else:
        best = min(
            results,
            key=lambda r: r.ppl_gap_pct if math.isfinite(r.ppl_gap_pct)
            else float("inf"),
        )
        print(f"  WARNING: No config meets <{target_gap:.0%} gap target.")
        print(f"  Best available: {best.name} (gap: {_fmt_gap(best.ppl_gap_pct)})")

    # Find the matching PROBE_CONFIGS entry
    for cfg in PROBE_CONFIGS:
        if cfg["name"] == best.name:
            return cfg, best

    # Fallback: reconstruct config from result
    return {"name": best.name, "top_n": 0}, best


def save_config(
    result: ConfigResult,
    model_id: str,
    threshold: float,
    baseline_ppl: float,
    output_path: Path,
) -> None:
    """Save the optimal config as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "model_id": model_id,
        "threshold": threshold,
        "baseline_ppl": round(baseline_ppl, 4),
        "mixed_precision": {
            "config_name": result.name,
            "protection_list": result.protection_list,
            "num_protected": result.num_protected,
            "num_ternary": result.num_ternary,
        },
        "results": {
            "ppl": round(result.ppl, 4) if math.isfinite(result.ppl) else None,
            "ppl_gap_pct": (
                round(result.ppl_gap_pct, 2)
                if math.isfinite(result.ppl_gap_pct)
                else None
            ),
            "compression_ratio": round(result.compression_ratio, 2),
            "sparsity": round(result.sparsity, 4),
        },
    }

    output_path.write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Config saved to {output_path}")


def update_results_md(report: MixedPrecisionReport) -> None:
    """Append mixed-precision results to RESULTS.md."""
    if not RESULTS_MD_PATH.exists():
        print(f"  Warning: {RESULTS_MD_PATH} not found.")
        return

    content = RESULTS_MD_PATH.read_text(encoding="utf-8")

    model_short = report.model_id.split("/")[-1]

    # Build probe table rows
    probe_rows = ""
    for r in report.probe_results:
        probe_rows += (
            f"| {r.name} | {r.num_protected} | {r.num_ternary} | "
            f"{_fmt_ppl(r.ppl)} | {_fmt_gap(r.ppl_gap_pct)} | "
            f"{r.compression_ratio:.1f}x |\n"
        )

    # Target result section
    target_section = ""
    if report.target_result:
        r = report.target_result
        target_section = f"""
### Full-Dataset Validation (338,535 tokens)

| Config | Protected | Ternary | PPL | Gap vs FP32 | Compression | Sparsity |
|--------|-----------|---------|-----|-------------|-------------|----------|
| {r.name} | {r.num_protected} | {r.num_ternary} | {_fmt_ppl(r.ppl)} | {_fmt_gap(r.ppl_gap_pct)} | {r.compression_ratio:.1f}x | {r.sparsity:.1%} |

- **FP32 baseline PPL**: {report.baseline_ppl:.2f}
- **Mixed-precision PPL**: {_fmt_ppl(r.ppl)}
- **Gap**: {_fmt_gap(r.ppl_gap_pct)}
- **Target met**: {"YES" if math.isfinite(r.ppl_gap_pct) and r.ppl_gap_pct <= report.target_ppl_gap * 100 else "NO"} (target: <{report.target_ppl_gap:.0%})
"""

    section = f"""## Mixed-Precision Evaluation (Patent 4)

Iterative protection search for **{model_short}** at threshold
{report.threshold}.  Protection list derived from Day 2 per-layer
sensitivity analysis (4,096 tokens, WikiText-2).

### Probe Results (2,048 tokens)

| Config | Protected | Ternary | PPL | Gap vs FP32 | Compression |
|--------|-----------|---------|-----|-------------|-------------|
{probe_rows}{target_section}
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
# Final output
# ═══════════════════════════════════════════════════════════════


def _print_final_summary(report: MixedPrecisionReport) -> None:
    """Print final evaluation summary."""
    print(f"\n{'=' * 70}")
    print("  Mixed-Precision Evaluation: Final Summary")
    print(f"{'=' * 70}\n")

    print(f"  Model: {report.model_id}")
    print(f"  FP32 Baseline PPL: {report.baseline_ppl:.2f}")
    print(f"  Target gap: <{report.target_ppl_gap:.0%}")

    if report.target_result:
        r = report.target_result
        met = (
            math.isfinite(r.ppl_gap_pct)
            and r.ppl_gap_pct <= report.target_ppl_gap * 100
        )

        print(f"\n  Best config: {r.name}")
        print(f"  Protected layers: {r.num_protected}")
        print(f"  Ternary layers: {r.num_ternary}")
        print(f"  PPL: {_fmt_ppl(r.ppl)}")
        print(f"  Gap: {_fmt_gap(r.ppl_gap_pct)}")
        print(f"  Compression: {r.compression_ratio:.1f}x")
        print(f"  Sparsity: {r.sparsity:.1%}")
        print(f"  Target met: {'YES' if met else 'NO'}")

    print(f"\n  Total time: {report.total_time_s:.0f}s")
    print()


def _config_result_to_dict(r: Optional[ConfigResult]) -> Optional[dict]:
    """Convert ConfigResult to JSON-serialisable dict."""
    if r is None:
        return None
    return {
        "name": r.name,
        "desc": r.desc,
        "protection_list": r.protection_list,
        "num_protected": r.num_protected,
        "num_ternary": r.num_ternary,
        "compression_ratio": round(r.compression_ratio, 2),
        "ppl": round(r.ppl, 4) if math.isfinite(r.ppl) else None,
        "ppl_gap_pct": (
            round(r.ppl_gap_pct, 2)
            if math.isfinite(r.ppl_gap_pct)
            else None
        ),
        "eval_time_s": round(r.eval_time_s, 1),
        "sparsity": round(r.sparsity, 4),
    }


def _report_to_dict(report: MixedPrecisionReport) -> dict:
    """Convert full report to JSON-serialisable dict."""
    return {
        "model_id": report.model_id,
        "threshold": report.threshold,
        "baseline_ppl": report.baseline_ppl,
        "target_ppl_gap": report.target_ppl_gap,
        "recon": _config_result_to_dict(report.recon_result),
        "probe": [_config_result_to_dict(r) for r in report.probe_results],
        "best_config": _config_result_to_dict(report.best_config),
        "target": _config_result_to_dict(report.target_result),
        "total_time_s": round(report.total_time_s, 1),
    }


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mixed-precision ternary evaluation: iterative config search"
        )
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
        "--baseline-ppl", type=float, default=DEFAULT_BASELINE_PPL,
        help="FP32 baseline perplexity (default: 7.19)",
    )
    parser.add_argument(
        "--target-ppl-gap", type=float, default=DEFAULT_TARGET_PPL_GAP,
        help="Target PPL gap as fraction (default: 0.05 = 5%%)",
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
        "--recon-only", action="store_true",
        help="Run recon phase only (512 tokens)",
    )
    parser.add_argument(
        "--skip-recon", action="store_true",
        help="Skip recon, go straight to probe",
    )
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Run probe phase only (no target validation)",
    )
    parser.add_argument(
        "--skip-target", action="store_true",
        help="Skip full-dataset target validation",
    )
    parser.add_argument(
        "--no-update-results", action="store_true",
        help="Do not modify RESULTS.md",
    )
    parser.add_argument(
        "--no-save-config", action="store_true",
        help="Do not save config JSON",
    )
    parser.add_argument(
        "--json-only", action="store_true",
        help="Output JSON only (no tables)",
    )
    args = parser.parse_args()

    t_total = time.perf_counter()

    report = MixedPrecisionReport(
        model_id=args.model,
        threshold=args.threshold,
        baseline_ppl=args.baseline_ppl,
        target_ppl_gap=args.target_ppl_gap,
    )

    # ─── Phase 1: Recon ─────────────────────────────────
    if not args.skip_recon:
        recon = run_recon(
            model_id=args.model,
            threshold=args.threshold,
            baseline_ppl=args.baseline_ppl,
            stride=args.stride,
            max_length=args.max_length,
        )
        report.recon_result = recon

        if args.recon_only:
            report.total_time_s = time.perf_counter() - t_total
            if args.json_only:
                print(json.dumps(_report_to_dict(report), indent=2))
            return

    # ─── Phase 2: Probe ─────────────────────────────────
    probe_results = run_probe(
        model_id=args.model,
        threshold=args.threshold,
        baseline_ppl=args.baseline_ppl,
        stride=args.stride,
        max_length=args.max_length,
    )
    report.probe_results = probe_results

    if not args.json_only:
        print_probe_summary(probe_results, args.baseline_ppl)

    # Find best config
    best_cfg, best_result = find_best_config(
        probe_results, args.target_ppl_gap,
    )
    report.best_config = best_result

    if args.probe_only or args.skip_target:
        report.total_time_s = time.perf_counter() - t_total
        if args.json_only:
            print(json.dumps(_report_to_dict(report), indent=2))
        elif not args.no_update_results:
            update_results_md(report)
        return

    # ─── Phase 3: Target ────────────────────────────────
    target = run_target(
        model_id=args.model,
        threshold=args.threshold,
        baseline_ppl=args.baseline_ppl,
        stride=args.stride,
        max_length=args.max_length,
        best_config=best_cfg,
    )
    report.target_result = target

    report.total_time_s = time.perf_counter() - t_total

    # ─── Phase 4: Report ────────────────────────────────
    if args.json_only:
        print(json.dumps(_report_to_dict(report), indent=2))
    else:
        _print_final_summary(report)

        if not args.no_update_results:
            update_results_md(report)

    # ─── Phase 5: Save config ───────────────────────────
    if not args.no_save_config:
        config_path = CONFIGS_DIR / "tinyllama_mixed_precision.json"
        save_config(
            target, args.model, args.threshold,
            args.baseline_ppl, config_path,
        )


if __name__ == "__main__":
    main()
