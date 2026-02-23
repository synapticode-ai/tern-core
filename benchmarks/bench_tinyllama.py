"""
Benchmark: TinyLlama-1.1B ternary conversion and text generation.

Downloads TinyLlama-1.1B from HuggingFace, converts to ternary via
TernaryInferenceEngine, and benchmarks text generation latency and
memory compression against the FP32 baseline.

Usage:
    python benchmarks/bench_tinyllama.py
    python benchmarks/bench_tinyllama.py --max-tokens 100
    python benchmarks/bench_tinyllama.py --skip-accel --json-only
    python benchmarks/bench_tinyllama.py --model-id "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

Patent 12: Auto binary-to-ternary conversion.
Patent 36: Deterministic execution guarantee.
Patent 40: Bandwidth optimisation — streaming ternary weight loader.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

# Ensure tern-core is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.accel import get_acceleration_info, is_accelerated
from terncore.hf_loader import (
    ConversionResult,
    GenerationResult,
    HFTernaryLoader,
)
from terncore.memory import profile_model_memory


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROMPT = "What is ternary computing? Explain in simple terms"
DEFAULT_MAX_TOKENS = 50
DEFAULT_THRESHOLD = 0.7
SEED = 0


# ═══════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════


@dataclass
class PhaseResult:
    """Result from one benchmark phase (baseline, ternary, or accel)."""

    phase: str
    generation: Optional[GenerationResult]
    memory_mb: float
    compression_ratio: float
    num_ternary_layers: int
    num_total_layers: int


@dataclass
class BenchReport:
    """Full benchmark report."""

    timestamp: str
    model_id: str
    platform: dict
    acceleration: dict
    config: dict
    model_info: dict
    conversion_time_s: float
    phases: list[PhaseResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════


def _model_memory_mb(model: torch.nn.Module) -> float:
    """Estimate model memory in MB from parameter + buffer storage."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


# ═══════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════


def run_benchmark(
    model_id: str = DEFAULT_MODEL_ID,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    threshold: float = DEFAULT_THRESHOLD,
    sensitivity: bool = False,
    skip_accel: bool = False,
    skip_generate: bool = False,
) -> BenchReport:
    """Run the full TinyLlama benchmark."""
    import platform as plat

    from terncore.hf_loader import require_transformers

    require_transformers()

    accel_info = get_acceleration_info()

    report = BenchReport(
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
            "prompt": prompt,
            "max_tokens": max_tokens,
            "threshold": threshold,
            "sensitivity_analysis": sensitivity,
            "seed": SEED,
            "skip_accel": skip_accel,
            "skip_generate": skip_generate,
        },
        model_info={},
        conversion_time_s=0.0,
    )

    # ─── Phase 1: Load model (FP32 baseline) ───────────────
    print(f"\n{'='*70}")
    print(f"  TinyLlama Ternary Benchmark")
    print(f"{'='*70}")
    print(f"\n  Model: {model_id}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Threshold: {threshold}")
    print()

    print("[1/4] Loading model from HuggingFace...")
    t0 = time.perf_counter()

    loader = HFTernaryLoader(
        threshold=threshold,
        sensitivity_analysis=sensitivity,
        use_accel=False,  # upgrade later if requested
    )

    # Load but don't convert yet — we need baseline first
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format prompt with chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            chat_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            print(f"  Chat template applied ({len(chat_prompt)} chars)")
            prompt = chat_prompt
        except Exception:
            pass  # Fall back to raw prompt

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    load_time = time.perf_counter() - t0
    baseline_mem = _model_memory_mb(model)

    # Gather model info
    model_info_obj = loader._gather_model_info(model, model_id)
    report.model_info = asdict(model_info_obj)

    print(f"       Loaded in {load_time:.1f}s")
    print(f"       Parameters: {model_info_obj.total_params:,}")
    print(f"       Eligible Linear layers: {model_info_obj.eligible_linear_layers}")
    print(f"       Protected layers: {len(model_info_obj.protected_layers)}")
    print(f"       Memory: {baseline_mem:.1f} MB")

    # ─── Phase 2: FP32 baseline generation ─────────────────
    print("\n[2/4] FP32 baseline generation...")

    gen_baseline = None
    if not skip_generate:
        gen_baseline = HFTernaryLoader.generate_text(
            model, tokenizer, prompt, max_tokens, seed=SEED
        )
        print(f"       Prefill: {gen_baseline.prefill_ms:.1f} ms")
        print(f"       Total: {gen_baseline.total_ms:.1f} ms")
        print(f"       Tokens: {gen_baseline.num_tokens_generated}")
        print(f"       Per token: {gen_baseline.per_token_ms:.1f} ms")
        print(f"       Text: {gen_baseline.generated_text[:120]}...")
    else:
        print("       (skipped)")

    report.phases.append(
        PhaseResult(
            phase="fp32_baseline",
            generation=gen_baseline,
            memory_mb=baseline_mem,
            compression_ratio=1.0,
            num_ternary_layers=0,
            num_total_layers=model_info_obj.num_linear_layers,
        )
    )

    # ─── Phase 3: Ternary conversion + generation ──────────
    print("\n[3/4] Converting to ternary...")

    t0 = time.perf_counter()
    conversion_report = loader.engine.convert(
        model, sensitivity_analysis=sensitivity
    )
    conversion_time = time.perf_counter() - t0
    report.conversion_time_s = conversion_time

    print(f"       Converted {conversion_report.converted_layers}/{conversion_report.total_layers} layers in {conversion_time:.1f}s")
    print(f"       Compression: {conversion_report.compression_ratio:.1f}x")

    # Measure memory profile (theoretical packed size)
    mem_profile = profile_model_memory(model)
    ternary_runtime_mem = _model_memory_mb(model)
    ternary_packed_mb = mem_profile.packed_bytes / (1024 * 1024) if hasattr(mem_profile, 'packed_bytes') else 0
    print(f"       Runtime memory: {ternary_runtime_mem:.1f} MB (FP32 cached ternary)")
    print(f"       Theoretical packed: {mem_profile.compression_ratio:.1f}x compression")

    gen_ternary = None
    if not skip_generate:
        gen_ternary = HFTernaryLoader.generate_text(
            model, tokenizer, prompt, max_tokens, seed=SEED
        )
        print(f"       Prefill: {gen_ternary.prefill_ms:.1f} ms")
        print(f"       Total: {gen_ternary.total_ms:.1f} ms")
        print(f"       Tokens: {gen_ternary.num_tokens_generated}")
        print(f"       Per token: {gen_ternary.per_token_ms:.1f} ms")
        print(f"       Text: {gen_ternary.generated_text[:120]}...")

    report.phases.append(
        PhaseResult(
            phase="ternary_pytorch",
            generation=gen_ternary,
            memory_mb=ternary_runtime_mem,
            compression_ratio=mem_profile.compression_ratio if hasattr(mem_profile, 'compression_ratio') else 1.0,
            num_ternary_layers=conversion_report.converted_layers,
            num_total_layers=conversion_report.total_layers,
        )
    )

    # ─── Phase 4: Accelerated ternary (optional) ───────────
    if not skip_accel and is_accelerated():
        print("\n[4/4] Upgrading to C+SIMD acceleration...")

        HFTernaryLoader._replace_with_accel(model)
        accel_mem = _model_memory_mb(model)
        print(f"       Memory: {accel_mem:.1f} MB")

        gen_accel = None
        if not skip_generate:
            gen_accel = HFTernaryLoader.generate_text(
                model, tokenizer, prompt, max_tokens, seed=SEED
            )
            print(f"       Prefill: {gen_accel.prefill_ms:.1f} ms")
            print(f"       Total: {gen_accel.total_ms:.1f} ms")
            print(f"       Tokens: {gen_accel.num_tokens_generated}")
            print(f"       Per token: {gen_accel.per_token_ms:.1f} ms")
            print(f"       Text: {gen_accel.generated_text[:120]}...")

        report.phases.append(
            PhaseResult(
                phase="ternary_accel",
                generation=gen_accel,
                memory_mb=accel_mem,
                compression_ratio=mem_profile.compression_ratio if hasattr(mem_profile, 'compression_ratio') else 1.0,
                num_ternary_layers=conversion_report.converted_layers,
                num_total_layers=conversion_report.total_layers,
            )
        )
    else:
        reason = "skipped by user" if skip_accel else "C library not available"
        print(f"\n[4/4] C+SIMD acceleration ({reason})")

    # Free original FP32 weights now that all generation phases are done
    HFTernaryLoader._free_original_weights(model)
    gc.collect()
    final_mem = _model_memory_mb(model)
    print(f"\n  Final memory after freeing FP32 weights: {final_mem:.1f} MB")

    return report


# ═══════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════


def print_summary(report: BenchReport) -> None:
    """Print a summary table of the benchmark results."""
    print(f"\n{'='*70}")
    print(f"  Summary: {report.model_id}")
    print(f"{'='*70}\n")

    mi = report.model_info
    print(f"  Architecture: {mi.get('model_class', 'N/A')}")
    print(f"  Parameters: {mi.get('total_params', 0):,}")
    print(f"  Linear layers: {mi.get('num_linear_layers', 0)} "
          f"({mi.get('eligible_linear_layers', 0)} eligible, "
          f"{len(mi.get('protected_layers', []))} protected)")
    print(f"  Conversion time: {report.conversion_time_s:.1f}s")
    print()

    # Latency table
    header = f"  {'Phase':<22s} {'Memory (MB)':>12s} {'Compress':>9s}"
    if any(p.generation for p in report.phases):
        header += f"  {'Prefill (ms)':>13s} {'Total (ms)':>11s} {'Tok/s':>7s} {'Tokens':>7s}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for phase in report.phases:
        line = f"  {phase.phase:<22s} {phase.memory_mb:>11.1f}  {phase.compression_ratio:>8.1f}x"

        if phase.generation:
            g = phase.generation
            tok_per_s = g.num_tokens_generated / (g.total_ms / 1000) if g.total_ms > 0 else 0
            line += f"  {g.prefill_ms:>12.1f}  {g.total_ms:>10.1f}  {tok_per_s:>6.1f}  {g.num_tokens_generated:>6d}"
        elif any(p.generation for p in report.phases):
            line += f"  {'N/A':>12s}  {'N/A':>10s}  {'N/A':>6s}  {'N/A':>6s}"

        print(line)

    print()

    # Text comparison
    gen_phases = [(p.phase, p.generation) for p in report.phases if p.generation]
    if gen_phases:
        print("  Generated Text Comparison:")
        print(f"  {'-'*60}")
        for phase_name, gen in gen_phases:
            text = gen.generated_text
            # Show first 200 chars
            display = text[:200] + ("..." if len(text) > 200 else "")
            print(f"\n  [{phase_name}]:")
            print(f"  {display}")
        print()

    # Determinism note
    print("  Notes:")
    print("  - Greedy decoding (do_sample=False) for determinism (Patent 36)")
    print("  - FP32 weights used for accurate quantisation thresholds")
    print("  - Compression ratio = theoretical packed size (2-bit + bitmap)")
    print("  - Runtime memory is higher (FP32 cached ternary for PyTorch dispatch)")
    print("  - First ternary forward pass includes one-time weight caching (~12s)")
    if report.conversion_time_s > 0:
        print(f"  - Conversion: {report.conversion_time_s:.1f}s (one-time cost)")
    print()


def report_to_dict(report: BenchReport) -> dict:
    """Convert report to a JSON-serialisable dict."""

    def _gen_dict(g: Optional[GenerationResult]) -> Optional[dict]:
        if g is None:
            return None
        return asdict(g)

    return {
        "timestamp": report.timestamp,
        "model_id": report.model_id,
        "platform": report.platform,
        "acceleration": report.acceleration,
        "config": report.config,
        "model_info": report.model_info,
        "conversion_time_s": round(report.conversion_time_s, 3),
        "phases": [
            {
                "phase": p.phase,
                "memory_mb": round(p.memory_mb, 2),
                "compression_ratio": round(p.compression_ratio, 2),
                "num_ternary_layers": p.num_ternary_layers,
                "num_total_layers": p.num_total_layers,
                "generation": _gen_dict(p.generation),
            }
            for p in report.phases
        ],
    }


def print_json(report: BenchReport) -> None:
    """Print results as formatted JSON."""
    print(json.dumps(report_to_dict(report), indent=2))


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TinyLlama-1.1B ternary conversion and generation"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Generation prompt (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Quantisation threshold (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run per-layer sensitivity analysis (slower)",
    )
    parser.add_argument(
        "--skip-accel",
        action="store_true",
        help="Skip C+SIMD acceleration phase",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip text generation (conversion + memory only)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only (no table)",
    )
    args = parser.parse_args()

    report = run_benchmark(
        model_id=args.model_id,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        threshold=args.threshold,
        sensitivity=args.sensitivity,
        skip_accel=args.skip_accel,
        skip_generate=args.skip_generate,
    )

    if args.json_only:
        print_json(report)
    else:
        print_summary(report)
        print("JSON output:\n")
        print_json(report)


if __name__ == "__main__":
    main()
