"""
Day 12: Performance Scaling Curve — tok/s Across Models and Sequence Lengths

Measures tok/s and memory across 4 causal models × 4 sequence lengths × 3 modes
(FP32, Ternary, Packed). Produces the headline performance numbers for the
evidence package.

Models:
  - DistilGPT-2 (82M)    — max_pos 1024
  - GPT-2 (124M)          — max_pos 1024
  - GPT-2-medium (355M)   — max_pos 1024
  - TinyLlama-1.1B        — max_pos 2048

BERT-base is encoder-only: measures forward pass latency, not tok/s.

Patent 36: Deterministic execution (do_sample=False, fixed seeds).
Patent 12: Auto binary-to-ternary conversion pipeline.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.

Run with:
    python benchmarks/bench_day12_performance.py          # full run
    python benchmarks/bench_day12_performance.py --recon   # DistilGPT-2 calibration only
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import resource
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class TimeoutError(Exception):
    """Raised when a measurement exceeds the hard timeout."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Hard timeout reached")

import torch
import torch.nn as nn

_BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BENCH_DIR.parent / "src"))

from terncore.engine.inference import TernaryInferenceEngine
from terncore.packed_linear import convert_model_to_packed

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SEED = 42
NUM_GEN_TOKENS = 32
WARMUP_RUNS = 1
MEASURED_RUNS = 3
RSS_LIMIT_MB = 14_000  # 14 GB guard
TIMEOUT_PER_MEASUREMENT_S = 120  # skip config if one measurement exceeds this

MODELS = [
    {
        "id": "distilgpt2",
        "type": "causal",
        "desc": "DistilGPT-2 (82M)",
        "max_pos": 1024,
    },
    {
        "id": "gpt2",
        "type": "causal",
        "desc": "GPT-2 (124M)",
        "max_pos": 1024,
    },
    {
        "id": "gpt2-medium",
        "type": "causal",
        "desc": "GPT-2-medium (355M)",
        "max_pos": 1024,
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "causal",
        "desc": "TinyLlama-1.1B",
        "max_pos": 2048,
    },
]

BERT_MODEL = {
    "id": "bert-base-uncased",
    "type": "encoder",
    "desc": "BERT-base (110M)",
    "max_pos": 512,
}

SEQ_LENS = [128, 512, 1024, 2048]

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


def banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def get_rss_mb() -> float:
    """Get current peak RSS in MB (macOS returns bytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS: ru_maxrss is in bytes; Linux: in KB
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    else:
        return usage.ru_maxrss / 1024


def model_memory_mb(model: nn.Module) -> float:
    """Estimate model memory from parameters + buffers."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def count_params_m(model: nn.Module) -> float:
    """Count total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def load_model(model_id: str, model_type: str):
    """Load model and tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "causal":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True,
        )
    else:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True,
        )

    model.eval()
    return model, tokenizer


def get_wikitext_text() -> str:
    """Load WikiText-2 test text (cached from prior days)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


def prepare_input_ids(tokenizer, text: str, seq_len: int) -> torch.Tensor:
    """Tokenize and truncate to seq_len."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=seq_len + 1024)
    input_ids = tokens["input_ids"][:, :seq_len]
    # Pad if text is shorter than seq_len (unlikely for WikiText-2)
    if input_ids.shape[1] < seq_len:
        pad_len = seq_len - input_ids.shape[1]
        pad_id = tokenizer.pad_token_id or 0
        padding = torch.full((1, pad_len), pad_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


# ═══════════════════════════════════════════════════════════════
# Measurement functions
# ═══════════════════════════════════════════════════════════════


def measure_generation(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_tokens: int = NUM_GEN_TOKENS,
    warmup: int = WARMUP_RUNS,
    runs: int = MEASURED_RUNS,
) -> Optional[dict]:
    """
    Measure generation speed (causal models).

    Returns dict with tok_s, total_s, prefill_ms, per_token_ms, or None on timeout.
    """
    torch.manual_seed(SEED)

    # Build attention_mask and pad_token_id to suppress warnings
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = getattr(model.config, "eos_token_id", 0)

    # Warmup with hard signal-based timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    for _ in range(warmup):
        t0 = time.perf_counter()
        try:
            signal.alarm(TIMEOUT_PER_MEASUREMENT_S)
            with torch.no_grad():
                model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=num_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            signal.alarm(0)
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            warmup_time = time.perf_counter() - t0
            print(f"    TIMEOUT: warmup interrupted at {warmup_time:.0f}s (limit {TIMEOUT_PER_MEASUREMENT_S}s)")
            return None
        warmup_time = time.perf_counter() - t0
        if warmup_time > TIMEOUT_PER_MEASUREMENT_S:
            signal.signal(signal.SIGALRM, old_handler)
            print(f"    TIMEOUT: warmup took {warmup_time:.1f}s > {TIMEOUT_PER_MEASUREMENT_S}s")
            return None
    signal.signal(signal.SIGALRM, old_handler)

    # Prefill timing (single forward pass on prompt)
    prefill_times = []
    for _ in range(runs):
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        prefill_times.append(time.perf_counter() - t0)

    # Generation timing
    gen_times = []
    for _ in range(runs):
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=num_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        elapsed = time.perf_counter() - t0
        gen_times.append(elapsed)
        if elapsed > TIMEOUT_PER_MEASUREMENT_S:
            print(f"    TIMEOUT: run took {elapsed:.1f}s > {TIMEOUT_PER_MEASUREMENT_S}s")
            return None

    # Median
    gen_times.sort()
    prefill_times.sort()
    median_gen = gen_times[len(gen_times) // 2]
    median_prefill = prefill_times[len(prefill_times) // 2]

    tok_s = num_tokens / median_gen
    per_token_ms = (median_gen * 1000) / num_tokens
    prefill_ms = median_prefill * 1000

    return {
        "tok_s": round(tok_s, 2),
        "total_s": round(median_gen, 3),
        "prefill_ms": round(prefill_ms, 1),
        "per_token_ms": round(per_token_ms, 1),
    }


def measure_forward(
    model: nn.Module,
    input_ids: torch.Tensor,
    warmup: int = WARMUP_RUNS,
    runs: int = MEASURED_RUNS,
) -> Optional[dict]:
    """
    Measure forward pass latency (encoder models like BERT).

    Returns dict with latency_ms.
    """
    torch.manual_seed(SEED)
    attention_mask = torch.ones_like(input_ids)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)

    # Measured runs
    times = []
    for _ in range(runs):
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
        times.append(time.perf_counter() - t0)

    times.sort()
    median_time = times[len(times) // 2]

    return {
        "latency_ms": round(median_time * 1000, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Per-model benchmark pipeline
# ═══════════════════════════════════════════════════════════════


def benchmark_causal_model(
    model_cfg: dict,
    seq_lens: list[int],
    wikitext_text: str,
    num_tokens: int = NUM_GEN_TOKENS,
) -> list[dict]:
    """Run FP32 → Ternary → Packed benchmark for a causal model."""
    model_id = model_cfg["id"]
    desc = model_cfg["desc"]
    max_pos = model_cfg["max_pos"]
    results = []

    banner(f"{desc} ({model_id})")

    # Load model
    print(f"  Loading {model_id}...")
    t0 = time.perf_counter()
    model, tokenizer = load_model(model_id, "causal")
    load_time = time.perf_counter() - t0
    params_m = count_params_m(model)
    print(f"  Loaded in {load_time:.1f}s — {params_m:.1f}M params")

    # Filter seq_lens: need room for generation (seq_len + num_tokens <= max_pos)
    valid_seq_lens = [s for s in seq_lens if s + num_tokens <= max_pos]
    skipped_seq_lens = [s for s in seq_lens if s not in valid_seq_lens]
    for s in skipped_seq_lens:
        print(f"\n  seq_len={s}: SKIP (need {s}+{num_tokens}={s+num_tokens} > max_pos={max_pos})")

    for seq_len in valid_seq_lens:
        print(f"\n  --- seq_len={seq_len} ---")

        input_ids = prepare_input_ids(tokenizer, wikitext_text, seq_len)
        print(f"  Input shape: {input_ids.shape}")

        # Check RSS guard
        rss = get_rss_mb()
        if rss > RSS_LIMIT_MB:
            print(f"  RSS {rss:.0f} MB > {RSS_LIMIT_MB} MB — skipping remaining")
            break

        # --- FP32 ---
        print(f"  FP32: measuring {num_tokens} tokens...")
        fp32_result = measure_generation(model, input_ids, num_tokens)
        fp32_rss = get_rss_mb()
        fp32_mem = model_memory_mb(model)

        if fp32_result:
            print(f"    {fp32_result['tok_s']} tok/s, prefill {fp32_result['prefill_ms']:.0f}ms, "
                  f"per-token {fp32_result['per_token_ms']:.1f}ms")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "FP32",
                "tok_s": fp32_result["tok_s"],
                "peak_mb": round(fp32_rss, 0),
                "model_mb": round(fp32_mem, 1),
                "prefill_ms": fp32_result["prefill_ms"],
                "per_token_ms": fp32_result["per_token_ms"],
            })
        else:
            print(f"    FP32 timed out — skipping this seq_len")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "FP32",
                "tok_s": None,
                "peak_mb": round(fp32_rss, 0),
                "model_mb": round(fp32_mem, 1),
                "prefill_ms": None,
                "per_token_ms": None,
            })
            continue

    # Convert to ternary (once, reuse for all remaining seq_lens)
    print(f"\n  Converting to ternary (sensitivity_analysis=False)...")
    t0 = time.perf_counter()
    engine = TernaryInferenceEngine(threshold=0.7)
    report = engine.convert(model, sensitivity_analysis=False)
    convert_time = time.perf_counter() - t0
    print(f"  Converted {report.converted_layers}/{report.total_layers} layers in {convert_time:.1f}s")
    tern_mem = model_memory_mb(model)

    tern_timed_out = False
    for seq_len in valid_seq_lens:
        if tern_timed_out:
            print(f"\n  --- Ternary seq_len={seq_len}: SKIP (previous timeout, larger will be slower) ---")
            continue

        print(f"\n  --- Ternary seq_len={seq_len} ---")
        input_ids = prepare_input_ids(tokenizer, wikitext_text, seq_len)

        rss = get_rss_mb()
        if rss > RSS_LIMIT_MB:
            print(f"  RSS {rss:.0f} MB > {RSS_LIMIT_MB} MB — skipping remaining")
            break

        print(f"  Ternary: measuring {num_tokens} tokens...")
        tern_result = measure_generation(model, input_ids, num_tokens)
        tern_rss = get_rss_mb()

        if tern_result:
            print(f"    {tern_result['tok_s']} tok/s, prefill {tern_result['prefill_ms']:.0f}ms, "
                  f"per-token {tern_result['per_token_ms']:.1f}ms")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "Ternary",
                "tok_s": tern_result["tok_s"],
                "peak_mb": round(tern_rss, 0),
                "model_mb": round(tern_mem, 1),
                "prefill_ms": tern_result["prefill_ms"],
                "per_token_ms": tern_result["per_token_ms"],
            })
        else:
            print(f"    Ternary timed out — skipping larger seq_lens")
            tern_timed_out = True

    # Convert to packed
    print(f"\n  Converting to PackedTernaryLinear...")
    t0 = time.perf_counter()
    pack_stats = convert_model_to_packed(model, threshold=0.7)
    pack_time = time.perf_counter() - t0
    print(f"  Packed {pack_stats['packed_layers']} layers in {pack_time:.1f}s")
    packed_mem = model_memory_mb(model)

    packed_timed_out = False
    for seq_len in valid_seq_lens:
        if packed_timed_out:
            print(f"\n  --- Packed seq_len={seq_len}: SKIP (previous timeout, larger will be slower) ---")
            continue

        print(f"\n  --- Packed seq_len={seq_len} ---")
        input_ids = prepare_input_ids(tokenizer, wikitext_text, seq_len)

        rss = get_rss_mb()
        if rss > RSS_LIMIT_MB:
            print(f"  RSS {rss:.0f} MB > {RSS_LIMIT_MB} MB — skipping remaining")
            break

        print(f"  Packed: measuring {num_tokens} tokens...")
        packed_result = measure_generation(model, input_ids, num_tokens)
        packed_rss = get_rss_mb()

        if packed_result:
            print(f"    {packed_result['tok_s']} tok/s, prefill {packed_result['prefill_ms']:.0f}ms, "
                  f"per-token {packed_result['per_token_ms']:.1f}ms")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "Packed",
                "tok_s": packed_result["tok_s"],
                "peak_mb": round(packed_rss, 0),
                "model_mb": round(packed_mem, 1),
                "prefill_ms": packed_result["prefill_ms"],
                "per_token_ms": packed_result["per_token_ms"],
            })
        else:
            print(f"    Packed timed out — skipping larger seq_lens")
            packed_timed_out = True

    # Cleanup
    del model, tokenizer
    gc.collect()
    print(f"\n  Cleanup done. RSS: {get_rss_mb():.0f} MB")

    return results


def benchmark_encoder_model(
    model_cfg: dict,
    seq_lens: list[int],
    wikitext_text: str,
) -> list[dict]:
    """Run FP32 → Ternary → Packed forward-pass benchmark for encoder model."""
    model_id = model_cfg["id"]
    desc = model_cfg["desc"]
    max_pos = model_cfg["max_pos"]
    results = []

    banner(f"{desc} ({model_id}) — Encoder Forward Pass")

    print(f"  Loading {model_id}...")
    t0 = time.perf_counter()
    model, tokenizer = load_model(model_id, "encoder")
    load_time = time.perf_counter() - t0
    params_m = count_params_m(model)
    print(f"  Loaded in {load_time:.1f}s — {params_m:.1f}M params")

    valid_seq_lens = [s for s in seq_lens if s <= max_pos]

    # FP32 forward
    for seq_len in valid_seq_lens:
        print(f"\n  FP32 seq_len={seq_len}:")
        input_ids = prepare_input_ids(tokenizer, wikitext_text, seq_len)
        fwd = measure_forward(model, input_ids)
        rss = get_rss_mb()
        mem = model_memory_mb(model)
        if fwd:
            print(f"    latency {fwd['latency_ms']:.1f}ms")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "FP32",
                "tok_s": None,
                "peak_mb": round(rss, 0),
                "model_mb": round(mem, 1),
                "prefill_ms": fwd["latency_ms"],
                "per_token_ms": None,
            })

    # Ternary
    print(f"\n  Converting to ternary...")
    t0 = time.perf_counter()
    engine = TernaryInferenceEngine(threshold=0.7)
    report = engine.convert(model, sensitivity_analysis=False)
    convert_time = time.perf_counter() - t0
    print(f"  Converted {report.converted_layers}/{report.total_layers} layers in {convert_time:.1f}s")

    for seq_len in valid_seq_lens:
        print(f"\n  Ternary seq_len={seq_len}:")
        input_ids = prepare_input_ids(tokenizer, wikitext_text, seq_len)
        fwd = measure_forward(model, input_ids)
        rss = get_rss_mb()
        mem = model_memory_mb(model)
        if fwd:
            print(f"    latency {fwd['latency_ms']:.1f}ms")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "Ternary",
                "tok_s": None,
                "peak_mb": round(rss, 0),
                "model_mb": round(mem, 1),
                "prefill_ms": fwd["latency_ms"],
                "per_token_ms": None,
            })

    # Packed
    print(f"\n  Converting to packed...")
    t0 = time.perf_counter()
    pack_stats = convert_model_to_packed(model, threshold=0.7)
    pack_time = time.perf_counter() - t0
    print(f"  Packed {pack_stats['packed_layers']} layers in {pack_time:.1f}s")

    for seq_len in valid_seq_lens:
        print(f"\n  Packed seq_len={seq_len}:")
        input_ids = prepare_input_ids(tokenizer, wikitext_text, seq_len)
        fwd = measure_forward(model, input_ids)
        rss = get_rss_mb()
        mem = model_memory_mb(model)
        if fwd:
            print(f"    latency {fwd['latency_ms']:.1f}ms")
            results.append({
                "model": desc,
                "params_m": round(params_m, 1),
                "seq_len": seq_len,
                "mode": "Packed",
                "tok_s": None,
                "peak_mb": round(rss, 0),
                "model_mb": round(mem, 1),
                "prefill_ms": fwd["latency_ms"],
                "per_token_ms": None,
            })

    del model, tokenizer
    gc.collect()
    print(f"\n  Cleanup done. RSS: {get_rss_mb():.0f} MB")

    return results


# ═══════════════════════════════════════════════════════════════
# Output generation
# ═══════════════════════════════════════════════════════════════


def write_csv(results: list[dict], path: Path) -> None:
    """Write results to CSV."""
    if not results:
        return
    fieldnames = ["model", "params_m", "seq_len", "mode", "tok_s",
                  "peak_mb", "model_mb", "prefill_ms", "per_token_ms"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nCSV written: {path}")


def write_markdown(results: list[dict], path: Path) -> None:
    """Write results to a markdown report."""
    causal = [r for r in results if r["tok_s"] is not None]
    encoder = [r for r in results if r["tok_s"] is None]

    lines = [
        "# Day 12: Performance Scaling Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Generation tokens: {NUM_GEN_TOKENS}",
        f"- Warmup runs: {WARMUP_RUNS}",
        f"- Measured runs: {MEASURED_RUNS} (median)",
        f"- Quantisation threshold: 0.7",
        f"- Sensitivity analysis: disabled",
        f"- Determinism: do_sample=False (Patent 36)",
        "",
    ]

    if causal:
        lines.append("## Causal Model Generation (tok/s)")
        lines.append("")
        lines.append("| Model | Params | Seq Len | FP32 tok/s | Ternary tok/s | Packed tok/s | Tern/FP32 | Pack/FP32 |")
        lines.append("|-------|--------|---------|-----------|--------------|-------------|-----------|-----------|")

        # Group by model + seq_len
        grouped: dict[tuple, dict[str, Optional[float]]] = {}
        for r in causal:
            key = (r["model"], r["params_m"], r["seq_len"])
            if key not in grouped:
                grouped[key] = {"FP32": None, "Ternary": None, "Packed": None}
            grouped[key][r["mode"]] = r["tok_s"]

        for (model, params_m, seq_len), modes in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2])):
            fp32 = modes.get("FP32")
            tern = modes.get("Ternary")
            packed = modes.get("Packed")

            fp32_s = f"{fp32:.1f}" if fp32 else "—"
            tern_s = f"{tern:.1f}" if tern else "—"
            packed_s = f"{packed:.1f}" if packed else "—"

            tern_ratio = f"{tern/fp32:.2f}x" if (tern and fp32) else "—"
            pack_ratio = f"{packed/fp32:.2f}x" if (packed and fp32) else "—"

            lines.append(
                f"| {model} | {params_m}M | {seq_len} | {fp32_s} | {tern_s} | {packed_s} | {tern_ratio} | {pack_ratio} |"
            )

        lines.append("")

        # Prefill table
        lines.append("## Prefill Latency (ms)")
        lines.append("")
        lines.append("| Model | Seq Len | FP32 | Ternary | Packed |")
        lines.append("|-------|---------|------|---------|--------|")

        prefill_grouped: dict[tuple, dict[str, Optional[float]]] = {}
        for r in causal:
            key = (r["model"], r["seq_len"])
            if key not in prefill_grouped:
                prefill_grouped[key] = {"FP32": None, "Ternary": None, "Packed": None}
            prefill_grouped[key][r["mode"]] = r["prefill_ms"]

        for (model, seq_len), modes in sorted(prefill_grouped.items()):
            fp32 = modes.get("FP32")
            tern = modes.get("Ternary")
            packed = modes.get("Packed")
            fp32_s = f"{fp32:.0f}" if fp32 else "—"
            tern_s = f"{tern:.0f}" if tern else "—"
            packed_s = f"{packed:.0f}" if packed else "—"
            lines.append(f"| {model} | {seq_len} | {fp32_s} | {tern_s} | {packed_s} |")

        lines.append("")

    # Memory table
    lines.append("## Memory Usage")
    lines.append("")
    lines.append("| Model | Mode | Model MB | Peak RSS MB |")
    lines.append("|-------|------|----------|-------------|")

    mem_seen: set[tuple] = set()
    for r in results:
        key = (r["model"], r["mode"])
        if key in mem_seen:
            continue
        mem_seen.add(key)
        lines.append(
            f"| {r['model']} | {r['mode']} | {r['model_mb']:.1f} | {r['peak_mb']:.0f} |"
        )
    lines.append("")

    if encoder:
        lines.append("## Encoder Model Forward Latency (ms)")
        lines.append("")
        lines.append("| Model | Seq Len | FP32 | Ternary | Packed |")
        lines.append("|-------|---------|------|---------|--------|")

        enc_grouped: dict[int, dict[str, Optional[float]]] = {}
        for r in encoder:
            if r["seq_len"] not in enc_grouped:
                enc_grouped[r["seq_len"]] = {"FP32": None, "Ternary": None, "Packed": None}
            enc_grouped[r["seq_len"]][r["mode"]] = r["prefill_ms"]

        for seq_len in sorted(enc_grouped.keys()):
            modes = enc_grouped[seq_len]
            fp32 = modes.get("FP32")
            tern = modes.get("Ternary")
            packed = modes.get("Packed")
            fp32_s = f"{fp32:.1f}" if fp32 else "—"
            tern_s = f"{tern:.1f}" if tern else "—"
            packed_s = f"{packed:.1f}" if packed else "—"
            lines.append(f"| BERT-base | {seq_len} | {fp32_s} | {tern_s} | {packed_s} |")

        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Auto-detect patterns
    if causal:
        fp32_rows = [r for r in causal if r["mode"] == "FP32" and r["tok_s"]]
        tern_rows = [r for r in causal if r["mode"] == "Ternary" and r["tok_s"]]
        packed_rows = [r for r in causal if r["mode"] == "Packed" and r["tok_s"]]

        if fp32_rows:
            best_fp32 = max(fp32_rows, key=lambda r: r["tok_s"])
            lines.append(f"- **Fastest FP32**: {best_fp32['model']} at seq_len={best_fp32['seq_len']}: "
                         f"{best_fp32['tok_s']:.1f} tok/s")

        if tern_rows:
            best_tern = max(tern_rows, key=lambda r: r["tok_s"])
            lines.append(f"- **Fastest Ternary**: {best_tern['model']} at seq_len={best_tern['seq_len']}: "
                         f"{best_tern['tok_s']:.1f} tok/s")

        if packed_rows:
            best_packed = max(packed_rows, key=lambda r: r["tok_s"])
            lines.append(f"- **Fastest Packed**: {best_packed['model']} at seq_len={best_packed['seq_len']}: "
                         f"{best_packed['tok_s']:.1f} tok/s")

        # Compute average ternary/FP32 ratio
        ratios = []
        for r in tern_rows:
            fp32_match = [f for f in fp32_rows if f["model"] == r["model"] and f["seq_len"] == r["seq_len"]]
            if fp32_match and fp32_match[0]["tok_s"]:
                ratios.append(r["tok_s"] / fp32_match[0]["tok_s"])
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            lines.append(f"- **Avg Ternary/FP32 ratio**: {avg_ratio:.2f}x (across {len(ratios)} configs)")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Markdown written: {path}")


def write_json(results: list[dict], meta: dict, path: Path) -> None:
    """Write results + metadata to JSON."""
    data = {
        "meta": meta,
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"JSON written: {path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Day 12: Performance Scaling Benchmark")
    parser.add_argument("--recon", action="store_true",
                        help="Recon mode: DistilGPT-2 at seq_len=128 only")
    parser.add_argument("--no-bert", action="store_true",
                        help="Skip BERT encoder benchmark")
    parser.add_argument("--tokens", type=int, default=NUM_GEN_TOKENS,
                        help=f"Number of tokens to generate (default: {NUM_GEN_TOKENS})")
    parser.add_argument("--json-only", action="store_true",
                        help="Only write JSON output, no markdown/CSV")
    args = parser.parse_args()

    num_tokens = args.tokens

    banner("Day 12: Performance Scaling Curve")
    print(f"  Tokens per measurement: {num_tokens}")
    print(f"  Warmup: {WARMUP_RUNS}, Measured: {MEASURED_RUNS} (median)")
    print(f"  RSS limit: {RSS_LIMIT_MB} MB")
    print(f"  Timeout: {TIMEOUT_PER_MEASUREMENT_S}s per measurement")
    start_time = time.perf_counter()

    # Load WikiText-2 text
    print("\n  Loading WikiText-2 test set...")
    wikitext_text = get_wikitext_text()
    print(f"  WikiText-2 loaded: {len(wikitext_text)} chars")

    all_results: list[dict] = []

    if args.recon:
        # Recon mode: DistilGPT-2 at seq_len=128 only
        recon_models = [MODELS[0]]  # distilgpt2
        recon_seq_lens = [128]
        print("\n  RECON MODE: DistilGPT-2 at seq_len=128 only")
    else:
        recon_models = MODELS
        recon_seq_lens = SEQ_LENS

    # Causal models
    for model_cfg in recon_models:
        results = benchmark_causal_model(
            model_cfg, recon_seq_lens, wikitext_text, num_tokens,
        )
        all_results.extend(results)

    # BERT (encoder)
    if not args.recon and not args.no_bert:
        bert_seq_lens = [s for s in SEQ_LENS if s <= BERT_MODEL["max_pos"]]
        results = benchmark_encoder_model(
            BERT_MODEL, bert_seq_lens, wikitext_text,
        )
        all_results.extend(results)

    # Output
    total_time = time.perf_counter() - start_time
    meta = {
        "timestamp": datetime.now().isoformat(),
        "total_time_s": round(total_time, 1),
        "num_tokens": num_tokens,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "threshold": 0.7,
        "seed": SEED,
        "recon_mode": args.recon,
    }

    banner("Results Summary")
    for r in all_results:
        tok_s = f"{r['tok_s']:.1f} tok/s" if r["tok_s"] else f"fwd {r['prefill_ms']:.1f}ms"
        print(f"  {r['model']:25s} seq={r['seq_len']:5d} {r['mode']:8s} → {tok_s}")

    # Write outputs
    out_dir = _BENCH_DIR
    json_path = out_dir / "day12_performance_data.json"
    write_json(all_results, meta, json_path)

    if not args.json_only:
        csv_path = out_dir / "day12_scaling_data.csv"
        write_csv(all_results, csv_path)

        md_path = out_dir / "day12_performance_results.md"
        write_markdown(all_results, md_path)

    print(f"\nTotal benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
