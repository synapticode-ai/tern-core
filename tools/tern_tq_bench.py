"""
TurboQuant KV cache benchmark — single-row measurement harness.

Parallel to tern_infer.py. Drives generate_streaming_turboquant() on a
provided HuggingFace causal LM, captures KPIs matching the schema in
benchmarks/tq_bench_results.json (TinyLlama-1.1B, 2026-03-30 baseline),
and emits one row to benchmarks/tq_bench_results_<model-slug>_<UTC>.json.

Purpose 3 + Fork α RTP validation: tests whether MixedPrecisionConverter
generalises beyond TinyLlama/Mistral to new model families.

Architecture note for Gemma 4: this harness loads via the model-class
hook (default: AutoModelForCausalLM, which for Gemma 4 maps to
Gemma4ForCausalLM — text-only path). The multimodal
Gemma4ForConditionalGeneration class is NOT exercised here; its
audio_tower and vision_tower contain nn.Linear modules that MPC's
pattern-based protection (embed/norm/lm_head) does not cover, so a
text-only invocation isolates the RTP question from the multimodal
protection-list-extension question.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.mixed_precision import MixedPrecisionConverter

# Reuse the proven TurboQuant streaming driver from tern_infer
from tern_infer import (
    _extract_kv_pairs,
    IncrementalTQCompressor,
)


def _rss_mb() -> float:
    """Process peak RSS in MiB (macOS reports maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _kv_bytes_per_head_dim(past_key_values) -> dict[int, int]:
    """Bytes per head_dim across all (k,v) tensors in a HF cache."""
    out: dict[int, int] = {}
    for item in past_key_values:
        if isinstance(item, (tuple, list)):
            k, v = item[0], item[1]
        else:
            k, v = item.keys, item.values
        head_dim = int(k.shape[-1])
        b = k.element_size() * k.numel() + v.element_size() * v.numel()
        out[head_dim] = out.get(head_dim, 0) + b
    return out


def _kv_bytes(past_key_values) -> int:
    return sum(_kv_bytes_per_head_dim(past_key_values).values())


def _tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.numel()
    return 0


def _walk_compressed_bytes(obj, seen: set | None = None) -> int:
    """Recursively sum tensor bytes inside a compressed-record nested
    structure, deduping tensors by id() so shared codebook + rotation
    references are not double-counted across records.
    """
    if seen is None:
        seen = set()
    if isinstance(obj, torch.Tensor):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        return _tensor_bytes(obj)
    if isinstance(obj, (tuple, list)):
        return sum(_walk_compressed_bytes(x, seen) for x in obj)
    if hasattr(obj, "__dict__"):
        return sum(_walk_compressed_bytes(v, seen) for v in vars(obj).values())
    return 0


def _compressed_bytes_per_head_dim(compressor: IncrementalTQCompressor) -> dict[int, int]:
    """Bytes per head_dim across compressed (k_c, v_c) records.

    Dedup is per-head_dim: codebook + rotation tensors are shared within
    a head_dim partition (one TurboQuantConfig per head_dim), so the
    "compressed bytes" attributable to a head_dim partition counts each
    shared tensor once across all records in that partition.
    """
    layer_to_dim: dict[int, int] = {}
    for d, layers in compressor.head_dim_layers.items():
        for l in layers:
            layer_to_dim[l] = d
    # group records by head_dim first, then dedup within each group
    grouped: dict[int, list] = {}
    for (l, h), records in compressor.compressed.items():
        d = layer_to_dim.get(l, -1)
        grouped.setdefault(d, []).extend(records)
    out: dict[int, int] = {}
    for d, all_records in grouped.items():
        seen: set = set()
        out[d] = _walk_compressed_bytes(all_records, seen)
    return out


def _compressed_bytes(compressor: IncrementalTQCompressor) -> int:
    return sum(_compressed_bytes_per_head_dim(compressor).values())


def _measure_ppl(model, tokenizer, passages: list[str], device=None) -> float:
    """Average per-token NLL over short passages, exponentiated.

    Lightweight PPL probe — single forward pass per passage, no sliding
    window. Sufficient for the v0 baseline-vs-TQ delta comparison; not
    comparable to WikiText-2 sliding-window numbers.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for passage in passages:
            ids = tokenizer(passage, return_tensors="pt").input_ids.to(device)
            if ids.shape[1] < 2:
                continue
            inputs = ids[:, :-1]
            targets = ids[:, 1:]
            out = model(inputs, use_cache=False)
            logits = out.logits  # (1, T, V)
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            nll = -log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
            total_nll += nll.sum().item()
            total_tokens += targets.numel()
    if total_tokens == 0:
        return float("nan")
    return math.exp(total_nll / total_tokens)


_PPL_PASSAGES = [
    "The future of computing lies in efficient quantisation of neural network weights.",
    "Ternary representation maps each weight to one of three discrete values, simplifying matrix multiplication to additions and subtractions.",
    "Large language models require careful balance between compression and accuracy preservation.",
    "Apple Silicon's unified memory architecture enables novel inference workloads that traditional GPUs cannot match.",
    "Streaming generation with incremental KV cache compression keeps long-context inference within bounded memory.",
]


def _generate_baseline(model, tokenizer, prompt: str, max_tokens: int, device):
    """Generate with HF kv_cache, no compression — captures uncompressed KV bytes."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()
    past = None
    t0 = time.perf_counter()
    prefill_ms = None
    tokens = 0
    final_pkv = None
    with torch.no_grad():
        for step in range(max_tokens):
            t_step = time.perf_counter()
            if past is not None:
                out = model(generated_ids[:, -1:], past_key_values=past, use_cache=True)
            else:
                out = model(generated_ids, use_cache=True)
                prefill_ms = (time.perf_counter() - t_step) * 1000.0
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)
            tokens += 1
            final_pkv = past
    elapsed = time.perf_counter() - t0
    tps = tokens / elapsed if elapsed > 0 else 0.0
    per_dim = _kv_bytes_per_head_dim(final_pkv) if final_pkv is not None else {}
    uncompressed_mb_per_dim = {d: b / (1024 * 1024) for d, b in per_dim.items()}
    uncompressed_mb = sum(uncompressed_mb_per_dim.values())
    return {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tps": tps,
        "prefill_ms": prefill_ms,
        "uncompressed_kv_mb": uncompressed_mb,
        "uncompressed_kv_mb_per_head_dim": uncompressed_mb_per_dim,
    }


def _generate_turboquant(model, tokenizer, prompt: str, max_tokens: int, device):
    """Generate with HF kv_cache + incremental TurboQuant compression.

    Uses the per-head_dim-lazy IncrementalTQCompressor — handles
    heterogeneous attention layouts (e.g. Gemma 4's 256/512 mix).
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()
    past = None
    compressor = IncrementalTQCompressor()
    t0 = time.perf_counter()
    prefill_ms = None
    encode_ms_samples = []
    first_encode_verified = False
    tokens = 0
    with torch.no_grad():
        for step in range(max_tokens):
            t_step = time.perf_counter()
            if past is not None:
                out = model(generated_ids[:, -1:], past_key_values=past, use_cache=True)
            else:
                out = model(generated_ids, use_cache=True)
                prefill_ms = (time.perf_counter() - t_step) * 1000.0
            past = out.past_key_values
            t_enc = time.perf_counter()
            compressor.append(past)
            encode_ms_samples.append((time.perf_counter() - t_enc) * 1000.0)
            if not first_encode_verified:
                summary = compressor.summary()
                print(
                    f"[bench] first encode OK — head_dim partition: "
                    + ", ".join(
                        f"d={d}:{info['n_layers']}L" for d, info in sorted(summary.items())
                    )
                )
                first_encode_verified = True
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)
            tokens += 1
    elapsed = time.perf_counter() - t0
    tps = tokens / elapsed if elapsed > 0 else 0.0

    per_dim_bytes = _compressed_bytes_per_head_dim(compressor)
    compressed_mb_per_dim = {d: b / (1024 * 1024) for d, b in per_dim_bytes.items()}
    compressed_mb = sum(compressed_mb_per_dim.values())

    # Per-token encode = mean of decode-step encodes (skip prefill-pass)
    per_token_encode_ms = (
        sum(encode_ms_samples[1:]) / max(1, len(encode_ms_samples) - 1)
        if len(encode_ms_samples) > 1 else float("nan")
    )
    prefill_encode_ms = encode_ms_samples[0] if encode_ms_samples else float("nan")
    return {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tps": tps,
        "prefill_ms": prefill_ms,
        "prefill_encode_ms": prefill_encode_ms,
        "per_token_encode_ms": per_token_encode_ms,
        "compressed_kv_mb": compressed_mb,
        "compressed_kv_mb_per_head_dim": compressed_mb_per_dim,
        "head_dim_layer_counts": {
            d: info["n_layers"] for d, info in compressor.summary().items()
        },
    }


_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _resolve_model_class(model_id: str, override: str | None) -> str:
    """Pick the HF model class. Default = AutoModelForCausalLM, but for
    google/gemma-4-* checkpoints AutoModelForCausalLM resolves to the
    multimodal Gemma4ForConditionalGeneration; if no override is given,
    upgrade to the explicit text-only Gemma4ForCausalLM.
    """
    if override and override != "auto":
        return override
    if "gemma-4" in model_id.lower():
        return "Gemma4ForCausalLM"
    return "AutoModelForCausalLM"


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model id")
    parser.add_argument("--prompt", default="The future of computing lies in")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--no-autoscan", action="store_true",
        help="Skip auto PPL scan; use pattern-based protection only (faster RTP gate)",
    )
    parser.add_argument(
        "--model-class", default="auto",
        help="HF model class for from_pretrained. 'auto' picks Gemma4ForCausalLM for gemma-4-* ids, else AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--dtype", choices=list(_DTYPE_MAP.keys()), default="fp16",
        help="Model dtype (default: fp16 — halves RSS vs fp32)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device: 'auto' (default — MPS if available, else CPU), 'cpu', 'mps'",
    )
    parser.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent.parent / "benchmarks"),
    )
    args = parser.parse_args()

    rss_start = _rss_mb()
    print(f"[bench] start RSS={rss_start:.0f} MiB")

    dtype = _DTYPE_MAP[args.dtype]
    device = _resolve_device(args.device)
    model_class_name = _resolve_model_class(args.model, args.model_class)
    print(f"[bench] dtype={args.dtype}  device={device}  model_class={model_class_name}")

    t_load0 = time.perf_counter()
    from transformers import AutoTokenizer
    import transformers as tx
    cls = getattr(tx, model_class_name)
    print(f"[bench] loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = cls.from_pretrained(args.model, dtype=dtype)
    if device.type != "cpu":
        try:
            model = model.to(device)
        except Exception as e:
            print(f"[bench] WARNING: .to({device}) failed ({type(e).__name__}: {e}); falling back to CPU")
            device = torch.device("cpu")
    model.eval()
    t_load = time.perf_counter() - t_load0
    rss_after_load = _rss_mb()
    print(f"[bench] load={t_load:.1f}s  RSS={rss_after_load:.0f} MiB  Δ={rss_after_load-rss_start:.0f}")

    # PPL baseline (on canonical short passages)
    print(f"[bench] measuring baseline PPL (dtype={args.dtype})...")
    t_ppl0 = time.perf_counter()
    ppl_fp = _measure_ppl(model, tokenizer, _PPL_PASSAGES, device=device)
    print(f"[bench] PPL_baseline={ppl_fp:.4f} ({time.perf_counter()-t_ppl0:.1f}s)")

    # MPC convert
    print(f"[bench] converting via MixedPrecisionConverter(threshold={args.threshold}, auto={not args.no_autoscan})...")
    t_conv0 = time.perf_counter()
    converter = MixedPrecisionConverter(
        threshold=args.threshold,
        auto=not args.no_autoscan,
    )
    # When auto=False, no model_id needed; when auto=True, pass model_id to trigger autoscan
    convert_kwargs = {} if args.no_autoscan else {"model_id": args.model}
    report = converter.convert(model, **convert_kwargs)
    t_conv = time.perf_counter() - t_conv0
    rss_after_conv = _rss_mb()
    print(
        f"[bench] convert={t_conv:.1f}s  RSS={rss_after_conv:.0f} MiB"
        f"  converted={report.converted_layers}/{report.total_layers}"
        f"  compression={report.compression_ratio:.2f}x"
    )

    # Baseline generation (uncompressed KV cache)
    print("[bench] baseline generation (uncompressed KV)...")
    base = _generate_baseline(model, tokenizer, args.prompt, args.max_tokens, device)
    print(
        f"[bench] base: {base['tokens']} tok, {base['tps']:.1f} tok/s, "
        f"KV={base['uncompressed_kv_mb']:.1f} MiB, prefill={base['prefill_ms']:.1f} ms"
    )
    for d, mb in sorted(base["uncompressed_kv_mb_per_head_dim"].items()):
        print(f"[bench]   uncompressed head_dim={d}: {mb:.2f} MiB")

    gc.collect()

    # TurboQuant generation
    print("[bench] turboquant generation (compressed KV)...")
    tq = _generate_turboquant(model, tokenizer, args.prompt, args.max_tokens, device)
    print(
        f"[bench] tq:   {tq['tokens']} tok, {tq['tps']:.1f} tok/s, "
        f"KV={tq['compressed_kv_mb']:.2f} MiB, encode={tq['per_token_encode_ms']:.3f} ms/tok"
    )
    for d, mb in sorted(tq["compressed_kv_mb_per_head_dim"].items()):
        layer_count = tq["head_dim_layer_counts"].get(d, "?")
        print(f"[bench]   compressed head_dim={d}: {mb:.2f} MiB ({layer_count} layers)")

    # PPL with ternary weights (post-convert)
    print("[bench] measuring ternary PPL...")
    ppl_tq = _measure_ppl(model, tokenizer, _PPL_PASSAGES, device=device)
    print(f"[bench] PPL_ternary={ppl_tq:.4f}")

    rss_peak = _rss_mb()
    compression_ratio = (
        base["uncompressed_kv_mb"] / tq["compressed_kv_mb"]
        if tq["compressed_kv_mb"] > 0 else float("nan")
    )
    ppl_delta_pct = (
        100.0 * (ppl_tq - ppl_fp) / ppl_fp if math.isfinite(ppl_fp) and ppl_fp > 0 else float("nan")
    )

    iso = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = args.model.split("/")[-1].replace("-", "_").replace(".", "_").lower()
    out_path = Path(args.out_dir) / f"tq_bench_results_{slug}_{iso}.json"

    # Per-head_dim compression breakdown
    per_head_dim_breakdown: dict[str, dict] = {}
    for d in sorted(set(base["uncompressed_kv_mb_per_head_dim"]) | set(tq["compressed_kv_mb_per_head_dim"])):
        unc = base["uncompressed_kv_mb_per_head_dim"].get(d, 0.0)
        com = tq["compressed_kv_mb_per_head_dim"].get(d, 0.0)
        ratio = (unc / com) if com > 0 else float("nan")
        per_head_dim_breakdown[str(d)] = {
            "layer_count": tq["head_dim_layer_counts"].get(d, 0),
            "uncompressed_mb": unc,
            "compressed_mb": com,
            "compression_ratio": ratio,
        }

    row = {
        "benchmark": "TurboQuant KV cache compression",
        "hardware": "Apple M4 Pro",
        "date_utc": iso,
        "model": args.model,
        "model_class": model_class_name,
        "method": "incremental TurboQuant (QJL + rotation, mixed-precision 3-bit MSE) — per-head_dim lazy compressor",
        "description": "Measures KV cache compression via TurboQuant hook during autoregressive generation. Prefill encodes full prompt KV in one batch; each decode step encodes only the new KV pair. Per-layer head_dim detected lazily — supports heterogeneous attention (e.g. Gemma 4 sliding/full mix).",
        "config": {
            "threshold": args.threshold,
            "autoscan": not args.no_autoscan,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "device": str(device),
            "b_mse": 3,
            "mixed_precision": True,
        },
        "results": {
            "kv_cache_compression_ratio_aggregate": compression_ratio,
            "per_head_dim_breakdown": per_head_dim_breakdown,
            "prefill_overhead_ms": tq.get("prefill_encode_ms"),
            "per_token_encode_ms": tq.get("per_token_encode_ms"),
            "peak_memory_uncompressed_mb": base["uncompressed_kv_mb"],
            "peak_memory_compressed_mb": tq["compressed_kv_mb"],
            "perplexity_baseline": ppl_fp,
            "perplexity_with_ternary_weights": ppl_tq,
            "perplexity_delta_pct": ppl_delta_pct,
            "ppl_note": (
                "PPL_baseline measured on a short canonical passage set "
                "(5 sentences), single forward pass each — not comparable to "
                "WikiText-2 sliding-window perplexity in tq_bench_results.json. "
                "ternary PPL reflects weight-only compression impact; the "
                "TurboQuant compressor encodes KV state but does not route "
                "back into forward in this harness. For Gemma 4 specifically, "
                "see Gemma-4-PPL-methodology backlog item — baseline numbers "
                "may be pathological without proper logit-softcap / chat-template "
                "preprocessing."
            ),
        },
        "tokens": {
            "baseline_tps": base["tps"],
            "turboquant_tps": tq["tps"],
            "tokens_generated_baseline": base["tokens"],
            "tokens_generated_tq": tq["tokens"],
        },
        "memory_mib": {
            "rss_start": rss_start,
            "rss_after_load": rss_after_load,
            "rss_after_convert": rss_after_conv,
            "rss_peak": rss_peak,
        },
        "conversion_report": {
            "total_layers": report.total_layers,
            "converted_layers": report.converted_layers,
            "skipped_layers": report.skipped_layers,
            "total_params": report.total_params,
            "ternary_params": report.ternary_params,
            "compression_ratio": report.compression_ratio,
            "original_size_mb": report.original_size_mb,
            "ternary_size_mb": report.ternary_size_mb,
        },
        "timings_s": {
            "model_load": t_load,
            "convert": t_conv,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
