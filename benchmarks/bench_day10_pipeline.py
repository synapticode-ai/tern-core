"""
Day 10: End-to-End Conversion Pipeline Benchmark

Measures:
1. Full pipeline timing (load -> convert -> write)
2. Per-stage timing breakdown
3. Output file size and compression ratio
4. Conversion throughput (params/second)

Runs on synthetic model (fast, no download).
Optional TinyLlama integration (requires model).

Patents 10-12: Automated binary-to-ternary conversion pipeline.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.

Run with: python benchmarks/bench_day10_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

from terncore.convert import TernaryConverter
from terncore.tern_model import TernModelReader


SEED = 42


def banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════
# Synthetic model definitions
# ═══════════════════════════════════════════════════════════════


class SyntheticTransformer(nn.Module):
    """Synthetic transformer-like model for benchmarking."""

    def __init__(self, hidden=256, layers=4, intermediate=512, vocab=1000):
        super().__init__()
        self.embed = nn.Linear(hidden, hidden, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "q_proj": nn.Linear(hidden, hidden, bias=False),
                "k_proj": nn.Linear(hidden, hidden, bias=False),
                "v_proj": nn.Linear(hidden, hidden, bias=False),
                "o_proj": nn.Linear(hidden, hidden, bias=False),
                "gate_proj": nn.Linear(hidden, intermediate, bias=False),
                "up_proj": nn.Linear(hidden, intermediate, bias=False),
                "down_proj": nn.Linear(intermediate, hidden, bias=False),
                "norm": nn.Linear(hidden, hidden, bias=False),
            }))
        self.final_norm = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block["q_proj"](x) + x
        return self.lm_head(x)


# ═══════════════════════════════════════════════════════════════
# 1. Synthetic Pipeline Timing
# ═══════════════════════════════════════════════════════════════


def bench_synthetic_pipeline():
    banner("Synthetic Pipeline Timing")

    torch.manual_seed(SEED)
    model = SyntheticTransformer(hidden=256, layers=4, intermediate=512, vocab=1000)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SyntheticTransformer")
    print(f"  Parameters: {total_params:,}")

    with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
        path = f.name

    try:
        converter = TernaryConverter(
            model_id="test/synthetic-transformer",
            output_path=path,
            threshold=0.7,
        )

        t0 = time.perf_counter()
        stats = converter.convert(verbose=False, model=model)
        total_time = time.perf_counter() - t0

        # Verify output
        reader = TernModelReader(path)
        valid = reader.verify()

        file_size = Path(path).stat().st_size

        throughput = total_params / total_time if total_time > 0 else 0

        result = {
            "total_params": total_params,
            "total_layers": stats["total_layers"],
            "ternary_layers": stats["ternary_layers"],
            "protected_layers": stats["protected_layers"],
            "file_size_bytes": file_size,
            "compression_ratio": stats["compression_ratio"],
            "total_time_s": round(total_time, 3),
            "throughput_params_per_s": round(throughput),
            "valid": valid,
        }

        print(f"  Layers: {stats['total_layers']} total, "
              f"{stats['ternary_layers']} ternary, "
              f"{stats['protected_layers']} protected")
        print(f"  File size: {file_size / 1024:.1f} KB")
        print(f"  Compression: {stats['compression_ratio']:.1f}x vs FP32")
        print(f"  Pipeline time: {total_time * 1000:.1f}ms")
        print(f"  Throughput: {throughput:,.0f} params/s")
        print(f"  Integrity: {'PASS' if valid else 'FAIL'}")

        return result
    finally:
        Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# 2. Scaling test — vary model size
# ═══════════════════════════════════════════════════════════════


def bench_scaling():
    banner("Pipeline Scaling (Varying Model Size)")

    configs = [
        {"hidden": 64, "layers": 2, "intermediate": 128, "vocab": 100},
        {"hidden": 128, "layers": 4, "intermediate": 256, "vocab": 500},
        {"hidden": 256, "layers": 4, "intermediate": 512, "vocab": 1000},
        {"hidden": 512, "layers": 4, "intermediate": 1024, "vocab": 2000},
    ]

    results = []
    for cfg in configs:
        torch.manual_seed(SEED)
        model = SyntheticTransformer(**cfg)
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())

        with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
            path = f.name

        try:
            converter = TernaryConverter(
                model_id="test/scaling",
                output_path=path,
                threshold=0.7,
            )

            t0 = time.perf_counter()
            stats = converter.convert(verbose=False, model=model)
            dt = time.perf_counter() - t0

            file_size = Path(path).stat().st_size
            throughput = total_params / dt if dt > 0 else 0

            entry = {
                "hidden": cfg["hidden"],
                "layers": cfg["layers"],
                "params": total_params,
                "time_ms": round(dt * 1000, 1),
                "file_kb": round(file_size / 1024, 1),
                "compression": stats["compression_ratio"],
                "throughput": round(throughput),
            }
            results.append(entry)

            print(f"  h={cfg['hidden']:>4}  L={cfg['layers']}  "
                  f"params={total_params:>10,}  "
                  f"time={dt * 1000:>8.1f}ms  "
                  f"size={file_size / 1024:>8.1f}KB  "
                  f"compress={stats['compression_ratio']:.1f}x  "
                  f"throughput={throughput:>12,.0f} p/s")
        finally:
            Path(path).unlink(missing_ok=True)

    return results


# ═══════════════════════════════════════════════════════════════
# 3. Round-trip correctness check
# ═══════════════════════════════════════════════════════════════


def bench_round_trip():
    banner("Round-Trip Correctness")

    torch.manual_seed(SEED)
    model = SyntheticTransformer(hidden=64, layers=2, intermediate=128, vocab=100)
    model.eval()

    x = torch.randn(2, 64)
    with torch.no_grad():
        original_out = model(x).clone()

    with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
        path = f.name

    try:
        converter = TernaryConverter(
            model_id="test/round-trip",
            output_path=path,
            threshold=0.7,
        )
        converter.convert(verbose=False, model=model)

        # Load into fresh model
        reader = TernModelReader(path)
        fresh = SyntheticTransformer(hidden=64, layers=2, intermediate=128, vocab=100)
        reader.load_as_model(fresh, strict=False)
        fresh.eval()

        with torch.no_grad():
            loaded_out = fresh(x)

        max_diff = (original_out - loaded_out).abs().max().item()
        mean_diff = (original_out - loaded_out).abs().mean().item()
        has_nan = torch.isnan(loaded_out).any().item()

        result = {
            "max_diff": round(max_diff, 6),
            "mean_diff": round(mean_diff, 6),
            "has_nan": has_nan,
            "output_shape": list(loaded_out.shape),
        }

        print(f"  Original output:  mean={original_out.mean().item():.4f}, "
              f"std={original_out.std().item():.4f}")
        print(f"  Loaded output:    mean={loaded_out.mean().item():.4f}, "
              f"std={loaded_out.std().item():.4f}")
        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")
        print(f"  NaN: {has_nan}")

        return result
    finally:
        Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# 4. Optional TinyLlama integration
# ═══════════════════════════════════════════════════════════════


def bench_tinyllama():
    banner("TinyLlama Integration (Optional)")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        print("  SKIPPED: transformers not installed")
        return None

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        # Quick check if model is cached
        from transformers import AutoConfig
        AutoConfig.from_pretrained(model_id, local_files_only=True)
    except Exception:
        print(f"  SKIPPED: {model_id} not cached locally")
        return None

    with tempfile.NamedTemporaryFile(suffix=".tern-model", delete=False) as f:
        path = f.name

    try:
        converter = TernaryConverter(
            model_id=model_id,
            output_path=path,
            threshold=0.7,
        )

        t0 = time.perf_counter()
        stats = converter.convert(verbose=True)
        total_time = time.perf_counter() - t0

        # Verify
        ok = converter.verify(verbose=True)

        file_size = Path(path).stat().st_size

        result = {
            "model": model_id,
            "total_time_s": round(total_time, 1),
            "file_size_mb": round(file_size / 1024 / 1024, 1),
            "compression_ratio": stats["compression_ratio"],
            "ternary_layers": stats["ternary_layers"],
            "protected_layers": stats["protected_layers"],
            "total_params": stats["total_params"],
            "valid": ok,
        }

        print(f"\n  Summary:")
        print(f"    Total time: {total_time:.1f}s")
        print(f"    File size: {file_size / 1024 / 1024:.1f} MB")
        print(f"    Compression: {stats['compression_ratio']:.1f}x vs FP32")
        print(f"    Ternary: {stats['ternary_layers']}/{stats['total_layers']} layers")
        print(f"    Valid: {ok}")

        return result
    finally:
        Path(path).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    json_only = "--json-only" in sys.argv

    if not json_only:
        print("Day 10: End-to-End Conversion Pipeline Benchmark")
        print(f"PyTorch {torch.__version__}")

    synthetic = bench_synthetic_pipeline()
    scaling = bench_scaling()
    round_trip = bench_round_trip()
    tinyllama = bench_tinyllama()

    all_results = {
        "synthetic_pipeline": synthetic,
        "scaling": scaling,
        "round_trip": round_trip,
        "tinyllama": tinyllama,
    }

    if not json_only:
        banner("Summary")
        print(f"Synthetic pipeline: {synthetic['total_time_s']}s, "
              f"{synthetic['compression_ratio']}x compression")
        print(f"Round-trip max diff: {round_trip['max_diff']}")
        if tinyllama:
            print(f"TinyLlama: {tinyllama['total_time_s']}s, "
                  f"{tinyllama['file_size_mb']} MB, "
                  f"{tinyllama['compression_ratio']}x")

    print("\n" + json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
