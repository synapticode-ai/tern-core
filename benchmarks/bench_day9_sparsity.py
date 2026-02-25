"""
Day 9: Sparsity Bitmap Zero-Skip Benchmark

Measures:
1. Bitmap caching speedup: cached vs rebuilt-per-call forward pass
2. C kernel zero-skip at varying sparsity levels (0% to 90%)
3. Block-level sparsity distribution at various element sparsity levels
4. Per-layer sparsity analysis for synthetic model

Patent 7: Sparsity-aware execution — cached bitmap.
Patent 9: Zero-skip via bitmap-driven sparse kernel.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.

Run with: python benchmarks/bench_day9_sparsity.py
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn as nn

from terncore.packed_linear import PackedTernaryLinear, _build_bitmap_from_packed
from terncore.packed_ops import packed_ternary_matmul, packed_ternary_matmul_fast
from terncore.sparse import (
    analyze_block_sparsity,
    model_sparsity_report,
    pack_ternary_weights,
)


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SEED = 42
WARMUP = 50
ITERS = 500
SIZE = (2048, 2048)  # Production-relevant matrix size

SPARSITY_LEVELS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9]


def banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def create_sparse_weights(
    shape: tuple[int, int], sparsity: float, seed: int = 42
) -> torch.Tensor:
    """Create ternary weights with specified sparsity level."""
    torch.manual_seed(seed)
    weights = torch.randn(shape)
    mask = torch.rand(shape) < sparsity
    weights[mask] = 0.0
    weights[~mask] = torch.sign(weights[~mask])
    return weights


# ═══════════════════════════════════════════════════════════════
# 1. Bitmap Caching Speedup
# ═══════════════════════════════════════════════════════════════

def bench_bitmap_caching():
    banner("Bitmap Caching Speedup: Cached vs Rebuilt-Per-Call")

    torch.manual_seed(SEED)
    out_f, in_f = SIZE
    linear = nn.Linear(in_f, out_f, bias=False)

    packed = PackedTernaryLinear.from_float(linear, threshold=0.7)
    packed.eval()
    x = torch.randn(1, in_f)

    # Measure: forward with cached bitmap (via PackedTernaryLinear.forward)
    with torch.no_grad():
        for _ in range(WARMUP):
            packed(x)

    cached_times = []
    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            packed(x)
            t1 = time.perf_counter()
            cached_times.append((t1 - t0) * 1e6)

    # Measure: forward with rebuilt bitmap (no pre-built bitmap)
    with torch.no_grad():
        for _ in range(WARMUP):
            packed_ternary_matmul_fast(
                x, packed.packed_weights, packed.alpha.item(),
                out_f, in_f, sparsity_bitmap=None,
            )

    rebuilt_times = []
    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            packed_ternary_matmul_fast(
                x, packed.packed_weights, packed.alpha.item(),
                out_f, in_f, sparsity_bitmap=None,
            )
            t1 = time.perf_counter()
            rebuilt_times.append((t1 - t0) * 1e6)

    # Measure: reference path (unpack → float → F.linear)
    with torch.no_grad():
        for _ in range(WARMUP):
            packed_ternary_matmul(
                x, packed.packed_weights, packed.alpha.item(), out_f, in_f,
            )

    ref_times = []
    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            packed_ternary_matmul(
                x, packed.packed_weights, packed.alpha.item(), out_f, in_f,
            )
            t1 = time.perf_counter()
            ref_times.append((t1 - t0) * 1e6)

    cached_mean = sum(cached_times) / len(cached_times)
    rebuilt_mean = sum(rebuilt_times) / len(rebuilt_times)
    ref_mean = sum(ref_times) / len(ref_times)

    speedup = rebuilt_mean / cached_mean if cached_mean > 0 else 0

    result = {
        "size": f"{out_f}x{in_f}",
        "cached_us": round(cached_mean, 1),
        "rebuilt_us": round(rebuilt_mean, 1),
        "reference_us": round(ref_mean, 1),
        "caching_speedup": round(speedup, 2),
    }

    print(f"  Cached bitmap:   {cached_mean:>10.1f} us")
    print(f"  Rebuilt per-call: {rebuilt_mean:>10.1f} us")
    print(f"  Reference (F.linear): {ref_mean:>10.1f} us")
    print(f"  Caching speedup: {speedup:.2f}x")

    return result


# ═══════════════════════════════════════════════════════════════
# 2. Zero-Skip at Varying Sparsity
# ═══════════════════════════════════════════════════════════════

def bench_sparsity_curve():
    banner("Zero-Skip Speedup at Varying Sparsity Levels")

    out_f, in_f = SIZE
    x = torch.randn(1, in_f)
    results = []

    for sparsity in SPARSITY_LEVELS:
        ternary = create_sparse_weights((out_f, in_f), sparsity, seed=SEED)
        packed, _ = pack_ternary_weights(ternary)
        alpha = 1.0

        # Build bitmap
        bitmap = _build_bitmap_from_packed(packed, out_f, in_f)

        # Warmup
        for _ in range(WARMUP):
            packed_ternary_matmul_fast(
                x, packed, alpha, out_f, in_f, sparsity_bitmap=bitmap,
            )

        # Measure with bitmap (zero-skip)
        skip_times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            packed_ternary_matmul_fast(
                x, packed, alpha, out_f, in_f, sparsity_bitmap=bitmap,
            )
            t1 = time.perf_counter()
            skip_times.append((t1 - t0) * 1e6)

        # Measure reference (no zero-skip)
        for _ in range(WARMUP):
            packed_ternary_matmul(x, packed, alpha, out_f, in_f)

        ref_times = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            packed_ternary_matmul(x, packed, alpha, out_f, in_f)
            t1 = time.perf_counter()
            ref_times.append((t1 - t0) * 1e6)

        skip_mean = sum(skip_times) / len(skip_times)
        ref_mean = sum(ref_times) / len(ref_times)
        ratio = ref_mean / skip_mean if skip_mean > 0 else 0

        # Block-level analysis
        block_analysis = analyze_block_sparsity(packed, out_f, in_f)

        actual_sparsity = (ternary == 0).float().mean().item()

        entry = {
            "target_sparsity": sparsity,
            "actual_sparsity": round(actual_sparsity, 3),
            "skip_us": round(skip_mean, 1),
            "ref_us": round(ref_mean, 1),
            "speedup": round(ratio, 2),
            "block_skip_ratio": block_analysis["block_skip_ratio"],
            "zero_blocks": block_analysis["zero_blocks"],
            "total_blocks": block_analysis["total_blocks"],
        }
        results.append(entry)

        print(
            f"  Sparsity {actual_sparsity:>5.1%}  "
            f"Skip={skip_mean:>8.1f} us  "
            f"Ref={ref_mean:>8.1f} us  "
            f"Speedup={ratio:.2f}x  "
            f"BlockSkip={block_analysis['block_skip_ratio']:.1%}"
        )

    return results


# ═══════════════════════════════════════════════════════════════
# 3. Block-Level Sparsity Distribution
# ═══════════════════════════════════════════════════════════════

def bench_block_distribution():
    banner("Block-Level Sparsity Distribution")

    out_f, in_f = SIZE
    results = []

    for sparsity in [0.3, 0.44, 0.65, 0.8, 0.9]:
        ternary = create_sparse_weights((out_f, in_f), sparsity, seed=SEED)
        packed, _ = pack_ternary_weights(ternary)

        for block_size in [64, 256]:
            analysis = analyze_block_sparsity(packed, out_f, in_f, block_size)
            actual = (ternary == 0).float().mean().item()

            entry = {
                "target_sparsity": sparsity,
                "actual_sparsity": round(actual, 3),
                "block_size": block_size,
                "total_blocks": analysis["total_blocks"],
                "zero_blocks": analysis["zero_blocks"],
                "block_skip_ratio": analysis["block_skip_ratio"],
                "mean_block_sparsity": analysis["mean_block_sparsity"],
            }
            results.append(entry)

            print(
                f"  Sparsity={actual:>5.1%}  "
                f"Block={block_size:>3}  "
                f"ZeroBlocks={analysis['zero_blocks']:>5}/{analysis['total_blocks']:<5}  "
                f"SkipRatio={analysis['block_skip_ratio']:>6.1%}"
            )

    return results


# ═══════════════════════════════════════════════════════════════
# 4. Model Sparsity Report
# ═══════════════════════════════════════════════════════════════

def bench_model_report():
    banner("Model Sparsity Report (Synthetic Multi-Layer)")

    torch.manual_seed(SEED)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2048, 2048)
            self.fc2 = nn.Linear(2048, 512)
            self.fc3 = nn.Linear(512, 256)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

    model = MLP()
    from terncore.packed_linear import convert_model_to_packed
    convert_model_to_packed(model, threshold=0.7)

    report = model_sparsity_report(model)

    for entry in report:
        print(
            f"  {entry['name']:<10}  "
            f"Weights={entry['total_weights']:>10,}  "
            f"Sparsity={entry['sparsity']:>5.1%}  "
            f"BlockSkip={entry['block_skip_ratio']:>5.1%}  "
            f"ZeroBlocks={entry['zero_blocks']}/{entry['total_blocks']}"
        )

    return report


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    json_only = "--json-only" in sys.argv

    if not json_only:
        print("Day 9: Sparsity Bitmap Zero-Skip Benchmark")
        print(f"PyTorch {torch.__version__}")
        print(f"Matrix size: {SIZE[0]}x{SIZE[1]}")

    caching = bench_bitmap_caching()
    curve = bench_sparsity_curve()
    blocks = bench_block_distribution()
    model_report = bench_model_report()

    all_results = {
        "bitmap_caching": caching,
        "sparsity_curve": curve,
        "block_distribution": blocks,
        "model_report": model_report,
    }

    if not json_only:
        banner("Summary")
        print(f"Bitmap caching speedup: {caching['caching_speedup']}x")
        if curve:
            best = max(curve, key=lambda x: x["speedup"])
            print(
                f"Best zero-skip speedup: {best['speedup']}x "
                f"at {best['actual_sparsity']:.0%} sparsity"
            )

    print("\n" + json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
