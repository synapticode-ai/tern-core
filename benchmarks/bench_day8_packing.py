"""
Day 8: 2-Bit Weight Packing Benchmark

Measures:
1. Memory reduction: nn.Linear vs TernaryLinear vs PackedTernaryLinear
2. Conversion overhead: time to convert model
3. Inference latency: packed vs unpacked forward pass
4. Compression ratio at various sizes

Sizes: [256x256, 512x512, 2048x2048, 2048x256 (v_proj shape)]

Patent 1: Ternary weight encoding.
Patent 5: Ternary execution path — packed weight computation.
Patent 39: Ternary-native memory — 2-bit packed storage format.

Copyright (c) 2025 Synapticode Co., Ltd. All rights reserved.

Run with: python benchmarks/bench_day8_packing.py
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.nn as nn

from terncore.arithmetic.linear import TernaryLinear
from terncore.packed_linear import PackedTernaryLinear, convert_model_to_packed


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

SIZES = [
    (256, 256),
    (512, 512),
    (2048, 2048),
    (2048, 256),   # TinyLlama v_proj shape
]

THRESHOLD = 0.7
WARMUP = 50
ITERS = 500
SEED = 42


def banner(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════
# 1. Memory Comparison
# ═══════════════════════════════════════════════════════════════

def bench_memory():
    banner("Memory Comparison: FP32 vs TernaryLinear vs PackedTernaryLinear")

    results = []

    for out_f, in_f in SIZES:
        torch.manual_seed(SEED)
        linear = nn.Linear(in_f, out_f, bias=False)

        # FP32 memory
        fp32_bytes = sum(p.nelement() * p.element_size() for p in linear.parameters())

        # TernaryLinear — stores FP32 weights + caches int8 ternary
        ternary = TernaryLinear(in_f, out_f, bias=False, threshold=THRESHOLD)
        ternary.weight.data.copy_(linear.weight.data)
        ternary_param_bytes = sum(
            p.nelement() * p.element_size() for p in ternary.parameters()
        )

        # PackedTernaryLinear — stores 2-bit packed
        packed = PackedTernaryLinear.from_float(linear, threshold=THRESHOLD)
        fp = packed.memory_footprint()

        ratio_vs_fp32 = fp32_bytes / fp["total_packed_bytes"]
        ratio_vs_ternary = ternary_param_bytes / fp["total_packed_bytes"]

        entry = {
            "size": f"{out_f}x{in_f}",
            "weights": out_f * in_f,
            "fp32_bytes": fp32_bytes,
            "ternary_param_bytes": ternary_param_bytes,
            "packed_bytes": fp["packed_bytes"],
            "alpha_bytes": fp["alpha_bytes"],
            "total_packed_bytes": fp["total_packed_bytes"],
            "compression_vs_fp32": round(ratio_vs_fp32, 1),
            "compression_vs_ternary": round(ratio_vs_ternary, 1),
        }
        results.append(entry)

        print(f"  {out_f:>5}x{in_f:<5}  "
              f"FP32={fp32_bytes:>12,} B  "
              f"Ternary={ternary_param_bytes:>12,} B  "
              f"Packed={fp['total_packed_bytes']:>10,} B  "
              f"({ratio_vs_fp32:.1f}x vs FP32, "
              f"{ratio_vs_ternary:.1f}x vs Ternary)")

    return results


# ═══════════════════════════════════════════════════════════════
# 2. Conversion Overhead
# ═══════════════════════════════════════════════════════════════

def bench_conversion():
    banner("Conversion Overhead: nn.Linear → PackedTernaryLinear")

    results = []

    for out_f, in_f in SIZES:
        torch.manual_seed(SEED)

        times = []
        for _ in range(10):
            linear = nn.Linear(in_f, out_f, bias=True)
            t0 = time.perf_counter()
            PackedTernaryLinear.from_float(linear, threshold=THRESHOLD)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        mean_ms = sum(times) / len(times)
        entry = {
            "size": f"{out_f}x{in_f}",
            "conversion_ms": round(mean_ms, 2),
        }
        results.append(entry)
        print(f"  {out_f:>5}x{in_f:<5}  conversion: {mean_ms:.2f} ms (avg of 10)")

    return results


# ═══════════════════════════════════════════════════════════════
# 3. Inference Latency
# ═══════════════════════════════════════════════════════════════

def bench_latency():
    banner("Inference Latency: nn.Linear vs PackedTernaryLinear")

    results = []

    for out_f, in_f in SIZES:
        torch.manual_seed(SEED)
        linear = nn.Linear(in_f, out_f, bias=True)
        linear.eval()

        packed = PackedTernaryLinear.from_float(linear, threshold=THRESHOLD)
        packed.eval()

        x = torch.randn(1, in_f)

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP):
                linear(x)
                packed(x)

        # Measure nn.Linear
        linear_times = []
        with torch.no_grad():
            for _ in range(ITERS):
                t0 = time.perf_counter()
                linear(x)
                t1 = time.perf_counter()
                linear_times.append((t1 - t0) * 1e6)  # us

        # Measure PackedTernaryLinear
        packed_times = []
        with torch.no_grad():
            for _ in range(ITERS):
                t0 = time.perf_counter()
                packed(x)
                t1 = time.perf_counter()
                packed_times.append((t1 - t0) * 1e6)  # us

        linear_mean = sum(linear_times) / len(linear_times)
        packed_mean = sum(packed_times) / len(packed_times)
        ratio = linear_mean / packed_mean if packed_mean > 0 else 0

        entry = {
            "size": f"{out_f}x{in_f}",
            "linear_us": round(linear_mean, 1),
            "packed_us": round(packed_mean, 1),
            "ratio": round(ratio, 2),
        }
        results.append(entry)

        faster = "PACKED" if ratio > 1 else "LINEAR"
        print(f"  {out_f:>5}x{in_f:<5}  "
              f"Linear={linear_mean:>8.1f} us  "
              f"Packed={packed_mean:>8.1f} us  "
              f"ratio={ratio:.2f}x  ({faster} faster)")

    return results


# ═══════════════════════════════════════════════════════════════
# 4. Model Conversion (multi-layer)
# ═══════════════════════════════════════════════════════════════

def bench_model_conversion():
    banner("Model Conversion: Multi-Layer Network")

    torch.manual_seed(SEED)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2048, 2048)
            self.fc2 = nn.Linear(2048, 2048)
            self.fc3 = nn.Linear(2048, 512)
            self.head = nn.Linear(512, 32000)  # like lm_head

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.head(x)

    model = MLP()
    model.eval()

    # Before: FP32 memory
    before_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())

    # Convert (protect head)
    t0 = time.perf_counter()
    stats = convert_model_to_packed(
        model, threshold=THRESHOLD, protection_list=["head"]
    )
    t1 = time.perf_counter()

    # After: packed + remaining params
    after_bytes = 0
    for buf in model.buffers():
        after_bytes += buf.nelement() * buf.element_size()
    for p in model.parameters():
        after_bytes += p.nelement() * p.element_size()

    conversion_ms = (t1 - t0) * 1000
    ratio = before_bytes / after_bytes if after_bytes > 0 else 0

    result = {
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "compression": round(ratio, 2),
        "conversion_ms": round(conversion_ms, 1),
        "packed_layers": stats["packed_layers"],
        "protected_layers": stats["protected_layers"],
    }

    print(f"  Before:     {before_bytes:>12,} bytes ({before_bytes / 1024 / 1024:.1f} MB)")
    print(f"  After:      {after_bytes:>12,} bytes ({after_bytes / 1024 / 1024:.1f} MB)")
    print(f"  Compression: {ratio:.2f}x")
    print(f"  Conversion:  {conversion_ms:.1f} ms")
    print(f"  Packed:      {stats['packed_layers']} layers")
    print(f"  Protected:   {stats['protected_layers']} layers")

    # Verify model still works
    x = torch.randn(1, 2048)
    with torch.no_grad():
        output = model(x)
    print(f"  Output OK:   shape={output.shape}")

    return result


# ═══════════════════════════════════════════════════════════════
# 5. Correctness Verification
# ═══════════════════════════════════════════════════════════════

def bench_correctness():
    banner("Correctness: PackedTernaryLinear vs TernaryLinear")

    results = []

    for out_f, in_f in SIZES:
        torch.manual_seed(SEED)
        linear = nn.Linear(in_f, out_f, bias=True)

        # TernaryLinear
        ternary = TernaryLinear(in_f, out_f, bias=True, threshold=THRESHOLD)
        ternary.weight.data.copy_(linear.weight.data)
        ternary.bias.data.copy_(linear.bias.data)
        ternary.eval()

        # PackedTernaryLinear
        packed = PackedTernaryLinear.from_float(linear, threshold=THRESHOLD)
        packed.eval()

        x = torch.randn(4, in_f)
        with torch.no_grad():
            t_out = ternary(x)
            p_out = packed(x)

        max_diff = (t_out - p_out).abs().max().item()
        matches = torch.allclose(t_out, p_out, atol=1e-5)

        entry = {
            "size": f"{out_f}x{in_f}",
            "max_diff": max_diff,
            "matches": matches,
        }
        results.append(entry)

        status = "PASS" if matches else "FAIL"
        print(f"  {out_f:>5}x{in_f:<5}  max_diff={max_diff:.2e}  [{status}]")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    json_only = "--json-only" in sys.argv

    if not json_only:
        print("Day 8: 2-Bit Weight Packing Benchmark")
        print(f"PyTorch {torch.__version__}, threshold={THRESHOLD}")

    memory = bench_memory()
    conversion = bench_conversion()
    latency = bench_latency()
    model_conv = bench_model_conversion()
    correctness = bench_correctness()

    all_results = {
        "memory": memory,
        "conversion": conversion,
        "latency": latency,
        "model_conversion": model_conv,
        "correctness": correctness,
    }

    if not json_only:
        banner("Summary")
        print("Memory: PackedTernaryLinear achieves 16x compression vs FP32")
        print("        (2-bit packed: 0.25 bytes/weight vs 4 bytes/weight)")
        all_correct = all(c["matches"] for c in correctness)
        print(f"Correctness: {'ALL PASS' if all_correct else 'SOME FAILED'}")

    # Always output JSON
    print("\n" + json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
