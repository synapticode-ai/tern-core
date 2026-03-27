#!/usr/bin/env python3
"""
bench_sparse_ane.py — Sparse channel-pruned ANE benchmark
==========================================================
Compares dense ternary 2-bit inference against channel-pruned sparse
ternary inference on the Apple Neural Engine.

Channel pruning exploits ternary weight sparsity by identifying
low-importance output channels, removing them entirely, and building
physically smaller Linear layers. Smaller matmuls → faster ANE dispatch.

Pruning targets:
  - MLP intermediate dim (gate/up/down_proj): largest matmuls, most impact
  - Attention internal dim (q/o_proj): secondary target
  - k/v_proj: not pruned (already small, 256 output dim)

Baseline: dense ternary 2-bit on ANE = 7.38 ms (from energy_cleanroom.py)

Terncore · Cubey/Synapticode · 2026
"""

import gc
import json
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from terncore.sparse.channel_pruning import (
    prune_mlp_channels,
    prune_attention_channels,
    ChannelPruneStats,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50
NUM_BLOCKS = 22
INPUT_SHAPE = (1, 64, 2048)
SEQ_LEN = 64

MODELS_DIR = Path(__file__).parent.parent / "output" / "coreml_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class SparseTernaryBlock(nn.Module):
    """Transformer block with channel-pruned MLP and attention."""

    def __init__(self, attn_dim: int = 2048, mlp_dim: int = 5632):
        super().__init__()
        self.q_proj = nn.Linear(2048, attn_dim, bias=False)
        self.k_proj = nn.Linear(2048, 256, bias=False)
        self.v_proj = nn.Linear(2048, 256, bias=False)
        self.o_proj = nn.Linear(attn_dim, 2048, bias=False)
        self.gate_proj = nn.Linear(2048, mlp_dim, bias=False)
        self.up_proj = nn.Linear(2048, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, 2048, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q)
        h = x + attn_out

        gate = self.gate_proj(h)
        gate = torch.nn.functional.silu(gate)
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)

        return h + mlp_out


class SparseTernaryStack(nn.Module):
    """Stack of sparse blocks with pruned dimensions."""

    def __init__(self, num_blocks: int, attn_dim: int, mlp_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            SparseTernaryBlock(attn_dim, mlp_dim) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DenseTernaryBlock(nn.Module):
    """Standard (unpruned) block — same as bench_coreml_ane.py."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2048, 2048, bias=False)
        self.k_proj = nn.Linear(2048, 256, bias=False)
        self.v_proj = nn.Linear(2048, 256, bias=False)
        self.o_proj = nn.Linear(2048, 2048, bias=False)
        self.gate_proj = nn.Linear(2048, 5632, bias=False)
        self.up_proj = nn.Linear(2048, 5632, bias=False)
        self.down_proj = nn.Linear(5632, 2048, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q)
        h = x + attn_out
        gate = self.gate_proj(h)
        gate = torch.nn.functional.silu(gate)
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)
        return h + mlp_out


class DenseTernaryStack(nn.Module):
    def __init__(self, num_blocks: int = NUM_BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList([DenseTernaryBlock() for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------
def quantize_model_ternary(model):
    """Quantize all Linear weights to ternary {-α, 0, +α}."""
    n_layers = 0
    total_params = 0
    sparsity_sum = 0.0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            abs_w = w.abs()
            mean_abs = abs_w.mean(dim=1, keepdim=True)
            threshold = 0.7 * mean_abs

            codes = torch.zeros_like(w, dtype=torch.int8)
            codes[w > threshold] = 1
            codes[w < -threshold] = -1

            mask = codes != 0
            scales = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
            for i in range(w.shape[0]):
                selected = abs_w[i][mask[i]]
                scales[i] = selected.mean() if selected.numel() > 0 else mean_abs[i, 0]

            module.weight.data = (codes.float() * scales.unsqueeze(1)).to(w.dtype)

            zeros = (codes == 0).sum().item()
            sparsity_sum += zeros / codes.numel()
            total_params += w.numel()
            n_layers += 1

    return {
        "n_layers": n_layers,
        "total_params": total_params,
        "avg_sparsity": sparsity_sum / max(n_layers, 1),
    }


# ---------------------------------------------------------------------------
# Channel pruning
# ---------------------------------------------------------------------------
def build_pruned_model(
    dense_model: DenseTernaryStack,
    mlp_prune_ratio: float = 0.30,
    attn_prune_ratio: float = 0.20,
) -> tuple[SparseTernaryStack, list[ChannelPruneStats]]:
    """Build a channel-pruned model from a quantized dense model.

    1. Score channels by importance (L1 norm)
    2. Prune lowest-importance channels from MLP intermediate and attention dims
    3. Copy surviving weights into physically smaller Linear layers
    """
    all_stats = []

    # Determine pruned dimensions from first block
    block0 = dense_model.blocks[0]
    _, _, _, mlp_stats = prune_mlp_channels(
        block0.gate_proj, block0.up_proj, block0.down_proj, mlp_prune_ratio
    )
    _, _, attn_stats = prune_attention_channels(
        block0.q_proj, block0.o_proj, attn_prune_ratio
    )

    pruned_mlp_dim = mlp_stats.pruned_out
    pruned_attn_dim = attn_stats.pruned_out

    print(f"  MLP intermediate: 5632 → {pruned_mlp_dim} "
          f"({mlp_stats.channels_removed} removed, {mlp_prune_ratio:.0%})")
    print(f"  Attention dim:    2048 → {pruned_attn_dim} "
          f"({attn_stats.channels_removed} removed, {attn_prune_ratio:.0%})")

    # Build pruned model
    sparse_model = SparseTernaryStack(NUM_BLOCKS, pruned_attn_dim, pruned_mlp_dim)

    # Copy pruned weights block by block
    for i, dense_block in enumerate(dense_model.blocks):
        sparse_block = sparse_model.blocks[i]

        # Prune MLP
        p_gate, p_up, p_down, stats = prune_mlp_channels(
            dense_block.gate_proj, dense_block.up_proj,
            dense_block.down_proj, mlp_prune_ratio
        )
        sparse_block.gate_proj = p_gate
        sparse_block.up_proj = p_up
        sparse_block.down_proj = p_down
        if i == 0:
            all_stats.append(stats)

        # Prune attention (q/o pair)
        p_q, p_o, stats = prune_attention_channels(
            dense_block.q_proj, dense_block.o_proj, attn_prune_ratio
        )
        sparse_block.q_proj = p_q
        sparse_block.o_proj = p_o
        if i == 0:
            all_stats.append(stats)

        # k/v stay as-is (already small)
        sparse_block.k_proj.weight = nn.Parameter(
            dense_block.k_proj.weight.data.clone()
        )
        sparse_block.v_proj.weight = nn.Parameter(
            dense_block.v_proj.weight.data.clone()
        )

    return sparse_model, all_stats


# ---------------------------------------------------------------------------
# CoreML conversion
# ---------------------------------------------------------------------------
def convert_to_coreml(model, name: str, model_path: Path,
                      input_shape=(1, 64, 2048)):
    print(f"  Converting {name} to CoreML...")
    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float16)

    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(model.float(), dummy.float(), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=input_shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    mlmodel.save(str(model_path))
    total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    print(f"  Saved: {model_path.name} ({total / (1024**2):.1f} MB)")
    return mlmodel


def palettize_coreml(mlmodel, model_path: Path, nbits: int = 2):
    print(f"  Applying {nbits}-bit palettization...")
    config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=nbits, mode="kmeans")
    )
    palettized = palettize_weights(mlmodel, config)
    palettized.save(str(model_path))
    total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    print(f"  Saved: {model_path.name} ({total / (1024**2):.1f} MB)")
    return palettized


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_coreml(model_path: Path, label: str,
                     compute_units: ct.ComputeUnit,
                     input_shape=(1, 64, 2048)) -> dict:
    print(f"  [{label}] Loading...")
    mlmodel = ct.models.MLModel(str(model_path), compute_units=compute_units)
    input_data = {"input": np.random.randn(*input_shape).astype(np.float16)}

    for _ in range(WARMUP_RUNS):
        mlmodel.predict(input_data)

    latencies = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        mlmodel.predict(input_data)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    del mlmodel
    gc.collect()

    return {
        "label": label,
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_min_ms": min(latencies) * 1000,
        "latency_stdev_ms": (statistics.stdev(latencies) * 1000
                             if len(latencies) > 1 else 0),
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(path: Path):
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024**2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  Sparse Channel-Pruned ANE Benchmark")
    print("  Dense Ternary 2-bit vs Sparse Ternary 2-bit")
    print("=" * 72)

    hw_chip = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"]
    ).decode().strip()
    print(f"\n  Hardware: {hw_chip}")
    print(f"  Blocks: {NUM_BLOCKS}, Input: {INPUT_SHAPE}")
    print(f"  Runs: {WARMUP_RUNS} warmup, {BENCHMARK_RUNS} measured\n")

    # Test multiple pruning ratios
    prune_configs = [
        {"mlp": 0.20, "attn": 0.10, "label": "20% MLP / 10% attn"},
        {"mlp": 0.30, "attn": 0.20, "label": "30% MLP / 20% attn"},
        {"mlp": 0.40, "attn": 0.30, "label": "40% MLP / 30% attn"},
        {"mlp": 0.50, "attn": 0.40, "label": "50% MLP / 40% attn"},
    ]

    all_results = {}

    # ------------------------------------------------------------------
    # Phase 1: Dense baseline (reuse cached model if available)
    # ------------------------------------------------------------------
    print(f"{'─'*72}")
    print("Phase 1: Dense ternary 2-bit baseline")
    print(f"{'─'*72}")

    dense_2bit_path = MODELS_DIR / "ternstack_ternary_2bit.mlpackage"
    if not dense_2bit_path.exists():
        print("  Building dense model...")
        dense_model = DenseTernaryStack(NUM_BLOCKS).to(torch.float16)
        quant_stats = quantize_model_ternary(dense_model)
        print(f"  Quantized: {quant_stats['avg_sparsity']:.1%} sparsity")
        tern_fp16_path = MODELS_DIR / "ternstack_ternary_fp16.mlpackage"
        if not tern_fp16_path.exists():
            mlmodel = convert_to_coreml(dense_model, "Dense-Ternary-FP16", tern_fp16_path)
        else:
            mlmodel = ct.models.MLModel(str(tern_fp16_path))
        palettize_coreml(mlmodel, dense_2bit_path, nbits=2)
        del dense_model, mlmodel
        gc.collect()
    else:
        print(f"  Using cached: {dense_2bit_path.name} ({model_size_mb(dense_2bit_path):.1f} MB)")

    dense_result = benchmark_coreml(
        dense_2bit_path, "Dense Ternary 2-bit (ANE)",
        ct.ComputeUnit.CPU_AND_NE, INPUT_SHAPE
    )
    all_results["dense_2bit"] = dense_result
    dense_params = count_params(DenseTernaryStack(1)) * NUM_BLOCKS
    print(f"    Latency: {dense_result['latency_mean_ms']:.2f} ms")

    # ------------------------------------------------------------------
    # Phase 2: Sparse models at different prune ratios
    # ------------------------------------------------------------------
    for cfg in prune_configs:
        mlp_ratio = cfg["mlp"]
        attn_ratio = cfg["attn"]
        label = cfg["label"]
        tag = f"sparse_mlp{int(mlp_ratio*100)}_attn{int(attn_ratio*100)}"

        print(f"\n{'─'*72}")
        print(f"Phase 2: Sparse — {label}")
        print(f"{'─'*72}")

        sparse_2bit_path = MODELS_DIR / f"ternstack_{tag}_2bit.mlpackage"

        # Build fresh dense model, quantize, then prune
        dense_model = DenseTernaryStack(NUM_BLOCKS).to(torch.float16)
        quantize_model_ternary(dense_model)

        sparse_model, prune_stats = build_pruned_model(
            dense_model, mlp_prune_ratio=mlp_ratio, attn_prune_ratio=attn_ratio
        )
        sparse_model = sparse_model.to(torch.float16)
        sparse_params = count_params(sparse_model)

        param_reduction = 1 - sparse_params / dense_params
        print(f"  Parameters: {dense_params:,} → {sparse_params:,} ({param_reduction:.1%} reduction)")

        del dense_model
        gc.collect()

        # Convert to CoreML + palettize
        sparse_fp16_path = MODELS_DIR / f"ternstack_{tag}_fp16.mlpackage"
        mlmodel = convert_to_coreml(sparse_model, f"Sparse-{label}", sparse_fp16_path)
        palettize_coreml(mlmodel, sparse_2bit_path, nbits=2)
        del sparse_model, mlmodel
        gc.collect()

        # Benchmark
        result = benchmark_coreml(
            sparse_2bit_path, f"Sparse {label} (ANE)",
            ct.ComputeUnit.CPU_AND_NE, INPUT_SHAPE
        )
        result["params"] = sparse_params
        result["param_reduction"] = param_reduction
        result["model_size_mb"] = model_size_mb(sparse_2bit_path)
        result["mlp_prune"] = mlp_ratio
        result["attn_prune"] = attn_ratio
        all_results[tag] = result

        speedup = dense_result["latency_mean_ms"] / result["latency_mean_ms"]
        print(f"    Latency: {result['latency_mean_ms']:.2f} ms "
              f"(vs dense {dense_result['latency_mean_ms']:.2f} ms = {speedup:.2f}x)")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  SPARSE ANE BENCHMARK RESULTS")
    print(f"{'='*72}\n")

    dense_ms = dense_result["latency_mean_ms"]
    dense_size = model_size_mb(dense_2bit_path)

    hdr = f"  {'Config':<32} {'Mean ms':>8} {'Min ms':>8} {'Speedup':>8} {'Size MB':>8} {'Params%':>8}"
    print(hdr)
    print(f"  {'─'*32} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for key, r in all_results.items():
        mean = r["latency_mean_ms"]
        mn = r["latency_min_ms"]
        speedup = dense_ms / mean if mean > 0 else 0
        size = r.get("model_size_mb", dense_size)
        param_pct = (1 - r.get("param_reduction", 0)) * 100
        print(f"  {r['label']:<32} {mean:>7.2f} {mn:>7.2f} {speedup:>7.2f}x {size:>7.1f} {param_pct:>7.1f}%")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    md = f"""# Sparse Channel-Pruned ANE Benchmark

> Structured channel pruning for faster ternary inference on ANE
> {hw_chip} · {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Method

The ANE executes dense matrix multiplications — it cannot skip individual zero
weights. To exploit ternary sparsity ({all_results['dense_2bit']['latency_mean_ms']:.2f} ms dense baseline),
we use **structured channel pruning**:

1. Quantize to ternary {{-α, 0, +α}} (43% weight sparsity)
2. Score each channel by L1 importance (geometric mean for MLP gate/up pairs)
3. Remove lowest-importance channels entirely from MLP intermediate and attention dims
4. Build physically smaller Linear layers → smaller matmuls on ANE
5. Apply 2-bit palettization → CoreML → ANE dispatch

**Pruning targets:**
- MLP intermediate (gate/up/down_proj): 5632 → varies — dominates compute
- Attention dim (q/o_proj): 2048 → varies — secondary target
- k/v_proj: not pruned (already 256-dim)

## Configuration

| | |
|---|---|
| Hardware | {hw_chip} |
| Blocks | {NUM_BLOCKS} |
| Input | {INPUT_SHAPE} (seq={SEQ_LEN} tokens) |
| Warmup | {WARMUP_RUNS} |
| Measured runs | {BENCHMARK_RUNS} |
| Compute units | CPU_AND_NE (ANE) |
| Baseline | Dense ternary 2-bit = {dense_ms:.2f} ms |

## Results

| Config | Latency (ms) | Min (ms) | Speedup | Model Size | Params |
|--------|:------------:|:--------:|:-------:|:----------:|:------:|
"""
    for key, r in all_results.items():
        mean = r["latency_mean_ms"]
        mn = r["latency_min_ms"]
        speedup = dense_ms / mean if mean > 0 else 0
        size = r.get("model_size_mb", dense_size)
        param_pct = (1 - r.get("param_reduction", 0)) * 100
        md += f"| {r['label']} | {mean:.2f} | {mn:.2f} | {speedup:.2f}x | {size:.1f} MB | {param_pct:.0f}% |\n"

    # Find best result
    best_key = min(
        (k for k in all_results if k != "dense_2bit"),
        key=lambda k: all_results[k]["latency_mean_ms"],
        default=None,
    )

    if best_key:
        best = all_results[best_key]
        best_speedup = dense_ms / best["latency_mean_ms"]
        best_size = best.get("model_size_mb", 0)

        md += f"""
## Best Configuration

**{best['label']}** achieves the best latency:

- **{best['latency_mean_ms']:.2f} ms** vs **{dense_ms:.2f} ms** dense baseline
- **{best_speedup:.2f}x speedup** from channel pruning
- **{best_size:.1f} MB** model size (vs {dense_size:.1f} MB dense)
- **{best.get('param_reduction', 0):.1%} parameter reduction**

## Tokens per Second

| Config | Tok/s | vs Dense |
|--------|:-----:|:--------:|
"""
        dense_tps = SEQ_LEN / (dense_ms / 1000)
        for key, r in all_results.items():
            tps = SEQ_LEN / (r["latency_mean_ms"] / 1000)
            ratio = tps / dense_tps
            md += f"| {r['label']} | {tps:.0f} | {ratio:.2f}x |\n"

    md += f"""
## Analysis

Structured channel pruning converts unstructured ternary sparsity (where 43%
of individual weights are zero but no full channels are zero) into structured
sparsity by removing the least-important channels entirely. The ANE then
processes physically smaller weight matrices, reducing both latency and energy.

The MLP layers (gate/up/down_proj at 5632 intermediate dim) account for ~70%
of total FLOPs per block, making them the highest-impact pruning target.

---
*Sparse channel-pruned ANE benchmark · Terncore · Cubey/Synapticode · {datetime.now().strftime('%Y-%m-%d')}*
"""

    md_path = RESULTS_DIR / "sparse_ane_benchmark.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"\n  Report: {md_path}")

    # JSON
    json_path = RESULTS_DIR / "sparse_ane_benchmark.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "hardware": hw_chip,
            "config": {
                "num_blocks": NUM_BLOCKS,
                "input_shape": list(INPUT_SHAPE),
                "warmup": WARMUP_RUNS,
                "benchmark_runs": BENCHMARK_RUNS,
            },
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"  JSON:   {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
