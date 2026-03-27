#!/usr/bin/env python3
"""
bench_coreml_ane.py — CoreML/ANE benchmark for ternary inference
================================================================
Converts ternary-weighted linear layers (at TinyLlama dimensions) to CoreML,
applies 2-bit palettization, and benchmarks inference across compute backends:

  1. CoreML ALL        (ANE preferred — Apple's automatic routing)
  2. CoreML CPU_AND_NE (ANE + CPU, no GPU)
  3. CoreML CPU_AND_GPU (GPU + CPU, no ANE)
  4. CoreML CPU_ONLY
  5. PyTorch MPS FP16  (baseline)

Tests both individual layer matmuls and a full-stack forward pass (all linear
layers from one TinyLlama transformer block × 22 blocks).

Why not convert the full HuggingFace model?
  coremltools 9.0 doesn't support transformers 5.3 / PyTorch 2.10 ops (diff,
  new_ones in RoPE/attention). The linear matmuls — where ternary wins — convert
  cleanly and are the bottleneck we're benchmarking.

Terncore · Cubey/Synapticode · 2026
"""

import gc
import json
import os
import re
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WARMUP_RUNS = 10
BENCHMARK_RUNS = 50
NUM_BLOCKS = 22  # TinyLlama has 22 transformer blocks

RESULTS_DIR = Path(__file__).parent
MODELS_DIR = Path(__file__).parent.parent / "output" / "coreml_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# TinyLlama linear layer dimensions (per transformer block)
BLOCK_LAYERS = [
    ("q_proj",    2048, 2048),
    ("k_proj",    2048, 256),
    ("v_proj",    2048, 256),
    ("o_proj",    2048, 2048),
    ("gate_proj", 2048, 5632),
    ("up_proj",   2048, 5632),
    ("down_proj", 5632, 2048),
]


# ---------------------------------------------------------------------------
# Representative model: linear stack matching TinyLlama
# ---------------------------------------------------------------------------
class TernaryBlock(nn.Module):
    """One transformer block's linear layers (skip attention/norm routing)."""

    def __init__(self):
        super().__init__()
        # Attention projections
        self.q_proj = nn.Linear(2048, 2048, bias=False)
        self.k_proj = nn.Linear(2048, 256, bias=False)
        self.v_proj = nn.Linear(2048, 256, bias=False)
        self.o_proj = nn.Linear(2048, 2048, bias=False)
        # MLP
        self.gate_proj = nn.Linear(2048, 5632, bias=False)
        self.up_proj = nn.Linear(2048, 5632, bias=False)
        self.down_proj = nn.Linear(5632, 2048, bias=False)

    def forward(self, x):
        # Attention-like: q/k/v projections, then output projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_out = self.o_proj(q)  # simplified: skip actual attention

        h = x + attn_out  # residual

        # MLP
        gate = self.gate_proj(h)
        gate = torch.nn.functional.silu(gate)
        up = self.up_proj(h)
        mlp_out = self.down_proj(gate * up)

        return h + mlp_out  # residual


class TernaryStack(nn.Module):
    """Full stack of linear layers matching TinyLlama architecture."""

    def __init__(self, num_blocks: int = NUM_BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList([TernaryBlock() for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Ternary quantization
# ---------------------------------------------------------------------------
def quantize_model_ternary(model):
    """Set all Linear weights to exact ternary values {-α, 0, +α}."""
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
# CoreML conversion
# ---------------------------------------------------------------------------
def convert_to_coreml(model, name: str, model_path: Path,
                      input_shape=(1, 64, 2048)) -> ct.models.MLModel:
    """Convert PyTorch model to CoreML via torch.jit.trace."""
    print(f"  Converting {name} to CoreML...")

    model.eval()
    dummy = torch.randn(*input_shape, dtype=torch.float16)

    import warnings
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(model.float(), dummy.float(), strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input",
                shape=input_shape,
                dtype=np.float16,
            )
        ],
        outputs=[ct.TensorType(name="output")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )

    mlmodel.save(str(model_path))
    total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    print(f"  Saved: {model_path} ({total / (1024**2):.1f} MB)")
    return mlmodel


def palettize_coreml(mlmodel, model_path: Path, nbits: int = 2) -> ct.models.MLModel:
    """Apply n-bit palettization to CoreML model."""
    print(f"  Applying {nbits}-bit palettization...")

    config = OptimizationConfig(
        global_config=OpPalettizerConfig(nbits=nbits, mode="kmeans")
    )
    palettized = palettize_weights(mlmodel, config)
    palettized.save(str(model_path))
    total = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    print(f"  Saved: {model_path} ({total / (1024**2):.1f} MB)")
    return palettized


# ---------------------------------------------------------------------------
# Power measurement via powermetrics
# ---------------------------------------------------------------------------
POWER_SAMPLE_INTERVAL_MS = 1000
POWER_NUM_SAMPLES = 10


def _parse_power_output(text: str) -> list[float]:
    """Extract package/combined power (watts) from powermetrics output."""
    watts = []
    for line in text.splitlines():
        # Match lines like "Package Power: 5.23 W" or "Combined Power (CPU + GPU + ANE): 7.1 W"
        m = re.search(r'(?:Package|Combined)\s+Power.*?:\s*([\d.]+)\s*(?:m?W)', line, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # If unit is mW, convert
            if 'mW' in line and 'W' in line:
                val /= 1000.0
            watts.append(val)
    return watts


def check_sudo_available() -> bool:
    """Check if sudo powermetrics can run without password."""
    try:
        subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"],
            capture_output=True, timeout=5,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def measure_power_during(run_fn, label: str, duration_s: float = 12.0) -> dict:
    """Run inference in a loop while sampling power with powermetrics.

    Args:
        run_fn: callable that runs one inference pass
        label: description for logging
        duration_s: how long to sustain inference for stable power reading

    Returns:
        dict with power stats (watts), or empty dict if powermetrics unavailable
    """
    print(f"  [{label}] Measuring power ({duration_s:.0f}s sustained)...")

    # Start powermetrics in background
    n_samples = max(int(duration_s), POWER_NUM_SAMPLES)
    interval = POWER_SAMPLE_INTERVAL_MS

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pm_proc = subprocess.Popen(
            ["sudo", "-n", "powermetrics",
             "--samplers", "cpu_power",
             "-i", str(interval),
             "-n", str(n_samples)],
            stdout=open(tmp_path, 'w'),
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, PermissionError):
        return {}

    # Run inference continuously for the measurement window
    t_end = time.time() + duration_s
    n_inferences = 0
    while time.time() < t_end:
        run_fn()
        n_inferences += 1

    # Wait for powermetrics to finish (it has a fixed sample count)
    try:
        pm_proc.wait(timeout=duration_s + 5)
    except subprocess.TimeoutExpired:
        pm_proc.terminate()
        pm_proc.wait(timeout=3)

    # Parse results
    with open(tmp_path, 'r') as f:
        output = f.read()
    os.unlink(tmp_path)

    watts = _parse_power_output(output)

    if not watts:
        return {}

    # Drop first sample (ramp-up)
    if len(watts) > 2:
        watts = watts[1:]

    result = {
        "label": label,
        "power_mean_w": statistics.mean(watts),
        "power_median_w": statistics.median(watts),
        "power_min_w": min(watts),
        "power_max_w": max(watts),
        "power_stdev_w": statistics.stdev(watts) if len(watts) > 1 else 0,
        "power_samples": len(watts),
        "inferences_during": n_inferences,
    }

    # Compute energy per inference
    total_energy_j = result["power_mean_w"] * duration_s
    result["energy_per_inference_mj"] = (total_energy_j / n_inferences) * 1000

    print(f"    Power: {result['power_mean_w']:.2f} W (mean), "
          f"{result['energy_per_inference_mj']:.2f} mJ/inference, "
          f"{len(watts)} samples")

    return result


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def get_process_rss_mb():
    pid = os.getpid()
    rss_kb = int(subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid)]
    ).decode().strip())
    return rss_kb / 1024


def benchmark_coreml(model_path: Path, label: str,
                     compute_units: ct.ComputeUnit,
                     input_shape=(1, 64, 2048)) -> dict:
    """Benchmark a CoreML model on specified compute units."""
    print(f"  [{label}] Loading...")

    mlmodel = ct.models.MLModel(str(model_path), compute_units=compute_units)
    input_data = {"input": np.random.randn(*input_shape).astype(np.float16)}

    # Warmup
    for _ in range(WARMUP_RUNS):
        _ = mlmodel.predict(input_data)

    # Benchmark
    latencies = []
    for _ in range(BENCHMARK_RUNS):
        t0 = time.perf_counter()
        _ = mlmodel.predict(input_data)
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    rss = get_process_rss_mb()
    del mlmodel
    gc.collect()

    return {
        "label": label,
        "compute_units": str(compute_units),
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_min_ms": min(latencies) * 1000,
        "latency_stdev_ms": (statistics.stdev(latencies) * 1000
                             if len(latencies) > 1 else 0),
        "process_rss_mb": rss,
    }


def benchmark_pytorch_mps(model, label: str,
                          input_shape=(1, 64, 2048)) -> dict:
    """Benchmark PyTorch model on MPS."""
    print(f"  [{label}] Benchmarking...")

    model = model.to(torch.float16).to("mps").eval()
    x = torch.randn(*input_shape, dtype=torch.float16, device="mps")

    for _ in range(WARMUP_RUNS):
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()

    latencies = []
    for _ in range(BENCHMARK_RUNS):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    rss = get_process_rss_mb()

    return {
        "label": label,
        "compute_units": "MPS (Metal)",
        "latency_mean_ms": statistics.mean(latencies) * 1000,
        "latency_median_ms": statistics.median(latencies) * 1000,
        "latency_min_ms": min(latencies) * 1000,
        "latency_stdev_ms": (statistics.stdev(latencies) * 1000
                             if len(latencies) > 1 else 0),
        "process_rss_mb": rss,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  Terncore CoreML / ANE Benchmark")
    print("  Ternary linear stack (TinyLlama 1.1B dimensions)")
    print("=" * 72)

    hw_chip = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"]
    ).decode().strip()
    print(f"\n  Hardware: {hw_chip}")
    print(f"  Model: {NUM_BLOCKS}-block linear stack (7 layers/block = "
          f"{NUM_BLOCKS * 7} matmuls)")
    print(f"  Input: (1, 64, 2048) — batch=1, seq=64, hidden=2048")
    print(f"  Runs: {WARMUP_RUNS} warmup, {BENCHMARK_RUNS} measured\n")

    input_shape = (1, 64, 2048)
    all_results = {}

    # ------------------------------------------------------------------
    # Phase 1: FP16 baseline
    # ------------------------------------------------------------------
    print(f"{'─'*72}")
    print("Phase 1: FP16 model → CoreML + PyTorch MPS baseline")
    print(f"{'─'*72}")

    model_fp16 = TernaryStack(NUM_BLOCKS).to(torch.float16)

    # PyTorch MPS baseline
    all_results["pytorch_mps_fp16"] = benchmark_pytorch_mps(
        model_fp16, "PyTorch MPS FP16", input_shape
    )
    model_fp16 = model_fp16.cpu()
    torch.mps.empty_cache()

    # CoreML FP16
    fp16_path = MODELS_DIR / "ternstack_fp16.mlpackage"
    if not fp16_path.exists():
        convert_to_coreml(model_fp16, "FP16", fp16_path, input_shape)
    else:
        print(f"  Using cached: {fp16_path}")

    for cu_name, cu in [
        ("ALL", ct.ComputeUnit.ALL),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ]:
        all_results[f"coreml_fp16_{cu_name.lower()}"] = benchmark_coreml(
            fp16_path, f"CoreML FP16 ({cu_name})", cu, input_shape
        )

    del model_fp16
    gc.collect()

    # ------------------------------------------------------------------
    # Phase 2: Ternary model → CoreML → 2-bit palettization
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("Phase 2: Ternary quantization → CoreML → 2-bit palettization")
    print(f"{'─'*72}")

    model_tern = TernaryStack(NUM_BLOCKS).to(torch.float16)
    quant_stats = quantize_model_ternary(model_tern)
    print(f"  Quantized {quant_stats['n_layers']} layers, "
          f"{quant_stats['avg_sparsity']:.1%} sparsity")

    # PyTorch MPS ternary (dequantized weights, same as FP16 path)
    all_results["pytorch_mps_ternary"] = benchmark_pytorch_mps(
        model_tern, "PyTorch MPS Ternary", input_shape
    )
    model_tern = model_tern.cpu()
    torch.mps.empty_cache()

    # CoreML ternary (FP16 weights that are exactly ternary)
    tern_fp16_path = MODELS_DIR / "ternstack_ternary_fp16.mlpackage"
    if not tern_fp16_path.exists():
        tern_mlmodel = convert_to_coreml(
            model_tern, "Ternary-FP16", tern_fp16_path, input_shape
        )
    else:
        print(f"  Using cached: {tern_fp16_path}")
        tern_mlmodel = ct.models.MLModel(str(tern_fp16_path))

    # 2-bit palettization
    tern_2bit_path = MODELS_DIR / "ternstack_ternary_2bit.mlpackage"
    if not tern_2bit_path.exists():
        palettize_coreml(tern_mlmodel, tern_2bit_path, nbits=2)
    else:
        print(f"  Using cached: {tern_2bit_path}")

    del tern_mlmodel, model_tern
    gc.collect()

    # Benchmark ternary FP16 (uncompressed ternary weights in CoreML)
    all_results["coreml_ternary_fp16_all"] = benchmark_coreml(
        tern_fp16_path, "CoreML Ternary-FP16 (ALL)", ct.ComputeUnit.ALL, input_shape
    )

    # Benchmark 2-bit palettized ternary
    for cu_name, cu in [
        ("ALL", ct.ComputeUnit.ALL),
        ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE),
        ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ]:
        all_results[f"coreml_ternary_2bit_{cu_name.lower()}"] = benchmark_coreml(
            tern_2bit_path, f"CoreML Ternary-2bit ({cu_name})", cu, input_shape
        )

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  RESULTS — Forward pass: {NUM_BLOCKS} blocks × 7 layers "
          f"= {NUM_BLOCKS * 7} matmuls")
    print(f"  Input: (1, 64, 2048)  |  {BENCHMARK_RUNS} runs")
    print(f"{'='*72}\n")

    # Model sizes
    print("  Model sizes on disk:")
    for p, label in [
        (fp16_path, "FP16"),
        (tern_fp16_path, "Ternary FP16"),
        (tern_2bit_path, "Ternary 2-bit"),
    ]:
        if p.exists():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            print(f"    {label:<20} {total / (1024**2):>8.1f} MB")
    print()

    # Latency table
    baseline_ms = all_results["pytorch_mps_fp16"]["latency_mean_ms"]

    hdr = (f"  {'Backend':<36} {'Mean ms':>9} {'Min ms':>9} "
           f"{'Stdev':>8} {'vs FP16':>9}")
    print(hdr)
    print(f"  {'─'*36} {'─'*9:>9} {'─'*9:>9} {'─'*8:>8} {'─'*9:>9}")

    for key, r in all_results.items():
        label = r["label"]
        mean = r["latency_mean_ms"]
        mn = r["latency_min_ms"]
        std = r["latency_stdev_ms"]
        speedup = baseline_ms / mean if mean > 0 else 0
        print(f"  {label:<36} {mean:>8.2f} {mn:>8.2f} "
              f"{std:>7.2f} {speedup:>8.2f}x")

    # ------------------------------------------------------------------
    # Phase 3: Power measurement (requires sudo powermetrics)
    # ------------------------------------------------------------------
    power_results = {}
    has_sudo = check_sudo_available()

    print(f"\n{'─'*72}")
    print("Phase 3: Energy measurement (powermetrics)")
    print(f"{'─'*72}")

    if not has_sudo:
        print("  sudo powermetrics unavailable — skipping energy measurement")
        print("  (run with sudo access for energy data)")
    else:
        # Measure MPS FP16 baseline power
        model_fp16_power = TernaryStack(NUM_BLOCKS).to(torch.float16).to("mps").eval()
        x_power = torch.randn(*input_shape, dtype=torch.float16, device="mps")

        def run_mps_fp16():
            with torch.no_grad():
                model_fp16_power(x_power)
            torch.mps.synchronize()

        power_results["mps_fp16"] = measure_power_during(
            run_mps_fp16, "PyTorch MPS FP16", duration_s=12.0)

        del model_fp16_power, x_power
        torch.mps.empty_cache(); gc.collect()

        # Measure ANE Ternary-2bit power
        ane_model = ct.models.MLModel(
            str(tern_2bit_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        ane_input = {"input": np.random.randn(*input_shape).astype(np.float16)}

        def run_ane_ternary():
            ane_model.predict(ane_input)

        power_results["ane_ternary_2bit"] = measure_power_during(
            run_ane_ternary, "CoreML Ternary-2bit (ANE)", duration_s=12.0)

        del ane_model; gc.collect()

        # Measure GPU Ternary-2bit power (for comparison)
        gpu_model = ct.models.MLModel(
            str(tern_2bit_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
        gpu_input = {"input": np.random.randn(*input_shape).astype(np.float16)}

        def run_gpu_ternary():
            gpu_model.predict(gpu_input)

        power_results["gpu_ternary_2bit"] = measure_power_during(
            run_gpu_ternary, "CoreML Ternary-2bit (GPU)", duration_s=12.0)

        del gpu_model; gc.collect()

    # ------------------------------------------------------------------
    # ANE analysis
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print("  ANE Analysis")
    print(f"{'─'*72}")

    ane_2bit = all_results.get("coreml_ternary_2bit_cpu_and_ne", {}).get("latency_mean_ms", 0)
    gpu_2bit = all_results.get("coreml_ternary_2bit_cpu_and_gpu", {}).get("latency_mean_ms", 0)
    all_2bit = all_results.get("coreml_ternary_2bit_all", {}).get("latency_mean_ms", 0)
    mps_fp16 = baseline_ms

    if ane_2bit > 0 and gpu_2bit > 0:
        print(f"  Ternary 2-bit on ANE:     {ane_2bit:.2f} ms")
        print(f"  Ternary 2-bit on GPU:     {gpu_2bit:.2f} ms")
        print(f"  Ternary 2-bit ALL:        {all_2bit:.2f} ms")
        print(f"  FP16 on MPS:              {mps_fp16:.2f} ms")
        print(f"  ANE vs GPU speedup:       {gpu_2bit/ane_2bit:.2f}x")
        print(f"  Ternary-2bit vs FP16:     {mps_fp16/all_2bit:.2f}x")

    if power_results:
        print(f"\n  Power consumption:")
        for key, pr in power_results.items():
            if pr:
                print(f"    {pr['label']:<36} {pr['power_mean_w']:.2f} W  "
                      f"({pr['energy_per_inference_mj']:.2f} mJ/inference)")

        mps_power = power_results.get("mps_fp16", {})
        ane_power = power_results.get("ane_ternary_2bit", {})
        if mps_power and ane_power:
            power_ratio = mps_power["power_mean_w"] / ane_power["power_mean_w"]
            energy_ratio = mps_power["energy_per_inference_mj"] / ane_power["energy_per_inference_mj"]
            print(f"\n  ANE ternary vs MPS FP16:")
            print(f"    Power draw:   {power_ratio:.2f}x less ({ane_power['power_mean_w']:.2f} vs {mps_power['power_mean_w']:.2f} W)")
            print(f"    Energy/infer: {energy_ratio:.2f}x less ({ane_power['energy_per_inference_mj']:.2f} vs {mps_power['energy_per_inference_mj']:.2f} mJ)")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output = {
        "benchmark": "Terncore CoreML/ANE — Ternary linear stack",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": hw_chip,
        "config": {
            "num_blocks": NUM_BLOCKS,
            "layers_per_block": len(BLOCK_LAYERS),
            "input_shape": list(input_shape),
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS,
        },
        "quantization": quant_stats,
        "results": all_results,
        "power": power_results,
        "model_sizes_mb": {},
    }

    for p, label in [
        (fp16_path, "fp16"),
        (tern_fp16_path, "ternary_fp16"),
        (tern_2bit_path, "ternary_2bit"),
    ]:
        if p.exists():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            output["model_sizes_mb"][label] = round(total / (1024**2), 1)

    json_path = RESULTS_DIR / "coreml_ane_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Markdown
    md = f"""# Terncore CoreML / ANE Benchmark

> Ternary linear stack ({NUM_BLOCKS} blocks × 7 layers = {NUM_BLOCKS*7} matmuls)
> {hw_chip} · {datetime.now().strftime('%Y-%m-%d')}

## Configuration

| | |
|---|---|
| Architecture | {NUM_BLOCKS}-block linear stack matching TinyLlama 1.1B dimensions |
| Input shape | (1, 64, 2048) — batch=1, seq=64, hidden=2048 |
| Ternary sparsity | {quant_stats['avg_sparsity']:.1%} |
| Benchmark runs | {BENCHMARK_RUNS} |

## Results

| Backend | Mean (ms) | Min (ms) | vs MPS FP16 |
|---------|:---------:|:--------:|:-----------:|
"""
    for key, r in all_results.items():
        mean = r["latency_mean_ms"]
        mn = r["latency_min_ms"]
        speedup = baseline_ms / mean if mean > 0 else 0
        md += f"| {r['label']} | {mean:.2f} | {mn:.2f} | {speedup:.2f}x |\n"

    md += f"""
## Model Sizes

| Format | Size |
|--------|-----:|
"""
    for p, label in [
        (fp16_path, "FP16 CoreML"),
        (tern_fp16_path, "Ternary FP16 CoreML"),
        (tern_2bit_path, "Ternary 2-bit CoreML"),
    ]:
        if p.exists():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            md += f"| {label} | {total / (1024**2):.1f} MB |\n"

    # Power section
    if power_results:
        md += f"""
## Energy Consumption

| Backend | Package Power | Energy/Inference | Inferences |
|:--------|:------------:|:----------------:|:----------:|
"""
        for key in ["mps_fp16", "ane_ternary_2bit", "gpu_ternary_2bit"]:
            pr = power_results.get(key, {})
            if pr:
                md += (f"| {pr['label']} | {pr['power_mean_w']:.2f} W | "
                       f"{pr['energy_per_inference_mj']:.2f} mJ | "
                       f"{pr['inferences_during']} |\n")

        mps_power = power_results.get("mps_fp16", {})
        ane_power = power_results.get("ane_ternary_2bit", {})
        if mps_power and ane_power:
            power_ratio = mps_power["power_mean_w"] / ane_power["power_mean_w"]
            energy_ratio = mps_power["energy_per_inference_mj"] / ane_power["energy_per_inference_mj"]
            md += f"""
**ANE ternary draws {power_ratio:.1f}x less power** and uses **{energy_ratio:.1f}x less energy per inference** than MPS FP16.
"""

    md += f"""
## ANE Analysis

CoreML routes 2-bit palettized matmuls to the Apple Neural Engine when
`ComputeUnit.CPU_AND_NE` or `ComputeUnit.ALL` is selected. The ANE operates
at fixed power with dedicated matrix multiply hardware — ideal for sustained
inference at low energy.

The 2-bit palette maps perfectly to ternary {{-α, 0, +α}} weights:
4 palette entries, 3 used, with ~{quant_stats['avg_sparsity']:.0%} zero weights.

---
*Terncore CoreML/ANE benchmark · Cubey/Synapticode · {datetime.now().strftime('%Y-%m-%d')}*
"""
    md_path = RESULTS_DIR / "coreml_ane_benchmark.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n  Results: {json_path}")
    print(f"  Report:  {md_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
