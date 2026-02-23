# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Synapticode** workspace — a multi-repository ternary AI computing stack (CNS Synaptic). The workspace contains scaffold definitions and active source code for ternary neural network inference.

Tern-core is the reference implementation for Synapticode's 56-patent ternary computing portfolio. It must demonstrate that ternary {-1, 0, +1} inference is measurably faster, smaller, and more power-efficient than conventional FP16/INT8 inference on standard CPU hardware.

The core idea: all neural network weights are quantised to {-1, 0, +1}. Multiplication is eliminated — replaced with compare-and-add (negate, skip, or pass-through). Typical ternary models have 60-70% zero weights, enabling massive compute savings via zero-skip.

## Repository Structure

The workspace contains two zip archives that define the full codebase:

- **`tern-core-stage1a.zip`** — Active implementation (Python + PyTorch). The CPU reference implementation of the ternary execution engine.
- **`synapticode-github-scaffold.zip`** — Scaffold for 9 GitHub repos under the `synapticode` org.

The `Synapticode_GitHub_Setup_Guide.md` describes how to create the GitHub org and push all repos.

## CNS Synaptic Stack (Bottom to Top)

```
tern-core       → Single-NPU ternary execution engine (ACTIVE — Stage 1A)
tern-compiler   → 5-stage compiler: FP16/INT8 → .tern-model (scaffold only)
tern-runtime    → Multi-NPU orchestration + deterministic scheduling (scaffold only)
tern-governance → Explainability, audit trail, regulatory compliance (scaffold only)
cns-edge        → Edge network, device clustering, .tern-model distribution (scaffold only)
tern-sdk        → Public developer API (Apache 2.0, scaffold only)
cubie-ui        → CUBIE dice-style 3D interface (scaffold only)
```

## Architecture Decisions

- **Python + C extension** (ctypes or pybind11) for performance-critical paths
- **SIMD acceleration**: AVX2/AVX-512 on x86, NEON on ARM
- **Quantisation**: post-training with learned thresholds (STE gradients)
- **Target model**: TinyLlama-1.1B (fallback: MNIST MLP)
- **Benchmark against**: PyTorch FP16, INT8 (via torch.quantization), INT4 (GPTQ)

## tern-core Development (Active Codebase)

### Setup

```bash
cd tern-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Build & Test

```bash
# Run all tests
pytest tests/ -v

# Run a single test class
pytest tests/test_stage1a.py::TestTernaryQuantizer -v

# Run a single test
pytest tests/test_stage1a.py::TestTernaryQuantizer::test_output_values -v

# Lint
ruff check src/
black --check src/

# Type check
mypy src/
```

### Key Configuration

- Python 3.11+, type hints mandatory
- C code: C11 standard, no undefined behaviour, bounds-checked
- Formatter: `black` (line-length 88)
- Linter: `ruff` (line-length 88, target py310)
- Test runner: `pytest` with `-v --tb=short` defaults (see `pyproject.toml`)
- Dependencies: `torch>=2.0.0`, `numpy>=1.24.0`
- Optional: `transformers`, `accelerate`, `sentencepiece` for HuggingFace model loading
- Benchmark results must be reproducible (fixed seeds, warm-up runs)

## tern-core Architecture

### Module Map

- **`arithmetic/quantizer.py`** — `TernaryQuantizer` maps FP weights to {-1, 0, +1} using adaptive threshold (Δ = threshold × mean(|W|)). `SensitivityAnalyzer` evaluates per-layer quantisation tolerance across multiple thresholds.
- **`arithmetic/linear.py`** — `TernaryLinear` and `TernaryConv2d` are drop-in replacements for `nn.Linear`/`nn.Conv2d`. Training uses straight-through estimator (STE) for gradient flow; eval mode uses cached ternary weights.
- **`engine/inference.py`** — `TernaryInferenceEngine` auto-converts a PyTorch model by replacing eligible Linear/Conv2d layers with ternary equivalents. Protects embeddings, LayerNorm, and LM head layers by default.
- **`sparse/__init__.py`** — 2-bit packing and sparsity bitmap generation for zero-skip. Supports both uint8 (4 weights/byte) and uint64 (32 trits/word) packing. Encoding: 01=+1, 10=-1, 00=0.
- **`memory/__init__.py`** — `profile_model_memory()` calculates compression stats for converted models.
- **`model_loader/__init__.py`** — `.tern-model` binary format: HEADER → JSON metadata → packed ternary weights → FP16 protected layers → SHA-256 checksum. `TernModelWriter`/`TernModelReader` for serialisation.

### Data Flow

```
PyTorch model (FP16)
  → TernaryInferenceEngine.convert()
    → SensitivityAnalyzer identifies precision-critical layers
    → nn.Linear → TernaryLinear (with per-layer threshold)
    → Embeddings/LayerNorm/LM head left in FP16
  → TernaryInferenceEngine.infer()
    → Deterministic forward pass (fixed seeds, no cudnn benchmark)
  → TernModelWriter.save()
    → Pack to 2-bit encoding + sparsity bitmap → .tern-model file
```

## Ternary Arithmetic Rules

- Weights ∈ {-1, 0, +1}. No other values.
- Multiply by +1 = pass-through (add). Multiply by -1 = negate (subtract). Multiply by 0 = skip entirely.
- MatMul becomes add/subtract/skip — no multiply instructions.
- Pack as 2-bit pairs: uint8 (4 weights/byte) for storage, uint64 (32 trits/word) for SIMD execution.
- Zero-weight sparsity (60-70% of trits are 0): skip entire blocks.
- Scaling factor α (mean |W| of non-zero entries) is stored per-layer, not per-weight.

## Critical Design Constraints

- **Determinism is non-negotiable**: same input + same model = bit-identical output. This is required for governance and audit (Patent 36).
- **Patent alignment**: every function documents which patent claim it implements. Preserve this convention.
- **Protected layers**: embeddings, LayerNorm/RMSNorm, and LM head are kept in FP16 by default — quantising them destroys model quality.
- **Threshold range**: quantiser threshold ∈ (0, 1), typical operating range 0.5–0.9, default 0.7. Higher threshold = more sparsity = more compression = less accuracy.

## Patent Alignment (Key Claims to Demonstrate)

- Patent 36: Biological neural mapping → ternary weight encoding
- Patent 37: Zero-weight clock-gating → sparsity-aware skip logic
- Patent 38: Configurable precision → dual-path ternary/INT8 execution
- Patent 39: Ternary-native memory → packed trit storage format
- Patent 40: Bandwidth optimisation → streaming ternary weight loader
- Patent 41: Compiler scheduling → execution plan generation

## Licence

tern-sdk is Apache 2.0. All other repositories are proprietary (Synapticode Co., Ltd.). Never commit model files (*.tern-model, *.pt, *.safetensors, etc.).
