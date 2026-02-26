# tern-core

**Ternary execution engine for neural network inference** — CNS Synaptic by Synapticode Co., Ltd.

## What This Is

A CPU reference implementation that converts standard neural network models to ternary {-1, 0, +1} weights. All multiplication is replaced with compare-and-add. Weights are packed to 2 bits each (16x vs FP32), zero-weights are skipped via sparsity bitmaps, and output is deterministic (bit-identical across runs).

Five architectures validated: TinyLlama-1.1B, GPT-2, GPT-2-medium, BERT-base, DistilGPT-2.

## Quick Start

```bash
# Clone and install
git clone https://github.com/synapticode/tern-core.git
cd tern-core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,transformers]"

# Convert DistilGPT-2 to ternary (~60s, downloads ~330MB model)
tern-convert distilgpt2 -o /tmp/distilgpt2.tern --verify

# Inspect the output
python tools/tern_loader.py --info /tmp/distilgpt2.tern
```

## What It Does

Every floating-point weight in a neural network is mapped to one of three values: **+1** (add), **-1** (subtract), or **0** (skip). This eliminates all multiply-accumulate operations in favour of simple compare-and-add arithmetic. A per-layer scaling factor (alpha) preserves magnitude.

Ternary models at threshold 0.7 have ~43-46% zero weights. These are identified by a sparsity bitmap and skipped entirely during inference. Weights are packed 4 per byte using a 2-bit encoding (01=+1, 10=-1, 00=0), achieving up to 8.4x file compression for TinyLlama (4,137 MB FP32 to 471.6 MB .tern-model).

The `.tern-model` v2 format stores a 256-byte header, JSON manifest with per-layer offsets, 32-byte-aligned weight data, and a CRC32 footer. Round-trip serialisation is bit-identical.

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+, NumPy 1.24+

```bash
# From source (recommended)
pip install -e ".[dev]"

# With HuggingFace model support
pip install -e ".[dev,transformers]"
```

The C acceleration library is built automatically on first import (requires a C compiler). On macOS, install `brew install libomp` for OpenMP support.

## Usage

### Convert a HuggingFace model

```bash
# Convert with default threshold (0.7)
tern-convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 -o model.tern

# Higher threshold = more compression, less accuracy
tern-convert distilgpt2 -o model.tern -t 0.8 --verify

# Show model info without converting
tern-convert distilgpt2 -o /dev/null --info
```

### Load and inspect a .tern-model

```bash
# File info (header, manifest, layer summary)
python tools/tern_loader.py --info model.tern

# Verify integrity (CRC32 check)
python tools/tern_loader.py --verify model.tern

# Run inference
python tools/tern_infer.py --model model.tern --prompt "Hello world"
```

### Programmatic API

```python
from terncore import TernaryQuantizer, TernaryLinear, TernaryInferenceEngine

# Quantise a weight tensor
q = TernaryQuantizer(threshold=0.7)
ternary, alpha = q.quantize(model.layer.weight.data)

# Convert a full model
engine = TernaryInferenceEngine()
report = engine.convert(model)  # replaces eligible Linear layers

# Inference with deterministic output
result = engine.infer(model, input_tensor)
```

## Architecture

```
src/terncore/
  __init__.py              Package entry point, public API
  arithmetic/
    quantizer.py           TernaryQuantizer, SensitivityAnalyzer
    linear.py              TernaryLinear, TernaryConv2d (drop-in replacements)
  engine/
    inference.py           TernaryInferenceEngine (auto-conversion + inference)
  sparse/
    __init__.py            Sparsity bitmap, 2-bit packing, zero-skip
  memory/
    __init__.py            Model memory profiling
  accel/
    __init__.py            C kernel acceleration (AVX2/NEON, OpenMP)
  model_loader/
    __init__.py            .tern-model v1 format (legacy)
  tern_model.py            .tern-model v2 format (TernModelWriter/Reader)
  packed_linear.py         PackedTernaryLinear (2-bit packed storage)
  ste.py                   Straight-through estimator for QAT
  ste_trainer.py           QAT trainer (gradient checkpointing, SGD)
  convert.py               End-to-end conversion pipeline + CLI
  hf_loader/
    __init__.py            HuggingFace model loader
  csrc/
    ternary_matmul.c       Scalar ternary matmul kernel
    ternary_avx2.c         AVX2 SIMD kernel
    ternary_neon.c         NEON SIMD kernel
    sparse_skip.c          Zero-skip kernel with CTZ bit-scan
    torch_bindings.cpp     PyTorch C++ extension bindings
```

## Benchmarks

Full results in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) and [`benchmarks/EVIDENCE_PACKAGE.md`](benchmarks/EVIDENCE_PACKAGE.md).

**Headline numbers:**

| Metric | Value |
|--------|-------|
| Architectures validated | 5 (TinyLlama, GPT-2, GPT-2-medium, BERT, DistilGPT-2) |
| TinyLlama compression | 8.4x (4,137 MB to 471.6 MB) |
| BERT compression | 13.5x (all 73 layers ternary) |
| SIMD speedup (AVX2+OpenMP) | 2.45x over BLAS at 2048x2048 |
| Zero-skip speedup | 5.28x at 90% sparsity |
| Deterministic output | Bit-identical across 100 runs |
| Test suite | 166 Python + 53 C tests passing |

## Testing

```bash
# All tests
pytest tests/ -v

# C kernel tests
cd src/terncore/csrc && make test

# Single test file
pytest tests/test_stage1a.py -v
```

## Patent Notice

This software implements technology described in the Synapticode patent portfolio (56 patents).
Inventor: Robert Lakelin. Key patents: 1 (ternary encoding), 36 (biological neural mapping),
37 (zero-skip), 38 (configurable precision), 39 (packed memory format).

## Licence

Proprietary. All rights reserved. Synapticode Co., Ltd.
