# Terncore CoreML / ANE Benchmark

> Ternary linear stack (22 blocks × 7 layers = 154 matmuls)
> Apple M4 Pro · 2026-03-27

## Configuration

| | |
|---|---|
| Architecture | 22-block linear stack matching TinyLlama 1.1B dimensions |
| Input shape | (1, 64, 2048) — batch=1, seq=64, hidden=2048 |
| Ternary sparsity | 35.0% |
| Benchmark runs | 50 |

## Results

| Backend | Mean (ms) | Min (ms) | vs MPS FP16 |
|---------|:---------:|:--------:|:-----------:|
| PyTorch MPS FP16 | 26.99 | 26.10 | 1.00x |
| CoreML FP16 (ALL) | 15.16 | 14.81 | 1.78x |
| CoreML FP16 (CPU_AND_NE) | 15.14 | 14.88 | 1.78x |
| CoreML FP16 (CPU_AND_GPU) | 24.78 | 23.77 | 1.09x |
| PyTorch MPS Ternary | 27.00 | 25.89 | 1.00x |
| CoreML Ternary-FP16 (ALL) | 15.11 | 14.80 | 1.79x |
| CoreML Ternary-2bit (ALL) | 7.29 | 7.18 | 3.70x |
| CoreML Ternary-2bit (CPU_AND_NE) | 7.50 | 7.19 | 3.60x |
| CoreML Ternary-2bit (CPU_AND_GPU) | 33.35 | 31.67 | 0.81x |

## Model Sizes

| Format | Size |
|--------|-----:|
| FP16 CoreML | 1804.1 MB |
| Ternary FP16 CoreML | 1804.1 MB |
| Ternary 2-bit CoreML | 225.6 MB |

## Energy Consumption

| Backend | Package Power | Energy/Inference | Inferences |
|:--------|:------------:|:----------------:|:----------:|

## ANE Analysis

CoreML routes 2-bit palettized matmuls to the Apple Neural Engine when
`ComputeUnit.CPU_AND_NE` or `ComputeUnit.ALL` is selected. The ANE operates
at fixed power with dedicated matrix multiply hardware — ideal for sustained
inference at low energy.

The 2-bit palette maps perfectly to ternary {-α, 0, +α} weights:
4 palette entries, 3 used, with ~35% zero weights.

---
*Terncore CoreML/ANE benchmark · Cubey/Synapticode · 2026-03-27*
