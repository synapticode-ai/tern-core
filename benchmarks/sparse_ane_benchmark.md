# Sparse Channel-Pruned ANE Benchmark

> Structured channel pruning for faster ternary inference on ANE
> Apple M4 Pro · 2026-03-27 13:29

## Method

The ANE executes dense matrix multiplications — it cannot skip individual zero
weights. To exploit ternary sparsity (7.30 ms dense baseline),
we use **structured channel pruning**:

1. Quantize to ternary {-α, 0, +α} (43% weight sparsity)
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
| Hardware | Apple M4 Pro |
| Blocks | 22 |
| Input | (1, 64, 2048) (seq=64 tokens) |
| Warmup | 10 |
| Measured runs | 50 |
| Compute units | CPU_AND_NE (ANE) |
| Baseline | Dense ternary 2-bit = 7.30 ms |

## Results

| Config | Latency (ms) | Min (ms) | Speedup | Model Size | Params |
|--------|:------------:|:--------:|:-------:|:----------:|:------:|
| Dense Ternary 2-bit (ANE) | 7.30 | 7.19 | 1.00x | 225.6 MB | 100% |
| Sparse 20% MLP / 10% attn (ANE) | 6.10 | 6.02 | 1.20x | 184.9 MB | 82% |
| Sparse 30% MLP / 20% attn (ANE) | 5.30 | 5.23 | 1.38x | 162.4 MB | 73% |
| Sparse 40% MLP / 30% attn (ANE) | 5.40 | 4.69 | 1.35x | 139.8 MB | 63% |
| Sparse 50% MLP / 40% attn (ANE) | 4.11 | 4.03 | 1.78x | 117.2 MB | 53% |

## Best Configuration

**Sparse 50% MLP / 40% attn (ANE)** achieves the best latency:

- **4.11 ms** vs **7.30 ms** dense baseline
- **1.78x speedup** from channel pruning
- **117.2 MB** model size (vs 225.6 MB dense)
- **46.9% parameter reduction**

## Tokens per Second

| Config | Tok/s | vs Dense |
|--------|:-----:|:--------:|
| Dense Ternary 2-bit (ANE) | 8769 | 1.00x |
| Sparse 20% MLP / 10% attn (ANE) | 10493 | 1.20x |
| Sparse 30% MLP / 20% attn (ANE) | 12066 | 1.38x |
| Sparse 40% MLP / 30% attn (ANE) | 11855 | 1.35x |
| Sparse 50% MLP / 40% attn (ANE) | 15581 | 1.78x |

## Analysis

Structured channel pruning converts unstructured ternary sparsity (where 43%
of individual weights are zero but no full channels are zero) into structured
sparsity by removing the least-important channels entirely. The ANE then
processes physically smaller weight matrices, reducing both latency and energy.

The MLP layers (gate/up/down_proj at 5632 intermediate dim) account for ~70%
of total FLOPs per block, making them the highest-impact pruning target.

---
*Sparse channel-pruned ANE benchmark · Terncore · Cubey/Synapticode · 2026-03-27*
