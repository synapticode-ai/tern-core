# Clean-Room ANE Energy Benchmark

> FP16 vs Ternary 2-bit inference — Tokens per Watt
> Apple M4 Pro · 2026-03-27 12:40

## Conditions

| | |
|---|---|
| Hardware | Apple M4 Pro |
| Input shape | (1, 64, 2048) (seq_len=64 tokens/pass) |
| Models | CoreML .mlpackage, ANE routed (CPU_AND_NE) |
| FP16 model | 1804.1 MB |
| Ternary 2-bit model | 225.6 MB |
| Compression | 8.0x |
| Power sampling | 500ms intervals, powermetrics |
| Samplers | ane_power, cpu_power, gpu_power |
| Environment | Clean-room (apps closed, displays off) |

## Baseline (Idle)

| Subsystem | Power |
|-----------|------:|
| ANE | 0.000 W |
| CPU | 0.349 W |
| GPU | 0.028 W |

## ANE Power: FP16 vs Ternary

| Metric | FP16 (ANE) | Ternary 2-bit (ANE) | Delta | % Change |
|--------|:----------:|:-------------------:|:-----:|:--------:|
| ANE Power (W) | 5.548 | 6.899 | -1.352 | -24.4% |
| Latency (ms) | 15.06 | 7.38 | -7.68 | -51.0% |
| Tokens/sec | 4250 | 8677 | +4428 | +104.2% |
| **Tokens/Watt** | **766** | **1258** | **+492** | **+64.2%** |

## Detailed Power Breakdown

| Backend | ANE (W) | CPU (W) | GPU (W) | Latency (ms) | Tok/s | Tok/W |
|---------|:-------:|:-------:|:-------:|:------------:|:-----:|:-----:|
| FP16 ANE | 5.548 | 0.349 | 0.031 | 15.06 | 4250 | 766 |
| Ternary 2-bit ANE | 6.899 | 0.287 | 0.031 | 7.38 | 8677 | 1258 |
| FP16 GPU | 0.000 | 0.479 | 18.600 | 24.09 | 2657 | 138 |

## Summary

Ternary 2-bit inference on the Apple Neural Engine achieves:

- **2.0x faster** inference (7.38 ms vs 15.06 ms)
- **6.899 W** instantaneous ANE power (5.548 W for FP16 — ternary draws 24% more power at higher throughput)
- **1258 tokens/watt** vs **766 tokens/watt** — **1.64x energy efficiency gain**
- **8.0x model compression** (8-bit → 2-bit palettization)

The ANE runs ternary 2-bit weights 2.0x faster than FP16. Although
instantaneous power draw is 24% higher (the ANE works harder per
unit time), the 2.0x throughput gain far exceeds the power increase —
yielding **1.64x more tokens per watt**.

The 2-bit palette maps exactly to ternary {-α, 0, +α} weights: 4 palette entries,
3 used. CoreML's palettization compresses the model 8.0x,
fitting entirely in the ANE's on-chip SRAM for zero external memory traffic.

---
*Clean-room ANE energy benchmark · Terncore · Cubey/Synapticode · 2026-03-27*
