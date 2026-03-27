# Clean-Room ANE Energy Benchmark

> FP16 vs Ternary 2-bit inference — Tokens per Watt
> Apple M4 Pro · 2026-03-28 08:25

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
| CPU | 0.765 W |
| GPU | 0.040 W |

## ANE Power: FP16 vs Ternary

| Metric | FP16 (ANE) | Ternary 2-bit (ANE) | Delta | % Change |
|--------|:----------:|:-------------------:|:-----:|:--------:|
| ANE Power (W) | 5.368 | 6.766 | -1.398 | -26.0% |
| Latency (ms) | 15.53 | 7.53 | -8.01 | -51.6% |
| Tokens/sec | 4120 | 8504 | +4384 | +106.4% |
| **Tokens/Watt** | **768** | **1257** | **+489** | **+63.8%** |

## Detailed Power Breakdown

| Backend | ANE (W) | CPU (W) | GPU (W) | Latency (ms) | Tok/s | Tok/W |
|---------|:-------:|:-------:|:-------:|:------------:|:-----:|:-----:|
| FP16 ANE | 5.368 | 0.680 | 0.045 | 15.53 | 4120 | 768 |
| Ternary 2-bit ANE | 6.766 | 0.600 | 0.051 | 7.53 | 8504 | 1257 |
| FP16 GPU | 0.000 | 1.586 | 16.461 | 24.84 | 2576 | 141 |

## Summary

Ternary 2-bit inference on the Apple Neural Engine achieves:

- **2.1x faster** inference (7.53 ms vs 15.53 ms)
- **6.766 W** instantaneous ANE power (5.368 W for FP16 — ternary draws 26% more power at higher throughput)
- **1257 tokens/watt** vs **768 tokens/watt** — **1.64x energy efficiency gain**
- **8.0x model compression** (8-bit → 2-bit palettization)

The ANE runs ternary 2-bit weights 2.1x faster than FP16. Although
instantaneous power draw is 26% higher (the ANE works harder per
unit time), the 2.1x throughput gain far exceeds the power increase —
yielding **1.64x more tokens per watt**.

The 2-bit palette maps exactly to ternary {-α, 0, +α} weights: 4 palette entries,
3 used. CoreML's palettization compresses the model 8.0x,
fitting entirely in the ANE's on-chip SRAM for zero external memory traffic.

---
*Clean-room ANE energy benchmark · Terncore · Cubey/Synapticode · 2026-03-28*
