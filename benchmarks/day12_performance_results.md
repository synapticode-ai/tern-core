# Day 12: Performance Scaling Results

Generated: 2026-02-25 21:01:04

## Configuration

- Generation tokens: 32
- Warmup runs: 1
- Measured runs: 3 (median)
- Quantisation threshold: 0.7
- Sensitivity analysis: disabled
- Determinism: do_sample=False (Patent 36)

## Causal Model Generation (tok/s)

| Model | Params | Seq Len | FP32 tok/s | Ternary tok/s | Packed tok/s | Tern/FP32 | Pack/FP32 |
|-------|--------|---------|-----------|--------------|-------------|-----------|-----------|
| DistilGPT-2 (82M) | 81.9M | 128 | 62.7 | 3.5 | 1.9 | 0.06x | 0.03x |
| DistilGPT-2 (82M) | 81.9M | 512 | 50.1 | 5.4 | 0.6 | 0.11x | 0.01x |
| GPT-2 (124M) | 124.4M | 128 | 37.0 | 1.7 | 0.9 | 0.05x | 0.03x |
| GPT-2 (124M) | 124.4M | 512 | 26.6 | 1.7 | 0.3 | 0.06x | 0.01x |
| GPT-2-medium (355M) | 354.8M | 128 | 13.4 | 0.5 | — | 0.03x | — |
| GPT-2-medium (355M) | 354.8M | 512 | 8.2 | 0.4 | — | 0.05x | — |
| TinyLlama-1.1B | 1100.0M | 128 | 5.8 | — | — | — | — |
| TinyLlama-1.1B | 1100.0M | 512 | 4.0 | — | — | — | — |
| TinyLlama-1.1B | 1100.0M | 1024 | 2.7 | — | — | — | — |

## Prefill Latency (ms)

| Model | Seq Len | FP32 | Ternary | Packed |
|-------|---------|------|---------|--------|
| DistilGPT-2 (82M) | 128 | 63 | 353 | 13168 |
| DistilGPT-2 (82M) | 512 | 214 | 599 | 52398 |
| GPT-2 (124M) | 128 | 95 | 687 | 26465 |
| GPT-2 (124M) | 512 | 374 | 1118 | 105936 |
| GPT-2-medium (355M) | 128 | 271 | 2531 | — |
| GPT-2-medium (355M) | 512 | 1036 | 3824 | — |
| TinyLlama-1.1B | 128 | 674 | — | — |
| TinyLlama-1.1B | 512 | 2427 | — | — |
| TinyLlama-1.1B | 1024 | 5088 | — | — |

## Memory Usage

| Model | Mode | Model MB | Peak RSS MB |
|-------|------|----------|-------------|
| DistilGPT-2 (82M) | FP32 | 318.5 | 808 |
| DistilGPT-2 (82M) | Ternary | 318.5 | 1159 |
| DistilGPT-2 (82M) | Packed | 171.7 | 1224 |
| GPT-2 (124M) | FP32 | 486.7 | 1224 |
| GPT-2 (124M) | Ternary | 486.7 | 1606 |
| GPT-2 (124M) | Packed | 193.1 | 1678 |
| GPT-2-medium (355M) | FP32 | 1377.5 | 2126 |
| GPT-2-medium (355M) | Ternary | 1377.5 | 3404 |
| TinyLlama-1.1B | FP32 | 4196.4 | 4361 |
| BERT-base (110M) | FP32 | 417.6 | 6674 |
| BERT-base (110M) | Ternary | 417.6 | 6674 |
| BERT-base (110M) | Packed | 122.0 | 6674 |

## Encoder Model Forward Latency (ms)

| Model | Seq Len | FP32 | Ternary | Packed |
|-------|---------|------|---------|--------|
| BERT-base | 128 | 52.6 | 293.6 | 27250.0 |
| BERT-base | 512 | 187.3 | 462.2 | 108295.2 |

## Key Findings

- **Fastest FP32**: DistilGPT-2 (82M) at seq_len=128: 62.7 tok/s
- **Fastest Ternary**: DistilGPT-2 (82M) at seq_len=512: 5.4 tok/s
- **Fastest Packed**: DistilGPT-2 (82M) at seq_len=128: 1.9 tok/s
- **Avg Ternary/FP32 ratio**: 0.06x (across 6 configs)
