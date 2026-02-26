# Ternary Evaluation Artefact: {Model Name}

**Date:** {YYYY-MM-DD}
**tern-core version:** {git commit hash}
**Hardware:** {CPU, RAM, OS}

---

## 1. Model Card

| Field | Value |
|-------|-------|
| Model | {HuggingFace ID or description} |
| Architecture | {decoder-only / encoder-only / encoder-decoder} |
| Parameters | {total, e.g. 1,100M} |
| Layers | {count total, count eligible, layer types} |
| Max sequence length | {max_position_embeddings} |
| Vocabulary size | {vocab_size} |
| Original precision | {FP32 / FP16 / BF16} |
| Original size on disk | {MB, FP32} |

## 2. Conversion Summary

| Field | Value |
|-------|-------|
| Threshold | {value} |
| Converted layers | {n} / {total eligible} |
| Protected layers | {n} |
| Conversion time | {seconds} |
| .tern-model file size | {MB} |
| Compression ratio | {original / ternary}x |

**Protected layers:**

| Layer | Reason |
|-------|--------|
| {layer_name} | {e.g. embedding, LayerNorm, lm_head} |

## 3. Weight Distribution

| Metric | Value |
|--------|-------|
| Sparsity (zero fraction) | {%} |
| +1 fraction | {%} |
| -1 fraction | {%} |
| Distribution consistency | {stable across layers / variable} |

**Per-type breakdown (threshold={value}):**

| Type | Count | Sparsity | Quant Error | Kurtosis |
|------|-------|----------|-------------|----------|
| {layer_type} | {n} | {%} | {value} | {value} |

## 4. Sensitivity Map

| Field | Value |
|-------|-------|
| Analysis method | {perturbation: per-layer ternary with full-model PPL evaluation} |
| Evaluation tokens | {count, source dataset} |
| Total analysis time | {seconds} |

**Sensitivity taxonomy:**

| Category | Criterion | Count | Percentage | Layers |
|----------|-----------|-------|------------|--------|
| Catastrophic | >10x baseline | {n} | {%} | {list} |
| High | 2-10x baseline | {n} | {%} | {list} |
| Moderate | 1.1-2x baseline | {n} | {%} | {list} |
| Tolerant | <1.1x baseline | {n} | {%} | {count too many to list} |

**Architecture-specific patterns:**
- {observation 1}
- {observation 2}

## 5. Performance (CPU Reference)

| Metric | FP32 | Ternary | Packed |
|--------|------|---------|--------|
| tok/s @ seq_len={n} | {value} | {value} | {value} |
| Prefill latency (ms) @ seq_len={n} | {value} | {value} | {value} |
| Per-token latency (ms) | {value} | {value} | {value} |
| Model memory (MB) | {value} | {value} | {value} |

**Microbenchmark (isolated matmul, C+SIMD vs BLAS):**

| Size | BLAS (us) | C+SIMD (us) | Speedup |
|------|-----------|-------------|---------|
| {NxN} | {value} | {value} | {value}x |

## 6. Quality Impact

| Metric | Value |
|--------|-------|
| FP32 baseline perplexity | {value} ({dataset}, {n} tokens) |
| Ternary perplexity (all layers, no STE) | {value} |
| Perplexity ratio | {ternary / FP32}x |

**Mixed-precision results:**

| Config | Ternary Layers | PPL | Gap vs FP32 |
|--------|---------------|-----|-------------|
| {config_name} | {n} | {value} | {%} |

**STE training results (if available):**

| Field | Value |
|-------|-------|
| Steps | {n} |
| Optimizer | {type}, LR={value} |
| Pre-STE perplexity | {value} |
| Post-STE perplexity | {value} |
| Improvement | {ratio}x |

## 7. .tern-model Format Details

| Field | Value |
|-------|-------|
| Format version | {1 or 2} |
| Header size | {bytes} |
| Manifest size | {bytes}, {field count} fields per layer |
| Weight alignment | {bytes} |
| Footer | CRC32 {hex value}, file_size, reverse magic |
| Round-trip verified | {yes / no} |
| Round-trip max diff | {value} |

## 8. Recommended Configuration

| Field | Value |
|-------|-------|
| Suggested threshold | {value} |
| Layers to protect | {pattern list or "all except v_proj late N"} |
| STE training recommended | {yes / no} |
| Rationale | {1-2 sentences} |
| Estimated STE steps for <5% PPL degradation | {range or "not yet determined"} |
| Target deployment | {edge / server / specific hardware} |

## 9. Reproducibility

All results in this document can be reproduced with the following commands on the specified hardware.

```bash
# Environment setup
git clone https://github.com/synapticode/tern-core.git
cd tern-core
git checkout {commit_hash}
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,transformers]"

# Build C library
cd src/terncore/csrc && make clean && make && cd ../../..

# Conversion (Section 2)
{command}

# Perplexity evaluation (Section 6)
{command}

# Sensitivity analysis (Section 4)
{command}

# Performance benchmark (Section 5)
{command}

# STE training (Section 6, if applicable)
{command}

# Weight analysis (Section 3)
{command}
```

## 10. Appendices

### A. Per-Layer Sensitivity Scores

Full table of all {n} layers, sorted by sensitivity ratio (highest first).

| Rank | Layer | PPL | Delta | Ratio | Params | Sparsity |
|------|-------|-----|-------|-------|--------|----------|
| {1} | {layer_name} | {value} | {value} | {value}x | {value} | {%} |

### B. Weight Distribution Data

Per-type weight statistics at threshold={value}.

| Type | Count | Mean |W| | Std | Sparsity | Quant Error | Kurtosis | Eff Rank |
|------|-------|---------|-----|----------|-------------|----------|----------|
| {type} | {n} | {value} | {value} | {%} | {value} | {value} | {value} |

### C. Conversion Log Excerpt

```
{tern-convert output showing layer-by-layer progress}
```

---

*Generated by tern-core. Synapticode Co., Ltd.*
