# Day 15 Brief — TEA Template (Ternary Evaluation Artefact)

## Context
Days 1-14 complete. Block 4 continues. The codebase is clean, documented, and tested. Now build the standardised evaluation format that NPU vendors and Apple engineers receive when assessing a model's ternary viability.

**Hardware:** 2019 iMac i9-9900K, 16GB DDR4.
**M4 Pro Status:** Mac Mini M4 Pro 12/16 64GB 1TB ordered from Apple AU. Pickup Brisbane City store ~March 30, 2026. Actively seeking earlier availability.
**Location:** Brisbane, Queensland, Australia.
**Doctrine:** No new computation. Template + reference example from existing data.
**Sprint goal:** All documentation prepared to Apple Core Quality engineering standards — as if an Apple CoreML engineer opens this repo tomorrow for a 4-hour evaluation.

## The Standard

The TEA is what a Rebellions engineer, an Apple CoreML evaluator, or a KSGC reviewer opens when they ask: "Show me what ternary does to this specific model." It answers every question they'd have in a single, self-contained document. Professional. Reproducible. No marketing language.

## Today's Deliverable

### `docs/TEA_TEMPLATE.md` — The Template

A standardised report format with placeholder sections. Anyone can fill this in for any model by running the existing benchmark scripts.

### `docs/TEA_TinyLlama_1.1B.md` — Reference Example

The template filled in with TinyLlama-1.1B data from Days 1-12. This is the proof that the template works and the reference example for all future TEAs.

## TEA Template Structure

```markdown
# Ternary Evaluation Artefact: {Model Name}

**Date:** {date}
**tern-core version:** {git commit hash}
**Hardware:** {CPU, RAM, OS}

## 1. Model Card
- Model: {HuggingFace ID or description}
- Architecture: {decoder-only / encoder-only / encoder-decoder}
- Parameters: {total}
- Layers: {count, types}
- Max sequence length: {max_position_embeddings}
- Vocabulary size: {vocab_size}
- Original precision: {FP32 / FP16 / BF16}
- Original size on disk: {MB}

## 2. Conversion Summary
- Threshold: {value}
- Converted layers: {n} / {total}
- Protected layers: {list with reasons}
- Conversion time: {seconds}
- .tern-model file size: {MB}
- Compression ratio: {original / ternary}x

## 3. Weight Distribution
- Sparsity (zero fraction): {%}
- +1 fraction: {%}
- -1 fraction: {%}
- Distribution consistency: {stable across layers / variable}

## 4. Sensitivity Map
- Analysis method: {gradient-based / perturbation / loss-delta}
- Evaluation tokens: {count, source}
- Sensitivity taxonomy:
  - Catastrophic layers (>10x baseline): {list}
  - High sensitivity (2-10x): {list}
  - Moderate (1.1-2x): {list}
  - Tolerant (<1.1x): {list}
- Safe layer percentage: {%} at threshold {value}
- Architecture-specific patterns: {observations}

## 5. Performance (CPU Reference)
- FP32 baseline tok/s: {value} @ seq_len={n}
- Ternary tok/s: {value} @ seq_len={n}
- Packed tok/s: {value} @ seq_len={n}
- Prefill latency: {ms} @ seq_len={n}
- Per-token latency: {ms}
- Memory: FP32 {MB} / Ternary {MB} / Packed {MB}

## 6. Quality Impact
- FP32 baseline perplexity: {value} (WikiText-2, {n} tokens)
- Ternary perplexity (naive, no STE): {value}
- Perplexity ratio: {ternary / FP32}x
- STE training results (if available):
  - Steps: {n}, Optimizer: {type}, LR: {value}
  - Post-STE perplexity: {value}
  - Improvement: {ratio}x

## 7. .tern-model Format Details
- Format version: {1 or 2}
- Header: {size} bytes
- Manifest: {size} bytes, {field count} fields per layer
- Alignment: {bytes}
- Footer: CRC32 {hex value}
- Round-trip verified: {yes/no}

## 8. Recommended Configuration
- Suggested threshold: {value}
- Layers to protect: {pattern list}
- STE training recommended: {yes/no, with rationale}
- Estimated STE steps for <5% PPL degradation: {range}
- Target deployment: {edge / server / specific hardware}

## 9. Reproducibility
Commands to reproduce every result in this document:
{exact commands with versions}

## 10. Appendices
- A. Per-layer sensitivity scores (full table)
- B. Weight distribution histograms (data, not images)
- C. Conversion log excerpt
```

## TinyLlama Reference TEA — Data Sources

All data already exists. No new computation needed.

| TEA Section | Source |
|-------------|--------|
| 1. Model Card | Day 1 model loading, HuggingFace model card |
| 2. Conversion Summary | Day 10 tern-convert output, Day 11 multi-model table |
| 3. Weight Distribution | Day 3 (43-46% sparsity), Day 11 confirmation |
| 4. Sensitivity Map | Day 2 (155-layer analysis), Day 5 (taxonomy), RESULTS.md |
| 5. Performance | Day 12 scaling data (FP32 5.8 tok/s @128, timeouts for ternary/packed) |
| 6. Quality Impact | Day 3 (FP32 PPL 7.19, ternary 130K), Day 4 STE (1.7K after 500 steps) |
| 7. .tern-model Format | Day 6-7 format spec, Day 8 round-trip verification |
| 8. Recommended Config | Synthesise from Days 2-5 findings |
| 9. Reproducibility | Existing benchmark commands from RESULTS.md |
| 10. Appendices | Day 2 sensitivity scores, Day 3 weight stats |

## Implementation Order

1. **Write TEA_TEMPLATE.md** (20 min) — the blank template with placeholder markers
2. **Write TEA_TinyLlama_1.1B.md** (40 min) — fill every section from existing data
3. **Cross-reference** (10 min) — verify every number in the TEA matches RESULTS.md / EVIDENCE_PACKAGE.md
4. **Review** (10 min) — read as a Rebellions engineer. Would you trust this? Would you know what to do next?
5. **Commit** (5 min)

## Exit Criteria
- [ ] `docs/TEA_TEMPLATE.md` complete with all 10 sections
- [ ] `docs/TEA_TinyLlama_1.1B.md` complete with real data in every section
- [ ] Every number traceable to a specific day's results
- [ ] Reproducibility section has working commands
- [ ] Sensitivity map includes the full 155-layer taxonomy
- [ ] No "TBD", "coming soon", or empty sections
- [ ] 166+ tests still pass (no source changes expected)
- [ ] Commit pushed

## Time Budget
| Phase | Estimate |
|-------|----------|
| TEA_TEMPLATE.md | 20 min |
| TEA_TinyLlama_1.1B.md | 40 min |
| Cross-reference verification | 10 min |
| Final review | 10 min |
| Commit | 5 min |
| **Total** | **~1.5 hours** |

## What NOT To Do
- Do NOT run any benchmarks. All data exists.
- Do NOT generate charts or images. Tables and numbers only.
- Do NOT write marketing copy. Engineering document, factual tone.
- Do NOT include projections or estimates. Only measured results.
- Do NOT modify any source code. Documentation only.
- Do NOT create TEAs for other models today. TinyLlama is the reference. Others come post-M4-Pro when we have complete data (ternary/packed didn't timeout).

## What This Enables
- Day 16: Patent-Code Mapping references the same source files documented here
- KSGC application: TEA is an attachment showing rigorous evaluation methodology
- Rebellions outreach: TEA_TinyLlama is the first deliverable they receive
- Apple evaluation: TEA demonstrates systematic, reproducible engineering methodology
- Post-M4-Pro: Generate TEAs for GPT-2, BERT, LLaMA-7B with complete performance data
- Partner conversations: "Here's our standardised evaluation for your model" — hand them a TEA
