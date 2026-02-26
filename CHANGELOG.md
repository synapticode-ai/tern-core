# Changelog

All notable changes to tern-core during the 20-day sprint.

## [0.4.0] - 2026-02-26 (Block 4: Documentation & Polish)

### Added
- Day 17: CI pipeline, changelog, complete project metadata, version tag
- Day 16: Patent-code mapping (14 patents linked to source, tests, and evidence)
- Day 15: TEA template and TinyLlama-1.1B reference evaluation
- Day 14: README rewrite, Google-style docstrings, `tern-convert` CLI entry point
- Day 13: Evidence consolidation (EVIDENCE_PACKAGE.md)

## [0.3.0] - 2026-02-25 (Block 3: Generalisation & Scaling)

### Added
- Day 12: Performance scaling curve (4 causal models + BERT, tok/s across sequence lengths)
- Day 11: Multi-model generalisation (5 architectures, Conv1D support for GPT-2 family)

## [0.2.0] - 2026-02-25 (Block 2: Format & Pipeline)

### Added
- Day 10: `tern-convert` CLI pipeline (TinyLlama 471.6 MB, 8.4x compression)
- Day 9: Cached sparsity bitmap and zero-skip benchmark (2.07x caching, 5.28x at 90%)
- Day 8: PackedTernaryLinear with 2-bit packed weight storage (16x compression)
- Day 7: TernModelReader, lazy loading API, bit-identical round-trip validation
- Day 6: .tern-model v2 format specification and TernModelWriter

## [0.1.0] - 2026-02-24 (Block 1: Core Engine)

### Added
- Day 5: Weight analysis, layer taxonomy, and gradient sensitivity tools
- Day 4: STE training proof-of-concept (PPL 77K -> 1.7K in 500 steps, 45.8x)
- Day 3: Mixed-precision ternary converter and compound error discovery
- Day 2: Per-layer sensitivity analysis (155 layers, 10,955s)
- Day 1: Perplexity benchmark framework (FP32: 7.19, all-ternary: 130,127)

## [0.0.1] - 2026-02-23 (Pre-Sprint: Foundation)

### Added
- Phase 4: Latency optimisation (torch C++ extension, OpenMP, AVX2 prefetch)
- Phase 3: HuggingFace TinyLlama-1.1B loader and end-to-end benchmark
- Stage 1B Phase 2: AVX2/NEON SIMD kernels, CPUID detection
- Stage 1B Phase 1: Scalar C kernels, ctypes bindings, Python accel wrapper
- Stage 1A: TernaryQuantizer, TernaryLinear, inference engine, sparse packing
- Initial scaffold (CNS Synaptic by Synapticode)
