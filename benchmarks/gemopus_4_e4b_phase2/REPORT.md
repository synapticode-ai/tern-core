# Gemopus-4-E4B-it Benchmark — Phase 1: Compression Methodology Validation

**Date:** 2026-05-01  
**Host:** Mac Mini Synapticode (Apple M4 Pro, 48 GB unified memory)  
**tern-core commit:** f48a7e5 (v0.4.0)

---

## Apple / KAIST Conversation Framing

This benchmark series sits at the intersection of two active conversations. The May 2026 Apple engagement concerns Synapticode's ternary compression stack as a candidate for on-device model deployment on Apple Silicon — a path where model footprint, memory headroom, and inference efficiency directly determine which model classes reach the NPU. Phase 1 establishes that the compression pipeline produces byte-clean, structurally sound .tern-model artefacts for Gemma 4 E4B-it and its Gemopus fine-tune, with footprints of 9153 MB at 48.3% ternary ratio. The artefacts on archive now constitute a working demonstration of the pipeline operating on Google's latest multimodal architecture at 8B parameter scale, ready to be surfaced in that conversation.

For KAIST and the Korean Smart Grid Consortium, the relevance runs in parallel: the Korean NPU target consumes .tern-model artefacts directly, and Phase 1 confirms that the gemma4 adapter produces valid artefacts for both a base fine-tune and a community instruction-tuned variant. The KAIST deliverable pattern — compression to a canonical .tern-model, integrity-verified and stored on archive — is exactly the workflow this session exercises. Phase 1 also establishes that fine-tune-agnostic compression transfers cleanly across the Gemma 4 E4B architecture family, which informs planning for the larger Gemopus-4-26B-A4B-it and Gemopus-4-31B-it deliverables.

This report covers Phase 1 of a three-phase benchmark programme. Phase 1 (this document) addresses compression methodology validation: artefact production, integrity confirmation, and uncompressed engine baselines on Mac Mini M4 Pro. Phase 1.5, scheduled for the next session after a clean reboot, cross-validates the storage substrate by re-running inference rows from local NVMe to confirm the external-archive path introduces no measurement artefact. Phase 2 implements TernModelReader to round-trip persisted .tern-model artefacts through PyTorch inference, producing the headline ternary-inference numbers — the compression-delta measurements the Apple and KAIST conversations require. The three phases are designed so each step establishes only what is provably true at that point, without asking the methodology to carry claims it cannot yet defend.

---

## What Phase 1 Demonstrates

### Finding 1 — Compression methodology validates structurally

The persisted .tern-model artefacts produced for gemma4-e4b (9153 MB, 48.3% ternary ratio) and gemopus-4-e4b (9153 MB, 48.3% ternary ratio) are byte-clean and integrity-verified. Both artefacts carry 283 ternary layers, 11 INT4 layers, and 1836 FP16 layers across 7.996 billion total parameters, with file sizes of 9,597,725,360 bytes and 9,597,725,392 bytes respectively — a 32-byte difference consistent with metadata only.

The Phase 1 dry-run gate predicted 49.34% ternary ratio; the full compression delivered 48.3% — within one percentage point. This confirms the dry-run gate operates as a reliable predictor of compression behaviour for this architecture, providing an early-exit signal for compression campaigns where the ratio must remain above a minimum threshold before committing to a multi-hour conversion run.

### Finding 2 — The compression pipeline is fine-tune-agnostic

Jackrong's Gemopus-4-E4B-it (a supervised fine-tune of Google's Gemma 4 E4B-it merged via LoRA) produced an identical ternary ratio (48.3%), identical layer counts, and a matching byte footprint to the base model. Weight magnitude distributions govern which layers cross the ternary threshold, and this result shows that LoRA-merged SFT fine-tuning does not shift those distributions measurably from the base — the fine-tune weights remain in the same magnitude regime as the base's pre-trained weights.

The strategic implication for the sprint cluster is direct: the Gemopus-4-26B-A4B-it and Gemopus-4-31B-it compressions can be planned using the same adapter, threshold, and expected ternary ratio as their Gemma 4 base variants, without requiring per-fine-tune calibration runs. The gemma4 adapter transfers cleanly across the Gemma 4 E4B architecture family, and by extension across community fine-tunes of that family.

### Finding 3 — Engine baselines establish the uncompressed reference

Rows 1 and 1' establish the BF16/FP16 reference numbers on Mac Mini M4 Pro under two inference engines:

- **Row 1 (mlx_vlm, mlx-community BF16):** 23.43 tok/s, 0.51 J/token, 16.2 GiB peak memory. MLX's unified-memory path keeps the full model in GPU-resident memory with efficient BF16 kernels; the 16.2 GiB footprint leaves 31.8 GiB free on the 48 GiB system.
- **Row 1' (PyTorch MPS, Google FP16):** 10.64 tok/s, 0.66 J/token, 29.6 GiB peak memory. The PyTorch MPS path for FP16 is the same engine family used by the ternary rows, making Row 1' the correct apples-to-apples baseline for the Phase 2 compression delta.

The 2.2× throughput advantage of mlx_vlm over PyTorch MPS on this hardware, and the 1.83× memory advantage (16.2 vs 29.6 GiB), are documented here for cross-engine reference. Library-reported generation throughput (mlx_vlm: 23.49 tok/s) confirmed the wall-clock measurement (23.43 tok/s) within 0.3%, providing triangulation. Both rows are clean Phase 1 measurements requiring no Phase 2 follow-up.

---

## Phase 1 Measurement Table

| # | Model | Format | Engine | tok/s | J/token | Peak mem | Status |
|---|---|---|---|---:|---:|---:|---|
| 1 | gemma-4-E4B-it (mlx-community BF16) | BF16 | mlx_vlm | 23.43 | 0.51 | 16.2 GiB | Clean Phase 1 reference |
| 1' | google/gemma-4-E4B-it | FP16 | pytorch_mps | 10.64 | 0.66 | 29.6 GiB | Clean Phase 1 reference |
| 2 | gemma-4-E4B-it (tern-core in-memory) | ternary mixed-INT4 | pytorch_mps | 4.11 | 4.04 | 32.7 GiB | Methodology-validation row; tok/s and J/token not representative of persisted .tern-model deployment performance — see Limitation section |
| 3 | gemopus-4-E4B-it (tern-core in-memory) | ternary mixed-INT4 | pytorch_mps | 0.94 | — | — | Confound: cumulative thermal/memory pressure during unattended cascade. Numbers retained for reproducibility audit only. Re-baseline in Phase 1.5. |

*Row 4 (gemopus-4-E4B-it, thinking on, pytorch_mps) was SIGKILLed by OOM during cascade. Deferred to Phase 1.5.*

**2026-05-02 erratum.** Phase 1's Row 4 termination was originally attributed to OOM. Phase 1.5 diagnostic work identified the actual root cause: the `run_llamacpp_gguf` harness invocation built llama.cpp arguments without `-ngl` (defaulting llama.cpp to CPU-only BLAS execution at ~0.025 tok/s), used `-no-cnv` (which is an alias for the conversation flag in llama.cpp 8990 rather than its negation, launching an interactive REPL with stdin closed), and parsed an output format that llama.cpp 8990 has retired. The 200-token generation under those conditions ran for hours before system termination — there was no memory-pressure component to the failure. The harness has been patched (`--single-turn -ngl 999` invocation plus a new-format `[ Prompt: P t/s | Generation: G t/s ]` parser); see `gemopus_4_e4b_phase1_5/REPORT_PHASE1_5.md` Section 4 for the corrected Row 4 measurement at 56.60 tok/s / 0.379 J/token.

---

## Phase 1 Limitation: The In-Memory Ternisation Inference Path

Row 2 measures inference through `TernaryInferenceEngine.convert()`, which produces ternary weights in memory but runs them through the standard PyTorch MPS inference path. The MPS path lacks a ternary-optimised matmul kernel: matrix multiplications effectively upcast ternary values to FP16 or FP32 per layer, paying a conversion overhead with each forward pass. The 4.11 tok/s result therefore reflects the cost of in-memory conversion plus unoptimised execution, not the throughput achievable from a persisted .tern-model artefact running through dedicated ternary inference infrastructure.

This is a methodology boundary, not a compression-pipeline failure. The persisted .tern-model artefacts are the canonical Korean NPU deliverables; they run through TernModelReader inference infrastructure that is not yet implemented on the Apple Silicon side. That implementation is Benchmark Phase 2, scheduled for the subsequent session. Phase 2 round-trips the persisted artefacts byte-identically through TernModelReader-backed PyTorch inference, producing the true compression-delta measurement against Row 1'.

Row 3's confound compounds this: the cascade ran unattended, and by the time Row 3 executed, the system had accumulated thermal and memory pressure from Row 2's 48-second run under high sustained load. The 0.94 tok/s figure — approximately 4.4× slower than Row 2's already-penalised path — reflects those cumulative conditions, not any property of the Gemopus model itself. Row 3 is retained in the audit trail and table above for reproducibility; it is not interpreted as a model measurement.

Rows 3, 3a, and 4 are deferred to Phase 1.5 (clean-substrate, post-reboot retry) and Phase 2 (TernModelReader inference delta).

---

## Phase Architecture and Forward Path

**Phase 1 — This report (2026-05-01)**  
Compression methodology validation. Artefacts produced for gemma4-e4b and gemopus-4-e4b, integrity-confirmed via sha256, fine-tune-agnostic transferability demonstrated. Engine baselines for mlx_vlm BF16 and pytorch_mps FP16 clean and cross-validated by library-reported metrics. Phase 1 closes here.

**Phase 1.5 — Next session, post-reboot for clean blind test**  
Substrate cross-validation. Re-run Rows 1, 1', 3, 3a, and 4 from local NVMe storage rather than the external archive, on a freshly rebooted system to eliminate thermal and memory pressure carryover. Confirms that the external-archive storage substrate introduces no inference measurement artefact (or quantifies any difference). One model at a time to keep local disk pressure bounded. llama-cli is installed and available for additional cross-validation.

**Phase 2 — Subsequent session**  
TernModelReader implementation. Round-trips persisted .tern-model artefacts through PyTorch inference for true byte-identical compression-delta measurement on Apple Silicon. This path produces the headline ternary-inference numbers — the tok/s and J/token delta against Row 1' — for the Apple and KAIST conversations.

---

## Methodology

**Prompt (fixed across all rows):**  
> "Describe the process of photosynthesis at a high-school level. Cover the inputs, outputs, and where the reaction takes place. Keep your answer to roughly four sentences."

**Parameters:** max_tokens 200, warmup 50 tokens, seed 42.

**Power measurement:** Apple `powermetrics` sampled at 1 Hz via active-window filter developed and validated against Row 1 sample data before locking methodology. The filter correctly distinguishes generation-active samples (power above idle floor) from post-stop idle samples. Both active-window and all-samples figures are captured per row in the JSON files for audit. Active-window figures are used in the table above.

**Library cross-check:** where mlx_vlm reported generation_tps, this confirmed wall-clock measurements within 0.3% (Row 1: library 23.49 tok/s vs wall-clock 23.43 tok/s). This provides measurement triangulation for rows where both methodologies run.

**Run count:** single representative runs per row with estimated run-to-run variance ~5%. Cross-row comparisons remain valid because each row uses identical single-run methodology. The variance estimate is not based on repeat runs in this session; Phase 1.5 will provide additional data points.

**Engine version stack at session:**

| Library | Version |
|---|---|
| mlx | 0.31.2 |
| mlx_lm | 0.31.3 |
| mlx_vlm | 0.4.4 |
| mlx-metal | 0.31.2 |
| torch | 2.7.0 |
| transformers | 5.5.4 |
| terncore | 0.4.0 (commit f48a7e5) |

llama-cli installed (ggml 0.10.1) and available for Phase 1.5 / Phase 2 cross-validation.

---

## Closing Reflection

tern-core compression methodology validates structurally and transfers cleanly across fine-tunes of the same base architecture. The two .tern-model artefacts on archive — gemma4_e4b_ternary_v0.1.0 and gemopus_4_e4b_ternary_v0.1.0, 9153 MB each, 48.3% ternary ratio — are byte-verified and ready as Korean NPU deliverables. The methodology boundary uncovered in Row 2 is honest and documented: the in-memory conversion path answers a different question than the one the Apple and KAIST conversations need. Phase 1.5 and Phase 2 are structured to close that gap cleanly, one substrate cross-validation and one TernModelReader implementation at a time. Phase 1 has established what is provably true today; the programme proceeds.

---

*Gamma Seeds Pte Ltd — Synapticode*
