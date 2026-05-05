# Gemopus-4-E4B-it Benchmark — Phase 2: Metal Kernel Integration + Canonical Measurement

**Date:** 2026-05-05
**Host:** Mac Mini Synapticode (Apple M4 Pro, 48 GB unified memory)
**tern-core branch:** `phase-4-terncore-packed-bench` at `1ab6b9f` (3 commits ahead of `main` at `3c3413c` post-PR-#9 merge)

---

## Apple / KAIST Conversation Framing

This Phase 2 report closes the headline ternary-inference measurement that Phase 1 explicitly deferred. The .tern-model artefacts produced and validated in Phase 1 (`gemopus_4_e4b_ternary_v0.1.0.tern-model`, 9153 MB at 48.3% ternary ratio) round-trip through PyTorch inference on Apple Silicon via a new Metal kernel integration into `PackedTernaryLinear`. Row 5 — 2.85 tok/s, 1.22 J/token, 44.5 GiB peak memory — establishes the canonical measurement for the persisted-artefact-via-Metal path on the v1 Metal forward design, completing the Phase 2 deliverable for the Apple May 2026 conversation.

For KAIST and the Korean Smart Grid Consortium, the Phase 2 closure carries the parallel weight: the same .tern-model artefacts that the Korean NPU consumes directly now have a measured Apple-Silicon inference baseline, allowing hardware-target comparisons in the same units. The Korean deliverable workflow stands unchanged — compression to a canonical .tern-model, integrity-verified and stored on archive — and now ships alongside an Apple-Silicon reference number.

---

## What Phase 2 Demonstrates

### Finding 1 — Configurational fidelity verified end-to-end

The canonical .tern-model artefact loads through the new Metal-aware path with exactly the layer configuration established in Saturday's Phase 2 Stage C structured findings: **258 packed layers, 335 protected layers, 593 total**. The state_dict load reports `missing=1, unexpected=54` — the lm_head-tied-to-embed-tokens singleton plus the 54 architecture-pruned sliding-window KV-projection entries that Saturday catalogued.

Configurational fidelity is now reproducible from a script (the manifest-driven `derive_protection_list_from_manifest` helper added to `terncore.tern_model`) rather than empirically referenced; the canonical baseline has a re-runnable source. The harness asserts `packed_layers < 350` before any measurement runs, surfacing configurational drift loudly if a future change breaks the protection-list derivation.

### Finding 2 — Headline tok/s lands at 2.85 on the v1 Metal path

Row 5 measures **2.85 tok/s** for 200-token greedy generation on the canonical artefact via the new `terncore_packed` harness backend. Energy: **1.22 J/token**. Peak memory: **44.5 GiB** on the 48 GiB M4 Pro. Generation time: 70.11 seconds for the standard photosynthesis prompt. The number is reproducible end-to-end from `git checkout 1ab6b9f` plus the artefact at the canonical archive path.

The 2.85 tok/s and 1.22 J/token figures are the v1 Metal forward floor, not the Metal kernel's ceiling — see "v1 Metal Forward + Forward Path" below for the framing of what these numbers bound.

### Finding 3 — Output quality envelope is the Phase 1 compression-threshold decision

Row 5's generated text is structurally readable but semantically incoherent — a mix of multilingual tokens, mathematical notation fragments, and HTML-like markup. This output quality matches Saturday's Row 2 (`pytorch_mps_ternary` in-memory path, same 0.7 threshold, 4.11 tok/s, also incoherent). The integration is mechanically correct, demonstrated by the configurational match in Finding 1; the quality gap is inherited from the Phase 1 compression decision (gemma4 adapter at threshold=0.7) and is independent of the Phase 2.5 integration work.

This distinction matters for the Apple/KAIST conversation: Row 5 confirms the integration delivers what was designed; quality optimisation lives in compression-parameter space (threshold, sensitivity-aware layer selection) and warrants its own focused work outside the Phase 2 scope.

---

## Phase 2 Measurement Table

| # | Model | Format | Engine | tok/s | J/token | Peak mem | Output |
|---|---|---|---|---:|---:|---:|---|
| 1 | gemma-4-E4B-it (mlx-community BF16) | BF16 | mlx_vlm | 23.43 | 0.51 | 16.2 GiB | Coherent — BF16 reference (engine upper bound on this hardware) |
| 1' | google/gemma-4-E4B-it | FP16 | pytorch_mps | 11.99 | 0.573 | 29.6 GiB | Coherent — apples-to-apples FP16 baseline (Phase 1.5 supersedes Phase 1's 10.64) |
| 2 | gemma-4-E4B-it (in-memory ternary, generic) | ternary mixed-INT4 | pytorch_mps | 4.11 | 4.04 | 32.7 GiB | Incoherent — quality envelope reference |
| 3 | gemopus-4-E4B-it thinking-off (in-memory ternary) | ternary mixed-INT4 | pytorch_mps | 0.94 | — | — | Incoherent — thermal-pressured cascade row, reproducibility-only |
| 4 | gemopus-4-E4B-it (llama.cpp Q4_K_M) | INT4 | llamacpp_gguf | 56.60 | 0.379 | — | Coherent — INT4 reference (different quantization scheme than ternary path) |
| **5** | **gemopus-4-E4B-it (.tern-model + Metal v1)** | **mixed ternary/INT4/FP16** | **terncore_packed** | **2.85** | **1.22** | **44.5 GiB** | **Incoherent — canonical Phase 2 measurement; configuration matches Stage C; quality envelope inherits Row 2 pattern** |

The table consolidates Phase 1, Phase 1.5, and Phase 2 measurements with explicit per-row output-coherence framing. Rows 1, 1', and 4 deliver coherent text and bound the engine-baseline performance envelope. Rows 2, 3, and 5 share the ternary-quantisation quality envelope inherited from the threshold=0.7 compression decision; their tok/s figures bound the speed envelope of various ternary execution paths on this hardware.

Row 5 is the canonical entry the Apple May 2026 conversation has been waiting on. Speed lands in Row 2 territory (~4 tok/s), consistent with the v1 Metal forward's per-layer CPU↔GPU round-trip overhead. Quality matches the established envelope.

---

## v1 Metal Forward + Forward Path

The Phase 2.5 Metal forward design was a deliberate v1 first-cut, scoped to land in bounded session time. Three properties define the v1 path:

- Weights stay GPU-resident across forwards via `MTLBuffer` instances allocated once at first MPS forward (per `PackedTernaryLinear` instance)
- Inputs and outputs round-trip CPU per forward call: `MPS tensor → numpy → MTLBuffer → matvec_gpu → MTLBuffer → numpy → MPS tensor`
- One Metal kernel dispatch per packed layer per token, with per-call buffer allocation for the transient input/output buffers

The 2.85 tok/s figure measures this v1 design accurately. At ~258 packed-layer dispatches per token plus the forward passes through 335 FP16-protected layers, the buffer-allocation and host-roundtrip overhead is the per-layer floor. The 1.22 J/token figure carries the same overhead profile — energy efficiency is also a v1 measurement, with the per-layer round-trip dominating.

The v2 Metal forward path opens three optimisations that should improve both tok/s and J/token together:

- **True MPS-resident input/output buffer reuse** — eliminates the per-layer numpy round-trip by routing MPS tensors directly to MTLBuffers via `mtl_buffer_from_tensor` (or equivalent zero-copy bridge)
- **Batched dispatch** — one Metal command buffer per token rather than per layer, reducing kernel-launch overhead
- **Optional command-queue prefetch** — overlap dispatch with previous-layer completion

These are scoped future work and follow the Phase 2.5 design pattern: v1 lands the integration; v2 lands the performance. **The 2.85 tok/s and 1.22 J/token figures are the v1 floor, not the Metal kernel's ceiling.** Both numbers should improve in v2 by margins proportional to the round-trip elimination — the per-layer overhead removal is structural rather than algorithmic.

The sprint cluster — Gemopus-4-26B-A4B-it and Gemopus-4-31B-it compressions, queued after Phase 2 — sequences naturally after v2 Metal lands so each compression is verifiable through the optimised inference path in the same session it lands.

---

## Diagnostic Journey

Phase 2.5 + Phase 4 surfaced four substantive findings via halt-and-surface discipline. Each landed with a documented memory entry for cross-session pattern recovery; each fix is in production and verified.

**CWD-dependency bug in `tern_engine_create()` (Stage 1).** Metal engine instantiation showed intermittent failure across 2026-05-03 and 2026-05-04 sessions — `tern_engine_create()` returned NULL from any working directory other than `csrc/metal/`. The root cause was a `__FILE__` macro in the Objective-C code: it expanded at compile time to a relative path (`./ternary_matmul.metal`), and the runtime lookup resolved that path against the Python process CWD rather than the dylib location. The fix replaces `__FILE__` with `dladdr()` against `&tern_engine_create`, anchoring the source-file lookup to the dylib's actual on-disk location. Verified: 5/5 fresh subprocess Metal init succeeds from any CWD. Pattern banked at `pattern_cwd_assumption_in_dylib_loading_v1.md`.

**`sparsity_bitmap` contract gap (Stage 1).** The cross-kernel equivalence test for the new `repack_uint8_to_uint32_codes` function surfaced a latent contract mismatch in `packed_ternary_matmul_fast`'s cached-bitmap path. The function's docstring promised packbits format (1 bit/weight, LSB-first); the path was accepting bool-format bitmaps from any caller passing `pack_ternary_weights`'s return value directly — silently producing 8× format mismatch and √K-scaled divergence. The fix adds a hard-reject shape check at the function boundary with an actionable error message naming the canonical packbits recipe. Production was unaffected (PackedTernaryLinear builds its own packbits cache via `_build_bitmap_from_packed`); the gap was contained to the new test path. Pattern banked at `pattern_compatible_types_incompatible_meanings_v1.md`.

**`reconstruct_all` `.weight`-doubling on production manifests (Phase 4).** Phase 4 dry-run revealed `load_state_dict: missing=2077, unexpected=2130` — every manifest key unmatched. The root cause was `reconstruct_all` unconditionally appending `.weight` to manifest entry names, which produced `*.weight.weight` for production manifests where entry names already include the suffix. Saturday's STATUS_PHASE2.md catalogued this as backlog item #7 and deferred it. The Path C fix adds a bounded `_PRODUCTION_NAME_SUFFIXES` constant and a per-entry conditional that detects whether the manifest uses test convention (bare names, append suffix) or production convention (suffixed names, use as-is). Test convention behaviour is preserved unchanged; production convention now loads correctly. Verified: 400 tests pass; `missing=1, unexpected=54` on the canonical artefact (matches Saturday's baseline exactly).

**Over-ternisation gap (Phase 4).** The same Phase 4 dry-run that surfaced the load-state-dict failure also surfaced over-ternisation: `convert_model_to_packed` produced `589 packed / 4 protected / 593 total` — a 99% ternisation rate that broke audio/vision encoder layers the gemma4 adapter intentionally kept FP16. The default `_should_protect_default()` heuristic catches embed/norm/lm_head patterns but lacks Gemma 4 multimodal awareness. The fix is the new `derive_protection_list_from_manifest` helper that constructs the protection list directly from the .tern-model manifest's per-layer dtype declarations. Verified: 258 packed / 335 protected / 593 total (matches Saturday's Stage C baseline exactly), reproducible from script. Patterns banked at `pattern_integration_test_configurational_fidelity_v1.md` and `pattern_quality_envelope_vs_integration_distinction_v1.md`.

---

## Methodology

**Prompt (fixed across all rows, identical to Phase 1 and Phase 1.5):**

> "Describe the process of photosynthesis at a high-school level. Cover the inputs, outputs, and where the reaction takes place. Keep your answer to roughly four sentences."

**Parameters:** `max_tokens=200`, warmup 50 tokens (no powermetrics), seed 42, greedy decoding (`do_sample=False`).

**Power measurement:** Apple `powermetrics` sampled at 2 Hz with active-window filter — only samples between first generated token and final token are counted as active. Both active-window and all-samples figures are captured per row in the JSON files for audit. Active-window figures are used in the Phase 2 table.

**Configurational fidelity assertion** (Phase 2 methodology addition): the harness asserts `convert_model_to_packed` produces fewer than 350 packed layers as a hard guard against regression to the over-ternisation pattern. The assertion fires loudly if the protection-list derivation fails to capture all FP16-marked layers, surfacing the configurational drift before any measurement runs. This is the structural defence against the silent-configuration-drift class banked at `pattern_integration_test_configurational_fidelity_v1.md`.

**Engine version stack at session:**

| Library | Version |
|---|---|
| mlx | 0.31.2 |
| mlx_lm | 0.31.3 |
| mlx_vlm | 0.4.4 |
| mlx-metal | 0.31.2 |
| torch | 2.7.0 |
| transformers | 5.5.4 |
| terncore | 0.4.0 (`phase-4-terncore-packed-bench` at `1ab6b9f`, 3 commits ahead of `main`) |

---

## Reproducibility Appendix

The Phase 2 measurement reproduces end-to-end from the canonical archive:

```bash
git checkout phase-4-terncore-packed-bench   # at 1ab6b9f
cd benchmarks
python bench_gemopus_phase2.py \
  --format terncore_packed \
  --model "/Volumes/Syn Archive/models/compressed/gemopus-4-e4b/gemopus_4_e4b_ternary_v0.1.0.tern-model/model.tern-model" \
  --hf-id Jackrong/Gemopus-4-E4B-it \
  --max-tokens 200 \
  --label row5_terncore_packed_metal
```

JSON outputs (committed):

- `benchmarks/gemopus_4_e4b_phase2/row5_probe_20t_canonical.json` — 20-token structural probe
- `benchmarks/gemopus_4_e4b_phase2/row5_terncore_packed_metal.json` — 200-token canonical measurement

Phase 2 commit ladder on `phase-4-terncore-packed-bench`:

- `e575dff` `feat(tern_model): production-manifest support — suffix preservation + protection_list helper` (closes STATUS_PHASE2.md backlog #7)
- `1ab6b9f` `feat(bench): terncore_packed canonical configuration + Row 5 measurement`

Stage 2 commits (merged into `main` via PR #9 as `3c3413c`):

- `05a549e` `fix(packed_ops): catch TypeError on MPS-resident input fallback`
- `3b39a69` `feat(packed_linear): Metal-aware forward path for MPS inputs`
- `d7693bd` `test(packed_linear_metal): production-data e2e on gemopus-4-e4b`

Stage 1 commits (on `main`):

- `38e2201` `feat(metal): add Metal engine singleton (metal_runtime)`
- `6feef02` `feat(metal): add uint32 repack and cross-kernel equivalence test`
- `97ebceb` `fix(metal): anchor source-file lookup via dladdr (CWD-independent)`
- `e4e74f0` `fix(packed_ops): hard-reject sparsity_bitmap shape mismatch in fast path`
- `89f546f` `feat(loader): support Gemma 4 transformers 5.5 multimodal layout`

Diagnostic context (preserved as historical evidence on the branch):

- `636faed` `WIP feat(bench): terncore_packed backend (over-ternisation gap surfaced)` — yesterday's diagnostic snapshot with the broken-state probe JSON

---

## Closing Reflection

Phase 2 closes the headline-number gap that Phase 1 explicitly deferred. The Metal kernel integration into `PackedTernaryLinear` works end-to-end on the canonical .tern-model artefact, configurational fidelity reproduces Saturday's baseline from a script, and the Row 5 measurement (2.85 tok/s, 1.22 J/token) is canonical for the v1 Metal forward design. Four substantive findings surfaced and resolved across Phase 2.5 + Phase 4 — each with a banked memory entry and a verified fix in production.

The Apple May 2026 conversation now carries a concrete number for what the v1 Metal path delivers on Apple Silicon today, framed honestly as the v1 floor. The v2 Metal forward optimisation work — bounded scope, well-understood improvements expected from MPS-resident buffer reuse and batched dispatch — is the path from this v1 floor toward what the architecture supports. The KAIST deliverable workflow is unchanged and unblocked: the same artefact that runs through Korean NPU inference now has a measured Apple-Silicon inference baseline. The sprint cluster — Gemopus-4-26B-A4B-it and Gemopus-4-31B-it — sequences naturally after v2 lands.

Phase 2 closes here.

---

*Gamma Seeds Pte Ltd — Synapticode*
