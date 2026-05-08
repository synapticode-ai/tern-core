# TN-003: Additive Compression Methodology Landscape

**Classification:** Gamma Seeds Pte Ltd — Internal Technical Note
**Author:** CC + Rob
**Date:** 2026-05-07 (Thursday evening preparation); 2026-05-08 (Friday morning infrastructure landed via PR #18; Friday afternoon harness build in flight)
**Status:** Friday morning infrastructure complete (PR #18 merged 2026-05-08 10:18:57 AEST); Friday afternoon measurement-infrastructure build in flight
**Prerequisite:** Per-expert ternary slicing (PR #14-#17), cross-architecture finding (Thursday morning, banked in `project_gemopus_26b_moe_compression_v1`)
**Patent relevance:** Patent 38 (Configurable precision), Patent 40 (Bandwidth optimisation), forward provisional candidates pending Friday outcomes

---

## Context

Today's cross-architecture sprint established per-expert ternary as a 4.1-4.9× compression baseline across 5 model instances spanning 4 architectures (Llama 3.1, Phi-4, Qwen3, Gemma 4). The cross-architecture FFN-clustering finding (median sparsity 0.42-0.43, spread 0.0072) is empirically established for the static-weight axis of compression.

This investigation extends the question to **additive compression**: which complementary techniques compose with the per-expert ternary baseline to push the unified-memory deployment ceiling higher? Today's 4.9× best ratio means a 60B-class MoE compresses to ~24 GB on M4 Pro 64 GB. If additive techniques push effective on-disk + runtime ratio to 6-8×, the ceiling rises to 80-120B-class models — material capability shift.

Rob references prior March 2026 KV cache comparative benchmarking work and an earlier ~300B-class compression attempt that didn't fit 64 GB. **Neither is in CC's visible context** (auto-memory file system effectively begins 2026-05-01; cf. `pattern_memory_coverage_boundary_v1`). This document reconstructs Rob's recollection collaboratively rather than asserting from incomplete traces.

### Methodology disciplines applied throughout

- **Epistemic markers per technique**: verified context (live integration, README-cited), pattern-matched recall (paper title + headline claim known, specifics not verified in this session), `[UNVERIFIED — needs paper lookup before Friday]` for techniques where CC's recall is genuinely uncertain.
- **Infrastructure-gap discipline** (`pattern_infrastructure_gap_discovery_v1`, banked tonight): probe actual integration surfaces before committing to measurements; two code paths producing structurally different outputs cannot be silently swapped.
- **Falsification readiness** (`pattern_methodology_memory_falsification_v1`): when measurements arrive Friday, if findings contradict published claims, the contradiction is the surface — investigate implementation correctness, metric fitness, comparison fairness before concluding the technique underperforms.

---

## Critical infrastructure gap discovered during preparation

The existing TurboQuant integration at `tools/tern_infer.py:149+` operates on the **v0.1.0 inference path**:

1. `AutoModelForCausalLM.from_pretrained(model_id)` — HuggingFace FP32 model
2. `MixedPrecisionConverter(threshold=0.7).convert(model)` — in-memory uniform-threshold ternary scan + conversion
3. `generate_streaming_turboquant()` — TurboQuant operates on the resulting in-memory model's `past_key_values`

**TurboQuant has NEVER been measured against tern-core's per-expert sliced `.tern-model` artefacts.** The two pipelines produce structurally different quantised models:

| Code path | Quantisation | Per-expert? | Output |
|---|---|---|---|
| `tools/tern_infer.py` (v0.1.0, TurboQuant-aware) | `MixedPrecisionConverter` uniform threshold | No (treats stacked experts as 3-D tensors) | In-memory model |
| `convert.py:full_convert` (PR #14-#17) | Per-tensor + per-expert slicing | Yes | `.tern-model` on disk |

**Status update 2026-05-08**: Friday morning's first deliverable — the `load_packed_model` rewrite — landed via PR #18 (merged 10:18:57 AEST). Two findings emerged from empirical retest cycles that adjust downstream scope: (a) Phi-4 ternary at threshold 0.7 produces repetition collapse (quality envelope, not load bug — disambiguated via cross-path methodology), and (b) gemma4-26b-a4b production retest surfaced per-expert-sliced MoE × stacked-tensor HF intersection requiring restacking dispatch (banked as backlog item, deferred to L5 sprint). The remaining Friday afternoon work — TurboQuant adapter from `.tern-model` path + standalone perplexity harness — is now unblocked.

---

## Categories of additive compression

### KV cache compression
Operates on the dynamic KV state during inference. Orthogonal to weight quantisation — addresses the runtime memory bottleneck weight compression does not touch. Most relevant to Apple/KAIST/NPU partner conversations because the unified-memory expansion problem (Llama-70B FP16 → ~116 GB at load) lives partly in KV state at long context. **Primary investigation target.**

### Activation quantisation
Quantises activations passed between layers during forward pass, separate from weight storage. Parallel to ternary rather than additive — typically replaces or adjusts the weight quantisation scheme rather than composing with it. AWQ, SmoothQuant, W4A8 fall here. Documented for completeness; **out of immediate scope** because the relationship to existing ternary baseline isn't additive.

### Structured pruning
Removes entire weight rows / columns / heads / layers based on importance metrics, producing a smaller dense model. Composes with ternary in principle (prune first, then ternarise the smaller surface). Implementation cost substantial. **Defer detailed investigation pending KV cache results land first**; if KV cache compression alone produces sufficient runtime envelope, structured pruning may not be needed.

### Runtime activation sparsity
Different from pruning — preserves all weights but skips weight columns at runtime based on activation magnitude per token. Independent of weight quantisation; can compose with ternary. **SpQt is Apple Silicon-specific** and therefore strategically interesting if implementable.

### Knowledge distillation
Trains a smaller "student" model to imitate a larger "teacher." Sequential pipeline (train → distill → quantise), not additive composition with existing artefacts. **Out of scope for this investigation** — would require fresh training infrastructure that doesn't exist in tern-core.

---

## Per-technique entries

### TurboQuant (ICLR 2026, Google Research) — INTEGRATED for v0.1.0 path, NOT YET MEASURED against per-expert `.tern-model` artefacts

**Description**: Two-stage KV cache compression — PolarQuant (random orthogonal rotation onto a sphere with codebook quantisation) + QJL residual correction (Johnson-Lindenstrauss-style sketch over residuals). Published claim: **6× memory reduction at 3-bit precision**, within 2.7× of information-theoretic optimal.

**Confidence**: Verified context. Live integration in tern-core; README v0.1.0 documents combined-stack benchmark; Bash probe earlier this session inspected the actual integration code at `tools/tern_infer.py:149-218`.

**Current tern-core integration**: `IncrementalTQCompressor` class at `tools/tern_infer.py:149+`. Operates on the v0.1.0 path (HF model + on-the-fly `MixedPrecisionConverter`). README v0.1.0 cites Mistral-7B at ~3.5 GB total Apple Silicon footprint against this path (combined with weight ternary).

**Infrastructure gap**: Not yet measured against per-expert sliced `.tern-model` artefacts (PR #14-#17 pipeline). See "Critical infrastructure gap" above.

**Required infrastructure for measurement** (status as of 2026-05-08 Friday afternoon):
- ✅ `load_packed_model` rewrite — landed via PR #18; Phi-4 in scope with quality-envelope characterisation; gemma4-26b-a4b xfail pending MoE restacking
- 🔄 Adapter from `.tern-model` inference path to TurboQuant's `IncrementalTQCompressor` — Friday afternoon build target (depends on `past_key_values` shape compatibility — verifiable now that rewrite has merged)
- 🔄 Standalone perplexity harness on WikiText-2 — Friday afternoon build target (extract from `MixedPrecisionConverter`'s auto-scan as standalone utility)

**Community caveat (worth measuring once infrastructure ready)**: Rob notes — QJL residual stage may hurt rather than help in autoregressive workloads per six independent community implementations; PolarQuant alone with MSE residual quantisation may outperform full pipeline. Worth measuring with and without QJL if the API supports both modes.

**Composability with per-expert ternary baseline**: Orthogonal in principle (KV cache vs static weights). Empirical composability gates on Friday's measurement infrastructure landing.

---

### KIVI (ICML 2024) — TARGET FOR FRIDAY

**Description**: 2-bit asymmetric quantisation of KV cache. Per-channel quantisation for keys, per-token quantisation for values. Published claim: **2.6× combined peak memory reduction** with minimal perplexity degradation. Most mature published alternative to TurboQuant in the published literature CC has context for.

**Confidence**: Pattern-matched recall — CC knows the technique by name, knows it's 2-bit + asymmetric + per-channel/per-token split, knows it's ICML 2024. Specific reduction percentages cited above are pattern-matched from impression of paper, NOT verified in this session. Friday's first-thing-Friday step: verify the published numbers via paper lookup.

**Implementation availability**: `[UNVERIFIED — needs Friday morning lookup]`. CC believes there's an open-source repo associated with the ICML paper but has not verified its current state, license, or maintenance status this session.

**Likely caveats** (pattern-matched from typical KV quantisation work, not verified for KIVI specifically):
- Custom CUDA kernels in published implementation — may not port cleanly to MPS / Apple Silicon
- Per-channel quantisation requires running statistics; may need calibration data or warm-up
- Performance cliff at very long context if outliers shift distribution

**Composability with per-expert ternary baseline**: Orthogonal in principle (operates on KV state, not weights).

**Friday integration estimate**: 3-4 hours focused engineering IF the published implementation has a Python-importable API that doesn't require CUDA kernel compilation; 6-8 hours if CUDA kernels require porting to MPS or replacement with PyTorch fallback.

---

### KVQuant — TARGET FOR FRIDAY

**Description**: Sub-4-bit KV cache quantisation, calibration-based. Published claim (pattern-matched): evaluated up to 10M context length, demonstrates that calibration-aware quantisation can preserve quality at very high compression ratios.

**Confidence**: Pattern-matched recall — CC knows the technique exists, knows it's calibration-based, knows the long-context evaluation framing. Specific numbers and methodology details NOT verified this session.

**Implementation availability**: `[UNVERIFIED — needs Friday morning lookup]`.

**Distinguishing characteristic**: Calibration cost. Unlike TurboQuant (training-free random orthogonal rotation) and KIVI (online statistics), KVQuant requires offline calibration pass over representative data. Calibration cost amortises across inference but adds engineering surface.

**Likely caveats**:
- Calibration data construction: needs representative WikiText-2 (or domain-appropriate) sample for the per-channel statistics
- Calibration needs re-running per model — 5 compressed models = 5 calibration passes
- Long-context evaluation may not generalise to shorter contexts (where most production deployments live)

**Composability with per-expert ternary baseline**: Orthogonal in principle.

**Friday integration estimate**: 4-6 hours including calibration data construction. May be tight for Friday alone; could pivot to "calibration infrastructure built Friday, measurement runs Saturday/following."

---

### kvtc (KV Cache Transform Coding) — TARGET FOR FRIDAY (maturity-dependent)

**Description**: Transform coding approach to KV cache compression. Published claim (per Rob's framing — CC has minimal independent context): up to 20× compression.

**Confidence**: `[UNVERIFIED — CC's recall uncertain that "kvtc" is the canonical name; may be informal acronym for a specific paper]`. Friday's verification step: identify the actual paper Rob is referring to (could be "KV Cache Transform Coding" as a literal title, could be an acronym/handle for a different specific paper — e.g., from arXiv 2024-2025 cohort).

**Implementation availability**: Unknown without verifying which paper. Newer arXiv work tends to have less mature open-source releases; package availability uncertain.

**Likely caveats**: Newness implies (a) implementation maturity uncertain, (b) community testing limited, (c) headline claims (20×) may not generalise outside specific evaluation conditions.

**Composability with per-expert ternary baseline**: Orthogonal in principle (KV cache).

**Friday integration estimate**: HIGHLY uncertain — could be 2 hours if there's a clean Python package, could be infeasible if it's paper-only or requires custom CUDA. Surface honestly Friday morning before committing.

---

### SpQt — TARGET FOR FRIDAY (Apple Silicon-specific, runtime activation sparsity)

**Description**: Apple Silicon-specific GPU runtime activation sparsity (per Rob's framing). November 2025 arXiv. Different category from KV cache compression — operates on activations during forward pass, skipping weight columns based on activation magnitude per token.

**Confidence**: `[UNVERIFIED — CC has minimal independent context for "SpQt" as a specific technique name]`. Friday's verification step: identify the actual paper. Apple Silicon-specific runtime sparsity work exists in literature but the specific naming Rob references needs lookup.

**Implementation availability**: Apple Silicon-specific GPU code typically requires Metal Shading Language or MPSGraph integration. No standard PyTorch package; CC believes any implementation would require Metal kernel work.

**Composability with per-expert ternary baseline**: In principle composable — runtime sparsity is independent of static weight quantisation. The Metal forward path landed in PR #9-#10 (Phase 2.5 + Phase 4) provides the integration surface; however, that path is currently the v1 floor (2.85 tok/s on E4B), not optimised for sparsity-aware dispatch.

**Friday integration estimate**: HIGHEST uncertainty in this scout. Apple Silicon GPU integration is non-trivial; likely requires a feasibility-scout-only outcome Friday with implementation deferred to a dedicated session.

---

### Out-of-immediate-scope techniques (documented for completeness)

#### MLA (Multi-head Latent Attention)
**Description**: Architectural change — projects KV onto a low-dimensional latent space, dramatically reducing KV cache memory. Used in DeepSeek-V2/V3.
**Confidence**: Verified context.
**Why out of scope**: Architectural — requires training from scratch or substantial weight surgery. Cannot be applied to existing compressed `.tern-model` artefacts. Relevant to long-term tern-core direction (designing compression schemes alongside MLA-aware models) but not to this investigation.

#### AWQ (Activation-aware Weight Quantisation)
**Description**: 4-bit weight quantisation that uses activation distribution to identify "salient" weights to preserve at higher precision.
**Confidence**: Verified context.
**Why out of scope**: Parallel rather than additive to ternary — replaces the weight quantisation scheme rather than composing on top. Could be a future alternative weight quantisation path for tern-core (AWQ-then-ternary hybrid?), but that's a different investigation.

#### SmoothQuant
**Description**: Migrates activation outliers into weight quantisation to enable W8A8 inference.
**Confidence**: Verified context.
**Why out of scope**: Same reasoning as AWQ — parallel to ternary rather than additive.

#### W4A8 / W4A4
**Description**: Joint weight + activation quantisation schemes (4-bit weights + 8-bit or 4-bit activations).
**Confidence**: Verified context.
**Why out of scope**: Replaces tern-core's weight quantisation; not additive.

#### SlideSparse
**Description**: `[UNVERIFIED — CC's recall uncertain that this is a canonical technique name]`. Possibly NVIDIA Tensor Core specific structured sparsity work.
**Why out of scope**: Hardware-incompatible with M4 Pro target if NVIDIA-specific. Apple's Tensor Core analogue (ANE matrix engine) has different sparsity primitives. If SlideSparse turns out to be the canonical name for something else portable, revisit.

#### CFSP / Compresso / QPruner
**Description**: Structured pruning literature — CFSP claims FFN pruning at varying ratios; Compresso and QPruner are pattern-matched as related structured pruning approaches but specifics `[UNVERIFIED]`.
**Why out of scope**: Substantial implementation work; defer until KV cache + activation sparsity comparisons land first to confirm whether structured pruning is needed on top of the runtime envelope KV compression alone produces.

#### "Apple Core ML 7.0 + TensorRT 9.2"
**Description**: `[POSSIBLY MISNAMED — TensorRT is NVIDIA, not normally used with CoreML; pairing is unusual and may indicate a specific paper title rather than a general technique combination]`. Friday verification step.

#### Knowledge distillation
**Description**: Train smaller student to imitate larger teacher.
**Why out of scope**: Sequential pipeline, not additive to existing compressed artefacts. Would require fresh training infrastructure.

---

## Implementation feasibility scout findings

Per-technique assessment of integration cost into the future per-expert `.tern-model` inference path (NOT the current v0.1.0 path). Tonight's findings are **CC's preliminary assessment based on existing context**; Friday morning starts with verification against actual repo state for KIVI / KVQuant / kvtc / SpQt.

| Technique | Implementation availability (CC's preliminary) | Installation effort | tern-core integration effort | Engineering estimate (Friday) | Notes |
|---|---|---|---|---|---|
| TurboQuant | INTEGRATED (v0.1.0 path) | n/a | Adapter from `.tern-model` path to existing `IncrementalTQCompressor` | 1-2 hr (after `load_packed_model` rewrite) | Already imported from `/Users/syn/synapticode/venv/src/turboquant`; community caveat about QJL worth testing both modes |
| KIVI | UNVERIFIED — published repo expected, state unknown | Likely pip install OR build from source | Drop-in if PyTorch-only API; non-trivial if CUDA-only | 3-4 hr (PyTorch API) / 6-8 hr (CUDA→MPS port) | ICML 2024 maturity; verify open-source state Friday morning before committing to integration time |
| KVQuant | UNVERIFIED — published repo expected, state unknown | Likely pip install OR build from source | Calibration infrastructure adds surface; dispatch is otherwise drop-in | 4-6 hr including calibration data construction | Long-context evaluation framing may need adjustment for shorter benchmarks; calibration cost amortises |
| kvtc | UNVERIFIED — paper itself needs identification | Unknown; newer arXiv work | Unknown without paper identified | 2 hr (clean package) to infeasible | Highest implementation maturity uncertainty in scout |
| SpQt | UNVERIFIED — paper itself needs identification | Apple Silicon GPU code; no standard package | Metal kernel work required | Feasibility scout only Friday; integration deferred to dedicated session | Highest engineering uncertainty; runtime sparsity integration into v1 Metal forward is substantial work |

**Caveat on the entire scout**: tonight's table is preliminary. Friday morning's first 30-60 min should verify each technique's repo state, package availability, and current maintenance posture before committing the engineering hours estimated above. The engineering estimates assume the optimistic case (clean Python API, drop-in dispatch); pessimistic cases may shift estimates materially.

---

## Friday investigation plan

### Friday morning (executed 2026-05-08): `load_packed_model` rewrite — INFRASTRUCTURE-CRITICAL — COMPLETE

Per the elevated backlog item, this gated everything downstream. Landed via PR #18 (commit 35240d1, merged 10:18:57 AEST):

1. ✅ **FP16 branch fix**: parameter-path-aware traversal handles production naming uniformly
2. ✅ **INT4 branch implementation**: B.1 dequantise-at-load with operator-visible log message; PackedINT4Linear (B.2) banked for future commercial-driven implementation
3. ✅ **`key_mapping` support**: `GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` and other future drift presets accepted
4. ✅ **Test coverage**: synthetic fixtures (7 tests, dense models) + production manifest integration tests (Phi-4 in scope; gemma4-26b-a4b xfail pending MoE restacking; gemma4-31b + qwen3-30b-a3b skipped per M4 Pro 64 GB ceiling; Mistral-7B compressed artefact not on disk per filesystem-verified 2026-05-07)
5. ✅ **Output sanity**: smoke probes via `expect_coherent_generation` parametrise field — Phi-4 disambiguated as quality-envelope at threshold 0.7

**Findings emerging from empirical retest cycles** (banked as backlog items):
- Phi-4 ternary recompression at lower threshold — quality-envelope characterisation
- load_packed_model: MoE per-expert restacking for stacked-tensor architectures (natural integration with L5 sprint week of 2026-05-12)
- 9th probe-before-committing instance banked to methodology memory (validation infrastructure scope extension)

### Friday late morning (1-2 hours): Build measurement infrastructure

1. **Perplexity harness on WikiText-2**: callable API `compute_perplexity(model, dataset_name="wikitext-2-raw-v1", split="validation")`. Currently PPL is internal to `MixedPrecisionConverter`'s auto-scan; expose as standalone utility.
2. **TurboQuant adapter** from new `.tern-model` inference path to `IncrementalTQCompressor`. Depends on `past_key_values` shape compatibility post-rewrite.
3. **Smoke test infrastructure on smallest model (Mistral-7B)**: verify TurboQuant compression reduces KV cache size as expected, perplexity stays within reasonable band of baseline.

### Friday afternoon (3-5 hours): Comparative measurements

Halt-and-surface discipline between technique integrations.

1. **TurboQuant baseline against per-expert `.tern-model` artefacts**: scope reduced to **Phi-4 alone** for first measurement target. Empirical findings from Friday morning's PR #18 retest constrain the original 5-model framing:
   - Mistral-7B compressed artefact NOT on disk (filesystem-verified 2026-05-07; the original "5 manifests on disk" assumption was wrong)
   - Qwen3-30B-A3B exceeds M4 Pro 64 GB practical ceiling (~57 GB FP16 base load)
   - gemma4-26b-a4b xfail pending MoE per-expert restacking
   - gemma4-31b exceeds hardware ceiling (~58 GB FP16)
   - Phi-4 in scope (27 GB FP16) — enters baselines as KV-cache-compression characterisation reference; underlying weight-compression quality envelope is documented but TurboQuant operates on inference-time KV cache, orthogonal to weight ternary

   Single-model first-measurement establishes infrastructure works end-to-end before fanning out. Cluster-wide TurboQuant baseline gates on (a) gemma4-26b-a4b MoE restacking landing (next-week L5 sprint) + (b) hardware unblock for 30B+ class.
2. **KIVI integration + comparison** if Friday-morning verification of feasibility scout confirms reasonable integration cost. Same model subset as TurboQuant baseline for apples-to-apples.
3. **KVQuant** likely deferred to following session given calibration infrastructure cost.
4. **kvtc + SpQt** deferred entirely to following sessions.

### Saturday/following session: KVQuant + kvtc + SpQt continuation

Build on Friday's foundation. Calibration infrastructure (KVQuant), paper identification (kvtc, SpQt), Apple Silicon GPU integration scout (SpQt) become the deliverables for the next focused session.

### Quality metric framework (apples-to-apples for all techniques)

- **Perplexity** on WikiText-2 validation split (`wikitext-2-raw-v1`, conventional choice for LLM perplexity benchmarking)
- **KV cache footprint** at 4K context length (chosen for tractable wall-clock; long-context evaluation deferred unless KVQuant's 10M-context claim becomes the focus)
- **Peak memory during inference** (RSS measurement)
- **Wall-clock per token** (throughput on Apple Silicon)

Single-model results don't generalise to "X always works." Multi-model comparison required for any defensible cross-technique claim.

### Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `load_packed_model` rewrite takes longer than 4-6 hr (e.g., production-manifest edge cases surface) | Medium | Halt-and-surface mid-rewrite; defer Friday afternoon comparative work to following session if needed |
| KIVI's CUDA kernels don't port cleanly to MPS | Medium | PyTorch fallback path costs 2-4× wall-clock; assess Friday morning before committing afternoon to KIVI specifically |
| Perplexity harness produces unexpected results (wildly off baseline) | Low | Apply `pattern_methodology_memory_falsification_v1`: investigate harness implementation correctness, dataset preprocessing, tokenisation alignment before concluding model is broken |
| KVQuant calibration data construction blows time budget | Medium | Defer KVQuant to following session if calibration takes >2 hr to scope |
| kvtc / SpQt papers can't be cleanly identified Friday morning | Medium | Document the identification gap honestly; defer to Saturday/following with a focused paper-search session |
| Wall-clock for 5-model perplexity sweep exceeds afternoon capacity | High | Triage to 3-model subset (smallest + biggest + MoE representative); preserve methodology defensibility over sample size |

---

## References

For Friday morning verification — CC has not fetched these in this session. Each technique entry should be cross-checked against its actual repo + paper before integration commits Friday.

- **TurboQuant** — ICLR 2026, Google Research. Local copy at `/Users/syn/synapticode/venv/src/turboquant`. README v0.1.0 references.
- **KIVI** — ICML 2024. Verify arXiv ID + repo URL Friday morning.
- **KVQuant** — Verify arXiv ID + repo URL Friday morning.
- **kvtc** — Identify canonical paper Friday morning.
- **SpQt** — Identify canonical paper Friday morning. November 2025 arXiv per Rob's framing.
- **MLA** — DeepSeek-V2 paper (arXiv:2405.04434 expected; verify).
- **AWQ** — `arXiv:2306.00978` expected.
- **SmoothQuant** — `arXiv:2211.10438` expected.
- `pattern_memory_coverage_boundary_v1` — auto-memory boundary 2026-05-01 (CC's pattern memory)
- `pattern_substring_pattern_discrimination_v1` — pattern discrimination methodology (CC's pattern memory)
- `pattern_methodology_memory_falsification_v1` — falsify-rather-than-restate discipline (CC's pattern memory)
- `pattern_infrastructure_gap_discovery_v1` — banked tonight; probe integration surfaces before committing to measurements

---

## Investigation status

| Phase | Status |
|---|---|
| Thursday evening (2026-05-07): TN-003 landscape + feasibility scout + Friday plan + backlog elevation | **Complete** |
| Friday morning (2026-05-08): `load_packed_model` rewrite — infrastructure-critical | **Complete (PR #18 merged 10:18:57 AEST)** |
| Friday early afternoon (in flight): probes A + B (perplexity computation surface; TurboQuant `.tern-model` interface) | **In flight** |
| Friday afternoon: Measurement infrastructure build (perplexity harness extract; TurboQuant adapter) + first-target measurement (Phi-4) | Planned |
| Saturday/following: KIVI integration + cluster expansion (gemma4-26b-a4b post-MoE-restacking; hardware-unblocked 30B+ class) + KVQuant calibration + kvtc/SpQt paper identification | Planned |

Updated incrementally as Friday's work lands. This document is the persistent reference scaffold.
