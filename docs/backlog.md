# tern-core backlog

Deferred items surfaced during integration sessions. Each item names the
session of origin, the exact code locations affected, and the recommended
fix scope. Closed items move to a "Closed" section at the bottom.

---

## Open

### load_packed_model rewrite — production manifest support

**Surfaced:** 2026-05-03 Phase 2 Stage C round-trip on production gemopus-4-e4b artefact (FP16 silent skip). **Scope expanded 2026-05-07** during Commit 1 design probe — discovered the ternary path has the same parameter-path traversal flaw, manifesting as silent module-structure corruption rather than silent skipping.

**Bugs surfaced for fix (shared root cause: parameter-path naming not detected):**

1. **FP16 silent skip** (originally documented). Parameter-path manifest naming (e.g., `model.layers.0.norm.weight`) walks `parts[:-1]` to the parent module; `getattr(parent, "weight")` returns a Parameter, not an `nn.Linear`; `isinstance(Parameter, nn.Linear)` is False; entry silently skipped. **853 of 2130 entries lost on gemopus-4-e4b** (852 `.weight` + 1 `.bias`). Affects every FP16 entry whose target is a non-Linear module (LayerNorm, RMSNorm, Embedding) or whose name uses parameter-path convention.

2. **Ternary load-time TypeError** (surfaced 2026-05-07 during Commit 1 design probe; framing empirically corrected when Commit 1's diagnostic test ran). Same parameter-path traversal: ternary entry `model.layers.0.q_proj.weight` walks to `model.layers.0.q_proj` (the Linear), then `setattr(q_proj, "weight", PackedTernaryLinear_instance)` attempts to replace the registered `weight` Parameter with a Module instance. **PyTorch's `nn.Module.__setattr__` guard catches this**: `TypeError: cannot assign 'PackedTernaryLinear' as parameter 'weight' (torch.nn.Parameter or None expected)`. Production manifests fail to load with this loud TypeError on the first ternary entry encountered.

   *Methodology trail*: originally framed as "silent corruption" based on probe-time pattern-matching from how the bug *would* work absent PyTorch's guard. Commit 1's diagnostic test surfaced PyTorch's `__setattr__` actually catches the corruption attempt. Loud failure rather than silent corruption — strictly better than the originally-framed mechanism (debuggable at load time rather than as confusing forward-pass errors), but still a bug because production manifests fail to load via `load_packed_model`. The contradiction between predicted mechanism and observed mechanism is itself the investigation finding (cf. `pattern_methodology_memory_falsification_v1`).

3. **`int4_block32` dtype not handled in `load_packed_model`.** `reconstruct_layer` handles it via `_reconstruct_int4` (`tern_model.py:847-848`), but `load_packed_model` has only `ternary2` + `float16` branches. 11 entries silently skipped on gemopus-4-e4b. INT4 routing also occurred on Wednesday's 26B-A4B compression (10 entries via cross-applied E4B sensitivity map), so this affects production manifests beyond gemopus-4-e4b.

4. **`key_mapping` translation needed for transformers API drift.** Already shipped as parameter on `load_as_model` (2026-05-03); `load_packed_model` needs the same parameter for parity. Preset `GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` already defined at module scope in `tern_model.py`.

5. **`missing`/`unexpected` reporting is also broken.** Discovered during Commit 1 design probe: the function returns `(missing, unexpected)` analogous to `load_state_dict`, but `loaded_keys.append(f"{name}.weight")` for FP16 entries with parameter-path naming produces double-suffixed entries (`"embedding.weight.weight"`). Comparison against `model.named_parameters()` keys is therefore unreliable. Fix needs to produce correct loaded_keys for parameter-path entries (just `name`, not `f"{name}.weight"` when name already ends in `.weight`).

The first two bugs share the same root cause: load path doesn't detect when manifest entry names use parameter-path naming (suffixed with `.weight` / `.bias`) versus module-path naming (bare module identifier, test convention). Fix requires uniform parameter-path-aware traversal in both ternary and FP16 branches.

**Test gap:** `tests/test_packed_linear.py` exercises module-path naming
only. `tests/test_sparsity.py:148-171` likewise. Production naming
convention untested until Phase 2.

**Recommended fix scope:**
- FP16 branch detects `.weight` / `.bias` / other-parameter-suffix at
  `parts[-1]`, walks to parent module, assigns to `module.<param>.data`.
  Handles `nn.Embedding`, `nn.LayerNorm` / RMSNorm, `nn.Linear`
  uniformly.
- `int4_block32` branch parallel to `ternary2` / `float16`,
  dequantising via `_reconstruct_int4` and assigning into the module's
  weight tensor.
- Add `key_mapping` parameter mirroring `load_as_model`.
- Regression test that loads a production-format manifest (built via
  `convert.py` or `streaming_convert.py`, not `add_layer`) and verifies
  every entry lands at its correct destination, with diffs against
  `from_pretrained` weights for a small reference model.

**Estimated effort:** 4–6 hours focused work with PR.

**Acceptance criteria (adjusted for both M4 Pro 64 GB hardware ceiling AND per-expert-sliced MoE restacking deferral):**
- All compressed manifests that **(a)** fit in M4 Pro 64 GB unified memory AND **(b)** use single-tensor-per-entry naming load via `load_packed_model` without silent entry skipping or silent corruption — verified via `tests/test_load_packed_model_production_integration.py`
- Phi-4 (dense, fits 27 GB) integration verified with quality-envelope characterisation (`expect_coherent_generation=False` per disambiguation finding 2026-05-08; load + clean logits asserted, repetition collapse documented as known outcome rather than asserted against)
- gemma4-26b-a4b (MoE, fits 48 GB) integration test `xfail`-marked pending MoE per-expert restacking — verified empirically 2026-05-08 that the loud-failure surface fires correctly on the per-expert-sliced × stacked-tensor architectural mismatch (cf. backlog item "load_packed_model: MoE per-expert restacking")
- 30B+ class manifests (gemma4-31b ~58 GB FP16, qwen3-30b-a3b ~57 GB FP16) explicitly skipped with documented hardware constraint per TN-001 / `pytest.mark.skip`; same M4 Pro 64 GB ceiling that makes Llama-3.1-70B "demo artefact only" applies
- Mistral-7B compressed artefact NOT on disk (verified via filesystem probe 2026-05-07); skipped with documented absence
- Synthetic-fixture coverage in `tests/test_load_packed_model_production_naming.py` verifies the same code paths on architecturally-equivalent small dense models — exercises every bug path independent of hardware ceiling. **Note:** synthetic dense fixtures do NOT cover per-expert-sliced MoE × stacked-tensor HF intersection; that gap surfaced empirically only via gemma4-26b-a4b production retest and is the validation-infrastructure finding banked as Instance 9 of `pattern_probe_before_committing_implementation_v1`.
- Loaded models produce either non-garbage outputs OR documented quality-envelope collapse on a 50-token smoke probe per in-scope model — pragmatic eyeball check via test stdout, gated by `expect_coherent_generation` parametrise field
- `key_mapping` parameter accepts `GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` and other future drift presets — verified empirically via gemma4-26b-a4b retest 2026-05-08 (key_mapping translation worked correctly; failure occurred in per-expert walk after translation, isolating the failure mode to the restacking gap rather than the key_mapping path)
- Loud-failure surface: `_resolve_module_or_raise` + `_resolve_parameter_or_raise` raise `ValueError` with diagnostic info naming manifest entry + missing path component when traversal fails — never silent skip / silent corruption / silent reporting drift; verified empirically by gemma4-26b-a4b retest surfacing per-expert MoE restacking gap as a clean loud-failure with full diagnostic context

Original criterion ("All 5 manifests load") was written without M4 Pro 64 GB factored in AND without the per-expert MoE restacking gap surfaced; adjusted to "manifests that fit in 64 GB AND use single-tensor-per-entry naming" framing with documented skip / xfail reasons for the rest. Future hardware (M4 Max, M5, Mac Studio 128+ GB) unblocks the hardware-ceiling cases without code changes; MoE restacking work (banked as separate backlog item, scheduled for L5 sprint week of 2026-05-12) unblocks the gemma4-26b-a4b xfail.

---

### Known architecture quirk: full_attention v_proj absence (gemma4 26B-A4B)

**Status:** Open (low priority, investigate later)
**Surfaced:** 2026-05-06 Session 3 design surface for per-expert slicing
**Observation:** The 5 full_attention layers (indices 5, 11, 17, 23, 29) carry 5 self_attn entries instead of 6 — they have no v_proj.weight, while q_proj doubles to [8192, 2816] (matches global_head_dim=512). The 25 sliding_attention layers carry 6 self_attn entries with v_proj present.
**Hypothesis:** Full-attention layers may reuse v from an adjacent layer or via a fused projection.
**Investigation:** Not blocking; the count formula in the Session 3 per-expert slicing rework incorporates the asymmetry. Worth confirming the architectural intent for thoroughness.

---

### dry_run_convert: stacked-tensor expansion not modelled

**Status:** Open (low priority, informational)
**Surfaced:** 2026-05-06 Session 3 per-expert slicing rework
**Observation:** `dry_run_convert` (`src/terncore/convert.py`) doesn't call `adapter.expand_stacked`, so MoE stacked-experts parent tensors (e.g., `experts.gate_up_proj` shape `[128, 1408, 2816]`) appear as 3-D ternary-eligible entries in the dry-run output rather than as 128 per-expert slices. Misleading for capacity planning on MoE models but doesn't affect actual compression correctness — `full_convert` (the active path) handles expansion correctly.
**Resolution scope:** Mirror the `stacked_plans` pattern from `full_convert` in `dry_run_convert`. Estimated 1-2 hours focused engineering. Not blocking any current work.

---

### Implement PackedINT4Linear for runtime-memory-preserved INT4 inference

**Status:** Open (low priority, future-when-needed)
**Surfaced:** 2026-05-07 Thursday evening rewrite Commit 3 design
**Observation:** `load_packed_model`'s Commit 3 INT4 branch dequantises INT4 entries to FP32 at load time (Scenario B.1 trade-off). Production exposure is small (~20 INT4 entries across all 5 manifests, mostly cross-applied sensitivity-map fallbacks for high-error layers; gemopus-4-e4b has 11, Wednesday's 26B-A4B compression has 10). Runtime memory difference vs a packed runtime module is probably <100 MB across the entire dataset. INFO-level log message at first INT4 entry encountered surfaces this trade-off explicitly so operators see it in load output (cf. `tern_model.py:load_packed_model` docstring).

**Resolution scope:** Implement `PackedINT4Linear` mirroring `PackedTernaryLinear`'s structure — holds packed INT4 weights + per-block scales, dequantises on forward (or implements INT4 matmul via Metal kernel for runtime-memory-preserved inference). ~2-3 hours focused engineering plus tests plus integration into `load_packed_model`.

**Trigger for unblocking:** commercial pressure that justifies the runtime-memory-preserved variant (e.g., a partner deployment where the INT4 entries dominate or the FP32 dequantisation cost matters at inference scale).

---

### Llama-3.1-70B-Instruct source weights: download deferred

**Status:** Open (low priority, future-when-needed)
**Surfaced:** 2026-05-06 Wednesday-night queue planning
**Observation:** License accepted on HF Hub 2026-04-09; source weights (~140 GB FP16) never pulled. Only `refs/main` SHA reference cached, no snapshots/blobs. The existing compressed `.tern-model` artefact at `/Volumes/Syn Archive/models/compressed/llama-3-1-70b/llama70b-v0.6.0-mixed.tern-model` (37 GB) already serves as cross-model reference for analysis work (used in Session 4 per-expert tolerance analysis 2026-05-06).
**Resolution scope:** Fire `hf download meta-llama/Llama-3.1-70B-Instruct` when a specific need arises that the existing compressed artefact can't satisfy (e.g., recompression with v2 Metal forward path improvements, full-precision baseline experiments, cross-validation). ~140 GB / ~1-2 hours overnight.

---

### Phi-4 ternary recompression at lower threshold — quality-envelope characterisation

**Status:** Open (medium priority, gated on TN-003 baseline measurement priorities)
**Surfaced:** 2026-05-08 by Rob during Friday morning Phi-4 disambiguation
**Observation:** April 2026 Phi-4 compression at threshold 0.7 (`phi4_14b_ternary_v0.1.1.tern-model`) produces repetition collapse in greedy generation ("at at at at..." after the input prompt). Disambiguated 2026-05-08 via cross-path methodology — loaded the same `.tern-model` artefact via `load_as_model` (the structurally independent dequantise-to-FP32 + `load_state_dict` path) and observed identical collapse with clean logits (no NaN/Inf, shape correct). Two structurally independent load paths producing identical observable outcome rules out load-infrastructure bugs and confirms the issue as a quality-envelope property of Phi-4 ternary at threshold 0.7. PR #16's structural-equivalence anchor for Phi3Adapter validation passed (manifest entries match), but inference quality was never empirically measured at integration time.

**Resolution scope:** Recompress Phi-4 at lower threshold and rerun the disambiguation smoke probe. Suggested first probe at 0.5 (operating-range floor per `~/synapticode/CLAUDE.md`'s typical operating range 0.5-0.9). If coherent generation returns at 0.5, sweep upward toward 0.6, 0.65 to find the compression-vs-quality optimum and bank the threshold curve as quality-envelope reference data for Phi-4. If 0.5 also collapses, the finding is below threshold-tuning intervention scope and requires different methodological approach (model-specific calibration, layer-wise sensitivity scan extension, possibly a Phi-4-specific protected-layers list). Applies the same per-model threshold calibration pattern that v0.6.0 sensitivity scan establishes for high-error layers, but at the model-level rather than layer-level granularity.

**Trigger:** Apply when TN-003 baseline measurement planning needs Phi-4 as a coherent-generation reference point. If Friday afternoon's TurboQuant baseline work proceeds with Phi-4 as quality-envelope-documented (acceptable for KV-cache compression characterisation), this item defers. If commercial conversations (Apple/KAIST briefing) require Phi-4 as a coherent-generation demo artefact, prioritise.

**Estimated effort:** ~2-3 hours per threshold pass (compression + smoke probe + disambiguation). Single-threshold Phi-4 recompression at 0.5 ~3 hours wall-clock. If 0.5 returns coherent generation, additional ~3 hours per upward-sweep threshold. If 0.5 collapses, scope shifts to a different methodological investigation (out of this item's bound).

**Methodology pattern reference:** This finding extends `pattern_probe_before_committing_implementation_v1` with the cross-path disambiguation as a concrete instance — empirical probe (`load_as_model` independent path) ruled out predicted mechanism (load infrastructure bug) and confirmed actual mechanism (quality envelope).

---

### load_packed_model: MoE per-expert restacking for stacked-tensor architectures

**Status:** Open (medium priority, natural integration with L5 sprint planned for week of 2026-05-12)
**Surfaced:** 2026-05-08 by gemma4-26b-a4b production manifest integration retest during PR #18 verification

**Observation:** PR #14's per-expert slicing rework produces compressed manifests with 128 separate `experts.N.{gate,up,down}_proj.weight` entries per layer (Gemma 4 family) for per-expert sparsity measurement granularity. PR #18's `load_packed_model` rewrite handles per-entry dispatch (one entry → one module replacement or parameter assignment). The intersection — loading per-expert-sliced MoE manifests back into HF MoE models that expose experts as stacked-tensor Parameters (e.g., `model.language_model.layers.0.experts.gate_up_proj` with first dim = 128) rather than ModuleList of individual expert modules — was never empirically tested before 2026-05-08's gemma4-26b-a4b retest.

`_resolve_module_or_raise` correctly raises `ValueError` on the first `experts.0` lookup since `experts` is a stacked-tensor `nn.Parameter`, not a `nn.ModuleList`. Loud-failure surface working as designed; the structural gap is in `load_packed_model`'s dispatch logic, not in the loud-failure helpers.

**Resolution scope:** Implement per-expert-naming detection + restacking dispatch in `load_packed_model`. Pattern: detect entries matching `experts.N.<projection>` naming → group by parent path + projection name → reconstruct stacked tensor by stacking 128 per-expert dequantised weights along dim 0 → assign to the parent's stacked-tensor parameter. Mirrors `reconstruct_all`'s restacking logic from PR #14 but applied at `PackedTernaryLinear` creation time (or, for stacked tensors, at parameter assignment time — design probe required to determine which is correct).

Test coverage: synthetic fixture mirroring Gemma 4 MoE expert structure (small N experts × small projections) for unit testing; production integration test against gemma4-26b-a4b (currently xfail) for end-to-end verification.

**Trigger:** Natural integration with next-week L5 sprint (week of 2026-05-12). The L5 lifecycle work on Qwen3-30B-A3B (Tier 1 per Rob's sprint planning context) and DeepSeek-V4-Flash (Tier 3) requires MoE expert paging + demand-loading + restacking; per-expert restacking in `load_packed_model` is a prerequisite for the demand-paging implementation. Sequencing: restacking work lands first as foundation, then L5 lifecycle work builds on top.

**Estimated effort:** ~3-4 hours focused work — design probe (per-expert naming detection + restacking dispatch placement) + implementation + synthetic fixture tests + xfail removal on gemma4-26b-a4b production test + retest cycle (~45 min wall-clock).

**Methodology pattern reference:** This is the 9th and most consequential instance of `pattern_probe_before_committing_implementation_v1` — caught a structural gap in validation infrastructure (synthetic dense fixtures don't cover per-expert-sliced MoE × stacked-tensor HF model intersection) that would have been a silent gap in PR #18 acceptance had we proceeded to merge based on synthetic-fixture coverage alone. Empirical production-manifest retest was required to surface the gap.

---

### Analysis plot tooling: include model name as plot title attribution

**Status:** Open (low priority, applied after current rewrite work lands)
**Surfaced:** 2026-05-08 by Rob during Friday morning Phi-4 disambiguation wait
**Observation:** `benchmarks/analyse_per_expert_tolerance.py` plot output (and any sibling plot generators) currently relies on filename + directory path for model attribution. This is fragile — plots saved standalone, embedded in briefing slides, sent to external reviewers (Eagle for IP review, Apple/KAIST partners), or aggregated for comparative work lose directory-path provenance.

**Resolution scope:** Update plot title format to include model identification visibly on the graph itself. Suggested format: `"Sparsity distribution: expert_ffn vs dense_ffn — <model_id>"` where `<model_id>` is derived from manifest metadata (e.g., "Phi-4", "Qwen3-30B-A3B", "Gemma 4 26B-A4B"). Apply across all plot types in the analysis pipeline (distribution overlay, per-layer scatter, per-expert heatmap, external reference comparison).

**Trigger:** Apply after current `load_packed_model` rewrite work lands (PR #18 merged). Should land before Friday afternoon's TurboQuant baseline comparative plots — those will be more numerous, more likely to be shared externally, and carry more commercial weight.

**Estimated effort:** ~30 min focused work + small test update.

---

### Additive compression methodology investigation

**Status:** Open (medium priority, follow-up research)
**Surfaced:** 2026-05-07 by Rob during Session Thursday close-out
**Observation:** Today's cross-architecture finding establishes per-expert ternary as a 4.1-4.9× compression baseline across architectures. tern-core already integrates two additive techniques: (1) **TurboQuant** (ICLR 2026, KV cache compression, 6× memory) live at `tools/tern_infer.py:149+` with combined-stack benchmark documented in `README.md` (Mistral-7B at ~3.5 GB total Apple Silicon footprint), and (2) **mixed ternary/INT4** (v0.6.0) routing high-error layers to INT4 via sensitivity scan. Rob references prior March 2026 KV cache comparative benchmarking work (per-model variation: TurboQuant outperformed alternatives for some models, alternatives outperformed for others) plus an earlier ~300B model compression attempt that didn't fit 64 GB. **Neither is in CC's visible context** — auto-memory file system effectively begins 1 May 2026; March work pre-dates it. Investigation reconstructs Rob's recollection via manual prompts rather than CC pattern-matching from incomplete traces.

**Investigation scope:**
- **Reconstruct prior history with Rob** via manual prompts: which specific models were benchmarked in March, which KV cache compression alternatives were tested — Rob's recollection should drive this; published candidates worth considering as starting points include KIVI, KVQuant, GEAR, H2O, AlphaCompress, but the actual March-tested set may have been different. Which ~300B-class model attempt was tried (possibly Llama 3.1 405B misremembered, possibly Falcon-180B, possibly a proprietary attempt) and what compression stack was applied.
- **Per-model recommendation matrix**: build a small matrix of (model class, KV compression technique, quality preserved, memory saved) so Apple/KAIST/NPU partners can answer "which technique on which model?" without re-running the comparison
- **Inventory additive methodologies** beyond what tern-core already integrates (activation quantisation, structured pruning, layer-wise mixed precision per-expert, sparse attention, knowledge distillation, others)
- **Compose with current per-expert ternary baseline**: which techniques stack cleanly (TurboQuant already proven), which conflict, which require infrastructure work
- **Re-examine TurboQuant default for new model classes** (Qwen3 MoE, Phi-4, future MiniMax) — March benchmarking may not generalise to architectures that didn't exist then
- **Estimate additional compression headroom** when measurable; note where mechanistic assumptions outrun current static-weight measurement (per Thursday's MoE structural-sparsity hedging in the briefing scaffold)

**Memory coverage gap noted:** auto-memory file system at `~/.claude/projects/-Users-syn/memory/` effectively begins 2026-05-01. Pre-May work (including March KV cache benchmarking + the ~300B attempt) lives in earlier conversation context that wasn't persisted. Future sessions should not assume CC has reliable recall of pre-May work without verifying against actual memory traces.

**Resolution scope:** Dedicated session (~2-3 hours) when current sprint cluster work concludes. Could also be triggered earlier if a specific commercial conversation (Apple ANE/CoreML partnership, KAIST/KSGC engagement) creates pressure for tighter compression than today's baseline.

**Why this matters commercially:** Today's 4.9× best ratio means a 60B-class MoE compresses to ~24 GB on M4 Pro 64 GB. If additive methodologies push the on-disk ratio to 6-8×, the unified-memory deployment ceiling rises from 60B-class to 80-120B-class models. Combined with TurboQuant's KV cache reduction (already proven 6× on small models), the total system footprint shrinks proportionally for inference workloads. The per-model recommendation matrix is itself commercially valuable — partners can scope deployments to specific model classes without exploratory benchmarking on their side.

---

## Closed

### reconstruct_all suffix-doubling — production manifest support

**Status:** Closed 2026-05-05  
**Closed by:** `e575dff` `feat(tern_model): production-manifest support — suffix preservation + protection_list helper` (PR #10)  
**Resolution:** Path C conditional in `reconstruct_all` detects per-entry whether the manifest uses test convention (bare names) or production convention (suffixed names) via a bounded `_PRODUCTION_NAME_SUFFIXES` constant; production manifests now load using entry names as-is, test convention preserved unchanged. Verified by canonical gemopus-4-e4b artefact loading with `missing=1, unexpected=54` matching Saturday's Stage C baseline exactly.  
**Related:** `pattern_integration_test_configurational_fidelity_v1.md`; REPORT_PHASE2.md (Diagnostic Journey, third paragraph).

---

### Metal kernel integration into PackedTernaryLinear (Phase 2.5)

**Status:** Closed 2026-05-05  
**Closed by:** Phase 2.5 Stage 1 + Stage 2 commit ladder, merged to main via PR #9. Stage 1: `38e2201` (metal_runtime singleton), `6feef02` (uint32 repack + cross-kernel test), `97ebceb` (dladdr CWD-independent source lookup), `e4e74f0` (sparsity_bitmap shape hard-reject), `89f546f` (Gemma 4 transformers 5.5 multimodal layout). Stage 2: `05a549e`, `3b39a69` (Metal-aware forward), `d7693bd` (production-data e2e test). Phase 4 measurement landed separately on PR #10.  
**Resolution:** TernaryEngine singleton with lazy MPS-resident weight buffer allocation; `PackedTernaryLinear.forward` branches on input device, dispatches Metal kernel for MPS-resident inputs and falls back to the C kernel for CPU inputs. Verified by Row 5 measurement (2.85 tok/s, 1.22 J/token, v1 Metal forward floor) on the canonical gemopus-4-e4b .tern-model artefact.  
**Related:** `pattern_cwd_assumption_in_dylib_loading_v1.md`, `pattern_compatible_types_incompatible_meanings_v1.md`, `pattern_test_path_routing_change_v1.md`; REPORT_PHASE2.md (full report; particularly Diagnostic Journey first/second paragraphs and v1 Metal Forward + Forward Path section).

---

### packed_ops.py MPS fallback — TypeError exception class

**Status:** Closed 2026-05-05  
**Closed by:** `05a549e` `fix(packed_ops): catch TypeError on MPS-resident input fallback` (PR #9 merged)  
**Resolution:** One-token addition of `TypeError` to the existing `except (ImportError, AttributeError)` clause in `packed_ternary_matmul_fast`'s C-kernel fast path; MPS-resident `packed_weights.numpy()` now falls through to the unpack→F.linear reference path. Verified by `tests/test_packed_linear.py::TestMPSFallback::test_mps_input_falls_back_without_crash` using `monkeypatch.setattr("terncore.metal_runtime.get_engine", lambda: None)` to force the F.linear path explicitly post-Commit-4 routing change.  
**Related:** `pattern_test_path_routing_change_v1.md` (banked specifically from this fix's interaction with Commit 4's Metal-aware forward routing); REPORT_PHASE2.md (Reproducibility Appendix lists `05a549e` under Stage 2 commits).
