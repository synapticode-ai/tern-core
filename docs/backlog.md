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

2. **Ternary silent corruption** (surfaced 2026-05-07 during Commit 1 design probe). Same parameter-path traversal: ternary entry `model.layers.0.q_proj.weight` walks to `model.layers.0.q_proj` (the Linear), then `setattr(q_proj, "weight", PackedTernaryLinear_instance)` — **replaces the `weight` Parameter with a Module instance**. Model loads "successfully" with no errors; corruption surfaces at first forward pass as a confusing `__matmul__` / type error. Affects every ternary entry on production manifests (~1,277 on gemopus-4-e4b). More dangerous than FP16 silent skip because failure is delayed and confusing rather than visible at load time.

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

### Llama-3.1-70B-Instruct source weights: download deferred

**Status:** Open (low priority, future-when-needed)
**Surfaced:** 2026-05-06 Wednesday-night queue planning
**Observation:** License accepted on HF Hub 2026-04-09; source weights (~140 GB FP16) never pulled. Only `refs/main` SHA reference cached, no snapshots/blobs. The existing compressed `.tern-model` artefact at `/Volumes/Syn Archive/models/compressed/llama-3-1-70b/llama70b-v0.6.0-mixed.tern-model` (37 GB) already serves as cross-model reference for analysis work (used in Session 4 per-expert tolerance analysis 2026-05-06).
**Resolution scope:** Fire `hf download meta-llama/Llama-3.1-70B-Instruct` when a specific need arises that the existing compressed artefact can't satisfy (e.g., recompression with v2 Metal forward path improvements, full-precision baseline experiments, cross-validation). ~140 GB / ~1-2 hours overnight.

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
