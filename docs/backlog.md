# tern-core backlog

Deferred items surfaced during integration sessions. Each item names the
session of origin, the exact code locations affected, and the recommended
fix scope. Closed items move to a "Closed" section at the bottom.

---

## Open

### load_packed_model rewrite — production manifest support

**Surfaced:** 2026-05-03 Phase 2 Stage C round-trip on production
gemopus-4-e4b artefact. Three issues identified:

1. **FP16 branch handles only module-path manifest naming**
   (`tern_model.py:1111-1124`). Test convention writes
   `add_layer("fc1", ...)` → manifest name `"fc1"`, and the loader walks
   `parts[:-1]` then `getattr(parent, parts[-1])` expecting a Linear
   submodule. Production `convert.py` writes parameter-path naming
   (`<module>.weight`, `<module>.bias`); the same loader code fetches
   `getattr(linear, "weight")` (a Parameter object) and
   `isinstance(Parameter, nn.Linear)` returns False, silently skipping
   853 of 2130 entries on gemopus-4-e4b (852 `.weight` + 1 `.bias`).

2. **`int4_block32` dtype not handled in `load_packed_model`.**
   `reconstruct_layer` handles it via `_reconstruct_int4`
   (`tern_model.py:847-848`), but `load_packed_model` has only
   `ternary2` + `float16` branches. 11 entries silently skipped on
   gemopus-4-e4b.

3. **`key_mapping` translation needed for transformers API drift.**
   Already shipped as parameter on `load_as_model` (2026-05-03);
   `load_packed_model` needs the same parameter for parity. Preset
   `GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` already defined at module
   scope in `tern_model.py`.

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
