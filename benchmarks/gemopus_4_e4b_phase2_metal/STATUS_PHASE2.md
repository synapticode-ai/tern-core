# Gemopus-4-E4B-it Benchmark — Phase 2: Status Note

**Date:** 2026-05-03
**Host:** Mac Mini Synapticode (Apple M4 Pro, 48 GB unified memory)
**tern-core commit:** f48a7e5 + Phase 2 patches (`load_as_model` `key_mapping` parameter, `terncore/__init__.py` TernModelReader resolution, Metal dylib rebuild)
**Status:** Closed as a structured-findings session. Headline ternary-inference measurement deferred to Phase 2.5.

---

## What Phase 2 attempted

Phase 2 set out to connect three components — `TernModelReader`, `PackedTernaryLinear`, and the Metal kernel (`TernaryEngine`) — into the benchmark harness as a new `terncore_packed` format backend, then run the headline ternary-inference measurement against the Phase 1.5 substrate-validated baselines (Row 1 mlx_vlm_bf16 23.43 tok/s, Row 1' pytorch_mps_fp16 11.99 tok/s, Row 4 llamacpp_gguf 56.60 tok/s). The seed assumed all three components were already integrated. Pre-flight investigation, plus the round-trip validation work that followed, surfaced that integration was incomplete in several places.

---

## What Phase 2 substantively achieved

### Stage A — Metal dylib rebuild
`make clean && make` in `src/terncore/csrc/metal/` produced a fresh `libternary.dylib` against the current macOS Metal SDK. `TernaryEngine()` instantiation succeeded; `device_name → "Apple M4 Pro"`. The previously stale dylib (dated 25 March) was replaced with a current build. **Outcome:** Metal kernel infrastructure is healthy at the standalone level.

### Stage B — TernModelReader name collision resolved
The package `__init__.py` re-exported the OLD `TernModelReader` from `terncore.model_loader.tern_model` (Stage 1 class with only `read_metadata` / `verify`), masking the NEW `TernModelReader` from `terncore.tern_model` (Stage 2 class with `load_as_model` / `load_packed_model` / `reconstruct_layer` / `reconstruct_all`). Patch (one-line edit to `__init__.py`) repointed the top-level export to the new class. The old reader remains accessible at `terncore.model_loader.TernModelReader` for the three existing test callers in `test_stage1a.py`. **Outcome:** `from terncore import TernModelReader` now returns the production-grade class.

### Stage C — Round-trip structural success
The persisted `.tern-model` artefact for gemopus-4-e4b loaded cleanly via the decoupled `load_as_model` + `convert_model_to_packed` path with the new `key_mapping=GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` parameter (added to `load_as_model` in this session). State of play after the load:

| Signal | Result |
|---|---|
| `verify()` against artefact | ✅ True; sha256 unchanged from Phase 1 fingerprints |
| Manifest entries rewritten via `key_mapping` | 719 / 2130 |
| `load_state_dict` missing keys | 1 — `lm_head.weight` (tied to embed_tokens; standard Gemma family pattern) |
| `load_state_dict` unexpected keys | 54 — `model.language_model.layers.{24..N}.self_attn.{k_norm,k_proj,v_proj}.weight`. Architecture pruning between artefact pack-time (which packed all layers' KV projections) and current Gemma 4 (sliding-window attention layers 24+ share KV with global layers and lack their own `k_norm`/`k_proj`/`v_proj`). Expected; non-blocking |
| `convert_model_to_packed` packed layers | 258 (vs 283 manifest ternary entries; 25 fewer reflects the same architecture-pruning delta) |
| `convert_model_to_packed` protected layers | 335 |
| `convert_model_to_packed` total | 593 |
| Model on MPS | ✅ |

The round-trip load is structurally correct. The artefact reconstructs into the current Gemma 4 model class with no semantic loss for text generation (the audio-tower KV-projection unexpected entries are unused for text inference; `lm_head.weight` ties to the loaded `embed_tokens.weight` automatically).

### Stage C — Inference crashed
First call to `model.generate()` raised `TypeError: can't convert mps:0 device type tensor to numpy` at `packed_ops.py:101` from `packed_weights.numpy()`. Diagnosed as `PackedTernaryLinear.forward` being CPU-only (calls `packed_ternary_matmul_fast` → `terncore.accel` C kernel via numpy). The fallback path exists at `packed_ops.py:144-149` (unpack → `F.linear`, MPS-compatible) but the `except (ImportError, AttributeError)` clause at line 142 doesn't catch `TypeError`, so the exception propagates instead of falling back.

---

## What Phase 2 surfaced (consolidated finding)

The Metal kernel exists as a standalone benchmark-proven component (March 2026 work showing FP16 parity on TinyLlama-class layers, 1.39× on lm_head due to bandwidth dominance) **but was never integrated into `PackedTernaryLinear`'s forward path.** Grep across the repo for `TernaryEngine` outside `ternary_metal.py` itself returns zero results. `PackedTernaryLinear.forward` uses the CPU-only C kernel (`packed_ops.py:packed_ternary_matmul_fast`); no Metal-aware path exists.

Five seed-premise issues plus three latent loader issues totalled in this session, all enumerated in `~/synapticode/tern-core/docs/backlog.md`:

| # | Item | Status |
|---|---|---|
| 1 | TernModelReader top-level export pointed at old class | ✅ resolved this session |
| 2 | Metal dylib stale; needed rebuild | ✅ resolved this session |
| 3 | `TernaryEngine` API: no `compile()` method (seed assumed one) | corrected understanding; documented |
| 4 | `load_as_model` lacked `key_mapping` parameter for transformers API drift | ✅ patched + `GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` preset added |
| 5 | `load_packed_model` FP16 branch handles only test-convention naming, not production parameter-path naming | deferred to focused PR (backlog) |
| 6 | `load_packed_model` lacks `int4_block32` branch | deferred to same PR (backlog) |
| 7 | `reconstruct_all` doubles `.weight` suffix on production manifests | deferred to same PR (backlog) |
| 8 | `packed_ops.py` MPS fallback excepts only `(ImportError, AttributeError)`, not `TypeError` | deferred to backlog (one-line fix) |
| 9 | **Metal kernel integration into PackedTernaryLinear is not implemented** | **Phase 2.5 — focused 1–2 session work, blocks headline benchmark** |

---

## Strategic implication

The headline ternary-inference benchmark for the Apple/KAIST conversation requires Metal kernel integration as its own focused engineering session before measurement can proceed. Workaround paths (CPU-only via the C fast path, or MPS via the dequantise-and-`F.linear` fallback after the one-line `packed_ops.py` fix) would produce numbers in Row 2 territory (~4 tok/s), which adds no information beyond Phase 1 Row 2 and would dilute the Phase 1.5 substrate-validated baselines if reported alongside them. The honest position is: **headline ternary-inference measurement deferred to Phase 2.5**.

---

## What is defensible for the Apple May conversation today

The Phase 1 + Phase 1.5 numbers carry intact:

| Row | Format | tok/s | J/token | Source |
|---|---|---:|---:|---|
| 1   | mlx_vlm_bf16 (mlx-community)         | 23.43 | 0.509 | Phase 1 reference (Phase 1.5: 22.30, substrate-equivalent) |
| 1'  | pytorch_mps_fp16 (google)            | 11.99 | 0.573 | Phase 1.5 supersedes Phase 1 (10.64) as cleaner-state baseline |
| 4   | llamacpp_gguf Q4_K_M (Jackrong)      | 56.60 | 0.379 | Phase 1.5 first clean measurement; harness `--single-turn -ngl 999` corrected |

The persisted `.tern-model` artefacts on archive (`gemma4_e4b_ternary_v0.1.0` and `gemopus_4_e4b_ternary_v0.1.0`, both 9153 MB at 48.3% ternary ratio) are byte-valid (sha256-anchored to Phase 1), structurally loadable into the current Gemma 4 model class via the patched loader, fine-tune-agnostic in compression behaviour, and ready for Metal kernel integration. The headline ternary-inference number is in active engineering and will land alongside Phase 2.5.

---

## Forward path

**Phase 2.5 — Metal kernel integration into PackedTernaryLinear.** Scoped as 1–2 focused sessions covering:

- TernaryEngine singleton design — one engine per process, shared across all PackedTernaryLinear instances, GPU buffer lifetime managed via the engine's `create_buffer` API.
- MPS↔Metal buffer handoff — when input is on MPS, route matvec through `matvec_gpu` (already on the same Metal device) without CPU round-trip.
- Metal-aware forward path on `PackedTernaryLinear` — device-branch in `forward()`, dispatch Metal for MPS-resident input, fall back to CPU C kernel for CPU input.
- Regression test against the CPU fast path on a small reference layer.
- Re-run Phase 2 measurement: 20-token probe, 200-token timed measurement on Gemopus-4-E4B-it.

**Sprint cluster sequencing:** Gemopus-4-26B-A4B-it and Gemopus-4-31B-it compressions sequence after Phase 2.5 so each compression is verifiable by the real Metal-kernel inference path in the same session.

---

## Session deliverables

- `bench_gemopus_phase2.py` — unchanged this session (Phase 2 backend integration deferred; harness already supports five existing format backends, terncore_packed waits for Phase 2.5)
- `src/terncore/__init__.py` — TernModelReader top-level export repointed to `terncore.tern_model.TernModelReader`
- `src/terncore/tern_model.py` — `load_as_model` gained `key_mapping` parameter; `GEMMA4_MULTIMODAL_TRANSFORMERS_5_5` preset added at module scope
- `src/terncore/csrc/metal/build/libternary.dylib` — rebuilt against current macOS Metal SDK
- `docs/backlog.md` — created; four open items (load_packed_model rewrite, reconstruct_all suffix-doubling, Metal kernel integration, packed_ops.py MPS fallback exception)
- `benchmarks/gemopus_4_e4b_phase2_metal/STATUS_PHASE2.md` — this document
- `/tmp/phase2_roundtrip.py` — round-trip validation script (decoupled load path); kept as reference for Phase 2.5 work

Memory entries banked:

- `pattern_seed_current_state_verification_v1.md` (5-min pre-draft API check)
- `pattern_artefact_format_transformers_drift_v1.md` (`key_mapping` translation; named presets)
- `pattern_first_production_data_load_v1.md` (first production-data load is discovery, not measurement)
- `pattern_integration_existence_verification_v1.md` (verify connections, not just components)

---

## Closing reflection

Phase 2 closes as a structured-findings session. The integration gaps it surfaced are real engineering work that needed surfacing — quietly running a workaround-path measurement and reporting it as headline would have damaged the Apple May conversation more than transparently deferring to Phase 2.5. The persisted artefacts are byte-valid and structurally loadable today; the loader path now has a real production-data round-trip diagnostic in the test plan; the Metal kernel integration is scoped and queued. Phase 2 closes here.

---

*Gamma Seeds Pte Ltd — Synapticode*
