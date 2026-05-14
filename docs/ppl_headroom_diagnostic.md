# PPL Headroom Diagnostic Methodology

**Document status:** Canonical (v1.0)
**Authored:** 2026-05-14
**Repository path:** `tern-core/docs/ppl_headroom_diagnostic.md`
**Companion document:** `wikitext2_ppl_methodology.md` (R7-A; defines the PPL measurement protocol this diagnostic consumes)
**Scope:** Methodology for sweeping ternary-compression parameters against a `ppl_headroom` ceiling to identify the maximum-compression configuration that remains within a target quality envelope.

---

## §1 Purpose

The `ppl_headroom` diagnostic is a parameter-sweep methodology that consumes the WikiText-2 PPL evaluation protocol (R7-A) to characterise the compression-quality frontier for a target model. For a fixed model and a target `ppl_headroom_ceiling`, the diagnostic answers:

> **What is the most-aggressive compression configuration that keeps `ppl_headroom` below the ceiling?**

It produces:

1. A **frontier curve** — per-sweep-point `(compression_config → ppl_headroom)` mapping
2. A **recommended operating point** — the maximum-compression point with `ppl_headroom < ceiling`
3. A **full sweep manifest** — every configuration evaluated, every result captured, every `.tern-model` artefact retained for forensic reproducibility

Outputs are consumed by:

- Production deployment configuration selection (which `.tern-model` ships for which deployment tier)
- Patent empirical-evidence collateral (which compression ratios are demonstrated achievable at which quality bands)
- Cross-architecture comparison (TinyLlama frontier vs Llama 3.2 frontier vs Gemma 4 frontier)
- The Eagle "constant per-token encode cost at scale" bracket (Bracket 3 from R4-C era) — frontier data corroborates compression-independent encode timing

---

## §2 Inputs

| Input | Type | Required | Notes |
|---|---|---|---|
| `source_model` | HF model_id or local path | yes | The FP16 baseline (re-compressed at each sweep point) |
| `baseline_ppl` | float | yes | Pre-computed FP16 PPL per R7-A v1.0 methodology; cached to avoid re-evaluation per sweep point |
| `ppl_headroom_ceiling` | float | yes | Upper bound on `ppl_headroom`; sweep terminates when crossed (default 0.50 per R8 first execution) |
| `sweep_grid` | dict | yes | Parameter space to enumerate (see §3) |
| `kv_pack_mode` | enum {OPT-A, OPT-B} | yes | Held fixed per sweep; primary R8 first execution uses OPT-B |
| `seed` | int | recommended | Compression-step determinism (FP16 PPL eval is independent of seed) |
| `output_dir` | path | yes | Where per-point JSONs, `.tern-model` artefacts, and aggregate frontier land |

---

## §3 Sweep grid

**Primary parameter:** `b_mse` — the bit-MSE control parameter for ternary compression aggressiveness. Lower `b_mse` → looser compression → lower `ppl_headroom`. Higher `b_mse` → tighter compression → higher `ppl_headroom`.

**Canonical sweep range (R8 v1.0):**

```
b_mse ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0}
```

11 points. Adjustable per model family via `sweep_grid` parameter; the canonical range is the documented starting point.

**Secondary parameters held FIXED in v1.0:**

- `ternary_ratio` — held at the model's existing calibrated value (e.g. 96.4% for TinyLlama-1.1B at b_mse=3)
- `kv_pack_mode` — held at OPT-B for primary sweep
- Per-layer head_dim variation — per D1 default (commit `5362bf0`)

Secondary parameters MAY be promoted to swept dimensions in v2.0 of this diagnostic; v1.0 deliberately constrains the sweep to a single primary axis for interpretability of the frontier.

**Sweep ordering:** Ascending `b_mse` (loosest to tightest compression). Rationale: aligns with monotonic expectation that `ppl_headroom` rises with `b_mse`; permits early termination when ceiling crossed.

---

## §4 Per-point procedure

For each `b_mse` value in the sweep grid:

1. **Compress** the source FP16 model with the configuration `{b_mse, ternary_ratio, kv_pack_mode, ...}`. Produces a `.tern-model` artefact.
2. **Compute sha256** of the produced `.tern-model` manifest. Record under `point.tern_model_manifest_sha256`.
3. **Load** the `.tern-model` artefact via R4-C `load_packed_model` zero-copy path.
4. **Evaluate PPL** per R7-A v1.0 methodology against WikiText-2 test split. Produces a results JSON conformant to `wikitext2_ppl/1.0` schema.
5. **Compute `ppl_headroom`** against the cached `baseline_ppl`.
6. **Record the point** in the sweep manifest (per §6 schema).
7. **Early-termination check:** if `ppl_headroom > ppl_headroom_ceiling`, mark the point `terminated_at_ceiling=true`; continue or stop per §5.

**Artefact retention:** Each per-point `.tern-model` artefact is retained on disk for the duration of the diagnostic run plus a configurable post-run window (default 30 days). Manifest sha256s are recorded in the aggregate output; physical artefacts may be culled later but the methodology requires their existence at point of recording.

---

## §5 Stopping criteria

The sweep terminates when ANY of:

1. **Ceiling crossed:** `ppl_headroom > ppl_headroom_ceiling` at the current `b_mse`. Mark `sweep.terminated_reason="ceiling_crossed"`.
2. **Grid exhausted:** All `b_mse` values in the sweep grid have been evaluated. Mark `sweep.terminated_reason="grid_exhausted"`.
3. **Compression failure:** `.tern-model` build fails at a sweep point (e.g. due to numerical instability at extreme `b_mse`). Mark `sweep.terminated_reason="compression_failure"` with `failed_at_b_mse` recorded.

**Continue-past-ceiling mode (diagnostic):** For research-mode runs that want frontier characterisation BEYOND the operating ceiling (e.g. mapping the marginal/fail bands per R7-A §7 threshold bands), the sweep MAY continue past `ppl_headroom_ceiling` if `continue_past_ceiling=true` is set. The recommended-operating-point selection (§7) still respects the original ceiling.

---

## §6 Output schema

The diagnostic emits TWO classes of JSON output:

### §6.1 Per-point JSON (one per sweep point)

```json
{
  "schema_version": "ppl_headroom_point/1.0",
  "diagnostic_run_id": "string (uuid4 — links to aggregate manifest)",
  "point_index": "int (0-indexed sweep position)",
  "config": {
    "b_mse": "float",
    "ternary_ratio": "float",
    "kv_pack_mode": "OPT-A | OPT-B",
    "model_id": "string"
  },
  "tern_model_manifest_sha256": "string",
  "ppl_eval_run_id": "string (links to wikitext2_ppl/1.0 results JSON)",
  "ppl_baseline": "float",
  "ppl_compressed": "float",
  "ppl_headroom": "float (4 sig fig)",
  "ppl_headroom_band": "Excellent | Acceptable | Marginal | Fail",
  "terminated_at_ceiling": "bool",
  "compression_failed": "bool",
  "notes": "string"
}
```

Filename: `point_<NN>_b_mse_<BMSE>_<timestamp>.json` (e.g. `point_03_b_mse_2.5_2026-05-14T021500Z.json`).

### §6.2 Aggregate sweep manifest (one per diagnostic run)

```json
{
  "schema_version": "ppl_headroom_sweep/1.0",
  "diagnostic_run_id": "string (uuid4)",
  "timestamp_utc": "ISO 8601",
  "tern_core_version": "string",
  "tern_core_git_commit": "string (10-char sha)",
  "spec_version": "ppl_headroom_diagnostic v1.0",
  "methodology_consumed": "wikitext2_ppl_methodology v1.0",

  "inputs": {
    "source_model": "string",
    "baseline_ppl": "float",
    "baseline_run_id": "string (links to baseline wikitext2_ppl results JSON)",
    "ppl_headroom_ceiling": "float",
    "sweep_grid": {"b_mse": ["float", ...]},
    "kv_pack_mode": "OPT-A | OPT-B",
    "seed": "int | null",
    "continue_past_ceiling": "bool"
  },

  "points": [
    "[references per-point JSONs by filename + point_index]"
  ],

  "frontier": [
    {
      "point_index": "int",
      "b_mse": "float",
      "ppl_headroom": "float",
      "ppl_headroom_band": "string"
    }
  ],

  "recommended_operating_point": {
    "point_index": "int | null",
    "b_mse": "float | null",
    "ppl_headroom": "float | null",
    "rationale": "string (e.g. 'maximum b_mse with ppl_headroom < ceiling')",
    "tern_model_manifest_sha256": "string | null"
  },

  "termination": {
    "terminated_reason": "ceiling_crossed | grid_exhausted | compression_failure",
    "terminated_at_point_index": "int | null",
    "failed_at_b_mse": "float | null"
  },

  "hardware": {
    "device": "string",
    "compression_wall_time_seconds": "float (total across sweep)",
    "ppl_eval_wall_time_seconds": "float (total across sweep)"
  },

  "notes": "string"
}
```

Filename: `sweep_<model_short>_<kv_pack_mode>_<timestamp>.json` (e.g. `sweep_tinyllama-1.1b_OPT-B_2026-05-14T021500Z.json`).

Stored alongside per-point JSONs in `tern-core/results/ppl_headroom/<diagnostic_run_id>/`.

---

## §7 Recommended operating-point selection

The recommended operating point is selected as:

> **The point with the highest `b_mse` such that `ppl_headroom < ppl_headroom_ceiling`.**

If NO point satisfies the constraint (i.e. even the lowest `b_mse` exceeds the ceiling), `recommended_operating_point` is `null` and the run is flagged as `compression_infeasible_under_ceiling` in the aggregate's `notes` field.

**Tiebreakers** (in the rare case of two points with identical highest `b_mse` admissible — should not occur given monotonic `b_mse` sweep but accounted for completeness):

1. Lower `ppl_headroom` wins
2. Lower wall-clock compression time wins
3. Lexicographic `point_index` (lower wins) — deterministic fallback

---

## §8 R8 v1.0 first execution — TinyLlama-1.1B re-scan

The canonical first execution of this diagnostic is a re-scan of TinyLlama-1.1B with `ppl_headroom_ceiling=0.50`.

**Rationale for TinyLlama-1.1B as calibration target:**

- Well-characterised under prior compression work (existing anchor: `b_mse=3` → `ppl_headroom=0.0409` per `tq_bench_results.json`, 2026-03-30)
- Compression + PPL eval cycle is fast (~1-3 min per sweep point on M4 Pro MPS)
- 11-point sweep fits comfortably in ~30-45 min wall time
- Result feeds R8 v1.0 spec-validation (does the diagnostic produce a sensible frontier?) without committing to longer-running 7B/31B sweeps prematurely

**First-execution parameters:**

```yaml
source_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0  # or the exact model_id Rob has on disk
baseline_ppl: <re-compute under R7-A v1.0 first, cache; the 7.82 figure was pre-v1.0>
ppl_headroom_ceiling: 0.50
sweep_grid:
  b_mse: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
kv_pack_mode: OPT-B
seed: 1337
continue_past_ceiling: false  # standard sweep; flip to true for frontier-mapping diagnostic mode
```

**Expected behaviour (informational; not pass/fail):**

- `b_mse=1.0` through `b_mse=3.0`: `ppl_headroom` in Excellent / Acceptable bands
- `b_mse=3.5` through `b_mse=5.0`: progressive movement into Marginal band
- `b_mse=5.5` to `6.0`: likely Fail band; possibly compression instability at the extreme
- `ceiling_crossed` termination likely somewhere in `b_mse=4.5-5.5` range

These expectations are NOT methodology assertions; they are pre-execution hypotheses to be validated by the first run. Material deviation from this curve shape is itself a research signal worth logging in `notes`.

**Post-execution validation gates:**

1. Aggregate sweep manifest is `ppl_headroom_sweep/1.0` schema-conformant
2. All per-point JSONs are `ppl_headroom_point/1.0` schema-conformant
3. `recommended_operating_point` is non-null AND `b_mse >= 2.5` (sanity floor — if even moderate compression fails, methodology or build pipeline is broken)
4. Frontier monotonicity check — `ppl_headroom` increases (non-strictly) with `b_mse`; non-monotonic frontier flags potential PPL-eval noise or compression non-determinism worth investigating before lifting the methodology to v1.1

---

## §9 Cross-architecture extension (post-R8 v1.0)

Once R8 v1.0 first execution validates the diagnostic on TinyLlama-1.1B, the methodology is applied across the canonical architecture set:

| Model | Status | Notes |
|---|---|---|
| TinyLlama-1.1B | R8 v1.0 first execution | Calibration target |
| Llama 3.2 1B | Pending R8 v1.0 validation | Independent architecture-family validation |
| Gemma 4 E4B | Pending R10 (KV-sharing adapter) | Blocked until per-layer K/V-sharing layout adapter lands |
| Gemma 4 31B | Pending R8 v1.0 validation | Headline memory-unlock demonstration target |
| Mistral 7B | Pending R8 v1.0 validation | Historical reference architecture |
| Phi-4 14B | Pending R8 v1.0 validation | Mid-scale architecture coverage |

Cross-architecture frontier curves feed:

- Eagle bracket 3 ("constant per-token encode cost at scale") empirical corroboration
- Patent collateral for cross-family generalisation claims
- Production deployment tier selection (different models for different memory / quality tiers)

Architecture sweep runs SHOULD reuse the v1.0 sweep grid where possible to permit direct cross-family frontier comparison; deviations MUST be documented in the run's `notes` field and the comparison reported with explicit caveats.

---

## §10 References

- `wikitext2_ppl_methodology.md` (R7-A, v1.0) — PPL measurement protocol consumed by this diagnostic
- tern-core `tq_bench_results.json` (2026-03-30) — TinyLlama-1.1B pre-v1.0 anchor data (`b_mse=3`, `ppl_headroom=0.0409`)
- tern-core R1+R6 calibration work (commits `ba71d30`, `df4c141`, 2026-05-13) — OPT-A / OPT-B packed-bits methodology + long-seq convergence validation
- tern-core R4-C native `.tern-model` bench harness (PR #19) — `load_packed_model` zero-copy artefact loading path consumed at §4 step 3
- R7-A §7 threshold bands — Excellent / Acceptable / Marginal / Fail dispositions referenced in `ppl_headroom_band` schema field

---

*Generated 2026-05-14 — tern-core canonical diagnostic methodology document. v1.0 first execution is the TinyLlama-1.1B re-scan per §8; results inform v1.1 refinement scope before cross-architecture extension (§9).*
