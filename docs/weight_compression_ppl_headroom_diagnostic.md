# Weight-Compression PPL Headroom Diagnostic Methodology — R8 v1.1

**Document status:** Canonical (v1.1)
**Authored:** 2026-05-14
**Repository path:** `tern-core/docs/weight_compression_ppl_headroom_diagnostic.md`
**Supersedes:** R8 v1.0 (`docs/ppl_headroom_diagnostic.md`) for the weight-compression scope. R8 v1.0 remains in the repository as documentary record per disposition note §5 (`docs/r8_v1.1_disposition_note.md`).
**Companion documents:**
- `wikitext2_ppl_methodology.md` (R7-A v1.0) — teacher-forcing PPL methodology consumed by §4 step 4
- `r8_v1.1_disposition_note.md` — ratification record for the split into R8 v1.1 (weight) + R12 (KV-cache)
**Scope:** Weight-compression parameter-sweep diagnostic that identifies the maximum-aggressive `threshold` configuration keeping `ppl_headroom` below a target quality ceiling.

---

## §1 Purpose

R8 v1.1 is a parameter-sweep methodology that consumes the R7-A v1.0 teacher-forcing PPL evaluation protocol to characterise the **weight-compression-quality frontier** for a target model. For a fixed model and a target `ppl_headroom_ceiling`, the diagnostic answers:

> **What is the most-aggressive weight-compression configuration that keeps `ppl_headroom` below the ceiling?**

It produces:

1. A **frontier curve** — per-sweep-point `(threshold → ppl_headroom)` mapping
2. A **recommended operating point** — the maximum-aggressive (highest `threshold`) point with `ppl_headroom < ceiling`
3. A **full sweep manifest** — every configuration evaluated, every result captured, every `.tern-model` artefact retained for forensic reproducibility

Outputs consumed by:

- Production deployment configuration selection (which `.tern-model` ships for which deployment tier)
- Patent empirical-evidence collateral (which weight-compression ratios are demonstrated achievable at which quality bands)
- Cross-architecture comparison (TinyLlama frontier vs Llama 3.2 frontier vs Gemma 4 frontier)
- Eagle Bracket 4 / P157 (`.tern-model` native bench harness — R8 v1.1 is one consumer of that harness)
- Eagle Bracket 3 (constant per-token encode cost at scale — weight-compression frontier shows that per-token encode is decoupled from compression-aggressiveness within the operating band)

### §1.1 Distinction from R8 v1.0 and from R12

R8 v1.0 §4 step 1 (commit `d74b093` KNOWN ISSUE) conflated two compression layers under one spec. R8 v1.1 resolves the conflation for the **weight-compression** half:

| Spec | Compression layer | Sweep parameter | Measurement methodology |
|---|---|---|---|
| R8 v1.0 | (conflated — superseded) | — | — |
| **R8 v1.1 (this document)** | Weight (build-time) | `threshold` | R7-A v1.0 teacher-forcing PPL |
| R12 (to draft) | KV cache (runtime) | `b_mse` | R7-B autoregressive PPL (to draft) |

R8 v1.1 and R12 are functionally independent diagnostics. Both can land on a single target model; comparison between the two frontiers is a separate analysis activity.

---

## §2 Inputs

| Input | Type | Required | Notes |
|---|---|---|---|
| `source_model` | HF model_id or local path | yes | The FP16 baseline (re-compressed at each sweep point) |
| `baseline_ppl` | float | yes | Pre-computed FP16 PPL under R7-A v1.0; cached to avoid re-evaluation per sweep point |
| `baseline_run_id` | string | yes | Links per-point JSONs to the canonical baseline JSON |
| `ppl_headroom_ceiling` | float | yes | Upper bound on `ppl_headroom`; sweep terminates when crossed (default 0.50 per R8 v1.1 first execution) |
| `sweep_grid` | dict | yes | Parameter space to enumerate (see §3) |
| `autoscan_mode` | enum {disabled, fixed_per_layer_budget} | yes | Held fixed per sweep; primary R8 v1.1 first execution uses `disabled` (pattern-based protection only) |
| `seed` | int | recommended | Compression-step determinism (forward-pass PPL eval is independent of seed) |
| `output_dir` | path | yes | Where per-point JSONs, `.tern-model` artefacts, and aggregate frontier land |

---

## §3 Sweep grid

**Primary parameter:** `threshold` — the ternarisation-eligibility threshold controlling which weights are converted to ternary representation. Higher `threshold` → more weights eligible for ternarisation → more aggressive compression → larger `ppl_headroom`. Lower `threshold` → fewer weights ternarised → less aggressive compression → smaller `ppl_headroom`.

The parameter lives on `main` at `src/terncore/convert.py TernaryInferenceEngine.convert(threshold: float = 0.7)`.

**Canonical sweep range (R8 v1.1):**

```
threshold ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}
```

10 points. Range chosen to bracket the production default (`threshold=0.7`) with reasonable variance both directions while remaining in the meaningful domain (values below ~0.5 produce essentially no ternarisation; values above ~0.95 produce numerical instability in extreme cases). Adjustable per model family via `sweep_grid` parameter; the canonical range is the documented starting point.

**Secondary parameters held FIXED in v1.1:**

- `autoscan_mode = disabled` — autoscan is NOT used during the sweep. Pattern-based protection (`auto=False` in `TernaryInferenceEngine.convert`) is used uniformly across all sweep points so the only variable is `threshold`. This is a deliberate methodology choice — see §3.1.
- `protection_list` — empty list (relies on pattern-based protection: embed/norm/lm_head)
- Per-layer head_dim variation — per D1 default (commit `5362bf0`)

Secondary parameters MAY be promoted to swept dimensions in v2.0 of this diagnostic; v1.1 deliberately constrains the sweep to a single primary axis for interpretability of the frontier.

### §3.1 Why `autoscan_mode = disabled` for the canonical sweep

If `autoscan_mode = fixed_per_layer_budget` were used during the sweep, autoscan's internal `ppl_headroom` parameter (the autoscan-internal sense per R13 terminology — max PPL degradation per added layer, used as a scan-time gate) would identify different protection lists at different `threshold` values. This produces a *confounded* frontier where `threshold` and "which layers autoscan judged safe" both vary, and the resulting `ppl_headroom` (the diagnostic outcome metric, distinct from autoscan's internal parameter — see R13 backlog item for terminology disambiguation) cannot be attributed cleanly to `threshold` alone.

`autoscan_mode = disabled` keeps the methodology clean: only `threshold` varies; pattern-based protection (embed/norm/lm_head) is held constant. R8 v1.1's canonical sweep is **about characterising `threshold`'s effect on quality**, not about characterising autoscan's effect — those are separable concerns.

**Sweep ordering:** Ascending `threshold` (lowest aggression to highest). Rationale: aligns with monotonic expectation that `ppl_headroom` rises with `threshold`; permits early termination when ceiling crossed.

---

## §4 Per-point procedure

For each `threshold` value in the sweep grid:

1. **Compress** the source FP16 model with `TernaryInferenceEngine.convert(threshold=<sweep_point>, auto=False)`. Produces a `.tern-model` artefact via `TernModelWriter`.
2. **Compute sha256** of the produced `.tern-model` manifest (manifest bytes only — matches `tern_ppl_bench.py` v1.0 convention for `tern_model_manifest_sha256`). Record under `point.tern_model_manifest_sha256`.
3. **Load** the `.tern-model` artefact via R4-C `load_packed_model` zero-copy path. Tokenizer resolves per R7-A v1.0 §3 — either explicit `--tokenizer` override OR manifest-resolved fallback.
4. **Evaluate PPL** per R7-A v1.0 methodology against WikiText-2 test split using `tools/tern_ppl_bench.py` (the R7-A-conformant tool, commit `5e74307`). Produces a results JSON conformant to `wikitext2_ppl/1.0` schema. Pass `--baseline-ppl <baseline_ppl>` and `--baseline-run-id <baseline_run_id>` so the per-run JSON contains `comparison.ppl_headroom` populated for that point.
5. **Compute `ppl_headroom`** against the cached `baseline_ppl`:
   ```
   ppl_headroom = (ppl_ternary - baseline_ppl) / baseline_ppl
   ```
   (Already computed inside `tern_ppl_bench.py` when `--baseline-ppl` is passed; the diagnostic just consumes it.)
6. **Record the point** in the sweep manifest (per §6 schema).
7. **Early-termination check:** if `ppl_headroom > ppl_headroom_ceiling` at the current `threshold`, mark the point `terminated_at_ceiling=true`; continue or stop per §5.

**Artefact retention:** Each per-point `.tern-model` artefact is retained on disk for the duration of the diagnostic run plus a configurable post-run window (default 30 days). Manifest sha256s are recorded in the aggregate output; physical artefacts may be culled later but the methodology requires their existence at point of recording.

**Wall-clock estimate** (TinyLlama-1.1B on M4 Pro MPS, per Phase C measurements):
- Compression per point: ~30-45s (TernaryInferenceEngine.convert wall-clock)
- PPL eval per point: ~150s (R7-A v1.0 sliding-window eval, per Phase C 146.9s wall-clock)
- Total per point: ~3-4 min
- 10-point sweep: ~30-45 min compute (plus baseline-load amortisation in initial iteration)

---

## §5 Stopping criteria

The sweep terminates when ANY of:

1. **Ceiling crossed:** `ppl_headroom > ppl_headroom_ceiling` at the current `threshold`. Mark `sweep.terminated_reason="ceiling_crossed"`.
2. **Grid exhausted:** All `threshold` values in the sweep grid have been evaluated. Mark `sweep.terminated_reason="grid_exhausted"`.
3. **Compression failure:** `TernaryInferenceEngine.convert` fails at a sweep point (e.g. due to numerical instability at extreme `threshold` values close to 1.0). Mark `sweep.terminated_reason="compression_failure"` with `failed_at_threshold` recorded.
4. **PPL eval failure:** `tern_ppl_bench.py` fails (OOM at unexpected scale, model-load failure, etc.). Mark `sweep.terminated_reason="ppl_eval_failure"` with `failed_at_threshold` recorded.

**Continue-past-ceiling mode (diagnostic):** For research-mode runs that want frontier characterisation BEYOND the operating ceiling (e.g. mapping the marginal/fail bands per R7-A §7 threshold bands), the sweep MAY continue past `ppl_headroom_ceiling` if `continue_past_ceiling=true` is set. The recommended-operating-point selection (§7) still respects the original ceiling.

---

## §6 Output schema

The diagnostic emits TWO classes of JSON output:

### §6.1 Per-point JSON (one per sweep point)

```json
{
  "schema_version": "ppl_headroom_weight_point/1.0",
  "diagnostic_run_id": "string (uuid4 — links to aggregate manifest)",
  "point_index": "int (0-indexed sweep position)",
  "config": {
    "threshold": "float",
    "autoscan_mode": "disabled | fixed_per_layer_budget",
    "model_id": "string"
  },
  "tern_model_manifest_sha256": "string",
  "ppl_eval_run_id": "string (links to wikitext2_ppl/1.0 results JSON)",
  "ppl_baseline": "float",
  "ppl_compressed": "float",
  "ppl_headroom": "float (4 sig fig)",
  "ppl_headroom_band": "Excellent | Acceptable | Marginal | Fail",
  "compression_ratio": "float (from TernaryInferenceEngine.convert report)",
  "terminated_at_ceiling": "bool",
  "compression_failed": "bool",
  "notes": "string"
}
```

Filename: `point_<NN>_threshold_<TT>_<timestamp>.json` (e.g. `point_03_threshold_0.65_2026-05-14T041500Z.json`).

### §6.2 Aggregate sweep manifest (one per diagnostic run)

```json
{
  "schema_version": "ppl_headroom_weight_sweep/1.0",
  "diagnostic_run_id": "string (uuid4)",
  "timestamp_utc": "ISO 8601",
  "tern_core_version": "string",
  "tern_core_git_commit": "string (10-char sha)",
  "spec_version": "weight_compression_ppl_headroom_diagnostic v1.1",
  "methodology_consumed": "wikitext2_ppl_methodology v1.0",

  "inputs": {
    "source_model": "string",
    "baseline_ppl": "float",
    "baseline_run_id": "string (links to baseline wikitext2_ppl results JSON)",
    "ppl_headroom_ceiling": "float",
    "sweep_grid": {"threshold": ["float", ...]},
    "autoscan_mode": "disabled | fixed_per_layer_budget",
    "seed": "int | null",
    "continue_past_ceiling": "bool"
  },

  "points": [
    "[references per-point JSONs by filename + point_index]"
  ],

  "frontier": [
    {
      "point_index": "int",
      "threshold": "float",
      "ppl_headroom": "float",
      "ppl_headroom_band": "string",
      "compression_ratio": "float"
    }
  ],

  "recommended_operating_point": {
    "point_index": "int | null",
    "threshold": "float | null",
    "ppl_headroom": "float | null",
    "compression_ratio": "float | null",
    "rationale": "string (e.g. 'maximum threshold with ppl_headroom < ceiling')",
    "tern_model_manifest_sha256": "string | null"
  },

  "termination": {
    "terminated_reason": "ceiling_crossed | grid_exhausted | compression_failure | ppl_eval_failure",
    "terminated_at_point_index": "int | null",
    "failed_at_threshold": "float | null"
  },

  "hardware": {
    "device": "string",
    "compression_wall_time_seconds": "float (total across sweep)",
    "ppl_eval_wall_time_seconds": "float (total across sweep)"
  },

  "notes": "string"
}
```

Filename: `sweep_<model_short>_<autoscan_mode>_<timestamp>.json` (e.g. `sweep_tinyllama-1.1b_disabled_2026-05-14T041500Z.json`).

Stored alongside per-point JSONs in `tern-core/results/ppl_headroom_weight/<diagnostic_run_id>/`. Note the directory split from R12's eventual `results/ppl_headroom_kv_cache/` to keep weight + KV diagnostic outputs cleanly separated.

---

## §7 Recommended operating-point selection

The recommended operating point is selected as:

> **The point with the highest `threshold` such that `ppl_headroom < ppl_headroom_ceiling`.**

If NO point satisfies the constraint (i.e. even the lowest `threshold` exceeds the ceiling), `recommended_operating_point` is `null` and the run is flagged as `compression_infeasible_under_ceiling` in the aggregate's `notes` field.

**Tiebreakers** (in the rare case of two points with identical highest `threshold` admissible — should not occur given monotonic `threshold` sweep but accounted for completeness):

1. Higher `compression_ratio` wins (more aggressive compression at same quality)
2. Lower `ppl_headroom` wins (better quality at same compression)
3. Lower wall-clock compression time wins
4. Lexicographic `point_index` (lower wins) — deterministic fallback

---

## §8 R8 v1.1 first execution — TinyLlama-1.1B threshold sweep

The canonical first execution of R8 v1.1 is a threshold sweep on TinyLlama-1.1B with `ppl_headroom_ceiling=0.50`.

**Rationale for TinyLlama-1.1B as calibration target:**

- v1.0-conformant FP16 baseline already cached on disk from Phase C (commit `77532e8`):
  - `baseline_ppl = 8.0307`
  - `baseline_run_id = 20260514T031257Z`
  - HF revision pinned at `b08601e04326c79dfdd32d625aee71d232d685c3`
- Compression + PPL eval cycle is fast (~3-4 min per sweep point on M4 Pro MPS)
- 10-point sweep fits comfortably in ~30-45 min wall time
- Result feeds v1.1 spec-validation (does the diagnostic produce a sensible frontier?) without committing to longer-running 7B/31B sweeps prematurely

**First-execution parameters:**

```yaml
source_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
baseline_ppl: 8.0307
baseline_run_id: 20260514T031257Z
ppl_headroom_ceiling: 0.50
sweep_grid:
  threshold: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
autoscan_mode: disabled
seed: 1337
continue_past_ceiling: false
```

**Expected behaviour (informational; not pass/fail):**

- `threshold = 0.50` through `0.65`: `ppl_headroom` in Excellent band (<2%); few weights ternarised; minimal quality impact
- `threshold = 0.70` through `0.80`: progressive movement through Acceptable band (2-10%); production-default `0.7` is the canonical operating point and should land Acceptable
- `threshold = 0.85` through `0.90`: Marginal band (10-25%)
- `threshold = 0.95`: likely Fail band (>25%); possibly compression instability at the extreme
- `ceiling_crossed` termination likely somewhere in `threshold = 0.85-0.95` range

These expectations are NOT methodology assertions; they are pre-execution hypotheses to be validated by the first run. Material deviation from this curve shape is itself a research signal worth logging in `notes`.

**Post-execution validation gates:**

1. Aggregate sweep manifest is `ppl_headroom_weight_sweep/1.0` schema-conformant
2. All per-point JSONs are `ppl_headroom_weight_point/1.0` schema-conformant
3. `recommended_operating_point` is non-null AND `threshold >= 0.65` (sanity floor — if even moderate compression fails, methodology or build pipeline is broken)
4. Frontier monotonicity check — `ppl_headroom` increases (non-strictly) with `threshold`; non-monotonic frontier flags potential PPL-eval noise or compression non-determinism worth investigating before lifting the methodology to v1.2
5. Production-default point (`threshold = 0.70`) lands in Acceptable band — sanity check that the current production configuration is in the documented quality zone

---

## §9 Cross-architecture extension (post-R8 v1.1 first execution)

Once R8 v1.1 first execution validates the diagnostic on TinyLlama-1.1B, the methodology is applied across the canonical architecture set:

| Model | Status | Notes |
|---|---|---|
| TinyLlama-1.1B | R8 v1.1 first execution | Calibration target |
| Llama 3.2 1B | Pending R8 v1.1 validation | Independent architecture-family validation; FP16 baseline must be re-established under R7-A v1.0 first |
| Gemma 4 E4B | Pending R8 v1.1 validation; PPL pathology per R9.3 backlog item must be resolved first | Gemma 4 PPL methodology calibration is a hard prerequisite |
| Gemma 4 31B | Pending R8 v1.1 validation; same PPL pathology dependency as E4B | Headline memory-unlock demonstration target |
| Mistral 7B | Pending R8 v1.1 validation | Historical reference architecture |
| Phi-4 14B | Pending R8 v1.1 validation | Mid-scale architecture coverage; no BOS prepend per R7-A v1.0 §4 |

Cross-architecture frontier curves feed:

- Eagle Bracket 4 / P157 empirical evidence (`.tern-model` native bench harness is the consumed tool)
- Patent collateral for cross-family generalisation claims
- Production deployment tier selection (different models for different memory / quality tiers)
- Eagle Bracket 3 (constant per-token encode cost) — secondary corroboration since weight compression aggressiveness should be largely orthogonal to per-token encode cost within a model family

Architecture sweep runs SHOULD reuse the v1.1 sweep grid where possible to permit direct cross-family frontier comparison; deviations MUST be documented in the run's `notes` field and the comparison reported with explicit caveats.

---

## §10 References

- `docs/wikitext2_ppl_methodology.md` (R7-A v1.0) — teacher-forcing PPL methodology consumed by §4 step 4
- `docs/r8_v1.1_disposition_note.md` — disposition record ratifying option (c) split into R8 v1.1 + R12
- `docs/ppl_headroom_diagnostic.md` (R8 v1.0 + KNOWN ISSUE at commit `d74b093`) — superseded for the weight-compression scope; preserved as documentary record
- `results/wikitext2_ppl/ppl_tinyllama-1.1b-chat-v1.0_fp16_20260514T031257Z.json` (commit `77532e8`) — R8 v1.1 baseline anchor (cached on disk from R7-A Phase C)
- `src/terncore/convert.py TernaryInferenceEngine.convert` — main-resident weight compression entry point; `threshold` parameter is R8 v1.1's primary sweep axis
- `tools/tern_ppl_bench.py` (commit `5e74307`) — R7-A v1.0-conformant PPL measurement tool consumed at §4 step 4
- `src/terncore/tern_model.py TernModelReader.load_packed_model` — R4-C zero-copy artefact loader consumed at §4 step 3
- Eagle brackets 1-4 (R4-C era, 2026-05-13, PR #19) — IP claim cluster; R8 v1.1 specifically supports Brackets 3 + 4
- `docs/backlog.md` R13 entry — `ppl_headroom` terminology disambiguation between autoscan-internal sense and R7-A/R8 v1.1 outcome metric

---

*Generated 2026-05-14 — tern-core canonical methodology document. v1.1 supersedes R8 v1.0 for the weight-compression scope; R12 (KV-cache) is the companion diagnostic with its own canonical methodology to be drafted alongside R7-B autoregressive PPL methodology.*
