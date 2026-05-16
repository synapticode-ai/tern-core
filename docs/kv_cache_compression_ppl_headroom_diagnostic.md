# KV-Cache-Compression PPL Headroom Diagnostic Methodology — R12 v1.0

**Document status:** v1.0 (ratified 2026-05-15) — promoted from v0.2 with §10 local-path-strip cleanup. v0.1 and v0.2 drafts retained at `ecc-ternary/uploads/R12_KV_CACHE_DIAGNOSTIC_SPEC_DRAFT_20260515T022418Z/` as historical record. Full change inventory at §11.
**Authored:** 2026-05-15
**Repository path (on ratification):** `tern-core/docs/kv_cache_compression_ppl_headroom_diagnostic.md`
**Companion documents:**
- `wikitext2_ppl_methodology_autoregressive.md` (R7-B v1.0, PR #23) — autoregressive PPL methodology consumed by §4 step 3
- `weight_compression_ppl_headroom_diagnostic.md` (R8 v1.1, PR #20) — sibling diagnostic for weight-compression scope; R12 mirrors its structural pattern §1-§9
- `r8_v1.1_disposition_note.md` (PR #20) — ratification record for the R8 v1.0 split into R8 v1.1 (weight) + R12 (KV-cache, this document)

**Scope:** KV-cache-compression parameter-sweep diagnostic that identifies the maximum-aggressive `b_mse` configuration keeping `ppl_headroom` below a target quality ceiling under R7-B v1.0 autoregressive methodology.

---

## §1 Purpose

R12 v1.0 is a parameter-sweep methodology that consumes the R7-B v1.0 autoregressive PPL evaluation protocol to characterise the **KV-cache-compression-quality frontier** for a target model. For a fixed model and a target `ppl_headroom_ceiling`, the diagnostic answers:

> **What is the most-aggressive KV-cache-compression configuration that keeps `ppl_headroom` below the ceiling?**

It produces:

1. A **frontier curve** — per-sweep-point `(b_mse → ppl_headroom)` mapping
2. A **recommended operating point** — the maximum-aggressive (lowest `b_mse`) point with `ppl_headroom < ceiling`
3. A **full sweep manifest** — every configuration evaluated, every result captured, every per-point JSON retained for forensic reproducibility

Outputs consumed by:

- Production deployment configuration selection (which `b_mse` ships for which deployment tier)
- Patent empirical-evidence collateral (which KV-cache-compression ratios are demonstrated achievable at which quality bands)
- Cross-architecture comparison (TinyLlama KV-cache frontier vs Llama 3.2 vs Gemma 4 vs Mistral 7B vs Phi-4 14B)
- **Eagle Bracket 2** (deployment-tier selection — R12 supplies the per-tier `b_mse` choice for memory/quality tradeoffs) **and Eagle Bracket 3** (constant per-token encode cost at scale — KV-cache compression directly affects per-token encode as cache grows; R12 supplies the quality-band attribution that complements Bracket 3's wall-clock evidence)

### §1.1 Distinction from R8 v1.1 and from the 12 May TQ bench anchor

R8 v1.0 (commit `d74b093` KNOWN ISSUE) conflated two compression layers under one spec. R8 v1.1 and R12 v1.0 together resolve the conflation; R12 handles the **KV-cache** half:

| Spec | Compression layer | Sweep parameter | Measurement methodology | Per-point cost |
|---|---|---|---|---|
| R8 v1.0 | (conflated — superseded) | — | — | — |
| R8 v1.1 (PR #20) | Weight (build-time) | `threshold` | R7-A v1.0 teacher-forcing PPL | ~3-4 min (TinyLlama M4 Pro MPS) |
| **R12 v1.0 (this document)** | KV cache (runtime) | `b_mse` | R7-B v1.0 autoregressive PPL (PR #23) | ~27 min (TinyLlama M4 Pro MPS) |

The 12 May `b_mse=3` PPL anchor in `benchmarks/tq_bench_results.json` (2026-03-30, `ppl_baseline=7.82 → ppl_ternary=8.14`) was measured under "5 sentences, single forward pass each" — teacher-forcing on short isolated sequences with no inter-sentence cache reuse. That methodology does NOT exercise KV-cache compression effects (per R7-B v1.0 §1.2). The 12 May anchor is preserved as documentary record but is NOT R12-conformant; R12 v1.0 first execution re-establishes the `b_mse` anchor under R7-B v1.0 methodology, which is the only way to legitimately attribute PPL deltas to KV-cache compression.

R8 v1.1 and R12 v1.0 are functionally independent diagnostics. Both can land on a single target model; comparison between the two frontiers is a separate analysis activity. Combined-frontier analysis (joint weight + KV-cache sweep) is out of scope for R12 v1.0 — proposed for a future v2.0 alongside an R8 v2.0 equivalent.

---

## §2 Inputs

| Input | Type | Required | Notes |
|---|---|---|---|
| `source_model` | HF model_id or local path | yes | The FP16 baseline. R12 v1.0 does NOT recompress weights per sweep point; KV cache compression is the only varied dimension. |
| `baseline_ppl_r7b` | float | yes | Pre-computed FP16 PPL under R7-B v1.0 (NOT R7-A — must be R7-B-conformant per R7-B §1.1 invariant). Cached to avoid re-evaluation per sweep point. |
| `baseline_run_id` | string | yes | Links per-point JSONs to the canonical R7-B baseline JSON. |
| `ppl_headroom_ceiling` | float | yes | Upper bound on `ppl_headroom`; sweep terminates when crossed (default 0.50 per R12 v1.0 first execution, matching R8 v1.1 convention). |
| `sweep_grid` | dict | yes | Parameter space to enumerate (see §3). |
| `num_sequences` | int | yes | R7-B v1.0 §4 canonical N=16. Held constant across sweep points for direct comparability. |
| `seq_len` | int | yes | R7-B v1.0 §4 canonical L=2048. Held constant. |
| `seed` | int | recommended | Sequence-sampling determinism per R7-B v1.0 §4. |
| `output_dir` | path | yes | Where per-point JSONs and aggregate frontier land. |

---

## §3 Sweep grid

**Primary parameter:** `b_mse` — the bits-of-MSE parameter controlling KV-cache quantisation aggressiveness via `IncrementalTQCompressor`. Lower `b_mse` → more aggressive compression → larger `ppl_headroom`. Higher `b_mse` → less aggressive compression → smaller `ppl_headroom`. The semantic is inverted relative to R8 v1.1's `threshold` (where higher → more aggressive); the methodology accommodates by stating ordering explicitly in §3.1.

The parameter surfaces in R12 via R7-B v1.0 §5.2's planned `make_b_mse_hook(b_mse: int)` factory, which constructs a `kv_cache_hook` callable invoking `IncrementalTQCompressor` between forward passes in the R7-B canonical loop. **Implementation prerequisite:** `IncrementalTQCompressor` currently hardcodes `b_mse=3` at construction time; the factory + parameterisation refactor must land first (see §8.1).

**Canonical sweep range (R12 v1.0):**

```
b_mse ∈ {6, 5, 4, 3, 2, 1}
```

6 points. Range chosen to bracket the 12 May empirical anchor (`b_mse=3`) with reasonable variance both directions while remaining in the meaningful domain (`b_mse=1` is extreme compression likely producing high-headroom signal; `b_mse=6` approaches identity within numerical precision). Adjustable per model family via `sweep_grid` parameter; the canonical range is the documented starting point.

**Secondary parameters held FIXED in v1.0:**

- `weight_threshold` — held at FP16 (no ternary weight compression at any sweep point). R12 v1.0 isolates KV-cache effects cleanly by keeping weights unchanged. See §3.2.
- `protection_list` — N/A (R12 doesn't touch weights)
- `kv_cache_hook` application scope — applied to ALL layers' KV pairs uniformly. Per-layer selective compression deferred to v2.0.
- R7-B v1.0 sequence-construction parameters (N=16, L=2048, seed) — held constant across sweep points so the only variable is `b_mse`.

Secondary parameters MAY be promoted to swept dimensions in v2.0 of this diagnostic; v1.0 deliberately constrains the sweep to a single primary axis for interpretability of the frontier.

### §3.1 Sweep ordering

Descending `b_mse` (highest `b_mse` first, lowest last) — equivalent to ascending compression aggressiveness, matching R8 v1.1's convention of "lowest aggression to highest". Rationale: aligns with monotonic expectation that `ppl_headroom` rises as `b_mse` decreases; permits early termination when ceiling crossed.

### §3.2 Why `weight_threshold = FP16` for the canonical v1.0 sweep

Mixing weight compression with KV-cache compression produces a confounded frontier where `b_mse` and weight-threshold both affect `ppl_headroom`, and the resulting signal cannot be attributed cleanly to KV-cache compression alone. Keeping weights at FP16 isolates the KV-cache-compression effect; per-axis methodology is the right v1.0 scope. Combined-frontier analysis (joint sweep over `threshold` × `b_mse`) is proposed for v2.0 of this diagnostic, alongside an R8 v2.0 equivalent.

This parallels R8 v1.1 §3.1's rationale for holding `autoscan_mode=disabled` during the canonical sweep — keep one dimension swept, the rest held constant, so the resulting signal attributes cleanly to the swept axis.

### §3.3 v1.0-simplification disclaimer

R12 v1.0 deliberately holds multiple dimensions fixed (`weight_threshold=FP16`, `kv_hook_application_scope=all_layers`, R7-B sequence parameters N/L/seed) for clean signal attribution to the `b_mse` axis. Joint weight × KV-cache frontier sweeps + per-layer selective compression scope are proposed for v2.0; both are deferred specifically because v1.0's job is to characterise the single-axis baseline first. Future cascade revisions (v1.x or v2.0) may promote any of these to a swept dimension if doing so would clarify a specific architecture's behaviour — at the cost of frontier-curve readability.

---

## §4 Per-point procedure

For each `b_mse` value in the sweep grid:

1. **Load** source FP16 model + tokenizer per R7-B v1.0 §3 (shared with R7-A v1.0 §3 — same conventions).
2. **Construct** the KV-cache compression hook via `make_b_mse_hook(b_mse=<sweep_point>)` per R7-B v1.0 §5.2 (factory to be authored per §8.1 implementation prerequisite).
3. **Evaluate PPL** per R7-B v1.0 methodology using `tools/tern_kv_ppl_bench.py` (to be authored as the R12-conformant tool, analogous to `tools/tern_ppl_bench.py` for R7-A). The hook is injected into `autoregressive_ppl(..., kv_cache_hook=hook)` per R7-B v1.0 §5 canonical loop. Tokeniser resolves per R7-B v1.0 §3.
4. **Compute `ppl_headroom`** against the cached `baseline_ppl_r7b`:
   ```
   ppl_headroom = (ppl_kv_compressed - baseline_ppl_r7b) / baseline_ppl_r7b
   ```
   (Computed inside `tern_kv_ppl_bench.py` when `--baseline-ppl` and `--baseline-run-id` are passed; the diagnostic consumes it.)
5. **Compute `kv_cache_compression_ratio`** externally — `IncrementalTQCompressor` does not natively surface this metric (verified 2026-05-15 against `tools/tern_infer.py:148`). `tern_kv_ppl_bench.py` measures `past_key_values` byte-size pre-hook and post-hook for the same forward call, then reports the ratio per the §6.1 schema field.
6. **Record the point** in the sweep manifest (per §6 schema).
7. **Early-termination check:** if `ppl_headroom > ppl_headroom_ceiling` at the current `b_mse`, mark `terminated_at_ceiling=true`; continue or stop per §5.

**No per-point compression artefact:** Unlike R8 v1.1 (which produces a per-point `.tern-model` via `TernModelWriter`), R12 v1.0 does not generate per-point artefacts. The varied dimension is runtime (hook application), not build-time. `kv_cache_hook` reconstruction per point is sub-second. The model weights remain on disk in their canonical FP16 form across the entire sweep.

**Wall-clock estimate** (TinyLlama-1.1B on M4 Pro MPS, per R7-B v1.0 §1 measurements):
- R7-B autoregressive PPL per point: ~27 min (N=16 × L=2048 = 32,768 forward calls)
- Hook construction + ratio measurement per point: <1s
- 6-point sweep: ~2.7-3 hours compute (substantially longer than R8 v1.1's ~30-45 min due to R7-B's autoregressive cost vs R7-A's teacher-forcing single-pass cost)

This is the methodological cost of legitimately exercising KV-cache compression: only autoregressive evaluation surfaces the cache-mediated quality signal (per R7-B v1.0 §1). A faster non-autoregressive approximation would forfeit R7-B-conformance and revert to the 12 May methodology shape that R7-B §1.2 explicitly retires.

---

## §5 Stopping criteria

The sweep terminates when ANY of:

1. **Ceiling crossed:** `ppl_headroom > ppl_headroom_ceiling` at the current `b_mse`. Mark `sweep.terminated_reason="ceiling_crossed"`.
2. **Grid exhausted:** All `b_mse` values in the sweep grid have been evaluated. Mark `sweep.terminated_reason="grid_exhausted"`.
3. **Hook construction failure:** `make_b_mse_hook` fails at an extreme `b_mse` value (e.g. `b_mse=0` or below the implementation's supported floor). Mark `sweep.terminated_reason="hook_construction_failure"` with `failed_at_b_mse` recorded.
4. **PPL eval failure:** `tern_kv_ppl_bench.py` fails (OOM at unexpected KV-cache scale on large models, model-load failure, numerical instability in the cache compressor, etc.). Mark `sweep.terminated_reason="ppl_eval_failure"` with `failed_at_b_mse` recorded.

**Continue-past-ceiling mode (diagnostic):** For research-mode runs that want frontier characterisation BEYOND the operating ceiling (e.g. mapping the marginal/fail bands per R7-A §7 threshold bands inherited by R7-B and R12), the sweep MAY continue past `ppl_headroom_ceiling` if `continue_past_ceiling=true` is set. The recommended-operating-point selection (§7) still respects the original ceiling.

---

## §6 Output schema

The diagnostic emits TWO classes of JSON output:

### §6.1 Per-point JSON (one per sweep point)

```json
{
  "schema_version": "ppl_headroom_kv_cache_point/1.0",
  "diagnostic_run_id": "string (uuid4 — links to aggregate manifest)",
  "point_index": "int (0-indexed sweep position)",
  "config": {
    "b_mse": "int",
    "weight_threshold": "FP16",
    "kv_hook_application_scope": "all_layers",
    "num_sequences": "int",
    "seq_len": "int",
    "model_id": "string"
  },
  "ppl_eval_run_id": "string (links to wikitext2_ppl_autoregressive/1.0 results JSON)",
  "baseline_ppl_r7b": "float",
  "ppl_kv_compressed": "float",
  "ppl_headroom": "float (4 sig fig)",
  "ppl_headroom_band": "Excellent | Acceptable | Marginal | Fail",
  "kv_cache_compression_ratio": "float (size_before / size_after, computed by tern_kv_ppl_bench.py from pre/post past_key_values byte sizes; IncrementalTQCompressor does NOT natively surface this — see §10 reference and §8.1 implementation prerequisite)",
  "terminated_at_ceiling": "bool",
  "hook_construction_failed": "bool",
  "notes": "string"
}
```

Filename: `point_<NN>_b_mse_<B>_<timestamp>.json` (e.g. `point_03_b_mse_3_2026-05-15T040000Z.json`).

### §6.2 Aggregate sweep manifest (one per diagnostic run)

```json
{
  "schema_version": "ppl_headroom_kv_cache_sweep/1.0",
  "diagnostic_run_id": "string (uuid4)",
  "timestamp_utc": "ISO 8601",
  "tern_core_version": "string",
  "tern_core_git_commit": "string (10-char sha)",
  "spec_version": "kv_cache_compression_ppl_headroom_diagnostic v1.0",
  "methodology_consumed": "wikitext2_ppl_methodology_autoregressive v1.0",

  "inputs": {
    "source_model": "string",
    "baseline_ppl_r7b": "float",
    "baseline_run_id": "string (links to baseline wikitext2_ppl_autoregressive results JSON)",
    "ppl_headroom_ceiling": "float",
    "sweep_grid": {"b_mse": ["int", ...]},
    "num_sequences": "int",
    "seq_len": "int",
    "seed": "int | null",
    "continue_past_ceiling": "bool"
  },

  "points": ["[references per-point JSONs by filename + point_index]"],

  "frontier": [
    {
      "point_index": "int",
      "b_mse": "int",
      "ppl_headroom": "float",
      "ppl_headroom_band": "string",
      "kv_cache_compression_ratio": "float"
    }
  ],

  "recommended_operating_point": {
    "point_index": "int | null",
    "b_mse": "int | null",
    "ppl_headroom": "float | null",
    "kv_cache_compression_ratio": "float | null",
    "rationale": "string (e.g. 'minimum b_mse with ppl_headroom < ceiling')"
  },

  "termination": {
    "terminated_reason": "ceiling_crossed | grid_exhausted | hook_construction_failure | ppl_eval_failure",
    "terminated_at_point_index": "int | null",
    "failed_at_b_mse": "int | null"
  },

  "hardware": {
    "device": "string",
    "ppl_eval_wall_time_seconds": "float (total across sweep)"
  },

  "notes": "string"
}
```

Filename: `sweep_<model_short>_kv_b_mse_<timestamp>.json` (e.g. `sweep_tinyllama-1.1b_kv_b_mse_2026-05-15T040000Z.json`).

Stored in `tern-core/results/ppl_headroom_kv_cache/<diagnostic_run_id>/` (parallel to R8 v1.1's `results/ppl_headroom_weight/` — directory split per R8 §6.2's pre-declared convention to keep weight + KV diagnostic outputs cleanly separated).

---

## §7 Recommended operating-point selection

The recommended operating point is selected as:

> **The point with the lowest `b_mse` such that `ppl_headroom < ppl_headroom_ceiling`.**

(Inverted relative to R8 v1.1's "highest threshold" — `b_mse` semantic is inverted; the selection rule mechanically preserves "maximum-aggressive compression below ceiling".)

If NO point satisfies the constraint (i.e. even the highest `b_mse` exceeds the ceiling), `recommended_operating_point` is `null` and the run is flagged as `kv_cache_compression_infeasible_under_ceiling` in the aggregate's `notes` field.

**Tiebreakers** (in the rare case of two points with identical lowest `b_mse` admissible — should not occur given monotonic `b_mse` sweep but accounted for completeness):

1. Higher `kv_cache_compression_ratio` wins (more aggressive compression at same quality)
2. Lower `ppl_headroom` wins (better quality at same compression)
3. Lexicographic `point_index` (lower wins) — deterministic fallback

---

## §8 R12 v1.0 first execution — TinyLlama-1.1B `b_mse` sweep

The canonical first execution of R12 v1.0 is a `b_mse` sweep on TinyLlama-1.1B with `ppl_headroom_ceiling=0.50`.

**Hard prerequisite (methodological):** R7-B v1.0 first execution (PR #23 §8) must produce a TinyLlama-1.1B R7-B-conformant FP16 baseline within `[7.99, 8.07]` (per R7-B §1.1 invariant against R7-A's `8.0307` anchor at run_id `20260514T031257Z`, commit `77532e8`). R12 v1.0 cannot proceed without this — using the R7-A baseline directly is explicitly non-R7-B-conformant (per R7-B §1.2) and would invalidate any KV-cache PPL attribution.

**Hard prerequisite (implementation):** see §8.1.

**Rationale for TinyLlama-1.1B as calibration target:**

- Shares calibration target with R8 v1.1 first execution; enables direct cross-diagnostic comparison on a single model
- R7-B autoregressive PPL cycle is ~27 min per point on M4 Pro MPS
- Wall-time characterisation moved to §8.2 (β1a ~26 hr, γ pure-Python ~157 hr; cascaded from 2026-05-16 empirical probes)
- Re-establishes the `b_mse=3` empirical anchor (retired from the 12 May TQ bench per R7-B §1.2) under R7-B-conformant methodology

**First-execution parameters:**

```yaml
source_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
baseline_ppl_r7b: <from R7-B v1.0 first execution; expected [7.99, 8.07]>
baseline_run_id: <from R7-B v1.0 first execution>
ppl_headroom_ceiling: 0.50
sweep_grid:
  b_mse: [6, 5, 4, 3, 2, 1]
num_sequences: 16
seq_len: 2048
seed: 1337
continue_past_ceiling: false
```

**Expected behaviour (informational; not pass/fail):**

- `b_mse = 6` through `4`: `ppl_headroom` in Excellent band (<2%); modest KV-cache reduction; minimal quality impact
- `b_mse = 3`: previously-measured 12 May anchor under non-R7-B methodology (`ppl_baseline=7.82 → ppl_ternary=8.14` ≈ headroom 4%) suggests Acceptable band under R7-B-conformant methodology, though direct numerical comparison is invalid
- `b_mse = 2`: progressive movement into Marginal band (10-25%)
- `b_mse = 1`: likely Fail band (>25%); extreme compression
- `ceiling_crossed` termination likely somewhere in `b_mse = 2-1` range

These expectations are NOT methodology assertions; they are pre-execution hypotheses to be validated by the first run. Material deviation from this curve shape is itself a research signal worth logging in `notes`.

**Post-execution validation gates:**

1. Aggregate sweep manifest is `ppl_headroom_kv_cache_sweep/1.0` schema-conformant
2. All per-point JSONs are `ppl_headroom_kv_cache_point/1.0` schema-conformant
3. `recommended_operating_point` is non-null (at least one point in the sweep grid lands below the ceiling). A specific `b_mse <= 4` floor as a pre-execution sanity gate would risk anchoring on the 12 May TQ bench shape (which is non-R7-B-conformant) — tighten this gate in v1.1 once first-execution data lands and the actual TinyLlama KV-cache-compression-quality frontier under R7-B is empirically characterised.
4. Frontier monotonicity check — `ppl_headroom` decreases (non-strictly) as `b_mse` increases; non-monotonic frontier flags potential PPL-eval noise or non-determinism worth investigating before lifting the methodology to v1.0
5. Cross-check against 12 May anchor: at `b_mse=3`, direction of effect should match (positive `ppl_headroom`); exact magnitudes legitimately differ due to methodology shape change

### §8.1 Implementation prerequisite — `b_mse` parameterisation

As of 2026-05-15 (verified against `tools/tern_infer.py:148` on `main`), `IncrementalTQCompressor.__init__` hardcodes `b_mse=3` inside its `TurboQuantConfig(d=head_dim, b_mse=3, ...)` call at line 162. The R7-B v1.0 §5.2 example `make_b_mse_hook(b_mse: int)` factory references a parameterisation that does not yet exist on the class.

**R12 v1.0 first execution requires a small refactor PR landing first:**

- Add `b_mse: int = 3` (or similar default) as a constructor kwarg on `IncrementalTQCompressor`
- Plumb it into the `TurboQuantConfig(d=head_dim, b_mse=<arg>, ...)` instantiation
- Update the existing consumer site at `tools/tern_infer.py:257` (`generate_streaming_turboquant`) to forward whatever the runtime decides — no behaviour change at default since `b_mse=3` remains the default
- Author `make_b_mse_hook(b_mse: int) -> Callable[[past_kv], past_kv]` factory per R7-B v1.0 §5.2 to construct the appropriate `kv_cache_hook` callable. The factory's hook body wraps `IncrementalTQCompressor` and applies it between forward passes as the R7-B canonical loop expects.

This refactor is **out of R12 v1.0 docs scope** — it's a separate implementation PR — but R12's first execution sweep is blocked on it. Estimated effort: <1 surgeon session. After the refactor lands, the canonical sweep `b_mse ∈ {6,5,4,3,2,1}` becomes mechanically achievable; until then, the methodology is documented but unexecutable.

**Empirical implementation cost (v1.1, cascaded from R-track sessions 2026-05-15 to 2026-05-16):** the prerequisite resolved as two PRs — parameterisation (PR #25, ~half session, merged) and round-trip factory (PR #26, ~full session, draft). A third workstream — Triton / `torch.compile` optimization survey plus β1a uniform-batched factory (PR #27, ~full session, draft) — emerged during the cascade as methodologically additive (see §8.2). The original `<1 surgeon session` estimate covered only the prerequisite and understated even that scope by approximately 2-3×; v1.1 surfaces this correction.

### §8.2 Optimization path (cascaded from R-track session 2026-05-16)

#### §8.2.1 Pure-Python baseline (γ)

- **L-scaling characterisation**: per-hook wall-time follows `cost(L) ≈ 556 ms + 2.24 ms × L` for `L ≥ 512` on TinyLlama-1.1B dims (`n_layers=22, n_heads=32, head_dim=64`) on M4 Pro CPU. Below `L ≈ 300` the cost sits on a Python-overhead floor; above it the vector-math regime is linear in `L`.
- **Full TinyLlama 6-point sweep**: integrating the per-call cost over the R7-B canonical loop (hook fires once per generated token, accumulating `O(L²)` per sequence) gives `~1.62 hr per 2048-token sequence`. At 16 sequences per sweep point and 6 sweep points, the full sweep is `~157 hr` wall time on pure-Python γ.
- **Cold-cost reattribution**: the ~10-minute "cold" first-hook cost in PR #26 traces to `TurboQuantConfig.get_mixed_config`'s lazy outlier-channel detection per `(layer, head)` pair on first call, not to `compute_lloyd_max_codebook` as originally framed in PR #26's body. Under `mixed_precision=False` (β1a), this short-circuits to `~0 cold`.
- **References**: per-hook latency at `L=8/32/64` (cold ~11 min, warm 620/645/699 ms) tabulated in PR #26 PR body; `L=256/512/1024/2048` measurements (1400/1719/2841/5140 ms warm) from the 2026-05-16 L-scaling probe (R-track session memory).
- **Why this matters**: §8's original `~3 hr` estimate was a pre-empirical projection assuming `~700 ms` constant per-call cost across all `L`; the L-scaling probe falsified that constancy assumption above `L ≈ 300`, understating true pure-γ wall-time by ~52×. v1.0 carried the estimate as aspirational rather than measured.

#### §8.2.2 Surveyed optimization paths

| Option | Status | Wall-time (full 6-point sweep) | Engineering | Methodological integrity |
|---|---|---|---|---|
| α — Triton kernels | **DEAD** | n/a | n/a | n/a (math-incompatible) |
| β1a — batched uniform | **LIVE** (PR #27) | ~26 hr | ~1.5 days | shift (no mixed precision) |
| β1b — batched mixed | **DEFERRED** | ~22-32 hr (projected) | ~3-5 days | preserved |
| β2 — `torch.compile` | **DEAD** | n/a (0.09× empirical) | n/a | n/a |
| γ — pure-Python (mixed) | **FALLBACK** | ~157 hr | none | preserved |

**α disqualification (dual)**: `turboquant/src/kernels.py` uses Rademacher (±1) QJL S matrices while `cache.py` uses Gaussian N(0,1) S matrices per the paper's Definition 1 (kernels.py lines 1-10 explicitly forbid mixing the two paths). Additionally, the Triton kernels assume CUDA per the self-test footer at `kernels.py:1268`; mainline Triton has no MPS backend in the installed environment.

**β2 disqualification**: empirical probe (R-track 2026-05-16) measured `torch.compile(make_b_mse_hook, mode='reduce-overhead')` at `0.09×` speedup (i.e., ~11× SLOWDOWN). Dynamo log shows `config.recompile_limit (8)` hit on `get_mixed_config` due to `head_idx` integer-specialisation triggering a fresh trace per head; `torch.Generator.__new__` (used by QJL Rademacher sampling) also flagged as untraceable. The structural mismatch with the per-(layer, head) Python loop is not fixable without refactoring the loop itself; out of v1.1 scope.

**β1b deferral**: methodologically preserves mixed precision via batched outlier-channel partitioning across pairs, at ~3-5 days engineering vs β1a's ~1.5 days. Not currently required given β1a + targeted γ-supplement strategy (§8.2.3) recovers mixed-precision PPL at deployment-relevant anchor points. Revisit if R12 sweep cadence increases or anchor-point coverage proves insufficient.

#### §8.2.3 Recommended R12 sweep workflow (post-v1.1)

**Primary headroom curve via β1a** (uniform precision):

- Tool: `make_b_mse_hook_uniform` (`tools/tern_infer.py`, PR #27)
- Coverage: full 6-point sweep `b_mse ∈ {6,5,4,3,2,1}` at uniform precision
- Wall-time: `~26 hr` (linear fit on β1a measured costs: `cost(L) ≈ 0.464 ms × L`, integrated over 2048-token sequences × 16 sequences × 6 points)

**Targeted mixed-precision anchors via γ**:

- Tool: `make_b_mse_hook` (`tools/tern_infer.py`, PR #26)
- Coverage: 2-3 deployment-relevant `b_mse` points (default: `b_mse ∈ {3, 4}` — the `b_mse=3` 12 May TQ bench legacy anchor + `b_mse=4` Excellent/Acceptable band boundary per §8 expected behaviour). Surgeon may override at runtime; document override in run `notes`.
- Wall-time: `~26 hr per anchor point` (γ per-sequence ~1.62 hr × 16 sequences; see §8.2.1 for derivation). 2 anchors = ~52 hr supplement; 3 anchors = ~78 hr supplement.
- Purpose: anchor the uniform-vs-mixed PPL delta empirically at deployment-relevant operating points.

**Total R&D cycle**: `~78-104 hr` (β1a + 2-3 γ anchors) vs `~157 hr` pure-γ — wall-time savings of **33-50%** over running R12 entirely on the mixed-precision reference.

**Cross-tabulation strategy**: at each anchor `b_mse` value, report both the β1a uniform PPL and the γ mixed-precision PPL. The cross-tabulation quantifies the outlier-channel-protection contribution to PPL at the sweep operating range. The uniform-precision headroom curve produced by β1a characterises a configuration that does not ship; the γ-supplement anchors recover the deployment-relevant numbers. Both datasets are required for the §7 operating-point selection to be deployment-meaningful.

**Empirical reference data point** (captured during PR #27 Phase E verification): `cos(β1a uniform, cache.py mixed) = 0.9896`, `MSE = 2.01e-02` at `b_mse=4`, layer 0, head 0. This single data point bounds the methodological delta but does not substitute for the full anchor-point sweep — PPL behaviour is downstream of single-vector reconstruction quality and requires its own measurement.

#### §8.2.4 Forward references

- **PR #25** (merged) — `b_mse` parameterisation of `IncrementalTQCompressor`; commit `03fe4f6`
- **PR #26** (draft) — `make_b_mse_hook` mixed-precision reference factory; head `227a675`
- **PR #27** (draft) — `make_b_mse_hook_uniform` β1a optimization factory; head `4002edc`
- **R7-B v1.1** (forthcoming) — hook-factory lifetime + per-call cost expectations; next queued workstream per session-end roadmap
- **R-track session memory 2026-05-16** — empirical baseline measurements (Triton-on-MPS survey, L-scaling probe, `torch.compile` probe, β1a landing + Phase E numbers)

---

## §9 Cross-architecture extension (post-R12 v1.0 first execution)

Once R12 v1.0 first execution validates the diagnostic on TinyLlama-1.1B, the methodology applies across the canonical architecture set:

| Model | Status | Notes |
|---|---|---|
| TinyLlama-1.1B | R12 v1.0 first execution | Calibration target |
| Llama 3.2 1B | Pending R12 validation | Independent architecture-family validation; R7-B-conformant baseline must be established first per §8 prerequisite |
| Gemma 4 E4B | Pending R12 validation; PPL pathology per R9.3 backlog item must be resolved first | Gemma 4 PPL methodology calibration is a hard prerequisite (per R9.3) — applies to R7-B baseline establishment, not just R7-A's |
| Gemma 4 31B | Pending R12 validation; same PPL pathology dependency as E4B; memory-accounting check per R7-B v1.0 §5.3 (KV cache ~2.5 GB at L=2048) | Headline KV-compression demonstration target |
| Mistral 7B | Pending R12 validation | Historical reference architecture |
| Phi-4 14B | Pending R12 validation; no-BOS-prepend convention per R7-A v1.0 §4 inherits via R7-B v1.0 §3 | Mid-scale architecture coverage |

Each architecture requires its own R7-B-conformant baseline run before its R12 sweep — per R7-B v1.0 §1.2's explicit retirement of R7-A baselines as non-R12-conformant.

Cross-architecture frontier curves feed:

- **Eagle Bracket 2** (deployment-tier selection — different `b_mse` choices for different memory/quality tiers; cross-architecture curves show which tier each model can sustain)
- **Eagle Bracket 3** (constant per-token encode cost at scale) — KV-cache compression directly affects per-token encode cost as cache grows; R12 supplies the quality-band attribution
- Patent collateral for cross-family generalisation claims around KV-cache ternarisation
- Production deployment tier selection (different `b_mse` choices for different memory / quality tiers)

Architecture sweep runs SHOULD reuse the v1.0 sweep grid where possible to permit direct cross-family frontier comparison; deviations MUST be documented in the run's `notes` field and the comparison reported with explicit caveats.

---

## §10 References

- `docs/wikitext2_ppl_methodology_autoregressive.md` (R7-B v1.0, PR #23) — autoregressive PPL methodology consumed by §4 step 3
- `docs/weight_compression_ppl_headroom_diagnostic.md` (R8 v1.1, PR #20) — sibling diagnostic for weight-compression scope; R12 structurally mirrors §1-§9
- `docs/r8_v1.1_disposition_note.md` (PR #20) — disposition record ratifying option (c) split into R8 v1.1 + R12
- `docs/wikitext2_ppl_methodology.md` (R7-A v1.0, PR #20) — teacher-forcing PPL methodology; sibling to R7-B but explicitly NOT R12-conformant per R7-B §1.2
- `benchmarks/tq_bench_results.json` (2026-03-30, `b_mse=3` 5-sentence anchor) — preserved as documentary record; retired by R7-B v1.0 §1.2; R12 v1.0 first execution re-establishes the anchor under R7-B-conformant methodology
- `tools/tern_kv_ppl_bench.py` (to be authored alongside R12 v1.0 ratification) — R12-conformant PPL + KV-cache-compression measurement tool, analogous to `tools/tern_ppl_bench.py` for R7-A
- `IncrementalTQCompressor` (tern-core adapter at `tools/tern_infer.py:148`; wraps third-party TurboQuant package) — KV-cache compression operator that R12 will consume via R7-B v1.0 §5.2's planned `make_b_mse_hook` factory. Current state: `b_mse` hardcoded at line 162 inside the `TurboQuantConfig` call; parameterisation refactor per §8.1 is the implementation prerequisite.
- `tools/tern_infer.py:62` — adjacent `MixedPrecisionConverter.convert(...).report.compression_ratio` surfacing pattern; weight-compression code path (NOT KV-cache); cited here so the cross-reference is unambiguous and future readers don't conflate the two ratio-surfacing patterns.
- `docs/backlog.md` R13 entry — `ppl_headroom` terminology disambiguation between autoscan-internal sense and R7-A/R8/R12 outcome metric (R12 inherits R8's outcome-metric sense unchanged)

---

## §11 Change log — v0.1 → v0.2 (2026-05-15)

Cascade applied 2026-05-15 per surgeon ratification of v0.1 leans. All edits sourced from the verification + corruption-fix pass landed earlier the same day. v0.1 retained alongside this file as `kv_cache_compression_ppl_headroom_diagnostic.md`.

### Substantive (methodology-affecting)

1. **Eagle Bracket attribution broadened from 3-only to 2+3** (§1 last paragraph + §9 cross-architecture-frontier feeds). v0.1 cited only Bracket 3 (constant per-token encode at scale). v0.2 adds Bracket 2 (deployment-tier selection — per-tier `b_mse` choices for memory/quality tradeoffs) since R12's per-architecture frontier is directly consumed by tier selection. Q2 from v0.1 SURGEON_BRIEF.
2. **§8 post-execution gate 3 softened** — v0.1 required `recommended_operating_point` non-null AND `b_mse <= 4`. v0.2 keeps the non-null requirement but removes the `b_mse <= 4` floor; pre-execution it would anchor on the 12 May TQ bench shape (non-R7-B-conformant) — tighten in v1.1 once first-execution data lands. Q from v0.1 SURGEON_BRIEF.
3. **§8.1 new section** — implementation prerequisite for `b_mse` parameterisation. Discovered during the verification pass: `IncrementalTQCompressor` currently hardcodes `b_mse=3` at `tools/tern_infer.py:162`; the `make_b_mse_hook` factory per R7-B v1.0 §5.2 does not yet exist on the class. R12 first execution is blocked on a small refactor PR (estimated <1 surgeon session). Spec is documented but unexecutable until that refactor lands.

### Documentary (clarity / accuracy without methodology change)

4. **§3 added explicit IncrementalTQCompressor hardcoding callout** — mid-section pointer to §8.1 prerequisite so any reader reaching the sweep-parameter discussion sees the blocker before getting to §8.
5. **§3.3 new v0.2-simplification disclaimer** — explicit one-paragraph note that v0.2 holds multiple dimensions fixed for signal-attribution cleanliness; v2.0 proposes joint sweeps + per-layer selectivity. Q3 from v0.1 SURGEON_BRIEF.
6. **§4 added explicit step 5 for external compression-ratio measurement** — v0.1 left the surfacing implicit ("from IncrementalTQCompressor report"). v0.2 adds a dedicated procedure step noting `tools/tern_kv_ppl_bench.py` computes the ratio externally from `past_key_values` byte sizes. Q4 from v0.1 SURGEON_BRIEF.
7. **§6.1 `kv_cache_compression_ratio` schema description rewritten** — v0.1 said `"float (size_before / size_after, from IncrementalTQCompressor report)"`. v0.2 says `"float (size_before / size_after, computed by tern_kv_ppl_bench.py from pre/post past_key_values byte sizes; IncrementalTQCompressor does NOT natively surface this — see §10 reference and §8.1 implementation prerequisite)"`. Q4 disposition.
8. **§10 IncrementalTQCompressor reference fixed** — v0.1 said `(tern-core internal) — surfaced via R7-B v1.0 §5.2 make_b_mse_hook factory`. v0.2 cites the concrete path `tools/tern_infer.py:148`, identifies it as an adapter wrapping third-party TurboQuant, and explicitly flags the `b_mse` hardcoding state at line 162 + the §8.1 refactor prerequisite. Q5 from v0.1 SURGEON_BRIEF.
9. **§10 added `tools/tern_infer.py:62` adjacent-pattern reference** — disambiguates the weight-compression `report.compression_ratio` surfacing at the file's earlier line from the KV-cache scope, so future readers don't conflate the two patterns when scanning the file.

### Unchanged from v0.1 (deliberate; surgeon may revisit in v0.3 if desired)

- **Q1 sweep direction** — descending `b_mse` (highest first). Keeps R8 v1.1's "lowest aggression to highest" convention. v0.1 lean confirmed.
- All schema versions (`ppl_headroom_kv_cache_point/1.0`, `ppl_headroom_kv_cache_sweep/1.0`) — decoupled from spec version per R7-A/R7-B/R8 v1.1 precedent.
- Canonical sweep range `b_mse ∈ {6,5,4,3,2,1}` — bracketing the 12 May anchor; adjustable per model family.
- §7 operating-point selection rule + tiebreakers — unchanged.
- §9 architecture coverage table — unchanged.

### v0.2 → v1.0 promotion (2026-05-15)

Ratification cascade applied per surgeon disposition. Two changes from v0.2:

10. **§10 local-path strip** — v0.2 cited the TurboQuant install location as a surgeon-local venv path (environment-brittle and not useful as repo documentation). v1.0 strips the path; the file:line reference at `tools/tern_infer.py:148` remains as the canonical adapter location.
11. **Version-string sweep** — all `v0.2` references in §1 through §10 and the footer rewritten to `v1.0`; §11 historical references to the v0.1 → v0.2 cascade preserved verbatim as documentary record. §3.3 line about "during v0.2 review" rephrased to "Future cascade revisions (v1.x or v2.0) may promote..." since v1.0 is now the ratified state.

All other v0.2 content unchanged.

### v1.0 → v1.1 cascade (2026-05-16) — empirical wall-time correction

Cascade applied 2026-05-16 per surgeon ratification of empirical-probe disposition. All edits sourced from the 2026-05-16 R-track session: Triton-on-MPS survey, L-scaling probe, `torch.compile` probe, β1a landing (PR #27 Phase E). v1.0 retained verbatim in §1-§7 and §9-§10; §8/§8.1 amended; §8.2 new.

#### Substantive (methodology-affecting)

12. **§8 wall-time estimate referenced to §8.2 (empirical)**. v1.0 cited `~3 hr` as a pre-empirical projection assuming `~700 ms` constant per-call cost; the 2026-05-16 L-scaling probe falsified the constancy assumption above `L ≈ 300` (`cost(L) ≈ 556 ms + 2.24 ms × L`). The original `~3 hr` understated true β1a wall-time by ~9× and true pure-γ wall-time by ~52×. v1.1 grounds wall-time in measurement.
13. **§8.1 scope correction surfaced (4 bullets understated by ~3×)**. Empirical resolution: parameterisation (PR #25, ~half session, merged) + round-trip factory (PR #26, ~full session, draft) as prerequisites, plus a third additive workstream — optimization survey + β1a (PR #27, ~full session, draft). Closing sentence added under v1.1; existing four bullets preserved as documentary record.
14. **§8.2 (new) — optimization path with empirical findings.** Subsections:
    - §8.2.1 pure-Python baseline (γ) — L-scaling characterisation, ~157 hr full sweep, cold-cost reattribution (`get_mixed_config` lazy outlier-detection, not codebook compute)
    - §8.2.2 surveyed paths — α (DEAD), β1a (LIVE, PR #27), β1b (DEFERRED), β2 (DEAD), γ (FALLBACK); disqualification reasons tabled
    - §8.2.3 recommended workflow — β1a primary + targeted γ-supplement; total R&D cycle `~78-104 hr` vs `~157 hr` pure-γ (33-50% savings)
    - §8.2.4 forward references

#### Documentary (clarity / accuracy without methodology change)

15. **Footer revised** — v1.0 footer claimed "Implementation execution blocked on §8.1 refactor PR." This is stale: prerequisite PR #25 has merged, PR #26 is in draft, and §8.2 provides the first-execution strategy. v1.1 footer reflects the unblocked state.

#### Banked lesson (methodology-adjacent)

Pre-empirical wall-time projections in spec documents are aspirational, not anchors. v1.0's `~3 hr` (§8) and `<1 surgeon session` (§8.1) projections both understated true cost — former by ~9-52× depending on optimization path, latter by ~3×. Future spec drafts SHOULD label pre-empirical projections explicitly as such and SHOULD cite them as upper-bound hypotheses to be falsified during first execution, not as commitments.

---

*Ratified 2026-05-16 — tern-core methodology document, R12 v1.1. Companion to R8 v1.1. Implementation execution unblocked: §8.1 prerequisite resolved (PR #25 merged, PR #26 draft); first-execution strategy per §8.2 (β1a primary sweep + targeted γ-supplement anchors).*
