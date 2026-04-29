# Mistral-7B Phase D — Energy Baselines

Capture date: 2026-04-15
Model: mistral_7b_ternary v0.3.0 (.tern-pkg, 2.27 GB)
mlpackage: mistral_7b_ternary.mlpackage
Hardware: Mac Mini M4 Pro, 64 GB unified memory
Methodology: 15 s sustained inference per row, 14 power samples,
10-run warmup, identical conditions across all three compute units.
Input shape per inference: `input_ids INT32 [1, 64]` (SEQ_LEN = 64).

## Headline framings

Two framings, both true. Use Framing 2 for the Apple-brief headline;
preserve Framing 1 for continuity with prior internal documents.

**Framing 1 — single-number:** "5.39 W on CPU_AND_NE."

**Framing 2 — best-of-three (preferred):** Two of three compute
units land within a 1.3 % energy envelope at ~5.4 W; the third
(CPU_AND_GPU) pays a 2.96× energy premium for a 27 % latency gain.
CPU_ONLY runs competitive with CPU_AND_NE on every axis and tighter
on latency stdev.

## Phase D — three compute units

| Compute Unit | tok/s | Mean ms | Latency stdev | Mean W | Stdev W | mJ/inference | mJ/token |
|---|---|---|---|---|---|---|---|
| CPU_ONLY    | 304.5 | 210.16 | 4.58 ms | 5.458  | 0.955 | 1137.1 | 17.77 |
| CPU_AND_NE  | 297.6 | 215.07 | 9.68 ms | 5.388  | 0.703 | 1122.6 | 17.54 |
| CPU_AND_GPU | 387.8 | 165.02 | 0.89 ms | 20.142 | 3.755 | 3320.1 | 51.88 |

mJ/token = mJ/inference ÷ 64 (SEQ_LEN per the powermetrics log
header `Input: input_ids INT32 [1, 64]`).

## Provenance

- `benchmarks/mistral7b_energy_baselines.log` — raw powermetrics
  output backing CPU_ONLY and CPU_AND_GPU verbatim. Force-added past
  the `*.log` gitignore in commit `fdcbe0e` because the file carries
  real measurement evidence rather than transient debug output.

- `benchmarks/mistral7b_phase2.json:79–89`
  (`energy.cpu_only_baseline` block) and `:91–101`
  (`energy.cpu_and_gpu_baseline` block) — structured JSON for
  CPU_ONLY and CPU_AND_GPU, derived from the powermetrics log.

- `benchmarks/mistral7b_phase2.json:68–77` (`energy.raw_best` block,
  label `raw_CPU_AND_NE`) — structured JSON for CPU_AND_NE energy
  figures from the Phase B+D combined run. Captured during the same
  2026-04-15 session and emitted separately from the standalone
  energy log.

- `benchmarks/mistral7b_phase2.json:37–48`
  (`compute_unit_benchmarks.CPU_AND_NE` block) — structured JSON
  for CPU_AND_NE Phase B latency context (215.07 ms mean, 9.68 ms
  stdev) used in the table above.

## Notes

- The 15-second sustained-inference methodology applies to every
  row, so the three compute units stand directly comparable.
- CPU_AND_NE retains the cold-start jitter advantage observed
  during Phase B (9.7 ms stdev against 245 ms outliers on
  CPU_AND_GPU under cold conditions). The steady-state energy run
  holds the thermal envelope long enough that CPU_AND_GPU's
  cold-start outliers fall away — its 0.89 ms stdev row above
  reflects warm steady-state, separate from the Phase B cold-start
  picture.
- The "best-of-three" framing carries the substrate-comparison
  argument for the Apple brief: ternary compression brings CPU and
  ANE to near-equivalence on energy, which is the deeper claim
  about what ternary does to compute substrate economics.

## Outstanding

- Phase 2 confirmation pass (fork iii) is queued as a separate
  sprint item: re-run Phase D from a clean thermal state via the
  archive-restored mlpackage, compare to this 2026-04-15 capture.
  If the delta on mean W stays under 5 %, treat as twice-confirmed.
