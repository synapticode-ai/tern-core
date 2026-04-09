# tern-core — CLAUDE.md

Extends the workspace-level CLAUDE.md with tern-core specific sprint targets.
Read the top-level CLAUDE.md first. This file adds to it — do not contradict it.

## Existing Architecture (do not alter)

The existing tern-core internals are stable and tested. The module map,
data flow, arithmetic rules, and patent alignment documented in the
top-level CLAUDE.md apply here in full. Do not refactor existing modules
to accommodate the new sprint components — add alongside, not into.

Existing tested components (READ ONLY for this sprint):
- arithmetic/quantizer.py   — TernaryQuantizer, SensitivityAnalyzer
- arithmetic/linear.py      — TernaryLinear, TernaryConv2d (STE gradients)
- engine/inference.py       — TernaryInferenceEngine (PyTorch-based)
- sparse/__init__.py        — 2-bit packing, sparsity bitmap, zero-skip
- memory/__init__.py        — profile_model_memory()
- model_loader/__init__.py  — TernModelWriter, TernModelReader, .tern-model format

Published benchmark results (locked — do not overstate or alter):
- Mistral-7B: 14.5GB → 2.27GB at 96.4% ternary sparsity ratio
- KV cache speedup: ~2.56×
- Perplexity-gated autoscan: active
- KV cache fix: O(n²) → cached (v0.2.0)

## Current Sprint: Ternary-Aware Fine-Tuning Harness (TFH)
Reference specification: documents/SPEC-TFH-001.docx

All new TFH components live under: `tern-core/harness/`
Do not place them inside existing src/ module directories.

### New components — build in this order

**Phase 1 — Foundation (P0, build first)**

1. `harness/projector.py` — TernaryProjector³
   - Soft ternary projection with temperature annealing (tau: 1.0 → 0.01)
   - MLX-native operations (not PyTorch — this is the MLX transition point)
   - Interfaces with existing TernaryEncoder from model_loader
   - Zero attractor enforces sparsity target of 0.90
   - Straight-through estimator extended to three-state case

2. `harness/objective.py` — ConfidenceObjective³
   - Composite loss: task_loss + alpha * (calibration_penalty + 0.1 * sparsity_penalty)
   - calibration_penalty: KL divergence between predicted and ground-truth
     epistemic distributions over {confirmed, uncertain, disconfirmed}
   - sparsity_penalty: L1 on non-zero activations vs sparsity_target
   - alpha anneals from 0.0 → 1.0 over training (governed by AdaptationScheduler³)

3. `harness/annotator.py` — EpistemicAnnotator³
   - Attaches ternary confidence triple to each weight cluster after update step
   - Triple vocabulary: confirmed | uncertain | disconfirmed
   - Writes annotations to ConfidenceEventLog³ at each step
   - Must not alter the weight values themselves — annotation only

4. `harness/scheduler.py` — AdaptationScheduler³
   - Controls temperature tau and loss-weighting alpha across training steps
   - Configurable via harness.yaml (see SPEC-TFH-001 Section 9)
   - No external dependencies — pure Python

**Phase 2 — Integration (P1, build after Phase 1 passes tests)**

5. `harness/trainer.py` — TernaryTrainer³
   - Master training loop integrating all Phase 1 components
   - MLX-native forward and backward pass
   - Gradient accumulation support (effective batch via accumulation steps)
   - Calls EpistemicAnnotator³ and ConfidenceEventLog³ after each step

6. `harness/checkpointer.py` — HarnessCheckpointer³
   - Persists model state + confidence annotations in .see3-wrapped format
   - Output must be directly loadable by MLXWeightLoader³ in tern-runtime
   - No conversion step between training checkpoint and inference deployment
   - Saves every N steps (configurable via harness.yaml)

7. `harness/event_log.py` — ConfidenceEventLog³ extension
   - Extends existing ConfidenceEventLog³ schema with training audit entries
   - Fields per step: step_idx, total_loss, task_loss, confidence_loss,
     mean_sparsity_ratio, triple_distribution
   - Backward compatible: existing inference log entries unchanged

### Epistemic label schema (attached per training example)
```json
{
  "epistemic_state":    "confirmed|uncertain|disconfirmed",
  "confidence_score":   0.0-1.0,
  "escalate":           true|false,
  "domain":             "factual|reasoning|creative|agentic",
  "source_reliability": 0.0-1.0
}
```

### Confidence thresholds (consistent across training and inference)
- confirmed:     top-1 prob >= 0.85
- uncertain:     top-1 prob >= 0.45
- disconfirmed:  top-1 prob <  0.45

### Sparsity target
0.90 — enforced via zero attractor in TernaryProjector³ soft projection.
Matches tern-core published 96.4% sparsity result as baseline.

### Config entry point
`harness/harness.yaml` — full reference in SPEC-TFH-001 Section 9.
CC should create this file as part of Phase 1 setup.

### Phase 1 success criteria (before moving to Phase 2)
- TernaryProjector³ anneals tau=1.0 → 0.01 without gradient collapse
- ConfidenceObjective³ total loss decreases monotonically on a clean dataset
- EpistemicAnnotator³ triple distribution matches dataset label priors
- All Phase 1 components have pytest coverage in harness/tests/

### Phase 3 validation targets (Mac Mini M4 Pro — run after Phase 2)
- Mistral-7B: perplexity ≤ base + 3%; ECE ≤ 0.10
- Peak memory ≤ 14GB at batch size 4
- Token throughput ≥ 500 tok/s during training step
- Checkpoint loads into MLXWeightLoader³ without conversion

## Patent alignment (new claims — document in all new functions)
- TernaryProjector³ soft projection → extends STE to three-state (no existing patent)
- ConfidenceObjective³ composite loss → novel training objective (no existing patent)
- EpistemicAnnotator³ weight annotation → candidate new provisional (flag to Rod)
- HarnessCheckpointer³ training→inference continuity → candidate new provisional

## Environment note
MLX is Apple-only. The harness/ components use MLX, not PyTorch.
The existing tern-core src/ components remain PyTorch-based.
Do not mix the two frameworks within the same module.


---
## REVISION NOTE — April 2026 (after codebase inspection)

The following files already exist in tern-core/src/terncore/ and are
directly relevant to the TFH sprint. Build ON these — do not duplicate.

### Existing files the TFH harness must extend, not replace:

`ste.py` + `ste_trainer.py`
  The STE trainer is the foundation of TernaryProjector³.
  TernaryProjector³ extends ste_trainer.py with:
  - Temperature annealing (tau: 1.0 → 0.01)
  - Three-state zero attractor (not binary STE)
  - MLX-native operations (ste_trainer.py is likely PyTorch — wrap, don't port)

`confidence.py`
  Confidence primitives already exist. EpistemicAnnotator³ should import
  and extend confidence.py — do not rewrite confidence logic from scratch.
  Inspect confidence.py first; add the triple annotation mechanism on top.

`routing.py`
  Routing logic exists. TernaryRouter³ integration in tern-runtime should
  reference this — confirm the interface before building GammaPlatformAdapter.

`queue.py`
  Queue primitives exist. ConfidenceQueue³ likely extends or wraps this.
  Inspect before building the orchestrator layer in tern-runtime.

`autoscan.py`
  Perplexity-gated autoscan is live. HarnessCheckpointer³ validation
  should call autoscan.py — do not reimplement perplexity gating.

`persistence.py`
  Persistence layer exists. HarnessCheckpointer³ .see3 wrapping should
  sit above persistence.py, not replace it.

`model_router.py`
  Model routing exists. Consult before implementing routing in harness/.

### Revised harness/ placement in light of existing files:

`harness/projector.py` — TernaryProjector³
  Import ste.py and ste_trainer.py. Extend with temperature annealing
  and three-state zero attractor. Do not rewrite STE from scratch.

`harness/objective.py` — ConfidenceObjective³
  Import confidence.py primitives. Add composite loss computation on top.

`harness/annotator.py` — EpistemicAnnotator³
  Import confidence.py. Add triple annotation to weight clusters.
  Write to ConfidenceEventLog³ via persistence.py.

`harness/scheduler.py` — AdaptationScheduler³
  Pure Python. No existing dependency. Safe to write fresh.

`harness/trainer.py` — TernaryTrainer³
  Orchestrates all above. MLX-native. Calls autoscan.py for validation.

`harness/checkpointer.py` — HarnessCheckpointer³
  Wraps persistence.py with .see3 metadata layer.
  Output must load directly into tern-runtime/loader/mlx_loader.py.

## Built — TFH Sprint
- pyproject.toml — added [project.optional-dependencies] harness =
  ["mlx>=0.18"]. PyTorch path under src/terncore/ untouched. Install
  via: pip install -e ".[harness]". Confirmed mlx 0.31.1 imports
  cleanly on arm64 macOS 26.4.
- harness/__init__.py — empty package marker.
- harness/epistemic_state.py — EpistemicState enum {confirmed,
  uncertain, disconfirmed} with lowercase string values. Domain enum
  {factual, reasoning, creative, agentic}. EpistemicLabel frozen
  dataclass mirroring SPEC-TFH-001 § 4.1 JSON schema with __post_init__
  range validation on confidence_score and source_reliability,
  to_dict / from_dict round-trip via JSON, from_string class methods
  with clear ValueError on unknown values. Distinct from
  terncore.confidence.RoutingConfidence — module docstring locks the
  vocabularies apart explicitly. String values match
  tern-runtime/inspector/confidence_emitter.EpistemicState byte-for-byte.
- harness/tests/test_epistemic_state.py — 16 tests covering enum
  basics, Domain enum, EpistemicLabel construction, frozen-dataclass
  immutability, range validation, JSON round-trip, missing-key
  rejection, unknown-state-string rejection, AND the day-one
  CROSS-REPO TRIP-WIRE: imports both this repo's EpistemicState and
  tern-runtime/inspector/confidence_emitter.EpistemicState, asserts
  the {name: value} maps are byte-identical. If that test ever fails,
  TFH training labels and LIS inference confidence have drifted apart
  and the TFH continuity claim is broken.
- harness/scheduler.py — AdaptationScheduler frozen dataclass, pure
  Python zero deps. Two annealing schedules consumed by Phase 1
  components: tau (linear hot→cold, default 1.0→0.01 over total_steps,
  consumed by TernaryProjector³) and alpha (held at 0.0 for the first
  alpha_warmup_steps, then linear 0.0→1.0 over the remainder, consumed
  by ConfidenceObjective³). Both schedules clamp at the boundaries so
  validation hooks and checkpoint loaders can pass out-of-range step
  indices safely. progress(step) returns {step, total_steps, tau,
  alpha, pct_complete, warmup_active} for ConfidenceEventLog³ entries
  and dashboards. Construction validates total_steps > 0,
  initial_tau > 0, final_tau > 0, initial_tau >= final_tau (rejects
  reverse schedule), 0 <= alpha_warmup_steps <= total_steps.
- harness/tests/test_scheduler.py — 25 tests, no deps: construction
  validation (six rejection cases), tau schedule (boundaries, midpoint
  linearity, clamps, monotonic decrease over the full range), alpha
  schedule (warmup zero, boundary, rise, completion, midpoint linearity,
  both-sided clamps, zero-warmup variant, monotonic non-decrease),
  progress dict shape and warmup flag, frozen dataclass mutation guard.
- harness/projector.py — TernaryProjector, the first MLX-touching file.
  Reimplements the ste.STEQuantize mathematical contract natively in
  MLX (ste.py is untouched and the PyTorch path is unaffected). Hard
  projection at tau <= HARD_TAU_EPSILON=1e-5 matches STEQuantize bit-
  for-bit; soft projection at tau > epsilon uses tanh(w/tau) inside
  the active band with the SAME threshold-based deadband mask, so the
  three-state zero attractor is preserved across the entire tau anneal.
  alpha is computed from the HARD active mask (not the soft output)
  so the per-step scaling factor stays stable as tau falls. Threshold
  formula: 0.7 * mean(|w|) per the documented STEQuantize default,
  configurable per-instance and per-call. Returns frozen
  ProjectionResult(weights_ternary, weights_dequant, alpha, sparsity,
  threshold). compute_threshold() exposed as a separate method so
  tests can verify the formula independently of the full projection.
  Handles all-zero weights gracefully via mean(|w|) fallback matching
  ste.py exactly.
- harness/tests/test_projector.py — 13 tests including the headline
  CONTRACT TRIP-WIRE: imports terncore.ste.STEQuantize as the PyTorch
  reference oracle, runs both side by side on the same numpy bytes
  (seed=42, shape=(64,32)), asserts MLX projector at tau=1e-6 matches
  PyTorch ternary output to 1e-4, threshold to 1e-6, alpha to 1e-5,
  sparsity to 1e-4. Plus: hard projection produces strictly {-1,0,+1},
  tau=0 == tau=epsilon/10, mx.grad through soft projection at tau=1.0
  produces non-zero gradients (verified the autograd graph survives
  the .item() calls used to extract threshold/alpha as Python floats),
  cosine similarity to hard increases as tau falls (anneal property
  check), sparsity strictly increases with threshold_scale, alpha
  equals mean(|w|) over active mask, ProjectionResult is frozen,
  all-zero weights produce alpha=0 sparsity=1 with no division by
  zero, construction validates threshold_scale > 0, project validates
  tau >= 0, per-call threshold_scale override does not mutate the
  projector instance.
- harness/annotator.py — EpistemicAnnotator, training-time analogue
  of ConfidenceEmitter. Maps a ProjectionResult to a StepAnnotation
  using the projector's sparsity ratio as the confidence proxy
  (higher sparsity = more decisive weight distribution = higher
  predicted confidence). Threshold class constants LOCKED at
  CONFIRMED_THRESHOLD=0.85 / UNCERTAIN_THRESHOLD=0.45 — same values
  as tern_runtime.inspector.confidence_emitter.ConfidenceEmitter.
  Stateless: every annotate() call is independent. annotate(),
  batch_annotate(), and summary() exposed. StepAnnotation frozen
  dataclass {epistemic_state, predicted_score, label_state,
  label_score, calibration_error, sparsity, is_correct}. summary()
  aggregates counts, mean sparsity, mean calibration error, accuracy
  for ConfidenceEventLog³ entries; empty input returns zeros (never
  raises) so logging is unconditional.
- harness/tests/test_annotator.py — 18 tests including the SECOND
  CROSS-REPO TRIP-WIRE: test_threshold_constants_match_confidence_
  emitter imports ConfidenceEmitter from tern-runtime and asserts
  CONFIRMED_THRESHOLD == 0.85 and UNCERTAIN_THRESHOLD == 0.45 in
  both repos byte-for-byte. Plus three-state classification, both
  threshold boundaries map upward, just-below-boundary maps down,
  calibration_error is the absolute difference (never negative),
  is_correct flag, batch_annotate length validation, summary counts
  and means, frozen StepAnnotation mutation guard, summary on empty
  list returns zeros.
- harness/objective.py — ConfidenceObjective, composite training loss
  per SPEC-TFH-001 § 3.4: total_loss = task_loss + alpha *
  (calibration_penalty + 0.10 * sparsity_penalty). Calibration penalty
  is KL(target || predicted) over 3-element {confirmed, uncertain,
  disconfirmed} distributions; each (state, score) pair is converted
  to a distribution by placing score at the named state and splitting
  (1-score)/2 across the other two. Sparsity penalty is the squared
  shortfall from the configurable sparsity_target (default 0.90),
  zero when actual >= target — never penalises excess sparsity.
  SPARSITY_PENALTY_WEIGHT locked at 0.10 as a class constant per the
  spec. Alpha gating: at alpha=0 the composite reduces to task_loss;
  at alpha=1 it includes the full confidence terms; linearly scales
  in between. Aggregates a list of ProjectionResults by mean sparsity
  (one per layer per step), aggregates a list of EpistemicLabels
  element-wise (one per example per batch). Phase 1 gradient note in
  docstring: the calibration and sparsity penalties are diagnostic /
  scheduling signals at this layer — they shape the LOGGED total loss
  but not yet the WEIGHT updates, because sparsity comes from the
  projector's .item() materialised float. Phase 2 will lift the
  sparsity penalty into a continuous MLX expression so it contributes
  to the gradient; the loss VALUE stays the same. ObjectiveResult
  frozen dataclass {total_loss, task_loss, calibration_penalty,
  sparsity_penalty, alpha_used, mean_predicted_sparsity}.
- harness/tests/test_objective.py — 22 tests, no MLX deps:
  construction validation, alpha=0 reduces composite to task_loss,
  alpha=1 includes confidence terms, total_loss scales linearly with
  alpha, sparsity_penalty=0 at or above target, squared shortfall
  formula, monotonic increase as sparsity falls, calibration penalty
  zero when distributions match, positive when states disagree, grows
  with disagreement strength, finite for degenerate one-hot labels,
  _state_to_distribution sums to 1 and places score at the named state,
  _kl_divergence properties (zero on identical, positive on different,
  handles zero target mass via the limit p log(p/q) → 0 as p → 0),
  multi-layer projection averaging, multi-example label averaging,
  frozen ObjectiveResult mutation guard.
- harness/trainer.py — TernaryTrainer, the master MLX training loop.
  Composition only — every primitive delegated to projector + annotator
  + objective + scheduler. Model-agnostic via injected loss_fn callable
  (params, x, y) → mx.array. train_step() reads tau/alpha from the
  scheduler at the given step, computes task_loss via loss_fn, projects
  every non-protected param via projector, builds an aggregated
  ProjectionResult representing the step's mean sparsity, annotates
  against each label, runs objective.compute() for the composite loss,
  computes grad_norm via mx.grad(loss_fn) on the dict-typed param arg.
  TrainStepResult frozen dataclass with all primitive Python types.
  log_step() flattens to a JSON-serialisable dict for ConfidenceEventLog³
  — no mx.arrays leak through. Default protect_patterns match
  terncore.ste_trainer.PROTECT_PATTERNS so QAT and TFH paths agree on
  which params are training-eligible. Phase 1 scope note in docstring:
  the loss_fn uses raw params; the projector runs over those same params
  as a parallel observation. Phase 2 will add an optional
  projected_loss_fn that lets gradients flow through the soft projection
  via the model's forward hook.
- harness/tests/test_trainer.py — 12 tests including the dict-typed
  mx.grad pattern verification: train_step structure, input validation,
  rejects when all params protected, scheduler integration (tau/alpha
  per step), warmup window reduces total_loss to task_loss, grad_norm
  positive, grad_norm matches manual mx.grad computation, log_step
  JSON-serialisable + zero mx.arrays, frozen TrainStepResult, default
  protect_patterns include the ste_trainer set, named protected params
  excluded from projection.
- harness/checkpointer.py — HarnessCheckpointer, .see3 JSON
  persistence following the persistence.py pattern (NOT a literal wrap
  of GuardianPersistence — the pattern is what we inherit, the
  contents are TFH-specific). save / load / exists / delete contract.
  Schema version locked at "tfh/1.0" — load() rejects any other version
  with a clear error. Filename: tfh_step_{step:06d}.see3 under a
  configurable output_dir. Atomic write: payload to .tmp first, then
  Path.rename moves into place — a crash mid-write leaves any prior
  checkpoint intact. CheckpointData frozen dataclass {step,
  schema_version, model_params, confidence_metadata, harness_config,
  saved_at}. model_params stored as nested Python lists via
  mx.array.tolist() — load() returns the lists as-is, caller lifts
  back to mx.array. confidence_metadata pulls annotation_summary +
  objective fields out of TrainStepResult. CRITICAL CONTINUITY note in
  docstring: the .see3 format is the input to a future tern-compiler
  v0.4+ step that will produce a .tern-pkg loadable by
  tern-runtime/loader/pkg_loader.py without re-quantisation. Phase 1
  honesty: the compiler step is not yet built; this file defines the
  on-disk format that the eventual compiler will consume.
- harness/tests/test_checkpointer.py — 14 tests: save creates the
  .see3 file, filename format zero-padded to 6 digits, output_dir
  auto-created, load round-trips step + schema_version + config +
  metadata + model_params (reconstructible into mx.arrays of the
  original shape), load rejects missing file, load rejects unknown
  schema_version (the format-drift trip-wire), schema constant locked
  at "tfh/1.0", saved payload carries the version, exists() true /
  false, delete() removes file and is idempotent, atomic write uses
  .tmp then Path.rename (verified by patching Path.rename and
  asserting the call), no .tmp left behind after success, save
  overwrites existing checkpoint cleanly with no duplicate or
  leftover files.

## TFH Sprint Status: COMPLETE
120/120 tests passing end-to-end. Phase 1: epistemic_state +
scheduler + projector + annotator + objective.  Phase 2: trainer +
checkpointer.  All four contract invariants holding green:
  1. Cross-repo string match (epistemic_state ↔ confidence_emitter)
  2. Cross-repo threshold match (annotator ↔ confidence_emitter)
  3. STE numerical agreement (projector ↔ ste.STEQuantize)
  4. Frozen-dataclass immutability across every result type
The training-time epistemic vocabulary now flows from raw weight
projections through annotation, composite loss, gradient computation,
and .see3 checkpoint serialisation — and the cross-repo trip-wires
guarantee a TFH checkpoint's labels round-trip into LIS inference
confidence without any translation layer.
