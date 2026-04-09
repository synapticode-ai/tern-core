# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — TFH master training
# loop integrating projector + annotator + objective + scheduler
# (flag to Rod).
"""
TernaryTrainer — master training loop for the Ternary-Aware Fine-Tuning
Harness.

Composition only. Every primitive is delegated:

  projector  → harness/projector.py        ternary projection per layer
  annotator  → harness/annotator.py        per-step EpistemicState
  objective  → harness/objective.py        composite loss
  scheduler  → harness/scheduler.py        tau / alpha annealing

The trainer's only original work is wiring those four together at each
step, computing the gradient norm via ``mx.grad``, and producing a
single ``TrainStepResult`` that the checkpointer and ConfidenceEventLog³
can both consume.

Model-agnostic by construction
==============================
The trainer accepts a ``loss_fn`` callable at construction:

    loss_fn(params: dict[str, mx.array], x: mx.array, y: mx.array) -> mx.array

The callable computes whatever forward pass and task loss the
downstream model requires. The trainer never sees the model
architecture — it only sees the params dict, the inputs, the targets,
and the resulting scalar loss. This means the same trainer can drive
a Mistral-7B fine-tune or a tiny linear regression test fixture
without any code change.

Phase 1 scope note
==================
The forward pass uses the RAW params (whatever the loss_fn does with
them); the projector runs over those same params to produce projection
results that the annotator and objective consume. The trainer does NOT
inject projected weights back into the loss_fn — that would require a
"forward pass through ternary weights" hook that the model layer must
provide. Phase 2 will add an optional ``projected_loss_fn`` that
receives the projected weights and lets the gradient flow through the
soft projection. For Phase 1 the projection is a parallel observation
on the model's weight distribution, not a transformation of the
forward pass.

This is the same staged decomposition as the LIS sprint: get the
composition right first, then upgrade individual links once the
end-to-end loop is proven.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import mlx.core as mx

from harness.annotator import EpistemicAnnotator, StepAnnotation
from harness.epistemic_state import EpistemicLabel
from harness.objective import ConfidenceObjective, ObjectiveResult
from harness.projector import ProjectionResult, TernaryProjector
from harness.scheduler import AdaptationScheduler


# ---------------------------------------------------------------------------
# Default skip patterns — same as ste_trainer.PROTECT_PATTERNS
# ---------------------------------------------------------------------------

DEFAULT_PROTECT_PATTERNS: tuple[str, ...] = (
    "embed", "layernorm", "layer_norm", "rmsnorm",
    "lm_head", "output", "classifier", "bias",
)


# ---------------------------------------------------------------------------
# Per-step result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainStepResult:
    """Frozen output of a single ``TernaryTrainer.train_step`` call.

    Fields are all primitive Python types so the checkpointer and
    ConfidenceEventLog³ can serialise them straight to JSON without
    any MLX-specific encoding. ``annotation_summary`` is the dict
    returned by ``EpistemicAnnotator.summary()``.
    """

    step: int
    total_loss: float
    task_loss: float
    calibration_penalty: float
    sparsity_penalty: float
    tau: float
    alpha: float
    mean_sparsity: float
    annotation_summary: dict
    grad_norm: float


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TernaryTrainer:
    """Composition layer that drives one TFH training step end to end.

    Args:
        projector: Constructed ``TernaryProjector``.
        annotator: Constructed ``EpistemicAnnotator``.
        objective: Constructed ``ConfidenceObjective``.
        scheduler: Constructed ``AdaptationScheduler``.
        loss_fn: A callable
            ``(params: dict, x: mx.array, y: mx.array) -> mx.array``
            that returns a scalar task loss. The trainer is
            model-agnostic — the loss_fn fully specifies the forward
            pass.
        config: Harness configuration dict (from harness.yaml). Read
            for ``protect_patterns`` only at this layer; everything
            else is consumed by the injected components at their own
            construction time.
    """

    def __init__(
        self,
        projector: TernaryProjector,
        annotator: EpistemicAnnotator,
        objective: ConfidenceObjective,
        scheduler: AdaptationScheduler,
        loss_fn: Callable[[dict, mx.array, mx.array], mx.array],
        config: Optional[dict] = None,
    ) -> None:
        self._projector = projector
        self._annotator = annotator
        self._objective = objective
        self._scheduler = scheduler
        self._loss_fn = loss_fn
        self._config = dict(config or {})

        # Eager-bind the gradient function once. mx.grad is pure, so
        # this is safe to share across train_step calls.
        self._grad_fn = mx.grad(self._loss_fn)

        # Names of params that should be projected. Anything matching
        # a protect pattern (case-insensitive substring) is left alone
        # — same convention as terncore.ste_trainer.STETrainer.
        protect = tuple(
            self._config.get("protect_patterns", DEFAULT_PROTECT_PATTERNS)
        )
        self._protect_patterns = tuple(p.lower() for p in protect)

    # ----------------------------------------------------------- accessors

    @property
    def config(self) -> dict:
        return dict(self._config)

    @property
    def protect_patterns(self) -> tuple[str, ...]:
        return self._protect_patterns

    # ----------------------------------------------------------- train_step

    def train_step(
        self,
        model_params: dict[str, mx.array],
        batch_inputs: mx.array,
        batch_targets: mx.array,
        labels: list[EpistemicLabel],
        step: int,
    ) -> TrainStepResult:
        """Execute one training step end to end.

        Args:
            model_params: Mapping from parameter name to MLX array.
                The trainer projects every entry whose name does NOT
                match a protect pattern.
            batch_inputs: Inputs for the loss_fn forward pass.
            batch_targets: Targets for the loss_fn forward pass.
            labels: One ``EpistemicLabel`` per example in the batch.
                The annotator and objective both consume this list.
            step: Current training step index. Drives the scheduler.

        Returns:
            ``TrainStepResult`` with composite loss components, scheduler
            values, projection sparsity, annotation summary, and the
            gradient norm.

        Raises:
            ValueError: empty params after filtering, empty labels, or
                negative step.
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")
        if not labels:
            raise ValueError("labels must be non-empty")

        # 1. Scheduler
        tau = self._scheduler.tau(step)
        alpha = self._scheduler.alpha(step)

        # 2. Forward pass — task loss only at this layer
        task_loss_array = self._loss_fn(model_params, batch_inputs, batch_targets)
        task_loss = float(task_loss_array.item())

        # 3. Project every non-protected param
        projection_results = self._project_params(model_params, tau)
        if not projection_results:
            raise ValueError(
                "No projectable params — every param matched a protect "
                "pattern. Check protect_patterns configuration."
            )

        # 4. Annotate against the per-example labels.
        # The annotator runs once per label, with each label compared
        # against the SAME aggregated projection summary for the step.
        # This produces one StepAnnotation per example, which the
        # summary() then aggregates into the dict the trainer returns.
        aggregated = self._aggregate_projections(projection_results)
        annotations: list[StepAnnotation] = [
            self._annotator.annotate(aggregated, label) for label in labels
        ]
        annotation_summary = self._annotator.summary(annotations)

        # 5. Composite loss
        objective_result: ObjectiveResult = self._objective.compute(
            task_loss=task_loss,
            projection_results=projection_results,
            labels=labels,
            alpha=alpha,
        )

        # 6. Gradient norm via mx.grad of loss_fn w.r.t. params
        grad_norm = self._compute_grad_norm(model_params, batch_inputs, batch_targets)

        # 7. Build result
        return TrainStepResult(
            step=step,
            total_loss=objective_result.total_loss,
            task_loss=objective_result.task_loss,
            calibration_penalty=objective_result.calibration_penalty,
            sparsity_penalty=objective_result.sparsity_penalty,
            tau=tau,
            alpha=alpha,
            mean_sparsity=objective_result.mean_predicted_sparsity,
            annotation_summary=annotation_summary,
            grad_norm=grad_norm,
        )

    # ----------------------------------------------------------- log_step

    @staticmethod
    def log_step(result: TrainStepResult) -> dict:
        """Flatten a TrainStepResult to a JSON-serialisable dict.

        Suitable for direct insertion into a ConfidenceEventLog³ entry
        or a harness dashboard. All values are primitive Python types
        — no mx.arrays, no enums, no custom objects.
        """
        return {
            "step": int(result.step),
            "total_loss": float(result.total_loss),
            "task_loss": float(result.task_loss),
            "calibration_penalty": float(result.calibration_penalty),
            "sparsity_penalty": float(result.sparsity_penalty),
            "tau": float(result.tau),
            "alpha": float(result.alpha),
            "mean_sparsity": float(result.mean_sparsity),
            "grad_norm": float(result.grad_norm),
            "annotation_summary": dict(result.annotation_summary),
        }

    # ----------------------------------------------------------- helpers

    def _is_protected(self, name: str) -> bool:
        lower = name.lower()
        return any(p in lower for p in self._protect_patterns)

    def _project_params(
        self, params: dict[str, mx.array], tau: float
    ) -> list[ProjectionResult]:
        results = []
        for name, weight in params.items():
            if self._is_protected(name):
                continue
            results.append(self._projector.project(weight, tau=tau))
        return results

    @staticmethod
    def _aggregate_projections(results: list[ProjectionResult]) -> ProjectionResult:
        """Build one representative ProjectionResult for the step.

        The annotator only reads ``sparsity`` (and stores ``alpha``,
        ``threshold`` as passthrough), so the aggregated record can
        leave the array fields as the first projection's arrays — the
        annotator never inspects them.
        """
        n = len(results)
        mean_sparsity = sum(r.sparsity for r in results) / n
        mean_alpha = sum(r.alpha for r in results) / n
        mean_threshold = sum(r.threshold for r in results) / n
        first = results[0]
        return ProjectionResult(
            weights_ternary=first.weights_ternary,
            weights_dequant=first.weights_dequant,
            alpha=mean_alpha,
            sparsity=mean_sparsity,
            threshold=mean_threshold,
        )

    def _compute_grad_norm(
        self,
        params: dict[str, mx.array],
        inputs: mx.array,
        targets: mx.array,
    ) -> float:
        """L2 norm of the flattened gradient dict."""
        grads = self._grad_fn(params, inputs, targets)
        norm_sq = 0.0
        for g in grads.values():
            norm_sq += float(mx.sum(g * g).item())
        return math.sqrt(norm_sq)
