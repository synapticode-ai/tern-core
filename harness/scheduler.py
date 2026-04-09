# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — TFH composite loss
# function annealing schedule (flag to Rod).
"""
AdaptationScheduler — annealing schedules for TernaryProjector³ and
ConfidenceObjective³.

Pure Python, zero dependencies. This file is the foundation that
``harness/projector.py`` (tau) and ``harness/objective.py`` (alpha)
both consume. Building it first means projector and objective can be
written without circular imports or framework-specific helpers.

Two schedules
=============

``tau`` — TernaryProjector³ soft-projection temperature.
    Linear anneal from ``initial_tau`` (default 1.0) to ``final_tau``
    (default 0.01) over ``total_steps``. Higher tau = softer projection
    (gradient flow through the ternary boundary). Lower tau = harder
    projection (forward pass approaches the discrete {-1, 0, +1}
    states). The schedule must end well before the final step so the
    last training updates see a near-discrete forward pass.

``alpha`` — ConfidenceObjective³ confidence-loss weight.
    Held at 0.0 for the first ``alpha_warmup_steps`` (default 2000),
    then rises linearly from 0.0 to 1.0 over the remaining steps.
    Reaches exactly 1.0 at ``total_steps`` and clamps there. The
    warmup window gives the model time to learn the task loss before
    the calibration penalty kicks in — without it, early-training
    instability tends to teach the model that under-confidence is
    safe, and it never recovers.

Both schedules clamp at the boundaries: negative steps return the
initial value, steps beyond ``total_steps`` return the final value.
This makes the scheduler safe to call from validation hooks and
checkpoint loaders that may pass step indices outside the training
range.

Reference: SPEC-TFH-001 § 9, harness.yaml ``ternary_projector`` and
``confidence_objective`` sections.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationScheduler:
    """Frozen schedule configuration. ``tau()``, ``alpha()``, and
    ``progress()`` are pure functions of ``step``.

    Args:
        total_steps: Total number of training steps. Must be > 0.
        initial_tau: Starting temperature for the soft projection.
            Default 1.0 per harness.yaml. Must be > 0 and >= final_tau.
        final_tau: Ending temperature. Default 0.01 per harness.yaml.
            Must be > 0.
        alpha_warmup_steps: Number of initial steps during which the
            confidence loss weight stays at 0.0. Default 2000 per
            harness.yaml. Must be in [0, total_steps].
    """

    total_steps: int
    initial_tau: float = 1.0
    final_tau: float = 0.01
    alpha_warmup_steps: int = 2000

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError(
                f"total_steps must be > 0, got {self.total_steps}"
            )
        if self.initial_tau <= 0:
            raise ValueError(
                f"initial_tau must be > 0, got {self.initial_tau}"
            )
        if self.final_tau <= 0:
            raise ValueError(
                f"final_tau must be > 0, got {self.final_tau}"
            )
        if self.initial_tau < self.final_tau:
            raise ValueError(
                f"initial_tau ({self.initial_tau}) must be >= "
                f"final_tau ({self.final_tau}); the schedule anneals "
                f"from hot to cold, not the reverse"
            )
        if self.alpha_warmup_steps < 0:
            raise ValueError(
                f"alpha_warmup_steps must be >= 0, got {self.alpha_warmup_steps}"
            )
        if self.alpha_warmup_steps > self.total_steps:
            raise ValueError(
                f"alpha_warmup_steps ({self.alpha_warmup_steps}) cannot "
                f"exceed total_steps ({self.total_steps}) — the rise "
                f"window would be negative"
            )

    # ----------------------------------------------------------- tau

    def tau(self, step: int) -> float:
        """Soft-projection temperature at ``step``.

        Linear interpolation from ``initial_tau`` (at step 0) to
        ``final_tau`` (at step ``total_steps``). Clamped at both ends.
        """
        if step <= 0:
            return self.initial_tau
        if step >= self.total_steps:
            return self.final_tau
        fraction = step / self.total_steps
        return self.initial_tau - fraction * (self.initial_tau - self.final_tau)

    # ----------------------------------------------------------- alpha

    def alpha(self, step: int) -> float:
        """Confidence-loss weight at ``step``.

        Returns 0.0 for ``step < alpha_warmup_steps``. Then rises
        linearly from 0.0 (at ``alpha_warmup_steps``) to 1.0 (at
        ``total_steps``). Clamped to [0.0, 1.0] at both ends.
        """
        if step < self.alpha_warmup_steps:
            return 0.0
        if step >= self.total_steps:
            return 1.0
        rise_window = self.total_steps - self.alpha_warmup_steps
        if rise_window <= 0:
            # Degenerate: warmup == total. Already handled by the
            # >= total branch above; this guard exists only as a
            # belt-and-braces against future refactors.
            return 1.0
        progress = (step - self.alpha_warmup_steps) / rise_window
        return max(0.0, min(1.0, progress))

    # ----------------------------------------------------------- progress

    def progress(self, step: int) -> dict:
        """Snapshot of the schedule state at ``step``.

        Suitable for ConfidenceEventLog³ entries and live dashboards.
        ``warmup_active`` is True iff alpha is still being held at 0.0.
        """
        clamped_step = max(0, min(step, self.total_steps))
        pct_complete = clamped_step / self.total_steps
        return {
            "step": step,
            "total_steps": self.total_steps,
            "tau": self.tau(step),
            "alpha": self.alpha(step),
            "pct_complete": pct_complete,
            "warmup_active": step < self.alpha_warmup_steps,
        }
