# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — TFH ternary projection
# with temperature annealing and three-state zero attractor (flag to Rod).
"""
TernaryProjector — soft ternary projection for the Ternary-Aware
Fine-Tuning Harness.

Mathematical contract from ``ste.STEQuantize``, reimplemented in MLX.
``ste.py`` is untouched and the PyTorch path is unaffected.

Why this is a re-implementation, not a wrap
============================================
PyTorch and MLX do not share an autograd graph — a literal wrap is
impossible. The intent is to preserve the mathematical contract
exactly: every value the projector produces at ``tau → 0`` must be
numerically indistinguishable from what
``terncore.ste.STEQuantize.apply(weights, threshold=0.7)`` produces in
PyTorch on the same input bytes. The numerical-agreement test
asserts this property against the actual PyTorch reference.

The contract (cribbed from ``ste.py`` lines 60-83):

    abs_w      = |w|
    threshold  = scale * mean(|w|)            # scale=0.7 by default
    ternary    = +1  where w >  threshold
                  0  where |w| <= threshold   # the deadband — sparsity
                 -1  where w < -threshold
    alpha      = mean(|w|)  over positions where ternary != 0
                 (fallback: mean(|w|) over all positions if all-zero)
    dequant    = ternary * alpha
    backward   = identity (gradient passes through unchanged)

Soft projection at tau > 0
==========================
At inference (tau → 0) the projection is hard. During training the
schedule starts at tau = 1.0 and anneals toward tau = 0.01, allowing
gradients to flow through a smoothed approximation of the sign
function:

    soft_active = tanh(w / tau)               # → sign(w) as tau → 0

The deadband — the three-state zero attractor — is enforced by the
SAME threshold mask in soft mode as in hard mode. Positions in the
deadband are exactly 0.0; positions in the active band are
``tanh(w/tau)``. As tau falls the active values saturate toward ±1
and the mask shape is unchanged, so the soft projection converges
smoothly to the hard projection.

This is the three-state extension of the standard binary STE: the
zero state is preserved through the entire schedule, not just at
the end.

Alpha is computed from the HARD mask (above-threshold positions
of the original weights), not from the soft output. This keeps
``alpha`` stable across the tau anneal — the soft projection is
just a gradient signal; the per-step scaling factor is the same one
the hard projection would produce on the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


# ---------------------------------------------------------------------------
# Projection result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectionResult:
    """Frozen output of a single ``TernaryProjector.project`` call.

    ``weights_ternary`` holds the ternary states. At hard projection
    (tau → 0) every entry is exactly -1, 0, or +1. At soft projection
    (tau > 0) the deadband entries are still exactly 0, but active
    entries are ``tanh(w/tau)`` continuous values that approach ±1
    as tau falls.

    ``weights_dequant`` is ``weights_ternary * alpha`` — the value
    used by the linear layer's matrix multiply during the forward
    pass.

    ``alpha``, ``sparsity``, and ``threshold`` are Python floats so
    they can be logged into ConfidenceEventLog³ and harness.yaml
    snapshots without MLX-specific serialisation.
    """

    weights_ternary: mx.array
    weights_dequant: mx.array
    alpha: float
    sparsity: float
    threshold: float


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class TernaryProjector:
    """MLX ternary projector. Reimplements the ``ste.STEQuantize``
    mathematical contract with a soft annealing path on top.

    Args:
        threshold_scale: Multiplier on ``mean(|w|)`` that defines the
            deadband cut point. Default 0.7 — matches the documented
            tern-core operating range and the ``STEQuantize`` default.
            Tunable per call via ``project(..., threshold_scale=...)``
            for sensitivity sweeps; not normally changed.
    """

    DEFAULT_THRESHOLD_SCALE = 0.7

    # Below this tau the soft formula is replaced by the exact hard
    # projection — avoids tanh(w / 0) and matches the ste.py contract
    # bit-for-bit. The numerical-agreement test runs at tau = 1e-6.
    HARD_TAU_EPSILON = 1e-5

    def __init__(self, threshold_scale: float = DEFAULT_THRESHOLD_SCALE) -> None:
        if threshold_scale <= 0:
            raise ValueError(
                f"threshold_scale must be > 0, got {threshold_scale}"
            )
        self.threshold_scale = threshold_scale

    # ----------------------------------------------------------- compute_threshold

    def compute_threshold(
        self,
        weights: mx.array,
        threshold_scale: Optional[float] = None,
    ) -> float:
        """Return the deadband cut point for the given weight tensor.

        Exposed as a separate method so tests can verify the threshold
        formula matches ``ste.STEQuantize`` exactly without going
        through the full projection. The formula is

            threshold = threshold_scale * mean(|w|)

        per the ``ste.py`` contract.
        """
        scale = threshold_scale if threshold_scale is not None else self.threshold_scale
        abs_w = mx.abs(weights)
        return float(scale * mx.mean(abs_w).item())

    # ----------------------------------------------------------- project

    def project(
        self,
        weights: mx.array,
        tau: float,
        threshold_scale: Optional[float] = None,
    ) -> ProjectionResult:
        """Project ``weights`` to the ternary state space at the given
        soft-projection temperature.

        Args:
            weights: MLX array of any shape. Treated element-wise.
            tau: Soft-projection temperature. ``tau <= HARD_TAU_EPSILON``
                triggers the exact hard projection (the ``ste.py``
                contract). ``tau > HARD_TAU_EPSILON`` uses the soft
                tanh approximation in the active band, with the same
                deadband mask as the hard path.
            threshold_scale: Optional override for this call. Default
                is the projector's instance value (0.7).

        Returns:
            ``ProjectionResult`` with ternary, dequantised, alpha,
            sparsity, and threshold fields.
        """
        if tau < 0:
            raise ValueError(f"tau must be >= 0, got {tau}")

        scale = threshold_scale if threshold_scale is not None else self.threshold_scale
        abs_w = mx.abs(weights)
        threshold = float(scale * mx.mean(abs_w).item())

        # Boolean masks — non-differentiable, used to define the
        # deadband and the alpha-averaging set.
        pos_mask = weights > threshold
        neg_mask = weights < -threshold
        active_mask = pos_mask | neg_mask

        if tau <= self.HARD_TAU_EPSILON:
            # Exact hard projection — same formula as ste.STEQuantize.
            ternary = mx.where(
                pos_mask,
                mx.ones_like(weights),
                mx.where(
                    neg_mask,
                    -mx.ones_like(weights),
                    mx.zeros_like(weights),
                ),
            )
        else:
            # Soft projection in the active band; deadband stays exactly 0.
            soft = mx.tanh(weights / tau)
            ternary = mx.where(active_mask, soft, mx.zeros_like(weights))

        # Alpha: mean of |w| over the hard active mask.
        # Computed from the hard mask (not the soft output) so alpha
        # is stable across the tau anneal — see module docstring.
        active_float = active_mask.astype(mx.float32)
        non_zero_count = float(mx.sum(active_float).item())
        if non_zero_count > 0:
            masked_abs_sum = float(mx.sum(abs_w * active_float).item())
            alpha = masked_abs_sum / non_zero_count
        else:
            # Fallback: all weights below threshold (or all zero).
            # Match the ste.py fallback exactly.
            alpha = float(mx.mean(abs_w).item())

        weights_dequant = ternary * alpha

        total_elements = int(weights.size)
        sparsity = (
            1.0 - (non_zero_count / total_elements)
            if total_elements > 0
            else 1.0
        )

        return ProjectionResult(
            weights_ternary=ternary,
            weights_dequant=weights_dequant,
            alpha=alpha,
            sparsity=sparsity,
            threshold=threshold,
        )
