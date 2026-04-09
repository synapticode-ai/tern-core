# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.projector — TernaryProjector MLX implementation.

The headline test is ``test_numerical_agreement_with_ste_quantize``:
it imports ``terncore.ste.STEQuantize`` (the existing PyTorch reference
implementation), runs both side by side on the same input bytes, and
asserts the MLX projector at hard tau matches the PyTorch reference
to floating-point tolerance. If that test ever fails, the TFH
projector has drifted from the contract baked into ste.py and the
"training-time math equals inference-time math" guarantee is broken.

ste.py is imported here ONLY as a reference oracle. It is not
modified, not extended, not wrapped — these tests prove the MLX
re-implementation agrees with it numerically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import mlx.core as mx

# tern-core/harness importable
HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

# tern-core/src importable for ste.STEQuantize reference
TERN_CORE_SRC = Path(__file__).resolve().parents[3] / "src"
if str(TERN_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(TERN_CORE_SRC))

from harness.projector import ProjectionResult, TernaryProjector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seeded_weights(shape=(32, 16), seed: int = 42) -> np.ndarray:
    """Reproducible Xavier-ish weight tensor as a numpy array."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.2, size=shape).astype(np.float32)


def _np_to_mx(arr: np.ndarray) -> mx.array:
    return mx.array(arr.tolist())


# ---------------------------------------------------------------------------
# Numerical agreement with ste.STEQuantize — the headline test
# ---------------------------------------------------------------------------

def test_numerical_agreement_with_ste_quantize():
    """Run the same weight bytes through STEQuantize (PyTorch reference)
    and TernaryProjector at hard tau, and assert they agree.

    Tolerance:
        ternary outputs:  1e-4   (any drift > this is real math drift)
        threshold:        1e-6   (computed from the same mean|w|)
        sparsity:         1e-4   (count-based, exact match expected)
        alpha:            1e-5   (mean over the same mask)
    """
    try:
        import torch
        from terncore.ste import STEQuantize
    except ImportError as e:
        pytest.skip(f"PyTorch / ste.py reference not importable: {e}")

    np_weights = _seeded_weights(shape=(64, 32), seed=42)

    # PyTorch reference path
    torch_weights = torch.from_numpy(np_weights.copy())
    torch_out = STEQuantize.apply(torch_weights, 0.7)
    torch_out_np = torch_out.detach().numpy()

    # Recover the reference threshold and alpha from STEQuantize's
    # documented formula, since the function returns ternary*alpha
    # not the components separately.
    abs_w = np.abs(np_weights)
    ref_threshold = 0.7 * float(np.mean(abs_w))
    ref_pos = np_weights > ref_threshold
    ref_neg = np_weights < -ref_threshold
    ref_active = ref_pos | ref_neg
    ref_alpha = (
        float(np.mean(abs_w[ref_active])) if ref_active.any()
        else float(np.mean(abs_w))
    )
    ref_ternary = np.zeros_like(np_weights)
    ref_ternary[ref_pos] = 1.0
    ref_ternary[ref_neg] = -1.0
    ref_sparsity = 1.0 - (ref_active.sum() / np_weights.size)

    # Sanity: STEQuantize output equals ref_ternary * ref_alpha to FP32 precision
    np.testing.assert_allclose(
        torch_out_np, ref_ternary * ref_alpha, atol=1e-6,
        err_msg="ste.STEQuantize output disagrees with manually-computed reference"
    )

    # MLX projector path
    mx_weights = _np_to_mx(np_weights)
    projector = TernaryProjector(threshold_scale=0.7)
    proj = projector.project(mx_weights, tau=1e-6)

    mx_ternary_np = np.array(proj.weights_ternary.tolist(), dtype=np.float32)
    mx_dequant_np = np.array(proj.weights_dequant.tolist(), dtype=np.float32)

    # The headline assertions
    np.testing.assert_allclose(
        mx_ternary_np, ref_ternary, atol=1e-4,
        err_msg="TernaryProjector ternary output diverged from STEQuantize"
    )
    np.testing.assert_allclose(
        mx_dequant_np, torch_out_np, atol=1e-4,
        err_msg="TernaryProjector dequantised output diverged from STEQuantize"
    )
    assert proj.threshold == pytest.approx(ref_threshold, abs=1e-6)
    assert proj.alpha == pytest.approx(ref_alpha, abs=1e-5)
    assert proj.sparsity == pytest.approx(ref_sparsity, abs=1e-4)


def test_compute_threshold_matches_ste_formula():
    """The compute_threshold static helper must produce the same value
    that ``0.7 * mean(|w|)`` does, byte-for-byte (within FP32 noise)."""
    np_weights = _seeded_weights(shape=(128,), seed=7)
    expected = 0.7 * float(np.mean(np.abs(np_weights)))
    projector = TernaryProjector(threshold_scale=0.7)
    actual = projector.compute_threshold(_np_to_mx(np_weights))
    assert actual == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Hard projection at tau = 0 — produces exact ternary states
# ---------------------------------------------------------------------------

def test_hard_projection_at_zero_tau():
    """tau=0 must produce a strictly-ternary output: every entry is
    exactly -1.0, 0.0, or +1.0 — no continuous values."""
    weights = _np_to_mx(_seeded_weights(seed=11))
    projector = TernaryProjector()
    proj = projector.project(weights, tau=0.0)

    ternary_np = np.array(proj.weights_ternary.tolist(), dtype=np.float32)
    unique_values = set(np.unique(ternary_np).tolist())
    assert unique_values.issubset({-1.0, 0.0, 1.0}), (
        f"Hard projection produced non-ternary values: {unique_values}"
    )
    # And at least one of each sign should be present in a sensible
    # weight tensor (sanity)
    assert -1.0 in unique_values
    assert 0.0 in unique_values
    assert 1.0 in unique_values


def test_hard_and_epsilon_tau_agree():
    """tau=0 and tau=HARD_TAU_EPSILON / 10 must produce identical
    output — both should fall through to the exact hard branch."""
    weights = _np_to_mx(_seeded_weights(seed=99))
    projector = TernaryProjector()
    a = projector.project(weights, tau=0.0)
    b = projector.project(weights, tau=TernaryProjector.HARD_TAU_EPSILON / 10)
    np.testing.assert_array_equal(
        np.array(a.weights_ternary.tolist()),
        np.array(b.weights_ternary.tolist()),
    )


# ---------------------------------------------------------------------------
# Soft projection differentiability
# ---------------------------------------------------------------------------

def test_soft_projection_is_differentiable():
    """At tau=1.0, mx.grad of sum(weights_ternary) w.r.t. weights must
    return a non-trivial gradient. This is the property that lets the
    soft projection participate in the training loop's backward pass."""
    weights = _np_to_mx(_seeded_weights(shape=(8, 8), seed=3))
    projector = TernaryProjector()

    def loss_fn(w: mx.array) -> mx.array:
        proj = projector.project(w, tau=1.0)
        return mx.sum(proj.weights_ternary)

    grad_fn = mx.grad(loss_fn)
    grads = grad_fn(weights)

    grads_np = np.array(grads.tolist(), dtype=np.float32)
    assert grads_np.shape == (8, 8)
    nonzero_count = int(np.sum(grads_np != 0))
    assert nonzero_count > 0, (
        f"Soft projection produced an all-zero gradient at tau=1.0 — "
        f"the active band must contribute non-zero gradients"
    )
    # The gradient at active positions is (1 - tanh²(w/tau))/tau, which
    # for typical weights at tau=1.0 sits in (0, 1).
    assert float(np.max(np.abs(grads_np))) <= 1.0 + 1e-6


def test_soft_projection_anneals_toward_hard():
    """As tau falls, the soft output must converge to the hard output.
    Cosine similarity at tau=1.0 should be lower than at tau=0.05,
    which in turn should be near 1.0 against the hard reference."""
    weights = _np_to_mx(_seeded_weights(shape=(64, 32), seed=21))
    projector = TernaryProjector()

    hard = np.array(projector.project(weights, tau=0.0).weights_ternary.tolist())
    soft_hot = np.array(projector.project(weights, tau=1.0).weights_ternary.tolist())
    soft_cold = np.array(projector.project(weights, tau=0.05).weights_ternary.tolist())

    def cos(a, b):
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
        return float(np.dot(a_flat, b_flat) / denom) if denom > 0 else 0.0

    sim_hot = cos(soft_hot, hard)
    sim_cold = cos(soft_cold, hard)

    assert sim_cold > sim_hot, (
        f"Soft projection should be closer to hard at lower tau: "
        f"sim(tau=0.05)={sim_cold:.4f} vs sim(tau=1.0)={sim_hot:.4f}"
    )
    assert sim_cold > 0.95, (
        f"Soft projection at tau=0.05 should be very close to hard, "
        f"got cosine similarity {sim_cold:.4f}"
    )


# ---------------------------------------------------------------------------
# Sparsity behaviour
# ---------------------------------------------------------------------------

def test_sparsity_increases_with_threshold_scale():
    """A higher threshold_scale widens the deadband and increases the
    fraction of zero weights. Strictly monotonic in the limit."""
    weights = _np_to_mx(_seeded_weights(shape=(128, 128), seed=5))

    sparsities = []
    for scale in [0.3, 0.7, 1.0, 1.5, 2.0]:
        projector = TernaryProjector(threshold_scale=scale)
        sparsities.append(projector.project(weights, tau=0.0).sparsity)

    # Strictly increasing
    for i in range(1, len(sparsities)):
        assert sparsities[i] > sparsities[i - 1], (
            f"Sparsity not increasing at scale step {i}: {sparsities}"
        )


# ---------------------------------------------------------------------------
# Alpha computation
# ---------------------------------------------------------------------------

def test_alpha_is_mean_of_nonzero_magnitudes():
    """alpha must equal mean(|w|) over the positions where ternary != 0."""
    np_weights = _seeded_weights(shape=(64, 64), seed=13)
    projector = TernaryProjector(threshold_scale=0.7)
    proj = projector.project(_np_to_mx(np_weights), tau=0.0)

    abs_w = np.abs(np_weights)
    threshold = 0.7 * float(np.mean(abs_w))
    active = (np_weights > threshold) | (np_weights < -threshold)
    expected_alpha = float(np.mean(abs_w[active]))

    assert proj.alpha == pytest.approx(expected_alpha, abs=1e-6)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_projection_result_is_frozen():
    """ProjectionResult is frozen — projection outputs cannot be mutated
    by downstream consumers."""
    import dataclasses
    weights = _np_to_mx(_seeded_weights(seed=1))
    proj = TernaryProjector().project(weights, tau=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        proj.alpha = 99.0  # type: ignore[misc]


def test_all_zero_weights_handled_gracefully():
    """An all-zero weight tensor must not divide by zero. The fallback
    matches ste.STEQuantize: alpha = mean(|w|) over the full tensor
    (which is also 0 in this case)."""
    weights = mx.zeros((16, 16), dtype=mx.float32)
    projector = TernaryProjector()
    proj = projector.project(weights, tau=0.0)
    assert proj.sparsity == 1.0
    assert proj.alpha == 0.0
    assert proj.threshold == 0.0
    ternary_np = np.array(proj.weights_ternary.tolist())
    assert np.all(ternary_np == 0.0)


def test_construction_validates_threshold_scale():
    with pytest.raises(ValueError, match="threshold_scale"):
        TernaryProjector(threshold_scale=0.0)
    with pytest.raises(ValueError, match="threshold_scale"):
        TernaryProjector(threshold_scale=-0.1)


def test_project_validates_tau():
    weights = _np_to_mx(_seeded_weights(seed=1))
    projector = TernaryProjector()
    with pytest.raises(ValueError, match="tau must be >= 0"):
        projector.project(weights, tau=-0.5)


def test_tau_override_does_not_mutate_projector():
    """Per-call threshold_scale override is local — the projector's
    instance value is not mutated."""
    weights = _np_to_mx(_seeded_weights(seed=8))
    projector = TernaryProjector(threshold_scale=0.7)
    proj = projector.project(weights, tau=0.0, threshold_scale=1.5)
    assert projector.threshold_scale == 0.7
    # The override produced a higher threshold than the default would
    proj_default = projector.project(weights, tau=0.0)
    assert proj.threshold > proj_default.threshold
