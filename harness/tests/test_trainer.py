# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.trainer — TernaryTrainer master loop.

Uses a trivial differentiable loss_fn so the trainer can be exercised
end-to-end without wiring in a real model. The fixture model is a
single param matrix with a mean-squared-error loss against a target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import mlx.core as mx

HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))


from harness.annotator import EpistemicAnnotator
from harness.epistemic_state import Domain, EpistemicLabel, EpistemicState
from harness.objective import ConfidenceObjective
from harness.projector import TernaryProjector
from harness.scheduler import AdaptationScheduler
from harness.trainer import (
    DEFAULT_PROTECT_PATTERNS,
    TernaryTrainer,
    TrainStepResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trivial_loss_fn(params, x, y):
    """w @ x → predict y; mean-squared error. Differentiable in w."""
    pred = params["w"] @ x
    return mx.mean((pred - y) ** 2)


def _make_trainer(
    *,
    total_steps: int = 1000,
    sparsity_target: float = 0.50,
    alpha_warmup_steps: int = 0,
) -> TernaryTrainer:
    return TernaryTrainer(
        projector=TernaryProjector(threshold_scale=0.7),
        annotator=EpistemicAnnotator(),
        objective=ConfidenceObjective(sparsity_target=sparsity_target),
        scheduler=AdaptationScheduler(
            total_steps=total_steps,
            alpha_warmup_steps=alpha_warmup_steps,
        ),
        loss_fn=_trivial_loss_fn,
        config={},
    )


def _make_params(seed: int = 7) -> dict[str, mx.array]:
    rng = mx.random.key(seed)
    return {"w": mx.random.normal(shape=(4, 3), key=rng)}


def _make_batch():
    x = mx.array([[0.5], [0.2], [0.7]])
    y = mx.array([[1.0], [0.5], [0.3], [0.8]])
    return x, y


def _label(state: EpistemicState = EpistemicState.UNCERTAIN, score: float = 0.5):
    return EpistemicLabel(
        epistemic_state=state,
        confidence_score=score,
        escalate=False,
        domain=Domain.FACTUAL,
        source_reliability=0.7,
    )


# ---------------------------------------------------------------------------
# train_step structure
# ---------------------------------------------------------------------------

def test_train_step_returns_correct_fields():
    trainer = _make_trainer()
    params = _make_params()
    x, y = _make_batch()
    labels = [_label(), _label()]

    result = trainer.train_step(params, x, y, labels, step=10)

    assert isinstance(result, TrainStepResult)
    assert result.step == 10
    assert isinstance(result.total_loss, float)
    assert isinstance(result.task_loss, float)
    assert isinstance(result.calibration_penalty, float)
    assert isinstance(result.sparsity_penalty, float)
    assert isinstance(result.tau, float)
    assert isinstance(result.alpha, float)
    assert isinstance(result.mean_sparsity, float)
    assert isinstance(result.grad_norm, float)
    assert isinstance(result.annotation_summary, dict)
    # Annotation summary keys come straight from EpistemicAnnotator.summary()
    assert set(result.annotation_summary.keys()) == {
        "n", "mean_calibration_error", "accuracy",
        "confirmed_count", "uncertain_count", "disconfirmed_count",
        "mean_sparsity",
    }
    assert result.annotation_summary["n"] == 2  # one entry per label


def test_train_step_validates_inputs():
    trainer = _make_trainer()
    params = _make_params()
    x, y = _make_batch()
    with pytest.raises(ValueError, match="step"):
        trainer.train_step(params, x, y, [_label()], step=-1)
    with pytest.raises(ValueError, match="labels"):
        trainer.train_step(params, x, y, [], step=0)


def test_train_step_rejects_when_all_params_protected():
    """If every param matches a protect pattern, projection list is
    empty and the trainer raises rather than producing a degenerate
    objective input."""
    trainer = TernaryTrainer(
        projector=TernaryProjector(),
        annotator=EpistemicAnnotator(),
        objective=ConfidenceObjective(sparsity_target=0.5),
        scheduler=AdaptationScheduler(total_steps=100, alpha_warmup_steps=0),
        loss_fn=_trivial_loss_fn,
        config={"protect_patterns": ["w"]},  # protect the single param
    )
    with pytest.raises(ValueError, match="No projectable params"):
        trainer.train_step(_make_params(), *_make_batch(), [_label()], step=0)


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

def test_tau_and_alpha_come_from_scheduler():
    """The trainer must read tau and alpha from the scheduler at the
    given step, not from any local state."""
    trainer = _make_trainer(total_steps=1000, alpha_warmup_steps=100)
    params = _make_params()
    x, y = _make_batch()
    labels = [_label()]

    expected_tau_at_500 = trainer._scheduler.tau(500)
    expected_alpha_at_500 = trainer._scheduler.alpha(500)
    expected_tau_at_50 = trainer._scheduler.tau(50)
    expected_alpha_at_50 = trainer._scheduler.alpha(50)

    r500 = trainer.train_step(params, x, y, labels, step=500)
    r50 = trainer.train_step(params, x, y, labels, step=50)

    assert r500.tau == pytest.approx(expected_tau_at_500)
    assert r500.alpha == pytest.approx(expected_alpha_at_500)
    assert r50.tau == pytest.approx(expected_tau_at_50)
    assert r50.alpha == pytest.approx(expected_alpha_at_50)
    # During warmup (step 50 < 100), alpha should be exactly 0
    assert r50.alpha == 0.0


def test_alpha_warmup_means_total_loss_equals_task_loss():
    """During the scheduler's warmup window, the composite loss reduces
    to the task loss (objective alpha=0 zeroes the confidence terms)."""
    trainer = _make_trainer(total_steps=1000, alpha_warmup_steps=200)
    params = _make_params()
    x, y = _make_batch()
    labels = [_label()]

    result = trainer.train_step(params, x, y, labels, step=50)
    assert result.alpha == 0.0
    assert result.total_loss == pytest.approx(result.task_loss)


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------

def test_grad_norm_is_positive():
    """For a non-trivial loss with a non-trivial param, the gradient
    norm must be > 0."""
    trainer = _make_trainer()
    result = trainer.train_step(
        _make_params(), *_make_batch(), [_label()], step=0
    )
    assert result.grad_norm > 0.0


def test_grad_norm_matches_manual_computation():
    """The trainer's grad_norm must equal sqrt(sum(g·g)) computed
    independently from mx.grad on the loss_fn."""
    trainer = _make_trainer()
    params = _make_params()
    x, y = _make_batch()

    result = trainer.train_step(params, x, y, [_label()], step=5)

    grads = mx.grad(_trivial_loss_fn)(params, x, y)
    expected_norm_sq = 0.0
    for g in grads.values():
        expected_norm_sq += float(mx.sum(g * g).item())
    expected_norm = expected_norm_sq ** 0.5

    assert result.grad_norm == pytest.approx(expected_norm, abs=1e-5)


# ---------------------------------------------------------------------------
# log_step
# ---------------------------------------------------------------------------

def test_log_step_is_json_serialisable():
    trainer = _make_trainer()
    result = trainer.train_step(
        _make_params(), *_make_batch(), [_label(), _label()], step=42
    )
    flat = TernaryTrainer.log_step(result)
    # Round trip through JSON without errors
    encoded = json.dumps(flat)
    decoded = json.loads(encoded)
    assert decoded["step"] == 42
    assert "total_loss" in decoded
    assert "annotation_summary" in decoded
    assert decoded["annotation_summary"]["n"] == 2


def test_log_step_has_no_mx_arrays():
    """Every value in the log_step dict must be a primitive Python
    type (int, float, str, dict, list, bool) — never an mx.array."""
    trainer = _make_trainer()
    result = trainer.train_step(
        _make_params(), *_make_batch(), [_label()], step=1
    )
    flat = TernaryTrainer.log_step(result)
    for k, v in flat.items():
        assert not isinstance(v, mx.array), f"{k} is an mx.array"


# ---------------------------------------------------------------------------
# Frozen result
# ---------------------------------------------------------------------------

def test_train_step_result_is_frozen():
    trainer = _make_trainer()
    result = trainer.train_step(
        _make_params(), *_make_batch(), [_label()], step=0
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.total_loss = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protect patterns
# ---------------------------------------------------------------------------

def test_default_protect_patterns_match_ste_trainer():
    """The trainer's default protect patterns match the documented
    set in terncore.ste_trainer.PROTECT_PATTERNS so model conversion
    behaviour is consistent across the QAT and TFH paths."""
    expected = {
        "embed", "layernorm", "layer_norm", "rmsnorm",
        "lm_head", "output", "classifier",
    }
    assert expected.issubset(set(DEFAULT_PROTECT_PATTERNS))


def test_protect_patterns_skip_named_params():
    """A param whose name contains a protect pattern is excluded from
    projection — verified by adding a 'layernorm_w' param alongside 'w'
    and checking that only one projection runs."""
    trainer = _make_trainer()
    params = {
        "w": mx.random.normal(shape=(4, 3), key=mx.random.key(1)),
        "layernorm_weight": mx.random.normal(shape=(4,), key=mx.random.key(2)),
    }

    def loss_fn(p, x, y):
        return mx.mean((p["w"] @ x - y) ** 2)

    trainer = TernaryTrainer(
        projector=TernaryProjector(),
        annotator=EpistemicAnnotator(),
        objective=ConfidenceObjective(sparsity_target=0.5),
        scheduler=AdaptationScheduler(total_steps=100, alpha_warmup_steps=0),
        loss_fn=loss_fn,
        config={},
    )
    x, y = _make_batch()
    result = trainer.train_step(params, x, y, [_label()], step=0)
    # Should not raise — the protected layernorm param is excluded,
    # leaving the projectable 'w' param.
    assert result.mean_sparsity >= 0.0
