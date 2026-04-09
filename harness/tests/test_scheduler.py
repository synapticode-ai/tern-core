# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.scheduler — pure stdlib, no deps."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from harness.scheduler import AdaptationScheduler


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

def test_construction_defaults():
    sched = AdaptationScheduler(total_steps=10000)
    assert sched.total_steps == 10000
    assert sched.initial_tau == 1.0
    assert sched.final_tau == 0.01
    assert sched.alpha_warmup_steps == 2000


def test_construction_validates_total_steps():
    with pytest.raises(ValueError, match="total_steps"):
        AdaptationScheduler(total_steps=0)
    with pytest.raises(ValueError, match="total_steps"):
        AdaptationScheduler(total_steps=-1)


def test_construction_validates_taus_positive():
    with pytest.raises(ValueError, match="initial_tau"):
        AdaptationScheduler(total_steps=100, initial_tau=0.0)
    with pytest.raises(ValueError, match="final_tau"):
        AdaptationScheduler(total_steps=100, final_tau=-0.01)


def test_construction_rejects_increasing_schedule():
    """The schedule anneals hot → cold. initial < final is rejected."""
    with pytest.raises(ValueError, match="anneals from hot to cold"):
        AdaptationScheduler(total_steps=100, initial_tau=0.01, final_tau=1.0)


def test_construction_rejects_warmup_exceeds_total():
    with pytest.raises(ValueError, match="cannot exceed total_steps"):
        AdaptationScheduler(total_steps=100, alpha_warmup_steps=200)


def test_construction_rejects_negative_warmup():
    with pytest.raises(ValueError, match="alpha_warmup_steps must be >= 0"):
        AdaptationScheduler(total_steps=100, alpha_warmup_steps=-1)


# ---------------------------------------------------------------------------
# tau schedule
# ---------------------------------------------------------------------------

# Tau-only tests use a small total_steps and zero warmup so they can
# focus on tau behaviour without colliding with the production default
# alpha_warmup_steps=2000 (which would be larger than total_steps=1000).

def _tau_only_sched(initial=1.0, final=0.01) -> AdaptationScheduler:
    return AdaptationScheduler(
        total_steps=1000,
        initial_tau=initial,
        final_tau=final,
        alpha_warmup_steps=0,
    )


def test_tau_at_step_zero_is_initial():
    sched = _tau_only_sched()
    assert sched.tau(0) == 1.0


def test_tau_at_final_step_is_final():
    sched = _tau_only_sched()
    assert sched.tau(1000) == 0.01


def test_tau_midpoint_is_linear():
    """At step total/2, tau must equal the average of initial and final.
    Uses positive final_tau because the validator (correctly) requires
    final_tau > 0; the linear-interp check is unaffected."""
    sched = _tau_only_sched(initial=1.0, final=0.1)
    assert sched.tau(500) == pytest.approx(0.55)  # (1.0 + 0.1) / 2

    sched2 = _tau_only_sched(initial=2.0, final=0.5)
    assert sched2.tau(500) == pytest.approx(1.25)  # (2.0 + 0.5) / 2


def test_tau_clamps_below_zero():
    sched = _tau_only_sched()
    assert sched.tau(-100) == 1.0
    assert sched.tau(-1) == 1.0


def test_tau_clamps_above_total():
    sched = _tau_only_sched()
    assert sched.tau(1001) == 0.01
    assert sched.tau(99999) == 0.01


def test_tau_monotonically_decreases():
    """Across the full range, tau should never increase between steps."""
    sched = _tau_only_sched()
    prev = sched.tau(0)
    for step in range(0, 1001, 50):
        current = sched.tau(step)
        assert current <= prev + 1e-12, (
            f"tau increased at step {step}: prev={prev} current={current}"
        )
        prev = current


# ---------------------------------------------------------------------------
# alpha schedule
# ---------------------------------------------------------------------------

def test_alpha_zero_during_warmup():
    """alpha must be 0.0 for every step strictly less than warmup."""
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    assert sched.alpha(0) == 0.0
    assert sched.alpha(1) == 0.0
    assert sched.alpha(1000) == 0.0
    assert sched.alpha(1999) == 0.0


def test_alpha_zero_at_warmup_boundary():
    """At step == alpha_warmup_steps, alpha is at the very start of the
    rise — still 0.0 — and grows from the next step onwards."""
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    assert sched.alpha(2000) == 0.0


def test_alpha_rises_after_warmup():
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    a1 = sched.alpha(2001)
    a2 = sched.alpha(3000)
    a3 = sched.alpha(5000)
    assert a1 > 0.0
    assert a2 > a1
    assert a3 > a2


def test_alpha_at_completion_is_one():
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    assert sched.alpha(10000) == 1.0


def test_alpha_midpoint_after_warmup_is_half():
    """Halfway through the rise window, alpha should be 0.5."""
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    # Rise window: 2000 → 10000 (8000 steps). Midpoint = 6000.
    assert sched.alpha(6000) == pytest.approx(0.5)


def test_alpha_clamps_to_one():
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    assert sched.alpha(10001) == 1.0
    assert sched.alpha(99999) == 1.0


def test_alpha_clamps_to_zero_for_negative_step():
    """Negative steps are below warmup; alpha should be 0.0."""
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    assert sched.alpha(-1) == 0.0
    assert sched.alpha(-1000) == 0.0


def test_alpha_with_zero_warmup_starts_rising_immediately():
    """alpha_warmup_steps = 0 → linear rise from step 0 to total_steps."""
    sched = AdaptationScheduler(total_steps=1000, alpha_warmup_steps=0)
    assert sched.alpha(0) == 0.0
    assert sched.alpha(500) == pytest.approx(0.5)
    assert sched.alpha(1000) == 1.0


def test_alpha_monotonically_non_decreasing():
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    prev = sched.alpha(0)
    for step in range(0, 10001, 100):
        current = sched.alpha(step)
        assert current >= prev - 1e-12, (
            f"alpha decreased at step {step}: prev={prev} current={current}"
        )
        prev = current


# ---------------------------------------------------------------------------
# progress dict
# ---------------------------------------------------------------------------

def test_progress_dict_has_correct_keys():
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    p = sched.progress(5000)
    assert set(p.keys()) == {
        "step", "total_steps", "tau", "alpha", "pct_complete", "warmup_active"
    }
    assert p["step"] == 5000
    assert p["total_steps"] == 10000
    assert p["tau"] == sched.tau(5000)
    assert p["alpha"] == sched.alpha(5000)
    assert p["pct_complete"] == pytest.approx(0.5)
    assert p["warmup_active"] is False


def test_warmup_active_flag():
    sched = AdaptationScheduler(total_steps=10000, alpha_warmup_steps=2000)
    assert sched.progress(0)["warmup_active"] is True
    assert sched.progress(1999)["warmup_active"] is True
    # At exactly the warmup boundary, the flag flips off — alpha is 0.0
    # at this step but is no longer "in warmup" by the < comparator.
    assert sched.progress(2000)["warmup_active"] is False
    assert sched.progress(5000)["warmup_active"] is False


def test_progress_pct_complete_clamps():
    sched = _tau_only_sched()
    assert sched.progress(-100)["pct_complete"] == 0.0
    assert sched.progress(0)["pct_complete"] == 0.0
    assert sched.progress(1000)["pct_complete"] == 1.0
    assert sched.progress(99999)["pct_complete"] == 1.0


# ---------------------------------------------------------------------------
# Frozen dataclass — schedule is configuration, not runtime state
# ---------------------------------------------------------------------------

def test_scheduler_is_frozen():
    """The schedule is configuration, not state. It must not be mutable
    so two callers asking for tau(step) always get the same answer."""
    import dataclasses
    sched = _tau_only_sched()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sched.total_steps = 2000  # type: ignore[misc]
