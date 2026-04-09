# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
"""Tests for harness.epistemic_state — pure-Python, no ML deps.

The headline test is ``test_cross_repo_string_match``: it imports
both this enum and the LIS-side enum from tern-runtime, and asserts
their string values agree byte-for-byte. That test is the day-one
trip-wire for the TFH's "training-time epistemic calibration survives
into inference-time agent behaviour" continuity claim. If it ever
fails, the two repos have drifted and a TFH-trained checkpoint's
labels no longer round-trip into the LIS runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


# Make tern-core/harness importable when pytest is run from anywhere
HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

# Make tern-runtime importable for the cross-repo string-match test
TERN_RUNTIME_SRC = (
    Path(__file__).resolve().parents[3] / "tern-runtime" / "src"
)
if str(TERN_RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(TERN_RUNTIME_SRC))


from harness.epistemic_state import (
    Domain,
    EpistemicLabel,
    EpistemicState,
)


# ---------------------------------------------------------------------------
# EpistemicState basics
# ---------------------------------------------------------------------------

def test_epistemic_state_string_values():
    """Three lowercase string values, no surprises."""
    assert EpistemicState.CONFIRMED.value == "confirmed"
    assert EpistemicState.UNCERTAIN.value == "uncertain"
    assert EpistemicState.DISCONFIRMED.value == "disconfirmed"


def test_epistemic_state_from_string_round_trip():
    for state in EpistemicState:
        assert EpistemicState.from_string(state.value) is state


def test_epistemic_state_from_string_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown EpistemicState"):
        EpistemicState.from_string("maybe")


def test_epistemic_state_from_string_rejects_uppercase():
    """The vocabulary is lowercase. Uppercase strings must not silently
    coerce — that would mask drift between this repo and tern-runtime."""
    with pytest.raises(ValueError):
        EpistemicState.from_string("CONFIRMED")


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def test_domain_string_values():
    assert Domain.FACTUAL.value == "factual"
    assert Domain.REASONING.value == "reasoning"
    assert Domain.CREATIVE.value == "creative"
    assert Domain.AGENTIC.value == "agentic"


def test_domain_from_string_round_trip():
    for d in Domain:
        assert Domain.from_string(d.value) is d


def test_domain_from_string_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown Domain"):
        Domain.from_string("scientific")


# ---------------------------------------------------------------------------
# EpistemicLabel — the per-example annotation record
# ---------------------------------------------------------------------------

def test_epistemic_label_construction():
    label = EpistemicLabel(
        epistemic_state=EpistemicState.CONFIRMED,
        confidence_score=0.92,
        escalate=False,
        domain=Domain.FACTUAL,
        source_reliability=0.85,
    )
    assert label.epistemic_state is EpistemicState.CONFIRMED
    assert label.confidence_score == 0.92
    assert label.escalate is False
    assert label.domain is Domain.FACTUAL
    assert label.source_reliability == 0.85


def test_epistemic_label_is_frozen():
    """EpistemicLabel must be immutable — labels travel through the
    training loop and must never be mutated mid-step."""
    label = EpistemicLabel(
        epistemic_state=EpistemicState.UNCERTAIN,
        confidence_score=0.50,
        escalate=True,
        domain=Domain.REASONING,
        source_reliability=0.60,
    )
    import dataclasses
    with pytest.raises(dataclasses.FrozenInstanceError):
        label.confidence_score = 0.99  # type: ignore[misc]


def test_epistemic_label_validates_confidence_score():
    """confidence_score must be in [0.0, 1.0]."""
    with pytest.raises(ValueError, match="confidence_score"):
        EpistemicLabel(
            epistemic_state=EpistemicState.CONFIRMED,
            confidence_score=1.5,
            escalate=False,
            domain=Domain.FACTUAL,
            source_reliability=0.9,
        )
    with pytest.raises(ValueError, match="confidence_score"):
        EpistemicLabel(
            epistemic_state=EpistemicState.CONFIRMED,
            confidence_score=-0.1,
            escalate=False,
            domain=Domain.FACTUAL,
            source_reliability=0.9,
        )


def test_epistemic_label_validates_source_reliability():
    with pytest.raises(ValueError, match="source_reliability"):
        EpistemicLabel(
            epistemic_state=EpistemicState.UNCERTAIN,
            confidence_score=0.5,
            escalate=False,
            domain=Domain.FACTUAL,
            source_reliability=2.0,
        )


def test_epistemic_label_to_dict_matches_spec_schema():
    """SPEC-TFH-001 § 4.1 declares an exact JSON shape — the round-trip
    must produce that exact set of keys with the right value types."""
    label = EpistemicLabel(
        epistemic_state=EpistemicState.DISCONFIRMED,
        confidence_score=0.10,
        escalate=True,
        domain=Domain.AGENTIC,
        source_reliability=0.30,
    )
    d = label.to_dict()
    assert set(d.keys()) == {
        "epistemic_state", "confidence_score", "escalate",
        "domain", "source_reliability",
    }
    assert d["epistemic_state"] == "disconfirmed"
    assert d["confidence_score"] == 0.10
    assert d["escalate"] is True
    assert d["domain"] == "agentic"
    assert d["source_reliability"] == 0.30


def test_epistemic_label_round_trips_via_json():
    """Encode → JSON → decode must preserve every field exactly."""
    original = EpistemicLabel(
        epistemic_state=EpistemicState.UNCERTAIN,
        confidence_score=0.55,
        escalate=False,
        domain=Domain.CREATIVE,
        source_reliability=0.72,
    )
    encoded = json.dumps(original.to_dict())
    decoded = EpistemicLabel.from_dict(json.loads(encoded))
    assert decoded == original


def test_epistemic_label_from_dict_rejects_missing_keys():
    incomplete = {
        "epistemic_state": "confirmed",
        "confidence_score": 0.9,
        # missing: escalate, domain, source_reliability
    }
    with pytest.raises(ValueError, match="missing required keys"):
        EpistemicLabel.from_dict(incomplete)


def test_epistemic_label_from_dict_rejects_unknown_state_string():
    bad = {
        "epistemic_state": "unsure",  # routing vocab leaking in
        "confidence_score": 0.5,
        "escalate": False,
        "domain": "factual",
        "source_reliability": 0.5,
    }
    with pytest.raises(ValueError, match="Unknown EpistemicState"):
        EpistemicLabel.from_dict(bad)


# ---------------------------------------------------------------------------
# CROSS-REPO STRING MATCH — the day-one trip-wire
# ---------------------------------------------------------------------------

def test_cross_repo_string_match():
    """The TFH EpistemicState (this file) and the LIS EpistemicState
    (tern-runtime/inspector/confidence_emitter.py) must declare
    byte-identical string values for every member.

    If this test fails, training-time labels and inference-time
    confidence have drifted apart. The TFH's whole architectural
    point — that a model's training-time epistemic calibration
    survives into inference-time agent behaviour without conversion
    — depends on these two enums agreeing forever.

    This is the trip-wire. Do not weaken this test by mapping or
    aliasing values. If a future schema bump requires changing the
    strings, change them in BOTH repos in the same commit and
    update this test along with them.
    """
    try:
        from tern_runtime.inspector.confidence_emitter import (
            EpistemicState as LIS_EpistemicState,
        )
    except ImportError as e:
        pytest.skip(
            f"tern-runtime not importable from this venv: {e}. "
            f"The trip-wire requires both repos in PYTHONPATH."
        )

    tfh_values = {state.name: state.value for state in EpistemicState}
    lis_values = {state.name: state.value for state in LIS_EpistemicState}

    assert tfh_values == lis_values, (
        f"TFH and LIS EpistemicState string values have diverged. "
        f"TFH={tfh_values} LIS={lis_values}. "
        f"Fix BOTH repos in the same commit before proceeding."
    )

    # Belt and braces — also check the three specific values directly,
    # so a failure message points at the exact string that broke.
    assert EpistemicState.CONFIRMED.value == LIS_EpistemicState.CONFIRMED.value == "confirmed"
    assert EpistemicState.UNCERTAIN.value == LIS_EpistemicState.UNCERTAIN.value == "uncertain"
    assert EpistemicState.DISCONFIRMED.value == LIS_EpistemicState.DISCONFIRMED.value == "disconfirmed"
