# Copyright 2026 Gamma Seeds Pte Ltd. Inventor: Robert Lakelin.
# Patent alignment: candidate new provisional — TFH composite loss
# function and EpistemicAnnotator³ weight annotation (flag to Rod).
"""
Epistemic state vocabulary for the Ternary-Aware Fine-Tuning Harness.

This module is the upstream side of the same vocabulary that
``tern-runtime/inspector/confidence_emitter.EpistemicState`` declares
on the inference side. The two enums use byte-identical lowercase
string values so a TFH-trained checkpoint's confidence labels round-
trip into the LIS runtime without any translation layer — that
continuity is the architectural core of the TFH design (SPEC-TFH-001
§ 6.2 "Inference-Time Continuity").

THIS VOCABULARY IS DISTINCT FROM ``terncore.confidence.RoutingConfidence``
=========================================================================

  RoutingConfidence  {SURE, UNSURE, UNKNOWN}
      Source: tern-core/src/terncore/confidence.py
      Domain: routing infrastructure (TernaryRouter, ConfidenceQueue)
      Lifetime: persistent across the agent loop
      Composition: weakest-dominates via stack_confidence()

  EpistemicState     {confirmed, uncertain, disconfirmed}
      Source: this file (training side) +
              tern-runtime/inspector/confidence_emitter.py (inference side)
      Domain: per-example / per-token model output certainty
      Lifetime: ephemeral (one training example, or one generated token)
      Composition: none — point measurements, aggregated by callers

Both enums must coexist. Never merge them, never alias one to the
other, and never let a function accept both as the same parameter.

Cross-repo invariant
====================
The string values declared here MUST equal the string values in
``tern_runtime.inspector.confidence_emitter.EpistemicState``. The
test ``tests/test_epistemic_state.py::test_cross_repo_string_match``
imports both modules and asserts equality — if that test ever fails,
the TFH's training-time labels and the LIS's inference-time
confidence have diverged and the whole continuity claim is broken.

Threshold constants
===================
The probability cut-points that map a top-1 probability to a state
are CLASS CONSTANTS, not configurable parameters. They live in
``harness/annotator.py`` (and in ``ConfidenceEmitter`` on the LIS
side) and must agree byte-for-byte. See SPEC-TFH-001 § 4 and
ARCH-LIS-001 § 4.3.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# EpistemicState — the three-state vocabulary
# ---------------------------------------------------------------------------

class EpistemicState(enum.Enum):
    """Per-example or per-token model output certainty.

    Values are lowercase strings so the enum serialises directly to the
    TernaryTokenEvent JSON schema, the per-example training annotation
    JSON, and the .see3 confidence-event-log entries — all without any
    mapping layer. Distinct from ``terncore.confidence.RoutingConfidence``
    (routing infrastructure state); the two enums must never be mixed.
    """

    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    DISCONFIRMED = "disconfirmed"

    @classmethod
    def from_string(cls, value: str) -> "EpistemicState":
        """Look up an EpistemicState by its lowercase string value.

        Raises ``ValueError`` if the string is not one of the three
        valid values — useful when parsing dataset annotation JSON
        from disk and you want a clear error rather than silent
        coercion to ``None``.
        """
        try:
            return cls(value)
        except ValueError as e:
            valid = sorted(s.value for s in cls)
            raise ValueError(
                f"Unknown EpistemicState string {value!r}. Valid: {valid}"
            ) from e


# ---------------------------------------------------------------------------
# Domain — coarse classifier for the per-example annotation schema
# ---------------------------------------------------------------------------

class Domain(enum.Enum):
    """Coarse domain label for a training example.

    Used by ``ConfidenceObjective³`` to weight the confidence
    calibration loss differently across domains (per harness.yaml
    → ``data.domain_weights``). String values match the SPEC-TFH-001
    § 4.1 schema.
    """

    FACTUAL = "factual"
    REASONING = "reasoning"
    CREATIVE = "creative"
    AGENTIC = "agentic"

    @classmethod
    def from_string(cls, value: str) -> "Domain":
        try:
            return cls(value)
        except ValueError as e:
            valid = sorted(d.value for d in cls)
            raise ValueError(
                f"Unknown Domain string {value!r}. Valid: {valid}"
            ) from e


# ---------------------------------------------------------------------------
# Per-example annotation schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpistemicLabel:
    """Per-example epistemic annotation attached to a training datum.

    Mirrors the JSON schema in SPEC-TFH-001 § 4.1:

        {
          "epistemic_state":   "confirmed|uncertain|disconfirmed",
          "confidence_score":  0.0-1.0,
          "escalate":          true|false,
          "domain":            "factual|reasoning|creative|agentic",
          "source_reliability": 0.0-1.0
        }

    Frozen so a label cannot be mutated mid-training-step. Use
    ``from_dict``/``to_dict`` to round-trip via JSON without losing
    type safety.
    """

    epistemic_state: EpistemicState
    confidence_score: float
    escalate: bool
    domain: Domain
    source_reliability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be in [0.0, 1.0], got {self.confidence_score}"
            )
        if not 0.0 <= self.source_reliability <= 1.0:
            raise ValueError(
                f"source_reliability must be in [0.0, 1.0], got {self.source_reliability}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "EpistemicLabel":
        """Decode an EpistemicLabel from the on-disk JSON shape.

        Strings are coerced to enums via ``from_string`` so unknown
        values raise a clear error instead of silently producing
        ``None`` enum members.
        """
        required = {
            "epistemic_state", "confidence_score", "escalate",
            "domain", "source_reliability",
        }
        missing = required - set(d.keys())
        if missing:
            raise ValueError(
                f"EpistemicLabel JSON missing required keys: {sorted(missing)}"
            )
        return cls(
            epistemic_state=EpistemicState.from_string(d["epistemic_state"]),
            confidence_score=float(d["confidence_score"]),
            escalate=bool(d["escalate"]),
            domain=Domain.from_string(d["domain"]),
            source_reliability=float(d["source_reliability"]),
        )

    def to_dict(self) -> dict:
        """Encode as the SPEC-TFH-001 § 4.1 JSON shape."""
        return {
            "epistemic_state": self.epistemic_state.value,
            "confidence_score": self.confidence_score,
            "escalate": self.escalate,
            "domain": self.domain.value,
            "source_reliability": self.source_reliability,
        }
