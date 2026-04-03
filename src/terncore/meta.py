"""
terncore.meta — MetaAgent: last resort before surfacing uncertainty.

Receives UNKNOWN escalations from TernaryRouter.
Attempts decompose → reframe → surface.
Never guesses.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from terncore.confidence import RoutingConfidence
from terncore.routing import RouteDecision, TernaryRouter


class ResolutionStrategy:
    DECOMPOSE = "decompose"
    REFRAME = "reframe"
    SURFACE = "surface"


@dataclass
class UncertaintyReport:
    """Surfaced uncertainty — the system is honest about what it doesn't know."""

    prompt: str
    strategies_tried: list[str]
    route_weight: float
    reason: str

    @property
    def summary(self) -> str:
        return (
            f"Prompt escalated after {len(self.strategies_tried)} "
            f"resolution attempt(s). Weight: {self.route_weight:.2f}."
        )


@dataclass
class ResolutionResult:
    """Outcome of MetaAgent.handle()."""

    outcome: str  # "resolved" or "surfaced"
    strategy: Optional[str] = None
    parts: Optional[list[str]] = None
    prompt: Optional[str] = None
    report: Optional[UncertaintyReport] = None


class MetaAgent:
    """
    Last resort before surfacing uncertainty.
    Receives UNKNOWN escalations from TernaryRouter.
    Attempts decompose → reframe → surface.
    Never guesses.
    """

    def __init__(
        self,
        router: TernaryRouter,
        assess: Callable[[str], RouteDecision],
    ):
        self._router = router
        self._assess = assess
        self.pending_reports: list[UncertaintyReport] = []
        self.resolved_count: int = 0

    def handle(self, prompt: str, decision: RouteDecision) -> ResolutionResult:
        """
        Attempt to resolve an UNKNOWN escalation.
        1. Decompose — split into parts, assess each
        2. Reframe — rephrase and re-route
        3. Surface — honest uncertainty report
        """
        # 1. Decompose
        parts = self._decompose(prompt)
        if parts and all(
            self._assess(p).confidence == RoutingConfidence.SURE for p in parts
        ):
            self.resolved_count += 1
            return ResolutionResult(
                outcome="resolved",
                strategy=ResolutionStrategy.DECOMPOSE,
                parts=parts,
            )

        # 2. Reframe
        reframed = f"Alternative framing: {prompt}"
        reframe_decision = self._router.route(reframed)
        if reframe_decision.confidence == RoutingConfidence.SURE:
            self.resolved_count += 1
            return ResolutionResult(
                outcome="resolved",
                strategy=ResolutionStrategy.REFRAME,
                prompt=reframed,
            )

        # 3. Surface
        report = UncertaintyReport(
            prompt=prompt,
            strategies_tried=[ResolutionStrategy.DECOMPOSE, ResolutionStrategy.REFRAME],
            route_weight=decision.weight,
            reason="decompose and reframe both failed",
        )
        self.pending_reports.append(report)
        return ResolutionResult(outcome="surfaced", report=report)

    def _decompose(self, prompt: str) -> Optional[list[str]]:
        """Split prompt into sentence-level parts."""
        parts = [s.strip() for s in prompt.split(".") if s.strip()]
        return parts if len(parts) > 1 else None
