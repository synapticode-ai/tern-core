"""
terncore.routing — TernaryRouter: confidence-weighted prompt dispatch.

Routes prompts to tools using ternary confidence scoring.
Each registered tool carries a scorer: prompt → float ∈ [-1, 1].

Three dispatch paths:
    SURE    (+1) → dispatch immediately
    UNSURE  ( 0) → hold in ConfidenceQueue
    UNKNOWN (-1) → escalate to MetaAgent

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from terncore.confidence import RoutingConfidence


@dataclass(frozen=True)
class RouteDecision:
    """A routing decision with ternary confidence."""

    tool_name: str
    confidence: RoutingConfidence
    weight: float  # ∈ [-1.0, 1.0]
    reason: Optional[str] = None

    @property
    def dispatchable(self) -> bool:
        """Ready for immediate dispatch."""
        return self.confidence == RoutingConfidence.SURE and self.weight > 0.0

    @property
    def should_defer(self) -> bool:
        """Should enter ConfidenceQueue."""
        return self.confidence == RoutingConfidence.UNSURE

    @property
    def should_escalate(self) -> bool:
        """Should escalate to MetaAgent."""
        return self.weight < 0.0


class TernaryRouter:
    """
    Routes prompts to tools using ternary confidence scoring.

    Each registered tool carries a scorer: prompt → float ∈ [-1, 1].
    Negative weight → UNKNOWN / escalation.
    Deferral band → UNSURE / ConfidenceQueue.
    Above band → SURE / immediate dispatch.
    """

    def __init__(
        self,
        escalation_threshold: float = 0.0,
        deferral_band: tuple[float, float] = (0.3, 0.6),
    ):
        self._tools: dict[str, Callable[[str], float]] = {}
        self._escalation_threshold = escalation_threshold
        self._deferral_lo, self._deferral_hi = deferral_band

    def register(self, tool_name: str, scorer: Callable[[str], float]) -> None:
        """Register a tool with its confidence scorer."""
        self._tools[tool_name] = scorer

    def route(self, prompt: str) -> RouteDecision:
        """Score all tools and return the best RouteDecision."""
        if not self._tools:
            return RouteDecision(
                "none", RoutingConfidence.UNKNOWN, -1.0, "no tools registered"
            )

        scores = {name: scorer(prompt) for name, scorer in self._tools.items()}
        best = max(scores, key=scores.__getitem__)
        w = max(-1.0, min(1.0, scores[best]))

        if w <= self._escalation_threshold:
            confidence = RoutingConfidence.UNKNOWN
        elif w < self._deferral_hi:
            confidence = RoutingConfidence.UNSURE
        else:
            confidence = RoutingConfidence.SURE

        return RouteDecision(best, confidence, w, f"best_score={w:.3f}")

    @property
    def tool_count(self) -> int:
        return len(self._tools)
