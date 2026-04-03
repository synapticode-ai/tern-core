"""
terncore.queue — ConfidenceQueue: context-accumulating holding layer.

Items don't wait for time — they wait for evidence.
When enough corroborating signal arrives, they release themselves.

Three release paths:
    evidence_threshold_met  — enough context chunks accumulated
    forced_after_max_deferrals — second deferral escape valve
    ttl_expired            — staleness guard

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from terncore.routing import RouteDecision


@dataclass
class QueuedRoute:
    """An UNSURE-routed item accumulating context."""

    id: str
    prompt: str
    decision: RouteDecision
    enqueued_at: datetime
    deferral_count: int = 0
    evidence: list[str] = field(default_factory=list)


class ReleaseReason:
    EVIDENCE_THRESHOLD_MET = "evidence_threshold_met"
    FORCED_MAX_DEFERRALS = "forced_after_max_deferrals"
    TTL_EXPIRED = "ttl_expired"


@dataclass
class ReleasedRoute:
    """A queued item that has been released for dispatch."""

    queued: QueuedRoute
    reason: str
    final_weight: float


class ConfidenceQueue:
    """
    Holds UNSURE routing decisions until evidence accumulates.

    Evidence threshold → auto-release with recalculated weight.
    Max deferrals → force dispatch.
    TTL → expire stale items.
    """

    def __init__(
        self,
        evidence_threshold: int = 3,
        max_deferrals: int = 2,
        ttl_seconds: int = 300,
        on_release: Optional[Callable[[ReleasedRoute], None]] = None,
    ):
        self._items: dict[str, QueuedRoute] = {}
        self.evidence_threshold = evidence_threshold
        self.max_deferrals = max_deferrals
        self.ttl = timedelta(seconds=ttl_seconds)
        self._on_release = on_release

    def enqueue(self, prompt: str, decision: RouteDecision) -> str:
        """Enqueue an UNSURE decision for context accumulation."""
        id = str(uuid.uuid4())
        self._items[id] = QueuedRoute(
            id=id,
            prompt=prompt,
            decision=decision,
            enqueued_at=datetime.now(),
        )
        return id

    def add_evidence(self, id: str, chunks: list[str]) -> Optional[ReleasedRoute]:
        """
        Feed evidence to a waiting item.
        Auto-releases when evidence threshold met.
        """
        if id not in self._items:
            return None
        item = self._items[id]
        item.evidence.extend(chunks)
        if len(item.evidence) >= self.evidence_threshold:
            return self._release(id, ReleaseReason.EVIDENCE_THRESHOLD_MET)
        return None

    def force_release(self, id: str) -> Optional[ReleasedRoute]:
        """Force-release a queued item (escape valve)."""
        if id not in self._items:
            return None
        return self._release(id, ReleaseReason.FORCED_MAX_DEFERRALS)

    def sweep_expired(self, now: Optional[datetime] = None) -> list[ReleasedRoute]:
        """Sweep items past TTL. Returns released items."""
        now = now or datetime.now()
        expired = [
            id
            for id, item in self._items.items()
            if now - item.enqueued_at > self.ttl
        ]
        return [self._release(id, ReleaseReason.TTL_EXPIRED) for id in expired]

    def get(self, id: str) -> Optional[QueuedRoute]:
        """Look up a queued item by ID."""
        return self._items.get(id)

    def _release(self, id: str, reason: str) -> ReleasedRoute:
        item = self._items.pop(id)
        evidence_boost = min(0.30, len(item.evidence) * 0.08)
        final_weight = min(1.0, item.decision.weight + evidence_boost)
        released = ReleasedRoute(item, reason, final_weight)
        if self._on_release:
            self._on_release(released)
        return released

    @property
    def pending_count(self) -> int:
        return len(self._items)
