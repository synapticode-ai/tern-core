"""
Tests for tern-core v0.2.0 — Ternary Confidence Layer.

TernaryRouter, ConfidenceQueue, MetaAgent, ConfidenceStacking.
Same architecture as the Swift implementation in gamma-platform.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from datetime import datetime, timedelta

import pytest

from terncore.confidence import RoutingConfidence, stack_confidence
from terncore.routing import TernaryRouter, RouteDecision
from terncore.queue import ConfidenceQueue, ReleaseReason
from terncore.meta import MetaAgent, ResolutionStrategy


# ── 1. TernaryRouter — SURE/UNSURE/UNKNOWN thresholds ──────────────


class TestTernaryRouter:
    def test_sure_threshold(self):
        """Score above deferral_hi → SURE."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool_a", lambda p: 0.85)

        decision = router.route("test prompt")
        assert decision.confidence == RoutingConfidence.SURE
        assert decision.tool_name == "tool_a"
        assert decision.dispatchable is True
        assert decision.should_defer is False

    def test_unsure_threshold(self):
        """Score in deferral band → UNSURE."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool_a", lambda p: 0.45)

        decision = router.route("test prompt")
        assert decision.confidence == RoutingConfidence.UNSURE
        assert decision.should_defer is True
        assert decision.dispatchable is False

    def test_unknown_threshold(self):
        """Score at or below escalation threshold → UNKNOWN."""
        router = TernaryRouter(escalation_threshold=0.0)
        router.register("tool_a", lambda p: -0.5)

        decision = router.route("test prompt")
        assert decision.confidence == RoutingConfidence.UNKNOWN
        assert decision.should_escalate is True

    def test_no_tools_registered(self):
        """No tools → UNKNOWN with weight -1.0."""
        router = TernaryRouter()
        decision = router.route("test")
        assert decision.confidence == RoutingConfidence.UNKNOWN
        assert decision.weight == -1.0

    def test_best_tool_selected(self):
        """Highest-scoring tool wins."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("weak", lambda p: 0.2)
        router.register("strong", lambda p: 0.9)

        decision = router.route("test")
        assert decision.tool_name == "strong"
        assert decision.confidence == RoutingConfidence.SURE

    def test_weight_clamped(self):
        """Weight clamped to [-1.0, 1.0]."""
        router = TernaryRouter()
        router.register("extreme", lambda p: 5.0)

        decision = router.route("test")
        assert decision.weight == 1.0


# ── 2. ConfidenceQueue — evidence accumulation and auto-release ─────


class TestConfidenceQueue:
    def test_enqueue_and_pending_count(self):
        queue = ConfidenceQueue(evidence_threshold=3)
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.4)

        id = queue.enqueue("prompt", decision)
        assert queue.pending_count == 1
        assert queue.get(id) is not None

    def test_add_evidence_below_threshold(self):
        """Evidence below threshold → stays queued."""
        queue = ConfidenceQueue(evidence_threshold=3)
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.4)
        id = queue.enqueue("prompt", decision)

        result = queue.add_evidence(id, ["chunk1", "chunk2"])
        assert result is None  # not yet released
        assert queue.pending_count == 1

    def test_add_evidence_auto_release(self):
        """Evidence at threshold → auto-release."""
        queue = ConfidenceQueue(evidence_threshold=3)
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.4)
        id = queue.enqueue("prompt", decision)

        queue.add_evidence(id, ["c1", "c2"])
        result = queue.add_evidence(id, ["c3"])

        assert result is not None
        assert result.reason == ReleaseReason.EVIDENCE_THRESHOLD_MET
        assert result.final_weight > decision.weight  # evidence boost
        assert queue.pending_count == 0

    def test_evidence_weight_boost(self):
        """Each evidence chunk adds 0.08, capped at +0.30."""
        queue = ConfidenceQueue(evidence_threshold=5)
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.4)
        id = queue.enqueue("prompt", decision)

        result = queue.add_evidence(id, ["c1", "c2", "c3", "c4", "c5"])
        assert result is not None
        # 5 chunks × 0.08 = 0.40, capped at 0.30, so final = 0.4 + 0.30 = 0.70
        assert abs(result.final_weight - 0.70) < 0.001

    def test_force_release(self):
        """Force-release removes item from queue."""
        queue = ConfidenceQueue()
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.5)
        id = queue.enqueue("prompt", decision)

        result = queue.force_release(id)
        assert result is not None
        assert result.reason == ReleaseReason.FORCED_MAX_DEFERRALS
        assert queue.pending_count == 0

    def test_sweep_expired(self):
        """Items past TTL are swept."""
        queue = ConfidenceQueue(ttl_seconds=60)
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.4)
        id = queue.enqueue("prompt", decision)

        # Manually backdate the enqueue time
        queue._items[id].enqueued_at = datetime.now() - timedelta(seconds=120)

        released = queue.sweep_expired()
        assert len(released) == 1
        assert released[0].reason == ReleaseReason.TTL_EXPIRED
        assert queue.pending_count == 0

    def test_on_release_callback(self):
        """Release callback fires on auto-release."""
        released_items = []
        queue = ConfidenceQueue(
            evidence_threshold=1, on_release=lambda r: released_items.append(r)
        )
        decision = RouteDecision("tool", RoutingConfidence.UNSURE, 0.4)
        id = queue.enqueue("prompt", decision)

        queue.add_evidence(id, ["chunk"])
        assert len(released_items) == 1

    def test_add_evidence_missing_id(self):
        """Evidence for non-existent ID returns None."""
        queue = ConfidenceQueue()
        result = queue.add_evidence("nonexistent", ["chunk"])
        assert result is None


# ── 3. MetaAgent — decompose, reframe, surface ──────────────────────


class TestMetaAgent:
    def _make_router_and_assess(self, reframe_score=0.1):
        """Helper: router that returns score based on prompt content."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register(
            "tool",
            lambda p: 0.9 if "Alternative framing" in p else reframe_score,
        )

        def assess(prompt: str) -> RouteDecision:
            return RouteDecision("tool", RoutingConfidence.SURE, 0.8)

        return router, assess

    def test_decompose_success(self):
        """Multi-sentence prompt decomposes into SURE parts → resolved."""
        router, assess = self._make_router_and_assess()
        agent = MetaAgent(router, assess)

        decision = RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.3)
        result = agent.handle("First part. Second part.", decision)

        assert result.outcome == "resolved"
        assert result.strategy == ResolutionStrategy.DECOMPOSE
        assert len(result.parts) == 2
        assert agent.resolved_count == 1

    def test_reframe_success(self):
        """Single-sentence prompt fails decompose, reframe succeeds."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register(
            "tool",
            lambda p: 0.9 if "Alternative framing" in p else -0.5,
        )
        # assess always returns UNKNOWN so decompose fails
        assess = lambda p: RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.5)

        agent = MetaAgent(router, assess)
        decision = RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.3)
        result = agent.handle("Single sentence prompt", decision)

        assert result.outcome == "resolved"
        assert result.strategy == ResolutionStrategy.REFRAME
        assert agent.resolved_count == 1

    def test_surface_when_all_fail(self):
        """Both decompose and reframe fail → surface uncertainty."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool", lambda p: -0.5)  # always UNKNOWN
        assess = lambda p: RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.5)

        agent = MetaAgent(router, assess)
        decision = RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.3)
        result = agent.handle("Opaque prompt", decision)

        assert result.outcome == "surfaced"
        assert result.report is not None
        assert len(result.report.strategies_tried) == 2
        assert ResolutionStrategy.DECOMPOSE in result.report.strategies_tried
        assert ResolutionStrategy.REFRAME in result.report.strategies_tried
        assert len(agent.pending_reports) == 1
        assert agent.resolved_count == 0

    def test_uncertainty_report_summary(self):
        """Report summary is human-readable."""
        router = TernaryRouter()
        router.register("tool", lambda p: -0.5)
        assess = lambda p: RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.5)

        agent = MetaAgent(router, assess)
        decision = RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.3)
        result = agent.handle("Test", decision)

        summary = result.report.summary
        assert "2 resolution attempt(s)" in summary
        assert "-0.30" in summary


# ── 4. ConfidenceStacking — all five cases ──────────────────────────


class TestConfidenceStacking:
    def test_sure_sure(self):
        assert stack_confidence(RoutingConfidence.SURE, RoutingConfidence.SURE) == RoutingConfidence.SURE

    def test_sure_unsure(self):
        assert stack_confidence(RoutingConfidence.SURE, RoutingConfidence.UNSURE) == RoutingConfidence.UNSURE

    def test_sure_unknown(self):
        assert stack_confidence(RoutingConfidence.SURE, RoutingConfidence.UNKNOWN) == RoutingConfidence.UNKNOWN

    def test_unsure_sure(self):
        """Evidence resolved the deferral."""
        assert stack_confidence(RoutingConfidence.UNSURE, RoutingConfidence.SURE) == RoutingConfidence.SURE

    def test_unsure_unsure(self):
        """Stacked uncertainty → escalate."""
        assert stack_confidence(RoutingConfidence.UNSURE, RoutingConfidence.UNSURE) == RoutingConfidence.UNKNOWN

    def test_unsure_unknown(self):
        assert stack_confidence(RoutingConfidence.UNSURE, RoutingConfidence.UNKNOWN) == RoutingConfidence.UNKNOWN

    def test_unknown_any(self):
        """UNKNOWN + anything → UNKNOWN."""
        for agent in RoutingConfidence:
            assert stack_confidence(RoutingConfidence.UNKNOWN, agent) == RoutingConfidence.UNKNOWN


# ── 5. Integration: router → queue → meta agent full chain ──────────


class TestIntegrationChain:
    def test_full_chain_sure(self):
        """SURE route → immediate dispatch, no queue or meta."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool", lambda p: 0.9)
        queue = ConfidenceQueue()

        decision = router.route("confident prompt")
        assert decision.dispatchable is True
        assert queue.pending_count == 0  # never entered queue

    def test_full_chain_unsure_to_sure(self):
        """UNSURE route → queue → evidence → auto-release as SURE."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool", lambda p: 0.45)
        queue = ConfidenceQueue(evidence_threshold=2)

        decision = router.route("uncertain prompt")
        assert decision.should_defer is True

        id = queue.enqueue("uncertain prompt", decision)
        assert queue.pending_count == 1

        queue.add_evidence(id, ["evidence_1"])
        assert queue.pending_count == 1  # still waiting

        released = queue.add_evidence(id, ["evidence_2"])
        assert released is not None
        assert released.final_weight > decision.weight
        assert queue.pending_count == 0

    def test_full_chain_unknown_to_surfaced(self):
        """UNKNOWN route → meta agent → surfaced uncertainty."""
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool", lambda p: -0.5)
        assess = lambda p: RouteDecision("tool", RoutingConfidence.UNKNOWN, -0.5)
        agent = MetaAgent(router, assess)

        decision = router.route("opaque prompt")
        assert decision.should_escalate is True

        result = agent.handle("opaque prompt", decision)
        assert result.outcome == "surfaced"
        assert len(agent.pending_reports) == 1
