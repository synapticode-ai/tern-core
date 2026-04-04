"""Tests for Guardian³ analytics."""

from datetime import datetime, timedelta
from terncore.confidence import RoutingConfidence
from terncore.cube import CubeAction, Guardian
from terncore.analytics import analyze, AnalyticsWindow


def _action(address, confidence, weight=0.9):
    return CubeAction(address=address, action="test", params={},
                       confidence=confidence, weight=weight)


class TestAnalytics:

    def test_empty_guardian(self):
        result = analyze(Guardian())
        assert result.total_actions == 0
        assert result.execute_rate == 0.0

    def test_counts_correct(self):
        g = Guardian()
        g.evaluate(_action("sales.invoices", RoutingConfidence.SURE))
        g.evaluate(_action("sales.crm", RoutingConfidence.SURE))
        g.evaluate(_action("operations.inventory", RoutingConfidence.UNSURE, 0.5))
        g.evaluate(_action("compliance.data_privacy", RoutingConfidence.UNKNOWN, -0.3))

        result = analyze(g, AnalyticsWindow.last_hours(1))
        assert result.total_actions == 4
        assert result.execute_count == 2
        assert result.gate_count == 1
        assert result.rollback_count == 1

    def test_execute_rate(self):
        g = Guardian()
        for _ in range(8):
            g.evaluate(_action("sales.invoices", RoutingConfidence.SURE))
        for _ in range(2):
            g.evaluate(_action("finance.bullion", RoutingConfidence.UNSURE, 0.5))

        result = analyze(g, AnalyticsWindow.last_hours(1))
        assert abs(result.execute_rate - 0.80) < 0.01

    def test_domain_stats(self):
        g = Guardian()
        g.evaluate(_action("sales.invoices", RoutingConfidence.SURE))
        g.evaluate(_action("sales.crm", RoutingConfidence.SURE))
        g.evaluate(_action("finance.bullion", RoutingConfidence.UNSURE, 0.5))

        result = analyze(g, AnalyticsWindow.last_hours(1))
        assert "sales" in result.domain_stats
        assert result.domain_stats["sales"].total_actions >= 2

    def test_hotspot_cells(self):
        g = Guardian(unknown_threshold=100, correlation_window=3600)
        for _ in range(5):
            g.evaluate(_action("compliance.data_privacy", RoutingConfidence.UNKNOWN, -0.3))
        for _ in range(2):
            g.evaluate(_action("finance.bullion", RoutingConfidence.UNKNOWN, -0.3))

        result = analyze(g, AnalyticsWindow.last_hours(1))
        assert len(result.hotspot_cells) >= 1
        assert result.hotspot_cells[0][0] == "compliance.data_privacy"
        assert result.hotspot_cells[0][1] == 5

    def test_domains_locked(self):
        g = Guardian(unknown_threshold=3)
        for i in range(3):
            g.evaluate(_action("compliance.data_privacy", RoutingConfidence.UNKNOWN, -0.3))
        assert g.is_protected("compliance")

        result = analyze(g, AnalyticsWindow.last_hours(1))
        assert result.domains_locked == 1

    def test_summary_string(self):
        g = Guardian()
        g.evaluate(_action("sales.invoices", RoutingConfidence.SURE))
        result = analyze(g, AnalyticsWindow.last_hours(1))
        s = result.summary()
        assert "Guardian³" in s
        assert "Execute" in s

    def test_window_filtering(self):
        g = Guardian()
        g.evaluate(_action("sales.invoices", RoutingConfidence.SURE))

        # Window in the past — should find nothing
        old = AnalyticsWindow(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now() - timedelta(days=29)
        )
        result = analyze(g, old)
        assert result.total_actions == 0

    def test_unknown_rate(self):
        g = Guardian()
        g.evaluate(_action("sales.invoices", RoutingConfidence.SURE))
        g.evaluate(_action("compliance.access_rights", RoutingConfidence.UNKNOWN, -0.3))

        result = analyze(g, AnalyticsWindow.last_hours(1))
        assert abs(result.unknown_rate - 0.50) < 0.01
