"""
terncore.analytics — Guardian³ organisational analytics.

Aggregates the event log across time. Domains locked per week.
Mean time to heal. Cells generating most UNKNOWN actions.
Makes the system self-aware at the organisational level.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from terncore.cube import Guardian, GuardianVerdict


@dataclass(frozen=True)
class AnalyticsWindow:
    """Time window for analytics queries."""
    start: datetime
    end: datetime

    @staticmethod
    def last_hours(n: int) -> AnalyticsWindow:
        end = datetime.now()
        return AnalyticsWindow(start=end - timedelta(hours=n), end=end)

    @staticmethod
    def last_days(n: int) -> AnalyticsWindow:
        end = datetime.now()
        return AnalyticsWindow(start=end - timedelta(days=n), end=end)

    @staticmethod
    def last_week() -> AnalyticsWindow:
        return AnalyticsWindow.last_days(7)

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600


@dataclass
class DomainStats:
    domain: str
    total_actions: int
    execute_count: int
    gate_count: int
    rollback_count: int
    anomaly_count: int
    is_protected: bool

    @property
    def unknown_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return (self.rollback_count + self.anomaly_count) / self.total_actions

    @property
    def gate_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.gate_count / self.total_actions


@dataclass
class GuardianAnalytics:
    """Organisational-level analytics from Guardian event log."""

    total_actions: int
    execute_count: int
    gate_count: int
    rollback_count: int
    anomaly_count: int
    domains_locked: int
    domain_stats: dict[str, DomainStats]
    hotspot_cells: list[tuple[str, int]]  # (address, unknown_count) sorted desc
    window: AnalyticsWindow

    @property
    def execute_rate(self) -> float:
        return self.execute_count / self.total_actions if self.total_actions else 0.0

    @property
    def gate_rate(self) -> float:
        return self.gate_count / self.total_actions if self.total_actions else 0.0

    @property
    def unknown_rate(self) -> float:
        return (self.rollback_count + self.anomaly_count) / self.total_actions if self.total_actions else 0.0

    def summary(self) -> str:
        return (
            f"Guardian³ — {self.window.duration_hours:.0f}h window\n"
            f"  Actions: {self.total_actions} total\n"
            f"  ◆ Execute: {self.execute_count} ({self.execute_rate:.0%})\n"
            f"  ◇ Gate:    {self.gate_count} ({self.gate_rate:.0%})\n"
            f"  ○ Unknown: {self.rollback_count + self.anomaly_count} ({self.unknown_rate:.0%})\n"
            f"  Domains locked: {self.domains_locked}\n"
            f"  Hotspots: {', '.join(f'{addr}({c})' for addr, c in self.hotspot_cells[:3])}"
        )


def analyze(guardian: Guardian, window: Optional[AnalyticsWindow] = None) -> GuardianAnalytics:
    """Analyze Guardian event log within a time window."""
    if window is None:
        window = AnalyticsWindow.last_week()

    events = [
        v for v in guardian.event_log
        if window.start <= v.timestamp <= window.end
    ]

    execute_count = sum(1 for v in events if v.verdict == "execute")
    gate_count = sum(1 for v in events if v.verdict == "gate")
    rollback_count = sum(1 for v in events if v.verdict == "rollback")
    anomaly_count = sum(1 for v in events if v.verdict == "anomaly")

    # Domain stats
    domain_actions: dict[str, list[GuardianVerdict]] = {}
    for v in events:
        # Extract domain from action_id context — we use recent_actions
        pass  # We'll build from recent_actions below

    # Build domain stats from recent actions
    domain_counter: dict[str, dict[str, int]] = {}
    for a in guardian._recent_actions:
        if not (window.start <= a.timestamp <= window.end):
            continue
        d = a.domain
        if d not in domain_counter:
            domain_counter[d] = {"total": 0, "execute": 0, "gate": 0, "rollback": 0, "anomaly": 0}
        domain_counter[d]["total"] += 1

    # Map verdicts to domains via event log order alignment
    for v in events:
        # Find matching action by action_id
        matching = [a for a in guardian._recent_actions if a.id == v.action_id]
        if matching:
            d = matching[0].domain
            if d not in domain_counter:
                domain_counter[d] = {"total": 0, "execute": 0, "gate": 0, "rollback": 0, "anomaly": 0}
            domain_counter[d][v.verdict] = domain_counter[d].get(v.verdict, 0) + 1

    domain_stats = {}
    for d, counts in domain_counter.items():
        domain_stats[d] = DomainStats(
            domain=d,
            total_actions=counts["total"],
            execute_count=counts.get("execute", 0),
            gate_count=counts.get("gate", 0),
            rollback_count=counts.get("rollback", 0),
            anomaly_count=counts.get("anomaly", 0),
            is_protected=guardian.is_protected(d),
        )

    # Hotspot cells — addresses generating most UNKNOWN actions
    unknown_actions = [
        a for a in guardian._recent_actions
        if a.confidence.value == -1 and window.start <= a.timestamp <= window.end
    ]
    cell_counter = Counter(a.address for a in unknown_actions)
    hotspots = cell_counter.most_common(10)

    return GuardianAnalytics(
        total_actions=len(events),
        execute_count=execute_count,
        gate_count=gate_count,
        rollback_count=rollback_count,
        anomaly_count=anomaly_count,
        domains_locked=len(guardian._protected_domains),
        domain_stats=domain_stats,
        hotspot_cells=hotspots,
        window=window,
    )
