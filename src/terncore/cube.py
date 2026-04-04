"""
terncore.cube — CubeAction Address Protocol.

Agents don't browse — they address. finance.swift, finance.bullion, sales.auction.
54 cells. One confidence model. One client.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from terncore.confidence import RoutingConfidence, stack_confidence
from terncore.routing import TernaryRouter, RouteDecision


# ── Address space ────────────────────────────────────────────────────────────

CUBE_ADDRESS_SPACE: dict[str, list[str]] = {
    "design": [
        "season_planning", "sampling", "merchandising",
        "production_sketches", "design", "design_library",
        "competitor_products", "forecasting_models", "merchandiser_review",
    ],
    "sales": [
        "crm", "lead_generation", "automations",
        "quotes", "sales", "access",
        "returns", "attrition", "invoices",
    ],
    "operations": [
        "inventory", "procurement", "production_tracking",
        "warehousing", "operations", "logistics",
        "supplier_compliance", "replenishment", "returns_flow",
    ],
    "compliance": [
        "data_privacy", "account_audit", "access_rights",
        "system_integrity", "compliance", "risk_alerts",
        "fraud_detection", "risk_log", "fraud_monitoring",
    ],
    "finance": [
        "accounting", "payroll", "sales_revenue",
        "purchases", "finance", "banking",
        "forecasts", "reports", "bullion",
    ],
    "cx": [
        "profiles", "support", "feedback",
        "loyalty", "cx", "recommendations",
        "returns_refunds", "nps", "escalations",
    ],
    "hr": [
        "roster", "onboarding", "probation",
        "performance", "hr", "payroll_hr",
        "offboarding", "compliance_hr", "training",
    ],
}

ALL_DOMAINS = list(CUBE_ADDRESS_SPACE.keys())


def validate_address(address: str) -> tuple[str, str]:
    """Validate and parse a cube address. Returns (domain, function)."""
    parts = address.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid address '{address}' — expected domain.function")
    domain, function = parts
    if domain not in CUBE_ADDRESS_SPACE:
        raise ValueError(f"Unknown domain '{domain}' — valid: {ALL_DOMAINS}")
    if function not in CUBE_ADDRESS_SPACE[domain]:
        raise ValueError(
            f"Unknown function '{function}' in domain '{domain}'"
        )
    return domain, function


# ── CubeAction ───────────────────────────────────────────────────────────────


@dataclass
class CubeAction:
    """Every operation in Agent³ eOS is a CubeAction."""

    address: str
    action: str
    params: dict[str, Any]
    confidence: RoutingConfidence
    weight: float
    tenant_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    domain: str = field(init=False)
    function: str = field(init=False)

    def __post_init__(self):
        self.domain, self.function = validate_address(self.address)
        self.weight = max(-1.0, min(1.0, self.weight))

    @property
    def name(self) -> str:
        return f"{self.address}.{self.action}"

    @property
    def is_sure(self) -> bool:
        return self.confidence == RoutingConfidence.SURE

    @property
    def is_unsure(self) -> bool:
        return self.confidence == RoutingConfidence.UNSURE

    @property
    def is_unknown(self) -> bool:
        return self.confidence == RoutingConfidence.UNKNOWN


# ── GuardianVerdict ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GuardianVerdict:
    action_id: str
    verdict: str  # "execute" | "gate" | "rollback" | "anomaly"
    reason: str
    confidence: RoutingConfidence
    weight: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def can_execute(self) -> bool:
        return self.verdict == "execute"

    @property
    def is_gated(self) -> bool:
        return self.verdict == "gate"

    @property
    def is_rolled_back(self) -> bool:
        return self.verdict == "rollback"


# ── Guardian ─────────────────────────────────────────────────────────────────


class Guardian:
    """Python Guardian³. Evaluates every CubeAction before execution."""

    def __init__(
        self,
        unknown_threshold: int = 3,
        correlation_window: float = 60.0,
    ):
        self._unknown_threshold = unknown_threshold
        self._correlation_window = correlation_window
        self._recent_actions: list[CubeAction] = []
        self._protected_domains: set[str] = set()
        self._event_log: list[GuardianVerdict] = []

    def evaluate(self, action: CubeAction) -> GuardianVerdict:
        """The single entry point — no action bypasses Guardian."""
        if action.domain in self._protected_domains:
            verdict = GuardianVerdict(
                action_id=action.id,
                verdict="rollback",
                reason=f"Domain '{action.domain}' is protected",
                confidence=RoutingConfidence.UNKNOWN,
                weight=-1.0,
            )
            self._event_log.append(verdict)
            return verdict

        self._recent_actions.append(action)
        self._sweep_expired()

        if action.confidence == RoutingConfidence.SURE:
            verdict = GuardianVerdict(
                action_id=action.id,
                verdict="execute",
                reason="SURE — auto-execute",
                confidence=RoutingConfidence.SURE,
                weight=action.weight,
            )

        elif action.confidence == RoutingConfidence.UNSURE:
            verdict = GuardianVerdict(
                action_id=action.id,
                verdict="gate",
                reason=f"UNSURE (weight={action.weight:.2f}) — human review",
                confidence=RoutingConfidence.UNSURE,
                weight=action.weight,
            )

        else:
            unknown_count = self._unknown_count(action.domain)
            if unknown_count >= self._unknown_threshold:
                self._protected_domains.add(action.domain)
                verdict = GuardianVerdict(
                    action_id=action.id,
                    verdict="anomaly",
                    reason=f"{unknown_count} UNKNOWNs in {action.domain} — domain locked",
                    confidence=RoutingConfidence.UNKNOWN,
                    weight=-1.0,
                )
            else:
                verdict = GuardianVerdict(
                    action_id=action.id,
                    verdict="rollback",
                    reason=f"UNKNOWN (weight={action.weight:.2f}) — rollback",
                    confidence=RoutingConfidence.UNKNOWN,
                    weight=action.weight,
                )

        self._event_log.append(verdict)
        return verdict

    def clear_domain(self, domain: str) -> None:
        self._protected_domains.discard(domain)

    def is_protected(self, domain: str) -> bool:
        return domain in self._protected_domains

    @property
    def pending_gates(self) -> list[GuardianVerdict]:
        return [v for v in self._event_log if v.is_gated]

    @property
    def recent_anomalies(self) -> list[GuardianVerdict]:
        return [v for v in self._event_log if v.verdict == "anomaly"]

    @property
    def event_log(self) -> list[GuardianVerdict]:
        return list(self._event_log)

    def _unknown_count(self, domain: str) -> int:
        now = datetime.now()
        return sum(
            1
            for a in self._recent_actions
            if a.domain == domain
            and a.confidence == RoutingConfidence.UNKNOWN
            and (now - a.timestamp).total_seconds() <= self._correlation_window
        )

    def _sweep_expired(self) -> None:
        now = datetime.now()
        self._recent_actions = [
            a
            for a in self._recent_actions
            if (now - a.timestamp).total_seconds() <= self._correlation_window
        ]


# ── CubeyClient ──────────────────────────────────────────────────────────────


class CubeyClient:
    """The Agent³ eOS Python client. Agents use this to address the cube."""

    def __init__(
        self,
        tenant_id: str,
        user_id: str,
        guardian: Optional[Guardian] = None,
        router: Optional[TernaryRouter] = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.guardian = guardian or Guardian()
        self.router = router

    def execute(
        self,
        address: str,
        action: str,
        params: dict[str, Any],
        confidence: RoutingConfidence,
        weight: float,
    ) -> GuardianVerdict:
        """Address a cube function and execute with confidence."""
        cube_action = CubeAction(
            address=address,
            action=action,
            params=params,
            confidence=confidence,
            weight=weight,
            tenant_id=self.tenant_id,
            user_id=self.user_id,
        )
        return self.guardian.evaluate(cube_action)

    def route_and_execute(
        self,
        address: str,
        action: str,
        params: dict[str, Any],
        prompt: str,
    ) -> GuardianVerdict:
        """Route prompt through TernaryRouter to determine confidence, then execute."""
        if not self.router:
            raise ValueError("Router required for route_and_execute")
        decision = self.router.route(prompt)
        return self.execute(
            address=address,
            action=action,
            params=params,
            confidence=decision.confidence,
            weight=decision.weight,
        )

    def address_space(self) -> dict[str, list[str]]:
        return dict(CUBE_ADDRESS_SPACE)

    def valid_addresses(self, domain: Optional[str] = None) -> list[str]:
        if domain:
            cells = CUBE_ADDRESS_SPACE.get(domain, [])
            return [f"{domain}.{cell}" for cell in cells]
        return [
            f"{d}.{f}" for d, fns in CUBE_ADDRESS_SPACE.items() for f in fns
        ]
