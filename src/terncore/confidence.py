"""
terncore.confidence — Ternary confidence types and stacking rules.

The same vocabulary as the Swift implementation in gamma-platform.
RoutingConfidence is the same type in both. The confidence propagation
claims in the patents are demonstrable in Python, not just in Swift.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from enum import IntEnum


class RoutingConfidence(IntEnum):
    """
    Ternary confidence state of a routing decision.

    SURE    (+1) — dispatch immediately
    UNSURE  ( 0) — hold in ConfidenceQueue until context accumulates
    UNKNOWN (-1) — escalate to MetaAgent
    """

    UNKNOWN = -1
    UNSURE = 0
    SURE = 1


def stack_confidence(
    route: RoutingConfidence,
    agent: RoutingConfidence,
) -> RoutingConfidence:
    """
    Combine routing confidence with agent confidence.

    The stacking rule:
        route=SURE   + agent=SURE   → SURE     (clean path)
        route=SURE   + agent=UNSURE → UNSURE   (agent uncertain)
        route=SURE   + agent=UNKNOWN→ UNKNOWN  (agent has no basis)
        route=UNSURE + agent=SURE   → SURE     (evidence resolved deferral)
        route=UNSURE + agent=UNSURE → UNKNOWN  (stacked uncertainty → escalate)
        route=UNSURE + agent=UNKNOWN→ UNKNOWN
        route=UNKNOWN + any         → UNKNOWN  (escalation already warranted)
    """
    if route == RoutingConfidence.UNKNOWN:
        return RoutingConfidence.UNKNOWN

    if route == RoutingConfidence.SURE and agent == RoutingConfidence.SURE:
        return RoutingConfidence.SURE
    if route == RoutingConfidence.UNSURE and agent == RoutingConfidence.SURE:
        return RoutingConfidence.SURE
    if route == RoutingConfidence.SURE and agent == RoutingConfidence.UNSURE:
        return RoutingConfidence.UNSURE

    # SURE + UNKNOWN, UNSURE + UNSURE, UNSURE + UNKNOWN
    return RoutingConfidence.UNKNOWN
