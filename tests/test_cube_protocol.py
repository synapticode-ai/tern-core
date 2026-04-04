"""
Tests for tern-core v0.4.0 — CubeAction Address Protocol.
Updated for 7 domains (63 cells) after HR domain added.
"""

import pytest
from terncore.confidence import RoutingConfidence
from terncore.routing import TernaryRouter
from terncore.cube import (
    CubeAction, CubeyClient, Guardian, GuardianVerdict,
    CUBE_ADDRESS_SPACE, validate_address,
)


class TestValidateAddress:
    def test_valid_finance_bullion(self):
        d, f = validate_address("finance.bullion")
        assert d == "finance" and f == "bullion"

    def test_valid_hr_probation(self):
        d, f = validate_address("hr.probation")
        assert d == "hr" and f == "probation"

    def test_valid_all_63_addresses(self):
        count = sum(1 for d, fns in CUBE_ADDRESS_SPACE.items() for f in fns
                    if validate_address(f"{d}.{f}"))
        assert count == 63

    def test_invalid_domain(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            validate_address("invalid.x")

    def test_invalid_function(self):
        with pytest.raises(ValueError, match="Unknown function"):
            validate_address("finance.nonexistent")

    def test_missing_dot(self):
        with pytest.raises(ValueError, match="expected domain.function"):
            validate_address("financeswift")


class TestCubeAction:
    def test_parses_address(self):
        a = CubeAction(address="finance.bullion", action="buy", params={},
                        confidence=RoutingConfidence.SURE, weight=0.92)
        assert a.domain == "finance" and a.function == "bullion"

    def test_weight_clamped(self):
        a = CubeAction(address="sales.invoices", action="x", params={},
                        confidence=RoutingConfidence.SURE, weight=5.0)
        assert a.weight == 1.0

    def test_name_property(self):
        a = CubeAction(address="finance.banking", action="transfer", params={},
                        confidence=RoutingConfidence.SURE, weight=0.9)
        assert a.name == "finance.banking.transfer"

    def test_confidence_properties(self):
        s = CubeAction(address="sales.crm", action="r", params={},
                        confidence=RoutingConfidence.SURE, weight=0.9)
        assert s.is_sure and not s.is_unsure
        u = CubeAction(address="sales.crm", action="r", params={},
                        confidence=RoutingConfidence.UNSURE, weight=0.5)
        assert u.is_unsure
        k = CubeAction(address="sales.crm", action="r", params={},
                        confidence=RoutingConfidence.UNKNOWN, weight=-0.3)
        assert k.is_unknown


class TestGuardian:
    def test_sure_executes(self):
        v = Guardian().evaluate(CubeAction(address="sales.invoices", action="create",
            params={}, confidence=RoutingConfidence.SURE, weight=0.92))
        assert v.can_execute

    def test_unsure_gates(self):
        v = Guardian().evaluate(CubeAction(address="operations.inventory", action="delete",
            params={}, confidence=RoutingConfidence.UNSURE, weight=0.51))
        assert v.is_gated

    def test_unknown_rolls_back(self):
        v = Guardian().evaluate(CubeAction(address="compliance.access_rights", action="elevate",
            params={}, confidence=RoutingConfidence.UNKNOWN, weight=-0.3))
        assert v.is_rolled_back

    def test_three_unknowns_locks(self):
        g = Guardian(unknown_threshold=3)
        for i in range(3):
            g.evaluate(CubeAction(address="compliance.data_privacy", action=f"p{i}",
                params={}, confidence=RoutingConfidence.UNKNOWN, weight=-0.3))
        assert g.is_protected("compliance")

    def test_protected_blocks_sure(self):
        g = Guardian()
        g._protected_domains.add("finance")
        v = g.evaluate(CubeAction(address="finance.accounting", action="read",
            params={}, confidence=RoutingConfidence.SURE, weight=0.99))
        assert v.is_rolled_back

    def test_clear_domain(self):
        g = Guardian()
        g._protected_domains.add("finance")
        g.clear_domain("finance")
        assert not g.is_protected("finance")

    def test_pending_gates(self):
        g = Guardian()
        g.evaluate(CubeAction(address="sales.quotes", action="send",
            params={}, confidence=RoutingConfidence.UNSURE, weight=0.5))
        assert len(g.pending_gates) == 1

    def test_anomalies(self):
        g = Guardian(unknown_threshold=2)
        for i in range(2):
            g.evaluate(CubeAction(address="cx.escalations", action=f"p{i}",
                params={}, confidence=RoutingConfidence.UNKNOWN, weight=-0.5))
        assert len(g.recent_anomalies) >= 1


class TestCubeyClient:
    def test_sure(self):
        assert CubeyClient("t", "u").execute("finance.bullion", "buy", {},
            RoutingConfidence.SURE, 0.92).can_execute

    def test_unsure(self):
        assert CubeyClient("t", "u").execute("operations.inventory", "delete", {},
            RoutingConfidence.UNSURE, 0.51).is_gated

    def test_unknown(self):
        assert CubeyClient("t", "u").execute("compliance.fraud_detection", "override", {},
            RoutingConfidence.UNKNOWN, -0.3).is_rolled_back

    def test_route_and_execute(self):
        r = TernaryRouter(deferral_band=(0.3, 0.6))
        r.register("tool", lambda p: 0.9)
        v = CubeyClient("t", "u", router=r).route_and_execute(
            "sales.invoices", "create", {}, "Create invoice")
        assert v.can_execute

    def test_no_router_raises(self):
        with pytest.raises(ValueError):
            CubeyClient("t", "u").route_and_execute("sales.crm", "r", {}, "t")

    def test_all_63_addresses(self):
        assert len(CubeyClient("t", "u").valid_addresses()) == 63

    def test_filtered_addresses(self):
        f = CubeyClient("t", "u").valid_addresses("finance")
        assert len(f) == 9 and all(a.startswith("finance.") for a in f)

    def test_hr_addresses(self):
        h = CubeyClient("t", "u").valid_addresses("hr")
        assert len(h) == 9 and all(a.startswith("hr.") for a in h)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            CubeyClient("t", "u").execute("invalid.bad", "x", {}, RoutingConfidence.SURE, 0.9)


class TestIntegration:
    def test_finance_sure(self):
        assert CubeyClient("t", "u").execute("finance.banking", "transfer",
            {"amount": 50000}, RoutingConfidence.SURE, 0.92).can_execute

    def test_finance_unknown(self):
        assert not CubeyClient("t", "u").execute("finance.banking", "transfer",
            {"amount": 800000}, RoutingConfidence.UNKNOWN, -0.3).can_execute

    def test_three_unknown_locks(self):
        g = Guardian(unknown_threshold=3)
        c = CubeyClient("t", "u", guardian=g)
        for i in range(3):
            c.execute("finance.bullion", f"p{i}", {}, RoutingConfidence.UNKNOWN, -0.3)
        assert g.is_protected("finance")
        assert c.execute("finance.accounting", "read", {},
            RoutingConfidence.SURE, 0.99).is_rolled_back

    def test_address_space_63_cells_7_domains(self):
        assert sum(len(fns) for fns in CUBE_ADDRESS_SPACE.values()) == 63
        assert len(CUBE_ADDRESS_SPACE) == 7

    def test_hr_probation_address(self):
        v = CubeyClient("t", "u").execute("hr.probation", "promote",
            {"employee": "james"}, RoutingConfidence.SURE, 0.90)
        assert v.can_execute
