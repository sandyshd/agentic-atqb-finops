from agentic_finops.models import (
    ActionType,
    CostSignal,
    Recommendation,
    RiskLevel,
)
from agentic_finops.policy.engine import evaluate


def _rec(confidence: float, risk: RiskLevel, savings: float) -> Recommendation:
    signal = CostSignal(
        timestamp="2026-01-01T00:00:00+00:00",
        cloud="azure",
        account_id="acct-1",
        service="compute",
        resource_id="vm-1",
        resource_name="vm-1",
        estimated_hourly_cost_usd=1.2,
        utilization_pct=5,
        anomaly_score=0.9,
    )
    return Recommendation(
        signal=signal,
        action_type=ActionType.rightsize_idle,
        reason="test",
        expected_monthly_savings_usd=savings,
        confidence=confidence,
        risk=risk,
    )


def test_low_risk_high_confidence_auto_allowed() -> None:
    decision = evaluate(_rec(confidence=0.95, risk=RiskLevel.low, savings=300))
    assert decision.allowed is True
    assert decision.requires_approval is False


def test_medium_risk_requires_approval() -> None:
    decision = evaluate(_rec(confidence=0.9, risk=RiskLevel.medium, savings=300))
    assert decision.allowed is True
    assert decision.requires_approval is True
