from __future__ import annotations

from agentic_finops.models import ActionType, CostSignal, Recommendation, RiskLevel


def recommend_action(signal: CostSignal) -> Recommendation | None:
    if signal.anomaly_score < 0.65:
        return None

    if signal.utilization_pct <= 10:
        action = ActionType.rightsize_idle
        reason = "Low utilization and sustained anomaly indicate overprovisioned capacity"
        risk = RiskLevel.low
        confidence = min(0.95, signal.anomaly_score + 0.15)
        savings = round(signal.estimated_hourly_cost_usd * 24 * 30 * 0.35, 2)
    elif signal.environment == "nonprod":
        action = ActionType.offhours_schedule
        reason = "Non-production workload can be stopped during off-hours"
        risk = RiskLevel.medium
        confidence = min(0.9, signal.anomaly_score + 0.1)
        savings = round(signal.estimated_hourly_cost_usd * 16 * 22, 2)
    else:
        action = ActionType.orphan_cleanup
        reason = "Resource lacks clear ownership or optimization baseline"
        risk = RiskLevel.medium
        confidence = min(0.85, signal.anomaly_score)
        savings = round(signal.estimated_hourly_cost_usd * 24 * 30 * 0.2, 2)

    return Recommendation(
        signal=signal,
        action_type=action,
        reason=reason,
        expected_monthly_savings_usd=savings,
        confidence=confidence,
        risk=risk,
    )
