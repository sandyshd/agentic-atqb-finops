from __future__ import annotations

from agentic_finops.config import settings
from agentic_finops.models import PolicyDecision, Recommendation, RiskLevel


def evaluate(recommendation: Recommendation) -> PolicyDecision:
    if recommendation.risk == RiskLevel.high:
        return PolicyDecision(
            allowed=False,
            requires_approval=True,
            explanation="High-risk actions are blocked in this phase",
        )

    if recommendation.expected_monthly_savings_usd > settings.max_auto_monthly_savings_usd:
        return PolicyDecision(
            allowed=False,
            requires_approval=True,
            explanation="Action exceeds auto-apply savings blast radius threshold",
        )

    if recommendation.confidence < settings.low_risk_confidence_threshold:
        return PolicyDecision(
            allowed=False,
            requires_approval=True,
            explanation="Confidence below auto-apply threshold; manual approval required",
        )

    if recommendation.risk == RiskLevel.medium:
        return PolicyDecision(
            allowed=True,
            requires_approval=True,
            explanation="Medium risk actions require human approval",
        )

    return PolicyDecision(
        allowed=True,
        requires_approval=False,
        explanation="Low risk action approved for guarded auto-remediation",
    )
