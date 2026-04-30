from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ActionType(str, Enum):
    rightsize_idle = "rightsize_idle"
    offhours_schedule = "offhours_schedule"
    orphan_cleanup = "orphan_cleanup"


class CostSignal(BaseModel):
    timestamp: str
    cloud: str
    account_id: str
    service: str
    resource_id: str
    resource_name: str
    owner: str | None = None
    environment: str = "nonprod"
    estimated_hourly_cost_usd: float
    utilization_pct: float = Field(ge=0, le=100)
    anomaly_score: float = Field(ge=0, le=1)
    tags: dict[str, str] = Field(default_factory=dict)


class Recommendation(BaseModel):
    signal: CostSignal
    action_type: ActionType
    reason: str
    expected_monthly_savings_usd: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    risk: RiskLevel


class PolicyDecision(BaseModel):
    allowed: bool
    requires_approval: bool
    explanation: str


class ActionRecord(BaseModel):
    recommendation: Recommendation
    decision: PolicyDecision
    state: str
    trace_id: str
