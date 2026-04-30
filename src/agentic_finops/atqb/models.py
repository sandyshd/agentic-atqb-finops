from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActuatorAction(str, Enum):
    """Actuator outputs of the ATQB control loop."""

    allow         = "allow"
    throttle      = "throttle"           # Reduce request rate
    quota_shift   = "quota_shift"        # Borrow capacity from low-TPW pools
    downgrade     = "downgrade_model"    # Switch to cheaper model tier
    quantize      = "quantize"           # Apply quantization to reduce cost
    hard_stop     = "hard_stop"          # Kill-switch: HLB enforced


class QuotaLedger(BaseModel):
    """Per-workload, per-period token and budget tracker."""

    workload_id: str
    period: str
    token_budget: int
    base_token_budget: int | None = None
    borrowed_tokens: int = 0
    lent_tokens: int = 0
    borrow_limit_pct: float = Field(default=0.30, ge=0.0, le=1.0)
    lend_floor_pct: float = Field(default=0.60, ge=0.0, le=1.0)
    last_tpw: float = Field(default=0.50, ge=0.0, le=1.0)
    tokens_used: int = 0
    hlb_usd: float = Field(description="Hard-Limit Budget in USD")
    cost_used_usd: float = 0.0

    @property
    def token_utilization(self) -> float:
        return self.tokens_used / max(1, self.token_budget)

    @property
    def budget_utilization(self) -> float:
        return self.cost_used_usd / max(0.0001, self.hlb_usd)

    @property
    def tokens_remaining(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    @property
    def effective_token_budget(self) -> int:
        return self.token_budget

    def model_post_init(self, __context) -> None:
        if self.base_token_budget is None:
            self.base_token_budget = self.token_budget


class QuotaTransfer(BaseModel):
    """Transfer record for quota borrowing between workloads."""

    from_workload_id: str
    to_workload_id: str
    tokens: int = Field(gt=0)
    reason: str
    timestamp: str


class RateLimitSignal(BaseModel):
    """Aggregated provider rate-limit telemetry for one workload."""

    workload_id: str
    provider: str = "azure"
    deployment_name: str | None = None
    http_429_count: int = Field(default=0, ge=0)
    retry_after_ms_p95: float = Field(default=0.0, ge=0.0)
    window_seconds: int = Field(default=60, ge=1)
    timestamp: str

    @property
    def pressure(self) -> float:
        # 429 burst and high retry-after both increase pressure toward 1.0
        count_component = min(1.0, self.http_429_count / 10.0)
        retry_component = min(1.0, self.retry_after_ms_p95 / 5000.0)
        return round(min(1.0, count_component * 0.7 + retry_component * 0.3), 4)


class ModelTierStats(BaseModel):
    """Observed quality/cost stats for a workload-model tier."""

    workload_id: str
    model_name: str
    attempts: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    avg_latency_ms: float = Field(default=0.0, ge=0.0)

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / max(1, self.attempts)


class ATQBDecision(BaseModel):
    """Output of one ATQB control-loop evaluation cycle."""

    workload_id: str
    action: ActuatorAction
    tpw: float
    current_uci: float
    budget_utilization: float
    token_utilization: float
    reason: str
    recommended_model: str | None = None
    rate_limit_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    mua_score: float | None = None
    p_success_current: float | None = None
    p_success_candidate: float | None = None
    expected_cost_current: float | None = None
    expected_cost_candidate: float | None = None
    timestamp: str = ""
