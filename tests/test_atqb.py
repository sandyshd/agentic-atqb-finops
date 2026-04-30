from datetime import datetime, timezone

from agentic_finops.atqb.controller import ATQBController
from agentic_finops.atqb.models import ActuatorAction, ModelTierStats, QuotaLedger
from agentic_finops.uci.models import UCIRecord


def _ledger(cost_pct: float, token_pct: float, hlb: float = 200.0) -> QuotaLedger:
    token_budget = 100_000
    period = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    l = QuotaLedger(workload_id="wl-test", period=period, token_budget=token_budget, hlb_usd=hlb)
    l.tokens_used = int(token_pct * token_budget)
    l.cost_used_usd = round(cost_pct * hlb, 4)
    return l


def _uci(val: float) -> UCIRecord:
    return UCIRecord(
        period="2026-01-01T00:00Z",
        workload_id="wl-test",
        c_tokens=val * 0.8,
        c_compute=val * 0.1,
        c_latency_penalty=val * 0.1,
        n_successful_tasks=10,
        uci=val,
    )


ctrl = ATQBController()


def test_hard_stop_at_100pct_budget() -> None:
    decision = ctrl.decide(_ledger(1.0, 0.5), _uci(0.01), tpw=0.5, current_model="gpt-4")
    assert decision.action == ActuatorAction.hard_stop


def test_hard_stop_at_100pct_tokens() -> None:
    decision = ctrl.decide(_ledger(0.5, 1.0), _uci(0.01), tpw=0.5, current_model="gpt-4")
    assert decision.action == ActuatorAction.hard_stop


def test_high_tpw_allowed_below_downgrade_threshold() -> None:
    # tpw=0.9, budget=0.75 (below 0.90) → should allow via quota borrow
    decision = ctrl.decide(_ledger(0.75, 0.5), _uci(0.01), tpw=0.9, current_model="gpt-4")
    assert decision.action == ActuatorAction.allow


def test_downgrade_at_90pct_budget() -> None:
    model_stats = {
        "gpt-5": ModelTierStats(
            workload_id="wl-test",
            model_name="gpt-5",
            attempts=140,
            successes=128,
            total_cost_usd=4.9,
            avg_latency_ms=780,
        ),
        "gpt-4": ModelTierStats(
            workload_id="wl-test",
            model_name="gpt-4",
            attempts=180,
            successes=153,
            total_cost_usd=3.2,
            avg_latency_ms=820,
        ),
    }
    decision = ctrl.decide(
        _ledger(0.92, 0.5),
        _uci(0.01),
        tpw=0.5,
        current_model="gpt-5",
        model_stats=model_stats,
    )
    assert decision.action == ActuatorAction.downgrade
    assert decision.recommended_model == "gpt-4"


def test_downgrade_with_measured_mua_under_pressure() -> None:
    model_stats = {
        "claude-4": ModelTierStats(
            workload_id="wl-test",
            model_name="claude-4",
            attempts=200,
            successes=182,
            total_cost_usd=4.8,
            avg_latency_ms=700,
        ),
        "llama-4": ModelTierStats(
            workload_id="wl-test",
            model_name="llama-4",
            attempts=220,
            successes=180,
            total_cost_usd=1.7,
            avg_latency_ms=760,
        ),
    }
    decision = ctrl.decide(
        _ledger(0.85, 0.5),
        _uci(0.02),
        tpw=0.5,
        current_model="claude-4",
        model_stats=model_stats,
    )
    assert decision.action == ActuatorAction.downgrade
    assert decision.recommended_model == "llama-4"
    assert decision.mua_score is not None
    assert decision.p_success_candidate is not None


def test_rate_limit_quota_shift_for_high_tpw() -> None:
    decision = ctrl.decide(
        _ledger(0.50, 0.30),
        _uci(0.01),
        tpw=0.9,
        current_model="gpt-4",
        rate_limit_pressure=0.8,
    )
    assert decision.action == ActuatorAction.quota_shift


def test_rate_limit_throttle_for_low_tpw() -> None:
    decision = ctrl.decide(
        _ledger(0.45, 0.40),
        _uci(0.01),
        tpw=0.4,
        current_model="gpt-4",
        rate_limit_pressure=0.8,
    )
    assert decision.action == ActuatorAction.throttle


def test_throttle_at_80pct_budget() -> None:
    decision = ctrl.decide(_ledger(0.82, 0.5), _uci(0.01), tpw=0.5, current_model="gpt-4")
    assert decision.action == ActuatorAction.throttle


def test_quota_shift_at_70pct() -> None:
    decision = ctrl.decide(_ledger(0.72, 0.5), _uci(0.01), tpw=0.5, current_model="gpt-4")
    assert decision.action == ActuatorAction.quota_shift


def test_allow_below_all_thresholds() -> None:
    decision = ctrl.decide(_ledger(0.30, 0.30), _uci(0.008), tpw=0.5, current_model="llama-4")
    assert decision.action == ActuatorAction.allow
