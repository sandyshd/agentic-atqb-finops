from __future__ import annotations

from datetime import datetime, timezone

from agentic_finops.atqb.models import ATQBDecision, ActuatorAction, ModelTierStats, QuotaLedger
from agentic_finops.uci.models import UCIRecord

# ---------------------------------------------------------------------------
# ATQB PID-inspired threshold ladder (from the paper / flow diagram)
# ---------------------------------------------------------------------------
BUDGET_QUOTA_SHIFT_PCT  = 0.70   # 70%  → start dynamic quota shifting
BUDGET_THROTTLE_PCT     = 0.80   # 80%  → throttle
BUDGET_DOWNGRADE_PCT    = 0.90   # 90%  → model downgrade / quantize
BUDGET_HARD_STOP_PCT    = 1.00   # 100% → Governor kill-switch

TOKEN_QUOTA_SHIFT_PCT   = 0.70
TOKEN_THROTTLE_PCT      = 0.85

SUCCESS_PROBABILITY_FLOOR = 0.80
UTILITY_GAIN_RATIO        = 1.03

# Marginal Utility of Accuracy (MUA) — model downgrade map
MODEL_DOWNGRADE_MAP: dict[str, str] = {
    "gpt-5":    "gpt-4",
    "claude-4": "llama-4",
    "gpt-4":    "llama-4",
    "default":  "llama-4",
}


class ATQBController:
    """
    Autonomous Token-Quota Balancing control loop.

    Evaluates the current quota/budget state and returns an ActuatorAction
    that preserves the Hard-Limit Budget (HLB) while maintaining SLO/SLA.

    Decision ladder (first matching rule wins):
      1. HLB exhausted OR tokens gone  → hard_stop   (Governor kill-switch)
      2. High-TPW task under 90% budget → allow       (borrow quota)
      3. Budget ≥ 90% OR UCI too high  → downgrade    (MUA optimization)
      4. Budget ≥ 80% OR tokens ≥ 85% → throttle
      5. Budget ≥ 70% OR tokens ≥ 70% → quota_shift  (liquid quota reallocation)
      6. Otherwise                     → allow
    """

    def _posterior_success(self, stats: ModelTierStats | None) -> float:
        # Beta posterior mean with mild prior to stabilize sparse tiers.
        alpha, beta = 2.0, 2.0
        if not stats:
            return alpha / (alpha + beta)
        return (stats.successes + alpha) / (stats.attempts + alpha + beta)

    def _avg_cost(self, stats: ModelTierStats | None, fallback_cost: float) -> float:
        if stats and stats.attempts > 0:
            return max(0.000001, stats.avg_cost_usd)
        return max(0.000001, fallback_cost)

    def _mua_evidence(
        self,
        *,
        current_model: str,
        fallback_cost: float,
        model_stats: dict[str, ModelTierStats],
    ) -> dict:
        current_key = current_model.lower()
        candidate_model = MODEL_DOWNGRADE_MAP.get(current_key, MODEL_DOWNGRADE_MAP["default"])
        candidate_key = candidate_model.lower()

        current_stats = model_stats.get(current_key)
        candidate_stats = model_stats.get(candidate_key)

        p_current = self._posterior_success(current_stats)
        p_candidate = self._posterior_success(candidate_stats)

        cost_current = self._avg_cost(current_stats, fallback_cost)
        # If candidate has no data, assume a cheaper but slightly lower-quality tier.
        if candidate_stats and candidate_stats.attempts > 0:
            cost_candidate = self._avg_cost(candidate_stats, fallback_cost * 0.7)
        else:
            cost_candidate = max(0.000001, cost_current * 0.7)
            p_candidate = max(0.65, p_current - 0.08)

        utility_current = p_current / cost_current
        utility_candidate = p_candidate / cost_candidate

        if cost_current > cost_candidate:
            mua_score = (p_current - p_candidate) / max(0.000001, (cost_current - cost_candidate))
        else:
            mua_score = -1.0

        should_downgrade = (
            p_candidate >= SUCCESS_PROBABILITY_FLOOR
            and utility_candidate >= utility_current * UTILITY_GAIN_RATIO
        )

        return {
            "candidate_model": candidate_model,
            "p_current": p_current,
            "p_candidate": p_candidate,
            "cost_current": cost_current,
            "cost_candidate": cost_candidate,
            "mua_score": mua_score,
            "should_downgrade": should_downgrade,
        }

    def decide(
        self,
        ledger: QuotaLedger,
        latest_uci: UCIRecord | None,
        tpw: float,
        current_model: str,
        *,
        rate_limit_pressure: float = 0.0,
        model_stats: dict[str, ModelTierStats] | None = None,
    ) -> ATQBDecision:
        bpct = ledger.budget_utilization
        tpct = ledger.token_utilization
        uci_val = latest_uci.uci if latest_uci else 0.0
        model_stats = model_stats or {}
        ts = datetime.now(timezone.utc).isoformat()

        # ── Rule 1: HLB / token exhaustion → Kill Switch ──────────────────
        if bpct >= BUDGET_HARD_STOP_PCT or tpct >= 1.0:
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.hard_stop,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason="Hard-Limit Budget (HLB) reached — Governor kill-switch engaged",
                rate_limit_pressure=rate_limit_pressure,
                timestamp=ts,
            )

        # ── Rule 2: Explicit 429/rate-limit pressure handling ────────────
        if rate_limit_pressure >= 0.60 and tpw >= 0.80 and bpct < BUDGET_HARD_STOP_PCT:
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.quota_shift,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason=(
                    f"Rate-limit pressure={rate_limit_pressure:.2f} on high-TPW workload — "
                    "requesting quota borrow from low-TPW pools"
                ),
                rate_limit_pressure=rate_limit_pressure,
                timestamp=ts,
            )

        if rate_limit_pressure >= 0.70 and tpw < 0.80:
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.throttle,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason=(
                    f"Rate-limit pressure={rate_limit_pressure:.2f} on non-critical workload — "
                    "throttling to protect high-TPW traffic"
                ),
                rate_limit_pressure=rate_limit_pressure,
                timestamp=ts,
            )

        # ── Rule 3: High-TPW critical task with headroom → allow + borrow ─
        if tpw >= 0.8 and bpct < BUDGET_DOWNGRADE_PCT:
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.allow,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason=f"High-TPW={tpw:.2f} task — quota borrowed from low-TPW pools",
                rate_limit_pressure=rate_limit_pressure,
                timestamp=ts,
            )

        # ── Rule 4: Measured MUA downgrade when pressure is rising ───────
        evidence = self._mua_evidence(
            current_model=current_model,
            fallback_cost=max(0.000001, uci_val),
            model_stats=model_stats,
        )
        if (
            evidence["should_downgrade"]
            and (
                bpct >= BUDGET_THROTTLE_PCT
                or tpct >= TOKEN_THROTTLE_PCT
                or rate_limit_pressure >= 0.40
            )
        ):
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.downgrade,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason=(
                    "Measured MUA indicates better success-per-dollar on "
                    f"{evidence['candidate_model']} under current pressure"
                ),
                recommended_model=evidence["candidate_model"],
                rate_limit_pressure=rate_limit_pressure,
                mua_score=round(evidence["mua_score"], 6),
                p_success_current=round(evidence["p_current"], 4),
                p_success_candidate=round(evidence["p_candidate"], 4),
                expected_cost_current=round(evidence["cost_current"], 6),
                expected_cost_candidate=round(evidence["cost_candidate"], 6),
                timestamp=ts,
            )

        # ── Rule 5: Budget ≥ 80% or tokens ≥ 85% → throttle ──────────────
        if bpct >= BUDGET_THROTTLE_PCT or tpct >= TOKEN_THROTTLE_PCT:
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.throttle,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason=f"Budget {bpct:.0%} / tokens {tpct:.0%} — throttling to preserve SLO",
                rate_limit_pressure=rate_limit_pressure,
                timestamp=ts,
            )

        # ── Rule 6: Budget ≥ 70% or tokens ≥ 70% → dynamic quota shift ───
        if bpct >= BUDGET_QUOTA_SHIFT_PCT or tpct >= TOKEN_QUOTA_SHIFT_PCT:
            return ATQBDecision(
                workload_id=ledger.workload_id,
                action=ActuatorAction.quota_shift,
                tpw=tpw, current_uci=uci_val,
                budget_utilization=bpct, token_utilization=tpct,
                reason="Token quota pressure building — reallocating from low-TPW workload pools",
                rate_limit_pressure=rate_limit_pressure,
                timestamp=ts,
            )

        # ── Rule 7: All clear ──────────────────────────────────────────────
        return ATQBDecision(
            workload_id=ledger.workload_id,
            action=ActuatorAction.allow,
            tpw=tpw, current_uci=uci_val,
            budget_utilization=bpct, token_utilization=tpct,
            reason="Within SLO and budget — proceeding normally",
            rate_limit_pressure=rate_limit_pressure,
            timestamp=ts,
        )
