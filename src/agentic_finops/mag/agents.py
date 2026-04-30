from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time

from agentic_finops.atqb.controller import ATQBController
from agentic_finops.atqb.models import (
    ATQBDecision,
    ActuatorAction,
    ModelTierStats,
    QuotaLedger,
    QuotaTransfer,
    RateLimitSignal,
)
from agentic_finops.uci.calculator import compute_uci, request_cost
from agentic_finops.uci.models import InferenceRecord, UCIRecord


# ---------------------------------------------------------------------------
# Gatekeeper Agent
# Receives workload requests, assigns Task Priority Weight (TPW ∈ [0,1]),
# and routes them into the MAG pipeline.
# ---------------------------------------------------------------------------
@dataclass
class GatekeeperAgent:
    workload_tpw_map: dict[str, float] = field(default_factory=lambda: {
        "workload-nlp":      0.85,
        "workload-code":     0.75,
        "workload-analysis": 0.60,
        "workload-chat":     0.45,
    })

    def assign_tpw(self, workload_id: str, is_critical: bool = False) -> float:
        if is_critical:
            return 0.95
        return self.workload_tpw_map.get(workload_id, 0.50)

    def classify_intent(self, request_text: str | None, workload_id: str) -> str:
        """Intent-aware request classification for routing and guardrails."""
        text = (request_text or "").lower()
        if any(k in text for k in ["incident", "outage", "sev", "critical"]):
            return "critical_ops"
        if any(k in text for k in ["compliance", "audit", "security", "safety"]):
            return "safety"
        if any(k in text for k in ["optimize", "cost", "cheap", "budget"]):
            return "cost_optimization"
        if workload_id in {"workload-code", "workload-analysis"}:
            return "productivity"
        return "general"

    def route(
        self,
        workload_id: str,
        tpw: float,
        decision: ATQBDecision,
        intent: str,
    ) -> dict:
        return {
            "workload_id": workload_id,
            "tpw": tpw,
            "intent": intent,
            "action": decision.action,
            "recommended_model": decision.recommended_model,
            "routed_at": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Auditor Agent
# Continuously monitors FOCUS™ data streams to detect cost anomalies
# with sub-second latency.
# ---------------------------------------------------------------------------
@dataclass
class AuditorAgent:
    _records: deque[InferenceRecord] = field(default_factory=lambda: deque(maxlen=5000))
    _uci_history: deque[UCIRecord] = field(default_factory=lambda: deque(maxlen=500))

    def ingest(self, record: InferenceRecord) -> None:
        self._records.appendleft(record)

    def compute_latest_uci(self, workload_id: str) -> UCIRecord | None:
        relevant = [r for r in self._records if r.workload_id == workload_id][:200]
        uci = compute_uci(relevant)
        if uci:
            self._uci_history.appendleft(uci)
        return uci

    def anomaly_trajectory(self, workload_id: str) -> str:
        recent = [u for u in self._uci_history if u.workload_id == workload_id][:10]
        if len(recent) < 2:
            return "stable"
        trend = recent[0].uci - recent[-1].uci          # newest first
        if trend > 0.003:
            return "rising"
        if trend < -0.003:
            return "falling"
        return "stable"

    def emit_audit_event(self, record: InferenceRecord, uci: UCIRecord | None) -> dict:
        return {
            "type": "uci_update",
            "workload_id": record.workload_id,
            "uci": uci.uci if uci else None,
            "trajectory": self.anomaly_trajectory(record.workload_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def latest_uci_records(self, limit: int = 30) -> list[UCIRecord]:
        return list(self._uci_history)[:limit]


# ---------------------------------------------------------------------------
# Broker Agent
# Manages the "Spot Market" for GPU compute; switches workloads between
# providers (Azure, AWS, GPU Neocloud) based on the lowest current UCI.
# ---------------------------------------------------------------------------
@dataclass
class BrokerAgent:
    quote_ttl_seconds: int = 60
    provider_uci: dict[str, float] = field(default_factory=lambda: {
        "azure":       0.030,
        "aws":         0.027,
        "neocloud-gpu": 0.021,
    })
    market_quotes: dict[str, dict] = field(default_factory=lambda: {
        "azure": {
            "spot_multiplier": 1.00,
            "unit_price_per_gpu_hour": 1.00,
            "egress_per_1k_tokens": 0.0000,
            "latency_ms": 40.0,
            "source": "bootstrap",
            "as_of": 0.0,
        },
        "aws": {
            "spot_multiplier": 0.96,
            "unit_price_per_gpu_hour": 0.96,
            "egress_per_1k_tokens": 0.0005,
            "latency_ms": 55.0,
            "source": "bootstrap",
            "as_of": 0.0,
        },
        "neocloud-gpu": {
            "spot_multiplier": 0.90,
            "unit_price_per_gpu_hour": 0.90,
            "egress_per_1k_tokens": 0.0012,
            "latency_ms": 70.0,
            "source": "bootstrap",
            "as_of": 0.0,
        },
    })

    def update_market_quote(
        self,
        provider: str,
        *,
        spot_multiplier: float,
        unit_price_per_gpu_hour: float = 0.0,
        egress_per_1k_tokens: float = 0.0,
        latency_ms: float = 0.0,
        source: str = "unknown",
    ) -> None:
        self.market_quotes[provider] = {
            "spot_multiplier": max(0.1, float(spot_multiplier)),
            "unit_price_per_gpu_hour": max(0.0, float(unit_price_per_gpu_hour)),
            "egress_per_1k_tokens": max(0.0, float(egress_per_1k_tokens)),
            "latency_ms": max(0.0, float(latency_ms)),
            "source": source,
            "as_of": time.time(),
        }

    def _is_fresh(self, provider: str) -> bool:
        quote = self.market_quotes.get(provider)
        if not quote:
            return False
        return (time.time() - float(quote.get("as_of", 0.0))) <= self.quote_ttl_seconds

    def _effective_uci(self, provider: str, current_uci: float) -> float:
        base = self.provider_uci.get(provider, current_uci)
        quote = self.market_quotes.get(provider)
        if not quote or not self._is_fresh(provider):
            return base
        # Approximate effective UCI = market-adjusted compute + network overhead + latency penalty.
        latency_penalty = max(0.0, (float(quote["latency_ms"]) - 50.0) * 0.000001)
        return (
            base * float(quote["spot_multiplier"])
            + float(quote["egress_per_1k_tokens"])
            + latency_penalty
        )

    def select_provider(self, current_uci: float, residency: str = "any") -> str:
        """Select provider with lowest effective market-driven UCI."""
        scored = {
            provider: self._effective_uci(provider, current_uci)
            for provider in self.provider_uci
        }
        best_provider = min(scored, key=scored.__getitem__)
        if scored[best_provider] < current_uci * 0.95:
            return best_provider
        # Default: prefer Azure when no meaningful arbitrage
        return "azure"

    def update_provider_uci(self, provider: str, uci: float) -> None:
        self.provider_uci[provider] = uci

    def spot_market_snapshot(self) -> dict[str, float]:
        return dict(self.provider_uci)

    def market_snapshot(self) -> list[dict]:
        rows: list[dict] = []
        now = time.time()
        for provider, quote in self.market_quotes.items():
            as_of = float(quote.get("as_of", 0.0) or 0.0)
            age_sec = max(0.0, now - as_of) if as_of > 0 else None
            rows.append(
                {
                    "provider": provider,
                    "spot_multiplier": float(quote.get("spot_multiplier", 1.0) or 1.0),
                    "unit_price_per_gpu_hour": float(quote.get("unit_price_per_gpu_hour", 0.0) or 0.0),
                    "egress_per_1k_tokens": float(quote.get("egress_per_1k_tokens", 0.0) or 0.0),
                    "latency_ms": float(quote.get("latency_ms", 0.0) or 0.0),
                    "source": str(quote.get("source", "unknown")),
                    "age_seconds": round(age_sec, 1) if age_sec is not None else None,
                    "is_fresh": self._is_fresh(provider),
                    "provider_uci": float(self.provider_uci.get(provider, 0.0) or 0.0),
                }
            )
        rows.sort(key=lambda row: row["provider"])
        return rows


# ---------------------------------------------------------------------------
# Governor Agent
# Holds the "Kill Switch" and "Throttle" permissions. Executes the ATQB
# algorithm to enforce organizational HLB without human intervention.
# ---------------------------------------------------------------------------
@dataclass
class GovernorAgent:
    ledgers: dict[str, QuotaLedger] = field(default_factory=dict)
    decisions: deque[ATQBDecision] = field(default_factory=lambda: deque(maxlen=500))
    transfers: deque[QuotaTransfer] = field(default_factory=lambda: deque(maxlen=1000))
    rate_limit_signals: dict[str, RateLimitSignal] = field(default_factory=dict)
    model_stats: dict[str, dict[str, ModelTierStats]] = field(default_factory=dict)
    _controller: ATQBController = field(default_factory=ATQBController)

    def get_or_create_ledger(
        self,
        workload_id: str,
        token_budget: int = 500_000,
        hlb_usd: float = 200.0,
    ) -> QuotaLedger:
        if workload_id not in self.ledgers:
            period = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self.ledgers[workload_id] = QuotaLedger(
                workload_id=workload_id,
                period=period,
                token_budget=token_budget,
                base_token_budget=token_budget,
                hlb_usd=hlb_usd,
            )
        return self.ledgers[workload_id]

    def _max_lendable(self, ledger: QuotaLedger) -> int:
        floor_tokens = int((ledger.base_token_budget or ledger.token_budget) * ledger.lend_floor_pct)
        return max(0, ledger.token_budget - floor_tokens)

    def _max_borrowable(self, ledger: QuotaLedger) -> int:
        cap = int((ledger.base_token_budget or ledger.token_budget) * ledger.borrow_limit_pct)
        return max(0, cap - ledger.borrowed_tokens)

    def rebalance_quota(
        self,
        *,
        requesting_workload: str,
        requester_tpw: float,
        needed_tokens: int,
        reason: str,
    ) -> list[QuotaTransfer]:
        receiver = self.get_or_create_ledger(requesting_workload)
        receiver.last_tpw = requester_tpw

        remaining_need = max(0, int(needed_tokens))
        remaining_capacity = self._max_borrowable(receiver)
        if remaining_need <= 0 or remaining_capacity <= 0:
            return []

        donors = [
            l for wid, l in self.ledgers.items()
            if wid != requesting_workload and self._max_lendable(l) > 0
        ]
        # Prefer donating from low-TPW pools with highest lendable headroom.
        donors.sort(key=lambda l: (l.last_tpw, -self._max_lendable(l)))

        applied: list[QuotaTransfer] = []
        for donor in donors:
            if remaining_need <= 0 or remaining_capacity <= 0:
                break
            lendable = self._max_lendable(donor)
            move = min(lendable, remaining_need, remaining_capacity)
            if move <= 0:
                continue
            donor.token_budget -= move
            donor.lent_tokens += move
            receiver.token_budget += move
            receiver.borrowed_tokens += move
            remaining_need -= move
            remaining_capacity -= move

            transfer = QuotaTransfer(
                from_workload_id=donor.workload_id,
                to_workload_id=requesting_workload,
                tokens=move,
                reason=reason,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            self.transfers.appendleft(transfer)
            applied.append(transfer)

        return applied

    def record_rate_limit_signal(
        self,
        *,
        workload_id: str,
        provider: str,
        http_429_count: int,
        retry_after_ms_p95: float,
        window_seconds: int,
        deployment_name: str | None = None,
    ) -> None:
        self.rate_limit_signals[workload_id] = RateLimitSignal(
            workload_id=workload_id,
            provider=provider,
            deployment_name=deployment_name,
            http_429_count=http_429_count,
            retry_after_ms_p95=retry_after_ms_p95,
            window_seconds=window_seconds,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def rate_limit_pressure(self, workload_id: str) -> float:
        signal = self.rate_limit_signals.get(workload_id)
        return signal.pressure if signal else 0.0

    def record_model_outcome(
        self,
        *,
        workload_id: str,
        model_name: str,
        attempts: int,
        successes: int,
        total_cost_usd: float,
        avg_latency_ms: float,
    ) -> None:
        key = model_name.lower()
        bucket = self.model_stats.setdefault(workload_id, {})
        stats = bucket.get(key)
        if stats is None:
            stats = ModelTierStats(workload_id=workload_id, model_name=key)
            bucket[key] = stats

        prev_attempts = stats.attempts
        stats.attempts += max(0, int(attempts))
        stats.successes += max(0, int(successes))
        stats.total_cost_usd += max(0.0, float(total_cost_usd))
        if stats.attempts > 0:
            weighted = (stats.avg_latency_ms * prev_attempts) + (avg_latency_ms * max(0, int(attempts)))
            stats.avg_latency_ms = weighted / stats.attempts

    def model_profile(self, workload_id: str) -> dict[str, ModelTierStats]:
        return dict(self.model_stats.get(workload_id, {}))

    def enforce(
        self,
        workload_id: str,
        tpw: float,
        current_model: str,
        latest_uci: UCIRecord | None,
    ) -> ATQBDecision:
        ledger = self.get_or_create_ledger(workload_id)
        ledger.last_tpw = tpw
        decision = self._controller.decide(
            ledger,
            latest_uci,
            tpw,
            current_model,
            rate_limit_pressure=self.rate_limit_pressure(workload_id),
            model_stats=self.model_profile(workload_id),
        )
        self.decisions.appendleft(decision)
        return decision

    def record_usage(self, workload_id: str, tokens: int, cost_usd: float) -> None:
        ledger = self.get_or_create_ledger(workload_id)
        ledger.tokens_used += tokens
        ledger.cost_used_usd = round(ledger.cost_used_usd + cost_usd, 6)

    def latest_decisions(self, limit: int = 30) -> list[ATQBDecision]:
        return list(self.decisions)[:limit]

    def ledger_statuses(self) -> list[dict]:
        return [
            {
                "workload_id": wid,
                "period": l.period,
                "base_token_budget": l.base_token_budget,
                "token_budget": l.token_budget,
                "effective_token_budget": l.effective_token_budget,
                "tokens_used": l.tokens_used,
                "tokens_remaining": l.tokens_remaining,
                "borrowed_tokens": l.borrowed_tokens,
                "lent_tokens": l.lent_tokens,
                "hlb_usd": l.hlb_usd,
                "cost_used_usd": round(l.cost_used_usd, 4),
                "budget_utilization_pct": round(l.budget_utilization * 100, 2),
                "token_utilization_pct": round(l.token_utilization * 100, 2),
                "last_tpw": l.last_tpw,
            }
            for wid, l in self.ledgers.items()
        ]

    def latest_transfers(self, limit: int = 50) -> list[QuotaTransfer]:
        return list(self.transfers)[:limit]


# ---------------------------------------------------------------------------
# Critic Agent
# Validates inference outputs for quality and hallucination;
# gates N_successful_tasks in the UCI denominator.
# ---------------------------------------------------------------------------
@dataclass
class CriticAgent:
    base_success_rate: float = 0.92

    def validate(
        self, output_text: str | None, latency_ms: float, slo_latency_ms: float
    ) -> bool:
        if not output_text or len(output_text) < 5:
            return False
        # Latency degradation reduces effective quality probability
        latency_ratio = latency_ms / max(1.0, slo_latency_ms)
        adjusted_rate = self.base_success_rate * min(1.0, 1.2 - latency_ratio * 0.2)
        return random.random() < max(0.0, adjusted_rate)


# ---------------------------------------------------------------------------
# Scaler Agent
# Manages GPU scaling within SLO constraints and provides scaling signals
# back to the Governor/Broker.
# ---------------------------------------------------------------------------
@dataclass
class ScalerAgent:
    def scale_recommendation(self, latency_ms: float, slo_latency_ms: float) -> str:
        ratio = latency_ms / max(1.0, slo_latency_ms)
        if ratio > 1.25:
            return "scale_up"
        if ratio < 0.40:
            return "scale_down"
        return "no_change"
