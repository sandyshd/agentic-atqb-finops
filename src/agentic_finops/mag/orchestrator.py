from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from agentic_finops.atqb.models import ActuatorAction

from agentic_finops.mag.agents import (
    AuditorAgent,
    BrokerAgent,
    CriticAgent,
    GatekeeperAgent,
    GovernorAgent,
    ScalerAgent,
)
from agentic_finops.mag.guardrails import PolicyGuardrails
from agentic_finops.mag.optimizer import OptimizationExecutor
from agentic_finops.uci.calculator import request_cost
from agentic_finops.uci.models import InferenceRecord


@dataclass
class MAGOrchestrator:
    """
    Multi-Agent Governance (MAG) Orchestrator.

    Implements the sequence-diagram interaction order:
      Workload → Gatekeeper (TPW + Route)
               → Governor (ATQB enforcement)
               → [Model Gateway] inference
               → Critic (validate success/quality)
               → Auditor (UCI update + audit event)
               → Broker (lowest-UCI provider selection)
               → Scaler (GPU scale recommendation)
               → Gatekeeper (compound response routing)

    No human intervention required for HLB enforcement.
    """

    gatekeeper: GatekeeperAgent = field(default_factory=GatekeeperAgent)
    auditor:    AuditorAgent    = field(default_factory=AuditorAgent)
    governor:   GovernorAgent   = field(default_factory=GovernorAgent)
    broker:     BrokerAgent     = field(default_factory=BrokerAgent)
    critic:     CriticAgent     = field(default_factory=CriticAgent)
    scaler:     ScalerAgent     = field(default_factory=ScalerAgent)
    guardrails: PolicyGuardrails = field(default_factory=PolicyGuardrails)
    optimizer:  OptimizationExecutor = field(default_factory=OptimizationExecutor)

    _results: deque[dict] = field(default_factory=lambda: deque(maxlen=500))

    def _build_sequence_trace(
        self,
        request_id: str,
        workload_id: str,
        intent: str,
        tpw: float,
        model_name: str,
        latest_uci: float,
        success: bool,
        decision,
        provider: str,
        scale_action: str,
        guardrail,
        execution: dict,
        audit_event: dict,
        route: dict,
        timestamp: str,
        scaler_payload: dict | None = None,
    ) -> list[dict]:
        return [
            {
                "step": 1,
                "from": "Workload",
                "to": "Gatekeeper",
                "box": "Intent Aware Request Handling",
                "arrow": "request",
                "detail": "Incoming workload request received",
                "payload": {
                    "request_id": request_id,
                    "workload_id": workload_id,
                    "intent": intent,
                },
                "timestamp": timestamp,
            },
            {
                "step": 2,
                "from": "Gatekeeper",
                "to": "Governor",
                "box": "Task Priority Weight Assignment",
                "arrow": "assign_tpw_and_route",
                "detail": "Intent classified and TPW assigned",
                "payload": {
                    "intent": intent,
                    "tpw": round(tpw, 2),
                    "model_name": model_name,
                },
                "timestamp": timestamp,
            },
            {
                "step": 3,
                "from": "Governor",
                "to": "ModelGateway",
                "box": "ATQB Enforcement",
                "arrow": "policy_decision",
                "detail": "ATQB evaluated budget and quota posture",
                "payload": {
                    "action": decision.action,
                    "reason": decision.reason,
                    "budget_utilization": round(decision.budget_utilization, 4),
                    "token_utilization": round(decision.token_utilization, 4),
                },
                "timestamp": timestamp,
            },
            {
                "step": 4,
                "from": "ModelGateway",
                "to": "Critic",
                "box": "Inference Plane",
                "arrow": "candidate_output",
                "detail": "Simulated model inference completed",
                "payload": {
                    "model_name": model_name,
                    "provider": provider,
                },
                "timestamp": timestamp,
            },
            {
                "step": 5,
                "from": "Critic",
                "to": "Auditor",
                "box": "Validation and Telemetry",
                "arrow": "validated_result",
                "detail": "Output quality validated for UCI denominator",
                "payload": {
                    "success": success,
                },
                "timestamp": timestamp,
            },
            {
                "step": 6,
                "from": "Auditor",
                "to": "Governor",
                "box": "FOCUS Telemetry and UCI",
                "arrow": "uci_update",
                "detail": "Auditor emitted UCI and anomaly trajectory",
                "payload": {
                    "uci": round(latest_uci, 6),
                    "audit_event": audit_event,
                },
                "timestamp": timestamp,
            },
            {
                "step": 7,
                "from": "Governor",
                "to": "Broker",
                "box": "Governance and Compliance",
                "arrow": "optimization_request",
                "detail": "Governor requested provider optimization under HLB",
                "payload": {
                    "requested_action": decision.action,
                    "recommended_model": decision.recommended_model or model_name,
                },
                "timestamp": timestamp,
            },
            {
                "step": 8,
                "from": "Broker",
                "to": "Scaler",
                "box": "Spot Market and Placement",
                "arrow": "provider_selection",
                "detail": "Broker selected lowest-UCI provider candidate",
                "payload": {
                    "provider": provider,
                    "uci": round(latest_uci, 6),
                },
                "timestamp": timestamp,
            },
            {
                "step": 9,
                "from": "Scaler",
                "to": "PolicyGuardrails",
                "box": "Scaling and Validation",
                "arrow": "scaling_signal",
                "detail": "Scaler produced elasticity recommendation",
                "payload": {
                    "scale_action": scale_action,
                    **(scaler_payload or {}),
                },
                "timestamp": timestamp,
            },
            {
                "step": 10,
                "from": "PolicyGuardrails",
                "to": "OptimizationExecutor",
                "box": "Policy Guardrails",
                "arrow": "guardrail_enforcement",
                "detail": "Policy guardrails validated or overrode the action",
                "payload": {
                    "allowed": guardrail.allowed,
                    "enforced_action": guardrail.enforced_action,
                    "reason": guardrail.reason,
                },
                "timestamp": timestamp,
            },
            {
                "step": 11,
                "from": "OptimizationExecutor",
                "to": "Gatekeeper",
                "box": "Autonomous Optimization",
                "arrow": "execution_result",
                "detail": "Optimization action executed autonomously",
                "payload": execution,
                "timestamp": timestamp,
            },
            {
                "step": 12,
                "from": "Gatekeeper",
                "to": "Workload",
                "box": "Compound Routed Response",
                "arrow": "final_response",
                "detail": "Final routed response returned to workload",
                "payload": route,
                "timestamp": timestamp,
            },
        ]

    def process_inference(
        self,
        workload_id: str,
        model_name: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        request_count: int = 1,
        slo_latency_ms: float = 2000.0,
        provider: str = "azure",
        rate_limit_429_count: int = 0,
        retry_after_ms_p95: float = 0.0,
        rate_limit_window_seconds: int = 60,
        is_critical: bool = False,
        request_text: str | None = None,
        output_text: str | None = "inference-output",
        success_override: bool | None = None,
    ) -> dict:
        request_id = f"req-{uuid4()}"
        ts = datetime.now(timezone.utc).isoformat()

        # ── Step 1: Gatekeeper assigns TPW ────────────────────────────────
        intent = self.gatekeeper.classify_intent(request_text, workload_id)
        tpw = self.gatekeeper.assign_tpw(workload_id, is_critical)

        # ── Step 2: Critic validates output quality ────────────────────────
        success = success_override if success_override is not None else self.critic.validate(output_text, latency_ms, slo_latency_ms)

        # ── Step 3: Build normalised inference record ──────────────────────
        record = InferenceRecord(
            request_id=request_id,
            workload_id=workload_id,
            model_name=model_name,
            tpw=tpw,
            request_count=request_count,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            slo_latency_ms=slo_latency_ms,
            success=success,
            timestamp=ts,
        )

        # Ingest explicit provider rate-limit telemetry for ATQB decisions.
        if rate_limit_429_count > 0 or retry_after_ms_p95 > 0:
            self.governor.record_rate_limit_signal(
                workload_id=workload_id,
                provider=provider,
                http_429_count=rate_limit_429_count,
                retry_after_ms_p95=retry_after_ms_p95,
                window_seconds=rate_limit_window_seconds,
            )

        # ── Step 4: Auditor ingests + computes UCI ─────────────────────────
        self.auditor.ingest(record)
        latest_uci = self.auditor.compute_latest_uci(workload_id)
        trajectory  = self.auditor.anomaly_trajectory(workload_id)
        audit_event = self.auditor.emit_audit_event(record, latest_uci)

        # ── Step 4b: Scaler observes demand + forecasts next window ───────
        # Done before the Governor so the forecast + cost-delta can feed back
        # into the ATQB decision context.
        total_tokens_window = int(tokens_input + tokens_output)
        self.scaler.observe(workload_id, total_tokens_window)
        uci_per_token = (
            (latest_uci.c_tokens / max(1, latest_uci.n_successful_tasks))
            if latest_uci and latest_uci.n_successful_tasks > 0
            else 0.0
        )
        scaler_rec = self.scaler.recommend(
            workload_id=workload_id,
            latency_ms=latency_ms,
            slo_latency_ms=slo_latency_ms,
            uci_per_token=uci_per_token,
        )

        # ── Step 5: Governor enforces ATQB / HLB ──────────────────────────
        decision = self.governor.enforce(workload_id, tpw, model_name, latest_uci)
        # Feedback loop: when the Scaler forecasts a sustained over-capacity
        # condition that would *add* projected cost, append the signal to the
        # Governor's reason chain so downstream consumers (ATQB ledger,
        # dashboard, audit trail) can attribute the action to scaling
        # pressure as well as budget pressure.
        if scaler_rec.action == "scale_up" and scaler_rec.cost_delta_usd > 0:
            decision.reason = (
                f"{decision.reason} | scaler: forecast pressure "
                f"{scaler_rec.pressure_ratio:.2f}× capacity, +"
                f"${scaler_rec.cost_delta_usd:.4f} projected"
            )

        transfers_applied: list[dict] = []
        if decision.action == ActuatorAction.quota_shift:
            ledger = self.governor.get_or_create_ledger(workload_id)
            total_tokens = max(0, int(tokens_input + tokens_output))
            needed_tokens = max(0, total_tokens - ledger.tokens_remaining)
            if needed_tokens > 0:
                transfers = self.governor.rebalance_quota(
                    requesting_workload=workload_id,
                    requester_tpw=tpw,
                    needed_tokens=needed_tokens,
                    reason=decision.reason,
                )
                transfers_applied = [t.model_dump() for t in transfers]

        # ── Step 6: Record actual usage in quota ledger ────────────────────
        actual_cost = request_cost(record)
        self.governor.record_usage(
            workload_id,
            tokens=tokens_input + tokens_output,
            cost_usd=actual_cost,
        )
        self.governor.record_model_outcome(
            workload_id=workload_id,
            model_name=model_name,
            # One validation gate per record: the Critic produces a single
            # success/failure decision regardless of how many calls the
            # record aggregates. Using request_count here would over-count
            # evidence in the Beta-posterior used by the MUA downgrade rule.
            attempts=1,
            successes=1 if success else 0,
            total_cost_usd=actual_cost,
            avg_latency_ms=latency_ms,
        )

        # ── Step 7: Broker — spot market provider selection ────────────────
        uci_val = latest_uci.uci if latest_uci else 0.0
        provider = self.broker.select_provider(uci_val)
        self.broker.update_provider_uci("azure", max(0.000001, uci_val * 1.02))
        self.broker.update_provider_uci("aws", max(0.000001, uci_val * 0.98))

        # ── Step 8: Scaler — GPU scale recommendation ──────────────────────
        # Recommendation was computed at step 4b; here we expose the verb so
        # the existing trace contract is preserved.
        scale_action = scaler_rec.action

        # ── Step 9: Policy guardrail check before autonomous action ───────
        guardrail = self.guardrails.evaluate(
            requested_action=decision.action,
            intent=intent,
            tpw=tpw,
            provider=provider,
        )

        # ── Step 10: Autonomous optimization execution under guardrails ───
        execution = self.optimizer.execute(
            action=guardrail.enforced_action,
            model_name=decision.recommended_model or model_name,
            provider=provider,
            scaler_recommendation=scaler_rec,
        )
        # Guardrails passed → commit the new replica count as the live state.
        if guardrail.allowed and scale_action != "no_change":
            self.scaler.apply(scaler_rec)

        # ── Step 11: Gatekeeper routes compound response ───────────────────
        route = self.gatekeeper.route(workload_id, tpw, decision, intent)
        sequence_trace = self._build_sequence_trace(
            request_id=request_id,
            workload_id=workload_id,
            intent=intent,
            tpw=tpw,
            model_name=model_name,
            latest_uci=uci_val,
            success=success,
            decision=decision,
            provider=provider,
            scale_action=scale_action,
            guardrail=guardrail,
            execution=execution,
            audit_event=audit_event,
            route=route,
            timestamp=ts,
            scaler_payload={
                "forecast_tpw": scaler_rec.forecast_tpw,
                "capacity_tpw": scaler_rec.capacity_tpw,
                "pressure_ratio": scaler_rec.pressure_ratio,
                "cost_delta_usd": scaler_rec.cost_delta_usd,
                "replicas_current": scaler_rec.replicas_current,
                "replicas_recommended": scaler_rec.replicas_recommended,
            },
        )

        result = {
            "request_id":          request_id,
            "workload_id":         workload_id,
            "intent":              intent,
            "model_name":          model_name,
            "request_count":       request_count,
            "tpw":                 round(tpw, 2),
            "atqb_action":         decision.action,
            "atqb_reason":         decision.reason,
            "guardrail_allowed":   guardrail.allowed,
            "guardrail_reason":    guardrail.reason,
            "guardrail_action":    guardrail.enforced_action,
            "recommended_model":   decision.recommended_model or model_name,
            "recommended_provider": provider,
            "uci":                 round(uci_val, 6),
            "uci_trajectory":      trajectory,
            "rate_limit_pressure": decision.rate_limit_pressure,
            "mua_score":           decision.mua_score,
            "p_success_current":   decision.p_success_current,
            "p_success_candidate": decision.p_success_candidate,
            "expected_cost_current": decision.expected_cost_current,
            "expected_cost_candidate": decision.expected_cost_candidate,
            "c_tokens":            latest_uci.c_tokens if latest_uci else 0,
            "c_compute":           latest_uci.c_compute if latest_uci else 0,
            "c_latency_penalty":   latest_uci.c_latency_penalty if latest_uci else 0,
            "n_successful_tasks":  latest_uci.n_successful_tasks if latest_uci else 0,
            "success":             success,
            "quota_transfers":     transfers_applied,
            "scale_action":        scale_action,
            "scaler":              {
                "action":              scaler_rec.action,
                "forecast_tpw":        scaler_rec.forecast_tpw,
                "capacity_tpw":        scaler_rec.capacity_tpw,
                "pressure_ratio":      scaler_rec.pressure_ratio,
                "cost_delta_usd":      scaler_rec.cost_delta_usd,
                "replicas_current":    scaler_rec.replicas_current,
                "replicas_recommended": scaler_rec.replicas_recommended,
                "latency_ratio":       scaler_rec.latency_ratio,
                "reasoning":           scaler_rec.reasoning,
            },
            "optimization_execution": execution,
            "sequence_trace":       sequence_trace,
            "budget_utilization":  round(decision.budget_utilization, 4),
            "token_utilization":   round(decision.token_utilization, 4),
            "audit_event":         audit_event,
            "route":               route,
            "timestamp":           ts,
        }
        self._results.appendleft(result)
        return result

    def latest_results(self, limit: int = 30) -> list[dict]:
        return list(self._results)[:limit]

    def latest_sequence_traces(self, limit: int = 30) -> list[dict]:
        return [
            {
                "request_id": result["request_id"],
                "workload_id": result["workload_id"],
                "intent": result["intent"],
                "timestamp": result["timestamp"],
                "sequence_trace": result["sequence_trace"],
            }
            for result in list(self._results)[:limit]
        ]

    def status(self) -> dict:
        recent = list(self._results)[:100]
        actions = [r["atqb_action"] for r in recent]
        return {
            "agents": ["Gatekeeper", "Auditor", "Governor", "Broker", "Critic", "Scaler"],
            "total_requests_processed": len(self._results),
            "atqb_action_counts": {
                a: actions.count(a) for a in set(actions)
            } if actions else {},
            "provider_uci": self.broker.spot_market_snapshot(),
            "workload_count": len(self.governor.ledgers),
        }
