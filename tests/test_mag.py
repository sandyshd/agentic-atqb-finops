from agentic_finops.mag.orchestrator import MAGOrchestrator


def _mag() -> MAGOrchestrator:
    return MAGOrchestrator()


def test_process_inference_returns_expected_keys() -> None:
    mag = _mag()
    result = mag.process_inference(
        workload_id="workload-nlp",
        model_name="gpt-4",
        tokens_input=500,
        tokens_output=200,
        latency_ms=800,
        slo_latency_ms=2000.0,
    )
    required_keys = {
        "request_id", "workload_id", "tpw", "atqb_action", "atqb_reason",
        "recommended_model", "recommended_provider", "uci", "uci_trajectory",
        "n_successful_tasks", "success", "budget_utilization", "token_utilization",
        "intent", "guardrail_allowed", "guardrail_action", "optimization_execution",
        "sequence_trace",
    }
    assert required_keys.issubset(result.keys())


def test_tpw_assigned_correctly() -> None:
    mag = _mag()
    result = mag.process_inference("workload-nlp", "gpt-4", 100, 50, 500)
    # workload-nlp maps to TPW 0.85
    assert result["tpw"] == 0.85


def test_critical_request_gets_high_tpw() -> None:
    mag = _mag()
    result = mag.process_inference("workload-chat", "llama-4", 100, 50, 500, is_critical=True)
    assert result["tpw"] == 0.95


def test_governor_ledger_created() -> None:
    mag = _mag()
    mag.process_inference("workload-code", "gpt-4", 1000, 400, 1200)
    assert "workload-code" in mag.governor.ledgers


def test_governor_ledger_usage_recorded() -> None:
    mag = _mag()
    mag.process_inference("workload-code", "gpt-4", 1000, 400, 1200)
    ledger = mag.governor.ledgers["workload-code"]
    assert ledger.tokens_used == 1400
    assert ledger.cost_used_usd > 0


def test_auditor_records_inference() -> None:
    mag = _mag()
    mag.process_inference("workload-analysis", "llama-4", 300, 100, 600)
    records = list(mag.auditor._records)
    assert any(r.workload_id == "workload-analysis" for r in records)


def test_atqb_decision_stored() -> None:
    mag = _mag()
    mag.process_inference("workload-nlp", "gpt-4", 500, 200, 900)
    decisions = mag.governor.latest_decisions()
    assert len(decisions) >= 1


def test_latest_results_stored() -> None:
    mag = _mag()
    mag.process_inference("workload-nlp", "gpt-4", 500, 200, 900)
    assert len(mag.latest_results()) == 1


def test_status_shows_all_agents() -> None:
    mag = _mag()
    status = mag.status()
    assert set(status["agents"]) == {
        "Gatekeeper", "Auditor", "Governor", "Broker", "Critic", "Scaler"
    }


def test_broker_selects_provider() -> None:
    mag = _mag()
    result = mag.process_inference("workload-code", "gpt-4", 200, 100, 400)
    assert result["recommended_provider"] in {"azure", "aws", "neocloud-gpu"}


def test_intent_classification_critical_ops() -> None:
    mag = _mag()
    result = mag.process_inference(
        "workload-chat", "gpt-4", 200, 100, 500, request_text="critical incident now"
    )
    assert result["intent"] == "critical_ops"


def test_guardrail_applies_and_execution_present() -> None:
    mag = _mag()
    result = mag.process_inference(
        "workload-analysis", "gpt-5", 2000, 1200, 3200,
        request_text="compliance audit run",
    )
    assert "execution_action" in result["optimization_execution"]
    assert result["guardrail_action"] is not None


def test_sequence_trace_mirrors_workflow_steps() -> None:
    mag = _mag()
    result = mag.process_inference(
        "workload-nlp",
        "gpt-4",
        500,
        200,
        800,
        request_text="optimize token cost for assistant pipeline",
    )
    trace = result["sequence_trace"]
    assert len(trace) == 12
    assert [step["step"] for step in trace] == list(range(1, 13))
    assert trace[0]["from"] == "Workload"
    assert trace[0]["to"] == "Gatekeeper"
    assert trace[-1]["from"] == "Gatekeeper"
    assert trace[-1]["to"] == "Workload"
    assert any(step["box"] == "Policy Guardrails" for step in trace)
    assert any(step["box"] == "Autonomous Optimization" for step in trace)


def test_latest_sequence_traces_returns_trace_wrapper() -> None:
    mag = _mag()
    mag.process_inference("workload-code", "gpt-4", 200, 100, 400)
    traces = mag.latest_sequence_traces(limit=1)
    assert len(traces) == 1
    assert "sequence_trace" in traces[0]
    assert traces[0]["sequence_trace"][0]["box"] == "Intent Aware Request Handling"


def test_quota_transfer_applied_when_quota_shift_needs_capacity() -> None:
    mag = _mag()
    requester = mag.governor.get_or_create_ledger("workload-hot")
    donor = mag.governor.get_or_create_ledger("workload-cold")
    requester.tokens_used = requester.token_budget - 40
    donor.last_tpw = 0.10

    result = mag.process_inference(
        workload_id="workload-hot",
        model_name="gpt-4",
        tokens_input=220,
        tokens_output=80,
        latency_ms=900,
        request_text="critical incident routing",
        is_critical=True,
        rate_limit_429_count=12,
        retry_after_ms_p95=3000,
    )

    assert result["atqb_action"] == "quota_shift"
    assert len(result["quota_transfers"]) >= 1
    assert any(t["to_workload_id"] == "workload-hot" for t in result["quota_transfers"])

# ──────────────────────────────────────────────────────────────────────────
# CriticAgent: deterministic quality gate (no randomness)
# ──────────────────────────────────────────────────────────────────────────

from agentic_finops.mag.agents import CriticAgent


def test_critic_rejects_empty_and_short_outputs() -> None:
    critic = CriticAgent()
    assert critic.validate(None, 100, 1000) is False
    assert critic.validate("", 100, 1000) is False
    assert critic.validate("   ", 100, 1000) is False
    assert critic.validate("hi", 100, 1000) is False


def test_critic_accepts_normal_output() -> None:
    critic = CriticAgent()
    assert critic.validate("This is a coherent multi-token answer.", 100, 1000) is True


def test_critic_rejects_known_failure_signatures() -> None:
    critic = CriticAgent()
    assert critic.validate("Sorry, I cannot help with that request.", 100, 1000) is False
    assert critic.validate("I'm unable to comply with this.", 100, 1000) is False
    assert critic.validate("As an AI language model, I do not have opinions.", 100, 1000) is False
    assert critic.validate("hello world <|endoftext|> garbage", 100, 1000) is False
    assert critic.validate("[ERROR] upstream timeout occurred", 100, 1000) is False


def test_critic_rejects_hard_latency_violation() -> None:
    critic = CriticAgent()
    # 2x SLO is the hard cutoff; 2.5x must fail even for a clean output
    assert critic.validate("A perfectly fine answer body.", 2500, 1000) is False
    # Just under threshold passes
    assert critic.validate("A perfectly fine answer body.", 1900, 1000) is True


def test_critic_rejects_degenerate_repetition() -> None:
    critic = CriticAgent()
    repeating = "spam " * 30
    assert critic.validate(repeating, 100, 1000) is False


def test_critic_rejects_json_error_envelope() -> None:
    critic = CriticAgent()
    assert critic.validate('{"error": "rate_limited"}', 100, 1000) is False


def test_critic_enforces_json_format_when_requested() -> None:
    critic = CriticAgent()
    assert critic.validate("not a json reply", 100, 1000, expected_format="json") is False
    assert critic.validate('{"answer": 42}', 100, 1000, expected_format="json") is True


def test_critic_is_deterministic() -> None:
    critic = CriticAgent()
    text = "Deterministic gate, repeated checks must agree."
    results = {critic.validate(text, 200, 1000) for _ in range(50)}
    assert results == {True}
