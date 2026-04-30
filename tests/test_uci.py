from agentic_finops.uci.calculator import compute_uci, token_cost, request_cost
from agentic_finops.uci.models import InferenceRecord


def _rec(
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    slo_ms: float,
    success: bool,
    model: str = "gpt-4",
    tpw: float = 0.8,
) -> InferenceRecord:
    return InferenceRecord(
        request_id="r1",
        workload_id="workload-test",
        model_name=model,
        tpw=tpw,
        tokens_input=tokens_in,
        tokens_output=tokens_out,
        latency_ms=latency_ms,
        slo_latency_ms=slo_ms,
        success=success,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_uci_formula_within_slo() -> None:
    records = [_rec(500, 200, 1000, 2000, True) for _ in range(10)]
    uci = compute_uci(records)
    assert uci is not None
    assert uci.n_successful_tasks == 10
    assert uci.c_latency_penalty == 0.0
    # UCI = (c_tokens + c_compute + 0) / 10 > 0
    assert uci.uci > 0


def test_uci_latency_penalty_applied() -> None:
    # SLO = 2000ms, latency = 4000ms → 2000ms breach each
    records = [_rec(400, 100, 4000, 2000, True) for _ in range(5)]
    uci = compute_uci(records)
    assert uci is not None
    assert uci.c_latency_penalty > 0


def test_uci_expensive_model_higher_than_cheap() -> None:
    gpt5_records = [_rec(1000, 500, 500, 2000, True, "gpt-5", 0.9) for _ in range(5)]
    llama_records = [_rec(1000, 500, 500, 2000, True, "llama-4", 0.9) for _ in range(5)]
    uci_gpt5  = compute_uci(gpt5_records)
    uci_llama = compute_uci(llama_records)
    assert uci_gpt5 is not None
    assert uci_llama is not None
    assert uci_gpt5.uci > uci_llama.uci


def test_uci_denominator_uses_successful_only() -> None:
    records = (
        [_rec(500, 200, 1000, 2000, True)] * 7 +
        [_rec(500, 200, 1000, 2000, False)] * 3
    )
    uci = compute_uci(records)
    assert uci is not None
    assert uci.n_successful_tasks == 7


def test_request_cost_positive() -> None:
    rec = _rec(1000, 500, 800, 2000, True)
    cost = request_cost(rec)
    assert cost > 0
