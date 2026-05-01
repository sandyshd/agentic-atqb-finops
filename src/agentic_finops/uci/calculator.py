from __future__ import annotations

from datetime import datetime, timezone

from agentic_finops.uci.models import InferenceRecord, UCIRecord

# ---------------------------------------------------------------------------
# Token price per 1 000 tokens — heterogeneous model tier pricing
# ---------------------------------------------------------------------------
TOKEN_PRICE_PER_1K: dict[str, float] = {
    "gpt-5":    0.015,
    "claude-4": 0.012,
    "gpt-4":    0.010,
    "llama-4":  0.002,
    "default":  0.008,
}

COMPUTE_OVERHEAD_PER_CALL_USD: float = 0.0005   # transient GPU/NPU orchestration
LATENCY_PENALTY_PER_MS_USD: float = 0.000002    # fiscal loss per ms over SLO


def token_cost(record: InferenceRecord) -> float:
    """Weighted token cost for a single inference call (C_tokens contribution)."""
    price = TOKEN_PRICE_PER_1K.get(record.model_name.lower(), TOKEN_PRICE_PER_1K["default"])
    return ((record.tokens_input + record.tokens_output) / 1_000) * price * record.tpw


def request_cost(record: InferenceRecord) -> float:
    """Actual cost of a single inference call for ledger tracking."""
    price = TOKEN_PRICE_PER_1K.get(record.model_name.lower(), TOKEN_PRICE_PER_1K["default"])
    return (
        ((record.tokens_input + record.tokens_output) / 1_000) * price
        + (record.request_count * COMPUTE_OVERHEAD_PER_CALL_USD)
    )


def compute_uci(records: list[InferenceRecord]) -> UCIRecord | None:
    """
    Compute UCI over a window of inference records.

    UCI = (C_tokens + C_compute + C_latency_penalty) / N_successful_tasks
    """
    if not records:
        return None

    workload_id = records[0].workload_id
    period = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00Z")

    c_tokens = sum(token_cost(r) for r in records)
    c_compute = sum(r.request_count for r in records) * COMPUTE_OVERHEAD_PER_CALL_USD
    c_latency_penalty = sum(
        max(0.0, r.latency_ms - r.slo_latency_ms) * LATENCY_PENALTY_PER_MS_USD
        for r in records
    )
    # One validation gate decision per record. `request_count` legitimately
    # scales cost (c_compute), but the UCI denominator must reflect the number
    # of *validated* outcomes — not how many calls were aggregated into a
    # single record. Using request_count here would amplify one Critic decision
    # by the batch size and bias UCI / MUA evidence.
    n_successful = sum(1 for r in records if r.success)
    uci = (c_tokens + c_compute + c_latency_penalty) / max(1, n_successful)

    return UCIRecord(
        period=period,
        workload_id=workload_id,
        c_tokens=round(c_tokens, 6),
        c_compute=round(c_compute, 6),
        c_latency_penalty=round(c_latency_penalty, 6),
        n_successful_tasks=n_successful,
        uci=round(uci, 6),
    )
