from agentic_finops.models import CostSignal
from agentic_finops.orchestration.engine import Orchestrator
from agentic_finops.store.memory_store import MemoryStore


def test_orchestrator_generates_action_for_anomaly() -> None:
    store = MemoryStore()
    orchestrator = Orchestrator(store=store)
    signal = CostSignal(
        timestamp="2026-01-01T00:00:00+00:00",
        cloud="aws",
        account_id="acct-2",
        service="compute",
        resource_id="i-1",
        resource_name="i-1",
        estimated_hourly_cost_usd=2.0,
        utilization_pct=5,
        anomaly_score=0.95,
        environment="nonprod",
    )

    record = orchestrator.process(signal)
    assert record is not None
    assert record.trace_id.startswith("trace-")
    assert len(store.latest_events()) == 1
    assert len(store.latest_actions()) == 1
