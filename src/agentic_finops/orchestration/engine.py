from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from agentic_finops.detection.anomaly import recommend_action
from agentic_finops.models import ActionRecord, CostSignal
from agentic_finops.policy.engine import evaluate
from agentic_finops.store.memory_store import MemoryStore


class Orchestrator:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def process(self, signal: CostSignal) -> ActionRecord | None:
        self.store.add_event(signal)
        recommendation = recommend_action(signal)
        if recommendation is None:
            return None

        decision = evaluate(recommendation)
        trace_id = f"trace-{uuid4()}"

        if not decision.allowed:
            state = "blocked"
        elif decision.requires_approval:
            state = "pending_approval"
        else:
            state = "auto_applied"

        record = ActionRecord(
            recommendation=recommendation,
            decision=decision,
            state=state,
            trace_id=trace_id,
        )
        self.store.add_action(record)
        return record

    def process_batch(self, signals: list[CostSignal]) -> list[ActionRecord]:
        records: list[ActionRecord] = []
        for signal in signals:
            if not signal.timestamp:
                signal.timestamp = datetime.now(timezone.utc).isoformat()
            record = self.process(signal)
            if record:
                records.append(record)
        return records
