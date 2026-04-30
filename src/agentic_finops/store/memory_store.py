from __future__ import annotations

from collections import deque

from agentic_finops.models import ActionRecord, CostSignal


class MemoryStore:
    def __init__(self, max_events: int = 1000, max_actions: int = 1000) -> None:
        self._events: deque[CostSignal] = deque(maxlen=max_events)
        self._actions: deque[ActionRecord] = deque(maxlen=max_actions)

    def add_event(self, event: CostSignal) -> None:
        self._events.appendleft(event)

    def add_action(self, action: ActionRecord) -> None:
        self._actions.appendleft(action)

    def latest_events(self, limit: int = 50) -> list[CostSignal]:
        return list(self._events)[:limit]

    def latest_actions(self, limit: int = 50) -> list[ActionRecord]:
        return list(self._actions)[:limit]

    def metrics(self) -> dict[str, float | int]:
        total_actions = len(self._actions)
        auto_applied = sum(1 for a in self._actions if a.state == "auto_applied")
        approval_pending = sum(1 for a in self._actions if a.state == "pending_approval")
        blocked = sum(1 for a in self._actions if a.state == "blocked")
        estimated_savings = round(
            sum(a.recommendation.expected_monthly_savings_usd for a in self._actions), 2
        )
        return {
            "event_count": len(self._events),
            "action_count": total_actions,
            "auto_applied_count": auto_applied,
            "pending_approval_count": approval_pending,
            "blocked_count": blocked,
            "estimated_monthly_savings_usd": estimated_savings,
        }
