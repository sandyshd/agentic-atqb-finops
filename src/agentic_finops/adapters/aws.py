from __future__ import annotations

from agentic_finops.models import CostSignal


class AwsAdapter:
    """Stub adapter for AWS cost + utilization signals."""

    def collect(self) -> list[CostSignal]:
        return []
