from __future__ import annotations

from typing import Protocol

from agentic_finops.models import CostSignal


class CostSignalAdapter(Protocol):
    def collect(self) -> list[CostSignal]:
        """Collect latest cost signals from a provider and normalize them."""
