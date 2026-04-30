from __future__ import annotations

from datetime import datetime, timezone
from random import Random

from agentic_finops.models import CostSignal


class SyntheticSignalGenerator:
    def __init__(self, seed: int = 42) -> None:
        self._random = Random(seed)
        self._counter = 0

    def next(self) -> CostSignal:
        self._counter += 1
        cloud = "azure" if self._counter % 2 == 0 else "aws"
        env = "nonprod" if self._counter % 3 else "prod"
        utilization = 5 + self._random.random() * 45
        anomaly = min(1.0, 0.55 + self._random.random() * 0.45)
        cost = round(0.2 + self._random.random() * 4.8, 2)

        return CostSignal(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cloud=cloud,
            account_id=f"acct-{1000 + self._counter}",
            service="compute",
            resource_id=f"{cloud}-vm-{self._counter}",
            resource_name=f"demo-{cloud}-vm-{self._counter}",
            owner=None if self._counter % 5 == 0 else "team-finops",
            environment=env,
            estimated_hourly_cost_usd=cost,
            utilization_pct=round(utilization, 2),
            anomaly_score=round(anomaly, 2),
            tags={"workload": "agentic-finops-demo"},
        )
