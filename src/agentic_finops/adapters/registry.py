from __future__ import annotations

from agentic_finops.adapters.aws import AwsAdapter
from agentic_finops.adapters.azure import AzureAdapter
from agentic_finops.adapters.base import CostSignalAdapter


def get_adapters() -> list[CostSignalAdapter]:
    return [AzureAdapter(), AwsAdapter()]
