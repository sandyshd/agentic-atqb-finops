"""FOCUS v1.0 normalization layer.

Maps the project's internal `CostSignal` and provider-specific cost rows
to the FinOps Open Cost and Usage Specification (FOCUS) v1.0 schema, so
that downstream consumers (BI tools, FinOps platforms, peer reviewers)
can interpret cost data using a vendor-neutral standard.

Reference: https://focus.finops.org/
"""

from agentic_finops.focus.schema import (
    FocusRow,
    cost_signal_to_focus,
    map_service_category,
)

__all__ = ["FocusRow", "cost_signal_to_focus", "map_service_category"]
