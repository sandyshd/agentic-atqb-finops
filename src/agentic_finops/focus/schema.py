"""FOCUS v1.0 row schema and converters.

This module defines `FocusRow`, a Pydantic model whose field set covers the
mandatory columns of the FinOps Open Cost and Usage Specification (FOCUS)
v1.0. The schema is intentionally a strict subset that can be losslessly
emitted from the project's `CostSignal` plus provider-specific metadata.

Provider-specific adapters (Azure, AWS) populate the FOCUS columns from
their native billing payloads and call `cost_signal_to_focus(...)` (or
construct `FocusRow` directly) to produce a normalized stream.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field

from agentic_finops.models import CostSignal


# ---------------------------------------------------------------------------
# FOCUS v1.0 — minimum mandatory columns
# Reference: https://focus.finops.org/#columns
# ---------------------------------------------------------------------------
class FocusRow(BaseModel):
    """A single FOCUS v1.0-aligned billing row.

    All field names match the FOCUS v1.0 column names so this model can be
    serialized directly to the canonical schema.
    """

    # Identifiers
    BillingAccountId: str
    BillingAccountName: str | None = None
    SubAccountId: str | None = None
    SubAccountName: str | None = None
    ResourceId: str | None = None
    ResourceName: str | None = None
    RegionId: str | None = None
    RegionName: str | None = None

    # Provider / issuer
    Provider: str
    Publisher: str
    InvoiceIssuer: str

    # Time window (ISO-8601, UTC)
    ChargePeriodStart: str
    ChargePeriodEnd: str
    BillingPeriodStart: str
    BillingPeriodEnd: str

    # Service classification
    ServiceName: str
    ServiceCategory: str

    # Cost
    BilledCost: float = Field(ge=0)
    EffectiveCost: float = Field(ge=0)
    ListCost: float = Field(ge=0)
    ContractedCost: float = Field(ge=0)

    # Pricing
    ListUnitPrice: float | None = Field(default=None, ge=0)
    ContractedUnitPrice: float | None = Field(default=None, ge=0)
    PricingCategory: str = "On-Demand"
    PricingQuantity: float | None = Field(default=None, ge=0)
    PricingUnit: str | None = None

    # Usage
    ConsumedQuantity: float | None = Field(default=None, ge=0)
    ConsumedUnit: str | None = None

    # Charge metadata
    ChargeCategory: str = "Usage"
    ChargeClass: str | None = None
    ChargeFrequency: str = "Usage-Based"
    ChargeDescription: str | None = None

    # Currency
    BillingCurrency: str = "USD"

    # Tags (FOCUS spec models this as a JSON-serializable object)
    Tags: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Service-category mapping
# ---------------------------------------------------------------------------
# Best-effort mapping from provider service tokens to FOCUS ServiceCategory
# (https://focus.finops.org/#allowed-values-servicecategory).
_SERVICE_CATEGORY_MAP: dict[str, str] = {
    # Compute
    "virtual-machines": "Compute",
    "compute": "Compute",
    "ec2": "Compute",
    "kubernetes-service": "Compute",
    "aks": "Compute",
    "container-instances": "Compute",
    "app-service": "Compute",
    "functions": "Compute",
    "lambda": "Compute",
    "batch": "Compute",
    # Storage
    "storage": "Storage",
    "blob-storage": "Storage",
    "s3": "Storage",
    "files": "Storage",
    "managed-disks": "Storage",
    # Databases
    "sql-database": "Databases",
    "cosmos-db": "Databases",
    "rds": "Databases",
    "dynamodb": "Databases",
    "postgresql": "Databases",
    "mysql": "Databases",
    "redis": "Databases",
    # Networking
    "virtual-network": "Networking",
    "load-balancer": "Networking",
    "application-gateway": "Networking",
    "vpn": "Networking",
    "expressroute": "Networking",
    "bandwidth": "Networking",
    "cloudfront": "Networking",
    # AI + ML
    "openai": "AI and Machine Learning",
    "azure-openai": "AI and Machine Learning",
    "cognitive-services": "AI and Machine Learning",
    "machine-learning": "AI and Machine Learning",
    "bedrock": "AI and Machine Learning",
    "sagemaker": "AI and Machine Learning",
    # Analytics
    "synapse": "Analytics",
    "data-factory": "Analytics",
    "redshift": "Analytics",
    "kusto": "Analytics",
    # Identity / Security
    "key-vault": "Identity",
    "active-directory": "Identity",
    "iam": "Identity",
    # Other
}


def map_service_category(service: str) -> str:
    """Map a normalized service token to a FOCUS ServiceCategory.

    Returns "Other" when no mapping exists, which is a valid FOCUS value.
    """
    if not service:
        return "Other"
    key = service.lower().strip()
    if key in _SERVICE_CATEGORY_MAP:
        return _SERVICE_CATEGORY_MAP[key]
    # Heuristic fallback: substring match against known categories
    for token, category in _SERVICE_CATEGORY_MAP.items():
        if token in key:
            return category
    return "Other"


# ---------------------------------------------------------------------------
# Provider metadata
# ---------------------------------------------------------------------------
_PROVIDER_METADATA: dict[str, dict[str, str]] = {
    "azure": {
        "Provider": "Microsoft",
        "Publisher": "Microsoft",
        "InvoiceIssuer": "Microsoft",
    },
    "aws": {
        "Provider": "AWS",
        "Publisher": "AWS",
        "InvoiceIssuer": "AWS",
    },
    "gcp": {
        "Provider": "Google",
        "Publisher": "Google",
        "InvoiceIssuer": "Google",
    },
}


def _provider_metadata(cloud: str) -> dict[str, str]:
    return _PROVIDER_METADATA.get(
        cloud.lower(),
        {"Provider": cloud, "Publisher": cloud, "InvoiceIssuer": cloud},
    )


# ---------------------------------------------------------------------------
# CostSignal -> FocusRow conversion
# ---------------------------------------------------------------------------
def cost_signal_to_focus(
    signal: CostSignal,
    *,
    lookback_hours: int = 1,
    contracted_discount_pct: float = 0.0,
    pricing_unit: str = "Hour",
    consumed_quantity: float | None = None,
    consumed_unit: str | None = None,
) -> FocusRow:
    """Convert a `CostSignal` into a FOCUS v1.0-aligned row.

    The signal carries an `estimated_hourly_cost_usd`. We project it across
    the `lookback_hours` window to compute `BilledCost`. `EffectiveCost` is
    the same as `BilledCost` minus any commitment discount specified by
    `contracted_discount_pct` (0.0 by default — On-Demand pricing).
    """
    try:
        end_dt = datetime.fromisoformat(signal.timestamp.replace("Z", "+00:00"))
    except ValueError:
        end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(hours=max(1, lookback_hours))

    list_cost = round(signal.estimated_hourly_cost_usd * max(1, lookback_hours), 6)
    discount = max(0.0, min(1.0, contracted_discount_pct))
    contracted_cost = round(list_cost * (1.0 - discount), 6)
    # In FOCUS, BilledCost is what the customer is invoiced; EffectiveCost
    # amortizes commitments. With no commitment, the three are equal.
    billed_cost = contracted_cost
    effective_cost = contracted_cost

    metadata = _provider_metadata(signal.cloud)
    period_start_iso = start_dt.astimezone(timezone.utc).isoformat()
    period_end_iso = end_dt.astimezone(timezone.utc).isoformat()
    # Billing period (calendar month containing the charge)
    billing_start = start_dt.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    if billing_start.month == 12:
        billing_end = billing_start.replace(year=billing_start.year + 1, month=1)
    else:
        billing_end = billing_start.replace(month=billing_start.month + 1)

    tags = dict(signal.tags or {})
    tags.setdefault("environment", signal.environment)
    if signal.owner:
        tags.setdefault("owner", signal.owner)

    return FocusRow(
        BillingAccountId=signal.account_id,
        BillingAccountName=signal.account_id,
        SubAccountId=signal.account_id,
        SubAccountName=signal.account_id,
        ResourceId=signal.resource_id,
        ResourceName=signal.resource_name,
        RegionId=tags.get("region"),
        RegionName=tags.get("region"),
        Provider=metadata["Provider"],
        Publisher=metadata["Publisher"],
        InvoiceIssuer=metadata["InvoiceIssuer"],
        ChargePeriodStart=period_start_iso,
        ChargePeriodEnd=period_end_iso,
        BillingPeriodStart=billing_start.isoformat(),
        BillingPeriodEnd=billing_end.isoformat(),
        ServiceName=signal.service,
        ServiceCategory=map_service_category(signal.service),
        BilledCost=billed_cost,
        EffectiveCost=effective_cost,
        ListCost=list_cost,
        ContractedCost=contracted_cost,
        ListUnitPrice=round(signal.estimated_hourly_cost_usd, 6),
        ContractedUnitPrice=round(signal.estimated_hourly_cost_usd * (1.0 - discount), 6),
        PricingCategory="Committed" if discount > 0 else "On-Demand",
        PricingQuantity=float(max(1, lookback_hours)),
        PricingUnit=pricing_unit,
        ConsumedQuantity=consumed_quantity,
        ConsumedUnit=consumed_unit,
        ChargeCategory="Usage",
        ChargeClass=None,
        ChargeFrequency="Usage-Based",
        ChargeDescription=f"{signal.service} usage on {signal.cloud}",
        BillingCurrency="USD",
        Tags=tags,
    )
