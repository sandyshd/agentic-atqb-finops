from agentic_finops.focus import FocusRow, cost_signal_to_focus, map_service_category
from agentic_finops.models import CostSignal


def _signal(**overrides) -> CostSignal:
    base = dict(
        timestamp="2026-04-30T12:00:00+00:00",
        cloud="azure",
        account_id="sub-1234",
        service="virtual-machines",
        resource_id="/subscriptions/sub-1234/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-a",
        resource_name="vm-a",
        owner=None,
        environment="prod",
        estimated_hourly_cost_usd=0.50,
        utilization_pct=42.0,
        anomaly_score=0.10,
        tags={"region": "eastus"},
    )
    base.update(overrides)
    return CostSignal(**base)


def test_focus_row_has_all_mandatory_columns() -> None:
    row = cost_signal_to_focus(_signal(), lookback_hours=1)
    mandatory = [
        "BillingAccountId", "Provider", "Publisher", "InvoiceIssuer",
        "ChargePeriodStart", "ChargePeriodEnd", "BillingPeriodStart", "BillingPeriodEnd",
        "ServiceName", "ServiceCategory",
        "BilledCost", "EffectiveCost", "ListCost", "ContractedCost",
        "ChargeCategory", "ChargeFrequency", "BillingCurrency",
    ]
    dump = row.model_dump()
    for col in mandatory:
        assert col in dump, f"missing FOCUS column: {col}"
        assert dump[col] is not None, f"FOCUS column {col} is null"


def test_focus_billed_equals_effective_when_no_discount() -> None:
    row = cost_signal_to_focus(_signal(), lookback_hours=1, contracted_discount_pct=0.0)
    assert row.BilledCost == row.EffectiveCost == row.ListCost == row.ContractedCost
    assert row.PricingCategory == "On-Demand"


def test_focus_applies_contracted_discount() -> None:
    row = cost_signal_to_focus(_signal(), lookback_hours=10, contracted_discount_pct=0.30)
    # ListCost = 0.50 * 10 = 5.00; ContractedCost = 5.00 * 0.70 = 3.50
    assert abs(row.ListCost - 5.00) < 1e-6
    assert abs(row.ContractedCost - 3.50) < 1e-6
    assert row.BilledCost == row.ContractedCost
    assert row.PricingCategory == "Committed"


def test_focus_provider_metadata_for_azure() -> None:
    row = cost_signal_to_focus(_signal(cloud="azure"))
    assert row.Provider == "Microsoft"
    assert row.Publisher == "Microsoft"
    assert row.InvoiceIssuer == "Microsoft"


def test_focus_provider_metadata_for_aws() -> None:
    row = cost_signal_to_focus(_signal(cloud="aws"))
    assert row.Provider == "AWS"
    assert row.Publisher == "AWS"
    assert row.InvoiceIssuer == "AWS"


def test_focus_service_category_mapping() -> None:
    assert map_service_category("virtual-machines") == "Compute"
    assert map_service_category("ec2") == "Compute"
    assert map_service_category("blob-storage") == "Storage"
    assert map_service_category("cosmos-db") == "Databases"
    assert map_service_category("azure-openai") == "AI and Machine Learning"
    assert map_service_category("expressroute") == "Networking"
    assert map_service_category("unknown-service-xyz") == "Other"


def test_focus_charge_period_uses_signal_timestamp() -> None:
    row = cost_signal_to_focus(
        _signal(timestamp="2026-04-30T12:00:00+00:00"),
        lookback_hours=2,
    )
    assert row.ChargePeriodEnd.startswith("2026-04-30T12:00")
    assert row.ChargePeriodStart.startswith("2026-04-30T10:00")


def test_focus_billing_period_is_calendar_month() -> None:
    row = cost_signal_to_focus(_signal(timestamp="2026-04-30T12:00:00+00:00"))
    assert row.BillingPeriodStart.startswith("2026-04-01T00:00")
    assert row.BillingPeriodEnd.startswith("2026-05-01T00:00")


def test_focus_tags_include_environment_and_owner() -> None:
    row = cost_signal_to_focus(_signal(owner="team-platform", environment="prod"))
    assert row.Tags["environment"] == "prod"
    assert row.Tags["owner"] == "team-platform"


def test_focus_row_is_serializable() -> None:
    row = cost_signal_to_focus(_signal())
    payload = row.model_dump_json()
    assert "BilledCost" in payload
    assert "ServiceCategory" in payload


def test_focus_row_pricing_quantity_scales_with_lookback() -> None:
    row = cost_signal_to_focus(_signal(), lookback_hours=24)
    assert row.PricingQuantity == 24.0
    assert row.PricingUnit == "Hour"
