from agentic_finops.adapters.azure import AzureAdapter


def test_normalize_query_result_maps_to_cost_signals() -> None:
    adapter = AzureAdapter(subscription_ids=["sub-123"], lookback_hours=24, cache_seconds=0)
    payload = {
        "properties": {
            "columns": [
                {"name": "PreTaxCost"},
                {"name": "ResourceId"},
                {"name": "ResourceType"},
                {"name": "ServiceName"},
                {"name": "ResourceGroupName"},
            ],
            "rows": [
                [12.0, "/subscriptions/sub-123/resourceGroups/rg-prod/providers/Microsoft.Compute/virtualMachines/vm-a", "Microsoft.Compute/virtualMachines", "Virtual Machines", "rg-prod"],
                [6.0, "/subscriptions/sub-123/resourceGroups/rg-dev/providers/Microsoft.Storage/storageAccounts/sa1", "Microsoft.Storage/storageAccounts", "Storage", "rg-dev"],
            ],
        }
    }

    signals = adapter._normalize_query_result(
        subscription_id="sub-123",
        payload=payload,
        timestamp="2026-01-01T00:00:00+00:00",
    )

    assert len(signals) == 2
    assert all(s.cloud == "azure" for s in signals)
    assert all(s.account_id == "sub-123" for s in signals)
    assert signals[0].estimated_hourly_cost_usd == 0.5
    assert signals[1].estimated_hourly_cost_usd == 0.25
    assert signals[0].environment == "prod"
    assert signals[1].environment == "nonprod"


def test_collect_returns_cached_data_when_token_missing() -> None:
    adapter = AzureAdapter(subscription_ids=["sub-123"], lookback_hours=24, cache_seconds=0)
    adapter._cached_signals = adapter._normalize_query_result(
        "sub-123",
        {
            "properties": {
                "columns": [
                    {"name": "PreTaxCost"},
                    {"name": "ResourceId"},
                    {"name": "ResourceType"},
                    {"name": "ServiceName"},
                    {"name": "ResourceGroupName"},
                ],
                "rows": [
                    [3.6, "/subscriptions/sub-123/resourceGroups/rg-dev/providers/Microsoft.Web/sites/app1", "Microsoft.Web/sites", "App Service", "rg-dev"],
                ],
            }
        },
        "2026-01-01T00:00:00+00:00",
    )

    adapter._get_access_token = lambda: None

    signals = adapter.collect()

    assert len(signals) == 1
    assert signals[0].resource_name == "app1"
