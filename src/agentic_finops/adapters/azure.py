from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agentic_finops.models import CostSignal


logger = logging.getLogger(__name__)


class _RateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: int = 0) -> None:
        super().__init__(message)
        self.retry_after_seconds = max(0, retry_after_seconds)


class AzureAdapter:
    """Collect Azure cost signals using Cost Management Query API.

    Required env vars for live data:
    - AZURE_SUBSCRIPTION_ID or AZURE_SUBSCRIPTION_IDS (comma-separated)
    - Azure credentials recognized by DefaultAzureCredential
    """

    _SCOPE = "https://management.azure.com/.default"

    def __init__(
        self,
        *,
        subscription_ids: list[str] | None = None,
        lookback_hours: int | None = None,
        cache_seconds: int | None = None,
        max_signals: int | None = None,
    ) -> None:
        ids = subscription_ids if subscription_ids is not None else self._subscription_ids_from_env()
        self.subscription_ids = [sid.strip() for sid in ids if sid.strip()]
        self.lookback_hours = max(1, lookback_hours or int(os.getenv("AZURE_COST_LOOKBACK_HOURS", "24")))
        self.cache_seconds = max(0, cache_seconds or int(os.getenv("AZURE_ADAPTER_CACHE_SECONDS", "300")))
        self.max_signals = max(1, max_signals or int(os.getenv("AZURE_ADAPTER_MAX_SIGNALS", "50")))
        self.min_query_interval_seconds = max(
            0,
            int(os.getenv("AZURE_ADAPTER_MIN_QUERY_INTERVAL_SECONDS", "300")),
        )
        self.initial_throttle_backoff_seconds = max(
            1,
            int(os.getenv("AZURE_ADAPTER_THROTTLE_BACKOFF_SECONDS", "60")),
        )
        self.max_throttle_backoff_seconds = max(
            self.initial_throttle_backoff_seconds,
            int(os.getenv("AZURE_ADAPTER_THROTTLE_MAX_BACKOFF_SECONDS", "1800")),
        )
        self.enabled = os.getenv("AZURE_ADAPTER_ENABLED", "true").lower() == "true"

        self._cached_signals: list[CostSignal] = []
        self._next_refresh_at: float = 0.0
        self._credential: Any | None = None
        self._current_backoff_seconds: int = self.initial_throttle_backoff_seconds

    def collect(self) -> list[CostSignal]:
        if not self.enabled or not self.subscription_ids:
            return []

        now = time.time()
        if now < self._next_refresh_at:
            return list(self._cached_signals)

        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            token = self._get_access_token()
            if not token:
                logger.info("AzureAdapter disabled for this cycle: no access token available")
                return list(self._cached_signals)

            all_signals: list[CostSignal] = []
            throttled_delays: list[int] = []
            for subscription_id in self.subscription_ids:
                try:
                    payload = self._query_subscription_costs(subscription_id, token)
                    all_signals.extend(self._normalize_query_result(subscription_id, payload, timestamp))
                except _RateLimitError as exc:
                    throttled_delays.append(exc.retry_after_seconds)
                    logger.warning(
                        "AzureAdapter throttled for subscription %s; retry_after=%ss",
                        subscription_id,
                        exc.retry_after_seconds,
                    )

            self._cached_signals = all_signals[: self.max_signals]
            if throttled_delays:
                throttle_delay = max(
                    self._current_backoff_seconds,
                    max(throttled_delays),
                    self.cache_seconds,
                    self.min_query_interval_seconds,
                )
                self._next_refresh_at = now + throttle_delay
                self._current_backoff_seconds = min(
                    self.max_throttle_backoff_seconds,
                    max(self._current_backoff_seconds * 2, self.initial_throttle_backoff_seconds),
                )
            else:
                self._next_refresh_at = now + max(self.cache_seconds, self.min_query_interval_seconds)
                self._current_backoff_seconds = self.initial_throttle_backoff_seconds
            return list(self._cached_signals)
        except Exception as exc:
            logger.warning("AzureAdapter collect failed: %s", exc)
            self._next_refresh_at = now + max(
                min(60, self.cache_seconds or 60),
                self.min_query_interval_seconds,
            )
            return list(self._cached_signals)

    def _subscription_ids_from_env(self) -> list[str]:
        many = os.getenv("AZURE_SUBSCRIPTION_IDS", "")
        single = os.getenv("AZURE_SUBSCRIPTION_ID", "")
        raw = many or single
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]

    def _get_access_token(self) -> str | None:
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential

                self._credential = DefaultAzureCredential()
            except Exception as exc:
                logger.info("Azure credential initialization failed: %s", exc)
                return None
        try:
            return self._credential.get_token(self._SCOPE).token
        except Exception as exc:
            logger.info("Azure token acquisition failed: %s", exc)
            return None

    def _query_subscription_costs(self, subscription_id: str, token: str) -> dict[str, Any]:
        to_dt = datetime.now(timezone.utc)
        from_dt = to_dt - timedelta(hours=self.lookback_hours)
        url = (
            "https://management.azure.com/subscriptions/"
            f"{subscription_id}/providers/Microsoft.CostManagement/query?api-version=2023-03-01"
        )
        body = {
            "type": "Usage",
            "timeframe": "Custom",
            "timePeriod": {
                "from": from_dt.isoformat(),
                "to": to_dt.isoformat(),
            },
            "dataset": {
                "granularity": "None",
                "aggregation": {
                    "totalCost": {"name": "PreTaxCost", "function": "Sum"},
                },
                "grouping": [
                    {"type": "Dimension", "name": "ResourceId"},
                    {"type": "Dimension", "name": "ResourceType"},
                    {"type": "Dimension", "name": "ServiceName"},
                    {"type": "Dimension", "name": "ResourceGroupName"},
                ],
            },
        }

        req = Request(
            url=url,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            if exc.code == 429:
                retry_after = self._parse_retry_after_seconds(exc)
                raise _RateLimitError(
                    f"Azure Cost query HTTP 429: {detail}",
                    retry_after_seconds=retry_after,
                ) from exc
            raise RuntimeError(f"Azure Cost query HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Azure Cost query network error: {exc.reason}") from exc

    @staticmethod
    def _parse_retry_after_seconds(exc: HTTPError) -> int:
        header_value = exc.headers.get("Retry-After") if exc.headers else None
        if not header_value:
            return 0
        try:
            return max(0, int(header_value))
        except (TypeError, ValueError):
            return 0

    def _normalize_query_result(
        self,
        subscription_id: str,
        payload: dict[str, Any],
        timestamp: str,
    ) -> list[CostSignal]:
        properties = payload.get("properties", {})
        columns = properties.get("columns", [])
        rows = properties.get("rows", [])
        if not columns or not rows:
            return []

        col_index = {c.get("name"): i for i, c in enumerate(columns)}
        cost_idx = col_index.get("PreTaxCost")
        if cost_idx is None:
            # The query may return the alias name in some API versions.
            cost_idx = col_index.get("totalCost")
        rid_idx = col_index.get("ResourceId")
        rtype_idx = col_index.get("ResourceType")
        svc_idx = col_index.get("ServiceName")
        rg_idx = col_index.get("ResourceGroupName")
        if cost_idx is None:
            return []

        interim: list[dict[str, Any]] = []
        for row in rows:
            try:
                raw_cost = float(row[cost_idx])
            except (TypeError, ValueError, IndexError):
                continue
            if raw_cost <= 0:
                continue

            resource_id = self._safe_row_value(row, rid_idx, default="unknown")
            resource_type = self._safe_row_value(row, rtype_idx, default="unknown")
            service_name = self._safe_row_value(row, svc_idx, default=resource_type)
            resource_group = self._safe_row_value(row, rg_idx, default="")
            hourly_cost = raw_cost / self.lookback_hours
            interim.append(
                {
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "service_name": service_name,
                    "resource_group": resource_group,
                    "hourly_cost": hourly_cost,
                }
            )

        if not interim:
            return []

        costs = [x["hourly_cost"] for x in interim]
        med = max(0.001, median(costs))
        max_cost = max(costs)

        signals: list[CostSignal] = []
        for item in interim:
            deviation = abs(item["hourly_cost"] - med) / med
            anomaly = min(1.0, round(deviation, 4))
            utilization_pct = 0.0 if max_cost <= 0 else min(100.0, (item["hourly_cost"] / max_cost) * 100.0)
            environment = self._infer_environment(item["resource_group"], item["resource_id"])

            signals.append(
                CostSignal(
                    timestamp=timestamp,
                    cloud="azure",
                    account_id=subscription_id,
                    service=self._normalize_service(item["service_name"], item["resource_type"]),
                    resource_id=item["resource_id"],
                    resource_name=self._resource_name(item["resource_id"]),
                    owner=None,
                    environment=environment,
                    estimated_hourly_cost_usd=round(item["hourly_cost"], 6),
                    utilization_pct=round(utilization_pct, 2),
                    anomaly_score=anomaly,
                    tags={
                        "source": "azure-cost-management",
                        "resource_type": item["resource_type"],
                        "resource_group": item["resource_group"],
                    },
                )
            )
        return signals

    @staticmethod
    def _safe_row_value(row: list[Any], idx: int | None, default: str) -> str:
        if idx is None or idx >= len(row):
            return default
        val = row[idx]
        return default if val in (None, "") else str(val)

    @staticmethod
    def _resource_name(resource_id: str) -> str:
        if not resource_id or resource_id == "unknown":
            return "unknown"
        return resource_id.rstrip("/").split("/")[-1]

    @staticmethod
    def _normalize_service(service_name: str, resource_type: str) -> str:
        if service_name and service_name != "unknown":
            return service_name.lower().replace(" ", "-")
        return resource_type.lower().replace("microsoft.", "").replace("/", "-")

    @staticmethod
    def _infer_environment(resource_group: str, resource_id: str) -> str:
        marker = f"{resource_group} {resource_id}".lower()
        if any(tok in marker for tok in ("prod", "prd", "production")):
            return "prod"
        return "nonprod"
