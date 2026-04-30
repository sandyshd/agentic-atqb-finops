"""Azure OpenAI / AI Services token-consumption adapter.

Queries Azure Monitor Metrics API for real token usage across all
Cognitive Services accounts (kinds: OpenAI, AIServices, CognitiveServices)
found in the configured subscriptions.

Env vars:
    AZURE_SUBSCRIPTION_ID          comma-separated subscription IDs
    AZURE_OPENAI_LOOKBACK_HOURS    hours of history to sum (default 24)
    AZURE_OPENAI_CACHE_SECONDS     cache TTL for metric results (default 300)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Metrics we want from Azure Monitor for Cognitive Services
# Actual names discovered from metricdefinitions API:
#   AIServices: InputTokens, OutputTokens, TotalTokens
#   Legacy OpenAI: ProcessedPromptTokens, GeneratedCompletionTokens, TokenTransaction
_TOKEN_METRICS_NEW = "InputTokens,OutputTokens,TotalTokens"
_TOKEN_METRICS_LEGACY = "ProcessedPromptTokens,GeneratedCompletionTokens,TokenTransaction"
_CS_KINDS = {"openai", "aiservices", "cognitiveservices"}
_ARM_BASE = "https://management.azure.com"


class AzureOpenAIMetricsAdapter:
    """Collect real Azure OpenAI token-consumption metrics from Azure Monitor."""

    def __init__(self) -> None:
        raw = os.getenv("AZURE_SUBSCRIPTION_ID", "")
        self.subscription_ids = [s.strip() for s in raw.split(",") if s.strip()]
        self.lookback_hours = max(1, int(os.getenv("AZURE_OPENAI_LOOKBACK_HOURS", "24")))
        self.cache_seconds = max(0, int(os.getenv("AZURE_OPENAI_CACHE_SECONDS", "300")))
        self.slo_latency_ms = max(1.0, float(os.getenv("AZURE_OPENAI_SLO_LATENCY_MS", "2000")))
        self.enabled = os.getenv("AZURE_ADAPTER_ENABLED", "true").lower() == "true"

        self._credential: Any | None = None
        self._cached: list[dict] = []
        self._deployment_cached: list[dict] = []
        self._pending_events: list[dict] = []
        self._emitted_event_keys: set[str] = set()
        self._next_refresh_at: float = 0.0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def collect(self) -> list[dict]:
        """Return token-consumption records, refreshing from Azure Monitor when cache expires."""
        self._refresh_if_needed()
        return list(self._cached)

    def deployment_breakdown(self) -> list[dict]:
        """Return real Azure Monitor usage split by resource and deployment."""
        self._refresh_if_needed()
        return list(self._deployment_cached)

    def collect_inference_events(self) -> list[dict]:
        """Return newly observed deployment usage buckets as MAG-ready inference events."""
        self._refresh_if_needed()
        pending = list(self._pending_events)
        self._pending_events.clear()
        return pending

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_access_token(self) -> str | None:
        try:
            if self._credential is None:
                from azure.identity import DefaultAzureCredential
                self._credential = DefaultAzureCredential()
            return self._credential.get_token("https://management.azure.com/.default").token
        except Exception as exc:
            logger.warning("AzureOpenAIMetricsAdapter: cannot get token: %s", exc)
            return None

    def _discover_cs_resources(self, token: str) -> list[dict]:
        """List all Cognitive Services accounts across subscriptions."""
        resources: list[dict] = []
        for sub in self.subscription_ids:
            url = (_ARM_BASE + "/subscriptions/" + sub
                   + "/providers/Microsoft.CognitiveServices/accounts?api-version=2023-05-01")
            try:
                data = self._arm_get(url, token)
                for acct in data.get("value", []):
                    kind = acct.get("kind", "").lower()
                    if kind in _CS_KINDS:
                        resources.append({
                            "id": acct["id"],
                            "name": acct["name"],
                            "kind": acct.get("kind", ""),
                            "subscription_id": sub,
                            "resource_group": acct["id"].split("/")[4],
                        })
            except Exception as exc:
                logger.warning("Resource discovery failed for sub %s: %s", sub, exc)
        return resources

    def _refresh_if_needed(self) -> None:
        if not self.enabled or not self.subscription_ids:
            return

        now = time.time()
        if now < self._next_refresh_at:
            return

        try:
            token = self._get_access_token()
            if not token:
                self._next_refresh_at = now + min(60, self.cache_seconds or 60)
                return

            resources = self._discover_cs_resources(token)
            deployment_rows: list[dict] = []
            resource_rows: list[dict] = []
            pending_events: list[dict] = []

            for resource in resources:
                try:
                    deployments = self._fetch_deployment_metrics(token, resource)
                except Exception as exc:
                    logger.warning("OpenAI deployment metrics fetch failed for %s: %s", resource.get("name"), exc)
                    continue

                if not deployments:
                    continue

                deployment_rows.extend(deployments)
                resource_rows.append(self._resource_summary(resource, deployments))
                pending_events.extend(self._build_events(resource, deployments))

            self._cached = sorted(resource_rows, key=lambda row: row["total_tokens"], reverse=True)
            self._deployment_cached = sorted(
                deployment_rows,
                key=lambda row: (row["total_tokens"], row["request_count"]),
                reverse=True,
            )
            self._pending_events.extend(pending_events)
            self._next_refresh_at = now + self.cache_seconds
            logger.info(
                "AzureOpenAIMetricsAdapter: refreshed %d resource(s), %d deployment(s), %d event(s)",
                len(self._cached),
                len(self._deployment_cached),
                len(pending_events),
            )
        except Exception as exc:
            logger.warning("AzureOpenAIMetricsAdapter refresh failed: %s", exc)
            self._next_refresh_at = now + min(60, self.cache_seconds or 60)

    def _fetch_deployment_metrics(self, token: str, resource: dict) -> list[dict]:
        token_data = self._query_metric_set(
            token,
            resource,
            metric_names=(_TOKEN_METRICS_NEW, _TOKEN_METRICS_LEGACY),
            aggregation="Total",
        )
        if not token_data:
            return []

        request_data = self._query_metric_set(
            token,
            resource,
            metric_names=("AzureOpenAIRequests", "ModelRequests"),
            aggregation="Total",
        )
        latency_data = self._query_metric_set(
            token,
            resource,
            metric_names=("TimeToResponse",),
            aggregation="Average",
        )

        deployments: dict[tuple[str, str], dict] = {}
        self._merge_metric_response(deployments, token_data)
        if request_data:
            self._merge_metric_response(deployments, request_data)
        if latency_data:
            self._merge_metric_response(deployments, latency_data)

        rows: list[dict] = []
        for (_, _), row in deployments.items():
            points = []
            total_requests = 0
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            latency_total = 0.0
            latency_count = 0
            last_activity_at = ""

            for timestamp, point in sorted(row["hourly_points"].items()):
                point_total = point["total_tokens"] or (point["prompt_tokens"] + point["completion_tokens"])
                point_latency = (
                    point["latency_ms_sum"] / point["latency_ms_count"]
                    if point["latency_ms_count"]
                    else 0.0
                )
                points.append(
                    {
                        "timestamp": timestamp,
                        "prompt_tokens": point["prompt_tokens"],
                        "completion_tokens": point["completion_tokens"],
                        "total_tokens": point_total,
                        "request_count": point["request_count"],
                        "latency_ms": round(point_latency, 2),
                    }
                )
                total_requests += point["request_count"]
                prompt_tokens += point["prompt_tokens"]
                completion_tokens += point["completion_tokens"]
                total_tokens += point_total
                if point["latency_ms_count"]:
                    latency_total += point["latency_ms_sum"]
                    latency_count += point["latency_ms_count"]
                if point_total > 0 or point["request_count"] > 0:
                    last_activity_at = timestamp

            if total_tokens == 0 and total_requests == 0:
                continue

            avg_latency_ms = round(latency_total / latency_count, 2) if latency_count else 0.0
            workload_id = self._workload_id(resource["name"], row["deployment_name"])
            rows.append(
                {
                    "workload_id": workload_id,
                    "resource_name": resource["name"],
                    "resource_group": resource["resource_group"],
                    "subscription_id": resource["subscription_id"],
                    "kind": resource["kind"],
                    "deployment_name": row["deployment_name"],
                    "model_name": row["model_name"],
                    "lookback_hours": self.lookback_hours,
                    "request_count": total_requests,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "avg_latency_ms": avg_latency_ms,
                    "last_activity_at": last_activity_at,
                    "hourly_points": points,
                    "refreshed_at": datetime.now(timezone.utc).isoformat(),
                }
            )

        return rows

    def _query_metric_set(
        self,
        token: str,
        resource: dict,
        *,
        metric_names: tuple[str, ...],
        aggregation: str,
    ) -> dict | None:
        for metric_name in metric_names:
            url = self._metrics_url(
                resource_id=resource["id"],
                metric_names=metric_name,
                aggregation=aggregation,
            )
            try:
                return self._arm_get(url, token)
            except HTTPError as exc:
                if exc.code in {400, 404}:
                    continue
                raise
        return None

    def _metrics_url(self, *, resource_id: str, metric_names: str, aggregation: str) -> str:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=self.lookback_hours)
        timespan = start_time.strftime("%Y-%m-%dT%H:%M:%SZ") + "/" + end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        metric_filter = quote("ModelDeploymentName eq '*' and ModelName eq '*'", safe="=$'()*,:/")
        return (
            _ARM_BASE
            + resource_id
            + "/providers/microsoft.insights/metrics"
            + "?api-version=2021-05-01"
            + "&metricnames=" + metric_names
            + "&timespan=" + timespan
            + "&aggregation=" + aggregation
            + "&interval=PT1H"
            + "&$filter=" + metric_filter
            + "&top=200"
        )

    def _merge_metric_response(self, deployments: dict[tuple[str, str], dict], data: dict) -> None:
        for metric in data.get("value", []):
            metric_name = metric.get("name", {}).get("value", "")
            for series in metric.get("timeseries", []):
                metadata = self._series_metadata(series)
                deployment_name = metadata.get("modeldeploymentname") or "unknown-deployment"
                model_name = metadata.get("modelname") or deployment_name
                key = (deployment_name, model_name)
                row = deployments.setdefault(
                    key,
                    {
                        "deployment_name": deployment_name,
                        "model_name": model_name,
                        "hourly_points": {},
                    },
                )
                for point in series.get("data", []):
                    timestamp = point.get("timeStamp", "")
                    bucket = row["hourly_points"].setdefault(
                        timestamp,
                        {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "request_count": 0,
                            "latency_ms_sum": 0.0,
                            "latency_ms_count": 0,
                        },
                    )
                    if metric_name in {"InputTokens", "ProcessedPromptTokens"}:
                        bucket["prompt_tokens"] += int(point.get("total", 0) or 0)
                    elif metric_name in {"OutputTokens", "GeneratedCompletionTokens"}:
                        bucket["completion_tokens"] += int(point.get("total", 0) or 0)
                    elif metric_name in {"TotalTokens", "TokenTransaction"}:
                        bucket["total_tokens"] += int(point.get("total", 0) or 0)
                    elif metric_name in {"AzureOpenAIRequests", "ModelRequests"}:
                        bucket["request_count"] += int(point.get("total", 0) or 0)
                    elif metric_name == "TimeToResponse":
                        avg = float(point.get("average", 0.0) or 0.0)
                        if avg > 0:
                            bucket["latency_ms_sum"] += avg
                            bucket["latency_ms_count"] += 1

    def _resource_summary(self, resource: dict, deployments: list[dict]) -> dict:
        return {
            "resource_name": resource["name"],
            "resource_group": resource["resource_group"],
            "subscription_id": resource["subscription_id"],
            "kind": resource["kind"],
            "lookback_hours": self.lookback_hours,
            "deployment_count": len(deployments),
            "request_count": sum(row["request_count"] for row in deployments),
            "prompt_tokens": sum(row["prompt_tokens"] for row in deployments),
            "completion_tokens": sum(row["completion_tokens"] for row in deployments),
            "total_tokens": sum(row["total_tokens"] for row in deployments),
            "avg_latency_ms": round(
                sum(row["avg_latency_ms"] for row in deployments if row["avg_latency_ms"] > 0)
                / max(1, sum(1 for row in deployments if row["avg_latency_ms"] > 0)),
                2,
            ),
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _build_events(self, resource: dict, deployments: list[dict]) -> list[dict]:
        events: list[dict] = []
        for row in deployments:
            for point in row["hourly_points"]:
                event_key = f"{row['workload_id']}|{point['timestamp']}"
                if event_key in self._emitted_event_keys:
                    continue
                if point["total_tokens"] <= 0 and point["request_count"] <= 0:
                    continue
                self._emitted_event_keys.add(event_key)
                events.append(
                    {
                        "workload_id": row["workload_id"],
                        "model_name": row["model_name"],
                        "request_count": max(1, int(point["request_count"] or 1)),
                        "tokens_input": int(point["prompt_tokens"]),
                        "tokens_output": int(point["completion_tokens"]),
                        "latency_ms": float(point["latency_ms"] or 0.0),
                        "slo_latency_ms": self.slo_latency_ms,
                        "is_critical": False,
                        "request_text": f"Azure Monitor telemetry for {resource['name']}/{row['deployment_name']}",
                        "output_text": "azure-monitor-ingested",
                        "success_override": True,
                    }
                )
        return sorted(events, key=lambda event: event["request_text"])

    def _series_metadata(self, series: dict) -> dict[str, str]:
        return {
            entry.get("name", {}).get("value", "").lower(): entry.get("value", "")
            for entry in series.get("metadatavalues", [])
        }

    def _workload_id(self, resource_name: str, deployment_name: str) -> str:
        raw = f"{resource_name}-{deployment_name}".lower()
        sanitized = "".join(ch if ch.isalnum() else "-" for ch in raw)
        while "--" in sanitized:
            sanitized = sanitized.replace("--", "-")
        return f"workload-{sanitized.strip('-')}"

    def _arm_get(self, url: str, token: str) -> dict:
        req = Request(url, headers={"Authorization": "Bearer " + token})
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
