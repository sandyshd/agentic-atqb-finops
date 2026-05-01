from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentic_finops.adapters.openai_metrics import AzureOpenAIMetricsAdapter
from agentic_finops.adapters.registry import get_adapters
from agentic_finops.config import settings
from agentic_finops.focus import FocusRow, cost_signal_to_focus
from agentic_finops.mag.orchestrator import MAGOrchestrator
from agentic_finops.mag.quote_adapters import ProviderQuoteManager
from agentic_finops.models import ActionRecord, CostSignal
from agentic_finops.orchestration.engine import Orchestrator
from agentic_finops.simulator.generator import SyntheticSignalGenerator
from agentic_finops.simulator.inference_generator import InferenceEventGenerator
from agentic_finops.store.memory_store import MemoryStore

# ── FinOps cost-signal pipeline (existing) ────────────────────────────────
store        = MemoryStore()
orchestrator = Orchestrator(store=store)
synthetic    = SyntheticSignalGenerator(seed=42)
adapters     = get_adapters()
openai_metrics = AzureOpenAIMetricsAdapter()

# ── MAG / UCI / ATQB pipeline (new) ──────────────────────────────────────
mag       = MAGOrchestrator()
infer_gen = InferenceEventGenerator(seed=17)
_mag_event_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=2000)
quote_manager = ProviderQuoteManager()

# Track the last adapter batch so cached signals are not re-processed each tick
_last_adapter_signals: list[CostSignal] = []


def _static_dir() -> Path:
    return Path(__file__).parent / "static"


async def _ingest_loop() -> None:
    """Cost-signal ingest: adapters first, optional synthetic fallback.

    Adapter signals are only processed when the batch differs from the previous
    one, preventing cached results from inflating event/action counts on every
    loop tick.
    """
    global _last_adapter_signals
    while True:
        try:
            adapter_signals: list[CostSignal] = []
            for adapter in adapters:
                adapter_signals.extend(adapter.collect())
            if adapter_signals:
                if adapter_signals != _last_adapter_signals:
                    orchestrator.process_batch(adapter_signals)
                    _last_adapter_signals = list(adapter_signals)
            elif settings.enable_cost_signal_simulation:
                orchestrator.process(synthetic.next())
        except Exception:
            pass
        await asyncio.sleep(settings.ingest_interval_seconds)


async def _mag_source_loop() -> None:
    """Collect live inference events and enqueue them for immediate MAG processing."""
    while True:
        try:
            live_events = openai_metrics.collect_inference_events()
            for event in live_events:
                if _mag_event_queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        _mag_event_queue.get_nowait()
                _mag_event_queue.put_nowait(event)
        except Exception:
            pass
        await asyncio.sleep(max(1, settings.ingest_interval_seconds // 4))


async def _mag_consumer_loop() -> None:
    """Process queued MAG events continuously; fallback to simulation when idle."""
    while True:
        try:
            event = await asyncio.wait_for(_mag_event_queue.get(), timeout=1.0)
            mag.process_inference(**event)
            continue
        except asyncio.TimeoutError:
            if settings.enable_mag_simulation:
                with contextlib.suppress(Exception):
                    mag.process_inference(**infer_gen.next())
        except Exception:
            pass
        await asyncio.sleep(0.05)


async def _broker_quote_loop() -> None:
    """Continuously refresh provider market quotes from real adapters."""
    while True:
        try:
            for quote in quote_manager.collect():
                mag.broker.update_market_quote(
                    quote.provider,
                    spot_multiplier=quote.spot_multiplier,
                    unit_price_per_gpu_hour=quote.unit_price_per_gpu_hour,
                    egress_per_1k_tokens=quote.egress_per_1k_tokens,
                    latency_ms=quote.latency_ms,
                    source=quote.source,
                )
        except Exception:
            pass
        await asyncio.sleep(max(5, settings.broker_quote_refresh_seconds))


@asynccontextmanager
async def lifespan(_: FastAPI):
    t1 = asyncio.create_task(_ingest_loop())
    t2 = asyncio.create_task(_mag_source_loop())
    t3 = asyncio.create_task(_mag_consumer_loop())
    t4 = asyncio.create_task(_broker_quote_loop())
    try:
        yield
    finally:
        t1.cancel()
        t2.cancel()
        t3.cancel()
        t4.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(t1, t2, t3, t4, return_exceptions=True)


app = FastAPI(title="Agentic FinOps", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(_static_dir())), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_static_dir() / "index.html")


@app.get("/pro")
def pro_dashboard() -> FileResponse:
    return FileResponse(_static_dir() / "pro-dashboard.html")


# ── FinOps signal endpoints ───────────────────────────────────────────────

@app.get("/api/events", response_model=list[CostSignal])
def events(limit: int = 25) -> list[CostSignal]:
    return store.latest_events(limit=limit)


@app.get("/api/actions", response_model=list[ActionRecord])
def actions(limit: int = 25) -> list[ActionRecord]:
    return store.latest_actions(limit=limit)


@app.get("/api/metrics")
def metrics() -> dict[str, float | int]:
    return store.metrics()


# ── FOCUS v1.0 endpoint ───────────────────────────────────────────────────

@app.get("/api/focus/rows", response_model=list[FocusRow])
def focus_rows(limit: int = 100) -> list[FocusRow]:
    """Return cost telemetry normalized to the FOCUS v1.0 schema.

    Combines:
      1. FOCUS rows derived from cached cost signals stored in the in-memory
         signal store (covers all adapter sources via `CostSignal`).
      2. Adapter-native FOCUS rows (currently Azure) when an adapter exposes
         `latest_focus_rows()`.
    """
    rows: list[FocusRow] = []

    for signal in store.latest_events(limit=limit):
        try:
            rows.append(cost_signal_to_focus(signal))
        except Exception:
            continue

    for adapter in adapters:
        provider = getattr(adapter, "latest_focus_rows", None)
        if callable(provider):
            try:
                rows.extend(provider())
            except Exception:
                continue

    # De-duplicate by (ResourceId, ChargePeriodStart) preserving first occurrence
    seen: set[tuple[str | None, str]] = set()
    unique_rows: list[FocusRow] = []
    for row in rows:
        key = (row.ResourceId, row.ChargePeriodStart)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    return unique_rows[: max(1, limit)]


# ── UCI endpoints ─────────────────────────────────────────────────────────

@app.get("/api/uci")
def uci(limit: int = 20) -> list[dict]:
    return [r.model_dump() for r in mag.auditor.latest_uci_records(limit=limit)]


# ── ATQB endpoints ────────────────────────────────────────────────────────

@app.get("/api/atqb/decisions")
def atqb_decisions(limit: int = 20) -> list[dict]:
    return [d.model_dump() for d in mag.governor.latest_decisions(limit=limit)]


@app.get("/api/atqb/ledgers")
def atqb_ledgers(workload_id: str | None = None) -> list[dict]:
    ledgers = mag.governor.ledger_statuses()
    if workload_id:
        ledgers = [l for l in ledgers if l["workload_id"] == workload_id]
    return ledgers


@app.get("/api/atqb/transfers")
def atqb_transfers(limit: int = 50) -> list[dict]:
    return [t.model_dump() for t in mag.governor.latest_transfers(limit=limit)]


@app.get("/api/atqb/rate-limits")
def atqb_rate_limits() -> list[dict]:
    return [s.model_dump() for s in mag.governor.rate_limit_signals.values()]


# ── MAG endpoints ─────────────────────────────────────────────────────────

@app.get("/api/mag/status")
def mag_status() -> dict:
    status = mag.status()
    if not settings.enable_broker_baseline_quotes and status.get("total_requests_processed", 0) == 0:
        status["provider_uci"] = {}
    return status


@app.get("/api/broker/quotes")
def broker_quotes() -> list[dict]:
    return mag.broker.market_snapshot()


@app.get("/api/mag/results")
def mag_results(limit: int = 20) -> list[dict]:
    return mag.latest_results(limit=limit)


@app.get("/api/mag/sequence-traces")
def mag_sequence_traces(limit: int = 20) -> list[dict]:
    return mag.latest_sequence_traces(limit=limit)


@app.get("/api/mag/scaler")
def mag_scaler(limit: int = 30) -> list[dict]:
    """Latest Scaler Agent recommendations (forecast + cost delta)."""
    from dataclasses import asdict

    return [asdict(r) for r in mag.scaler.recent(limit=limit)]


# ── Azure OpenAI / AI Services token consumption ─────────────────────────

@app.get("/api/openai/usage")
def openai_usage(days: int = 7) -> list[dict]:
    """Real token consumption from Azure Monitor for all OpenAI/AI Services resources.

    Args:
        days: Number of days of history to return (default 7, max 90).
    """
    from datetime import datetime, timedelta, timezone

    records = openai_metrics.collect()
    cutoff = datetime.now(timezone.utc) - timedelta(days=min(days, 90))
    return [
        r for r in records
        if not r.get("timestamp") or datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) >= cutoff
    ]


@app.get("/api/openai/deployments")
def openai_deployments() -> list[dict]:
    """Real token, request, and latency breakdown by Azure OpenAI deployment."""
    return openai_metrics.deployment_breakdown()


@app.get("/api/openai/token-history")
def openai_token_history(days: int = 7) -> dict:
    """Per-deployment hourly token/request/latency timeseries for charting.

    Args:
        days: Number of days of history to return (default 7, max 90).

    Returns::

        {
          "labels": ["2026-04-09T14:00:00Z", ...],   # sorted hourly timestamps
          "series": [
            {
              "workload_id": "workload-santechlab-gpt-4o",
              "deployment_name": "gpt-4o",
              "resource_name": "santechlab",
              "model_name": "gpt-4o",
              "total_tokens":   [0, 1234, ...],
              "request_count":  [0,    3, ...],
              "avg_latency_ms": [0.0, 861.0, ...],
            },
            ...
          ]
        }
    """
    from datetime import datetime, timedelta, timezone

    rows = openai_metrics.deployment_breakdown()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min(days, 90))).isoformat()
    all_ts: set[str] = set()
    for row in rows:
        for pt in row.get("hourly_points", []):
            ts = pt.get("timestamp", "")
            if ts and ts >= cutoff:
                all_ts.add(ts)

    labels = sorted(all_ts)
    if not labels:
        return {"labels": [], "series": []}

    ts_index = {ts: i for i, ts in enumerate(labels)}
    series = []
    for row in rows:
        n = len(labels)
        total_tokens: list[int] = [0] * n
        request_count: list[int] = [0] * n
        avg_latency_ms: list[float] = [0.0] * n
        for pt in row.get("hourly_points", []):
            idx = ts_index.get(pt.get("timestamp", ""))
            if idx is not None:
                total_tokens[idx] = int(pt.get("total_tokens", 0) or 0)
                request_count[idx] = int(pt.get("request_count", 0) or 0)
                avg_latency_ms[idx] = float(pt.get("latency_ms", 0.0) or 0.0)
        series.append(
            {
                "workload_id": row["workload_id"],
                "deployment_name": row["deployment_name"],
                "resource_name": row["resource_name"],
                "model_name": row["model_name"],
                "total_tokens": total_tokens,
                "request_count": request_count,
                "avg_latency_ms": avg_latency_ms,
            }
        )

    return {"labels": labels, "series": series}
