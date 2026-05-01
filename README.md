# Agentic FinOps — Multi-Agent GPU Token & Cost Governance

**Autonomous framework for real-time governance of large language model (LLM) costs in multi-cloud architectures.**

Agentic FinOps implements **Multi-Agent Governance (MAG)** and **Autonomous Token-Quota Balancing (ATQB)** to autonomously manage GPU token budgets, optimize model selection, and enforce cost constraints without human intervention. It intelligently routes inference requests across cloud providers, triggers quota borrowing under rate-limit pressure, measures model quality metrics, and provides real-time observability into governance decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Use Cases](#use-cases)
3. [Core Architecture](#core-architecture)
4. [Key Features](#key-features)
5. [Project Structure](#project-structure)
6. [API Endpoints](#api-endpoints)
7. [Configuration](#configuration)
8. [Quick Start](#quick-start)
9. [Dashboard](#dashboard)
10. [Testing](#testing)

---

## Overview

### What is Agentic FinOps?

Agentic FinOps is a **governance platform** that autonomously manages generative AI costs by:

- **Tracking token budgets** per workload with borrowing/lending mechanics
- **Measuring model quality** (success rates, latency, cost) to drive intelligent downgrade decisions
- **Monitoring rate limits** from cloud providers and triggering quota rebalancing
- **Selecting providers** based on real-time spot pricing (Azure Retail Prices, AWS Spot, NeoCloud)
- **Enforcing hard limits** via token throttling, model downgrade, and quota shifting
- **Operating autonomously** without human approval for cost-containment actions

### Research Foundation

This implementation is based on two research algorithms:

1. **Multi-Agent Governance (MAG)**: A 6-agent orchestration system (Gatekeeper → Governor → Broker → Auditor → Critic → Scaler) that coordinates real-time cost control decisions.
2. **Autonomous Token-Quota Balancing (ATQB)**: A hierarchical decision ladder that enforces token budgets, adapts to rate-limit signals, and optimizes model tier selection based on measured success probability and cost utility.

---

## Use Cases

### 1. **Cost Containment for Multi-Tenant LLM Platforms**
   - Enforce per-team token budgets
   - Prevent runaway batch jobs from consuming entire monthly budgets
   - Automatically throttle or downgrade model tiers when approaching limits

### 2. **Rate-Limit Handling Under Workload Spikes**
   - Detect 429 responses from cloud providers
   - Dynamically shift high-priority traffic to low-pressure providers
   - Borrow tokens from idle teams to prioritize critical work

### 3. **Multi-Cloud Provider Selection**
   - Compare spot prices across Azure, AWS, NeoCloud in real-time
   - Route inferences to the lowest-cost provider at request time
   - Handle provider fallback gracefully when APIs unavailable

### 4. **Model Quality Optimization**
   - Track success rates per model tier (e.g., GPT-4 vs LLaMA)
   - Measure cost-per-successful-completion
   - Auto-downgrade to cheaper models if success rate maintained and utility improves

### 5. **Operational Observability**
   - Real-time dashboard showing quota transfers, rate-limit pressure, provider health
   - Audit trail of all governance decisions with full 12-step orchestration traces
   - KPI tracking: token velocity, cost trajectory, workload pressure trends

---

## Core Architecture

### Multi-Agent Governance (MAG) — 6-Agent Orchestra

The orchestrator sequences decisions through six specialized agents:

```
Workload Request
    ↓ [Step 1]
┌─────────────────────────────────────────────────────────┐
│ GATEKEEPER AGENT: Intent Classification & TPW Assignment │
│  • Routes request by intent (chat, batch, fine-tune)     │
│  • Assigns Task Priority Weight (TPW) ∈ [0.45, 0.95]   │
│  • 0.95 TPW for critical=true (SLA-protected)            │
└─────────────────────────────────────────────────────────┘
    ↓ [Step 2]
┌──────────────────────────────────────────────────────────┐
│ GOVERNOR AGENT: ATQB Enforcement & Quota Transfers       │
│  • Computes token utilization (TPW)                      │
│  • Injects rate-limit pressure signals into decision     │
│  • Triggers quota borrowing (quota_shift) from donors    │
│  • Maintains transfer ledger & borrow/lend limits        │
└──────────────────────────────────────────────────────────┘
    ↓ [Step 3]
┌──────────────────────────────────────────────────────────┐
│ INFERENCE EXECUTION: Model Gateway (OpenAI/Ollama)       │
│  • Execute inference on recommended model tier           │
│  • Capture latency, tokens, success signal               │
└──────────────────────────────────────────────────────────┘
    ↓ [Step 4]
┌──────────────────────────────────────────────────────────┐
│ CRITIC AGENT: Quality Validation                         │
│  • Validate response quality and success                 │
│  • Detect hallucinations, truncations, errors            │
│  • Compute success_override for cold-start models        │
└──────────────────────────────────────────────────────────┘
    ↓ [Step 5]
┌──────────────────────────────────────────────────────────┐
│ AUDITOR AGENT: Usage Cost Calculation & Audit Logging    │
│  • Compute Utility Cost Index (UCI) per inference        │
│  • Update workload token/cost ledgers                    │
│  • Record audit event with full metadata                 │
└──────────────────────────────────────────────────────────┘
    ↓ [Step 6]
┌──────────────────────────────────────────────────────────┐
│ BROKER AGENT: Provider Selection & Market Data           │
│  • Query live spot pricing (Azure, AWS, NeoCloud)        │
│  • Recommend lowest-UCI provider for next inference      │
│  • Track market quotes with freshness and source         │
└──────────────────────────────────────────────────────────┘
    ↓ [Step 7]
┌──────────────────────────────────────────────────────────┐
│ SCALER AGENT: GPU Capacity Recommendations               │
│  • EWMA forecast of token demand per workload (α=0.4)    │
│  • Pressure = forecast / capacity (replicas × 50k tok)   │
│  • Recommends scale_up at ≥0.85 or latency ≥1.25× SLO    │
│  • Recommends scale_down at <0.30 (min 1 replica)        │
│  • Cost delta in USD using UCI per-token, fed back into  │
│    Governor reasoning + Optimizer side-action            │
└──────────────────────────────────────────────────────────┘
    ↓ [Step 8-12]
Policy Guardrails → Optimization Executor → Gatekeeper → Workload Response
```

### Autonomous Token-Quota Balancing (ATQB) — Decision Ladder

The Governor executes a hierarchical decision ladder (7 rules, first-match-wins):

| Rule | Condition | Action | Rationale |
|------|-----------|--------|-----------|
| 1 | Hard-stop: token_util ≥ 100% OR budget_util ≥ 100% | `hard_stop` | Kill-switch: HLB enforced |
| 2a | Rate-limit pressure ≥ 0.60 AND TPW ≥ 0.80 AND budget < 90% | `quota_shift` | High-priority under pressure → borrow capacity |
| 2b | Rate-limit pressure ≥ 0.70 AND TPW < 0.80 | `throttle` | Low-priority under pressure → reduce request rate |
| 3 | TPW ≥ 0.80 AND budget < 90% | `allow` | High-priority → allow if room; expect borrow |
| 4 | Measured MUA favorable (utility_candidate ≥ 1.03 × utility_current) AND success_candidate ≥ 80% | `downgrade_model` | Cheaper model still succeeds → save cost |
| 5 | Budget ≥ 80% OR token ≥ 85% | `throttle` | Approaching limit → reduce velocity |
| 6 | Budget ≥ 70% OR token ≥ 70% | `quota_shift` | Reserve capacity → signal for borrowing |
| 7 | (else) | `allow` | Default: allow unless constrained |

### Quota Transfer Engine

When `quota_shift` is triggered and tokens are needed:

1. **Donor Selection**: Sort workloads by (TPW ascending, lendable_tokens descending)
2. **Transfer**: Move tokens from low-TPW donors to high-TPW requester
3. **Ledger Update**: Mutate `borrowed_tokens`, `lent_tokens`, update `effective_token_budget`
4. **History**: Append `QuotaTransfer` record (from, to, tokens, reason, timestamp) to rolling deque (max 1000)

### Rate-Limit Pressure Scoring

Rate-limit pressure = (429_count / 10.0) × 0.7 + (retry_after_ms_p95 / 5000.0) × 0.3, clamped to [0.0, 1.0]

- **0.0**: No pressure
- **0.4–0.7**: Moderate pressure → throttle low-TPW; consider quota_shift for high-TPW
- **≥ 0.8**: Severe pressure → aggressive quota_shift or hard_stop

### Model Quality Measurement & MUA Downgrade

**Model Utility Analysis (MUA)** tracks per-workload-model-tier:

- **Success Rate**: Bayesian posterior P_success = (successes + α) / (attempts + α + β), α=β=2
- **Cost**: Average cost per inference attempt
- **Utility**: utility = P_success / cost_per_call
- **Downgrade Rule**: Downgrade to candidate tier if:
  - P_candidate_success ≥ 0.80 (success floor maintained)
  - AND utility_candidate ≥ 1.03 × utility_current (utility improves by ≥3%)

### Provider Quote Adapters

Real-time market data from three sources:

1. **Azure Retail Prices API**: Queries spot SKUs; fallback to any GPU VM if no spot available
2. **AWS EC2 Spot Pricing**: boto3 `describe_spot_price_history` (requires AWS credentials & region config)
3. **NeoCloud Custom Endpoint**: HTTP GET returning JSON {price, spot_multiplier, egress, latency}

Refresh cadence: ~45 seconds (configurable via `BROKER_QUOTE_REFRESH_SECONDS`)

---

## Key Features

✅ **Autonomous Governance**
- No human approval needed for cost-containment actions
- Fully deterministic decision rules with audit trails

✅ **Multi-Cloud & Multi-Model**
- Unified token/cost model across Azure OpenAI, AWS, NeoCloud
- Dynamic model tier downgrade based on measured utility

✅ **Real-Time Rate-Limit Resilience**
- Ingests 429 signals and retry-after headers
- Automatically rebalances traffic across providers and workload priorities

✅ **Live Spot Pricing Integration**
- Azure Retail Prices API for spot GPU availability
- AWS Spot History for price discovery
- Broker selects lowest-cost provider per request

✅ **Quota Borrowing & Lending**
- Cross-workload token transfers with donor prioritization
- Borrow/lend limits enforced; transfer history audited

✅ **Observability Dashboard**
- Real-time KPI cards: transfer count, tokens shifted, workload pressure, rate-limit bars
- Transfer table showing cross-workload token movements
- Provider health badges (LIVE vs FALLBACK)
- 12-step orchestration traces per inference

✅ **Bayesian Model Quality**
- Cold-start-safe success probability estimation
- Cost-per-successful-completion optimization

✅ **Streaming Control Plane**
- Queue-based MAG processing replaces polling
- Sub-second decision latency

---

## Project Structure

```
AgenticFinOps/
├── src/agentic_finops/
│   ├── api/
│   │   ├── app.py                   # FastAPI endpoint definitions
│   │   ├── static/
│   │   │   ├── index.html           # Real-time dashboard
│   │   │   └── style.css
│   │   └── __init__.py
│   │
│   ├── atqb/
│   │   ├── models.py                # ATQB data types (ActuatorAction, QuotaLedger, ATQBDecision)
│   │   ├── controller.py            # ATQB decision ladder (7 rules)
│   │   └── __init__.py
│   │
│   ├── mag/
│   │   ├── agents.py                # 6 agents (Gatekeeper, Governor, Broker, Auditor, Critic, Scaler)
│   │   ├── orchestrator.py          # MAG sequence orchestration
│   │   ├── quote_adapters.py        # Provider quote collectors (Azure, AWS, NeoCloud)
│   │   ├── guardrails.py            # Policy enforcement layer
│   │   └── optimizer.py             # Execution engine
│   │
│   ├── uci/
│   │   ├── calculator.py            # Utility Cost Index (UCI) computation
│   │   ├── models.py                # InferenceRecord, UCIRecord
│   │   └── __init__.py
│   │
│   ├── config.py                    # Environment-based configuration
│   ├── models.py                    # Common data types (RiskLevel, ActionType, CostSignal)
│   ├── main.py                      # App entry point
│   └── __init__.py
│
├── tests/
│   ├── test_atqb.py                 # ATQB controller tests (downgrade, quota_shift, throttle)
│   ├── test_mag.py                  # MAG orchestration tests
│   ├── test_uci.py                  # UCI calculation tests
│   ├── test_focus.py                # FOCUS v1.0 schema + converter tests
│   ├── test_scaler.py               # Scaler EWMA forecast + cost-delta tests
│   └── __init__.py
│
├── .env.example                     # Configuration template
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── Agentic_FinOps_IEEE_Formatted.docx # Research paper reference
└── .venv/                           # Virtual environment

```

---

## API Endpoints

### Status & Configuration

- `GET /` — Dashboard HTML page
- `GET /api/metrics` — System health metrics

### ATQB Governance

- `GET /api/atqb/decisions?limit=50` — Latest ATQB decisions (action, TPW, budget%, token%)
- `GET /api/atqb/ledgers?workload_id=...` — Quota ledger per workload (budget, borrowed, lent, TPW)
- `GET /api/atqb/transfers?limit=50` — Quota transfer history (from, to, tokens, reason)
- `GET /api/atqb/rate-limits` — Current rate-limit pressure per workload (429s, retry_p95, pressure score)

### Multi-Agent Governance

- `GET /api/mag/status` — Current MAG state (6 agents, ledgers, latest rules)
- `GET /api/mag/results?limit=50` — Inference results with governance decisions
- `GET /api/mag/sequence-traces?limit=50` — Full 12-step orchestration traces
- `GET /api/mag/scaler?limit=30` — Scaler Agent recommendations (forecast tok/window, capacity, pressure ratio, replicas current → recommended, projected cost delta, reasoning)

### Provider & Cost

- `GET /api/broker/quotes` — Live spot prices per provider (source, freshness, spot_multiplier)
- `GET /api/openai/usage?days=7` — OpenAI token usage history
- `GET /api/openai/deployments` — Available model deployments
- `GET /api/openai/token-history?days=7` — Token burn rate over time
- `GET /api/uci` — Latest UCI computations per provider

### Cost Events & Actions (Original Adapters)

- `GET /api/events?limit=25` — Cost signal events
- `GET /api/actions?limit=25` — Recommended actions (with approval state)

### FOCUS v1.0 Compatibility

- `GET /api/focus/rows?limit=100` — Cost telemetry normalized to the
  [FinOps Open Cost and Usage Specification (FOCUS) v1.0](https://focus.finops.org/)
  schema. The endpoint converts the platform's internal `CostSignal` records
  into FOCUS rows containing the v1.0 mandatory columns:

  | Group | Columns |
  | --- | --- |
  | Identifiers | `BillingAccountId`, `BillingAccountName`, `SubAccountId`, `SubAccountName`, `ResourceId`, `ResourceName`, `RegionId`, `RegionName` |
  | Provider | `Provider`, `Publisher`, `InvoiceIssuer` |
  | Time | `ChargePeriodStart`, `ChargePeriodEnd`, `BillingPeriodStart`, `BillingPeriodEnd` |
  | Service | `ServiceName`, `ServiceCategory` |
  | Cost | `BilledCost`, `EffectiveCost`, `ListCost`, `ContractedCost` |
  | Pricing | `ListUnitPrice`, `ContractedUnitPrice`, `PricingCategory`, `PricingQuantity`, `PricingUnit` |
  | Usage | `ConsumedQuantity`, `ConsumedUnit` |
  | Charge | `ChargeCategory`, `ChargeClass`, `ChargeFrequency`, `ChargeDescription` |
  | Other | `BillingCurrency`, `Tags` |

  Provider attribution is mapped automatically (Azure → `Microsoft`, AWS →
  `AWS`, GCP → `Google`). Service-category mapping covers Compute, Storage,
  Databases, Networking, AI and Machine Learning, Analytics, and Identity
  per the FOCUS allowed-values list, with `Other` as fallback. Currency is
  USD; commitment-discount fields are populated when contracted pricing is
  applied.

---

## Configuration

Create a `.env` file from `.env.example`:

```bash
# Core governance
FINOPS_ENV=demo
ENABLE_MAG_SIMULATION=false          # Use real adapters vs. synthetic inference
ACTION_MODE=guardrailed-auto        # Autonomous decision execution

# Token ingest
INGEST_INTERVAL_SECONDS=20

# ATQB thresholds (optional; defaults in controller.py)
# BUDGET_HARD_STOP_PCT=1.0
# BUDGET_DOWNGRADE_PCT=0.90
# MODEL_DOWNGRADE_MAP={"claude-4":"llama-4"}

# Azure authentication (for live cost signals)
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_TENANT_ID=<your-tenant-id>
AZURE_ADAPTER_ENABLED=true
AZURE_COST_LOOKBACK_HOURS=72
AZURE_OPENAI_LOOKBACK_HOURS=24      # Hours of token-history pulled from Azure Monitor
                                    # (must be >= window shown by Token Burn Rate chart)

# Broker quote settings
BROKER_QUOTE_REFRESH_SECONDS=45     # Refresh provider quotes every 45s
BROKER_AZURE_REGION=eastus          # Azure region for spot queries

# AWS (optional; requires credentials)
BROKER_AWS_REGION=us-west-2
BROKER_AWS_INSTANCE_TYPE=g4dn.xlarge

# NeoCloud (optional; requires external endpoint)
BROKER_NEOCLOUD_QUOTE_URL=          # e.g., https://quotes.neocloud.ai/spot
```

---

## Quick Start

### 1. Clone & Setup

```powershell
cd c:\Project\AgenticFinOps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```powershell
# Copy .env.example to .env and set your Azure subscription
cp .env.example .env
# Edit .env with your settings
```

### 3. Run

```powershell
$env:PYTHONPATH = "src"
uvicorn agentic_finops.main:app --reload --port 8000
```

### 4. Access Dashboard

Open browser: **http://127.0.0.1:8000**

---

## Dashboard

The real-time dashboard displays:

### Top Row: KPI Cards
- **Transfer Count**: Cross-workload quota transfers in last 1 hour
- **Transfer Tokens**: Total tokens shifted in last 1 hour
- **Workloads Under Pressure**: Count of workloads with rate-limit pressure ≥ 0.4
- **Max Rate Pressure**: Highest pressure score (0–1 scale)

### Spot Market Section
- Provider cards showing UCI, spot multiplier, freshness, source
- **LIVE** badge (green) for real adapters; **FALLBACK** badge (orange) for bootstrap pricing

### Recent Transfer Table
- From → To workload, tokens shifted, reason

### Rate-Limit Pressure Table
- Workload, provider, 429 count, retry-after p95, pressure score (sorted DESC)
- Visual bar chart per row showing pressure intensity

### Inference Metrics
- Token burn rate (tokens/sec)
- Cost trajectory (USD/hour estimated)
- Model distribution pie chart

---

## Testing

```powershell
# Run all tests
$env:PYTHONPATH = "src"
python -m pytest -v

# Specific test suite
python -m pytest tests/test_atqb.py -v       # ATQB decision ladder
python -m pytest tests/test_mag.py -v        # MAG orchestration
python -m pytest tests/test_uci.py -v        # UCI calculations
python -m pytest tests/test_focus.py -v      # FOCUS v1.0 mapping
python -m pytest tests/test_scaler.py -v     # Scaler forecast + cost delta

# Test coverage
python -m pytest --cov=src.agentic_finops tests/
```

### Test Coverage

- **ATQB**: 7 decision rules, rate-limit handling, MUA downgrade, quota transfers
- **MAG**: 12-step orchestration traces, agent ordering, audit ledger
- **UCI**: Provider pricing, spot multipliers, egress cost
- **Guardrails**: Hard-limit enforcement, approval workflows
- **FOCUS v1.0**: 29 mandatory columns, billed/effective cost math, contracted-discount handling, Azure/AWS provider mapping, service-category mapping, calendar-month billing period, JSON serialization
- **Scaler**: EWMA build-up, no_change / scale_up (pressure & latency) / scale_down branches, replica state commit on apply, cost-delta scaling with UCI per-token, negative-input clamping, legacy string API

**Current Status**: 35+ tests passing (100% pass rate)

---

## Architecture Diagram (Text)

```
                    ┌──────────────────────┐
                    │   Workload Request   │
                    │  (tokens, model)     │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │    GATEKEEPER       │
                    │  Intent→TPW,Route   │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼────────────────────┐
           │                   │                    │
    ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
    │  GOVERNOR   │     │   AUDITOR   │     │   BROKER    │
    │ATQB+Quota   │     │ UCI + Cost  │     │Spot Pricing │
    │Transfers    │     │   Tracking  │     │  Selection  │
    └─────┬───────┘     └─────┬───────┘     └──────┬──────┘
          │                   │                    │
          ├───────────────────┼────────────────────┤
          │                   │                    │
          │ ┌──────────────────▼──────────────────┐│
          │ │  [Inference Execution]              ││
          │ │  OpenAI/Ollama Model Gateway        ││
          │ │  (tokens, latency, success)         ││
          │ └───────────────┬──────────────────────┘│
          │                 │                        │
          ├─────────────────┼────────────────────────┤
          │                 │                        │
    ┌─────▼─────┐    ┌──────▼──────┐    ┌───────────▼─┐
    │  CRITIC    │    │   AUDITOR   │    │   SCALER    │
    │ Validate   │    │  Log+Ledger │    │Forecast &   │
    │ Quality    │    │  Update     │    │  Recommend  │
    └───────────┬┘    └──────┬──────┘    └───────┬─────┘
                │             │                  │
                └─────────────┴──────────────────┘
                             │
                ┌────────────▼────────────┐
                │ Policy Guardrails       │
                │ (HLB, Approval, Risk)   │
                └────────────┬────────────┘
                             │
                ┌────────────▼────────────┐
                │ Optimization Executor   │
                │ (Action Provisioning)   │
                └────────────┬────────────┘
                             │
                    ┌────────▼────────┐
                    │  Response Sent  │
                    │ to Workload     │
                    └─────────────────┘
```

---

## Next Steps & Roadmap

- [ ] **AWS Adapter Activation**: Set `BROKER_AWS_REGION` + credentials
- [ ] **NeoCloud Provider**: Configure `BROKER_NEOCLOUD_QUOTE_URL` for custom market
- [ ] **Provider Status Webhooks**: Notify ops of provider outages or price spikes
- [ ] **Approval Workflow API**: Add `/api/actions/{id}/approve` for high-risk decisions
- [ ] **Audit Persistence**: Land 12-step traces + ledger snapshots to durable storage (Cosmos DB / PostgreSQL)
- [ ] **Alert Thresholds**: Configurable per-workload alerts for pressure, transfers, cost anomalies
- [ ] **Rate-Limit Auto-Detection**: Integrate Azure Monitor response headers / Application Insights
- [ ] **Cost Attribution**: Per-team billing module with chargeback calculations

---

## Research & References

- **Paper**: *Agentic FinOps: Autonomous Token-Quota Balancing for LLM Cost Governance* (included: `Agentic_FinOps_IEEE_Formatted.docx`)
- **ATQB Algorithm**: Hierarchical token-budget enforcement with measured model quality optimization
- **MAG Framework**: Six-agent orchestration for decoupled governance decisions

---

## License

Proprietary (internal use only)

---

## Support

For issues, feature requests, or research inquiries:
- Review `tests/test_*.py` for usage examples
- Check `.env.example` for configuration schema
- Inspect orchestration traces via `/api/mag/sequence-traces` for debugging

**Status**: Production-ready for internal FinOps governance. Deploy with confidence.
