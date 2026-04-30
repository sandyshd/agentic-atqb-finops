from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceRecord(BaseModel):
    """A single instrumented AI inference call."""

    request_id: str
    workload_id: str
    model_name: str
    provider: str = "azure"
    tpw: float = Field(ge=0, le=1, description="Task Priority Weight assigned by Gatekeeper")
    request_count: int = Field(default=1, ge=1, description="Number of requests represented by this record")
    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    slo_latency_ms: float = Field(ge=0, description="SLO threshold in milliseconds")
    success: bool = Field(description="Output validated by Critic Agent (non-hallucinatory)")
    timestamp: str


class UCIRecord(BaseModel):
    """
    UCI = (C_tokens + C_compute + C_latency_penalty) / N_successful_tasks

    C_tokens          : Weighted cost of input/output tokens across heterogeneous models.
    C_compute         : Overhead of transient GPU/NPU orchestration.
    C_latency_penalty : Fiscal loss when inference time exceeds application SLO.
    N_successful_tasks: Count of validated, non-hallucinatory outputs (Critic Agent verified).
    """

    period: str
    workload_id: str
    c_tokens: float
    c_compute: float
    c_latency_penalty: float
    n_successful_tasks: int
    uci: float
