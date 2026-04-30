from __future__ import annotations

from random import Random


class InferenceEventGenerator:
    """
    Generates synthetic inference call parameters for demo purposes.
    Covers heterogeneous models, varying TPW workloads, SLO breaches,
    and periodic critical-priority requests to exercise all ATQB actuator paths.
    """

    MODELS    = ["gpt-5", "claude-4", "gpt-4", "llama-4"]
    WORKLOADS = ["workload-nlp", "workload-code", "workload-analysis", "workload-chat"]
    REQUEST_TEXTS = [
        "optimize token cost for assistant pipeline",
        "compliance audit summary required",
        "critical incident response assistant",
        "general chat interaction",
    ]
    SLO_MS    = 2000.0

    def __init__(self, seed: int = 17) -> None:
        self._rng     = Random(seed)
        self._counter = 0

    def next(self) -> dict:
        self._counter += 1
        workload   = self.WORKLOADS[self._counter % len(self.WORKLOADS)]
        model      = self.MODELS[self._counter % len(self.MODELS)]

        # Occasional expensive model access pushes UCI up
        if self._counter % 11 == 0:
            model = "gpt-5"

        tokens_input  = int(self._rng.uniform(80, 3000))
        tokens_output = int(self._rng.uniform(40, 1000))

        # Inject SLO breaches periodically to exercise latency penalty path
        if self._counter % 8 == 0:
            latency_ms = round(self._rng.uniform(2200, 5000), 1)
        else:
            latency_ms = round(self._rng.uniform(150, 1800), 1)

        # Critical requests every 7 ticks (high TPW, quota borrowing path)
        is_critical = self._counter % 7 == 0

        return {
            "workload_id":    workload,
            "model_name":     model,
            "tokens_input":   tokens_input,
            "tokens_output":  tokens_output,
            "latency_ms":     latency_ms,
            "slo_latency_ms": self.SLO_MS,
            "is_critical":    is_critical,
            "request_text":   self.REQUEST_TEXTS[self._counter % len(self.REQUEST_TEXTS)],
            "output_text":    f"response-{self._counter}",
        }
