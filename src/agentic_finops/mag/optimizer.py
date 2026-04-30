from __future__ import annotations

from dataclasses import dataclass

from agentic_finops.atqb.models import ActuatorAction


@dataclass
class OptimizationExecutor:
    """Executes autonomous optimization actions selected by ATQB under guardrails."""

    def execute(
        self,
        action: ActuatorAction,
        model_name: str,
        provider: str,
    ) -> dict:
        if action == ActuatorAction.allow:
            return {
                "execution_state": "executed",
                "execution_action": "no_change",
                "effective_model": model_name,
                "effective_provider": provider,
                "optimization_note": "No intervention required",
            }

        if action == ActuatorAction.throttle:
            return {
                "execution_state": "executed",
                "execution_action": "throttle_20pct",
                "effective_model": model_name,
                "effective_provider": provider,
                "optimization_note": "Applied rate limit reduction to preserve HLB",
            }

        if action == ActuatorAction.quota_shift:
            return {
                "execution_state": "executed",
                "execution_action": "reallocate_quota",
                "effective_model": model_name,
                "effective_provider": provider,
                "optimization_note": "Shifted quota from low-TPW pools",
            }

        if action == ActuatorAction.downgrade:
            cheaper_model = "llama-4" if model_name != "llama-4" else model_name
            return {
                "execution_state": "executed",
                "execution_action": "model_downgrade",
                "effective_model": cheaper_model,
                "effective_provider": provider,
                "optimization_note": "Switched model to lower-UCI tier",
            }

        if action == ActuatorAction.quantize:
            return {
                "execution_state": "executed",
                "execution_action": "enable_quantization",
                "effective_model": model_name,
                "effective_provider": provider,
                "optimization_note": "Enabled quantization for efficiency",
            }

        return {
            "execution_state": "executed",
            "execution_action": "kill_switch_stop",
            "effective_model": model_name,
            "effective_provider": provider,
            "optimization_note": "Governor kill-switch engaged due to HLB breach",
        }
