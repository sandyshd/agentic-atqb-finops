from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_finops.atqb.models import ActuatorAction

if TYPE_CHECKING:
    from agentic_finops.mag.agents import ScalerRecommendation


@dataclass
class OptimizationExecutor:
    """Executes autonomous optimization actions selected by ATQB under guardrails."""

    def execute(
        self,
        action: ActuatorAction,
        model_name: str,
        provider: str,
        scaler_recommendation: "ScalerRecommendation | None" = None,
    ) -> dict:
        result = self._execute_primary(action, model_name, provider)
        # Side action: the Scaler's elasticity recommendation runs in parallel
        # with the primary ATQB action so over-/under-provisioning is acted on
        # even when the governance verb is "allow" or "throttle".
        if scaler_recommendation is not None:
            result["scaler"] = {
                "action": scaler_recommendation.action,
                "replicas_current": scaler_recommendation.replicas_current,
                "replicas_recommended": scaler_recommendation.replicas_recommended,
                "forecast_tpw": scaler_recommendation.forecast_tpw,
                "capacity_tpw": scaler_recommendation.capacity_tpw,
                "pressure_ratio": scaler_recommendation.pressure_ratio,
                "cost_delta_usd": scaler_recommendation.cost_delta_usd,
                "reasoning": scaler_recommendation.reasoning,
            }
        return result

    def _execute_primary(
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
