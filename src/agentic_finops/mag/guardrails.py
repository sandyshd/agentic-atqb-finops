from __future__ import annotations

from dataclasses import dataclass

from agentic_finops.atqb.models import ActuatorAction


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    enforced_action: ActuatorAction
    reason: str


@dataclass
class PolicyGuardrails:
    """Policy enforcement before any autonomous optimization action executes."""

    allow_cross_cloud_shift: bool = True
    require_human_for_hard_stop: bool = False

    def evaluate(
        self,
        requested_action: ActuatorAction,
        intent: str,
        tpw: float,
        provider: str,
    ) -> GuardrailDecision:
        # Protect high-priority intent from aggressive optimization actions.
        if intent in {"critical_ops", "safety"} and requested_action in {
            ActuatorAction.downgrade,
            ActuatorAction.quantize,
        }:
            return GuardrailDecision(
                allowed=False,
                enforced_action=ActuatorAction.throttle,
                reason="Critical intent guardrail: downgrade/quantize blocked; throttle enforced",
            )

        if requested_action == ActuatorAction.hard_stop and self.require_human_for_hard_stop:
            return GuardrailDecision(
                allowed=False,
                enforced_action=ActuatorAction.throttle,
                reason="Hard-stop requires human approval in this policy profile",
            )

        if not self.allow_cross_cloud_shift and provider != "azure":
            return GuardrailDecision(
                allowed=False,
                enforced_action=ActuatorAction.allow,
                reason="Cross-cloud shifting disabled by policy; staying on approved provider",
            )

        return GuardrailDecision(
            allowed=True,
            enforced_action=requested_action,
            reason="Policy guardrails passed",
        )
