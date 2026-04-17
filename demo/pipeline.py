"""Pure-function eval pipeline for the wearable agent demo.

Extracted from demo/app.py to enable unit testing and CLI reuse without
the Streamlit dependency.  The single public entry point is
:func:`run_eval_pipeline`.

Pipeline steps
--------------
1. :class:`~src.data.wearable_generator.WearableLogGenerator` — synthetic
   sensor/audio logs with Gaussian DP noise (ε=1.0).
2. :func:`_privacy_gate_result` — consent check.  Blocks REVOKED consent and
   AMBIENT consent when the scenario is ``privacy_sensitive`` (intimate /
   healthcare contexts require at minimum IMPLIED consent).
3. :class:`~src.eval.hitl_trigger.HITLTriggerEvaluator` — flags steps that
   require human review before the agent proceeds.
4. :class:`~src.eval.agentic_eval.FACTSGroundingScorer` — parametric +
   search + grounding sub-scores (DeepMind FACTS).
5. :class:`~src.eval.agentic_eval.AgenticEvaluator` — 5-layer
   TrajectoryScorer + 4 PIA rubric dimensions.
"""

from __future__ import annotations

import logging
from typing import Any

from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import ScenarioType, WearableLog, WearableLogGenerator
from src.eval.agentic_eval import AgenticEvaluator, FACTSGroundingScorer
from src.eval.hitl_trigger import HITLTrigger, HITLTriggerEvaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------


def _privacy_gate_result(
    log: WearableLog,
    gate_enabled: bool,
) -> tuple[bool, str]:
    """Return ``(blocked, reason)`` for a single wearable log.

    Blocking conditions (only evaluated when ``gate_enabled=True``):

    * :attr:`~src.data.privacy_gate.ConsentModel.REVOKED` consent — always
      blocked regardless of scenario.
    * ``privacy_sensitive`` scenario with
      :attr:`~src.data.privacy_gate.ConsentModel.AMBIENT` consent — blocked
      because intimate / healthcare contexts require at minimum IMPLIED consent.

    Args:
        log: The wearable log to evaluate.
        gate_enabled: When ``False`` the gate is bypassed and ``(False,
            "gate disabled")`` is returned immediately.

    Returns:
        Tuple of ``(blocked: bool, reason: str)``.
    """
    if not gate_enabled:
        return False, "gate disabled"

    consent_val: str = log.consent_model.value
    scenario_val: str = str(log.scenario_type)

    if consent_val == ConsentModel.REVOKED.value:
        return True, "consent REVOKED — capture suppressed"

    # Ambient consent is insufficient for privacy-sensitive contexts.
    if (
        scenario_val == ScenarioType.PRIVACY_SENSITIVE.value
        and consent_val == ConsentModel.AMBIENT.value
    ):
        return (
            True,
            "privacy_sensitive scenario with AMBIENT consent — capture suppressed",
        )

    if consent_val == ConsentModel.AMBIENT.value:
        return False, "ambient consent — reduced trust"

    return False, f"consent: {consent_val}"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_eval_pipeline(
    scenario: str,
    num_trajectories: int = 3,
    privacy_gate_enabled: bool = True,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run the full 5-step wearable eval pipeline and return per-trajectory results.

    All evaluation is performed in ``dry_run=True`` mode so no live API calls
    are made.  Results are deterministic for a given ``seed``.

    Args:
        scenario: A :class:`~src.data.wearable_generator.ScenarioType` value
            string, e.g. ``"health_alert"``.
        num_trajectories: Number of logs to generate and evaluate.  Must be
            ≥ 1.
        privacy_gate_enabled: When ``True`` the privacy gate is applied
            (REVOKED + privacy_sensitive/AMBIENT → blocked).
        seed: Integer RNG seed forwarded to
            :class:`~src.data.wearable_generator.WearableLogGenerator`.

    Returns:
        List of result dicts, one per trajectory.  Each dict exposes:

        * ``log_id`` (str): UUID of the generated log.
        * ``scenario`` (str): ScenarioType value.
        * ``consent_model`` (str): ConsentModel value.
        * ``privacy_blocked`` (bool): Whether the privacy gate blocked this log.
        * ``privacy_reason`` (str): Human-readable gate decision explanation.
        * ``privacy_gate_enabled`` (bool): Echo of the input flag.
        * ``facts_score`` (float): DeepMind FACTS overall mean in [0, 1].
        * ``facts_detail`` / ``facts`` (dict): Per sub-score breakdown.
        * ``traj_quality`` (float): 5-layer weighted composite in [0, 1].
        * ``tool_accuracy`` (float): Kore.ai tool_invocation_accuracy in [0, 1].
        * ``traj_success`` (float): Kore.ai trajectory_success_rate in [0, 1].
        * ``privacy_leak_detected`` (bool): PII scan result.
        * ``hitl_triggered`` (bool): ``True`` if ≥ 1 step fired a trigger.
        * ``hitl_count`` (int): Number of HITL triggers fired.
        * ``hitl_triggers`` (list[dict]): Serialised trigger details.
        * ``triggers`` (list[HITLTrigger]): Raw trigger objects (for UI use).
        * ``eval`` (dict): Full :class:`~src.eval.agentic_eval.AgenticEvaluator`
            output (17 keys).
        * ``trajectory_steps`` (list[dict]): Step-by-step trajectory.
        * ``log`` (WearableLog): Raw log object (for UI JSON preview).
        * ``is_mock`` (bool): Always ``False``; present for UI compatibility.
    """
    generator = WearableLogGenerator(seed=seed)
    logs: list[WearableLog] = generator.generate_batch(
        count=num_trajectories,
        scenario_filter=[ScenarioType(scenario)],
    )

    evaluator = AgenticEvaluator(dry_run=True)
    facts_scorer = FACTSGroundingScorer()
    hitl_evaluator = HITLTriggerEvaluator()

    results: list[dict[str, Any]] = []
    for log in logs:
        # Step 2 — privacy gate
        blocked, privacy_reason = _privacy_gate_result(log, privacy_gate_enabled)

        # Step 3 — HITL triggers
        hitl_steps = [
            {
                "action": step.action,
                "output": step.observation,
                "confidence": step.confidence,
                "tool_calls": [step.action] if step.action else [],
            }
            for step in log.trajectory
        ]
        triggers: list[HITLTrigger] = hitl_evaluator.evaluate_trajectory(
            hitl_steps, log.log_id
        )

        # Step 4 — FACTS grounding
        act_obs: str = log.trajectory[-1].observation if log.trajectory else ""
        source_docs = [s.observation for s in log.trajectory[:-1]]
        facts = facts_scorer.score(act_obs, source_docs or [act_obs])

        # Step 5 — 5-layer trajectory scoring + PIA dimensions
        eval_dict = evaluator.evaluate_with_trajectory_score(log)

        logger.debug(
            "log=%s blocked=%s triggers=%d facts=%.3f quality=%.3f",
            log.log_id[:8],
            blocked,
            len(triggers),
            facts["overall_facts_score"],
            eval_dict["weighted_total"],
        )

        results.append(
            {
                # metadata
                "log_id": log.log_id,
                "scenario": str(log.scenario_type),
                "consent_model": log.consent_model.value,
                # privacy gate
                "privacy_blocked": blocked,
                "privacy_reason": privacy_reason,
                "privacy_gate_enabled": privacy_gate_enabled,
                # flat score keys (used by tests and score history)
                "facts_score": facts["overall_facts_score"],
                "traj_quality": eval_dict["weighted_total"],
                "tool_accuracy": eval_dict["kore_tool_invocation_accuracy"],
                "traj_success": eval_dict["kore_trajectory_success"],
                "privacy_leak_detected": eval_dict["kore_privacy_leak_detected"],
                # HITL
                "hitl_triggered": len(triggers) > 0,
                "hitl_count": len(triggers),
                "hitl_triggers": [
                    {
                        "trigger_type": t.trigger_type.value,
                        "severity": t.severity,
                        "step_index": t.step_index,
                    }
                    for t in triggers
                ],
                # rich objects — used by demo/app.py UI rendering
                "triggers": triggers,
                "facts": facts,
                "facts_detail": facts,
                "eval": eval_dict,
                "trajectory_steps": [
                    {
                        "step_index": s.step_index,
                        "step_name": s.step_name,
                        "observation": s.observation,
                        "reasoning": s.reasoning,
                        "action": s.action,
                        "confidence": s.confidence,
                    }
                    for s in log.trajectory
                ],
                "log": log,
                "is_mock": False,
            }
        )

    return results
