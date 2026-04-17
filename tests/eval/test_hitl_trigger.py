"""Tests for src/eval/hitl_trigger.py — 12 cases covering all four trigger types."""

from __future__ import annotations

import pytest

from src.eval.hitl_trigger import (
    HITLTriggerEvaluator,
    TriggerType,
)


@pytest.fixture()
def evaluator() -> HITLTriggerEvaluator:
    return HITLTriggerEvaluator(confidence_threshold=0.70)


# ---------------------------------------------------------------------------
# 1. Confidence trigger fires
# ---------------------------------------------------------------------------


def test_confidence_trigger_fires(evaluator: HITLTriggerEvaluator) -> None:
    step = {"confidence": 0.50}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.CONFIDENCE_BELOW_THRESHOLD


# ---------------------------------------------------------------------------
# 2. Confidence trigger suppressed when score is above threshold
# ---------------------------------------------------------------------------


def test_confidence_trigger_suppressed(evaluator: HITLTriggerEvaluator) -> None:
    step = {"confidence": 0.85}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is None


# ---------------------------------------------------------------------------
# 3. Safety-adjacent trigger fires on emergency action
# ---------------------------------------------------------------------------


def test_safety_adjacent_fires(evaluator: HITLTriggerEvaluator) -> None:
    step = {"action": "escalate_to_emergency", "confidence": 0.95}
    trigger = evaluator.evaluate_step(step, step_index=1, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.SAFETY_ADJACENT_ACTION


# ---------------------------------------------------------------------------
# 4. Novel tool trigger fires for unregistered tool
# ---------------------------------------------------------------------------


def test_novel_tool_fires(evaluator: HITLTriggerEvaluator) -> None:
    step = {"tool_calls": ["unknown_db_tool"], "confidence": 0.95}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.NOVEL_TOOL_PATTERN


# ---------------------------------------------------------------------------
# 5. Known tools do not fire novel-tool trigger
# ---------------------------------------------------------------------------


def test_known_tool_suppressed(evaluator: HITLTriggerEvaluator) -> None:
    # All tools in KNOWN_TOOLS; no other trigger conditions present.
    step = {
        "tool_calls": ["search", "log"],
        "confidence": 0.95,
        "action": "lookup",
        "output": "ok",
    }
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is None


# ---------------------------------------------------------------------------
# 6. Medical domain trigger fires on heart-rate output
# ---------------------------------------------------------------------------


def test_medical_domain_fires(evaluator: HITLTriggerEvaluator) -> None:
    step = {"output": "heart rate 45 bpm detected", "confidence": 0.95}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.DOMAIN_EXPERTISE_REQUIRED
    assert trigger.domain_flag == "medical"


# ---------------------------------------------------------------------------
# 7. Legal domain trigger fires on GDPR action
# ---------------------------------------------------------------------------


def test_legal_domain_fires(evaluator: HITLTriggerEvaluator) -> None:
    step = {"action": "gdpr compliance check", "confidence": 0.95}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.DOMAIN_EXPERTISE_REQUIRED
    assert trigger.domain_flag == "legal"


# ---------------------------------------------------------------------------
# 8. Financial domain trigger fires on fraud output
# ---------------------------------------------------------------------------


def test_financial_domain_fires(evaluator: HITLTriggerEvaluator) -> None:
    step = {"output": "transaction fraud risk score: 0.91", "confidence": 0.95}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.DOMAIN_EXPERTISE_REQUIRED
    assert trigger.domain_flag == "financial"


# ---------------------------------------------------------------------------
# 9. evaluate_trajectory returns exactly the expected number of triggers
# ---------------------------------------------------------------------------


def test_evaluate_trajectory_multiple(evaluator: HITLTriggerEvaluator) -> None:
    trajectory = [
        # step 0 — triggers: confidence low
        {"confidence": 0.45},
        # step 1 — triggers: novel tool
        {
            "tool_calls": ["custom_sensor_api"],
            "confidence": 0.95,
            "action": "fetch",
            "output": "data",
        },
        # step 2 — clean
        {
            "tool_calls": ["search"],
            "confidence": 0.90,
            "action": "lookup",
            "output": "results",
        },
        # step 3 — triggers: safety
        {"action": "escalate_to_emergency", "confidence": 0.95},
        # step 4 — clean
        {
            "tool_calls": ["log"],
            "confidence": 0.80,
            "action": "record",
            "output": "done",
        },
    ]
    triggers = evaluator.evaluate_trajectory(trajectory, trajectory_id="multi_test")
    assert len(triggers) == 3


# ---------------------------------------------------------------------------
# 10. summary() output contains all required keys
# ---------------------------------------------------------------------------


def test_summary_keys(evaluator: HITLTriggerEvaluator) -> None:
    step = {"action": "escalate_to_emergency"}
    triggers = evaluator.evaluate_trajectory([step], trajectory_id="t1")
    result = evaluator.summary(triggers)

    required_keys = {
        "total_triggers",
        "by_type",
        "critical_count",
        "high_count",
        "trigger_rate",
        "requires_immediate_review",
    }
    assert required_keys.issubset(result.keys())


# ---------------------------------------------------------------------------
# 11. Clean trajectory returns empty trigger list
# ---------------------------------------------------------------------------


def test_no_triggers_trajectory(evaluator: HITLTriggerEvaluator) -> None:
    trajectory = [
        {
            "tool_calls": ["search"],
            "confidence": 0.95,
            "action": "lookup",
            "output": "ok",
        },
        {
            "tool_calls": ["log"],
            "confidence": 0.80,
            "action": "record",
            "output": "saved",
        },
        {
            "tool_calls": ["notify"],
            "confidence": 0.92,
            "action": "send",
            "output": "sent",
        },
    ]
    triggers = evaluator.evaluate_trajectory(trajectory, trajectory_id="clean")
    assert triggers == []


# ---------------------------------------------------------------------------
# 12. Safety trigger with "emergency" in action has severity == "critical"
# ---------------------------------------------------------------------------


def test_severity_critical_on_safety(evaluator: HITLTriggerEvaluator) -> None:
    step = {"action": "escalate_to_emergency", "confidence": 0.95}
    trigger = evaluator.evaluate_step(step, step_index=0, trajectory_id="t1")
    assert trigger is not None
    assert trigger.trigger_type == TriggerType.SAFETY_ADJACENT_ACTION
    assert trigger.severity == "critical"
