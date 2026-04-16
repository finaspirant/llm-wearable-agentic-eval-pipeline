"""Tests for src/eval/agentic_eval.py.

Covers KoraiMetrics, FACTSGroundingScorer, DeepEvalJudge, and the
compute_overall_score weighted composite. All tests are deterministic
and require no LLM API key.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.eval.agentic_eval import (
    AgenticEvalResult,
    DeepEvalJudge,
    FACTSGroundingScorer,
    KoraiMetrics,
    compute_overall_score,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# tool_call mirrors tool_name — score_tool_invocation reads tool_call.
# step 5: tool_call="retry_log" is NOT in expected_tools=["log_event"] → wrong.
SAMPLE_TRAJECTORY: list[dict] = [
    {
        "step": 1,
        "agent_role": "health_monitor",
        "expected_role": "health_monitor",
        "tool_name": "read_sensor",
        "tool_call": "read_sensor",
        "expected_tools": ["read_sensor"],
        "goal_achieved": True,
        "tool_output": "heart_rate: 92bpm",
    },
    {
        "step": 2,
        "agent_role": "privacy_gate",
        "expected_role": "privacy_gate",
        "tool_name": "check_context",
        "tool_call": "check_context",
        "expected_tools": ["check_context"],
        "goal_achieved": True,
        "tool_output": "context: private_meeting",
    },
    {
        "step": 3,
        "agent_role": "privacy_gate",
        "expected_role": "privacy_gate",
        "tool_name": "suppress_alert",
        "tool_call": "suppress_alert",
        "expected_tools": ["suppress_alert"],
        "goal_achieved": True,
        "tool_output": "alert suppressed",
    },
    {
        "step": 4,
        "agent_role": "health_monitor",
        "expected_role": "health_monitor",
        "tool_name": "log_event",
        "tool_call": "log_event",
        "expected_tools": ["log_event"],
        "goal_achieved": False,
        "tool_output": "logging failed",
    },
    {
        "step": 5,
        "agent_role": "health_monitor",
        "expected_role": "health_monitor",
        "tool_name": "retry_log",
        "tool_call": "retry_log",
        "expected_tools": ["log_event"],
        "goal_achieved": True,
        "tool_output": "logged",
    },
]


@pytest.fixture()
def metrics() -> KoraiMetrics:
    return KoraiMetrics()


@pytest.fixture()
def facts_scorer() -> FACTSGroundingScorer:
    return FACTSGroundingScorer()


@pytest.fixture()
def judge() -> DeepEvalJudge:
    return DeepEvalJudge()


def _make_result(**overrides: float | bool | str) -> AgenticEvalResult:
    """Build a minimal AgenticEvalResult for composite score tests."""
    defaults: dict = {
        "trajectory_id": "t-test",
        "task_id": "task-test",
        "framework": "langgraph",
        "trajectory_success_rate": 0.8,
        "tool_invocation_accuracy": 0.8,
        "groundedness_score": 0.75,
        "privacy_leak_detected": False,
        "orchestrator_correctness": 1.0,
        "latency_sla_compliance": 1.0,
        "overall_score": 0.0,
        "eval_timestamp": "2026-04-16T00:00:00+00:00",
    }
    defaults.update(overrides)
    return AgenticEvalResult(**defaults)


# ---------------------------------------------------------------------------
# KoraiMetrics — trajectory success
# ---------------------------------------------------------------------------


def test_trajectory_success_rate(metrics: KoraiMetrics) -> None:
    # steps 1, 2, 3, 5 are True; step 4 is False → 4/5
    rate = metrics.score_trajectory_success(SAMPLE_TRAJECTORY)
    assert rate == pytest.approx(0.8)


def test_trajectory_success_empty(metrics: KoraiMetrics) -> None:
    assert metrics.score_trajectory_success([]) == 0.0


# ---------------------------------------------------------------------------
# KoraiMetrics — tool invocation accuracy
# ---------------------------------------------------------------------------


def test_tool_invocation_accuracy(metrics: KoraiMetrics) -> None:
    # steps 1–4 correct; step 5 tool_call="retry_log" ∉ ["log_event"] → 4/5
    acc = metrics.score_tool_invocation(SAMPLE_TRAJECTORY)
    assert acc == pytest.approx(0.8)


def test_tool_invocation_no_tool_calls(metrics: KoraiMetrics) -> None:
    # no tool_call field → returns 1.0 (nothing to penalise)
    traj = [{"step": 1, "goal_achieved": True}]
    assert metrics.score_tool_invocation(traj) == 1.0


# ---------------------------------------------------------------------------
# KoraiMetrics — privacy leak detection
# ---------------------------------------------------------------------------


def test_privacy_leak_none(metrics: KoraiMetrics) -> None:
    assert metrics.detect_privacy_leak(SAMPLE_TRAJECTORY) is False


def test_privacy_leak_detected(metrics: KoraiMetrics) -> None:
    leaked = [*SAMPLE_TRAJECTORY, {"step": 6, "tool_output": "email: user@test.com"}]
    assert metrics.detect_privacy_leak(leaked) is True


def test_privacy_leak_phone(metrics: KoraiMetrics) -> None:
    traj = [{"tool_output": "contact: 555-867-5309"}]
    assert metrics.detect_privacy_leak(traj) is True


def test_privacy_leak_ssn(metrics: KoraiMetrics) -> None:
    traj = [{"tool_output": "ssn: 123-45-6789"}]
    assert metrics.detect_privacy_leak(traj) is True


# ---------------------------------------------------------------------------
# KoraiMetrics — orchestrator correctness
# ---------------------------------------------------------------------------


def test_orchestrator_correctness(metrics: KoraiMetrics) -> None:
    # all agent_role == expected_role → 5/5 = 1.0
    score = metrics.score_orchestrator_correctness(SAMPLE_TRAJECTORY)
    assert score == pytest.approx(1.0)


def test_orchestrator_correctness_mismatch(metrics: KoraiMetrics) -> None:
    traj = [
        {"agent_role": "health_monitor", "expected_role": "privacy_gate"},
        {"agent_role": "privacy_gate", "expected_role": "privacy_gate"},
    ]
    assert metrics.score_orchestrator_correctness(traj) == pytest.approx(0.5)


def test_orchestrator_correctness_empty(metrics: KoraiMetrics) -> None:
    assert metrics.score_orchestrator_correctness([]) == 1.0


# ---------------------------------------------------------------------------
# KoraiMetrics — latency SLA
# ---------------------------------------------------------------------------


def test_latency_sla_pass(metrics: KoraiMetrics) -> None:
    assert metrics.score_latency_sla(3000.0, sla_ms=5000.0) == pytest.approx(1.0)


def test_latency_sla_at_boundary(metrics: KoraiMetrics) -> None:
    assert metrics.score_latency_sla(5000.0, sla_ms=5000.0) == pytest.approx(1.0)


def test_latency_sla_fail(metrics: KoraiMetrics) -> None:
    # 8000ms with 5000ms SLA → 1 - (3000/5000) = 0.4
    score = metrics.score_latency_sla(8000.0, sla_ms=5000.0)
    assert 0.0 < score < 0.5


def test_latency_sla_extreme_clamp(metrics: KoraiMetrics) -> None:
    # far over SLA → clamped to 0.0
    assert metrics.score_latency_sla(100_000.0, sla_ms=5000.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_overall_score — weighted composite
# ---------------------------------------------------------------------------


def test_overall_score_range() -> None:
    result = _make_result()
    score = compute_overall_score(result)
    assert 0.0 <= score <= 1.0


def test_overall_score_perfect() -> None:
    result = _make_result(
        trajectory_success_rate=1.0,
        tool_invocation_accuracy=1.0,
        groundedness_score=1.0,
        privacy_leak_detected=False,
        orchestrator_correctness=1.0,
        latency_sla_compliance=1.0,
    )
    assert compute_overall_score(result) == pytest.approx(1.0)


def test_overall_score_privacy_leak_penalty() -> None:
    clean = _make_result(privacy_leak_detected=False)
    leaked = _make_result(privacy_leak_detected=True)
    # privacy weight = 0.10; leaked score must be lower
    assert compute_overall_score(leaked) < compute_overall_score(clean)


# ---------------------------------------------------------------------------
# FACTSGroundingScorer
# ---------------------------------------------------------------------------


def test_facts_scorer_search_score(facts_scorer: FACTSGroundingScorer) -> None:
    docs = ["heart rate elevated at 102 bpm", "tachycardia threshold is 100 bpm"]
    response = "The heart rate is 102 bpm, exceeding the tachycardia threshold."
    result = facts_scorer.score(response, docs)

    assert set(result.keys()) == {
        "parametric_score",
        "search_score",
        "grounding_score",
        "overall_facts_score",
    }
    assert result["parametric_score"] == pytest.approx(0.70)
    # response tokens heavily overlap with docs → high search score
    assert result["search_score"] > 0.5
    assert 0.0 <= result["search_score"] <= 1.0


def test_facts_scorer_no_overlap(facts_scorer: FACTSGroundingScorer) -> None:
    docs = ["apple orange banana"]
    response = "The satellite orbited Jupiter."
    result = facts_scorer.score(response, docs)
    assert result["search_score"] == pytest.approx(0.0)


def test_facts_scorer_overall_is_mean(facts_scorer: FACTSGroundingScorer) -> None:
    docs = ["sensor data shows normal vitals"]
    response = "Vitals are normal."
    result = facts_scorer.score(response, docs)
    expected_mean = (
        result["parametric_score"] + result["search_score"] + result["grounding_score"]
    ) / 3.0
    assert result["overall_facts_score"] == pytest.approx(expected_mean)


def test_facts_scorer_empty_response(facts_scorer: FACTSGroundingScorer) -> None:
    result = facts_scorer.score("", ["some context"])
    assert 0.0 <= result["overall_facts_score"] <= 1.0


# ---------------------------------------------------------------------------
# DeepEvalJudge
# ---------------------------------------------------------------------------


def test_deepeval_judge_returns_dict(judge: DeepEvalJudge) -> None:
    mock_metric = MagicMock()
    mock_metric.measure.return_value = 0.85

    with patch("deepeval.metrics.GEval", return_value=mock_metric):
        result = judge.judge_trajectory(
            SAMPLE_TRAJECTORY, "Monitor wearable health data and suppress alerts."
        )

    assert set(result.keys()) == {
        "trajectory_quality",
        "error_recovery",
        "goal_alignment",
        "ensemble_score",
    }


def test_deepeval_judge_ensemble_is_mean(judge: DeepEvalJudge) -> None:
    mock_metric = MagicMock()
    mock_metric.measure.return_value = 0.9

    with patch("deepeval.metrics.GEval", return_value=mock_metric):
        result = judge.judge_trajectory(SAMPLE_TRAJECTORY, "Health monitoring task.")

    expected = (
        result["trajectory_quality"]
        + result["error_recovery"]
        + result["goal_alignment"]
    ) / 3.0
    assert result["ensemble_score"] == pytest.approx(expected)


def test_deepeval_judge_fallback_on_error(judge: DeepEvalJudge) -> None:
    with patch("deepeval.metrics.GEval", side_effect=RuntimeError("no model")):
        result = judge.judge_trajectory(SAMPLE_TRAJECTORY, "Any task.")

    assert result["trajectory_quality"] == pytest.approx(0.7)
    assert result["error_recovery"] == pytest.approx(0.7)
    assert result["goal_alignment"] == pytest.approx(0.7)
    assert result["ensemble_score"] == pytest.approx(0.7)


def test_deepeval_judge_scores_clamped(judge: DeepEvalJudge) -> None:
    mock_metric = MagicMock()
    mock_metric.measure.return_value = 1.5  # out-of-range raw value

    with patch("deepeval.metrics.GEval", return_value=mock_metric):
        result = judge.judge_trajectory(SAMPLE_TRAJECTORY, "Health monitoring task.")

    for key in ("trajectory_quality", "error_recovery", "goal_alignment"):
        assert result[key] <= 1.0
