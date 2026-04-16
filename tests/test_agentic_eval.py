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


# ---------------------------------------------------------------------------
# AgenticEvaluator — integration with TrajectoryScorer
# ---------------------------------------------------------------------------

import math  # noqa: E402 — placed here to keep existing imports unmodified

from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import (
    AgentAction,
    AudioTranscript,
    ScenarioType,
    SensorData,
    TrajectoryStep,
    WearableLog,
)
from src.eval.agentic_eval import AgenticEvaluator, _wearable_steps_to_kore_dicts


def _make_wearable_log(
    final_action: str = AgentAction.SEND_ALERT,
    log_id: str = "eval-log-001",
) -> WearableLog:
    sensor = SensorData(
        heart_rate=90.0,
        spo2=97.0,
        steps=50.0,
        gps_lat=37.77,
        gps_lon=-122.41,
        noise_db=55.0,
        skin_temp_c=36.8,
    )
    audio = AudioTranscript(
        text="",
        language="en-US",
        confidence=0.9,
        duration_s=1.0,
        keywords_detected=[],
    )
    steps = [
        TrajectoryStep(0, "sense", "obs", "reason", "", 0.9),
        TrajectoryStep(1, "plan", "obs", "reason", "", 0.9),
        TrajectoryStep(2, "act", "obs", "reason", final_action, 0.9),
    ]
    return WearableLog(
        log_id=log_id,
        timestamp="2026-04-16T00:00:00Z",
        scenario_type=ScenarioType.HEALTH_ALERT,
        consent_model=ConsentModel.EXPLICIT,
        sensor_data=sensor,
        audio_transcript=audio,
        context_metadata={},
        trajectory=steps,
        ground_truth_action=final_action,
    )


@pytest.fixture
def evaluator() -> AgenticEvaluator:
    return AgenticEvaluator(dry_run=True)


class TestAgenticEvaluatorInit:
    def test_trajectory_scorer_instantiated(self, evaluator: AgenticEvaluator) -> None:
        from src.eval.trajectory_scorer import TrajectoryScorer

        assert isinstance(evaluator._trajectory_scorer, TrajectoryScorer)

    def test_kore_metrics_instantiated(self, evaluator: AgenticEvaluator) -> None:
        assert isinstance(evaluator._kore_metrics, KoraiMetrics)

    def test_dry_run_propagated(self) -> None:
        ev = AgenticEvaluator(dry_run=False)
        assert ev._trajectory_scorer._dry_run is False


class TestWearableStepsToKoreDicts:
    def test_length_matches_trajectory(self) -> None:
        log = _make_wearable_log()
        dicts = _wearable_steps_to_kore_dicts(log)
        assert len(dicts) == len(log.trajectory)

    def test_final_step_has_expected_tools(self) -> None:
        log = _make_wearable_log(final_action=AgentAction.SEND_ALERT)
        dicts = _wearable_steps_to_kore_dicts(log)
        assert log.ground_truth_action in dicts[-1]["expected_tools"]

    def test_intermediate_steps_have_empty_expected_tools(self) -> None:
        log = _make_wearable_log()
        dicts = _wearable_steps_to_kore_dicts(log)
        for d in dicts[:-1]:
            assert d["expected_tools"] == []

    def test_tool_call_maps_to_action(self) -> None:
        log = _make_wearable_log(final_action=AgentAction.SEND_ALERT)
        dicts = _wearable_steps_to_kore_dicts(log)
        assert dicts[-1]["tool_call"] == AgentAction.SEND_ALERT

    def test_sense_step_tool_call_is_none(self) -> None:
        log = _make_wearable_log()
        dicts = _wearable_steps_to_kore_dicts(log)
        assert dicts[0]["tool_call"] is None

    def test_agent_role_equals_expected_role(self) -> None:
        log = _make_wearable_log()
        for d in _wearable_steps_to_kore_dicts(log):
            assert d["agent_role"] == d["expected_role"]


class TestEvaluateWithTrajectoryScore:
    _EXPECTED_KEYS = {
        "trajectory_id",
        "kore_trajectory_success",
        "kore_tool_invocation_accuracy",
        "kore_groundedness",
        "kore_privacy_leak_detected",
        "kore_orchestrator_correctness",
        "kore_latency_sla_compliance",
        "layer_intent",
        "layer_planning",
        "layer_tool_calls",
        "layer_recovery",
        "layer_outcome",
        "pia_planning_quality",
        "pia_error_recovery",
        "pia_goal_alignment",
        "pia_tool_precision",
        "weighted_total",
    }

    def test_returns_all_expected_keys(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        result = evaluator.evaluate_with_trajectory_score(log)
        assert set(result.keys()) == self._EXPECTED_KEYS

    def test_trajectory_id_matches_log(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log(log_id="specific-id-001")
        result = evaluator.evaluate_with_trajectory_score(log)
        assert result["trajectory_id"] == "specific-id-001"

    def test_weighted_total_in_unit_interval(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        result = evaluator.evaluate_with_trajectory_score(log)
        assert 0.0 <= result["weighted_total"] <= 1.0

    def test_all_float_values_finite(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        result = evaluator.evaluate_with_trajectory_score(log)
        for key, val in result.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{key}={val} is not finite"

    def test_kore_groundedness_is_fallback(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        result = evaluator.evaluate_with_trajectory_score(log)
        assert result["kore_groundedness"] == pytest.approx(0.75)

    def test_kore_latency_sla_is_one(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        result = evaluator.evaluate_with_trajectory_score(log)
        assert result["kore_latency_sla_compliance"] == pytest.approx(1.0)

    def test_pia_keys_in_unit_interval(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        result = evaluator.evaluate_with_trajectory_score(log)
        for key in ("pia_planning_quality", "pia_error_recovery", "pia_goal_alignment", "pia_tool_precision"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range"

    def test_layer_recovery_none_for_non_escalation(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log(final_action=AgentAction.SEND_ALERT)
        result = evaluator.evaluate_with_trajectory_score(log)
        assert result["layer_recovery"] is None

    def test_layer_recovery_float_for_escalation(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log(final_action=AgentAction.ESCALATE_TO_EMERGENCY)
        result = evaluator.evaluate_with_trajectory_score(log)
        assert result["layer_recovery"] == pytest.approx(0.70)


class TestBatchEvaluateWithTrajectoryScore:
    def test_length_matches_input(self, evaluator: AgenticEvaluator) -> None:
        logs = [_make_wearable_log(log_id=f"log-{i}") for i in range(5)]
        results = evaluator.batch_evaluate_with_trajectory_score(logs)
        assert len(results) == 5

    def test_empty_input_returns_empty(self, evaluator: AgenticEvaluator) -> None:
        assert evaluator.batch_evaluate_with_trajectory_score([]) == []

    def test_order_preserved(self, evaluator: AgenticEvaluator) -> None:
        ids = [f"log-{i:03d}" for i in range(4)]
        logs = [_make_wearable_log(log_id=lid) for lid in ids]
        results = evaluator.batch_evaluate_with_trajectory_score(logs)
        assert [r["trajectory_id"] for r in results] == ids

    def test_all_results_have_weighted_total(self, evaluator: AgenticEvaluator) -> None:
        logs = [_make_wearable_log(log_id=f"log-{i}") for i in range(3)]
        for r in evaluator.batch_evaluate_with_trajectory_score(logs):
            assert "weighted_total" in r
            assert math.isfinite(r["weighted_total"])


class TestComputeBatchNondeterminism:
    def test_returns_key_per_task(self, evaluator: AgenticEvaluator) -> None:
        groups = {
            "task_a": [_make_wearable_log(log_id=f"a-{i}") for i in range(3)],
            "task_b": [_make_wearable_log(log_id=f"b-{i}") for i in range(3)],
        }
        result = evaluator.compute_batch_nondeterminism(groups)
        assert set(result.keys()) == {"task_a", "task_b"}

    def test_variance_dict_has_required_keys(self, evaluator: AgenticEvaluator) -> None:
        groups = {
            "task_x": [_make_wearable_log(log_id=f"x-{i}") for i in range(3)],
        }
        result = evaluator.compute_batch_nondeterminism(groups)
        assert set(result["task_x"].keys()) == {
            "score_std",
            "pia_planning_std",
            "pia_recovery_std",
            "pia_goal_std",
            "pia_tool_std",
            "max_variance_layer",
        }

    def test_single_run_task_skipped(self, evaluator: AgenticEvaluator) -> None:
        groups = {"solo": [_make_wearable_log()]}
        result = evaluator.compute_batch_nondeterminism(groups)
        assert "solo" not in result

    def test_empty_groups_returns_empty(self, evaluator: AgenticEvaluator) -> None:
        assert evaluator.compute_batch_nondeterminism({}) == {}

    def test_identical_runs_zero_score_std(self, evaluator: AgenticEvaluator) -> None:
        log = _make_wearable_log()
        groups = {"same": [_make_wearable_log(log_id=f"s-{i}") for i in range(3)]}
        result = evaluator.compute_batch_nondeterminism(groups)
        assert result["same"]["score_std"] == pytest.approx(0.0, abs=1e-9)
