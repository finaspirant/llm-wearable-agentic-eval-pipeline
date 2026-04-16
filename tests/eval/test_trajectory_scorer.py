"""Tests for src.eval.trajectory_scorer.

Covers all public methods of TrajectoryScorer without API calls
(dry_run=True throughout).
"""

from __future__ import annotations

import pytest

from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import (
    AgentAction,
    AudioTranscript,
    ScenarioType,
    SensorData,
    TrajectoryStep,
    WearableLog,
)
from src.eval.trajectory_scorer import (
    _DEFAULT_WEIGHTS,
    TrajectoryScorer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(index: int, name: str, action: str = "") -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        step_name=name,
        observation=f"obs_{name}",
        reasoning=f"reason_{name}",
        action=action,
        confidence=0.9,
    )


def _make_sensor() -> SensorData:
    return SensorData(
        heart_rate=75.0,
        spo2=98.0,
        steps=100.0,
        gps_lat=37.7749,
        gps_lon=-122.4194,
        noise_db=50.0,
        skin_temp_c=36.6,
    )


def _make_audio() -> AudioTranscript:
    return AudioTranscript(
        text="",
        language="en-US",
        confidence=0.9,
        duration_s=1.0,
        keywords_detected=[],
    )


def _make_log(
    scenario: ScenarioType = ScenarioType.HEALTH_ALERT,
    final_action: str = AgentAction.SEND_ALERT,
    n_steps: int = 3,
    log_id: str = "test-log-001",
) -> WearableLog:
    """Build a WearableLog with n_steps steps ending in final_action."""
    steps: list[TrajectoryStep] = [_make_step(0, "sense")]
    for i in range(1, n_steps - 1):
        steps.append(_make_step(i, f"plan_{i}"))
    steps.append(_make_step(n_steps - 1, "act", action=final_action))
    return WearableLog(
        log_id=log_id,
        timestamp="2026-04-16T00:00:00Z",
        scenario_type=scenario,
        consent_model=ConsentModel.EXPLICIT,
        sensor_data=_make_sensor(),
        audio_transcript=_make_audio(),
        context_metadata={},
        trajectory=steps,
        ground_truth_action=final_action,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_trajectory() -> WearableLog:
    """3-step health_alert trajectory ending with SEND_ALERT."""
    return _make_log(
        scenario=ScenarioType.HEALTH_ALERT,
        final_action=AgentAction.SEND_ALERT,
        n_steps=3,
        log_id="minimal-001",
    )


@pytest.fixture
def escalation_trajectory() -> WearableLog:
    """3-step health_alert trajectory ending with ESCALATE_TO_EMERGENCY."""
    return _make_log(
        scenario=ScenarioType.HEALTH_ALERT,
        final_action=AgentAction.ESCALATE_TO_EMERGENCY,
        n_steps=3,
        log_id="escalation-001",
    )


@pytest.fixture
def over_engineered_trajectory() -> WearableLog:
    """8-step trajectory — low step efficiency."""
    return _make_log(
        scenario=ScenarioType.HEALTH_ALERT,
        final_action=AgentAction.SEND_ALERT,
        n_steps=8,
        log_id="overengineered-001",
    )


@pytest.fixture
def three_run_batch() -> list[WearableLog]:
    """3 copies of minimal_trajectory with distinct log_ids (tiny variation)."""
    return [
        _make_log(
            scenario=ScenarioType.HEALTH_ALERT,
            final_action=AgentAction.SEND_ALERT,
            n_steps=3,
            log_id=f"run-{i:03d}",
        )
        for i in range(3)
    ]


@pytest.fixture
def scorer() -> TrajectoryScorer:
    return TrajectoryScorer(dry_run=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_score_intent_valid_scenario(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    result = scorer.score_intent(minimal_trajectory)
    assert result.score > 0.5
    assert result.matched_goal is True


def test_score_intent_no_scenario(scorer: TrajectoryScorer) -> None:
    log = _make_log()
    object.__setattr__(log, "scenario_type", "not_a_real_scenario")
    result = scorer.score_intent(log)
    assert result.score < 0.5
    assert result.matched_goal is False


def test_score_planning_efficient(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    result = scorer.score_planning(minimal_trajectory)
    # 3-step trajectory → step_efficiency = 1.0 > 0.6 → score = 0.80 > 0.7
    assert result.step_efficiency > 0.6
    assert result.score > 0.7


def test_score_planning_inefficient(
    scorer: TrajectoryScorer, over_engineered_trajectory: WearableLog
) -> None:
    result = scorer.score_planning(over_engineered_trajectory)
    # 8 steps → step_efficiency = 3/8 = 0.375 < 0.6 → score = 0.55
    assert result.step_efficiency < 0.6
    assert result.score == pytest.approx(0.55)


def test_score_tool_calls_all_valid(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    result = scorer.score_tool_calls(minimal_trajectory)
    assert result.precision == pytest.approx(1.0)
    assert result.false_positives == 0


def test_score_recovery_no_escalation(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    result = scorer.score_recovery(minimal_trajectory)
    assert result.had_error is False
    assert result.score is None


def test_score_recovery_with_escalation(
    scorer: TrajectoryScorer, escalation_trajectory: WearableLog
) -> None:
    result = scorer.score_recovery(escalation_trajectory)
    assert result.had_error is True
    assert result.score == pytest.approx(0.70)


def test_score_outcome_terminal_action(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    result = scorer.score_outcome(minimal_trajectory)
    assert result.goal_achieved is True
    assert result.score == pytest.approx(1.0)


def test_aggregate_weights_sum_to_1(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    """When recovery is None, remaining weights renormalize; verify the math."""
    ts = scorer.score_trajectory(minimal_trajectory)
    # recovery.score is None for minimal_trajectory (no escalation)
    assert ts.recovery.score is None

    active_weights = {k: v for k, v in _DEFAULT_WEIGHTS.items() if k != "recovery"}
    total_w = sum(active_weights.values())
    expected = (
        ts.intent.score * _DEFAULT_WEIGHTS["intent"]
        + ts.planning.score * _DEFAULT_WEIGHTS["planning"]
        + ts.tool_calls.score * _DEFAULT_WEIGHTS["tool_calls"]
        + ts.outcome.score * _DEFAULT_WEIGHTS["outcome"]
    ) / total_w
    assert ts.weighted_total == pytest.approx(expected, abs=1e-9)


def test_score_trajectory_returns_all_layers(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    from src.eval.trajectory_scorer import (
        IntentScore,
        OutcomeScore,
        PlanningScore,
        RecoveryScore,
        ToolCallScore,
    )

    ts = scorer.score_trajectory(minimal_trajectory)
    assert isinstance(ts.intent, IntentScore)
    assert isinstance(ts.planning, PlanningScore)
    assert isinstance(ts.tool_calls, ToolCallScore)
    assert isinstance(ts.recovery, RecoveryScore)
    assert isinstance(ts.outcome, OutcomeScore)


def test_pia_dimensions_keys(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    dims = scorer.score_pia_dimensions(minimal_trajectory)
    assert set(dims.keys()) == {
        "planning_quality",
        "error_recovery",
        "goal_alignment",
        "tool_precision",
    }


def test_nondeterminism_variance_keys(
    scorer: TrajectoryScorer, three_run_batch: list[WearableLog]
) -> None:
    result = scorer.compute_nondeterminism_variance("task-1", three_run_batch)
    assert set(result.keys()) == {
        "score_std",
        "pia_planning_std",
        "pia_recovery_std",
        "pia_goal_std",
        "pia_tool_std",
        "max_variance_layer",
    }


def test_batch_score_length(
    scorer: TrajectoryScorer, three_run_batch: list[WearableLog]
) -> None:
    results = scorer.batch_score(three_run_batch)
    assert len(results) == len(three_run_batch)


def test_weighted_total_range(
    scorer: TrajectoryScorer, minimal_trajectory: WearableLog
) -> None:
    ts = scorer.score_trajectory(minimal_trajectory)
    assert 0.0 <= ts.weighted_total <= 1.0
