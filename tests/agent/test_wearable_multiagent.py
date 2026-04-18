"""Tests for src.agent.wearable_multiagent.

All tests are purely heuristic (no LLM API calls). Fixtures construct
WearableLogs with controlled sensor values so threshold-driven assertions
are deterministic.
"""

from __future__ import annotations

import json

import pytest

from src.agent.wearable_multiagent import (
    AgentRoleType,
    MultiAgentPipeline,
    MultiAgentResult,
    RoleAnnotation,
)
from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import (
    AgentAction,
    AudioTranscript,
    ScenarioType,
    SensorData,
    TrajectoryStep,
    WearableLog,
)
from src.eval.role_attribution import RoleAttributionScorer

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_sensor(
    hr: float = 75.0,
    spo2: float = 98.0,
    noise_db: float = 55.0,
    gps_lat: float = 37.7749,
    gps_lon: float = -122.4194,
) -> SensorData:
    """Return a SensorData with both raw and noised fields set identically.

    Setting noised == raw avoids the need for a live PrivacyGate and makes
    threshold assertions on hr_noised / spo2_noised deterministic.
    """
    return SensorData(
        heart_rate=hr,
        spo2=spo2,
        steps=200.0,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        noise_db=noise_db,
        skin_temp_c=36.6,
        heart_rate_noised=hr,
        spo2_noised=spo2,
        steps_noised=200.0,
        noise_db_noised=noise_db,
        gps_lat_noised=gps_lat,
        gps_lon_noised=gps_lon,
    )


def _make_audio(keywords: list[str] | None = None) -> AudioTranscript:
    return AudioTranscript(
        text=" ".join(keywords or []),
        language="en-US",
        confidence=0.95,
        duration_s=2.0,
        keywords_detected=keywords or [],
    )


def _make_step(index: int, name: str, action: str = "") -> TrajectoryStep:
    return TrajectoryStep(
        step_index=index,
        step_name=name,
        observation=f"obs_{name}",
        reasoning=f"reason_{name}",
        action=action,
        confidence=0.9,
    )


def _make_log(
    scenario: ScenarioType,
    consent: ConsentModel = ConsentModel.EXPLICIT,
    hr: float = 75.0,
    spo2: float = 98.0,
    noise_db: float = 55.0,
    keywords: list[str] | None = None,
    log_id: str = "test-log-0001",
) -> WearableLog:
    """Construct a minimal WearableLog with controlled sensor values."""
    return WearableLog(
        log_id=log_id,
        timestamp="2026-04-17T12:00:00Z",
        scenario_type=scenario,
        consent_model=consent,
        sensor_data=_make_sensor(hr=hr, spo2=spo2, noise_db=noise_db),
        audio_transcript=_make_audio(keywords),
        context_metadata={"device": "test-wearable"},
        trajectory=[
            _make_step(0, "sense"),
            _make_step(1, "plan"),
            _make_step(2, "act", action=AgentAction.LOG_AND_MONITOR.value),
        ],
        ground_truth_action=AgentAction.SEND_ALERT.value,
    )


# ---------------------------------------------------------------------------
# Shared pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> MultiAgentPipeline:
    """Instantiate the pipeline once per module (graph compile is slow)."""
    return MultiAgentPipeline()


# ---------------------------------------------------------------------------
# 1. test_orchestrator_routes_health_alert
# ---------------------------------------------------------------------------


class TestOrchestratorRoutesHealthAlert:
    """Verify that a health_alert log is routed to HealthAgent."""

    @pytest.fixture
    def result(self, pipeline: MultiAgentPipeline) -> MultiAgentResult:
        # HR=145 exceeds _HR_ALERT_THRESHOLD (140) → SEND_ALERT territory.
        # spo2=92 is at _SPO2_ALERT_THRESHOLD (92) — not strictly above it,
        # so only HR drives the alert path.
        log = _make_log(
            scenario=ScenarioType.HEALTH_ALERT,
            hr=145.0,
            spo2=92.0,
        )
        return pipeline.run(log)

    def test_health_agent_in_role_annotations(
        self, result: MultiAgentResult
    ) -> None:
        roles = [a.agent_role for a in result.role_annotations]
        assert "health_agent" in roles

    def test_final_action_is_health_terminal(
        self, result: MultiAgentResult
    ) -> None:
        assert result.final_action in {
            AgentAction.SEND_ALERT,
            AgentAction.ESCALATE_TO_EMERGENCY,
        }

    def test_orchestrator_has_handoff_quality(
        self, result: MultiAgentResult
    ) -> None:
        orchestrator = next(
            (a for a in result.role_annotations if a.agent_role == "orchestrator"),
            None,
        )
        assert orchestrator is not None, "No orchestrator in role_annotations"
        assert orchestrator.handoff_quality is not None

    def test_trajectory_has_three_steps(self, result: MultiAgentResult) -> None:
        assert len(result.trajectory) == 3

    def test_step_names_are_sense_plan_act(
        self, result: MultiAgentResult
    ) -> None:
        names = [s.step_name for s in result.trajectory]
        assert names == ["sense", "plan", "act"]


# ---------------------------------------------------------------------------
# 2. test_privacy_gate_blocks_revoked_consent
# ---------------------------------------------------------------------------


class TestPrivacyGateBlocksRevokedConsent:
    """Verify that REVOKED consent routes to PrivacyGateAgent → suppress."""

    @pytest.fixture
    def result(self, pipeline: MultiAgentPipeline) -> MultiAgentResult:
        log = _make_log(
            scenario=ScenarioType.PRIVACY_SENSITIVE,
            consent=ConsentModel.REVOKED,
        )
        return pipeline.run(log)

    def test_privacy_gate_agent_in_role_annotations(
        self, result: MultiAgentResult
    ) -> None:
        roles = [a.agent_role for a in result.role_annotations]
        assert "privacy_gate_agent" in roles

    def test_final_action_suppresses_or_requests_consent(
        self, result: MultiAgentResult
    ) -> None:
        # REVOKED consent must resolve to suppress_capture per the ConsentModel
        # decision matrix (priority 1: categorical enforcement).
        assert result.final_action in {
            AgentAction.SUPPRESS_CAPTURE,
            AgentAction.REQUEST_CONSENT,
        }

    def test_final_action_is_suppress_for_revoked(
        self, result: MultiAgentResult
    ) -> None:
        # Stricter assertion: REVOKED → suppress_capture specifically.
        assert result.final_action == AgentAction.SUPPRESS_CAPTURE

    def test_privacy_gate_authority_appropriate(
        self, result: MultiAgentResult
    ) -> None:
        privacy_agent = next(
            (
                a
                for a in result.role_annotations
                if a.agent_role == "privacy_gate_agent"
            ),
            None,
        )
        assert privacy_agent is not None, "No privacy_gate_agent in role_annotations"
        assert privacy_agent.authority_appropriate is True

    def test_privacy_gate_has_no_handoff_quality(
        self, result: MultiAgentResult
    ) -> None:
        """Non-orchestrator roles must not carry handoff_quality per schema."""
        privacy_agent = next(
            a
            for a in result.role_annotations
            if a.agent_role == "privacy_gate_agent"
        )
        assert privacy_agent.handoff_quality is None
        # Confirm to_dict() also omits the field (schema compliance).
        assert "handoff_quality" not in privacy_agent.to_dict()


# ---------------------------------------------------------------------------
# 3. test_attribution_report_cascade_risk_flag
# ---------------------------------------------------------------------------


class TestAttributionReportCascadeRiskFlag:
    """Verify cascade_risk is raised when no agent owns a failure."""

    @pytest.fixture
    def all_unaccountable_annotations(self) -> list[RoleAnnotation]:
        """Inject a result where every agent has accountability_clear=False."""
        return [
            RoleAnnotation(
                agent_id="orchestrator_01",
                agent_role="orchestrator",
                delegation_quality=3,
                authority_appropriate=True,
                accountability_clear=False,
                handoff_quality=3,
            ),
            RoleAnnotation(
                agent_id="health_agent_01",
                agent_role="health_agent",
                delegation_quality=3,
                authority_appropriate=True,
                accountability_clear=False,
            ),
        ]

    def test_cascade_risk_true_when_none_accountable(
        self, all_unaccountable_annotations: list[RoleAnnotation]
    ) -> None:
        scorer = RoleAttributionScorer()
        report = scorer.score(
            all_unaccountable_annotations, goal_achieved=False
        )
        assert report.cascade_risk is True

    def test_cascade_risk_false_when_goal_achieved(
        self, all_unaccountable_annotations: list[RoleAnnotation]
    ) -> None:
        scorer = RoleAttributionScorer()
        report = scorer.score(
            all_unaccountable_annotations, goal_achieved=True
        )
        assert report.cascade_risk is False

    def test_cascade_risk_false_when_one_agent_accountable(self) -> None:
        annotations = [
            RoleAnnotation(
                agent_id="orchestrator_01",
                agent_role="orchestrator",
                delegation_quality=4,
                authority_appropriate=True,
                accountability_clear=False,
                handoff_quality=4,
            ),
            RoleAnnotation(
                agent_id="privacy_gate_01",
                agent_role="privacy_gate_agent",
                delegation_quality=4,
                authority_appropriate=True,
                accountability_clear=True,  # this one owns it
            ),
        ]
        scorer = RoleAttributionScorer()
        report = scorer.score(annotations, goal_achieved=False)
        assert report.cascade_risk is False

    def test_accountability_coverage_zero_for_failed_unaccountable(
        self, all_unaccountable_annotations: list[RoleAnnotation]
    ) -> None:
        scorer = RoleAttributionScorer()
        report = scorer.score(
            all_unaccountable_annotations, goal_achieved=False
        )
        assert report.accountability_coverage == 0.0

    def test_scorer_raises_on_empty_annotations(self) -> None:
        scorer = RoleAttributionScorer()
        with pytest.raises(ValueError, match="at least one record"):
            scorer.score([], goal_achieved=False)


# ---------------------------------------------------------------------------
# 4. test_multiagent_result_json_serialisable
# ---------------------------------------------------------------------------


class TestMultiAgentResultJsonSerialisable:
    """Verify MultiAgentResult.to_dict() produces JSON-safe output."""

    @pytest.fixture
    def result(self, pipeline: MultiAgentPipeline) -> MultiAgentResult:
        log = _make_log(
            scenario=ScenarioType.CALENDAR_REMINDER,
            consent=ConsentModel.EXPLICIT,
        )
        return pipeline.run(log)

    def test_to_dict_raises_no_exception(self, result: MultiAgentResult) -> None:
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_json_dumps_raises_no_exception(
        self, result: MultiAgentResult
    ) -> None:
        serialised = json.dumps(result.to_dict())
        assert isinstance(serialised, str)
        assert len(serialised) > 0

    def test_json_round_trip_preserves_log_id(
        self, result: MultiAgentResult
    ) -> None:
        d = json.loads(json.dumps(result.to_dict()))
        assert d["log_id"] == result.log_id

    def test_json_round_trip_preserves_final_action(
        self, result: MultiAgentResult
    ) -> None:
        d = json.loads(json.dumps(result.to_dict()))
        assert d["final_action"] == result.final_action.value

    def test_role_annotations_serialised_as_list(
        self, result: MultiAgentResult
    ) -> None:
        d = result.to_dict()
        assert isinstance(d["role_annotations"], list)
        assert len(d["role_annotations"]) >= 1

    def test_trajectory_serialised_as_list_of_dicts(
        self, result: MultiAgentResult
    ) -> None:
        d = result.to_dict()
        assert isinstance(d["trajectory"], list)
        for step in d["trajectory"]:
            assert isinstance(step, dict)
            assert "step_name" in step
            assert "action" in step

    def test_latency_ms_is_positive_float(
        self, result: MultiAgentResult
    ) -> None:
        d = result.to_dict()
        assert isinstance(d["latency_ms"], float)
        assert d["latency_ms"] >= 0.0


# ---------------------------------------------------------------------------
# 5. Additional edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Catch regressions in routing and trajectory structure."""

    def test_calendar_reminder_direct_route_produces_three_steps(
        self, pipeline: MultiAgentPipeline
    ) -> None:
        """calendar_reminder bypasses specialists; ActionAgent adds plan+act."""
        log = _make_log(scenario=ScenarioType.CALENDAR_REMINDER)
        result = pipeline.run(log)
        assert len(result.trajectory) == 3
        assert result.final_action == AgentAction.SURFACE_REMINDER

    def test_location_trigger_routes_to_privacy_gate(
        self, pipeline: MultiAgentPipeline
    ) -> None:
        log = _make_log(
            scenario=ScenarioType.LOCATION_TRIGGER,
            consent=ConsentModel.EXPLICIT,
        )
        result = pipeline.run(log)
        roles: list[AgentRoleType] = [a.agent_role for a in result.role_annotations]
        assert "privacy_gate_agent" in roles

    def test_ambient_noise_routes_to_health_agent(
        self, pipeline: MultiAgentPipeline
    ) -> None:
        log = _make_log(
            scenario=ScenarioType.AMBIENT_NOISE,
            noise_db=92.0,
        )
        result = pipeline.run(log)
        roles = [a.agent_role for a in result.role_annotations]
        assert "health_agent" in roles
        assert result.final_action == AgentAction.ADJUST_NOISE_PROFILE

    def test_escalate_emergency_when_dual_modality_triggered(
        self, pipeline: MultiAgentPipeline
    ) -> None:
        """HR > 160 + 'dizzy' audio keyword → escalate_to_emergency."""
        log = _make_log(
            scenario=ScenarioType.HEALTH_ALERT,
            hr=165.0,
            spo2=95.0,
            keywords=["dizzy"],
        )
        result = pipeline.run(log)
        assert result.final_action == AgentAction.ESCALATE_TO_EMERGENCY

    def test_result_has_two_role_annotations_for_specialist_path(
        self, pipeline: MultiAgentPipeline
    ) -> None:
        """Orchestrator + specialist = 2 annotations; calendar = 1."""
        health_log = _make_log(scenario=ScenarioType.HEALTH_ALERT, hr=145.0)
        result = pipeline.run(health_log)
        # orchestrator + health_agent + calendar_action_agent = 3
        assert len(result.role_annotations) >= 2
