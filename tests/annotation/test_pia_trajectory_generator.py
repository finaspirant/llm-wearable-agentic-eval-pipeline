"""Tests for src/annotation/pia_trajectory_generator.py.

No network or API calls — the generator is fully deterministic given a seed.
All tests use the default seed=42.

Test inventory:
  TestPairCount              — generate_all_pairs() returns exactly 10 pairs
  TestPairSchema             — TrajectoryPair fields present and typed correctly
  TestAgentTrajectories      — agent_a is direct (3 steps), agent_b is indirect (4–5)
  TestStepSchema             — each PairStep has all required fields
  TestStepTypes              — direct path has only standard steps; indirect has detours
  TestSensorContext          — sensor_context contains all expected keys
  TestDeterminism            — same seed → identical output on two independent runs
  TestSeedVariance           — different seeds → different sensor readings
  TestScenarioCoverage       — all 5 scenario types present in 10 pairs (2 each)
  TestConsentModels          — consent models are valid enum values
  TestTerminalActions        — both agents share the same terminal_action
  TestGoalAchieved           — both agents always achieve the goal
  TestStepConfidence         — confidence values are floats in [0.50, 0.99]
  TestSaveAndLoad            — save_pairs() writes valid JSON files to output_dir
  TestSavedFileCount         — save_pairs() creates exactly 10 numbered files
  TestSavedFileNaming        — files named pair_01.json … pair_10.json
  TestFormatMapSafety        — no KeyError from unresolved {placeholder} tokens
  TestDPNoiseApplied         — DP-noised sensor values differ from raw baselines

Run:
    pytest tests/annotation/test_pia_trajectory_generator.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.annotation.pia_trajectory_generator import (
    _SCENARIO_GPS,
    AgentTrajectory,
    PIATrajectoryGenerator,
    TrajectoryPair,
    _make_format_context,
)
from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import ScenarioType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_CONSENT_VALUES = {cm.value for cm in ConsentModel}
_VALID_SCENARIO_VALUES = {s.value for s in ScenarioType}
_VALID_ACTIONS = {
    "send_alert",
    "suppress_capture",
    "trigger_geofence",
    "adjust_noise_profile",
    "surface_reminder",
    "log_and_monitor",
    "request_consent",
    "escalate_to_emergency",
    # detour steps use no-op actions
    "",
}


@pytest.fixture(scope="module")
def generator() -> PIATrajectoryGenerator:
    return PIATrajectoryGenerator(seed=42)


@pytest.fixture(scope="module")
def pairs(generator: PIATrajectoryGenerator) -> list[TrajectoryPair]:
    return generator.generate_all_pairs()


# ---------------------------------------------------------------------------
# TestPairCount
# ---------------------------------------------------------------------------


class TestPairCount:
    def test_exactly_ten_pairs(self, pairs: list[TrajectoryPair]) -> None:
        assert len(pairs) == 10

    def test_pair_ids_sequential(self, pairs: list[TrajectoryPair]) -> None:
        ids = [p.pair_id for p in pairs]
        assert ids == [f"{i:02d}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# TestPairSchema
# ---------------------------------------------------------------------------


class TestPairSchema:
    def test_required_string_fields(self, pairs: list[TrajectoryPair]) -> None:
        required = (
            "pair_id",
            "scenario",
            "goal",
            "consent_model",
            "ground_truth_outcome",
            "shared_terminal_action",
            "path_divergence_description",
            "standard_kappa_prediction",
            "pia_rubric_prediction",
        )
        for pair in pairs:
            d = pair.to_dict()
            for field in required:
                assert field in d, f"{pair.pair_id} missing field: {field}"
                assert isinstance(d[field], str), (
                    f"{pair.pair_id}.{field} must be str, got {type(d[field])}"
                )

    def test_sensor_context_is_dict(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert isinstance(pair.sensor_context, dict), pair.pair_id

    def test_agent_a_and_b_present(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert isinstance(pair.agent_a, AgentTrajectory), pair.pair_id
            assert isinstance(pair.agent_b, AgentTrajectory), pair.pair_id

    def test_to_dict_is_json_serialisable(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            try:
                json.dumps(pair.to_dict())
            except (TypeError, ValueError) as exc:
                pytest.fail(f"{pair.pair_id} not JSON-serialisable: {exc}")


# ---------------------------------------------------------------------------
# TestAgentTrajectories
# ---------------------------------------------------------------------------


class TestAgentTrajectories:
    def test_agent_a_is_direct(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.agent_a.path_style == "direct", pair.pair_id
            n = pair.agent_a.n_steps
            assert n == 3, f"{pair.pair_id} agent_a should have 3 steps, got {n}"

    def test_agent_b_is_indirect(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.agent_b.path_style == "indirect", pair.pair_id
            n = pair.agent_b.n_steps
            assert n in {4, 5}, f"{pair.pair_id} agent_b should have 4–5 steps, got {n}"

    def test_agent_b_longer_than_a(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.agent_b.n_steps > pair.agent_a.n_steps, pair.pair_id

    def test_agent_ids(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.agent_a.agent_id == "agent_a", pair.pair_id
            assert pair.agent_b.agent_id == "agent_b", pair.pair_id

    def test_step_count_matches_list_length(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                assert len(traj.steps) == traj.n_steps, (
                    f"{pair.pair_id}/{traj.agent_id}: n_steps={traj.n_steps} "
                    f"but len(steps)={len(traj.steps)}"
                )


# ---------------------------------------------------------------------------
# TestStepSchema
# ---------------------------------------------------------------------------


class TestStepSchema:
    _REQUIRED_KEYS = {
        "step_index",
        "step_name",
        "step_type",
        "observation",
        "reasoning",
        "action",
        "confidence",
    }

    def test_all_step_fields_present(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    missing = self._REQUIRED_KEYS - set(step.to_dict())
                    assert not missing, (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        f"missing: {missing}"
                    )

    def test_step_index_zero_based_sequential(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                indices = [s.step_index for s in traj.steps]
                assert indices == list(range(len(traj.steps))), (
                    f"{pair.pair_id}/{traj.agent_id} non-sequential indices: {indices}"
                )

    def test_step_name_nonempty(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    assert step.step_name, (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        "has empty step_name"
                    )

    def test_observation_nonempty(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    assert step.observation.strip(), (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        "has empty observation"
                    )

    def test_reasoning_nonempty(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    assert step.reasoning.strip(), (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        "has empty reasoning"
                    )


# ---------------------------------------------------------------------------
# TestStepTypes
# ---------------------------------------------------------------------------


class TestStepTypes:
    def test_direct_path_has_only_standard_steps(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        for pair in pairs:
            for step in pair.agent_a.steps:
                assert step.step_type == "standard", (
                    f"{pair.pair_id} agent_a step {step.step_index} "
                    f"has step_type={step.step_type!r}, expected 'standard'"
                )

    def test_indirect_path_has_at_least_one_detour(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        for pair in pairs:
            detour_steps = [s for s in pair.agent_b.steps if s.step_type == "detour"]
            assert detour_steps, f"{pair.pair_id} agent_b has no detour steps"

    def test_indirect_path_ends_with_standard_step(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        for pair in pairs:
            last = pair.agent_b.steps[-1]
            assert last.step_type == "standard", (
                f"{pair.pair_id} agent_b final step has step_type={last.step_type!r}"
            )

    def test_valid_step_type_values(self, pairs: list[TrajectoryPair]) -> None:
        valid = {"standard", "detour"}
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    assert step.step_type in valid, (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        f"invalid step_type={step.step_type!r}"
                    )


# ---------------------------------------------------------------------------
# TestSensorContext
# ---------------------------------------------------------------------------


class TestSensorContext:
    _REQUIRED_KEYS = {
        "heart_rate_noised",
        "spo2_noised",
        "steps",
        "noise_db",
        "skin_temp_c",
        "gps_lat_noised",
        "gps_lon_noised",
        "audio_text",
        "audio_keywords",
        "audio_confidence",
        "activity",
        "consent_model",
    }

    def test_all_sensor_keys_present(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            missing = self._REQUIRED_KEYS - set(pair.sensor_context)
            assert not missing, f"{pair.pair_id} sensor_context missing: {missing}"

    def test_numeric_fields_are_finite(self, pairs: list[TrajectoryPair]) -> None:
        numeric_keys = {
            "heart_rate_noised",
            "spo2_noised",
            "steps",
            "noise_db",
            "skin_temp_c",
            "gps_lat_noised",
            "gps_lon_noised",
        }
        for pair in pairs:
            for k in numeric_keys:
                val = pair.sensor_context[k]
                assert math.isfinite(float(val)), (
                    f"{pair.pair_id} sensor_context[{k!r}]={val} is not finite"
                )

    def test_steps_nonnegative(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.sensor_context["steps"] >= 0.0, pair.pair_id

    def test_audio_keywords_is_list(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert isinstance(pair.sensor_context["audio_keywords"], list), pair.pair_id

    def test_consent_model_valid(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.sensor_context["consent_model"] in _VALID_CONSENT_VALUES, (
                f"{pair.pair_id} invalid consent_model: "
                f"{pair.sensor_context['consent_model']!r}"
            )


# ---------------------------------------------------------------------------
# TestDeterminism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        g1 = PIATrajectoryGenerator(seed=42)
        g2 = PIATrajectoryGenerator(seed=42)
        p1 = g1.generate_all_pairs()
        p2 = g2.generate_all_pairs()
        for a, b in zip(p1, p2, strict=True):
            assert a.to_dict() == b.to_dict(), f"{a.pair_id} output differs"

    def test_deterministic_sensor_readings(self) -> None:
        g = PIATrajectoryGenerator(seed=99)
        run1 = g.generate_all_pairs()
        g2 = PIATrajectoryGenerator(seed=99)
        run2 = g2.generate_all_pairs()
        for a, b in zip(run1, run2, strict=True):
            assert a.sensor_context == b.sensor_context, f"{a.pair_id} sensor diverged"


# ---------------------------------------------------------------------------
# TestSeedVariance
# ---------------------------------------------------------------------------


class TestSeedVariance:
    def test_different_seeds_produce_different_readings(self) -> None:
        g1 = PIATrajectoryGenerator(seed=1)
        g2 = PIATrajectoryGenerator(seed=2)
        p1 = g1.generate_all_pairs()
        p2 = g2.generate_all_pairs()
        diffs = sum(
            1
            for a, b in zip(p1, p2)
            if a.sensor_context["heart_rate_noised"]
            != b.sensor_context["heart_rate_noised"]
        )
        assert diffs > 0, (
            "seed=1 and seed=2 produced identical heart_rate_noised values"
        )


# ---------------------------------------------------------------------------
# TestScenarioCoverage
# ---------------------------------------------------------------------------


class TestScenarioCoverage:
    def test_all_five_scenarios_present(self, pairs: list[TrajectoryPair]) -> None:
        scenarios = {p.scenario for p in pairs}
        assert scenarios == _VALID_SCENARIO_VALUES, (
            f"Missing scenarios: {_VALID_SCENARIO_VALUES - scenarios}"
        )

    def test_two_pairs_per_scenario(self, pairs: list[TrajectoryPair]) -> None:
        from collections import Counter

        counts = Counter(p.scenario for p in pairs)
        for scenario, count in counts.items():
            assert count == 2, f"Expected 2 pairs for {scenario}, got {count}"


# ---------------------------------------------------------------------------
# TestConsentModels
# ---------------------------------------------------------------------------


class TestConsentModels:
    def test_consent_model_valid_enum_value(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.consent_model in _VALID_CONSENT_VALUES, (
                f"{pair.pair_id} invalid consent_model: {pair.consent_model!r}"
            )

    def test_sensor_context_consent_matches_pair(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        for pair in pairs:
            assert pair.sensor_context["consent_model"] == pair.consent_model, (
                f"{pair.pair_id} consent mismatch: "
                f"pair={pair.consent_model!r}, "
                f"context={pair.sensor_context['consent_model']!r}"
            )


# ---------------------------------------------------------------------------
# TestTerminalActions
# ---------------------------------------------------------------------------


class TestTerminalActions:
    def test_shared_terminal_action_nonempty(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.shared_terminal_action, pair.pair_id

    def test_agent_a_terminal_matches_shared(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            a_terminal = pair.agent_a.terminal_action
            assert a_terminal == pair.shared_terminal_action, (
                f"{pair.pair_id} agent_a.terminal_action={a_terminal!r} != "
                f"shared={pair.shared_terminal_action!r}"
            )

    def test_agent_b_terminal_matches_shared(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            b_terminal = pair.agent_b.terminal_action
            assert b_terminal == pair.shared_terminal_action, (
                f"{pair.pair_id} agent_b.terminal_action={b_terminal!r} != "
                f"shared={pair.shared_terminal_action!r}"
            )

    def test_terminal_action_in_valid_set(self, pairs: list[TrajectoryPair]) -> None:
        valid = {
            "send_alert",
            "suppress_capture",
            "trigger_geofence",
            "adjust_noise_profile",
            "surface_reminder",
            "log_and_monitor",
            "request_consent",
            "escalate_to_emergency",
        }
        for pair in pairs:
            assert pair.shared_terminal_action in valid, (
                f"{pair.pair_id} unrecognised terminal_action: "
                f"{pair.shared_terminal_action!r}"
            )


# ---------------------------------------------------------------------------
# TestGoalAchieved
# ---------------------------------------------------------------------------


class TestGoalAchieved:
    def test_agent_a_goal_achieved(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.agent_a.overall_goal_achieved is True, (
                f"{pair.pair_id} agent_a.overall_goal_achieved is not True"
            )

    def test_agent_b_goal_achieved(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            assert pair.agent_b.overall_goal_achieved is True, (
                f"{pair.pair_id} agent_b.overall_goal_achieved is not True"
            )

    def test_session_outcome_success(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                assert traj.session_outcome == "success", (
                    f"{pair.pair_id}/{traj.agent_id} session_outcome="
                    f"{traj.session_outcome!r}"
                )


# ---------------------------------------------------------------------------
# TestStepConfidence
# ---------------------------------------------------------------------------


class TestStepConfidence:
    def test_confidence_in_valid_range(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    assert 0.50 <= step.confidence <= 0.99, (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        f"confidence={step.confidence:.3f} out of [0.50, 0.99]"
                    )

    def test_confidence_is_float(self, pairs: list[TrajectoryPair]) -> None:
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    assert isinstance(step.confidence, float), (
                        f"{pair.pair_id}/{traj.agent_id} step {step.step_index} "
                        f"confidence is {type(step.confidence)}"
                    )


# ---------------------------------------------------------------------------
# TestSaveAndLoad
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    def test_saved_json_is_valid(self, tmp_path: Path) -> None:
        g = PIATrajectoryGenerator(seed=42, output_dir=tmp_path)
        pairs = g.generate_all_pairs()
        paths = g.save_pairs(pairs)
        for p in paths:
            raw = p.read_text()
            obj = json.loads(raw)
            assert isinstance(obj, dict), f"{p.name} is not a JSON object"

    def test_loaded_pair_id_matches_filename(self, tmp_path: Path) -> None:
        g = PIATrajectoryGenerator(seed=42, output_dir=tmp_path)
        pairs = g.generate_all_pairs()
        paths = g.save_pairs(pairs)
        for p in paths:
            obj = json.loads(p.read_text())
            # file is pair_01.json; pair_id inside is "01"
            numeric_id = p.stem.replace("pair_", "")  # "01" etc.
            assert obj["pair_id"] == numeric_id, (
                f"{p.name}: pair_id={obj['pair_id']!r} != {numeric_id!r}"
            )

    def test_round_trip_preserves_agent_steps(self, tmp_path: Path) -> None:
        g = PIATrajectoryGenerator(seed=42, output_dir=tmp_path)
        pairs = g.generate_all_pairs()
        paths = g.save_pairs(pairs)
        for pair, path in zip(pairs, paths, strict=True):
            obj = json.loads(path.read_text())
            assert len(obj["agent_a"]["steps"]) == pair.agent_a.n_steps
            assert len(obj["agent_b"]["steps"]) == pair.agent_b.n_steps


# ---------------------------------------------------------------------------
# TestSavedFileCount
# ---------------------------------------------------------------------------


class TestSavedFileCount:
    def test_exactly_ten_files_written(self, tmp_path: Path) -> None:
        g = PIATrajectoryGenerator(seed=42, output_dir=tmp_path)
        paths = g.save_pairs(g.generate_all_pairs())
        assert len(paths) == 10

    def test_output_dir_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        assert not nested.exists()
        g = PIATrajectoryGenerator(seed=42, output_dir=nested)
        g.save_pairs(g.generate_all_pairs())
        assert nested.is_dir()


# ---------------------------------------------------------------------------
# TestSavedFileNaming
# ---------------------------------------------------------------------------


class TestSavedFileNaming:
    def test_file_names_match_pattern(self, tmp_path: Path) -> None:
        g = PIATrajectoryGenerator(seed=42, output_dir=tmp_path)
        paths = g.save_pairs(g.generate_all_pairs())
        names = sorted(p.name for p in paths)
        expected = [f"pair_{i:02d}.json" for i in range(1, 11)]
        assert names == expected


# ---------------------------------------------------------------------------
# TestFormatMapSafety
# ---------------------------------------------------------------------------


class TestFormatMapSafety:
    """Verify that _FormatContext silently passes through unknown placeholders."""

    def test_known_keys_resolve(self) -> None:
        ctx: dict[str, Any] = {
            "heart_rate_noised": 75.0,
            "spo2_noised": 97.0,
            "steps": 300.0,
            "noise_db": 55.0,
            "skin_temp_c": 36.6,
            "gps_lat_noised": 37.7749,
            "gps_lon_noised": -122.4194,
            "audio_text": "test audio",
            "audio_keywords": ["keyword"],
            "audio_confidence": 0.85,
            "activity": "walking",
            "environment": "outdoor",
            "consent_model": "explicit",
        }
        fmt = _make_format_context(ctx)
        result = "{hr} bpm, {spo2}% SpO2, {steps} steps".format_map(fmt)
        assert "75.0 bpm" in result
        assert "97.0% SpO2" in result

    def test_unknown_key_preserved_literally(self) -> None:
        ctx: dict[str, Any] = {
            "heart_rate_noised": 75.0,
            "spo2_noised": 97.0,
            "steps": 0.0,
            "noise_db": 50.0,
            "skin_temp_c": 36.5,
            "gps_lat_noised": 0.0,
            "gps_lon_noised": 0.0,
            "audio_text": "",
            "audio_keywords": [],
            "audio_confidence": 0.0,
            "activity": "still",
            "environment": None,
            "consent_model": "explicit",
        }
        fmt = _make_format_context(ctx)
        # {unknown_field} should NOT raise KeyError — it should be returned as-is
        result = "value is {unknown_field}".format_map(fmt)
        assert result == "value is {unknown_field}"

    def test_no_format_spec_error_in_all_templates(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        """All instantiated observation/reasoning strings must be non-empty."""
        for pair in pairs:
            for traj in (pair.agent_a, pair.agent_b):
                for step in traj.steps:
                    # Observations should not contain unresolved {placeholders}
                    has_open = "{" in step.observation
                    has_close = "}" in step.observation
                    loc = f"{pair.pair_id}/{traj.agent_id} step {step.step_index}"
                    assert not (has_open and has_close), (
                        f"{loc} has unresolved placeholder in observation"
                    )


# ---------------------------------------------------------------------------
# TestDPNoiseApplied
# ---------------------------------------------------------------------------


class TestDPNoiseApplied:
    """GPS and sensor values differ from baselines due to DP noise."""

    def test_gps_differs_from_exact_baseline(self, pairs: list[TrajectoryPair]) -> None:
        """At seed=42, DP GPS noise moves lat/lon away from exact baseline values."""
        exact_matches = 0
        for pair in pairs:
            scenario = ScenarioType(pair.scenario)
            lat_base, lon_base = _SCENARIO_GPS[scenario]
            lat_noised = pair.sensor_context["gps_lat_noised"]
            lon_noised = pair.sensor_context["gps_lon_noised"]
            if lat_noised == lat_base and lon_noised == lon_base:
                exact_matches += 1
        # With DP noise (σ=0.001°), probability of exact match is essentially zero
        assert exact_matches == 0, (
            f"{exact_matches}/10 pairs had GPS coordinates matching base exactly"
        )

    def test_heart_rate_plausible_given_dp_noise(
        self, pairs: list[TrajectoryPair]
    ) -> None:
        """At ε=1.0, σ≈48 bpm; [-100, 300] bpm range almost never violated."""
        for pair in pairs:
            hr = pair.sensor_context["heart_rate_noised"]
            assert math.isfinite(hr), f"{pair.pair_id} HR={hr} is not finite"
            assert -100.0 <= hr <= 300.0, (
                f"{pair.pair_id} HR={hr:.1f} outside sanity range (DP noise too large?)"
            )
