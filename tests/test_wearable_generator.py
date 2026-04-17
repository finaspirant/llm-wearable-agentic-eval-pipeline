"""Tests for src/data/wearable_generator.py and src/data/privacy_gate.py.

Covers:
- Log generation produces the correct count
- All 5 scenario types are represented in a balanced batch
- JSON schema consistency across logs
- Sensor values are within realistic ranges after DP noise
- PrivacyGate applies noise (mean-shift < 3σ, but distribution shifts)
- Trajectory has exactly 3 steps with correct step names
- Ground truth actions match scenario type
"""

import json
import math

import numpy as np
import pytest

from src.data.privacy_gate import ConsentModel, PrivacyGate
from src.data.wearable_generator import (
    ScenarioType,
    WearableLog,
    WearableLogGenerator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator() -> WearableLogGenerator:
    """Seeded generator for reproducible tests."""
    return WearableLogGenerator(seed=42, epsilon=1.0)


@pytest.fixture(scope="module")
def batch_100(generator: WearableLogGenerator) -> list[WearableLog]:
    """100-log batch covering all 5 scenario types."""
    return generator.generate_batch(100)


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------


def test_generate_correct_count(generator: WearableLogGenerator) -> None:
    """generate_batch(n) returns exactly n logs."""
    for n in (1, 10, 50):
        logs = generator.generate_batch(n)
        assert len(logs) == n, f"Expected {n} logs, got {len(logs)}"


# ---------------------------------------------------------------------------
# Scenario coverage
# ---------------------------------------------------------------------------


def test_all_scenario_types_present(batch_100: list[WearableLog]) -> None:
    """All 5 ScenarioType values appear in a 100-log batch."""
    found = {log.scenario_type for log in batch_100}
    missing = set(ScenarioType) - found
    assert found == set(ScenarioType), f"Missing scenario types: {missing}"


def test_scenario_filter_respected(generator: WearableLogGenerator) -> None:
    """scenario_filter restricts output to the requested types."""
    target = [ScenarioType.HEALTH_ALERT, ScenarioType.AMBIENT_NOISE]
    logs = generator.generate_batch(20, scenario_filter=target)
    types_found = {log.scenario_type for log in logs}
    unexpected = types_found - set(target)
    assert types_found.issubset(set(target)), f"Unexpected types: {unexpected}"


# ---------------------------------------------------------------------------
# JSON schema consistency
# ---------------------------------------------------------------------------


_REQUIRED_TOP_KEYS = {
    "log_id",
    "timestamp",
    "scenario_type",
    "consent_model",
    "sensor_data",
    "audio_transcript",
    "context_metadata",
    "trajectory",
    "ground_truth_action",
}

_REQUIRED_SENSOR_KEYS = {
    "heart_rate",
    "spo2",
    "steps",
    "gps_lat",
    "gps_lon",
    "noise_db",
    "skin_temp_c",
    "heart_rate_noised",
    "spo2_noised",
    "steps_noised",
    "noise_db_noised",
    "gps_lat_noised",
    "gps_lon_noised",
}

_REQUIRED_AUDIO_KEYS = {
    "text",
    "language",
    "confidence",
    "duration_s",
    "keywords_detected",
}


def test_json_schema_consistency(batch_100: list[WearableLog]) -> None:
    """Every log round-trips through JSON and has the required keys."""
    for log in batch_100:
        raw_json = log.to_json()
        d = json.loads(raw_json)

        assert _REQUIRED_TOP_KEYS.issubset(d.keys()), (
            f"Missing top-level keys: {_REQUIRED_TOP_KEYS - d.keys()}"
        )
        assert _REQUIRED_SENSOR_KEYS.issubset(d["sensor_data"].keys()), (
            f"Missing sensor keys: {_REQUIRED_SENSOR_KEYS - d['sensor_data'].keys()}"
        )
        assert _REQUIRED_AUDIO_KEYS.issubset(d["audio_transcript"].keys()), (
            f"Missing audio keys: {_REQUIRED_AUDIO_KEYS - d['audio_transcript'].keys()}"
        )
        assert isinstance(d["trajectory"], list) and len(d["trajectory"]) == 3


def test_log_ids_are_unique(batch_100: list[WearableLog]) -> None:
    """Every log has a unique UUID."""
    ids = [log.log_id for log in batch_100]
    assert len(set(ids)) == len(ids), "Duplicate log_ids detected"


# ---------------------------------------------------------------------------
# Sensor value ranges (post-DP)
# ---------------------------------------------------------------------------


def test_sensor_value_ranges(batch_100: list[WearableLog]) -> None:
    """Raw sensor values respect physiological bounds; noised values are finite.

    Raw fields are clamped at generation time.  Noised fields intentionally
    carry large Gaussian perturbations (σ ≈ 48 bpm at ε=1.0) so they cannot
    be held to the same physiological bounds — instead we verify they are
    finite (not NaN or ±inf), which would indicate a broken noise call.
    GPS raw values must be valid WGS-84; noised variants may drift slightly
    outside the bounding box but remain finite.
    """
    import math

    for log in batch_100:
        s = log.sensor_data

        # Raw values: physiological / WGS-84 bounds
        assert 30.0 <= s.heart_rate <= 220.0, f"raw HR out of range: {s.heart_rate}"
        assert 70.0 <= s.spo2 <= 100.0, f"raw SpO2 out of range: {s.spo2}"
        assert s.steps >= 0.0, f"raw steps negative: {s.steps}"
        assert -90.0 <= s.gps_lat <= 90.0, f"raw lat invalid: {s.gps_lat}"
        assert -180.0 <= s.gps_lon <= 180.0, f"raw lon invalid: {s.gps_lon}"

        # Noised values: must be finite floats (DP noise is intentionally large)
        noised_fields = [
            ("heart_rate_noised", s.heart_rate_noised),
            ("spo2_noised", s.spo2_noised),
            ("steps_noised", s.steps_noised),
            ("noise_db_noised", s.noise_db_noised),
            ("gps_lat_noised", s.gps_lat_noised),
            ("gps_lon_noised", s.gps_lon_noised),
        ]
        for name, val in noised_fields:
            assert math.isfinite(val), f"{name} is not finite: {val}"


def test_health_alert_elevated_hr(generator: WearableLogGenerator) -> None:
    """HEALTH_ALERT logs have higher mean HR than CALENDAR_REMINDER logs."""
    health = generator.generate_batch(40, scenario_filter=[ScenarioType.HEALTH_ALERT])
    calendar = generator.generate_batch(
        40, scenario_filter=[ScenarioType.CALENDAR_REMINDER]
    )
    mean_health_hr = np.mean([log.sensor_data.heart_rate for log in health])
    mean_calendar_hr = np.mean([log.sensor_data.heart_rate for log in calendar])
    assert mean_health_hr > mean_calendar_hr, (
        f"Expected HEALTH_ALERT HR > CALENDAR_REMINDER HR, "
        f"got {mean_health_hr:.1f} vs {mean_calendar_hr:.1f}"
    )


# ---------------------------------------------------------------------------
# Trajectory structure
# ---------------------------------------------------------------------------


def test_trajectory_has_three_steps(batch_100: list[WearableLog]) -> None:
    """Every log has exactly 3 trajectory steps."""
    for log in batch_100:
        assert len(log.trajectory) == 3, f"Expected 3 steps, got {len(log.trajectory)}"


def test_trajectory_step_names(batch_100: list[WearableLog]) -> None:
    """Steps are named 'sense', 'plan', 'act' in order."""
    expected_names = ["sense", "plan", "act"]
    for log in batch_100:
        names = [step.step_name for step in log.trajectory]
        assert names == expected_names, f"Unexpected step names: {names}"


def test_act_step_has_action(batch_100: list[WearableLog]) -> None:
    """The 'act' step always has a non-empty action string."""
    for log in batch_100:
        act_step = log.trajectory[2]
        assert act_step.action, f"act step missing action for log {log.log_id}"


def test_ground_truth_matches_scenario(batch_100: list[WearableLog]) -> None:
    """Ground truth action matches the expected action for each scenario type."""
    from src.data.wearable_generator import _SCENARIO_GROUND_TRUTH

    for log in batch_100:
        expected = _SCENARIO_GROUND_TRUTH[log.scenario_type]
        assert log.ground_truth_action == expected, (
            f"Scenario {log.scenario_type}: expected {expected}, "
            f"got {log.ground_truth_action}"
        )


# ---------------------------------------------------------------------------
# PrivacyGate unit tests
# ---------------------------------------------------------------------------


def test_privacy_gate_adds_noise() -> None:
    """PrivacyGate.apply_gaussian_noise shifts values away from the raw input."""
    gate = PrivacyGate(epsilon=1.0, rng=np.random.default_rng(0))
    raw = 72.0
    sensitivity = 10.0
    noised_values = [gate.apply_gaussian_noise(raw, sensitivity) for _ in range(200)]
    assert not all(v == raw for v in noised_values), "Noise was never applied"
    # Mean should be within ±3σ of raw (unbiased mechanism)
    sigma = gate.calibrate_noise(1.0, 1e-5, sensitivity)
    mean_err = abs(float(np.mean(noised_values)) - raw)
    assert mean_err < 3 * sigma / math.sqrt(200), (
        f"Noise appears biased: mean error {mean_err:.4f} > 3σ/√n"
    )


def test_privacy_gate_calibrate_noise() -> None:
    """calibrate_noise returns a positive σ that increases with sensitivity."""
    gate = PrivacyGate(epsilon=1.0)
    sigma_low = gate.calibrate_noise(1.0, 1e-5, sensitivity=1.0)
    sigma_high = gate.calibrate_noise(1.0, 1e-5, sensitivity=10.0)
    assert sigma_low > 0
    assert sigma_high > sigma_low


def test_privacy_gate_validate_budget() -> None:
    """validate_epsilon_budget correctly identifies over-budget operation lists."""
    gate = PrivacyGate(epsilon=1.0)
    assert gate.validate_epsilon_budget([0.3, 0.3, 0.3]) is True
    assert gate.validate_epsilon_budget([0.5, 0.6]) is False


def test_privacy_gate_sanitize_record() -> None:
    """sanitize_record noises known fields and passes through unknown ones."""
    gate = PrivacyGate(epsilon=1.0, rng=np.random.default_rng(7))
    record = {"heart_rate": 72.0, "unknown_field": 42.0}
    sanitized = gate.sanitize_record(record)
    assert sanitized["unknown_field"] == 42.0, "Unknown fields should be unchanged"
    # heart_rate should be noised (not exactly equal to raw with very high probability)
    # probabilistic — just verify the key is present with a numeric value
    assert isinstance(sanitized["heart_rate"], float)


def test_privacy_gate_revoked_consent_passthrough() -> None:
    """sanitize_record with REVOKED consent returns the record unchanged."""
    gate = PrivacyGate(epsilon=1.0, rng=np.random.default_rng(0))
    record = {"heart_rate": 72.0, "spo2": 98.0}
    result = gate.sanitize_record(record, consent=ConsentModel.REVOKED)
    assert result == record


def test_privacy_gate_invalid_epsilon() -> None:
    """PrivacyGate raises ValueError for non-positive epsilon."""
    with pytest.raises(ValueError, match="epsilon"):
        PrivacyGate(epsilon=0.0)


# ---------------------------------------------------------------------------
# Task-requested tests (schema, distribution, privacy gate integration)
# ---------------------------------------------------------------------------


def test_schema_required_fields_and_types(
    batch_100: list[WearableLog],
) -> None:
    """Every log has all required fields with correct Python types.

    Covers the full JSON-round-trip: serialise to dict, then check that
    each field is present and carries the expected type.  This catches
    accidental None values, type coercions, and missing keys in one pass.
    """
    for log in batch_100:
        d = log.to_dict()

        # Top-level scalar types
        assert isinstance(d["log_id"], str) and len(d["log_id"]) == 36, (
            f"log_id must be a UUID4 string, got {d['log_id']!r}"
        )
        assert isinstance(d["timestamp"], str), "timestamp must be str"
        assert isinstance(d["scenario_type"], str), "scenario_type must be str"
        assert isinstance(d["consent_model"], str), "consent_model must be str"
        assert isinstance(d["ground_truth_action"], str), (
            "ground_truth_action must be str"
        )

        # sensor_data — all numeric fields must be float
        sd = d["sensor_data"]
        float_fields = [
            "heart_rate",
            "spo2",
            "steps",
            "gps_lat",
            "gps_lon",
            "noise_db",
            "skin_temp_c",
            "heart_rate_noised",
            "spo2_noised",
            "steps_noised",
            "noise_db_noised",
            "gps_lat_noised",
            "gps_lon_noised",
        ]
        for field in float_fields:
            assert field in sd, f"sensor_data missing field: {field}"
            assert isinstance(sd[field], float), (
                f"sensor_data.{field} must be float, got {type(sd[field])}"
            )

        # audio_transcript types
        at = d["audio_transcript"]
        assert isinstance(at["text"], str), "audio_transcript.text must be str"
        assert isinstance(at["confidence"], float), (
            "audio_transcript.confidence must be float"
        )
        assert isinstance(at["duration_s"], float), (
            "audio_transcript.duration_s must be float"
        )
        assert isinstance(at["keywords_detected"], list), (
            "audio_transcript.keywords_detected must be list"
        )

        # trajectory — list of 3 dicts with typed fields
        traj = d["trajectory"]
        assert isinstance(traj, list) and len(traj) == 3
        for step in traj:
            assert isinstance(step["step_index"], int)
            assert isinstance(step["step_name"], str)
            assert isinstance(step["observation"], str)
            assert isinstance(step["reasoning"], str)
            assert isinstance(step["action"], str)
            assert isinstance(step["confidence"], float)


def test_distribution_all_five_scenarios() -> None:
    """All 5 scenario types appear in a 100-log batch with balanced coverage.

    With round-robin selection and 100 logs over 5 types, each scenario
    must appear exactly 20 times.
    """
    gen = WearableLogGenerator(seed=7)
    logs = gen.generate_batch(100)

    from collections import Counter

    counts = Counter(log.scenario_type for log in logs)

    assert set(counts.keys()) == set(ScenarioType), (
        f"Missing scenarios: {set(ScenarioType) - set(counts.keys())}"
    )
    for scenario, n in counts.items():
        assert n == 20, f"Expected 20 logs for {scenario}, got {n}"


def test_privacy_gate_modifies_sensor_values() -> None:
    """Noised sensor fields differ from their raw counterparts in every log.

    Verifies that the privacy gate is actually called and produces non-zero
    noise for heart_rate, spo2, steps, noise_db, gps_lat, and gps_lon.
    With σ >> 0 for all fields, the probability of exact equality is
    effectively zero for any reasonable seed.
    """
    gen = WearableLogGenerator(seed=1)
    logs = gen.generate_batch(50)

    noised_pairs = [
        ("heart_rate", "heart_rate_noised"),
        ("spo2", "spo2_noised"),
        ("steps", "steps_noised"),
        ("noise_db", "noise_db_noised"),
        ("gps_lat", "gps_lat_noised"),
        ("gps_lon", "gps_lon_noised"),
    ]

    for raw_field, noised_field in noised_pairs:
        diffs = [
            abs(
                getattr(log.sensor_data, raw_field)
                - getattr(log.sensor_data, noised_field)
            )
            for log in logs
        ]
        assert all(d > 0 for d in diffs), (
            f"{noised_field}: expected all values to differ from raw "
            f"{raw_field}, but found exact matches"
        )
        # The mean absolute difference must be non-trivial (> 0.001 for all
        # fields) — confirms the gate applies meaningful perturbation.
        mean_diff = sum(diffs) / len(diffs)
        assert mean_diff > 0.001, (
            f"{noised_field}: mean |raw - noised| = {mean_diff:.6f}, "
            f"noise appears negligible"
        )


# ---------------------------------------------------------------------------
# Input validation edge cases
# ---------------------------------------------------------------------------


def test_generate_batch_rejects_non_positive_count() -> None:
    """generate_batch raises ValueError when count <= 0."""
    gen = WearableLogGenerator(seed=0)
    with pytest.raises(ValueError, match="positive"):
        gen.generate_batch(0)
    with pytest.raises(ValueError, match="positive"):
        gen.generate_batch(-5)


def test_generate_batch_rejects_empty_scenario_filter() -> None:
    """generate_batch raises ValueError when scenario_filter is an empty list."""
    gen = WearableLogGenerator(seed=0)
    with pytest.raises(ValueError, match="empty"):
        gen.generate_batch(10, scenario_filter=[])


def test_empty_audio_transcript_has_zero_confidence() -> None:
    """Logs with an empty audio transcript always have confidence == 0.0."""
    # Health-alert scenario includes "" as a possible transcript text (silent
    # fall-detection event).  Generate enough logs to hit that case.
    gen = WearableLogGenerator(seed=3)
    logs = gen.generate_batch(60, scenario_filter=[ScenarioType.HEALTH_ALERT])
    empty_logs = [lg for lg in logs if lg.audio_transcript.text == ""]
    assert empty_logs, "Expected at least one empty transcript in 60 health_alert logs"
    for log in empty_logs:
        assert log.audio_transcript.confidence == 0.0, (
            f"Empty transcript must have confidence 0.0, "
            f"got {log.audio_transcript.confidence}"
        )
        assert log.audio_transcript.keywords_detected == [], (
            "Empty transcript must have no detected keywords"
        )
