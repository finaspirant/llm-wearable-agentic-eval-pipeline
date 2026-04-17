"""Tests for demo/pipeline.py — pure-function eval pipeline.

All tests run in dry_run mode (AgenticEvaluator dry_run=True) so no live
API calls are made.  Results are deterministic for a given seed.

Scenario-specific behaviour verified here is grounded in the wearable
generator's seeded output (seed=42):

- health_alert:   step 0 observation contains 'bpm'/'spo2' → DOMAIN trigger;
                  step 2 action 'send_alert' contains 'alert' → SAFETY trigger.
                  Both fire deterministically for every health_alert trajectory.

- privacy_sensitive + seed=42: log index 1 has AMBIENT consent.
  Pipeline policy: privacy_sensitive + AMBIENT → blocked (intimate /
  healthcare contexts require at minimum IMPLIED consent).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.pipeline import run_eval_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keys whose values must be floats in [0.0, 1.0].
_SCORE_KEYS: tuple[str, ...] = (
    "facts_score",
    "traj_quality",
    "tool_accuracy",
    "traj_success",
)

# Required top-level keys every result dict must expose.
_REQUIRED_KEYS: tuple[str, ...] = (
    "log_id",
    "scenario",
    "consent_model",
    "privacy_blocked",
    "privacy_reason",
    "privacy_gate_enabled",
    "facts_score",
    "traj_quality",
    "tool_accuracy",
    "traj_success",
    "privacy_leak_detected",
    "hitl_triggered",
    "hitl_count",
    "hitl_triggers",
    "eval",
    "trajectory_steps",
)


# ---------------------------------------------------------------------------
# Shape and required keys
# ---------------------------------------------------------------------------


def test_pipeline_returns_results() -> None:
    """run_eval_pipeline returns a list with one dict per trajectory."""
    results = run_eval_pipeline("health_alert", num_trajectories=2, seed=42)

    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, dict)


def test_result_has_required_keys() -> None:
    """Every result dict exposes all required top-level keys."""
    results = run_eval_pipeline("health_alert", num_trajectories=1, seed=42)

    assert len(results) == 1
    for key in _REQUIRED_KEYS:
        assert key in results[0], f"Missing required key: {key!r}"


def test_single_trajectory_list_length() -> None:
    """num_trajectories=1 returns a list of exactly one element."""
    results = run_eval_pipeline("ambient_noise", num_trajectories=1, seed=7)
    assert len(results) == 1


def test_five_trajectories_list_length() -> None:
    """num_trajectories=5 returns a list of exactly five elements."""
    results = run_eval_pipeline("calendar_reminder", num_trajectories=5, seed=99)
    assert len(results) == 5


# ---------------------------------------------------------------------------
# Score ranges
# ---------------------------------------------------------------------------


def test_all_scores_are_floats() -> None:
    """All float score fields in every result dict are in [0.0, 1.0]."""
    results = run_eval_pipeline("health_alert", num_trajectories=2, seed=42)

    for r in results:
        for key in _SCORE_KEYS:
            val = r[key]
            assert isinstance(val, float), (
                f"{key} should be float, got {type(val).__name__}"
            )
            assert 0.0 <= val <= 1.0, f"{key}={val:.4f} is outside [0.0, 1.0]"


def test_eval_weighted_total_in_range() -> None:
    """The full eval dict's weighted_total is also in [0.0, 1.0]."""
    results = run_eval_pipeline("location_trigger", num_trajectories=2, seed=42)

    for r in results:
        wt = r["eval"]["weighted_total"]
        assert isinstance(wt, float)
        assert 0.0 <= wt <= 1.0


# ---------------------------------------------------------------------------
# Privacy gate
# ---------------------------------------------------------------------------


def test_privacy_gate_blocks_sensitive() -> None:
    """privacy_sensitive + gate=True blocks at least 1 trajectory.

    seed=42 yields AMBIENT consent on log index 1.  The gate policy for
    privacy_sensitive + AMBIENT is block (see pipeline._privacy_gate_result).
    """
    results = run_eval_pipeline(
        "privacy_sensitive",
        num_trajectories=2,
        privacy_gate_enabled=True,
        seed=42,
    )

    blocked = [r for r in results if r["privacy_blocked"]]
    assert len(blocked) >= 1, (
        "Expected at least 1 blocked trajectory for privacy_sensitive "
        "with gate=True and seed=42"
    )


def test_privacy_gate_disabled_never_blocks() -> None:
    """With gate disabled no trajectory is blocked regardless of consent."""
    results = run_eval_pipeline(
        "privacy_sensitive",
        num_trajectories=3,
        privacy_gate_enabled=False,
        seed=42,
    )

    assert not any(r["privacy_blocked"] for r in results), (
        "No trajectory should be blocked when privacy_gate_enabled=False"
    )


def test_blocked_result_has_reason() -> None:
    """Blocked trajectories carry a non-empty privacy_reason string."""
    results = run_eval_pipeline(
        "privacy_sensitive",
        num_trajectories=2,
        privacy_gate_enabled=True,
        seed=42,
    )

    for r in results:
        if r["privacy_blocked"]:
            assert isinstance(r["privacy_reason"], str)
            assert len(r["privacy_reason"]) > 0


# ---------------------------------------------------------------------------
# HITL triggers
# ---------------------------------------------------------------------------


def test_hitl_fires_on_health_alert() -> None:
    """health_alert always fires HITL triggers.

    Step 0 observation contains 'bpm'/'spo2' → DOMAIN_EXPERTISE_REQUIRED.
    Step 2 action 'send_alert' contains 'alert' → SAFETY_ADJACENT_ACTION.
    Both fire deterministically for every health_alert trajectory.
    """
    results = run_eval_pipeline("health_alert", num_trajectories=1, seed=42)

    assert len(results) == 1
    assert results[0]["hitl_triggered"] is True
    assert results[0]["hitl_count"] >= 1


def test_hitl_trigger_dict_has_required_keys() -> None:
    """Each entry in hitl_triggers has trigger_type, severity, step_index."""
    results = run_eval_pipeline("health_alert", num_trajectories=1, seed=42)

    assert results[0]["hitl_triggered"] is True
    for t in results[0]["hitl_triggers"]:
        assert "trigger_type" in t
        assert "severity" in t
        assert "step_index" in t


def test_hitl_count_matches_triggers_list_length() -> None:
    """hitl_count equals len(hitl_triggers) for every result."""
    results = run_eval_pipeline("health_alert", num_trajectories=3, seed=42)

    for r in results:
        assert r["hitl_count"] == len(r["hitl_triggers"])


# ---------------------------------------------------------------------------
# FACTS score
# ---------------------------------------------------------------------------


def test_facts_score_present() -> None:
    """Every result dict has a 'facts_score' key with a float value."""
    results = run_eval_pipeline("ambient_noise", num_trajectories=3, seed=42)

    for r in results:
        assert "facts_score" in r, "result missing 'facts_score'"
        assert isinstance(r["facts_score"], float)


def test_facts_detail_has_sub_scores() -> None:
    """facts_detail exposes parametric_score, search_score, grounding_score."""
    results = run_eval_pipeline("calendar_reminder", num_trajectories=1, seed=42)

    fd = results[0]["facts_detail"]
    for sub_key in (
        "parametric_score",
        "search_score",
        "grounding_score",
        "overall_facts_score",
    ):
        assert sub_key in fd, f"facts_detail missing '{sub_key}'"
        assert isinstance(fd[sub_key], float)


# ---------------------------------------------------------------------------
# Trajectory steps
# ---------------------------------------------------------------------------


def test_trajectory_steps_present() -> None:
    """Every result has a non-empty trajectory_steps list."""
    results = run_eval_pipeline("health_alert", num_trajectories=1, seed=42)

    assert len(results[0]["trajectory_steps"]) > 0


def test_trajectory_steps_have_required_keys() -> None:
    """Every step dict has step_index, step_name, observation, action, confidence."""
    results = run_eval_pipeline("location_trigger", num_trajectories=1, seed=42)

    _step_keys = (
        "step_index",
        "step_name",
        "observation",
        "reasoning",
        "action",
        "confidence",
    )
    for step in results[0]["trajectory_steps"]:
        for key in _step_keys:
            assert key in step, f"step missing key: {key!r}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_results() -> None:
    """Two calls with the same seed return identical scores and consent models.

    log_ids use uuid.uuid4() (os.urandom) so they are not reproducible; scores
    and consent models are derived from the numpy-seeded RNG and are stable.
    """
    r1 = run_eval_pipeline("health_alert", num_trajectories=2, seed=123)
    r2 = run_eval_pipeline("health_alert", num_trajectories=2, seed=123)

    assert [r["traj_quality"] for r in r1] == [r["traj_quality"] for r in r2]
    assert [r["consent_model"] for r in r1] == [r["consent_model"] for r in r2]
    assert [r["hitl_count"] for r in r1] == [r["hitl_count"] for r in r2]


def test_different_seeds_produce_different_log_ids() -> None:
    """Two calls with different seeds yield different log UUIDs."""
    r1 = run_eval_pipeline("health_alert", num_trajectories=1, seed=10)
    r2 = run_eval_pipeline("health_alert", num_trajectories=1, seed=99)

    assert r1[0]["log_id"] != r2[0]["log_id"]
