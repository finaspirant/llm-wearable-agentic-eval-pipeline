"""Tests for src/eval/ab_experiment.py.

Ten focused test classes verifying split logic, metric coverage, target
deltas, output files, dry-run behaviour, and result schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from src.eval.ab_experiment import (
    _METRIC_NAMES,
    ABExperiment,
    ABResult,
    app,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_WEARABLE_LOGS = Path("data/raw/synthetic_wearable_logs.jsonl")
_REAL_SCORES = Path("data/processed/trajectory_scores.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_entry(tid: str, weighted_total: float = 0.897) -> dict[str, Any]:
    return {
        "trajectory_id": tid,
        "weighted_total": weighted_total,
        "intent": {"score": 0.75, "matched_goal": True, "reasoning": "ok"},
        "planning": {"score": 0.80, "step_efficiency": 1.0, "reasoning": "ok"},
        "tool_calls": {
            "score": 1.0,
            "precision": 1.0,
            "false_positives": 0,
            "reasoning": "ok",
        },
        "recovery": {"score": None, "had_error": False, "reasoning": "n/a"},
        "outcome": {"score": 1.0, "goal_achieved": True, "reasoning": "ok"},
        "metadata": {"scenario_type": "health_alert"},
    }


def _scores_payload(n: int = 100, varied: bool = False) -> dict[str, Any]:
    scores = [
        _score_entry(f"traj_{i:04d}", i / n if varied else 0.897)
        for i in range(n)
    ]
    return {
        "generated_at": "2026-04-16T00:00:00+00:00",
        "dry_run": True,
        "n_trajectories": n,
        "scores": scores,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synth_scores(tmp_path: Path) -> Path:
    """Synthetic scores — IDs do not match real WearableLogs."""
    p = tmp_path / "scores.json"
    p.write_text(json.dumps(_scores_payload()))
    return p


@pytest.fixture()
def varied_scores(tmp_path: Path) -> Path:
    """Scores with ascending weighted_total (0.00 → 0.99)."""
    p = tmp_path / "varied.json"
    p.write_text(json.dumps(_scores_payload(varied=True)))
    return p


@pytest.fixture()
def experiment() -> ABExperiment:
    """ABExperiment backed by real WearableLogs."""
    return ABExperiment(wearable_logs_path=_WEARABLE_LOGS, rng_seed=42)


@pytest.fixture()
def ab_result(experiment: ABExperiment, tmp_path: Path) -> ABResult:
    """Full ABResult from real trajectory scores + real WearableLogs."""
    return experiment.run(_REAL_SCORES, tmp_path / "ab_out")


# ---------------------------------------------------------------------------
# 1. TestSplitBalance
# ---------------------------------------------------------------------------


class TestSplitBalance:
    """Raw and curated groups each have exactly 50 trajectories."""

    def test_both_groups_have_50(
        self,
        experiment: ABExperiment,
        synth_scores: Path,
        tmp_path: Path,
    ) -> None:
        raw, curated = experiment.load_and_split(synth_scores, tmp_path / "out")
        assert len(raw) == 50
        assert len(curated) == 50


# ---------------------------------------------------------------------------
# 2. TestNoOverlap
# ---------------------------------------------------------------------------


class TestNoOverlap:
    """No trajectory ID appears in both groups."""

    def test_trajectory_ids_are_disjoint(
        self,
        experiment: ABExperiment,
        synth_scores: Path,
        tmp_path: Path,
    ) -> None:
        raw, curated = experiment.load_and_split(synth_scores, tmp_path / "out")
        raw_ids = {s["trajectory_id"] for s in raw}
        curated_ids = {s["trajectory_id"] for s in curated}
        assert raw_ids.isdisjoint(curated_ids)


# ---------------------------------------------------------------------------
# 3. TestMetricKeys
# ---------------------------------------------------------------------------


class TestMetricKeys:
    """ab_results.json contains all 6 Kore.ai metric keys in every section."""

    def test_all_sections_contain_all_metric_keys(
        self, ab_result: ABResult
    ) -> None:
        d = ab_result.to_dict()
        expected = set(_METRIC_NAMES)
        assert expected <= set(d["raw"].keys())
        assert expected <= set(d["curated"].keys())
        assert set(d["delta"].keys()) == expected
        assert set(d["pct_improvement"].keys()) == expected


# ---------------------------------------------------------------------------
# 4. TestCuratedHigherScore
# ---------------------------------------------------------------------------


class TestCuratedHigherScore:
    """Curated group mean weighted_total > raw group mean weighted_total."""

    def test_curated_mean_exceeds_raw(
        self,
        experiment: ABExperiment,
        varied_scores: Path,
        tmp_path: Path,
    ) -> None:
        raw, curated = experiment.load_and_split(varied_scores, tmp_path / "out")
        raw_mean = sum(s["weighted_total"] for s in raw) / len(raw)
        curated_mean = sum(s["weighted_total"] for s in curated) / len(curated)
        assert curated_mean > raw_mean


# ---------------------------------------------------------------------------
# 5. TestToolAccuracyTarget
# ---------------------------------------------------------------------------


class TestToolAccuracyTarget:
    """tool_invocation_accuracy pct_improvement >= 25.0."""

    def test_tool_invocation_pct_improvement(self, ab_result: ABResult) -> None:
        pct = ab_result.pct_improvement["tool_invocation_accuracy"]
        assert pct >= 25.0, f"tool_invocation_accuracy pct={pct:.1f}% < 25%"


# ---------------------------------------------------------------------------
# 6. TestSuccessRateTarget
# ---------------------------------------------------------------------------


class TestSuccessRateTarget:
    """trajectory_success_rate pct_improvement >= 15.0."""

    def test_success_rate_pct_improvement(self, ab_result: ABResult) -> None:
        pct = ab_result.pct_improvement["trajectory_success_rate"]
        assert pct >= 15.0, f"trajectory_success_rate pct={pct:.1f}% < 15%"


# ---------------------------------------------------------------------------
# 7. TestOutputFilesExist
# ---------------------------------------------------------------------------


class TestOutputFilesExist:
    """Both split files + ab_results.json are created on disk after run()."""

    def test_all_three_output_files_written(
        self, experiment: ABExperiment, tmp_path: Path
    ) -> None:
        out = tmp_path / "ab_out"
        experiment.run(_REAL_SCORES, out)
        assert (out / "raw_trajectories.json").exists()
        assert (out / "curated_trajectories.json").exists()
        assert (out / "ab_results.json").exists()


# ---------------------------------------------------------------------------
# 8. TestDryRunNoWrite
# ---------------------------------------------------------------------------


class TestDryRunNoWrite:
    """--dry-run flag completes successfully without writing to --output dir."""

    def test_dry_run_does_not_write_output_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        out = tmp_path / "dry_out"
        result = runner.invoke(
            app,
            [
                "--input",
                str(_REAL_SCORES),
                "--output",
                str(out),
                "--wearable-logs",
                str(_WEARABLE_LOGS),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert not (out / "ab_results.json").exists()


# ---------------------------------------------------------------------------
# 9. TestDeltaSign
# ---------------------------------------------------------------------------


class TestDeltaSign:
    """All delta values are >= 0.0 (curated never scores below raw)."""

    def test_all_deltas_non_negative(self, ab_result: ABResult) -> None:
        for metric, d in ab_result.delta.items():
            assert d >= 0.0, f"{metric} delta={d:.4f} is negative"


# ---------------------------------------------------------------------------
# 10. TestResultSchema
# ---------------------------------------------------------------------------


class TestResultSchema:
    """ab_results.json is valid JSON with correct top-level keys."""

    def test_ab_results_json_schema(
        self, experiment: ABExperiment, tmp_path: Path
    ) -> None:
        out = tmp_path / "ab_out"
        experiment.run(_REAL_SCORES, out)
        data: dict[str, Any] = json.loads((out / "ab_results.json").read_text())
        assert isinstance(data, dict)
        required = {
            "raw",
            "curated",
            "delta",
            "pct_improvement",
            "experiment_timestamp",
        }
        assert required <= data.keys()
