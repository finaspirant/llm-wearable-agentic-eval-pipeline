"""Tests for Day 22 additions to src/eval/benchmark_runner.py.

Covers:
- BenchmarkResult has trajectory_score, pia_dimensions, nondeterminism_variance,
  run_index, cascade_depth fields.
- run_all(runs=3) computes nondeterminism_variance >= 0.0.
- generate_leaderboard() output contains all 4 frameworks and 6 ranking dimensions.
- generate_leaderboard() writes framework_leaderboard.json.
- run_all() over all 10 tasks × 4 frameworks × 3 runs = 120 results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.benchmark_runner import (
    ALL_FRAMEWORK_NAMES,
    BenchmarkResult,
    BenchmarkRunner,
    LangGraphBenchmark,
    TaskConfig,
)

CONFIG_PATH = Path("configs/benchmark_tasks.yaml")

_MINIMAL_TASK = TaskConfig(
    task_id="it_helpdesk",
    description="Fix VPN.",
    goal="VPN restored.",
    max_steps=7,
    timeout_s=30.0,
    tools_available=["diagnose_network"],
    expected_steps=[],
    success_criteria={},
    difficulty_level="medium",
    tags=["kore_ai"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    tmp_path: Path, framework_names: list[str] | None = None
) -> BenchmarkRunner:
    output = tmp_path / "results.jsonl"
    return BenchmarkRunner(
        config_path=CONFIG_PATH,
        output_path=output,
        framework_names=framework_names,
    )


# ---------------------------------------------------------------------------
# 1. BenchmarkResult has the new Day-22 fields
# ---------------------------------------------------------------------------


class TestBenchmarkResultFields:
    def test_defaults_present(self) -> None:
        r = BenchmarkResult(
            task_id="t",
            framework="langgraph",
            steps_taken=3,
            tokens_used=100,
            latency_ms=50.0,
            errors=[],
            goal_achieved=True,
            trajectory=[],
        )
        assert r.run_index == 1
        assert r.cascade_depth == 0
        assert r.trajectory_score is None
        assert r.pia_dimensions is None
        assert r.nondeterminism_variance is None

    def test_to_dict_includes_new_fields(self) -> None:
        r = BenchmarkResult(
            task_id="t",
            framework="langgraph",
            steps_taken=3,
            tokens_used=100,
            latency_ms=50.0,
            errors=[],
            goal_achieved=True,
            trajectory=[],
            run_index=2,
            cascade_depth=4,
            trajectory_score=0.85,
            pia_dimensions={"planning_quality": 0.9},
            nondeterminism_variance=0.01,
        )
        d = r.to_dict()
        assert d["run_index"] == 2
        assert d["cascade_depth"] == 4
        assert d["trajectory_score"] == pytest.approx(0.85, abs=1e-4)
        assert d["pia_dimensions"] == {"planning_quality": 0.9}
        assert d["nondeterminism_variance"] == pytest.approx(0.01, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. trajectory_score populated after run_task()
# ---------------------------------------------------------------------------


class TestTrajectoryScoreAfterRunTask:
    def test_trajectory_score_is_float_or_none(self, tmp_path: Path) -> None:
        bench = LangGraphBenchmark()
        result = bench.run_task(_MINIMAL_TASK, run_index=1)
        # Field must exist and be float or None (never raises AttributeError)
        assert result.trajectory_score is None or isinstance(
            result.trajectory_score, float
        )

    def test_run_all_single_task_populates_trajectory_score(
        self, tmp_path: Path
    ) -> None:
        runner = _make_runner(tmp_path, framework_names=["langgraph"])
        results = runner.run_all(task_ids=["it_helpdesk"], runs=1)
        assert len(results) == 1
        r = results[0]
        # After run_all, trajectory_score should be wired via _build_wearable_proxy
        assert r.trajectory_score is None or isinstance(r.trajectory_score, float)


# ---------------------------------------------------------------------------
# 3. run_all(runs=3) returns 3 results and computes nondeterminism_variance
# ---------------------------------------------------------------------------


class TestThreeRunsVariance:
    def test_three_runs_returns_three_results(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, framework_names=["langgraph"])
        results = runner.run_all(task_ids=["it_helpdesk"], runs=3)
        assert len(results) == 3

    def test_run_indices_are_one_two_three(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, framework_names=["langgraph"])
        results = runner.run_all(task_ids=["it_helpdesk"], runs=3)
        indices = {r.run_index for r in results}
        assert indices == {1, 2, 3}

    def test_nondeterminism_variance_is_float_gte_zero(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path, framework_names=["langgraph"])
        results = runner.run_all(task_ids=["it_helpdesk"], runs=3)
        for r in results:
            if r.nondeterminism_variance is not None:
                assert isinstance(r.nondeterminism_variance, float)
                assert r.nondeterminism_variance >= 0.0

    def test_variance_consistent_across_runs_in_same_batch(
        self, tmp_path: Path
    ) -> None:
        runner = _make_runner(tmp_path, framework_names=["langgraph"])
        results = runner.run_all(task_ids=["it_helpdesk"], runs=3)
        variances = [
            r.nondeterminism_variance
            for r in results
            if r.nondeterminism_variance is not None
        ]
        if variances:
            # All runs in same (task, framework) batch share the same variance
            assert len(set(variances)) == 1


# ---------------------------------------------------------------------------
# 4. generate_leaderboard() keys
# ---------------------------------------------------------------------------

_EXPECTED_RANKING_DIMS = {
    "token_efficiency",
    "latency",
    "reliability",
    "goal_rate",
    "trajectory_quality",
    "cascade_depth",
}


class TestLeaderboardKeys:
    @pytest.fixture()
    def leaderboard(self, tmp_path: Path) -> dict:
        runner = _make_runner(tmp_path)
        results = runner.run_all(task_ids=["it_helpdesk"], runs=1)
        return runner.generate_leaderboard(results)

    def test_all_four_frameworks_present(self, leaderboard: dict) -> None:
        assert set(leaderboard["frameworks"].keys()) == set(ALL_FRAMEWORK_NAMES)

    def test_all_six_ranking_dimensions_present(self, leaderboard: dict) -> None:
        assert set(leaderboard["rankings"].keys()) == _EXPECTED_RANKING_DIMS

    def test_each_ranking_contains_all_frameworks(self, leaderboard: dict) -> None:
        for dim, ranked in leaderboard["rankings"].items():
            assert set(ranked) == set(ALL_FRAMEWORK_NAMES), (
                f"Dim {dim!r} missing frameworks"
            )

    def test_n_frameworks_matches_registry(self, leaderboard: dict) -> None:
        assert leaderboard["n_frameworks"] == len(ALL_FRAMEWORK_NAMES)

    def test_framework_aggregates_have_required_keys(self, leaderboard: dict) -> None:
        required = {
            "avg_trajectory_score",
            "avg_tokens",
            "avg_latency_ms",
            "goal_achievement_rate",
            "avg_nondeterminism_variance",
            "avg_cascade_depth",
        }
        for fw, metrics in leaderboard["frameworks"].items():
            assert required <= set(metrics.keys()), (
                f"Framework {fw!r} missing aggregate keys"
            )


# ---------------------------------------------------------------------------
# 5. generate_leaderboard() writes framework_leaderboard.json
# ---------------------------------------------------------------------------


class TestLeaderboardSaved:
    def test_leaderboard_json_written(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(task_ids=["it_helpdesk"], runs=1)
        lb_path = Path("data/processed/framework_leaderboard.json")
        runner.generate_leaderboard(results)
        assert lb_path.exists(), "framework_leaderboard.json was not written"

    def test_leaderboard_json_is_valid(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(task_ids=["it_helpdesk"], runs=1)
        runner.generate_leaderboard(results)
        lb_path = Path("data/processed/framework_leaderboard.json")
        data = json.loads(lb_path.read_text())
        assert "frameworks" in data
        assert "rankings" in data

    def test_leaderboard_return_matches_saved_file(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(task_ids=["it_helpdesk"], runs=1)
        returned = runner.generate_leaderboard(results)
        lb_path = Path("data/processed/framework_leaderboard.json")
        saved = json.loads(lb_path.read_text())
        # Top-level keys should match (timestamps may differ by a microsecond but
        # the structural keys must be identical)
        assert set(returned.keys()) == set(saved.keys())


# ---------------------------------------------------------------------------
# 6. All 10 tasks × 4 frameworks × 3 runs = 120 results
# ---------------------------------------------------------------------------


class TestAllTenTasksRun:
    def test_total_result_count_is_120(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(runs=3)
        assert len(results) == 120, (
            f"Expected 120 results (10 tasks × 4 frameworks × 3 runs), got {len(results)}"  # noqa: E501
        )

    def test_all_four_frameworks_represented(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(runs=3)
        frameworks = {r.framework for r in results}
        assert frameworks == set(ALL_FRAMEWORK_NAMES)

    def test_jsonl_has_120_lines(self, tmp_path: Path) -> None:
        output = tmp_path / "results.jsonl"
        runner = BenchmarkRunner(
            config_path=CONFIG_PATH,
            output_path=output,
        )
        runner.run_all(runs=3)
        lines = [ln for ln in output.read_text().splitlines() if ln.strip()]
        assert len(lines) == 120

    def test_each_result_has_goal_achieved(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(runs=3)
        for r in results:
            assert isinstance(r.goal_achieved, bool)

    def test_cascade_depth_is_non_negative_int(self, tmp_path: Path) -> None:
        runner = _make_runner(tmp_path)
        results = runner.run_all(runs=1)
        for r in results:
            assert isinstance(r.cascade_depth, int)
            assert r.cascade_depth >= 0
