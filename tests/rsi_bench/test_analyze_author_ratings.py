"""Tests for src/rsi_bench/analyze_author_ratings.py.

No test here makes a network call or touches the real committed
author_ratings.csv / human_rating_key.json -- every test builds a small,
synthetic fixture in-process (or via tmp_path for the file-loading tests).

Test inventory:
  TestLoadAuthorRatings     -- CSV round trip
  TestLoadRatingKey         -- JSON round trip
  TestJoinRecords           -- join correctness, judge_mean, missing-case KeyError
  TestScopesAndTargetValue  -- family filtering, judge vs. judge_mean lookup
  TestSpearmanPoint         -- matches scipy; constant-column handling
  TestBootstrapCi           -- determinism, all-None-rep handling
  TestSpearmanWithCi        -- constant skip, normal-case CI bounds
  TestAgreementRates        -- exact/within_1, rounding, empty input
  TestMeanOffsetWithCi      -- correctness, determinism, empty input
  TestAuthorVsTargetsReport -- end-to-end structure on a small fixture
  TestJudgePairwiseReport   -- constant-column handling among judges
  TestOutcomeConsistencyDistributions -- counts on a small fixture
  TestRunAnalysisDeterminism -- same seed -> byte-identical output
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import spearmanr

from src.rsi_bench.analyze_author_ratings import (
    _bootstrap_ci,
    _scopes,
    _spearman_point,
    _target_value,
    agreement_rates,
    author_vs_targets_report,
    join_records,
    judge_pairwise_report,
    load_author_ratings,
    load_rating_key,
    mean_offset_with_ci,
    outcome_consistency_distributions,
    run_analysis,
    spearman_with_ci,
)

_JUDGES = ("GameTheoreticRigor", "LiteralGroundedness", "OpponentResponsiveness")
_DIMS = ("valuation_reasoning", "horizon_strategy_planning", "outcome_consistency")


def _key_entry(family: str, scores: dict[str, dict[str, int]]) -> dict[str, Any]:
    """One human_rating_key.json entry: ``scores`` is judge -> dim -> int."""
    return {
        "game_family": family,
        "cluster_id": f"{family}_001",
        "member_id": "g1:1:offer",
        "game_id": "g1",
        "round": 1,
        "phase": "offer",
        "judge_scores": scores,
        "cluster_kappa_per_dimension": {dim: 0.0 for dim in _DIMS},
    }


def _small_key() -> dict[str, Any]:
    """6 cases: 3 bargaining, 2 negotiation, 1 persuasion."""
    key: dict[str, Any] = {}
    families = [
        "bargaining",
        "bargaining",
        "bargaining",
        "negotiation",
        "negotiation",
        "persuasion",
    ]
    # valuation_reasoning varies; outcome_consistency constant=5 for every judge
    # (mirrors the real data's degenerate column, exercised deliberately here).
    val_scores = [1, 2, 3, 4, 5, 2]
    for i, family in enumerate(families, start=1):
        scores = {
            judge: {
                "valuation_reasoning": val_scores[i - 1],
                "horizon_strategy_planning": (val_scores[i - 1] + j) % 5 + 1,
                "outcome_consistency": 5,
            }
            for j, judge in enumerate(_JUDGES)
        }
        key[f"case_{i:02d}"] = _key_entry(family, scores)
    return key


def _small_author_rows() -> list[dict[str, Any]]:
    return [
        {
            "case_id": i,
            "valuation_reasoning": v,
            "horizon_strategy_planning": v,
            "outcome_consistency": 5,
        }
        for i, v in enumerate([1, 2, 2, 4, 5, 1], start=1)
    ]


class TestLoadAuthorRatings:
    def test_round_trips_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "author_ratings.csv"
        path.write_text(
            "case_id,valuation_reasoning,horizon_strategy_planning,outcome_consistency\n"
            "1,3,3,5\n2,4,2,5\n"
        )
        rows = load_author_ratings(path)
        assert rows == [
            {
                "case_id": 1,
                "valuation_reasoning": 3,
                "horizon_strategy_planning": 3,
                "outcome_consistency": 5,
            },
            {
                "case_id": 2,
                "valuation_reasoning": 4,
                "horizon_strategy_planning": 2,
                "outcome_consistency": 5,
            },
        ]


class TestLoadRatingKey:
    def test_round_trips_json(self, tmp_path: Path) -> None:
        path = tmp_path / "key.json"
        data = {
            "case_01": _key_entry(
                "bargaining", {j: {d: 3 for d in _DIMS} for j in _JUDGES}
            )
        }
        path.write_text(json.dumps(data))
        loaded = load_rating_key(path)
        assert loaded == data


class TestJoinRecords:
    def test_joins_by_zero_padded_case_id(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        assert len(joined) == 6
        assert joined[0]["case_id"] == 1
        assert joined[0]["game_family"] == "bargaining"
        assert joined[0]["author"]["valuation_reasoning"] == 1

    def test_judge_mean_is_correct(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        case1 = joined[0]
        expected = sum(case1["judges"][j]["valuation_reasoning"] for j in _JUDGES) / 3
        assert case1["judge_mean"]["valuation_reasoning"] == pytest.approx(expected)

    def test_missing_case_raises_keyerror(self) -> None:
        key = _small_key()
        rows = [
            {
                "case_id": 99,
                "valuation_reasoning": 1,
                "horizon_strategy_planning": 1,
                "outcome_consistency": 1,
            }
        ]
        with pytest.raises(KeyError):
            join_records(rows, key)


class TestScopesAndTargetValue:
    def test_scopes_filters_by_family(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        scoped = _scopes(joined)
        assert len(scoped["overall"]) == 6
        assert len(scoped["bargaining"]) == 3
        assert len(scoped["negotiation"]) == 2
        assert len(scoped["persuasion"]) == 1

    def test_target_value_judge_vs_judge_mean(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        record = joined[0]
        assert (
            _target_value(record, "valuation_reasoning", "GameTheoreticRigor")
            == (record["judges"]["GameTheoreticRigor"]["valuation_reasoning"])
        )
        assert (
            _target_value(record, "valuation_reasoning", "judge_mean")
            == (record["judge_mean"]["valuation_reasoning"])
        )


class TestSpearmanPoint:
    def test_matches_scipy_on_varied_input(self) -> None:
        xs = [1, 2, 3, 4, 5]
        ys = [2, 1, 4, 3, 5]
        result = _spearman_point(xs, ys)
        expected_rho, expected_p = spearmanr(xs, ys)
        assert result["rho"] == pytest.approx(float(expected_rho))
        assert result["p"] == pytest.approx(float(expected_p))
        assert result["note"] is None

    def test_constant_x_is_undefined(self) -> None:
        result = _spearman_point([3, 3, 3], [1, 2, 3])
        assert result["rho"] is None
        assert result["p"] is None
        assert result["note"] == "undefined (constant)"

    def test_constant_y_is_undefined(self) -> None:
        result = _spearman_point([1, 2, 3], [5, 5, 5])
        assert result["note"] == "undefined (constant)"

    def test_both_constant_is_undefined(self) -> None:
        result = _spearman_point([5, 5, 5], [5, 5, 5])
        assert result["note"] == "undefined (constant)"


class TestBootstrapCi:
    def test_deterministic_same_seed(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        def stat_fn(indices: list[int]) -> float:
            return sum(values[i] for i in indices) / len(indices)

        ci1 = _bootstrap_ci(stat_fn, len(values), random.Random(42), 500)
        ci2 = _bootstrap_ci(stat_fn, len(values), random.Random(42), 500)
        assert ci1 == ci2

    def test_all_none_reps_gives_none_ci_not_fabricated(self) -> None:
        ci = _bootstrap_ci(lambda indices: None, 5, random.Random(1), 50)
        assert ci == {
            "ci_low": None,
            "ci_high": None,
            "n_valid_reps": 0,
            "n_total_reps": 50,
        }


class TestSpearmanWithCi:
    def test_constant_input_skips_bootstrap_entirely(self) -> None:
        result = spearman_with_ci(
            [5, 5, 5, 5], [1, 2, 3, 4], random.Random(1), n_resamples=50
        )
        assert result["note"] == "undefined (constant)"
        assert result["ci_low"] is None
        assert result["n_valid_reps"] == 0

    def test_normal_case_produces_ci_containing_point_estimate_region(self) -> None:
        xs = [1, 2, 3, 4, 5, 6, 7, 8]
        ys = [1, 3, 2, 5, 4, 7, 6, 8]
        result = spearman_with_ci(xs, ys, random.Random(7), n_resamples=2000)
        assert result["note"] is None
        assert result["rho"] is not None
        assert (
            result["ci_low"] <= result["rho"] <= result["ci_high"]
            or result["n_valid_reps"] < 2000
        )

    def test_deterministic_same_seed(self) -> None:
        xs = [1, 2, 3, 4, 5]
        ys = [2, 1, 5, 3, 4]
        r1 = spearman_with_ci(xs, ys, random.Random(99), n_resamples=300)
        r2 = spearman_with_ci(xs, ys, random.Random(99), n_resamples=300)
        assert r1 == r2


class TestAgreementRates:
    def test_exact_and_within_1(self) -> None:
        author = [1, 2, 3, 4, 5]
        target = [1, 2, 3, 5, 3]
        result = agreement_rates(author, target)
        assert result["exact"] == pytest.approx(3 / 5)
        # |4-5|=1 (within), |5-3|=2 (not within) -> 4/5 within-1
        assert result["within_1"] == pytest.approx(4 / 5)

    def test_rounds_target_before_comparing(self) -> None:
        # round(2.5) == 2 (Python banker's rounding) -- author=2 should match exactly
        result = agreement_rates([2], [2.5])
        assert result["exact"] == pytest.approx(1.0)

    def test_empty_input_gives_none_not_fabricated(self) -> None:
        result = agreement_rates([], [])
        assert result == {"exact": None, "within_1": None}


class TestMeanOffsetWithCi:
    def test_mean_offset_correctness(self) -> None:
        author = [3, 4, 5]
        target = [1, 2, 3]
        result = mean_offset_with_ci(author, target, random.Random(1), n_resamples=100)
        assert result["mean_offset"] == pytest.approx(2.0)

    def test_deterministic_same_seed(self) -> None:
        author = [3, 4, 5, 1, 2]
        target = [1.5, 2.5, 3.5, 0.5, 1.5]
        r1 = mean_offset_with_ci(author, target, random.Random(5), n_resamples=200)
        r2 = mean_offset_with_ci(author, target, random.Random(5), n_resamples=200)
        assert r1 == r2

    def test_empty_input_gives_none_not_fabricated(self) -> None:
        result = mean_offset_with_ci([], [], random.Random(1), n_resamples=10)
        assert result["mean_offset"] is None
        assert result["ci_low"] is None
        assert result["n_valid_reps"] == 0


class TestAuthorVsTargetsReport:
    def test_covers_every_dimension_scope_target(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        report = author_vs_targets_report(joined, random.Random(1), n_resamples=20)
        for dim in _DIMS:
            for scope in ("overall", "bargaining", "negotiation", "persuasion"):
                for target in (*_JUDGES, "judge_mean"):
                    cell = report[dim][scope][target]
                    assert "spearman" in cell
                    assert "agreement" in cell
                    assert "offset" in cell

    def test_outcome_consistency_is_undefined_constant_everywhere(self) -> None:
        # outcome_consistency is constant=5 for author AND every judge in
        # the fixture -- this is the constant-column handling the task asks
        # for, mirroring the real author_ratings.csv data exactly.
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        report = author_vs_targets_report(joined, random.Random(1), n_resamples=20)
        for scope in ("overall", "bargaining", "negotiation", "persuasion"):
            for target in (*_JUDGES, "judge_mean"):
                cell = report["outcome_consistency"][scope][target]
                assert cell["spearman"]["note"] == "undefined (constant)"
                assert cell["offset"]["mean_offset"] == pytest.approx(0.0)


class TestJudgePairwiseReport:
    def test_constant_dimension_is_undefined_between_judges(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        report = judge_pairwise_report(joined)
        for scope in ("overall", "bargaining", "negotiation", "persuasion"):
            for pair_key, cell in report["outcome_consistency"][scope].items():
                assert cell["spearman"]["note"] == "undefined (constant)"
                assert cell["mean_offset"] == pytest.approx(0.0)

    def test_varied_dimension_has_defined_pairwise_stats(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        report = judge_pairwise_report(joined)
        cell = report["valuation_reasoning"]["overall"][
            "GameTheoreticRigor|LiteralGroundedness"
        ]
        assert cell["n_cases"] == 6
        # all 3 judges share the identical valuation_reasoning score per
        # case in this fixture (see _small_key) -- so this pair IS constant
        # relative to each other on a per-judge basis, but offset is exact 0
        assert cell["mean_offset"] == pytest.approx(0.0)


class TestOutcomeConsistencyDistributions:
    def test_counts_match_fixture(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        joined = join_records(rows, key)
        dist = outcome_consistency_distributions(joined)
        assert dist["n_cases"] == 6
        assert dist["author"] == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 6}
        for judge in _JUDGES:
            assert dist["judges"][judge] == {"1": 0, "2": 0, "3": 0, "4": 0, "5": 6}


class TestRunAnalysisDeterminism:
    def test_same_seed_gives_identical_output(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        result1 = run_analysis(rows, key, seed=20261002, n_resamples=50)
        result2 = run_analysis(rows, key, seed=20261002, n_resamples=50)
        assert result1 == result2

    def test_n_cases_and_by_family_counts(self) -> None:
        key = _small_key()
        rows = _small_author_rows()
        result = run_analysis(rows, key, seed=1, n_resamples=10)
        assert result["n_cases"] == 6
        assert result["by_family_n_cases"] == {
            "bargaining": 3,
            "negotiation": 2,
            "persuasion": 1,
        }
