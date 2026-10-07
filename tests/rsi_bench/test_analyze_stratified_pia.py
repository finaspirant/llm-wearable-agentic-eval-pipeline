"""Tests for src/rsi_bench/analyze_stratified_pia.py.

No test here touches the Anthropic API, the real GLEE trajectory fixture,
or the real (committed) scored/manifest files -- every test builds a
small, synthetic scored-cluster dataset in-process. Ground-truth kappa
values are produced by calling the module's own
:func:`~src.rsi_bench.analyze_stratified_pia.recompute_cluster_kappa`
(itself a thin wrapper around
:func:`~src.rsi_bench.glee_pia_baseline._dimension_kappa`), so records
are internally self-consistent by construction -- this file tests the
*analysis* logic (integrity diffing, aggregation arithmetic, degenerate
detection, bootstrap determinism), not Fleiss' kappa itself, which
already has dedicated coverage in ``tests/annotation/test_irr_calculator.py``
and ``tests/rsi_bench/test_run_stratified_pia.py::TestRawScoreRoundTrip``.

Test inventory:
  TestLoading                 -- JSONL/JSON round trip, missing-file errors
  TestIntegrity                -- 0 mismatches on clean data, 1 on corrupted
  TestAggregationArithmetic    -- unweighted/weighted/pooled formulas
  TestDegenerateDetection       -- forced-1.0 entries found, excluded correctly
  TestBootstrapDeterminism     -- same seed -> identical output, no fabrication
  TestJudgeAgreementReport      -- distributions/means/pairwise stats vs manual calc
  TestShuffledClusterFloor      -- cluster-size preservation, determinism, no fab.
  TestDependenceReport          -- shared-game counting on a controlled fixture
  TestBuildRecordsWithoutDegenerate -- strips only flagged dims, no mutation
  TestCompletePanelScoresSubset -- regression: judges-subset filtering bug
  TestGameLevelBootstrap        -- two-stage game/cluster resample, determinism
  TestOffsetRobustAgreement     -- centered ICC(3,1) + Krippendorff alpha
  TestLeaveOneJudgeOut          -- 2-judge kappa recompute + matching floor
  TestVarianceDecomposition     -- real vs shuffled eta-squared
  TestExtractActionFeature      -- per-family/phase action classification
  TestParseMemberIdAndHistoryLength -- member_id parsing, history length
  TestLoadRawStepsIndex         -- gz JSONL load, dedup, malformed-line skip
  TestResolveMemberAndClusterFeatures -- raw-step resolution, unresolved count
  TestActionSpreadAndHelpers    -- _sd/_modal_share/_action_spread arithmetic
  TestActionOnlyBaselineReport  -- per-cluster rows + family summary, no judge scores
  TestActionShuffledClusterFloor -- determinism, cluster-size preservation
  TestMembershipHomogeneityReport -- round/phase/history-length modal shares
  TestClusterJudgeMeanSpread    -- SD of judge-mean score, averaged over dims
  TestActionJudgeSpreadCorrelation -- Spearman rho sign/None-on-insufficient-data
"""

from __future__ import annotations

import copy
import gzip
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.stats import spearmanr

from src.annotation.irr_calculator import IRRCalculator
from src.rsi_bench.analyze_stratified_pia import (
    _action_pool,
    _action_spread,
    _chunk_kappa,
    _cluster_components,
    _cluster_judge_mean_spread,
    _cluster_membership_features,
    _cluster_pool,
    _complete_panel_scores,
    _eta_squared,
    _family_judge_means,
    _history_length,
    _icc_3_1,
    _judge_mean_score,
    _modal_share,
    _parse_member_id,
    _recompute_kappa_for_judge_subset,
    _resolve_member,
    _sd,
    _split_features,
    action_judge_spread_correlation,
    action_only_baseline_report,
    action_shuffled_cluster_floor,
    audit_game_level_bootstrap,
    bootstrap_primary_aggregate,
    build_aggregates,
    build_records_without_degenerate,
    check_integrity,
    cluster_components_report,
    dependence_report,
    extract_action_feature,
    find_degenerate_entries,
    game_level_bootstrap,
    judge_agreement_report,
    leave_one_judge_out_report,
    load_manifest,
    load_raw_steps_index,
    load_scored_records,
    membership_homogeneity_report,
    offset_robust_agreement_report,
    pooled_dimension_kappa,
    recompute_cluster_kappa,
    shuffled_cluster_floor,
    variance_decomposition,
)
from src.rsi_bench.glee_pia_baseline import _JUDGES as _REAL_JUDGES

_DIMS = [
    "valuation_reasoning",
    "horizon_strategy_planning",
    "concession_handling",
    "outcome_consistency",
]
_JUDGES = list(_REAL_JUDGES)  # recompute_cluster_kappa -> _dimension_kappa is
# hardwired to these exact 3 judge names (not parameterized), so synthetic
# data must use them for the integrity-recomputation path to find any scores.


def _patterned_score(member_idx: int, judge_idx: int, dim_idx: int, salt: int) -> int:
    """Deterministic 1-5 score, varied enough to avoid accidental degeneracy.

    Coefficients are chosen coprime-ish with the mod-5 scale so that
    ``judge_idx`` actually moves the result (a coefficient that is a
    multiple of 5, e.g. 5 itself, would vanish under ``% 5`` and make
    every judge agree unanimously within an item by construction --
    exactly the "legitimate kappa=1.0" trap this avoids).
    """
    return ((member_idx * 3 + judge_idx * 2 + dim_idx * 7 + salt) % 5) + 1


def _build_record(
    cluster_id: str,
    family: str,
    members_judges_scores: dict[str, dict[str, dict[str, int | None]]],
) -> dict[str, Any]:
    """Build a self-consistent synthetic scored-cluster record.

    ``kappa_per_dimension``/``kappa_overall`` are computed via
    :func:`recompute_cluster_kappa` so the record is guaranteed
    stored == recomputed (the "clean" ground truth for integrity tests).
    """
    raw_scores = []
    for member_id, by_judge in members_judges_scores.items():
        for judge, dims in by_judge.items():
            raw_scores.append(
                {
                    "member_id": member_id,
                    "judge_name": judge,
                    "valuation_reasoning": dims.get("valuation_reasoning"),
                    "horizon_strategy_planning": dims.get("horizon_strategy_planning"),
                    "concession_handling": dims.get("concession_handling"),
                    "outcome_consistency": dims.get("outcome_consistency"),
                    "rationale": "synthetic",
                    "judge_failed_dimensions": [],
                }
            )
    record: dict[str, Any] = {
        "cluster_id": cluster_id,
        "game_family": family,
        "n_members": len(members_judges_scores),
        "n_fallback": 0,
        "fallback_rate": 0.0,
        "all_non_fallback_member_ids": list(members_judges_scores.keys()),
        "sampled_member_ids": list(members_judges_scores.keys()),
        "n_judged_members": len(members_judges_scores),
        "raw_scores": raw_scores,
        "partial_panel_excluded": {},
        "action_format_compliance_rate": None,
        "judge_model": "test-model",
        "temperature": "api_default",
        "scored_at": "2026-01-01T00:00:00+00:00",
    }
    irr = IRRCalculator()
    recomputed = recompute_cluster_kappa(record, _DIMS, irr)
    record["kappa_per_dimension"] = recomputed
    record["kappa_overall"] = (
        sum(recomputed.values()) / len(recomputed) if recomputed else 0.0
    )
    return record


def _make_varied_cluster(
    cluster_id: str,
    family: str,
    n_members: int,
    salt: int,
    concession_all_none: bool = False,
) -> dict[str, Any]:
    """A cluster with patterned (non-degenerate) scores on every dimension."""
    members_judges_scores: dict[str, dict[str, dict[str, int | None]]] = {}
    for mi in range(n_members):
        by_judge: dict[str, dict[str, int | None]] = {}
        for ji, judge in enumerate(_JUDGES):
            dims: dict[str, int | None] = {}
            for di, dim in enumerate(_DIMS):
                if dim == "concession_handling" and concession_all_none:
                    dims[dim] = None
                else:
                    dims[dim] = _patterned_score(mi, ji, di, salt)
            by_judge[judge] = dims
        members_judges_scores[f"m{mi}"] = by_judge
    return _build_record(cluster_id, family, members_judges_scores)


def _make_degenerate_cluster(
    cluster_id: str,
    family: str,
    n_members: int,
    degenerate_dim: str,
    degenerate_value: int,
    salt: int,
) -> dict[str, Any]:
    """A cluster where every judge gives the same score for ``degenerate_dim``."""
    members_judges_scores: dict[str, dict[str, dict[str, int | None]]] = {}
    for mi in range(n_members):
        by_judge: dict[str, dict[str, int | None]] = {}
        for ji, judge in enumerate(_JUDGES):
            dims: dict[str, int | None] = {}
            for di, dim in enumerate(_DIMS):
                if dim == degenerate_dim:
                    dims[dim] = degenerate_value
                elif dim == "concession_handling":
                    dims[dim] = None
                else:
                    dims[dim] = _patterned_score(mi, ji, di, salt)
            by_judge[judge] = dims
        members_judges_scores[f"m{mi}"] = by_judge
    return _build_record(cluster_id, family, members_judges_scores)


def _synthetic_dataset() -> list[dict[str, Any]]:
    """6 clusters (2 per family); one deliberately degenerate; persuasion
    always lacks concession_handling (mirrors the real GLEE data's
    "no prior rejection" N/A pattern for that family)."""
    return [
        _make_varied_cluster("bargaining_001", "bargaining", 4, salt=1),
        _make_degenerate_cluster(
            "bargaining_002", "bargaining", 3, "outcome_consistency", 5, salt=2
        ),
        _make_varied_cluster("negotiation_001", "negotiation", 4, salt=3),
        _make_varied_cluster("negotiation_002", "negotiation", 5, salt=4),
        _make_varied_cluster(
            "persuasion_001", "persuasion", 4, salt=5, concession_all_none=True
        ),
        _make_varied_cluster(
            "persuasion_002", "persuasion", 4, salt=6, concession_all_none=True
        ),
    ]


def _synthetic_manifest() -> dict[str, Any]:
    return {"config": {"dimensions": _DIMS, "judges": _JUDGES}}


class TestLoading:
    def test_load_scored_records_round_trips(self, tmp_path: Path) -> None:
        records = _synthetic_dataset()
        path = tmp_path / "scored.jsonl"
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        loaded = load_scored_records(path)
        assert len(loaded) == len(records)
        assert [r["cluster_id"] for r in loaded] == [r["cluster_id"] for r in records]

    def test_load_scored_records_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_scored_records(tmp_path / "does_not_exist.jsonl")

    def test_load_manifest_round_trips(self, tmp_path: Path) -> None:
        manifest = _synthetic_manifest()
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        assert load_manifest(path) == manifest


class TestIntegrity:
    def test_clean_data_has_zero_mismatches(self) -> None:
        records = _synthetic_dataset()
        report = check_integrity(records, _DIMS)
        assert report["n_mismatches"] == 0
        assert report["mismatches"] == []

    def test_corrupted_dimension_value_is_detected(self) -> None:
        records = _synthetic_dataset()
        corrupted = copy.deepcopy(records)
        corrupted[0]["kappa_per_dimension"]["valuation_reasoning"] += 0.3

        report = check_integrity(corrupted, _DIMS)

        assert report["n_mismatches"] == 1
        mismatch = report["mismatches"][0]
        assert mismatch["cluster_id"] == corrupted[0]["cluster_id"]
        assert mismatch["dimension"] == "valuation_reasoning"
        assert mismatch["abs_diff"] == pytest.approx(0.3)

    def test_kappa_overall_mismatch_is_detected_independently(self) -> None:
        records = _synthetic_dataset()
        corrupted = copy.deepcopy(records)
        corrupted[0]["kappa_overall"] += 0.5

        report = check_integrity(corrupted, _DIMS)

        dims_flagged = {m["dimension"] for m in report["mismatches"]}
        assert "kappa_overall" in dims_flagged
        # Only kappa_overall was corrupted -- no per-dimension entry should
        # also fire for this cluster.
        assert "valuation_reasoning" not in dims_flagged


class TestAggregationArithmetic:
    def test_unweighted_mean_matches_plain_average(self) -> None:
        records = _synthetic_dataset()
        bargaining = [r for r in records if r["game_family"] == "bargaining"]
        assert len(bargaining) == 2

        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        aggregates = build_aggregates(records, _DIMS, _JUDGES, degenerate)

        dim = "horizon_strategy_planning"  # untouched by the degenerate cluster
        expected = sum(r["kappa_per_dimension"][dim] for r in bargaining) / 2
        actual = aggregates["with_degenerate"][dim]["bargaining"]["unweighted_mean"]
        assert actual == pytest.approx(expected)

    def test_member_weighted_mean_matches_manual_weighting(self) -> None:
        records = _synthetic_dataset()
        negotiation = [r for r in records if r["game_family"] == "negotiation"]
        assert len(negotiation) == 2  # n_judged_members 4 and 5 -- unequal weights

        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        aggregates = build_aggregates(records, _DIMS, _JUDGES, degenerate)

        dim = "valuation_reasoning"
        num = sum(
            r["kappa_per_dimension"][dim] * r["n_judged_members"] for r in negotiation
        )
        den = sum(r["n_judged_members"] for r in negotiation)
        expected = num / den
        actual = aggregates["with_degenerate"][dim]["negotiation"][
            "member_weighted_mean"
        ]
        assert actual == pytest.approx(expected)
        # Sanity: with unequal weights, weighted != plain average.
        plain_average = sum(r["kappa_per_dimension"][dim] for r in negotiation) / 2
        assert actual != pytest.approx(plain_average)

    def test_pooled_matches_direct_fleiss_call_on_merged_matrix(self) -> None:
        records = _synthetic_dataset()
        bargaining = [r for r in records if r["game_family"] == "bargaining"]
        dim = "horizon_strategy_planning"

        pooled, n_items = pooled_dimension_kappa(bargaining, dim, _JUDGES, set())

        # Independently build the merged matrix by hand and call the same
        # underlying Fleiss' kappa machinery directly -- this is a second,
        # independent computation, not a hand-derived magic literal.
        from src.annotation.pia_calculator import _build_label_matrix, _fleiss_kappa

        item_ids = []
        score_map = {}
        for r in bargaining:
            by_member: dict[str, dict[str, int]] = {}
            for rs in r["raw_scores"]:
                by_member.setdefault(rs["member_id"], {})[rs["judge_name"]] = rs[dim]
            for member_id, scores in by_member.items():
                item_id = f"{r['cluster_id']}::{member_id}"
                item_ids.append(item_id)
                for judge, score in scores.items():
                    score_map[(item_id, judge)] = score

        matrix = _build_label_matrix(item_ids, _JUDGES, score_map, scale_offset=1)
        expected = _fleiss_kappa(IRRCalculator(), matrix, 5)

        assert n_items == len(item_ids)
        assert pooled == pytest.approx(expected)

        # Pooling is NOT the same as averaging the two clusters' own kappas.
        naive_average = (
            bargaining[0]["kappa_per_dimension"][dim]
            + bargaining[1]["kappa_per_dimension"][dim]
        ) / 2
        assert pooled != pytest.approx(naive_average)

    def test_pooled_excludes_cluster_when_asked(self) -> None:
        records = _synthetic_dataset()
        bargaining = [r for r in records if r["game_family"] == "bargaining"]
        dim = "horizon_strategy_planning"

        excluded_id = bargaining[1]["cluster_id"]
        pooled_without, n_without = pooled_dimension_kappa(
            bargaining, dim, _JUDGES, {excluded_id}
        )
        pooled_with, n_with = pooled_dimension_kappa(bargaining, dim, _JUDGES, set())

        assert n_without < n_with
        assert pooled_without != pytest.approx(pooled_with)


class TestDegenerateDetection:
    def test_forced_entry_found_with_expected_fields(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)

        assert len(degenerate) == 1
        entry = degenerate[0]
        assert entry["cluster_id"] == "bargaining_002"
        assert entry["dimension"] == "outcome_consistency"
        assert entry["uniform_score_value"] == 5
        assert entry["n_items_used_in_kappa"] == 3
        assert entry["n_judged_members"] == 3

    def test_non_degenerate_clusters_not_flagged(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        flagged_ids = {e["cluster_id"] for e in degenerate}
        assert "bargaining_001" not in flagged_ids
        assert "negotiation_001" not in flagged_ids
        assert "negotiation_002" not in flagged_ids

    def test_without_degenerate_drops_that_cluster_from_its_dimension_only(
        self,
    ) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        aggregates = build_aggregates(records, _DIMS, _JUDGES, degenerate)

        with_cell = aggregates["with_degenerate"]["outcome_consistency"]["bargaining"]
        without_cell = aggregates["without_degenerate"]["outcome_consistency"][
            "bargaining"
        ]
        assert with_cell["n_clusters"] == 2
        assert without_cell["n_clusters"] == 1

        # The degenerate cluster's OTHER dimension is untouched by "without".
        other_dim = "valuation_reasoning"
        with_other = aggregates["with_degenerate"][other_dim]["bargaining"]
        without_other = aggregates["without_degenerate"][other_dim]["bargaining"]
        assert with_other["n_clusters"] == without_other["n_clusters"] == 2


class TestBootstrapDeterminism:
    def test_same_seed_gives_identical_output(self) -> None:
        records = _synthetic_dataset()
        result1 = bootstrap_primary_aggregate(
            records, _DIMS, seed=20261002, n_resamples=200
        )
        result2 = bootstrap_primary_aggregate(
            records, _DIMS, seed=20261002, n_resamples=200
        )
        assert result1 == result2

    def test_different_seed_can_give_different_output(self) -> None:
        records = _synthetic_dataset()
        result_a = bootstrap_primary_aggregate(
            records, _DIMS, seed=20261002, n_resamples=200
        )
        result_b = bootstrap_primary_aggregate(records, _DIMS, seed=1, n_resamples=200)
        assert result_a != result_b

    def test_structurally_all_null_dimension_is_not_fabricated(self) -> None:
        records = _synthetic_dataset()
        result = bootstrap_primary_aggregate(
            records, _DIMS, seed=20261002, n_resamples=200
        )
        cell = result["ci_95"]["concession_handling"]["persuasion"]
        assert cell == {
            "low": None,
            "high": None,
            "n_valid_reps": 0,
            "n_total_reps": 200,
        }
        assert result["fraction_positive"]["concession_handling"]["persuasion"] is None

    def test_ci_scopes_cover_overall_and_every_family(self) -> None:
        records = _synthetic_dataset()
        result = bootstrap_primary_aggregate(
            records, _DIMS, seed=20261002, n_resamples=50
        )
        for stat in ["kappa_overall", *_DIMS]:
            assert set(result["ci_95"][stat].keys()) == {
                "overall",
                "bargaining",
                "negotiation",
                "persuasion",
            }


class TestJudgeAgreementReport:
    def test_distribution_and_fractions_match_manual_pooled_count(self) -> None:
        records = _synthetic_dataset()
        report = judge_agreement_report(records, _DIMS, _JUDGES)

        dim, family = "horizon_strategy_planning", "negotiation"
        family_records = [r for r in records if r["game_family"] == family]
        pooled = [
            rs[dim]
            for r in family_records
            for rs in r["raw_scores"]
            if rs[dim] is not None
        ]
        cell = report[dim][family]
        assert cell["n_scores"] == len(pooled)
        assert cell["distribution"] == {str(v): pooled.count(v) for v in range(1, 6)}
        assert cell["frac_ge_4"] == pytest.approx(
            sum(1 for s in pooled if s >= 4) / len(pooled)
        )
        assert cell["frac_eq_5"] == pytest.approx(
            sum(1 for s in pooled if s == 5) / len(pooled)
        )

    def test_per_judge_mean_matches_manual_average(self) -> None:
        records = _synthetic_dataset()
        report = judge_agreement_report(records, _DIMS, _JUDGES)

        dim, family, judge = "outcome_consistency", "bargaining", _JUDGES[0]
        family_records = [r for r in records if r["game_family"] == family]
        scores = [
            rs[dim]
            for r in family_records
            for rs in r["raw_scores"]
            if rs["judge_name"] == judge and rs[dim] is not None
        ]
        expected = sum(scores) / len(scores)
        assert report[dim][family]["per_judge_mean"][judge] == pytest.approx(expected)

    def test_pairwise_exact_and_within1_agreement_match_manual_calc(self) -> None:
        records = _synthetic_dataset()
        report = judge_agreement_report(records, _DIMS, _JUDGES)

        dim, family = "valuation_reasoning", "bargaining"
        j1, j2 = _JUDGES[0], _JUDGES[1]
        family_records = [r for r in records if r["game_family"] == family]
        by_member: dict[str, dict[str, int | None]] = {}
        for r in family_records:
            for rs in r["raw_scores"]:
                by_member.setdefault(f"{r['cluster_id']}::{rs['member_id']}", {})[
                    rs["judge_name"]
                ] = rs[dim]
        paired = [
            (v[j1], v[j2])
            for v in by_member.values()
            if v.get(j1) is not None and v.get(j2) is not None
        ]
        expected_exact = sum(1 for a, b in paired if a == b) / len(paired)
        expected_within1 = sum(1 for a, b in paired if abs(a - b) <= 1) / len(paired)

        pair = report[dim][family]["pairwise"][f"{j1}|{j2}"]
        assert pair["n_items_paired"] == len(paired)
        assert pair["exact_agreement"] == pytest.approx(expected_exact)
        assert pair["within_1_agreement"] == pytest.approx(expected_within1)

    def test_cohens_kappa_matches_direct_irr_call(self) -> None:
        records = _synthetic_dataset()
        report = judge_agreement_report(records, _DIMS, _JUDGES)

        dim, family = "valuation_reasoning", "negotiation"
        j1, j2 = _JUDGES[0], _JUDGES[2]
        members = {}
        for r in records:
            if r["game_family"] != family:
                continue
            for rs in r["raw_scores"]:
                members.setdefault(f"{r['cluster_id']}::{rs['member_id']}", {})[
                    rs["judge_name"]
                ] = rs[dim]
        a = [
            v[j1]
            for v in members.values()
            if v.get(j1) is not None and v.get(j2) is not None
        ]
        b = [
            v[j2]
            for v in members.values()
            if v.get(j1) is not None and v.get(j2) is not None
        ]
        expected = IRRCalculator().cohens_kappa(a, b)["kappa"]

        actual = report[dim][family]["pairwise"][f"{j1}|{j2}"]["cohens_kappa"]
        assert actual == pytest.approx(expected)

    def test_spearman_matches_scipy_direct_call(self) -> None:
        records = _synthetic_dataset()
        report = judge_agreement_report(records, _DIMS, _JUDGES)

        dim, family = "outcome_consistency", "negotiation"
        j1, j2 = _JUDGES[0], _JUDGES[1]
        members = {}
        for r in records:
            if r["game_family"] != family:
                continue
            for rs in r["raw_scores"]:
                members.setdefault(f"{r['cluster_id']}::{rs['member_id']}", {})[
                    rs["judge_name"]
                ] = rs[dim]
        a = [
            v[j1]
            for v in members.values()
            if v.get(j1) is not None and v.get(j2) is not None
        ]
        b = [
            v[j2]
            for v in members.values()
            if v.get(j1) is not None and v.get(j2) is not None
        ]

        actual = report[dim][family]["pairwise"][f"{j1}|{j2}"]["spearman_rho"]
        if len(set(a)) > 1 and len(set(b)) > 1:
            expected_rho, _ = spearmanr(a, b)
            assert actual == pytest.approx(float(expected_rho))
        else:
            assert actual is None

    def test_persuasion_concession_handling_all_none_is_not_fabricated(self) -> None:
        records = _synthetic_dataset()
        report = judge_agreement_report(records, _DIMS, _JUDGES)

        cell = report["concession_handling"]["persuasion"]
        assert cell["n_scores"] == 0
        assert cell["frac_ge_4"] is None
        assert cell["frac_eq_5"] is None
        assert all(v is None for v in cell["per_judge_mean"].values())
        for pair in cell["pairwise"].values():
            assert pair["n_items_paired"] == 0
            assert pair["exact_agreement"] is None
            assert pair["cohens_kappa"] is None


class TestShuffledClusterFloor:
    def test_cluster_pool_sizes_sum_to_pool_length(self) -> None:
        records = _synthetic_dataset()
        bargaining = [r for r in records if r["game_family"] == "bargaining"]
        sizes, pool = _cluster_pool(bargaining, _DIMS)
        assert sizes == [r["n_judged_members"] for r in bargaining]
        assert sum(sizes) == len(pool)

    def test_chunk_kappa_matches_pooled_dimension_kappa_on_whole_pool(self) -> None:
        records = _synthetic_dataset()
        bargaining = [r for r in records if r["game_family"] == "bargaining"]
        _sizes, pool = _cluster_pool(bargaining, _DIMS)
        irr = IRRCalculator()

        # Treat the WHOLE unshuffled pool as a single chunk: this must equal
        # pooled_dimension_kappa() over the same two clusters (both compute
        # one Fleiss' kappa over every complete-panel item, unweighted by
        # cluster boundaries).
        dim = "horizon_strategy_planning"
        chunk_result = _chunk_kappa(pool, dim, _JUDGES, irr)
        pooled_result, _n = pooled_dimension_kappa(bargaining, dim, _JUDGES, set())
        assert chunk_result == pytest.approx(pooled_result)

    def test_same_seed_gives_identical_output(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        aggregates = build_aggregates(records, _DIMS, _JUDGES, degenerate)

        result1 = shuffled_cluster_floor(
            records, _DIMS, _JUDGES, aggregates, seed=20261002, n_permutations=25
        )
        result2 = shuffled_cluster_floor(
            records, _DIMS, _JUDGES, aggregates, seed=20261002, n_permutations=25
        )
        assert result1 == result2

    def test_real_value_matches_aggregates_input(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        aggregates = build_aggregates(records, _DIMS, _JUDGES, degenerate)

        result = shuffled_cluster_floor(
            records, _DIMS, _JUDGES, aggregates, seed=20261002, n_permutations=25
        )
        for stat in ["kappa_overall", *_DIMS]:
            for family in ["bargaining", "negotiation", "persuasion"]:
                expected = aggregates["with_degenerate"][stat][family][
                    "unweighted_mean"
                ]
                assert result[stat][family]["real"] == expected

    def test_persuasion_concession_handling_floor_is_not_fabricated(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        aggregates = build_aggregates(records, _DIMS, _JUDGES, degenerate)

        result = shuffled_cluster_floor(
            records, _DIMS, _JUDGES, aggregates, seed=20261002, n_permutations=25
        )
        cell = result["concession_handling"]["persuasion"]
        assert cell["shuffled_mean"] is None
        assert cell["ci_low"] is None
        assert cell["n_valid_perms"] == 0


class TestDependenceReport:
    def _fixture(self) -> list[dict[str, Any]]:
        """2 bargaining clusters sharing one game, 1 fully independent."""
        return [
            {
                "cluster_id": "bargaining_001",
                "game_family": "bargaining",
                "sampled_member_ids": [
                    "gameA:1:offer",
                    "gameB:1:offer",
                ],
            },
            {
                "cluster_id": "bargaining_002",
                "game_family": "bargaining",
                "sampled_member_ids": [
                    "gameA:2:decision",  # shares gameA with bargaining_001
                    "gameC:1:offer",
                ],
            },
            {
                "cluster_id": "bargaining_003",
                "game_family": "bargaining",
                "sampled_member_ids": [
                    "gameD:1:offer",  # shares nothing with any other cluster
                ],
            },
        ]

    def test_distinct_games_and_sharing_fraction(self) -> None:
        records = self._fixture()
        report = dependence_report(records)

        bargaining = report["bargaining"]
        assert bargaining["n_clusters"] == 3
        assert bargaining["distinct_games"] == 4  # gameA, gameB, gameC, gameD
        # 2 of 3 clusters (001, 002) share gameA with each other; 003 shares
        # nothing.
        assert bargaining["share_clusters_sharing_a_game"] == pytest.approx(2 / 3)

    def test_mean_clusters_per_game(self) -> None:
        records = self._fixture()
        report = dependence_report(records)
        # gameA -> 2 clusters, gameB -> 1, gameC -> 1, gameD -> 1
        expected = (2 + 1 + 1 + 1) / 4
        assert report["bargaining"]["mean_clusters_per_game"] == pytest.approx(expected)

    def test_other_families_empty_when_absent(self) -> None:
        records = self._fixture()
        report = dependence_report(records)
        for family in ("negotiation", "persuasion"):
            assert report[family]["n_clusters"] == 0
            assert report[family]["distinct_games"] == 0
            assert report[family]["share_clusters_sharing_a_game"] is None


class TestBuildRecordsWithoutDegenerate:
    def test_strips_only_the_flagged_dimension(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        assert len(degenerate) == 1  # bargaining_002 / outcome_consistency

        adjusted = build_records_without_degenerate(records, degenerate)
        adjusted_by_id = {r["cluster_id"]: r for r in adjusted}

        flagged = adjusted_by_id["bargaining_002"]
        assert "outcome_consistency" not in flagged["kappa_per_dimension"]
        assert "valuation_reasoning" in flagged["kappa_per_dimension"]

        untouched = adjusted_by_id["bargaining_001"]
        original = next(r for r in records if r["cluster_id"] == "bargaining_001")
        assert untouched["kappa_per_dimension"] == original["kappa_per_dimension"]

    def test_kappa_overall_recomputed_as_mean_of_remaining_dimensions(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        adjusted = build_records_without_degenerate(records, degenerate)

        flagged = next(r for r in adjusted if r["cluster_id"] == "bargaining_002")
        expected = sum(flagged["kappa_per_dimension"].values()) / len(
            flagged["kappa_per_dimension"]
        )
        assert flagged["kappa_overall"] == pytest.approx(expected)

    def test_original_records_not_mutated(self) -> None:
        records = _synthetic_dataset()
        original_keys = {
            r["cluster_id"]: set(r["kappa_per_dimension"].keys()) for r in records
        }
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        build_records_without_degenerate(records, degenerate)

        for r in records:
            assert (
                set(r["kappa_per_dimension"].keys()) == original_keys[r["cluster_id"]]
            )


class TestCompletePanelScoresSubset:
    """Regression test: completeness must be judged against the REQUESTED
    judges subset, not against however many judges happen to be in the
    raw data (the bug this guards against made every judges-subset call
    -- e.g. a 2-of-3 leave-one-out -- silently return zero items)."""

    def _record_with_three_judges(self) -> dict[str, Any]:
        raw_scores = []
        for mi in range(3):
            for ji, judge in enumerate(_JUDGES):
                raw_scores.append(
                    {
                        "member_id": f"m{mi}",
                        "judge_name": judge,
                        "valuation_reasoning": mi + ji + 1,
                        "horizon_strategy_planning": None,
                        "concession_handling": None,
                        "outcome_consistency": None,
                        "rationale": "x",
                        "judge_failed_dimensions": [],
                    }
                )
        return {"raw_scores": raw_scores}

    def test_full_judges_list_returns_all_complete_members(self) -> None:
        record = self._record_with_three_judges()
        complete = _complete_panel_scores(record, "valuation_reasoning", _JUDGES)
        assert len(complete) == 3
        for scores in complete.values():
            assert set(scores.keys()) == set(_JUDGES)

    def test_two_judge_subset_still_returns_members(self) -> None:
        record = self._record_with_three_judges()
        subset = _JUDGES[:2]
        complete = _complete_panel_scores(record, "valuation_reasoning", subset)
        assert len(complete) == 3  # NOT empty -- this is exactly the regression
        for scores in complete.values():
            assert set(scores.keys()) == set(subset)
            assert _JUDGES[2] not in scores

    def test_dimension_with_no_scores_returns_empty(self) -> None:
        record = self._record_with_three_judges()
        complete = _complete_panel_scores(record, "horizon_strategy_planning", _JUDGES)
        assert complete == {}


class TestGameLevelBootstrap:
    def test_same_seed_gives_identical_output(self) -> None:
        records = _synthetic_dataset()
        result1 = game_level_bootstrap(records, _DIMS, seed=20261002, n_resamples=50)
        result2 = game_level_bootstrap(records, _DIMS, seed=20261002, n_resamples=50)
        assert result1 == result2

    def test_shape_matches_cluster_level_bootstrap(self) -> None:
        records = _synthetic_dataset()
        game_result = game_level_bootstrap(records, _DIMS, seed=1, n_resamples=20)
        cluster_result = bootstrap_primary_aggregate(
            records, _DIMS, seed=1, n_resamples=20
        )
        assert set(game_result["ci_95"].keys()) == set(cluster_result["ci_95"].keys())
        for stat in game_result["ci_95"]:
            assert set(game_result["ci_95"][stat].keys()) == {
                "overall",
                "bargaining",
                "negotiation",
                "persuasion",
            }

    def test_persuasion_concession_handling_not_fabricated(self) -> None:
        records = _synthetic_dataset()
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        adjusted = build_records_without_degenerate(records, degenerate)
        result = game_level_bootstrap(adjusted, _DIMS, seed=20261002, n_resamples=50)
        cell = result["ci_95"]["concession_handling"]["persuasion"]
        assert cell["low"] is None
        assert cell["n_valid_reps"] == 0


class TestOffsetRobustAgreement:
    def test_icc_perfect_agreement_is_one(self) -> None:
        # Every rater agrees exactly within each item; items differ -- the
        # textbook ICC(3,1) = 1.0 case.
        matrix = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0]])
        assert _icc_3_1(matrix) == pytest.approx(1.0)

    def test_icc_invariant_to_constant_per_rater_shift(self) -> None:
        rng = np.random.default_rng(0)
        base = rng.integers(1, 6, size=(12, 3)).astype(float)
        shifted = base + np.array([0.0, 1.5, -2.0])  # constant per-column shift
        assert _icc_3_1(base) == pytest.approx(_icc_3_1(shifted))

    def test_icc_zero_variance_is_zero_not_nan(self) -> None:
        matrix = np.full((5, 3), 3.0)
        assert _icc_3_1(matrix) == 0.0

    def test_report_matches_independent_recomputation(self) -> None:
        records = _synthetic_dataset()
        report = offset_robust_agreement_report(records, _DIMS, _JUDGES)

        dim, family = "horizon_strategy_planning", "negotiation"
        family_records = [r for r in records if r["game_family"] == family]
        members = []
        for r in family_records:
            by_member: dict[str, dict[str, int | None]] = {}
            for rs in r["raw_scores"]:
                by_member.setdefault(rs["member_id"], {})[rs["judge_name"]] = rs[dim]
            members.extend(by_member.values())
        per_judge_mean = {
            j: sum(s for m in members if (s := m.get(j)) is not None)
            / sum(1 for m in members if m.get(j) is not None)
            for j in _JUDGES
        }
        complete = [m for m in members if all(m.get(j) is not None for j in _JUDGES)]
        centered = np.array(
            [[m[j] - per_judge_mean[j] for j in _JUDGES] for m in complete]
        )
        expected_icc = _icc_3_1(centered)

        cell = report[dim][family]
        assert cell["n_items"] == len(complete)
        assert cell["icc_3_1_consistency"] == pytest.approx(expected_icc)

    def test_persuasion_concession_handling_is_null_not_fabricated(self) -> None:
        records = _synthetic_dataset()
        report = offset_robust_agreement_report(records, _DIMS, _JUDGES)
        cell = report["concession_handling"]["persuasion"]
        assert cell["icc_3_1_consistency"] is None
        assert cell["krippendorff_alpha_ordinal"] is None
        assert cell["n_items"] == 0


class TestLeaveOneJudgeOut:
    def test_recompute_is_nonempty_for_clusters_with_enough_members(self) -> None:
        records = _synthetic_dataset()
        remaining = [j for j in _JUDGES if j != "LiteralGroundedness"]
        adjusted = _recompute_kappa_for_judge_subset(records, _DIMS, remaining)
        for original, recomputed in zip(records, adjusted, strict=True):
            if original["n_judged_members"] >= 2:
                assert recomputed["kappa_per_dimension"] != {}

    def test_recompute_matches_pooled_dimension_kappa_directly(self) -> None:
        records = _synthetic_dataset()
        remaining = [j for j in _JUDGES if j != "LiteralGroundedness"]
        adjusted = _recompute_kappa_for_judge_subset(records, _DIMS, remaining)

        record = records[0]
        recomputed = next(
            r for r in adjusted if r["cluster_id"] == record["cluster_id"]
        )
        for dim in _DIMS:
            expected, _n = pooled_dimension_kappa([record], dim, remaining, set())
            assert recomputed["kappa_per_dimension"].get(dim) == (
                pytest.approx(expected) if expected is not None else None
            )

    def test_report_shape_and_excluded_judge(self) -> None:
        records = _synthetic_dataset()
        report = leave_one_judge_out_report(
            records,
            _DIMS,
            _JUDGES,
            excluded_judge="LiteralGroundedness",
            seed=20261002,
            n_permutations=20,
        )
        assert report["excluded_judge"] == "LiteralGroundedness"
        assert "LiteralGroundedness" not in report["remaining_judges"]
        assert set(report["remaining_judges"]) == set(_JUDGES) - {"LiteralGroundedness"}
        assert "aggregates" in report
        assert "shuffled_cluster_floor" in report

    def test_floor_real_value_matches_aggregates(self) -> None:
        records = _synthetic_dataset()
        report = leave_one_judge_out_report(
            records,
            _DIMS,
            _JUDGES,
            excluded_judge="LiteralGroundedness",
            seed=20261002,
            n_permutations=20,
        )
        for stat in ["kappa_overall", *_DIMS]:
            for family in ["bargaining", "negotiation", "persuasion"]:
                expected = report["aggregates"]["with_degenerate"][stat][family][
                    "unweighted_mean"
                ]
                assert report["shuffled_cluster_floor"][stat][family]["real"] == (
                    expected
                )


class TestVarianceDecomposition:
    def test_eta_squared_perfect_separation(self) -> None:
        values = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0]
        sizes = [3, 3]
        assert _eta_squared(values, sizes) == pytest.approx(1.0)

    def test_eta_squared_zero_between_group_variance(self) -> None:
        values = [1.0, 5.0, 1.0, 5.0]
        sizes = [2, 2]
        assert _eta_squared(values, sizes) == pytest.approx(0.0)

    def test_eta_squared_zero_total_variance_is_zero_not_nan(self) -> None:
        assert _eta_squared([3.0, 3.0, 3.0], [1, 2]) == 0.0

    def test_eta_squared_too_few_values_is_none(self) -> None:
        assert _eta_squared([1.0], [1]) is None

    def test_judge_mean_score_averages_available_judges(self) -> None:
        member = {
            _JUDGES[0]: {"valuation_reasoning": 2},
            _JUDGES[1]: {"valuation_reasoning": 4},
            _JUDGES[2]: {"valuation_reasoning": None},
        }
        result = _judge_mean_score(member, "valuation_reasoning", _JUDGES)
        assert result == pytest.approx(3.0)

    def test_judge_mean_score_none_when_all_null(self) -> None:
        member = {j: {"concession_handling": None} for j in _JUDGES}
        assert _judge_mean_score(member, "concession_handling", _JUDGES) is None

    def test_family_judge_means_drops_members_with_zero_signal(self) -> None:
        records = _synthetic_dataset()
        persuasion = [r for r in records if r["game_family"] == "persuasion"]
        values, sizes = _family_judge_means(persuasion, "concession_handling", _JUDGES)
        assert values == []
        assert sizes == []

    def test_same_seed_gives_identical_output(self) -> None:
        records = _synthetic_dataset()
        result1 = variance_decomposition(
            records, _DIMS, _JUDGES, seed=20261002, n_permutations=20
        )
        result2 = variance_decomposition(
            records, _DIMS, _JUDGES, seed=20261002, n_permutations=20
        )
        assert result1 == result2

    def test_real_eta_squared_matches_manual_computation(self) -> None:
        records = _synthetic_dataset()
        result = variance_decomposition(
            records, _DIMS, _JUDGES, seed=20261002, n_permutations=10
        )
        dim, family = "outcome_consistency", "negotiation"
        family_records = [r for r in records if r["game_family"] == family]
        values, sizes = _family_judge_means(family_records, dim, _JUDGES)
        expected = _eta_squared(values, sizes)
        assert result[dim][family]["real_eta_squared"] == pytest.approx(expected)
        assert result[dim][family]["n_members"] == len(values)
        assert result[dim][family]["n_clusters"] == len(sizes)

    def test_overall_member_and_cluster_counts_sum_across_families(self) -> None:
        records = _synthetic_dataset()
        result = variance_decomposition(
            records, _DIMS, _JUDGES, seed=20261002, n_permutations=10
        )
        dim = "horizon_strategy_planning"
        family_members = sum(
            result[dim][f]["n_members"]
            for f in ("bargaining", "negotiation", "persuasion")
        )
        family_clusters = sum(
            result[dim][f]["n_clusters"]
            for f in ("bargaining", "negotiation", "persuasion")
        )
        assert result[dim]["overall"]["n_members"] == family_members
        assert result[dim]["overall"]["n_clusters"] == family_clusters

    def test_persuasion_concession_handling_is_null_not_fabricated(self) -> None:
        records = _synthetic_dataset()
        result = variance_decomposition(
            records, _DIMS, _JUDGES, seed=20261002, n_permutations=10
        )
        cell = result["concession_handling"]["persuasion"]
        assert cell["real_eta_squared"] is None
        assert cell["shuffled_mean"] is None
        assert cell["n_members"] == 0


def _make_cluster_with_games(
    cluster_id: str, family: str, game_ids: list[str], salt: int
) -> dict[str, Any]:
    """A cluster with one member per given game_id (``f"{game_id}:1:offer"``).

    Uses ``random.Random(salt)`` (not :func:`_patterned_score`) for the
    scores: ``_patterned_score``'s ``salt`` term is a constant additive
    shift mod 5, which only relabels categories and leaves Fleiss' kappa
    completely unchanged -- every cluster built with it ends up with the
    IDENTICAL kappa regardless of ``salt``, which is correct kappa
    behaviour but useless for a test that needs genuinely different
    kappa values across clusters (e.g. to get a non-degenerate CI).
    """
    rng = random.Random(salt)
    members_judges_scores: dict[str, dict[str, dict[str, int | None]]] = {}
    for game_id in game_ids:
        by_judge: dict[str, dict[str, int | None]] = {}
        for judge in _JUDGES:
            dims: dict[str, int | None] = {}
            for dim in _DIMS:
                dims[dim] = None if dim == "concession_handling" else rng.randint(1, 5)
            by_judge[judge] = dims
        members_judges_scores[f"{game_id}:1:offer"] = by_judge
    return _build_record(cluster_id, family, members_judges_scores)


class TestClusterComponents:
    def test_disjoint_games_give_singleton_components(self) -> None:
        cluster_a = _make_cluster_with_games(
            "bargaining_001", "bargaining", ["g1", "g2", "g3", "g4", "g5"], salt=1
        )
        cluster_b = _make_cluster_with_games(
            "bargaining_002", "bargaining", ["g6"], salt=2
        )
        components = _cluster_components([cluster_a, cluster_b])
        assert sorted(components) == [["bargaining_001"], ["bargaining_002"]]

    def test_shared_game_merges_clusters_into_one_component(self) -> None:
        cluster_a = _make_cluster_with_games(
            "bargaining_001", "bargaining", ["g1", "g_shared"], salt=1
        )
        cluster_b = _make_cluster_with_games(
            "bargaining_002", "bargaining", ["g_shared", "g2"], salt=2
        )
        cluster_c = _make_cluster_with_games(
            "bargaining_003", "bargaining", ["g3"], salt=3
        )
        components = _cluster_components([cluster_a, cluster_b, cluster_c])
        assert sorted(components) == [
            ["bargaining_001", "bargaining_002"],
            ["bargaining_003"],
        ]

    def test_report_matches_component_function(self) -> None:
        records = _synthetic_dataset()
        report = cluster_components_report(records)
        for family in ("bargaining", "negotiation", "persuasion"):
            family_records = [r for r in records if r["game_family"] == family]
            expected = _cluster_components(family_records)
            assert report[family]["n_components"] == len(expected)
            assert report[family]["n_clusters"] == len(family_records)
            assert report[family]["largest_component_size"] == max(
                (len(c) for c in expected), default=0
            )


def _make_component_clone_cluster(
    cluster_id: str,
    family: str,
    shared_game: str,
    unique_games: list[str],
    base_rng_seed: int,
) -> dict[str, Any]:
    """A cluster sharing ``shared_game`` with others built from the SAME seed.

    Reusing the identical ``base_rng_seed`` across several clusters
    (varying only their own-exclusive ``unique_games``, which don't
    affect the shared slot's scores) gives them genuinely CORRELATED --
    here, identical -- kappa values, simulating the real concern behind
    game-level dependence: clusters sharing a game are not independent
    evidence. Only with this kind of induced correlation does block
    (component) resampling reliably widen the CI relative to i.i.d.
    cluster resampling in a small toy -- independently-random same-sized
    clusters sharing a game purely by graph topology (no score
    correlation) don't reliably exhibit the effect at n=6, since the
    direction of a block bootstrap's width change relative to i.i.d. is
    a genuine finite-sample function of the within-block correlation,
    not a one-line inequality that always points the same way.
    """
    rng = random.Random(base_rng_seed)
    shared_row = {
        judge: {
            dim: (None if dim == "concession_handling" else rng.randint(1, 5))
            for dim in _DIMS
        }
        for judge in _JUDGES
    }
    members_judges_scores = {f"{shared_game}:1:offer": copy.deepcopy(shared_row)}
    for game_id in unique_games:
        members_judges_scores[f"{game_id}:1:offer"] = copy.deepcopy(shared_row)
    return _build_record(cluster_id, family, members_judges_scores)


class TestGameLevelBootstrapCorrectness:
    """Regression tests for the game-vs-cluster-count bug: the original
    implementation drew ``n_games`` cluster-picks per family per
    repetition instead of ``n_clusters``, which over-weighted clusters
    that happen to touch many EXCLUSIVE games relative to clusters
    touching only one -- even when neither shares anything with any
    other cluster. The fix resamples CONNECTED COMPONENTS, so a cluster
    touching 5 exclusive games and a cluster touching 1 exclusive game
    are both singleton components and get equal weight, exactly as a
    plain cluster bootstrap would."""

    def test_reduces_to_cluster_bootstrap_when_components_are_singletons(
        self,
    ) -> None:
        # Cluster A touches 5 EXCLUSIVE games; cluster B touches only 1.
        # Neither shares a game with the other -- both are singleton
        # components despite the lopsided game counts. Under the old
        # (buggy) game-draw implementation, A would have been drawn ~5x
        # as often as B per repetition; under the fix, both get the
        # same 1/2 draw probability as a plain 2-cluster bootstrap.
        cluster_a = _make_cluster_with_games(
            "bargaining_001", "bargaining", ["g1", "g2", "g3", "g4", "g5"], salt=1
        )
        cluster_b = _make_cluster_with_games(
            "bargaining_002", "bargaining", ["g6"], salt=2
        )
        records = [cluster_a, cluster_b]  # negotiation/persuasion absent -> 0 draws

        cluster_result = bootstrap_primary_aggregate(
            records, _DIMS, seed=20261002, n_resamples=500
        )
        game_result = game_level_bootstrap(
            records, _DIMS, seed=20261002, n_resamples=500
        )
        assert cluster_result == game_result

    def test_same_property_holds_after_degenerate_removal(self) -> None:
        """(b): both bootstraps must see the SAME (degenerate-stripped)
        per-cluster kappa values -- exercised here by running both on the
        output of build_records_without_degenerate and confirming the
        singleton-component equivalence still holds on that adjusted
        input."""
        cluster_a = _make_cluster_with_games(
            "bargaining_001", "bargaining", ["g1", "g2", "g3"], salt=1
        )
        cluster_b = _make_cluster_with_games(
            "bargaining_002", "bargaining", ["g4"], salt=2
        )
        records = [cluster_a, cluster_b]
        degenerate = find_degenerate_entries(records, _DIMS, _JUDGES)
        adjusted = build_records_without_degenerate(records, degenerate)

        cluster_result = bootstrap_primary_aggregate(
            adjusted, _DIMS, seed=1, n_resamples=300
        )
        game_result = game_level_bootstrap(adjusted, _DIMS, seed=1, n_resamples=300)
        assert cluster_result == game_result

    def test_shared_component_widens_ci_relative_to_cluster_bootstrap(self) -> None:
        """(d): a component of 4 clusters with genuinely CORRELATED kappa
        (built from the same seed via :func:`_make_component_clone_cluster`
        -- see its docstring for why correlation, not just shared graph
        topology, is what the test needs) must measurably widen the
        game-level CI relative to the plain cluster-level CI, and the
        real point estimate must still fall inside it."""
        clusters = [
            _make_component_clone_cluster(
                "bargaining_001", "bargaining", "g_common", ["a1", "a2", "a3", "a4"], 42
            ),
            _make_component_clone_cluster(
                "bargaining_002", "bargaining", "g_common", ["b1", "b2", "b3", "b4"], 42
            ),
            _make_component_clone_cluster(
                "bargaining_003", "bargaining", "g_common", ["c1", "c2", "c3", "c4"], 42
            ),
            _make_component_clone_cluster(
                "bargaining_004", "bargaining", "g_common", ["d1", "d2", "d3", "d4"], 42
            ),
            _make_cluster_with_games(
                "bargaining_005", "bargaining", ["e1", "e2", "e3", "e4", "e5"], salt=5
            ),
            _make_cluster_with_games(
                "bargaining_006", "bargaining", ["f1", "f2", "f3", "f4", "f5"], salt=6
            ),
        ]
        components = _cluster_components(clusters)
        assert sorted(len(c) for c in components) == [1, 1, 4]

        degenerate = find_degenerate_entries(clusters, _DIMS, _JUDGES)
        aggregates = build_aggregates(clusters, _DIMS, _JUDGES, degenerate)
        cluster_boot = bootstrap_primary_aggregate(
            clusters, _DIMS, seed=20261002, n_resamples=5000
        )
        game_boot = game_level_bootstrap(
            clusters, _DIMS, seed=20261002, n_resamples=5000
        )
        audit = audit_game_level_bootstrap(
            aggregates["without_degenerate"], cluster_boot, game_boot, _DIMS
        )

        cell = audit["kappa_overall"]["bargaining"]
        assert cell["point_estimate_in_game_ci"] is True
        assert cell["game_ci_at_least_as_wide_as_cluster_ci"] is True
        assert cell["game_ci_width"] > cell["cluster_ci_width"]


class TestAuditGameLevelBootstrap:
    """Unit tests for audit_game_level_bootstrap's own arithmetic, fed
    hand-built CI dicts rather than going through the real bootstraps."""

    _SCOPES = ("overall", "bargaining", "negotiation", "persuasion")

    def _ci_block(self, low: float | None, high: float | None) -> dict[str, Any]:
        cell = {"low": low, "high": high, "n_valid_reps": 10, "n_total_reps": 10}
        return {"ci_95": {"kappa_overall": {scope: cell for scope in self._SCOPES}}}

    def _point(self, unweighted: float) -> dict[str, Any]:
        cell = {
            "unweighted_mean": unweighted,
            "member_weighted_mean": unweighted,
            "pooled": unweighted,
            "n_clusters": 5,
        }
        return {"kappa_overall": {scope: cell for scope in self._SCOPES}}

    def test_flags_narrower_game_ci_as_violation(self) -> None:
        aggregates = self._point(0.05)
        cluster_boot = self._ci_block(-0.1, 0.2)
        game_boot = self._ci_block(0.0, 0.1)
        audit = audit_game_level_bootstrap(aggregates, cluster_boot, game_boot, [])
        cell = audit["kappa_overall"]["overall"]
        assert cell["point_estimate_in_cluster_ci"] is True
        assert cell["point_estimate_in_game_ci"] is True
        assert cell["game_ci_at_least_as_wide_as_cluster_ci"] is False

    def test_flags_point_estimate_outside_game_ci(self) -> None:
        aggregates = self._point(0.007)
        cluster_boot = self._ci_block(-0.02, 0.04)
        game_boot = self._ci_block(0.07, 0.09)
        audit = audit_game_level_bootstrap(aggregates, cluster_boot, game_boot, [])
        cell = audit["kappa_overall"]["overall"]
        assert cell["point_estimate_in_game_ci"] is False

    def test_passes_when_wider_and_contains_point_estimate(self) -> None:
        aggregates = self._point(0.05)
        cluster_boot = self._ci_block(0.0, 0.1)
        game_boot = self._ci_block(-0.1, 0.2)
        audit = audit_game_level_bootstrap(aggregates, cluster_boot, game_boot, [])
        cell = audit["kappa_overall"]["overall"]
        assert cell["point_estimate_in_cluster_ci"] is True
        assert cell["point_estimate_in_game_ci"] is True
        assert cell["game_ci_at_least_as_wide_as_cluster_ci"] is True

    def test_null_ci_gives_null_checks_not_fabricated(self) -> None:
        aggregates = self._point(0.05)
        cluster_boot = self._ci_block(None, None)
        game_boot = self._ci_block(None, None)
        audit = audit_game_level_bootstrap(aggregates, cluster_boot, game_boot, [])
        cell = audit["kappa_overall"]["overall"]
        assert cell["point_estimate_in_cluster_ci"] is None
        assert cell["point_estimate_in_game_ci"] is None
        assert cell["game_ci_at_least_as_wide_as_cluster_ci"] is None


# ---------------------------------------------------------------------------
# Action-only within-cluster baseline (no judge scores)
# ---------------------------------------------------------------------------


def _raw_step(
    game_id: str,
    round_num: int,
    phase: str,
    game_family: str,
    action: dict[str, Any] | str,
    game_state: dict[str, Any] | None = None,
    your_player: str = "player_1",
) -> dict[str, Any]:
    """A minimal raw GLEE trajectory record, shaped as
    :func:`~src.rsi_bench.analyze_stratified_pia.load_raw_steps_index` would
    parse from the gzipped JSONL log."""
    fields = list(action.keys()) if isinstance(action, dict) else []
    return {
        "game_id": game_id,
        "game_family": game_family,
        "your_player": your_player,
        "phase": phase,
        "round": round_num,
        "game_state": game_state or {},
        "valid_actions": {"type": phase, "fields": {k: "n/a" for k in fields}},
        "reasoning": "synthetic",
        "action": action,
        "fallback_used": False,
        "model": "test-model",
    }


def _raw_index_from_steps(
    steps: list[dict[str, Any]],
) -> dict[tuple[str, int, str], dict[str, Any]]:
    return {(s["game_id"], s["round"], s["phase"]): s for s in steps}


def _action_cluster_record(
    cluster_id: str, family: str, member_ids: list[str]
) -> dict[str, Any]:
    """A minimal scored-cluster record carrying only what the action-only
    baseline reads (``cluster_id``, ``game_family``, ``sampled_member_ids``)
    -- no ``raw_scores`` key at all, so any accidental judge-score read
    would raise ``KeyError`` rather than silently pass."""
    return {
        "cluster_id": cluster_id,
        "game_family": family,
        "sampled_member_ids": member_ids,
    }


class TestExtractActionFeature:
    def test_bargaining_offer_is_continuous_alice_share(self) -> None:
        step = _raw_step(
            "g1",
            1,
            "offer",
            "bargaining",
            {"alice_gain": 600, "bob_gain": 400},
            {"money_to_divide": 1000},
        )
        feature = extract_action_feature(step, "bargaining")
        assert feature["feature_type"] == "continuous"
        assert feature["value"] == pytest.approx(0.6)

    def test_bargaining_decision_is_categorical(self) -> None:
        step = _raw_step("g1", 1, "decision", "bargaining", {"decision": "Reject"})
        feature = extract_action_feature(step, "bargaining")
        assert feature["feature_type"] == "categorical"
        assert feature["label"] == "decision:reject"

    def test_bargaining_offer_missing_money_is_none(self) -> None:
        step = _raw_step("g1", 1, "offer", "bargaining", {"alice_gain": 600}, {})
        feature = extract_action_feature(step, "bargaining")
        assert feature["feature_type"] is None

    def test_negotiation_offer_is_continuous_price_ratio(self) -> None:
        step = _raw_step(
            "g1",
            1,
            "offer",
            "negotiation",
            {"product_price": 150},
            {"current_player": "player_1", "player_1_value": 75},
        )
        feature = extract_action_feature(step, "negotiation")
        assert feature["feature_type"] == "continuous"
        assert feature["value"] == pytest.approx(2.0)

    def test_negotiation_decision_is_categorical(self) -> None:
        step = _raw_step(
            "g1", 1, "decision", "negotiation", {"decision": "RejectOffer"}
        )
        feature = extract_action_feature(step, "negotiation")
        assert feature["feature_type"] == "categorical"
        assert feature["label"] == "decision:rejectoffer"

    def test_negotiation_offer_missing_value_is_none(self) -> None:
        step = _raw_step("g1", 1, "offer", "negotiation", {"product_price": 150}, {})
        feature = extract_action_feature(step, "negotiation")
        assert feature["feature_type"] is None

    def test_persuasion_buyer_decision_is_categorical(self) -> None:
        step = _raw_step("g1", 1, "buyer_decision", "persuasion", {"decision": "yes"})
        feature = extract_action_feature(step, "persuasion")
        assert feature["feature_type"] == "categorical"
        assert feature["label"] == "decision:yes"

    def test_persuasion_seller_message_binary_decision(self) -> None:
        step = _raw_step("g1", 1, "seller_message", "persuasion", {"decision": "no"})
        feature = extract_action_feature(step, "persuasion")
        assert feature["label"] == "decision:no"

    def test_persuasion_message_present(self) -> None:
        step = _raw_step(
            "g1", 1, "seller_message", "persuasion", {"message": "hello there"}
        )
        feature = extract_action_feature(step, "persuasion")
        assert feature["feature_type"] == "categorical"
        assert feature["label"] == "message:present"

    def test_persuasion_message_absent_for_empty_string(self) -> None:
        step = _raw_step("g1", 1, "seller_message", "persuasion", {"message": ""})
        feature = extract_action_feature(step, "persuasion")
        assert feature["label"] == "message:absent"

    def test_non_dict_action_is_none(self) -> None:
        step = _raw_step("g1", 1, "offer", "bargaining", "not-a-dict")
        feature = extract_action_feature(step, "bargaining")
        assert feature["feature_type"] is None

    def test_unknown_family_is_none(self) -> None:
        step = _raw_step("g1", 1, "offer", "mystery", {"decision": "yes"})
        feature = extract_action_feature(step, "mystery")
        assert feature["feature_type"] is None


class TestParseMemberIdAndHistoryLength:
    def test_parses_three_colon_parts(self) -> None:
        assert _parse_member_id("game-1:3:offer") == ("game-1", 3, "offer")

    def test_rejects_wrong_part_count(self) -> None:
        assert _parse_member_id("game-1-offer") is None

    def test_rejects_non_integer_round(self) -> None:
        assert _parse_member_id("game-1:x:offer") is None

    def test_history_length_counts_list(self) -> None:
        step = _raw_step("g1", 1, "decision", "bargaining", {}, {"history": [{}, {}]})
        assert _history_length(step) == 2

    def test_history_length_none_when_missing(self) -> None:
        step = _raw_step("g1", 1, "decision", "bargaining", {}, {})
        assert _history_length(step) is None


class TestLoadRawStepsIndex:
    def test_loads_and_dedupes(self, tmp_path: Path) -> None:
        path = tmp_path / "trajectories.jsonl.gz"
        step1 = _raw_step(
            "g1", 1, "offer", "bargaining", {"alice_gain": 1, "bob_gain": 1}
        )
        step2 = _raw_step("g2", 1, "decision", "bargaining", {"decision": "accept"})
        lines = [json.dumps(step1), json.dumps(step2), json.dumps(step2)]
        with gzip.open(path, "wt") as f:
            f.write("\n".join(lines) + "\n")
        index = load_raw_steps_index(path)
        assert len(index) == 2
        assert index[("g1", 1, "offer")]["action"]["alice_gain"] == 1
        assert index[("g2", 1, "decision")]["action"]["decision"] == "accept"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "trajectories.jsonl.gz"
        step1 = _raw_step(
            "g1", 1, "offer", "bargaining", {"alice_gain": 1, "bob_gain": 1}
        )
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(step1) + "\n")
            f.write("{not valid json\n")
            f.write(json.dumps({"game_id": "g2"}) + "\n")
        index = load_raw_steps_index(path)
        assert len(index) == 1


class TestResolveMemberAndClusterFeatures:
    def test_resolve_member_returns_none_for_unresolvable_id(self) -> None:
        assert _resolve_member("not-colon-formatted", "bargaining", {}) is None
        assert _resolve_member("g1:1:offer", "bargaining", {}) is None

    def test_resolve_member_extracts_round_phase_history_feature(self) -> None:
        step = _raw_step(
            "g1",
            2,
            "offer",
            "bargaining",
            {"alice_gain": 300, "bob_gain": 700},
            {"money_to_divide": 1000, "history": [{"decision": "reject"}]},
        )
        index = {("g1", 2, "offer"): step}
        resolved = _resolve_member("g1:2:offer", "bargaining", index)
        assert resolved is not None
        assert resolved["round"] == 2
        assert resolved["phase"] == "offer"
        assert resolved["history_length"] == 1
        assert resolved["feature_type"] == "continuous"
        assert resolved["value"] == pytest.approx(0.3)

    def test_cluster_membership_features_counts_unresolved(self) -> None:
        step = _raw_step("g1", 1, "decision", "bargaining", {"decision": "accept"})
        index = {("g1", 1, "decision"): step}
        record = _action_cluster_record(
            "bargaining_001", "bargaining", ["g1:1:decision", "g2:1:decision"]
        )
        feats = _cluster_membership_features(record, index)
        assert feats["n_sampled_members"] == 2
        assert feats["n_resolved"] == 1
        assert feats["n_unresolved"] == 1
        assert len(feats["members"]) == 1


class TestActionSpreadAndHelpers:
    def test_sd_requires_two_values(self) -> None:
        assert _sd([1.0]) is None
        assert _sd([1.0, 3.0]) == pytest.approx(float(np.std([1.0, 3.0], ddof=1)))

    def test_modal_share_empty(self) -> None:
        assert _modal_share([]) == (None, None, 0)

    def test_modal_share_basic(self) -> None:
        share, label, n_distinct = _modal_share(["a", "a", "b"])
        assert share == pytest.approx(2 / 3)
        assert label == "a"
        assert n_distinct == 2

    def test_split_features(self) -> None:
        members = [
            {"feature_type": "continuous", "value": 0.5, "label": None},
            {"feature_type": "categorical", "value": None, "label": "decision:yes"},
        ]
        continuous, categorical = _split_features(members)
        assert continuous == [0.5]
        assert categorical == ["decision:yes"]

    def test_action_spread_prefers_continuous_when_available(self) -> None:
        members = [
            {"feature_type": "continuous", "value": 0.2, "label": None},
            {"feature_type": "continuous", "value": 0.8, "label": None},
            {"feature_type": "categorical", "value": None, "label": "decision:yes"},
            {"feature_type": "categorical", "value": None, "label": "decision:no"},
        ]
        spread, source = _action_spread(members)
        assert source == "continuous_sd"
        assert spread == pytest.approx(float(np.std([0.2, 0.8], ddof=1)))

    def test_action_spread_falls_back_to_categorical(self) -> None:
        members = [
            {"feature_type": "categorical", "value": None, "label": "decision:yes"},
            {"feature_type": "categorical", "value": None, "label": "decision:yes"},
            {"feature_type": "categorical", "value": None, "label": "decision:no"},
        ]
        spread, source = _action_spread(members)
        assert source == "categorical_1_minus_modal_share"
        assert spread == pytest.approx(1 - 2 / 3)

    def test_action_spread_none_when_insufficient(self) -> None:
        members = [{"feature_type": "continuous", "value": 0.5, "label": None}]
        spread, source = _action_spread(members)
        assert spread is None
        assert source is None


class TestActionOnlyBaselineReport:
    def _bargaining_fixture(
        self,
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
        steps = [
            _raw_step(
                "g1",
                1,
                "offer",
                "bargaining",
                {"alice_gain": 200, "bob_gain": 800},
                {"money_to_divide": 1000},
            ),
            _raw_step(
                "g2",
                1,
                "offer",
                "bargaining",
                {"alice_gain": 800, "bob_gain": 200},
                {"money_to_divide": 1000},
            ),
            _raw_step("g3", 1, "decision", "bargaining", {"decision": "accept"}),
            _raw_step("g4", 1, "decision", "bargaining", {"decision": "reject"}),
        ]
        index = _raw_index_from_steps(steps)
        record = _action_cluster_record(
            "bargaining_001",
            "bargaining",
            ["g1:1:offer", "g2:1:offer", "g3:1:decision", "g4:1:decision"],
        )
        return [record], index

    def test_per_cluster_row_prefers_continuous(self) -> None:
        records, index = self._bargaining_fixture()
        report = action_only_baseline_report(records, index)
        row = report["per_cluster"]["bargaining_001"]
        assert row["n_continuous"] == 2
        assert row["n_categorical"] == 2
        assert row["action_spread_source"] == "continuous_sd"
        assert row["continuous_sd"] == pytest.approx(float(np.std([0.2, 0.8], ddof=1)))
        assert row["action_spread"] == row["continuous_sd"]

    def test_family_summary_means_over_defined_clusters_only(self) -> None:
        records, index = self._bargaining_fixture()
        lone_step = _raw_step("g5", 1, "decision", "bargaining", {"decision": "accept"})
        index[("g5", 1, "decision")] = lone_step
        records.append(
            _action_cluster_record("bargaining_002", "bargaining", ["g5:1:decision"])
        )
        report = action_only_baseline_report(records, index)
        summary = report["family_summary"]["bargaining"]
        assert summary["n_clusters"] == 2
        assert summary["n_clusters_with_action_spread"] == 1
        assert summary["mean_action_spread"] == pytest.approx(
            report["per_cluster"]["bargaining_001"]["action_spread"]
        )

    def test_never_reads_judge_scores(self) -> None:
        records, index = self._bargaining_fixture()
        assert "raw_scores" not in records[0]
        action_only_baseline_report(records, index)  # must not raise KeyError


class TestActionShuffledClusterFloor:
    def _fixture(
        self,
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
        steps = []
        member_ids = []
        for i in range(10):
            gid = f"g{i}"
            decision = "accept" if i % 3 else "reject"
            steps.append(
                _raw_step(gid, 1, "decision", "bargaining", {"decision": decision})
            )
            member_ids.append(f"{gid}:1:decision")
        index = _raw_index_from_steps(steps)
        records = [
            _action_cluster_record("bargaining_001", "bargaining", member_ids[:5]),
            _action_cluster_record("bargaining_002", "bargaining", member_ids[5:]),
        ]
        return records, index

    def test_deterministic_same_seed(self) -> None:
        records, index = self._fixture()
        baseline = action_only_baseline_report(records, index)
        floor1 = action_shuffled_cluster_floor(
            records, index, baseline, seed=42, n_permutations=25
        )
        floor2 = action_shuffled_cluster_floor(
            records, index, baseline, seed=42, n_permutations=25
        )
        assert floor1 == floor2

    def test_preserves_cluster_sizes_pool(self) -> None:
        records, index = self._fixture()
        sizes, pool = _action_pool(records, index)
        assert sizes == [5, 5]
        assert len(pool) == 10

    def test_real_value_matches_baseline(self) -> None:
        records, index = self._fixture()
        baseline = action_only_baseline_report(records, index)
        floor = action_shuffled_cluster_floor(
            records, index, baseline, seed=1, n_permutations=10
        )
        assert floor["bargaining"]["real"] == pytest.approx(
            baseline["family_summary"]["bargaining"]["mean_action_spread"]
        )

    def test_no_value_fabricated_for_empty_family(self) -> None:
        records, index = self._fixture()
        baseline = action_only_baseline_report(records, index)
        floor = action_shuffled_cluster_floor(
            records, index, baseline, seed=1, n_permutations=5
        )
        assert floor["negotiation"]["real"] is None
        assert floor["negotiation"]["shuffled_mean"] is None
        assert floor["negotiation"]["n_valid_perms"] == 0


class TestMembershipHomogeneityReport:
    def test_detects_mixed_phase_cluster(self) -> None:
        steps = [
            _raw_step(
                "g1",
                1,
                "offer",
                "bargaining",
                {"alice_gain": 1, "bob_gain": 1},
                {"money_to_divide": 2},
            ),
            _raw_step("g2", 1, "decision", "bargaining", {"decision": "accept"}),
        ]
        index = _raw_index_from_steps(steps)
        records = [
            _action_cluster_record(
                "bargaining_001", "bargaining", ["g1:1:offer", "g2:1:decision"]
            )
        ]
        report = membership_homogeneity_report(records, index)
        assert report["bargaining"]["n_clusters_mixed_phase"] == 1
        assert report["bargaining"]["mean_phase_modal_share"] == pytest.approx(0.5)

    def test_homogeneous_cluster_has_full_modal_share(self) -> None:
        steps = [
            _raw_step(
                "g1",
                1,
                "decision",
                "bargaining",
                {"decision": "accept"},
                {"history": []},
            ),
            _raw_step(
                "g2",
                1,
                "decision",
                "bargaining",
                {"decision": "reject"},
                {"history": []},
            ),
        ]
        index = _raw_index_from_steps(steps)
        records = [
            _action_cluster_record(
                "bargaining_001", "bargaining", ["g1:1:decision", "g2:1:decision"]
            )
        ]
        report = membership_homogeneity_report(records, index)
        assert report["bargaining"]["n_clusters_mixed_phase"] == 0
        assert report["bargaining"]["mean_phase_modal_share"] == pytest.approx(1.0)
        assert report["bargaining"]["mean_round_modal_share"] == pytest.approx(1.0)
        assert report["bargaining"]["mean_history_length_sd"] == pytest.approx(0.0)

    def test_empty_family_gives_none_not_fabricated(self) -> None:
        report = membership_homogeneity_report([], {})
        assert report["bargaining"]["mean_phase_modal_share"] is None
        assert report["bargaining"]["n_clusters"] == 0


def _judge_score_record(
    cluster_id: str, family: str, member_scores: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """A minimal scored-cluster record carrying only ``raw_scores`` for
    ``valuation_reasoning`` -- enough for :func:`_cluster_judge_mean_spread`
    without needing a full kappa setup. ``member_scores``: member_id ->
    ``{judge_name: score}``."""
    raw_scores = []
    for member_id, by_judge in member_scores.items():
        for judge, score in by_judge.items():
            raw_scores.append(
                {
                    "member_id": member_id,
                    "judge_name": judge,
                    "valuation_reasoning": score,
                    "horizon_strategy_planning": None,
                    "concession_handling": None,
                    "outcome_consistency": None,
                    "rationale": "synthetic",
                    "judge_failed_dimensions": [],
                }
            )
    return {"cluster_id": cluster_id, "game_family": family, "raw_scores": raw_scores}


class TestClusterJudgeMeanSpread:
    def test_sd_of_judge_mean_scores(self) -> None:
        record = _judge_score_record(
            "bargaining_001",
            "bargaining",
            {
                "m1": {"GameTheoreticRigor": 1, "LiteralGroundedness": 1},
                "m2": {"GameTheoreticRigor": 5, "LiteralGroundedness": 5},
            },
        )
        spread = _cluster_judge_mean_spread(
            record, ["valuation_reasoning"], list(_JUDGES)
        )
        assert spread == pytest.approx(float(np.std([1.0, 5.0], ddof=1)))

    def test_none_when_fewer_than_two_members(self) -> None:
        record = _judge_score_record(
            "bargaining_001", "bargaining", {"m1": {"GameTheoreticRigor": 3}}
        )
        spread = _cluster_judge_mean_spread(
            record, ["valuation_reasoning"], list(_JUDGES)
        )
        assert spread is None

    def test_averages_across_defined_dimensions(self) -> None:
        raw_scores = [
            {
                "member_id": "m1",
                "judge_name": "GameTheoreticRigor",
                "valuation_reasoning": 1,
                "horizon_strategy_planning": 2,
                "concession_handling": None,
                "outcome_consistency": None,
                "rationale": "x",
                "judge_failed_dimensions": [],
            },
            {
                "member_id": "m2",
                "judge_name": "GameTheoreticRigor",
                "valuation_reasoning": 5,
                "horizon_strategy_planning": 2,
                "concession_handling": None,
                "outcome_consistency": None,
                "rationale": "x",
                "judge_failed_dimensions": [],
            },
        ]
        record = {
            "cluster_id": "c1",
            "game_family": "bargaining",
            "raw_scores": raw_scores,
        }
        spread = _cluster_judge_mean_spread(
            record, ["valuation_reasoning", "horizon_strategy_planning"], list(_JUDGES)
        )
        expected = (float(np.std([1.0, 5.0], ddof=1)) + 0.0) / 2
        assert spread == pytest.approx(expected)


class TestActionJudgeSpreadCorrelation:
    """Builds 4 bargaining clusters whose action spread and judge-mean
    spread both increase strictly cluster-over-cluster (by construction,
    not by chance), so Spearman rho is exactly 1.0 -- an unambiguous
    rank-correlation check rather than a fuzzy sign assertion. Negotiation
    and persuasion are left with zero clusters to exercise the
    insufficient-data -> None branch."""

    def _build(
        self,
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]]]:
        # Cluster A: 4x "accept" -> modal_share=1.0 -> action_spread=0.0
        # Cluster B: 3x "accept" + 1x "reject" -> modal_share=0.75 -> spread=0.25
        # Cluster C: 2x "accept" + 2x "reject" -> modal_share=0.5 -> spread=0.5
        # Cluster D: "accept"/"reject"/"walkaway" (3 distinct, 3 members)
        #   -> modal_share=1/3 -> spread=2/3
        decision_plan = {
            "bargaining_A": ["accept"] * 4,
            "bargaining_B": ["accept", "accept", "accept", "reject"],
            "bargaining_C": ["accept", "accept", "reject", "reject"],
            "bargaining_D": ["accept", "reject", "walkaway"],
        }
        # Judge-mean values chosen so their SD is strictly increasing in the
        # same cluster order as the action spread above.
        judge_plan = {
            "bargaining_A": [3, 3, 3, 3],
            "bargaining_B": [3, 3, 3, 4],
            "bargaining_C": [2, 3, 4, 5],
            "bargaining_D": [1, 2, 5],
        }
        steps: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        game_counter = 0
        for cluster_id, decisions in decision_plan.items():
            member_ids = []
            raw_scores = []
            judge_values = judge_plan[cluster_id]
            for decision, judge_value in zip(decisions, judge_values, strict=True):
                game_counter += 1
                game_id = f"g{game_counter}"
                steps.append(
                    _raw_step(
                        game_id, 1, "decision", "bargaining", {"decision": decision}
                    )
                )
                member_id = f"{game_id}:1:decision"
                member_ids.append(member_id)
                raw_scores.append(
                    {
                        "member_id": member_id,
                        "judge_name": "GameTheoreticRigor",
                        "valuation_reasoning": judge_value,
                        "horizon_strategy_planning": None,
                        "concession_handling": None,
                        "outcome_consistency": None,
                        "rationale": "x",
                        "judge_failed_dimensions": [],
                    }
                )
            records.append(
                {
                    "cluster_id": cluster_id,
                    "game_family": "bargaining",
                    "sampled_member_ids": member_ids,
                    "raw_scores": raw_scores,
                }
            )
        index = _raw_index_from_steps(steps)
        return records, index

    def test_perfect_rank_correlation_detected(self) -> None:
        records, index = self._build()
        baseline = action_only_baseline_report(records, index)
        # Sanity-check the designed monotonic action spreads before
        # trusting the correlation result.
        spreads = [
            baseline["per_cluster"][cid]["action_spread"]
            for cid in ("bargaining_A", "bargaining_B", "bargaining_C", "bargaining_D")
        ]
        assert spreads == sorted(spreads)
        assert len(set(spreads)) == 4

        result = action_judge_spread_correlation(
            records, baseline, ["valuation_reasoning"], ["GameTheoreticRigor"]
        )
        assert result["bargaining"]["n_clusters_used"] == 4
        assert result["bargaining"]["spearman_rho"] == pytest.approx(1.0)
        assert result["bargaining"]["spearman_p"] is not None

    def test_none_when_family_has_no_clusters(self) -> None:
        records, index = self._build()
        baseline = action_only_baseline_report(records, index)
        result = action_judge_spread_correlation(
            records, baseline, ["valuation_reasoning"], ["GameTheoreticRigor"]
        )
        for family in ("negotiation", "persuasion"):
            assert result[family]["n_clusters_used"] == 0
            assert result[family]["spearman_rho"] is None
            assert result[family]["spearman_p"] is None
