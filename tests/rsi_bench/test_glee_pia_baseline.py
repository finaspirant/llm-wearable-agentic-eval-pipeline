"""Tests for src/rsi_bench/glee_pia_baseline.py.

No test in this file touches the Anthropic API: GLEEPIABaseline and
_score_cluster are always exercised with a _StubJudgeClient injected via
the judge_client constructor argument, matching the stub-injection pattern
used in tests/annotation/test_pia_scorer.py.

Test inventory:
  TestLoadGleeLog          — JSONL loading, malformed-line skip
  TestActionFormatCompliance — deterministic action/valid_actions matching
  TestConcessionApplicable — prior-rejection heuristic
  TestClusterSignatures    — per-family coarse signature functions
  TestClusterDecisions     — grouping, min_cluster_size, unclustered count
  TestExtractJson          — bare / fenced / embedded JSON parsing
  TestAnnotateCluster      — fallback exclusion, concession None-ing
  TestDimensionKappa       — reuse of pia_calculator's Fleiss' kappa path
  TestScoreCluster         — per-cluster aggregation, compliance diagnostic
  TestGLEEPIABaseline      — end-to-end run() with stub judge, output schema
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.annotation.irr_calculator import IRRCalculator
from src.rsi_bench.glee_pia_baseline import (
    _JUDGES,
    _MAX_RETRIES,
    DecisionCluster,
    GLEEPIABaseline,
    GLEEReasoningAnnotation,
    GLEEStep,
    _annotate_cluster,
    _bargaining_signature,
    _concession_handling_applicable,
    _dimension_kappa,
    _extract_json,
    _missing_required_dimensions,
    _negotiation_signature,
    _persuasion_signature,
    _score_cluster,
    check_action_format_compliance,
    cluster_decisions,
    load_glee_log,
)

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _bargaining_step(
    *,
    game_id: str = "g1",
    round_: int = 1,
    money: float = 100.0,
    offer: float | None = 50.0,
    max_rounds: int = 10,
    history: list[dict[str, Any]] | None = None,
    reasoning: str = "I should offer half the pot.",
    action: dict[str, Any] | str | None = None,
    fallback_used: bool = False,
    model: str = "claude-sonnet-4-6",
) -> GLEEStep:
    # game_state shape matches a real captured bargaining record:
    # "history" is a server-provided list of prior-round dicts, each with
    # a "decision" field ("reject"/"accept") -- see
    # _concession_handling_applicable.
    game_state: dict[str, Any] = {
        "money_to_divide": money,
        "offer_on_table": offer,
        "max_rounds": max_rounds,
    }
    if history is not None:
        game_state["history"] = history
    return GLEEStep(
        game_id=game_id,
        game_family="bargaining",
        your_player="player_1",
        phase="offer",
        round=round_,
        game_state=game_state,
        valid_actions={"fields": ["offer_amount"]},
        reasoning=reasoning,
        action=action if action is not None else {"offer_amount": 50.0},
        fallback_used=fallback_used,
        model=model,
    )


def _negotiation_step(
    *,
    game_id: str = "n1",
    round_: int = 1,
    role: str = "buyer",
    price: float = 80.0,
    valuation: float = 100.0,
    reasoning: str = "Price is below my valuation, worth countering.",
) -> GLEEStep:
    # game_state shape matches a real captured negotiation record:
    # current_player + {current_player}_role / {current_player}_value are
    # templated keys, not static ones -- see _negotiation_signature.
    return GLEEStep(
        game_id=game_id,
        game_family="negotiation",
        your_player="player_1",
        phase="counter",
        round=round_,
        game_state={
            "current_player": "player_1",
            "player_1_role": role,
            "player_1_value": valuation,
            "price": price,
        },
        valid_actions={"fields": ["counter_price"]},
        reasoning=reasoning,
        action={"counter_price": 85.0},
        fallback_used=False,
        model="claude-sonnet-4-6",
    )


def _persuasion_step(
    *,
    game_id: str = "p1",
    round_: int = 1,
    p: float = 0.5,
    v: float = 1.0,
    u: float = 0.3,
    reasoning: str = "Signal is informative given the prior.",
) -> GLEEStep:
    return GLEEStep(
        game_id=game_id,
        game_family="persuasion",
        your_player="sender",
        phase="signal",
        round=round_,
        game_state={"p": p, "v": v, "u": u},
        valid_actions={"fields": ["signal"]},
        reasoning=reasoning,
        action={"signal": "high"},
        fallback_used=False,
        model="claude-sonnet-4-6",
    )


class _StubJudgeClient:
    """Deterministic judge — no network calls.

    Score depends only on (member_index parity, judge_name) so tests can
    predict exact agreement/disagreement patterns.
    """

    def __init__(self, agree: bool = True) -> None:
        self._agree = agree

    def score_member(
        self, cluster: DecisionCluster, member: GLEEStep, judge_name: str
    ) -> dict[str, Any]:
        if self._agree:
            base = 4
        else:
            # Vary by judge so raters disagree.
            base = {
                "GameTheoreticRigor": 5,
                "LiteralGroundedness": 2,
                "OpponentResponsiveness": 3,
            }[judge_name]
        return {
            "valuation_reasoning": base,
            "horizon_strategy_planning": base,
            "concession_handling": base,
            "outcome_consistency": base,
            "rationale": f"[stub] {judge_name} scored {base}",
        }


class _FlakyMissingKeyJudgeClient:
    """Omits "horizon_strategy_planning" from its response for the first
    ``fail_calls`` calls, then returns a complete response -- simulates a
    judge returning valid-but-incomplete JSON (the real bargaining_007
    failure mode: valid JSON, missing required key, no ValueError raised
    by _extract_json). Set fail_calls higher than _MAX_RETRIES to
    simulate retries being exhausted."""

    def __init__(self, fail_calls: int) -> None:
        self.fail_calls = fail_calls
        self.call_count = 0

    def score_member(
        self, cluster: DecisionCluster, member: GLEEStep, judge_name: str
    ) -> dict[str, Any]:
        self.call_count += 1
        response = {
            "valuation_reasoning": 4,
            "concession_handling": None,
            "outcome_consistency": 4,
            "rationale": "[flaky stub]",
        }
        if self.call_count > self.fail_calls:
            response["horizon_strategy_planning"] = 4
        return response


# ---------------------------------------------------------------------------
# TestLoadGleeLog
# ---------------------------------------------------------------------------


class TestLoadGleeLog:
    def test_loads_valid_records(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        record = {
            "game_id": "g1",
            "game_family": "bargaining",
            "your_player": "player_1",
            "phase": "offer",
            "round": 1,
            "game_state": {"money_to_divide": 100},
            "valid_actions": {"fields": ["offer_amount"]},
            "reasoning": "test",
            "action": {"offer_amount": 50},
            "fallback_used": False,
            "model": "claude-sonnet-4-6",
        }
        path.write_text(json.dumps(record) + "\n")
        steps = load_glee_log(path)
        assert len(steps) == 1
        assert steps[0].game_id == "g1"

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        good = _bargaining_step()
        path.write_text(
            "not json\n"
            + json.dumps(
                {
                    "game_id": good.game_id,
                    "game_family": good.game_family,
                    "your_player": good.your_player,
                    "phase": good.phase,
                    "round": good.round,
                    "game_state": good.game_state,
                    "valid_actions": good.valid_actions,
                    "reasoning": good.reasoning,
                    "action": good.action,
                    "fallback_used": good.fallback_used,
                    "model": good.model,
                }
            )
            + "\n"
            + json.dumps({"missing": "fields"})
            + "\n"
        )
        steps = load_glee_log(path)
        assert len(steps) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_glee_log(tmp_path / "nonexistent.jsonl")

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "log.jsonl"
        record = {
            "game_id": "g1",
            "game_family": "bargaining",
            "your_player": "player_1",
            "phase": "offer",
            "round": 1,
            "game_state": {},
            "valid_actions": {},
            "reasoning": "test",
            "action": {},
            "fallback_used": False,
            "model": "m",
        }
        path.write_text("\n\n" + json.dumps(record) + "\n\n")
        assert len(load_glee_log(path)) == 1


# ---------------------------------------------------------------------------
# TestActionFormatCompliance
# ---------------------------------------------------------------------------


class TestActionFormatCompliance:
    def test_exact_field_match_is_compliant(self) -> None:
        step = _bargaining_step(action={"offer_amount": 50.0})
        assert check_action_format_compliance(step) is True

    def test_extra_field_is_noncompliant(self) -> None:
        step = _bargaining_step(action={"offer_amount": 50.0, "note": "x"})
        assert check_action_format_compliance(step) is False

    def test_missing_field_is_noncompliant(self) -> None:
        step = _bargaining_step(action={})
        assert check_action_format_compliance(step) is False

    def test_no_fields_key_returns_none(self) -> None:
        step = _bargaining_step()
        step.valid_actions = {"type": "structured"}
        assert check_action_format_compliance(step) is None

    def test_non_dict_action_returns_none(self) -> None:
        step = _bargaining_step()
        step.action = "reject"
        assert check_action_format_compliance(step) is None


# ---------------------------------------------------------------------------
# TestConcessionApplicable
# ---------------------------------------------------------------------------


class TestConcessionApplicable:
    def test_true_when_history_has_a_rejection(self) -> None:
        step = _bargaining_step(history=[{"round": 1, "decision": "reject"}])
        assert _concession_handling_applicable(step) is True

    def test_true_when_history_has_a_negotiation_style_rejection(self) -> None:
        # negotiation's real "decision" value is "RejectOffer", not "reject"
        step = _bargaining_step(history=[{"round": 1, "decision": "RejectOffer"}])
        assert _concession_handling_applicable(step) is True

    def test_false_when_history_absent(self) -> None:
        step = _bargaining_step()
        assert _concession_handling_applicable(step) is False

    def test_false_when_history_empty(self) -> None:
        step = _bargaining_step(history=[])
        assert _concession_handling_applicable(step) is False

    def test_false_when_history_has_no_rejection(self) -> None:
        step = _bargaining_step(history=[{"round": 1, "decision": "accept"}])
        assert _concession_handling_applicable(step) is False

    def test_false_when_history_entries_have_no_decision_key(self) -> None:
        # matches real persuasion history entries -- no "decision" field at
        # all, so this must stay False rather than error.
        step = _bargaining_step(history=[{"round": 1, "quality": "low"}])
        assert _concession_handling_applicable(step) is False


# ---------------------------------------------------------------------------
# TestClusterSignatures
# ---------------------------------------------------------------------------


class TestClusterSignatures:
    def test_bargaining_same_state_same_signature(self) -> None:
        a = _bargaining_step(game_id="a", money=100, offer=50, round_=3, max_rounds=10)
        b = _bargaining_step(game_id="b", money=100, offer=50, round_=3, max_rounds=10)
        assert _bargaining_signature(a) == _bargaining_signature(b)

    def test_bargaining_different_money_different_signature(self) -> None:
        a = _bargaining_step(money=100)
        b = _bargaining_step(money=200)
        assert _bargaining_signature(a) != _bargaining_signature(b)

    def test_bargaining_missing_money_returns_none(self) -> None:
        step = _bargaining_step()
        step.game_state = {}
        assert _bargaining_signature(step) is None

    def test_negotiation_same_role_ratio_same_signature(self) -> None:
        a = _negotiation_step(role="buyer", price=80, valuation=100)
        b = _negotiation_step(role="buyer", price=81, valuation=100)
        assert _negotiation_signature(a) == _negotiation_signature(b)

    def test_negotiation_missing_role_returns_none(self) -> None:
        step = _negotiation_step()
        step.game_state = {"price": 80, "valuation": 100}
        assert _negotiation_signature(step) is None

    def test_persuasion_same_pvu_same_signature(self) -> None:
        a = _persuasion_step(p=0.5, v=1.0, u=0.3)
        b = _persuasion_step(p=0.51, v=1.0, u=0.3)
        assert _persuasion_signature(a) == _persuasion_signature(b)

    def test_persuasion_all_missing_returns_none(self) -> None:
        step = _persuasion_step()
        step.game_state = {}
        assert _persuasion_signature(step) is None


# ---------------------------------------------------------------------------
# TestClusterDecisions
# ---------------------------------------------------------------------------


class TestClusterDecisions:
    def test_groups_equivalent_states_across_histories(self) -> None:
        steps = [
            _bargaining_step(game_id="a", reasoning="path A reasoning"),
            _bargaining_step(
                game_id="b", reasoning="totally different path B reasoning"
            ),
        ]
        clusters, unclustered = cluster_decisions(steps, min_cluster_size=2)
        assert len(clusters) == 1
        assert len(clusters[0].members) == 2
        assert unclustered == 0

    def test_singleton_cluster_dropped_by_default_min_size(self) -> None:
        steps = [
            _bargaining_step(game_id="a", money=100),
            _bargaining_step(game_id="b", money=999),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        assert clusters == []

    def test_unrecognised_game_family_counted_unclustered(self) -> None:
        step = _bargaining_step()
        step.game_family = "auction"
        _, unclustered = cluster_decisions([step], min_cluster_size=1)
        assert unclustered == 1

    def test_insufficient_state_counted_unclustered(self) -> None:
        step = _bargaining_step()
        step.game_state = {}
        _, unclustered = cluster_decisions([step], min_cluster_size=1)
        assert unclustered == 1

    def test_cluster_ids_prefixed_by_family(self) -> None:
        steps = [
            _bargaining_step(game_id="a"),
            _bargaining_step(game_id="b"),
            _negotiation_step(game_id="c"),
            _negotiation_step(game_id="d"),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        families = {c.game_family for c in clusters}
        assert families == {"bargaining", "negotiation"}
        for c in clusters:
            assert c.cluster_id.startswith(c.game_family)

    def test_mixed_families_do_not_merge(self) -> None:
        steps = [_bargaining_step(game_id="a"), _negotiation_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=1)
        assert len(clusters) == 2


# ---------------------------------------------------------------------------
# TestExtractJson
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_bare_json(self) -> None:
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_embedded_in_prose(self) -> None:
        text = 'Here is my answer: {"a": 1} — hope that helps.'
        assert _extract_json(text) == {"a": 1}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError):
            _extract_json("no json here at all")


# ---------------------------------------------------------------------------
# TestAnnotateCluster
# ---------------------------------------------------------------------------


class TestAnnotateCluster:
    def test_annotation_count_matches_members_times_judges(self) -> None:
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        annotations, skipped = _annotate_cluster(clusters[0], _StubJudgeClient())
        assert len(annotations) == 2 * 3  # 2 members x 3 judges
        assert skipped == 0

    def test_fallback_members_excluded(self) -> None:
        steps = [
            _bargaining_step(game_id="a"),
            _bargaining_step(game_id="b", fallback_used=True),
            _bargaining_step(game_id="c"),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        annotations, skipped = _annotate_cluster(clusters[0], _StubJudgeClient())
        assert skipped == 1
        assert len(annotations) == 2 * 3  # only 2 non-fallback members

    def test_concession_handling_none_when_not_applicable(self) -> None:
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        annotations, _ = _annotate_cluster(clusters[0], _StubJudgeClient())
        assert all(a.concession_handling is None for a in annotations)

    def test_concession_handling_scored_when_applicable(self) -> None:
        rejected_history = [{"round": 1, "decision": "reject"}]
        steps = [
            _bargaining_step(game_id="a", history=rejected_history),
            _bargaining_step(game_id="b", history=rejected_history),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        annotations, _ = _annotate_cluster(clusters[0], _StubJudgeClient())
        assert all(a.concession_handling is not None for a in annotations)

    def test_missing_required_key_retries_then_succeeds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Real failure mode (bargaining_007): judge returns valid JSON
        missing a required dimension. Should retry via score_member
        rather than crash with a KeyError, and recover if a later
        attempt is complete."""
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        # fail_calls=1: only the very first call (member 0, first judge)
        # fails; the client's fail counter is shared across the whole
        # cluster, so every other (member, judge) pair -- 5 more, since
        # this cluster is 2 members x 3 judges = 6 pairs total -- succeeds
        # on its first attempt.
        client = _FlakyMissingKeyJudgeClient(fail_calls=1)
        with caplog.at_level("WARNING", logger="src.rsi_bench.glee_pia_baseline"):
            annotations, _ = _annotate_cluster(clusters[0], client)
        # no crash -- got a full set of annotations
        assert len(annotations) == 2 * 3
        first = annotations[0]
        assert first.horizon_strategy_planning == 4
        assert first.judge_failed_dimensions == ()
        # first pair: 1 failed attempt + 1 successful retry = 2 calls;
        # remaining 5 pairs: 1 call each (already past fail_calls) = 5.
        assert client.call_count == 2 + 5
        assert any("incomplete" in r.message for r in caplog.records)

    def test_missing_required_key_exhausts_retries_without_crashing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Judge never produces the key within _MAX_RETRIES attempts: must
        record that dimension as None + list it in
        judge_failed_dimensions (NOT silently treated like a legitimate
        concession_handling N/A), log at ERROR, and keep going rather
        than crash the whole cluster."""
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        # fail_calls far larger than any call count this small test could
        # ever reach (2 members x 3 judges x _MAX_RETRIES attempts each,
        # at most), so EVERY pair exhausts all _MAX_RETRIES attempts --
        # avoids the shared-counter "later pairs start succeeding" effect
        # from the smaller fail_calls used in the recovery test above.
        client = _FlakyMissingKeyJudgeClient(fail_calls=100_000)
        with caplog.at_level("WARNING", logger="src.rsi_bench.glee_pia_baseline"):
            annotations, skipped = _annotate_cluster(clusters[0], client)
        assert skipped == 0
        assert len(annotations) == 2 * 3  # still one annotation per (member, judge)
        first = annotations[0]
        assert first.horizon_strategy_planning is None
        assert first.judge_failed_dimensions == ("horizon_strategy_planning",)
        # other dimensions from the same (incomplete) response are untouched
        assert first.valuation_reasoning == 4
        assert first.outcome_consistency == 4
        # every one of the 6 (member, judge) pairs exhausts _MAX_RETRIES
        assert client.call_count == 6 * _MAX_RETRIES
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_judge_failure_stays_distinguishable_from_legitimate_na(self) -> None:
        """concession_handling=None (legitimate N/A) and
        horizon_strategy_planning=None (judge failure) must not collapse
        into the same signal -- judge_failed_dimensions is what tells
        them apart in the output, not just the logs."""
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        client = _FlakyMissingKeyJudgeClient(fail_calls=100_000)
        annotations, _ = _annotate_cluster(clusters[0], client)
        ann = annotations[0]
        assert ann.concession_handling is None
        assert ann.horizon_strategy_planning is None
        assert "concession_handling" not in ann.judge_failed_dimensions
        assert "horizon_strategy_planning" in ann.judge_failed_dimensions


# ---------------------------------------------------------------------------
# TestMissingRequiredDimensions
# ---------------------------------------------------------------------------


class TestMissingRequiredDimensions:
    def test_all_present_returns_empty(self) -> None:
        raw = {
            "valuation_reasoning": 4,
            "horizon_strategy_planning": 3,
            "outcome_consistency": 5,
        }
        assert _missing_required_dimensions(raw) == ()

    def test_absent_key_is_missing(self) -> None:
        raw = {"valuation_reasoning": 4, "outcome_consistency": 5}
        assert _missing_required_dimensions(raw) == ("horizon_strategy_planning",)

    def test_null_value_is_missing(self) -> None:
        raw = {
            "valuation_reasoning": 4,
            "horizon_strategy_planning": None,
            "outcome_consistency": 5,
        }
        assert _missing_required_dimensions(raw) == ("horizon_strategy_planning",)

    def test_non_numeric_value_is_missing(self) -> None:
        raw = {
            "valuation_reasoning": 4,
            "horizon_strategy_planning": "not a number",
            "outcome_consistency": 5,
        }
        assert _missing_required_dimensions(raw) == ("horizon_strategy_planning",)

    def test_concession_handling_never_flagged(self) -> None:
        # concession_handling absent entirely -- must not appear as missing,
        # it's an optional field checked elsewhere.
        raw = {
            "valuation_reasoning": 4,
            "horizon_strategy_planning": 3,
            "outcome_consistency": 5,
        }
        assert "concession_handling" not in _missing_required_dimensions(raw)


# ---------------------------------------------------------------------------
# TestDimensionKappa
# ---------------------------------------------------------------------------


class TestDimensionKappa:
    def test_identical_scores_yield_kappa_one(self) -> None:
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        annotations, _ = _annotate_cluster(clusters[0], _StubJudgeClient(agree=True))
        kappa, n_excluded = _dimension_kappa(
            IRRCalculator(), "valuation_reasoning", annotations
        )
        assert kappa == 1.0
        assert n_excluded == 0

    def test_disagreement_yields_lower_kappa(self) -> None:
        steps = [
            _bargaining_step(game_id=f"g{i}", reasoning=f"reasoning {i}")
            for i in range(4)
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        annotations, _ = _annotate_cluster(clusters[0], _StubJudgeClient(agree=False))
        kappa, n_excluded = _dimension_kappa(
            IRRCalculator(), "valuation_reasoning", annotations
        )
        assert kappa is not None
        assert -1.0 <= kappa < 1.0
        assert n_excluded == 0

    def test_fewer_than_two_items_returns_none(self) -> None:
        steps = [
            _bargaining_step(game_id="a", history=[{"round": 1, "decision": "reject"}])
        ]
        # Build a single-member "cluster" annotation set manually.
        clusters, _ = cluster_decisions(
            [steps[0], _bargaining_step(game_id="b")], min_cluster_size=2
        )
        annotations, _ = _annotate_cluster(clusters[0], _StubJudgeClient())
        # only member "a" has concession_handling applicable -> 1 valid
        # item, below the 2-item minimum for kappa.
        kappa, n_excluded = _dimension_kappa(
            IRRCalculator(), "concession_handling", annotations
        )
        assert kappa is None
        assert n_excluded == 0  # member "b" is uniformly N/A, not a partial panel

    def test_partial_panel_excludes_member_without_crashing(self) -> None:
        """The actual bargaining_007 -> pia_calculator KeyError chain,
        reproduced directly: one judge is None on horizon_strategy_planning
        for member 0 (2/3 judges scored it, 1/3 didn't -- simulating
        exhausted retries) while members 1 and 2 have a complete panel on
        every dimension. horizon_strategy_planning must exclude member 0
        without crashing _build_label_matrix, while valuation_reasoning
        (complete for all 3 members) is scored normally with zero
        exclusions."""
        judges = list(_JUDGES)

        def complete(
            member_index: int, val: int, horizon: int, outcome: int
        ) -> list[GLEEReasoningAnnotation]:
            return [
                GLEEReasoningAnnotation(
                    annotation_id=f"c/{member_index}/{j}",
                    cluster_id="c",
                    member_index=member_index,
                    judge_name=j,
                    valuation_reasoning=val,
                    horizon_strategy_planning=horizon,
                    concession_handling=None,
                    outcome_consistency=outcome,
                    rationale="",
                )
                for j in judges
            ]

        annotations = [
            # member 0: horizon_strategy_planning has a PARTIAL panel --
            # judges[2] returned None (simulating exhausted retries).
            GLEEReasoningAnnotation(
                annotation_id="c/0/j0", cluster_id="c", member_index=0,
                judge_name=judges[0], valuation_reasoning=4,
                horizon_strategy_planning=3, concession_handling=None,
                outcome_consistency=5, rationale="",
            ),
            GLEEReasoningAnnotation(
                annotation_id="c/0/j1", cluster_id="c", member_index=0,
                judge_name=judges[1], valuation_reasoning=4,
                horizon_strategy_planning=3, concession_handling=None,
                outcome_consistency=5, rationale="",
            ),
            GLEEReasoningAnnotation(
                annotation_id="c/0/j2", cluster_id="c", member_index=0,
                judge_name=judges[2], valuation_reasoning=4,
                horizon_strategy_planning=None, concession_handling=None,
                outcome_consistency=5, rationale="",
                judge_failed_dimensions=("horizon_strategy_planning",),
            ),
            *complete(1, val=4, horizon=3, outcome=5),
            *complete(2, val=2, horizon=1, outcome=3),
        ]

        irr = IRRCalculator()

        # horizon_strategy_planning: member 0 is a partial panel -> excluded;
        # members 1+2 have complete panels -> kappa computed over those 2,
        # no crash despite the ragged input.
        kappa_h, n_excluded_h = _dimension_kappa(
            irr, "horizon_strategy_planning", annotations
        )
        assert n_excluded_h == 1
        assert kappa_h is not None

        # valuation_reasoning: every member has a complete panel -> zero
        # partial-panel exclusions, scored normally.
        kappa_v, n_excluded_v = _dimension_kappa(
            irr, "valuation_reasoning", annotations
        )
        assert n_excluded_v == 0
        assert kappa_v is not None


# ---------------------------------------------------------------------------
# TestScoreCluster
# ---------------------------------------------------------------------------


class TestScoreCluster:
    def test_returns_result_for_valid_cluster(self) -> None:
        steps = [_bargaining_step(game_id="a"), _bargaining_step(game_id="b")]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        result = _score_cluster(IRRCalculator(), clusters[0], _StubJudgeClient())
        assert result is not None
        assert result.n_members == 2
        assert result.n_judged_members == 2
        assert 0.0 <= result.kappa_overall or result.kappa_overall <= 1.0

    def test_returns_none_when_all_but_one_fallback(self) -> None:
        steps = [
            _bargaining_step(game_id="a"),
            _bargaining_step(game_id="b", fallback_used=True),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        result = _score_cluster(IRRCalculator(), clusters[0], _StubJudgeClient())
        assert result is None

    def test_action_format_compliance_rate_computed(self) -> None:
        steps = [
            _bargaining_step(game_id="a", action={"offer_amount": 50.0}),
            _bargaining_step(game_id="b", action={"offer_amount": 50.0, "extra": 1}),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        result = _score_cluster(IRRCalculator(), clusters[0], _StubJudgeClient())
        assert result is not None
        assert result.action_format_compliance_rate == 0.5

    def test_fallback_rate_reported(self) -> None:
        steps = [
            _bargaining_step(game_id="a"),
            _bargaining_step(game_id="b"),
            _bargaining_step(game_id="c", fallback_used=True),
        ]
        clusters, _ = cluster_decisions(steps, min_cluster_size=2)
        result = _score_cluster(IRRCalculator(), clusters[0], _StubJudgeClient())
        assert result is not None
        assert result.fallback_rate == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# TestGLEEPIABaseline
# ---------------------------------------------------------------------------


def _write_log(path: Path, steps: list[GLEEStep]) -> None:
    with path.open("w") as f:
        for s in steps:
            f.write(
                json.dumps(
                    {
                        "game_id": s.game_id,
                        "game_family": s.game_family,
                        "your_player": s.your_player,
                        "phase": s.phase,
                        "round": s.round,
                        "game_state": s.game_state,
                        "valid_actions": s.valid_actions,
                        "reasoning": s.reasoning,
                        "action": s.action,
                        "fallback_used": s.fallback_used,
                        "model": s.model,
                    }
                )
                + "\n"
            )


class TestGLEEPIABaseline:
    def _make_log(self, tmp_path: Path) -> Path:
        steps = [
            _bargaining_step(
                game_id="a", money=100, offer=50, round_=1, model="model-x"
            ),
            _bargaining_step(
                game_id="b", money=100, offer=50, round_=1, model="model-y"
            ),
            _negotiation_step(game_id="c", role="buyer", price=80, valuation=100),
            _negotiation_step(game_id="d", role="buyer", price=81, valuation=100),
        ]
        path = tmp_path / "glee_log.jsonl"
        _write_log(path, steps)
        return path

    def test_run_produces_result(self, tmp_path: Path) -> None:
        log_path = self._make_log(tmp_path)
        baseline = GLEEPIABaseline(
            log_path=log_path,
            output_path=tmp_path / "out.json",
            judge_client=_StubJudgeClient(),
        )
        result = baseline.run()
        assert result.n_clusters == 2
        assert result.n_clusters_skipped == 0
        assert set(result.models_represented) >= {"model-x", "model-y"}

    def test_output_file_written(self, tmp_path: Path) -> None:
        log_path = self._make_log(tmp_path)
        out_path = tmp_path / "out.json"
        baseline = GLEEPIABaseline(
            log_path=log_path, output_path=out_path, judge_client=_StubJudgeClient()
        )
        baseline.run()
        assert out_path.exists()
        obj = json.loads(out_path.read_text())
        required = {
            "generated_at",
            "model",
            "judges",
            "n_clusters",
            "per_dimension_kappa",
            "overall_kappa",
            "interpretation",
            "per_cluster",
            "by_game_family",
            "action_format_compliance_overall",
            "notes",
        }
        assert required <= set(obj.keys())

    def test_overall_kappa_is_one_when_judges_agree(self, tmp_path: Path) -> None:
        log_path = self._make_log(tmp_path)
        baseline = GLEEPIABaseline(
            log_path=log_path,
            output_path=tmp_path / "out.json",
            judge_client=_StubJudgeClient(agree=True),
        )
        result = baseline.run()
        assert result.overall_kappa == 1.0
        assert result.interpretation == "almost perfect"

    def test_by_game_family_covers_both_families(self, tmp_path: Path) -> None:
        log_path = self._make_log(tmp_path)
        baseline = GLEEPIABaseline(
            log_path=log_path,
            output_path=tmp_path / "out.json",
            judge_client=_StubJudgeClient(),
        )
        result = baseline.run()
        assert set(result.by_game_family.keys()) == {"bargaining", "negotiation"}

    def test_limit_clusters_caps_scored_clusters(self, tmp_path: Path) -> None:
        log_path = self._make_log(tmp_path)
        baseline = GLEEPIABaseline(
            log_path=log_path,
            output_path=tmp_path / "out.json",
            judge_client=_StubJudgeClient(),
            limit_clusters=1,
        )
        result = baseline.run()
        assert result.n_clusters == 1

    def test_notes_flag_deterministic_and_uncalibrated_caveats(
        self, tmp_path: Path
    ) -> None:
        log_path = self._make_log(tmp_path)
        baseline = GLEEPIABaseline(
            log_path=log_path,
            output_path=tmp_path / "out.json",
            judge_client=_StubJudgeClient(),
        )
        result = baseline.run()
        joined = " ".join(result.notes)
        assert "excluded from overall_kappa" in joined
        assert "not calibrated" in joined

    def test_missing_log_raises(self, tmp_path: Path) -> None:
        baseline = GLEEPIABaseline(
            log_path=tmp_path / "nonexistent.jsonl",
            output_path=tmp_path / "out.json",
            judge_client=_StubJudgeClient(),
        )
        with pytest.raises(FileNotFoundError):
            baseline.run()
