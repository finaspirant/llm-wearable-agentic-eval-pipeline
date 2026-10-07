"""Tests for src/rsi_bench/run_intrastate_replay.py.

No test here makes a network call. Every test that would otherwise hit
the Anthropic API injects a fake client (``_FakeClient`` below) via
:func:`~src.rsi_bench.run_intrastate_replay.run_replay`'s ``client=``
parameter -- a structural stand-in for ``anthropic.Anthropic`` exposing
only ``.messages.create(**kwargs)``, never importing or constructing a
real ``anthropic.Anthropic()``.

Test inventory:
  TestInferActionType          -- phase->type rule per family
  TestBuildValidActions         -- canonical schema lookup + error case
  TestUnwrapSchemaEcho          -- schema-echoed vs. already-flat actions
  TestLoadIntrastateStates      -- synthetic-fixture load, state_id, original_action
  TestBuildPromptForState       -- deviation-1 empty prompt line, history slicing
  TestBudget                    -- max_calls / max_cost_usd cap detection
  TestCircuitBreaker            -- consecutive-failure trip + success reset
  TestSampleOnce                -- parse success/failure on a fake response
  TestLoadCompletedKeys         -- resume-key extraction from an existing JSONL
  TestRunReplayWithMockedClient -- full run: concurrency, resume, budget cap,
                                    circuit breaker, incremental persistence
  TestExtractSampleFeature      -- wraps extract_action_feature correctly
  TestPerStateTempReport        -- agreement/mean-abs-diff arithmetic
  TestAggregateReplay           -- bootstrap determinism, None-on-empty
  TestCompareWithObservationalBaseline -- reads a fixture glee_pia_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.rsi_bench.run_intrastate_replay import (
    _Budget,
    _CircuitBreaker,
    _load_completed_keys,
    _sample_once,
    aggregate_replay,
    build_per_state_temp_reports,
    build_per_state_temp_reports_with_failures_as_category,
    build_prompt_for_state,
    build_valid_actions,
    compare_with_observational_baseline,
    extract_sample_feature,
    infer_action_type,
    load_intrastate_states,
    per_state_temp_report,
    per_state_temp_report_with_failures_as_category,
    run_replay,
    unwrap_schema_echo,
)

# ---------------------------------------------------------------------------
# Fake Anthropic client (no network)
# ---------------------------------------------------------------------------


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(
        self, text: str, input_tokens: int = 100, output_tokens: int = 50
    ) -> None:
        self.content = [_FakeBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, responder: Any) -> None:  # noqa: ANN401
        self._responder = responder

    def create(self, **kwargs: Any) -> _FakeResponse:  # noqa: ANN401
        return self._responder(kwargs)


class _FakeClient:
    """Structural stand-in for ``anthropic.Anthropic`` -- `.messages.create`
    only, driven by a caller-supplied ``responder(kwargs) -> _FakeResponse``
    (or one that raises, to simulate an API-level failure)."""

    def __init__(self, responder: Any) -> None:  # noqa: ANN401
        self.messages = _FakeMessages(responder)


def _make_state(
    state_id: str = "00_bargaining_g1_1",
    game_family: str = "bargaining",
    phase: str = "decision",
    game_state: dict[str, Any] | None = None,
    original_action: Any = None,  # noqa: ANN401
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "index": 0,
        "source_tag": "test",
        "source_file": "test.jsonl",
        "source_description": "synthetic test fixture",
        "game_id": "g1",
        "round": 1,
        "game_family": game_family,
        "phase": phase,
        "game_state": game_state or {"phase": phase},
        "n_archived_calls": 1,
        "action_differs_across_calls_archived": False,
        "original_action": original_action
        if original_action is not None
        else {"decision": "accept"},
    }


class TestInferActionType:
    def test_bargaining_phase_is_type(self) -> None:
        assert infer_action_type("bargaining", "offer", {}) == "offer"
        assert infer_action_type("bargaining", "decision", {}) == "decision"

    def test_negotiation_phase_is_type(self) -> None:
        assert infer_action_type("negotiation", "offer", {}) == "offer"
        assert infer_action_type("negotiation", "decision", {}) == "decision"

    def test_persuasion_buyer_decision(self) -> None:
        assert infer_action_type("persuasion", "buyer_decision", {}) == "buyer_decision"

    def test_persuasion_seller_message_binary(self) -> None:
        state = {"seller_message_type": "binary"}
        assert (
            infer_action_type("persuasion", "seller_message", state)
            == "seller_recommendation"
        )

    def test_persuasion_seller_message_text(self) -> None:
        state = {"seller_message_type": "text"}
        assert (
            infer_action_type("persuasion", "seller_message", state) == "seller_message"
        )

    def test_unknown_family_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer action type"):
            infer_action_type("mystery", "offer", {})


class TestBuildValidActions:
    def test_bargaining_offer_schema(self) -> None:
        va = build_valid_actions("bargaining", "offer", {})
        assert va["type"] == "offer"
        assert set(va["fields"]) == {"alice_gain", "bob_gain", "message"}

    def test_persuasion_seller_message_binary_schema(self) -> None:
        va = build_valid_actions(
            "persuasion", "seller_message", {"seller_message_type": "binary"}
        )
        assert va["type"] == "seller_recommendation"
        assert set(va["fields"]) == {"decision"}

    def test_missing_schema_raises(self) -> None:
        with pytest.raises(ValueError, match="No canonical schema"):
            build_valid_actions("bargaining", "mystery_phase", {})


class TestUnwrapSchemaEcho:
    def test_unwraps_schema_echoed_action(self) -> None:
        wrapped = {"type": "offer", "fields": {"product_price": 100}}
        assert unwrap_schema_echo(wrapped) == {"product_price": 100}

    def test_leaves_flat_action_untouched(self) -> None:
        flat = {"decision": "accept"}
        assert unwrap_schema_echo(flat) == flat

    def test_leaves_non_dict_untouched(self) -> None:
        assert unwrap_schema_echo("not-a-dict") == "not-a-dict"


class TestLoadIntrastateStates:
    def _write_fixture(self, tmp_path: Path) -> Path:
        path = tmp_path / "intrastate_variance.json"
        data = [
            {
                "source_file": "a.jsonl",
                "source_tag": "tag_a",
                "source_description": "desc a",
                "game_id": "gA",
                "round": 3,
                "game_family": "bargaining",
                "action_differs_across_calls": True,
                "game_state": {"phase": "offer", "money_to_divide": 1000},
                "calls": [
                    {
                        "reasoning": "r1",
                        "action": {
                            "type": "offer",
                            "fields": {"alice_gain": 400, "bob_gain": 600},
                        },
                    },
                    {
                        "reasoning": "r2",
                        "action": {
                            "type": "offer",
                            "fields": {"alice_gain": 500, "bob_gain": 500},
                        },
                    },
                ],
            },
            {
                "source_file": "b.jsonl",
                "source_tag": "tag_b",
                "source_description": "desc b",
                "game_id": "gB",
                "round": 1,
                "game_family": "negotiation",
                "action_differs_across_calls": False,
                "game_state": {"phase": "decision", "current_player": "player_1"},
                "calls": [{"reasoning": "r1", "action": {"decision": "AcceptOffer"}}],
            },
        ]
        path.write_text(json.dumps(data))
        return path

    def test_loads_all_states_with_stable_ids(self, tmp_path: Path) -> None:
        path = self._write_fixture(tmp_path)
        states = load_intrastate_states(path)
        assert len(states) == 2
        assert states[0]["state_id"] == "00_bargaining_gA_3"
        assert states[1]["state_id"] == "01_negotiation_gB_1"

    def test_original_action_is_calls_zero_unwrapped(self, tmp_path: Path) -> None:
        path = self._write_fixture(tmp_path)
        states = load_intrastate_states(path)
        assert states[0]["original_action"] == {"alice_gain": 400, "bob_gain": 600}
        assert states[1]["original_action"] == {"decision": "AcceptOffer"}

    def test_real_fixture_has_18_states_in_expected_split(self) -> None:
        states = load_intrastate_states()
        assert len(states) == 18
        tags = [s["source_tag"] for s in states]
        assert sum(1 for t in tags if "run2" in t) == 4
        assert sum(1 for t in tags if "run3" in t) == 12
        assert sum(1 for t in tags if "live" in t) == 2
        assert len({s["state_id"] for s in states}) == 18


class TestBuildPromptForState:
    def test_empty_prompt_line_is_blank_not_keyerror(self) -> None:
        state = _make_state(
            game_state={"phase": "decision", "history": []},
        )
        prompt, game = build_prompt_for_state(state)
        assert prompt.startswith("\n")  # f"{''}\n" -- deviation 1
        assert game["prompt"] == ""

    def test_includes_flat_example_hint(self) -> None:
        state = _make_state(
            game_family="negotiation",
            phase="offer",
            game_state={
                "phase": "offer",
                "current_player": "player_1",
                "player_1_value": 80,
                "history": [],
            },
        )
        prompt, _game = build_prompt_for_state(state)
        assert "Example of a correctly-formatted action for this exact schema" in prompt
        assert (
            '"type"'
            not in prompt.split("Example of")[1].split("Your action must look")[0]
        )

    def test_history_includes_all_prior_rounds_not_the_placeholder(self) -> None:
        state = _make_state(
            game_state={
                "phase": "decision",
                "history": [{"decision": "reject"}, {"decision": "reject"}],
            },
        )
        prompt, _game = build_prompt_for_state(state)
        history_block = prompt.split("Full game state visible to you:")[0]
        # one complete json.dumps(entry) line per history entry (see
        # _format_history_block in the committed agent)
        assert history_block.count('"decision": "reject"') == 2
        assert "_replay_current_round_placeholder" not in prompt


class TestBudget:
    def test_trips_on_max_calls(self) -> None:
        budget = _Budget(max_calls=2, max_cost_usd=None)
        assert budget.record_and_check(10, 10) is None
        assert budget.record_and_check(10, 10) == "max_calls"
        assert budget.tripped() is True

    def test_trips_on_max_cost(self) -> None:
        budget = _Budget(max_calls=None, max_cost_usd=0.001)
        # 1_000_000 input tokens @ $3/MTok = $3 -- trips immediately
        reason = budget.record_and_check(1_000_000, 0)
        assert reason == "max_cost_usd"

    def test_never_trips_within_caps(self) -> None:
        budget = _Budget(max_calls=100, max_cost_usd=100.0)
        for _ in range(5):
            assert budget.record_and_check(100, 50) is None
        assert budget.tripped() is False

    def test_cost_usd_uses_3_and_15_per_mtok(self) -> None:
        budget = _Budget(max_calls=None, max_cost_usd=None)
        budget.record_and_check(1_000_000, 1_000_000)
        assert budget.cost_usd() == pytest.approx(3.0 + 15.0)


class TestCircuitBreaker:
    def test_trips_after_threshold_consecutive_failures(self) -> None:
        breaker = _CircuitBreaker(threshold=3)
        assert breaker.note_failure() is False
        assert breaker.note_failure() is False
        assert breaker.note_failure() is True
        assert breaker.tripped is True

    def test_success_resets_counter(self) -> None:
        breaker = _CircuitBreaker(threshold=3)
        breaker.note_failure()
        breaker.note_failure()
        breaker.note_success()
        assert breaker.consecutive_failures == 0
        assert breaker.note_failure() is False
        assert breaker.tripped is False


class TestSampleOnce:
    def test_successful_parse(self) -> None:
        client = _FakeClient(
            lambda kwargs: _FakeResponse(
                'REASONING: ok\nACTION: {"decision": "accept"}'
            )
        )
        result = _sample_once(client, "claude-sonnet-4-6", "prompt text", None, 60, 600)
        assert result["ok"] is True
        assert result["action"] == {"decision": "accept"}
        assert result["parse_error"] is None
        assert result["usage"]["input_tokens"] == 100

    def test_parse_failure_is_recorded_not_raised(self) -> None:
        client = _FakeClient(lambda kwargs: _FakeResponse("garbage, no action at all"))
        result = _sample_once(client, "claude-sonnet-4-6", "prompt text", None, 60, 600)
        assert result["ok"] is False
        assert result["action"] is None
        assert result["parse_error"] is not None
        assert result["usage"]["input_tokens"] == 100

    def test_temperature_omitted_when_none(self) -> None:
        seen_kwargs = {}

        def responder(kwargs: Any) -> _FakeResponse:  # noqa: ANN401
            seen_kwargs.update(kwargs)
            return _FakeResponse('REASONING: x\nACTION: {"decision": "accept"}')

        client = _FakeClient(responder)
        _sample_once(client, "claude-sonnet-4-6", "prompt", None, 60, 600)
        assert "temperature" not in seen_kwargs

    def test_temperature_passed_when_given(self) -> None:
        seen_kwargs = {}

        def responder(kwargs: Any) -> _FakeResponse:  # noqa: ANN401
            seen_kwargs.update(kwargs)
            return _FakeResponse('REASONING: x\nACTION: {"decision": "accept"}')

        client = _FakeClient(responder)
        _sample_once(client, "claude-sonnet-4-6", "prompt", 0.0, 60, 600)
        assert seen_kwargs["temperature"] == 0.0


class TestLoadCompletedKeys:
    def test_empty_when_file_missing(self, tmp_path: Path) -> None:
        assert _load_completed_keys(tmp_path / "missing.jsonl") == set()

    def test_reads_existing_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "scored.jsonl"
        path.write_text(
            json.dumps({"state_id": "s1", "temp_label": "t0", "sample_idx": 0})
            + "\n"
            + json.dumps({"state_id": "s1", "temp_label": "t0", "sample_idx": 1})
            + "\n"
        )
        keys = _load_completed_keys(path)
        assert keys == {("s1", "t0", 0), ("s1", "t0", 1)}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "scored.jsonl"
        path.write_text(
            "{not json\n"
            + json.dumps({"state_id": "s1", "temp_label": "t0", "sample_idx": 0})
            + "\n"
        )
        assert _load_completed_keys(path) == {("s1", "t0", 0)}


class TestRunReplayWithMockedClient:
    def _states(self) -> list[dict[str, Any]]:
        return [
            _make_state(state_id="00_bargaining_g1_1"),
            _make_state(state_id="01_negotiation_g2_1", game_family="negotiation"),
        ]

    def test_full_run_writes_all_samples(self, tmp_path: Path) -> None:
        output = tmp_path / "scored.jsonl"
        client = _FakeClient(
            lambda kwargs: _FakeResponse('REASONING: x\nACTION: {"decision": "accept"}')
        )
        stats = run_replay(
            self._states(),
            output,
            n_samples_per_temp=3,
            concurrency=2,
            max_calls=1000,
            max_cost_usd=1000.0,
            client=client,
        )
        # 2 states * 2 temps * 3 samples = 12
        assert stats["n_calls"] == 12
        assert stats["n_attempted"] == 12
        assert stats["stopped_reason"] is None
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 12
        recorded_keys = {
            (
                json.loads(line)["state_id"],
                json.loads(line)["temp_label"],
                json.loads(line)["sample_idx"],
            )
            for line in lines
        }
        assert len(recorded_keys) == 12

    def test_resume_makes_zero_additional_calls(self, tmp_path: Path) -> None:
        output = tmp_path / "scored.jsonl"
        call_count = {"n": 0}

        def responder(kwargs: Any) -> _FakeResponse:  # noqa: ANN401
            call_count["n"] += 1
            return _FakeResponse('REASONING: x\nACTION: {"decision": "accept"}')

        client = _FakeClient(responder)
        run_replay(
            self._states(),
            output,
            n_samples_per_temp=2,
            concurrency=2,
            max_calls=1000,
            max_cost_usd=1000.0,
            client=client,
        )
        first_run_calls = call_count["n"]
        assert first_run_calls == 8  # 2 states * 2 temps * 2 samples

        stats_resumed = run_replay(
            self._states(),
            output,
            n_samples_per_temp=2,
            concurrency=2,
            max_calls=1000,
            max_cost_usd=1000.0,
            client=client,
        )
        assert call_count["n"] == first_run_calls  # no new calls made
        assert stats_resumed["n_attempted"] == 0
        assert stats_resumed["n_resumed_skipped"] == 8

    def test_max_calls_cap_stops_new_dispatch(self, tmp_path: Path) -> None:
        output = tmp_path / "scored.jsonl"
        client = _FakeClient(
            lambda kwargs: _FakeResponse('REASONING: x\nACTION: {"decision": "accept"}')
        )
        stats = run_replay(
            self._states(),
            output,
            n_samples_per_temp=3,
            concurrency=1,
            max_calls=2,
            max_cost_usd=1000.0,
            client=client,
        )
        assert (
            stats["n_calls"] <= 3
        )  # capped near 2, concurrency=1 so no overshoot race
        assert stats["stopped_reason"] == "max_calls"

    def test_circuit_breaker_stops_on_consecutive_api_failures(
        self, tmp_path: Path
    ) -> None:
        output = tmp_path / "scored.jsonl"

        def always_fails(kwargs: Any) -> _FakeResponse:  # noqa: ANN401
            raise RuntimeError("simulated network failure")

        client = _FakeClient(always_fails)
        stats = run_replay(
            self._states(),
            output,
            n_samples_per_temp=10,
            concurrency=1,
            max_calls=1000,
            max_cost_usd=1000.0,
            circuit_breaker_threshold=3,
            client=client,
        )
        assert stats["circuit_breaker_tripped"] is True
        assert stats["n_calls"] == 0  # every attempt was an API-level failure
        lines = output.read_text().strip().splitlines()
        assert all(json.loads(line)["api_error"] is not None for line in lines)
        # stopped well short of the full 2*2*10=40 task grid
        assert len(lines) < 40

    def test_parse_failures_never_dropped(self, tmp_path: Path) -> None:
        output = tmp_path / "scored.jsonl"
        client = _FakeClient(
            lambda kwargs: _FakeResponse("no REASONING or ACTION heading")
        )
        stats = run_replay(
            self._states(),
            output,
            n_samples_per_temp=2,
            concurrency=2,
            max_calls=1000,
            max_cost_usd=1000.0,
            client=client,
        )
        assert stats["outcome_counts"].get("parse_failure", 0) == 8
        lines = output.read_text().strip().splitlines()
        assert len(lines) == 8
        assert all(json.loads(line)["ok"] is False for line in lines)
        assert all(json.loads(line)["parse_error"] is not None for line in lines)


class TestExtractSampleFeature:
    def test_bargaining_offer_continuous(self) -> None:
        state = _make_state(
            game_family="bargaining",
            phase="offer",
            game_state={"phase": "offer", "money_to_divide": 1000},
        )
        feature = extract_sample_feature({"alice_gain": 300, "bob_gain": 700}, state)
        assert feature["feature_type"] == "continuous"
        assert feature["value"] == pytest.approx(0.3)

    def test_bargaining_decision_categorical(self) -> None:
        state = _make_state(game_family="bargaining", phase="decision")
        feature = extract_sample_feature({"decision": "Reject"}, state)
        assert feature["feature_type"] == "categorical"
        assert feature["label"] == "decision:reject"


class TestPerStateTempReport:
    def _samples(self, actions: list[Any], ok: bool = True) -> list[dict[str, Any]]:
        return [
            {
                "state_id": "s",
                "temp_label": "t0",
                "sample_idx": i,
                "ok": ok,
                "api_error": None,
                "parse_error": None if ok else "bad",
                "action": a if ok else None,
            }
            for i, a in enumerate(actions)
        ]

    def test_categorical_agreement_with_original(self) -> None:
        state = _make_state(
            game_family="bargaining",
            phase="decision",
            original_action={"decision": "accept"},
        )
        samples = self._samples(
            [{"decision": "accept"}, {"decision": "accept"}, {"decision": "reject"}]
        )
        report = per_state_temp_report(state, samples)
        assert report["original_action_feature_type"] == "categorical"
        assert report["agreement_with_original"] == pytest.approx(2 / 3)
        assert report["n_ok"] == 3
        assert report["n_parse_failures"] == 0

    def test_continuous_mean_abs_diff_from_original(self) -> None:
        state = _make_state(
            game_family="bargaining",
            phase="offer",
            game_state={"phase": "offer", "money_to_divide": 1000},
            original_action={"alice_gain": 500, "bob_gain": 500},
        )
        samples = self._samples(
            [
                {"alice_gain": 500, "bob_gain": 500},
                {"alice_gain": 700, "bob_gain": 300},
            ]
        )
        report = per_state_temp_report(state, samples)
        assert report["original_action_feature_type"] == "continuous"
        assert report["agreement_with_original"] is None
        assert report["mean_abs_diff_from_original"] == pytest.approx((0.0 + 0.2) / 2)

    def test_parse_failures_counted_and_excluded_from_features(self) -> None:
        state = _make_state(game_family="bargaining", phase="decision")
        ok_samples = self._samples([{"decision": "accept"}])
        bad_samples = self._samples([None, None], ok=False)
        report = per_state_temp_report(state, ok_samples + bad_samples)
        assert report["n_samples_total"] == 3
        assert report["n_ok"] == 1
        assert report["n_parse_failures"] == 2
        assert report["n_categorical"] == 1

    def test_build_per_state_temp_reports_covers_every_temp(self) -> None:
        state = _make_state(state_id="s1", game_family="bargaining", phase="decision")
        records = [
            {
                "state_id": "s1",
                "temp_label": "api_default",
                "sample_idx": 0,
                "ok": True,
                "api_error": None,
                "parse_error": None,
                "action": {"decision": "accept"},
            }
        ]
        reports = build_per_state_temp_reports([state], records)
        assert ("s1", "api_default") in reports
        assert ("s1", "t0") in reports
        assert reports[("s1", "t0")]["n_samples_total"] == 0


class TestPerStateTempReportWithFailuresAsCategory:
    def _samples(
        self, actions: list[Any | None], ok_flags: list[bool]
    ) -> list[dict[str, Any]]:
        return [
            {
                "state_id": "s",
                "temp_label": "t0",
                "sample_idx": i,
                "ok": ok,
                "api_error": None,
                "parse_error": None if ok else "bad",
                "action": a,
            }
            for i, (a, ok) in enumerate(zip(actions, ok_flags, strict=True))
        ]

    def test_failures_pool_into_one_shared_category(self) -> None:
        state = _make_state(game_family="bargaining", phase="decision")
        samples = self._samples(
            actions=[{"decision": "accept"}, None, None, None],
            ok_flags=[True, False, False, False],
        )
        report = per_state_temp_report_with_failures_as_category(state, samples)
        assert report["n_samples_total"] == 4
        assert report["n_distinct_labels"] == 2  # "decision:accept" + "parse_failure"
        assert report["modal_label"] == "parse_failure"
        assert report["modal_share"] == pytest.approx(3 / 4)
        assert report["action_spread"] == pytest.approx(1 / 4)

    def test_never_excludes_a_sample_unlike_per_state_temp_report(self) -> None:
        state = _make_state(game_family="bargaining", phase="decision")
        samples = self._samples(
            actions=[{"decision": "accept"}, None],
            ok_flags=[True, False],
        )
        with_failures = per_state_temp_report_with_failures_as_category(state, samples)
        without_failures = per_state_temp_report(state, samples)
        assert with_failures["n_samples_total"] == 2
        # the exclusion-based report's categorical pool has only 1 member
        # (the failure is dropped, not counted in n_categorical)
        assert without_failures["n_categorical"] == 1
        assert without_failures["n_parse_failures"] == 1

    def test_continuous_feature_values_collapse_only_when_identical(self) -> None:
        state = _make_state(
            game_family="bargaining",
            phase="offer",
            game_state={"phase": "offer", "money_to_divide": 1000},
        )
        samples = self._samples(
            actions=[
                {"alice_gain": 500, "bob_gain": 500},
                {"alice_gain": 500, "bob_gain": 500},
                {"alice_gain": 700, "bob_gain": 300},
            ],
            ok_flags=[True, True, True],
        )
        report = per_state_temp_report_with_failures_as_category(state, samples)
        assert report["n_distinct_labels"] == 2  # "value:0.5" twice, "value:0.7" once
        assert report["modal_share"] == pytest.approx(2 / 3)
        assert report["action_spread"] == pytest.approx(1 / 3)

    def test_empty_samples_gives_none_not_fabricated(self) -> None:
        state = _make_state(game_family="bargaining", phase="decision")
        report = per_state_temp_report_with_failures_as_category(state, [])
        assert report["modal_share"] is None
        assert report["action_spread"] is None
        assert report["n_distinct_labels"] == 0

    def test_build_per_state_temp_reports_with_failures_matches_single_call(
        self,
    ) -> None:
        state = _make_state(state_id="s1", game_family="bargaining", phase="decision")
        records = [
            {
                "state_id": "s1",
                "temp_label": "api_default",
                "sample_idx": 0,
                "ok": True,
                "api_error": None,
                "parse_error": None,
                "action": {"decision": "accept"},
            },
            {
                "state_id": "s1",
                "temp_label": "api_default",
                "sample_idx": 1,
                "ok": False,
                "api_error": None,
                "parse_error": "bad",
                "action": None,
            },
        ]
        reports = build_per_state_temp_reports_with_failures_as_category(
            [state], records
        )
        expected = per_state_temp_report_with_failures_as_category(state, records)
        assert reports[("s1", "api_default")] == expected
        assert reports[("s1", "t0")]["n_samples_total"] == 0


class TestAggregateReplay:
    def _per_state_temp(self) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            ("s1", "api_default"): {"action_spread": 0.2},
            ("s2", "api_default"): {"action_spread": 0.4},
            ("s1", "t0"): {"action_spread": 0.0},
            ("s2", "t0"): {"action_spread": 0.0},
        }

    def _states(self) -> list[dict[str, Any]]:
        return [
            _make_state(state_id="s1", game_family="bargaining"),
            _make_state(state_id="s2", game_family="bargaining"),
        ]

    def test_deterministic_same_seed(self) -> None:
        states = self._states()
        per_state_temp = self._per_state_temp()
        result1 = aggregate_replay(states, per_state_temp, seed=42, n_resamples=100)
        result2 = aggregate_replay(states, per_state_temp, seed=42, n_resamples=100)
        assert result1 == result2

    def test_mean_matches_manual_computation(self) -> None:
        states = self._states()
        per_state_temp = self._per_state_temp()
        result = aggregate_replay(states, per_state_temp, seed=1, n_resamples=50)
        assert result["bargaining"]["api_default"][
            "mean_action_spread"
        ] == pytest.approx(0.3)
        assert result["bargaining"]["t0"]["mean_action_spread"] == pytest.approx(0.0)

    def test_empty_family_gives_none_not_fabricated(self) -> None:
        states = self._states()
        per_state_temp = self._per_state_temp()
        result = aggregate_replay(states, per_state_temp, seed=1, n_resamples=50)
        assert result["negotiation"]["api_default"]["mean_action_spread"] is None
        assert result["negotiation"]["api_default"]["ci_low"] is None


class TestCompareWithObservationalBaseline:
    def test_reads_family_summary_from_baseline_json(self, tmp_path: Path) -> None:
        baseline_path = tmp_path / "glee_pia_analysis.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "action_only_baseline": {
                        "family_summary": {
                            "bargaining": {
                                "mean_action_spread": 0.1030,
                                "n_clusters_with_action_spread": 26,
                            },
                            "negotiation": {
                                "mean_action_spread": 0.0599,
                                "n_clusters_with_action_spread": 35,
                            },
                            "persuasion": {
                                "mean_action_spread": 0.3486,
                                "n_clusters_with_action_spread": 35,
                            },
                        }
                    }
                }
            )
        )
        replay_aggregate = {
            "bargaining": {
                "api_default": {
                    "mean_action_spread": 0.15,
                    "ci_low": 0.1,
                    "ci_high": 0.2,
                    "n_states_with_spread": 5,
                }
            },
            "negotiation": {
                "api_default": {
                    "mean_action_spread": 0.05,
                    "ci_low": 0.0,
                    "ci_high": 0.1,
                    "n_states_with_spread": 6,
                }
            },
            "persuasion": {
                "api_default": {
                    "mean_action_spread": 0.3,
                    "ci_low": 0.2,
                    "ci_high": 0.4,
                    "n_states_with_spread": 5,
                }
            },
        }
        comparison = compare_with_observational_baseline(
            replay_aggregate, baseline_path
        )
        assert comparison["bargaining"]["replay_mean_action_spread_api_default"] == 0.15
        assert comparison["bargaining"]["observational_mean_action_spread"] == 0.1030
        assert (
            comparison["persuasion"]["observational_n_clusters_with_action_spread"]
            == 35
        )
