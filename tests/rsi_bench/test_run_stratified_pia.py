"""Tests for src/rsi_bench/run_stratified_pia.py.

No test in this file touches the Anthropic API or the real (large)
tests/experiments/glee/trajectories.jsonl.gz fixture: selection and
scoring are exercised against small, synthetic GLEEStep sets, and judging
is always a stub implementing the same protocol as
tests/rsi_bench/test_glee_pia_baseline.py's _StubJudgeClient.

Test inventory:
  TestInputFreezing        — sha256 + gzip streaming load, no /tmp copy
  TestSelectionDeterminism — manifest identical across two runs, same seed
  TestSelectionMechanics   — eligibility, binning, member cap, fallback_rate
  TestRawScoreRoundTrip    — persisted raw scores recompute the same kappa
  TestResume               — resume skips completed clusters, no re-judging
  TestManifestOnlySafety   — --manifest-only never constructs an API client
  TestUsageCapture         — real token usage wrapped at the driver level
  TestJudgeTemperatureRejected — --judge-temperature raises, doesn't no-op
  TestBudgetCaps           — --max-calls / --max-cost-usd stop cleanly
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest
import typer

from src.annotation.irr_calculator import IRRCalculator
from src.rsi_bench.glee_pia_baseline import (
    _GLEE_DIMENSIONS,
    DecisionCluster,
    GLEEReasoningAnnotation,
    GLEEStep,
    _dimension_kappa,
)
from src.rsi_bench.run_stratified_pia import (
    InputMeta,
    SelectionConfig,
    _build_scoring_cluster,
    _member_id,
    _read_completed_cluster_ids,
    _run_summary_path,
    _wrap_for_usage,
    build_manifest,
    fallback_rate_summary,
    load_glee_log_from_gz,
    main,
    score_selected_clusters,
    select_clusters,
    sha256_of_file,
)

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------


def _step(
    family: str,
    game_id: str,
    cluster_key: int,
    *,
    fallback: bool = False,
) -> GLEEStep:
    """One synthetic GLEEStep. ``cluster_key`` controls which coarse
    signature (and therefore which cluster) the step lands in; steps
    sharing a ``(family, cluster_key)`` pair always cluster together."""
    common: dict[str, Any] = {
        "your_player": "player_1",
        "phase": "decide",
        "round": 1,
        "reasoning": f"reasoning for {game_id}",
        "fallback_used": fallback,
        "model": "stub-model",
    }
    if family == "bargaining":
        return GLEEStep(
            game_id=game_id,
            game_family=family,
            game_state={
                "money_to_divide": 100.0 + cluster_key,
                "offer_on_table": None,
                "max_rounds": 10,
            },
            valid_actions={"fields": ["offer_amount"]},
            action={"offer_amount": 50.0},
            **common,
        )
    if family == "negotiation":
        return GLEEStep(
            game_id=game_id,
            game_family=family,
            game_state={
                "current_player": "player_1",
                "player_1_role": "buyer",
                "player_1_value": 100.0 + cluster_key,
                "price": 50.0,
            },
            valid_actions={"fields": ["action_type"]},
            action={"action_type": "accept"},
            **common,
        )
    if family == "persuasion":
        return GLEEStep(
            game_id=game_id,
            game_family=family,
            game_state={"p": 0.5, "v": 1.0 + cluster_key, "u": 0.3},
            valid_actions={"fields": ["message"]},
            action={"message": "buy it"},
            **common,
        )
    raise ValueError(f"unknown family: {family}")


def _make_family_clusters(
    family: str, sizes: list[int], *, fallback_fraction: float = 0.0
) -> list[GLEEStep]:
    """``len(sizes)`` distinct clusters for ``family``, sized per ``sizes``."""
    steps: list[GLEEStep] = []
    for key, size in enumerate(sizes):
        n_fallback = round(size * fallback_fraction)
        for i in range(size):
            steps.append(
                _step(family, f"{family}-{key}-{i}", key, fallback=(i < n_fallback))
            )
    return steps


_SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


def _all_families_steps(**kwargs: Any) -> list[GLEEStep]:  # noqa: ANN401
    return (
        _make_family_clusters("bargaining", _SIZES, **kwargs)
        + _make_family_clusters("negotiation", _SIZES, **kwargs)
        + _make_family_clusters("persuasion", _SIZES, **kwargs)
    )


class _StubJudgeClient:
    """Deterministic judge, no network calls — mirrors
    test_glee_pia_baseline._StubJudgeClient."""

    def __init__(self, agree: bool = True) -> None:
        self._agree = agree
        self.n_calls = 0

    def score_member(
        self, cluster: DecisionCluster, member: GLEEStep, judge_name: str
    ) -> dict[str, Any]:
        self.n_calls += 1
        if self._agree:
            base = 4
        else:
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


def _small_config(**overrides: Any) -> SelectionConfig:  # noqa: ANN401
    defaults: dict[str, Any] = {
        "seed": 42,
        "n_bins": 3,
        "clusters_per_bin": 2,
        "member_cap": 5,
        "model": "stub-model",
    }
    defaults.update(overrides)
    return SelectionConfig(**defaults)


def _fake_input_meta(n_steps: int) -> InputMeta:
    return InputMeta(
        path="fake.jsonl.gz",
        sha256="0" * 64,
        raw_line_count=n_steps,
        parsed_step_count=n_steps,
    )


# ---------------------------------------------------------------------------
# TestInputFreezing
# ---------------------------------------------------------------------------


class TestInputFreezing:
    def test_sha256_is_deterministic_and_matches_hashlib(self, tmp_path: Path) -> None:
        path = tmp_path / "data.bin"
        path.write_bytes(b"some bytes to hash" * 1000)
        import hashlib

        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha256_of_file(path) == expected
        assert sha256_of_file(path) == sha256_of_file(path)

    def test_load_from_gz_parses_and_skips_malformed(self, tmp_path: Path) -> None:
        gz_path = tmp_path / "log.jsonl.gz"
        good = _step("bargaining", "g1", 0)
        good_raw = {
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
        lines = [
            json.dumps(good_raw),
            "not even json",
            json.dumps({"game_id": "missing_fields_only"}),
            "",
        ]
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        steps, raw_line_count = load_glee_log_from_gz(gz_path)

        assert len(steps) == 1
        assert steps[0].game_id == "g1"
        assert raw_line_count == 4

    def test_load_from_gz_never_writes_to_tmp_other_than_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard against regressions that shell out to gunzip into /tmp."""
        gz_path = tmp_path / "log.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write("")

        import pathlib

        original_open = pathlib.Path.open

        def _guard(self: pathlib.Path, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if "w" in (args[0] if args else kwargs.get("mode", "r")):
                raise AssertionError(
                    f"load_glee_log_from_gz must never open a file for "
                    f"writing, but tried: {self}"
                )
            return original_open(self, *args, **kwargs)

        monkeypatch.setattr(pathlib.Path, "open", _guard)
        load_glee_log_from_gz(gz_path)  # must not raise


# ---------------------------------------------------------------------------
# TestSelectionDeterminism
# ---------------------------------------------------------------------------


class TestSelectionDeterminism:
    def test_manifest_identical_across_runs_same_seed(self) -> None:
        steps = _all_families_steps()
        config = _small_config(seed=123)
        input_meta = _fake_input_meta(len(steps))

        manifest1 = build_manifest(input_meta, config, select_clusters(steps, config))
        manifest2 = build_manifest(input_meta, config, select_clusters(steps, config))

        assert manifest1 == manifest2

    def test_different_seeds_can_select_differently(self) -> None:
        steps = _all_families_steps()
        result_a = select_clusters(steps, _small_config(seed=1))
        result_b = select_clusters(steps, _small_config(seed=2))

        ids_a = {c.cluster_id for c in result_a.clusters}
        ids_b = {c.cluster_id for c in result_b.clusters}
        # Not a hard requirement that they differ, but with this many
        # eligible clusters and a tight per-bin cap it would be
        # suspicious if two different seeds always picked identically.
        assert ids_a != ids_b or result_a.clusters != result_b.clusters

    def test_bin_boundaries_independent_of_seed(self) -> None:
        steps = _all_families_steps()
        result_a = select_clusters(steps, _small_config(seed=1))
        result_b = select_clusters(steps, _small_config(seed=999))
        assert result_a.by_family_bin_sizes == result_b.by_family_bin_sizes
        assert result_a.by_family_eligible_count == result_b.by_family_eligible_count


# ---------------------------------------------------------------------------
# TestSelectionMechanics
# ---------------------------------------------------------------------------


class TestSelectionMechanics:
    def test_eligibility_excludes_small_clusters(self) -> None:
        # size=1 cluster has only 1 non-fallback member -> ineligible.
        steps = _make_family_clusters("bargaining", [1, 2, 3])
        config = _small_config(n_bins=1, clusters_per_bin=10)
        result = select_clusters(steps, config)
        assert result.by_family_eligible_count["bargaining"] == 2
        assert result.by_family_eligible_count["negotiation"] == 0
        assert result.by_family_eligible_count["persuasion"] == 0

    def test_member_cap_applied(self) -> None:
        steps = _make_family_clusters("bargaining", [20])
        config = _small_config(n_bins=1, clusters_per_bin=1, member_cap=5)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 1
        sel = result.clusters[0]
        assert len(sel.non_fallback_member_ids) == 20
        assert len(sel.sampled_member_ids) == 5
        assert set(sel.sampled_member_ids).issubset(set(sel.non_fallback_member_ids))

    def test_member_cap_not_applied_when_under_cap(self) -> None:
        steps = _make_family_clusters("bargaining", [3])
        config = _small_config(n_bins=1, clusters_per_bin=1, member_cap=10)
        result = select_clusters(steps, config)
        sel = result.clusters[0]
        assert sel.sampled_member_ids == sel.non_fallback_member_ids

    def test_fallback_rate_both_ways(self) -> None:
        # Cluster A: 10 members, 5 fallback -> rate 0.5.
        # Cluster B: 4 members, 1 fallback -> rate 0.25.
        cluster_a = [
            _step("bargaining", f"a-{i}", 0, fallback=(i < 5)) for i in range(10)
        ]
        cluster_b = [
            _step("bargaining", f"b-{i}", 1, fallback=(i < 1)) for i in range(4)
        ]
        config = _small_config(n_bins=1, clusters_per_bin=2, member_cap=10)
        result = select_clusters(cluster_a + cluster_b, config)
        assert len(result.clusters) == 2

        by_id = {c.cluster_id: c for c in result.clusters}
        rates = sorted(c.fallback_rate for c in by_id.values())
        assert rates == pytest.approx([0.25, 0.5])

        summary = fallback_rate_summary(result.clusters)
        assert summary["unweighted_mean"] == pytest.approx((0.5 + 0.25) / 2)
        assert summary["member_weighted"] == pytest.approx((5 + 1) / (10 + 4))
        assert summary["unweighted_mean"] != summary["member_weighted"]

    def test_short_bin_samples_all_available_without_raising(self) -> None:
        # Only 3 eligible clusters total, but clusters_per_bin=7 and
        # n_bins=5 would normally want up to 35.
        steps = _make_family_clusters("bargaining", [2, 3, 4])
        config = _small_config(n_bins=5, clusters_per_bin=7, member_cap=10)
        result = select_clusters(steps, config)
        fam_selected = [c for c in result.clusters if c.game_family == "bargaining"]
        assert len(fam_selected) == 3  # all eligible, none dropped, no raise

    def test_selected_count_never_exceeds_eligible_count(self) -> None:
        steps = _all_families_steps()
        config = _small_config()
        result = select_clusters(steps, config)
        for family in ("bargaining", "negotiation", "persuasion"):
            selected_in_family = sum(
                1 for c in result.clusters if c.game_family == family
            )
            assert selected_in_family <= result.by_family_eligible_count[family]


# ---------------------------------------------------------------------------
# TestRawScoreRoundTrip
# ---------------------------------------------------------------------------


class TestRawScoreRoundTrip:
    def test_persisted_raw_scores_recompute_same_kappa(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [12])
        config = _small_config(n_bins=1, clusters_per_bin=1, member_cap=10, seed=7)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 1

        judge = _StubJudgeClient(agree=False)  # disagreement -> non-trivial kappa
        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(steps, result, config, judge, output_path)

        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])

        sampled_ids = record["sampled_member_ids"]
        annotations = [
            GLEEReasoningAnnotation(
                annotation_id=f"{record['cluster_id']}/{sampled_ids.index(rs['member_id'])}/{rs['judge_name']}",
                cluster_id=record["cluster_id"],
                member_index=sampled_ids.index(rs["member_id"]),
                judge_name=rs["judge_name"],
                valuation_reasoning=rs["valuation_reasoning"],
                horizon_strategy_planning=rs["horizon_strategy_planning"],
                concession_handling=rs["concession_handling"],
                outcome_consistency=rs["outcome_consistency"],
                rationale=rs["rationale"],
                judge_failed_dimensions=tuple(rs["judge_failed_dimensions"]),
            )
            for rs in record["raw_scores"]
        ]

        irr = IRRCalculator()
        recomputed: dict[str, float] = {}
        for dim in _GLEE_DIMENSIONS:
            kappa, _ = _dimension_kappa(irr, dim, annotations)
            if kappa is not None:
                recomputed[dim] = kappa

        assert recomputed.keys() == record["kappa_per_dimension"].keys()
        for dim, value in recomputed.items():
            assert value == pytest.approx(record["kappa_per_dimension"][dim])
        assert sum(recomputed.values()) / len(recomputed) == pytest.approx(
            record["kappa_overall"]
        )

    def test_raw_scores_cover_every_sampled_member_and_judge(
        self, tmp_path: Path
    ) -> None:
        steps = _make_family_clusters("bargaining", [6])
        config = _small_config(n_bins=1, clusters_per_bin=1, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        judge = _StubJudgeClient(agree=True)
        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(steps, result, config, judge, output_path)

        record = json.loads(output_path.read_text().strip())
        member_judge_pairs = {
            (rs["member_id"], rs["judge_name"]) for rs in record["raw_scores"]
        }
        expected = {
            (mid, judge_name)
            for mid in record["sampled_member_ids"]
            for judge_name in (
                "GameTheoreticRigor",
                "LiteralGroundedness",
                "OpponentResponsiveness",
            )
        }
        assert member_judge_pairs == expected
        assert record["usage"]["n_score_member_calls"] == len(record["raw_scores"])
        assert record["usage"]["input_tokens"] is None


# ---------------------------------------------------------------------------
# TestResume
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_skips_completed_clusters_no_rejudging(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [5, 6])
        config = _small_config(n_bins=1, clusters_per_bin=2, member_cap=10, seed=3)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 2

        output_path = tmp_path / "scored.jsonl"
        judge = _StubJudgeClient(agree=True)

        score_selected_clusters(steps, result, config, judge, output_path)
        calls_after_first = judge.n_calls
        lines_after_first = output_path.read_text().strip().splitlines()
        assert len(lines_after_first) == 2

        score_selected_clusters(steps, result, config, judge, output_path)
        assert judge.n_calls == calls_after_first  # no re-judging
        lines_after_second = output_path.read_text().strip().splitlines()
        assert len(lines_after_second) == 2  # no duplicate records
        cluster_ids = {json.loads(line)["cluster_id"] for line in lines_after_second}
        assert cluster_ids == {c.cluster_id for c in result.clusters}

    def test_read_completed_cluster_ids_empty_when_absent(self, tmp_path: Path) -> None:
        assert _read_completed_cluster_ids(tmp_path / "nope.jsonl") == set()

    def test_build_scoring_cluster_preserves_sampled_order(self) -> None:
        steps = [_step("bargaining", f"g{i}", 0) for i in range(5)]
        original = DecisionCluster(
            cluster_id="bargaining_001",
            game_family="bargaining",
            state_signature=(),
            members=steps,
        )
        ids = [_member_id(s) for s in steps]
        reversed_ids = tuple(reversed(ids[:3]))
        scoring_cluster = _build_scoring_cluster(original, reversed_ids)
        assert [_member_id(m) for m in scoring_cluster.members] == list(reversed_ids)


# ---------------------------------------------------------------------------
# TestManifestOnlySafety
# ---------------------------------------------------------------------------


class TestManifestOnlySafety:
    def test_manifest_only_constructs_no_api_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise AssertionError(
                "manifest-only mode must never construct _AnthropicJudgeClient"
            )

        monkeypatch.setattr(
            "src.rsi_bench.run_stratified_pia._AnthropicJudgeClient", _boom
        )

        gz_path = tmp_path / "log.jsonl.gz"
        steps = _make_family_clusters("bargaining", [3, 4])
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
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

        manifest_output = tmp_path / "manifest.json"
        scored_output = tmp_path / "scored.jsonl"

        # Calling the typer-decorated command function directly, with
        # every parameter given explicitly (bypasses Click's CLI parsing
        # entirely, so typer.Option default sentinels are never touched).
        main(
            input_path=gz_path,
            seed=1,
            manifest_only=True,
            manifest_output=manifest_output,
            scored_output=scored_output,
            model="stub-model",
            judge_temperature=None,
            cost_per_call=0.0056,
            n_bins=1,
            clusters_per_bin=2,
            member_cap=5,
            verbose=False,
        )

        assert manifest_output.exists()
        assert not scored_output.exists()
        manifest = json.loads(manifest_output.read_text())
        assert manifest["config"]["seed"] == 1
        assert "generated_at" in manifest

    def test_real_anthropic_client_is_never_imported_as_side_effect(self) -> None:
        """Sanity check: importing the module must not instantiate a
        client or require ANTHROPIC_API_KEY to be set."""
        import importlib

        import src.rsi_bench.run_stratified_pia as module

        importlib.reload(module)  # must not raise even with no API key


# ---------------------------------------------------------------------------
# Fakes for usage-capture tests (structural stand-ins for the Anthropic SDK
# shapes _wrap_for_usage expects -- no real anthropic objects, no network)
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeMessage:
    def __init__(self, text: str, usage: _FakeUsage) -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = usage


class _FakeMessagesResource:
    """Stands in for anthropic.resources.Messages -- just .create()."""

    def __init__(self, make_response: Any) -> None:  # noqa: ANN401
        self._make_response = make_response
        self.n_create_calls = 0

    def create(self, **kwargs: Any) -> Any:  # noqa: ANN401
        self.n_create_calls += 1
        return self._make_response()


class _FakeAnthropicClient:
    def __init__(self, make_response: Any) -> None:  # noqa: ANN401
        self.messages = _FakeMessagesResource(make_response)


class _FakeAnthropicJudgeClient:
    """Structural stand-in for _AnthropicJudgeClient: same
    ``_client.messages.create`` call shape, so _wrap_for_usage and the
    usage-capture path can be exercised without the real anthropic SDK
    or any network call. Deliberately NOT a subclass of
    _AnthropicJudgeClient -- that class always constructs a real
    anthropic.Anthropic() in __init__, which this has no need for."""

    def __init__(
        self, *, input_tokens: int = 10, output_tokens: int = 5, base_score: int = 4
    ) -> None:
        self._usage = _FakeUsage(input_tokens, output_tokens)
        self._base_score = base_score
        self._client = _FakeAnthropicClient(self._make_response)

    def _make_response(self) -> _FakeMessage:
        payload = json.dumps(
            {
                "valuation_reasoning": self._base_score,
                "horizon_strategy_planning": self._base_score,
                "concession_handling": self._base_score,
                "outcome_consistency": self._base_score,
                "rationale": "[fake] scored",
            }
        )
        return _FakeMessage(payload, self._usage)

    def score_member(
        self, cluster: DecisionCluster, member: GLEEStep, judge_name: str
    ) -> dict[str, Any]:
        message = self._client.messages.create(
            model="x", max_tokens=10, system=[], messages=[]
        )
        return json.loads(message.content[0].text)


# ---------------------------------------------------------------------------
# TestUsageCapture
# ---------------------------------------------------------------------------


class TestUsageCapture:
    def test_wrap_for_usage_returns_none_for_plain_stub(self) -> None:
        assert _wrap_for_usage(_StubJudgeClient()) is None

    def test_wrap_for_usage_captures_calls(self) -> None:
        fake = _FakeAnthropicJudgeClient(input_tokens=100, output_tokens=50)
        accumulator = _wrap_for_usage(fake)
        assert accumulator is not None

        for _ in range(3):
            fake.score_member(cluster=None, member=None, judge_name="x")  # type: ignore[arg-type]

        totals = accumulator.totals()
        assert totals["n_calls"] == 3
        assert totals["input_tokens"] == 300
        assert totals["output_tokens"] == 150

    def test_scored_record_has_real_token_usage(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [3])
        config = _small_config(n_bins=1, clusters_per_bin=1, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 1

        fake = _FakeAnthropicJudgeClient(input_tokens=100, output_tokens=50)
        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(steps, result, config, fake, output_path)

        record = json.loads(output_path.read_text().strip())
        n_calls = record["usage"]["n_score_member_calls"]
        assert n_calls == 3 * 3  # 3 members x 3 judges
        assert record["usage"]["input_tokens"] == n_calls * 100
        assert record["usage"]["output_tokens"] == n_calls * 50
        assert record["usage"]["note"] is None

    def test_plain_stub_usage_stays_null_with_note(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [3])
        config = _small_config(n_bins=1, clusters_per_bin=1, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(steps, result, config, _StubJudgeClient(), output_path)

        record = json.loads(output_path.read_text().strip())
        assert record["usage"]["input_tokens"] is None
        assert record["usage"]["output_tokens"] is None
        assert record["usage"]["note"] is not None

    def test_run_summary_written_with_token_totals(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [3, 4])
        config = _small_config(n_bins=1, clusters_per_bin=2, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 2

        fake = _FakeAnthropicJudgeClient(input_tokens=1000, output_tokens=200)
        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(steps, result, config, fake, output_path)

        summary = json.loads(_run_summary_path(output_path).read_text())
        assert summary["n_clusters_scored"] == 2
        assert summary["cost_capturable"] is True
        expected_calls = (3 + 4) * 3
        assert summary["total_score_member_calls"] == expected_calls
        assert summary["token_usage"]["input_tokens"] == expected_calls * 1000
        assert summary["token_usage"]["output_tokens"] == expected_calls * 200
        expected_cost = (
            expected_calls * 1000 / 1_000_000 * 3
            + expected_calls * 200 / 1_000_000 * 15
        )
        assert summary["estimated_cost_usd"] == pytest.approx(expected_cost, abs=1e-4)
        assert summary["stopped_reason"] is None


# ---------------------------------------------------------------------------
# TestJudgeTemperatureRejected
# ---------------------------------------------------------------------------


class TestJudgeTemperatureRejected:
    def test_selection_config_rejects_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            SelectionConfig(seed=1, judge_temperature=0.5)

    def test_selection_config_accepts_none(self) -> None:
        SelectionConfig(seed=1, judge_temperature=None)  # must not raise

    def test_cli_rejects_judge_temperature(self, tmp_path: Path) -> None:
        gz_path = tmp_path / "log.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            f.write("")

        with pytest.raises(typer.BadParameter):
            main(
                input_path=gz_path,
                seed=1,
                manifest_only=True,
                manifest_output=tmp_path / "manifest.json",
                scored_output=tmp_path / "scored.jsonl",
                model="stub-model",
                judge_temperature=0.7,
                cost_per_call=0.0056,
                max_calls=2700,
                max_cost_usd=16.0,
                n_bins=1,
                clusters_per_bin=1,
                member_cap=5,
                verbose=False,
            )
        assert not (tmp_path / "manifest.json").exists()


# ---------------------------------------------------------------------------
# TestBudgetCaps
# ---------------------------------------------------------------------------


class TestBudgetCaps:
    def test_max_calls_stops_cleanly(self, tmp_path: Path) -> None:
        # 3 clusters x 6 members x 3 judges = 54 total calls available;
        # cap well below that so at least one cluster is left unscored.
        steps = _make_family_clusters("bargaining", [6, 6, 6])
        config = _small_config(n_bins=1, clusters_per_bin=3, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 3

        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(
            steps, result, config, _StubJudgeClient(), output_path, max_calls=18
        )

        lines = output_path.read_text().strip().splitlines()
        assert 0 < len(lines) < 3  # stopped partway, not all 3 clusters scored

        summary = json.loads(_run_summary_path(output_path).read_text())
        assert summary["stopped_reason"] == "max_calls"
        assert summary["total_score_member_calls"] >= 18

    def test_max_cost_usd_stops_cleanly_with_real_usage(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [2, 2, 2])
        config = _small_config(n_bins=1, clusters_per_bin=3, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 3

        # 1 cluster = 2 members x 3 judges = 6 calls. Each call: 1e6 in +
        # 1e6 out tokens -> $3 + $15 = $18/call -> one cluster alone costs
        # $108, instantly blowing past a $10 cap.
        fake = _FakeAnthropicJudgeClient(
            input_tokens=1_000_000, output_tokens=1_000_000
        )
        output_path = tmp_path / "scored.jsonl"
        score_selected_clusters(
            steps, result, config, fake, output_path, max_calls=None, max_cost_usd=10.0
        )

        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 1  # stopped after the first cluster

        summary = json.loads(_run_summary_path(output_path).read_text())
        assert summary["stopped_reason"] == "max_cost_usd"
        assert summary["estimated_cost_usd"] >= 10.0

    def test_max_cost_usd_skipped_without_usage_capture(self, tmp_path: Path) -> None:
        steps = _make_family_clusters("bargaining", [2, 2])
        config = _small_config(n_bins=1, clusters_per_bin=2, member_cap=10, seed=1)
        result = select_clusters(steps, config)
        assert len(result.clusters) == 2

        output_path = tmp_path / "scored.jsonl"
        # Absurdly low cost cap, but with a non-instrumentable stub this
        # must never trigger -- only max_calls can stop a stub-judged run.
        score_selected_clusters(
            steps,
            result,
            config,
            _StubJudgeClient(),
            output_path,
            max_calls=None,
            max_cost_usd=0.0001,
        )

        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 2  # both clusters scored; cost cap was inert

        summary = json.loads(_run_summary_path(output_path).read_text())
        assert summary["stopped_reason"] is None
        assert summary["cost_capturable"] is False
