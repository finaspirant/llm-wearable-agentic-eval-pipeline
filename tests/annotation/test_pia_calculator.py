"""Tests for src/annotation/pia_calculator.py.

All tests use dry_run=True — no Anthropic API calls.  The dry-run mode uses
deterministic table-driven scoring from _PIA_SCORES and seeded SHA-256 from
_STANDARD_STEP_BIAS, making every test reproducible without network access.

Test inventory:
  TestStepAnnotation       — StepAnnotation dataclass fields and serialisation
  TestPIAAnnotation        — PIAAnnotation dataclass fields and serialisation
  TestStandardStepAnnotator — Mode A annotation shapes and score values
  TestPIADimensionAnnotator — Mode B annotation shapes and score values
  TestBuildLabelMatrix     — _build_label_matrix produces correct 0-indexed matrix
  TestStandardIRRComputer  — per-pair κ is a float, step counts match
  TestPIAIRRComputer       — per-dimension and per-pair κ values and shapes
  TestPIACalculator        — end-to-end run(), save(), delta direction
  TestOutputSchema         — pia_results.json structure and key values

Run:
    pytest tests/annotation/test_pia_calculator.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.annotation.pia_calculator import (
    _DETOUR_SCORES,
    _PERSONAS,
    _PIA_DIMENSIONS,
    _PIA_SCORES,
    _STANDARD_STEP_BIAS,
    PIAAnnotation,
    PIACalculator,
    StepAnnotation,
    _build_label_matrix,
    _dry_run_pia_score,
    _dry_run_step_score,
    _kappa_interpretation,
    _PIADimensionAnnotator,
    _StandardStepAnnotator,
)
from src.annotation.pia_trajectory_generator import PIATrajectoryGenerator

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_PAIRS_DIR = Path("data/trajectories/pia_pairs")


@pytest.fixture(scope="module")
def pairs() -> list:  # type: ignore[type-arg]
    return PIATrajectoryGenerator(seed=42).generate_all_pairs()


@pytest.fixture(scope="module")
def calculator() -> PIACalculator:
    return PIACalculator(pairs_dir=_PAIRS_DIR, dry_run=True)


@pytest.fixture(scope="module")
def loaded_pairs(calculator: PIACalculator) -> list:  # type: ignore[type-arg]
    return calculator.load_pairs()


# ---------------------------------------------------------------------------
# TestStepAnnotation
# ---------------------------------------------------------------------------


class TestStepAnnotation:
    def test_to_dict_keys(self) -> None:
        ann = StepAnnotation(
            step_id="01/agent_a/0",
            pair_id="01",
            agent_id="agent_a",
            step_index=0,
            step_type="standard",
            scenario="health_alert",
            persona_name="PrivacyMaximalist",
            step_quality=3,
            rationale="test",
        )
        keys = set(ann.to_dict())
        expected = {
            "step_id",
            "pair_id",
            "agent_id",
            "step_index",
            "step_type",
            "scenario",
            "persona_name",
            "step_quality",
            "rationale",
        }
        assert keys == expected

    def test_step_quality_in_range(self) -> None:
        ann = StepAnnotation(
            step_id="01/agent_b/2",
            pair_id="01",
            agent_id="agent_b",
            step_index=2,
            step_type="detour",
            scenario="health_alert",
            persona_name="OutcomeOptimist",
            step_quality=4,
            rationale="x",
        )
        assert 1 <= ann.step_quality <= 4

    def test_json_serialisable(self) -> None:
        ann = StepAnnotation(
            step_id="02/agent_a/1",
            pair_id="02",
            agent_id="agent_a",
            step_index=1,
            step_type="standard",
            scenario="privacy_sensitive",
            persona_name="ProcessPurist",
            step_quality=2,
            rationale="ok",
        )
        json.dumps(ann.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# TestPIAAnnotation
# ---------------------------------------------------------------------------


class TestPIAAnnotation:
    def test_to_dict_keys(self) -> None:
        import uuid

        ann = PIAAnnotation(
            annotation_id=str(uuid.uuid4()),
            pair_id="01",
            agent_id="agent_a",
            scenario="health_alert",
            persona_name="PrivacyMaximalist",
            planning_quality=4,
            error_recovery=None,
            goal_alignment=5,
            rationale="test",
        )
        keys = set(ann.to_dict())
        expected = {
            "annotation_id",
            "pair_id",
            "agent_id",
            "scenario",
            "persona_name",
            "planning_quality",
            "error_recovery",
            "goal_alignment",
            "rationale",
        }
        assert keys == expected

    def test_error_recovery_none_for_direct(self) -> None:
        import uuid

        ann = PIAAnnotation(
            annotation_id=str(uuid.uuid4()),
            pair_id="01",
            agent_id="agent_a",
            scenario="health_alert",
            persona_name="RecoverySkeptic",
            planning_quality=3,
            error_recovery=None,
            goal_alignment=4,
            rationale="direct path",
        )
        assert ann.error_recovery is None


# ---------------------------------------------------------------------------
# TestStandardStepAnnotator
# ---------------------------------------------------------------------------


class TestStandardStepAnnotator:
    def test_annotate_pair_count(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _StandardStepAnnotator(dry_run=True)
        pair = pairs[0]  # pair_01
        results = ann.annotate_pair(pair)
        expected = len(_PERSONAS) * (pair.agent_a.n_steps + pair.agent_b.n_steps)
        assert len(results) == expected

    def test_annotate_all_count(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _StandardStepAnnotator(dry_run=True)
        results = ann.annotate_all(pairs)
        total_steps = sum(p.agent_a.n_steps + p.agent_b.n_steps for p in pairs)
        assert len(results) == len(_PERSONAS) * total_steps

    def test_step_quality_in_scale(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _StandardStepAnnotator(dry_run=True)
        for a in ann.annotate_all(pairs):
            assert 1 <= a.step_quality <= 4, (
                f"{a.step_id} {a.persona_name}: step_quality={a.step_quality}"
            )

    def test_detour_scores_match_table(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _StandardStepAnnotator(dry_run=True)
        for pair in pairs:
            for result in ann.annotate_pair(pair):
                if result.step_type == "detour":
                    expected = _DETOUR_SCORES[result.persona_name]
                    assert result.step_quality == expected, (
                        f"{result.step_id} {result.persona_name}: "
                        f"got {result.step_quality}, want {expected}"
                    )

    def test_detour_scores_have_wide_spread(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        """Detour scores span [1, 4] across personas — required for low κ."""
        detour_scores = set(_DETOUR_SCORES.values())
        assert min(detour_scores) == 1
        assert max(detour_scores) == 4

    def test_standard_step_scores_persona_variation(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        """Different personas must produce different standard step scores."""
        ann = _StandardStepAnnotator(dry_run=True)
        pair = pairs[0]
        standard_only = [
            a for a in ann.annotate_pair(pair) if a.step_type == "standard"
        ]
        scores_by_persona = {p: [] for p in _PERSONAS}
        for a in standard_only:
            scores_by_persona[a.persona_name].append(a.step_quality)
        # At least two personas must have different mean scores.
        means = [sum(v) / len(v) for v in scores_by_persona.values() if v]
        assert max(means) - min(means) > 0, (
            "All personas produced identical standard step scores"
        )

    def test_live_mode_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _StandardStepAnnotator(dry_run=False)


# ---------------------------------------------------------------------------
# TestPIADimensionAnnotator
# ---------------------------------------------------------------------------


class TestPIADimensionAnnotator:
    def test_annotate_pair_count(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _PIADimensionAnnotator(dry_run=True)
        results = ann.annotate_pair(pairs[0])
        assert len(results) == len(_PERSONAS) * 2  # 5 × 2 agents

    def test_annotate_all_count(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _PIADimensionAnnotator(dry_run=True)
        results = ann.annotate_all(pairs)
        assert len(results) == len(_PERSONAS) * 2 * len(pairs)  # 100

    def test_error_recovery_none_for_direct_agents(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _PIADimensionAnnotator(dry_run=True)
        for pair in pairs:
            for a in ann.annotate_pair(pair):
                if a.agent_id == "agent_a":
                    assert a.error_recovery is None, (
                        f"{a.pair_id}/agent_a should have error_recovery=None"
                    )

    def test_error_recovery_scored_for_indirect_agents(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _PIADimensionAnnotator(dry_run=True)
        for pair in pairs:
            for a in ann.annotate_pair(pair):
                if a.agent_id == "agent_b":
                    assert a.error_recovery is not None, (
                        f"{a.pair_id}/agent_b should have error_recovery score"
                    )
                    assert 1 <= a.error_recovery <= 5

    def test_pia_scores_in_valid_range(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _PIADimensionAnnotator(dry_run=True)
        for a in ann.annotate_all(pairs):
            assert 1 <= a.planning_quality <= 5
            assert 1 <= a.goal_alignment <= 5
            if a.error_recovery is not None:
                assert 1 <= a.error_recovery <= 5

    def test_scores_match_pia_scores_table(
        self,
        pairs: list,  # type: ignore[type-arg]
    ) -> None:
        ann = _PIADimensionAnnotator(dry_run=True)
        for pair in pairs:
            for a in ann.annotate_pair(pair):
                path = "direct" if a.agent_id == "agent_a" else "indirect"
                expected_pq = _PIA_SCORES[pair.scenario][path]["planning_quality"][
                    a.persona_name
                ]
                assert a.planning_quality == expected_pq, (
                    f"{a.pair_id}/{a.agent_id} {a.persona_name} "
                    f"planning_quality: got {a.planning_quality}, "
                    f"want {expected_pq}"
                )

    def test_live_mode_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _PIADimensionAnnotator(dry_run=False)


# ---------------------------------------------------------------------------
# TestBuildLabelMatrix
# ---------------------------------------------------------------------------


class TestBuildLabelMatrix:
    def test_shape_n_items_x_n_raters(self) -> None:
        items = ["a", "b", "c"]
        raters = ["r1", "r2"]
        scores = {
            ("a", "r1"): 3,
            ("a", "r2"): 4,
            ("b", "r1"): 2,
            ("b", "r2"): 2,
            ("c", "r1"): 4,
            ("c", "r2"): 3,
        }
        matrix = _build_label_matrix(items, raters, scores, scale_offset=1)
        assert len(matrix) == 3
        assert all(len(row) == 2 for row in matrix)

    def test_zero_indexed(self) -> None:
        items = ["x"]
        raters = ["r1", "r2"]
        scores = {("x", "r1"): 1, ("x", "r2"): 3}
        matrix = _build_label_matrix(items, raters, scores, scale_offset=1)
        assert matrix == [[0, 2]]

    def test_score_offset_applied(self) -> None:
        items = ["x"]
        raters = ["r1"]
        scores = {("x", "r1"): 5}
        matrix = _build_label_matrix(items, raters, scores, scale_offset=1)
        assert matrix[0][0] == 4  # 5 - 1

    def test_row_ordering_matches_item_list(self) -> None:
        items = ["b", "a"]
        raters = ["r1"]
        scores = {("a", "r1"): 2, ("b", "r1"): 4}
        matrix = _build_label_matrix(items, raters, scores, scale_offset=1)
        assert matrix[0][0] == 3  # "b" first → 4-1=3
        assert matrix[1][0] == 1  # "a" second → 2-1=1


# ---------------------------------------------------------------------------
# TestStandardIRRComputer
# ---------------------------------------------------------------------------


class TestStandardIRRComputer:
    def test_run_standard_irr_returns_all_pairs(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        results = calculator.run_standard_irr(loaded_pairs)
        assert set(results.keys()) == {p.pair_id for p in loaded_pairs}

    def test_kappa_is_float_in_range(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        results = calculator.run_standard_irr(loaded_pairs)
        for pid, r in results.items():
            assert isinstance(r.kappa, float), f"{pid}: kappa not float"
            assert -1.0 <= r.kappa <= 1.0, f"{pid}: kappa={r.kappa} out of range"

    def test_step_counts_match_trajectories(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        results = calculator.run_standard_irr(loaded_pairs)
        for pair in loaded_pairs:
            r = results[pair.pair_id]
            assert r.n_steps_a == pair.agent_a.n_steps
            assert r.n_steps_b == pair.agent_b.n_steps
            assert r.total_steps == r.n_steps_a + r.n_steps_b

    def test_overall_kappa_is_poor(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        """Standard IRR must demonstrate the IRR-breaks-for-agents problem."""
        results = calculator.run_standard_irr(loaded_pairs)
        from src.annotation.irr_calculator import IRRCalculator
        from src.annotation.pia_calculator import _StandardIRRComputer

        computer = _StandardIRRComputer(IRRCalculator())
        overall = computer.compute_overall(results)
        assert overall < 0.40, (
            f"Standard IRR κ={overall:.4f} should be < 0.40 (fair) "
            "to demonstrate the path-comparison failure"
        )


# ---------------------------------------------------------------------------
# TestPIAIRRComputer
# ---------------------------------------------------------------------------


class TestPIAIRRComputer:
    def test_run_pia_irr_returns_all_pairs(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        per_pair, _ = calculator.run_pia_irr(loaded_pairs)
        assert set(per_pair.keys()) == {p.pair_id for p in loaded_pairs}

    def test_per_dimension_kappa_keys(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        _, per_dim = calculator.run_pia_irr(loaded_pairs)
        assert set(per_dim.keys()) == set(_PIA_DIMENSIONS)

    def test_per_dimension_kappa_range(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        _, per_dim = calculator.run_pia_irr(loaded_pairs)
        for dim, kappa in per_dim.items():
            assert -1.0 <= kappa <= 1.0, f"{dim}: κ={kappa} out of range"

    def test_pia_overall_kappa_is_substantial(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        """PIA must demonstrate substantial agreement to validate the method."""
        _, per_dim = calculator.run_pia_irr(loaded_pairs)
        overall = sum(per_dim.values()) / len(per_dim)
        assert overall >= 0.60, (
            f"PIA overall κ={overall:.4f} should be ≥ 0.60 (moderate+) "
            "to demonstrate rubric-dimension agreement"
        )

    def test_per_pair_error_recovery_is_none(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        """Per-pair error_recovery κ must be None (only 1 agent has data)."""
        per_pair, _ = calculator.run_pia_irr(loaded_pairs)
        for pid, r in per_pair.items():
            assert r.kappa_error_recovery is None, (
                f"{pid}: expected kappa_error_recovery=None, "
                f"got {r.kappa_error_recovery}"
            )

    def test_per_pair_kappa_overall_is_float(
        self,
        calculator: PIACalculator,
        loaded_pairs: list,  # type: ignore[type-arg]
    ) -> None:
        per_pair, _ = calculator.run_pia_irr(loaded_pairs)
        for pid, r in per_pair.items():
            assert isinstance(r.kappa_overall, float), pid
            assert -1.0 <= r.kappa_overall <= 1.0, pid


# ---------------------------------------------------------------------------
# TestPIACalculator
# ---------------------------------------------------------------------------


class TestPIACalculator:
    def test_load_pairs_count(self, calculator: PIACalculator) -> None:
        pairs = calculator.load_pairs()
        assert len(pairs) == 10

    def test_load_pairs_missing_dir_raises(self, tmp_path: Path) -> None:
        calc = PIACalculator(pairs_dir=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            calc.load_pairs()

    def test_run_produces_result(
        self, calculator: PIACalculator, tmp_path: Path
    ) -> None:
        calc = PIACalculator(
            pairs_dir=_PAIRS_DIR,
            output_path=tmp_path / "pia_results.json",
            dry_run=True,
        )
        result = calc.run()
        assert result.n_pairs == 10
        assert result.n_annotators == 5

    def test_delta_is_positive(self, calculator: PIACalculator, tmp_path: Path) -> None:
        """PIA must outperform standard IRR — the core claim."""
        calc = PIACalculator(
            pairs_dir=_PAIRS_DIR,
            output_path=tmp_path / "pia_results.json",
            dry_run=True,
        )
        result = calc.run()
        assert result.delta > 0, (
            f"delta={result.delta:.4f} must be positive (PIA κ > standard κ)"
        )

    def test_standard_interpretation_poor_or_slight(
        self, calculator: PIACalculator, tmp_path: Path
    ) -> None:
        calc = PIACalculator(
            pairs_dir=_PAIRS_DIR,
            output_path=tmp_path / "pia_results.json",
            dry_run=True,
        )
        result = calc.run()
        assert result.standard_interpretation in {"poor", "slight", "fair"}, (
            f"Standard IRR should be poor/slight/fair, "
            f"got {result.standard_interpretation}"
        )

    def test_pia_interpretation_moderate_or_better(
        self, calculator: PIACalculator, tmp_path: Path
    ) -> None:
        calc = PIACalculator(
            pairs_dir=_PAIRS_DIR,
            output_path=tmp_path / "pia_results.json",
            dry_run=True,
        )
        result = calc.run()
        assert result.pia_interpretation in {
            "moderate",
            "substantial",
            "almost perfect",
        }, f"PIA IRR should be moderate+, got {result.pia_interpretation}"

    def test_by_scenario_all_five_present(
        self, calculator: PIACalculator, tmp_path: Path
    ) -> None:
        calc = PIACalculator(
            pairs_dir=_PAIRS_DIR,
            output_path=tmp_path / "pia_results.json",
            dry_run=True,
        )
        result = calc.run()
        expected = {
            "health_alert",
            "privacy_sensitive",
            "location_trigger",
            "ambient_noise",
            "calendar_reminder",
        }
        assert set(result.by_scenario.keys()) == expected

    def test_by_scenario_delta_positive(
        self, calculator: PIACalculator, tmp_path: Path
    ) -> None:
        calc = PIACalculator(
            pairs_dir=_PAIRS_DIR,
            output_path=tmp_path / "pia_results.json",
            dry_run=True,
        )
        result = calc.run()
        for sc, comp in result.by_scenario.items():
            assert comp.delta > 0, f"{sc}: delta={comp.delta:.4f} must be positive"


# ---------------------------------------------------------------------------
# TestOutputSchema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    @pytest.fixture(scope="class")
    def result_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        out = tmp_path_factory.mktemp("pia") / "pia_results.json"
        calc = PIACalculator(pairs_dir=_PAIRS_DIR, output_path=out, dry_run=True)
        calc.run()
        return out

    def test_file_is_valid_json(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert isinstance(obj, dict)

    def test_top_level_keys(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        required = {
            "generated_at",
            "seed",
            "n_pairs",
            "n_annotators",
            "personas",
            "standard_total_steps",
            "standard_overall_kappa",
            "standard_interpretation",
            "standard_per_pair",
            "pia_total_agents",
            "pia_per_dimension_kappa",
            "pia_overall_kappa",
            "pia_interpretation",
            "pia_per_pair",
            "delta",
            "delta_headline",
            "by_scenario",
        }
        assert required <= set(obj.keys())

    def test_n_pairs_is_ten(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert obj["n_pairs"] == 10

    def test_n_annotators_is_five(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert obj["n_annotators"] == 5

    def test_standard_per_pair_has_ten_entries(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert len(obj["standard_per_pair"]) == 10

    def test_pia_per_pair_has_ten_entries(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert len(obj["pia_per_pair"]) == 10

    def test_pia_per_dimension_kappa_has_three_dims(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        dims = set(obj["pia_per_dimension_kappa"].keys())
        assert dims == {"planning_quality", "error_recovery", "goal_alignment"}

    def test_delta_positive(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert float(obj["delta"]) > 0

    def test_delta_headline_is_string(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        assert isinstance(obj["delta_headline"], str)
        assert len(obj["delta_headline"]) > 10

    def test_by_scenario_n_pairs_two_each(self, result_path: Path) -> None:
        obj = json.loads(result_path.read_text())
        for sc, comp in obj["by_scenario"].items():
            assert comp["n_pairs"] == 2, f"{sc}: n_pairs={comp['n_pairs']}"


# ---------------------------------------------------------------------------
# TestDryRunHelpers
# ---------------------------------------------------------------------------


class TestDryRunHelpers:
    def test_kappa_interpretation_labels(self) -> None:
        assert _kappa_interpretation(-0.1) == "poor"
        assert _kappa_interpretation(0.0) == "slight"
        assert _kappa_interpretation(0.20) == "slight"
        assert _kappa_interpretation(0.21) == "fair"
        assert _kappa_interpretation(0.40) == "fair"
        assert _kappa_interpretation(0.41) == "moderate"
        assert _kappa_interpretation(0.60) == "moderate"
        assert _kappa_interpretation(0.61) == "substantial"
        assert _kappa_interpretation(0.80) == "substantial"
        assert _kappa_interpretation(0.81) == "almost perfect"

    def test_dry_run_step_score_detour_fixed(self) -> None:
        for persona, expected in _DETOUR_SCORES.items():
            score = _dry_run_step_score("01", "agent_b", 1, "detour", persona)
            assert score == expected, f"{persona}: got {score}, want {expected}"

    def test_dry_run_step_score_standard_in_range(self) -> None:
        for persona, (lo, hi) in _STANDARD_STEP_BIAS.items():
            score = _dry_run_step_score("03", "agent_a", 0, "standard", persona)
            assert lo <= score <= hi, f"{persona}: score={score} outside [{lo}, {hi}]"

    def test_dry_run_step_score_deterministic(self) -> None:
        s1 = _dry_run_step_score("05", "agent_a", 2, "standard", "ProcessPurist")
        s2 = _dry_run_step_score("05", "agent_a", 2, "standard", "ProcessPurist")
        assert s1 == s2

    def test_dry_run_pia_score_direct_error_recovery_none(self) -> None:
        for scenario in _PIA_SCORES:
            for persona in _PERSONAS:
                result = _dry_run_pia_score(
                    scenario, "direct", "error_recovery", persona
                )
                assert result is None, (
                    f"{scenario}/direct/error_recovery/{persona}: "
                    f"expected None, got {result}"
                )

    def test_dry_run_pia_score_indirect_scores_in_range(self) -> None:
        for scenario in _PIA_SCORES:
            for dim in ("planning_quality", "error_recovery", "goal_alignment"):
                for persona in _PERSONAS:
                    score = _dry_run_pia_score(scenario, "indirect", dim, persona)
                    assert score is not None
                    assert 1 <= score <= 5, (
                        f"{scenario}/indirect/{dim}/{persona}: "
                        f"score={score} out of [1,5]"
                    )

    def test_pia_scores_table_covers_all_scenarios(self) -> None:
        expected_scenarios = {
            "health_alert",
            "privacy_sensitive",
            "location_trigger",
            "ambient_noise",
            "calendar_reminder",
        }
        assert set(_PIA_SCORES.keys()) == expected_scenarios

    def test_pia_scores_table_all_personas_present(self) -> None:
        for scenario in _PIA_SCORES:
            for path_style in ("direct", "indirect"):
                dims = _PIA_SCORES[scenario][path_style]
                for dim, persona_map in dims.items():
                    missing = set(_PERSONAS) - set(persona_map.keys())
                    assert not missing, (
                        f"{scenario}/{path_style}/{dim} missing personas: {missing}"
                    )
