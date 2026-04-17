"""Tests for src/annotation/annotator_simulator.py.

All tests use ``dry_run=True`` — no Anthropic API calls are made.  The dry-run
mode generates deterministic scores from per-persona bias tables seeded by
``(log_id, persona_name)``, making every test reproducible without mocking
the network.

Test inventory:
  TestOutputShape          — annotate_all() produces the correct number of records
  TestScoreRange           — all dimension scores are integers in [1, 4]
  TestAnnotationRecord     — AnnotationRecord fields, serialisation, rationale length
  TestDryRunReproducibility — identical inputs → identical scores across calls
  TestFleissKappaComputed  — compute_irr() returns float κ in [-1, 1] per dimension
  TestOutputSavedToJSONL   — output file is created and contains valid JSONL
  TestDisagreementHotspots — find_disagreement_hotspots() ranks by lowest κ first
  TestPersonaBiasDirection — persona biases produce expected score tendencies

Run:
    pytest tests/annotation/test_annotator_simulator.py -v
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from src.annotation.annotator_simulator import (
    _DIMENSIONS,
    _DRY_RUN_BIAS,
    _PERSONAS,
    AnnotationRecord,
    AnnotatorSimulator,
    compute_irr,
    find_disagreement_hotspots,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SCENARIO_TYPES = (
    "health_alert",
    "privacy_sensitive",
    "location_trigger",
    "ambient_noise",
    "calendar_reminder",
)

_CONSENT_MODELS = ("explicit", "implied", "ambient", "revoked")

_ACTIONS = (
    "send_alert",
    "log_and_monitor",
    "suppress_capture",
    "request_consent",
    "escalate_to_emergency",
    "transcribe_and_store",
    "summarise_and_notify",
    "no_action",
)


def _make_log(
    scenario_type: str = "health_alert",
    consent_model: str = "explicit",
    ground_truth_action: str = "send_alert",
) -> dict[str, Any]:
    """Build a minimal but structurally complete wearable log dict."""
    log_id = str(uuid.uuid4())
    return {
        "log_id": log_id,
        "scenario_type": scenario_type,
        "consent_model": consent_model,
        "ground_truth_action": ground_truth_action,
        "audio_transcript": {
            "text": "Patient reported chest discomfort.",
            "keywords_detected": ["chest", "pain"],
        },
        "context_metadata": {
            "location": "home",
            "time_of_day": "morning",
        },
        "trajectory": [
            {
                "step_index": 0,
                "step_name": "sense",
                "observation": "Heart rate 138 bpm; SpO2 97.2%; no motion.",
                "reasoning": "Elevated HR + reduced SpO2 consistent with cardiac event.",  # noqa: E501
                "action": "",
                "confidence": 0.93,
            },
            {
                "step_index": 1,
                "step_name": "plan",
                "observation": "5-min HR trend: persistent elevation.",
                "reasoning": "Dual-modality confirmation exceeds 0.85 threshold.",
                "action": "",
                "confidence": 0.88,
            },
            {
                "step_index": 2,
                "step_name": "act",
                "observation": "Alert threshold confirmed.",
                "reasoning": "Escalate to emergency services per protocol.",
                "action": "send_alert",
                "confidence": 0.95,
            },
        ],
    }


def _make_logs(n: int, **kwargs: str) -> list[dict[str, Any]]:
    """Return n distinct wearable logs, cycling through scenario types."""
    scenarios = list(_SCENARIO_TYPES)
    return [
        _make_log(
            scenario_type=kwargs.get("scenario_type", scenarios[i % len(scenarios)]),
            consent_model=kwargs.get("consent_model", "explicit"),
            ground_truth_action=kwargs.get("ground_truth_action", "send_alert"),
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def sim(tmp_path_factory: pytest.TempPathFactory) -> AnnotatorSimulator:
    """Module-scoped simulator in dry_run mode, writing to a temp directory."""
    out = tmp_path_factory.mktemp("annotations") / "test_annotations.jsonl"
    return AnnotatorSimulator(dry_run=True, output_path=out)


@pytest.fixture(scope="module")
def three_logs() -> list[dict[str, Any]]:
    """Three toy trajectories covering three distinct scenario types."""
    return _make_logs(3)


@pytest.fixture(scope="module")
def three_records(
    sim: AnnotatorSimulator,
    three_logs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """25 annotation records (5 logs × 5 personas) from a module-level run."""
    # Use 5 logs so compute_irr always has ≥ 2 complete trajectories.
    logs = _make_logs(5)
    return sim.annotate_all(logs)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    """annotate_all() produces exactly n_logs × 5 records."""

    def test_three_logs_produce_fifteen_records(
        self,
        sim: AnnotatorSimulator,
        three_logs: list[dict[str, Any]],
    ) -> None:
        """3 trajectories × 5 personas = 15 records."""
        records = sim.annotate_all(three_logs)
        assert len(records) == 3 * len(_PERSONAS)

    def test_each_record_has_all_four_dimensions(
        self,
        sim: AnnotatorSimulator,
        three_logs: list[dict[str, Any]],
    ) -> None:
        records = sim.annotate_all(three_logs)
        for rec in records:
            for dim in _DIMENSIONS:
                assert dim in rec, f"Missing dimension {dim!r} in record {rec}"

    def test_all_five_personas_represented(
        self,
        sim: AnnotatorSimulator,
        three_logs: list[dict[str, Any]],
    ) -> None:
        records = sim.annotate_all(three_logs)
        persona_names = {r["persona_name"] for r in records}
        assert persona_names == set(_PERSONAS.keys())

    def test_each_log_has_five_records(
        self,
        sim: AnnotatorSimulator,
        three_logs: list[dict[str, Any]],
    ) -> None:
        records = sim.annotate_all(three_logs)
        log_ids = [log["log_id"] for log in three_logs]
        for log_id in log_ids:
            count = sum(1 for r in records if r["log_id"] == log_id)
            assert count == len(_PERSONAS), (
                f"Expected {len(_PERSONAS)} records for log {log_id}, got {count}"
            )


# ---------------------------------------------------------------------------
# 2. Score range
# ---------------------------------------------------------------------------


class TestScoreRange:
    """All dimension scores are integers in [1, 4]."""

    @pytest.mark.parametrize("dim", _DIMENSIONS)
    def test_scores_in_valid_range(
        self,
        three_records: list[dict[str, Any]],
        dim: str,
    ) -> None:
        for rec in three_records:
            val = rec[dim]
            assert isinstance(val, int), (
                f"{dim} score is {type(val).__name__}, expected int"
            )
            assert 1 <= val <= 4, (
                f"{dim} score {val} out of [1, 4] for persona {rec['persona_name']}"
            )

    def test_scores_not_all_identical_across_personas(
        self,
        three_records: list[dict[str, Any]],
    ) -> None:
        """Persona bias must produce at least some score variation."""
        for dim in _DIMENSIONS:
            all_scores = [r[dim] for r in three_records]
            assert len(set(all_scores)) > 1, (
                f"All {dim} scores are identical ({all_scores[0]}) — "
                "bias ranges may be too narrow"
            )


# ---------------------------------------------------------------------------
# 3. AnnotationRecord dataclass
# ---------------------------------------------------------------------------


class TestAnnotationRecord:
    """AnnotationRecord fields, to_dict serialisation, and rationale length."""

    def test_to_dict_returns_all_required_keys(self) -> None:
        rec = AnnotationRecord(
            annotation_id="a1",
            log_id="l1",
            persona_name="OutcomeOptimist",
            scenario_type="health_alert",
            consent_model="explicit",
            ground_truth_action="send_alert",
            step_quality=3,
            privacy_compliance=3,
            goal_alignment=4,
            error_recovery=3,
            rationale="This is a rationale of sufficient length for the test.",
            created_at="2026-04-14T00:00:00+00:00",
        )
        d = rec.to_dict()
        expected_keys = {
            "annotation_id",
            "log_id",
            "persona_name",
            "scenario_type",
            "consent_model",
            "ground_truth_action",
            "step_quality",
            "privacy_compliance",
            "goal_alignment",
            "error_recovery",
            "rationale",
            "created_at",
        }
        assert expected_keys == set(d.keys())

    def test_to_dict_is_json_serialisable(self) -> None:
        rec = AnnotationRecord(
            annotation_id="a1",
            log_id="l1",
            persona_name="ProcessPurist",
            scenario_type="ambient_noise",
            consent_model="ambient",
            ground_truth_action="log_and_monitor",
            step_quality=2,
            privacy_compliance=2,
            goal_alignment=3,
            error_recovery=1,
            rationale="Rationale text that exceeds forty characters for this test.",
            created_at="2026-04-14T00:00:00+00:00",
        )
        serialised = json.dumps(rec.to_dict())
        assert isinstance(serialised, str)

    def test_dry_run_rationale_meets_minimum_length(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        """Every rationale must be at least 40 characters."""
        for rec in three_records:
            assert len(rec["rationale"]) >= 40, (
                f"rationale too short ({len(rec['rationale'])} chars) "
                f"for persona {rec['persona_name']}: {rec['rationale']!r}"
            )

    def test_annotation_id_is_unique_per_record(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        ids = [r["annotation_id"] for r in three_records]
        assert len(ids) == len(set(ids)), "Duplicate annotation_id values detected"


# ---------------------------------------------------------------------------
# 4. Dry-run reproducibility
# ---------------------------------------------------------------------------


class TestDryRunReproducibility:
    """Dry-run produces identical scores when called twice with the same log."""

    def test_same_log_same_persona_identical_scores(self, tmp_path: Path) -> None:
        log = _make_log()
        sim_a = AnnotatorSimulator(
            dry_run=True,
            output_path=tmp_path / "a.jsonl",
        )
        sim_b = AnnotatorSimulator(
            dry_run=True,
            output_path=tmp_path / "b.jsonl",
        )
        for persona in _PERSONAS:
            rec_a = sim_a.annotate_trajectory(log, persona)
            rec_b = sim_b.annotate_trajectory(log, persona)
            for dim in _DIMENSIONS:
                assert rec_a[dim] == rec_b[dim], (
                    f"Non-deterministic score for {persona}/{dim}: "
                    f"{rec_a[dim]} vs {rec_b[dim]}"
                )

    def test_different_log_ids_produce_different_scores_for_same_persona(
        self, tmp_path: Path
    ) -> None:
        """Different log IDs must (probabilistically) differ on ≥1 dimension."""
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "c.jsonl")
        logs = _make_logs(10)
        persona = "OutcomeOptimist"
        score_vectors = [
            tuple(sim.annotate_trajectory(log, persona)[dim] for dim in _DIMENSIONS)
            for log in logs
        ]
        # With 10 logs and 4-point scales, at least 2 distinct score vectors expected.
        assert len(set(score_vectors)) > 1, (
            "All 10 logs produced identical scores for OutcomeOptimist — "
            "seeding may be broken"
        )


# ---------------------------------------------------------------------------
# 5. Fleiss' κ computation
# ---------------------------------------------------------------------------


class TestFleissKappaComputed:
    """compute_irr() returns a float κ in [-1, 1] for every dimension."""

    def test_irr_returns_kappa_for_all_dimensions(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        for dim in _DIMENSIONS:
            assert dim in irr, f"Dimension {dim!r} missing from IRR result"
            assert "kappa" in irr[dim]

    @pytest.mark.parametrize("dim", _DIMENSIONS)
    def test_kappa_is_float_in_valid_range(
        self, three_records: list[dict[str, Any]], dim: str
    ) -> None:
        irr = compute_irr(three_records)
        kappa = float(irr[dim]["kappa"])
        assert -1.0 <= kappa <= 1.0, f"κ for {dim} = {kappa:.4f} is outside [-1, 1]"

    def test_overall_kappa_is_mean_of_dimensions(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        dim_kappas = [float(irr[d]["kappa"]) for d in _DIMENSIONS]
        expected_mean = sum(dim_kappas) / len(dim_kappas)
        assert abs(float(irr["overall"]["kappa"]) - expected_mean) < 1e-9

    def test_irr_includes_interpretation_string(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        valid_labels = {
            "poor",
            "slight",
            "fair",
            "moderate",
            "substantial",
            "almost perfect",
        }
        for dim in _DIMENSIONS:
            label = str(irr[dim]["interpretation"]).lower()
            assert label in valid_labels, (
                f"Unexpected interpretation {label!r} for {dim}"
            )

    def test_compute_irr_raises_on_too_few_complete_logs(self, tmp_path: Path) -> None:
        """Fewer than 2 complete trajectories must raise ValueError."""
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "few.jsonl")
        # Only 1 log → only 1 complete trajectory → should raise.
        one_log = _make_logs(1)
        records = sim.annotate_all(one_log)
        with pytest.raises(ValueError, match="Fleiss"):
            compute_irr(records)

    def test_persona_biases_produce_low_agreement(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        """With 5 biased personas, overall κ should be below 0.4 (fair or worse)."""
        irr = compute_irr(three_records)
        overall_kappa = float(irr["overall"]["kappa"])
        assert overall_kappa < 0.4, (
            f"Overall κ = {overall_kappa:.4f} is unexpectedly high; "
            "persona biases may not be producing genuine disagreement"
        )


# ---------------------------------------------------------------------------
# 6. Output saved to JSONL
# ---------------------------------------------------------------------------


class TestOutputSavedToJSONL:
    """Output file is created and every line is valid JSON."""

    def test_output_file_is_created(self, tmp_path: Path) -> None:
        out = tmp_path / "output.jsonl"
        sim = AnnotatorSimulator(dry_run=True, output_path=out)
        sim.annotate_all(_make_logs(2))
        assert out.exists(), "Output JSONL file was not created"

    def test_output_file_has_correct_line_count(self, tmp_path: Path) -> None:
        n_logs = 3
        out = tmp_path / "count_test.jsonl"
        sim = AnnotatorSimulator(dry_run=True, output_path=out)
        sim.annotate_all(_make_logs(n_logs))
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == n_logs * len(_PERSONAS)

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        out = tmp_path / "valid_json.jsonl"
        sim = AnnotatorSimulator(dry_run=True, output_path=out)
        sim.annotate_all(_make_logs(2))
        for i, line in enumerate(out.read_text(encoding="utf-8").splitlines()):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                pytest.fail(f"Line {i} is not valid JSON: {exc}\n{line!r}")
            assert isinstance(obj, dict), f"Line {i} did not parse to a dict"

    def test_jsonl_records_contain_all_required_keys(self, tmp_path: Path) -> None:
        required_keys = {
            "annotation_id",
            "log_id",
            "persona_name",
            "scenario_type",
            "consent_model",
            "ground_truth_action",
            *_DIMENSIONS,
            "rationale",
            "created_at",
        }
        out = tmp_path / "keys_test.jsonl"
        sim = AnnotatorSimulator(dry_run=True, output_path=out)
        sim.annotate_all(_make_logs(2))
        for line in out.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            missing = required_keys - rec.keys()
            assert not missing, f"Record missing keys: {missing}"

    def test_output_path_parent_directory_created(self, tmp_path: Path) -> None:
        nested_out = tmp_path / "nested" / "deep" / "annotations.jsonl"
        sim = AnnotatorSimulator(dry_run=True, output_path=nested_out)
        sim.annotate_all(_make_logs(2))
        assert nested_out.exists(), "annotate_all() did not create parent directories"


# ---------------------------------------------------------------------------
# 7. Disagreement hotspots
# ---------------------------------------------------------------------------


class TestDisagreementHotspots:
    """find_disagreement_hotspots() identifies the 3 lowest-κ dimensions."""

    def test_returns_top_three_hotspots(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=3)
        assert len(hotspots) == 3

    def test_hotspots_ordered_by_ascending_kappa(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        """Rank 1 must have the lowest κ (most disagreement)."""
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=3)
        kappas = [h["kappa"] for h in hotspots]
        assert kappas == sorted(kappas), (
            f"Hotspots are not sorted ascending by κ: {kappas}"
        )

    def test_hotspot_rank_starts_at_one(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=3)
        assert hotspots[0]["rank"] == 1
        assert hotspots[1]["rank"] == 2
        assert hotspots[2]["rank"] == 3

    def test_each_hotspot_has_required_keys(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=3)
        required = {
            "rank",
            "dimension",
            "kappa",
            "interpretation",
            "top_variance_log_ids",
            "top_variances",
        }
        for h in hotspots:
            missing = required - h.keys()
            assert not missing, f"Hotspot missing keys: {missing}"

    def test_hotspot_dimensions_are_valid(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=3)
        for h in hotspots:
            assert h["dimension"] in _DIMENSIONS

    def test_top_n_one_returns_single_hotspot(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=1)
        assert len(hotspots) == 1

    def test_top_variance_log_ids_are_strings(
        self, three_records: list[dict[str, Any]]
    ) -> None:
        irr = compute_irr(three_records)
        hotspots = find_disagreement_hotspots(three_records, irr, top_n=3)
        for h in hotspots:
            for log_id in h["top_variance_log_ids"]:
                assert isinstance(log_id, str), (
                    f"log_id {log_id!r} is {type(log_id).__name__}, expected str"
                )

    def test_known_variance_pattern_produces_expected_top_dimension(
        self, tmp_path: Path
    ) -> None:
        """Manually constructed records with extreme variance on step_quality.

        By giving PrivacyMaximalist score=1 and OutcomeOptimist score=4 for
        step_quality on every log (with other dimensions artificially uniform),
        step_quality must be the top disagreement dimension.
        """
        persona_names = list(_PERSONAS.keys())
        n_logs = 5
        logs = _make_logs(n_logs)
        records: list[dict[str, Any]] = []

        for log in logs:
            for i, persona in enumerate(persona_names):
                # step_quality: extreme spread (1 and 4 alternate by persona index)
                sq = 1 if i % 2 == 0 else 4
                # all other dims: uniform across personas → κ ≈ 1
                records.append(
                    {
                        "annotation_id": str(uuid.uuid4()),
                        "log_id": log["log_id"],
                        "persona_name": persona,
                        "scenario_type": log["scenario_type"],
                        "consent_model": log["consent_model"],
                        "ground_truth_action": log["ground_truth_action"],
                        "step_quality": sq,
                        "privacy_compliance": 3,
                        "goal_alignment": 3,
                        "error_recovery": 3,
                        "rationale": "Synthetic record for variance pattern test.",
                        "created_at": "2026-04-14T00:00:00+00:00",
                    }
                )

        irr = compute_irr(records)
        hotspots = find_disagreement_hotspots(records, irr, top_n=1)
        assert hotspots[0]["dimension"] == "step_quality", (
            f"Expected step_quality as top hotspot, got {hotspots[0]['dimension']}"
        )


# ---------------------------------------------------------------------------
# 8. Persona bias direction
# ---------------------------------------------------------------------------


class TestPersonaBiasDirection:
    """Persona bias ranges produce the documented score tendencies."""

    def test_privacy_maximalist_privacy_compliance_low(self, tmp_path: Path) -> None:
        """PrivacyMaximalist's privacy_compliance bias range is (1, 2)."""
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "pm.jsonl")
        logs = _make_logs(20, scenario_type="privacy_sensitive")
        records = sim.annotate_all(logs)
        pm_scores = [
            r["privacy_compliance"]
            for r in records
            if r["persona_name"] == "PrivacyMaximalist"
        ]
        lo, hi = _DRY_RUN_BIAS["PrivacyMaximalist"]["privacy_compliance"]
        assert all(lo <= s <= hi for s in pm_scores), (
            f"PrivacyMaximalist privacy_compliance scores outside [{lo},{hi}]: {pm_scores}"  # noqa: E501
        )

    def test_outcome_optimist_goal_alignment_high(self, tmp_path: Path) -> None:
        """OutcomeOptimist's goal_alignment bias range is (3, 4)."""
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "oo.jsonl")
        logs = _make_logs(20)
        records = sim.annotate_all(logs)
        oo_scores = [
            r["goal_alignment"]
            for r in records
            if r["persona_name"] == "OutcomeOptimist"
        ]
        lo, hi = _DRY_RUN_BIAS["OutcomeOptimist"]["goal_alignment"]
        assert all(lo <= s <= hi for s in oo_scores), (
            f"OutcomeOptimist goal_alignment scores outside [{lo},{hi}]: {oo_scores}"
        )

    def test_clinical_safety_first_health_alert_goal_alignment_high(
        self, tmp_path: Path
    ) -> None:
        """ClinicalSafetyFirst scores goal_alignment 3–4 on health_alert scenarios."""
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "csf.jsonl")
        logs = _make_logs(20, scenario_type="health_alert")
        records = sim.annotate_all(logs)
        csf_scores = [
            r["goal_alignment"]
            for r in records
            if r["persona_name"] == "ClinicalSafetyFirst"
        ]
        assert all(3 <= s <= 4 for s in csf_scores), (
            f"ClinicalSafetyFirst goal_alignment on health_alert outside [3,4]: {csf_scores}"  # noqa: E501
        )

    def test_clinical_safety_first_non_health_goal_alignment_low(
        self, tmp_path: Path
    ) -> None:
        """ClinicalSafetyFirst scores goal_alignment 1–2 on non-health scenarios."""
        sim = AnnotatorSimulator(
            dry_run=True, output_path=tmp_path / "csf_nonhealth.jsonl"
        )
        logs = _make_logs(20, scenario_type="calendar_reminder")
        records = sim.annotate_all(logs)
        csf_scores = [
            r["goal_alignment"]
            for r in records
            if r["persona_name"] == "ClinicalSafetyFirst"
        ]
        assert all(1 <= s <= 2 for s in csf_scores), (
            f"ClinicalSafetyFirst goal_alignment on calendar_reminder outside [1,2]: {csf_scores}"  # noqa: E501
        )

    def test_recovery_skeptic_error_recovery_low(self, tmp_path: Path) -> None:
        """RecoverySkeptic's error_recovery bias range is (1, 2)."""
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "rs.jsonl")
        logs = _make_logs(20)
        records = sim.annotate_all(logs)
        rs_scores = [
            r["error_recovery"]
            for r in records
            if r["persona_name"] == "RecoverySkeptic"
        ]
        lo, hi = _DRY_RUN_BIAS["RecoverySkeptic"]["error_recovery"]
        assert all(lo <= s <= hi for s in rs_scores), (
            f"RecoverySkeptic error_recovery scores outside [{lo},{hi}]: {rs_scores}"
        )

    def test_unknown_persona_raises_value_error(self, tmp_path: Path) -> None:
        sim = AnnotatorSimulator(dry_run=True, output_path=tmp_path / "err.jsonl")
        log = _make_log()
        with pytest.raises(ValueError, match="Unknown persona"):
            sim.annotate_trajectory(log, "NonExistentAnnotator")
