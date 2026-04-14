"""Integration tests: agenteval-schema-v1.json Layer 3 ↔ IRRCalculator.

Validates that the annotation schema's Layer 3 fields (nominal labels,
ordinal scores, and free-text rationales) are directly compatible with
every IRRCalculator method.  This is the named gap vs. Cohere Command A
(arXiv 2504.00698): 65 annotators, zero agreement statistics reported.

Test inventory:

1. test_schema_loads_and_validates — JSON Schema structure + constraint checks
2. test_nominal_fields_work_with_cohens_kappa — action_correct_for_context → κ
3. test_ordinal_fields_work_with_krippendorff — error_recovery_quality → α
4. test_rationale_field_works_with_bertscore — annotator_rationale → F1
5. test_full_pipeline_smoke — WearableLogGenerator → records → compute_all

Run:
    pytest tests/annotation/test_schema_irr_integration.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.annotation.argilla_loader import ArgillaTrajectoryLoader, PRS_DECODE
from src.annotation.irr_calculator import IRRCalculator
from src.data.wearable_generator import WearableLogGenerator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    Path(__file__).parents[2] / "data" / "annotations" / "agenteval-schema-v1.json"
)

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    """Load agenteval-schema-v1.json once for all tests in this module."""
    assert SCHEMA_PATH.exists(), (
        f"Schema not found at {SCHEMA_PATH}. "
        "Run: python configs/argilla/argilla_setup.py"
    )
    with SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def calc() -> IRRCalculator:
    """Shared calculator instance — stateless, safe to reuse."""
    return IRRCalculator()


@pytest.fixture(scope="module")
def loader() -> ArgillaTrajectoryLoader:
    """Loader instantiated without connecting to Argilla (lazy connection)."""
    return ArgillaTrajectoryLoader()


# ---------------------------------------------------------------------------
# 1. Schema structure
# ---------------------------------------------------------------------------


class TestSchemaLoadsAndValidates:
    """agenteval-schema-v1.json covers all three annotation layers."""

    def test_schema_file_is_valid_json(self, schema: dict[str, Any]) -> None:
        assert isinstance(schema, dict)

    def test_top_level_required_fields(self, schema: dict[str, Any]) -> None:
        required = set(schema["required"])
        assert "session" in required
        assert "roles" in required
        assert "steps" in required

    def test_layer_1_session_required_fields(self, schema: dict[str, Any]) -> None:
        session_props = schema["properties"]["session"]["properties"]
        for field in (
            "overall_goal_achieved",
            "session_outcome",
            "privacy_compliance_overall",
            "latency_acceptable",
        ):
            assert field in session_props, f"Layer 1 missing field: {field}"

    def test_layer_2_roles_is_array(self, schema: dict[str, Any]) -> None:
        roles = schema["properties"]["roles"]
        assert roles["type"] == "array"
        assert "items" in roles

    def test_layer_2_orchestrator_requires_handoff_quality(
        self, schema: dict[str, Any]
    ) -> None:
        """if/then/else should require handoff_quality for orchestrators only."""
        role_item = schema["properties"]["roles"]["items"]
        assert "if" in role_item, "Layer 2 missing if/then/else for handoff_quality"
        assert "then" in role_item
        assert "else" in role_item

    def test_layer_3_steps_is_array(self, schema: dict[str, Any]) -> None:
        steps = schema["properties"]["steps"]
        assert steps["type"] == "array"
        assert "items" in steps

    def test_layer_3_process_reward_score_range(self, schema: dict[str, Any]) -> None:
        prs = schema["properties"]["steps"]["items"]["properties"][
            "process_reward_score"
        ]
        assert prs["type"] == "number"
        assert prs["minimum"] == -1.0
        assert prs["maximum"] == 1.0

    def test_layer_3_partial_credit_range(self, schema: dict[str, Any]) -> None:
        pc = schema["properties"]["steps"]["items"]["properties"]["partial_credit"]
        assert pc["type"] == "number"
        assert pc["minimum"] == 0.0
        assert pc["maximum"] == 1.0

    def test_layer_3_annotator_rationale_min_length(
        self, schema: dict[str, Any]
    ) -> None:
        rationale = schema["properties"]["steps"]["items"]["properties"][
            "annotator_rationale"
        ]
        assert rationale["minLength"] == 20, (
            "minLength=20 enforces non-trivial rationale quality for BERTScore IRR"
        )

    def test_layer_3_tool_called_enum(self, schema: dict[str, Any]) -> None:
        tool_called = schema["properties"]["steps"]["items"]["properties"]["tool_called"]
        enum_vals = tool_called["enum"]
        expected_actions = {
            "send_alert",
            "suppress_capture",
            "trigger_geofence",
            "adjust_noise_profile",
            "surface_reminder",
            "log_and_monitor",
            "request_consent",
            "escalate_to_emergency",
            "",  # sense / plan steps have no action
        }
        assert set(enum_vals) == expected_actions

    def test_schema_has_examples(self, schema: dict[str, Any]) -> None:
        assert "examples" in schema
        assert len(schema["examples"]) >= 1, "Schema should include at least one example"

    def test_schema_metadata_version(self, schema: dict[str, Any]) -> None:
        meta = schema.get("schema_metadata", {})
        assert meta.get("version") == "1.0.0"
        assert "irr_integration" in meta, "Missing IRR integration note in schema_metadata"
        assert "prm_integration" in meta, "Missing PRM integration note in schema_metadata"


# ---------------------------------------------------------------------------
# 2. Nominal field → Cohen's κ
# ---------------------------------------------------------------------------


class TestNominalFieldsWithCohensKappa:
    """action_correct_for_context (correct=2, acceptable=1, incorrect=0) → κ."""

    # Encoding mirrors the natural ordinal ordering of the field values.
    # correct=2 / acceptable=1 / incorrect=0 so that κ is meaningful.
    _ENCODE: dict[str, int] = {"correct": 2, "acceptable": 1, "incorrect": 0}

    def _encode(self, labels: list[str]) -> list[int]:
        return [self._ENCODE[lbl] for lbl in labels]

    def test_returns_kappa_and_interpretation_keys(
        self, calc: IRRCalculator
    ) -> None:
        r1 = self._encode(["correct", "correct", "incorrect", "acceptable", "correct"])
        r2 = self._encode(["correct", "incorrect", "incorrect", "acceptable", "correct"])
        result = calc.cohens_kappa(r1, r2)
        assert "kappa" in result
        assert "interpretation" in result
        assert "n_items" in result

    def test_kappa_is_float_in_valid_range(self, calc: IRRCalculator) -> None:
        r1 = self._encode(["correct", "acceptable", "incorrect", "correct"])
        r2 = self._encode(["correct", "acceptable", "incorrect", "incorrect"])
        result = calc.cohens_kappa(r1, r2)
        kappa = float(result["kappa"])  # type: ignore[arg-type]
        assert -1.0 <= kappa <= 1.0

    def test_perfect_agreement_gives_kappa_one(self, calc: IRRCalculator) -> None:
        r1 = self._encode(["correct", "acceptable", "incorrect", "correct"])
        result = calc.cohens_kappa(r1, r1)
        assert float(result["kappa"]) == pytest.approx(1.0)  # type: ignore[arg-type]

    def test_complete_disagreement_gives_negative_or_low_kappa(
        self, calc: IRRCalculator
    ) -> None:
        """Systematic reversal across all 3 labels should produce κ < 0."""
        # Annotator A always says "correct" or "incorrect"; B always flips.
        r1 = [2, 0, 2, 0, 2, 0, 2, 0]
        r2 = [0, 2, 0, 2, 0, 2, 0, 2]
        result = calc.cohens_kappa(r1, r2)
        assert float(result["kappa"]) < 0.0  # type: ignore[arg-type]

    def test_n_items_matches_input_length(self, calc: IRRCalculator) -> None:
        labels = ["correct", "acceptable", "incorrect", "correct", "correct"]
        r = self._encode(labels)
        result = calc.cohens_kappa(r, r)
        assert result["n_items"] == len(labels)

    def test_privacy_compliant_binary_field_works(self, calc: IRRCalculator) -> None:
        """tool_call_privacy_compliant (0=non_compliant, 1=compliant) → κ."""
        r1 = [1, 1, 0, 1, 0, 1, 1, 0]
        r2 = [1, 0, 0, 1, 1, 1, 0, 0]
        result = calc.cohens_kappa(r1, r2)
        assert "kappa" in result
        assert -1.0 <= float(result["kappa"]) <= 1.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Ordinal field → Krippendorff's α
# ---------------------------------------------------------------------------


class TestOrdinalFieldsWithKrippendorff:
    """error_recovery_quality (not_applicable=0, poor=1, adequate=2, excellent=3) → α."""

    _ENCODE: dict[str, int] = {
        "not_applicable": 0,
        "poor": 1,
        "adequate": 2,
        "excellent": 3,
    }

    def _encode(self, labels: list[str]) -> list[float]:
        return [float(self._ENCODE[lbl]) for lbl in labels]

    def test_returns_alpha_and_required_keys(self, calc: IRRCalculator) -> None:
        rater_a = self._encode(
            ["excellent", "adequate", "poor", "not_applicable", "excellent"]
        )
        rater_b = self._encode(
            ["excellent", "poor", "poor", "not_applicable", "adequate"]
        )
        rater_c = self._encode(
            ["adequate", "adequate", "poor", "not_applicable", "excellent"]
        )
        result = calc.krippendorffs_alpha(
            [rater_a, rater_b, rater_c], level_of_measurement="ordinal"
        )
        for key in ("alpha", "interpretation", "n_raters", "n_items", "level_of_measurement"):
            assert key in result

    def test_alpha_in_valid_range(self, calc: IRRCalculator) -> None:
        rater_a = self._encode(
            ["excellent", "adequate", "poor", "not_applicable", "excellent"]
        )
        rater_b = self._encode(
            ["excellent", "adequate", "adequate", "not_applicable", "excellent"]
        )
        result = calc.krippendorffs_alpha(
            [rater_a, rater_b], level_of_measurement="ordinal"
        )
        alpha = float(result["alpha"])  # type: ignore[arg-type]
        assert -1.0 <= alpha <= 1.0

    def test_perfect_ordinal_agreement_gives_alpha_one(
        self, calc: IRRCalculator
    ) -> None:
        ratings = self._encode(["excellent", "adequate", "poor", "not_applicable"])
        result = calc.krippendorffs_alpha(
            [ratings, ratings, ratings], level_of_measurement="ordinal"
        )
        assert float(result["alpha"]) == pytest.approx(1.0, abs=1e-6)  # type: ignore[arg-type]

    def test_level_of_measurement_is_preserved(self, calc: IRRCalculator) -> None:
        rater_a = self._encode(["excellent", "adequate", "poor", "not_applicable"])
        rater_b = self._encode(["adequate", "adequate", "poor", "not_applicable"])
        result = calc.krippendorffs_alpha(
            [rater_a, rater_b], level_of_measurement="ordinal"
        )
        assert result["level_of_measurement"] == "ordinal"

    def test_three_raters_n_raters_correct(self, calc: IRRCalculator) -> None:
        row = self._encode(["excellent", "adequate", "poor", "not_applicable", "poor"])
        result = calc.krippendorffs_alpha(
            [row, row, row], level_of_measurement="ordinal"
        )
        assert result["n_raters"] == 3
        assert result["n_items"] == 5

    def test_missing_values_are_handled(self, calc: IRRCalculator) -> None:
        """None encodes a step that annotator B did not review."""
        rater_a: list[float | None] = [3.0, 2.0, 1.0, 0.0, 3.0]
        rater_b: list[float | None] = [3.0, None, 1.0, 0.0, 2.0]  # step 1 not rated
        result = calc.krippendorffs_alpha(
            [rater_a, rater_b], level_of_measurement="ordinal"
        )
        assert result["missing_values"] == 1
        assert -1.0 <= float(result["alpha"]) <= 1.0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. Rationale field → BERTScore
# ---------------------------------------------------------------------------


class TestRationaleFieldWithBERTScore:
    """annotator_rationale free-text → bertscore_agreement().

    BERTScore F1 interpretations (from IRRCalculator.bertscore_agreement):
      ≥ 0.85 → "high semantic agreement"
      ≥ 0.70 → "moderate semantic agreement"
      < 0.70 → "low semantic agreement"

    The named gap vs. Cohere Command A: they reported zero κ or α values for
    65 annotators.  BERTScore on rationale strings catches the case where
    raters pick the same label for completely different reasons — a failure
    mode invisible to label-agreement metrics alone.
    """

    # Semantically similar pairs — both describe the same policy trigger.
    _SPECIFIC_A = [
        (
            "Agent correctly calls suppress_capture under AMBIENT consent because "
            "audio keywords include 'private' and the environment is classified as "
            "therapy_office. No data leaves the device. Matches policy §3.2."
        ),
        (
            "SpO2=88% is below the 90% alert threshold and HR=158bpm exceeds the "
            "140bpm resting threshold. Dual-modality confirmation met. "
            "send_alert is the required action under health_agent authority."
        ),
        (
            "ConsentModel=REVOKED. log_and_monitor persists audio to cloud storage, "
            "which is a categorical policy violation — REVOKED means zero capture. "
            "Non-compliant score is definitive, not a judgment call."
        ),
    ]
    _SPECIFIC_B = [
        (
            "suppress_capture is correct here: AMBIENT consent applies, the keyword "
            "'private' was detected, and the location context is a therapy office. "
            "Policy requires local-only processing in this scenario."
        ),
        (
            "Both biometric signals exceed alert thresholds simultaneously: SpO2 at "
            "88% (threshold 90%) and heart rate at 158 (threshold 140 at rest). "
            "send_alert is the only defensible action per the dual-modality rule."
        ),
        (
            "REVOKED consent prohibits any data storage. The agent called "
            "log_and_monitor which stores data to the cloud — this violates the "
            "explicit REVOKED policy regardless of the audio content."
        ),
    ]

    # Vague pairs — generic one-liners with no policy grounding.
    _VAGUE_A = [
        "Looks correct to me.",
        "Good action.",
        "The agent did the right thing here.",
    ]
    _VAGUE_B = [
        "Seems fine.",
        "Action seems appropriate.",
        "Seems like the right decision was made.",
    ]

    def test_specific_rationales_return_required_keys(
        self, calc: IRRCalculator
    ) -> None:
        result = calc.bertscore_agreement(
            self._SPECIFIC_A,
            self._SPECIFIC_B,
            model_type="distilbert-base-uncased",
        )
        for key in (
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "f1_per_pair",
            "model",
            "n_pairs",
            "interpretation",
        ):
            assert key in result

    def test_specific_rationales_moderate_or_high_f1(
        self, calc: IRRCalculator
    ) -> None:
        """Pairs that cite the same policy triggers should reach at least
        'moderate semantic agreement' (F1 ≥ 0.70) with distilbert.
        """
        result = calc.bertscore_agreement(
            self._SPECIFIC_A,
            self._SPECIFIC_B,
            model_type="distilbert-base-uncased",
        )
        f1 = float(result["f1_mean"])  # type: ignore[arg-type]
        assert f1 >= 0.70, (
            f"Expected F1 ≥ 0.70 for policy-grounded rationale pairs; got {f1:.4f}. "
            "Low F1 here signals that the annotator rationale schema is not producing "
            "grounded justifications — a training/calibration issue."
        )

    def test_f1_per_pair_length_matches_input(
        self, calc: IRRCalculator
    ) -> None:
        result = calc.bertscore_agreement(
            self._SPECIFIC_A,
            self._SPECIFIC_B,
            model_type="distilbert-base-uncased",
        )
        assert len(result["f1_per_pair"]) == len(self._SPECIFIC_A)  # type: ignore[arg-type]

    def test_vague_vs_specific_f1_ordering(
        self, calc: IRRCalculator
    ) -> None:
        """Specific pairs must achieve higher F1 than vague pairs.

        This is the core Cohere gap test: rationale quality matters.
        If vague rationales score just as high as specific ones, the BERTScore
        signal is not capturing policy grounding — and the annotation schema
        is not working as a quality gate.
        """
        specific_result = calc.bertscore_agreement(
            self._SPECIFIC_A,
            self._SPECIFIC_B,
            model_type="distilbert-base-uncased",
        )
        vague_result = calc.bertscore_agreement(
            self._VAGUE_A,
            self._VAGUE_B,
            model_type="distilbert-base-uncased",
        )
        specific_f1 = float(specific_result["f1_mean"])  # type: ignore[arg-type]
        vague_f1 = float(vague_result["f1_mean"])  # type: ignore[arg-type]
        assert specific_f1 >= vague_f1, (
            f"Expected specific_f1 ({specific_f1:.4f}) ≥ vague_f1 ({vague_f1:.4f}). "
            "Vague rationales ('Looks fine') should not match better than policy-grounded ones."
        )

    def test_n_pairs_reported_correctly(self, calc: IRRCalculator) -> None:
        result = calc.bertscore_agreement(
            self._SPECIFIC_A[:2],
            self._SPECIFIC_B[:2],
            model_type="distilbert-base-uncased",
        )
        assert result["n_pairs"] == 2

    def test_identical_rationales_give_f1_one(
        self, calc: IRRCalculator
    ) -> None:
        texts = [self._SPECIFIC_A[0], self._SPECIFIC_A[1]]
        result = calc.bertscore_agreement(
            texts,
            texts,
            model_type="distilbert-base-uncased",
        )
        assert float(result["f1_mean"]) == pytest.approx(1.0, abs=1e-4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Full pipeline smoke test
# ---------------------------------------------------------------------------


class TestFullPipelineSmoke:
    """WearableLogGenerator → ArgillaTrajectoryLoader.trajectory_to_records()
    → label encoding → IRRCalculator.compute_all().

    No Argilla server required — trajectory_to_records() is a pure in-memory
    conversion; the lazy _connect() is never called.
    """

    def test_generate_logs_and_build_records(
        self, loader: ArgillaTrajectoryLoader
    ) -> None:
        """5 logs × 3 steps = 15 records, all with required fields set."""
        gen = WearableLogGenerator(seed=99)
        logs = gen.generate_batch(5)

        all_records = []
        for log in logs:
            records = loader.trajectory_to_records(log)
            all_records.extend(records)

        assert len(all_records) == 15  # 5 logs × 3 steps
        for rec in all_records:
            assert rec.fields["step_observation"]
            assert rec.fields["step_reasoning"]
            assert rec.metadata["step_id"]
            assert rec.metadata["scenario_type"]

    def test_step_id_format_is_deterministic(
        self, loader: ArgillaTrajectoryLoader
    ) -> None:
        """step_id must be '{log_id}_{step_index}' for join key integrity."""
        gen = WearableLogGenerator(seed=42)
        log = gen.generate_batch(1)[0]
        records = loader.trajectory_to_records(log)

        for i, rec in enumerate(records):
            expected_id = f"{log.log_id}_{i}"
            assert rec.id == expected_id
            assert rec.metadata["step_id"] == expected_id

    def test_privacy_suggestion_present_on_all_records(
        self, loader: ArgillaTrajectoryLoader
    ) -> None:
        gen = WearableLogGenerator(seed=77)
        logs = gen.generate_batch(3)
        for log in logs:
            for rec in loader.trajectory_to_records(log):
                assert len(rec.suggestions) >= 1
                # RecordSuggestions is keyed by question_name, not integer index.
                suggestion = rec.suggestions["tool_call_privacy_compliant"]
                assert suggestion.question_name == "tool_call_privacy_compliant"
                assert suggestion.value in {"compliant", "non_compliant"}
                assert 0.0 < suggestion.score <= 1.0

    def test_revoked_consent_flags_non_compliant(
        self, loader: ArgillaTrajectoryLoader
    ) -> None:
        """Any act-step under REVOKED consent with a non-allowed action must
        receive a non_compliant pre-fill suggestion with score=0.95."""
        from src.data.privacy_gate import ConsentModel

        label, score = ArgillaTrajectoryLoader._suggest_privacy_compliant(
            "log_and_monitor", ConsentModel.REVOKED
        )
        assert label == "non_compliant"
        assert score == pytest.approx(0.95)

    def test_ambient_blocked_action_flags_non_compliant(
        self, loader: ArgillaTrajectoryLoader
    ) -> None:
        from src.data.privacy_gate import ConsentModel

        label, score = ArgillaTrajectoryLoader._suggest_privacy_compliant(
            "trigger_geofence", ConsentModel.AMBIENT
        )
        assert label == "non_compliant"
        assert score == pytest.approx(0.80)

    def test_compute_all_on_simulated_annotation_labels(
        self, calc: IRRCalculator
    ) -> None:
        """Simulate two annotators scoring action_correct_for_context on 5 steps.

        correct=2, acceptable=1, incorrect=0 — same encoding as
        TestNominalFieldsWithCohensKappa.  compute_all() must return the
        quality_gate_passed key in its summary dict.
        """
        rater1 = [2, 2, 1, 0, 2]
        rater2 = [2, 1, 1, 0, 2]
        result = calc.compute_all(rater1, rater2)

        assert "cohens_kappa" in result
        assert "fleiss_kappa" in result
        assert "krippendorffs_alpha" in result
        assert "summary" in result
        assert "quality_gate_passed" in result["summary"]
        assert isinstance(result["summary"]["quality_gate_passed"], bool)

    def test_compute_all_min_kappa_is_conservative_lower_bound(
        self, calc: IRRCalculator
    ) -> None:
        """min_kappa must equal the smallest of the three label-agreement values."""
        rater1 = [2, 2, 0, 1, 2, 0, 2, 1]
        rater2 = [2, 1, 0, 1, 1, 0, 2, 2]
        result = calc.compute_all(rater1, rater2)
        cohens = float(result["cohens_kappa"]["kappa"])  # type: ignore[index]
        fleiss = float(result["fleiss_kappa"]["kappa"])  # type: ignore[index]
        kripp = float(result["krippendorffs_alpha"]["alpha"])  # type: ignore[index]
        min_kappa = float(result["summary"]["min_kappa"])  # type: ignore[index]
        assert min_kappa == pytest.approx(min(cohens, fleiss, kripp))

    def test_prs_decode_table_covers_all_argilla_rating_values(self) -> None:
        """PRS_DECODE must cover every integer Argilla rating value [0..8].

        Argilla v2 RatingQuestion requires integers in [0, 10].  We use [0..8]
        to encode the [-1.0, +1.0] PRM signal.  The decode table must be
        complete — missing entries would cause KeyError in export_annotations().
        """
        for rating in range(9):
            assert rating in PRS_DECODE
            prs = PRS_DECODE[rating]
            assert -1.0 <= prs <= 1.0
            # Decode formula: prs = (rating - 4) * 0.25
            assert prs == pytest.approx((rating - 4) * 0.25)

    def test_all_five_scenario_types_produce_records(
        self, loader: ArgillaTrajectoryLoader
    ) -> None:
        """Every ScenarioType must appear in the generated batch so all
        scenario-type metadata paths are exercised."""
        from src.data.wearable_generator import ScenarioType

        gen = WearableLogGenerator(seed=0)
        # 25 logs at 5 per scenario type (generator distributes evenly at seed=0)
        logs = gen.generate_batch(25)
        scenario_types_seen: set[str] = set()
        for log in logs:
            records = loader.trajectory_to_records(log)
            for rec in records:
                scenario_types_seen.add(str(rec.metadata["scenario_type"]))

        expected = {st.value for st in ScenarioType}
        assert scenario_types_seen == expected, (
            f"Missing scenario types: {expected - scenario_types_seen}"
        )
