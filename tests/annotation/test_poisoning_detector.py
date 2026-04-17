"""Tests for src.annotation.poisoning_detector.

Five test classes covering the PoisoningDetector public API:

  TestDetectOutlierAnnotators  — suspicion scores on real annotation data
  TestInjectSyntheticPoisoners — augmented pool structure + ground-truth fields
  TestPoisonersDetectable      — controlled synthetic pool confirms detection logic
  TestEvaluateDetection        — output schema and metric properties
  TestCleanlabLabelQuality     — cleanlab integration (skipped if not installed)
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

import pytest

from src.annotation.poisoning_detector import (
    RUBRIC_DIMENSIONS,
    PoisoningDetector,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANNOTATIONS_PATH = Path("data/annotations/day12_annotations.jsonl")
_REAL_PERSONA_NAMES = {
    "PrivacyMaximalist",
    "OutcomeOptimist",
    "ProcessPurist",
    "ClinicalSafetyFirst",
    "RecoverySkeptic",
}


@pytest.fixture(scope="module")
def real_annotations() -> list[dict[str, Any]]:
    """Load the real 25-record annotation file once for the whole module."""
    records: list[dict[str, Any]] = []
    with _ANNOTATIONS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert records, f"Annotation file is empty: {_ANNOTATIONS_PATH}"
    return records


def _make_synthetic_pool(
    honest_personas: list[str],
    honest_score: int,
    poisoner_name: str,
    poisoner_score: int,
    n_logs: int = 5,
) -> list[dict[str, Any]]:
    """Build a minimal annotation pool with unanimous honest annotators and one outlier.

    All honest annotators rate every log with ``honest_score`` on all four
    dimensions.  The poisoner rates every log with ``poisoner_score`` on all
    four dimensions.  With no within-group variance, the MAD-based detector
    has a clean signal to work with.
    """
    records: list[dict[str, Any]] = []
    log_ids = [f"synthetic-log-{i:02d}" for i in range(n_logs)]
    all_personas = honest_personas + [poisoner_name]
    for log_id in log_ids:
        for persona in all_personas:
            score = honest_score if persona in honest_personas else poisoner_score
            records.append(
                {
                    "annotation_id": str(uuid.uuid4()),
                    "log_id": log_id,
                    "persona_name": persona,
                    "scenario_type": "health_alert",
                    "consent_model": "explicit",
                    "ground_truth_action": "send_alert",
                    "step_quality": score,
                    "privacy_compliance": score,
                    "goal_alignment": score,
                    "error_recovery": score,
                    "rationale": f"[TEST] {persona} on {log_id}: score={score}",
                    "created_at": "2026-04-15T00:00:00+00:00",
                }
            )
    return records


# ---------------------------------------------------------------------------
# 1. detect_outlier_annotators — real data
# ---------------------------------------------------------------------------


class TestDetectOutlierAnnotators:
    """Suspicion scores on the real 25-record annotation pool."""

    def test_returns_all_personas(self, real_annotations: list[dict[str, Any]]) -> None:
        """All five persona names must appear as keys in the output dict."""
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(real_annotations)
        assert set(scores.keys()) == _REAL_PERSONA_NAMES

    def test_scores_are_floats_in_unit_interval(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Every suspicion score must be a finite float in [0.0, 1.0]."""
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(real_annotations)
        for persona, score in scores.items():
            assert isinstance(score, float), f"{persona} score is not float"
            assert math.isfinite(score), f"{persona} score is not finite: {score}"
            assert 0.0 <= score <= 1.0, (
                f"{persona} score {score:.4f} is outside [0.0, 1.0]"
            )

    def test_max_score_is_one(self, real_annotations: list[dict[str, Any]]) -> None:
        """The most deviant annotator must receive a normalised score of exactly 1.0."""
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(real_annotations)
        assert max(scores.values()) == pytest.approx(1.0)

    def test_min_score_is_zero(self, real_annotations: list[dict[str, Any]]) -> None:
        """The least deviant annotator must receive a normalised score of exactly 0.0."""  # noqa: E501
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(real_annotations)
        assert min(scores.values()) == pytest.approx(0.0)

    def test_raises_on_empty_input(self) -> None:
        """detect_outlier_annotators must raise ValueError on an empty list."""
        detector = PoisoningDetector()
        with pytest.raises(ValueError, match="must not be empty"):
            detector.detect_outlier_annotators([])

    def test_deterministic(self, real_annotations: list[dict[str, Any]]) -> None:
        """Calling the method twice must return identical scores (no randomness)."""
        detector = PoisoningDetector()
        scores_a = detector.detect_outlier_annotators(real_annotations)
        scores_b = detector.detect_outlier_annotators(real_annotations)
        for persona in scores_a:
            assert scores_a[persona] == pytest.approx(scores_b[persona])


# ---------------------------------------------------------------------------
# 2. inject_synthetic_poisoners — pool structure and ground-truth fields
# ---------------------------------------------------------------------------


class TestInjectSyntheticPoisoners:
    """Augmented pool structure, injected record fields, and label integrity."""

    def test_augmented_count_is_original_plus_injected(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Augmented list length = original + (n_malicious × n_unique_logs)."""
        detector = PoisoningDetector()
        n_malicious = 3
        augmented = detector.inject_synthetic_poisoners(
            real_annotations, n_malicious=n_malicious
        )
        n_logs = len({r["log_id"] for r in real_annotations})
        expected = len(real_annotations) + n_malicious * n_logs
        assert len(augmented) == expected

    def test_all_three_poisoner_names_present(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """All three canonical poisoner names must appear in the augmented pool."""
        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        augmented_personas = {r["persona_name"] for r in augmented}
        for name in ("Poisoner_A", "Poisoner_B", "Poisoner_C"):
            assert name in augmented_personas, f"{name} missing from augmented pool"

    def test_injected_records_have_ground_truth_flag(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Every injected record must carry is_injected_poisoner == True."""
        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        injected = [r for r in augmented if r["persona_name"].startswith("Poisoner_")]
        assert injected, "No injected records found"
        for record in injected:
            assert record.get("is_injected_poisoner") is True, (
                f"Record {record['annotation_id']} missing is_injected_poisoner=True"
            )

    def test_original_records_lack_ground_truth_flag(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Original records must NOT have is_injected_poisoner set."""
        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        original_augmented = [
            r for r in augmented if not r["persona_name"].startswith("Poisoner_")
        ]
        for record in original_augmented:
            assert "is_injected_poisoner" not in record, (
                f"Original record {record['annotation_id']} unexpectedly has "
                "is_injected_poisoner field"
            )

    def test_injected_scores_respect_label_bounds(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Poisoner dimension scores must stay within the valid 1–4 range."""
        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        injected = [r for r in augmented if r.get("is_injected_poisoner")]
        for record in injected:
            for dim in RUBRIC_DIMENSIONS:
                score = record[dim]
                assert 1 <= score <= 4, (
                    f"Poisoner score {score} for '{dim}' is outside [1, 4]"
                )

    def test_injected_privacy_score_below_consensus(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Poisoners must suppress privacy_compliance below the original consensus."""
        # Compute original per-log consensus for privacy_compliance.
        from collections import defaultdict

        dim_scores: dict[str, list[float]] = defaultdict(list)
        for r in real_annotations:
            dim_scores[r["log_id"]].append(float(r["privacy_compliance"]))
        original_consensus = {
            log_id: sum(vals) / len(vals) for log_id, vals in dim_scores.items()
        }

        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        injected = [r for r in augmented if r.get("is_injected_poisoner")]

        for record in injected:
            log_id = record["log_id"]
            consensus = original_consensus[log_id]
            # Poisoner should be ≤ consensus (biased downward).
            assert record["privacy_compliance"] <= round(consensus), (
                f"Poisoner privacy_compliance {record['privacy_compliance']} "
                f"exceeds consensus {consensus:.2f} for log {log_id}"
            )

    def test_injected_step_quality_above_consensus(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Poisoners must inflate step_quality above the original consensus."""
        from collections import defaultdict

        dim_scores: dict[str, list[float]] = defaultdict(list)
        for r in real_annotations:
            dim_scores[r["log_id"]].append(float(r["step_quality"]))
        original_consensus = {
            log_id: sum(vals) / len(vals) for log_id, vals in dim_scores.items()
        }

        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        injected = [r for r in augmented if r.get("is_injected_poisoner")]

        for record in injected:
            log_id = record["log_id"]
            consensus = original_consensus[log_id]
            # Poisoner should be ≥ consensus (biased upward).
            assert record["step_quality"] >= round(consensus), (
                f"Poisoner step_quality {record['step_quality']} "
                f"is below consensus {consensus:.2f} for log {log_id}"
            )

    def test_original_annotations_not_mutated(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """inject_synthetic_poisoners must not modify the input list in place."""
        original_ids = {r["annotation_id"] for r in real_annotations}
        detector = PoisoningDetector()
        detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)
        # The input list itself must be unchanged.
        assert {r["annotation_id"] for r in real_annotations} == original_ids

    def test_n_malicious_exceeds_three_raises(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """n_malicious > 3 must raise ValueError (only 3 names defined)."""
        detector = PoisoningDetector()
        with pytest.raises(ValueError, match="n_malicious must be"):
            detector.inject_synthetic_poisoners(real_annotations, n_malicious=4)

    def test_seed_produces_deterministic_records(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Same seed must produce records with the same dimension scores."""
        detector = PoisoningDetector()
        aug_a = detector.inject_synthetic_poisoners(real_annotations, seed=7)
        aug_b = detector.inject_synthetic_poisoners(real_annotations, seed=7)
        injected_a = [r for r in aug_a if r.get("is_injected_poisoner")]
        injected_b = [r for r in aug_b if r.get("is_injected_poisoner")]
        assert len(injected_a) == len(injected_b)
        # Sort by (persona_name, log_id) to align records across runs.
        key = lambda r: (r["persona_name"], r["log_id"])  # noqa: E731
        for r_a, r_b in zip(sorted(injected_a, key=key), sorted(injected_b, key=key)):
            for dim in RUBRIC_DIMENSIONS:
                assert r_a[dim] == r_b[dim]


# ---------------------------------------------------------------------------
# 3. Detection logic — controlled synthetic pool
# ---------------------------------------------------------------------------


class TestPoisonersDetectable:
    """Verify the MAD-based detector on a minimal, controlled annotation pool.

    Using real annotation data here would conflate detection performance with
    the biases already embedded in the five legitimate personas.  A synthetic
    pool with unanimous honest raters and a single clear outlier gives a
    clean unit test for the detection algorithm itself.
    """

    def test_unanimous_pool_single_outlier_scores_highest(self) -> None:
        """A single outlier who rates every log at 1 while the pool rates at 3
        must receive a suspicion score of exactly 1.0 (maximum)."""
        records = _make_synthetic_pool(
            honest_personas=["Honest_A", "Honest_B", "Honest_C", "Honest_D"],
            honest_score=3,
            poisoner_name="Poisoner_A",
            poisoner_score=1,
            n_logs=5,
        )
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(records)
        assert scores["Poisoner_A"] == pytest.approx(1.0), (
            f"Outlier scored {scores['Poisoner_A']:.4f}, expected 1.0"
        )

    def test_unanimous_honest_annotators_score_zero(self) -> None:
        """All honest annotators in the controlled pool must share score 0.0."""
        records = _make_synthetic_pool(
            honest_personas=["Honest_A", "Honest_B", "Honest_C", "Honest_D"],
            honest_score=3,
            poisoner_name="Poisoner_A",
            poisoner_score=1,
        )
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(records)
        for persona in ("Honest_A", "Honest_B", "Honest_C", "Honest_D"):
            assert scores[persona] == pytest.approx(0.0), (
                f"{persona} scored {scores[persona]:.4f}, expected 0.0"
            )

    def test_outlier_ranked_above_all_honest(self) -> None:
        """The outlier's suspicion score must exceed every honest annotator's score."""
        records = _make_synthetic_pool(
            honest_personas=["Honest_A", "Honest_B", "Honest_C"],
            honest_score=3,
            poisoner_name="Badactor",
            poisoner_score=1,
        )
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(records)
        poisoner_score = scores["Badactor"]
        for persona in ("Honest_A", "Honest_B", "Honest_C"):
            assert poisoner_score > scores[persona], (
                f"Badactor ({poisoner_score:.4f}) not ranked above "
                f"{persona} ({scores[persona]:.4f})"
            )

    def test_inject_and_detect_single_poisoner_in_clean_pool(self) -> None:
        """inject_synthetic_poisoners then detect_outlier_annotators on a clean pool.

        Start with four unanimous honest annotators all rating 3 on every log.
        Inject 1 synthetic poisoner.  The poisoner must rank as the most
        suspicious annotator in the augmented pool.
        """
        # Build a clean pool where all 4 honest annotators agree perfectly.
        records = _make_synthetic_pool(
            honest_personas=["Honest_A", "Honest_B", "Honest_C", "Honest_D"],
            honest_score=3,
            # Use a fifth honest annotator as placeholder — will be replaced.
            poisoner_name="Filler",
            poisoner_score=3,  # Filler is also honest; n_malicious injects real bias.
        )
        # Remove the filler — we only want 4 honest annotators.
        records = [r for r in records if r["persona_name"] != "Filler"]

        detector = PoisoningDetector()
        augmented = detector.inject_synthetic_poisoners(records, n_malicious=1)

        injected = [r for r in augmented if r.get("is_injected_poisoner")]
        assert len(injected) > 0, "No injected records found"
        poisoner_name = injected[0]["persona_name"]

        scores = detector.detect_outlier_annotators(augmented)
        poisoner_score = scores[poisoner_name]
        honest_scores = [s for name, s in scores.items() if name != poisoner_name]
        # The single injected poisoner should outrank every honest annotator.
        assert poisoner_score > max(honest_scores), (
            f"Poisoner ({poisoner_score:.4f}) did not outscore all honest "
            f"annotators (max honest={max(honest_scores):.4f})"
        )

    def test_identical_annotators_produce_zero_range(self) -> None:
        """When all annotators agree perfectly, all suspicion scores must be 0.0."""
        records = _make_synthetic_pool(
            honest_personas=["A", "B", "C"],
            honest_score=2,
            poisoner_name="D",
            poisoner_score=2,  # Also identical — no outlier.
        )
        detector = PoisoningDetector()
        scores = detector.detect_outlier_annotators(records)
        for persona, score in scores.items():
            assert score == pytest.approx(0.0), (
                f"Expected 0.0 for {persona} in unanimous pool, got {score:.4f}"
            )


# ---------------------------------------------------------------------------
# 4. evaluate_detection — output schema and metric properties
# ---------------------------------------------------------------------------


class TestEvaluateDetection:
    """Output schema, metric properties, and edge-case semantics."""

    _INJECTED_NAMES = ["Poisoner_A", "Poisoner_B", "Poisoner_C"]

    @pytest.fixture(scope="class")
    def augmented_pool(
        self, real_annotations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        detector = PoisoningDetector()
        return detector.inject_synthetic_poisoners(real_annotations, n_malicious=3)

    def test_output_has_required_keys(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """evaluate_detection must return a dict with all seven required keys."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(augmented_pool, self._INJECTED_NAMES)
        required_keys = {
            "threshold",
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
            "per_annotator_scores",
        }
        assert required_keys.issubset(report.keys()), (
            f"Missing keys: {required_keys - report.keys()}"
        )

    def test_per_annotator_scores_is_dict(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """per_annotator_scores must be a dict mapping persona name → float."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(augmented_pool, self._INJECTED_NAMES)
        pas = report["per_annotator_scores"]
        assert isinstance(pas, dict)
        for name, score in pas.items():
            assert isinstance(name, str)
            assert isinstance(score, float)

    def test_per_annotator_scores_covers_all_personas(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """per_annotator_scores must include both legitimate and injected annotators."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(augmented_pool, self._INJECTED_NAMES)
        pas_names = set(report["per_annotator_scores"].keys())
        for name in _REAL_PERSONA_NAMES | set(self._INJECTED_NAMES):
            assert name in pas_names, f"{name} missing from per_annotator_scores"

    def test_tp_plus_fn_equals_ground_truth_positives(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """TP + FN must equal the number of injected annotator names."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(augmented_pool, self._INJECTED_NAMES)
        assert report["true_positives"] + report["false_negatives"] == len(
            self._INJECTED_NAMES
        )

    def test_precision_recall_in_unit_interval(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """Precision and recall must be floats in [0.0, 1.0]."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(augmented_pool, self._INJECTED_NAMES)
        assert 0.0 <= report["precision"] <= 1.0
        assert 0.0 <= report["recall"] <= 1.0

    def test_f1_is_harmonic_mean_of_precision_and_recall(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """F1 must equal the harmonic mean of precision and recall."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(augmented_pool, self._INJECTED_NAMES)
        p = report["precision"]
        r = report["recall"]
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert report["f1"] == pytest.approx(expected_f1, abs=1e-4)
        else:
            assert report["f1"] == pytest.approx(0.0)

    def test_threshold_echoed_in_output(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """The threshold value passed in must appear as-is in the output dict."""
        detector = PoisoningDetector()
        for threshold in (0.3, 0.5, 0.7, 0.9):
            report = detector.evaluate_detection(
                augmented_pool, self._INJECTED_NAMES, threshold=threshold
            )
            assert report["threshold"] == threshold

    def test_threshold_zero_flags_everyone(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """threshold=0.0 must flag every annotator (all scores ≥ 0.0)."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(
            augmented_pool, self._INJECTED_NAMES, threshold=0.0
        )
        # All injected annotators are flagged → recall = 1.0, FN = 0.
        assert report["false_negatives"] == 0
        assert report["recall"] == pytest.approx(1.0)

    def test_threshold_one_flags_only_top_scorer(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """threshold=1.0 must flag at most one annotator (the top scorer)."""
        detector = PoisoningDetector()
        report = detector.evaluate_detection(
            augmented_pool, self._INJECTED_NAMES, threshold=1.0
        )
        flagged_count = report["true_positives"] + report["false_positives"]
        assert flagged_count <= 1

    def test_invalid_threshold_raises(
        self,
        augmented_pool: list[dict[str, Any]],
    ) -> None:
        """threshold outside [0.0, 1.0] must raise ValueError."""
        detector = PoisoningDetector()
        with pytest.raises(ValueError, match="threshold must be in"):
            detector.evaluate_detection(
                augmented_pool, self._INJECTED_NAMES, threshold=1.5
            )

    def test_empty_annotations_raises(self) -> None:
        """evaluate_detection must raise ValueError on an empty annotation list."""
        detector = PoisoningDetector()
        with pytest.raises(ValueError, match="must not be empty"):
            detector.evaluate_detection([], ["Poisoner_A"])


# ---------------------------------------------------------------------------
# 5. cleanlab_label_quality — integration + skip guard
# ---------------------------------------------------------------------------


class TestCleanlabLabelQuality:
    """Cleanlab integration tests — skipped when cleanlab is not installed."""

    @pytest.fixture(autouse=True)
    def require_cleanlab(self) -> None:
        pytest.importorskip("cleanlab", reason="cleanlab not installed")

    def test_output_has_required_keys(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Output dict must contain all five required keys."""
        detector = PoisoningDetector()
        result = detector.cleanlab_label_quality(
            real_annotations, dimension="privacy_compliance"
        )
        required = {
            "dimension",
            "n_logs",
            "n_issues_found",
            "flagged_log_ids",
            "quality_scores",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_dimension_echoed_in_output(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """The dimension name must be echoed back in the output dict."""
        detector = PoisoningDetector()
        for dim in RUBRIC_DIMENSIONS:
            result = detector.cleanlab_label_quality(real_annotations, dimension=dim)
            assert result["dimension"] == dim

    def test_n_logs_matches_unique_log_ids(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """n_logs must equal the count of unique log_ids in the annotation pool."""
        detector = PoisoningDetector()
        result = detector.cleanlab_label_quality(
            real_annotations, dimension="step_quality"
        )
        expected_n_logs = len({r["log_id"] for r in real_annotations})
        assert result["n_logs"] == expected_n_logs

    def test_n_issues_found_matches_flagged_list(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """n_issues_found must equal len(flagged_log_ids)."""
        detector = PoisoningDetector()
        result = detector.cleanlab_label_quality(
            real_annotations, dimension="goal_alignment"
        )
        assert result["n_issues_found"] == len(result["flagged_log_ids"])

    def test_quality_scores_cover_all_logs(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """quality_scores must contain one entry per unique log_id."""
        detector = PoisoningDetector()
        result = detector.cleanlab_label_quality(
            real_annotations, dimension="error_recovery"
        )
        all_log_ids = {r["log_id"] for r in real_annotations}
        assert set(result["quality_scores"].keys()) == all_log_ids

    def test_quality_scores_are_in_unit_interval(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """Every quality score must be a finite float in [0.0, 1.0]."""
        detector = PoisoningDetector()
        result = detector.cleanlab_label_quality(
            real_annotations, dimension="privacy_compliance"
        )
        for log_id, score in result["quality_scores"].items():
            assert isinstance(score, float), f"{log_id} score is not float"
            assert math.isfinite(score), f"{log_id} score is not finite"
            assert 0.0 <= score <= 1.0, (
                f"{log_id} quality score {score:.4f} outside [0.0, 1.0]"
            )

    def test_flagged_log_ids_are_subset_of_all_logs(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """flagged_log_ids must be a subset of the log_ids present in the pool."""
        detector = PoisoningDetector()
        result = detector.cleanlab_label_quality(
            real_annotations, dimension="step_quality"
        )
        all_log_ids = {r["log_id"] for r in real_annotations}
        assert set(result["flagged_log_ids"]).issubset(all_log_ids)

    def test_invalid_dimension_raises(
        self, real_annotations: list[dict[str, Any]]
    ) -> None:
        """An unknown dimension name must raise ValueError."""
        detector = PoisoningDetector()
        with pytest.raises(ValueError, match="dimension must be one of"):
            detector.cleanlab_label_quality(
                real_annotations, dimension="nonexistent_field"
            )

    def test_empty_annotations_raises(self) -> None:
        """cleanlab_label_quality must raise ValueError on empty input."""
        detector = PoisoningDetector()
        with pytest.raises(ValueError, match="must not be empty"):
            detector.cleanlab_label_quality([], dimension="step_quality")
