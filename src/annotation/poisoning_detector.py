"""Annotator poisoning and outlier detection module.

Implements detection heuristics inspired by Anthropic's 250-doc backdoor
finding (Oct 2025): only 250 malicious documents needed to backdoor models
of any size; attack success is count-based, not proportion-based.

Applied to annotation pipelines: a small number of compromised annotators
can systematically bias training data without triggering proportion-based
quality gates.  The analogous threat model here is a persona (or real
annotator) that applies a consistent, directional bias — e.g. always
suppressing privacy_compliance scores by 1 point while inflating
step_quality — rather than random noise.

Detection layers implemented:

1. **Deviation-based outlier scoring** (``detect_outlier_annotators``):
   Computes each annotator's mean absolute deviation (MAD) from the panel
   consensus across all four rubric dimensions.  Annotators whose MAD is
   furthest from the group centroid receive suspicion scores approaching 1.0.
   This mirrors the perplexity-differential signal from Anthropic's backdoor
   detection: triggered vs. non-triggered pattern divergence, adapted from
   per-document to per-annotator granularity.

2. **Synthetic poisoner injection** (``inject_synthetic_poisoners``):
   Injects ``n_malicious`` fake annotators with a deterministic directional
   bias for controlled detection experiments.  The bias pattern — suppress
   ``privacy_compliance`` by 1, inflate ``step_quality`` by 1 — is a
   realistic threat model for an annotator trying to produce training data
   that de-emphasises privacy constraints.

3. **Detection evaluation** (``evaluate_detection``):
   Measures the precision/recall/F1 of the deviation-based detector at a
   configurable suspicion-score threshold, using injected ground-truth labels.

4. **Cleanlab label quality** (``cleanlab_label_quality``):
   Applies Confident Learning (Northcutt et al., 2021) to identify individual
   annotation records whose given label is likely erroneous, using the
   distribution of annotator ratings per log as a proxy for predicted class
   probabilities.

CLI::

    python -m src.annotation.poisoning_detector \\
        --input  data/annotations/day12_annotations.jsonl \\
        --n-malicious 3 \\
        --threshold 0.6 \\
        --dimension privacy_compliance
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import typer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The four rubric dimensions present in every annotation record.
RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "step_quality",
    "privacy_compliance",
    "goal_alignment",
    "error_recovery",
)

#: 1-indexed label range for all rubric dimensions.
LABEL_MIN: int = 1
LABEL_MAX: int = 4
N_CLASSES: int = LABEL_MAX - LABEL_MIN + 1  # 4

#: Names used for injected synthetic poisoners.
_POISONER_NAMES: tuple[str, ...] = ("Poisoner_A", "Poisoner_B", "Poisoner_C")

#: Small epsilon added to denominators to prevent zero-division.
_EPS: float = 1e-9

#: Softmax temperature for pred_probs estimation from annotator agreement.
#: Lower → sharper; higher → smoother.  1.0 keeps raw vote proportions.
_PRED_PROB_TEMPERATURE: float = 1.0

#: Laplace smoothing applied to raw vote counts before normalisation.
_LAPLACE_ALPHA: float = 0.1


# ---------------------------------------------------------------------------
# PoisoningDetector
# ---------------------------------------------------------------------------


class PoisoningDetector:
    """Detect systematically biased annotators in a multi-rater annotation pool.

    Implements four methods that together provide a full detection pipeline:

    1. :meth:`detect_outlier_annotators` — per-persona suspicion score
       based on mean absolute deviation from panel consensus.
    2. :meth:`inject_synthetic_poisoners` — controlled ground-truth
       injection for evaluation purposes.
    3. :meth:`evaluate_detection` — precision/recall/F1 at a configurable
       suspicion threshold.
    4. :meth:`cleanlab_label_quality` — Confident Learning label issue
       detection for a single rubric dimension.

    All methods are stateless and accept annotation records as plain dicts
    matching the ``day12_annotations.jsonl`` schema.

    Example:
        >>> detector = PoisoningDetector()
        >>> records = [...]  # loaded from day12_annotations.jsonl
        >>> scores = detector.detect_outlier_annotators(records)
        >>> augmented = detector.inject_synthetic_poisoners(records, n_malicious=3)
        >>> report = detector.evaluate_detection(
        ...     augmented, ["Poisoner_A", "Poisoner_B", "Poisoner_C"]
        ... )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_outlier_annotators(
        self,
        annotations: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute a suspicion score for each annotator in the pool.

        For every ``log_id``, the panel consensus is the mean rating per
        rubric dimension across all annotators who rated that log.  Each
        annotator's deviation from that consensus is measured as the mean
        absolute deviation (MAD) across all four dimensions and all logs
        they rated.  The raw MAD values are then min-max normalised across
        the annotator pool so that the highest-deviating annotator receives
        a score approaching 1.0.

        This mirrors Anthropic's backdoor perplexity-differential signal:
        triggered (malicious) annotation patterns diverge from the non-triggered
        (legitimate) baseline in a detectable, directional way.

        Args:
            annotations: Flat list of annotation record dicts.  Each record
                must contain ``"persona_name"``, ``"log_id"``, and the four
                rubric dimension keys (``"step_quality"``, ``"privacy_compliance"``,
                ``"goal_alignment"``, ``"error_recovery"``).

        Returns:
            A dict mapping each ``persona_name`` to a suspicion score in
            ``[0.0, 1.0]``.  A score near 1.0 indicates the annotator's
            ratings deviate most from the panel consensus; 0.0 indicates the
            least deviant annotator.  When fewer than 2 annotators are present,
            all scores are 0.0 (no reference panel).

        Raises:
            ValueError: If ``annotations`` is empty.

        Example:
            >>> detector = PoisoningDetector()
            >>> scores = detector.detect_outlier_annotators(records)
            >>> sorted(scores.items(), key=lambda x: -x[1])[0]  # highest suspect
            ('Poisoner_A', 0.9...)
        """
        if not annotations:
            raise ValueError("annotations must not be empty.")

        # Step 1: Group ratings by (log_id, persona_name).
        # ratings[log_id][persona_name][dimension] = score
        ratings: dict[str, dict[str, dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for record in annotations:
            log_id: str = record["log_id"]
            persona: str = record["persona_name"]
            for dim in RUBRIC_DIMENSIONS:
                if dim in record:
                    ratings[log_id][persona][dim] = float(record[dim])

        all_personas: set[str] = {r["persona_name"] for r in annotations}

        if len(all_personas) < 2:
            logger.warning(
                "detect_outlier_annotators: fewer than 2 annotators — "
                "cannot compute consensus deviation; returning 0.0 for all."
            )
            return {p: 0.0 for p in all_personas}

        # Step 2: For each log_id, compute panel mean per dimension.
        # consensus[log_id][dimension] = mean across all annotators who rated it.
        consensus: dict[str, dict[str, float]] = {}
        for log_id, persona_ratings in ratings.items():
            dim_values: dict[str, list[float]] = defaultdict(list)
            for _persona, dim_scores in persona_ratings.items():
                for dim, score in dim_scores.items():
                    dim_values[dim].append(score)
            consensus[log_id] = {
                dim: float(np.mean(scores))
                for dim, scores in dim_values.items()
                if scores
            }

        # Step 3: Compute each annotator's mean absolute deviation across
        # all logs and dimensions.
        persona_mad: dict[str, float] = {}
        for persona in all_personas:
            deviations: list[float] = []
            for log_id, persona_ratings in ratings.items():
                if persona not in persona_ratings:
                    continue
                panel_mean = consensus.get(log_id, {})
                for dim, score in persona_ratings[persona].items():
                    if dim in panel_mean:
                        deviations.append(abs(score - panel_mean[dim]))
            persona_mad[persona] = float(np.mean(deviations)) if deviations else 0.0
            logger.debug(
                "detect_outlier_annotators | persona=%s mad=%.4f",
                persona,
                persona_mad[persona],
            )

        # Step 4: Min-max normalise to [0.0, 1.0].
        mad_values = list(persona_mad.values())
        mad_min = min(mad_values)
        mad_max = max(mad_values)
        mad_range = mad_max - mad_min

        suspicion_scores: dict[str, float] = {}
        for persona, mad in persona_mad.items():
            if mad_range < _EPS:
                # All annotators deviate equally — no discriminative signal.
                suspicion_scores[persona] = 0.0
            else:
                suspicion_scores[persona] = (mad - mad_min) / mad_range

        logger.info(
            "detect_outlier_annotators | n_annotators=%d n_logs=%d "
            "max_suspicion=%.4f (persona=%s)",
            len(all_personas),
            len(ratings),
            max(suspicion_scores.values()),
            max(suspicion_scores, key=lambda k: suspicion_scores[k]),
        )
        return suspicion_scores

    def inject_synthetic_poisoners(
        self,
        annotations: list[dict[str, Any]],
        n_malicious: int = 3,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Augment the annotation pool with synthetic malicious annotators.

        Each injected poisoner applies a systematic directional bias to
        every log in the pool:

        - ``privacy_compliance``: decreased by 1 from consensus (floor: 1).
        - ``step_quality``: increased by 1 from consensus (ceiling: 4).

        All other dimensions are set to the rounded panel consensus.  This
        models a realistic adversary who wants to produce training data that
        de-emphasises privacy constraints while inflating perceived step
        quality — a pattern consistent with Anthropic's 250-doc backdoor
        finding where the attack is directional and count-based.

        Args:
            annotations: Original annotation pool.  Must be non-empty.
            n_malicious: Number of synthetic poisoners to inject.  Names are
                drawn from ``("Poisoner_A", "Poisoner_B", "Poisoner_C")``; at
                most 3 are supported.  Defaults to 3.
            seed: Random seed for any stochastic tie-breaking.  Defaults to 42.
                In the current deterministic implementation this seed is stored
                but not consumed; it is reserved for future extensions that
                add noise on top of the directional bias.

        Returns:
            A new list containing all original records followed by one record
            per injected poisoner per log.  Each synthetic record includes an
            ``"is_injected_poisoner": True`` field for ground-truth evaluation.
            Original records are not mutated; they do **not** gain the
            ``"is_injected_poisoner"`` key.

        Raises:
            ValueError: If ``annotations`` is empty or ``n_malicious`` > 3.

        Example:
            >>> augmented = detector.inject_synthetic_poisoners(records, n_malicious=2)
            >>> poisoner_records = [
            ...     r for r in augmented if r.get("is_injected_poisoner")
            ... ]
            >>> len(poisoner_records) == 2 * n_unique_logs
            True
        """
        if not annotations:
            raise ValueError("annotations must not be empty.")
        if n_malicious > len(_POISONER_NAMES):
            raise ValueError(
                f"n_malicious must be ≤ {len(_POISONER_NAMES)}; got {n_malicious}."
            )

        rng = np.random.default_rng(seed)  # reserved for future noise layer
        _ = rng  # suppress unused-variable linting

        # Build consensus per (log_id, dimension) from the original pool.
        dim_sums: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in annotations:
            log_id = record["log_id"]
            for dim in RUBRIC_DIMENSIONS:
                if dim in record:
                    dim_sums[log_id][dim].append(float(record[dim]))

        consensus: dict[str, dict[str, float]] = {
            log_id: {dim: float(np.mean(scores)) for dim, scores in dim_vals.items()}
            for log_id, dim_vals in dim_sums.items()
        }

        # Build a representative template record per log_id from the first
        # annotation seen, so we can inherit metadata fields.
        template_by_log: dict[str, dict[str, Any]] = {}
        for record in annotations:
            log_id = record["log_id"]
            if log_id not in template_by_log:
                template_by_log[log_id] = record

        poisoner_names = _POISONER_NAMES[:n_malicious]
        synthetic_records: list[dict[str, Any]] = []

        for poisoner_name in poisoner_names:
            for log_id, panel_mean in consensus.items():
                template = template_by_log[log_id]

                # Apply directional bias: suppress privacy, inflate step_quality.
                step_quality_raw = panel_mean.get("step_quality", 2.5)
                privacy_raw = panel_mean.get("privacy_compliance", 2.5)
                goal_raw = panel_mean.get("goal_alignment", 2.5)
                recovery_raw = panel_mean.get("error_recovery", 2.5)

                step_quality = int(min(LABEL_MAX, round(step_quality_raw) + 1))
                privacy_compliance = int(max(LABEL_MIN, round(privacy_raw) - 1))
                goal_alignment = int(max(LABEL_MIN, min(LABEL_MAX, round(goal_raw))))
                error_recovery = int(
                    max(LABEL_MIN, min(LABEL_MAX, round(recovery_raw)))
                )

                synthetic_records.append(
                    {
                        "annotation_id": str(uuid.uuid4()),
                        "log_id": log_id,
                        "persona_name": poisoner_name,
                        "scenario_type": template.get("scenario_type", ""),
                        "consent_model": template.get("consent_model", ""),
                        "ground_truth_action": template.get("ground_truth_action", ""),
                        "step_quality": step_quality,
                        "privacy_compliance": privacy_compliance,
                        "goal_alignment": goal_alignment,
                        "error_recovery": error_recovery,
                        "rationale": (
                            f"[SYNTHETIC-POISONER] {poisoner_name}: "
                            f"inflated step_quality by +1, suppressed "
                            f"privacy_compliance by -1 from panel consensus."
                        ),
                        "created_at": template.get("created_at", ""),
                        "is_injected_poisoner": True,
                    }
                )

        augmented = list(annotations) + synthetic_records
        logger.info(
            "inject_synthetic_poisoners | original=%d injected=%d total=%d",
            len(annotations),
            len(synthetic_records),
            len(augmented),
        )
        return augmented

    def evaluate_detection(
        self,
        annotations: list[dict[str, Any]],
        injected_names: list[str],
        threshold: float = 0.6,
    ) -> dict[str, Any]:
        """Evaluate the deviation-based detector against injected ground truth.

        Runs :meth:`detect_outlier_annotators` on the augmented pool and
        computes standard information-retrieval metrics at the given suspicion
        score threshold.

        An annotator is classified as **poisoned** if their suspicion score
        meets or exceeds ``threshold``.  True positives are names that appear
        in both ``injected_names`` and the set of flagged annotators.

        Args:
            annotations: Augmented pool (original + injected records).
                Typically the output of :meth:`inject_synthetic_poisoners`.
            injected_names: Ground-truth list of injected annotator names (e.g.
                ``["Poisoner_A", "Poisoner_B", "Poisoner_C"]``).
            threshold: Suspicion score cut-off for classifying an annotator as
                poisoned.  Defaults to 0.6.

        Returns:
            A dict with the following keys:

            - ``"threshold"`` (float): The threshold used.
            - ``"true_positives"`` (int): Injected annotators correctly flagged.
            - ``"false_positives"`` (int): Legitimate annotators incorrectly flagged.
            - ``"false_negatives"`` (int): Injected annotators not flagged.
            - ``"precision"`` (float): TP / (TP + FP). 0.0 when no positives predicted.
            - ``"recall"`` (float): TP / (TP + FN). 0.0 when no ground-truth positives.
            - ``"f1"`` (float): Harmonic mean of precision and recall.
            - ``"per_annotator_scores"`` (dict[str, float]): Raw suspicion scores
              from :meth:`detect_outlier_annotators`.

        Raises:
            ValueError: If ``annotations`` is empty or ``threshold`` is outside
                ``[0.0, 1.0]``.

        Example:
            >>> report = detector.evaluate_detection(augmented, ["Poisoner_A"])
            >>> report["f1"] > 0.5
            True
        """
        if not annotations:
            raise ValueError("annotations must not be empty.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0]; got {threshold}.")

        per_annotator_scores = self.detect_outlier_annotators(annotations)

        injected_set = set(injected_names)
        flagged_set = {
            persona
            for persona, score in per_annotator_scores.items()
            if score >= threshold
        }

        tp = len(flagged_set & injected_set)
        fp = len(flagged_set - injected_set)
        fn = len(injected_set - flagged_set)

        precision = tp / (tp + fp + _EPS) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + _EPS) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall + _EPS)
            if (precision + recall) > 0
            else 0.0
        )

        # Strip the epsilon additions from reported metrics so values read cleanly.
        precision = round(precision, 6)
        recall = round(recall, 6)
        f1 = round(f1, 6)

        logger.info(
            "evaluate_detection | threshold=%.2f TP=%d FP=%d FN=%d "
            "precision=%.4f recall=%.4f f1=%.4f",
            threshold,
            tp,
            fp,
            fn,
            precision,
            recall,
            f1,
        )
        return {
            "threshold": threshold,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_annotator_scores": per_annotator_scores,
        }

    def cleanlab_label_quality(
        self,
        annotations: list[dict[str, Any]],
        dimension: str = "privacy_compliance",
    ) -> dict[str, Any]:
        """Identify annotation records with likely label errors via Confident Learning.

        Adapts Cleanlab's ``find_label_issues`` (Northcutt et al., 2021) to the
        multi-rater annotation setting.  For each ``log_id``, the distribution of
        annotator ratings on ``dimension`` is used as a proxy for predicted class
        probabilities (``pred_probs``).  The majority-vote rating is used as the
        "given label".  Cleanlab then flags log-level label issues where the
        observed label distribution conflicts with the majority signal — a direct
        indicator of annotation contention.

        This is a creative adaptation: Cleanlab is designed for ML classifier
        outputs, not multi-rater surveys.  The key mapping is:

        - Each ``log_id`` is a data point.
        - The majority-vote annotation is the "given label".
        - The normalised rating count distribution per log is ``pred_probs``.
        - Cleanlab flags log_ids where the given label conflicts with the
          predicted probability distribution — exactly the logs where annotators
          disagree most.

        Args:
            annotations: Flat list of annotation records.  All records must
                contain ``"log_id"``, ``"persona_name"``, and the ``dimension``
                key with an integer value in ``[1, 4]``.
            dimension: Which rubric dimension to analyse.  Must be one of
                ``"step_quality"``, ``"privacy_compliance"``,
                ``"goal_alignment"``, ``"error_recovery"``.  Defaults to
                ``"privacy_compliance"``.

        Returns:
            A dict with the following keys:

            - ``"dimension"`` (str): The analysed dimension.
            - ``"n_logs"`` (int): Number of unique log_ids analysed.
            - ``"n_issues_found"`` (int): Number of logs flagged by Cleanlab.
            - ``"flagged_log_ids"`` (list[str]): log_id values flagged as
              having label quality issues.
            - ``"quality_scores"`` (dict[str, float]): Per-log quality score
              in ``[0.0, 1.0]``; lower means more likely a label issue.

        Raises:
            ValueError: If ``annotations`` is empty or ``dimension`` is not
                one of the four rubric dimensions.
            ImportError: If the ``cleanlab`` package is not installed.

        Example:
            >>> result = detector.cleanlab_label_quality(records, "privacy_compliance")
            >>> result["n_issues_found"]
            3
        """
        try:
            from cleanlab.filter import find_label_issues
            from cleanlab.rank import get_label_quality_scores
        except ImportError as exc:
            raise ImportError(
                "cleanlab is required for cleanlab_label_quality. "
                "Install it with: uv add cleanlab"
            ) from exc

        if not annotations:
            raise ValueError("annotations must not be empty.")
        if dimension not in RUBRIC_DIMENSIONS:
            raise ValueError(
                f"dimension must be one of {RUBRIC_DIMENSIONS}; got {repr(dimension)}."
            )

        # Pivot: vote_counts[log_id][label_0indexed] = count of annotations
        # Labels are 1-4; map to 0-indexed 0-3 for cleanlab.
        vote_counts: dict[str, list[int]] = {}
        for record in annotations:
            log_id = record["log_id"]
            raw_label = record.get(dimension)
            if raw_label is None:
                continue
            if log_id not in vote_counts:
                vote_counts[log_id] = [0] * N_CLASSES
            label_idx = int(raw_label) - LABEL_MIN  # 0-indexed
            label_idx = max(0, min(N_CLASSES - 1, label_idx))
            vote_counts[log_id][label_idx] += 1

        if len(vote_counts) < 2:
            logger.warning(
                "cleanlab_label_quality: fewer than 2 log_ids — returning empty result."
            )
            return {
                "dimension": dimension,
                "n_logs": len(vote_counts),
                "n_issues_found": 0,
                "flagged_log_ids": [],
                "quality_scores": {},
            }

        log_ids_ordered = sorted(vote_counts.keys())
        n_logs = len(log_ids_ordered)

        # Build pred_probs (N, K) via Laplace-smoothed normalisation.
        # This converts vote counts into a probability distribution over labels.
        pred_probs = np.zeros((n_logs, N_CLASSES), dtype=np.float64)
        given_labels = np.zeros(n_logs, dtype=np.int64)

        for i, log_id in enumerate(log_ids_ordered):
            counts = np.array(vote_counts[log_id], dtype=np.float64)
            smoothed = counts + _LAPLACE_ALPHA
            pred_probs[i] = smoothed / smoothed.sum()
            # Given label = majority vote (argmax of raw counts), 0-indexed.
            given_labels[i] = int(np.argmax(counts))

        logger.debug(
            "cleanlab_label_quality | dimension=%s n_logs=%d n_classes=%d",
            dimension,
            n_logs,
            N_CLASSES,
        )

        # Run Cleanlab's Confident Learning detector.
        try:
            issue_mask: np.ndarray = find_label_issues(
                labels=given_labels,
                pred_probs=pred_probs,
                return_indices_ranked_by=None,
                filter_by="prune_by_noise_rate",
                n_jobs=1,
                verbose=False,
            )
            quality_scores_arr: np.ndarray = get_label_quality_scores(
                labels=given_labels,
                pred_probs=pred_probs,
                method="self_confidence",
            )
        except Exception as exc:
            logger.error("cleanlab_label_quality | cleanlab raised an error: %s", exc)
            raise

        flagged_indices: list[int] = [int(i) for i in np.where(issue_mask)[0]]
        flagged_log_ids: list[str] = [log_ids_ordered[i] for i in flagged_indices]
        quality_scores: dict[str, float] = {
            log_id: float(quality_scores_arr[i])
            for i, log_id in enumerate(log_ids_ordered)
        }

        n_issues = len(flagged_log_ids)
        logger.info(
            "cleanlab_label_quality | dimension=%s n_logs=%d n_issues=%d flagged=%s",
            dimension,
            n_logs,
            n_issues,
            flagged_log_ids[:5],
        )
        return {
            "dimension": dimension,
            "n_logs": n_logs,
            "n_issues_found": n_issues,
            "flagged_log_ids": flagged_log_ids,
            "quality_scores": quality_scores,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="poisoning-detector",
    help="Detect malicious annotators and label quality issues in annotation pools.",
    add_completion=False,
)


@app.command()
def detect(
    input: Path = typer.Option(
        Path("data/annotations/day12_annotations.jsonl"),
        "--input",
        "-i",
        help="Path to annotation JSONL file.",
    ),
    n_malicious: int = typer.Option(
        3,
        "--n-malicious",
        help="Number of synthetic poisoners to inject for evaluation.",
    ),
    threshold: float = typer.Option(
        0.6,
        "--threshold",
        "-t",
        help="Suspicion score threshold for classifying an annotator as poisoned.",
    ),
    dimension: str = typer.Option(
        "privacy_compliance",
        "--dimension",
        "-d",
        help="Rubric dimension for cleanlab label quality analysis.",
    ),
    output: Path = typer.Option(
        Path("data/annotations/poisoning_report.json"),
        "--output",
        "-o",
        help="Path for JSON output report.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable DEBUG logging."
    ),
) -> None:
    """Run the full poisoning detection pipeline.

    Steps:
      1. Load annotation pool from --input.
      2. Run outlier detection on the original pool.
      3. Inject --n-malicious synthetic poisoners.
      4. Evaluate detection precision/recall/F1 at --threshold.
      5. Run cleanlab label quality on --dimension.
      6. Write JSON report to --output.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not input.exists():
        typer.echo(f"Input file not found: {input}", err=True)
        raise typer.Exit(code=1)

    annotations: list[dict[str, Any]] = []
    with input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))

    typer.echo(f"Loaded {len(annotations)} annotation records from {input}")

    detector = PoisoningDetector()

    # Step 1: Outlier detection on original pool.
    typer.echo("\n[1/4] Running outlier detection on original pool …")
    original_scores = detector.detect_outlier_annotators(annotations)

    # Step 2: Inject synthetic poisoners.
    typer.echo(f"\n[2/4] Injecting {n_malicious} synthetic poisoner(s) …")
    injected_names = list(_POISONER_NAMES[:n_malicious])
    augmented = detector.inject_synthetic_poisoners(
        annotations, n_malicious=n_malicious
    )
    typer.echo(f"      Augmented pool: {len(augmented)} records.")

    # Step 3: Evaluate detection.
    typer.echo(f"\n[3/4] Evaluating detection at threshold={threshold} …")
    eval_report = detector.evaluate_detection(
        augmented, injected_names=injected_names, threshold=threshold
    )

    # Step 4: Cleanlab label quality.
    typer.echo(f"\n[4/4] Running cleanlab label quality on '{dimension}' …")
    lq_report = detector.cleanlab_label_quality(annotations, dimension=dimension)

    # Assemble full report.
    report: dict[str, Any] = {
        "input_file": str(input),
        "n_original_records": len(annotations),
        "n_augmented_records": len(augmented),
        "n_malicious_injected": n_malicious,
        "original_suspicion_scores": original_scores,
        "detection_evaluation": eval_report,
        "cleanlab_label_quality": lq_report,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    typer.echo(f"\nReport written to {output}")

    # Rich summary.
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        scores_table = Table(
            title="Annotator Suspicion Scores (original pool)",
            show_header=True,
            header_style="bold cyan",
        )
        scores_table.add_column("Annotator", style="bold")
        scores_table.add_column("Suspicion Score", justify="right")
        scores_table.add_column("Flagged (≥ threshold)", justify="center")

        for persona, score in sorted(original_scores.items(), key=lambda x: -x[1]):
            flagged = score >= threshold
            flag_str = "[red]YES[/red]" if flagged else "[green]no[/green]"
            scores_table.add_row(persona, f"{score:.4f}", flag_str)

        console.print()
        console.print(scores_table)

        eval_table = Table(
            title=f"Detection Evaluation (threshold={threshold})",
            show_header=True,
            header_style="bold cyan",
        )
        eval_table.add_column("Metric", style="bold")
        eval_table.add_column("Value", justify="right")
        eval_table.add_row("True Positives", str(eval_report["true_positives"]))
        eval_table.add_row("False Positives", str(eval_report["false_positives"]))
        eval_table.add_row("False Negatives", str(eval_report["false_negatives"]))
        eval_table.add_row("Precision", f"{eval_report['precision']:.4f}")
        eval_table.add_row("Recall", f"{eval_report['recall']:.4f}")
        eval_table.add_row("F1", f"{eval_report['f1']:.4f}")
        console.print()
        console.print(eval_table)

        console.print()
        console.print(
            f"[bold]Cleanlab ({dimension}):[/bold] "
            f"{lq_report['n_issues_found']} / {lq_report['n_logs']} "
            f"logs flagged as label quality issues."
        )
        console.print()

    except ImportError:
        typer.echo("\n=== Annotator Suspicion Scores (original pool) ===")
        for persona, score in sorted(original_scores.items(), key=lambda x: -x[1]):
            flag = " ← FLAGGED" if score >= threshold else ""
            typer.echo(f"  {persona}: {score:.4f}{flag}")
        typer.echo(
            f"\nDetection F1={eval_report['f1']:.4f}  "
            f"P={eval_report['precision']:.4f}  "
            f"R={eval_report['recall']:.4f}"
        )
        typer.echo(
            f"Cleanlab ({dimension}): "
            f"{lq_report['n_issues_found']}/{lq_report['n_logs']} logs flagged."
        )


if __name__ == "__main__":
    app()
