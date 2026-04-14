"""Inter-Rater Reliability (IRR) calculator for annotation quality.

USE CASES BY METRIC:
  Cohen's κ   — 2 raters, nominal labels. Best for pairwise QA review.
                Corrects for chance agreement. Range: [-1, 1].
                Use when: validating two human annotators on same batch.

  Fleiss' κ   — 3+ raters, fixed category set, nominal labels.
                Extends Cohen's to multi-rater pools.
                Use when: crowdsourced annotation (e.g. MTurk) with
                consistent label schema.

  Krippendorff's α — Any # raters, any measurement level
                (nominal/ordinal/interval/ratio). Handles missing data.
                The most general metric — preferred for mixed-mode
                annotation or when raters annotate different subsets.
                Use when: HH-RLHF analysis, multi-annotator pipelines
                where not every rater sees every item.

  BERTScore Agreement — Semantic similarity between free-text rationales.
                Uses contextual embeddings (default: roberta-large).
                Measures whether two annotators MEAN the same thing even
                if they word it differently.
                Use when: annotators write justifications, not just labels.

STRATEGIC CONTEXT:
  Cohere Command A (2504.00698): 800 prompts, 65 annotators, 5-point
  scale — zero agreement statistics reported. This module fills that gap.
  Target threshold for annotation quality: κ > 0.8 (Landis & Koch).

CLI:
  python -m src.annotation.irr_calculator --metric cohens_kappa \\
      --rater1 data/annotations/r1.json \\
      --rater2 data/annotations/r2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import krippendorff
import numpy as np
from bert_score import score as bert_score_fn
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Landis & Koch (1977) interpretation scale — shared by all κ/α metrics.
# ---------------------------------------------------------------------------


def _interpret_kappa(kappa: float) -> str:
    """Return a Landis & Koch (1977) qualitative label for a kappa/alpha value.

    Boundaries are upper-inclusive: 0.20 → "slight", 0.21 → "fair", etc.

    Args:
        kappa: Observed kappa or alpha value.

    Returns:
        One of: "poor", "slight", "fair", "moderate", "substantial",
        "almost perfect".
    """
    if kappa < 0:
        return "poor"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class IRRCalculator:
    """Inter-Rater Reliability calculator for annotation quality assessment.

    Supports Cohen's κ (2 raters), Fleiss' κ (multi-rater), Krippendorff's α
    (any raters, any measurement level), and BERTScore semantic agreement.

    Each method returns a plain dict so results can be serialised to JSON or
    logged directly without further conversion.

    Note:
        BERTScore measures semantic agreement between FREE-TEXT rationales.
        It answers "do these two annotators mean the same thing?"
        Cohen's/Fleiss/Krippendorff answer "do they pick the same label?"
        Use BERTScore when your annotation schema requires written justifications
        (e.g. wearable agent trajectory annotation with per-step rationale).

    Example:
        >>> calc = IRRCalculator()
        >>> result = calc.cohens_kappa([1, 2, 3, 1, 2], [1, 2, 3, 2, 2])
        >>> result["kappa"]
        0.6666...
    """

    # ------------------------------------------------------------------
    # Cohen's κ
    # ------------------------------------------------------------------

    def cohens_kappa(
        self,
        rater1: list[int],
        rater2: list[int],
    ) -> dict[str, object]:
        """Compute Cohen's kappa for two raters on a nominal label set.

        Cohen's κ corrects for the agreement expected by chance:
            κ = (p_o − p_e) / (1 − p_e)
        where p_o is observed agreement and p_e is expected chance agreement
        derived from each rater's marginal label distribution.

        Args:
            rater1: Ordered list of integer category labels from annotator 1.
            rater2: Ordered list of integer category labels from annotator 2.
                Must be the same length as rater1 and share the same label
                vocabulary.

        Returns:
            A dict with the following keys:

            - ``"kappa"`` (float): Cohen's κ in [-1, 1].
            - ``"interpretation"`` (str): Landis & Koch qualitative label.
            - ``"n_items"`` (int): Number of rated items.

        Raises:
            ValueError: If either list is empty, the lists have different
                lengths, or fewer than 2 items are provided.

        Example:
            >>> calc = IRRCalculator()
            >>> calc.cohens_kappa([1, 1, 2, 2], [1, 2, 2, 2])
            {'kappa': 0.5, 'interpretation': 'moderate', 'n_items': 4}
        """
        if not rater1 or not rater2:
            raise ValueError("rater1 and rater2 must not be empty.")
        if len(rater1) != len(rater2):
            raise ValueError(
                f"rater1 and rater2 must have the same length; "
                f"got {len(rater1)} vs {len(rater2)}."
            )
        if len(rater1) < 2:
            raise ValueError(
                "At least 2 items are required to compute Cohen's κ; "
                f"got {len(rater1)}."
            )

        kappa = float(cohen_kappa_score(rater1, rater2))
        interpretation = _interpret_kappa(kappa)

        logger.info(
            "Cohen's κ = %.4f (%s)  [n=%d]",
            kappa,
            interpretation,
            len(rater1),
        )
        return {
            "kappa": kappa,
            "interpretation": interpretation,
            "n_items": len(rater1),
        }

    # ------------------------------------------------------------------
    # Fleiss' κ
    # ------------------------------------------------------------------

    def fleiss_kappa(
        self,
        ratings_matrix: list[list[int]],
        n_categories: int,
    ) -> dict[str, object]:
        """Compute Fleiss' kappa for 3+ raters on a fixed nominal label set.

        Extends Cohen's κ to pools of more than two raters. Each item must
        be rated by the same number of raters (no missing data). Categories
        are 0-indexed integers in the range [0, n_categories − 1].

        Algorithm (Fleiss 1971):
            1. Build count matrix N[i, j] = # raters assigning category j
               to item i.
            2. P_i = proportion of agreeing rater pairs for item i:
                   P_i = (Σ_j N[i,j]² − n) / (n(n−1))
            3. P̄ = mean of all P_i  (observed agreement).
            4. p_j = marginal proportion of all assignments in category j.
            5. P̄_e = Σ_j p_j²  (expected chance agreement).
            6. κ = (P̄ − P̄_e) / (1 − P̄_e).

        Args:
            ratings_matrix: 2-D list of shape [n_items × n_raters]. Each
                cell contains the 0-indexed integer category label assigned
                by that rater to that item. Every row must have the same
                length (balanced design — all raters rate every item).
            n_categories: Total number of distinct categories in the label
                schema. Must be ≥ 2. Categories not present in
                ratings_matrix still count toward the expected-agreement
                denominator.

        Returns:
            A dict with the following keys:

            - ``"kappa"`` (float): Fleiss' κ in [-1, 1].
            - ``"interpretation"`` (str): Landis & Koch qualitative label.
            - ``"n_items"`` (int): Number of rated items (rows).
            - ``"n_raters"`` (int): Number of raters per item (columns).

        Raises:
            ValueError: If the matrix is empty, rows have inconsistent
                lengths, any label is outside [0, n_categories − 1],
                n_categories < 2, or n_raters < 2.

        Example:
            >>> calc = IRRCalculator()
            >>> matrix = [[0, 0, 1], [1, 1, 1], [0, 1, 0]]
            >>> calc.fleiss_kappa(matrix, n_categories=2)
            {'kappa': ..., 'interpretation': ..., 'n_items': 3, 'n_raters': 3}
        """
        if not ratings_matrix:
            raise ValueError("ratings_matrix must not be empty.")
        if n_categories < 2:
            raise ValueError(f"n_categories must be ≥ 2; got {n_categories}.")

        n_items = len(ratings_matrix)
        n_raters = len(ratings_matrix[0])

        if n_raters < 2:
            raise ValueError(f"Fleiss' κ requires ≥ 2 raters per item; got {n_raters}.")

        # Validate consistent row length and label range.
        for i, row in enumerate(ratings_matrix):
            if len(row) != n_raters:
                raise ValueError(
                    f"All rows must have the same length (balanced design); "
                    f"row 0 has {n_raters} raters but row {i} has {len(row)}."
                )
            for label in row:
                if label < 0 or label >= n_categories:
                    raise ValueError(
                        f"Label {label} in row {i} is outside the valid range "
                        f"[0, {n_categories - 1}]."
                    )

        # Step 1: Build count matrix N[i, j] using numpy for efficiency.
        arr = np.array(ratings_matrix, dtype=np.int64)  # shape: (n_items, n_raters)
        count = np.zeros((n_items, n_categories), dtype=np.int64)
        for j in range(n_categories):
            count[:, j] = np.sum(arr == j, axis=1)

        n = n_raters  # shorthand

        # Step 2–3: Per-item observed agreement, then mean.
        # P_i = (Σ_j count[i,j]² − n) / (n(n−1))
        P_i = (np.sum(count**2, axis=1) - n) / (n * (n - 1))
        P_bar = float(np.mean(P_i))

        # Step 4–5: Marginal category proportions and expected agreement.
        p_j = np.sum(count, axis=0) / (n_items * n)  # shape: (n_categories,)
        P_e_bar = float(np.sum(p_j**2))

        # Step 6: Kappa.
        denominator = 1.0 - P_e_bar
        if abs(denominator) < 1e-12:
            # All raters always assign the same category: perfect agreement.
            logger.warning(
                "Denominator (1 − P̄_e) ≈ 0; expected agreement ≈ 1. Returning κ = 1.0."
            )
            kappa = 1.0
        else:
            kappa = (P_bar - P_e_bar) / denominator

        interpretation = _interpret_kappa(kappa)
        logger.info(
            "Fleiss' κ = %.4f (%s)  [n_items=%d, n_raters=%d, n_categories=%d]",
            kappa,
            interpretation,
            n_items,
            n_raters,
            n_categories,
        )
        return {
            "kappa": kappa,
            "interpretation": interpretation,
            "n_items": n_items,
            "n_raters": n_raters,
        }

    # ------------------------------------------------------------------
    # BERTScore agreement
    # ------------------------------------------------------------------

    def bertscore_agreement(
        self,
        rationales_a: list[str],
        rationales_b: list[str],
        model_type: str = "distilbert-base-uncased",
        lang: str = "en",
    ) -> dict[str, object]:
        """Compute semantic agreement between free-text rationales via BERTScore.

        BERTScore uses contextual token embeddings to measure whether two
        annotators *mean the same thing* even when they word it differently.
        This complements label-level kappa metrics: two raters may pick the
        same label for different reasons (high κ, low BERTScore) or write
        near-identical justifications yet choose different labels (low κ,
        high BERTScore).

        Internally delegates to ``bert_score.score``, which returns
        precision, recall, and F1 tensors per rationale pair. Aggregate
        means and per-pair F1 are returned for downstream logging and
        calibration reporting.

        Args:
            rationales_a: Ordered list of free-text justification strings
                from annotator A. Must be the same length as ``rationales_b``.
                No element may be an empty string.
            rationales_b: Ordered list of free-text justification strings
                from annotator B. Must be the same length as ``rationales_a``.
                No element may be an empty string.
            model_type: HuggingFace model identifier used for contextual
                embeddings. Defaults to ``"distilbert-base-uncased"`` (~250 MB,
                fast CPU inference). Pass ``"roberta-large"`` for
                publication-quality results at the cost of ~500 MB download
                and slower inference.
            lang: Language code passed to ``bert_score.score``. Defaults to
                ``"en"``.

        Returns:
            A dict with the following keys:

            - ``"precision_mean"`` (float): Mean BERTScore precision across
              all rationale pairs.
            - ``"recall_mean"`` (float): Mean BERTScore recall across all
              rationale pairs.
            - ``"f1_mean"`` (float): Mean BERTScore F1 across all rationale
              pairs. Primary agreement metric.
            - ``"f1_per_pair"`` (list[float]): Per-pair F1 scores; same
              length as the input lists.
            - ``"model"`` (str): The model identifier used for scoring.
            - ``"n_pairs"`` (int): Number of rationale pairs scored.
            - ``"interpretation"`` (str): Qualitative agreement label:
              ``"low semantic agreement"`` (F1 < 0.70),
              ``"moderate semantic agreement"`` (0.70 ≤ F1 < 0.85), or
              ``"high semantic agreement"`` (F1 ≥ 0.85).

        Raises:
            ValueError: If ``rationales_a`` and ``rationales_b`` have
                different lengths, either list is empty, or any element is
                an empty string.

        Example:
            >>> calc = IRRCalculator()
            >>> a = ["The agent correctly escalated the alert."]
            >>> b = ["Alert was escalated appropriately by the agent."]
            >>> result = calc.bertscore_agreement(a, b)
            >>> result["f1_mean"] > 0.85
            True
        """
        if not rationales_a or not rationales_b:
            raise ValueError("rationales_a and rationales_b must not be empty.")
        if len(rationales_a) != len(rationales_b):
            raise ValueError(
                f"rationales_a and rationales_b must have the same length; "
                f"got {len(rationales_a)} vs {len(rationales_b)}."
            )
        for side, rationales in (
            ("rationales_a", rationales_a),
            ("rationales_b", rationales_b),
        ):
            for i, text in enumerate(rationales):
                if not text:
                    raise ValueError(
                        f"BERTScore requires non-empty rationale strings; "
                        f"{side}[{i}] is empty."
                    )

        n_pairs = len(rationales_a)
        if n_pairs > 1000:
            logger.warning(
                "BERTScore on CPU is slow at scale: scoring %d pairs with %s. "
                "Consider batching or GPU acceleration.",
                n_pairs,
                model_type,
            )

        P, R, F1 = bert_score_fn(
            rationales_a,
            rationales_b,
            model_type=model_type,
            lang=lang,
            verbose=False,
        )

        precision_mean = float(P.mean().item())
        recall_mean = float(R.mean().item())
        f1_mean = float(F1.mean().item())
        f1_per_pair: list[float] = F1.tolist()

        if f1_mean >= 0.85:
            interpretation = "high semantic agreement"
        elif f1_mean >= 0.70:
            interpretation = "moderate semantic agreement"
        else:
            interpretation = "low semantic agreement"

        logger.info(
            "BERTScore agreement: F1=%.4f (%s)  [n_pairs=%d, model=%s]",
            f1_mean,
            interpretation,
            n_pairs,
            model_type,
        )
        return {
            "precision_mean": precision_mean,
            "recall_mean": recall_mean,
            "f1_mean": f1_mean,
            "f1_per_pair": f1_per_pair,
            "model": model_type,
            "n_pairs": n_pairs,
            "interpretation": interpretation,
        }

    # ------------------------------------------------------------------
    # Krippendorff's α
    # ------------------------------------------------------------------

    def krippendorffs_alpha(
        self,
        reliability_data: list[list[float | None]],
        level_of_measurement: str = "ordinal",
    ) -> dict[str, object]:
        """Compute Krippendorff's α for any number of raters on any scale.

        Krippendorff's α is the most general IRR metric: it handles any
        number of raters, any measurement level (nominal, ordinal, interval,
        ratio), and missing data encoded as ``None``. It is the preferred
        primary metric for mixed-mode annotation pipelines such as HH-RLHF
        analysis where not every rater sees every item.

        Internally delegates to the ``krippendorff`` package, which
        implements the exact algorithm from Krippendorff (2011). The input
        array follows the rater-major convention expected by that library:
        rows are raters, columns are items.

        Args:
            reliability_data: 2-D list of shape ``[n_raters × n_items]``.
                Each cell contains a numeric rating or ``None`` for missing.
                Rows represent raters; columns represent items. Every row
                must have the same length. At least 2 raters and 2 items
                are required.
            level_of_measurement: Distance metric applied when computing
                observed and expected disagreement. One of:

                - ``"nominal"``  — categorical labels; order is irrelevant.
                - ``"ordinal"``  — rank-ordered categories; rank differences
                  matter. Default and recommended for 5-point annotation
                  scales (e.g. Cohere Command A methodology).
                - ``"interval"`` — equal-spaced numeric scale.
                - ``"ratio"``    — ratio-scaled numeric data with a true zero.

        Returns:
            A dict with the following keys:

            - ``"alpha"`` (float): Krippendorff's α in [-1, 1]. Values near
              1 indicate high agreement; 0 indicates chance-level agreement.
            - ``"interpretation"`` (str): Landis & Koch qualitative label
              (shared scale with Cohen's/Fleiss' κ for easy comparison).
            - ``"n_raters"`` (int): Number of rater rows.
            - ``"n_items"`` (int): Number of item columns.
            - ``"level_of_measurement"`` (str): The distance metric used.

        Raises:
            ValueError: If ``reliability_data`` is empty, has fewer than
                2 raters, has rows of inconsistent length, or has fewer
                than 2 items.

        Example:
            >>> calc = IRRCalculator()
            >>> data = [[0, 1, 0, 1], [0, 1, 1, 0]]
            >>> calc.krippendorffs_alpha(data)["alpha"]  # ordinal by default
            0.0
        """
        if not reliability_data:
            raise ValueError("reliability_data must not be empty.")
        n_raters = len(reliability_data)
        if n_raters < 2:
            raise ValueError(f"Krippendorff's α requires ≥ 2 raters; got {n_raters}.")
        n_items = len(reliability_data[0])
        if n_items < 2:
            raise ValueError(f"Krippendorff's α requires ≥ 2 items; got {n_items}.")
        for i, row in enumerate(reliability_data):
            if len(row) != n_items:
                raise ValueError(
                    f"All rows must have the same length; "
                    f"row 0 has {n_items} items but row {i} has {len(row)}."
                )
        _valid_levels = {"nominal", "ordinal", "interval", "ratio"}
        if level_of_measurement not in _valid_levels:
            raise ValueError(
                f"level_of_measurement must be one of {sorted(_valid_levels)}; "
                f"got {level_of_measurement!r}."
            )

        missing_values = sum(1 for row in reliability_data for v in row if v is None)

        # krippendorff.alpha expects a numpy array with np.nan for missing.
        arr = np.array(
            [[np.nan if v is None else v for v in row] for row in reliability_data],
            dtype=np.float64,
        )
        alpha = float(
            krippendorff.alpha(
                reliability_data=arr,
                level_of_measurement=level_of_measurement,
            )
        )
        interpretation = _interpret_kappa(alpha)
        logger.info(
            "Krippendorff's α = %.4f (%s)  "
            "[n_raters=%d, n_items=%d, level=%s, missing=%d]",
            alpha,
            interpretation,
            n_raters,
            n_items,
            level_of_measurement,
            missing_values,
        )
        return {
            "alpha": alpha,
            "interpretation": interpretation,
            "n_raters": n_raters,
            "n_items": n_items,
            "missing_values": missing_values,
            "level_of_measurement": level_of_measurement,
        }

    # ------------------------------------------------------------------
    # Convenience wrapper
    # ------------------------------------------------------------------

    def compute_all(
        self,
        rater1: list[int],
        rater2: list[int],
        rationales_a: list[str] | None = None,
        rationales_b: list[str] | None = None,
    ) -> dict[str, object]:
        """Run all applicable IRR metrics in a single call and return a summary.

        Convenience wrapper that executes Cohen's κ, Fleiss' κ, and
        Krippendorff's α on the same two-rater label data, plus an optional
        BERTScore pass when free-text rationales are supplied. All three
        label-agreement metrics operate on the same ``rater1``/``rater2``
        vectors so their results are directly comparable — divergence between
        them signals edge cases worth investigating (e.g. label imbalance
        inflating Cohen's κ).

        Recommended metric:
            Krippendorff's α is recommended as the primary metric because it
            is the most general: it handles any number of raters, any
            measurement level, and missing values. Use κ values for
            cross-validation and as sanity checks.

        Args:
            rater1: Ordered list of integer category labels from annotator 1.
            rater2: Ordered list of integer category labels from annotator 2.
                Must be the same length as ``rater1``.
            rationales_a: Optional free-text justifications from annotator A,
                one per rated item. BERTScore is run only when both
                ``rationales_a`` and ``rationales_b`` are provided.
            rationales_b: Optional free-text justifications from annotator B.
                Must be the same length as ``rationales_a`` when provided.

        Returns:
            A nested dict with the following top-level keys:

            - ``"cohens_kappa"`` (dict): Full result from
              :meth:`cohens_kappa`.
            - ``"fleiss_kappa"`` (dict): Full result from
              :meth:`fleiss_kappa`.
            - ``"krippendorffs_alpha"`` (dict): Full result from
              :meth:`krippendorffs_alpha`.
            - ``"bertscore_agreement"`` (dict | None): Full result from
              :meth:`bertscore_agreement`, or ``None`` if rationales were
              not provided.
            - ``"summary"`` (dict): Aggregated quality indicators:

              - ``"recommended_metric"`` (str): Always ``"krippendorff"`` —
                see class docstring for rationale.
              - ``"min_kappa"`` (float): Minimum of Cohen's κ, Fleiss' κ,
                and Krippendorff's α. The conservative lower bound on
                agreement across all three metrics.
              - ``"quality_gate_passed"`` (bool): ``True`` when
                Krippendorff's α ≥ 0.800 (Landis & Koch "almost perfect"
                threshold).

        Raises:
            ValueError: Propagated from the underlying metric methods if
                inputs are invalid.

        Example:
            >>> calc = IRRCalculator()
            >>> result = calc.compute_all([1, 2, 1, 2], [1, 2, 2, 2])
            >>> result["summary"]["quality_gate_passed"]
            False
        """
        cohens = self.cohens_kappa(rater1, rater2)

        matrix = [[r1, r2] for r1, r2 in zip(rater1, rater2)]
        n_categories = max(set(rater1 + rater2)) + 1
        fleiss = self.fleiss_kappa(matrix, n_categories=n_categories)

        reliability_data: list[list[float | None]] = [
            [float(v) for v in rater1],
            [float(v) for v in rater2],
        ]
        kripp = self.krippendorffs_alpha(reliability_data)

        bertscore: dict[str, object] | None = None
        if rationales_a is not None and rationales_b is not None:
            bertscore = self.bertscore_agreement(rationales_a, rationales_b)

        cohens_kappa_val = float(cohens["kappa"])  # type: ignore[arg-type]
        fleiss_kappa_val = float(fleiss["kappa"])  # type: ignore[arg-type]
        kripp_alpha_val = float(kripp["alpha"])  # type: ignore[arg-type]

        min_kappa = min(cohens_kappa_val, fleiss_kappa_val, kripp_alpha_val)
        quality_gate_passed = kripp_alpha_val >= 0.800

        logger.info(
            "compute_all summary: min_kappa=%.4f quality_gate=%s",
            min_kappa,
            quality_gate_passed,
        )
        return {
            "cohens_kappa": cohens,
            "fleiss_kappa": fleiss,
            "krippendorffs_alpha": kripp,
            "bertscore_agreement": bertscore,
            "summary": {
                "recommended_metric": "krippendorff",
                "min_kappa": min_kappa,
                "quality_gate_passed": quality_gate_passed,
            },
        }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.annotation.irr_calculator",
        description="Inter-Rater Reliability calculator for annotation quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.annotation.irr_calculator --dataset toy --metric all\n"
            "  python -m src.annotation.irr_calculator --metric krippendorff \\\n"
            "      --rater1 data/annotations/r1.json \\\n"
            "      --rater2 data/annotations/r2.json \\\n"
            "      --level ordinal --output results/irr_day10.json"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["cohens_kappa", "fleiss_kappa", "krippendorff", "bertscore", "all"],
        default="all",
        help="Which IRR metric(s) to compute (default: all).",
    )
    parser.add_argument(
        "--dataset",
        choices=["hh_rlhf", "toy"],
        default="toy",
        help=(
            "Built-in dataset to use when --rater1/--rater2 are not provided "
            "(default: toy)."
        ),
    )
    parser.add_argument(
        "--rater1",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a JSON file containing rater 1 labels as a list of ints.",
    )
    parser.add_argument(
        "--rater2",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a JSON file containing rater 2 labels as a list of ints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write JSON results to this file instead of stdout.",
    )
    parser.add_argument(
        "--level",
        choices=["nominal", "ordinal", "interval", "ratio"],
        default="ordinal",
        help="Measurement level for Krippendorff's α (default: ordinal).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _load_toy_dataset() -> dict[str, object]:
    """Load the bundled toy annotation dataset from configs/."""
    # Resolve relative to the repo root (two levels above this file's package).
    config_path = (
        Path(__file__).parent.parent.parent / "configs" / "toy_annotation_data.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(
            f"Toy dataset not found at {config_path}. "
            "Run from the repository root or check configs/toy_annotation_data.json."
        )
    with config_path.open() as fh:
        return dict(json.load(fh))


def _print_summary(results: dict[str, object]) -> None:
    """Print a human-readable summary, using rich if available."""
    summary = results.get("summary", {})
    gate = summary.get("quality_gate_passed") if isinstance(summary, dict) else None
    min_k = summary.get("min_kappa") if isinstance(summary, dict) else None

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title="IRR Calculator Results", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_column("Interpretation")

        ck = results.get("cohens_kappa")
        fk = results.get("fleiss_kappa")
        ka = results.get("krippendorffs_alpha")
        bs = results.get("bertscore_agreement")

        if isinstance(ck, dict):
            table.add_row(
                "Cohen's κ",
                f"{ck['kappa']:.4f}",
                str(ck["interpretation"]),
            )
        if isinstance(fk, dict):
            table.add_row(
                "Fleiss' κ",
                f"{fk['kappa']:.4f}",
                str(fk["interpretation"]),
            )
        if isinstance(ka, dict):
            table.add_row(
                "Krippendorff's α",
                f"{ka['alpha']:.4f}",
                str(ka["interpretation"]),
            )
        if isinstance(bs, dict):
            table.add_row(
                "BERTScore F1",
                f"{bs['f1_mean']:.4f}",
                str(bs["interpretation"]),
            )

        console.print(table)

        gate_str = "[green]PASSED[/green]" if gate else "[red]FAILED[/red]"
        console.print(f"\nQuality gate (α ≥ 0.800): {gate_str}   min κ/α = {min_k:.4f}")
    except ImportError:
        # Plain fallback when rich is not installed.
        print("=== IRR Calculator Results ===")
        for key in (
            "cohens_kappa",
            "fleiss_kappa",
            "krippendorffs_alpha",
            "bertscore_agreement",
        ):
            val = results.get(key)
            if isinstance(val, dict):
                metric_val = val.get("alpha", val.get("kappa", val.get("f1_mean")))
                interp = val.get("interpretation", "")
                print(f"  {key}: {metric_val:.4f}  ({interp})")
        gate_label = "PASSED" if gate else "FAILED"
        print(f"\nQuality gate (α ≥ 0.800): {gate_label}   min κ/α = {min_k:.4f}")


def _main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------ #
    # Dataset / label loading                                              #
    # ------------------------------------------------------------------ #
    if args.dataset == "hh_rlhf" and args.rater1 is None:
        print(
            "HH-RLHF analysis coming in Day 10. "
            "Run with --dataset toy to validate reference values today."
        )
        sys.exit(0)

    rationales_a: list[str] | None = None
    rationales_b: list[str] | None = None

    if args.rater1 is not None and args.rater2 is not None:
        rater1: list[int] = json.loads(args.rater1.read_text())
        rater2: list[int] = json.loads(args.rater2.read_text())
    else:
        toy = _load_toy_dataset()
        ck_toy = toy["cohens_kappa_toy"]
        bs_toy = toy["bertscore_toy"]
        assert isinstance(ck_toy, dict)
        assert isinstance(bs_toy, dict)
        rater1 = list(ck_toy["rater1"])
        rater2 = list(ck_toy["rater2"])
        rationales_a = list(bs_toy["rationales_a"])
        rationales_b = list(bs_toy["rationales_b"])
        logger.debug(
            "Loaded toy dataset: %d label items, %d rationale pairs",
            len(rater1),
            len(rationales_a),
        )

    # ------------------------------------------------------------------ #
    # Compute                                                              #
    # ------------------------------------------------------------------ #
    calc = IRRCalculator()

    if args.metric == "cohens_kappa":
        results: dict[str, object] = {"cohens_kappa": calc.cohens_kappa(rater1, rater2)}
    elif args.metric == "fleiss_kappa":
        matrix = [[r1, r2] for r1, r2 in zip(rater1, rater2)]
        n_cat = max(set(rater1 + rater2)) + 1
        results = {"fleiss_kappa": calc.fleiss_kappa(matrix, n_categories=n_cat)}
    elif args.metric == "krippendorff":
        reliability: list[list[float | None]] = [
            [float(v) for v in rater1],
            [float(v) for v in rater2],
        ]
        results = {
            "krippendorffs_alpha": calc.krippendorffs_alpha(
                reliability, level_of_measurement=args.level
            )
        }
    elif args.metric == "bertscore":
        if rationales_a is None or rationales_b is None:
            parser.error(
                "--metric bertscore requires rationales. "
                "Use --dataset toy or supply annotated rationale files."
            )
        results = {
            "bertscore_agreement": calc.bertscore_agreement(rationales_a, rationales_b)
        }
    else:  # "all"
        results = calc.compute_all(rater1, rater2, rationales_a, rationales_b)

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    if "summary" in results:
        _print_summary(results)

    json_out = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_out)
        print(f"\nResults written to {args.output}")
    else:
        print(json_out)


if __name__ == "__main__":
    _main()
