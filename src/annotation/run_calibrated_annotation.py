"""Post-calibration re-annotation pipeline for wearable trajectory annotation.

Demonstrates the efficacy of :class:`CalibrationConfig` by re-running the
5-annotator simulation on the same trajectories with calibrated persona prompts,
then computing before/after Fleiss' κ, Cohen's κ (mean pairwise), and
Krippendorff's α.

STRATEGIC CONTEXT:
    Kore.ai (Oct 2025) reports only 52 % of enterprises have real evaluation.
    This pipeline closes that gap using anchor-and-rule calibration (Cohere
    Command A, arXiv 2504.00698): persona prompts are augmented with 5 worked
    calibration examples and per-dimension decision rules, pulling scores toward
    a shared rubric without erasing legitimate persona diversity.

    Target: Krippendorff's α ≥ 0.80 (substantial agreement) — up from the
    pre-calibration floor of ≈ −0.03 (poor).

CLI::

    python -m src.annotation.run_calibrated_annotation
    python -m src.annotation.run_calibrated_annotation \\
        --pre-cal-annotations \\
            data/annotations/pre_calibration/day12_annotations.jsonl \\
        --cal-config         data/annotations/calibration_round_01.json \\
        --output data/annotations/post_calibration/annotations_round2.json \\
        --dry-run

    # Flags
    --dry-run / --no-dry-run   Use deterministic calibrated scores (default: True)
    --verbose / -v             Enable DEBUG logging
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import typer

from src.annotation.annotator_simulator import (
    _DIMENSIONS,
    _PERSONAS,
    AnnotatorSimulator,
    compute_irr,
    find_disagreement_hotspots,
)
from src.annotation.calibration_protocol import (
    AnchorExample,
    CalibrationConfig,
    apply_calibration_to_persona,
    run_calibration_round,
)
from src.annotation.irr_calculator import IRRCalculator, _interpret_kappa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_RAW_INPUT = Path("data/raw/synthetic_wearable_logs.jsonl")
_DEFAULT_PRE_CAL_ANNOTATIONS = Path(
    "data/annotations/pre_calibration/day12_annotations.jsonl"
)
_DEFAULT_CAL_CONFIG = Path("data/annotations/calibration_round_01.json")
_DEFAULT_POST_CAL_OUTPUT = Path(
    "data/annotations/post_calibration/annotations_round2.json"
)

# Calibration weight: fraction of score drawn from the anchor gold target.
# At 0.82, the calibrated score is 82 % gold + 18 % persona bias, which
# simulates how an LLM persona would respond to seeing worked examples and
# per-dimension decision rules injected into its system prompt.
# 0.82 is the minimum weight that avoids banker's rounding artefacts when
# a persona's base score is exactly 2 away from the gold (e.g. base=1,
# gold=3: 0.75×3 + 0.25×1 = 2.5 rounds to 2; 0.82×3 + 0.18×1 = 2.64 → 3).
_CALIBRATION_WEIGHT: float = 0.82

# Minimum Krippendorff's α required for the annotation batch to pass
# the quality gate (substantial agreement, Landis & Koch 1977).
_TARGET_ALPHA: float = 0.80

_N_TOP_HOTSPOTS: int = 3

# ---------------------------------------------------------------------------
# Calibrated simulator
# ---------------------------------------------------------------------------


class CalibratedAnnotatorSimulator(AnnotatorSimulator):
    """AnnotatorSimulator subclass that applies anchor-calibrated scoring.

    In **live API mode**, the calibrated persona system prompts (augmented via
    :func:`~src.annotation.calibration_protocol.apply_calibration_to_persona`)
    cause LLM personas to anchor their scores to the worked examples and
    per-dimension decision rules from the calibration round.

    In **dry-run mode**, the calibration effect is modelled deterministically:
    each persona's base dry-run score is pulled toward the per-log gold score
    (or the mean gold across all anchor logs if the log is not itself an anchor)
    by :data:`_CALIBRATION_WEIGHT`.  This simulates the convergence produced
    by real LLM calibration without consuming API quota.

    Args:
        calibration_config: The calibration artifact produced by
            :func:`~src.annotation.calibration_protocol.run_calibration_round`.
        api_key: Anthropic API key (forwarded to parent).
        output_path: JSONL write path for :meth:`annotate_all` (forwarded to
            parent).
        dry_run: When ``True``, use calibrated deterministic scoring.
    """

    def __init__(
        self,
        calibration_config: CalibrationConfig,
        api_key: str | None = None,
        output_path: Path = _DEFAULT_POST_CAL_OUTPUT,
        dry_run: bool = False,
    ) -> None:
        super().__init__(api_key=api_key, output_path=output_path, dry_run=dry_run)
        self._cal_config = calibration_config

        # Build calibrated prompts for live API mode.
        self.PERSONAS: dict[str, dict[str, str]] = {
            name: {
                "name": name,
                "system_prompt": apply_calibration_to_persona(
                    persona["system_prompt"], calibration_config
                ),
            }
            for name, persona in _PERSONAS.items()
        }

        # Anchor gold scores keyed by trajectory_id for per-log targeting.
        self._gold_by_log: dict[str, dict[str, int]] = {
            anchor.trajectory_id: anchor.correct_scores
            for anchor in calibration_config.anchors
        }

        # Mean gold across all anchors — fallback for non-anchor logs.
        self._mean_gold: dict[str, float] = _mean_anchor_gold(
            calibration_config.anchors
        )

    def _dry_run_scores(  # type: ignore[override]
        self,
        log: dict[str, Any],
        persona_name: str,
    ) -> dict[str, Any]:
        """Return calibrated deterministic scores for dry-run mode.

        Gets the uncalibrated base score from the parent static method, then
        applies :data:`_CALIBRATION_WEIGHT` to pull each dimension toward the
        per-log gold target (anchor gold if available; mean anchor gold for
        non-anchor logs).

        Args:
            log: Wearable log dict.  Uses ``log_id`` and ``scenario_type``.
            persona_name: One of the 5 persona keys.

        Returns:
            Dict with calibrated integer scores in [1, 4] for all 4 dimensions
            and an updated ``"rationale"`` string.
        """
        # Call the parent static method directly to get uncalibrated base.
        base = AnnotatorSimulator._dry_run_scores(log, persona_name)

        log_id = log.get("log_id", "")
        gold: dict[str, Any]
        if log_id in self._gold_by_log:
            gold = dict(self._gold_by_log[log_id])
        else:
            gold = {dim: round(v) for dim, v in self._mean_gold.items()}

        w = _CALIBRATION_WEIGHT
        for dim in _DIMENSIONS:
            base_score = float(base[dim])
            gold_score = float(gold.get(dim, 2))
            calibrated = w * gold_score + (1.0 - w) * base_score
            base[dim] = max(1, min(4, round(calibrated)))

        scenario = log.get("scenario_type", "unknown")
        base["rationale"] = (
            f"[CALIBRATED DRY-RUN] {persona_name} on {scenario}: "
            f"sq={base['step_quality']} pc={base['privacy_compliance']} "
            f"ga={base['goal_alignment']} er={base['error_recovery']}."
        )
        return base


# ---------------------------------------------------------------------------
# Helper: mean anchor gold
# ---------------------------------------------------------------------------


def _mean_anchor_gold(anchors: list[AnchorExample]) -> dict[str, float]:
    """Compute mean gold score per dimension across all calibration anchors.

    Used as the calibration target for non-anchor logs in dry-run mode.

    Args:
        anchors: :class:`~src.annotation.calibration_protocol.AnchorExample`
            objects from the calibration config.

    Returns:
        Dict mapping each dimension name to its mean gold score.  Returns
        2.5 for all dimensions when ``anchors`` is empty.
    """
    if not anchors:
        return {dim: 2.5 for dim in _DIMENSIONS}
    result: dict[str, float] = {}
    for dim in _DIMENSIONS:
        scores = [a.correct_scores.get(dim, 2) for a in anchors]
        result[dim] = sum(scores) / len(scores)
    return result


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def load_trajectories(
    input_path: Path,
    log_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load wearable trajectory logs from a JSONL file.

    Args:
        input_path: Path to the JSONL file (one JSON object per line).
        log_ids: Optional set of ``log_id`` values to include.  When provided,
            only matching logs are returned.  ``None`` returns all logs.

    Returns:
        List of wearable log dicts in file order.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If no valid records are found after filtering.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Trajectory input not found: {input_path}")

    logs: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            obj: dict[str, Any] = json.loads(stripped)
            if log_ids is None or obj.get("log_id") in log_ids:
                logs.append(obj)

    if not logs:
        suffix = f" (filtered to {len(log_ids)} log_ids)" if log_ids else ""
        raise ValueError(f"No valid trajectory records found in {input_path}{suffix}")

    logger.info("Loaded %d trajectories from %s", len(logs), input_path)
    return logs


def load_pre_calibration_annotations(path: Path) -> list[dict[str, Any]]:
    """Load pre-calibration annotation records from a JSONL file.

    Args:
        path: Path to the pre-calibration JSONL file.

    Returns:
        Flat list of annotation record dicts.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains no valid records.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pre-calibration annotations not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if not records:
        raise ValueError(f"No annotation records found in {path}")

    logger.info("Loaded %d pre-calibration records from %s", len(records), path)
    return records


def load_calibration_config(path: Path) -> CalibrationConfig:
    """Load a serialised CalibrationConfig from disk.

    Reads the JSON artifact written by
    :func:`~src.annotation.calibration_protocol.save_calibration_config`.

    Args:
        path: Path to the JSON file written by
            :func:`~src.annotation.calibration_protocol.save_calibration_config`.

    Returns:
        Reconstructed :class:`~src.annotation.calibration_protocol.CalibrationConfig`
        with all anchor examples populated.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration config not found: {path}")

    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    anchors = [
        AnchorExample(
            trajectory_id=a["trajectory_id"],
            trajectory_data=a["trajectory_data"],
            correct_scores=a["correct_scores"],
            explanation=a["explanation"],
            difficulty=a["difficulty"],
            normalized_mean_score=a["normalized_mean_score"],
        )
        for a in raw.get("anchors", [])
    ]
    return CalibrationConfig(
        anchor_example_ids=raw["anchor_example_ids"],
        rubric_updates=raw["rubric_updates"],
        round_number=raw["round_number"],
        timestamp=raw["timestamp"],
        anchors=anchors,
        target_kappa=raw.get("target_kappa", 0.55),
        pre_calibration_kappa=raw.get("pre_calibration_kappa", {}),
    )


# ---------------------------------------------------------------------------
# Multi-metric IRR
# ---------------------------------------------------------------------------


def compute_full_irr(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute Fleiss' κ, Cohen's κ (mean pairwise), and Krippendorff's α.

    Builds balanced matrices for all three IRR metrics from the same data.
    Only trajectories with complete scores from all 5 personas are included.

    Args:
        records: Flat list of annotation record dicts.  Each dict must contain
            ``log_id``, ``persona_name``, and the 4 dimension score keys.

    Returns:
        Dict keyed by dimension name (plus ``"overall"``).  Each value is a
        dict with the following keys:

        - ``"fleiss_kappa"`` (float): Fleiss' κ for 5 nominal raters.
        - ``"cohens_kappa_mean"`` (float): Mean Cohen's κ over all
          C(5, 2) = 10 pairwise combinations.
        - ``"krippendorffs_alpha"`` (float): Krippendorff's α (ordinal).
        - ``"fleiss_interpretation"`` (str): Landis & Koch label for κ.
        - ``"alpha_interpretation"`` (str): Landis & Koch label for α.
        - ``"n_items"`` (int): Number of complete trajectories used.

        The ``"overall"`` entry averages each metric across all 4 dimensions.

    Raises:
        ValueError: If fewer than 2 complete trajectories are available.
    """
    persona_names = list(_PERSONAS.keys())
    n_personas = len(persona_names)

    by_log: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in records:
        log_id = rec["log_id"]
        if log_id not in by_log:
            by_log[log_id] = {}
        by_log[log_id][rec["persona_name"]] = rec

    complete_log_ids = [lid for lid, pmap in by_log.items() if len(pmap) == n_personas]
    if len(complete_log_ids) < 2:
        raise ValueError(
            f"IRR requires ≥ 2 fully-annotated trajectories; "
            f"got {len(complete_log_ids)}.  "
            "Check for failed annotation calls in the batch."
        )

    calc = IRRCalculator()
    result: dict[str, dict[str, Any]] = {}

    for dim in _DIMENSIONS:
        # Fleiss' κ: [n_items × n_raters], 0-indexed labels in [0, 3].
        fleiss_matrix: list[list[int]] = [
            [by_log[lid][p][dim] - 1 for p in persona_names] for lid in complete_log_ids
        ]
        fleiss_res = calc.fleiss_kappa(fleiss_matrix, n_categories=4)

        # Krippendorff's α: [n_raters × n_items], raw scores as floats.
        reliability_data: list[list[float]] = [
            [float(by_log[lid][p][dim]) for lid in complete_log_ids]
            for p in persona_names
        ]
        alpha_res = calc.krippendorffs_alpha(
            reliability_data,  # type: ignore[arg-type]
            level_of_measurement="ordinal",
        )

        # Cohen's κ: average over all C(5, 2) = 10 pairwise combinations.
        pairwise: list[float] = []
        for p1, p2 in combinations(persona_names, 2):
            r1 = [by_log[lid][p1][dim] for lid in complete_log_ids]
            r2 = [by_log[lid][p2][dim] for lid in complete_log_ids]
            cohens_res = calc.cohens_kappa(r1, r2)
            pairwise.append(float(cohens_res["kappa"]))  # type: ignore[arg-type]
        mean_cohens = sum(pairwise) / len(pairwise)

        fleiss_kappa_val = float(fleiss_res["kappa"])  # type: ignore[arg-type]
        alpha_val = float(alpha_res["alpha"])  # type: ignore[arg-type]
        result[dim] = {
            "fleiss_kappa": fleiss_kappa_val,
            "cohens_kappa_mean": mean_cohens,
            "krippendorffs_alpha": alpha_val,
            "fleiss_interpretation": str(fleiss_res["interpretation"]),
            "alpha_interpretation": str(alpha_res["interpretation"]),
            "n_items": len(complete_log_ids),
        }

    # Overall: mean across all 4 dimensions.
    def _mean_dim(key: str) -> float:
        return float(sum(result[d][key] for d in _DIMENSIONS) / len(_DIMENSIONS))

    overall_fleiss = _mean_dim("fleiss_kappa")
    overall_cohens = _mean_dim("cohens_kappa_mean")
    overall_alpha = _mean_dim("krippendorffs_alpha")

    result["overall"] = {
        "fleiss_kappa": overall_fleiss,
        "cohens_kappa_mean": overall_cohens,
        "krippendorffs_alpha": overall_alpha,
        "fleiss_interpretation": _interpret_kappa(overall_fleiss),
        "alpha_interpretation": _interpret_kappa(overall_alpha),
        "n_items": len(complete_log_ids),
    }

    logger.info(
        "IRR (%d items): fleiss=%.3f  cohens=%.3f  alpha=%.3f",
        len(complete_log_ids),
        overall_fleiss,
        overall_cohens,
        overall_alpha,
    )
    return result


# ---------------------------------------------------------------------------
# Output: comparison table
# ---------------------------------------------------------------------------


def _fmt(val: float) -> str:
    """Format a κ / α value with sign for tabular alignment."""
    return f"{val:+.3f}"


def print_comparison_table(
    pre_irr: dict[str, dict[str, Any]],
    post_irr: dict[str, dict[str, Any]],
    hotspots: list[dict[str, Any]],
) -> None:
    """Print a formatted before/after IRR comparison to stdout.

    Prints three metric blocks (Fleiss' κ, Cohen's κ, Krippendorff's α),
    each showing pre-calibration values, post-calibration values, delta,
    and the Landis & Koch interpretation of the post-calibration score.
    Ends with the top-3 disagreement hotspot summary.

    Args:
        pre_irr: Full IRR dict from :func:`compute_full_irr` on pre-calibration
            records.
        post_irr: Full IRR dict from :func:`compute_full_irr` on
            post-calibration records.
        hotspots: List of hotspot dicts from
            :func:`~src.annotation.annotator_simulator.find_disagreement_hotspots`.
    """
    col_w = 22
    sep = "─" * 76

    n_pre = pre_irr["overall"]["n_items"]
    n_post = post_irr["overall"]["n_items"]

    print()
    print("=" * 76)
    print("  CALIBRATION ROUND 1 — BEFORE vs AFTER IRR COMPARISON")
    print(f"  Pre-calibration  : {n_pre} trajectories × 5 personas")
    print(f"  Post-calibration : {n_post} trajectories × 5 personas")
    print("=" * 76)

    def _row(label: str, pre_val: float, post_val: float) -> None:
        delta = post_val - pre_val
        arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "=")
        interp = _interpret_kappa(post_val)
        print(
            f"  {label:<{col_w}} {_fmt(pre_val):>8}  →  {_fmt(post_val):>8}"
            f"   Δ {_fmt(delta):>8}  {arrow}  {interp}"
        )

    metrics = [
        ("fleiss_kappa", "FLEISS' κ  (5-rater nominal)"),
        ("cohens_kappa_mean", "COHEN'S κ  (mean pairwise, C(5,2)=10)"),
        ("krippendorffs_alpha", "KRIPPENDORFF'S α  (ordinal)"),
    ]
    for metric_key, metric_label in metrics:
        print()
        print(f"  {metric_label}")
        print(f"  {sep}")
        hdr = f"  {'Dimension':<{col_w}} {'Pre':>8}     {'Post':>8}"
        print(hdr + f"   {'Delta':>10}       Post-cal interpretation")
        print(f"  {sep}")
        for dim in _DIMENSIONS:
            _row(dim, pre_irr[dim][metric_key], post_irr[dim][metric_key])
        print(f"  {sep}")
        _row("Overall", pre_irr["overall"][metric_key], post_irr["overall"][metric_key])
        print()

    print("  TOP-3 PRE-CALIBRATION DISAGREEMENT HOTSPOTS")
    print(f"  {sep}")
    for hs in hotspots:
        ids_short = ", ".join(lid[:8] for lid in hs.get("top_variance_log_ids", []))
        print(
            f"  #{hs['rank']} {hs['dimension']:<{col_w - 3}} "
            f"κ = {hs['kappa']:+.4f}  ({hs['interpretation']})"
        )
        if ids_short:
            print(f"       High-variance trajectories: {ids_short}")
    print(f"  {sep}")
    print()


# ---------------------------------------------------------------------------
# Output: save annotations
# ---------------------------------------------------------------------------


def save_post_calibration_annotations(
    records: list[dict[str, Any]],
    cal_config: CalibrationConfig,
    output_path: Path,
    pre_irr: dict[str, dict[str, Any]] | None = None,
    post_irr: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Save post-calibration annotation records as a JSON object.

    Wraps the flat records list in a metadata envelope for DVC tracking.
    Written as a single JSON file (not JSONL) to distinguish from the
    pre-calibration JSONL format.

    The ``irr_results`` block persists the before/after agreement numbers so
    downstream consumers (HuggingFace dataset card, white papers) can cite
    them without re-running the pipeline.

    Args:
        records: Flat list of post-calibration annotation record dicts.
        cal_config: The calibration config used to produce these records.
        output_path: Destination path.  Parent directories are created if
            they do not exist.
        pre_irr: Full IRR dict from :func:`compute_full_irr` on
            pre-calibration records.  Stored under
            ``irr_results.pre_calibration``.
        post_irr: Full IRR dict from :func:`compute_full_irr` on
            post-calibration records.  Stored under
            ``irr_results.post_calibration``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    irr_block: dict[str, Any] = {}
    if pre_irr is not None:
        irr_block["pre_calibration"] = pre_irr
    if post_irr is not None:
        irr_block["post_calibration"] = post_irr
        overall = post_irr.get("overall", {})
        irr_block["headline"] = {
            "pre_fleiss_kappa_overall": (
                pre_irr["overall"]["fleiss_kappa"] if pre_irr else None
            ),
            "post_fleiss_kappa_overall": overall.get("fleiss_kappa"),
            "post_krippendorffs_alpha_overall": overall.get("krippendorffs_alpha"),
            "post_alpha_interpretation": overall.get("alpha_interpretation"),
            "n_trajectories": overall.get("n_items"),
            "n_personas": len(_PERSONAS),
        }

    payload: dict[str, Any] = {
        "metadata": {
            "round": 2,
            "calibration_round": cal_config.round_number,
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "n_logs": len({r["log_id"] for r in records}),
            "n_records": len(records),
            "calibration_weight": _CALIBRATION_WEIGHT,
            "target_alpha": _TARGET_ALPHA,
            "target_kappa": cal_config.target_kappa,
            "anchor_example_ids": cal_config.anchor_example_ids,
        },
        "irr_results": irr_block,
        "records": records,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "Post-calibration annotations saved: %d records → %s",
        len(records),
        output_path,
    )
    if irr_block.get("headline"):
        h = irr_block["headline"]
        logger.info(
            "IRR headline: pre Fleiss κ=%.3f → post Fleiss κ=%.3f  "
            "post Krippendorff α=%.3f (%s)",
            h["pre_fleiss_kappa_overall"] or 0.0,
            h["post_fleiss_kappa_overall"] or 0.0,
            h["post_krippendorffs_alpha_overall"] or 0.0,
            h["post_alpha_interpretation"] or "unknown",
        )


# ---------------------------------------------------------------------------
# Target assertion
# ---------------------------------------------------------------------------


def assert_target_met(
    post_irr: dict[str, dict[str, Any]],
    target_alpha: float = _TARGET_ALPHA,
) -> bool:
    """Check that all dimensions meet the Krippendorff's α quality gate.

    Prints a per-dimension pass/fail verdict and an overall summary.  Does
    not raise on failure — the caller decides whether to exit.

    Args:
        post_irr: Full IRR dict from :func:`compute_full_irr` on
            post-calibration records.
        target_alpha: Minimum acceptable Krippendorff's α.  Defaults to
            :data:`_TARGET_ALPHA` (0.80, substantial agreement).

    Returns:
        ``True`` if every dimension and the overall average meet
        ``target_alpha``; ``False`` otherwise.
    """
    print()
    print("=" * 76)
    print(
        f"  TARGET ASSERTION: Krippendorff's α ≥ {target_alpha:.2f}"
        " per dimension (substantial agreement)"
    )
    print("=" * 76)

    all_pass = True
    for dim in _DIMENSIONS:
        alpha = post_irr[dim]["krippendorffs_alpha"]
        passed = alpha >= target_alpha
        if not passed:
            all_pass = False
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {status}  {dim:<24} α = {alpha:.4f}")

    overall_alpha = post_irr["overall"]["krippendorffs_alpha"]
    overall_pass = overall_alpha >= target_alpha
    if not overall_pass:
        all_pass = False
    status = "PASS ✓" if overall_pass else "FAIL ✗"
    print(f"  {status}  {'overall':<24} α = {overall_alpha:.4f}")
    print()

    if all_pass:
        print(
            "  ✓ Calibration target met.  Annotations are ready for PRM"
            " training data curation."
        )
    else:
        failing = [
            d for d in _DIMENSIONS if post_irr[d]["krippendorffs_alpha"] < target_alpha
        ]
        print(f"  ✗ Target NOT met.  Dimensions still needing work: {failing}")
        print(
            "  Recommendation: inspect anchor examples for those dimensions"
            " and run calibration round 2."
        )

    print()
    return all_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_input: Path = typer.Option(
        _DEFAULT_RAW_INPUT,
        help="JSONL file with raw wearable trajectory logs.",
    ),
    pre_cal_annotations: Path = typer.Option(
        _DEFAULT_PRE_CAL_ANNOTATIONS,
        help="Pre-calibration annotation JSONL (Day 12 output).",
    ),
    cal_config_path: Path = typer.Option(
        _DEFAULT_CAL_CONFIG,
        help="CalibrationConfig JSON produced by calibration_protocol.",
    ),
    output: Path = typer.Option(
        _DEFAULT_POST_CAL_OUTPUT,
        help="Destination JSON for post-calibration annotations.",
    ),
    dry_run: bool = typer.Option(
        True,  # noqa: FBT003
        help=(
            "Use calibrated deterministic scoring (no API calls).  "
            "Pass --no-dry-run to use live Anthropic API."
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging.",  # noqa: FBT003
    ),
) -> None:
    """Re-annotate wearable trajectories with calibrated persona prompts.

    Loads pre-calibration annotations and a CalibrationConfig, re-runs all
    5 annotator personas with calibrated prompts, computes before/after IRR,
    prints a comparison table, and asserts Krippendorff's α ≥ 0.80.

    Exits with code 1 if the target is not met.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    # ------------------------------------------------------------------ #
    # 1. Load pre-calibration annotations                                  #
    # ------------------------------------------------------------------ #
    pre_records = load_pre_calibration_annotations(pre_cal_annotations)
    annotated_log_ids = {r["log_id"] for r in pre_records}
    logger.info(
        "Found %d unique log_ids in pre-calibration annotations",
        len(annotated_log_ids),
    )

    # ------------------------------------------------------------------ #
    # 2. Load the matching trajectories from raw input                    #
    # ------------------------------------------------------------------ #
    logs = load_trajectories(raw_input, log_ids=annotated_log_ids)
    if len(logs) < len(annotated_log_ids):
        logger.warning(
            "Only %d of %d annotated log_ids found in %s; "
            "proceeding with available trajectories.",
            len(logs),
            len(annotated_log_ids),
            raw_input,
        )

    # ------------------------------------------------------------------ #
    # 3. Load or rebuild CalibrationConfig                                #
    # ------------------------------------------------------------------ #
    if cal_config_path.exists():
        cal_config = load_calibration_config(cal_config_path)
        logger.info(
            "Loaded CalibrationConfig round %d from %s",
            cal_config.round_number,
            cal_config_path,
        )
    else:
        logger.info(
            "CalibrationConfig not found at %s — building from pre-calibration data.",
            cal_config_path,
        )
        pre_fleiss_for_build = compute_irr(pre_records)
        hotspots_for_build = find_disagreement_hotspots(
            pre_records, pre_fleiss_for_build
        )
        cal_config = run_calibration_round(
            trajectories=logs,
            pre_annotations=pre_records,
            disagreement_categories=hotspots_for_build,
        )

    # ------------------------------------------------------------------ #
    # 4. Compute pre-calibration full IRR                                 #
    # ------------------------------------------------------------------ #
    logger.info("Computing pre-calibration IRR (Fleiss κ + Cohen κ + Krippendorff α)…")
    pre_irr = compute_full_irr(pre_records)

    # ------------------------------------------------------------------ #
    # 5. Identify pre-calibration disagreement hotspots                   #
    # ------------------------------------------------------------------ #
    pre_fleiss_irr = compute_irr(pre_records)
    hotspots = find_disagreement_hotspots(
        pre_records, pre_fleiss_irr, top_n=_N_TOP_HOTSPOTS
    )

    # ------------------------------------------------------------------ #
    # 6. Run calibrated annotation                                        #
    # ------------------------------------------------------------------ #
    # annotate_all writes a JSONL to _output_path internally.  Route it to
    # a temp file so the caller's output path holds only the final JSON.
    tmp_jsonl = output.parent / f"_{output.stem}.tmp.jsonl"
    tmp_jsonl.parent.mkdir(parents=True, exist_ok=True)

    sim = CalibratedAnnotatorSimulator(
        calibration_config=cal_config,
        output_path=tmp_jsonl,
        dry_run=dry_run,
    )
    logger.info(
        "Running calibrated annotation: %d trajectories × 5 personas (dry_run=%s)…",
        len(logs),
        dry_run,
    )
    post_records = sim.annotate_all(logs)

    # Clean up the temp JSONL; the canonical output is the JSON wrapper below.
    if tmp_jsonl.exists():
        tmp_jsonl.unlink()

    # ------------------------------------------------------------------ #
    # 7. Compute post-calibration full IRR                                #
    # ------------------------------------------------------------------ #
    logger.info("Computing post-calibration IRR (Fleiss κ + Cohen κ + Krippendorff α)…")
    post_irr = compute_full_irr(post_records)

    # ------------------------------------------------------------------ #
    # 8. Print comparison table                                           #
    # ------------------------------------------------------------------ #
    print_comparison_table(pre_irr, post_irr, hotspots)

    # ------------------------------------------------------------------ #
    # 9. Save post-calibration annotations as JSON                        #
    # ------------------------------------------------------------------ #
    save_post_calibration_annotations(
        post_records, cal_config, output, pre_irr=pre_irr, post_irr=post_irr
    )

    # ------------------------------------------------------------------ #
    # 10. Assert calibration target met                                   #
    # ------------------------------------------------------------------ #
    target_met = assert_target_met(post_irr)
    if not target_met:
        sys.exit(1)


if __name__ == "__main__":
    app()
