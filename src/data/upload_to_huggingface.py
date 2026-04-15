"""Package and upload the annotated wearable trajectory dataset to HuggingFace Hub.

Scans data/annotations/ for all annotation JSONL/JSON files, joins phase-level
IRR metrics from the post-calibration results, attaches the README.md dataset
card, and either prints a dry-run summary or pushes to the Hub.

Usage::

    # Inspect without uploading
    uv run python -m src.data.upload_to_huggingface --dry-run

    # Upload to HuggingFace Hub (requires HF_TOKEN env var)
    uv run python -m src.data.upload_to_huggingface --push \
        --repo-id your-username/wearable-agent-trajectory-annotations
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from huggingface_hub import DatasetCard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ANNOTATIONS_DIR = Path("data/annotations")
_README_PATH = _ANNOTATIONS_DIR / "README.md"

# Files that live under data/annotations/ but are NOT annotation records.
# Content-based detection (checking for annotation_id) is the primary filter;
# this set provides a fast-path skip for known non-record files.
_NON_RECORD_FILES = frozenset(
    {
        "agenteval-schema-v1.json",
        "calibration_round_01.json",
        "pia_results.json",
        "README.md",
        "wearable_annotation_rubric.md",
    }
)

# Canonical field that all annotation records must have.
_RECORD_KEY = "annotation_id"

# Map from ground_truth_action to a human-readable session_outcome label.
_ACTION_TO_OUTCOME: dict[str, str] = {
    "send_alert": "escalated",
    "suppress_capture": "suppressed",
    "trigger_geofence": "geofenced",
    "adjust_noise_profile": "adjusted",
    "surface_reminder": "reminded",
    "log_and_monitor": "monitored",
    "request_consent": "consent_requested",
    "escalate_to_emergency": "emergency_escalated",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_irr_lookup(annotations_dir: Path) -> dict[str, dict[str, float]]:
    """Return phase-level IRR metrics keyed by ``"pre"`` and ``"post"``.

    Reads from the post-calibration results file, which contains both
    pre- and post-calibration overall agreement scores.

    Args:
        annotations_dir: Root of the annotations directory tree.

    Returns:
        Mapping from phase name to a dict with keys
        ``fleiss_kappa``, ``cohens_kappa_mean``, ``krippendorffs_alpha``.
        Returns empty dicts for each phase if the results file is absent.
    """
    post_cal_file = annotations_dir / "post_calibration" / "annotations_round2.json"
    empty: dict[str, float] = {}
    if not post_cal_file.exists():
        logger.warning("Post-calibration results not found at %s", post_cal_file)
        return {"pre": empty, "post": empty}
    with post_cal_file.open() as fh:
        payload: dict[str, Any] = json.load(fh)
    irr = payload.get("irr_results", {})
    return {
        "pre": irr.get("pre_calibration", {}).get("overall", empty),
        "post": irr.get("post_calibration", {}).get("overall", empty),
    }


def _is_annotation_file(path: Path) -> bool:
    """Return True if *path* looks like a flat annotation record file.

    Checks that the file is not in the known non-record set and that its
    first parseable record contains ``annotation_id``.

    Args:
        path: Path to a candidate file.

    Returns:
        True if the file contains annotation records, False otherwise.
    """
    if path.name in _NON_RECORD_FILES:
        return False
    try:
        with path.open() as fh:
            if path.suffix == ".jsonl":
                first_line = fh.readline().strip()
                if not first_line:
                    return False
                record: dict[str, Any] = json.loads(first_line)
                return _RECORD_KEY in record
            else:  # .json
                payload: Any = json.load(fh)
                if isinstance(payload, list) and payload:
                    return _RECORD_KEY in payload[0]
                if isinstance(payload, dict):
                    # Support {"records": [...]} envelope (post-cal format)
                    records = payload.get("records", [])
                    return bool(records) and _RECORD_KEY in records[0]
    except (json.JSONDecodeError, OSError, KeyError):
        pass
    return False


def _read_records_from_file(path: Path) -> list[dict[str, Any]]:
    """Read all annotation records from a JSONL or JSON annotation file.

    Args:
        path: Path to a validated annotation file.

    Returns:
        List of raw annotation record dicts.
    """
    with path.open() as fh:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in fh if line.strip()]
        payload: Any = json.load(fh)
        records: list[dict[str, Any]]
        if isinstance(payload, list):
            records = payload
        else:
            records = payload.get("records", [])
        return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_annotations(
    annotations_dir: Path = _ANNOTATIONS_DIR,
    readme_path: Path = _README_PATH,
) -> tuple[Dataset, str]:
    """Load all annotation files and build a HuggingFace Dataset.

    Scans *annotations_dir* recursively for ``.jsonl`` and ``.json`` files
    that contain annotation records (identified by the presence of
    ``annotation_id``). Joins phase-level IRR metrics from the
    post-calibration results file. Deduplicates on ``annotation_id`` so that
    superseded subsets (e.g. the root-level ``day12_annotations.jsonl``) do
    not produce duplicate rows.

    The returned Dataset has the following columns:

    * ``trajectory_id`` — UUID of the source wearable log.
    * ``annotator_id`` — Persona name of the LLM-simulated annotator.
    * ``session_outcome`` — Human-readable label derived from
      ``ground_truth_action`` (e.g. ``"escalated"``, ``"suppressed"``).
    * ``overall_goal_achieved`` — ``True`` for all records; the synthetic
      dataset contains only success trajectories by design.
    * ``privacy_compliance_overall`` — Integer rubric score 1–4.
    * ``pre_calibration`` — ``True`` if annotated before calibration.
    * ``kappa_cohens`` — Phase-level mean pairwise Cohen's κ.
    * ``kappa_fleiss`` — Phase-level Fleiss' κ.
    * ``alpha_krippendorff`` — Phase-level Krippendorff's α.

    Args:
        annotations_dir: Root directory containing annotation files.
        readme_path: Path to the README.md dataset card.

    Returns:
        A tuple of ``(Dataset, card_text)`` where ``card_text`` is the
        raw README.md content (empty string if the file is absent).

    Raises:
        FileNotFoundError: If *annotations_dir* does not exist.
        ValueError: If no annotation records are found after deduplication.
    """
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    irr_lookup = _load_irr_lookup(annotations_dir)

    # Collect records; track seen annotation_ids to skip duplicates.
    seen_ids: set[str] = set()
    raw_rows: list[dict[str, Any]] = []

    # Only scan subdirectories — files directly under annotations_dir are either
    # schema/config files or superseded partial copies (e.g. the root-level
    # day12_annotations.jsonl which is a 5-trajectory subset of the authoritative
    # pre_calibration/ version).
    candidate_files = sorted(
        p
        for p in annotations_dir.rglob("*")
        if p.is_file()
        and p.suffix in {".jsonl", ".json"}
        and p.parent != annotations_dir
    )

    for path in candidate_files:
        if not _is_annotation_file(path):
            logger.debug("Skipping non-annotation file: %s", path)
            continue
        records = _read_records_from_file(path)
        # Infer phase from file path
        phase = "pre" if "pre_calibration" in str(path) else "post"
        irr = irr_lookup.get(phase, {})
        loaded = skipped = 0
        for rec in records:
            aid = rec.get(_RECORD_KEY)
            if not aid or aid in seen_ids:
                skipped += 1
                continue
            seen_ids.add(aid)
            raw_rows.append(
                {
                    "trajectory_id": rec.get("log_id", ""),
                    "annotator_id": rec.get("persona_name", ""),
                    "session_outcome": _ACTION_TO_OUTCOME.get(
                        rec.get("ground_truth_action", ""), "unknown"
                    ),
                    "overall_goal_achieved": True,
                    "privacy_compliance_overall": int(rec.get("privacy_compliance", 0)),
                    "pre_calibration": phase == "pre",
                    "kappa_cohens": irr.get("cohens_kappa_mean", float("nan")),
                    "kappa_fleiss": irr.get("fleiss_kappa", float("nan")),
                    "alpha_krippendorff": irr.get("krippendorffs_alpha", float("nan")),
                }
            )
            loaded += 1
        logger.info(
            "Loaded %d records from %s (%d duplicates skipped)",
            loaded,
            path,
            skipped,
        )

    if not raw_rows:
        raise ValueError(
            f"No annotation records found under {annotations_dir}. "
            "Check that pre_calibration/ and post_calibration/ subdirectories exist."
        )

    df = pd.DataFrame(raw_rows)
    df["privacy_compliance_overall"] = df["privacy_compliance_overall"].astype("int8")
    dataset = Dataset.from_pandas(df, preserve_index=False)

    card_text = ""
    if readme_path.exists():
        card_text = readme_path.read_text(encoding="utf-8")
    else:
        logger.warning(
            "README.md not found at %s — dataset card will be empty", readme_path
        )

    return dataset, card_text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package and upload the annotated wearable trajectory dataset"
            " to HuggingFace Hub."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print dataset stats without uploading.",
    )
    parser.add_argument(
        "--repo-id",
        default="wearable-agent-trajectory-annotations",
        help=(
            "HuggingFace repository ID"
            " (e.g. your-username/wearable-agent-trajectory-annotations)."
        ),
    )
    parser.add_argument(
        "--push",
        action="store_true",
        default=False,
        help="Actually upload to HuggingFace Hub (requires HF_TOKEN env var).",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=_ANNOTATIONS_DIR,
        help="Path to the annotations directory.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=_README_PATH,
        help="Path to the README.md dataset card.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the HuggingFace upload CLI.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Returns:
        Exit code (0 on success, 1 on error).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dry_run and not args.push:
        parser.error("Specify --dry-run to inspect or --push to upload.")

    try:
        dataset, card_text = load_annotations(
            annotations_dir=args.annotations_dir,
            readme_path=args.readme,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load annotations: %s", exc)
        return 1

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — no data will be uploaded")
        print("=" * 60)
        print(f"\nRecords : {len(dataset)}")
        print(f"Columns : {dataset.column_names}")
        print("\nSample rows (first 3):")
        df = dataset.to_pandas()
        print(df.head(3).to_string(index=False))
        print()
        if card_text:
            first_line = card_text.splitlines()[0] if card_text.splitlines() else ""
            print(
                f"Dataset card : README.md attached"
                f" ({len(card_text)} chars, starts: {first_line!r})"
            )
        else:
            print("Dataset card : NOT FOUND — upload will proceed without a card")
        print()
        phase_counts = df["pre_calibration"].value_counts().to_dict()
        n_pre = phase_counts.get(True, 0)
        n_post = phase_counts.get(False, 0)
        print(f"Phase breakdown : pre={n_pre}  post={n_post}")
        print(f"Unique trajectories : {df['trajectory_id'].nunique()}")
        annotators = sorted(df["annotator_id"].unique())
        print(f"Unique annotators   : {df['annotator_id'].nunique()} ({annotators})")
        kappa_pre = df.loc[df["pre_calibration"], "kappa_cohens"].iloc[0]
        kappa_post = df.loc[~df["pre_calibration"], "kappa_cohens"].iloc[0]
        print(f"Cohen's κ           : pre={kappa_pre:.4f}  post={kappa_post:.4f}")
        print()

    if args.push:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.error(
                "HF_TOKEN environment variable not set. "
                "Run: export HF_TOKEN=hf_... then retry."
            )
            return 1
        logger.info("Pushing dataset to Hub as %s …", args.repo_id)
        dataset.push_to_hub(
            args.repo_id,
            token=hf_token,
            commit_message=(
                "Add annotated wearable trajectory dataset"
                " (30 trajectories, 5 personas, 2 phases)"
            ),
        )
        if card_text:
            card = DatasetCard(card_text)
            card.push_to_hub(args.repo_id, token=hf_token)
            logger.info("Dataset card pushed.")
        logger.info("Upload complete: https://huggingface.co/datasets/%s", args.repo_id)

    return 0


if __name__ == "__main__":
    sys.exit(main())
