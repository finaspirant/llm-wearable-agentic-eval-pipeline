"""Build HuggingFace-loadable parquet for the 30-trajectory annotated wearable dataset.

Joins pre-calibration and post-calibration annotations with trajectory metadata
from the raw wearable logs. Output: data/processed/wearable_annotated_30.parquet

Usage:
    uv run python -m src.data.build_hf_dataset
    uv run python -m src.data.build_hf_dataset --output data/processed/custom.parquet
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import typer

logger = logging.getLogger(__name__)

_DEFAULT_PRE_CAL = Path("data/annotations/pre_calibration/day12_annotations.jsonl")
_DEFAULT_POST_CAL = Path("data/annotations/post_calibration/annotations_round2.json")
_DEFAULT_RAW_LOGS = Path("data/raw/synthetic_wearable_logs.jsonl")
_DEFAULT_OUTPUT = Path("data/processed/wearable_annotated_30.parquet")

_SCORE_DIMS = ["step_quality", "privacy_compliance", "goal_alignment", "error_recovery"]


def _load_pre_calibration(path: Path) -> pd.DataFrame:
    """Load JSONL pre-calibration annotations into a DataFrame."""
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["calibration_phase"] = "pre"
    return df


def _load_post_calibration(path: Path) -> pd.DataFrame:
    """Load JSON post-calibration annotations into a DataFrame."""
    with path.open() as fh:
        payload = json.load(fh)
    df = pd.DataFrame(payload["records"])
    df["calibration_phase"] = "post"
    return df


def _load_trajectory_metadata(path: Path) -> pd.DataFrame:
    """Load raw wearable logs, extracting flat trajectory-level metadata."""
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            log = json.loads(line)
            traj = log.get("trajectory", [])
            sensor = log.get("sensor_data", {})
            ctx = log.get("context_metadata", {})
            rows.append(
                {
                    "log_id": log["log_id"],
                    "log_timestamp": log.get("timestamp"),
                    "n_trajectory_steps": len(traj),
                    "final_action": traj[-1].get("action", "") if traj else "",
                    "mean_step_confidence": (
                        sum(s.get("confidence", 0.0) for s in traj) / len(traj)
                        if traj
                        else None
                    ),
                    "heart_rate": sensor.get("heart_rate"),
                    "spo2": sensor.get("spo2"),
                    "noise_db": sensor.get("noise_db"),
                    "device_model": ctx.get("device_model"),
                    "activity": ctx.get("activity"),
                    "alert_severity": ctx.get("alert_severity"),
                }
            )
    return pd.DataFrame(rows)


def build_dataset(
    pre_cal_path: Path,
    post_cal_path: Path,
    raw_logs_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Produce the consolidated annotated dataset parquet.

    Args:
        pre_cal_path: Path to pre-calibration JSONL annotations.
        post_cal_path: Path to post-calibration JSON annotations.
        raw_logs_path: Path to raw wearable logs JSONL.
        output_path: Destination parquet file path.

    Returns:
        The assembled DataFrame (also written to output_path).
    """
    logger.info("Loading pre-calibration annotations from %s", pre_cal_path)
    pre_df = _load_pre_calibration(pre_cal_path)

    logger.info("Loading post-calibration annotations from %s", post_cal_path)
    post_df = _load_post_calibration(post_cal_path)

    logger.info("Loading trajectory metadata from %s", raw_logs_path)
    meta_df = _load_trajectory_metadata(raw_logs_path)

    # Stack pre + post into one long-format annotation table
    annotations = pd.concat([pre_df, post_df], ignore_index=True)
    logger.info("Total annotation records: %d", len(annotations))

    # Join trajectory metadata (left join — every annotation gets metadata)
    df = annotations.merge(meta_df, on="log_id", how="left")

    # Enforce column ordering: identifiers → calibration_phase → scores → metadata
    id_cols = [
        "annotation_id",
        "log_id",
        "calibration_phase",
        "persona_name",
        "scenario_type",
        "consent_model",
        "ground_truth_action",
    ]
    score_cols = _SCORE_DIMS
    meta_cols = [
        "log_timestamp",
        "n_trajectory_steps",
        "final_action",
        "mean_step_confidence",
        "heart_rate",
        "spo2",
        "noise_db",
        "device_model",
        "activity",
        "alert_severity",
        "rationale",
        "created_at",
    ]
    present = set(df.columns)
    ordered = (
        [c for c in id_cols if c in present]
        + [c for c in score_cols if c in present]
        + [c for c in meta_cols if c in present]
    )
    df = df[ordered]

    # Cast score columns to nullable int8 (ordinal 1–5)
    for col in score_cols:
        if col in df.columns:
            df[col] = df[col].astype("Int8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Wrote %d rows × %d cols to %s", len(df), len(df.columns), output_path)
    return df


app = typer.Typer(add_completion=False)


@app.command()
def main(
    pre_cal: Path = typer.Option(
        _DEFAULT_PRE_CAL, help="Pre-calibration JSONL annotations"
    ),
    post_cal: Path = typer.Option(
        _DEFAULT_POST_CAL, help="Post-calibration JSON annotations"
    ),
    raw_logs: Path = typer.Option(_DEFAULT_RAW_LOGS, help="Raw wearable logs JSONL"),
    output: Path = typer.Option(_DEFAULT_OUTPUT, help="Output parquet path"),
) -> None:
    """Build consolidated HuggingFace-loadable parquet for wearable annotation dataset."""  # noqa: E501
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    df = build_dataset(pre_cal, post_cal, raw_logs, output)

    # Print quick summary to stdout
    n_logs = df["log_id"].nunique()
    n_personas = df["persona_name"].nunique()
    for phase in ["pre", "post"]:
        phase_df = df[df["calibration_phase"] == phase]
        mean_scores = phase_df[_SCORE_DIMS].mean().round(3).to_dict()
        logger.info(
            "%s-calibration: %d records, mean scores %s",
            phase,
            len(phase_df),
            mean_scores,
        )
    logger.info(
        "Dataset summary: %d trajectories × %d personas × 2 phases = %d records",
        n_logs,
        n_personas,
        len(df),
    )


if __name__ == "__main__":
    app()
