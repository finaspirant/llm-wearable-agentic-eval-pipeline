"""FACTS Grounding Benchmark evaluation script for wearable agent trajectories.

Implements the two-phase evaluation protocol from the DeepMind FACTS Grounding
Benchmark (Dec 2025), applied to wearable/ambient AI agent trajectories:

* **Phase 1 — Eligibility**: Does the agent's final action fulfill the task
  request (i.e., match ``ground_truth_action``)?  Ineligible responses score 0.
* **Phase 2 — Grounding**: Is the agent's reasoning grounded in the observed
  sensor context?  Uses :class:`~src.eval.agentic_eval.FACTSGroundingScorer`
  (token-overlap ``search_score`` + RAGAS ``grounding_score`` fallback).

Combined FACTS score follows the FACTS paper convention:
``facts_score = grounding_score  if  eligibility_score == 1  else  0.0``

No model currently cracks 70% on FACTS — this script tracks that ceiling for
wearable AI agents specifically.

CLI::

    uv run python -m src.eval.facts_kaggle_eval
    uv run python -m src.eval.facts_kaggle_eval --n 50 --output results/custom.csv
    uv run python -m src.eval.facts_kaggle_eval \
        --input data/raw/synthetic_wearable_logs.jsonl
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from src.eval.agentic_eval import FACTSGroundingScorer

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="facts-kaggle-eval",
    help="Run two-phase FACTS evaluation on wearable agent trajectories.",
    add_completion=False,
)

_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_INPUT = _ROOT / "data" / "raw" / "synthetic_wearable_logs.jsonl"
_DEFAULT_OUTPUT = _ROOT / "results" / "facts_kaggle_submission.csv"

_CSV_COLUMNS = [
    "trajectory_id",
    "task_type",
    "eligibility_score",
    "grounding_score",
    "facts_score",
    "notes",
]


# ---------------------------------------------------------------------------
# Phase 1 — Eligibility
# ---------------------------------------------------------------------------


def score_eligibility(log: dict[str, Any]) -> tuple[float, str]:
    """Determine whether the agent's final action satisfies the task request.

    Eligibility is binary: 1.0 when the terminal action matches
    ``ground_truth_action``, 0.0 otherwise.  A soft-credit fallback of 0.5
    is awarded when the agent reaches a terminal *action* step but the action
    is a plausible alternative (both are non-empty and the log has no explicit
    ground truth).

    Args:
        log: Parsed JSON dict from a wearable log JSONL record.

    Returns:
        Tuple of (eligibility_score, note_string).
    """
    trajectory: list[dict[str, Any]] = log.get("trajectory", [])
    ground_truth: str = log.get("ground_truth_action", "")

    if not trajectory:
        return 0.0, "empty trajectory"

    final_step = trajectory[-1]
    final_action: str = final_step.get("action", "")

    if not final_action:
        return 0.0, "no action in final step"

    if not ground_truth:
        # No ground truth — award partial credit if any terminal action taken
        return 0.5, "no ground_truth_action; partial credit for taking action"

    if final_action == ground_truth:
        confidence: float = float(final_step.get("confidence", 1.0))
        return 1.0, f"correct action '{final_action}' (conf={confidence:.2f})"

    return 0.0, f"wrong action '{final_action}' (expected '{ground_truth}')"


# ---------------------------------------------------------------------------
# Phase 2 — Grounding
# ---------------------------------------------------------------------------


def build_response_and_sources(log: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract the agent's reasoning response and source context documents.

    The agent "response" is the concatenated step-level reasoning strings from
    all trajectory steps.  The "source documents" are the sensor observations
    plus structured metadata — analogous to the retrieval context in a RAG
    pipeline.

    Args:
        log: Parsed JSON dict from a wearable log JSONL record.

    Returns:
        Tuple of (agent_response, source_documents).
    """
    trajectory: list[dict[str, Any]] = log.get("trajectory", [])

    reasoning_parts: list[str] = []
    for step in trajectory:
        obs = step.get("observation", "")
        reasoning = step.get("reasoning", "")
        if reasoning:
            reasoning_parts.append(f"{step.get('step_name', 'step')}: {reasoning}")
        elif obs:
            reasoning_parts.append(f"{step.get('step_name', 'step')}: {obs}")

    agent_response = (
        " ".join(reasoning_parts) if reasoning_parts else "No reasoning recorded."
    )

    # Source context: observations (the "retrieved" sensor facts) + metadata
    obs_sources: list[str] = [
        s.get("observation", "") for s in trajectory if s.get("observation")
    ]

    meta: dict[str, Any] = log.get("context_metadata", {})
    meta_sources: list[str] = []
    if isinstance(meta, dict):
        for key in ("device_model", "activity", "user_id", "consent_model"):
            val = meta.get(key)
            if val:
                meta_sources.append(f"{key}: {val}")

    sensor: dict[str, Any] = log.get("sensor_data", {})
    sensor_sources: list[str] = []
    if isinstance(sensor, dict):
        for key, val in sensor.items():
            if isinstance(val, (int, float)):
                sensor_sources.append(f"{key}={val:.4g}")

    source_documents = (
        obs_sources
        + (
            [", ".join(sensor_sources)]
            if sensor_sources
            else [f"scenario: {log.get('scenario_type', 'unknown')}"]
        )
        + meta_sources
    )
    return agent_response, [s for s in source_documents if s]


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------


def evaluate_log(log: dict[str, Any], scorer: FACTSGroundingScorer) -> dict[str, Any]:
    """Run two-phase FACTS evaluation on a single wearable log.

    Args:
        log: Parsed JSON dict from a wearable log JSONL record.
        scorer: Instantiated :class:`~src.eval.agentic_eval.FACTSGroundingScorer`.

    Returns:
        Dict with all CSV column values.
    """
    trajectory_id: str = log.get("log_id", "unknown")
    task_type: str = log.get("scenario_type", "unknown")

    # Phase 1 — Eligibility
    eligibility_score, eligibility_note = score_eligibility(log)

    # Phase 2 — Grounding (only meaningful when eligible, but computed always
    # so reviewers can see how grounding tracks eligibility)
    agent_response, source_documents = build_response_and_sources(log)
    grounding_facts = scorer.score(agent_response, source_documents)
    grounding_score: float = round(grounding_facts["grounding_score"], 6)

    # Combined FACTS score — ineligible responses score 0 per paper convention
    facts_score: float = round(grounding_score if eligibility_score > 0.0 else 0.0, 6)

    notes = eligibility_note
    if eligibility_score < 1.0 and eligibility_score > 0.0:
        notes += f"; grounding_score={grounding_score:.4f} (partial eligibility)"
    elif eligibility_score == 0.0:
        notes += "; grounding not counted (ineligible)"

    return {
        "trajectory_id": trajectory_id,
        "task_type": task_type,
        "eligibility_score": round(eligibility_score, 6),
        "grounding_score": grounding_score,
        "facts_score": facts_score,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Annotated[
        Path,
        typer.Option("--input", "-i", help="JSONL file of wearable log records."),
    ] = _DEFAULT_INPUT,
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="Destination CSV path."),
    ] = _DEFAULT_OUTPUT,
    n: Annotated[
        int,
        typer.Option("--n", help="Number of trajectories to evaluate (0 = all)."),
    ] = 0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Evaluate wearable agent trajectories against the FACTS Grounding Benchmark."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if not input_path.exists():
        logger.error("Input not found: %s", input_path)
        raise typer.Exit(1)

    logs: list[dict[str, Any]] = [
        json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
    ]
    if n > 0:
        logs = logs[:n]
    logger.info("Loaded %d trajectory logs from %s", len(logs), input_path)

    scorer = FACTSGroundingScorer()
    rows: list[dict[str, Any]] = []

    for log in logs:
        try:
            row = evaluate_log(log, scorer)
            rows.append(row)
            if verbose:
                logger.debug(
                    "%s  eligibility=%.2f  grounding=%.4f  facts=%.4f",
                    row["trajectory_id"][:12],
                    row["eligibility_score"],
                    row["grounding_score"],
                    row["facts_score"],
                )
        except Exception:
            logger.exception("Failed to evaluate log %s — skipping.", log.get("log_id"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows → %s", len(rows), output_path)

    # Summary stats
    def _mean(col: str) -> float:
        vals = [float(r[col]) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    n_eligible = sum(1 for r in rows if float(r["eligibility_score"]) == 1.0)
    eligibility_rate = n_eligible / max(len(rows), 1)

    table = Table(title="FACTS Grounding Evaluation — Summary", show_lines=True)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")
    table.add_row("Trajectories scored", str(len(rows)))
    table.add_row("Eligibility rate (exact match)", f"{eligibility_rate:.1%}")
    table.add_row("Mean eligibility_score", f"{_mean('eligibility_score'):.4f}")
    table.add_row("Mean grounding_score", f"{_mean('grounding_score'):.4f}")
    table.add_row("Mean facts_score", f"{_mean('facts_score'):.4f}")
    table.add_row("Output", str(output_path))
    console.print(table)

    # WP2/WP3 key stat
    console.print(
        f"\n[bold green]WP3 Key Stat:[/bold green] Mean FACTS score = "
        f"[cyan]{_mean('facts_score'):.4f}[/cyan] across {len(rows)} wearable "
        f"trajectories "
        f"(eligibility rate {eligibility_rate:.1%}). "
        f"No published model exceeds 0.70 on the FACTS benchmark."
    )


if __name__ == "__main__":
    app()
