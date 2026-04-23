"""Day 41 Kaggle FACTS leaderboard submission artifact.

Loads wearable agent trajectory scores, runs each through
FACTSGroundingScorer across three factuality dimensions
(parametric, search, grounding), and writes a CSV in the format
expected by the DeepMind FACTS Grounding Benchmark Kaggle competition.

Output: results/facts_kaggle_submission.csv

Usage::

    uv run python scripts/kaggle_facts_submission.py
    uv run python scripts/kaggle_facts_submission.py --n 50
"""

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer

# Ensure project root is importable when called directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.eval.agentic_eval import FACTSGroundingScorer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TRAJECTORY_SCORES = _ROOT / "data" / "processed" / "trajectory_scores.json"
_OUTPUT_CSV = _ROOT / "results" / "facts_kaggle_submission.csv"
_CSV_COLUMNS = [
    "id",
    "parametric_score",
    "search_score",
    "grounding_score",
    "overall_facts_score",
]


def _build_response_and_sources(entry: dict[str, Any]) -> tuple[str, list[str]]:
    """Derive a natural-language response and source documents from a score entry.

    The trajectory scorer records per-layer reasoning strings. We concatenate
    them as the agent "response" and treat the scenario metadata as the
    source context — a proxy for the retrieval context that would be present
    in a live deployment.

    Args:
        entry: One element from trajectory_scores.json ``scores`` list.

    Returns:
        Tuple of (agent_response, source_documents).
    """
    layers = ["intent", "planning", "tool_calls", "recovery", "outcome"]
    reasoning_parts: list[str] = []
    for layer in layers:
        layer_data = entry.get(layer) or {}
        reasoning = layer_data.get("reasoning", "")
        if reasoning:
            reasoning_parts.append(f"{layer}: {reasoning}")

    agent_response = (
        " ".join(reasoning_parts) if reasoning_parts else "No reasoning recorded."
    )

    meta = entry.get("metadata", {})
    source_documents = [
        f"Scenario type: {meta.get('scenario_type', 'unknown')}.",
        f"Consent model: {meta.get('consent_model', 'unknown')}.",
        f"Steps taken: {meta.get('n_steps', 0)}.",
        f"Trajectory weighted total: {entry.get('weighted_total', 0.0):.4f}.",
    ]
    return agent_response, source_documents


def run(
    n: int = typer.Option(10, "--n", help="Number of trajectories to score."),
    input_path: Path = typer.Option(
        _TRAJECTORY_SCORES, "--input", help="Path to trajectory_scores.json."
    ),
    output_path: Path = typer.Option(
        _OUTPUT_CSV, "--output", help="Destination CSV path."
    ),
) -> None:
    """Score trajectories and write a Kaggle FACTS submission CSV."""
    logger.info("Loading trajectories from %s", input_path)
    with input_path.open() as fh:
        data = json.load(fh)

    scores_list: list[dict[str, Any]] = data["scores"]
    selected = scores_list[:n]
    logger.info("Scoring %d / %d trajectories", len(selected), len(scores_list))

    scorer = FACTSGroundingScorer()
    rows: list[dict[str, float | str]] = []

    for entry in selected:
        trajectory_id: str = entry["trajectory_id"]
        agent_response, source_documents = _build_response_and_sources(entry)
        facts = scorer.score(agent_response, source_documents)
        rows.append(
            {
                "id": trajectory_id,
                "parametric_score": round(facts["parametric_score"], 6),
                "search_score": round(facts["search_score"], 6),
                "grounding_score": round(facts["grounding_score"], 6),
                "overall_facts_score": round(facts["overall_facts_score"], 6),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows → %s", len(rows), output_path)

    # Summary
    dims = [
        "parametric_score", "search_score", "grounding_score", "overall_facts_score"
    ]
    means = {dim: sum(float(r[dim]) for r in rows) / len(rows) for dim in dims}

    print("\n── FACTS Submission Summary ──────────────────────────────")
    print(f"  Trajectories scored : {len(rows)}")
    print(f"  parametric_score    : {means['parametric_score']:.4f}")
    print(f"  search_score        : {means['search_score']:.4f}")
    print(f"  grounding_score     : {means['grounding_score']:.4f}")
    print(f"  overall_facts_score : {means['overall_facts_score']:.4f}")
    print(f"  Output              : {output_path}")
    print("──────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    typer.run(run)
