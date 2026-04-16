"""Run the full trajectory scoring pipeline against synthetic wearable data.

Produces:
  data/processed/trajectory_scores.json     — per-trajectory 5-layer scores
  data/processed/nondeterminism_report.json — per-scenario variance report

WP2 Section 3 empirical backing.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.eval.trajectory_scorer import TrajectoryScorer, _load_trajectories

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()

JSONL_PATH = Path("data/raw/synthetic_wearable_logs.jsonl")
SCORES_OUT = Path("data/processed/trajectory_scores.json")
REPORT_OUT = Path("data/processed/nondeterminism_report.json")


def main() -> None:
    # 1. Load logs
    logger.info("Loading trajectories from %s", JSONL_PATH)
    logs = _load_trajectories(JSONL_PATH, limit=None)
    logger.info("Loaded %d trajectories", len(logs))

    scorer = TrajectoryScorer(dry_run=True)

    # 2. Batch score all 100 logs
    logger.info("Scoring %d trajectories…", len(logs))
    results = scorer.batch_score(logs)

    # 3. Save trajectory_scores.json
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": True,
        "n_trajectories": len(results),
        "scores": [r.to_dict() for r in results],
    }
    SCORES_OUT.write_text(json.dumps(scores_payload, indent=2))
    logger.info("Wrote scores → %s", SCORES_OUT)

    # 4. Group by scenario_type
    groups: dict[str, list] = defaultdict(list)
    for log, ts in zip(logs, results):
        groups[str(log.scenario_type)].append((log, ts))

    # 5. Compute nondeterminism variance per scenario group (≥3 runs)
    scenario_groups: dict[str, dict] = {}
    for scenario, pairs in sorted(groups.items()):
        scenario_logs = [p[0] for p in pairs]
        scenario_scores = [p[1] for p in pairs]
        n = len(scenario_logs)
        mean_wt = statistics.mean(s.weighted_total for s in scenario_scores)

        if n >= 3:
            variance = scorer.compute_nondeterminism_variance(scenario, scenario_logs)
            scenario_groups[scenario] = {
                "n_runs": n,
                "mean_weighted_total": round(mean_wt, 6),
                "score_std": round(float(variance["score_std"]), 6),
                "max_variance_layer": variance["max_variance_layer"],
                "pia_planning_std": round(float(variance["pia_planning_std"]), 6),
                "pia_recovery_std": round(float(variance["pia_recovery_std"]), 6),
                "pia_goal_std": round(float(variance["pia_goal_std"]), 6),
                "pia_tool_std": round(float(variance["pia_tool_std"]), 6),
            }
            logger.info(
                "  %s: n=%d mean=%.4f score_std=%.4f max_layer=%s",
                scenario, n, mean_wt,
                float(variance["score_std"]),
                variance["max_variance_layer"],
            )
        else:
            logger.info("  %s: n=%d — skipped (< 3 runs)", scenario, n)

    # Build headline: scenario + layer with highest score_std
    if scenario_groups:
        worst = max(scenario_groups.items(), key=lambda kv: kv[1]["score_std"])
        headline = (
            f"Highest nondeterminism in {worst[1]['max_variance_layer']} layer "
            f"(std={worst[1]['score_std']:.4f}) — scenario: {worst[0]}"
        )
    else:
        headline = "No scenario groups with ≥3 runs."

    # 6. Save nondeterminism_report.json
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "total_trajectories": len(logs),
        "scenario_groups": scenario_groups,
        "headline": headline,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    logger.info("Wrote nondeterminism report → %s", REPORT_OUT)

    # 7. Rich table
    table = Table(title="Trajectory Nondeterminism by Scenario", show_lines=True)
    table.add_column("scenario", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("mean_score", justify="right")
    table.add_column("score_std", justify="right")
    table.add_column("max_variance_layer", style="yellow")

    for scenario, stats in sorted(scenario_groups.items()):
        table.add_row(
            scenario,
            str(stats["n_runs"]),
            f"{stats['mean_weighted_total']:.4f}",
            f"{stats['score_std']:.4f}",
            stats["max_variance_layer"],
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold]Headline:[/bold] {headline}")
    console.print(
        f"\n[dim]trajectory_scores.json → {SCORES_OUT}\n"
        f"nondeterminism_report.json → {REPORT_OUT}[/dim]"
    )


if __name__ == "__main__":
    main()
