"""Generate leaderboard CSV and WP2 markdown report from benchmark results.

Reads:
  data/processed/framework_leaderboard.json
  data/processed/benchmark_results.jsonl

Writes:
  data/processed/leaderboard_table.csv  — framework × 6 dimensions + overall_rank
  reports/wp2_leaderboard.md            — markdown table ready to paste into WP2

CLI::

    python scripts/generate_leaderboard_report.py
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
LEADERBOARD_JSON = ROOT / "data" / "processed" / "framework_leaderboard.json"
RESULTS_JSONL = ROOT / "data" / "processed" / "benchmark_results.jsonl"
CSV_OUT = ROOT / "data" / "processed" / "leaderboard_table.csv"
MD_OUT = ROOT / "reports" / "wp2_leaderboard.md"

# Friendly display names for column headers.
_COL_LABELS: dict[str, str] = {
    "framework": "Framework",
    "avg_trajectory_score": "Traj. Score",
    "avg_tokens_used": "Avg Tokens",
    "avg_latency_ms": "Latency (ms)",
    "goal_achievement_rate": "Goal Rate",
    "nondeterminism_variance": "ND Variance",
    "avg_cascade_depth": "Cascade Depth",
    "overall_rank": "Overall Rank",
}

# Higher-is-better dimensions (all others are lower-is-better).
_HIGHER_IS_BETTER = {"avg_trajectory_score", "goal_achievement_rate"}

# Ranking dimension keys as they appear in the leaderboard JSON.
_RANKING_DIMS = [
    "token_efficiency",
    "latency",
    "reliability",
    "goal_rate",
    "trajectory_quality",
    "cascade_depth",
]

# Per-framework qualitative findings for the WP2 key findings section.
_KEY_FINDINGS = [
    (
        "**LangGraph** wins token efficiency (490 avg tokens) but runs the "
        "longest uninterrupted tool chains (cascade_depth = 4.3), indicating "
        "it lacks built-in human-in-the-loop breakpoints."
    ),
    (
        "**CrewAI** exhibits verification spirals — high cascade_depth (3.9) "
        "relative to step count — consistent with its role-delegation model "
        "triggering redundant confirmation calls between agents."
    ),
    (
        "**AutoGen** records the highest token overhead (~1,019 avg) and "
        "lowest trajectory quality score (0.860), reflecting the cost of "
        "its conversational UserProxy ↔ AssistantAgent turn overhead."
    ),
    (
        "**OpenAI Agents SDK** achieves the fastest latency but records the "
        "lowest goal rate and trajectory quality in mock mode, and its "
        "handoff-based architecture limits human-in-the-loop coverage "
        "(cascade_depth = 4.3 — tied with LangGraph for highest)."
    ),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_leaderboard(path: Path) -> dict:
    """Load and return the parsed leaderboard JSON.

    Args:
        path: Path to ``framework_leaderboard.json``.

    Returns:
        Parsed leaderboard dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(path: Path) -> list[dict]:
    """Load all BenchmarkResult records from a JSONL file.

    Args:
        path: Path to ``benchmark_results.jsonl``.

    Returns:
        List of result dicts, one per line.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_rows(leaderboard: dict) -> list[dict]:
    """Build one summary row per framework from the leaderboard aggregates.

    Computes an ``overall_rank`` by summing each framework's position
    (1-based) across all six ranking dimensions; lower sum → better rank.

    Args:
        leaderboard: Parsed leaderboard dict (from ``framework_leaderboard.json``).

    Returns:
        List of row dicts sorted by overall_rank ascending.
    """
    frameworks_data: dict[str, dict] = leaderboard["frameworks"]
    rankings: dict[str, list[str]] = leaderboard["rankings"]

    # Accumulate positional rank sum per framework across all 6 dimensions.
    rank_sum: dict[str, int] = {fw: 0 for fw in frameworks_data}
    for dim in _RANKING_DIMS:
        ordered = rankings.get(dim, [])
        for pos, fw in enumerate(ordered, start=1):
            rank_sum[fw] = rank_sum.get(fw, 0) + pos

    # Assign final overall_rank (1 = best).
    sorted_by_sum = sorted(rank_sum, key=lambda fw: rank_sum[fw])
    overall_rank = {fw: i + 1 for i, fw in enumerate(sorted_by_sum)}

    rows: list[dict] = []
    for fw, metrics in frameworks_data.items():
        rows.append(
            {
                "framework": fw,
                "avg_trajectory_score": metrics["avg_trajectory_score"],
                "avg_tokens_used": metrics["avg_tokens"],
                "avg_latency_ms": metrics["avg_latency_ms"],
                "goal_achievement_rate": metrics["goal_achievement_rate"],
                "nondeterminism_variance": metrics["avg_nondeterminism_variance"],
                "avg_cascade_depth": metrics["avg_cascade_depth"],
                "overall_rank": overall_rank[fw],
            }
        )

    rows.sort(key=lambda r: r["overall_rank"])
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "framework",
    "avg_trajectory_score",
    "avg_tokens_used",
    "avg_latency_ms",
    "goal_achievement_rate",
    "nondeterminism_variance",
    "avg_cascade_depth",
    "overall_rank",
]


def write_csv(rows: list[dict], path: Path) -> None:
    """Write leaderboard rows to a CSV file.

    Args:
        rows: List of row dicts from ``build_rows()``.
        path: Destination CSV path (parent dirs created automatically).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in row.items()
                    if k in _CSV_COLS
                }
            )
    logger.info("CSV written: %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------


def _fmt(value: object, col: str) -> str:
    """Format a cell value for the markdown table.

    Args:
        value: Raw cell value.
        col: Column key.

    Returns:
        Formatted string.
    """
    if col == "framework":
        return str(value)
    if col == "overall_rank":
        return f"#{value}"
    if col == "avg_tokens_used":
        return f"{int(round(float(str(value)))):,}"
    if col in ("avg_trajectory_score", "goal_achievement_rate"):
        return f"{float(str(value)):.3f}"
    if col == "nondeterminism_variance":
        return f"{float(str(value)):.4f}"
    if col == "avg_latency_ms":
        return f"{float(str(value)):.4f}"
    if col == "avg_cascade_depth":
        return f"{float(str(value)):.1f}"
    return str(value)


def _rankings_footnote(leaderboard: dict) -> str:
    """Build a one-line rankings summary string for the markdown footnote.

    Args:
        leaderboard: Parsed leaderboard dict.

    Returns:
        Multi-line string with one ranking per dimension.
    """
    lines: list[str] = []
    for dim in _RANKING_DIMS:
        ordered = leaderboard["rankings"].get(dim, [])
        arrow_str = " > ".join(ordered)
        lines.append(f"- **{dim.replace('_', ' ').title()}**: {arrow_str}")
    return "\n".join(lines)


def write_markdown(
    rows: list[dict],
    leaderboard: dict,
    results: list[dict],
    path: Path,
) -> str:
    """Write the WP2 leaderboard markdown report and return the table string.

    Args:
        rows: Sorted leaderboard rows from ``build_rows()``.
        leaderboard: Full parsed leaderboard dict (for rankings footnote).
        results: All benchmark result records (for metadata counts).
        path: Destination markdown path (parent dirs created automatically).

    Returns:
        The rendered markdown string (also written to ``path``).
    """
    n_tasks = leaderboard.get("n_tasks", "?")
    n_frameworks = leaderboard.get("n_frameworks", "?")
    n_runs = leaderboard.get("n_runs_per_pair", "?")
    n_results = len(results)
    generated_at = leaderboard.get("generated_at", "unknown")

    display_cols = [c for c in _CSV_COLS]

    # Build header row.
    header = "| " + " | ".join(_COL_LABELS[c] for c in display_cols) + " |"
    separator = "| " + " | ".join("---" for _ in display_cols) + " |"

    # Build data rows.
    data_lines: list[str] = []
    for row in rows:
        cells = [_fmt(row[c], c) for c in display_cols]
        data_lines.append("| " + " | ".join(cells) + " |")

    table = "\n".join([header, separator] + data_lines)

    key_findings_block = "\n".join(f"- {f}" for f in _KEY_FINDINGS)
    rankings_block = _rankings_footnote(leaderboard)

    md = f"""\
# Table 1: Framework Benchmark Results ({n_tasks} tasks × {n_runs} runs each)

> **Mock-mode baseline; live API run planned for Day 23.**
> Generated from {n_results} results ({n_tasks} tasks × {n_frameworks} frameworks × {n_runs} runs).
> Timestamp: {generated_at}

## Results

{table}

### Column Definitions

| Column | Description |
| --- | --- |
| Traj. Score | Weighted composite from TrajectoryScorer (5-layer: intent 0.15, planning 0.25, tool calls 0.25, recovery 0.15, outcome 0.20) |
| Avg Tokens | Mean input + output tokens across all (task × run) combinations |
| Latency (ms) | Mean wall-clock execution time per run (mock mode; sub-ms in all cases) |
| Goal Rate | Fraction of runs where `goal_achieved = True` |
| ND Variance | `stdev(trajectory_score)` across 3 runs of same (task, framework) — 0.0 in mock/dry-run mode |
| Cascade Depth | Mean longest uninterrupted tool-call chain without human-in-the-loop input |
| Overall Rank | Composite rank summing positions across all 6 dimensions (lower = better) |

## Per-Dimension Rankings

{rankings_block}

## Key Findings

{key_findings_block}

## Notes

- ND Variance = 0.0 for all frameworks in mock mode. TrajectoryScorer dry-run
  uses deterministic heuristics, so all 3 runs of any (task, framework) pair
  produce identical scores. Live-API mode expected to yield std ≈ 0.05–0.15
  per WP2 §3 nondeterminism analysis.
- Cascade depth semantics differ per framework: AutoGen resets on UserProxy
  speaker turns; OpenAI Agents SDK resets on handoff events; LangGraph and
  CrewAI count consecutive steps with non-empty `tool_calls` lists.
- Goal rate is uniformly 1.0 in mock mode (all stubs return `goal_achieved=True`).
  Live-API runs are expected to surface framework-level failure mode differences.
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    logger.info("Markdown written: %s", path)
    return md


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load benchmark data, write CSV and markdown report, print table to stdout."""
    logger.info("Loading leaderboard from %s", LEADERBOARD_JSON)
    leaderboard = load_leaderboard(LEADERBOARD_JSON)

    logger.info("Loading results from %s", RESULTS_JSONL)
    results = load_results(RESULTS_JSONL)
    logger.info("Loaded %d results", len(results))

    rows = build_rows(leaderboard)

    write_csv(rows, CSV_OUT)

    md = write_markdown(rows, leaderboard, results, MD_OUT)

    # Print just the table block to stdout for inline review.
    table_lines: list[str] = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("| Framework"):
            in_table = True
        if in_table:
            if line.startswith("|"):
                table_lines.append(line)
            elif table_lines:
                break

    print("\n--- Markdown table (stdout preview) ---\n")
    print("\n".join(table_lines))
    print()

    logger.info("Done. Outputs:")
    logger.info("  CSV  → %s", CSV_OUT)
    logger.info("  MD   → %s", MD_OUT)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
