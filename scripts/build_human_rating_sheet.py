"""Build a blind human-rating sheet from already-scored GLEE PIA clusters.

Zero API calls: reads only already-committed local files --
``data/rsi_bench/glee_pia_stratified_scored.jsonl`` (judge scores, read
ONLY to build the answer key -- never rendered into the sheet itself),
``data/rsi_bench/glee_pia_analysis.json`` (for the 12 degenerate
(cluster, dimension) entries to exclude), and
``tests/experiments/glee/trajectories.jsonl.gz`` (the frozen raw
trajectory log, for each selected member's game_state/valid_actions/
reasoning/action -- the exact same raw-log index machinery
:mod:`src.rsi_bench.analyze_stratified_pia` uses for the action-only
baseline).

Selection
---------------------------------------------------------------------------
Population: every (cluster_id, member_id) pair drawn from a scored
cluster's ``sampled_member_ids`` (the exact scored members), EXCLUDING:

- any cluster_id that owns at least one of the 12 degenerate
  (cluster_id, dimension) entries from ``glee_pia_analysis.json``'s
  ``degenerate_entries`` -- 10 distinct cluster_ids
  (bargaining_062, negotiation_002/064/081,
  persuasion_011/015/103/113/184/188). A whole cluster is dropped, not
  just the degenerate dimension, since the task is "exclude the 12
  degenerate entries" and a member cannot be presented as "from a
  non-degenerate cluster" while its cluster partially is one.
- any member whose raw trajectory step has ``fallback_used: true``.
  ``sampled_member_ids`` is already drawn from each cluster's
  ``all_non_fallback_member_ids`` (so this should never actually trigger),
  but the task explicitly says to exclude fallback members, so this is
  checked directly against ground truth rather than assumed.

20 members are drawn stratified 7 bargaining / 7 negotiation / 6
persuasion, from one ``random.Random(20261002)`` instance consumed in
fixed family order (bargaining, negotiation, persuasion) via
``rng.sample(sorted(pool), k)`` per family -- the sorted pool makes the
draw a pure function of (seed, pool membership), independent of Python's
dict/set iteration order.

Outputs
---------------------------------------------------------------------------
- ``data/rsi_bench/human_rating_sheet.md`` -- the rubric anchors (shown
  once, verbatim from what the judges were actually given -- see
  :data:`_RUBRIC_TEXT`), then 20 cases (family, game state summary, valid
  actions, reasoning, action, and 3 blank 1-5 rating fields). NO judge
  score of any kind appears in this file.
- ``data/rsi_bench/human_rating_key.json`` -- case_id -> {cluster_id,
  member_id, game_id, round, phase, and the panel's actual raw scores +
  kappa for the 3 rated dimensions}, kept separate so a human rater
  given only the ``.md`` file cannot see the answer.

CLI:
    python -m scripts.build_human_rating_sheet
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

_SCORED_INPUT = Path("data/rsi_bench/glee_pia_stratified_scored.jsonl")
_ANALYSIS_INPUT = Path("data/rsi_bench/glee_pia_analysis.json")
_RAW_TRAJECTORIES_INPUT = Path("tests/experiments/glee/trajectories.jsonl.gz")
_SHEET_OUTPUT = Path("data/rsi_bench/human_rating_sheet.md")
_KEY_OUTPUT = Path("data/rsi_bench/human_rating_key.json")

_SEED = 20261002
_GAME_FAMILIES: tuple[str, ...] = ("bargaining", "negotiation", "persuasion")
_STRATUM_COUNTS: dict[str, int] = {"bargaining": 7, "negotiation": 7, "persuasion": 6}
_RATED_DIMENSIONS: tuple[str, ...] = (
    "valuation_reasoning",
    "horizon_strategy_planning",
    "outcome_consistency",
)
_JUDGES: tuple[str, ...] = (
    "GameTheoreticRigor",
    "LiteralGroundedness",
    "OpponentResponsiveness",
)

# The rubric as it was ACTUALLY given to the judges (src/rsi_bench/
# glee_pia_baseline.py's _SYSTEM_PROMPTS): a shared 1-5 scale whose only
# defined anchors are the two endpoints, plus the dimension's role as
# described in that same shared instruction text. No per-level (2/3/4)
# anchor text exists anywhere in the codebase to show -- stated here
# rather than invented.
_RUBRIC_TEXT = """\
## Rubric (verbatim from what the judges were given)

Each dimension is scored on a **1-5 integer scale: 5 = excellent, 1 = badly
wrong**. This is the ONLY scale-anchor text that exists anywhere in the
judge prompts -- there is no further per-level (2/3/4) definition to show;
showing one here would be inventing detail the judges themselves were never
given.

The 3 dimensions you are rating, exactly as described in the judges'
shared instructions:

- **valuation_reasoning** -- does the reasoning correctly derive the
  agent's reservation value / expected value from the `game_state` it was
  given?
- **horizon_strategy_planning** -- does the strategy correctly account
  for the round horizon (concession schedules, inflation/decay effects,
  endgame behavior near `max_rounds`)?
- **outcome_consistency** -- are the reasoning and the final action
  mutually consistent (the agent doesn't reason its way to one action and
  then take a different one)?

You are NOT shown the panel's scores, each other's ratings, or which
cluster/game this case came from -- rate each case strictly on the
reasoning and action shown below it.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_raw_index(path: Path) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Same (game_id, round, phase) -> raw step index as
    :func:`src.rsi_bench.analyze_stratified_pia.load_raw_steps_index`."""
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            key = (raw["game_id"], int(raw["round"]), raw["phase"])
            index.setdefault(key, raw)
    return index


def parse_member_id(member_id: str) -> tuple[str, int, str]:
    game_id, round_str, phase = member_id.split(":", 2)
    return game_id, int(round_str), phase


def excluded_cluster_ids(analysis: dict[str, Any]) -> set[str]:
    return {e["cluster_id"] for e in analysis["degenerate_entries"]}


def build_population(
    scored_records: list[dict[str, Any]],
    raw_index: dict[tuple[str, int, str], dict[str, Any]],
    excluded_clusters: set[str],
) -> dict[str, list[tuple[str, str]]]:
    """``game_family -> sorted [(cluster_id, member_id), ...]`` pool, after
    dropping excluded clusters and any member whose raw step is a
    confirmed fallback."""
    pool: dict[str, list[tuple[str, str]]] = {family: [] for family in _GAME_FAMILIES}
    for record in scored_records:
        cluster_id = record["cluster_id"]
        if cluster_id in excluded_clusters:
            continue
        family = record["game_family"]
        for member_id in record["sampled_member_ids"]:
            game_id, round_num, phase = parse_member_id(member_id)
            raw_step = raw_index.get((game_id, round_num, phase))
            if raw_step is None or raw_step.get("fallback_used"):
                continue
            pool[family].append((cluster_id, member_id))
    for family in _GAME_FAMILIES:
        pool[family].sort()
    return pool


def select_cases(
    pool: dict[str, list[tuple[str, str]]], seed: int = _SEED
) -> list[tuple[str, str, str]]:
    """``[(game_family, cluster_id, member_id), ...]``, 20 total, stratified
    7/7/6, one random.Random(seed) consumed in fixed family order."""
    rng = random.Random(seed)
    selected: list[tuple[str, str, str]] = []
    for family in _GAME_FAMILIES:
        k = _STRATUM_COUNTS[family]
        chosen = rng.sample(pool[family], k=k)
        for cluster_id, member_id in chosen:
            selected.append((family, cluster_id, member_id))
    return selected


def render_sheet(
    cases: list[tuple[str, str, str]],
    raw_index: dict[tuple[str, int, str], dict[str, Any]],
) -> str:
    lines: list[str] = ["# GLEE PIA -- Human Rating Sheet", ""]
    lines.append(
        "20 cases, stratified 7 bargaining / 7 negotiation / 6 persuasion, "
        f"seed={_SEED}, drawn from already-scored clusters excluding the 12 "
        "degenerate (cluster, dimension) entries and any fallback member. "
        "No judge score is shown anywhere below."
    )
    lines.append("")
    lines.append(_RUBRIC_TEXT)

    for i, (family, _cluster_id, member_id) in enumerate(cases, start=1):
        game_id, round_num, phase = parse_member_id(member_id)
        raw_step = raw_index[(game_id, round_num, phase)]
        game_state = raw_step["game_state"]

        lines.append(f"## Case {i:02d}")
        lines.append("")
        lines.append(
            f"**Family:** {family}  **Phase:** {phase}  **Round:** {round_num}"
        )
        lines.append("")
        lines.append("**Game state:**")
        lines.append("```json")
        lines.append(json.dumps(game_state, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        lines.append("**Valid actions (schema for this turn):**")
        lines.append("```json")
        lines.append(json.dumps(raw_step["valid_actions"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        lines.append("**Agent's reasoning:**")
        lines.append("")
        lines.append(f"> {raw_step['reasoning']}")
        lines.append("")
        lines.append("**Agent's action:**")
        lines.append("```json")
        lines.append(json.dumps(raw_step["action"], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
        lines.append("**Your ratings (1-5 each):**")
        lines.append("")
        lines.append("| dimension | your score |")
        lines.append("|---|---|")
        for dim in _RATED_DIMENSIONS:
            lines.append(f"| {dim} | ____ |")
        lines.append("")

    return "\n".join(lines)


def build_key(
    cases: list[tuple[str, str, str]],
    scored_records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_cluster_id = {r["cluster_id"]: r for r in scored_records}
    key: dict[str, Any] = {}
    for i, (family, cluster_id, member_id) in enumerate(cases, start=1):
        record = by_cluster_id[cluster_id]
        game_id, round_num, phase = parse_member_id(member_id)
        judge_scores = {
            rs["judge_name"]: {dim: rs.get(dim) for dim in _RATED_DIMENSIONS}
            for rs in record["raw_scores"]
            if rs["member_id"] == member_id
        }
        key[f"case_{i:02d}"] = {
            "game_family": family,
            "cluster_id": cluster_id,
            "member_id": member_id,
            "game_id": game_id,
            "round": round_num,
            "phase": phase,
            "judge_scores": judge_scores,
            "cluster_kappa_per_dimension": {
                dim: record["kappa_per_dimension"].get(dim) for dim in _RATED_DIMENSIONS
            },
        }
    return key


def main() -> None:
    scored_records = load_jsonl(_SCORED_INPUT)
    analysis = json.loads(_ANALYSIS_INPUT.read_text())
    raw_index = load_raw_index(_RAW_TRAJECTORIES_INPUT)
    excluded = excluded_cluster_ids(analysis)

    pool = build_population(scored_records, raw_index, excluded)
    cases = select_cases(pool)

    _SHEET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _SHEET_OUTPUT.write_text(render_sheet(cases, raw_index))
    _KEY_OUTPUT.write_text(json.dumps(build_key(cases, scored_records), indent=2))

    print(f"Excluded {len(excluded)} degenerate cluster_ids: {sorted(excluded)}")
    print("Pool sizes:")
    for family in _GAME_FAMILIES:
        print(f"  {family}: {len(pool[family])} eligible members")
    print(f"Sheet written to {_SHEET_OUTPUT}")
    print(f"Key written to {_KEY_OUTPUT}")


if __name__ == "__main__":
    main()
