"""Compare a human author's blind ratings against the GLEE PIA judge panel.

Pure analysis, zero API calls: reads only two already-committed local
files -- ``data/rsi_bench/author_ratings.csv`` (the author's 1-5 ratings
on the 20 cases in ``data/rsi_bench/human_rating_sheet.md``) and
``data/rsi_bench/human_rating_key.json`` (the answer key for those same
20 cases: cluster_id/member_id/family plus each of the 3 judges' raw
scores). There is no Anthropic import anywhere in this module.

DESIGN DECISIONS (read before changing anything)
--------------------------------------------------------------------------
Case identity
    The CSV's ``case_id`` (plain integer 1-20) maps to the key's
    ``f"case_{case_id:02d}"``. Joining is a direct dict lookup -- there is
    no fuzzy matching and a missing key raises immediately rather than
    silently dropping a case.

"undefined (constant)"
    Spearman's rho is undefined when either input is constant (zero
    variance) -- :func:`scipy.stats.spearmanr` returns ``nan`` in that
    case, which this module never passes through silently. Every Spearman
    computation here (point estimate AND the point estimate of each
    bootstrap repetition) checks ``len(set(values)) > 1`` on BOTH sides
    first and reports the literal string ``"undefined (constant)"``
    instead of a fabricated number. This is not a hypothetical edge case
    in this dataset: the author's own ``outcome_consistency`` column is
    5 on all 20 cases (constant by construction), so every
    outcome_consistency Spearman figure against the author is genuinely
    undefined, overall and in every family.

Rounding (agreement only, never offset)
    The judge MEAN of 3 integer 1-5 scores is not generally an integer
    (e.g. 3.33). Exact/within-1 agreement rounds it first (Python's
    banker's-rounding ``round()``) since "agreement" is inherently a
    same-integer comparison; the MEAN OFFSET statistic uses the raw,
    unrounded judge mean throughout, since an offset is a continuous
    quantity and rounding it first would just be a different, less
    precise statistic -- the task scopes "(rounded)" to agreement only.

Bootstrap determinism
    One ``random.Random(seed)`` instance is consumed in a single fixed
    nested order -- dimension (``_DIMENSIONS`` order), then scope
    (``"overall"``, then ``_GAME_FAMILIES`` order), then comparison target
    in fixed order (3 judges, then judge_mean) -- across BOTH the
    author-vs-target Spearman bootstrap and the author-vs-target
    mean-offset bootstrap, so two runs of :func:`run_analysis` with the
    same seed are byte-identical. Each bootstrap repetition resamples
    case INDICES (not per-dimension values) with replacement, the same
    size as the scope's case count; a repetition that lands on a
    constant resample for either side contributes no value to that
    cell's distribution (never a fabricated one).
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import typer
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="analyze-author-ratings",
    help=(
        "Compare blind author ratings against the GLEE PIA judge panel. Zero API calls."
    ),
    add_completion=False,
)

_DIMENSIONS: tuple[str, ...] = (
    "valuation_reasoning",
    "horizon_strategy_planning",
    "outcome_consistency",
)
_JUDGES: tuple[str, ...] = (
    "GameTheoreticRigor",
    "LiteralGroundedness",
    "OpponentResponsiveness",
)
_GAME_FAMILIES: tuple[str, ...] = ("bargaining", "negotiation", "persuasion")
_JUDGE_MEAN_KEY = "judge_mean"
_COMPARISON_TARGETS: tuple[str, ...] = (*_JUDGES, _JUDGE_MEAN_KEY)

_DEFAULT_SEED = 20261002
_DEFAULT_N_RESAMPLES = 10_000
_CONSTANT_NOTE = "undefined (constant)"

_DEFAULT_CSV_INPUT = Path("data/rsi_bench/author_ratings.csv")
_DEFAULT_KEY_INPUT = Path("data/rsi_bench/human_rating_key.json")
_DEFAULT_JSON_OUTPUT = Path("data/rsi_bench/author_ratings_analysis.json")
_DEFAULT_MARKDOWN_OUTPUT = Path("data/rsi_bench/author_ratings_analysis.md")


# ---------------------------------------------------------------------------
# Loading + joining
# ---------------------------------------------------------------------------


def load_author_ratings(path: Path = _DEFAULT_CSV_INPUT) -> list[dict[str, Any]]:
    """Load ``author_ratings.csv`` -- one row per case, in file order."""
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for raw_row in csv.DictReader(f):
            rows.append(
                {
                    "case_id": int(raw_row["case_id"]),
                    **{dim: int(raw_row[dim]) for dim in _DIMENSIONS},
                }
            )
    return rows


def load_rating_key(path: Path = _DEFAULT_KEY_INPUT) -> dict[str, Any]:
    """Load ``human_rating_key.json``."""
    return json.loads(Path(path).read_text())


def join_records(
    author_rows: list[dict[str, Any]], key: dict[str, Any]
) -> list[dict[str, Any]]:
    """Join each author row to its key entry by ``f"case_{case_id:02d}"``.

    Returns:
        One dict per case, in ``author_rows`` order:
        ``{"case_id", "game_family", "author": {dim: int}, "judges":
        {judge: {dim: int}}, "judge_mean": {dim: float}}``. Raises
        ``KeyError`` immediately if a case_id has no matching key entry
        (never silently dropped).
    """
    joined: list[dict[str, Any]] = []
    for row in author_rows:
        case_key = f"case_{row['case_id']:02d}"
        entry = key[case_key]
        judges = entry["judge_scores"]
        judge_mean = {
            dim: sum(judges[j][dim] for j in _JUDGES) / len(_JUDGES)
            for dim in _DIMENSIONS
        }
        joined.append(
            {
                "case_id": row["case_id"],
                "game_family": entry["game_family"],
                "author": {dim: row[dim] for dim in _DIMENSIONS},
                "judges": judges,
                "judge_mean": judge_mean,
            }
        )
    return joined


def _scopes(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """``"overall" -> all records`` plus one entry per family, in fixed order."""
    scoped: dict[str, list[dict[str, Any]]] = {"overall": records}
    for family in _GAME_FAMILIES:
        scoped[family] = [r for r in records if r["game_family"] == family]
    return scoped


def _target_value(record: dict[str, Any], dim: str, target: str) -> float:
    """One record's score for a comparison target (a judge name, or judge_mean)."""
    if target == _JUDGE_MEAN_KEY:
        return record["judge_mean"][dim]
    return record["judges"][target][dim]


# ---------------------------------------------------------------------------
# Spearman (point + bootstrap CI), agreement, offset
# ---------------------------------------------------------------------------


def _spearman_point(xs: list[float], ys: list[float]) -> dict[str, Any]:
    """Point-estimate Spearman rho/p, or :data:`_CONSTANT_NOTE` if either side
    has zero variance.

    ``p`` is ``None`` (not a fabricated/non-JSON-safe NaN) when n=2: with
    exactly 2 non-constant points rho is always well-defined (+-1.0) but
    scipy's p-value relies on a t-distribution with ``n - 2 == 0`` degrees
    of freedom, which is undefined and comes back as NaN. ``NaN != NaN``
    in Python, which would otherwise make two byte-identical runs compare
    unequal and silently break the determinism guarantee -- caught here,
    not downstream.
    """
    if len(set(xs)) <= 1 or len(set(ys)) <= 1:
        return {"rho": None, "p": None, "note": _CONSTANT_NOTE}
    rho, p = spearmanr(xs, ys)
    rho = float(rho)
    p = float(p)
    if p != p:  # nan check without importing math
        p = None
    return {"rho": rho, "p": p, "note": None}


def _bootstrap_ci(
    stat_fn: Any,  # noqa: ANN401 -- callable(indices) -> float | None
    n: int,
    rng: random.Random,
    n_resamples: int,
) -> dict[str, Any]:
    """Generic percentile-CI bootstrap, resampling case indices with replacement.

    Args:
        stat_fn: ``indices -> float | None`` for one repetition's resample
            (``None`` when that resample cannot support the statistic,
            e.g. a constant Spearman input -- never fabricated).
        n: Number of cases in this scope.
        rng: Shared RNG, consumed ``n_resamples`` times here (each
            repetition draws ``n`` indices via ``rng.randrange``).
        n_resamples: Number of bootstrap repetitions.

    Returns:
        ``{"ci_low", "ci_high", "n_valid_reps", "n_total_reps"}``.
    """
    series: list[float | None] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        series.append(stat_fn(indices))
    valid = [v for v in series if v is not None]
    if not valid:
        return {
            "ci_low": None,
            "ci_high": None,
            "n_valid_reps": 0,
            "n_total_reps": n_resamples,
        }
    low, high = np.percentile(np.array(valid), [2.5, 97.5])
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "n_valid_reps": len(valid),
        "n_total_reps": n_resamples,
    }


def spearman_with_ci(
    xs: list[float],
    ys: list[float],
    rng: random.Random,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
) -> dict[str, Any]:
    """Point Spearman rho (+ p) and its bootstrap 95% CI over case indices.

    Returns:
        ``{"rho", "p", "note", "ci_low", "ci_high", "n_valid_reps",
        "n_total_reps"}``. ``note`` is :data:`_CONSTANT_NOTE` (and
        ``rho``/``p``/CI fields are all ``None``) when ``xs`` or ``ys`` has
        zero variance; the CI bootstrap is skipped entirely in that case
        (there's nothing to resample toward).
    """
    point = _spearman_point(xs, ys)
    if point["note"] is not None:
        return {
            **point,
            "ci_low": None,
            "ci_high": None,
            "n_valid_reps": 0,
            "n_total_reps": n_resamples,
        }

    n = len(xs)

    def _rep(indices: list[int]) -> float | None:
        rx = [xs[i] for i in indices]
        ry = [ys[i] for i in indices]
        if len(set(rx)) <= 1 or len(set(ry)) <= 1:
            return None
        rho, _p = spearmanr(rx, ry)
        return float(rho)

    ci = _bootstrap_ci(_rep, n, rng, n_resamples)
    return {**point, **ci}


def agreement_rates(
    author_values: list[int], target_values: list[float]
) -> dict[str, float | None]:
    """Exact and within-1 agreement, rounding ``target_values`` first.

    Returns:
        ``{"exact", "within_1"}``, or ``None`` for both if there are no
        cases.
    """
    if not author_values:
        return {"exact": None, "within_1": None}
    n = len(author_values)
    exact = 0
    within_1 = 0
    for a, t in zip(author_values, target_values, strict=True):
        rt = round(t)
        if a == rt:
            exact += 1
        if abs(a - rt) <= 1:
            within_1 += 1
    return {"exact": exact / n, "within_1": within_1 / n}


def mean_offset_with_ci(
    author_values: list[float],
    target_values: list[float],
    rng: random.Random,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
) -> dict[str, Any]:
    """Mean(author - target) (raw, unrounded) + bootstrap 95% CI over case indices."""
    if not author_values:
        return {
            "mean_offset": None,
            "ci_low": None,
            "ci_high": None,
            "n_valid_reps": 0,
            "n_total_reps": n_resamples,
        }
    offsets = [a - t for a, t in zip(author_values, target_values, strict=True)]
    mean_offset = sum(offsets) / len(offsets)
    n = len(offsets)

    def _rep(indices: list[int]) -> float:
        return sum(offsets[i] for i in indices) / n

    ci = _bootstrap_ci(_rep, n, rng, n_resamples)
    return {"mean_offset": mean_offset, **ci}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def author_vs_targets_report(
    records: list[dict[str, Any]],
    rng: random.Random,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per dimension x scope x target: Spearman(+CI), agreement, mean offset(+CI).

    Consumes ``rng`` in the fixed order documented in the module
    docstring: dimension, then scope, then target -- Spearman bootstrap
    first, then the mean-offset bootstrap, for each (dimension, scope,
    target) cell before moving to the next.

    Returns:
        ``dimension -> scope -> target -> {"spearman": {...},
        "agreement": {...}, "offset": {...}, "n_cases"}``.
    """
    scoped = _scopes(records)
    report: dict[str, dict[str, dict[str, Any]]] = {dim: {} for dim in _DIMENSIONS}
    for dim in _DIMENSIONS:
        for scope_name in ("overall", *_GAME_FAMILIES):
            scope_records = scoped[scope_name]
            author_values = [r["author"][dim] for r in scope_records]
            report[dim][scope_name] = {}
            for target in _COMPARISON_TARGETS:
                target_values = [_target_value(r, dim, target) for r in scope_records]
                spearman = spearman_with_ci(
                    author_values, target_values, rng, n_resamples
                )
                agreement = agreement_rates(author_values, target_values)
                offset = mean_offset_with_ci(
                    author_values, target_values, rng, n_resamples
                )
                report[dim][scope_name][target] = {
                    "n_cases": len(scope_records),
                    "spearman": spearman,
                    "agreement": agreement,
                    "offset": offset,
                }
    return report


def judge_pairwise_report(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per dimension x scope: pairwise Spearman + mean offset among the 3
    judges, on these same cases (point estimates only -- no bootstrap CI
    was requested for this comparison).

    Returns:
        ``dimension -> scope -> "<judge1>|<judge2>" -> {"spearman",
        "mean_offset", "n_cases"}``.
    """
    scoped = _scopes(records)
    report: dict[str, dict[str, dict[str, Any]]] = {dim: {} for dim in _DIMENSIONS}
    pairs = [
        (_JUDGES[i], _JUDGES[j])
        for i in range(len(_JUDGES))
        for j in range(i + 1, len(_JUDGES))
    ]
    for dim in _DIMENSIONS:
        for scope_name in ("overall", *_GAME_FAMILIES):
            scope_records = scoped[scope_name]
            report[dim][scope_name] = {}
            for j1, j2 in pairs:
                v1 = [r["judges"][j1][dim] for r in scope_records]
                v2 = [r["judges"][j2][dim] for r in scope_records]
                spearman = _spearman_point(v1, v2)
                mean_offset = (
                    sum(a - b for a, b in zip(v1, v2, strict=True)) / len(v1)
                    if v1
                    else None
                )
                report[dim][scope_name][f"{j1}|{j2}"] = {
                    "n_cases": len(scope_records),
                    "spearman": spearman,
                    "mean_offset": mean_offset,
                }
    return report


def outcome_consistency_distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Rating distributions (1-5 counts) for the author and each judge, on
    ``outcome_consistency``, over all 20 cases."""
    dim = "outcome_consistency"
    author_counts = Counter(r["author"][dim] for r in records)
    judge_counts = {
        judge: Counter(r["judges"][judge][dim] for r in records) for judge in _JUDGES
    }
    return {
        "n_cases": len(records),
        "author": {str(v): author_counts.get(v, 0) for v in range(1, 6)},
        "judges": {
            judge: {str(v): judge_counts[judge].get(v, 0) for v in range(1, 6)}
            for judge in _JUDGES
        },
    }


def run_analysis(
    author_rows: list[dict[str, Any]],
    key: dict[str, Any],
    seed: int = _DEFAULT_SEED,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
) -> dict[str, Any]:
    """Join + run every report, assemble the full output dict."""
    records = join_records(author_rows, key)
    rng = random.Random(seed)
    return {
        "seed": seed,
        "n_resamples": n_resamples,
        "n_cases": len(records),
        "by_family_n_cases": {
            family: sum(1 for r in records if r["game_family"] == family)
            for family in _GAME_FAMILIES
        },
        "author_vs_targets": author_vs_targets_report(records, rng, n_resamples),
        "judge_pairwise": judge_pairwise_report(records),
        "outcome_consistency_distributions": outcome_consistency_distributions(records),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_spearman(cell: dict[str, Any]) -> str:
    if cell.get("note"):
        return cell["note"]
    return _fmt(cell["rho"])


def render_markdown(analysis: dict[str, Any]) -> str:
    lines: list[str] = ["# Author ratings vs. judge panel -- analysis", ""]
    lines.append(f"seed={analysis['seed']}  n_resamples={analysis['n_resamples']}")
    lines.append(
        f"n_cases={analysis['n_cases']}  by_family={analysis['by_family_n_cases']}"
    )
    lines.append("")

    scopes = ("overall", *_GAME_FAMILIES)
    for dim in _DIMENSIONS:
        lines.append(f"## {dim}")
        lines.append("")
        lines.append("### a) Spearman rho vs. author (+ 95% bootstrap CI)")
        lines.append("")
        lines.append("| scope | target | n | rho | ci_low | ci_high | n_valid_reps |")
        lines.append("|---|---|---|---|---|---|---|")
        for scope in scopes:
            for target in _COMPARISON_TARGETS:
                cell = analysis["author_vs_targets"][dim][scope][target]
                sp = cell["spearman"]
                lines.append(
                    f"| {scope} | {target} | {cell['n_cases']} | {_fmt_spearman(sp)} "
                    f"| {_fmt(sp.get('ci_low'))} | {_fmt(sp.get('ci_high'))} "
                    f"| {sp.get('n_valid_reps', 0)}/{sp.get('n_total_reps', 0)} |"
                )
        lines.append("")

        lines.append("### b) Exact / within-1 agreement vs. author (rounded)")
        lines.append("")
        lines.append("| scope | target | n | exact | within_1 |")
        lines.append("|---|---|---|---|---|")
        for scope in scopes:
            for target in _COMPARISON_TARGETS:
                cell = analysis["author_vs_targets"][dim][scope][target]
                ag = cell["agreement"]
                lines.append(
                    f"| {scope} | {target} | {cell['n_cases']} "
                    f"| {_fmt(ag['exact'])} | {_fmt(ag['within_1'])} |"
                )
        lines.append("")

        lines.append("### c) Mean offset (author - target), raw (+ 95% bootstrap CI)")
        lines.append("")
        lines.append(
            "| scope | target | n | mean_offset | ci_low | ci_high | n_valid_reps |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for scope in scopes:
            for target in _COMPARISON_TARGETS:
                cell = analysis["author_vs_targets"][dim][scope][target]
                off = cell["offset"]
                lines.append(
                    f"| {scope} | {target} | {cell['n_cases']} "
                    f"| {_fmt(off['mean_offset'])} | {_fmt(off['ci_low'])} "
                    f"| {_fmt(off['ci_high'])} "
                    f"| {off['n_valid_reps']}/{off['n_total_reps']} |"
                )
        lines.append("")

        lines.append(
            "### d) Pairwise judge Spearman + mean offset (point estimates only)"
        )
        lines.append("")
        lines.append("| scope | judge_pair | n | spearman_rho | mean_offset |")
        lines.append("|---|---|---|---|---|")
        for scope in scopes:
            for pair_key, cell in analysis["judge_pairwise"][dim][scope].items():
                lines.append(
                    f"| {scope} | {pair_key} | {cell['n_cases']} "
                    f"| {_fmt_spearman(cell['spearman'])} "
                    f"| {_fmt(cell['mean_offset'])} |"
                )
        lines.append("")

    lines.append("## outcome_consistency rating distributions (n=20)")
    lines.append("")
    dist = analysis["outcome_consistency_distributions"]
    lines.append("| rater | 1 | 2 | 3 | 4 | 5 |")
    lines.append("|---|---|---|---|---|---|")
    author_row = dist["author"]
    lines.append(
        f"| author | {author_row['1']} | {author_row['2']} | {author_row['3']} "
        f"| {author_row['4']} | {author_row['5']} |"
    )
    for judge in _JUDGES:
        row = dist["judges"][judge]
        lines.append(
            f"| {judge} | {row['1']} | {row['2']} | {row['3']} "
            f"| {row['4']} | {row['5']} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    csv_input: Path = typer.Option(_DEFAULT_CSV_INPUT, "--csv-input"),
    key_input: Path = typer.Option(_DEFAULT_KEY_INPUT, "--key-input"),
    json_output: Path = typer.Option(_DEFAULT_JSON_OUTPUT, "--json-output"),
    markdown_output: Path = typer.Option(_DEFAULT_MARKDOWN_OUTPUT, "--markdown-output"),
    seed: int = typer.Option(_DEFAULT_SEED, "--seed"),
    n_resamples: int = typer.Option(_DEFAULT_N_RESAMPLES, "--n-resamples"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Compare author_ratings.csv against the judge panel. Makes zero API calls."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    author_rows = load_author_ratings(csv_input)
    key = load_rating_key(key_input)
    analysis = run_analysis(author_rows, key, seed=seed, n_resamples=n_resamples)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(analysis, indent=2))
    typer.echo(f"Analysis written to {json_output}")

    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(analysis))
    typer.echo(f"Markdown report written to {markdown_output}")


if __name__ == "__main__":
    app()
