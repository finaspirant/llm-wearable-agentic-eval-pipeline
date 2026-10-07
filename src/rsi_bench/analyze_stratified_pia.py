"""Pure analysis of an already-scored stratified GLEE PIA run.

Reads only two already-committed files --
:data:`_DEFAULT_SCORED_INPUT` (the per-cluster scored JSONL produced by
``src.rsi_bench.run_stratified_pia``) and :data:`_DEFAULT_MANIFEST_INPUT`
(its frozen selection manifest) -- and performs no network I/O of any
kind. There is no Anthropic client anywhere in this module and no import
of ``anthropic``; running it can never spend a live judge call.

This module changes NO scoring logic and reuses the existing kappa
machinery rather than re-deriving it:

- :func:`~src.rsi_bench.glee_pia_baseline._dimension_kappa` -- the exact
  per-cluster, per-dimension Fleiss' kappa function
  ``run_stratified_pia.score_selected_clusters`` already calls while
  scoring -- is reused for the integrity recomputation in step 1. It is
  fed reconstructed :class:`~src.rsi_bench.glee_pia_baseline.GLEEReasoningAnnotation`
  objects built from each record's persisted ``raw_scores``, the same
  reconstruction technique already exercised in
  ``tests/rsi_bench/test_run_stratified_pia.py::TestRawScoreRoundTrip``.
- :func:`~src.annotation.pia_calculator._build_label_matrix` and
  :func:`~src.annotation.pia_calculator._fleiss_kappa` are reused
  *directly* for the pooled aggregation in step 2(c), since
  ``_dimension_kappa`` only ever operates within a single cluster and
  pooling needs one combined matrix spanning many clusters' items.

DESIGN DECISIONS (read before changing aggregation behaviour)
--------------------------------------------------------------------------
Degenerate-cluster detection
    ``IRRCalculator.fleiss_kappa`` forces kappa to 1.0 exactly when
    ``abs(1 - P_e_bar) < 1e-12`` (see ``irr_calculator.py``), and
    ``P_e_bar == 1.0`` iff every single cell of the label matrix carries
    the same category. So a (cluster_id, dimension) pair is flagged
    degenerate here iff it has >= 2 complete-panel items AND every
    (item, judge) score for it is identical -- no re-derivation of the
    kappa formula is needed, just that one equivalent, exact condition.

"With and without" aggregates
    Degenerate status is a (cluster_id, dimension) pair, not a whole
    cluster. For a single dimension's aggregate, "without" means: drop
    that cluster's contribution to *that dimension's* mean/weighted-
    mean/pooled computation only -- its other dimensions are untouched.
    For ``kappa_overall`` (defined, same as in the scoring driver, as the
    mean of a cluster's own non-null per-dimension kappas), "without"
    recomputes each affected cluster's kappa_overall as the mean of only
    its *non-degenerate* dimensions, then aggregates those adjusted
    per-cluster values as usual. Pooled kappa_overall is, in both modes,
    the mean of the 4 pooled per-dimension values for that scope.

Bootstrap
    One ``random.Random(seed)`` instance, consumed in a fixed order --
    for each of ``n_resamples`` repetitions, families are resampled (with
    replacement, same size as the family) in the fixed
    ``_GAME_FAMILIES`` order, which is what makes two runs with the same
    seed byte-identical. Each repetition yields both a per-family
    statistic (from that family's own resample) and an overall
    "stratified" statistic (from the concatenation of all three
    families' resamples for that repetition) -- one resampling pass
    serves both the "stratified by family for the overall figure" and
    "within family for per-family figures" requirements at once. A
    repetition whose resample has zero non-null observations for a given
    scope/dimension contributes no value to that cell's distribution
    (never a fabricated 0.0) and is excluded from its percentile/
    fraction-positive calculation; ``n_valid_reps`` records how many
    repetitions actually contributed.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import typer
from scipy.stats import spearmanr

from src.annotation.irr_calculator import IRRCalculator
from src.annotation.pia_calculator import _build_label_matrix, _fleiss_kappa
from src.rsi_bench.glee_pia_baseline import (
    _GAME_FAMILIES,
    _GLEE_SCALE,
    GLEEReasoningAnnotation,
    _dimension_kappa,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="analyze-stratified-pia",
    help="Pure (zero-API-call) analysis of an already-scored stratified GLEE PIA run.",
    add_completion=False,
)

_DEFAULT_SCORED_INPUT = Path("data/rsi_bench/glee_pia_stratified_scored.jsonl")
_DEFAULT_MANIFEST_INPUT = Path("data/rsi_bench/glee_pia_manifest.json")
_DEFAULT_JSON_OUTPUT = Path("data/rsi_bench/glee_pia_analysis.json")
_DEFAULT_MARKDOWN_OUTPUT = Path("data/rsi_bench/glee_pia_analysis.md")
_DEFAULT_SEED = 20261002
_DEFAULT_N_RESAMPLES = 10_000
_DEFAULT_N_PERMUTATIONS = 1_000
_KAPPA_OVERALL_KEY = "kappa_overall"
_MISMATCH_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_scored_records(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per line from a scored-cluster JSONL file.

    Args:
        path: Path to ``glee_pia_stratified_scored.jsonl`` (or a
            synthetic file with the same record schema).

    Returns:
        List of per-cluster record dicts, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Scored-cluster file not found: {path}")
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d scored cluster records from %s.", len(records), path)
    return records


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the frozen selection manifest.

    Args:
        path: Path to ``glee_pia_manifest.json``.

    Returns:
        The parsed manifest dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Step 1: Integrity
# ---------------------------------------------------------------------------


def _reconstruct_annotations(record: dict[str, Any]) -> list[GLEEReasoningAnnotation]:
    """Rebuild :class:`GLEEReasoningAnnotation` objects from one record's raw scores.

    Mirrors ``tests/rsi_bench/test_run_stratified_pia.py::TestRawScoreRoundTrip``'s
    reconstruction technique, but assigns ``member_index`` by first-seen
    order of ``member_id`` within this record rather than looking it up in
    ``sampled_member_ids`` -- equivalent (each real member gets one stable
    index, consistent across its judges) and does not depend on that list
    still being present or in the original order.

    Args:
        record: One parsed scored-cluster record.

    Returns:
        One :class:`GLEEReasoningAnnotation` per ``raw_scores`` entry.
    """
    member_index_by_id: dict[str, int] = {}
    annotations: list[GLEEReasoningAnnotation] = []
    for rs in record["raw_scores"]:
        member_id = rs["member_id"]
        idx = member_index_by_id.setdefault(member_id, len(member_index_by_id))
        annotations.append(
            GLEEReasoningAnnotation(
                annotation_id=f"{record['cluster_id']}/{idx}/{rs['judge_name']}",
                cluster_id=record["cluster_id"],
                member_index=idx,
                judge_name=rs["judge_name"],
                valuation_reasoning=rs["valuation_reasoning"],
                horizon_strategy_planning=rs["horizon_strategy_planning"],
                concession_handling=rs["concession_handling"],
                outcome_consistency=rs["outcome_consistency"],
                rationale=rs["rationale"],
                judge_failed_dimensions=tuple(rs.get("judge_failed_dimensions", [])),
            )
        )
    return annotations


def recompute_cluster_kappa(
    record: dict[str, Any], dimensions: list[str], irr: IRRCalculator
) -> dict[str, float]:
    """Recompute a cluster's per-dimension kappa from its persisted raw scores.

    Reuses :func:`~src.rsi_bench.glee_pia_baseline._dimension_kappa`
    exactly -- the same function the live scoring run calls.

    Args:
        record: One parsed scored-cluster record.
        dimensions: Dimension names to recompute.
        irr: Shared :class:`IRRCalculator` instance.

    Returns:
        Mapping of dimension -> recomputed kappa, omitting dimensions with
        fewer than 2 complete-panel items (kappa undefined).
    """
    annotations = _reconstruct_annotations(record)
    recomputed: dict[str, float] = {}
    for dim in dimensions:
        kappa, _n_excluded = _dimension_kappa(irr, dim, annotations)
        if kappa is not None:
            recomputed[dim] = kappa
    return recomputed


def check_integrity(
    records: list[dict[str, Any]], dimensions: list[str]
) -> dict[str, Any]:
    """Recompute every cluster's kappa and diff it against the stored value.

    Also checks ``kappa_overall`` (stored as the mean of the stored
    per-dimension values) against the mean of the *recomputed*
    per-dimension values, under the pseudo-dimension name
    ``"kappa_overall"``.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.

    Returns:
        ``{"n_mismatches": int, "mismatches": [...]}``. Each mismatch is
        ``{"cluster_id", "dimension", "stored", "recomputed", "abs_diff"}``
        (``stored``/``recomputed`` may be ``None`` if only one side has a
        defined value).
    """
    irr = IRRCalculator()
    mismatches: list[dict[str, Any]] = []
    for record in records:
        recomputed = recompute_cluster_kappa(record, dimensions, irr)
        stored = record["kappa_per_dimension"]
        for dim in dimensions:
            s = stored.get(dim)
            r = recomputed.get(dim)
            if (s is None) != (r is None):
                mismatches.append(
                    {
                        "cluster_id": record["cluster_id"],
                        "dimension": dim,
                        "stored": s,
                        "recomputed": r,
                        "abs_diff": None,
                    }
                )
            elif s is not None and r is not None:
                diff = abs(s - r)
                if diff > _MISMATCH_TOLERANCE:
                    mismatches.append(
                        {
                            "cluster_id": record["cluster_id"],
                            "dimension": dim,
                            "stored": s,
                            "recomputed": r,
                            "abs_diff": diff,
                        }
                    )

        stored_overall = record[_KAPPA_OVERALL_KEY]
        recomputed_overall = (
            sum(recomputed.values()) / len(recomputed) if recomputed else None
        )
        if recomputed_overall is None or abs(stored_overall - recomputed_overall) > (
            _MISMATCH_TOLERANCE
        ):
            mismatches.append(
                {
                    "cluster_id": record["cluster_id"],
                    "dimension": _KAPPA_OVERALL_KEY,
                    "stored": stored_overall,
                    "recomputed": recomputed_overall,
                    "abs_diff": (
                        None
                        if recomputed_overall is None
                        else abs(stored_overall - recomputed_overall)
                    ),
                }
            )

    return {"n_mismatches": len(mismatches), "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Degenerate (denominator ~ 0) clusters
# ---------------------------------------------------------------------------


def _member_scores_for_dimension(
    record: dict[str, Any], dimension: str
) -> dict[str, dict[str, int | None]]:
    """Group one cluster's raw scores by member for a single dimension.

    Args:
        record: One parsed scored-cluster record.
        dimension: Dimension name to extract.

    Returns:
        ``member_id -> {judge_name: score_or_None}``.
    """
    by_member: dict[str, dict[str, int | None]] = {}
    for rs in record["raw_scores"]:
        by_member.setdefault(rs["member_id"], {})[rs["judge_name"]] = rs.get(dimension)
    return by_member


def _complete_panel_scores(
    record: dict[str, Any], dimension: str, judges: list[str]
) -> dict[str, dict[str, int]]:
    """Return only the members with a non-null score from every judge.

    Matches the inclusion rule inside
    :func:`~src.rsi_bench.glee_pia_baseline._dimension_kappa` exactly: a
    member is included in a dimension's kappa only if every judge in
    ``judges`` produced a non-``None`` score for it.

    Args:
        record: One parsed scored-cluster record.
        dimension: Dimension name.
        judges: Judge names that must all be present.

    Returns:
        ``member_id -> {judge_name: score}`` for members with a complete
        panel.
    """
    by_member = _member_scores_for_dimension(record, dimension)
    complete: dict[str, dict[str, int]] = {}
    for member_id, scores in by_member.items():
        # Filter to the REQUESTED judges first -- `scores` may contain
        # other judges too (e.g. the full 3 when `judges` is a 2-judge
        # leave-one-out subset), and completeness must be judged only
        # against the requested subset, not against whatever else
        # happens to be in the raw data.
        present = {j: scores.get(j) for j in judges if scores.get(j) is not None}
        if len(present) == len(judges):
            complete[member_id] = present  # type: ignore[assignment]
    return complete


def find_degenerate_entries(
    records: list[dict[str, Any]], dimensions: list[str], judges: list[str]
) -> list[dict[str, Any]]:
    """Find every (cluster, dimension) whose kappa was forced to 1.0.

    See the module docstring's "Degenerate-cluster detection" note for
    the exact (and exactly equivalent) condition used here.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Judge names.

    Returns:
        One entry per degenerate (cluster, dimension) pair:
        ``{"cluster_id", "game_family", "dimension", "n_judged_members",
        "n_items_used_in_kappa", "uniform_score_value", "member_ids"}``.
    """
    degenerate: list[dict[str, Any]] = []
    for record in records:
        for dim in dimensions:
            complete = _complete_panel_scores(record, dim, judges)
            if len(complete) < 2:
                continue
            all_values = {s for scores in complete.values() for s in scores.values()}
            if len(all_values) == 1:
                degenerate.append(
                    {
                        "cluster_id": record["cluster_id"],
                        "game_family": record["game_family"],
                        "dimension": dim,
                        "n_judged_members": record["n_judged_members"],
                        "n_items_used_in_kappa": len(complete),
                        "uniform_score_value": next(iter(all_values)),
                        "member_ids": sorted(complete.keys()),
                    }
                )
    return degenerate


# ---------------------------------------------------------------------------
# Step 2: Aggregations
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    """Arithmetic mean, or ``None`` for an empty list."""
    return sum(values) / len(values) if values else None


def _weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    """Weighted mean of ``(value, weight)`` pairs, or ``None`` if empty/zero-weight."""
    total_weight = sum(w for _v, w in pairs)
    if total_weight == 0:
        return None
    return sum(v * w for v, w in pairs) / total_weight


def effective_kappa_overall(
    record: dict[str, Any], excluded_dimensions: set[str]
) -> float | None:
    """Per-cluster kappa_overall, optionally excluding given dimensions.

    With an empty ``excluded_dimensions``, this reproduces exactly
    ``record["kappa_overall"]`` (both are the mean of the record's
    non-null ``kappa_per_dimension`` values) -- a useful self-check.

    Args:
        record: One parsed scored-cluster record.
        excluded_dimensions: Dimension names to drop before averaging
            (used for the "without degenerate clusters" aggregate).

    Returns:
        Mean of the remaining per-dimension kappa values, or ``None`` if
        none remain.
    """
    remaining = [
        v
        for dim, v in record["kappa_per_dimension"].items()
        if dim not in excluded_dimensions
    ]
    return _mean(remaining)


def pooled_dimension_kappa(
    records: list[dict[str, Any]],
    dimension: str,
    judges: list[str],
    excluded_cluster_ids: set[str],
) -> tuple[float | None, int]:
    """Pool every cluster's complete-panel items into one label matrix.

    Directly reuses
    :func:`~src.annotation.pia_calculator._build_label_matrix` and
    :func:`~src.annotation.pia_calculator._fleiss_kappa` -- this is the
    one aggregation ``_dimension_kappa`` cannot do, since it only ever
    sees a single cluster.

    Args:
        records: Clusters to pool (already scoped to a family, or all).
        dimension: Dimension name.
        judges: Judge names (fixed column order across all items).
        excluded_cluster_ids: Clusters to drop entirely from this
            dimension's pool (used for the "without degenerate clusters"
            aggregate).

    Returns:
        ``(pooled_kappa_or_None, n_items_pooled)``.
    """
    irr = IRRCalculator()
    item_ids: list[str] = []
    score_map: dict[tuple[str, str], int] = {}
    for record in records:
        if record["cluster_id"] in excluded_cluster_ids:
            continue
        complete = _complete_panel_scores(record, dimension, judges)
        for member_id, scores in complete.items():
            item_id = f"{record['cluster_id']}::{member_id}"
            item_ids.append(item_id)
            for judge, score in scores.items():
                score_map[(item_id, judge)] = score

    if len(item_ids) < 2:
        return None, len(item_ids)
    matrix = _build_label_matrix(item_ids, judges, score_map, scale_offset=1)
    return _fleiss_kappa(irr, matrix, _GLEE_SCALE), len(item_ids)


def _scopes(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split records into the 4 reporting scopes: overall + each family."""
    scopes: dict[str, list[dict[str, Any]]] = {"overall": records}
    for family in _GAME_FAMILIES:
        scopes[family] = [r for r in records if r["game_family"] == family]
    return scopes


def build_aggregates(
    records: list[dict[str, Any]],
    dimensions: list[str],
    judges: list[str],
    degenerate_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full (a)/(b)/(c) aggregate grid, with and without degenerates.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Judge names.
        degenerate_entries: Output of :func:`find_degenerate_entries`.

    Returns:
        ``{"with_degenerate": {...}, "without_degenerate": {...}}``, each
        keyed by dimension name (plus ``"kappa_overall"``), then by scope
        (``"overall"`` + each family), each a dict with
        ``unweighted_mean``, ``member_weighted_mean``, ``pooled``, and
        ``n_clusters``.
    """
    scopes = _scopes(records)
    degenerate_cluster_ids_by_dim: dict[str, set[str]] = {
        dim: set() for dim in dimensions
    }
    degenerate_dims_by_cluster: dict[str, set[str]] = {}
    for entry in degenerate_entries:
        degenerate_cluster_ids_by_dim[entry["dimension"]].add(entry["cluster_id"])
        degenerate_dims_by_cluster.setdefault(entry["cluster_id"], set()).add(
            entry["dimension"]
        )

    output: dict[str, Any] = {}
    for mode, strip in (("with_degenerate", False), ("without_degenerate", True)):
        per_dim: dict[str, dict[str, Any]] = {}
        for dim in dimensions:
            per_scope: dict[str, Any] = {}
            for scope_name, subset in scopes.items():
                exclude_ids = degenerate_cluster_ids_by_dim[dim] if strip else set()
                vals = [
                    r["kappa_per_dimension"][dim]
                    for r in subset
                    if r["cluster_id"] not in exclude_ids
                    and dim in r["kappa_per_dimension"]
                ]
                wpairs = [
                    (r["kappa_per_dimension"][dim], r["n_judged_members"])
                    for r in subset
                    if r["cluster_id"] not in exclude_ids
                    and dim in r["kappa_per_dimension"]
                ]
                pooled, n_pooled = pooled_dimension_kappa(
                    subset, dim, judges, exclude_ids
                )
                per_scope[scope_name] = {
                    "unweighted_mean": _mean(vals),
                    "member_weighted_mean": _weighted_mean(wpairs),
                    "pooled": pooled,
                    "n_clusters": len(vals),
                    "n_items_pooled": n_pooled,
                }
            per_dim[dim] = per_scope

        overall_scope: dict[str, Any] = {}
        for scope_name, subset in scopes.items():
            vals = []
            wpairs = []
            for r in subset:
                excluded = (
                    degenerate_dims_by_cluster.get(r["cluster_id"], set())
                    if strip
                    else set()
                )
                v = effective_kappa_overall(r, excluded)
                if v is None:
                    continue
                vals.append(v)
                wpairs.append((v, r["n_judged_members"]))
            pooled_dim_values = [
                per_dim[dim][scope_name]["pooled"]
                for dim in dimensions
                if per_dim[dim][scope_name]["pooled"] is not None
            ]
            overall_scope[scope_name] = {
                "unweighted_mean": _mean(vals),
                "member_weighted_mean": _weighted_mean(wpairs),
                "pooled": _mean(pooled_dim_values),
                "n_clusters": len(vals),
                "n_items_pooled": None,
            }
        per_dim[_KAPPA_OVERALL_KEY] = overall_scope
        output[mode] = per_dim

    return output


# ---------------------------------------------------------------------------
# Step 4: Cluster bootstrap
# ---------------------------------------------------------------------------


def _percentile_ci(values: list[float]) -> dict[str, Any]:
    """95% percentile CI (2.5th/97.5th) over non-``None`` bootstrap values.

    Args:
        values: One value per bootstrap repetition (``None`` where that
            repetition's resample had no non-null observations).

    Returns:
        ``{"low", "high", "n_valid_reps", "n_total_reps"}``. ``low``/
        ``high`` are ``None`` if no repetition produced a value.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return {
            "low": None,
            "high": None,
            "n_valid_reps": 0,
            "n_total_reps": len(values),
        }
    low, high = np.percentile(np.array(valid), [2.5, 97.5])
    return {
        "low": float(low),
        "high": float(high),
        "n_valid_reps": len(valid),
        "n_total_reps": len(values),
    }


def _fraction_positive(values: list[float]) -> float | None:
    """Fraction of non-``None`` bootstrap values that are strictly > 0."""
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(1 for v in valid if v > 0) / len(valid)


def bootstrap_primary_aggregate(
    records: list[dict[str, Any]],
    dimensions: list[str],
    seed: int = _DEFAULT_SEED,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
) -> dict[str, Any]:
    """Cluster bootstrap CIs for the primary aggregate (unweighted mean).

    See the module docstring's "Bootstrap" note for the exact resampling
    order that makes this deterministic given a fixed seed.

    Args:
        records: Parsed scored-cluster records (the full, with-degenerate
            dataset -- this function does not take a without-degenerate
            mode).
        dimensions: Judge-scored dimension names.
        seed: RNG seed.
        n_resamples: Number of bootstrap repetitions.

    Returns:
        ``{"seed", "n_resamples", "ci_95": {stat_name: {scope: ci}},
        "fraction_positive": {stat_name: {scope: fraction_or_None}}}``
        where ``stat_name`` ranges over ``"kappa_overall"`` and every
        dimension, and ``scope`` over ``"overall"`` + each family.
    """
    rng = random.Random(seed)
    by_family = {
        family: [r for r in records if r["game_family"] == family]
        for family in _GAME_FAMILIES
    }
    stat_names = [_KAPPA_OVERALL_KEY, *dimensions]
    series: dict[str, dict[str, list[float | None]]] = {
        stat: {scope: [] for scope in ("overall", *_GAME_FAMILIES)}
        for stat in stat_names
    }

    def _stat(subset: list[dict[str, Any]], stat: str) -> float | None:
        if stat == _KAPPA_OVERALL_KEY:
            return _mean([r[_KAPPA_OVERALL_KEY] for r in subset])
        return _mean(
            [
                r["kappa_per_dimension"][stat]
                for r in subset
                if stat in r["kappa_per_dimension"]
            ]
        )

    for _ in range(n_resamples):
        resampled_all: list[dict[str, Any]] = []
        per_family_resample: dict[str, list[dict[str, Any]]] = {}
        for family in _GAME_FAMILIES:
            pool = by_family[family]
            n = len(pool)
            resample = [pool[rng.randrange(n)] for _ in range(n)] if n else []
            per_family_resample[family] = resample
            resampled_all.extend(resample)

        for stat in stat_names:
            for family in _GAME_FAMILIES:
                series[stat][family].append(_stat(per_family_resample[family], stat))
            series[stat]["overall"].append(_stat(resampled_all, stat))

    ci_95 = {
        stat: {scope: _percentile_ci(series[stat][scope]) for scope in series[stat]}
        for stat in stat_names
    }
    fraction_positive = {
        stat: {scope: _fraction_positive(series[stat][scope]) for scope in series[stat]}
        for stat in stat_names
    }
    return {
        "seed": seed,
        "n_resamples": n_resamples,
        "ci_95": ci_95,
        "fraction_positive": fraction_positive,
    }


def build_records_without_degenerate(
    records: list[dict[str, Any]], degenerate_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Deep-copy ``records`` with every degenerate (cluster, dimension) entry removed.

    For each affected cluster, the degenerate dimension is dropped from
    ``kappa_per_dimension`` and ``kappa_overall`` is recomputed as the mean
    of whatever dimensions remain (matching :func:`effective_kappa_overall`
    with that cluster's degenerate dimensions excluded). The result can be
    fed straight back into :func:`bootstrap_primary_aggregate` (or
    :func:`build_aggregates`) unchanged -- this function only edits the
    *input*, not the aggregation logic.

    Args:
        records: Parsed scored-cluster records.
        degenerate_entries: Output of :func:`find_degenerate_entries`.

    Returns:
        A new list of records; ``records`` itself is not mutated.
    """
    degenerate_dims_by_cluster: dict[str, set[str]] = {}
    for entry in degenerate_entries:
        degenerate_dims_by_cluster.setdefault(entry["cluster_id"], set()).add(
            entry["dimension"]
        )

    adjusted: list[dict[str, Any]] = []
    for record in records:
        excluded = degenerate_dims_by_cluster.get(record["cluster_id"])
        if not excluded:
            adjusted.append(record)
            continue
        copied = copy.deepcopy(record)
        for dim in excluded:
            copied["kappa_per_dimension"].pop(dim, None)
        remaining = list(copied["kappa_per_dimension"].values())
        copied[_KAPPA_OVERALL_KEY] = (
            sum(remaining) / len(remaining) if remaining else 0.0
        )
        adjusted.append(copied)
    return adjusted


# ---------------------------------------------------------------------------
# Judge agreement diagnostics (score distributions, pairwise agreement)
# ---------------------------------------------------------------------------


def _members_by_judge(
    records: list[dict[str, Any]], dimension: str
) -> list[dict[str, int | None]]:
    """One dict per judged member across ``records``: ``judge_name -> score``.

    Args:
        records: Records to pool (already scoped to one family).
        dimension: Dimension to extract.

    Returns:
        One ``{judge_name: score_or_None}`` dict per member, across every
        record's ``raw_scores``.
    """
    members: list[dict[str, int | None]] = []
    for record in records:
        by_member: dict[str, dict[str, int | None]] = {}
        for rs in record["raw_scores"]:
            by_member.setdefault(rs["member_id"], {})[rs["judge_name"]] = rs.get(
                dimension
            )
        members.extend(by_member.values())
    return members


def judge_agreement_report(
    records: list[dict[str, Any]], dimensions: list[str], judges: list[str]
) -> dict[str, dict[str, Any]]:
    """Score distributions and pairwise judge-agreement stats, per dimension/family.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Judge names.

    Returns:
        ``dimension -> family -> {"distribution", "n_scores",
        "frac_ge_4", "frac_eq_5", "per_judge_mean", "pairwise": {
        "<judge1>|<judge2>": {"n_items_paired", "exact_agreement",
        "within_1_agreement", "spearman_rho", "spearman_p",
        "cohens_kappa"}}}``. Any statistic undefined for lack of data
        (e.g. a dimension/family with zero non-null scores, or a judge
        pair with constant scores for Spearman) is ``None`` rather than
        fabricated.
    """
    report: dict[str, dict[str, Any]] = {dim: {} for dim in dimensions}
    for dim in dimensions:
        for family in _GAME_FAMILIES:
            family_records = [r for r in records if r["game_family"] == family]
            members = _members_by_judge(family_records, dim)

            pooled = [s for m in members for s in m.values() if s is not None]
            distribution = {str(v): pooled.count(v) for v in range(1, 6)}
            n_scores = len(pooled)
            frac_ge_4 = (
                (sum(1 for s in pooled if s >= 4) / n_scores) if n_scores else None
            )
            frac_eq_5 = (
                (sum(1 for s in pooled if s == 5) / n_scores) if n_scores else None
            )

            per_judge_mean: dict[str, float | None] = {}
            for judge in judges:
                judge_scores = [m[judge] for m in members if m.get(judge) is not None]
                per_judge_mean[judge] = _mean(judge_scores)

            pairwise: dict[str, dict[str, Any]] = {}
            for j1, j2 in combinations(judges, 2):
                paired = [
                    (m[j1], m[j2])
                    for m in members
                    if m.get(j1) is not None and m.get(j2) is not None
                ]
                key = f"{j1}|{j2}"
                if len(paired) < 2:
                    pairwise[key] = {
                        "n_items_paired": len(paired),
                        "exact_agreement": None,
                        "within_1_agreement": None,
                        "spearman_rho": None,
                        "spearman_p": None,
                        "cohens_kappa": None,
                    }
                    continue
                a = [p[0] for p in paired]
                b = [p[1] for p in paired]
                exact = sum(1 for x, y in paired if x == y) / len(paired)
                within_1 = sum(1 for x, y in paired if abs(x - y) <= 1) / len(paired)
                if len(set(a)) > 1 and len(set(b)) > 1:
                    rho, p_value = spearmanr(a, b)
                    rho = float(rho)
                    p_value = float(p_value)
                else:
                    rho, p_value = None, None
                try:
                    kappa_result = IRRCalculator().cohens_kappa(a, b)
                    kappa = float(kappa_result["kappa"])  # type: ignore[arg-type]
                    if kappa != kappa:  # nan check without importing math
                        kappa = None
                except ValueError:
                    kappa = None
                pairwise[key] = {
                    "n_items_paired": len(paired),
                    "exact_agreement": exact,
                    "within_1_agreement": within_1,
                    "spearman_rho": rho,
                    "spearman_p": p_value,
                    "cohens_kappa": kappa,
                }

            report[dim][family] = {
                "distribution": distribution,
                "n_scores": n_scores,
                "frac_ge_4": frac_ge_4,
                "frac_eq_5": frac_eq_5,
                "per_judge_mean": per_judge_mean,
                "pairwise": pairwise,
            }
    return report


# ---------------------------------------------------------------------------
# Shuffled-cluster floor
# ---------------------------------------------------------------------------


def _cluster_pool(
    records: list[dict[str, Any]], dimensions: list[str]
) -> tuple[list[int], list[dict[str, dict[str, int | None]]]]:
    """Flatten a family's clusters into (sizes, member pool).

    Args:
        records: Records for one family.
        dimensions: Dimension names to carry per member.

    Returns:
        ``(cluster_sizes, pool)`` where ``pool[i]`` is
        ``{judge_name: {dimension: score_or_None}}`` for one judged
        member, and ``sum(cluster_sizes) == len(pool)``.
    """
    cluster_sizes: list[int] = []
    pool: list[dict[str, dict[str, int | None]]] = []
    for record in records:
        by_member: dict[str, dict[str, dict[str, int | None]]] = {}
        for rs in record["raw_scores"]:
            by_member.setdefault(rs["member_id"], {})[rs["judge_name"]] = {
                dim: rs.get(dim) for dim in dimensions
            }
        cluster_sizes.append(len(by_member))
        pool.extend(by_member.values())
    return cluster_sizes, pool


def _chunk_kappa(
    chunk: list[dict[str, dict[str, int | None]]],
    dimension: str,
    judges: list[str],
    irr: IRRCalculator,
) -> float | None:
    """Fleiss' kappa for one dimension over one shuffled "cluster" chunk."""
    item_ids: list[str] = []
    score_map: dict[tuple[str, str], int] = {}
    for mi, member in enumerate(chunk):
        present = {
            j: member[j][dimension]
            for j in judges
            if j in member and member[j].get(dimension) is not None
        }
        if len(present) == len(judges):
            item_id = str(mi)
            item_ids.append(item_id)
            for judge, score in present.items():
                score_map[(item_id, judge)] = score
    if len(item_ids) < 2:
        return None
    matrix = _build_label_matrix(item_ids, judges, score_map, scale_offset=1)
    return _fleiss_kappa(irr, matrix, _GLEE_SCALE)


def shuffled_cluster_floor(
    records: list[dict[str, Any]],
    dimensions: list[str],
    judges: list[str],
    aggregates: dict[str, Any],
    seed: int = _DEFAULT_SEED,
    n_permutations: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Null floor: reassign judged members to same-sized clusters at random.

    Within each family, the pool of already-judged members (their real,
    unmodified per-judge score vectors) is repeatedly reshuffled into
    "clusters" of the SAME sizes as the real selected clusters, and the
    primary aggregate (unweighted mean of per-cluster kappa) is
    recomputed on each shuffle. If the real value falls inside this
    floor's typical range, the real clustering is not contributing
    agreement beyond what same-sized random grouping of this family's
    score pool would already produce.

    One ``random.Random(seed)`` instance is consumed in a fixed
    family order, and within each family a single
    ``rng.shuffle(order)`` per repetition drives every dimension's
    (and ``kappa_overall``'s) statistic for that repetition -- so two
    runs with the same seed, data, and ``n_permutations`` are
    byte-identical.

    Args:
        records: Parsed scored-cluster records (full dataset).
        dimensions: Judge-scored dimension names.
        judges: Judge names.
        aggregates: Output of :func:`build_aggregates` -- used only to
            read off the real ``with_degenerate`` unweighted-mean value
            to report alongside the floor.
        seed: RNG seed.
        n_permutations: Number of reshuffles per family.

    Returns:
        ``stat_name -> family -> {"real", "shuffled_mean", "ci_low",
        "ci_high", "n_valid_perms", "n_total_perms"}`` where
        ``stat_name`` ranges over every dimension and
        ``"kappa_overall"``.
    """
    rng = random.Random(seed)
    irr = IRRCalculator()
    stat_names = [*dimensions, _KAPPA_OVERALL_KEY]
    result: dict[str, dict[str, Any]] = {stat: {} for stat in stat_names}

    for family in _GAME_FAMILIES:
        family_records = [r for r in records if r["game_family"] == family]
        cluster_sizes, pool = _cluster_pool(family_records, dimensions)
        n_total = len(pool)

        series: dict[str, list[float | None]] = {stat: [] for stat in stat_names}
        for _ in range(n_permutations):
            order = list(range(n_total))
            rng.shuffle(order)
            shuffled = [pool[i] for i in order]
            chunks: list[list[dict[str, dict[str, int | None]]]] = []
            start = 0
            for size in cluster_sizes:
                chunks.append(shuffled[start : start + size])
                start += size

            per_dim_chunk_kappas: dict[str, list[float]] = {
                dim: [] for dim in dimensions
            }
            cluster_overalls: list[float] = []
            for chunk in chunks:
                chunk_dim_kappas: dict[str, float] = {}
                for dim in dimensions:
                    kappa = _chunk_kappa(chunk, dim, judges, irr)
                    if kappa is not None:
                        chunk_dim_kappas[dim] = kappa
                        per_dim_chunk_kappas[dim].append(kappa)
                if chunk_dim_kappas:
                    cluster_overalls.append(
                        sum(chunk_dim_kappas.values()) / len(chunk_dim_kappas)
                    )

            for dim in dimensions:
                vals = per_dim_chunk_kappas[dim]
                series[dim].append(_mean(vals) if vals else None)
            series[_KAPPA_OVERALL_KEY].append(
                _mean(cluster_overalls) if cluster_overalls else None
            )

        for stat in stat_names:
            valid = [v for v in series[stat] if v is not None]
            if valid:
                low, high = np.percentile(np.array(valid), [2.5, 97.5])
                shuffled_mean = _mean(valid)
            else:
                low = high = shuffled_mean = None
            real = aggregates["with_degenerate"][stat][family]["unweighted_mean"]
            result[stat][family] = {
                "real": real,
                "shuffled_mean": shuffled_mean,
                "ci_low": None if low is None else float(low),
                "ci_high": None if high is None else float(high),
                "n_valid_perms": len(valid),
                "n_total_perms": n_permutations,
            }
    return result


# ---------------------------------------------------------------------------
# Cluster dependence (shared games across selected clusters)
# ---------------------------------------------------------------------------


def _cluster_game_ids(record: dict[str, Any]) -> set[str]:
    """Distinct game_ids touched by one cluster's sampled members.

    ``game_id`` is derived from each sampled member's log id (the part
    before the first ``":"`` -- see
    :func:`~src.rsi_bench.run_stratified_pia._member_id`).
    """
    return {mid.split(":", 1)[0] for mid in record["sampled_member_ids"]}


def _game_to_cluster_ids(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """``game_id -> sorted list of cluster_ids touching that game``.

    Args:
        records: Records already scoped to one family.
    """
    game_to_clusters: dict[str, set[str]] = {}
    for record in records:
        for game_id in _cluster_game_ids(record):
            game_to_clusters.setdefault(game_id, set()).add(record["cluster_id"])
    return {g: sorted(cs) for g, cs in game_to_clusters.items()}


def dependence_report(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per family: how much do selected clusters share underlying games?

    Args:
        records: Parsed scored-cluster records.

    Returns:
        ``family -> {"n_clusters", "distinct_games",
        "mean_clusters_per_game", "share_clusters_sharing_a_game"}``.
    """
    result: dict[str, dict[str, Any]] = {}
    for family in _GAME_FAMILIES:
        family_records = [r for r in records if r["game_family"] == family]
        cluster_games = {r["cluster_id"]: _cluster_game_ids(r) for r in family_records}
        game_to_clusters = _game_to_cluster_ids(family_records)

        n_clusters = len(family_records)
        distinct_games = len(game_to_clusters)
        mean_clusters_per_game = _mean(
            [float(len(cs)) for cs in game_to_clusters.values()]
        )
        n_sharing = sum(
            1
            for cluster_id, games in cluster_games.items()
            if any(len(game_to_clusters[g]) > 1 for g in games)
        )
        result[family] = {
            "n_clusters": n_clusters,
            "distinct_games": distinct_games,
            "mean_clusters_per_game": mean_clusters_per_game,
            "share_clusters_sharing_a_game": (
                n_sharing / n_clusters if n_clusters else None
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Game-level (two-stage) bootstrap
# ---------------------------------------------------------------------------


def _cluster_components(records: list[dict[str, Any]]) -> list[list[str]]:
    """Connected components of the cluster-cluster "shares a game" graph.

    Two clusters are joined by an edge iff they share at least one
    ``game_id`` (via :func:`_game_to_cluster_ids`). A cluster bootstrap
    that resamples individual clusters independently is invalid here --
    clusters in the same component are entangled through a chain of
    shared games and must move together as one block. Implemented as a
    plain union-find over ``record["cluster_id"]``.

    Args:
        records: Records already scoped to one family.

    Returns:
        Components as lists of ``cluster_id``, each sorted, and the
        outer list sorted by its first (smallest) ``cluster_id`` -- a
        canonical, deterministic order independent of dict-iteration
        order.
    """
    cluster_ids = [r["cluster_id"] for r in records]
    parent: dict[str, str] = {cid: cid for cid in cluster_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cluster_group in _game_to_cluster_ids(records).values():
        ordered = sorted(cluster_group)
        for cid in ordered[1:]:
            union(ordered[0], cid)

    members_by_root: dict[str, list[str]] = {}
    for cid in cluster_ids:
        members_by_root.setdefault(find(cid), []).append(cid)

    components = [sorted(members) for members in members_by_root.values()]
    components.sort(key=lambda comp: comp[0])
    return components


def cluster_components_report(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per family: connected-component structure of the cluster-game graph.

    Args:
        records: Parsed scored-cluster records.

    Returns:
        ``family -> {"n_clusters", "n_components", "component_sizes"
        (descending), "largest_component_size"}``.
    """
    result: dict[str, dict[str, Any]] = {}
    for family in _GAME_FAMILIES:
        family_records = [r for r in records if r["game_family"] == family]
        components = _cluster_components(family_records)
        sizes = sorted((len(c) for c in components), reverse=True)
        result[family] = {
            "n_clusters": len(family_records),
            "n_components": len(components),
            "component_sizes": sizes,
            "largest_component_size": sizes[0] if sizes else 0,
        }
    return result


def game_level_bootstrap(
    records: list[dict[str, Any]],
    dimensions: list[str],
    seed: int = _DEFAULT_SEED,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
) -> dict[str, Any]:
    """Block bootstrap over connected components of the cluster-game graph.

    The plain cluster bootstrap (:func:`bootstrap_primary_aggregate`)
    resamples clusters as if they were independent. They are not: per
    :func:`dependence_report`, many clusters share an underlying game.
    An earlier version of this function resampled at the GAME level
    (draw ``n_games`` games, one cluster per game) -- that is WRONG: when
    games vastly outnumber clusters (e.g. persuasion has 246 games over
    only 35 clusters), it inflates the effective per-repetition sample
    size far past the real cluster count and over-weights clusters that
    touch many games, which both shrinks the CI (opposite of the
    dependence-correction this function exists to provide) and shifts
    its centre away from the plain per-cluster mean. See
    ``tests/rsi_bench/test_analyze_stratified_pia.py::TestGameLevelBootstrap::test_reduces_to_cluster_bootstrap_when_no_sharing``
    for the regression this replacement is pinned against.

    The correct unit to resample is the CONNECTED COMPONENT of the
    cluster-cluster "shares a game" graph (:func:`_cluster_components`):
    for each family, ``n_components`` draws are made (with replacement)
    from that family's components; each draw contributes EVERY cluster
    in that component (never a subset of one). When every component is a
    singleton (no sharing at all), this is byte-for-byte the plain i.i.d.
    cluster bootstrap. The more clusters are entangled into large
    components, the fewer effective independent draws a repetition has,
    which is what correctly WIDENS the CI relative to
    :func:`bootstrap_primary_aggregate` rather than narrowing it.

    The per-repetition statistic is the exact same estimator as
    :func:`bootstrap_primary_aggregate` -- the unweighted mean of
    per-cluster ``kappa_overall``/``kappa_per_dimension`` over whatever
    clusters end up in the resample -- so the two are directly
    comparable side by side.

    Args:
        records: Parsed scored-cluster records (pass the without-degenerate
            adjusted list -- see :func:`build_records_without_degenerate`
            -- to exclude the 12 degenerate entries, matching how this is
            used in :func:`run_analysis`).
        dimensions: Judge-scored dimension names.
        seed: RNG seed.
        n_resamples: Number of bootstrap repetitions.

    Returns:
        Same shape as :func:`bootstrap_primary_aggregate`'s return value:
        ``{"seed", "n_resamples", "ci_95": {...}, "fraction_positive": {...}}``.
    """
    rng = random.Random(seed)
    by_cluster_id = {r["cluster_id"]: r for r in records}
    family_components: dict[str, list[list[str]]] = {}
    for family in _GAME_FAMILIES:
        family_records = [r for r in records if r["game_family"] == family]
        family_components[family] = _cluster_components(family_records)

    stat_names = [_KAPPA_OVERALL_KEY, *dimensions]
    series: dict[str, dict[str, list[float | None]]] = {
        stat: {scope: [] for scope in ("overall", *_GAME_FAMILIES)}
        for stat in stat_names
    }

    def _stat(subset: list[dict[str, Any]], stat: str) -> float | None:
        if stat == _KAPPA_OVERALL_KEY:
            return _mean([r[_KAPPA_OVERALL_KEY] for r in subset])
        return _mean(
            [
                r["kappa_per_dimension"][stat]
                for r in subset
                if stat in r["kappa_per_dimension"]
            ]
        )

    for _ in range(n_resamples):
        resampled_all: list[dict[str, Any]] = []
        per_family_picks: dict[str, list[dict[str, Any]]] = {}
        for family in _GAME_FAMILIES:
            components = family_components[family]
            n_components = len(components)
            picks: list[dict[str, Any]] = []
            for _ in range(n_components):
                component = components[rng.randrange(n_components)]
                picks.extend(by_cluster_id[cid] for cid in component)
            per_family_picks[family] = picks
            resampled_all.extend(picks)

        for stat in stat_names:
            for family in _GAME_FAMILIES:
                series[stat][family].append(_stat(per_family_picks[family], stat))
            series[stat]["overall"].append(_stat(resampled_all, stat))

    ci_95 = {
        stat: {scope: _percentile_ci(series[stat][scope]) for scope in series[stat]}
        for stat in stat_names
    }
    fraction_positive = {
        stat: {scope: _fraction_positive(series[stat][scope]) for scope in series[stat]}
        for stat in stat_names
    }
    return {
        "seed": seed,
        "n_resamples": n_resamples,
        "ci_95": ci_95,
        "fraction_positive": fraction_positive,
    }


def audit_game_level_bootstrap(
    aggregates_without_degenerate: dict[str, Any],
    cluster_bootstrap: dict[str, Any],
    game_bootstrap: dict[str, Any],
    dimensions: list[str],
) -> dict[str, dict[str, Any]]:
    """Cross-check the game-level bootstrap against the headline estimator.

    Reports, per (stat, scope): the three point-estimate variants already
    computed by :func:`build_aggregates` -- ``unweighted_mean`` (the
    headline per-cluster mean kappa, un-pooled and un-weighted),
    ``member_weighted_mean``, and ``pooled`` (one global Fleiss' kappa
    over every pooled item) -- next to both bootstraps' CIs, plus the two
    checks :func:`run_analysis` is expected to satisfy: the headline
    point estimate falls inside the game-level CI, and the game-level CI
    is at least as wide as the cluster-level CI (dependence should widen
    a CI, never narrow it).

    Args:
        aggregates_without_degenerate: ``build_aggregates(...)[
            "without_degenerate"]`` -- the three point-estimate variants.
        cluster_bootstrap: :func:`bootstrap_primary_aggregate`'s output on
            the without-degenerate records.
        game_bootstrap: :func:`game_level_bootstrap`'s output on the same
            records.
        dimensions: Judge-scored dimension names.

    Returns:
        ``stat -> scope -> {"unweighted_mean", "member_weighted_mean",
        "pooled", "cluster_ci_low", "cluster_ci_high", "game_ci_low",
        "game_ci_high", "cluster_ci_width", "game_ci_width",
        "point_estimate_in_cluster_ci", "point_estimate_in_game_ci",
        "game_ci_at_least_as_wide_as_cluster_ci"}``. Any ``None`` CI bound
        (e.g. ``concession_handling``/persuasion, 0 valid reps on both
        sides) makes the corresponding boolean checks ``None`` rather
        than a fabricated pass/fail.
    """
    stat_names = [_KAPPA_OVERALL_KEY, *dimensions]
    scopes = ("overall", *_GAME_FAMILIES)
    result: dict[str, dict[str, Any]] = {stat: {} for stat in stat_names}
    for stat in stat_names:
        for scope in scopes:
            point = aggregates_without_degenerate[stat][scope]
            cci = cluster_bootstrap["ci_95"][stat][scope]
            gci = game_bootstrap["ci_95"][stat][scope]
            unweighted = point["unweighted_mean"]

            if cci["low"] is None or gci["low"] is None:
                in_cluster_ci = None
                in_game_ci = None
                widths_check = None
                cluster_width = None
                game_width = None
            else:
                in_cluster_ci = (
                    unweighted is not None and cci["low"] <= unweighted <= cci["high"]
                )
                in_game_ci = (
                    unweighted is not None and gci["low"] <= unweighted <= gci["high"]
                )
                cluster_width = cci["high"] - cci["low"]
                game_width = gci["high"] - gci["low"]
                widths_check = game_width >= cluster_width

            result[stat][scope] = {
                "unweighted_mean": unweighted,
                "member_weighted_mean": point["member_weighted_mean"],
                "pooled": point["pooled"],
                "cluster_ci_low": cci["low"],
                "cluster_ci_high": cci["high"],
                "game_ci_low": gci["low"],
                "game_ci_high": gci["high"],
                "cluster_ci_width": cluster_width,
                "game_ci_width": game_width,
                "point_estimate_in_cluster_ci": in_cluster_ci,
                "point_estimate_in_game_ci": in_game_ci,
                "game_ci_at_least_as_wide_as_cluster_ci": widths_check,
            }
    return result


# ---------------------------------------------------------------------------
# Offset-robust agreement: per-judge-centered ICC(3,1) + Krippendorff's alpha
# ---------------------------------------------------------------------------


def _icc_3_1(matrix: np.ndarray) -> float:
    """Two-way mixed-effects ICC(3,1), consistency, single measurement.

    Shrout & Fleiss (1979) / McGraw & Wong (1996) formula, computed from a
    balanced (no missing) one-way-repeated-measures ANOVA:

    - ``BMS`` = between-items (rows) mean square
    - ``EMS`` = residual (item x rater interaction) mean square
    - ``ICC(3,1) = (BMS - EMS) / (BMS + (k - 1) * EMS)``

    This definition already removes each rater's main effect (their
    column mean), which is exactly what makes it an appropriate
    "offset-robust" companion to pre-centering the input -- centering
    first and running ICC(3,1) is consistent (ICC(3,1) is invariant to a
    constant per-rater shift either way).

    Args:
        matrix: ``(n_items, n_raters)`` array, no missing cells.

    Returns:
        ICC(3,1) as a float. ``0.0`` in the degenerate case of zero total
        variance (every cell identical), rather than a 0/0 NaN.
    """
    n, k = matrix.shape
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ss_total = float(((matrix - grand_mean) ** 2).sum())
    ss_rows = float(k * ((row_means - grand_mean) ** 2).sum())
    ss_cols = float(n * ((col_means - grand_mean) ** 2).sum())
    ss_error = ss_total - ss_rows - ss_cols
    df_rows = n - 1
    df_error = (n - 1) * (k - 1)
    bms = ss_rows / df_rows if df_rows else 0.0
    ems = ss_error / df_error if df_error else 0.0
    denom = bms + (k - 1) * ems
    if denom == 0:
        return 0.0
    return (bms - ems) / denom


def offset_robust_agreement_report(
    records: list[dict[str, Any]], dimensions: list[str], judges: list[str]
) -> dict[str, dict[str, Any]]:
    """Per-judge-centered ICC(3,1) consistency and Krippendorff's alpha (ordinal).

    Fleiss' kappa requires integer category labels; re-binning
    per-judge-centered (now non-integer) scores back onto 1-5 would
    discard the centering, so this reports two metrics that run directly
    on continuous data instead. For each (dimension, family), each
    judge's scores are centered by subtracting THAT judge's own mean
    score (over every non-null score for that judge in that
    dimension/family -- the same quantity as
    :func:`judge_agreement_report`'s ``per_judge_mean``), restricted to
    members with a complete (all-judges-present) panel -- the same
    eligibility rule used everywhere else in this module.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Judge names.

    Returns:
        ``dimension -> family -> {"icc_3_1_consistency",
        "krippendorff_alpha_ordinal", "n_items", "n_judges"}``. ``None``
        values when fewer than 2 complete-panel items exist (e.g.
        ``concession_handling``/persuasion).
    """
    report: dict[str, dict[str, Any]] = {dim: {} for dim in dimensions}
    for dim in dimensions:
        for family in _GAME_FAMILIES:
            family_records = [r for r in records if r["game_family"] == family]
            members = _members_by_judge(family_records, dim)

            per_judge_mean: dict[str, float | None] = {}
            for judge in judges:
                scores = [m[judge] for m in members if m.get(judge) is not None]
                per_judge_mean[judge] = _mean(scores)

            complete = [m for m in members if all(m.get(j) is not None for j in judges)]
            if len(complete) < 2 or any(per_judge_mean[j] is None for j in judges):
                report[dim][family] = {
                    "icc_3_1_consistency": None,
                    "krippendorff_alpha_ordinal": None,
                    "n_items": len(complete),
                    "n_judges": len(judges),
                }
                continue

            centered = np.array(
                [[m[j] - per_judge_mean[j] for j in judges] for m in complete]
            )
            icc = _icc_3_1(centered)
            alpha_data = centered.T.tolist()  # rater-major for krippendorffs_alpha
            alpha = IRRCalculator().krippendorffs_alpha(
                alpha_data, level_of_measurement="ordinal"
            )["alpha"]
            report[dim][family] = {
                "icc_3_1_consistency": icc,
                "krippendorff_alpha_ordinal": float(alpha),  # type: ignore[arg-type]
                "n_items": len(complete),
                "n_judges": len(judges),
            }
    return report


# ---------------------------------------------------------------------------
# Leave-one-judge-out kappa + matching shuffled-cluster floor
# ---------------------------------------------------------------------------


def _recompute_kappa_for_judge_subset(
    records: list[dict[str, Any]], dimensions: list[str], judges: list[str]
) -> list[dict[str, Any]]:
    """Deep-copy records with kappa recomputed from an arbitrary judge subset.

    Each cluster's per-dimension kappa is recomputed via
    :func:`pooled_dimension_kappa` applied to that ONE cluster (a
    single-cluster "pool" is just that cluster's own complete-panel
    items under ``judges``) -- this is the same machinery the real
    pooled aggregation uses, just scoped to one cluster at a time so the
    result is directly comparable to the stored (3-judge) per-cluster
    ``kappa_per_dimension``.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Judge subset to recompute with (e.g. 2 of the real 3).

    Returns:
        Deep copies of ``records`` with ``kappa_per_dimension`` and
        ``kappa_overall`` replaced; ``records`` itself is untouched.
    """
    adjusted: list[dict[str, Any]] = []
    for record in records:
        copied = copy.deepcopy(record)
        kappa_per_dim: dict[str, float] = {}
        for dim in dimensions:
            kappa, _n_items = pooled_dimension_kappa([record], dim, judges, set())
            if kappa is not None:
                kappa_per_dim[dim] = kappa
        copied["kappa_per_dimension"] = kappa_per_dim
        copied[_KAPPA_OVERALL_KEY] = (
            sum(kappa_per_dim.values()) / len(kappa_per_dim) if kappa_per_dim else 0.0
        )
        adjusted.append(copied)
    return adjusted


def leave_one_judge_out_report(
    records: list[dict[str, Any]],
    dimensions: list[str],
    judges: list[str],
    excluded_judge: str,
    seed: int = _DEFAULT_SEED,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
) -> dict[str, Any]:
    """Kappa with one judge dropped, plus the matching 2-rater shuffled floor.

    Reuses :func:`build_aggregates` and :func:`shuffled_cluster_floor`
    unchanged -- both already take ``judges`` as a parameter -- so this
    function's only new work is recomputing per-cluster kappa for the
    2-judge subset (see :func:`_recompute_kappa_for_judge_subset`) before
    handing it to them.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Full (3-judge) judge list.
        excluded_judge: Judge to drop (e.g. ``"LiteralGroundedness"``).
        seed: RNG seed for the matching shuffled floor.
        n_permutations: Number of floor repetitions.

    Returns:
        ``{"excluded_judge", "remaining_judges", "aggregates",
        "degenerate_entries", "shuffled_cluster_floor"}``.
    """
    remaining = [j for j in judges if j != excluded_judge]
    adjusted = _recompute_kappa_for_judge_subset(records, dimensions, remaining)
    degenerate = find_degenerate_entries(records, dimensions, remaining)
    aggregates = build_aggregates(adjusted, dimensions, remaining, degenerate)
    floor = shuffled_cluster_floor(
        records,
        dimensions,
        remaining,
        aggregates,
        seed=seed,
        n_permutations=n_permutations,
    )
    return {
        "excluded_judge": excluded_judge,
        "remaining_judges": remaining,
        "aggregates": aggregates,
        "degenerate_entries": degenerate,
        "shuffled_cluster_floor": floor,
    }


# ---------------------------------------------------------------------------
# Within-cluster vs. pooled variance decomposition (judge-mean score)
# ---------------------------------------------------------------------------


def _judge_mean_score(
    member: dict[str, dict[str, int | None]], dimension: str, judges: list[str]
) -> float | None:
    """Mean of whatever non-null judge scores a member has for ``dimension``."""
    scores = [
        member[j][dimension]
        for j in judges
        if j in member and member[j].get(dimension) is not None
    ]
    return sum(scores) / len(scores) if scores else None


def _eta_squared(values: list[float], group_sizes: list[int]) -> float | None:
    """One-way share of variance explained by group membership.

    Args:
        values: Flattened values, ordered so the first ``group_sizes[0]``
            belong to group 0, the next ``group_sizes[1]`` to group 1, etc.
        group_sizes: Sizes of each group; ``sum(group_sizes) ==
            len(values)``.

    Returns:
        ``SS_between / SS_total``, or ``None`` if fewer than 2 values.
        ``0.0`` (not NaN) if every value is identical (``SS_total == 0``).
    """
    n = len(values)
    if n < 2:
        return None
    grand_mean = sum(values) / n
    ss_total = sum((v - grand_mean) ** 2 for v in values)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    idx = 0
    for size in group_sizes:
        if size == 0:
            continue
        group_vals = values[idx : idx + size]
        idx += size
        group_mean = sum(group_vals) / size
        ss_between += size * (group_mean - grand_mean) ** 2
    return ss_between / ss_total


def _family_judge_means(
    records: list[dict[str, Any]], dimension: str, judges: list[str]
) -> tuple[list[float], list[int]]:
    """Flattened judge-mean scores + per-cluster group sizes for one family.

    Members with zero non-null scores for ``dimension`` (e.g. every
    ``concession_handling`` member in persuasion) are dropped entirely
    rather than imputed -- the returned group sizes reflect the post-drop
    counts, which is also what keeps :func:`_eta_squared`'s grouping
    consistent with the values actually passed to it.
    """
    cluster_sizes, pool = _cluster_pool(records, [dimension])
    values: list[float] = []
    sizes: list[int] = []
    idx = 0
    for size in cluster_sizes:
        chunk = pool[idx : idx + size]
        idx += size
        chunk_values = [
            v
            for v in (_judge_mean_score(member, dimension, judges) for member in chunk)
            if v is not None
        ]
        if chunk_values:
            values.extend(chunk_values)
            sizes.append(len(chunk_values))
    return values, sizes


def variance_decomposition(
    records: list[dict[str, Any]],
    dimensions: list[str],
    judges: list[str],
    seed: int = _DEFAULT_SEED,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
) -> dict[str, dict[str, Any]]:
    """Real vs. shuffled share of judge-mean-score variance explained by cluster.

    For each dimension, each member's judge-mean score (mean of whatever
    non-null judge scores it has) is grouped by its real cluster, and
    :func:`_eta_squared` gives the real share of variance "explained" by
    cluster membership. The same pool of judge-mean scores is then
    reshuffled into same-sized groups (:func:`_cluster_pool`'s sizes,
    after the member-dropping in :func:`_family_judge_means`) exactly as
    in :func:`shuffled_cluster_floor`, for ``n_permutations`` repetitions
    with one ``random.Random(seed)`` consumed in fixed family order --
    same determinism guarantee as the rest of this module. ``"overall"``
    combines each repetition's three per-family shuffles (never a
    cross-family shuffle), matching :func:`shuffled_cluster_floor`'s
    convention.

    Args:
        records: Parsed scored-cluster records.
        dimensions: Judge-scored dimension names.
        judges: Judge names.
        seed: RNG seed.
        n_permutations: Number of reshuffles per family.

    Returns:
        ``dimension -> scope -> {"real_eta_squared", "shuffled_mean",
        "ci_low", "ci_high", "n_valid_perms", "n_total_perms",
        "n_members", "n_clusters"}`` where ``scope`` ranges over
        ``"overall"`` and each family.
    """
    rng = random.Random(seed)
    result: dict[str, dict[str, Any]] = {}

    for dim in dimensions:
        family_data: dict[str, tuple[list[float], list[int]]] = {}
        for family in _GAME_FAMILIES:
            family_records = [r for r in records if r["game_family"] == family]
            family_data[family] = _family_judge_means(family_records, dim, judges)

        real_by_scope: dict[str, float | None] = {}
        n_members_by_scope: dict[str, int] = {}
        n_clusters_by_scope: dict[str, int] = {}
        overall_values: list[float] = []
        overall_sizes: list[int] = []
        for family in _GAME_FAMILIES:
            values, sizes = family_data[family]
            real_by_scope[family] = _eta_squared(values, sizes)
            n_members_by_scope[family] = len(values)
            n_clusters_by_scope[family] = len(sizes)
            overall_values.extend(values)
            overall_sizes.extend(sizes)
        real_by_scope["overall"] = _eta_squared(overall_values, overall_sizes)
        n_members_by_scope["overall"] = len(overall_values)
        n_clusters_by_scope["overall"] = len(overall_sizes)

        series: dict[str, list[float | None]] = {
            scope: [] for scope in ("overall", *_GAME_FAMILIES)
        }
        for _ in range(n_permutations):
            combined_values: list[float] = []
            combined_sizes: list[int] = []
            for family in _GAME_FAMILIES:
                values, sizes = family_data[family]
                n_total = len(values)
                order = list(range(n_total))
                rng.shuffle(order)
                shuffled = [values[i] for i in order]
                series[family].append(_eta_squared(shuffled, sizes))
                combined_values.extend(shuffled)
                combined_sizes.extend(sizes)
            series["overall"].append(_eta_squared(combined_values, combined_sizes))

        dim_out: dict[str, Any] = {}
        for scope in ("overall", *_GAME_FAMILIES):
            ci = _percentile_ci(series[scope])
            valid = [v for v in series[scope] if v is not None]
            dim_out[scope] = {
                "real_eta_squared": real_by_scope[scope],
                "shuffled_mean": _mean(valid) if valid else None,
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "n_valid_perms": ci["n_valid_reps"],
                "n_total_perms": n_permutations,
                "n_members": n_members_by_scope[scope],
                "n_clusters": n_clusters_by_scope[scope],
            }
        result[dim] = dim_out
    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    """Format a nullable float for the markdown report."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(analysis: dict[str, Any], dimensions: list[str]) -> str:
    """Render the analysis dict as a markdown report.

    Args:
        analysis: Full output of :func:`run_analysis`.
        dimensions: Judge-scored dimension names, for table row order.

    Returns:
        Markdown text.
    """
    lines: list[str] = ["# Stratified GLEE PIA -- post-hoc analysis", ""]

    lines.append("## 1. Integrity")
    lines.append("")
    n_mismatches = analysis["integrity"]["n_mismatches"]
    lines.append(f"Mismatches between stored and recomputed kappa: **{n_mismatches}**")
    lines.append("")
    if n_mismatches:
        lines.append("| cluster_id | dimension | stored | recomputed | abs_diff |")
        lines.append("|---|---|---|---|---|")
        for m in analysis["integrity"]["mismatches"]:
            lines.append(
                f"| {m['cluster_id']} | {m['dimension']} | {_fmt(m['stored'])} "
                f"| {_fmt(m['recomputed'])} | {_fmt(m['abs_diff'])} |"
            )
        lines.append("")

    stat_names = [_KAPPA_OVERALL_KEY, *dimensions]
    scopes = ["overall", *_GAME_FAMILIES]
    for mode in ("with_degenerate", "without_degenerate"):
        lines.append(f"## 2. Aggregates -- {mode}")
        lines.append("")
        for stat in stat_names:
            lines.append(f"### {stat}")
            lines.append("")
            lines.append(
                "| scope | unweighted_mean | member_weighted_mean "
                "| pooled | n_clusters |"
            )
            lines.append("|---|---|---|---|---|")
            for scope in scopes:
                cell = analysis["aggregates"][mode][stat][scope]
                lines.append(
                    f"| {scope} | {_fmt(cell['unweighted_mean'])} "
                    f"| {_fmt(cell['member_weighted_mean'])} | {_fmt(cell['pooled'])} "
                    f"| {cell['n_clusters']} |"
                )
            lines.append("")

    lines.append("## 3. Degenerate clusters (denominator ~ 0, kappa forced to 1.0)")
    lines.append("")
    degenerate = analysis["degenerate_entries"]
    lines.append(f"Count: **{len(degenerate)}**")
    lines.append("")
    if degenerate:
        lines.append(
            "| cluster_id | family | dimension | n_judged_members | "
            "n_items_used_in_kappa | uniform_score_value |"
        )
        lines.append("|---|---|---|---|---|---|")
        for entry in degenerate:
            lines.append(
                f"| {entry['cluster_id']} | {entry['game_family']} "
                f"| {entry['dimension']} | {entry['n_judged_members']} "
                f"| {entry['n_items_used_in_kappa']} "
                f"| {entry['uniform_score_value']} |"
            )
        lines.append("")

    def _bootstrap_table(boot: dict[str, Any]) -> list[str]:
        out = [f"seed={boot['seed']}  n_resamples={boot['n_resamples']}", ""]
        out.append(
            "| stat | scope | ci_low | ci_high | n_valid_reps | fraction_kappa_gt_0 |"
        )
        out.append("|---|---|---|---|---|---|")
        for stat in stat_names:
            for scope in scopes:
                ci = boot["ci_95"][stat][scope]
                frac = boot["fraction_positive"][stat][scope]
                out.append(
                    f"| {stat} | {scope} | {_fmt(ci['low'])} | {_fmt(ci['high'])} "
                    f"| {ci['n_valid_reps']}/{ci['n_total_reps']} | {_fmt(frac)} |"
                )
        out.append("")
        return out

    lines.append("## 4. Cluster bootstrap (primary aggregate: unweighted mean)")
    lines.append("")
    lines.extend(_bootstrap_table(analysis["bootstrap"]))

    lines.append("## 5. Cluster bootstrap, 12 degenerate entries excluded")
    lines.append("")
    lines.extend(_bootstrap_table(analysis["bootstrap_without_degenerate"]))

    lines.append("## 6. Judge agreement diagnostics")
    lines.append("")
    for dim in dimensions:
        lines.append(f"### {dim}")
        lines.append("")
        for family in _GAME_FAMILIES:
            cell = analysis["judge_agreement"][dim][family]
            per_judge_mean_str = {j: _fmt(v) for j, v in cell["per_judge_mean"].items()}
            lines.append(
                f"**{family}** -- n_scores={cell['n_scores']}, "
                f"distribution={cell['distribution']}, "
                f"frac_ge_4={_fmt(cell['frac_ge_4'])}, "
                f"frac_eq_5={_fmt(cell['frac_eq_5'])}, "
                f"per_judge_mean={per_judge_mean_str}"
            )
            lines.append("")
            lines.append(
                "| judge_pair | n_items_paired | exact_agreement "
                "| within_1_agreement | spearman_rho | cohens_kappa |"
            )
            lines.append("|---|---|---|---|---|---|")
            for pair_key, pair in cell["pairwise"].items():
                lines.append(
                    f"| {pair_key} | {pair['n_items_paired']} "
                    f"| {_fmt(pair['exact_agreement'])} "
                    f"| {_fmt(pair['within_1_agreement'])} "
                    f"| {_fmt(pair['spearman_rho'])} | {_fmt(pair['cohens_kappa'])} |"
                )
            lines.append("")

    lines.append("## 7. Shuffled-cluster floor")
    lines.append("")
    lines.append(
        "| stat | family | real | shuffled_mean | ci_low | ci_high | n_valid_perms |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for stat in stat_names:
        for family in _GAME_FAMILIES:
            cell = analysis["shuffled_cluster_floor"][stat][family]
            lines.append(
                f"| {stat} | {family} | {_fmt(cell['real'])} "
                f"| {_fmt(cell['shuffled_mean'])} | {_fmt(cell['ci_low'])} "
                f"| {_fmt(cell['ci_high'])} "
                f"| {cell['n_valid_perms']}/{cell['n_total_perms']} |"
            )
    lines.append("")

    lines.append("## 8. Cluster dependence (shared games)")
    lines.append("")
    lines.append(
        "| family | n_clusters | distinct_games | mean_clusters_per_game "
        "| share_clusters_sharing_a_game |"
    )
    lines.append("|---|---|---|---|---|")
    for family in _GAME_FAMILIES:
        cell = analysis["dependence"][family]
        lines.append(
            f"| {family} | {cell['n_clusters']} | {cell['distinct_games']} "
            f"| {_fmt(cell['mean_clusters_per_game'])} "
            f"| {_fmt(cell['share_clusters_sharing_a_game'])} |"
        )
    lines.append("")

    lines.append("## 9. Cluster-game connected components (resampling unit)")
    lines.append("")
    lines.append(
        "| family | n_clusters | n_components | largest_component_size "
        "| component_sizes (desc) |"
    )
    lines.append("|---|---|---|---|---|")
    for family in _GAME_FAMILIES:
        cell = analysis["cluster_components"][family]
        lines.append(
            f"| {family} | {cell['n_clusters']} | {cell['n_components']} "
            f"| {cell['largest_component_size']} | {cell['component_sizes']} |"
        )
    lines.append("")

    lines.append(
        "## 10. Cluster-level vs. game-level bootstrap "
        "(both: 12 degenerate entries excluded)"
    )
    lines.append("")
    cluster_boot = analysis["bootstrap_without_degenerate"]
    game_boot = analysis["game_level_bootstrap"]
    lines.append(
        f"cluster-level: seed={cluster_boot['seed']} "
        f"n_resamples={cluster_boot['n_resamples']}  |  "
        f"game-level: seed={game_boot['seed']} n_resamples={game_boot['n_resamples']}"
    )
    lines.append("")
    lines.append(
        "| stat | scope | cluster_ci_low | cluster_ci_high | game_ci_low "
        "| game_ci_high | cluster_frac>0 | game_frac>0 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for stat in stat_names:
        for scope in scopes:
            cci = cluster_boot["ci_95"][stat][scope]
            gci = game_boot["ci_95"][stat][scope]
            cfrac = cluster_boot["fraction_positive"][stat][scope]
            gfrac = game_boot["fraction_positive"][stat][scope]
            lines.append(
                f"| {stat} | {scope} | {_fmt(cci['low'])} | {_fmt(cci['high'])} "
                f"| {_fmt(gci['low'])} | {_fmt(gci['high'])} "
                f"| {_fmt(cfrac)} | {_fmt(gfrac)} |"
            )
    lines.append("")

    lines.append(
        "## 11. Game-level bootstrap audit "
        "(estimator cross-check + CI containment/width checks)"
    )
    lines.append("")
    lines.append(
        "| stat | scope | unweighted_mean | member_weighted_mean | pooled "
        "| in_cluster_ci | in_game_ci | game_width>=cluster_width |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for stat in stat_names:
        for scope in scopes:
            cell = analysis["game_level_audit"][stat][scope]
            lines.append(
                f"| {stat} | {scope} | {_fmt(cell['unweighted_mean'])} "
                f"| {_fmt(cell['member_weighted_mean'])} | {_fmt(cell['pooled'])} "
                f"| {cell['point_estimate_in_cluster_ci']} "
                f"| {cell['point_estimate_in_game_ci']} "
                f"| {cell['game_ci_at_least_as_wide_as_cluster_ci']} |"
            )
    lines.append("")

    lines.append(
        "## 12. Offset-robust agreement "
        "(per-judge-centered ICC(3,1) + Krippendorff's alpha, ordinal)"
    )
    lines.append("")
    lines.append(
        "| dimension | family | icc_3_1_consistency | krippendorff_alpha_ordinal "
        "| n_items |"
    )
    lines.append("|---|---|---|---|---|")
    for dim in dimensions:
        for family in _GAME_FAMILIES:
            cell = analysis["offset_robust_agreement"][dim][family]
            lines.append(
                f"| {dim} | {family} | {_fmt(cell['icc_3_1_consistency'])} "
                f"| {_fmt(cell['krippendorff_alpha_ordinal'])} "
                f"| {cell['n_items']} |"
            )
    lines.append("")

    lines.append("## 13. Leave-one-judge-out (LiteralGroundedness removed)")
    lines.append("")
    loo = analysis["leave_one_judge_out"]
    lines.append(f"remaining_judges={loo['remaining_judges']}")
    lines.append("")
    lines.append(
        "| stat | scope | unweighted_mean | member_weighted_mean | pooled "
        "| n_clusters |"
    )
    lines.append("|---|---|---|---|---|---|")
    for stat in stat_names:
        for scope in scopes:
            cell = loo["aggregates"]["with_degenerate"][stat][scope]
            lines.append(
                f"| {stat} | {scope} | {_fmt(cell['unweighted_mean'])} "
                f"| {_fmt(cell['member_weighted_mean'])} | {_fmt(cell['pooled'])} "
                f"| {cell['n_clusters']} |"
            )
    lines.append("")
    lines.append("2-rater shuffled-cluster floor (1,000 perms, same seed):")
    lines.append("")
    lines.append(
        "| stat | family | real | shuffled_mean | ci_low | ci_high | n_valid_perms |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for stat in stat_names:
        for family in _GAME_FAMILIES:
            cell = loo["shuffled_cluster_floor"][stat][family]
            lines.append(
                f"| {stat} | {family} | {_fmt(cell['real'])} "
                f"| {_fmt(cell['shuffled_mean'])} | {_fmt(cell['ci_low'])} "
                f"| {_fmt(cell['ci_high'])} "
                f"| {cell['n_valid_perms']}/{cell['n_total_perms']} |"
            )
    lines.append("")

    lines.append(
        "## 14. Variance decomposition of judge-mean score "
        "(share explained by cluster membership)"
    )
    lines.append("")
    lines.append(
        "| dimension | scope | real_eta_sq | shuffled_mean | ci_low | ci_high "
        "| n_valid_perms | n_members | n_clusters |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for dim in dimensions:
        for scope in scopes:
            cell = analysis["variance_decomposition"][dim][scope]
            lines.append(
                f"| {dim} | {scope} | {_fmt(cell['real_eta_squared'])} "
                f"| {_fmt(cell['shuffled_mean'])} | {_fmt(cell['ci_low'])} "
                f"| {_fmt(cell['ci_high'])} "
                f"| {cell['n_valid_perms']}/{cell['n_total_perms']} "
                f"| {cell['n_members']} | {cell['n_clusters']} |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_analysis(
    scored_records: list[dict[str, Any]],
    manifest: dict[str, Any],
    seed: int = _DEFAULT_SEED,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    n_permutations: int = _DEFAULT_N_PERMUTATIONS,
) -> dict[str, Any]:
    """Run every analysis step and assemble the full output dict.

    Args:
        scored_records: Parsed scored-cluster records.
        manifest: Parsed selection manifest (source of truth for the
            dimension and judge name lists).
        seed: Bootstrap/permutation RNG seed.
        n_resamples: Number of bootstrap repetitions.
        n_permutations: Number of shuffled-cluster-floor repetitions.

    Returns:
        ``{"seed", "n_clusters", "integrity", "degenerate_entries",
        "aggregates", "bootstrap", "bootstrap_without_degenerate",
        "judge_agreement", "shuffled_cluster_floor", "dependence",
        "game_level_bootstrap", "offset_robust_agreement",
        "leave_one_judge_out", "variance_decomposition",
        "by_family_n_clusters"}``.
    """
    dimensions = list(manifest["config"]["dimensions"])
    judges = list(manifest["config"]["judges"])

    integrity = check_integrity(scored_records, dimensions)
    degenerate_entries = find_degenerate_entries(scored_records, dimensions, judges)
    aggregates = build_aggregates(
        scored_records, dimensions, judges, degenerate_entries
    )
    bootstrap = bootstrap_primary_aggregate(
        scored_records, dimensions, seed=seed, n_resamples=n_resamples
    )
    records_without_degenerate = build_records_without_degenerate(
        scored_records, degenerate_entries
    )
    bootstrap_without_degenerate = bootstrap_primary_aggregate(
        records_without_degenerate, dimensions, seed=seed, n_resamples=n_resamples
    )
    judge_agreement = judge_agreement_report(scored_records, dimensions, judges)
    floor = shuffled_cluster_floor(
        scored_records,
        dimensions,
        judges,
        aggregates,
        seed=seed,
        n_permutations=n_permutations,
    )
    dependence = dependence_report(scored_records)
    cluster_components = cluster_components_report(records_without_degenerate)
    game_level = game_level_bootstrap(
        records_without_degenerate, dimensions, seed=seed, n_resamples=n_resamples
    )
    game_level_audit = audit_game_level_bootstrap(
        aggregates["without_degenerate"],
        bootstrap_without_degenerate,
        game_level,
        dimensions,
    )
    offset_robust = offset_robust_agreement_report(scored_records, dimensions, judges)
    loo = leave_one_judge_out_report(
        scored_records,
        dimensions,
        judges,
        excluded_judge="LiteralGroundedness",
        seed=seed,
        n_permutations=n_permutations,
    )
    variance_decomp = variance_decomposition(
        scored_records, dimensions, judges, seed=seed, n_permutations=n_permutations
    )

    by_family_n_clusters = {
        family: sum(1 for r in scored_records if r["game_family"] == family)
        for family in _GAME_FAMILIES
    }

    return {
        "seed": seed,
        "n_clusters": len(scored_records),
        "by_family_n_clusters": by_family_n_clusters,
        "dimensions": dimensions,
        "judges": judges,
        "integrity": integrity,
        "degenerate_entries": degenerate_entries,
        "aggregates": aggregates,
        "bootstrap": bootstrap,
        "bootstrap_without_degenerate": bootstrap_without_degenerate,
        "judge_agreement": judge_agreement,
        "shuffled_cluster_floor": floor,
        "dependence": dependence,
        "cluster_components": cluster_components,
        "game_level_bootstrap": game_level,
        "game_level_audit": game_level_audit,
        "offset_robust_agreement": offset_robust,
        "leave_one_judge_out": loo,
        "variance_decomposition": variance_decomp,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    scored_input: Path = typer.Option(_DEFAULT_SCORED_INPUT, "--scored-input"),
    manifest_input: Path = typer.Option(_DEFAULT_MANIFEST_INPUT, "--manifest-input"),
    json_output: Path = typer.Option(_DEFAULT_JSON_OUTPUT, "--json-output"),
    markdown_output: Path = typer.Option(_DEFAULT_MARKDOWN_OUTPUT, "--markdown-output"),
    seed: int = typer.Option(_DEFAULT_SEED, "--seed"),
    n_resamples: int = typer.Option(_DEFAULT_N_RESAMPLES, "--n-resamples"),
    n_permutations: int = typer.Option(_DEFAULT_N_PERMUTATIONS, "--n-permutations"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Analyze an already-scored stratified GLEE PIA run. Makes zero API calls."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    records = load_scored_records(scored_input)
    manifest = load_manifest(manifest_input)
    analysis = run_analysis(
        records,
        manifest,
        seed=seed,
        n_resamples=n_resamples,
        n_permutations=n_permutations,
    )

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(analysis, indent=2))
    typer.echo(f"Analysis written to {json_output}")

    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text(render_markdown(analysis, analysis["dimensions"]))
    typer.echo(f"Markdown report written to {markdown_output}")

    by_family = analysis["by_family_n_clusters"]
    typer.echo(f"\nn_clusters={analysis['n_clusters']}  by_family={by_family}")
    typer.echo(f"integrity mismatches: {analysis['integrity']['n_mismatches']}")
    n_degenerate = len(analysis["degenerate_entries"])
    typer.echo(f"degenerate (cluster, dimension) entries: {n_degenerate}")


if __name__ == "__main__":
    app()
