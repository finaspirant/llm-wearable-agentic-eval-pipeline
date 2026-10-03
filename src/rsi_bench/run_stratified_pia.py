"""Seeded, auditable driver for the stratified GLEE PIA scoring run.

Replaces the uncommitted, ad-hoc driver that originally produced
``data/rsi_bench/glee_pia_stratified105_incremental.jsonl`` -- a file that
stored aggregates only (no raw per-member/per-judge scores, no member
identifiers, no sampling seed), and whose generating script was never
committed anywhere in this repository (confirmed: no script matching
``*stratified*`` exists in ``git log --all``).

This module changes NO scoring logic. It imports and reuses, unmodified,
from :mod:`src.rsi_bench.glee_pia_baseline`:

- :func:`~src.rsi_bench.glee_pia_baseline.cluster_decisions` -- clustering
- :func:`~src.rsi_bench.glee_pia_baseline._annotate_cluster` -- judging
- :func:`~src.rsi_bench.glee_pia_baseline._dimension_kappa` -- which calls
  ``src.annotation.pia_calculator._build_label_matrix`` /
  ``_fleiss_kappa``, the same Fleiss' kappa path used by the wearable PIA
  pilot and the committed (non-stratified) GLEE baseline.
- :func:`~src.rsi_bench.glee_pia_baseline.check_action_format_compliance`
- :class:`~src.rsi_bench.glee_pia_baseline._AnthropicJudgeClient`

The only new logic here is (1) deterministic, seeded cluster/member
*selection* and (2) a persistence format that keeps raw scores instead of
discarding them after computing the aggregate.

DESIGN DECISIONS (read before changing selection behaviour)
-------------------------------------------------------------------------
``log_id``
    ``GLEEStep`` has no native id field (unlike the wearable
    ``WearableLog``). The closest stable per-step identifier available is
    ``f"{game_id}:{round}:{phase}"`` (see :func:`_member_id`); this module
    uses that everywhere "log_id" is asked for, and logs a warning if two
    members of the same cluster ever collide on it.

Clustering vs. eligibility
    Clusters are built ONCE from the full step set, fallback members
    included, via ``cluster_decisions(steps, min_cluster_size=1)`` --
    exactly what ``GLEEClusterResult`` already documents as "all members
    incl. fallback". That is what makes ``fallback_rate`` a real,
    non-degenerate number. Eligibility (`>= 2` non-fallback members),
    quantile binning, and the per-cluster member cap all then operate on
    each cluster's NON-FALLBACK member count, since fallback members are
    never sent to a judge (``_annotate_cluster`` skips them) and so
    contribute nothing to judge-call counts, kappa, or cost. Clustering
    on the full set first -- rather than pre-filtering fallback steps out
    of ``cluster_decisions``'s input -- is what makes a genuine
    (non-zero-by-construction) fallback_rate possible to report at all.

Quantile bins
    Bin boundaries are computed from the data alone: eligible clusters
    per family are sorted by non-fallback member count and split into
    ``n_bins`` contiguous, as-equal-as-possible groups. Boundaries are
    therefore identical across seeds; only which clusters get *sampled
    within* each bin depends on ``--seed``.

Token-level usage
    ``_JudgeClient.score_member`` returns only the parsed rubric dict --
    there is no protocol-level hook for the raw Anthropic response or its
    ``.usage``. Rather than edit ``_AnthropicJudgeClient`` (out of scope)
    or reimplement its HTTP call (drift risk), :func:`_wrap_for_usage`
    monkeypatches the *already-constructed* client's
    ``_client.messages.create`` -- a bound method on an object this
    module already holds a reference to -- so every real call's
    ``response.usage`` is captured by an external
    :class:`_UsageAccumulator`, while ``score_member``'s own behaviour
    (retries, JSON parsing) is completely untouched. Clusters are judged
    strictly sequentially and synchronously, so the accumulator's log can
    be sliced per cluster by index range. When the judge client has no
    ``_client.messages`` (e.g. a plain stub implementing only the
    ``_JudgeClient`` protocol), wrapping is skipped and usage fields stay
    ``null`` with a note -- still never fabricated.

``--judge-temperature``
    Raises immediately (``typer.BadParameter`` at the CLI layer,
    ``ValueError`` from :class:`SelectionConfig` itself if constructed
    directly) rather than silently doing nothing: there is still no
    temperature hook in ``_AnthropicJudgeClient.score_member``, so
    honoring this flag would require either editing that module (out of
    scope) or a parallel reimplementation of its retry loop (drift risk).
    Remove this guard only once one of those is actually done -- do not
    just delete the raise to make the flag "work" without it genuinely
    reaching the API call.

``UsageLimitReached``
    Not a literal exception class in the installed ``anthropic`` SDK
    (confirmed via introspection: ``anthropic`` 0.94.0 exposes
    ``BadRequestError`` but no ``UsageLimitReached``). Detection here is
    via ``anthropic.BadRequestError`` (explicitly named in the brief) plus
    a message-pattern check for usage/credit-limit language on any
    ``anthropic.APIStatusError`` -- see :func:`_looks_like_usage_limit`.

``--max-calls`` / ``--max-cost-usd``
    Local, driver-side budget caps, checked after each cluster is scored
    and flushed. ``--max-calls`` counts ``score_member`` invocations
    directly (always enforceable, even with a plain stub).
    ``--max-cost-usd`` is computed from the real captured token usage at
    $3 / $15 per MTok input/output -- it can only be enforced when usage
    capture is active (see "Token-level usage" above); with a
    non-instrumentable client it is skipped with a one-time warning
    rather than silently never triggering without explanation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import typer

from src.annotation.irr_calculator import IRRCalculator
from src.rsi_bench.glee_pia_baseline import (
    _GAME_FAMILIES,
    _GLEE_DIMENSIONS,
    _JUDGES,
    _MODEL,
    DecisionCluster,
    GLEEStep,
    _annotate_cluster,
    _AnthropicJudgeClient,
    _dimension_kappa,
    _JudgeClient,
    check_action_format_compliance,
    cluster_decisions,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="run-stratified-pia",
    help="Seeded, auditable stratified-sampling driver for the GLEE PIA baseline.",
    add_completion=False,
)

_DEFAULT_SEED = 20261002
_DEFAULT_N_BINS = 5
_DEFAULT_CLUSTERS_PER_BIN = 7
_DEFAULT_MEMBER_CAP = 10
_DEFAULT_ELIGIBLE_MIN_NON_FALLBACK = 2
_DEFAULT_COST_PER_CALL_USD = 0.0056  # per the brief: "last run's actual usage"
_DEFAULT_INPUT = Path("tests/experiments/glee/trajectories.jsonl.gz")
_DEFAULT_MANIFEST_OUTPUT = Path("data/rsi_bench/glee_pia_manifest.json")
_DEFAULT_SCORED_OUTPUT = Path("data/rsi_bench/glee_pia_stratified_scored.jsonl")

_DEFAULT_MAX_CALLS = 2700
_DEFAULT_MAX_COST_USD = 16.0
_INPUT_PRICE_PER_MTOK = 3.0  # USD per 1e6 input tokens
_OUTPUT_PRICE_PER_MTOK = 15.0  # USD per 1e6 output tokens

_USAGE_LIMIT_PATTERNS: tuple[str, ...] = (
    "credit balance",
    "usage limit",
    "quota",
    "exceeded your",
    "insufficient_quota",
)


# ---------------------------------------------------------------------------
# Input freezing
# ---------------------------------------------------------------------------


def sha256_of_file(path: Path) -> str:
    """Return the sha256 hex digest of a file, read in chunks.

    Args:
        path: File to hash.

    Returns:
        64-character hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InputMeta:
    """Identity of the frozen input file this run was computed against.

    Args:
        path: Path as given on the command line (not resolved/absolute,
            so the manifest stays portable across checkouts).
        sha256: Digest of the ``.gz`` file exactly as distributed.
        raw_line_count: Total decompressed lines scanned (including any
            blank or malformed ones).
        parsed_step_count: Number of lines that parsed into a
            :class:`~src.rsi_bench.glee_pia_baseline.GLEEStep`.
    """

    path: str
    sha256: str
    raw_line_count: int
    parsed_step_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_glee_log_from_gz(path: Path) -> tuple[list[GLEEStep], int]:
    """Stream-parse a gzip-compressed GLEE JSONL log with no on-disk copy.

    Mirrors
    :func:`~src.rsi_bench.glee_pia_baseline.load_glee_log`'s per-line
    parsing exactly (same field mapping, same malformed-line skip
    behaviour) -- that function only accepts a plain-text ``Path``, so
    this exists purely to add gzip streaming, never to change parsing
    semantics. Nothing is ever written to ``/tmp`` or anywhere else.

    Args:
        path: Path to a ``.gz`` file containing one GLEE step JSON object
            per decompressed line.

    Returns:
        A 2-tuple of (parsed steps, total decompressed line count).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"GLEE log not found: {path}")

    steps: list[GLEEStep] = []
    raw_line_count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line_count = line_no
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                steps.append(
                    GLEEStep(
                        game_id=raw["game_id"],
                        game_family=raw["game_family"],
                        your_player=raw["your_player"],
                        phase=raw["phase"],
                        round=int(raw["round"]),
                        game_state=dict(raw["game_state"]),
                        valid_actions=dict(raw["valid_actions"]),
                        reasoning=raw["reasoning"],
                        action=raw["action"],
                        fallback_used=bool(raw["fallback_used"]),
                        model=raw["model"],
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed GLEE log line %d: %s", line_no, exc)

    logger.info(
        "Loaded %d GLEE steps from %s (gzip, %d raw lines).",
        len(steps),
        path,
        raw_line_count,
    )
    return steps, raw_line_count


def _member_id(step: GLEEStep) -> str:
    """Stable per-step identifier; GLEEStep has no native id field.

    Args:
        step: The step to identify.

    Returns:
        ``"{game_id}:{round}:{phase}"``.
    """
    return f"{step.game_id}:{step.round}:{step.phase}"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionConfig:
    """All knobs that determine selection, independent of the input file.

    Args:
        seed: Single seed driving every random draw in this run (bin
            sampling AND the per-cluster member cap), consumed in a fixed
            order (family, then bin, then cluster) so two runs with the
            same seed and the same input always select the same thing.
        n_bins: Quantile bins per family, by non-fallback member count.
        clusters_per_bin: Target clusters sampled per bin (short bins
            sample everything available instead of raising).
        member_cap: Max non-fallback members judged per selected cluster.
        eligible_min_non_fallback: Minimum non-fallback members for a
            cluster to be eligible for selection at all.
        model: Anthropic model id for judging.
        judge_temperature: Must be ``None`` -- see ``__post_init__``.
            Field kept (rather than removed) so it still appears,
            explicitly null, in ``to_dict()``/the manifest, and so the
            CLI has somewhere to store the rejected value before raising.
        cost_per_call_usd: Per-``score_member``-call cost estimate.
    """

    seed: int
    n_bins: int = _DEFAULT_N_BINS
    clusters_per_bin: int = _DEFAULT_CLUSTERS_PER_BIN
    member_cap: int = _DEFAULT_MEMBER_CAP
    eligible_min_non_fallback: int = _DEFAULT_ELIGIBLE_MIN_NON_FALLBACK
    model: str = _MODEL
    judge_temperature: float | None = None
    cost_per_call_usd: float = _DEFAULT_COST_PER_CALL_USD

    def __post_init__(self) -> None:
        """Reject a non-``None`` ``judge_temperature`` at construction time.

        Defense in depth alongside the CLI-layer ``typer.BadParameter``
        in :func:`main` -- this fires even if ``SelectionConfig`` is
        constructed directly (e.g. from a script or a future caller),
        not only through the CLI. See the module docstring's
        ``--judge-temperature`` note for why this raises instead of
        silently doing nothing.

        Raises:
            ValueError: If ``judge_temperature`` is not ``None``.
        """
        if self.judge_temperature is not None:
            raise ValueError(
                "judge_temperature is not yet threaded into the live "
                "Anthropic call -- _AnthropicJudgeClient.score_member has "
                "no temperature hook, and this driver does not duplicate "
                "its retry loop to add one (see module docstring). "
                "Passing a value here would silently have no effect, so "
                "this raises instead. Omit it until that hook exists "
                "upstream."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_bins": self.n_bins,
            "clusters_per_bin": self.clusters_per_bin,
            "member_cap": self.member_cap,
            "eligible_min_non_fallback": self.eligible_min_non_fallback,
            "model": self.model,
            "judges": list(_JUDGES),
            "dimensions": list(_GLEE_DIMENSIONS),
            "temperature": (
                "api_default"
                if self.judge_temperature is None
                else self.judge_temperature
            ),
            "cost_per_call_usd": self.cost_per_call_usd,
        }


@dataclass(frozen=True)
class SelectedCluster:
    """One cluster chosen by stratified sampling, plus its member ids.

    Args:
        cluster_id: Stable id from :func:`cluster_decisions`.
        game_family: One of the three GLEE families.
        n_members: Total members in the cluster (incl. fallback).
        n_fallback: Fallback-action members in the cluster.
        fallback_rate: ``n_fallback / n_members``.
        non_fallback_member_ids: ALL eligible (non-fallback) member ids
            in the cluster, sorted -- the full pool judging could draw
            from, not just what was actually sampled.
        sampled_member_ids: The (<= ``member_cap``) ids actually selected
            for judging, sorted.
    """

    cluster_id: str
    game_family: str
    n_members: int
    n_fallback: int
    fallback_rate: float
    non_fallback_member_ids: tuple[str, ...]
    sampled_member_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "game_family": self.game_family,
            "n_members": self.n_members,
            "n_fallback": self.n_fallback,
            "fallback_rate": self.fallback_rate,
            "all_non_fallback_member_ids": list(self.non_fallback_member_ids),
            "sampled_member_ids": list(self.sampled_member_ids),
        }


@dataclass(frozen=True)
class SelectionResult:
    """Everything :func:`select_clusters` produces for one run.

    Args:
        clusters: All selected clusters, across all families.
        unclustered_step_count: Steps ``cluster_decisions`` excluded
            (unrecognised family or insufficient ``game_state``).
        by_family_eligible_count: Family -> count of clusters meeting
            ``eligible_min_non_fallback`` (before any sampling).
        by_family_bin_sizes: Family -> list of eligible-cluster counts
            per quantile bin, in bin order (diagnostic: a short bin means
            fewer than ``clusters_per_bin`` were available there).
    """

    clusters: tuple[SelectedCluster, ...]
    unclustered_step_count: int
    by_family_eligible_count: dict[str, int]
    by_family_bin_sizes: dict[str, list[int]]


def _split_into_bins(
    items: list[tuple[DecisionCluster, list[GLEEStep]]], n_bins: int
) -> list[list[tuple[DecisionCluster, list[GLEEStep]]]]:
    """Split a pre-sorted list into ``n_bins`` contiguous, near-equal groups.

    Deterministic given the (already-sorted) input -- carries no
    randomness, so bin boundaries never depend on ``--seed``.

    Args:
        items: Pre-sorted ``(cluster, non_fallback_members)`` pairs.
        n_bins: Number of bins to produce.

    Returns:
        ``n_bins`` lists whose concatenation reproduces ``items``.
    """
    n = len(items)
    bins: list[list[tuple[DecisionCluster, list[GLEEStep]]]] = []
    start = 0
    for i in range(n_bins):
        size = n // n_bins + (1 if i < n % n_bins else 0)
        bins.append(items[start : start + size])
        start += size
    return bins


def select_clusters(steps: list[GLEEStep], config: SelectionConfig) -> SelectionResult:
    """Deterministically select clusters and members for judging.

    Args:
        steps: All loaded GLEE steps (fallback included).
        config: Selection parameters, including the seed.

    Returns:
        :class:`SelectionResult`.
    """
    all_clusters, unclustered = cluster_decisions(steps, min_cluster_size=1)
    rng = random.Random(config.seed)

    selected: list[SelectedCluster] = []
    eligible_counts: dict[str, int] = {}
    bin_sizes: dict[str, list[int]] = {}

    for family in _GAME_FAMILIES:
        family_clusters = sorted(
            (c for c in all_clusters if c.game_family == family),
            key=lambda c: c.cluster_id,
        )
        annotated: list[tuple[DecisionCluster, list[GLEEStep]]] = []
        for cluster in family_clusters:
            non_fallback = [m for m in cluster.members if not m.fallback_used]
            if len(non_fallback) >= config.eligible_min_non_fallback:
                annotated.append((cluster, non_fallback))
        eligible_counts[family] = len(annotated)

        # Sort by (non-fallback member count, cluster_id) for deterministic
        # quantile boundaries independent of dict/scan ordering upstream.
        annotated.sort(key=lambda pair: (len(pair[1]), pair[0].cluster_id))

        bins = _split_into_bins(annotated, config.n_bins)
        bin_sizes[family] = [len(b) for b in bins]
        if any(len(b) < config.clusters_per_bin for b in bins):
            logger.warning(
                "Family %s: one or more quantile bins have fewer than "
                "%d eligible clusters (bin sizes=%s); sampling all "
                "available in short bins instead of raising.",
                family,
                config.clusters_per_bin,
                [len(b) for b in bins],
            )

        for bin_items in bins:
            k = min(config.clusters_per_bin, len(bin_items))
            chosen = rng.sample(bin_items, k) if bin_items else []
            chosen = sorted(chosen, key=lambda pair: pair[0].cluster_id)
            for cluster, non_fallback in chosen:
                ids = [_member_id(m) for m in non_fallback]
                if len(set(ids)) != len(ids):
                    logger.warning(
                        "Duplicate member_id within cluster %s -- "
                        "game_id/round/phase collision.",
                        cluster.cluster_id,
                    )
                if len(non_fallback) > config.member_cap:
                    sampled = rng.sample(non_fallback, config.member_cap)
                else:
                    sampled = list(non_fallback)
                n_fallback = len(cluster.members) - len(non_fallback)
                n_members = len(cluster.members)
                selected.append(
                    SelectedCluster(
                        cluster_id=cluster.cluster_id,
                        game_family=family,
                        n_members=n_members,
                        n_fallback=n_fallback,
                        fallback_rate=(n_fallback / n_members if n_members else 0.0),
                        non_fallback_member_ids=tuple(sorted(ids)),
                        sampled_member_ids=tuple(
                            sorted(_member_id(m) for m in sampled)
                        ),
                    )
                )

    return SelectionResult(
        clusters=tuple(selected),
        unclustered_step_count=unclustered,
        by_family_eligible_count=eligible_counts,
        by_family_bin_sizes=bin_sizes,
    )


def fallback_rate_summary(
    selected: tuple[SelectedCluster, ...],
) -> dict[str, float | int]:
    """Aggregate fallback_rate two ways, since they disagree materially.

    Args:
        selected: Clusters to aggregate over.

    Returns:
        Dict with ``unweighted_mean`` (mean of per-cluster rates),
        ``member_weighted`` (total fallback members / total members),
        and ``n_clusters``.
    """
    if not selected:
        return {"unweighted_mean": 0.0, "member_weighted": 0.0, "n_clusters": 0}
    unweighted = sum(c.fallback_rate for c in selected) / len(selected)
    total_members = sum(c.n_members for c in selected)
    weighted = (
        sum(c.fallback_rate * c.n_members for c in selected) / total_members
        if total_members
        else 0.0
    )
    return {
        "unweighted_mean": unweighted,
        "member_weighted": weighted,
        "n_clusters": len(selected),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    input_meta: InputMeta, config: SelectionConfig, result: SelectionResult
) -> dict[str, Any]:
    """Build the deterministic manifest dict (no wall-clock fields).

    Deliberately excludes any timestamp so that two calls with identical
    arguments are byte-for-byte equal (the CLI layer adds
    ``generated_at`` itself, only when writing to disk).

    Args:
        input_meta: Frozen-input identity.
        config: Selection configuration used.
        result: Output of :func:`select_clusters`.

    Returns:
        JSON-serialisable manifest dict.
    """
    by_family: dict[str, Any] = {}
    for family in _GAME_FAMILIES:
        fam_selected = [c for c in result.clusters if c.game_family == family]
        n_sampled_members = sum(len(c.sampled_member_ids) for c in fam_selected)
        n_judge_calls = n_sampled_members * len(_JUDGES)
        by_family[family] = {
            "n_eligible_clusters": result.by_family_eligible_count.get(family, 0),
            "n_selected_clusters": len(fam_selected),
            "bin_sizes": result.by_family_bin_sizes.get(family, []),
            "n_sampled_members": n_sampled_members,
            "estimated_judge_calls": n_judge_calls,
            "estimated_cost_usd": round(n_judge_calls * config.cost_per_call_usd, 4),
        }

    total_judge_calls = sum(v["estimated_judge_calls"] for v in by_family.values())

    return {
        "input": input_meta.to_dict(),
        "config": config.to_dict(),
        "unclustered_step_count": result.unclustered_step_count,
        "n_selected_clusters_total": len(result.clusters),
        "fallback_rate": fallback_rate_summary(result.clusters),
        "by_family": by_family,
        "total_estimated_judge_calls": total_judge_calls,
        "total_estimated_cost_usd": round(
            total_judge_calls * config.cost_per_call_usd, 4
        ),
        "clusters": [
            c.to_dict() for c in sorted(result.clusters, key=lambda c: c.cluster_id)
        ],
    }


def _print_summary(manifest: dict[str, Any]) -> None:
    """Print the human-readable summary the brief asks for."""
    typer.echo("\n── Stratified GLEE PIA selection ──────────────────────────")
    typer.echo(f"  Input sha256   : {manifest['input']['sha256']}")
    typer.echo(f"  Input lines    : {manifest['input']['raw_line_count']}")
    typer.echo(f"  Seed           : {manifest['config']['seed']}")
    typer.echo(f"  Clusters total : {manifest['n_selected_clusters_total']}")
    fb = manifest["fallback_rate"]
    typer.echo(
        f"  fallback_rate  : unweighted_mean={fb['unweighted_mean']:.4f}  "
        f"member_weighted={fb['member_weighted']:.4f}  (n={fb['n_clusters']})"
    )
    typer.echo("\n  By family:")
    for family, stats in manifest["by_family"].items():
        typer.echo(
            f"    {family:<12} clusters={stats['n_selected_clusters']:<3} "
            f"(eligible={stats['n_eligible_clusters']:<4}) "
            f"sampled_members={stats['n_sampled_members']:<4} "
            f"calls≈{stats['estimated_judge_calls']:<5} "
            f"cost≈${stats['estimated_cost_usd']:.2f}"
        )
    typer.echo(
        f"\n  TOTAL estimated judge calls : {manifest['total_estimated_judge_calls']}"
    )
    typer.echo(
        f"  TOTAL estimated cost        : ${manifest['total_estimated_cost_usd']:.2f}"
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _looks_like_usage_limit(exc: Exception) -> bool:
    """Heuristic match for a usage/credit-limit condition.

    See module docstring's ``UsageLimitReached`` note: the SDK has no
    dedicated exception class for this, so this is a message-pattern
    check layered on top of the real exception types.

    Args:
        exc: The caught exception.

    Returns:
        ``True`` if the message looks like a usage/credit limit.
    """
    text = str(exc).lower()
    return any(pattern in text for pattern in _USAGE_LIMIT_PATTERNS)


def _read_completed_cluster_ids(output_path: Path) -> set[str]:
    """Cluster ids already present in a (possibly partial) output file.

    Args:
        output_path: Scored-output JSONL path.

    Returns:
        Set of ``cluster_id`` values found; empty if the file is absent.
    """
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["cluster_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _build_scoring_cluster(
    original: DecisionCluster, sampled_ids: tuple[str, ...]
) -> DecisionCluster:
    """Rebuild a :class:`DecisionCluster` containing only the sampled members.

    ``member_index`` inside ``_annotate_cluster``/``_dimension_kappa`` is
    positional within ``.members`` -- building a cluster with exactly the
    sampled subset, in ``sampled_ids`` order, is what lets the scored
    output map ``member_index`` back to a ``member_id`` deterministically.

    Args:
        original: The full cluster as reconstructed by
            ``cluster_decisions`` on the frozen input.
        sampled_ids: Ids to include, in the order they should appear.

    Returns:
        A new :class:`DecisionCluster` with ``members`` set to exactly
        the requested ids, in the requested order.

    Raises:
        KeyError: If a requested id isn't a non-fallback member of
            ``original`` (selection and scoring were run against
            different inputs).
    """
    non_fallback = [m for m in original.members if not m.fallback_used]
    by_id: dict[str, GLEEStep] = {}
    for member in non_fallback:
        by_id.setdefault(_member_id(member), member)
    members = [by_id[member_id] for member_id in sampled_ids]
    return DecisionCluster(
        cluster_id=original.cluster_id,
        game_family=original.game_family,
        state_signature=original.state_signature,
        members=members,
    )


class _UsageAccumulator:
    """Ordered log of per-call token usage, captured by :func:`_wrap_for_usage`.

    Not a cache or semantic layer -- just a flat list the driver can
    slice per cluster. Safe to slice this way only because
    :func:`score_selected_clusters` scores clusters strictly
    sequentially and synchronously (one cluster's calls fully complete
    before the next cluster starts).
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def record(self, usage: Any) -> None:  # noqa: ANN401
        """Append one call's usage, tolerant of a partial/odd ``usage`` shape.

        Args:
            usage: The ``.usage`` attribute of an Anthropic
                ``Message`` response (duck-typed -- any object exposing
                these attributes works, including a test double).
        """
        self.calls.append(
            {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
                "cache_creation_input_tokens": int(
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                ),
                "cache_read_input_tokens": int(
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                ),
            }
        )

    def slice_since(self, start_index: int) -> list[dict[str, int]]:
        """Calls recorded since ``start_index`` (inclusive)."""
        return self.calls[start_index:]

    def totals(self) -> dict[str, int]:
        """Running totals across every call recorded so far."""
        return {
            "n_calls": len(self.calls),
            "input_tokens": sum(c["input_tokens"] for c in self.calls),
            "output_tokens": sum(c["output_tokens"] for c in self.calls),
            "cache_creation_input_tokens": sum(
                c["cache_creation_input_tokens"] for c in self.calls
            ),
            "cache_read_input_tokens": sum(
                c["cache_read_input_tokens"] for c in self.calls
            ),
        }


def _estimate_cost_usd(totals: dict[str, int]) -> float:
    """$3/$15 per MTok input/output, per the brief's pricing."""
    return (
        totals["input_tokens"] / 1_000_000 * _INPUT_PRICE_PER_MTOK
        + totals["output_tokens"] / 1_000_000 * _OUTPUT_PRICE_PER_MTOK
    )


def _wrap_for_usage(judge_client: _JudgeClient) -> _UsageAccumulator | None:
    """Monkeypatch a judge client's underlying ``messages.create`` to
    record real token usage, without touching ``glee_pia_baseline.py``.

    ``_AnthropicJudgeClient.score_member`` calls
    ``self._client.messages.create(...)`` and returns only the parsed
    JSON -- usage is otherwise invisible outside that class. This
    replaces the bound ``create`` method on the already-constructed
    client's ``.messages`` resource object with a wrapper that calls the
    real one, records ``response.usage``, and returns the response
    completely unchanged -- every other behaviour of ``score_member``
    (retries, JSON extraction) is untouched because its code never
    changes, only the thing it calls through to does.

    Args:
        judge_client: Any ``_JudgeClient``. Accessing the private
            ``_client``/``messages``/``create`` attribute path is
            deliberate -- this *is* "wrapping at the driver level", the
            alternative to editing the module. Duck-typed, not
            isinstance-gated, so a structurally-matching test double
            (not a real ``_AnthropicJudgeClient``) works identically.

    Returns:
        A fresh :class:`_UsageAccumulator` if ``judge_client`` exposes
        the expected ``_client.messages.create`` shape; ``None`` if it
        doesn't (e.g. a plain ``_JudgeClient``-protocol stub), in which
        case nothing is patched and usage stays unobservable, as before.
    """
    real_client = getattr(judge_client, "_client", None)
    messages = getattr(real_client, "messages", None)
    original_create = getattr(messages, "create", None)
    if messages is None or original_create is None or not callable(original_create):
        return None

    accumulator = _UsageAccumulator()

    def _tracked_create(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        response = original_create(*args, **kwargs)
        accumulator.record(getattr(response, "usage", None))
        return response

    messages.create = _tracked_create
    return accumulator


def _run_summary_path(scored_output: Path) -> Path:
    """Sidecar path for the run summary, next to the scored output."""
    return scored_output.with_name(scored_output.name + ".summary.json")


def _write_run_summary(
    output_path: Path,
    *,
    accumulator: _UsageAccumulator | None,
    total_calls: int,
    n_clusters_scored: int,
    config: SelectionConfig,
    max_calls: int | None,
    max_cost_usd: float | None,
    stopped_reason: str | None,
) -> None:
    """Write (overwrite) the run summary, so it stays accurate even on
    an early/clean stop -- see :func:`score_selected_clusters`.

    Args:
        output_path: Scored-output JSONL path (summary is written beside it).
        accumulator: Usage accumulator, or ``None`` if usage wasn't observable.
        total_calls: ``score_member`` invocations across all clusters so far.
        n_clusters_scored: Clusters newly scored (not counting resumed skips).
        config: Run configuration, for the model id.
        max_calls: The ``--max-calls`` cap in effect, if any.
        max_cost_usd: The ``--max-cost-usd`` cap in effect, if any.
        stopped_reason: ``None`` on natural completion, else e.g.
            ``"max_calls"``, ``"max_cost_usd"``, ``"bad_request"``,
            ``"usage_limit"``.
    """
    totals = accumulator.totals() if accumulator is not None else None
    cost = _estimate_cost_usd(totals) if totals is not None else None
    summary = {
        "n_clusters_scored": n_clusters_scored,
        "total_score_member_calls": total_calls,
        "token_usage": totals,
        "estimated_cost_usd": round(cost, 4) if cost is not None else None,
        "cost_capturable": accumulator is not None,
        "max_calls": max_calls,
        "max_cost_usd": max_cost_usd,
        "stopped_reason": stopped_reason,
        "model": config.model,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    _run_summary_path(output_path).write_text(json.dumps(summary, indent=2))


def score_selected_clusters(
    steps: list[GLEEStep],
    selection: SelectionResult,
    config: SelectionConfig,
    judge_client: _JudgeClient,
    output_path: Path,
    *,
    max_calls: int | None = _DEFAULT_MAX_CALLS,
    max_cost_usd: float | None = _DEFAULT_MAX_COST_USD,
) -> None:
    """Judge every selected cluster, appending one JSONL record per cluster.

    Resumable: clusters whose ``cluster_id`` is already present in
    ``output_path`` are skipped without calling the judge client again.
    Exits cleanly (returns rather than raising) on
    ``anthropic.BadRequestError``, anything that looks like a
    usage/credit-limit condition, or hitting ``max_calls``/``max_cost_usd``
    -- whatever was written before that point is already on disk, since
    each cluster is appended and flushed immediately on completion, and
    the run summary (see :func:`_write_run_summary`) is rewritten after
    every cluster so it is never stale relative to the JSONL.

    Real per-call token usage is captured via :func:`_wrap_for_usage` when
    ``judge_client`` supports it (a real ``_AnthropicJudgeClient`` or a
    structurally-equivalent double); otherwise usage stays ``null`` with
    a note, exactly as before, and only ``max_calls`` (not
    ``max_cost_usd``) can be enforced.

    Args:
        steps: All loaded GLEE steps (fallback included) -- MUST be the
            same input the selection was computed against.
        selection: Output of :func:`select_clusters`.
        config: Selection configuration (for model/temperature logging).
        judge_client: Live or stub judge implementation.
        output_path: JSONL file to append to.
        max_calls: Stop cleanly once total ``score_member`` calls reach
            this. ``None`` disables the check.
        max_cost_usd: Stop cleanly once estimated cost reaches this.
            ``None`` disables the check; also inert when usage isn't
            observable (see above).
    """
    irr = IRRCalculator()
    all_clusters, _ = cluster_decisions(steps, min_cluster_size=1)
    by_cluster_id = {c.cluster_id: c for c in all_clusters}

    done_ids = _read_completed_cluster_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accumulator = _wrap_for_usage(judge_client)
    if accumulator is None:
        logger.warning(
            "Judge client has no instrumentable _client.messages.create "
            "-- per-call token usage will be null in scored output, and "
            "--max-cost-usd cannot be enforced this run. --max-calls is "
            "still enforced via call count."
        )

    total_calls = 0
    n_clusters_scored = 0

    with output_path.open("a") as out:
        for sel in sorted(selection.clusters, key=lambda c: c.cluster_id):
            if sel.cluster_id in done_ids:
                logger.info("Resume: skipping already-scored %s", sel.cluster_id)
                continue

            original = by_cluster_id.get(sel.cluster_id)
            if original is None:
                logger.error(
                    "Selected cluster %s not found when reclustering the "
                    "input -- selection and scoring were run against "
                    "different data. Skipping.",
                    sel.cluster_id,
                )
                continue

            scoring_cluster = _build_scoring_cluster(original, sel.sampled_member_ids)
            usage_start = len(accumulator.calls) if accumulator is not None else 0

            try:
                annotations, skipped_fallback = _annotate_cluster(
                    scoring_cluster, judge_client
                )
            except anthropic.BadRequestError as exc:
                logger.error(
                    "BadRequestError scoring %s (%s). Stopping cleanly -- "
                    "progress so far is saved in %s.",
                    sel.cluster_id,
                    exc,
                    output_path,
                )
                _write_run_summary(
                    output_path,
                    accumulator=accumulator,
                    total_calls=total_calls,
                    n_clusters_scored=n_clusters_scored,
                    config=config,
                    max_calls=max_calls,
                    max_cost_usd=max_cost_usd,
                    stopped_reason="bad_request",
                )
                return
            except anthropic.APIStatusError as exc:
                if _looks_like_usage_limit(exc):
                    logger.error(
                        "Usage/credit limit reached scoring %s (%s). "
                        "Stopping cleanly -- progress so far is saved in %s.",
                        sel.cluster_id,
                        exc,
                        output_path,
                    )
                    _write_run_summary(
                        output_path,
                        accumulator=accumulator,
                        total_calls=total_calls,
                        n_clusters_scored=n_clusters_scored,
                        config=config,
                        max_calls=max_calls,
                        max_cost_usd=max_cost_usd,
                        stopped_reason="usage_limit",
                    )
                    return
                raise

            kappa_per_dim: dict[str, float] = {}
            partial_excluded: dict[str, int] = {}
            for dim in _GLEE_DIMENSIONS:
                kappa, n_excluded = _dimension_kappa(irr, dim, annotations)
                if kappa is not None:
                    kappa_per_dim[dim] = kappa
                if n_excluded:
                    partial_excluded[dim] = n_excluded
            kappa_overall = (
                sum(kappa_per_dim.values()) / len(kappa_per_dim)
                if kappa_per_dim
                else 0.0
            )

            compliance_values = [
                v
                for m in scoring_cluster.members
                if not m.fallback_used
                for v in (check_action_format_compliance(m),)
                if v is not None
            ]
            compliance_rate = (
                sum(compliance_values) / len(compliance_values)
                if compliance_values
                else None
            )

            member_id_by_index = {
                idx: _member_id(m) for idx, m in enumerate(scoring_cluster.members)
            }
            raw_scores = [
                {
                    "member_id": member_id_by_index[ann.member_index],
                    "judge_name": ann.judge_name,
                    "valuation_reasoning": ann.valuation_reasoning,
                    "horizon_strategy_planning": ann.horizon_strategy_planning,
                    "concession_handling": ann.concession_handling,
                    "outcome_consistency": ann.outcome_consistency,
                    "rationale": ann.rationale,
                    "judge_failed_dimensions": list(ann.judge_failed_dimensions),
                }
                for ann in annotations
            ]

            if accumulator is not None:
                cluster_calls = accumulator.slice_since(usage_start)
                usage_block = {
                    "n_score_member_calls": len(raw_scores),
                    "input_tokens": sum(c["input_tokens"] for c in cluster_calls),
                    "output_tokens": sum(c["output_tokens"] for c in cluster_calls),
                    "cache_creation_input_tokens": sum(
                        c["cache_creation_input_tokens"] for c in cluster_calls
                    ),
                    "cache_read_input_tokens": sum(
                        c["cache_read_input_tokens"] for c in cluster_calls
                    ),
                    "note": None,
                }
            else:
                usage_block = {
                    "n_score_member_calls": len(raw_scores),
                    "input_tokens": None,
                    "output_tokens": None,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": None,
                    "note": (
                        "Token-level usage is not observable for this "
                        "judge client (no instrumentable "
                        "_client.messages.create); left null rather than "
                        "fabricated. See module docstring."
                    ),
                }

            record = {
                "cluster_id": sel.cluster_id,
                "game_family": sel.game_family,
                "n_members": sel.n_members,
                "n_fallback": sel.n_fallback,
                "fallback_rate": sel.fallback_rate,
                "all_non_fallback_member_ids": list(sel.non_fallback_member_ids),
                "sampled_member_ids": list(sel.sampled_member_ids),
                "n_judged_members": len(scoring_cluster.members) - skipped_fallback,
                "raw_scores": raw_scores,
                "kappa_per_dimension": kappa_per_dim,
                "kappa_overall": kappa_overall,
                "partial_panel_excluded": partial_excluded,
                "action_format_compliance_rate": compliance_rate,
                "usage": usage_block,
                "judge_model": config.model,
                "temperature": (
                    "api_default"
                    if config.judge_temperature is None
                    else config.judge_temperature
                ),
                "scored_at": datetime.now(UTC).isoformat(),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()

            total_calls += len(raw_scores)
            n_clusters_scored += 1
            logger.info(
                "Scored %s: kappa_overall=%.4f, n_judged=%d",
                sel.cluster_id,
                kappa_overall,
                record["n_judged_members"],
            )

            _write_run_summary(
                output_path,
                accumulator=accumulator,
                total_calls=total_calls,
                n_clusters_scored=n_clusters_scored,
                config=config,
                max_calls=max_calls,
                max_cost_usd=max_cost_usd,
                stopped_reason=None,
            )

            if max_calls is not None and total_calls >= max_calls:
                logger.error(
                    "Reached --max-calls=%d (total=%d) after %s. "
                    "Stopping cleanly -- progress so far is saved in %s.",
                    max_calls,
                    total_calls,
                    sel.cluster_id,
                    output_path,
                )
                _write_run_summary(
                    output_path,
                    accumulator=accumulator,
                    total_calls=total_calls,
                    n_clusters_scored=n_clusters_scored,
                    config=config,
                    max_calls=max_calls,
                    max_cost_usd=max_cost_usd,
                    stopped_reason="max_calls",
                )
                return

            if accumulator is not None and max_cost_usd is not None:
                cost_so_far = _estimate_cost_usd(accumulator.totals())
                if cost_so_far >= max_cost_usd:
                    logger.error(
                        "Reached --max-cost-usd=%.2f (actual=$%.4f) after "
                        "%s. Stopping cleanly -- progress so far is saved "
                        "in %s.",
                        max_cost_usd,
                        cost_so_far,
                        sel.cluster_id,
                        output_path,
                    )
                    _write_run_summary(
                        output_path,
                        accumulator=accumulator,
                        total_calls=total_calls,
                        n_clusters_scored=n_clusters_scored,
                        config=config,
                        max_calls=max_calls,
                        max_cost_usd=max_cost_usd,
                        stopped_reason="max_cost_usd",
                    )
                    return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = typer.Option(
        _DEFAULT_INPUT, "--input", help="Gzip-compressed GLEE JSONL trajectory log."
    ),
    seed: int = typer.Option(
        _DEFAULT_SEED, "--seed", help="Single seed for all sampling in this run."
    ),
    manifest_only: bool = typer.Option(
        False,
        "--manifest-only",
        help="Write the manifest and exit. Never constructs an API client.",
    ),
    manifest_output: Path = typer.Option(_DEFAULT_MANIFEST_OUTPUT, "--manifest-output"),
    scored_output: Path = typer.Option(_DEFAULT_SCORED_OUTPUT, "--scored-output"),
    model: str = typer.Option(_MODEL, "--model", help="Anthropic model id."),
    judge_temperature: float | None = typer.Option(
        None,
        "--judge-temperature",
        help="NOT YET SUPPORTED -- raises immediately if given. See module docstring.",
    ),
    cost_per_call: float = typer.Option(
        _DEFAULT_COST_PER_CALL_USD, "--cost-per-call-usd"
    ),
    max_calls: int | None = typer.Option(
        _DEFAULT_MAX_CALLS,
        "--max-calls",
        help="Stop cleanly once total score_member calls reach this.",
    ),
    max_cost_usd: float | None = typer.Option(
        _DEFAULT_MAX_COST_USD,
        "--max-cost-usd",
        help="Stop cleanly once estimated cost (from captured usage, "
        "$3/$15 per MTok in/out) reaches this.",
    ),
    n_bins: int = typer.Option(_DEFAULT_N_BINS, "--n-bins"),
    clusters_per_bin: int = typer.Option(
        _DEFAULT_CLUSTERS_PER_BIN, "--clusters-per-bin"
    ),
    member_cap: int = typer.Option(_DEFAULT_MEMBER_CAP, "--member-cap"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Select, manifest, and (unless --manifest-only) score a stratified
    GLEE PIA sample. Always calls the live Anthropic API once scoring
    starts -- there is no --dry-run mode, matching
    ``glee_pia_baseline.main``'s own policy."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if judge_temperature is not None:
        raise typer.BadParameter(
            "--judge-temperature is not yet threaded into the live "
            "Anthropic call -- _AnthropicJudgeClient.score_member has no "
            "temperature hook, and this driver does not duplicate its "
            "retry loop to add one (see module docstring). Passing a "
            "value here would silently have no effect, so this raises "
            "instead of accepting it. Omit the flag.",
            param_hint="--judge-temperature",
        )

    input_sha256 = sha256_of_file(input_path)
    steps, raw_line_count = load_glee_log_from_gz(input_path)
    input_meta = InputMeta(
        path=str(input_path),
        sha256=input_sha256,
        raw_line_count=raw_line_count,
        parsed_step_count=len(steps),
    )

    config = SelectionConfig(
        seed=seed,
        n_bins=n_bins,
        clusters_per_bin=clusters_per_bin,
        member_cap=member_cap,
        model=model,
        judge_temperature=judge_temperature,
        cost_per_call_usd=cost_per_call,
    )

    result = select_clusters(steps, config)
    manifest = build_manifest(input_meta, config, result)
    manifest["generated_at"] = datetime.now(UTC).isoformat()

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"\nManifest written to {manifest_output}")

    _print_summary(manifest)

    if manifest_only:
        typer.echo(
            "\n--manifest-only: no judge client constructed, zero API calls made."
        )
        return

    judge_client = _AnthropicJudgeClient(model=model)
    score_selected_clusters(
        steps,
        result,
        config,
        judge_client,
        scored_output,
        max_calls=max_calls,
        max_cost_usd=max_cost_usd,
    )
    typer.echo(f"\nScored output written to {scored_output}")
    typer.echo(f"Run summary written to {_run_summary_path(scored_output)}")


if __name__ == "__main__":
    app()
