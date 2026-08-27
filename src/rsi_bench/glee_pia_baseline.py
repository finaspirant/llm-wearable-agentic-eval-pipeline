"""GLEE Path-Invariant Agreement (PIA) baseline.

Adapts the "Mode B" mechanism from :mod:`src.annotation.pia_calculator` —
rubric-dimension scoring by a rater panel, blind to the specific history
that produced the item being scored — to GLEE economic-game trajectories
(bargaining / negotiation / persuasion).

WHAT IS PORTED VERBATIM (the actual methodological IP):
    - The comparison mechanism: N raters independently score each item on
      rubric dimensions; :func:`~src.annotation.irr_calculator.IRRCalculator.
      fleiss_kappa` measures whether raters agree, exactly as
      ``pia_calculator._PIAIRRComputer`` does for wearable trajectories.
      This module imports and reuses ``_build_label_matrix``,
      ``_fleiss_kappa``, and ``_kappa_interpretation`` from
      :mod:`src.annotation.pia_calculator` directly rather than
      reimplementing them.
    - The "item" unit generalizes from "one agent within a trajectory pair"
      (wearable: 2 agents, direct vs indirect path, same terminal outcome)
      to "one trajectory step within a decision cluster" (GLEE: N members
      reaching an equivalent decision state via different game histories).
    - The optional-dimension pattern: wearable's ``error_recovery`` is
      ``None`` for direct-path agents that had no escalation to recover
      from; GLEE's ``concession_handling`` is ``None`` for cluster members
      with no prior rejection to respond to. Both are excluded from that
      dimension's matrix, not scored as failures.

WHAT IS DOMAIN-SPECIFIC (reinterpreted, not reused):
    - The rubric dimensions themselves. Wearable dimensions
      (tool_precision, privacy compliance) do not exist in an economic
      game; GLEE dimensions are mapped from the same 5-layer decomposition
      (intent -> planning -> tool-call precision -> recovery -> outcome)
      onto: valuation_reasoning, horizon_strategy_planning,
      action_format_compliance, concession_handling, outcome_consistency.
    - ``action_format_compliance`` is NOT judge-scored. It is objectively
      checkable (does ``action`` match ``valid_actions["fields"]``
      exactly?), so it is computed deterministically and reported as a
      diagnostic. It is deliberately EXCLUDED from the kappa computation:
      a deterministic check produces identical output for every "rater" by
      construction, so folding it into inter-rater agreement would inflate
      the headline kappa with a dimension that was never actually at risk
      of disagreement.

INTEGRITY NOTE ON LIVE JUDGING:
    Unlike ``pia_calculator.py``'s dry-run tables (which are hand-tuned to
    hit a target kappa -- see that module's "Calibration proof" comment),
    this module has no dry-run mode and makes no assumption about what
    kappa should result. The 3 judge personas (:data:`_JUDGES`) differ in
    evaluation *angle* (game-theoretic correctness, groundedness in actual
    game_state values, responsiveness to opponent history), not in
    deliberately engineered scoring bias. Whatever kappa comes out of a
    live run is the real number -- there is no fallback path that fabricates
    or pre-calibrates it. Tests inject a stub :class:`_JudgeClient` so the
    suite runs without API calls, but production/CLI use always calls the
    Anthropic API; there is intentionally no ``--dry-run`` flag.

INTEGRITY NOTE ON SCHEMA ASSUMPTIONS:
    The original ``game_state`` key names per game_family were guessed, not
    confirmed, at the time this module was first written. On 2026-08-22 a
    subset was checked against the SDK's own reference implementation
    (``sdk/examples/simple_agent.py`` in ``eilamshapira/GLEE_competition``)
    and :data:`_STATE_KEY_CANDIDATES` was updated accordingly -- see the
    inline ``# confirmed`` / ``# speculative`` comments on each entry.
    Confirmation status is per-*key*, not per-field: some fields still have
    no confirmed literal key (e.g. negotiation's per-player valuation is
    exposed as ``{current_player}_value`` -- a template keyed off another
    field's value, which :func:`_first_present`'s static candidate-list
    lookup cannot resolve; that requires a clustering-logic change, not a
    candidates-table change, and is deliberately deferred). Steps whose
    game_family is unrecognized, or whose game_state lacks enough of the
    expected keys to build a signature, are excluded from clustering and
    counted in :attr:`GLEEPIAResult.unclustered_step_count` rather than
    silently guessed at.

CLI:
    python -m src.rsi_bench.glee_pia_baseline --log-path <path> --output <path>
    python -m src.rsi_bench.glee_pia_baseline --log-path <path> --limit-clusters 5
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import anthropic
import typer

from src.annotation.irr_calculator import IRRCalculator
from src.annotation.pia_calculator import (
    _build_label_matrix,
    _fleiss_kappa,
    _kappa_interpretation,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="glee-pia-baseline",
    help="PIA rubric-agreement baseline for GLEE economic-game trajectories.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 512
_MAX_RETRIES = 3
_RETRY_BASE_SLEEP_S = 2.0
_INTER_CALL_SLEEP_S = 0.25

_GAME_FAMILIES: tuple[str, ...] = ("bargaining", "negotiation", "persuasion")

# 3 independent evaluation angles, not deliberately biased scoring ranges
# (contrast with pia_calculator._STANDARD_STEP_BIAS / _DETOUR_SCORES, which
# are hand-tuned to produce a target kappa for the wearable demonstration).
_JUDGES: tuple[str, ...] = (
    "GameTheoreticRigor",
    "LiteralGroundedness",
    "OpponentResponsiveness",
)

# Judge-scored dimensions. action_format_compliance is deliberately absent
# here -- it is computed deterministically, not by the judge panel.
_GLEE_DIMENSIONS: tuple[str, ...] = (
    "valuation_reasoning",
    "horizon_strategy_planning",
    "concession_handling",
    "outcome_consistency",
)
_GLEE_SCALE: int = 5  # 1-5 integer scale, matches PIA rubric (Mode B) scale

_MIN_CLUSTER_SIZE_DEFAULT = 2

# Candidate game_state key names per field, tried in order (first present
# key wins). See the "INTEGRITY NOTE ON SCHEMA ASSUMPTIONS" module docstring
# section. Within each tuple, keys checked against sdk/examples/
# simple_agent.py (eilamshapira/GLEE_competition, 2026-08-22) come first and
# are marked "confirmed"; the original pre-verification guesses follow as
# fallbacks and are marked "speculative" -- kept, not deleted, in case a
# schema variant doesn't match the reference script exactly.
_STATE_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "money_to_divide": ("money_to_divide",),  # confirmed
    "offer_on_table": (
        "last_offer",  # confirmed -- NOTE: SDK value is a dict of
        # {player}_gain amounts (e.g. {"alice_gain": 50, "bob_gain": 50}),
        # not a scalar. _bargaining_signature's isinstance(offer, (int,
        # float)) check will correctly treat this as "absent" (ratio=None)
        # rather than crash, but it means the offer-ratio signature
        # component is not actually populated from real logs yet -- that
        # needs a clustering-logic change (extracting the current player's
        # gain from the dict), deliberately deferred.
        "offer_on_table",  # speculative
        "current_offer",  # speculative
        "offer",  # speculative
    ),
    # Not yet consumed by _bargaining_signature -- needed to interpret
    # last_offer's {player}_gain keys once that follow-up lands.
    "current_player": ("current_player",),  # confirmed
    "price": ("price", "current_price", "offer_price"),  # speculative --
    # no confirmed flat "price" key was found in the reference script
    "valuation": ("valuation", "my_valuation", "reservation_value"),  # speculative --
    # NOTE: the confirmed real field is NOT a static key at all. The SDK
    # exposes it as f"{current_player}_value" (see negotiation_strategy in
    # simple_agent.py: `my_value = state[f"{me}_value"]` where
    # `me = state["current_player"]`). A static candidate-list lookup
    # cannot resolve a template keyed off another field's runtime value --
    # resolving this needs a clustering-logic change (read current_player
    # first, then look up f"{current_player}_value"), deliberately
    # deferred rather than papering over with a wrong literal guess here.
    "role": ("role",),  # speculative -- SDK derives role via
    # state[f"{current_player}_role"], same dynamic-key limitation as
    # "valuation" above.
    "p": ("p",),  # confirmed
    "v": ("v",),  # confirmed
    "u": ("u",),  # confirmed
    # Not yet consumed by _persuasion_signature.
    "product_price": ("product_price",),  # confirmed
    "max_rounds": ("max_rounds", "total_rounds", "n_rounds"),  # speculative
    # NOTE: an "opponent_last_action" entry used to live here for
    # _concession_handling_applicable(). Confirmed wrong against real data
    # (0/13 clusters scored concession_handling in a live run) and removed
    # rather than left as a dead/misleading guess -- see that function's
    # docstring for the real mechanism (game_state["history"]).
}

_ROUND_BUCKET_STEP = 0.25
_RATIO_BUCKET_STEP = 0.1

_SYSTEM_PROMPTS: dict[str, str] = {
    "GameTheoreticRigor": (
        "You are an expert game-theory evaluator scoring an AI agent's "
        "reasoning trace from a single decision point in an economic game "
        "(bargaining, negotiation, or persuasion). You care about formal "
        "correctness: does the reasoning correctly derive the agent's "
        "reservation value / expected value from the game_state it was "
        "given, and does its strategy account for the round horizon "
        "correctly (e.g. concession schedules, inflation/decay effects, "
        "endgame behavior near max_rounds)? You do not see any other "
        "decision points -- score this one reasoning trace on its own "
        "merits.\n\n"
        "Score each dimension on a 1-5 integer scale (5 = excellent, "
        "1 = badly wrong) and return ONLY a JSON object with these keys: "
        '"valuation_reasoning" (int 1-5), "horizon_strategy_planning" '
        '(int 1-5), "concession_handling" (int 1-5, or null if the '
        "context provided has no prior rejection to respond to), "
        '"outcome_consistency" (int 1-5), "rationale" (string, 1-2 '
        "sentences). No text outside the JSON object."
    ),
    "LiteralGroundedness": (
        "You are a fact-checking evaluator scoring an AI agent's reasoning "
        "trace from a single decision point in an economic game. You care "
        "about groundedness: does every number and claim in the reasoning "
        "actually trace back to a value present in the game_state it was "
        "given, or does it invent, misremember, or misread values? Penalize "
        "reasoning that cites a number not present in game_state, or that "
        "contradicts itself between reasoning and the final action. You do "
        "not see any other decision points -- score this one reasoning "
        "trace on its own merits.\n\n"
        "Score each dimension on a 1-5 integer scale (5 = excellent, "
        "1 = badly wrong) and return ONLY a JSON object with these keys: "
        '"valuation_reasoning" (int 1-5), "horizon_strategy_planning" '
        '(int 1-5), "concession_handling" (int 1-5, or null if the '
        "context provided has no prior rejection to respond to), "
        '"outcome_consistency" (int 1-5), "rationale" (string, 1-2 '
        "sentences). No text outside the JSON object."
    ),
    "OpponentResponsiveness": (
        "You are an evaluator scoring an AI agent's reasoning trace from a "
        "single decision point in an economic game. You care about "
        "responsiveness: when prior opponent behavior (a rejection, a "
        "counter-offer, a stated preference) is available in the context, "
        "does the reasoning actually engage with WHY the opponent acted "
        "that way, or does it ignore history and repeat a generic script? "
        "You do not see any other decision points -- score this one "
        "reasoning trace on its own merits.\n\n"
        "Score each dimension on a 1-5 integer scale (5 = excellent, "
        "1 = badly wrong) and return ONLY a JSON object with these keys: "
        '"valuation_reasoning" (int 1-5), "horizon_strategy_planning" '
        '(int 1-5), "concession_handling" (int 1-5, or null if the '
        "context provided has no prior rejection to respond to), "
        '"outcome_consistency" (int 1-5), "rationale" (string, 1-2 '
        "sentences). No text outside the JSON object."
    ),
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class GLEEStep:
    """One trajectory record as logged by the GLEE competition agent.

    Args:
        game_id: Unique game identifier.
        game_family: One of :data:`_GAME_FAMILIES`.
        your_player: Player identifier for the logging agent.
        phase: Game phase label.
        round: 1-indexed round number.
        game_state: Domain-specific state dict; key names vary by
            game_family (see :data:`_STATE_KEY_CANDIDATES`).
        valid_actions: Action schema for this decision point. Expected to
            contain a ``"fields"`` key (list of required action field
            names) when the action is a structured dict.
        reasoning: Free-text reasoning trace produced by the agent.
        action: The action taken; typically a dict matching
            ``valid_actions["fields"]``.
        fallback_used: True when the logged action was a fallback (e.g.
            parse failure recovery) rather than a genuine model output.
        model: Identifier of the model that produced this step.
    """

    game_id: str
    game_family: str
    your_player: str
    phase: str
    round: int
    game_state: dict[str, Any]
    valid_actions: dict[str, Any]
    reasoning: str
    action: Any
    fallback_used: bool
    model: str


@dataclass
class DecisionCluster:
    """Group of :class:`GLEEStep` records reached via different histories
    but judged to represent the same coarse decision state.

    Args:
        cluster_id: Stable identifier, ``"<game_family>_<index>"``.
        game_family: One of :data:`_GAME_FAMILIES`.
        state_signature: The coarse signature tuple members were grouped
            on; kept for debugging/audit, not used after clustering.
        members: Ordered list of steps in this cluster.
    """

    cluster_id: str
    game_family: str
    state_signature: tuple[Any, ...]
    members: list[GLEEStep]


@dataclass
class GLEEReasoningAnnotation:
    """One judge's rubric scores for one cluster member's reasoning.

    Args:
        annotation_id: ``"<cluster_id>/<member_index>/<judge_name>"``.
        cluster_id: Owning cluster identifier.
        member_index: 0-based position of the scored step within the
            cluster's ``members`` list.
        judge_name: One of :data:`_JUDGES`.
        valuation_reasoning: Integer score in [1, 5], or ``None`` if the
            judge failed to produce it after retries (see
            ``judge_failed_dimensions`` -- this is NOT the same as a
            legitimate N/A and must stay distinguishable from one).
        horizon_strategy_planning: Integer score in [1, 5], or ``None``
            (judge failure -- see ``valuation_reasoning``).
        concession_handling: Integer score in [1, 5], or ``None`` when the
            member had no prior rejection to respond to (a genuine,
            expected N/A -- see :func:`_concession_handling_applicable`).
        outcome_consistency: Integer score in [1, 5], or ``None`` (judge
            failure -- see ``valuation_reasoning``).
        rationale: Judge's brief explanation.
        judge_failed_dimensions: Names of required dimensions the judge
            failed to produce after :data:`_MAX_RETRIES` attempts, if any.
            Empty when everything the judge owed us came back. This is
            what distinguishes "judge failed to produce a required field"
            (a real gap, logged at ERROR when it happens) from
            ``concession_handling``'s ``None``, which means "genuinely
            doesn't apply here" and is never a failure. Both end up
            excluded from that dimension's kappa the same way, but only
            one of them belongs in this tuple.
    """

    annotation_id: str
    cluster_id: str
    member_index: int
    judge_name: str
    valuation_reasoning: int | None
    horizon_strategy_planning: int | None
    concession_handling: int | None
    outcome_consistency: int | None
    rationale: str
    judge_failed_dimensions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return asdict(self)


@dataclass
class GLEEClusterResult:
    """Per-cluster PIA agreement result.

    Args:
        cluster_id: Cluster identifier.
        game_family: One of :data:`_GAME_FAMILIES`.
        n_members: Total steps in the cluster (including any excluded
            from judging, e.g. fallback actions).
        n_judged_members: Members actually sent to the judge panel
            (``n_members`` minus fallback-action members).
        kappa_per_dimension: Mapping of dimension -> Fleiss' kappa,
            computed across this cluster's judged members only. A
            dimension is omitted here (not present as a key) when fewer
            than 2 members have a non-``None`` score for it.
        kappa_overall: Mean of the dimensions present in
            ``kappa_per_dimension``.
        action_format_compliance_rate: Deterministic diagnostic — fraction
            of judged members whose action exactly matched
            ``valid_actions["fields"]``. ``None`` when it could not be
            evaluated for any member (unrecognised valid_actions shape).
            NOT included in ``kappa_overall`` — see module docstring.
        fallback_rate: Fraction of all cluster members whose action was a
            fallback (excluded from judging).
        partial_panel_excluded: Mapping of dimension -> count of members
            excluded from that dimension's kappa specifically for having
            a partial judge panel (some judges scored them, at least one
            didn't after exhausting retries) -- see
            :func:`_dimension_kappa`. Distinct from members with a
            uniform ``None`` across all judges (e.g. concession_handling
            not applicable), which are never counted here since that's
            expected, not a judge failure. A dimension absent from this
            dict had zero such exclusions.
    """

    cluster_id: str
    game_family: str
    n_members: int
    n_judged_members: int
    kappa_per_dimension: dict[str, float]
    kappa_overall: float
    action_format_compliance_rate: float | None
    fallback_rate: float
    partial_panel_excluded: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return asdict(self)


@dataclass
class GameFamilyComparison:
    """Aggregate PIA result for one game_family.

    Args:
        game_family: One of :data:`_GAME_FAMILIES`.
        n_clusters: Number of scored clusters in this family.
        kappa_overall: Mean of per-cluster ``kappa_overall`` values.
    """

    game_family: str
    n_clusters: int
    kappa_overall: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return asdict(self)


@dataclass
class GLEEPIAResult:
    """Top-level result — serialises to the GLEE PIA baseline output JSON.

    Field naming mirrors
    :class:`~src.annotation.pia_calculator.PIAComparisonResult` so this
    output is directly comparable to ``pia_results.json``.

    Args:
        generated_at: ISO-8601 timestamp.
        model: Anthropic model id used for judging.
        judges: Ordered list of judge names.
        n_clusters: Number of clusters with at least 2 judged members.
        n_clusters_skipped: Clusters found but excluded (below
            ``min_cluster_size`` after removing fallback-action members).
        unclustered_step_count: Steps excluded before clustering
            (unrecognised game_family or insufficient game_state keys).
        models_represented: Distinct ``model`` values seen in the input log.
        per_dimension_kappa: Mapping of dimension -> overall Fleiss' kappa
            across all judged members in all scored clusters.
        overall_kappa: Mean of ``per_dimension_kappa`` values.
        interpretation: Landis & Koch label for ``overall_kappa``.
        per_cluster: Mapping of cluster_id -> :class:`GLEEClusterResult`.
        by_game_family: Mapping of game_family -> :class:`GameFamilyComparison`.
        action_format_compliance_overall: Deterministic diagnostic across
            all judged members; NOT part of ``overall_kappa``.
        notes: Methodology caveats worth carrying alongside the numbers.
    """

    generated_at: str
    model: str
    judges: list[str]
    n_clusters: int
    n_clusters_skipped: int
    unclustered_step_count: int
    models_represented: list[str]
    per_dimension_kappa: dict[str, float]
    overall_kappa: float
    interpretation: str
    per_cluster: dict[str, GLEEClusterResult]
    by_game_family: dict[str, GameFamilyComparison]
    action_format_compliance_overall: float | None
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full result to a JSON-safe nested dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_glee_log(path: Path) -> list[GLEEStep]:
    """Load GLEE trajectory records from a JSONL file.

    Malformed lines are skipped with a warning rather than aborting the
    whole load, matching the per-record error handling convention used in
    :mod:`src.annotation.argilla_loader`.

    Args:
        path: Path to a JSONL file, one GLEE step record per line.

    Returns:
        List of :class:`GLEEStep` objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"GLEE log not found: {path}")

    steps: list[GLEEStep] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
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

    logger.info("Loaded %d GLEE steps from %s.", len(steps), path)
    return steps


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _first_present(d: dict[str, Any], field: str) -> Any | None:  # noqa: ANN401
    """Return the first value found for ``field`` under any candidate key.

    Args:
        d: ``game_state`` dict to search.
        field: Logical field name; looked up via
            :data:`_STATE_KEY_CANDIDATES`.

    Returns:
        The first matching value, or ``None`` if no candidate key is
        present.
    """
    for key in _STATE_KEY_CANDIDATES.get(field, (field,)):
        if key in d and d[key] is not None:
            return d[key]
    return None


def _bucket(value: float, step: float) -> float:
    """Round ``value`` to the nearest multiple of ``step`` for coarse binning."""
    return round(value / step) * step


def _round_position(step: GLEEStep) -> tuple[str, float | int]:
    """Coarse round-position component of a decision-cluster signature.

    Normalises by ``max_rounds`` when available (bucketed to
    :data:`_ROUND_BUCKET_STEP`); falls back to the raw round number,
    tagged distinctly so a raw round is never confused with a normalised
    ratio in the same signature space.

    Args:
        step: The GLEE step to derive a round position for.

    Returns:
        ``("norm", bucketed_ratio)`` or ``("raw", round)``.
    """
    max_rounds = _first_present(step.game_state, "max_rounds")
    if isinstance(max_rounds, (int, float)) and max_rounds > 0:
        return ("norm", _bucket(step.round / max_rounds, _ROUND_BUCKET_STEP))
    return ("raw", step.round)


def _bargaining_signature(step: GLEEStep) -> tuple[Any, ...] | None:
    """Coarse signature: round position + money_to_divide + offer ratio.

    Returns:
        Signature tuple, or ``None`` if ``money_to_divide`` is missing.
    """
    money = _first_present(step.game_state, "money_to_divide")
    if not isinstance(money, (int, float)) or money == 0:
        return None
    offer = _first_present(step.game_state, "offer_on_table")
    offer_ratio = (
        _bucket(offer / money, _RATIO_BUCKET_STEP)
        if isinstance(offer, (int, float))
        else None
    )
    return (_round_position(step), money, offer_ratio)


def _negotiation_signature(step: GLEEStep) -> tuple[Any, ...] | None:
    """Coarse signature: round position + role + price/valuation ratio.

    ``role`` and ``valuation`` are NOT static keys in real GLEE negotiation
    state -- the SDK exposes them templated off ``current_player``, e.g.
    ``current_player="player_2"`` -> ``player_2_role`` / ``player_2_value``
    (confirmed against a live trajectory: see git history / PR discussion).
    :data:`_STATE_KEY_CANDIDATES`-based :func:`_first_present` cannot
    resolve a key whose name depends on another field's runtime value, so
    those two fields are looked up directly here instead of going through
    the static candidates table (which remains correct for genuinely
    static fields like ``money_to_divide`` or ``p``/``v``/``u``).

    Returns:
        Signature tuple, or ``None`` if ``current_player`` is missing from
        ``game_state`` (nothing to template the key names off of) or the
        resulting ``{current_player}_role`` key is absent.
    """
    current_player = step.game_state.get("current_player")
    if current_player is None:
        return None
    role = step.game_state.get(f"{current_player}_role")
    if role is None:
        return None
    valuation = step.game_state.get(f"{current_player}_value")
    price = _first_present(step.game_state, "price")
    ratio = (
        _bucket(price / valuation, _RATIO_BUCKET_STEP)
        if isinstance(price, (int, float))
        and isinstance(valuation, (int, float))
        and valuation != 0
        else None
    )
    return (_round_position(step), role, ratio)


def _persuasion_signature(step: GLEEStep) -> tuple[Any, ...] | None:
    """Coarse signature: p/v/u buckets + round position.

    Returns:
        Signature tuple, or ``None`` if all of p/v/u are missing.
    """
    p = _first_present(step.game_state, "p")
    v = _first_present(step.game_state, "v")
    u = _first_present(step.game_state, "u")
    if p is None and v is None and u is None:
        return None
    bucketed = tuple(
        _bucket(x, _RATIO_BUCKET_STEP) if isinstance(x, (int, float)) else None
        for x in (p, v, u)
    )
    return (_round_position(step), *bucketed)


_SIGNATURE_FNS: dict[str, Any] = {
    "bargaining": _bargaining_signature,
    "negotiation": _negotiation_signature,
    "persuasion": _persuasion_signature,
}


def cluster_decisions(
    steps: list[GLEEStep],
    *,
    min_cluster_size: int = _MIN_CLUSTER_SIZE_DEFAULT,
) -> tuple[list[DecisionCluster], int]:
    """Group steps into decision clusters by coarse state signature.

    First-pass heuristic clustering, per game_family, on
    (round-normalized-position, family-specific coarse state features).
    See the module docstring's "INTEGRITY NOTE ON SCHEMA ASSUMPTIONS".

    Args:
        steps: All loaded :class:`GLEEStep` records.
        min_cluster_size: Clusters with fewer members than this are
            dropped (there is nothing to measure path-invariant agreement
            over with a single history).

    Returns:
        A 2-tuple of:

        - Ordered list of :class:`DecisionCluster` with
          ``len(members) >= min_cluster_size``.
        - Count of steps excluded before clustering (unrecognised
          game_family or a signature function returning ``None``).
    """
    buckets: dict[tuple[str, tuple[Any, ...]], list[GLEEStep]] = {}
    unclustered = 0

    for step in steps:
        sig_fn = _SIGNATURE_FNS.get(step.game_family)
        if sig_fn is None:
            logger.warning("Unrecognised game_family %r; excluding.", step.game_family)
            unclustered += 1
            continue
        signature = sig_fn(step)
        if signature is None:
            logger.warning(
                "Insufficient game_state to sign game_id=%s (family=%s); excluding.",
                step.game_id,
                step.game_family,
            )
            unclustered += 1
            continue
        key = (step.game_family, signature)
        buckets.setdefault(key, []).append(step)

    clusters: list[DecisionCluster] = []
    per_family_index: dict[str, int] = {family: 0 for family in _GAME_FAMILIES}
    for (family, signature), members in buckets.items():
        if len(members) < min_cluster_size:
            continue
        per_family_index[family] += 1
        clusters.append(
            DecisionCluster(
                cluster_id=f"{family}_{per_family_index[family]:03d}",
                game_family=family,
                state_signature=signature,
                members=members,
            )
        )

    logger.info(
        "Clustered %d steps into %d clusters (min_size=%d); %d unclustered.",
        len(steps),
        len(clusters),
        min_cluster_size,
        unclustered,
    )
    return clusters, unclustered


# ---------------------------------------------------------------------------
# Deterministic action_format_compliance
# ---------------------------------------------------------------------------


def check_action_format_compliance(step: GLEEStep) -> bool | None:
    """Deterministically check whether ``action`` matches
    ``valid_actions["fields"]`` exactly.

    Not judge-scored — see the module docstring's rationale for excluding
    this dimension from the kappa computation.

    Args:
        step: The GLEE step to check.

    Returns:
        ``True``/``False`` when evaluable, ``None`` when ``valid_actions``
        has no ``"fields"`` list or ``action`` is not a dict (cannot be
        evaluated with this rule under the current schema assumptions).
    """
    fields = step.valid_actions.get("fields")
    if not isinstance(fields, list) or not isinstance(step.action, dict):
        return None
    return set(step.action.keys()) == set(fields)


def _concession_handling_applicable(step: GLEEStep) -> bool:
    """Heuristic: does this step have a prior rejection to respond to?

    The original :data:`_STATE_KEY_CANDIDATES` ``"opponent_last_action"``
    candidates (``opponent_last_action`` / ``last_offer_rejected`` /
    ``prior_offer`` / ``offer_history``) don't exist in real GLEE data —
    confirmed 0/13 clusters scored this dimension in a live run. Checked
    against a real captured record instead of guessing again: the server
    itself provides ``game_state["history"]``, a list of prior-round dicts
    each carrying a ``"decision"`` field — ``"reject"`` for bargaining,
    ``"RejectOffer"`` for negotiation (confirmed from a real bargaining
    record where the agent's own reasoning referenced the opponent's
    rigidity: game_id 7a9f5e7f..., round 6, history[0..4] each showing
    ``"decision": "reject"``). This is server-side truth already present
    in every logged step, not something that needs our own in-memory
    per-game_id history tracking.

    Persuasion's history entries have no ``"decision"`` field at all (it's
    a signaling game — buyer_decision is "yes"/"no" on whether to buy, not
    a reject/accept of a counter-offer). This function correctly returns
    ``False`` for every persuasion step as a result, which is a genuine
    structural fact about that game family, not a gap to fix.

    Returns:
        ``True`` if ``game_state["history"]`` contains at least one prior
        round whose ``"decision"`` value contains "reject"
        (case-insensitive); ``False`` otherwise, including when
        ``"history"`` is missing, not a list, or empty (e.g. round 1).
    """
    history = step.game_state.get("history")
    if not isinstance(history, list):
        return False
    return any(
        isinstance(entry, dict)
        and isinstance(entry.get("decision"), str)
        and "reject" in entry["decision"].lower()
        for entry in history
    )


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


class _JudgeClient(Protocol):
    """Structural interface for scoring one cluster member's reasoning.

    Tests inject a stub implementing this protocol so the suite never
    calls a live API.
    """

    def score_member(
        self,
        cluster: DecisionCluster,
        member: GLEEStep,
        judge_name: str,
    ) -> dict[str, Any]:
        """Return raw judge scores; keys match :data:`_GLEE_DIMENSIONS`
        plus ``"rationale"``."""
        ...


def _extract_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object from a model response.

    Handles a bare JSON object, a ```json fenced block, or a JSON object
    embedded in surrounding prose — mirrors
    :meth:`src.annotation.annotator_simulator.AnnotatorSimulator._extract_json`.

    Args:
        text: Raw model response text.

    Returns:
        Parsed JSON dict.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    stripped = text.strip()
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    if "```" in stripped:
        fenced = stripped.split("```")
        for block in fenced:
            block = block.removeprefix("json").strip()
            try:
                result = json.loads(block)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(stripped[start : end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract a JSON object from response: {text[:200]!r}")


def _build_user_prompt(cluster: DecisionCluster, member: GLEEStep) -> str:
    """Build the per-member user prompt.

    Deliberately shows only this one member's context — no cross-member
    disclosure, no mention of cluster size or position, no reference
    answer. This is the GLEE analog of Mode B withholding step count/type
    from wearable raters: nothing about the *other* histories in the
    cluster can anchor the judge's score.

    Args:
        cluster: Owning cluster (used for game_family only).
        member: The step being scored.

    Returns:
        Multi-line prompt string.
    """
    lines = [
        f"Game family: {cluster.game_family}",
        f"Phase: {member.phase}  Round: {member.round}",
        f"game_state: {json.dumps(member.game_state, default=str)}",
        f"valid_actions: {json.dumps(member.valid_actions, default=str)}",
        "",
        "=== AGENT REASONING ===",
        member.reasoning,
        "",
        f"=== AGENT ACTION ===\n{json.dumps(member.action, default=str)}",
        "",
        "Return ONLY the JSON object — no explanation outside the JSON.",
    ]
    return "\n".join(lines)


class _AnthropicJudgeClient:
    """Live judge implementation backed by the Anthropic Messages API.

    Args:
        model: Anthropic model id.
        api_key: Defaults to the ``ANTHROPIC_API_KEY`` environment
            variable via the SDK's own resolution.
    """

    def __init__(self, model: str = _MODEL, api_key: str | None = None) -> None:
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def score_member(
        self,
        cluster: DecisionCluster,
        member: GLEEStep,
        judge_name: str,
    ) -> dict[str, Any]:
        """Call the Anthropic API and return parsed judge scores.

        Retries up to :data:`_MAX_RETRIES` times with exponential
        back-off on rate-limit/overload errors, mirroring
        :meth:`src.annotation.annotator_simulator.AnnotatorSimulator._call_api`.

        Args:
            cluster: Owning cluster.
            member: The step being scored.
            judge_name: One of :data:`_JUDGES`.

        Returns:
            Parsed JSON dict with the 4 dimension keys plus ``rationale``.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        system_prompt = _SYSTEM_PROMPTS[judge_name]
        user_prompt = _build_user_prompt(cluster, member)
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                message = self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    system=[
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
                first_block = message.content[0]
                if not isinstance(first_block, anthropic.types.TextBlock):
                    raise ValueError(
                        f"Expected TextBlock, got {type(first_block).__name__}"
                    )
                return _extract_json(first_block.text)

            except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
                wait = _RETRY_BASE_SLEEP_S * (2 ** (attempt - 1))
                logger.warning(
                    "API error on attempt %d/%d (%s); retrying in %.1fs",
                    attempt,
                    _MAX_RETRIES,
                    type(exc).__name__,
                    wait,
                )
                last_exc = exc
                time.sleep(wait)

            except (ValueError, KeyError, IndexError) as exc:
                logger.warning(
                    "Parse error on attempt %d/%d: %s", attempt, _MAX_RETRIES, exc
                )
                last_exc = exc

        raise RuntimeError(
            f"Judge API call failed after {_MAX_RETRIES} attempts"
        ) from last_exc


# ---------------------------------------------------------------------------
# IRR computation
# ---------------------------------------------------------------------------


# Dimensions every judge response MUST carry. concession_handling is
# deliberately excluded -- it's legitimately absent whenever
# _concession_handling_applicable() is False, which is not a failure.
_REQUIRED_JUDGE_DIMENSIONS: tuple[str, ...] = (
    "valuation_reasoning",
    "horizon_strategy_planning",
    "outcome_consistency",
)


def _missing_required_dimensions(raw: dict[str, Any]) -> tuple[str, ...]:
    """Which of :data:`_REQUIRED_JUDGE_DIMENSIONS` ``raw`` is missing, or
    has a value that can't be coerced to ``int`` (e.g. absent, ``null``,
    or a non-numeric string).

    A judge response can be perfectly valid JSON (so ``_extract_json``
    never raises) and still be missing a required key -- that failure
    mode lives one layer above JSON parsing, which is why it isn't
    handled inside :class:`_AnthropicJudgeClient`.

    Args:
        raw: A judge's parsed response dict.

    Returns:
        Tuple of missing/invalid dimension names, in
        :data:`_REQUIRED_JUDGE_DIMENSIONS` order; empty if all present.
    """
    missing = []
    for key in _REQUIRED_JUDGE_DIMENSIONS:
        value = raw.get(key)
        if value is None:
            missing.append(key)
            continue
        try:
            int(value)
        except (TypeError, ValueError):
            missing.append(key)
    return tuple(missing)


def _annotate_cluster(
    cluster: DecisionCluster,
    judge_client: _JudgeClient,
) -> tuple[list[GLEEReasoningAnnotation], int]:
    """Score every non-fallback member of ``cluster`` with every judge.

    If a judge's response is missing a required dimension (see
    :func:`_missing_required_dimensions`), retries up to
    :data:`_MAX_RETRIES` times -- the same budget
    :class:`_AnthropicJudgeClient` uses for parse failures, since this is
    the same class of failure one layer up (valid JSON, wrong shape,
    rather than invalid JSON). Every retry attempt logs the judge's raw
    parsed response at WARNING so a systematic pattern (as opposed to a
    one-off) is visible in the logs rather than silently retried away.
    If still missing after retries, that specific dimension is recorded
    as ``None`` on the annotation AND named in
    ``judge_failed_dimensions`` (logged at ERROR) -- deliberately NOT
    treated the same as ``concession_handling``'s legitimate ``None``,
    even though both end up excluded from that dimension's kappa the same
    way.

    Args:
        cluster: The cluster to annotate.
        judge_client: Live or stub judge implementation.

    Returns:
        A 2-tuple of (flat annotation list, count of members skipped for
        having ``fallback_used=True``).
    """
    annotations: list[GLEEReasoningAnnotation] = []
    skipped_fallback = 0

    for idx, member in enumerate(cluster.members):
        if member.fallback_used:
            skipped_fallback += 1
            continue
        applicable = _concession_handling_applicable(member)
        for judge_name in _JUDGES:
            raw: dict[str, Any] = {}
            missing: tuple[str, ...] = ()
            for attempt in range(1, _MAX_RETRIES + 1):
                raw = judge_client.score_member(cluster, member, judge_name)
                missing = _missing_required_dimensions(raw)
                if not missing:
                    break
                logger.warning(
                    "Judge %s incomplete for %s member %d (attempt %d/%d) "
                    "-- missing %s. Raw parsed response: %s",
                    judge_name,
                    cluster.cluster_id,
                    idx,
                    attempt,
                    _MAX_RETRIES,
                    missing,
                    json.dumps(raw, default=str),
                )
            if missing:
                logger.error(
                    "Judge %s failed to produce %s for %s member %d after "
                    "%d attempts -- recording as missing (NOT as N/A) and "
                    "excluding from that dimension's kappa.",
                    judge_name,
                    missing,
                    cluster.cluster_id,
                    idx,
                    _MAX_RETRIES,
                )

            concession = raw.get("concession_handling") if applicable else None
            annotations.append(
                GLEEReasoningAnnotation(
                    annotation_id=f"{cluster.cluster_id}/{idx}/{judge_name}",
                    cluster_id=cluster.cluster_id,
                    member_index=idx,
                    judge_name=judge_name,
                    valuation_reasoning=(
                        int(raw["valuation_reasoning"])
                        if "valuation_reasoning" not in missing
                        else None
                    ),
                    horizon_strategy_planning=(
                        int(raw["horizon_strategy_planning"])
                        if "horizon_strategy_planning" not in missing
                        else None
                    ),
                    concession_handling=(
                        int(concession) if concession is not None else None
                    ),
                    outcome_consistency=(
                        int(raw["outcome_consistency"])
                        if "outcome_consistency" not in missing
                        else None
                    ),
                    rationale=str(raw.get("rationale", "")),
                    judge_failed_dimensions=missing,
                )
            )
            time.sleep(_INTER_CALL_SLEEP_S)

    return annotations, skipped_fallback


def _dimension_kappa(
    irr: IRRCalculator,
    dimension: str,
    annotations: list[GLEEReasoningAnnotation],
) -> tuple[float | None, int]:
    """Fleiss' kappa for one dimension across a set of annotations.

    An item (``member_index``) is included only if EVERY judge in
    :data:`_JUDGES` has a non-``None`` score for ``dimension`` -- Fleiss'
    kappa needs a complete rater panel per item, and
    ``pia_calculator._build_label_matrix`` has no tolerance for a ragged
    one (it indexes every (item, judge) pair unconditionally). A member
    with a PARTIAL panel -- some judges scored it, at least one didn't --
    is excluded from this dimension's kappa entirely and logged at
    WARNING, since that's a real judge-reliability signal (e.g. a
    required dimension surviving retries for 2/3 judges but not the
    third). This is deliberately distinct from a member where EVERY judge
    agrees the dimension doesn't apply at all (concession_handling's
    ordinary case) -- that's expected and is never logged.

    Args:
        irr: Shared :class:`IRRCalculator` instance.
        dimension: One of :data:`_GLEE_DIMENSIONS`.
        annotations: Annotations to compute over (scoped to one cluster).

    Returns:
        ``(kappa, n_partial_panel_excluded)``. ``kappa`` is ``None`` if
        fewer than 2 members have a complete panel for this dimension.
        ``n_partial_panel_excluded`` counts members dropped specifically
        for a partial (not uniformly-``None``) panel -- worth surfacing
        separately, since a high count on one dimension is a judge
        reliability problem, not just noise.
    """
    by_member: dict[int, dict[str, int | None]] = {}
    for ann in annotations:
        by_member.setdefault(ann.member_index, {})[ann.judge_name] = getattr(
            ann, dimension
        )

    item_ids: list[int] = []
    n_partial_excluded = 0
    for member_index, scores_by_judge in by_member.items():
        present = {j: s for j, s in scores_by_judge.items() if s is not None}
        if len(present) == len(_JUDGES):
            item_ids.append(member_index)
        elif present:
            missing = [j for j in _JUDGES if j not in present]
            cluster_id = next(
                a.cluster_id for a in annotations if a.member_index == member_index
            )
            logger.warning(
                "Partial judge panel for %s member %d, dimension %r -- "
                "missing %s, present %s. Excluding this member from this "
                "dimension's kappa (Fleiss' kappa needs a complete panel).",
                cluster_id,
                member_index,
                dimension,
                missing,
                list(present),
            )
            n_partial_excluded += 1
        # else: every judge is None -- legitimate uniform N/A, skip quietly.

    if len(item_ids) < 2:
        return None, n_partial_excluded

    score_map: dict[tuple[int, str], int] = {
        (member_index, judge_name): score
        for member_index in item_ids
        for judge_name, score in by_member[member_index].items()
        if score is not None
    }

    matrix = _build_label_matrix(
        [str(i) for i in item_ids],
        list(_JUDGES),
        {(str(k[0]), k[1]): v for k, v in score_map.items()},
        scale_offset=1,
    )
    return _fleiss_kappa(irr, matrix, _GLEE_SCALE), n_partial_excluded


def _score_cluster(
    irr: IRRCalculator,
    cluster: DecisionCluster,
    judge_client: _JudgeClient,
) -> GLEEClusterResult | None:
    """Annotate and score one cluster.

    Args:
        irr: Shared :class:`IRRCalculator` instance.
        cluster: Cluster to score.
        judge_client: Live or stub judge implementation.

    Returns:
        :class:`GLEEClusterResult`, or ``None`` if fewer than 2 non-
        fallback members remain to judge.
    """
    annotations, skipped_fallback = _annotate_cluster(cluster, judge_client)
    n_judged = len(cluster.members) - skipped_fallback
    if n_judged < 2:
        logger.warning(
            "Cluster %s has only %d judgeable members after excluding "
            "fallback actions; skipping.",
            cluster.cluster_id,
            n_judged,
        )
        return None

    kappa_per_dim: dict[str, float] = {}
    partial_panel_excluded: dict[str, int] = {}
    for dim in _GLEE_DIMENSIONS:
        kappa, n_excluded = _dimension_kappa(irr, dim, annotations)
        if kappa is not None:
            kappa_per_dim[dim] = kappa
        if n_excluded:
            partial_panel_excluded[dim] = n_excluded

    kappa_overall = (
        sum(kappa_per_dim.values()) / len(kappa_per_dim) if kappa_per_dim else 0.0
    )

    compliance_values = [
        v
        for m in cluster.members
        if not m.fallback_used
        for v in (check_action_format_compliance(m),)
        if v is not None
    ]
    compliance_rate = (
        sum(compliance_values) / len(compliance_values) if compliance_values else None
    )

    return GLEEClusterResult(
        cluster_id=cluster.cluster_id,
        game_family=cluster.game_family,
        n_members=len(cluster.members),
        n_judged_members=n_judged,
        kappa_per_dimension=kappa_per_dim,
        kappa_overall=kappa_overall,
        action_format_compliance_rate=compliance_rate,
        fallback_rate=skipped_fallback / len(cluster.members),
        partial_panel_excluded=partial_panel_excluded,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class GLEEPIABaseline:
    """Entry point: load -> cluster -> judge -> score -> save.

    Args:
        log_path: Path to the GLEE JSONL trajectory log.
        output_path: Where to write the result JSON.
        model: Anthropic model id for the judge panel.
        min_cluster_size: Minimum members for a cluster to be scored.
        limit_clusters: If set, only score the first N clusters (cost
            control for a first pass) — reported clusters are still a
            representative cross-section, not a truncated-but-labeled-
            complete run.
        judge_client: Injected judge implementation. Defaults to a live
            :class:`_AnthropicJudgeClient`. Tests inject a stub.
    """

    def __init__(
        self,
        log_path: Path,
        output_path: Path = Path("data/rsi_bench/glee_pia_results.json"),
        model: str = _MODEL,
        min_cluster_size: int = _MIN_CLUSTER_SIZE_DEFAULT,
        limit_clusters: int | None = None,
        judge_client: _JudgeClient | None = None,
    ) -> None:
        self.log_path = log_path
        self.output_path = output_path
        self.model = model
        self.min_cluster_size = min_cluster_size
        self.limit_clusters = limit_clusters
        self._judge_client: _JudgeClient = judge_client or _AnthropicJudgeClient(
            model=model
        )
        self._irr = IRRCalculator()

    def run(self) -> GLEEPIAResult:
        """Full pipeline: load, cluster, judge every cluster, aggregate, save.

        Returns:
            :class:`GLEEPIAResult` after writing the output JSON.
        """
        steps = load_glee_log(self.log_path)
        models_represented = sorted({s.model for s in steps})

        clusters, unclustered = cluster_decisions(
            steps, min_cluster_size=self.min_cluster_size
        )
        if self.limit_clusters is not None:
            clusters = clusters[: self.limit_clusters]

        per_cluster: dict[str, GLEEClusterResult] = {}
        n_skipped = 0
        for cluster in clusters:
            cluster_result = _score_cluster(self._irr, cluster, self._judge_client)
            if cluster_result is None:
                n_skipped += 1
                continue
            per_cluster[cluster.cluster_id] = cluster_result

        result = self._aggregate(
            per_cluster=per_cluster,
            n_clusters_skipped=n_skipped,
            unclustered_step_count=unclustered,
            models_represented=models_represented,
        )
        self.save(result)
        return result

    def _aggregate(
        self,
        per_cluster: dict[str, GLEEClusterResult],
        n_clusters_skipped: int,
        unclustered_step_count: int,
        models_represented: list[str],
    ) -> GLEEPIAResult:
        """Assemble per-dimension, overall, and per-game_family aggregates.

        Args:
            per_cluster: Scored clusters.
            n_clusters_skipped: Clusters found but not scored.
            unclustered_step_count: Steps excluded before clustering.
            models_represented: Distinct model ids seen in the input log.

        Returns:
            :class:`GLEEPIAResult` ready for serialisation.
        """
        per_dimension: dict[str, list[float]] = {dim: [] for dim in _GLEE_DIMENSIONS}
        for cr in per_cluster.values():
            for dim, kappa in cr.kappa_per_dimension.items():
                per_dimension[dim].append(kappa)

        per_dimension_kappa = {
            dim: sum(vals) / len(vals) for dim, vals in per_dimension.items() if vals
        }
        overall_kappa = (
            sum(per_dimension_kappa.values()) / len(per_dimension_kappa)
            if per_dimension_kappa
            else 0.0
        )

        by_family: dict[str, list[float]] = {family: [] for family in _GAME_FAMILIES}
        for cr in per_cluster.values():
            by_family[cr.game_family].append(cr.kappa_overall)
        by_game_family = {
            family: GameFamilyComparison(
                game_family=family,
                n_clusters=len(vals),
                kappa_overall=sum(vals) / len(vals),
            )
            for family, vals in by_family.items()
            if vals
        }

        compliance_values = [
            cr.action_format_compliance_rate
            for cr in per_cluster.values()
            if cr.action_format_compliance_rate is not None
        ]
        compliance_overall = (
            sum(compliance_values) / len(compliance_values)
            if compliance_values
            else None
        )

        notes = [
            "action_format_compliance is computed deterministically and is "
            "excluded from overall_kappa (see module docstring).",
            "Judges are not calibrated to a target kappa; this is a live, "
            "uncalibrated measurement, unlike pia_calculator.py's dry-run "
            "tables.",
            "Steps with fallback_used=True are excluded from judging "
            "(no genuine reasoning to score) and reported via fallback_rate.",
        ]
        if unclustered_step_count:
            notes.append(
                f"{unclustered_step_count} steps were excluded before "
                "clustering (unrecognised game_family or insufficient "
                "game_state keys under current schema assumptions)."
            )

        return GLEEPIAResult(
            generated_at=datetime.now(UTC).isoformat(),
            model=self.model,
            judges=list(_JUDGES),
            n_clusters=len(per_cluster),
            n_clusters_skipped=n_clusters_skipped,
            unclustered_step_count=unclustered_step_count,
            models_represented=models_represented,
            per_dimension_kappa={
                k: round(v, 4) for k, v in per_dimension_kappa.items()
            },
            overall_kappa=round(overall_kappa, 4),
            interpretation=_kappa_interpretation(overall_kappa),
            per_cluster=per_cluster,
            by_game_family=by_game_family,
            action_format_compliance_overall=compliance_overall,
            notes=notes,
        )

    def save(self, result: GLEEPIAResult) -> Path:
        """Write ``result`` to :attr:`output_path` as indented JSON.

        Args:
            result: Result to serialise.

        Returns:
            The path written to.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        logger.info("Wrote GLEE PIA results to %s.", self.output_path)
        return self.output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    log_path: Path = typer.Option(..., "--log-path", help="GLEE JSONL trajectory log."),
    output: Path = typer.Option(
        Path("data/rsi_bench/glee_pia_results.json"),
        "--output",
        help="Output path for the result JSON.",
    ),
    model: str = typer.Option(
        _MODEL, "--model", help="Anthropic model id for judging."
    ),
    min_cluster_size: int = typer.Option(
        _MIN_CLUSTER_SIZE_DEFAULT,
        "--min-cluster-size",
        help="Minimum members for a cluster to be scored.",
    ),
    limit_clusters: int | None = typer.Option(
        None,
        "--limit-clusters",
        help="Score only the first N clusters (cost control for a first pass).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the GLEE PIA baseline. Always calls the live Anthropic API —
    there is no --dry-run mode (see module docstring)."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    baseline = GLEEPIABaseline(
        log_path=log_path,
        output_path=output,
        model=model,
        min_cluster_size=min_cluster_size,
        limit_clusters=limit_clusters,
    )
    result = baseline.run()

    typer.echo("\n── GLEE PIA Baseline ────────────────────────────────────")
    typer.echo(
        f"  Clusters scored : {result.n_clusters} "
        f"(skipped: {result.n_clusters_skipped})"
    )
    typer.echo(
        f"  Overall κ       = {result.overall_kappa:.4f} ({result.interpretation})"
    )
    typer.echo("\n── Per dimension ────────────────────────────────────────")
    for dim, kappa in result.per_dimension_kappa.items():
        typer.echo(f"  {dim:<26} κ = {kappa:.4f}")
    typer.echo("\n── By game family ───────────────────────────────────────")
    for family, comp in result.by_game_family.items():
        typer.echo(
            f"  {family:<12} n={comp.n_clusters:<3} κ = {comp.kappa_overall:.4f}"
        )
    if result.action_format_compliance_overall is not None:
        typer.echo(
            f"\n  action_format_compliance (diagnostic, not in κ): "
            f"{result.action_format_compliance_overall:.2%}"
        )
    typer.echo(f"\nResults written to {output}\n")


if __name__ == "__main__":
    app()
