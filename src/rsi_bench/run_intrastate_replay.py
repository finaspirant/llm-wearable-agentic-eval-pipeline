"""Same-state replay: resample the committed GLEE agent at saved decision points.

Measures how much the committed agent's own action varies when it is asked
the SAME decision again, independent of judge scoring entirely -- this
module never imports ``src.rsi_bench.glee_pia_baseline`` and makes no judge
calls of any kind. It makes real calls to the Anthropic API for the AGENT
itself (``claude-sonnet-4-6``, the model the committed agent used), gated by
hard cost/call caps.

Reused UNMODIFIED from the committed agent (:mod:`src.glee.claude_glee_agent`):

- :data:`~src.glee.claude_glee_agent.SYSTEM_PROMPT`
- :func:`~src.glee.claude_glee_agent.build_user_prompt` (includes the
  runtime flat-example-action hint via its internal
  ``_build_example_action`` call)
- :func:`~src.glee.claude_glee_agent.parse_response` (includes the
  schema-echo unwrap)

Reused UNMODIFIED from the rest of this package:

- :func:`~src.rsi_bench.analyze_stratified_pia.extract_action_feature` --
  the exact extractor used by the action-only baseline (section 15 of
  ``glee_pia_analysis.md``) -- plus its ``_sd``/``_modal_share``/
  ``_split_features``/``_action_spread``/``_mean``/``_percentile_ci``
  helpers.
- :data:`~src.rsi_bench.run_stratified_pia._INPUT_PRICE_PER_MTOK` /
  :data:`~src.rsi_bench.run_stratified_pia._OUTPUT_PRICE_PER_MTOK` -- the
  same $3/$15-per-MTok real-usage cost estimate already used for the judge
  scoring driver, applied here to the agent's own token usage instead.

DEVIATIONS FROM THE COMMITTED AGENT'S REAL RUNTIME BEHAVIOUR (read before
trusting a replay number as "what the real agent would have done")
--------------------------------------------------------------------------
1. ``game["prompt"]`` -- the game's own rules/situation prose that
   :func:`build_user_prompt` interpolates first -- was never logged
   anywhere (:class:`~src.glee.claude_glee_agent` logs only
   ``game_state``/``valid_actions``/``reasoning``/``action``, never the
   raw ``game`` payload's ``"prompt"`` key; confirmed by grepping every
   field name written in ``log_trajectory_step``). It is reconstructed
   here as an empty string (``game["prompt"] = ""``), not omitted from the
   call entirely (the committed function does ``f"{game['prompt']}\\n"``
   unconditionally and would raise ``KeyError`` on a missing key) -- so
   every replayed prompt is missing one paragraph of game-rules prose the
   real agent saw. This is reported in every output file's metadata, not
   hidden.
2. ``valid_actions`` -- also never logged per-state in
   ``intrastate_variance.json`` (that file stores only ``game_state`` plus
   each archived call's raw reasoning/action). Reconstructed via
   :data:`_CANONICAL_ACTION_SCHEMAS`, a lookup table confirmed by a direct
   one-time scan of the frozen trajectory log
   (``tests/experiments/glee/trajectories.jsonl.gz``, 26,319 raw steps)
   keyed by ``(game_family, action_type)``. ``action_type`` itself is
   inferred by :func:`infer_action_type` (``phase`` directly for
   bargaining/negotiation; ``game_state["seller_message_type"]`` to
   disambiguate persuasion's two ``seller_message``-phase sub-schemas).
   Where the scan showed more than one field-set variant for a given
   ``(family, action_type)`` pair (an optional ``"message"`` field present
   or absent server-side, or negotiation's final-round "no counteroffer
   possible" decision variant), the SUPERSET variant is used -- i.e. every
   replay sees the fuller schema, which may occasionally offer a field the
   real decision point did not actually have available.
3. No retry-on-parse-failure and no ``safe_action`` fallback. The
   committed agent's ``ask_claude`` retries once (feeding the bad reply
   back) and ``_play`` falls back to a guaranteed-legal ``safe_action`` on
   total failure, because in real play a dropped move costs a real game
   turn. Here nothing is at stake except the measurement itself, and
   retrying-until-parseable would systematically bias the very
   parse-failure-rate and action distribution this module exists to
   measure -- so each of the 20 samples per state is exactly ONE real API
   call, recorded whatever the outcome (success, parse failure, or API
   failure), never retried, never dropped, never replaced by a fallback
   action.
4. The committed agent's own circuit breaker
   (``CIRCUIT_BREAKER_THRESHOLD = 20``) is reproduced here as
   :class:`_CircuitBreaker` with the same threshold and the same
   "a successful HTTP response resets the counter even if parsing then
   fails" rule, but as an independent instance scoped to one
   :func:`run_replay` call -- it does not share state with (or get reset
   by) any other process.

Determinism / resume
---------------------------------------------------------------------------
Every row written to the output JSONL carries ``(state_id, temp_label,
sample_idx)``. Before dispatching any new call, :func:`run_replay` reads
every ``(state_id, temp_label, sample_idx)`` triple already present in the
output file and skips re-requesting it -- an interrupted run can be resumed
by re-invoking with the same output path, and a resumed run performs ZERO
additional API calls for work already recorded. The bootstrap in
:func:`aggregate_replay` resamples at the STATE level (not the
sample level) within each family, one ``random.Random(seed)`` consumed in
fixed family order, matching the determinism convention used throughout
:mod:`src.rsi_bench.analyze_stratified_pia`.

CLI:
    python -m src.rsi_bench.run_intrastate_replay \\
        --max-cost-usd 4 --max-calls 400
"""

from __future__ import annotations

import copy
import json
import logging
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic
import typer

from src.glee.claude_glee_agent import SYSTEM_PROMPT, build_user_prompt, parse_response
from src.rsi_bench.analyze_stratified_pia import (
    _action_spread,
    _mean,
    _modal_share,
    _percentile_ci,
    _sd,
    _split_features,
    extract_action_feature,
)
from src.rsi_bench.run_stratified_pia import (
    _INPUT_PRICE_PER_MTOK,
    _OUTPUT_PRICE_PER_MTOK,
)

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="run-intrastate-replay",
    help=(
        "Same-state replay of the committed GLEE agent. Makes real agent "
        "API calls; zero judge calls."
    ),
    add_completion=False,
)

_GAME_FAMILIES: tuple[str, ...] = ("bargaining", "negotiation", "persuasion")

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_STATES_INPUT = Path("tests/experiments/glee/intrastate_variance.json")
_DEFAULT_SCORED_OUTPUT = Path("data/rsi_bench/glee_replay_scored.jsonl")
_DEFAULT_SUMMARY_OUTPUT = Path("data/rsi_bench/glee_replay_summary.json")
_DEFAULT_BASELINE_ANALYSIS_INPUT = Path("data/rsi_bench/glee_pia_analysis.json")
_DEFAULT_SEED = 20261002
_DEFAULT_N_SAMPLES_PER_TEMP = 10
_DEFAULT_CONCURRENCY = 4
_DEFAULT_TIMEOUT_S = 60
_DEFAULT_MAX_TOKENS = 600  # matches src.glee.claude_glee_agent.ask_claude
_DEFAULT_MAX_CALLS = 400
_DEFAULT_MAX_COST_USD = 4.0
_DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 20
_DEFAULT_N_BOOTSTRAP_RESAMPLES = 10_000

# (temp_label, temperature) -- temperature=None means the kwarg is omitted
# from the API call entirely, i.e. genuinely the API's own default, not a
# locally-chosen "default value" passed explicitly.
_TEMPERATURES: tuple[tuple[str, float | None], ...] = (
    ("api_default", None),
    ("t0", 0.0),
)

# Canonical valid_actions field schemas -- see module docstring deviation 2.
_CANONICAL_ACTION_SCHEMAS: dict[tuple[str, str], dict[str, str]] = {
    ("bargaining", "offer"): {
        "alice_gain": "number (your proposed amount for Alice)",
        "bob_gain": "number (your proposed amount for Bob)",
        "message": "string (optional message to opponent)",
    },
    ("bargaining", "decision"): {
        "decision": (
            "'accept', 'reject', or 'walkaway' (leave the bargaining "
            "— the money is not divided, both get $0)"
        ),
    },
    ("negotiation", "offer"): {
        "product_price": "number (your proposed price)",
        "message": "string (optional message)",
    },
    ("negotiation", "decision"): {
        "decision": "'AcceptOffer', 'RejectOffer', or 'WalkAway'",
        "product_price": "number (required if RejectOffer - your counteroffer)",
        "message": "string (optional)",
    },
    ("persuasion", "seller_recommendation"): {
        "decision": "'yes' (recommend) or 'no' (don't recommend)",
    },
    ("persuasion", "seller_message"): {
        "message": "string (your message to the buyer)",
    },
    ("persuasion", "buyer_decision"): {
        "decision": "'yes' (buy) or 'no' (don't buy)",
    },
}


# ---------------------------------------------------------------------------
# State loading + prompt reconstruction
# ---------------------------------------------------------------------------


def infer_action_type(game_family: str, phase: str, game_state: dict[str, Any]) -> str:
    """Infer ``valid_actions["type"]`` for one state.

    ``phase`` IS the action type for bargaining/negotiation (confirmed by
    direct scan of the frozen trajectory log: every bargaining/negotiation
    step's ``valid_actions["type"] == step["phase"]``). Persuasion's
    ``seller_message`` phase covers two different action schemas
    disambiguated by ``game_state["seller_message_type"]``
    (``"binary"`` -> ``"seller_recommendation"``, else -> ``"seller_message"``).

    Raises:
        ValueError: If ``game_family``/``phase`` is not one of the 7
            combinations seen in the frozen log.
    """
    if game_family in ("bargaining", "negotiation"):
        return phase
    if game_family == "persuasion":
        if phase == "buyer_decision":
            return "buyer_decision"
        if phase == "seller_message":
            smt = game_state.get("seller_message_type")
            return "seller_recommendation" if smt == "binary" else "seller_message"
    raise ValueError(
        f"Cannot infer action type for game_family={game_family!r} phase={phase!r}"
    )


def build_valid_actions(
    game_family: str, phase: str, game_state: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct a ``valid_actions`` dict for one state -- see deviation 2."""
    action_type = infer_action_type(game_family, phase, game_state)
    fields = _CANONICAL_ACTION_SCHEMAS.get((game_family, action_type))
    if fields is None:
        raise ValueError(f"No canonical schema for ({game_family!r}, {action_type!r})")
    return {"type": action_type, "fields": dict(fields)}


def unwrap_schema_echo(action: Any) -> Any:  # noqa: ANN401 -- arbitrary parsed JSON
    """Same unwrap rule as :func:`src.glee.claude_glee_agent.parse_response`.

    Applied to the archived ``calls[0]["action"]`` too (not just live
    replay samples) so the "originally logged action" comparison target
    is on the same flat footing as every replay sample.
    """
    if isinstance(action, dict) and isinstance(action.get("fields"), dict):
        return action["fields"]
    return action


def load_intrastate_states(path: Path = _DEFAULT_STATES_INPUT) -> list[dict[str, Any]]:
    """Load the 18 saved states and derive a stable ``state_id`` for each.

    ``state_id`` is ``f"{index:02d}_{game_family}_{game_id}_{round}"`` --
    stable across runs because it is derived purely from the input file's
    fixed order plus content, never randomly generated.

    "Originally logged action" (for the agreement comparison) is taken as
    ``calls[0]["action"]`` -- the first chronologically-logged call for
    this exact ``(game_id, round)`` in the state's own ``source_file`` --
    unwrapped via :func:`unwrap_schema_echo`. ``intrastate_variance.json``
    itself documents each record's ``calls`` as consecutive log lines for
    one state; the first one is what was actually logged (and, for the
    pre-pollfix archived runs, what the poll-loop race then uselessly
    redispatched).
    """
    raw = json.loads(Path(path).read_text())
    states: list[dict[str, Any]] = []
    for i, rec in enumerate(raw):
        game_state = rec["game_state"]
        phase = game_state["phase"]
        family = rec["game_family"]
        state_id = f"{i:02d}_{family}_{rec['game_id']}_{rec['round']}"
        original_action = unwrap_schema_echo(copy.deepcopy(rec["calls"][0]["action"]))
        states.append(
            {
                "state_id": state_id,
                "index": i,
                "source_tag": rec["source_tag"],
                "source_file": rec["source_file"],
                "source_description": rec["source_description"],
                "game_id": rec["game_id"],
                "round": rec["round"],
                "game_family": family,
                "phase": phase,
                "game_state": game_state,
                "n_archived_calls": len(rec["calls"]),
                "action_differs_across_calls_archived": rec.get(
                    "action_differs_across_calls"
                ),
                "original_action": original_action,
            }
        )
    return states


def build_prompt_for_state(state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Reconstruct the user-turn prompt for one state via the committed
    agent's own :func:`build_user_prompt`, unmodified.

    Returns:
        ``(prompt_text, game_dict_used)`` -- ``game_dict_used`` is
        returned for metadata/debugging only, never reused across calls.
    """
    game_state = state["game_state"]
    valid_actions = build_valid_actions(
        state["game_family"], state["phase"], game_state
    )
    game = {
        "prompt": "",  # deviation 1 -- see module docstring
        "game_id": state["game_id"],
        "game_family": state["game_family"],
        "your_player": game_state.get("current_player"),
        "phase": state["phase"],
        "game_state": game_state,
        "valid_actions": valid_actions,
    }
    # build_user_prompt renders history[:-1] (it treats the LAST entry as
    # "the current round", already reflected in game_state itself) -- so a
    # placeholder is appended to make the full game_state["history"] (which
    # already excludes the current decision) the part that gets rendered.
    history_arg = [
        *game_state.get("history", []),
        {"_replay_current_round_placeholder": True},
    ]
    prompt = build_user_prompt(game, history_arg)
    return prompt, game


# ---------------------------------------------------------------------------
# Budget / circuit breaker
# ---------------------------------------------------------------------------


class _Budget:
    """Thread-safe running cost/call tracker with hard caps.

    Mirrors ``src.rsi_bench.run_stratified_pia``'s check-after-call
    convention: a cap is detected after the call that crosses it, not
    pre-empted before -- so total calls/cost can overshoot by up to
    ``concurrency`` in-flight calls, same as that driver.
    """

    def __init__(self, max_calls: int | None, max_cost_usd: float | None) -> None:
        self._lock = threading.Lock()
        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.n_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.stopped_reason: str | None = None

    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * _INPUT_PRICE_PER_MTOK
            + self.output_tokens / 1_000_000 * _OUTPUT_PRICE_PER_MTOK
        )

    def record_and_check(self, input_tokens: int, output_tokens: int) -> str | None:
        """Record one call's usage; return a stop reason if a cap is now hit."""
        with self._lock:
            self.n_calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if self.stopped_reason is None:
                if self.max_calls is not None and self.n_calls >= self.max_calls:
                    self.stopped_reason = "max_calls"
                elif (
                    self.max_cost_usd is not None
                    and self.cost_usd() >= self.max_cost_usd
                ):
                    self.stopped_reason = "max_cost_usd"
            return self.stopped_reason

    def tripped(self) -> bool:
        with self._lock:
            return self.stopped_reason is not None


class _CircuitBreaker:
    """Stops new dispatch after ``threshold`` consecutive API-level failures.

    A parse failure (the call succeeded; the reply didn't parse) does NOT
    count -- :meth:`note_success` is called on any successful HTTP
    response, matching ``claude_glee_agent._note_api_success``'s "even if
    the reply then fails to parse" rule exactly.
    """

    def __init__(self, threshold: int) -> None:
        self._lock = threading.Lock()
        self.threshold = threshold
        self.consecutive_failures = 0
        self.tripped = False

    def note_success(self) -> None:
        with self._lock:
            self.consecutive_failures = 0

    def note_failure(self) -> bool:
        with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.threshold:
                self.tripped = True
            return self.tripped


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_once(
    client: Any,  # noqa: ANN401 -- Anthropic-shaped client or test mock
    model: str,
    prompt: str,
    temp_value: float | None,
    timeout_s: int,
    max_tokens: int,
) -> dict[str, Any]:
    """Exactly one real API call + parse attempt. Raises on an API-level
    failure (network/timeout/provider error); returns a result dict on any
    HTTP-successful response, whether or not it then parses.

    Returns:
        ``{"ok", "reasoning", "action", "usage", "raw_text", "parse_error"}``.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": timeout_s,
    }
    if temp_value is not None:
        kwargs["temperature"] = temp_value
    response = client.messages.create(**kwargs)
    text = "".join(block.text for block in response.content if block.type == "text")
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    try:
        reasoning, action = parse_response(text)
        return {
            "ok": True,
            "reasoning": reasoning,
            "action": action,
            "usage": usage,
            "raw_text": text,
            "parse_error": None,
        }
    except Exception as exc:  # noqa: BLE001 -- any parse failure, counted not dropped
        return {
            "ok": False,
            "reasoning": None,
            "action": None,
            "usage": usage,
            "raw_text": text,
            "parse_error": f"{type(exc).__name__}: {exc}",
        }


def _load_completed_keys(path: Path) -> set[tuple[str, str, int]]:
    """``(state_id, temp_label, sample_idx)`` triples already in ``path``."""
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            completed.add((rec["state_id"], rec["temp_label"], rec["sample_idx"]))
    return completed


def run_replay(
    states: list[dict[str, Any]],
    output_path: Path,
    *,
    model: str = _DEFAULT_MODEL,
    n_samples_per_temp: int = _DEFAULT_N_SAMPLES_PER_TEMP,
    concurrency: int = _DEFAULT_CONCURRENCY,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_calls: int | None = _DEFAULT_MAX_CALLS,
    max_cost_usd: float | None = _DEFAULT_MAX_COST_USD,
    circuit_breaker_threshold: int = _DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
    client: Any | None = None,  # noqa: ANN401 -- Anthropic-shaped client or test mock
) -> dict[str, Any]:
    """Run (or resume) the replay sampling job.

    Args:
        states: Output of :func:`load_intrastate_states`.
        output_path: Incremental-JSONL output; read first for resume.
        client: Injectable Anthropic-shaped client (``.messages.create``)
            -- tests pass a mock here so the suite makes zero network
            calls; production use constructs a real ``anthropic.Anthropic()``
            when left ``None``.

    Returns:
        Run-level stats: ``{"n_total_tasks", "n_resumed_skipped",
        "n_attempted", "n_calls", "input_tokens", "output_tokens",
        "estimated_cost_usd", "stopped_reason", "circuit_breaker_tripped",
        "outcome_counts"}``.
    """
    client = client or anthropic.Anthropic()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_keys(output_path)
    budget = _Budget(max_calls, max_cost_usd)
    breaker = _CircuitBreaker(circuit_breaker_threshold)
    write_lock = threading.Lock()
    stop_event = threading.Event()

    prompts_by_state_id = {s["state_id"]: build_prompt_for_state(s)[0] for s in states}

    all_keys = [
        (state, temp_label, temp_value, sample_idx)
        for state in states
        for temp_label, temp_value in _TEMPERATURES
        for sample_idx in range(n_samples_per_temp)
    ]
    tasks = [t for t in all_keys if (t[0]["state_id"], t[1], t[3]) not in completed]
    n_resumed_skipped = len(all_keys) - len(tasks)

    def _write(record: dict[str, Any]) -> None:
        with write_lock:
            with output_path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def _worker(task: tuple[dict[str, Any], str, float | None, int]) -> str:
        state, temp_label, temp_value, sample_idx = task
        if stop_event.is_set() or budget.tripped():
            stop_event.set()
            return "skipped_budget"
        prompt = prompts_by_state_id[state["state_id"]]
        base_record = {
            "state_id": state["state_id"],
            "game_id": state["game_id"],
            "round": state["round"],
            "game_family": state["game_family"],
            "temp_label": temp_label,
            "temp_value": temp_value,
            "sample_idx": sample_idx,
            "sampled_at": datetime.now(UTC).isoformat(),
        }
        try:
            result = _sample_once(
                client, model, prompt, temp_value, timeout_s, max_tokens
            )
        except Exception as exc:  # noqa: BLE001 -- API-level failure
            tripped = breaker.note_failure()
            _write(
                {
                    **base_record,
                    "ok": False,
                    "api_error": f"{type(exc).__name__}: {exc}",
                    "parse_error": None,
                    "reasoning": None,
                    "action": None,
                    "usage": None,
                }
            )
            if tripped:
                stop_event.set()
                logger.error(
                    "Circuit breaker tripped after %d consecutive API failures",
                    breaker.consecutive_failures,
                )
                return "circuit_breaker"
            return "api_failure"

        breaker.note_success()
        usage = result["usage"]
        stop_reason = budget.record_and_check(
            usage["input_tokens"], usage["output_tokens"]
        )
        _write(
            {
                **base_record,
                "ok": result["ok"],
                "api_error": None,
                "parse_error": result["parse_error"],
                "reasoning": result["reasoning"],
                "action": result["action"],
                "usage": usage,
            }
        )
        if stop_reason:
            stop_event.set()
            logger.warning("Budget cap reached: %s", stop_reason)
            return stop_reason
        return "ok" if result["ok"] else "parse_failure"

    outcome_counts: Counter[str] = Counter()
    if tasks:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_worker, t): t for t in tasks}
            for fut in as_completed(futures):
                outcome_counts[fut.result()] += 1

    return {
        "n_total_tasks": len(all_keys),
        "n_resumed_skipped": n_resumed_skipped,
        "n_attempted": len(tasks),
        "n_calls": budget.n_calls,
        "input_tokens": budget.input_tokens,
        "output_tokens": budget.output_tokens,
        "estimated_cost_usd": round(budget.cost_usd(), 4),
        "stopped_reason": budget.stopped_reason,
        "circuit_breaker_tripped": breaker.tripped,
        "outcome_counts": dict(outcome_counts),
    }


# ---------------------------------------------------------------------------
# Feature extraction + aggregation (reuses analyze_stratified_pia's extractor)
# ---------------------------------------------------------------------------


def extract_sample_feature(
    sample_action: Any,  # noqa: ANN401 -- arbitrary parsed JSON action
    state: dict[str, Any],
) -> dict[str, Any]:
    """:func:`extract_action_feature`, fed a synthetic raw_step built from
    one replay sample's action plus the state's own phase/game_state."""
    raw_step = {
        "action": sample_action,
        "phase": state["phase"],
        "game_state": state["game_state"],
    }
    return extract_action_feature(raw_step, state["game_family"])


def load_scored_replay_records(path: Path) -> list[dict[str, Any]]:
    """Load ``glee_replay_scored.jsonl`` -- one line per real sample."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def per_state_temp_report(
    state: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """Per-(state, one temperature) action spread / modal share / agreement.

    Args:
        state: One entry from :func:`load_intrastate_states`.
        samples: This state+temperature's recorded replay rows (any
            ``sample_idx``), as loaded from the scored JSONL.

    Returns:
        See module-level field names below; ``None`` wherever a statistic
        does not apply (never a fabricated value).
    """
    ok_samples = [s for s in samples if s["ok"]]
    n_parse_failures = sum(1 for s in samples if s["api_error"] is None and not s["ok"])
    n_api_failures = sum(1 for s in samples if s["api_error"] is not None)

    features = [extract_sample_feature(s["action"], state) for s in ok_samples]
    members = [
        {"feature_type": f["feature_type"], "value": f["value"], "label": f["label"]}
        for f in features
    ]
    continuous_values, categorical_labels = _split_features(members)
    spread, spread_source = _action_spread(members)
    modal_share, modal_label, n_distinct = _modal_share(categorical_labels)

    original_feature = extract_sample_feature(state["original_action"], state)
    agreement_with_original: float | None = None
    mean_abs_diff_from_original: float | None = None
    if original_feature["feature_type"] == "categorical" and categorical_labels:
        agreement_with_original = sum(
            1 for lbl in categorical_labels if lbl == original_feature["label"]
        ) / len(categorical_labels)
    elif original_feature["feature_type"] == "continuous" and continuous_values:
        mean_abs_diff_from_original = _mean(
            [abs(v - original_feature["value"]) for v in continuous_values]
        )

    return {
        "state_id": state["state_id"],
        "game_family": state["game_family"],
        "n_samples_total": len(samples),
        "n_ok": len(ok_samples),
        "n_parse_failures": n_parse_failures,
        "n_api_failures": n_api_failures,
        "n_continuous": len(continuous_values),
        "continuous_sd": _sd(continuous_values),
        "n_categorical": len(categorical_labels),
        "categorical_n_distinct_labels": n_distinct,
        "categorical_modal_label": modal_label,
        "categorical_modal_share": modal_share,
        "action_spread": spread,
        "action_spread_source": spread_source,
        "original_action_feature_type": original_feature["feature_type"],
        "original_action_label": original_feature["label"],
        "original_action_value": original_feature["value"],
        "agreement_with_original": agreement_with_original,
        "mean_abs_diff_from_original": mean_abs_diff_from_original,
    }


def build_per_state_temp_reports(
    states: list[dict[str, Any]], scored_records: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """``(state_id, temp_label) -> per_state_temp_report(...)`` for every state."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in scored_records:
        grouped.setdefault((rec["state_id"], rec["temp_label"]), []).append(rec)

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for state in states:
        for temp_label, _ in _TEMPERATURES:
            samples = grouped.get((state["state_id"], temp_label), [])
            result[(state["state_id"], temp_label)] = per_state_temp_report(
                state, samples
            )
    return result


def aggregate_replay(
    states: list[dict[str, Any]],
    per_state_temp: dict[tuple[str, str], dict[str, Any]],
    seed: int = _DEFAULT_SEED,
    n_resamples: int = _DEFAULT_N_BOOTSTRAP_RESAMPLES,
) -> dict[str, dict[str, Any]]:
    """Per family/temperature mean action spread + a bootstrap CI over states.

    One ``random.Random(seed)`` instance, consumed in fixed
    ``(_GAME_FAMILIES, _TEMPERATURES)`` order, resampling each family's
    per-state ``action_spread`` values (with replacement, same size as the
    number of states in that family with a defined spread) --
    :func:`~src.rsi_bench.analyze_stratified_pia._percentile_ci`'s
    "contribute no value, never a fabricated one" convention applies when
    a family/temperature has zero states with a defined spread.
    """
    rng = random.Random(seed)
    result: dict[str, dict[str, Any]] = {family: {} for family in _GAME_FAMILIES}
    for family in _GAME_FAMILIES:
        family_state_ids = [s["state_id"] for s in states if s["game_family"] == family]
        for temp_label, _ in _TEMPERATURES:
            values = [
                per_state_temp[(sid, temp_label)]["action_spread"]
                for sid in family_state_ids
                if per_state_temp[(sid, temp_label)]["action_spread"] is not None
            ]
            n = len(values)
            series: list[float | None] = []
            for _ in range(n_resamples):
                if n == 0:
                    series.append(None)
                else:
                    series.append(_mean([values[rng.randrange(n)] for _ in range(n)]))
            ci = _percentile_ci(series)
            result[family][temp_label] = {
                "n_states": len(family_state_ids),
                "n_states_with_spread": n,
                "mean_action_spread": _mean(values) if values else None,
                "ci_low": ci["low"],
                "ci_high": ci["high"],
                "n_valid_reps": ci["n_valid_reps"],
                "n_total_reps": ci["n_total_reps"],
            }
    return result


def compare_with_observational_baseline(
    replay_aggregate: dict[str, dict[str, Any]],
    baseline_analysis_path: Path = _DEFAULT_BASELINE_ANALYSIS_INPUT,
) -> dict[str, dict[str, Any]]:
    """Side-by-side: replay mean spread (``"api_default"`` temperature) vs.
    the observational within-cluster action spread from
    ``glee_pia_analysis.json``'s ``action_only_baseline.family_summary``
    (the data behind ``glee_pia_analysis.md`` section 15)."""
    baseline = json.loads(Path(baseline_analysis_path).read_text())
    family_summary = baseline["action_only_baseline"]["family_summary"]
    result: dict[str, dict[str, Any]] = {}
    for family in _GAME_FAMILIES:
        replay_cell = replay_aggregate[family]["api_default"]
        observational_mean = family_summary[family]["mean_action_spread"]
        result[family] = {
            "replay_mean_action_spread_api_default": replay_cell["mean_action_spread"],
            "replay_ci_low": replay_cell["ci_low"],
            "replay_ci_high": replay_cell["ci_high"],
            "replay_n_states_with_spread": replay_cell["n_states_with_spread"],
            "observational_mean_action_spread": observational_mean,
            "observational_n_clusters_with_action_spread": family_summary[family][
                "n_clusters_with_action_spread"
            ],
        }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    states_input: Path = typer.Option(_DEFAULT_STATES_INPUT, "--states-input"),
    scored_output: Path = typer.Option(_DEFAULT_SCORED_OUTPUT, "--scored-output"),
    summary_output: Path = typer.Option(_DEFAULT_SUMMARY_OUTPUT, "--summary-output"),
    baseline_analysis_input: Path = typer.Option(
        _DEFAULT_BASELINE_ANALYSIS_INPUT, "--baseline-analysis-input"
    ),
    model: str = typer.Option(_DEFAULT_MODEL, "--model"),
    n_samples_per_temp: int = typer.Option(
        _DEFAULT_N_SAMPLES_PER_TEMP, "--n-samples-per-temp"
    ),
    concurrency: int = typer.Option(_DEFAULT_CONCURRENCY, "--concurrency"),
    timeout_s: int = typer.Option(_DEFAULT_TIMEOUT_S, "--timeout-s"),
    max_calls: int = typer.Option(_DEFAULT_MAX_CALLS, "--max-calls"),
    max_cost_usd: float = typer.Option(_DEFAULT_MAX_COST_USD, "--max-cost-usd"),
    seed: int = typer.Option(_DEFAULT_SEED, "--seed"),
    n_resamples: int = typer.Option(_DEFAULT_N_BOOTSTRAP_RESAMPLES, "--n-resamples"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Replay the 18 saved states, 20 real agent samples each, then aggregate."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    states = load_intrastate_states(states_input)
    run_stats = run_replay(
        states,
        scored_output,
        model=model,
        n_samples_per_temp=n_samples_per_temp,
        concurrency=concurrency,
        timeout_s=timeout_s,
        max_calls=max_calls,
        max_cost_usd=max_cost_usd,
    )

    scored_records = load_scored_replay_records(scored_output)
    per_state_temp = build_per_state_temp_reports(states, scored_records)
    aggregate = aggregate_replay(
        states, per_state_temp, seed=seed, n_resamples=n_resamples
    )
    comparison = (
        compare_with_observational_baseline(aggregate, baseline_analysis_input)
        if baseline_analysis_input.exists()
        else None
    )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "run_stats": run_stats,
        "prompt_construction_notes": [
            "game['prompt'] prose was never logged anywhere; reconstructed as "
            "an empty string (see module docstring deviation 1) -- every "
            "replayed prompt is missing that rules/situation paragraph.",
            "valid_actions was never logged per-state in "
            "intrastate_variance.json; reconstructed from a canonical "
            "(family, action_type) -> fields schema table confirmed by a "
            "direct scan of trajectories.jsonl.gz, using the superset field "
            "variant where more than one was observed (see deviation 2).",
            "No retry-on-parse-failure and no safe_action fallback -- each "
            "sample is exactly one real API call, recorded as-is (see "
            "deviation 3).",
        ],
        "by_state_temp": {
            f"{sid}|{tl}": report for (sid, tl), report in per_state_temp.items()
        },
        "aggregate_by_family_temp": aggregate,
        "comparison_with_observational_baseline": comparison,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2))
    typer.echo(f"Scored replay written to {scored_output}")
    typer.echo(f"Summary written to {summary_output}")
    typer.echo(
        f"n_calls={run_stats['n_calls']} cost=${run_stats['estimated_cost_usd']:.4f} "
        f"stopped_reason={run_stats['stopped_reason']}"
    )


if __name__ == "__main__":
    app()
