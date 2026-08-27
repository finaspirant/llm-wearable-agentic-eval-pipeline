"""
Claude-powered agent for the GLEE Competition (bargaining / negotiation / persuasion).

Design notes (read before editing):
  - One strategy function per game family + a dispatcher, per the SDK's recommended
    pattern. This is what lets `client.run()` play all three families off one queue
    and lets you tune one family without touching the others.
  - Every call to Claude is asked for explicit step-by-step reasoning *before* the
    action JSON. The reasoning is logged to a JSONL trajectory file alongside the
    action. This is the hook for PIA: PIA scores whether reasoning is consistent
    across different paths to equivalent decision points, so the reasoning text
    has to exist and be retrievable per-decision, not just the final action.
  - Defensive JSON parsing + a guaranteed-legal fallback action, same pattern as
    the SDK's own llm_agent.py example, because a malformed reply or a hung API
    call should cost you one move, never the game.
  - Simple in-memory opponent model (per game_id) — round-by-round offer history
    summarized back into the prompt, so the agent reasons about concession
    patterns rather than reacting to each offer in isolation.
  - Real token usage is captured from every response and logged per-step
    (input/output/cache_creation/cache_read). Prompt caching via
    cache_control was tried and reverted -- see build_user_prompt's
    docstring -- so the cache_* fields are always 0 for now; they're kept
    in the schema since re-enabling caching is a config change, not a
    schema change, if it's revisited later.
  - A module-level circuit breaker stops the process after too many
    consecutive Claude failures in a row (the credits-exhausted failure
    mode from an earlier overnight run, which otherwise spins for hours
    burning fallback moves with no way to notice unattended).

Usage:
    pip install glee-sdk anthropic
    export GLEE_API_KEY=glee_...
    export ANTHROPIC_API_KEY=sk-ant-...
    python claude_glee_agent.py
"""

import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path

from anthropic import Anthropic
from glee_sdk import GleeClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("claude_glee_agent")

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
TRAJECTORY_LOG = Path(
    os.environ.get("GLEE_TRAJECTORY_LOG", "tests/experiments/glee/trajectories.jsonl")
)

anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY from env

# ---------------------------------------------------------------------------
# Opponent-history tracking (in-memory; keyed by game_id)
# ---------------------------------------------------------------------------

_game_histories: dict[str, list[dict]] = {}


def _record_state(game: dict) -> list[dict]:
    """Append this call's state snapshot to the game's running history and
    return the full history so the prompt can show concession patterns."""
    hist = _game_histories.setdefault(game["game_id"], [])
    hist.append(
        {
            "round": game["game_state"].get("round"),
            "phase": game.get("phase"),
            "last_offer": game["game_state"].get("last_offer"),
        }
    )
    return hist


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a strategic player in a language-based economic game (part of \
the GLEE benchmark: bargaining, negotiation, or persuasion). You will be shown the game's \
own rules/situation as prose, the full machine-readable state visible to you (including \
round history), and the exact legal action format for this turn.

Think about this like a careful game theorist, not just a pattern-matcher:
1. Identify what family of game this is and what phase you're in.
2. Work out your reservation value / walk-away point from the state given (e.g. money_to_divide, \
your own valuation, expected value from p/v/u for persuasion).
3. Model the opponent from the history: are they conceding, anchoring, stalling? Is there a \
round cap or inflation pressure that changes the value of waiting?
4. Decide the action that maximizes YOUR OWN payoff over the whole game (not fairness, not \
speed), subject to the rules of the current phase.

First write 2-4 sentences of reasoning under a "REASONING:" heading. Then, on a new line, \
write "ACTION:" followed by ONLY the JSON action object -- no markdown fences, no trailing \
commentary. The reasoning and the action must be consistent with each other: don't reason \
your way to accepting an offer and then reject it.
"""


def _example_value_for_field(description: str) -> object:
    """Infer a plausible placeholder value for one valid_actions field from
    its human-readable type description (e.g. "number (your proposed
    amount for Alice)" -> 0, "string (optional message)" -> "example",
    "'AcceptOffer', 'RejectOffer', or 'WalkAway'" -> "AcceptOffer").

    Args:
        description: The field's description string as given in
            ``game["valid_actions"]["fields"]``.

    Returns:
        A representative placeholder value for the flat example action.
    """
    lowered = description.strip().lower()
    if lowered.startswith("number"):
        return 0
    if lowered.startswith("string"):
        return "example"
    quoted = re.findall(r"'([^']+)'", description)
    if quoted:
        return quoted[0]
    return "example"


def _build_example_action(valid_actions: dict) -> dict:
    """Build a flat example action from a valid_actions schema at runtime.

    Generated fresh per call rather than hardcoded, since the field set
    differs by game_family and phase. Returns ``{}`` when ``valid_actions``
    has no ``"fields"`` dict to build from.

    Args:
        valid_actions: ``game["valid_actions"]`` for the current turn.

    Returns:
        A flat dict of field_name -> placeholder value, with no
        ``"type"``/``"fields"`` wrapper.
    """
    fields = valid_actions.get("fields")
    if not isinstance(fields, dict):
        return {}
    return {name: _example_value_for_field(str(desc)) for name, desc in fields.items()}


def _format_history_block(history: list[dict]) -> str:
    """Render prior-round snapshots as an append-only, byte-stable block.

    One complete JSON object per line, rather than a single
    ``json.dumps(list, indent=2)`` array: an array's closing bracket moves
    every time an element is appended, so round N's serialization is NOT a
    byte-prefix of round N+1's -- that would silently defeat Anthropic's
    prefix-match prompt cache (see build_user_prompt). Rendering one
    complete, never-rewritten line per round guarantees round N's block is
    always exactly round (N-1)'s block plus one new line appended at the
    end, which is what prefix matching needs to get a partial cache hit.

    ``sort_keys=True`` removes any dependency on dict insertion order being
    identical call to call (our own history entries already insert keys in
    a fixed order, but ``last_offer`` is copied verbatim from the server's
    JSON response, whose key order isn't a guarantee we control).

    Args:
        history: Prior-round snapshots for this game, oldest first. Must
            NOT include the current round (callers pass ``history[:-1]``).

    Returns:
        One ``json.dumps(...)`` line per entry, each newline-terminated.
        Empty string when ``history`` is empty (round 1).
    """
    return "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in history)


# Constant, so prepending it to the cached prefix never breaks the
# append-only-prefix guarantee -- it's identical on every call, including
# round 1 when there's no history yet (see build_user_prompt).
_HISTORY_HEADER = (
    "Running history of this game so far, one JSON snapshot per line, "
    "oldest first (use this to model the opponent's pattern, not just the "
    "latest offer; empty if this is round 1):\n"
)


def build_user_prompt(game: dict, history: list[dict]) -> str:
    """Build the full user-turn prompt for this round.

    Prompt caching (cache_control) was tried and reverted here after two
    live runs: attempt 1 (game["prompt"] + history as one cached block)
    got cache_creation_input_tokens growing every round but
    cache_read_input_tokens=0 across 100/100 calls -- a full fresh cache
    write every round, never reused (a live run's cost analysis showed
    this made the run ~5% MORE expensive than no caching at all, not
    less). Attempt 2 isolated the cached block to just the accumulated
    history (moving game["prompt"] out, per the hypothesis that it was
    round-varying and poisoning the prefix match) -- that run showed
    cache_creation_input_tokens=0 too: the history-only block never grew
    past Anthropic's ~1024-token minimum-to-cache size within a 20-round
    game, so caching never activated at all, write or read. Two different
    failure modes, zero net benefit either time -- reverted rather than
    spending more live-API budget on a third design tonight.

    Args:
        game: The current game payload from the GLEE SDK.
        history: This game's snapshot history INCLUDING the current round,
            as returned by _record_state (current round is the last entry).

    Returns:
        The full prompt text for the user turn.
    """
    history_block = _format_history_block(history[:-1])
    history_note = f"\n{_HISTORY_HEADER}{history_block}" if history_block else ""

    example_action = _build_example_action(game["valid_actions"])
    example_note = ""
    if example_action:
        example_note = (
            "\nExample of a correctly-formatted action for this exact schema: "
            f"{json.dumps(example_action, sort_keys=True)}. Your action must look "
            'like this shape -- values only, no "type" or "fields" wrapper keys.\n'
        )
    return (
        f"{game['prompt']}\n"
        f"{history_note}\n"
        f"\nFull game state visible to you:\n"
        f"{json.dumps(game['game_state'], indent=2, sort_keys=True)}\n"
        f"Your action must conform to this schema:\n"
        f"{json.dumps(game['valid_actions'], indent=2, sort_keys=True)}\n"
        f"{example_note}\n"
        'Respond with "REASONING:" then your reasoning, then "ACTION:" then the JSON object only.'
    )


def parse_response(text: str) -> tuple[str, dict]:
    """Split a REASONING:/ACTION: reply into (reasoning, action_dict)."""
    action_split = re.split(r"ACTION:\s*", text, maxsplit=1)
    reasoning = re.sub(r"^REASONING:\s*", "", action_split[0].strip()) if action_split else ""
    action_text = action_split[1] if len(action_split) > 1 else text
    action_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", action_text.strip())
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", action_text, re.DOTALL)
        if not match:
            raise
        action = json.loads(match.group(0))

    # Defense in depth: Claude sometimes echoes the *shape* of valid_actions
    # ({"type": ..., "fields": {...}}) instead of a flat instance of just
    # the field values -- confirmed root cause of 40 server-side rejections
    # / 12 duplicate-round groups / at least 1 confirmed game loss in a real
    # run. build_user_prompt()'s worked example is the primary fix; this
    # unwrap is the safety net for whenever that alone isn't enough. None of
    # the real GLEE schemas observed so far have a legitimate field literally
    # named "fields", so this check is safe against the actions we know of.
    if isinstance(action, dict) and isinstance(action.get("fields"), dict):
        logger.warning(
            "parse_response: unwrapping schema-echoed action %r -> %r",
            action,
            action["fields"],
        )
        action = action["fields"]

    return reasoning, action


def safe_action(game: dict) -> dict:
    """Guaranteed-legal conservative fallback -- same role as in the SDK's own
    llm_agent.py example: cheap insurance against a bad API call or bad JSON."""
    family = game["game_family"]
    action_type = game["valid_actions"]["type"]
    state = game["game_state"]

    if action_type == "offer":
        if family == "bargaining":
            half = state["money_to_divide"] / 2
            return {"alice_gain": half, "bob_gain": half}
        me = state["current_player"]
        return {"product_price": state[f"{me}_value"]}
    if action_type == "seller_message":
        return {"message": "This product is a fair deal at the listed price."}
    if action_type == "seller_recommendation":
        return {"decision": "yes"}
    if family == "bargaining":
        return {"decision": "accept"}
    if family == "negotiation":
        return {"decision": "AcceptOffer"}
    return {"decision": "yes"}  # persuasion buyer_decision


# ---------------------------------------------------------------------------
# Trajectory logging (feeds PIA scoring later -- see integration plan)
# ---------------------------------------------------------------------------


def log_trajectory_step(
    game: dict,
    reasoning: str,
    action: dict,
    fallback_used: bool,
    usage: dict[str, int] | None = None,
) -> None:
    """Append one trajectory record. ``usage`` is None for fallback steps
    (no successful Claude call happened, so there's no token count to log)
    and is otherwise ask_claude()'s per-call usage dict -- real
    input/output tokens plus cache_creation/cache_read tokens, which is
    what lets you verify from the log alone whether prompt caching (see
    build_user_prompt) is actually landing hits, rather than assuming it.
    """
    usage = usage or {}
    record = {
        "log_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "game_id": game["game_id"],
        "game_family": game["game_family"],
        "your_player": game.get("your_player"),
        "phase": game.get("phase"),
        "round": game["game_state"].get("round"),
        "game_state": game["game_state"],
        "valid_actions": game["valid_actions"],
        "reasoning": reasoning,
        "action": action,
        "fallback_used": fallback_used,
        "model": MODEL,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
    }
    with TRAJECTORY_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------

# Defense-in-depth against the poll-loop race that produced duplicate
# submissions for the same turn (see the poll_interval comment in
# __main__): _inflight_calls' keys are the set of game_ids currently inside
# ask_claude(), added on entry and removed on exit/exception. The Future
# values let a second call for the same game_id -- dispatched by the SDK
# before the first one has registered its move server-side -- wait for and
# reuse that call's result instead of burning a second real API call.
_inflight_lock = threading.Lock()
_inflight_calls: dict[str, "Future[tuple[str, dict, dict[str, int]]]"] = {}

# Circuit breaker: stops the process after too many consecutive Claude
# failures in a row (e.g. exhausted credits), rather than spinning for
# hours logging "credit balance too low" and burning fallback moves with
# nobody watching -- exactly what happened in the unattended overnight run
# that motivated this. Shared across concurrency workers (hence the lock):
# a real outage fails every worker's calls roughly together, so the count
# climbs fast regardless of which thread happens to observe each failure.
_failure_lock = threading.Lock()
consecutive_failures = 0
CIRCUIT_BREAKER_THRESHOLD = 20


def _note_api_success() -> None:
    """Reset the circuit breaker after any successful Claude API response,
    even if the reply then fails to parse -- a parse failure is a Claude
    output-quality problem, not evidence of an outage/exhausted credits."""
    global consecutive_failures
    with _failure_lock:
        consecutive_failures = 0


def _note_api_failure() -> int:
    """Increment the circuit breaker counter and return the new total."""
    global consecutive_failures
    with _failure_lock:
        consecutive_failures += 1
        return consecutive_failures


def ask_claude(game: dict, history: list[dict]) -> tuple[str, dict, dict[str, int]]:
    game_id = game["game_id"]
    future: Future[tuple[str, dict, dict[str, int]]] = Future()
    with _inflight_lock:
        _inflight_calls[game_id] = future
    try:
        # No cache_control -- see build_user_prompt's docstring for why
        # (reverted after two live runs both produced zero net benefit).
        messages: list[dict] = [
            {"role": "user", "content": build_user_prompt(game, history)}
        ]
        last_text = ""
        for attempt in range(2):
            try:
                response = anthropic_client.messages.create(
                    model=MODEL,
                    max_tokens=600,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    timeout=60,
                )
                _note_api_success()
                last_text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                reasoning, action = parse_response(last_text)
                usage: dict[str, int] = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": (
                        response.usage.cache_creation_input_tokens or 0
                    ),
                    "cache_read_input_tokens": (
                        response.usage.cache_read_input_tokens or 0
                    ),
                }
                future.set_result((reasoning, action, usage))
                return reasoning, action, usage
            except Exception as e:  # noqa: BLE001 -- provider error, timeout, or bad JSON
                failures = _note_api_failure()
                if failures > CIRCUIT_BREAKER_THRESHOLD:
                    breaker_error = RuntimeError(
                        f"Circuit breaker tripped after {failures} consecutive failures"
                    )
                    future.set_exception(breaker_error)
                    logger.error(
                        "Circuit breaker: %d consecutive failures, likely "
                        "exhausted credits -- stopping",
                        failures,
                    )
                    raise SystemExit(1) from e
                logger.warning("Claude attempt %d failed: %s", attempt + 1, e)
                messages.append({"role": "assistant", "content": last_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That reply didn't parse ({e}). Reply again with "
                            '"REASONING:" then "ACTION:" then ONLY the corrected JSON object.'
                        ),
                    }
                )
        error = RuntimeError("Claude failed to produce a parseable action after retries")
        future.set_exception(error)
        raise error
    finally:
        with _inflight_lock:
            _inflight_calls.pop(game_id, None)


# ---------------------------------------------------------------------------
# Per-family strategies + dispatcher
# ---------------------------------------------------------------------------


def _play(game: dict) -> dict:
    history = _record_state(game)
    game_id = game["game_id"]

    with _inflight_lock:
        duplicate_future = _inflight_calls.get(game_id)

    try:
        if duplicate_future is not None:
            logger.warning(
                "Game %s dispatched again while a Claude call for it is "
                "already in flight (poll-loop race) -- waiting on that "
                "call instead of making a second API request.",
                game_id,
            )
            reasoning, action, usage = duplicate_future.result()
        else:
            reasoning, action, usage = ask_claude(game, history)
        log_trajectory_step(game, reasoning, action, fallback_used=False, usage=usage)
        logger.info(
            "[%s/%s] round=%s -> %s (in=%d out=%d cache_read=%d cache_write=%d)",
            game["game_family"],
            game["valid_actions"]["type"],
            game["game_state"].get("round"),
            action,
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cache_read_input_tokens"],
            usage["cache_creation_input_tokens"],
        )
        return action
    except SystemExit:
        # Circuit breaker tripped inside ask_claude -- must NOT be treated
        # as an ordinary per-move failure (that's the whole point: stop the
        # process, don't fall back and keep spinning). Re-raise past the
        # broad `except Exception` below, which doesn't catch this anyway
        # since SystemExit isn't an Exception subclass, but being explicit
        # here documents the intent rather than relying on that subtlety.
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("Falling back to safe_action for game %s: %s", game_id, e)
        action = safe_action(game)
        log_trajectory_step(
            game,
            reasoning="[fallback: LLM call/parse failed]",
            action=action,
            fallback_used=True,
            usage=None,
        )
        return action


def bargaining_strategy(game: dict) -> dict:
    return _play(game)


def negotiation_strategy(game: dict) -> dict:
    return _play(game)


def persuasion_strategy(game: dict) -> dict:
    return _play(game)


STRATEGIES = {
    "bargaining": bargaining_strategy,
    "negotiation": negotiation_strategy,
    "persuasion": persuasion_strategy,
}


def strategy(game: dict) -> dict:
    return STRATEGIES[game["game_family"]](game)


if __name__ == "__main__":
    api_key = os.environ.get("GLEE_API_KEY", "")
    if not api_key:
        print("Set GLEE_API_KEY environment variable (get it from your glee-competition.com dashboard)")
        raise SystemExit(1)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY environment variable")
        raise SystemExit(1)

    base_url = os.environ.get("GLEE_API_URL")
    client = GleeClient(api_key=api_key, base_url=base_url) if base_url else GleeClient(api_key=api_key)

    print(f"Agent stats: {client.stats()}")
    print(f"Logging trajectories to: {TRAJECTORY_LOG.resolve()}")

    # poll_interval=20: default is 2.0s, but observed Claude call latency in
    # real runs was 5-11s, so the SDK's poll loop could see a game as still
    # "pending" and redispatch it before the prior move registered
    # server-side -- 5 duplicate submissions on two negotiation games and 3
    # on a bargaining game in one run. 20s gives comfortable margin above
    # observed latency without meaningfully slowing throughput here.
    #
    # No max_games: unbounded run for real data collection. requeue stays
    # True (default), so this runs until stopped (Ctrl+C / process kill),
    # the competition closes, or an unhandled error. Prompt caching was
    # tried and reverted (see build_user_prompt) -- every call is at the
    # plain uncached rate now.
    client.run(strategy, concurrency=6, poll_interval=20)
