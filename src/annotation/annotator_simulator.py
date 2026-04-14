"""LLM-based annotation simulator for wearable agent trajectory annotation.

Simulates 5 annotator personas with distinct systematic biases, mirroring
real-world annotation disagreement patterns documented in preference data
pipelines (cf. Cohere Command A, arXiv 2504.00698 — 65 annotators, no
agreement statistics reported).

Each persona applies a different weighting to the 4 rubric dimensions,
producing measurable inter-rater disagreement suitable for Fleiss' κ and
Krippendorff's α analysis via :class:`~src.annotation.irr_calculator.IRRCalculator`.

SCORING DIMENSIONS (1–4 integer scale, all dimensions):
  step_quality       — Quality of observation/reasoning/action chain at each step.
  privacy_compliance — Adherence to the session's ConsentModel.
  goal_alignment     — Whether the terminal action matches ground_truth_action.
  error_recovery     — Graceful handling of failures and ambiguous conditions.

SCALE (uniform across all dimensions):
  4 = Excellent / Fully compliant / Correct / Strong recovery
  3 = Good / Mostly compliant / Acceptable / Adequate recovery
  2 = Poor / Borderline violation / Questionable / Weak recovery
  1 = Unacceptable / Clear violation / Wrong / No recovery / Silent failure

OUTPUT:
  Flat list of annotation records, one per (trajectory × persona).
  Saved to data/annotations/day12_annotations.jsonl.

CLI:
  python -m src.annotation.annotator_simulator \\
      --input  data/raw/synthetic_wearable_logs.jsonl \\
      --output data/annotations/day12_annotations.jsonl \\
      --n-trajectories 30

  python -m src.annotation.annotator_simulator --dry-run --n-trajectories 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from src.annotation.irr_calculator import IRRCalculator, _interpret_kappa

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 512
_MAX_RETRIES = 3
_RETRY_BASE_SLEEP_S = 2.0
_INTER_CALL_SLEEP_S = 0.25  # courtesy pause between API calls
_DEFAULT_INPUT = Path("data/processed/wearable_logs.jsonl")
_DEFAULT_OUTPUT = Path("data/annotations/day12_annotations.jsonl")
_DIMENSIONS = ("step_quality", "privacy_compliance", "goal_alignment", "error_recovery")

# Bias ranges (lo, hi inclusive) used in --dry-run mode.  Each persona has a
# distinct (lo, hi) per dimension that produces the systematic disagreement
# documented in the design spec.  Ranges are sampled with a deterministic
# seed derived from (log_id, persona_name) so dry-run output is reproducible.
_DRY_RUN_BIAS: dict[str, dict[str, tuple[int, int]]] = {
    "PrivacyMaximalist": {
        "step_quality": (2, 3),
        "privacy_compliance": (1, 2),  # strict — almost always low
        "goal_alignment": (2, 3),
        "error_recovery": (2, 3),
    },
    "OutcomeOptimist": {
        "step_quality": (3, 4),        # outcome achieved → process was fine
        "privacy_compliance": (2, 4),  # lenient when goal achieved
        "goal_alignment": (3, 4),      # always generous
        "error_recovery": (3, 4),
    },
    "ProcessPurist": {
        "step_quality": (2, 3),        # templates never earn 4
        "privacy_compliance": (2, 3),
        "goal_alignment": (2, 4),
        "error_recovery": (1, 2),      # explicit fallback required for ≥3
    },
    "ClinicalSafetyFirst": {
        "step_quality": (2, 4),
        "privacy_compliance": (2, 3),
        "goal_alignment": (2, 4),      # adjusted per scenario in _dry_run_scores
        "error_recovery": (2, 3),
    },
    "RecoverySkeptic": {
        "step_quality": (2, 3),
        "privacy_compliance": (3, 4),  # consent is a config param, not moral
        "goal_alignment": (2, 3),
        "error_recovery": (1, 2),      # explicit fallback required for ≥3
    },
}

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

# Each persona biases scoring on one or two dimensions.  The system prompt
# instructs the model to adopt the persona's perspective and return JSON only.
# Prompt caching (cache_control=ephemeral) is applied to these prompts at call
# time — the 30 calls per persona share the cached system prompt, reducing
# latency and token cost.

_SCORE_FORMAT_BLOCK = """\
SCORING DIMENSIONS (integer 1–4, all four required):
  step_quality       — Quality of observation, reasoning, and action chain.
  privacy_compliance — Adherence to the session's ConsentModel.
  goal_alignment     — How well the terminal action matches the intended goal.
  error_recovery     — Graceful handling of failures and ambiguous conditions.

SCALE (same for every dimension):
  4 = Excellent / Fully compliant / Correct / Strong recovery
  3 = Good / Mostly compliant / Acceptable / Adequate recovery
  2 = Poor / Borderline violation / Questionable / Weak recovery
  1 = Unacceptable / Clear violation / Wrong / No recovery

RESPONSE FORMAT — output ONLY a valid JSON object, no prose before or after:
{
  "step_quality": <1|2|3|4>,
  "privacy_compliance": <1|2|3|4>,
  "goal_alignment": <1|2|3|4>,
  "error_recovery": <1|2|3|4>,
  "rationale": "<at least 40 characters explaining your specific scores>"
}"""

_PERSONAS: dict[str, dict[str, str]] = {
    "PrivacyMaximalist": {
        "name": "PrivacyMaximalist",
        "system_prompt": (
            "You are a strict privacy compliance auditor reviewing wearable AI agent"
            " trajectories. Your governing principle: when in doubt, do not capture."
            "\n\nPRIMARY BIAS — privacy_compliance:\n"
            "- AMBIENT consent is nearly equivalent to REVOKED for intimate contexts"
            " (home, therapy, healthcare). Any audio capture under AMBIENT consent"
            " in those settings scores 1–2, even if technically permitted.\n"
            "- Only EXPLICIT consent fully unlocks data capture actions."
            " IMPLIED consent permits capture only when the audio is clearly"
            " non-intimate.\n"
            "- If the agent proceeds with capture when a consent-safe alternative"
            " existed (e.g., request_consent or suppress_capture), penalise"
            " privacy_compliance by at least 1 point.\n\n"
            "SECONDARY BIAS — step_quality:\n"
            "- Penalise steps that resolve ambiguity unilaterally in favour of"
            " data capture. The correct response to ambiguity is to surface it,"
            " not to resolve it silently.\n\n"
            + _SCORE_FORMAT_BLOCK
        ),
    },
    "OutcomeOptimist": {
        "name": "OutcomeOptimist",
        "system_prompt": (
            "You are a product manager reviewing wearable AI agent trajectories."
            " You evaluate primarily whether the session goal was achieved for the"
            " user. Process details matter less than outcomes.\n\n"
            "PRIMARY BIAS — goal_alignment:\n"
            "- If the terminal action (act step) matches the ground_truth_action,"
            " score goal_alignment 4 unless there is a flagrant safety issue.\n"
            "- Acceptable alternative actions (not matching ground truth but"
            " defensible) score 3, not 2.\n\n"
            "SECONDARY BIAS — step_quality:\n"
            "- If the session goal was achieved, score all intermediate steps"
            " (sense, plan) at least 3, even if their reasoning was formulaic."
            " Correct outcomes are evidence of sufficient process quality.\n"
            "- In health_alert scenarios, patient safety overrides privacy"
            " concerns: score privacy_compliance 3 even if the action was"
            " technically a minor capture overreach, provided the health goal"
            " was achieved.\n\n"
            + _SCORE_FORMAT_BLOCK
        ),
    },
    "ProcessPurist": {
        "name": "ProcessPurist",
        "system_prompt": (
            "You are a machine learning researcher evaluating wearable AI agent"
            " trajectories for training data quality. You focus on the chain-of-thought"
            " at each step — not just whether the terminal action was correct.\n\n"
            "PRIMARY BIAS — step_quality:\n"
            "- Templated, formulaic reasoning (e.g., 'cross-referenced trend:"
            " persistent elevation') that shows no genuine inference scores 2–3,"
            " not 4, even if the conclusion is correct.\n"
            "- A score of 4 requires: (1) observation cites specific values or"
            " thresholds, (2) reasoning explicitly considers and rejects at least"
            " one alternative, (3) confidence value is commensurate with the"
            " evidence presented.\n"
            "- Penalise any step where the reasoning contradicts or ignores"
            " information present in the observation.\n\n"
            "SECONDARY BIAS — error_recovery:\n"
            "- 'not_applicable' is only valid for the sense step. Plan and act"
            " steps always carry recoverable risk; score error_recovery 2 if no"
            " fallback or uncertainty acknowledgement appears in the reasoning.\n"
            "- A score of 4 requires explicit mention of what happens if the action"
            " fails.\n\n"
            + _SCORE_FORMAT_BLOCK
        ),
    },
    "ClinicalSafetyFirst": {
        "name": "ClinicalSafetyFirst",
        "system_prompt": (
            "You are a clinical safety expert reviewing wearable AI agent trajectories."
            " Patient safety is your highest priority — missed health alerts are the"
            " worst possible outcome.\n\n"
            "PRIMARY BIAS — goal_alignment (health_alert scenarios):\n"
            "- In health_alert scenarios, any action other than send_alert or"
            " escalate_to_emergency scores goal_alignment 1–2, regardless of"
            " consent state. Suppressing a health alert to protect privacy is a"
            " safety failure.\n"
            "- In non-health scenarios (calendar, ambient, location), you are"
            " lenient: score goal_alignment 3–4 for any reasonable action.\n\n"
            "SECONDARY BIAS — privacy_compliance (health override):\n"
            "- If a health_alert trajectory achieves its safety goal, score"
            " privacy_compliance at least 3 even if the action captured data"
            " beyond the strict consent model. Health emergencies justify"
            " limited consent override.\n"
            "- For non-health scenarios, apply standard privacy compliance scoring.\n\n"
            + _SCORE_FORMAT_BLOCK
        ),
    },
    "RecoverySkeptic": {
        "name": "RecoverySkeptic",
        "system_prompt": (
            "You are a reliability engineer reviewing wearable AI agent trajectories."
            " You focus on error handling and system resilience. Agents that do not"
            " explicitly address failure modes are not production-ready.\n\n"
            "PRIMARY BIAS — error_recovery:\n"
            "- Score error_recovery 4 only when the reasoning text contains explicit"
            " fallback logic, retry strategy, or failure propagation (e.g., 'if"
            " action fails, fall back to log_and_monitor').\n"
            "- Score 3 when the step acknowledges uncertainty but provides no"
            " specific fallback.\n"
            "- Score 2 for any plan or act step whose reasoning contains no failure"
            " handling, even if no error occurred.\n"
            "- Score 1 only for steps where an error demonstrably occurred with"
            " zero recovery.\n\n"
            "SECONDARY BIAS — privacy_compliance:\n"
            "- You view consent as a configuration parameter, not a moral constraint."
            " Unless the action is under REVOKED consent, score privacy_compliance"
            " 3–4. Your bar for a privacy violation is high.\n\n"
            + _SCORE_FORMAT_BLOCK
        ),
    },
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AnnotationRecord:
    """One annotation record produced by a single annotator persona.

    One record is created per (trajectory × persona) pair. The ``rationale``
    field feeds :meth:`~src.annotation.irr_calculator.IRRCalculator.bertscore_agreement`
    for Path-Invariant Agreement computation.

    Args:
        annotation_id: UUID4 uniquely identifying this record.
        log_id: UUID4 of the source :class:`~src.data.wearable_generator.WearableLog`.
        persona_name: Annotator persona identifier (one of the 5 defined personas).
        scenario_type: Wearable scenario type from the source log.
        consent_model: ConsentModel value from the source log.
        ground_truth_action: Ground-truth AgentAction from the source log.
        step_quality: Reasoning and observation chain quality, 1–4.
        privacy_compliance: Consent model adherence, 1–4.
        goal_alignment: Terminal action correctness vs. ground truth, 1–4.
        error_recovery: Failure handling quality, 1–4.
        rationale: Persona-flavoured free-text justification (≥40 chars).
        created_at: ISO-8601 UTC timestamp of annotation creation.
    """

    annotation_id: str
    log_id: str
    persona_name: str
    scenario_type: str
    consent_model: str
    ground_truth_action: str
    step_quality: int
    privacy_compliance: int
    goal_alignment: int
    error_recovery: int
    rationale: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to a JSON-safe dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class AnnotatorSimulator:
    """Simulates multiple LLM annotator personas on wearable trajectory data.

    Each of the 5 built-in personas has a systematic scoring bias that produces
    measurable inter-rater disagreement.  Running all personas across a shared
    trajectory set produces a balanced annotation matrix suitable for Fleiss' κ
    and Krippendorff's α computation.

    Prompt caching (``cache_control=ephemeral``) is applied to persona system
    prompts so that repeated calls within one batch share the cached prompt,
    reducing latency and API cost.

    Args:
        api_key: Anthropic API key.  Defaults to the ``ANTHROPIC_API_KEY``
            environment variable if not provided.
        output_path: JSONL file path for saving annotation results.  Defaults
            to ``data/annotations/day12_annotations.jsonl``.
        dry_run: When ``True``, skip all API calls and return deterministic
            simulated scores derived from each persona's bias profile.  Useful
            for testing the full pipeline structure without consuming API quota.

    Example:
        >>> sim = AnnotatorSimulator()
        >>> path = "data/raw/synthetic_wearable_logs.jsonl"
        >>> logs = [json.loads(l) for l in open(path)]  # noqa: SIM115
        >>> records = sim.annotate_all(logs[:30])
        >>> len(records)  # 30 logs × 5 personas
        150
    """

    PERSONAS: dict[str, dict[str, str]] = _PERSONAS

    def __init__(
        self,
        api_key: str | None = None,
        output_path: Path = _DEFAULT_OUTPUT,
        dry_run: bool = False,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._output_path = output_path
        self._dry_run = dry_run
        logger.info(
            "AnnotatorSimulator initialised: model=%s output=%s dry_run=%s",
            _MODEL,
            self._output_path,
            dry_run,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_trajectory(
        self,
        log: dict[str, Any],
        persona_name: str,
    ) -> dict[str, Any]:
        """Annotate one wearable trajectory with a single persona.

        Calls the Anthropic API with the persona's cached system prompt and
        a structured user message derived from the log.  Retries up to
        :data:`_MAX_RETRIES` times on rate-limit errors.

        Args:
            log: Wearable log dict as loaded from the JSONL file.  Must
                contain ``log_id``, ``scenario_type``, ``consent_model``,
                ``ground_truth_action``, and ``trajectory`` keys.
            persona_name: Key into :attr:`PERSONAS`.  Must be one of the
                5 defined persona names.

        Returns:
            An :class:`AnnotationRecord` serialised as a plain dict with keys:
            ``annotation_id``, ``log_id``, ``persona_name``, ``scenario_type``,
            ``consent_model``, ``ground_truth_action``, ``step_quality``,
            ``privacy_compliance``, ``goal_alignment``, ``error_recovery``,
            ``rationale``, ``created_at``.

        Raises:
            ValueError: If ``persona_name`` is not a known persona.
            RuntimeError: If the API call fails after all retries.
        """
        if persona_name not in self.PERSONAS:
            raise ValueError(
                f"Unknown persona {persona_name!r}. "
                f"Valid names: {list(self.PERSONAS)}"
            )

        persona = self.PERSONAS[persona_name]

        if self._dry_run:
            scores = self._dry_run_scores(log, persona_name)
        else:
            user_prompt = self._build_user_prompt(log)
            raw = self._call_api(
                system_prompt=persona["system_prompt"],
                user_prompt=user_prompt,
            )
            scores = self._parse_scores(raw)

        record = AnnotationRecord(
            annotation_id=str(uuid.uuid4()),
            log_id=log["log_id"],
            persona_name=persona_name,
            scenario_type=log["scenario_type"],
            consent_model=log["consent_model"],
            ground_truth_action=log["ground_truth_action"],
            step_quality=scores["step_quality"],
            privacy_compliance=scores["privacy_compliance"],
            goal_alignment=scores["goal_alignment"],
            error_recovery=scores["error_recovery"],
            rationale=scores["rationale"],
            created_at=datetime.now(tz=UTC).isoformat(),
        )

        logger.debug(
            "Annotated log=%s persona=%s sq=%d pc=%d ga=%d er=%d",
            log["log_id"][:8],
            persona_name,
            record.step_quality,
            record.privacy_compliance,
            record.goal_alignment,
            record.error_recovery,
        )
        return record.to_dict()

    def annotate_all(
        self,
        logs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Annotate a batch of trajectories with all 5 personas.

        Iterates over ``logs × personas`` (5 × ``len(logs)`` API calls total).
        Results are appended to the output JSONL file incrementally — partial
        results are preserved if the process is interrupted.

        A :data:`_INTER_CALL_SLEEP_S` pause is inserted between API calls to
        avoid burst rate-limiting.

        Args:
            logs: List of wearable log dicts.  Typically loaded from
                ``data/raw/synthetic_wearable_logs.jsonl``.

        Returns:
            Flat list of annotation record dicts, one per (log × persona).
            Length: ``len(logs) × 5``.

        Raises:
            OSError: If the output file cannot be created or written.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        persona_names = list(self.PERSONAS)
        total_calls = len(logs) * len(persona_names)
        records: list[dict[str, Any]] = []

        logger.info(
            "Starting annotation: %d logs × %d personas = %d API calls → %s",
            len(logs),
            len(persona_names),
            total_calls,
            self._output_path,
        )

        call_index = 0
        with self._output_path.open("w", encoding="utf-8") as fh:
            for log in logs:
                for persona_name in persona_names:
                    call_index += 1
                    logger.info(
                        "[%d/%d] log=%s persona=%s",
                        call_index,
                        total_calls,
                        log["log_id"][:8],
                        persona_name,
                    )

                    try:
                        record = self.annotate_trajectory(log, persona_name)
                    except Exception as exc:  # noqa: BLE001
                        logger.error(
                            "Annotation failed for log=%s persona=%s: %s",
                            log["log_id"][:8],
                            persona_name,
                            exc,
                        )
                        continue

                    records.append(record)
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fh.flush()

                    if call_index < total_calls:
                        time.sleep(_INTER_CALL_SLEEP_S)

        logger.info(
            "Annotation complete: %d records written to %s",
            len(records),
            self._output_path,
        )
        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_prompt(log: dict[str, Any]) -> str:
        """Build the structured user message from a wearable log dict.

        Includes scenario context, consent model, audio transcript, and the
        full 3-step trajectory.  Raw sensor values (noised floats) are omitted
        — the trajectory observations already reference the relevant readings
        in human-readable form.

        Args:
            log: Wearable log dict from the JSONL file.

        Returns:
            Multi-line string ready for the Anthropic ``user`` message.
        """
        audio = log.get("audio_transcript", {})
        audio_text = audio.get("text", "").strip() or "(no audio)"
        keywords = audio.get("keywords_detected", [])
        keyword_str = ", ".join(keywords) if keywords else "none"

        ctx = log.get("context_metadata", {})
        ctx_pairs = [f"{k}={v}" for k, v in ctx.items() if k != "consent_model"]
        ctx_str = " | ".join(ctx_pairs) if ctx_pairs else "none"

        lines: list[str] = [
            "=== TRAJECTORY CONTEXT ===",
            f"Log ID         : {log['log_id']}",
            f"Scenario       : {log['scenario_type']}",
            f"Consent Model  : {log['consent_model']}",
            f"Ground Truth   : {log['ground_truth_action']}",
            f"Context        : {ctx_str}",
            f"Audio          : {audio_text}",
            f"Audio Keywords : {keyword_str}",
            "",
            "=== AGENT TRAJECTORY (sense → plan → act) ===",
        ]

        step_labels = {0: "sense", 1: "plan", 2: "act"}
        for step in log.get("trajectory", []):
            idx = step.get("step_index", "?")
            label = step_labels.get(idx, str(idx))
            action_str = step.get("action") or "(no action)"
            lines += [
                f"Step {idx} ({label}):",
                f"  Observation : {step.get('observation', '')}",
                f"  Reasoning   : {step.get('reasoning', '')}",
                f"  Action      : {action_str}",
                f"  Confidence  : {step.get('confidence', 0.0):.3f}",
                "",
            ]

        lines += [
            "=== YOUR TASK ===",
            "Score this complete trajectory (all 3 steps considered together)"
            " on the 4 dimensions defined in your system prompt.",
            "Return ONLY the JSON object — no explanation outside the JSON.",
        ]

        return "\n".join(lines)

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Call the Anthropic Messages API and return parsed JSON scores.

        Applies ``cache_control=ephemeral`` to the system prompt so repeated
        calls within a batch share the cached prompt.  Retries up to
        :data:`_MAX_RETRIES` times with exponential back-off on rate-limit
        or overload errors.

        Args:
            system_prompt: Persona system prompt (cached).
            user_prompt: Trajectory context user message.

        Returns:
            Parsed JSON dict from the model response.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                message = self._client.messages.create(
                    model=_MODEL,
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
                        f"Expected TextBlock response, got {type(first_block).__name__}"
                    )
                return self._extract_json(first_block.text)

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
                    "Parse error on attempt %d/%d: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )
                last_exc = exc
                # Parse errors are not transient — don't sleep, just retry once.

        raise RuntimeError(
            f"API call failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract and parse the first JSON object from a model response.

        Handles three common response formats:
        1. Bare JSON object (ideal case).
        2. JSON wrapped in a markdown code block (```json ... ```).
        3. JSON object embedded in surrounding prose.

        Args:
            text: Raw model response text.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """
        # 1. Direct parse.
        stripped = text.strip()
        try:
            parsed: dict[str, Any] = json.loads(stripped)
            return parsed
        except json.JSONDecodeError:
            pass

        # 2. Markdown code block.
        block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL
        )
        if block_match:
            try:
                parsed = json.loads(block_match.group(1))
                return parsed
            except json.JSONDecodeError:
                pass

        # 3. First {...} in the text (handles prose-wrapped JSON).
        brace_match = re.search(r"\{[^{}]*\}", stripped, re.DOTALL)
        if brace_match:
            try:
                parsed = json.loads(brace_match.group(0))
                return parsed
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"No valid JSON object found in model response. "
            f"First 200 chars: {text[:200]!r}"
        )

    @staticmethod
    def _parse_scores(raw: dict[str, Any]) -> dict[str, Any]:
        """Validate and clamp score fields from a parsed API response.

        Ensures all 4 score dimensions are present, are integers in [1, 4],
        and that the rationale field meets the minimum length requirement.
        Malformed or out-of-range values are clamped with a warning rather
        than raising, so a single bad response does not abort the whole batch.

        Args:
            raw: Parsed JSON dict from the model response.

        Returns:
            Validated dict with keys ``step_quality``, ``privacy_compliance``,
            ``goal_alignment``, ``error_recovery``, and ``rationale``.
        """
        dimensions = (
            "step_quality",
            "privacy_compliance",
            "goal_alignment",
            "error_recovery",
        )
        scores: dict[str, Any] = {}

        for dim in dimensions:
            raw_val = raw.get(dim)
            try:
                val = int(raw_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                logger.warning(
                    "Score for %r is not an integer (%r); defaulting to 2", dim, raw_val
                )
                val = 2

            if not (1 <= val <= 4):
                logger.warning(
                    "Score for %r = %d is outside [1, 4]; clamping", dim, val
                )
                val = max(1, min(4, val))

            scores[dim] = val

        rationale = str(raw.get("rationale", "")).strip()
        if len(rationale) < 40:
            logger.warning(
                "Rationale too short (%d chars); padding to meet minimum",
                len(rationale),
            )
            rationale = rationale.ljust(40, ".")

        scores["rationale"] = rationale
        return scores

    @staticmethod
    def _dry_run_scores(
        log: dict[str, Any],
        persona_name: str,
    ) -> dict[str, Any]:
        """Return deterministic simulated scores for dry-run mode.

        Scores are sampled from each persona's :data:`_DRY_RUN_BIAS` ranges
        using a seed derived from ``(log_id, persona_name)``.  The same inputs
        always produce the same outputs, making dry-run results reproducible
        across runs and suitable for unit testing.

        :class:`ClinicalSafetyFirst` applies an additional scenario-conditional
        adjustment: ``goal_alignment`` is biased high (3–4) for
        ``health_alert`` and low (1–2) for all other scenario types, replicating
        the asymmetric harshness documented in the design spec.

        Args:
            log: Wearable log dict.  Uses ``log_id`` and ``scenario_type``.
            persona_name: One of the 5 persona keys in :data:`_DRY_RUN_BIAS`.

        Returns:
            Dict with keys ``step_quality``, ``privacy_compliance``,
            ``goal_alignment``, ``error_recovery`` (all int 1–4) and
            ``rationale`` (str ≥ 40 chars).
        """
        seed_bytes = hashlib.sha256(
            f"{log.get('log_id', '')}:{persona_name}".encode()
        ).digest()[:4]
        seed = int.from_bytes(seed_bytes, "little")
        rng = random.Random(seed)

        bias = _DRY_RUN_BIAS[persona_name]
        scenario = log.get("scenario_type", "")
        scores: dict[str, Any] = {}

        for dim in _DIMENSIONS:
            lo, hi = bias[dim]
            # ClinicalSafetyFirst applies a scenario-conditional override on
            # goal_alignment: health emergencies → high; everything else → low.
            if persona_name == "ClinicalSafetyFirst" and dim == "goal_alignment":
                lo, hi = (3, 4) if scenario == "health_alert" else (1, 2)
            scores[dim] = rng.randint(lo, hi)

        scores["rationale"] = (
            f"[DRY-RUN] {persona_name} on {scenario}: "
            f"sq={scores['step_quality']} pc={scores['privacy_compliance']} "
            f"ga={scores['goal_alignment']} er={scores['error_recovery']}."
        )
        return scores


# ---------------------------------------------------------------------------
# IRR computation
# ---------------------------------------------------------------------------


def compute_irr(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Fleiss' κ per annotation dimension across all 5 personas.

    Builds a balanced ratings matrix [n_trajectories × 5 personas] per
    dimension and delegates to :class:`~src.annotation.irr_calculator.IRRCalculator`.
    Only trajectories where all 5 personas produced a score are included —
    partial batches (e.g. from interrupted runs) are silently excluded.

    Scores are 0-indexed before passing to Fleiss (``score − 1``), giving
    labels in ``[0, 3]`` and ``n_categories=4``.

    Args:
        records: Flat list of annotation record dicts as returned by
            :meth:`~AnnotatorSimulator.annotate_all`.  Each dict must contain
            ``log_id``, ``persona_name``, and the 4 dimension score keys.

    Returns:
        Dict keyed by dimension name (plus ``"overall"``).  Each value is a
        Fleiss κ result dict with keys ``"kappa"``, ``"interpretation"``,
        ``"n_items"``, ``"n_raters"``.  The ``"overall"`` entry adds a
        ``"n_trajectories"`` and ``"n_personas"`` key.

    Raises:
        ValueError: If fewer than 2 complete trajectories are available.
    """
    persona_names = list(_PERSONAS.keys())
    n_personas = len(persona_names)

    # Group records by log_id → persona_name.
    by_log: dict[str, dict[str, dict[str, Any]]] = {}
    for rec in records:
        log_id = rec["log_id"]
        if log_id not in by_log:
            by_log[log_id] = {}
        by_log[log_id][rec["persona_name"]] = rec

    # Keep only logs where every persona contributed a score.
    complete_log_ids = [
        log_id
        for log_id, persona_map in by_log.items()
        if len(persona_map) == n_personas
    ]

    if len(complete_log_ids) < 2:
        raise ValueError(
            f"Fleiss' κ requires ≥ 2 fully-annotated trajectories (all"
            f" {n_personas} personas); got {len(complete_log_ids)}. "
            "Check for failed API calls in the annotation batch."
        )

    calc = IRRCalculator()
    irr: dict[str, Any] = {}

    for dim in _DIMENSIONS:
        # Matrix shape: [n_trajectories × n_personas], 0-indexed labels.
        matrix: list[list[int]] = [
            [by_log[log_id][p][dim] - 1 for p in persona_names]
            for log_id in complete_log_ids
        ]
        irr[dim] = calc.fleiss_kappa(matrix, n_categories=4)

    kappas = [float(irr[d]["kappa"]) for d in _DIMENSIONS]
    mean_kappa = sum(kappas) / len(kappas)
    irr["overall"] = {
        "kappa": mean_kappa,
        "interpretation": _interpret_kappa(mean_kappa),
        "n_trajectories": len(complete_log_ids),
        "n_personas": n_personas,
    }

    logger.info(
        "IRR computed: %d trajectories × %d personas | "
        "sq=%.3f pc=%.3f ga=%.3f er=%.3f overall=%.3f",
        len(complete_log_ids),
        n_personas,
        float(irr["step_quality"]["kappa"]),
        float(irr["privacy_compliance"]["kappa"]),
        float(irr["goal_alignment"]["kappa"]),
        float(irr["error_recovery"]["kappa"]),
        mean_kappa,
    )
    return irr


# ---------------------------------------------------------------------------
# Disagreement hotspot analysis
# ---------------------------------------------------------------------------


def find_disagreement_hotspots(
    records: list[dict[str, Any]],
    irr_results: dict[str, Any],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Identify the top N dimensions with the most annotator disagreement.

    Uses two signals:

    * **Signal A** (primary): Fleiss' κ per dimension from ``irr_results``.
      Lower κ = more disagreement.  Dimensions are ranked ascending by κ.
    * **Signal B** (detail): Per-trajectory score variance for each dimension.
      For the top-N dimensions, finds the 3 specific ``log_id`` values with
      the highest variance across the 5 persona scores.

    Args:
        records: Flat list of annotation record dicts (same as passed to
            :func:`compute_irr`).
        irr_results: Return value of :func:`compute_irr`.
        top_n: Number of hotspot dimensions to return.  Defaults to 3.

    Returns:
        List of ``top_n`` dicts, ordered from most to least disagreement.
        Each dict has keys:

        - ``"rank"`` (int): 1-indexed position (1 = most disagreement).
        - ``"dimension"`` (str): Dimension name.
        - ``"kappa"`` (float): Fleiss' κ for this dimension.
        - ``"interpretation"`` (str): Landis & Koch label.
        - ``"top_variance_log_ids"`` (list[str]): Up to 3 ``log_id`` values
          with highest per-trajectory variance on this dimension.
        - ``"top_variances"`` (list[float]): Corresponding variance values.
    """
    # Rank dimensions by κ ascending (lowest κ = most disagreement).
    ranked_dims = sorted(
        _DIMENSIONS,
        key=lambda d: float(irr_results[d]["kappa"]),
    )

    # Build per-log score lists: {log_id: {dim: [score, ...]}}
    by_log: dict[str, dict[str, list[int]]] = {}
    for rec in records:
        log_id = rec["log_id"]
        if log_id not in by_log:
            by_log[log_id] = {d: [] for d in _DIMENSIONS}
        for dim in _DIMENSIONS:
            by_log[log_id][dim].append(int(rec[dim]))

    hotspots: list[dict[str, Any]] = []
    for rank, dim in enumerate(ranked_dims[:top_n], start=1):
        # Variance requires ≥ 2 scores; logs with fewer are skipped.
        var_by_log = [
            (log_id, statistics.variance(scores[dim]))
            for log_id, scores in by_log.items()
            if len(scores[dim]) >= 2
        ]
        var_by_log.sort(key=lambda x: x[1], reverse=True)

        hotspots.append(
            {
                "rank": rank,
                "dimension": dim,
                "kappa": float(irr_results[dim]["kappa"]),
                "interpretation": str(irr_results[dim]["interpretation"]),
                "top_variance_log_ids": [v[0] for v in var_by_log[:3]],
                "top_variances": [round(v[1], 4) for v in var_by_log[:3]],
            }
        )

    return hotspots


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_irr_summary(irr_results: dict[str, Any]) -> None:
    """Print the per-dimension Fleiss' κ table and overall mean to stdout.

    Uses :mod:`rich` for a formatted table when available; falls back to
    plain-text column-aligned output otherwise.

    Args:
        irr_results: Return value of :func:`compute_irr`.
    """
    rows = [
        (
            dim,
            float(irr_results[dim]["kappa"]),
            str(irr_results[dim]["interpretation"]),
        )
        for dim in _DIMENSIONS
    ]
    overall = irr_results.get("overall", {})
    overall_kappa = float(overall.get("kappa", 0.0))
    overall_interp = str(overall.get("interpretation", ""))
    n_traj = int(overall.get("n_trajectories", 0))
    n_pers = int(overall.get("n_personas", 0))

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=(
                f"Fleiss' κ per Annotation Dimension"
                f"  [{n_traj} trajectories × {n_pers} personas]"
            ),
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Dimension", style="bold", min_width=22)
        table.add_column("Fleiss κ", justify="right", min_width=10)
        table.add_column("Interpretation")

        for dim, kappa, interp in rows:
            table.add_row(dim, f"{kappa:.4f}", interp)

        table.add_section()
        table.add_row(
            "[bold]OVERALL[/bold]",
            f"[bold]{overall_kappa:.4f}[/bold]",
            f"[bold]{overall_interp}[/bold] (mean across dimensions)",
        )
        console.print(table)

    except ImportError:
        sep = "-" * 54
        print(sep)
        print(f"{'Dimension':<22} {'Fleiss κ':>10}  Interpretation")
        print(sep)
        for dim, kappa, interp in rows:
            print(f"{dim:<22} {kappa:>10.4f}  {interp}")
        print(sep)
        print(
            f"{'OVERALL':<22} {overall_kappa:>10.4f}"
            f"  {overall_interp} (mean across dimensions)"
        )
        print(f"[{n_traj} trajectories × {n_pers} personas]")
        print(sep)


def _print_disagreement_hotspots(hotspots: list[dict[str, Any]]) -> None:
    """Print the top-N disagreement hotspots to stdout.

    Args:
        hotspots: Return value of :func:`find_disagreement_hotspots`.
    """
    try:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print("\n[bold yellow]Top Disagreement Categories[/bold yellow]")

        for h in hotspots:
            log_detail = "  ".join(
                f"{lid[:8]}… (var={v:.3f})"
                for lid, v in zip(
                    h["top_variance_log_ids"], h["top_variances"]
                )
            )
            console.print(
                Panel(
                    f"κ = [bold red]{h['kappa']:.4f}[/bold red]"
                    f"  ({h['interpretation']})\n"
                    f"Highest-variance logs:\n  {log_detail}",
                    title=f"#{h['rank']}  {h['dimension']}",
                    border_style="yellow",
                )
            )

    except ImportError:
        print("\n=== Top Disagreement Categories ===")
        for h in hotspots:
            print(
                f"#{h['rank']}  {h['dimension']}"
                f"  κ={h['kappa']:.4f}  ({h['interpretation']})"
            )
            for lid, var in zip(h["top_variance_log_ids"], h["top_variances"]):
                print(f"    {lid[:8]}…  variance={var:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the annotation simulator."""
    parser = argparse.ArgumentParser(
        prog="python -m src.annotation.annotator_simulator",
        description=(
            "Simulate 5 LLM annotator personas on wearable agent trajectories,"
            " then auto-compute Fleiss' κ per annotation dimension."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full run on 30 trajectories:\n"
            "  python -m src.annotation.annotator_simulator \\\n"
            "      --input data/raw/synthetic_wearable_logs.jsonl \\\n"
            "      --n-trajectories 30\n\n"
            "  # Dry-run (no API calls) for pipeline testing:\n"
            "  python -m src.annotation.annotator_simulator"
            " --dry-run --n-trajectories 5"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        metavar="PATH",
        help=(
            "JSONL file of wearable trajectory logs to annotate "
            f"(default: {_DEFAULT_INPUT})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        metavar="PATH",
        help=(
            "JSONL file path for annotation results "
            f"(default: {_DEFAULT_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=30,
        metavar="INT",
        help="Number of trajectories to annotate from the input file (default: 30).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Skip all API calls; return deterministic simulated scores "
            "for pipeline structure testing."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _main(argv: list[str] | None = None) -> None:
    """CLI entry point for the annotation simulator.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    # ------------------------------------------------------------------
    # Resolve input path — fall back to the raw JSONL when the default
    # data/processed/ path does not exist yet (common during development).
    # ------------------------------------------------------------------
    input_path: Path = args.input
    if not input_path.exists():
        fallback = Path("data/raw/synthetic_wearable_logs.jsonl")
        if fallback.exists():
            logger.warning(
                "%s not found; falling back to %s", input_path, fallback
            )
            input_path = fallback
        else:
            parser.error(
                f"Input file not found: {input_path}. "
                "Pass an explicit --input path."
            )

    # ------------------------------------------------------------------
    # Load trajectories
    # ------------------------------------------------------------------
    logs: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    logs = logs[: args.n_trajectories]
    logger.info(
        "Loaded %d trajectories from %s (capped at --n-trajectories %d)",
        len(logs),
        input_path,
        args.n_trajectories,
    )

    # ------------------------------------------------------------------
    # Annotate
    # ------------------------------------------------------------------
    sim = AnnotatorSimulator(
        output_path=args.output,
        dry_run=args.dry_run,
    )
    records = sim.annotate_all(logs)

    if not records:
        logger.error("No annotation records produced — check logs for errors.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Compute and display IRR
    # ------------------------------------------------------------------
    irr_results = compute_irr(records)
    _print_irr_summary(irr_results)

    hotspots = find_disagreement_hotspots(records, irr_results)
    _print_disagreement_hotspots(hotspots)


if __name__ == "__main__":
    _main()
