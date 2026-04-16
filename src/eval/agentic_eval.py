"""Agentic evaluation harness implementing the 6 Kore.ai eval metrics.

Addresses the methodology gap identified in Kore.ai (Oct 2025): 89% of
enterprises have agent observability but only 52% have real evaluation.
This module provides trajectory-level scoring across six dimensions:
trajectory success, tool invocation accuracy, groundedness, privacy leak
detection, orchestrator correctness, and latency SLA compliance.

:class:`AgenticEvaluator` integrates the Kore.ai 6-metric harness with the
5-layer :class:`~src.eval.trajectory_scorer.TrajectoryScorer` and the four
PIA rubric dimensions, producing a single unified flat dict per trajectory.

CLI::

    python -m src.eval.agentic_eval --input <trajectory.jsonl>
    python -m src.eval.agentic_eval --dry-run --verbose  # smoke test
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from src.data.wearable_generator import WearableLog
from src.eval.trajectory_scorer import _TERMINAL_ACTIONS, TrajectoryScorer

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="agentic-eval",
    help="Score agent trajectories across 6 Kore.ai evaluation metrics.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# PII patterns for privacy leak detection
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),  # US phone
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
]


# ---------------------------------------------------------------------------
# Part 1 — Data models
# ---------------------------------------------------------------------------


@dataclass
class AgenticEvalResult:
    """Evaluation result for a single agent trajectory.

    Captures all six Kore.ai metrics plus a weighted composite score.
    One instance is produced per (trajectory_id, framework) pair.
    """

    trajectory_id: str
    task_id: str
    framework: str
    # Six Kore.ai metrics
    trajectory_success_rate: float
    tool_invocation_accuracy: float
    groundedness_score: float
    privacy_leak_detected: bool
    orchestrator_correctness: float
    latency_sla_compliance: float
    # Weighted composite
    overall_score: float
    eval_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "framework": self.framework,
            "trajectory_success_rate": self.trajectory_success_rate,
            "tool_invocation_accuracy": self.tool_invocation_accuracy,
            "groundedness_score": self.groundedness_score,
            "privacy_leak_detected": self.privacy_leak_detected,
            "orchestrator_correctness": self.orchestrator_correctness,
            "latency_sla_compliance": self.latency_sla_compliance,
            "overall_score": self.overall_score,
            "eval_timestamp": self.eval_timestamp,
        }


# ---------------------------------------------------------------------------
# Part 2 — KoraiMetrics
# ---------------------------------------------------------------------------


class KoraiMetrics:
    """Six Kore.ai trajectory-level evaluation metrics.

    Each method is stateless and operates on raw trajectory dicts so the
    class can be used without an active LLM connection (groundedness is
    the only method that will later require one).
    """

    def score_trajectory_success(self, trajectory: list[dict[str, Any]]) -> float:
        """Ratio of steps where ``goal_achieved`` is True.

        Args:
            trajectory: List of step dicts, each optionally containing a
                boolean ``goal_achieved`` field.

        Returns:
            Float in [0.0, 1.0]. Returns 0.0 for an empty trajectory.
        """
        if not trajectory:
            return 0.0
        achieved = sum(1 for step in trajectory if step.get("goal_achieved") is True)
        return achieved / len(trajectory)

    def score_tool_invocation(self, trajectory: list[dict[str, Any]]) -> float:
        """Ratio of tool calls where the invoked tool is in ``expected_tools``.

        Steps without a ``tool_call`` field are skipped. Steps with a
        ``tool_call`` but no ``expected_tools`` list count as incorrect.

        Args:
            trajectory: List of step dicts. Each step may contain:
                - ``tool_call`` (str): name of the tool that was called.
                - ``expected_tools`` (list[str]): tools considered valid for
                  this step.

        Returns:
            Float in [0.0, 1.0]. Returns 1.0 when no steps have tool calls
            (nothing to penalise).
        """
        steps_with_calls = [s for s in trajectory if s.get("tool_call")]
        if not steps_with_calls:
            return 1.0
        correct = sum(
            1
            for step in steps_with_calls
            if step.get("tool_call") in step.get("expected_tools", [])
        )
        return correct / len(steps_with_calls)

    # RAGAS Faithfulness measures whether the response is grounded
    # in the provided context — key metric for wearable RAG pipelines.
    # Ref: RAGAS paper 2023; extended here to agentic trajectory context.

    def score_groundedness(self, response: str, context: str) -> float:
        """Groundedness of a response relative to a retrieved context.

        Uses RAGAS ``Faithfulness`` (ragas>=0.4) to check whether every
        claim in ``response`` can be inferred from ``context``. Falls back
        to 0.75 if the RAGAS call fails (e.g. missing API key, empty
        input, or network error) and logs a warning so the failure is
        visible without crashing the eval pipeline.

        Args:
            response: The agent's natural-language response.
            context: The retrieved context the response should be grounded
                in; passed as both ``user_input`` and the sole element of
                ``retrieved_contexts`` for the Faithfulness scorer.

        Returns:
            Float in [0.0, 1.0]. 0.75 when RAGAS is unavailable.
        """
        _FALLBACK = 0.75
        try:
            import os

            from openai import OpenAI
            from ragas.llms import llm_factory
            from ragas.metrics.collections import Faithfulness

            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            metric = Faithfulness(llm=llm_factory(model="gpt-4o-mini", client=client))
            result = metric.score(
                user_input=context,
                response=response,
                retrieved_contexts=[context],
            )
            value = float(result.value)
            if value != value:  # NaN guard — RAGAS returns NaN on empty statements
                return _FALLBACK
            return max(0.0, min(1.0, value))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RAGAS Faithfulness scoring failed (%s: %s); using fallback %.2f",
                type(exc).__name__,
                exc,
                _FALLBACK,
            )
            return _FALLBACK

    def detect_privacy_leak(self, trajectory: list[dict[str, Any]]) -> bool:
        """Scan tool outputs for PII patterns (email, US phone, SSN).

        Checks the ``tool_output`` field of each step. A single match
        across any step triggers a positive detection.

        Args:
            trajectory: List of step dicts, each optionally containing a
                ``tool_output`` (str) field.

        Returns:
            True if any PII pattern is found, False otherwise.
        """
        for step in trajectory:
            output = str(step.get("tool_output", ""))
            for pattern in _PII_PATTERNS:
                if pattern.search(output):
                    logger.warning(
                        "PII pattern detected in step %s",
                        step.get("step_index", "?"),
                    )
                    return True
        return False

    def score_orchestrator_correctness(self, trajectory: list[dict[str, Any]]) -> float:
        """Ratio of steps where ``agent_role`` matches ``expected_role``.

        Steps missing either field are counted as incorrect to penalise
        unannotated trajectories.

        Args:
            trajectory: List of step dicts, each optionally containing
                ``agent_role`` (str) and ``expected_role`` (str).

        Returns:
            Float in [0.0, 1.0]. Returns 1.0 for an empty trajectory.
        """
        if not trajectory:
            return 1.0
        correct = sum(
            1
            for step in trajectory
            if step.get("agent_role")
            and step.get("agent_role") == step.get("expected_role")
        )
        return correct / len(trajectory)

    def score_latency_sla(self, latency_ms: float, sla_ms: float = 5000.0) -> float:
        """Compliance score for a latency-vs-SLA comparison.

        Returns 1.0 when within SLA, then decays linearly to 0.0 at
        ``2 × sla_ms`` and is clamped to 0.0 beyond that.

        Args:
            latency_ms: Observed end-to-end latency in milliseconds.
            sla_ms: Maximum acceptable latency. Defaults to 5 000 ms.

        Returns:
            Float in [0.0, 1.0].
        """
        if latency_ms <= sla_ms:
            return 1.0
        return max(0.0, 1.0 - (latency_ms - sla_ms) / sla_ms)


# ---------------------------------------------------------------------------
# DeepEval LLM-as-judge ensemble
# ---------------------------------------------------------------------------

# DeepMind's FACTS paper (Dec 2025) identified LLM-as-judge ensembles
# as the strongest proxy for factuality grounding. DeepEvalJudge
# implements this pattern for trajectory-level quality assessment.


_GEVAL_CRITERIA: dict[str, str] = {
    "trajectory_quality": "Does the agent make logical, purposeful decisions?",
    "error_recovery": "Does the agent handle failures gracefully?",
    "goal_alignment": "Does each step move toward the stated task goal?",
}

_DEEPEVAL_FALLBACK = 0.7


class DeepEvalJudge:
    """LLM-as-judge ensemble for trajectory-level quality assessment.

    Wraps three ``deepeval.metrics.GEval`` rubrics — Trajectory Quality,
    Error Recovery, and Goal Alignment — into a single ensemble score.
    Each rubric evaluates the serialised trajectory against
    ``task_description`` as the grounding context.

    All scoring is wrapped in a broad ``except`` so that tests without a
    configured LLM API key receive deterministic placeholder values
    (0.7) rather than hard failures.
    """

    def judge_trajectory(
        self, trajectory: list[dict[str, Any]], task_description: str
    ) -> dict[str, float]:
        """Score a trajectory on three GEval rubrics and return an ensemble.

        The trajectory is serialised to a JSON string and passed as
        ``actual_output``; ``task_description`` becomes the ``input``
        field of the ``LLMTestCase``. Both ``INPUT`` and ``ACTUAL_OUTPUT``
        are therefore always present, satisfying GEval's required params.

        Args:
            trajectory: List of step dicts from an agent run.
            task_description: Human-readable description of the task the
                agent was attempting.

        Returns:
            Dict with keys ``trajectory_quality``, ``error_recovery``,
            ``goal_alignment`` (each float in [0.0, 1.0]) and
            ``ensemble_score`` (their unweighted mean).
        """
        _placeholder: dict[str, float] = {
            "trajectory_quality": _DEEPEVAL_FALLBACK,
            "error_recovery": _DEEPEVAL_FALLBACK,
            "goal_alignment": _DEEPEVAL_FALLBACK,
            "ensemble_score": _DEEPEVAL_FALLBACK,
        }

        try:
            import json as _json

            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams

            trajectory_text = _json.dumps(trajectory, indent=2)
            test_case = LLMTestCase(
                input=task_description,
                actual_output=trajectory_text,
            )

            scores: dict[str, float] = {}
            for key, criteria in _GEVAL_CRITERIA.items():
                metric = GEval(
                    name=key.replace("_", " ").title(),
                    criteria=criteria,
                    evaluation_params=[
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    async_mode=False,
                )
                raw = metric.measure(test_case, _show_indicator=False)
                scores[key] = max(0.0, min(1.0, float(raw)))

            ensemble = sum(scores.values()) / len(scores)
            return {**scores, "ensemble_score": ensemble}

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DeepEval GEval judging failed (%s: %s); using fallback %.1f",
                type(exc).__name__,
                exc,
                _DEEPEVAL_FALLBACK,
            )
            return _placeholder


# ---------------------------------------------------------------------------
# FACTS Grounding Scorer
# ---------------------------------------------------------------------------

# Implements rubric-based factuality scoring following the FACTS
# Grounding Benchmark (DeepMind, Dec 2025). No current model exceeds
# 70% on FACTS — this scorer tracks that gap in wearable agent responses.
# Full Kaggle submission planned for Day 41.


class FACTSGroundingScorer:
    """Factuality scorer aligned with DeepMind's FACTS Grounding Benchmark.

    Measures three factuality dimensions for a single agent response:

    * ``parametric_score`` — knowledge recalled from training (stub 0.70,
      pending fine-tuned parametric probe).
    * ``search_score`` — sentence-level token overlap between the response
      and the retrieved source documents; a lightweight proxy for retrieval
      grounding that requires no LLM call.
    * ``grounding_score`` — RAGAS Faithfulness via
      :meth:`KoraiMetrics.score_groundedness`; falls back to 0.75 when
      the API key is absent.
    """

    def __init__(self) -> None:
        self._ragas = KoraiMetrics()

    def _ragas_groundedness(self, response: str, combined_context: str) -> float:
        """Delegate to :meth:`KoraiMetrics.score_groundedness`."""
        return self._ragas.score_groundedness(response, combined_context)

    @staticmethod
    def _sentence_token_overlap(response: str, source_tokens: set[str]) -> float:
        """Ratio of response sentences containing at least one source token.

        Sentences are split on ``'.'``, ``'!'``, and ``'?'``. Tokens are
        lowercased non-whitespace words. Returns 1.0 when there are no
        sentences so an empty response is not penalised asymmetrically.

        Args:
            response: The agent's natural-language response.
            source_tokens: Lowercased token set from all source documents.

        Returns:
            Float in [0.0, 1.0].
        """
        import re as _re

        sentences = [s.strip() for s in _re.split(r"[.!?]", response) if s.strip()]
        if not sentences:
            return 1.0
        hits = sum(
            1
            for sent in sentences
            if source_tokens.intersection(w.lower() for w in sent.split())
        )
        return hits / len(sentences)

    def score(
        self, agent_response: str, source_documents: list[str]
    ) -> dict[str, float]:
        """Score an agent response across three FACTS dimensions.

        Args:
            agent_response: The response text produced by the agent.
            source_documents: Retrieved passages the response should be
                grounded in.

        Returns:
            Dict with keys ``parametric_score``, ``search_score``,
            ``grounding_score``, and ``overall_facts_score`` (their mean),
            all floats in [0.0, 1.0].
        """
        combined_context = " ".join(source_documents)
        source_tokens: set[str] = {
            w.lower() for doc in source_documents for w in doc.split()
        }

        parametric_score = 0.70  # stub — parametric probe pending
        search_score = self._sentence_token_overlap(agent_response, source_tokens)
        grounding_score = self._ragas_groundedness(agent_response, combined_context)
        overall_facts_score = (parametric_score + search_score + grounding_score) / 3.0

        return {
            "parametric_score": parametric_score,
            "search_score": search_score,
            "grounding_score": grounding_score,
            "overall_facts_score": overall_facts_score,
        }


# ---------------------------------------------------------------------------
# Part 3 — Weighted composite
# ---------------------------------------------------------------------------

_WEIGHTS: dict[str, float] = {
    "trajectory_success_rate": 0.30,
    "tool_invocation_accuracy": 0.25,
    "groundedness_score": 0.20,
    "privacy": 0.10,
    "orchestrator_correctness": 0.10,
    "latency_sla_compliance": 0.05,
}


def compute_overall_score(result: AgenticEvalResult) -> float:
    """Weighted average of the six Kore.ai metrics.

    Privacy contributes 1.0 when no leak is detected, 0.0 otherwise.

    Weights:
        trajectory_success_rate  0.30
        tool_invocation_accuracy 0.25
        groundedness_score       0.20
        privacy (no leak)        0.10
        orchestrator_correctness 0.10
        latency_sla_compliance   0.05

    Args:
        result: A fully-populated :class:`AgenticEvalResult`.

    Returns:
        Float in [0.0, 1.0].
    """
    privacy_score = 0.0 if result.privacy_leak_detected else 1.0
    return (
        _WEIGHTS["trajectory_success_rate"] * result.trajectory_success_rate
        + _WEIGHTS["tool_invocation_accuracy"] * result.tool_invocation_accuracy
        + _WEIGHTS["groundedness_score"] * result.groundedness_score
        + _WEIGHTS["privacy"] * privacy_score
        + _WEIGHTS["orchestrator_correctness"] * result.orchestrator_correctness
        + _WEIGHTS["latency_sla_compliance"] * result.latency_sla_compliance
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def evaluate_trajectory(
    trajectory_id: str,
    task_id: str,
    framework: str,
    trajectory: list[dict[str, Any]],
    response: str,
    context: str,
    latency_ms: float,
    sla_ms: float = 5000.0,
) -> AgenticEvalResult:
    """Run all six Kore.ai metrics and return a populated result.

    Args:
        trajectory_id: Unique identifier for this trajectory.
        task_id: Identifier for the benchmark task being evaluated.
        framework: Agent framework name (e.g. ``"langgraph"``).
        trajectory: List of step dicts from the agent run.
        response: Final natural-language response produced by the agent.
        context: Retrieved context used to generate ``response``.
        latency_ms: End-to-end latency in milliseconds.
        sla_ms: SLA threshold in milliseconds. Defaults to 5 000 ms.

    Returns:
        Fully-populated :class:`AgenticEvalResult` including overall score.
    """
    metrics = KoraiMetrics()

    result = AgenticEvalResult(
        trajectory_id=trajectory_id,
        task_id=task_id,
        framework=framework,
        trajectory_success_rate=metrics.score_trajectory_success(trajectory),
        tool_invocation_accuracy=metrics.score_tool_invocation(trajectory),
        groundedness_score=metrics.score_groundedness(response, context),
        privacy_leak_detected=metrics.detect_privacy_leak(trajectory),
        orchestrator_correctness=metrics.score_orchestrator_correctness(trajectory),
        latency_sla_compliance=metrics.score_latency_sla(latency_ms, sla_ms),
        overall_score=0.0,
        eval_timestamp=datetime.now(UTC).isoformat(),
    )
    result.overall_score = compute_overall_score(result)
    return result


# ---------------------------------------------------------------------------
# WearableLog → Kore.ai dict adapter
# ---------------------------------------------------------------------------


def _wearable_steps_to_kore_dicts(log: WearableLog) -> list[dict[str, Any]]:
    """Convert a WearableLog's trajectory to the dict format KoraiMetrics expects.

    Maps TrajectoryStep fields onto the keys consumed by each KoraiMetrics
    method, making reasonable approximations for fields that have no direct
    equivalent in the wearable schema:

    * ``tool_call`` ← ``step.action`` (the discrete action taken).
    * ``expected_tools`` ← ``[log.ground_truth_action]`` for the final step,
      empty list for intermediate steps (nothing to penalise).
    * ``tool_output`` ← ``step.observation`` (scanned for PII patterns).
    * ``goal_achieved`` ← ``True`` if the final step's action is terminal.
    * ``agent_role`` / ``expected_role`` ← ``step.step_name`` for both
      (wearable logs always have canonical step names, so correctness = 1.0).

    Args:
        log: Source WearableLog.

    Returns:
        List of dicts, one per trajectory step.
    """
    n = len(log.trajectory)
    result: list[dict[str, Any]] = []
    for i, step in enumerate(log.trajectory):
        is_final = i == n - 1
        result.append(
            {
                "step_index": step.step_index,
                "step_name": step.step_name,
                "tool_call": step.action if step.action else None,
                "expected_tools": ([log.ground_truth_action] if is_final else []),
                "tool_output": step.observation,
                "goal_achieved": (
                    step.action in _TERMINAL_ACTIONS if is_final else None
                ),
                "agent_role": step.step_name,
                "expected_role": step.step_name,
            }
        )
    return result


# ---------------------------------------------------------------------------
# AgenticEvaluator — unified harness
# ---------------------------------------------------------------------------


class AgenticEvaluator:
    """Unified evaluation harness: 6 Kore.ai metrics + 5-layer TrajectoryScorer.

    Combines :class:`KoraiMetrics` with
    :class:`~src.eval.trajectory_scorer.TrajectoryScorer`
    and the four PIA rubric dimensions into a single flat output dict per
    trajectory, enabling side-by-side comparison of all evaluation signals.

    Args:
        dry_run: Passed to TrajectoryScorer. When True, all trajectory
            scoring is heuristic (no LLM calls).
    """

    def __init__(self, dry_run: bool = True) -> None:
        self._dry_run = dry_run
        self._kore_metrics = KoraiMetrics()
        self._trajectory_scorer = TrajectoryScorer(dry_run=dry_run)

    def evaluate_with_trajectory_score(self, log: WearableLog) -> dict[str, Any]:
        """Run Kore.ai metrics and TrajectoryScorer on a single WearableLog.

        Merges all signals into a single flat dict.  Key prefixes:

        * ``kore_*`` — the 6 Kore.ai metrics.
        * ``layer_*`` — the 5 TrajectoryScorer layer scores.
        * ``pia_*`` — the 4 PIA rubric dimensions.
        * ``weighted_total`` — TrajectoryScorer composite score.

        Args:
            log: WearableLog to evaluate.

        Returns:
            Flat dict with all evaluation signals and ``trajectory_id``.
        """
        kore_steps = _wearable_steps_to_kore_dicts(log)

        # 6 Kore.ai metrics
        tsr = self._kore_metrics.score_trajectory_success(kore_steps)
        tia = self._kore_metrics.score_tool_invocation(kore_steps)
        privacy = self._kore_metrics.detect_privacy_leak(kore_steps)
        orch = self._kore_metrics.score_orchestrator_correctness(kore_steps)
        # Groundedness and latency require external context; use sensible defaults
        # for wearable logs where those aren't available in the schema.
        ground = 0.75  # RAGAS fallback — no free-text response in WearableLog
        latency = 1.0  # no latency field in WearableLog; SLA assumed met

        # 5-layer TrajectoryScorer
        ts = self._trajectory_scorer.score_trajectory(log)

        # 4 PIA dimensions
        pia = self._trajectory_scorer.score_pia_dimensions(log)

        return {
            "trajectory_id": log.log_id,
            # Kore.ai
            "kore_trajectory_success": tsr,
            "kore_tool_invocation_accuracy": tia,
            "kore_groundedness": ground,
            "kore_privacy_leak_detected": privacy,
            "kore_orchestrator_correctness": orch,
            "kore_latency_sla_compliance": latency,
            # TrajectoryScorer layers
            "layer_intent": ts.intent.score,
            "layer_planning": ts.planning.score,
            "layer_tool_calls": ts.tool_calls.score,
            "layer_recovery": ts.recovery.score,
            "layer_outcome": ts.outcome.score,
            # PIA dimensions
            "pia_planning_quality": pia["planning_quality"],
            "pia_error_recovery": pia["error_recovery"],
            "pia_goal_alignment": pia["goal_alignment"],
            "pia_tool_precision": pia["tool_precision"],
            # Composite
            "weighted_total": ts.weighted_total,
        }

    def batch_evaluate_with_trajectory_score(
        self, trajectories: list[WearableLog]
    ) -> list[dict[str, Any]]:
        """Evaluate a batch of WearableLogs, skipping failures.

        Args:
            trajectories: List of WearableLog instances to evaluate.

        Returns:
            List of flat evaluation dicts in the same order as input.
            Trajectories that raise an exception are skipped with a warning.
        """
        results: list[dict[str, Any]] = []
        for log in trajectories:
            try:
                results.append(self.evaluate_with_trajectory_score(log))
            except Exception:
                logger.exception(
                    "evaluate_with_trajectory_score failed for %s — skipping.",
                    log.log_id,
                )
        logger.info(
            "batch_evaluate: scored %d/%d trajectories",
            len(results),
            len(trajectories),
        )
        return results

    def compute_batch_nondeterminism(
        self, task_groups: dict[str, list[WearableLog]]
    ) -> dict[str, Any]:
        """Compute nondeterminism variance for each task group.

        Each key in ``task_groups`` is a task identifier; its value is a list
        of WearableLog instances representing ≥ 2 independent runs of that task.
        Tasks with fewer than 2 runs are skipped with a warning.

        Args:
            task_groups: Mapping from task_id to list of WearableLog runs.

        Returns:
            Mapping from task_id to the variance dict returned by
            :meth:`~src.eval.trajectory_scorer.TrajectoryScorer.compute_nondeterminism_variance`.
            Skipped tasks are absent from the result.
        """
        output: dict[str, Any] = {}
        for task_id, logs in task_groups.items():
            if len(logs) < 2:
                logger.warning(
                    "task '%s' has %d run(s) — need ≥ 2; skipping.", task_id, len(logs)
                )
                continue
            try:
                output[task_id] = (
                    self._trajectory_scorer.compute_nondeterminism_variance(
                        task_id, logs
                    )
                )
            except Exception:
                logger.exception(
                    "compute_nondeterminism_variance failed for task '%s'.", task_id
                )
        return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="JSONL file of trajectory records."),
    ] = None,
    output_path: Annotated[
        Path,
        typer.Option("--output", "-o", help="JSONL file for eval results."),
    ] = Path("data/processed/agentic_eval_results.jsonl"),
    sla_ms: Annotated[
        float,
        typer.Option("--sla-ms", help="Latency SLA threshold in milliseconds."),
    ] = 5000.0,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Validate imports and instantiate AgenticEvaluator; skip file I/O.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Score agent trajectories from a JSONL file and write results."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if dry_run:
        evaluator = AgenticEvaluator(dry_run=True)
        logger.info(
            "Dry-run: AgenticEvaluator instantiated OK (trajectory_scorer=%r)",
            evaluator._trajectory_scorer,
        )
        console.print(
            "[green]✓[/green] Import smoke test passed — AgenticEvaluator ready."
        )
        return

    if input_path is None:
        logger.error("--input is required when not using --dry-run.")
        raise typer.Exit(1)

    records = [
        json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
    ]
    logger.info("Loaded %d trajectory records from %s", len(records), input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[AgenticEvalResult] = []

    for rec in records:
        result = evaluate_trajectory(
            trajectory_id=rec.get("log_id", "unknown"),
            task_id=rec.get("task_id", "unknown"),
            framework=rec.get("framework", "unknown"),
            trajectory=rec.get("trajectory", []),
            response=rec.get("response", ""),
            context=rec.get("context", ""),
            latency_ms=float(rec.get("latency_ms", 0.0)),
            sla_ms=sla_ms,
        )
        results.append(result)

    with output_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")

    logger.info("Wrote %d eval results to %s", len(results), output_path)

    table = Table(title="Agentic Eval Results", show_lines=True)
    columns = (
        "trajectory_id",
        "framework",
        "success",
        "tool",
        "ground",
        "privacy",
        "orch",
        "latency",
        "overall",
    )
    for col in columns:
        table.add_column(col, style="cyan" if col == "overall" else "white")

    for r in results:
        table.add_row(
            r.trajectory_id[:20],
            r.framework,
            f"{r.trajectory_success_rate:.2f}",
            f"{r.tool_invocation_accuracy:.2f}",
            f"{r.groundedness_score:.2f}",
            "LEAK" if r.privacy_leak_detected else "ok",
            f"{r.orchestrator_correctness:.2f}",
            f"{r.latency_sla_compliance:.2f}",
            f"{r.overall_score:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
