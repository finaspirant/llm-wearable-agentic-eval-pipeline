"""Step-level Process Reward Model annotator with partial credit.

Implements a simplified version of the AgentPRM annotation protocol
(arXiv 2502.10325, "Process Reward Models for Agentic Tasks via Monte Carlo
Rollouts") for step-level reward assignment in wearable agent trajectories.

Addresses the **gradient conflict problem** identified in ReasonRAG (NeurIPS 2025,
arXiv 2505.14069): outcome-only reward (ORM) assigns zero reward to every step
in a failed trajectory — including steps that were correct — because the terminal
step failed.  This penalises the model for correct sensing and planning when the
failure is attributable to a downstream configuration error (e.g. missing
emergency contact, DP-noised GPS coordinate displacing the geofence hit).

The fix is three-field reward annotation per step:

  ``process_reward_score``
      Per-step PRM signal in [−1.0, +1.0].  Measures whether this step, in
      isolation, contributed positively or negatively to the trajectory's
      eventual outcome.  Negative scores are valid and important — they are
      DPO negative-example training signal.

  ``outcome_reward``
      ORM signal, non-zero only on the terminal ``act`` step.
      ``+1.0`` → session succeeded; ``−1.0`` → session failed; ``0.0`` on
      all non-terminal steps.  Used exclusively for gradient conflict detection.

  ``partial_credit``
      Step-level correctness score in [0.0, 1.0] for non-terminal steps,
      assessed independently of the terminal outcome.  A correct ``sense``
      step receives ``partial_credit = 1.0`` even when the ``act`` step
      fails because no emergency contact was on file — preserving the
      learning signal that sensing was done right.

**Gradient conflict statistic (WP1 §3 empirical hook):**

    % of failed trajectories (``outcome_reward ≤ 0``) where ≥ 50 % of
    steps carry ``process_reward_score > 0``

If this proportion is high — e.g. 60–70 % — ORM would penalise trajectories
whose reasoning process was largely correct, motivating PRM + partial credit
as the training-data curation strategy.  This is the central empirical
claim of "Beyond Preference Pairs" (WP1).

Key references:

  - **ReasonRAG** (NeurIPS 2025, arXiv 2505.14069): Process-supervised DPO
    outperforms outcome-supervised RL with 18× fewer training queries.
    MCTS exploration + SPRE reward assignment.  Core citation for PRM
    motivation.

  - **AgentPRM** (arXiv 2502.10325): Monte Carlo rollout annotation for
    step-level rewards in agentic tasks.  This module implements a
    deterministic approximation of AgentPRM's MC rollout scoring, replacing
    stochastic rollouts with rubric-grounded heuristics over the wearable
    trajectory schema.

CLI::

    python -m src.annotation.prm_annotator \\
        --input  data/raw/synthetic_wearable_logs.jsonl \\
        --output data/annotations/prm_annotations_20.jsonl \\
        --limit  20 \\
        --summary-output data/annotations/prm_summary_stats.json
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Tool names that represent a decisive positive terminal action.  A terminal
# step calling one of these when the session succeeded earns +1.0 process
# reward; no special case applies when the session failed (the heuristic
# branch handles that path).  Mirrors the AgentAction enum subset that maps
# to life-safety or critical-alert outcomes.
_ALERT_TOOLS: frozenset[str] = frozenset({"alert_user", "notify_emergency"})

# Step quality scores from the annotation schema use a 1–4 integer scale.
# This linear transform maps them to [−1.0, +1.0]:
#   1 → −1.00  (unacceptable)
#   2 → −0.33  (poor)
#   3 → +0.33  (good)
#   4 → +1.00  (excellent)
_STEP_QUALITY_SCALE_MIN: float = 1.0
_STEP_QUALITY_SCALE_MAX: float = 4.0
_STEP_QUALITY_MIDPOINT: float = (
    (_STEP_QUALITY_SCALE_MIN + _STEP_QUALITY_SCALE_MAX) / 2  # 2.5
)
_STEP_QUALITY_HALF_RANGE: float = (
    (_STEP_QUALITY_SCALE_MAX - _STEP_QUALITY_SCALE_MIN) / 2  # 1.5
)


# ---------------------------------------------------------------------------
# Scoring configuration
# ---------------------------------------------------------------------------


@dataclass
class PRMScoringConfig:
    """Reward assignment constants for PRM step annotation.

    All four values are used by the annotator to assign
    ``process_reward_score``, ``outcome_reward``, and
    ``partial_credit`` fields on :class:`StepReward` records.

    Using a dataclass rather than bare module constants makes the
    configuration injectable in tests — callers can override individual
    thresholds without monkey-patching.

    Args:
        CORRECT_TERMINAL_REWARD: ``outcome_reward`` assigned to the terminal
            step of a trajectory where ``overall_goal_achieved = True``.
        FAILED_TERMINAL_REWARD: ``outcome_reward`` assigned to the terminal
            step of a trajectory where ``overall_goal_achieved = False``.
        NEUTRAL_STEP_REWARD: ``outcome_reward`` assigned to all non-terminal
            steps.  Always 0.0 — the ORM signal carries no information at
            intermediate steps by definition.
        GRADIENT_CONFLICT_THRESHOLD: Fraction of steps with
            ``process_reward_score > 0`` required to classify a failed
            trajectory as a gradient conflict instance.  A trajectory is
            flagged when this fraction meets or exceeds the threshold.
    """

    CORRECT_TERMINAL_REWARD: float = 1.0
    FAILED_TERMINAL_REWARD: float = -1.0
    NEUTRAL_STEP_REWARD: float = 0.0
    GRADIENT_CONFLICT_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Step-level reward record
# ---------------------------------------------------------------------------


@dataclass
class StepReward:
    """PRM reward annotation for one step in a wearable agent trajectory.

    One ``StepReward`` is produced per step per trajectory.  For a standard
    3-step sense → plan → act trajectory, each log yields three records.

    ``process_reward_score`` and ``partial_credit`` are the primary PRM
    training signals.  ``outcome_reward`` is the ORM signal and is non-zero
    only on the terminal step — it exists on every record to simplify
    serialisation, but callers must check ``is_terminal`` before using it.

    Args:
        step_index: 0-based index within the trajectory.  For the canonical
            3-step schema: 0 = sense, 1 = plan, 2 = act.
        step_type: Human-readable step label — one of ``"sense"``,
            ``"plan"``, or ``"act"``.  Matches ``TrajectoryStep.step_name``
            in the wearable log schema.
        process_reward_score: PRM training signal for this step in [−1.0,
            +1.0].  Measures whether the step contributed positively or
            negatively to the trajectory outcome, independent of whether the
            session ultimately succeeded.  Negative values are valid and
            intentional — they are DPO negative-example signal per
            ReasonRAG §3.2.
        partial_credit: Step-level correctness score in [0.0, 1.0] for
            non-terminal steps.  Answers: "if we strip away terminal
            outcome, was this step's output correct and causally useful?"
            For the terminal ``act`` step, equals
            ``max(0.0, process_reward_score)`` normalised to [0, 1].
        outcome_reward: ORM signal in [−1.0, +1.0].  ``+1.0`` for a
            successful terminal step, ``−1.0`` for a failed terminal step,
            ``0.0`` for all non-terminal steps.  Set from
            :attr:`PRMScoringConfig.CORRECT_TERMINAL_REWARD`,
            :attr:`PRMScoringConfig.FAILED_TERMINAL_REWARD`, or
            :attr:`PRMScoringConfig.NEUTRAL_STEP_REWARD`.
        is_terminal: ``True`` only for the final ``act`` step.  When
            ``True``, ``outcome_reward`` carries the ORM signal; when
            ``False``, ``outcome_reward`` is always 0.0.
        annotator_rationale: Free-text justification for the assigned
            scores.  Must reference at least one policy condition (e.g.
            consent model, sensor threshold, role authority boundary) to
            meet the BERTScore quality gate (F1 ≥ 0.70) defined in
            ``agenteval-schema-v1``.  Minimum 20 characters.
    """

    step_index: int
    step_type: str
    process_reward_score: float
    partial_credit: float
    outcome_reward: float
    is_terminal: bool
    annotator_rationale: str


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------


class PRMAnnotator:
    """Rule-based Process Reward Model annotator for wearable trajectories.

    Assigns three reward fields per trajectory step without requiring live
    LLM calls, making it deterministic and suitable for offline batch
    annotation of the 20-trajectory WP1 dataset.

    The scoring logic is a rubric-grounded approximation of the AgentPRM
    Monte Carlo rollout protocol (arXiv 2502.10325).  Stochastic rollouts
    are replaced by three ordered heuristics:

    1. **Tool-match heuristic** — decisive alert tools on successful terminal
       steps earn +1.0; ``no_action`` on a failed terminal step earns −1.0.
    2. **Step-quality heuristic** — if the step dict carries a ``step_quality``
       annotation (1–4 scale from the Day 12 annotator pipeline), it is
       linearly mapped to [−1.0, +1.0].
    3. **Positional heuristic** — fallback when neither of the above applies.
       Early steps (sense) get a small positive prior (+0.30); later steps
       decay toward −0.50, reflecting the increasing risk of compounding
       errors.  Clamped to [−0.5, +0.5] to keep the range conservative.

    Args:
        config: Reward assignment constants.  Pass a custom
            :class:`PRMScoringConfig` to override thresholds in tests.
    """

    def __init__(self, config: PRMScoringConfig | None = None) -> None:
        self._config: PRMScoringConfig = (
            config if config is not None else PRMScoringConfig()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate_step(
        self,
        step: dict[str, Any],
        step_index: int,
        is_terminal: bool,
        outcome_success: bool,
    ) -> StepReward:
        """Produce a :class:`StepReward` for one trajectory step.

        The method is pure (no side effects) and deterministic for a given
        ``(step, step_index, is_terminal, outcome_success)`` tuple.

        Args:
            step: A trajectory step dict.  Accepts both the wearable log
                schema (keys: ``step_name``, ``action``) and the annotation
                schema (keys: ``step_type``, ``tool_called``).  When both
                aliases exist, the annotation-schema key takes priority.
            step_index: 0-based position within the trajectory (0 = sense,
                1 = plan, 2 = act for the canonical 3-step schema).
            is_terminal: ``True`` only for the final step.  Controls whether
                ``outcome_reward`` carries the ORM signal or 0.0.
            outcome_success: ``True`` when the session achieved its goal
                (``overall_goal_achieved`` in the wearable log schema).

        Returns:
            A fully populated :class:`StepReward` record.
        """
        tool_called: str = step.get("tool_called") or step.get("action") or ""
        step_type: str = (
            step.get("step_type") or step.get("step_name") or f"step_{step_index}"
        )

        outcome_reward = self._compute_outcome_reward(is_terminal, outcome_success)
        process_reward_score = self._compute_process_reward(
            tool_called, step, step_index, outcome_success
        )
        partial_credit = self._compute_partial_credit(process_reward_score, is_terminal)
        outcome_label = "success" if outcome_success else "failure"
        annotator_rationale = (
            f"Step {step_index} ({tool_called}): process={process_reward_score:.2f}, "
            f"terminal={is_terminal}, outcome={outcome_label}"
        )

        logger.debug(
            "annotate_step | index=%d type=%s tool=%s prs=%.2f pc=%.2f or=%.1f",
            step_index,
            step_type,
            tool_called,
            process_reward_score,
            partial_credit,
            outcome_reward,
        )

        return StepReward(
            step_index=step_index,
            step_type=step_type,
            process_reward_score=process_reward_score,
            partial_credit=partial_credit,
            outcome_reward=outcome_reward,
            is_terminal=is_terminal,
            annotator_rationale=annotator_rationale,
        )

    def annotate_trajectory(self, trajectory: dict[str, Any]) -> list[StepReward]:
        """Annotate every step in a trajectory dict.

        Args:
            trajectory: A dict with at minimum:

                - ``"steps"`` (*list[dict]*) — ordered step records.  Falls
                  back to the wearable log key ``"trajectory"`` when
                  ``"steps"`` is absent.
                - ``"outcome_success"`` (*bool*, optional) — whether the
                  session succeeded.  Defaults to ``False`` when absent.
                - ``"final_action"`` (*str*, optional) — terminal action
                  label; present for documentation but not used in scoring
                  (the terminal step's ``action`` / ``tool_called`` field
                  carries this information).

        Returns:
            One :class:`StepReward` per step, in trajectory order.  The
            last element always has ``is_terminal=True``.

        Raises:
            ValueError: If ``trajectory`` contains no steps.
        """
        steps: list[dict[str, Any]] = (
            trajectory.get("steps") or trajectory.get("trajectory") or []
        )
        if not steps:
            raise ValueError(
                "trajectory must contain at least one step under 'steps' "
                "or 'trajectory' key."
            )

        outcome_success: bool = bool(trajectory.get("outcome_success", False))
        n_steps = len(steps)

        rewards: list[StepReward] = []
        for i, step in enumerate(steps):
            is_terminal = i == n_steps - 1
            rewards.append(self.annotate_step(step, i, is_terminal, outcome_success))

        logger.debug(
            "annotate_trajectory | steps=%d outcome=%s conflicts_candidate=%s",
            n_steps,
            outcome_success,
            self.is_gradient_conflict(rewards),
        )
        return rewards

    def is_gradient_conflict(self, step_rewards: list[StepReward]) -> bool:
        """Return ``True`` if this trajectory is a gradient conflict instance.

        A gradient conflict occurs when ORM would penalise a trajectory
        (``outcome_reward < 0`` on the terminal step) even though the
        majority of non-terminal steps had positive process reward — i.e.
        the process was largely correct but the terminal step failed for an
        unrelated reason (e.g. missing emergency contact, DP-noised GPS).

        Formally, a trajectory is flagged when **both** conditions hold:

        1. The terminal step has ``outcome_reward < 0``.
        2. The fraction of non-terminal steps with
           ``process_reward_score > 0`` ≥
           :attr:`PRMScoringConfig.GRADIENT_CONFLICT_THRESHOLD`.

        Args:
            step_rewards: Output of :meth:`annotate_trajectory`.

        Returns:
            ``True`` if the trajectory is a gradient conflict instance,
            ``False`` otherwise (including when there are no non-terminal
            steps to evaluate).
        """
        terminal = next((r for r in reversed(step_rewards) if r.is_terminal), None)
        if terminal is None or terminal.outcome_reward >= 0.0:
            return False

        non_terminal = [r for r in step_rewards if not r.is_terminal]
        if not non_terminal:
            return False

        positive_count = sum(1 for r in non_terminal if r.process_reward_score > 0.0)
        positive_fraction = positive_count / len(non_terminal)
        return positive_fraction >= self._config.GRADIENT_CONFLICT_THRESHOLD

    def annotate_dataset(
        self, trajectories: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Annotate a batch of trajectories and return one record per trajectory.

        Each record contains the original ``log_id`` and ``outcome_success``
        flag, a ``gradient_conflict`` boolean, and the full list of per-step
        reward dicts serialised via :func:`dataclasses.asdict`.

        Args:
            trajectories: List of trajectory dicts, each accepted by
                :meth:`annotate_trajectory`.  Trajectories with no steps
                (which would raise :exc:`ValueError` in
                :meth:`annotate_trajectory`) are skipped with a WARNING log.

        Returns:
            One dict per trajectory::

                {
                    "log_id": str,
                    "outcome_success": bool,
                    "gradient_conflict": bool,
                    "steps": [
                        {
                            "step_index": int,
                            "step_type": str,
                            "process_reward_score": float,
                            "partial_credit": float,
                            "outcome_reward": float,
                            "is_terminal": bool,
                            "annotator_rationale": str,
                        },
                        ...
                    ],
                }
        """
        results: list[dict[str, Any]] = []
        for trajectory in trajectories:
            log_id: str = trajectory.get("log_id", "")
            outcome_success: bool = bool(trajectory.get("outcome_success", False))
            try:
                step_rewards = self.annotate_trajectory(trajectory)
            except ValueError:
                logger.warning(
                    "annotate_dataset | skipping log_id=%s — no steps found",
                    log_id,
                )
                continue
            results.append(
                {
                    "log_id": log_id,
                    "outcome_success": outcome_success,
                    "gradient_conflict": self.is_gradient_conflict(step_rewards),
                    "steps": [dataclasses.asdict(sr) for sr in step_rewards],
                }
            )
        logger.debug(
            "annotate_dataset | processed=%d skipped=%d",
            len(results),
            len(trajectories) - len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_outcome_reward(
        self, is_terminal: bool, outcome_success: bool
    ) -> float:
        """Return the ORM signal for this step position."""
        if not is_terminal:
            return self._config.NEUTRAL_STEP_REWARD
        return (
            self._config.CORRECT_TERMINAL_REWARD
            if outcome_success
            else self._config.FAILED_TERMINAL_REWARD
        )

    def _compute_process_reward(
        self,
        tool_called: str,
        step: dict[str, Any],
        step_index: int,
        outcome_success: bool,
    ) -> float:
        """Return process_reward_score via the three-heuristic cascade.

        Priority order:
        1. Tool-match heuristic (decisive tools + outcome).
        2. Step-quality annotation, if present in ``step``.
        3. Positional heuristic (fallback).
        """
        # Heuristic 1 — tool-match
        if tool_called in _ALERT_TOOLS and outcome_success:
            return 1.0
        if tool_called == "no_action" and not outcome_success:
            return -1.0

        # Heuristic 2 — step_quality annotation (1–4 ordinal → [−1, +1])
        raw_quality = step.get("step_quality")
        if raw_quality is not None:
            sq = float(raw_quality)
            return (sq - _STEP_QUALITY_MIDPOINT) / _STEP_QUALITY_HALF_RANGE

        # Heuristic 3 — positional fallback
        # Sense steps get a small positive prior; reward decays 0.25 per
        # position.  Range is clamped to [−0.5, +0.5] to stay conservative.
        return max(-0.5, min(0.5, 0.3 - 0.25 * step_index))

    @staticmethod
    def _compute_partial_credit(
        process_reward_score: float, is_terminal: bool
    ) -> float:
        """Return partial_credit normalised to [0.0, 1.0].

        For non-terminal steps: ``max(0.0, process_reward_score)``.
        For the terminal step: ``(process_reward_score + 1.0) / 2.0``,
        which maps [−1, +1] → [0, 1] symmetrically.
        """
        if is_terminal:
            return (process_reward_score + 1.0) / 2.0
        return max(0.0, process_reward_score)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="prm-annotator",
    help="Run step-level PRM annotation on a wearable trajectory JSONL file.",
    add_completion=False,
)


@app.command()
def annotate(
    input: Path = typer.Option(
        Path("data/raw/synthetic_wearable_logs.jsonl"),
        "--input",
        "-i",
        help="Path to input JSONL file of wearable trajectories.",
    ),
    output: Path = typer.Option(
        Path("data/annotations/prm_annotations_20.jsonl"),
        "--output",
        "-o",
        help="Path for per-trajectory annotated output (JSONL).",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum number of trajectories to process.",
    ),
    summary_output: Path = typer.Option(
        Path("data/annotations/prm_summary_stats.json"),
        "--summary-output",
        help="Path for summary statistics JSON.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Annotate wearable trajectories with step-level PRM rewards.

    Reads --limit trajectories from --input, runs PRMAnnotator.annotate_dataset,
    writes one JSONL record per trajectory to --output, and writes gradient
    conflict summary statistics to --summary-output.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    if not input.exists():
        typer.echo(f"Input file not found: {input}", err=True)
        raise typer.Exit(code=1)

    # Load trajectories
    trajectories: list[dict[str, Any]] = []
    with input.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
            if len(trajectories) >= limit:
                break

    typer.echo(f"Loaded {len(trajectories)} trajectories from {input}")

    # Annotate
    annotator = PRMAnnotator()
    results = annotator.annotate_dataset(trajectories)

    # Write JSONL output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record) + "\n")

    typer.echo(f"Wrote {len(results)} annotated trajectories to {output}")

    # Compute summary stats
    failed = [r for r in results if not r["outcome_success"]]
    conflicts = [r for r in results if r["gradient_conflict"]]

    gradient_conflict_rate: float = len(conflicts) / len(failed) if failed else 0.0
    pct_failed_majority_correct: float = gradient_conflict_rate * 100

    non_terminal_steps = [
        step for r in results for step in r["steps"] if not step["is_terminal"]
    ]
    mean_prs: float = (
        sum(s["process_reward_score"] for s in non_terminal_steps)
        / len(non_terminal_steps)
        if non_terminal_steps
        else 0.0
    )
    mean_pc: float = (
        sum(s["partial_credit"] for s in non_terminal_steps) / len(non_terminal_steps)
        if non_terminal_steps
        else 0.0
    )

    summary: dict[str, Any] = {
        "total_trajectories": len(results),
        "failed_trajectories": len(failed),
        "gradient_conflict_count": len(conflicts),
        "gradient_conflict_rate": round(gradient_conflict_rate, 4),
        "pct_failed_with_majority_correct_steps": round(pct_failed_majority_correct, 2),
        "mean_process_reward_non_terminal": round(mean_prs, 4),
        "mean_partial_credit_non_terminal": round(mean_pc, 4),
    }

    # Write summary JSON
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    typer.echo(f"Summary stats written to {summary_output}")

    # Rich summary table
    from rich.console import Console
    from rich.table import Table

    console = Console()

    table = Table(
        title="PRM Annotation Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold", min_width=44)
    table.add_column("Value", justify="right", min_width=12)

    table.add_row("Total trajectories", str(summary["total_trajectories"]))
    table.add_row(
        "Failed trajectories (outcome_success=False)",
        str(summary["failed_trajectories"]),
    )
    table.add_row("Gradient conflict count", str(summary["gradient_conflict_count"]))
    table.add_row(
        "Gradient conflict rate (conflicts / failed)",
        f"{summary['gradient_conflict_rate']:.4f}",
    )
    table.add_row(
        "% failed with majority-correct steps",
        f"{summary['pct_failed_with_majority_correct_steps']:.2f}%",
    )
    table.add_row(
        "Mean process_reward_score (non-terminal)",
        f"{summary['mean_process_reward_non_terminal']:.4f}",
    )
    table.add_row(
        "Mean partial_credit (non-terminal)",
        f"{summary['mean_partial_credit_non_terminal']:.4f}",
    )

    console.print()
    console.print(table)
    console.print()
    console.print(
        f"[bold yellow]WP1 Key Stat:[/bold yellow] "
        f"[bold]{pct_failed_majority_correct:.1f}%[/bold] of outcome-failed "
        f"trajectories "
        f"had majority-correct intermediate steps (gradient conflict instances)"
    )


if __name__ == "__main__":
    app()
