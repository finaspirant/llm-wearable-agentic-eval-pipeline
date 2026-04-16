"""Trajectory-level scorer with 5-layer decomposition.

Decomposes agent trajectories into 5 evaluation layers:
1. Intent parsing — did the agent correctly understand the request?
2. Planning quality — was the plan reasonable and efficient?
3. Tool call precision — were tools called correctly with right args?
4. Recovery behavior — how did the agent handle errors/failures?
5. Outcome — did the agent achieve the goal?

Each layer is scored independently, enabling fine-grained diagnosis of where
agent performance breaks down.  In ``dry_run`` mode all scoring is heuristic
(no LLM calls), making the module usable without API credentials.

PIA integration: :meth:`TrajectoryScorer.score_pia_dimensions` maps the 5
layers onto the four Path-Invariant Agreement rubric dimensions proved in
:mod:`src.annotation.pia_calculator` (planning_quality, error_recovery,
goal_alignment, tool_precision).

Nondeterminism variance: :meth:`TrajectoryScorer.compute_nondeterminism_variance`
accepts 3 runs of the same task and returns per-layer standard deviations,
identifying which evaluation layer is least stable.

CLI::

    python -m src.eval.trajectory_scorer \\
        --input data/raw/synthetic_wearable_logs.jsonl \\
        --output data/processed/trajectory_scores.json \\
        --dry-run --verbose
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from src.data.wearable_generator import AgentAction, ScenarioType, WearableLog

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="trajectory-scorer",
    help="Score agent trajectories across 5 decomposed evaluation layers.",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "intent": 0.15,
    "planning": 0.25,
    "tool_calls": 0.25,
    "recovery": 0.15,
    "outcome": 0.20,
}

# Actions that represent a decisive terminal outcome (goal achieved).
# SUPPRESS_CAPTURE, TRIGGER_GEOFENCE, ADJUST_NOISE_PROFILE, SURFACE_REMINDER
# are scenario-specific decisive actions; LOG_AND_MONITOR is monitoring-only
# and is intentionally excluded.
_TERMINAL_ACTIONS: frozenset[str] = frozenset(
    {
        AgentAction.SEND_ALERT,
        AgentAction.SUPPRESS_CAPTURE,
        AgentAction.TRIGGER_GEOFENCE,
        AgentAction.ADJUST_NOISE_PROFILE,
        AgentAction.SURFACE_REMINDER,
        AgentAction.ESCALATE_TO_EMERGENCY,
    }
)

# Recovery is inferred from escalation — the agent acknowledged an error
# state and handed off to a higher authority.
_RECOVERY_ACTION: str = AgentAction.ESCALATE_TO_EMERGENCY

# Valid AgentAction string values for precision check.
_VALID_ACTIONS: frozenset[str] = frozenset(a.value for a in AgentAction)


# ---------------------------------------------------------------------------
# Score dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IntentScore:
    """Score for the intent-parsing layer.

    Args:
        score: Float in [0, 1].
        reasoning: Explanation of how the score was derived.
        matched_goal: Whether the agent's stated goal matched the scenario type.
    """

    score: float
    reasoning: str
    matched_goal: bool


@dataclass
class PlanningScore:
    """Score for the planning-quality layer.

    Args:
        score: Float in [0, 1].
        reasoning: Explanation of how the score was derived.
        step_efficiency: Ratio of ideal steps (3) to actual steps, capped at 1.0.
    """

    score: float
    reasoning: str
    step_efficiency: float


@dataclass
class ToolCallScore:
    """Score for the tool-call-precision layer.

    Args:
        score: Float in [0, 1].
        reasoning: Explanation of how the score was derived.
        precision: Fraction of step actions that are valid AgentAction members.
        false_positives: Count of steps with actions not in the valid action set.
    """

    score: float
    reasoning: str
    precision: float
    false_positives: int


@dataclass
class RecoveryScore:
    """Score for the recovery-behavior layer.

    Args:
        score: Float in [0, 1] if an error was detected, else None
            (layer is not applicable and is excluded from aggregation).
        reasoning: Explanation of how the score was derived.
        had_error: True if an escalation action was detected in the trajectory.
    """

    score: float | None
    reasoning: str
    had_error: bool


@dataclass
class OutcomeScore:
    """Score for the outcome layer.

    Args:
        score: Float in [0, 1].
        reasoning: Explanation of how the score was derived.
        goal_achieved: Whether the final trajectory step took a terminal action.
    """

    score: float
    reasoning: str
    goal_achieved: bool


@dataclass
class TrajectoryScore:
    """Aggregated score across all 5 evaluation layers.

    Args:
        trajectory_id: Log ID from the source WearableLog.
        intent: IntentScore for this trajectory.
        planning: PlanningScore for this trajectory.
        tool_calls: ToolCallScore for this trajectory.
        recovery: RecoveryScore for this trajectory.
        outcome: OutcomeScore for this trajectory.
        weighted_total: Weighted composite score in [0, 1].  Recovery is
            excluded and remaining weights are renormalized when recovery
            is not applicable.
        metadata: Freeform dict capturing scenario_type, step count, etc.
    """

    trajectory_id: str
    intent: IntentScore
    planning: PlanningScore
    tool_calls: ToolCallScore
    recovery: RecoveryScore
    outcome: OutcomeScore
    weighted_total: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "trajectory_id": self.trajectory_id,
            "weighted_total": self.weighted_total,
            "intent": asdict(self.intent),
            "planning": asdict(self.planning),
            "tool_calls": asdict(self.tool_calls),
            "recovery": asdict(self.recovery),
            "outcome": asdict(self.outcome),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# TrajectoryScorer
# ---------------------------------------------------------------------------


class TrajectoryScorer:
    """Score WearableLog trajectories across 5 decomposed evaluation layers.

    In ``dry_run=True`` mode (default) all scoring is purely heuristic —
    no LLM API calls are made.  This is the safe default for CI, testing,
    and batch baseline runs.

    Args:
        weights: Per-layer weights for the weighted composite score.
            Must sum to 1.0.  Defaults to the CLAUDE.md-specified split.
        dry_run: When True, use deterministic heuristic scoring instead of
            LLM-as-judge calls.

    Raises:
        ValueError: If ``weights`` keys don't cover all 5 layers.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        dry_run: bool = True,
    ) -> None:
        self._weights = weights if weights is not None else dict(_DEFAULT_WEIGHTS)
        self._dry_run = dry_run
        required = {"intent", "planning", "tool_calls", "recovery", "outcome"}
        missing = required - self._weights.keys()
        if missing:
            raise ValueError(f"weights missing keys: {missing}")

    # ------------------------------------------------------------------
    # Layer scorers
    # ------------------------------------------------------------------

    def score_intent(self, trajectory: WearableLog) -> IntentScore:
        """Score whether the agent correctly parsed the user's intent.

        In dry_run mode, a valid ``scenario_type`` field is used as a proxy
        for correct intent parsing: a recognised scenario implies the agent
        could identify what kind of event it was handling.

        Args:
            trajectory: Wearable log containing sensor and trajectory data.

        Returns:
            IntentScore with score 0.75 (recognised scenario) or 0.40
            (unrecognised).
        """
        try:
            ScenarioType(trajectory.scenario_type)
            matched = True
            score = 0.75
            reasoning = (
                f"Scenario type '{trajectory.scenario_type}' is a recognised "
                "ScenarioType — agent intent parsing assumed correct."
            )
        except ValueError:
            matched = False
            score = 0.40
            reasoning = (
                f"Scenario type '{trajectory.scenario_type}' is not a recognised "
                "ScenarioType — intent parsing likely incorrect."
            )
        logger.debug("intent score=%.2f matched=%s", score, matched)
        return IntentScore(score=score, reasoning=reasoning, matched_goal=matched)

    def score_planning(self, trajectory: WearableLog) -> PlanningScore:
        """Score the quality and efficiency of the agent's plan.

        Step efficiency is the ratio of the ideal step count (3: sense →
        plan → act) to the actual step count, capped at 1.0.  Longer
        trajectories indicate unnecessary detours.

        Args:
            trajectory: Wearable log containing the agent trajectory.

        Returns:
            PlanningScore with step_efficiency in (0, 1] and score 0.80
            (efficiency > 0.60) or 0.55 (efficiency ≤ 0.60).
        """
        n_steps = max(1, len(trajectory.trajectory))
        step_efficiency = min(1.0, 3 / n_steps)
        score = 0.80 if step_efficiency > 0.6 else 0.55
        reasoning = (
            f"{n_steps} steps taken; step_efficiency={step_efficiency:.3f} "
            f"({'efficient' if score >= 0.80 else 'suboptimal'} plan)."
        )
        logger.debug("planning score=%.2f efficiency=%.3f", score, step_efficiency)
        return PlanningScore(
            score=score, reasoning=reasoning, step_efficiency=step_efficiency
        )

    def score_tool_calls(self, trajectory: WearableLog) -> ToolCallScore:
        """Score precision of tool/action calls across all trajectory steps.

        Each step's ``action`` field is checked against the AgentAction enum.
        Steps with an empty action string (sense and plan steps) are not
        penalised — only non-empty action strings outside the valid set count
        as false positives.

        Args:
            trajectory: Wearable log containing the agent trajectory.

        Returns:
            ToolCallScore with precision in [0, 1] and false_positives count.
        """
        total = len(trajectory.trajectory)
        valid_count = 0
        false_positives = 0
        for step in trajectory.trajectory:
            action = step.action
            if not action:
                valid_count += 1  # empty action on sense/plan steps is correct
            elif action in _VALID_ACTIONS:
                valid_count += 1
            else:
                false_positives += 1

        precision = valid_count / total if total > 0 else 0.0
        score = precision
        reasoning = (
            f"{valid_count}/{total} steps have valid actions; "
            f"{false_positives} false positive(s); precision={precision:.3f}."
        )
        logger.debug(
            "tool_calls score=%.2f precision=%.3f fp=%d",
            score,
            precision,
            false_positives,
        )
        return ToolCallScore(
            score=score,
            reasoning=reasoning,
            precision=precision,
            false_positives=false_positives,
        )

    def score_recovery(self, trajectory: WearableLog) -> RecoveryScore:
        """Score recovery behavior when an error state is detected.

        Recovery is detected by the presence of an ``escalate_to_emergency``
        action in any trajectory step (indicating the agent acknowledged an
        error and delegated to a higher authority).

        If no error is detected the layer is not applicable and score=None.
        The aggregate method will renormalize weights to exclude this layer.

        Args:
            trajectory: Wearable log containing the agent trajectory.

        Returns:
            RecoveryScore with score=0.70 if an escalation was found (partial
            recovery credit), score=None if no error was detected.
        """
        had_error = any(
            step.action == _RECOVERY_ACTION for step in trajectory.trajectory
        )
        if had_error:
            score: float | None = 0.70
            reasoning = (
                "Escalation action detected — agent recognised an error state "
                "and handed off appropriately (partial recovery credit)."
            )
        else:
            score = None
            reasoning = "No escalation detected — recovery layer not applicable."
        logger.debug("recovery had_error=%s score=%s", had_error, score)
        return RecoveryScore(score=score, reasoning=reasoning, had_error=had_error)

    def score_outcome(self, trajectory: WearableLog) -> OutcomeScore:
        """Score whether the agent achieved the session goal.

        Goal achievement is proxied by the final step's action: a decisive
        terminal action (send_alert, suppress_capture, trigger_geofence,
        adjust_noise_profile, surface_reminder, escalate_to_emergency)
        indicates goal completion.  Monitoring-only actions (log_and_monitor)
        or missing actions indicate goal not yet achieved.

        Args:
            trajectory: Wearable log containing the agent trajectory.

        Returns:
            OutcomeScore with goal_achieved=True and score=1.0 if terminal,
            else goal_achieved=False and score=0.0.
        """
        if not trajectory.trajectory:
            return OutcomeScore(
                score=0.0,
                reasoning="Empty trajectory — outcome cannot be assessed.",
                goal_achieved=False,
            )
        final_action = trajectory.trajectory[-1].action
        goal_achieved = final_action in _TERMINAL_ACTIONS
        score = 1.0 if goal_achieved else 0.0
        reasoning = (
            f"Final action '{final_action}' is "
            f"{'a terminal' if goal_achieved else 'not a terminal'} action."
        )
        logger.debug("outcome score=%.2f goal_achieved=%s", score, goal_achieved)
        return OutcomeScore(
            score=score, reasoning=reasoning, goal_achieved=goal_achieved
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        intent: IntentScore,
        planning: PlanningScore,
        tool_calls: ToolCallScore,
        recovery: RecoveryScore,
        outcome: OutcomeScore,
    ) -> float:
        """Compute the weighted composite score across all applicable layers.

        When recovery.score is None (no error detected), the recovery weight
        is redistributed proportionally across the remaining four layers.

        Args:
            intent: Scored intent layer.
            planning: Scored planning layer.
            tool_calls: Scored tool-call layer.
            recovery: Scored recovery layer (score may be None).
            outcome: Scored outcome layer.

        Returns:
            Weighted composite score in [0, 1].
        """
        layer_scores: dict[str, float] = {
            "intent": intent.score,
            "planning": planning.score,
            "tool_calls": tool_calls.score,
            "outcome": outcome.score,
        }
        if recovery.score is not None:
            layer_scores["recovery"] = recovery.score

        active_weights = {k: self._weights[k] for k in layer_scores}
        total_weight = sum(active_weights.values())
        weighted_total = sum(
            active_weights[k] * v / total_weight for k, v in layer_scores.items()
        )
        logger.debug(
            "weighted_total=%.4f active_layers=%s", weighted_total, list(layer_scores)
        )
        return weighted_total

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def score_trajectory(self, trajectory: WearableLog) -> TrajectoryScore:
        """Score a single trajectory across all 5 evaluation layers.

        Args:
            trajectory: Wearable log to evaluate.

        Returns:
            TrajectoryScore containing per-layer scores and a weighted total.
        """
        intent = self.score_intent(trajectory)
        planning = self.score_planning(trajectory)
        tool_calls = self.score_tool_calls(trajectory)
        recovery = self.score_recovery(trajectory)
        outcome = self.score_outcome(trajectory)
        weighted_total = self.aggregate(intent, planning, tool_calls, recovery, outcome)

        metadata: dict[str, Any] = {
            "scenario_type": str(trajectory.scenario_type),
            "consent_model": trajectory.consent_model.value,
            "n_steps": len(trajectory.trajectory),
            "scored_at": datetime.now(UTC).isoformat(),
            "dry_run": self._dry_run,
        }
        return TrajectoryScore(
            trajectory_id=trajectory.log_id,
            intent=intent,
            planning=planning,
            tool_calls=tool_calls,
            recovery=recovery,
            outcome=outcome,
            weighted_total=weighted_total,
            metadata=metadata,
        )

    def batch_score(self, trajectories: list[WearableLog]) -> list[TrajectoryScore]:
        """Score a batch of trajectories.

        Args:
            trajectories: List of wearable logs to evaluate.

        Returns:
            List of TrajectoryScore objects in the same order as input.
        """
        results: list[TrajectoryScore] = []
        for traj in trajectories:
            try:
                results.append(self.score_trajectory(traj))
            except Exception:
                logger.exception(
                    "Failed to score trajectory %s — skipping.", traj.log_id
                )
        logger.info(
            "batch_score: scored %d/%d trajectories", len(results), len(trajectories)
        )
        return results

    # ------------------------------------------------------------------
    # PIA integration
    # ------------------------------------------------------------------

    def score_pia_dimensions(self, trajectory: WearableLog) -> dict[str, float]:
        """Map trajectory scores onto the four PIA rubric dimensions.

        The four dimensions mirror those proved significant in
        :mod:`src.annotation.pia_calculator` (planning_quality, error_recovery,
        goal_alignment, tool_precision).  This allows a single trajectory to
        be positioned in the PIA rubric space without a full annotator panel.

        Mapping logic:
        - planning_quality  → PlanningScore.score
        - error_recovery    → RecoveryScore.score (0.5 if not applicable)
        - goal_alignment    → mean(IntentScore.score, OutcomeScore.score)
        - tool_precision    → ToolCallScore.precision

        Args:
            trajectory: Wearable log to evaluate.

        Returns:
            Dict with keys planning_quality, error_recovery, goal_alignment,
            tool_precision; all values in [0, 1].
        """
        intent = self.score_intent(trajectory)
        planning = self.score_planning(trajectory)
        tool_calls = self.score_tool_calls(trajectory)
        recovery = self.score_recovery(trajectory)
        outcome = self.score_outcome(trajectory)

        return {
            "planning_quality": planning.score,
            "error_recovery": recovery.score if recovery.score is not None else 0.5,
            "goal_alignment": (intent.score + outcome.score) / 2.0,
            "tool_precision": tool_calls.precision,
        }

    # ------------------------------------------------------------------
    # Nondeterminism variance
    # ------------------------------------------------------------------

    def compute_nondeterminism_variance(
        self,
        task_id: str,
        trajectories: list[WearableLog],
    ) -> dict[str, float | str]:
        """Measure score variance across repeated runs of the same task.

        Designed for 3 runs of the same task (e.g. the same prompt submitted
        3 times to the same agent framework).  Returns per-layer standard
        deviations and identifies which evaluation layer is least stable.

        Args:
            task_id: Identifier for the task being evaluated (logged only).
            trajectories: Exactly 3 WearableLog instances representing 3
                independent runs of the same task.  Fewer than 2 raises
                ValueError (std dev requires ≥ 2 samples).

        Returns:
            Dict with keys::

                score_std          — std dev of weighted_total across runs
                pia_planning_std   — std dev of planning_quality
                pia_recovery_std   — std dev of error_recovery
                pia_goal_std       — std dev of goal_alignment
                pia_tool_std       — std dev of tool_precision
                max_variance_layer — which of the 5 layers varies most

        Raises:
            ValueError: If fewer than 2 trajectories are provided.
        """
        if len(trajectories) < 2:
            raise ValueError(
                f"compute_nondeterminism_variance requires ≥ 2 trajectories, "
                f"got {len(trajectories)} for task '{task_id}'."
            )

        scores = [self.score_trajectory(t) for t in trajectories]
        pia_dims = [self.score_pia_dimensions(t) for t in trajectories]

        totals = [s.weighted_total for s in scores]
        planning_vals = [d["planning_quality"] for d in pia_dims]
        recovery_vals = [d["error_recovery"] for d in pia_dims]
        goal_vals = [d["goal_alignment"] for d in pia_dims]
        tool_vals = [d["tool_precision"] for d in pia_dims]

        score_std = statistics.stdev(totals)
        pia_planning_std = statistics.stdev(planning_vals)
        pia_recovery_std = statistics.stdev(recovery_vals)
        pia_goal_std = statistics.stdev(goal_vals)
        pia_tool_std = statistics.stdev(tool_vals)

        layer_stds: dict[str, float] = {
            "intent": statistics.stdev([s.intent.score for s in scores]),
            "planning": pia_planning_std,
            "tool_calls": pia_tool_std,
            "recovery": pia_recovery_std,
            "outcome": statistics.stdev([s.outcome.score for s in scores]),
        }
        max_variance_layer = max(layer_stds, key=lambda k: layer_stds[k])

        logger.info(
            "nondeterminism variance task=%s score_std=%.4f max_variance_layer=%s",
            task_id,
            score_std,
            max_variance_layer,
        )
        return {
            "score_std": score_std,
            "pia_planning_std": pia_planning_std,
            "pia_recovery_std": pia_recovery_std,
            "pia_goal_std": pia_goal_std,
            "pia_tool_std": pia_tool_std,
            "max_variance_layer": max_variance_layer,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_trajectories(path: Path, limit: int | None) -> list[WearableLog]:
    """Load WearableLog instances from a JSONL file.

    Args:
        path: Path to the JSONL file.
        limit: Maximum number of logs to load (None = all).

    Returns:
        List of WearableLog dataclasses reconstructed from JSON.
    """
    from src.data.privacy_gate import ConsentModel
    from src.data.wearable_generator import (
        AudioTranscript,
        SensorData,
        TrajectoryStep,
    )

    logs: list[WearableLog] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            raw = json.loads(line)
            sensor_raw = raw["sensor_data"]
            sensor = SensorData(**sensor_raw)
            audio_raw = raw["audio_transcript"]
            audio = AudioTranscript(**audio_raw)
            steps = [TrajectoryStep(**s) for s in raw["trajectory"]]
            log = WearableLog(
                log_id=raw["log_id"],
                timestamp=raw["timestamp"],
                scenario_type=ScenarioType(raw["scenario_type"]),
                consent_model=ConsentModel(raw["consent_model"]),
                sensor_data=sensor,
                audio_transcript=audio,
                context_metadata=raw["context_metadata"],
                trajectory=steps,
                ground_truth_action=raw["ground_truth_action"],
            )
            logs.append(log)
    return logs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Annotated[
        Path,
        typer.Option("--input", help="Path to JSONL file of WearableLog records."),
    ] = Path("data/raw/synthetic_wearable_logs.jsonl"),
    output_path: Annotated[
        Path,
        typer.Option("--output", help="Path for JSON output of scored trajectories."),
    ] = Path("data/processed/trajectory_scores.json"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Max trajectories to score (default: all)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run/--no-dry-run", help="Heuristic scoring (no LLM)."),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose/--quiet"),
    ] = False,
) -> None:
    """Score WearableLog trajectories across 5 decomposed evaluation layers."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise typer.Exit(1)

    logger.info("Loading trajectories from %s", input_path)
    trajectories = _load_trajectories(input_path, limit)
    logger.info("Loaded %d trajectories", len(trajectories))

    scorer = TrajectoryScorer(dry_run=dry_run)
    results = scorer.batch_score(trajectories)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scored_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "n_trajectories": len(results),
        "scores": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %d scores to %s", len(results), output_path)

    # Rich summary table
    table = Table(title="Trajectory Scores (5-Layer Decomposition)")
    table.add_column("log_id", style="dim", max_width=12)
    table.add_column("scenario")
    table.add_column("intent", justify="right")
    table.add_column("planning", justify="right")
    table.add_column("tools", justify="right")
    table.add_column("recovery", justify="right")
    table.add_column("outcome", justify="right")
    table.add_column("total", justify="right", style="bold")

    for r in results:
        rec_str = f"{r.recovery.score:.2f}" if r.recovery.score is not None else "N/A"
        table.add_row(
            r.trajectory_id[:8],
            str(r.metadata.get("scenario_type", "")),
            f"{r.intent.score:.2f}",
            f"{r.planning.score:.2f}",
            f"{r.tool_calls.score:.2f}",
            rec_str,
            f"{r.outcome.score:.2f}",
            f"{r.weighted_total:.3f}",
        )

    console.print(table)
    mean_total = (
        sum(r.weighted_total for r in results) / len(results) if results else 0.0
    )
    console.print(
        f"\n[bold]Mean weighted total:[/bold] {mean_total:.4f} "
        f"across {len(results)} trajectories"
    )


if __name__ == "__main__":
    app()
