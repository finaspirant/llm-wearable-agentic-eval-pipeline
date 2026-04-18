"""Comparison: single-agent mock vs multi-agent pipeline on 10 wearable logs.

Loads 2 logs per scenario type (10 total), runs both pipelines, scores each
with :class:`~src.eval.trajectory_scorer.TrajectoryScorer`, computes role
attribution for the multi-agent run, and prints a side-by-side comparison
table.

Single-agent baseline
~~~~~~~~~~~~~~~~~~~~~
:class:`~src.agent.wearable_agent_langgraph.WearableAgentLangGraph` is still
a stub (Day 19 TODO).  This module uses :class:`_MockSingleAgentPipeline` as
a stand-in: a deterministic sense → plan → act agent that applies simpler
decision heuristics than the multi-agent pipeline.

Key intentional differences:
- ``privacy_sensitive`` scenarios: single-agent always outputs
  ``log_and_monitor`` (no dedicated privacy specialist); multi-agent applies
  the full ConsentModel matrix (→ ``suppress_capture`` / ``request_consent``).
- No role delegation or attribution in the single-agent path.
- Slightly higher latency variance in multi-agent due to graph traversal.

``log_and_monitor`` is deliberately excluded from ``_TERMINAL_ACTIONS`` in
:mod:`~src.eval.trajectory_scorer`, so ``trajectory_success`` will be
``False`` for single-agent on ``privacy_sensitive`` runs — a meaningful,
semantically grounded difference.

CLI::

    python -m src.eval.multiagent_vs_single_comparison
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from tabulate import tabulate

from src.agent.wearable_multiagent import (
    _EMERGENCY_AUDIO_KEYWORDS,
    _HR_ALERT_THRESHOLD,
    _HR_EMERGENCY_THRESHOLD,
    _NOISE_HEARING_THRESHOLD,
    _PRIVACY_AUDIO_KEYWORDS,
    _SPO2_ALERT_THRESHOLD,
    _SPO2_EMERGENCY_THRESHOLD,
    MultiAgentPipeline,
    MultiAgentResult,
)
from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import (
    AgentAction,
    AudioTranscript,
    ScenarioType,
    SensorData,
    TrajectoryStep,
    WearableLog,
)
from src.eval.role_attribution import AttributionReport, RoleAttributionScorer
from src.eval.trajectory_scorer import TrajectoryScorer

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="multiagent-vs-single",
    help="Compare single-agent mock vs multi-agent pipeline on 10 wearable logs.",
    add_completion=False,
)

_DEFAULT_INPUT = Path("data/raw/synthetic_wearable_logs.jsonl")
_DEFAULT_OUTPUT = Path("data/processed/multiagent_comparison.json")
_LOGS_PER_SCENARIO = 2


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RunMetrics:
    """Evaluation metrics for one pipeline run on one log.

    Args:
        log_id: Source log UUID.
        scenario_type: Scenario type string.
        pipeline: ``"single_agent"`` or ``"multi_agent"``.
        trajectory_success: Whether the final action is a terminal action.
        tool_invocation_accuracy: Tool-call precision score from
            :class:`~src.eval.trajectory_scorer.TrajectoryScorer`.
        privacy_compliance: Whether the chosen action is compatible with the
            log's consent model and any sensitive audio keywords.
        latency_ms: Wall-clock execution time in milliseconds.
        weighted_total: Composite trajectory score in [0, 1].
        attribution: :class:`~src.eval.role_attribution.AttributionReport`
            for multi-agent runs; ``None`` for single-agent.
    """

    log_id: str
    scenario_type: str
    pipeline: str
    trajectory_success: bool
    tool_invocation_accuracy: float
    privacy_compliance: bool
    latency_ms: float
    weighted_total: float
    attribution: AttributionReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        d: dict[str, Any] = {
            "log_id": self.log_id,
            "scenario_type": self.scenario_type,
            "pipeline": self.pipeline,
            "trajectory_success": self.trajectory_success,
            "tool_invocation_accuracy": round(self.tool_invocation_accuracy, 4),
            "privacy_compliance": self.privacy_compliance,
            "latency_ms": round(self.latency_ms, 3),
            "weighted_total": round(self.weighted_total, 4),
        }
        if self.attribution is not None:
            d["attribution"] = self.attribution.to_dict()
        return d


# ---------------------------------------------------------------------------
# Privacy compliance helper
# ---------------------------------------------------------------------------


def _is_privacy_compliant(action: str, log: WearableLog) -> bool:
    """Return True if the chosen action is compatible with the consent model.

    Args:
        action: String value of the chosen :class:`AgentAction`.
        log: Source :class:`WearableLog`.

    Returns:
        ``False`` when the action captures or processes data in a way that
        violates the applicable consent constraint; ``True`` otherwise.
    """
    consent = log.consent_model
    keywords = set(log.audio_transcript.keywords_detected)
    has_sensitive = bool(_PRIVACY_AUDIO_KEYWORDS & keywords)

    if consent == ConsentModel.REVOKED:
        # Only suppress_capture fully honours revoked consent.
        return action == AgentAction.SUPPRESS_CAPTURE.value

    if consent == ConsentModel.AMBIENT and has_sensitive:
        # Ambiguous consent + private audio → must stop or request consent.
        return action in {
            AgentAction.SUPPRESS_CAPTURE.value,
            AgentAction.REQUEST_CONSENT.value,
        }

    # EXPLICIT or IMPLIED consent with no sensitive triggers: all actions compliant.
    return True


# ---------------------------------------------------------------------------
# Mock single-agent pipeline
# ---------------------------------------------------------------------------


class _MockSingleAgentPipeline:
    """Deterministic single-agent baseline.

    Implements the same sense → plan → act structure as
    :class:`~src.agent.wearable_multiagent.MultiAgentPipeline` but applies
    simpler decision heuristics: no specialist delegation, no ConsentModel
    matrix for location or privacy scenarios.

    This is used in place of
    :class:`~src.agent.wearable_agent_langgraph.WearableAgentLangGraph`,
    which is still a stub.

    Design note — intentional weaknesses vs multi-agent:
        * ``privacy_sensitive``: always outputs ``log_and_monitor`` (no
          dedicated privacy specialist → not a terminal action → goal fails).
        * ``location_trigger``: always outputs ``trigger_geofence`` with no
          consent check (may violate REVOKED consent if such logs existed).
        * No role attribution layer.
    """

    def run(self, log: WearableLog) -> WearableLog:
        """Execute the single-agent pipeline and return a scored WearableLog.

        The returned log is a copy of the input with the
        :attr:`~src.data.wearable_generator.WearableLog.trajectory` field
        replaced by the single-agent's three-step trajectory.

        Args:
            log: Source wearable event log.

        Returns:
            A new :class:`~src.data.wearable_generator.WearableLog` whose
            trajectory reflects the single-agent's sense → plan → act path.
        """
        action, confidence, reasoning = self._decide_action(log)

        sense_step = TrajectoryStep(
            step_index=0,
            step_name="sense",
            observation=(
                f"Single-agent sensed: scenario={log.scenario_type.value},"
                f" consent={log.consent_model.value},"
                f" HR={log.sensor_data.heart_rate_noised:.0f} bpm,"
                f" SpO2={log.sensor_data.spo2_noised:.1f}%,"
                f" noise={log.sensor_data.noise_db_noised:.0f} dB."
            ),
            reasoning="Sensor fusion complete — passing to plan step.",
            action="",
            confidence=0.90,
        )
        plan_step = TrajectoryStep(
            step_index=1,
            step_name="plan",
            observation=f"Scenario classified as {log.scenario_type.value}.",
            reasoning=reasoning,
            action="",
            confidence=confidence,
        )
        act_step = TrajectoryStep(
            step_index=2,
            step_name="act",
            observation=f"Executing {action}.",
            reasoning=f"Single-agent selected {action} based on sensor state.",
            action=action,
            confidence=confidence,
        )
        return dataclasses.replace(log, trajectory=[sense_step, plan_step, act_step])

    def _decide_action(self, log: WearableLog) -> tuple[str, float, str]:
        """Choose an action using simplified (no-specialist) heuristics.

        Args:
            log: Source wearable event log.

        Returns:
            Tuple of (action_value, confidence, reasoning_string).
        """
        scenario = log.scenario_type
        consent = log.consent_model
        s = log.sensor_data
        keywords = set(log.audio_transcript.keywords_detected)

        if scenario == ScenarioType.HEALTH_ALERT:
            return self._health_heuristic(s.heart_rate_noised, s.spo2_noised, keywords)

        if scenario == ScenarioType.PRIVACY_SENSITIVE:
            # Single-agent has no dedicated privacy specialist: defaults to
            # log_and_monitor instead of applying the ConsentModel matrix.
            if consent == ConsentModel.REVOKED:
                return (
                    AgentAction.SUPPRESS_CAPTURE.value,
                    0.95,
                    "Revoked consent detected; suppressing capture.",
                )
            return (
                AgentAction.LOG_AND_MONITOR.value,
                0.65,
                (
                    "Privacy-sensitive scenario detected. Single-agent defaults"
                    " to log_and_monitor — no specialist available to apply"
                    " full ConsentModel decision matrix."
                ),
            )

        if scenario == ScenarioType.LOCATION_TRIGGER:
            # Single-agent triggers geofence without consulting a privacy gate.
            return (
                AgentAction.TRIGGER_GEOFENCE.value,
                0.80,
                "Location trigger detected; firing geofence without consent audit.",
            )

        if scenario == ScenarioType.AMBIENT_NOISE:
            return self._noise_heuristic(s.noise_db_noised)

        # calendar_reminder
        return (
            AgentAction.SURFACE_REMINDER.value,
            0.92,
            "Calendar event detected; surfacing reminder to user.",
        )

    @staticmethod
    def _health_heuristic(
        hr: float, spo2: float, keywords: set[str]
    ) -> tuple[str, float, str]:
        """Apply biometric thresholds (identical to HealthAgent heuristics).

        Args:
            hr: DP-noised heart rate (bpm).
            spo2: DP-noised SpO2 (%).
            keywords: Audio keyword set from the transcript.

        Returns:
            Tuple of (action_value, confidence, reasoning_string).
        """
        has_emergency_audio = bool(_EMERGENCY_AUDIO_KEYWORDS & keywords)

        if (
            hr > _HR_EMERGENCY_THRESHOLD or spo2 < _SPO2_EMERGENCY_THRESHOLD
        ) and has_emergency_audio:
            return (
                AgentAction.ESCALATE_TO_EMERGENCY.value,
                0.98,
                "Critical biometrics + emergency audio → escalate.",
            )
        if hr > _HR_ALERT_THRESHOLD or spo2 < _SPO2_ALERT_THRESHOLD:
            return (AgentAction.SEND_ALERT.value, 0.87, "Alert biometrics → alert.")
        return (
            AgentAction.LOG_AND_MONITOR.value,
            0.75,
            "Biometrics within alert range → log and monitor.",
        )

    @staticmethod
    def _noise_heuristic(noise_db: float) -> tuple[str, float, str]:
        """Decide noise-management action from ambient dB level.

        Args:
            noise_db: DP-noised ambient noise level (dB SPL).

        Returns:
            Tuple of (action_value, confidence, reasoning_string).
        """
        if noise_db > _NOISE_HEARING_THRESHOLD:
            return (
                AgentAction.ADJUST_NOISE_PROFILE.value,
                0.90,
                f"Noise {noise_db:.0f} dB exceeds WHO threshold → adjust profile.",
            )
        return (
            AgentAction.LOG_AND_MONITOR.value,
            0.78,
            f"Noise level {noise_db:.0f} dB below threshold → log and monitor.",
        )


# ---------------------------------------------------------------------------
# Log loader
# ---------------------------------------------------------------------------


def _load_logs_by_scenario(
    path: Path,
    n_per_scenario: int,
) -> list[WearableLog]:
    """Load n_per_scenario logs for each of the 5 scenario types.

    Args:
        path: Path to a JSONL file produced by
            :mod:`src.data.wearable_generator`.
        n_per_scenario: Number of logs to include per scenario.

    Returns:
        Ordered list of :class:`~src.data.wearable_generator.WearableLog`
        instances (total = n_per_scenario × 5).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file contains fewer than ``n_per_scenario`` logs
            for any scenario type.
    """
    buckets: dict[str, list[WearableLog]] = defaultdict(list)

    with path.open() as fh:
        for line in fh:
            raw = json.loads(line)
            scenario = raw["scenario_type"]
            if len(buckets[scenario]) >= n_per_scenario:
                continue
            buckets[scenario].append(
                WearableLog(
                    log_id=raw["log_id"],
                    timestamp=raw["timestamp"],
                    scenario_type=ScenarioType(raw["scenario_type"]),
                    consent_model=ConsentModel(raw["consent_model"]),
                    sensor_data=SensorData(**raw["sensor_data"]),
                    audio_transcript=AudioTranscript(**raw["audio_transcript"]),
                    context_metadata=raw["context_metadata"],
                    trajectory=[TrajectoryStep(**s) for s in raw["trajectory"]],
                    ground_truth_action=raw["ground_truth_action"],
                )
            )

    scenario_names = [s.value for s in ScenarioType]
    for name in scenario_names:
        got = len(buckets.get(name, []))
        if got < n_per_scenario:
            raise ValueError(
                f"Insufficient logs for scenario '{name}': "
                f"need {n_per_scenario}, found {got}."
            )

    result: list[WearableLog] = []
    for name in scenario_names:
        result.extend(buckets[name])
    return result


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_single(
    log: WearableLog,
    mock: _MockSingleAgentPipeline,
    scorer: TrajectoryScorer,
) -> RunMetrics:
    """Run and score one log through the single-agent mock.

    Args:
        log: Source wearable log.
        mock: Instantiated :class:`_MockSingleAgentPipeline`.
        scorer: Instantiated :class:`~src.eval.trajectory_scorer.TrajectoryScorer`.

    Returns:
        :class:`RunMetrics` for the single-agent run.
    """
    t0 = time.monotonic()
    scored_log = mock.run(log)
    latency_ms = (time.monotonic() - t0) * 1000.0

    ts = scorer.score_trajectory(scored_log)
    final_action = scored_log.trajectory[-1].action if scored_log.trajectory else ""

    return RunMetrics(
        log_id=log.log_id,
        scenario_type=log.scenario_type.value,
        pipeline="single_agent",
        trajectory_success=ts.outcome.goal_achieved,
        tool_invocation_accuracy=ts.tool_calls.score,
        privacy_compliance=_is_privacy_compliant(final_action, log),
        latency_ms=latency_ms,
        weighted_total=ts.weighted_total,
        attribution=None,
    )


def _score_multi(
    log: WearableLog,
    pipeline: MultiAgentPipeline,
    scorer: TrajectoryScorer,
    attr_scorer: RoleAttributionScorer,
) -> RunMetrics:
    """Run and score one log through the multi-agent pipeline.

    Args:
        log: Source wearable log.
        pipeline: Instantiated :class:`MultiAgentPipeline`.
        scorer: Instantiated :class:`TrajectoryScorer`.
        attr_scorer: Instantiated :class:`RoleAttributionScorer`.

    Returns:
        :class:`RunMetrics` for the multi-agent run.
    """
    result: MultiAgentResult = pipeline.run(log)

    scored_log = dataclasses.replace(log, trajectory=result.trajectory)
    ts = scorer.score_trajectory(scored_log)
    final_action = result.final_action.value

    attribution = attr_scorer.score(
        result.role_annotations,
        goal_achieved=ts.outcome.goal_achieved,
    )

    return RunMetrics(
        log_id=log.log_id,
        scenario_type=log.scenario_type.value,
        pipeline="multi_agent",
        trajectory_success=ts.outcome.goal_achieved,
        tool_invocation_accuracy=ts.tool_calls.score,
        privacy_compliance=_is_privacy_compliant(final_action, log),
        latency_ms=result.latency_ms,
        weighted_total=ts.weighted_total,
        attribution=attribution,
    )


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------


def _winner(single: RunMetrics, multi: RunMetrics) -> str:
    """Determine the winner based on weighted_total composite score.

    Args:
        single: Single-agent :class:`RunMetrics`.
        multi: Multi-agent :class:`RunMetrics`.

    Returns:
        ``"multi_agent"``, ``"single_agent"``, or ``"tie"`` string.
    """
    if multi.weighted_total > single.weighted_total + 1e-6:
        return "multi_agent"
    if single.weighted_total > multi.weighted_total + 1e-6:
        return "single_agent"
    return "tie"


def _build_table(pairs: list[tuple[RunMetrics, RunMetrics]]) -> str:
    """Render the comparison table as a string.

    Args:
        pairs: List of (single_metrics, multi_metrics) tuples, one per log.

    Returns:
        Formatted tabulate table string ready for stdout.
    """
    headers = [
        "Scenario",
        "Log (short)",
        "Single — success/tool/priv/score",
        "Multi  — success/tool/priv/score",
        "Winner",
    ]
    rows = []
    for single, multi in pairs:

        def _fmt(m: RunMetrics) -> str:
            s = "✓" if m.trajectory_success else "✗"
            p = "✓" if m.privacy_compliance else "✗"
            tool = f"{m.tool_invocation_accuracy:.2f}"
            return f"{s} / {tool} / {p} / {m.weighted_total:.3f}"

        rows.append(
            [
                single.scenario_type,
                single.log_id[:8],
                _fmt(single),
                _fmt(multi),
                _winner(single, multi),
            ]
        )
    return str(tabulate(rows, headers=headers, tablefmt="github"))


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------


def run_comparison(
    input_path: Path = _DEFAULT_INPUT,
    output_path: Path = _DEFAULT_OUTPUT,
) -> list[dict[str, Any]]:
    """Load logs, run both pipelines, score, print table, and save JSON.

    Args:
        input_path: Path to JSONL wearable log file.
        output_path: Destination path for JSON comparison results.

    Returns:
        List of result dicts (one per log), each containing ``single_agent``
        and ``multi_agent`` sub-dicts plus a ``winner`` key.
    """
    logs = _load_logs_by_scenario(input_path, _LOGS_PER_SCENARIO)
    logger.info("Loaded %d logs (%d per scenario).", len(logs), _LOGS_PER_SCENARIO)

    mock = _MockSingleAgentPipeline()
    pipeline = MultiAgentPipeline()
    scorer = TrajectoryScorer(dry_run=True)
    attr_scorer = RoleAttributionScorer()

    pairs: list[tuple[RunMetrics, RunMetrics]] = []
    output_records: list[dict[str, Any]] = []

    for log in logs:
        single = _score_single(log, mock, scorer)
        multi = _score_multi(log, pipeline, scorer, attr_scorer)
        pairs.append((single, multi))

        output_records.append(
            {
                "log_id": log.log_id,
                "scenario_type": log.scenario_type.value,
                "consent_model": log.consent_model.value,
                "single_agent": single.to_dict(),
                "multi_agent": multi.to_dict(),
                "winner": _winner(single, multi),
            }
        )
        logger.debug(
            "log_id=%s scenario=%s winner=%s",
            log.log_id[:8],
            log.scenario_type.value,
            _winner(single, multi),
        )

    table = _build_table(pairs)
    console.print("\n" + table + "\n")

    _print_summary(output_records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results saved to %s", output_path)

    return output_records


def _print_summary(records: list[dict[str, Any]]) -> None:
    """Print aggregate win counts and mean score deltas.

    Args:
        records: List of comparison result dicts from :func:`run_comparison`.
    """
    wins: dict[str, int] = {"multi_agent": 0, "single_agent": 0, "tie": 0}
    multi_totals: list[float] = []
    single_totals: list[float] = []

    for r in records:
        wins[r["winner"]] += 1
        multi_totals.append(r["multi_agent"]["weighted_total"])
        single_totals.append(r["single_agent"]["weighted_total"])

    mean_multi = sum(multi_totals) / len(multi_totals)
    mean_single = sum(single_totals) / len(single_totals)

    console.print(
        f"Wins  — multi_agent: {wins['multi_agent']}  "
        f"single_agent: {wins['single_agent']}  tie: {wins['tie']}"
    )
    console.print(
        f"Mean weighted_total — multi: {mean_multi:.3f}  "
        f"single: {mean_single:.3f}  "
        f"delta: {mean_multi - mean_single:+.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = typer.Option(
        _DEFAULT_INPUT, "--input", help="JSONL wearable log file."
    ),
    output_path: Path = typer.Option(
        _DEFAULT_OUTPUT, "--output", help="JSON output path."
    ),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Compare single-agent mock vs multi-agent pipeline on 10 wearable logs."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
    run_comparison(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    app()
