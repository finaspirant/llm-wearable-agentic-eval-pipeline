"""Day 19 smoke test: end-to-end agentic eval pipeline.

Generates 5 synthetic wearable trajectories (variants of the canonical
5-step fixture), runs KoraiMetrics + DeepEvalJudge + FACTSGroundingScorer
on each, and writes results to data/processed/day19_smoke_test_results.jsonl.

Usage:
    uv run python scripts/run_day19_eval_smoke_test.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure repo root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.agentic_eval import (  # noqa: E402
    AgenticEvalResult,
    DeepEvalJudge,
    FACTSGroundingScorer,
    KoraiMetrics,
    compute_overall_score,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
console = Console()

OUTPUT_PATH = Path("data/processed/day19_smoke_test_results.jsonl")

# ---------------------------------------------------------------------------
# Synthetic trajectory factory
# ---------------------------------------------------------------------------

_BASE_STEPS: list[dict] = [
    {
        "agent_role": "health_monitor",
        "expected_role": "health_monitor",
        "tool_call": "read_sensor",
        "expected_tools": ["read_sensor"],
        "tool_output": "heart_rate: 92bpm",
    },
    {
        "agent_role": "privacy_gate",
        "expected_role": "privacy_gate",
        "tool_call": "check_context",
        "expected_tools": ["check_context"],
        "tool_output": "context: private_meeting",
    },
    {
        "agent_role": "privacy_gate",
        "expected_role": "privacy_gate",
        "tool_call": "suppress_alert",
        "expected_tools": ["suppress_alert"],
        "tool_output": "alert suppressed",
    },
    {
        "agent_role": "health_monitor",
        "expected_role": "health_monitor",
        "tool_call": "log_event",
        "expected_tools": ["log_event"],
        "tool_output": "logging failed",
    },
    {
        "agent_role": "health_monitor",
        "expected_role": "health_monitor",
        "tool_call": "retry_log",   # intentionally wrong tool for step 5
        "expected_tools": ["log_event"],
        "tool_output": "logged",
    },
]

# Per-trajectory goal_achieved patterns — vary to produce different success rates.
_GOAL_PATTERNS: list[list[bool]] = [
    [True,  True,  True,  False, True ],   # traj-0: 4/5 = 0.80
    [True,  True,  True,  True,  True ],   # traj-1: 5/5 = 1.00
    [True,  False, False, False, True ],   # traj-2: 2/5 = 0.40
    [False, True,  True,  True,  False],   # traj-3: 3/5 = 0.60
    [True,  True,  False, True,  True ],   # traj-4: 4/5 = 0.80
]

_TASK_DESCRIPTIONS: list[str] = [
    "Monitor heart rate during private meeting; suppress alerts if context warrants.",
    "Detect elevated heart rate and escalate via preferred notification channel.",
    "Check ambient noise levels and adjust microphone sensitivity during active call.",
    "Log wearable sensor events to cloud storage with retry on transient failures.",
    "Enforce privacy gate: suppress biometric alerts on sensitive calendar events.",
]

_SOURCE_DOCS: list[list[str]] = [
    [
        "Heart rate above 90 bpm during rest may indicate elevated stress.",
        "Private meeting context: suppress all non-critical biometric alerts.",
    ],
    [
        "Tachycardia threshold: sustained rate above 100 bpm requires notification.",
        "Preferred escalation channel: wrist haptic, then mobile push.",
    ],
    [
        "Noise levels above 85 dB may impair call quality; reduce mic gain.",
        "Ambient noise sensor: reports in dB SPL at 1 kHz reference.",
    ],
    [
        "Cloud logging endpoint: POST /api/v1/events with idempotency key.",
        "Retry policy: 3 attempts with exponential back-off, max 30 s.",
    ],
    [
        "Privacy gate rule: any calendar event tagged CONFIDENTIAL suppresses alerts.",
        "GDPR Art. 9 restricts health data processing without explicit consent.",
    ],
]

# Simulated latencies per trajectory (ms).
_LATENCIES_MS: list[float] = [3200.0, 4850.0, 6100.0, 2750.0, 9400.0]


def _build_trajectory(index: int) -> list[dict]:
    goals = _GOAL_PATTERNS[index]
    return [
        {"step": i + 1, "goal_achieved": goals[i], **step}
        for i, step in enumerate(_BASE_STEPS)
    ]


def _agent_response_for(index: int) -> str:
    return _TASK_DESCRIPTIONS[index]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_smoke_test() -> None:
    """Run the Day 19 end-to-end eval smoke test and write results."""
    metrics = KoraiMetrics()
    judge = DeepEvalJudge()
    facts = FACTSGroundingScorer()

    results: list[AgenticEvalResult] = []

    console.print("\n[bold cyan]Day 19 — Agentic Eval Smoke Test[/bold cyan]\n")

    for i in range(5):
        traj_id = f"traj-day19-{i:02d}"
        trajectory = _build_trajectory(i)
        response = _agent_response_for(i)
        source_docs = _SOURCE_DOCS[i]
        latency = _LATENCIES_MS[i]

        # 1. KoraiMetrics — all 6 scores
        success = metrics.score_trajectory_success(trajectory)
        tool_acc = metrics.score_tool_invocation(trajectory)
        privacy_leak = metrics.detect_privacy_leak(trajectory)
        orchestrator = metrics.score_orchestrator_correctness(trajectory)
        latency_sla = metrics.score_latency_sla(latency)
        groundedness = metrics.score_groundedness(response, " ".join(source_docs))

        # 2. DeepEvalJudge (graceful fallback built in)
        judge_scores = judge.judge_trajectory(trajectory, response)

        # 3. FACTSGroundingScorer
        facts_scores = facts.score(response, source_docs)

        result = AgenticEvalResult(
            trajectory_id=traj_id,
            task_id=f"task-{i:02d}",
            framework="smoke_test",
            trajectory_success_rate=success,
            tool_invocation_accuracy=tool_acc,
            groundedness_score=groundedness,
            privacy_leak_detected=privacy_leak,
            orchestrator_correctness=orchestrator,
            latency_sla_compliance=latency_sla,
            overall_score=0.0,
            eval_timestamp=datetime.now(UTC).isoformat(),
        )
        result.overall_score = compute_overall_score(result)
        results.append(result)

        logger.debug(
            "%s | judge=%s | facts_overall=%.3f",
            traj_id,
            judge_scores,
            facts_scores["overall_facts_score"],
        )

    # ---------------------------------------------------------------------------
    # Rich table
    # ---------------------------------------------------------------------------
    table = Table(
        title="Day 19 Smoke Test — Kore.ai Eval Results",
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("trajectory_id", style="white")
    table.add_column("success", justify="right")
    table.add_column("tool_acc", justify="right")
    table.add_column("groundedness", justify="right")
    table.add_column("privacy_ok", justify="center")
    table.add_column("orchestrator", justify="right")
    table.add_column("latency", justify="right")
    table.add_column("overall", justify="right", style="bold cyan")

    for r in results:
        table.add_row(
            r.trajectory_id,
            f"{r.trajectory_success_rate:.2f}",
            f"{r.tool_invocation_accuracy:.2f}",
            f"{r.groundedness_score:.2f}",
            "[green]✓[/green]" if not r.privacy_leak_detected else "[red]LEAK[/red]",
            f"{r.orchestrator_correctness:.2f}",
            f"{r.latency_sla_compliance:.2f}",
            f"{r.overall_score:.3f}",
        )

    console.print(table)

    # ---------------------------------------------------------------------------
    # Write JSONL
    # ---------------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")

    console.print(
        f"\n[green]✓[/green] Wrote {len(results)} results → [bold]{OUTPUT_PATH}[/bold]"
    )
    print(str(OUTPUT_PATH.resolve()))


if __name__ == "__main__":
    run_smoke_test()
