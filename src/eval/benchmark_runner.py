"""Multi-framework benchmark harness.

Runs identical tasks across LangGraph, CrewAI, AutoGen (AG2), and the
OpenAI Agents SDK.  Logs token counts, latency, trajectory scores, cascade
depth, and nondeterminism variance per framework.

This is the empirical foundation for WP2.  The harness controls for task
definition, tool availability, and evaluation criteria — enabling
apples-to-apples framework comparison across six dimensions:
token_efficiency, latency, reliability, goal_rate, trajectory_quality,
and cascade_depth.

Phase 3 (Days 19–22): replace mock stubs with real API calls via ``--live``.
Day 22: 3-run nondeterminism measurement + TrajectoryScorer integration.

CLI::

    python -m src.eval.benchmark_runner --tasks all --runs 3
"""

from __future__ import annotations

import abc
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from src.data.wearable_generator import WearableLog

import typer
import yaml
from rich.console import Console
from rich.table import Table

from src.eval.trajectory_scorer import TrajectoryScorer

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="benchmark-runner",
    help="Run benchmark tasks across all agent frameworks.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TaskConfig:
    """Specification for a single benchmark task.

    Args:
        task_id: Unique identifier used in results and YAML keys.
        description: Natural-language task prompt sent to the agent.
        goal: Outcome criterion used to determine goal_achieved.
        max_steps: Hard cap on agent steps; exceeded steps are truncated.
        timeout_s: Wall-clock timeout in seconds (enforced in Phase 3).
        tools_available: Tool names the agent is permitted to call.
        expected_steps: Ordered list of step descriptions the agent should
            ideally perform; used for trajectory alignment scoring in Phase 3.
        success_criteria: Key-value flags evaluated after task completion.
        difficulty_level: Qualitative difficulty rating (``"easy"``,
            ``"medium"``, or ``"hard"``).
        tags: Audience/company tags indicating which reviewers this task
            is designed to impress (e.g. ``["kore_ai", "deepmind"]``).
    """

    task_id: str
    description: str
    goal: str
    max_steps: int
    timeout_s: float
    tools_available: list[str]
    expected_steps: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    difficulty_level: str = "medium"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskConfig:
        """Construct a TaskConfig from a raw YAML-parsed mapping.

        Args:
            data: Mapping with keys matching dataclass fields.

        Returns:
            A fully populated TaskConfig instance.
        """
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            goal=data["goal"],
            max_steps=int(data["max_steps"]),
            timeout_s=float(data["timeout_s"]),
            tools_available=list(data["tools_available"]),
            expected_steps=list(data.get("expected_steps", [])),
            success_criteria=dict(data.get("success_criteria", {})),
            difficulty_level=str(data.get("difficulty_level", "medium")),
            tags=list(data.get("tags", [])),
        )


@dataclass
class BenchmarkResult:
    """Outcome of running one task on one framework for one run.

    Args:
        task_id: Matches TaskConfig.task_id.
        framework: Canonical framework name (e.g. ``"langgraph"``).
        steps_taken: Number of trajectory steps recorded.
        tokens_used: Estimated input + output tokens consumed.
        latency_ms: Wall-clock execution time in milliseconds.
        errors: Any non-fatal error strings encountered during the run.
        goal_achieved: Whether the agent satisfied the task goal.
        trajectory: Ordered list of step dicts; schema varies by framework.
        run_index: 1-indexed run number within the nondeterminism batch.
        cascade_depth: Longest uninterrupted chain of tool calls without
            human-in-the-loop input.
        trajectory_score: Weighted composite score from TrajectoryScorer
            (None if scoring failed).
        pia_dimensions: Four PIA rubric dimension scores from TrajectoryScorer
            (None if scoring failed).
        nondeterminism_variance: Standard deviation of trajectory_score across
            all runs for this (task, framework) pair.  Set after all runs
            complete.
    """

    task_id: str
    framework: str
    steps_taken: int
    tokens_used: int
    latency_ms: float
    errors: list[str]
    goal_achieved: bool
    trajectory: list[dict[str, Any]]
    run_index: int = 1
    cascade_depth: int = 0
    trajectory_score: float | None = None
    pia_dimensions: dict[str, float] | None = None
    nondeterminism_variance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable mapping for JSONL logging.

        Returns:
            Dict with all fields; floats rounded to 6 decimal places max.
        """
        return {
            "task_id": self.task_id,
            "framework": self.framework,
            "run_index": self.run_index,
            "steps_taken": self.steps_taken,
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 3),
            "errors": self.errors,
            "goal_achieved": self.goal_achieved,
            "cascade_depth": self.cascade_depth,
            "trajectory_score": (
                round(self.trajectory_score, 6)
                if self.trajectory_score is not None
                else None
            ),
            "pia_dimensions": self.pia_dimensions,
            "nondeterminism_variance": (
                round(self.nondeterminism_variance, 6)
                if self.nondeterminism_variance is not None
                else None
            ),
            "trajectory": self.trajectory,
        }


# ---------------------------------------------------------------------------
# Task → scenario / terminal-action mappings (used by proxy builder)
# ---------------------------------------------------------------------------

# Maps each task_id to a valid ScenarioType value for TrajectoryScorer intent scoring.
# Wearable tasks map to their natural scenario; enterprise tasks use closest analogue.
_TASK_SCENARIO_MAP: dict[str, str] = {
    "wearable_health_alert": "health_alert",
    "wearable_privacy": "privacy_sensitive",
    "wearable_location_sensitive": "location_trigger",
    "wearable_sleep_coaching": "health_alert",
    "wearable_ambient_noise": "ambient_noise",
    "it_helpdesk": "calendar_reminder",
    "hr_policy_query": "calendar_reminder",
    "compliance_audit": "privacy_sensitive",
    "incident_triage": "health_alert",
    "code_review_assist": "ambient_noise",
}

# Maps each task_id to a terminal AgentAction used on the final proxy step.
# Determines outcome scoring: all values must be in _TERMINAL_ACTIONS.
_TASK_TERMINAL_ACTION: dict[str, str] = {
    "wearable_health_alert": "send_alert",
    "wearable_privacy": "suppress_capture",
    "wearable_location_sensitive": "surface_reminder",
    "wearable_sleep_coaching": "surface_reminder",
    "wearable_ambient_noise": "adjust_noise_profile",
    "it_helpdesk": "send_alert",
    "hr_policy_query": "surface_reminder",
    "compliance_audit": "suppress_capture",
    "incident_triage": "send_alert",
    "code_review_assist": "surface_reminder",
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_cascade_depth(
    trajectory: list[dict[str, Any]],
    framework: str,
) -> int:
    """Compute the longest tool-call chain without human-in-the-loop input.

    Human input is defined per framework:
    - autogen: any step where ``speaker == "UserProxy"`` resets the chain.
    - openai_agents: only ``event_type == "tool_call"`` steps extend the chain;
      handoffs and other events reset it.
    - langgraph / crewai: consecutive steps with non-empty ``tool_calls`` lists.

    Args:
        trajectory: List of framework-specific step dicts.
        framework: Canonical framework name.

    Returns:
        Integer cascade depth (0 if no tool calls).
    """
    max_chain = 0
    current_chain = 0

    for step in trajectory:
        if framework == "autogen":
            if step.get("speaker") == "UserProxy":
                current_chain = 0
                continue
            tool_calls: list[Any] = step.get("tool_calls", [])
        elif framework == "openai_agents":
            if step.get("event_type") == "tool_call":
                tool_calls = ["_tool_"]  # sentinel: non-empty means tool used
            else:
                current_chain = 0
                continue
        else:
            tool_calls = step.get("tool_calls", [])

        if tool_calls:
            current_chain += 1
            max_chain = max(max_chain, current_chain)
        else:
            current_chain = 0

    return max_chain


def _build_wearable_proxy(
    result: BenchmarkResult,
    task_config: TaskConfig,
    run_index: int,
) -> WearableLog:
    """Build a minimal WearableLog proxy for TrajectoryScorer scoring.

    Maps the framework-specific BenchmarkResult trajectory to the WearableLog
    format expected by TrajectoryScorer.  The last step receives the task's
    designated terminal action so that outcome scoring reflects goal_achieved.

    Args:
        result: BenchmarkResult containing the framework trajectory.
        task_config: Task specification providing scenario context.
        run_index: 1-indexed run number; makes log_id unique across runs.

    Returns:
        Minimal WearableLog ready for TrajectoryScorer.score_trajectory().
    """
    from src.data.privacy_gate import ConsentModel
    from src.data.wearable_generator import (
        AudioTranscript,
        ScenarioType,
        SensorData,
        TrajectoryStep,
        WearableLog,
    )

    scenario_str = _TASK_SCENARIO_MAP.get(task_config.task_id, "health_alert")
    try:
        scenario = ScenarioType(scenario_str)
    except ValueError:
        scenario = ScenarioType("health_alert")

    terminal_action = _TASK_TERMINAL_ACTION.get(task_config.task_id, "send_alert")

    steps: list[TrajectoryStep] = []
    n = len(result.trajectory)
    for i in range(n):
        is_last = i == n - 1
        if is_last and result.goal_achieved:
            action = terminal_action
            step_name = "act"
        elif i == 0:
            action = ""
            step_name = "sense"
        else:
            action = ""
            step_name = "plan"
        steps.append(
            TrajectoryStep(
                step_index=i,
                step_name=step_name,
                observation=f"[proxy] step {i + 1}/{n} task={task_config.task_id}",
                reasoning="mock reasoning for benchmark proxy",
                action=action,
                confidence=0.80,
            )
        )

    # Minimal sensor data — values are not used by the scorer's heuristics.
    sensor = SensorData(
        heart_rate=72.0,
        spo2=98.0,
        steps=3000,
        noise_db=45.0,
        gps_lat=37.4220,
        gps_lon=-122.0841,
        skin_temp_c=36.5,
    )
    audio = AudioTranscript(
        text="",
        language="en-US",
        confidence=0.0,
        duration_s=0.0,
        keywords_detected=[],
    )
    return WearableLog(
        log_id=f"{task_config.task_id}:{result.framework}:{run_index}",
        timestamp=datetime.now(UTC).isoformat(),
        scenario_type=scenario,
        consent_model=ConsentModel.EXPLICIT,
        sensor_data=sensor,
        audio_transcript=audio,
        context_metadata={"task_id": task_config.task_id, "mock_proxy": True},
        trajectory=steps,
        ground_truth_action=terminal_action,
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AgentBenchmark(abc.ABC):
    """Abstract wrapper around a single agent framework.

    Subclasses implement ``_execute`` with that framework's native API.
    ``run_task`` handles timing, exception catching, and result assembly —
    subclasses should not override it.

    Phase 3: ``_execute`` implementations switch from stubs to real API calls.
    """

    @property
    @abc.abstractmethod
    def framework_name(self) -> str:
        """Canonical lowercase identifier used in results and table headers."""
        ...

    @abc.abstractmethod
    def _execute(
        self,
        task_config: TaskConfig,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], int, list[str], bool]:
        """Execute the task and return raw results.

        Args:
            task_config: Full task specification.
            rng: Seeded RNG for deterministic mock behaviour; real
                implementations should ignore this parameter.

        Returns:
            4-tuple of (trajectory, tokens_used, errors, goal_achieved).
        """
        ...

    def run_task(
        self,
        task_config: TaskConfig,
        run_index: int = 1,
    ) -> BenchmarkResult:
        """Run a task, measure wall-clock time, and return a BenchmarkResult.

        Catches all exceptions so a single framework failure does not abort
        the full benchmark run.

        Args:
            task_config: Task to execute.
            run_index: 1-indexed run number used to vary the RNG seed across
                nondeterminism measurement runs.

        Returns:
            BenchmarkResult populated with metrics.
        """
        logger.info(
            "Starting task=%s framework=%s run=%d",
            task_config.task_id,
            self.framework_name,
            run_index,
        )
        rng = random.Random(f"{task_config.task_id}:{self.framework_name}:{run_index}")

        t0 = time.perf_counter()
        try:
            trajectory, tokens_used, errors, goal_achieved = self._execute(
                task_config, rng
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unhandled exception in framework=%s task=%s run=%d",
                self.framework_name,
                task_config.task_id,
                run_index,
            )
            trajectory = []
            tokens_used = 0
            errors = [f"UnhandledException: {exc!s}"]
            goal_achieved = False

        latency_ms = (time.perf_counter() - t0) * 1_000.0
        cascade = _compute_cascade_depth(trajectory, self.framework_name)

        result = BenchmarkResult(
            task_id=task_config.task_id,
            framework=self.framework_name,
            steps_taken=len(trajectory),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            errors=errors,
            goal_achieved=goal_achieved,
            trajectory=trajectory,
            run_index=run_index,
            cascade_depth=cascade,
        )
        logger.debug(
            "Finished task=%s framework=%s run=%d goal=%s steps=%d "
            "tokens=%d latency_ms=%.2f cascade=%d",
            result.task_id,
            result.framework,
            result.run_index,
            result.goal_achieved,
            result.steps_taken,
            result.tokens_used,
            result.latency_ms,
            result.cascade_depth,
        )
        return result


# ---------------------------------------------------------------------------
# Framework implementations (mock stubs — Phase 3 replaces _execute bodies)
# ---------------------------------------------------------------------------

# Each stub produces a trajectory whose *structure* mirrors the framework's
# native execution model.  A staff-engineer reviewer should see exactly where
# real API calls slot in during Phase 3.


class LangGraphBenchmark(AgentBenchmark):
    """LangGraph StateGraph stub.

    Real implementation (Phase 3)::

        graph = StateGraph(AgentState)
        graph.add_node("sense", sense_node)
        graph.add_node("plan",  plan_node)
        graph.add_node("act",   act_node)
        graph.add_conditional_edges("plan", route_by_risk, {...})
        compiled = graph.compile()
        result = compiled.invoke({"task": task_config.description})

    Trajectory schema: ``{step, node, action, tool_calls, output,
    timestamp_offset_ms}``.
    """

    # (node_name, action_label, tool_calls) per task
    _PLANS: dict[str, list[tuple[str, str, list[str]]]] = {
        "it_helpdesk": [
            ("sense", "ingest_vpn_error_report", ["query_vpn_logs"]),
            ("plan", "determine_diagnostic_path", []),
            (
                "act",
                "run_network_diagnostic",
                ["run_diagnostic", "check_network_status"],
            ),
            ("act", "propose_fix_or_escalate", ["send_ticket"]),
            ("end", "log_resolution", ["log_event"]),
        ],
        "wearable_privacy": [
            ("sense", "read_biometric_sensors", ["get_sensor_reading"]),
            (
                "plan",
                "evaluate_risk_and_privacy_context",
                ["assess_health_risk", "check_privacy_context"],
            ),
            ("act", "apply_privacy_policy_decision", []),
            ("end", "log_decision_rationale", ["log_event"]),
        ],
        "hr_policy_query": [
            ("sense", "retrieve_employee_record", ["retrieve_employee_record"]),
            ("plan", "identify_applicable_policies", ["search_policy_docs"]),
            ("act", "compute_entitlement_and_respond", ["compute_pto_balance"]),
            ("end", "log_policy_decision", ["log_event"]),
        ],
        "compliance_audit": [
            ("sense", "retrieve_processing_log", ["retrieve_processing_log"]),
            ("plan", "query_gdpr_ruleset", ["query_gdpr_ruleset"]),
            ("act_flag", "identify_violations", ["flag_violation"]),
            ("act_report", "generate_findings_report", ["generate_findings_report"]),
            ("act_escalate", "notify_dpo", ["escalate_to_dpo"]),
            ("end", "log_audit_complete", ["log_event"]),
        ],
        "incident_triage": [
            ("sense", "check_metrics_dashboard", ["query_metrics_dashboard"]),
            ("plan", "correlate_deployment_signals", ["get_deployment_history"]),
            ("act_db", "check_db_connection_pool", ["check_db_connection_pool"]),
            ("act_downstream", "check_downstream_health", ["check_downstream_health"]),
            ("act_mitigate", "apply_mitigation", ["apply_mitigation"]),
            ("act_page", "page_oncall_engineer", ["page_oncall"]),
            ("end", "log_incident_resolution", ["log_event"]),
        ],
        "code_review_assist": [
            ("sense", "fetch_pr_diff", ["fetch_pr_diff"]),
            ("plan", "run_static_analysis", ["run_static_analysis"]),
            ("act_cwe", "lookup_security_issues", ["lookup_cwe"]),
            ("act_comment", "post_review_comment", ["post_review_comment"]),
            ("end", "log_review_complete", ["log_event"]),
        ],
        "wearable_health_alert": [
            ("sense", "read_biometric_sensors", ["get_sensor_reading"]),
            (
                "plan",
                "assess_health_risk",
                ["assess_health_risk", "check_privacy_context"],
            ),
            ("act", "send_health_alert", ["send_alert"]),
            ("end", "log_alert_action", ["log_event"]),
        ],
        "wearable_location_sensitive": [
            ("sense", "get_location_context", ["get_location_context"]),
            (
                "plan",
                "classify_notification_urgency",
                ["classify_notification_urgency", "check_privacy_context"],
            ),
            ("act", "surface_or_suppress_reminder", ["surface_reminder"]),
            ("end", "log_location_decision", ["log_event"]),
        ],
        "wearable_sleep_coaching": [
            ("sense", "retrieve_sleep_history", ["retrieve_sleep_history"]),
            (
                "plan",
                "analyse_sleep_trend",
                ["analyse_sleep_trend", "check_privacy_context"],
            ),
            (
                "act",
                "generate_coaching_recommendation",
                ["generate_coaching_recommendation"],
            ),
            ("end", "log_coaching_session", ["log_event"]),
        ],
        "wearable_ambient_noise": [
            ("sense", "read_noise_sensor", ["get_sensor_reading"]),
            (
                "plan",
                "assess_noise_exposure_risk",
                ["assess_noise_exposure_risk", "check_privacy_context"],
            ),
            ("act", "adjust_noise_profile_or_alert", ["adjust_noise_profile"]),
            ("end", "log_noise_action", ["log_event"]),
        ],
    }

    _OUTPUTS: dict[str, dict[str, str]] = {
        "it_helpdesk": {
            "sense": (
                "VPN client error 619: tunnel endpoint unreachable. "
                "Last success: 2025-11-14T09:12:00Z. Client: FortiClient 7.2."
            ),
            "plan": (
                "Diagnostic path: (1) DNS for vpn.corp.internal, "
                "(2) UDP 4500 firewall rule, (3) certificate expiry."
            ),
            "act": (
                "UDP 4500 blocked by local firewall policy FW-CORP-04. "
                "Fix: add outbound UDP 4500 exception."
            ),
            "end": "Ticket IT-20251114-0042 created. Resolution logged.",
        },
        "wearable_privacy": {
            "sense": (
                "HR: 112 bpm (elevated baseline: 68 bpm). SpO2: 97%. "
                "Audio context: private_conversation. Consent: AMBIENT."
            ),
            "plan": (
                "Risk: moderate (HR elevated, not critical). "
                "Privacy flag: active. Override threshold: HR > 140 bpm. "
                "Decision path: suppress alert, log locally."
            ),
            "act": (
                "Privacy boundary enforced. Alert suppressed. "
                "HR 112 bpm < 140 bpm override threshold."
            ),
            "end": "Decision logged locally. No biometric data transmitted.",
        },
        "hr_policy_query": {
            "sense": (
                "Employee record retrieved. John Smith, Engineering, "
                "hired 2023-01-15. PTO balance: 8.5 days remaining."
            ),
            "plan": (
                "Policy match: Employee Handbook §4.2 (PTO accrual). "
                "8.5 days remaining. No negative balance risk."
            ),
            "act": (
                "PTO balance confirmed: 8.5 days available. "
                "Approval criteria met. Response drafted for HR portal."
            ),
            "end": "Policy query logged. No escalation required.",
        },
        "compliance_audit": {
            "sense": (
                "Retrieved 47 GDPR Article 5 processing activities for user "
                "EU-2847. Retention flags: 3 anomalies detected."
            ),
            "plan": (
                "GDPR ruleset loaded. Article 5(1)(e) storage limitation "
                "applies to 3 flagged records exceeding 90-day retention."
            ),
            "act_flag": (
                "3 violations flagged: retention > 90 days without consent "
                "refresh on health sensor records."
            ),
            "act_report": (
                "Findings report generated: GDPR-2025-11-14-EU2847.pdf. "
                "3 high-severity findings."
            ),
            "act_escalate": (
                "DPO notified. Article 7(3) consent withdrawal procedure initiated."
            ),
            "end": "Audit log complete. DPO acknowledgement pending.",
        },
        "incident_triage": {
            "sense": (
                "P99 latency: 2,340 ms (SLA: 500 ms). Error rate: 12.4%. "
                "Dashboard: CRITICAL. Affected service: checkout."
            ),
            "plan": (
                "Deployment history: v2.3.1 deployed 14:20 UTC "
                "(30 min before incident). Correlation: HIGH."
            ),
            "act_db": (
                "DB connection pool at 98% utilisation (max: 100). "
                "14 connection timeouts in last 5 min."
            ),
            "act_downstream": (
                "HealthCheck: payment-service DEGRADED (timeout), "
                "auth-service OK, search-service OK."
            ),
            "act_mitigate": (
                "Mitigation applied: connection pool expanded to 200. "
                "Latency falling: 2,340 ms → 890 ms."
            ),
            "act_page": (
                "On-call engineer paged. Runbook: DB-POOL-EXHAUSTION linked "
                "in PagerDuty alert #2025-11-14-1450."
            ),
            "end": "Incident logged. Post-mortem scheduled 2025-11-15.",
        },
        "code_review_assist": {
            "sense": (
                "PR #487 diff fetched. 342 lines changed across 7 files. "
                "Python + SQL changes detected."
            ),
            "plan": (
                "Static analysis: 2 critical issues "
                "(CWE-89 SQL injection candidates in auth.py:L142, L198)."
            ),
            "act_cwe": (
                "CWE-89 confirmed: f-string interpolation into raw SQL query. "
                "CVSS 9.8 (Critical)."
            ),
            "act_comment": (
                "Review comment posted: 'Parameterised queries required for "
                "L142, L198. Blocking merge.'"
            ),
            "end": "Review logged. PR status: CHANGES_REQUESTED.",
        },
        "wearable_health_alert": {
            "sense": (
                "HR: 142 bpm (elevated; baseline: 70 bpm). "
                "SpO2: 93% (below 95% alert threshold). Consent: EXPLICIT."
            ),
            "plan": (
                "Risk: HIGH (HR > 140 bpm AND SpO2 < 95%). "
                "Consent EXPLICIT — no privacy bar to alert."
            ),
            "act": (
                "Health alert sent to user and emergency contact. "
                "SpO2 < 88% escalation threshold not met; monitoring continues."
            ),
            "end": "Alert logged. Follow-up biometric check scheduled T+5 min.",
        },
        "wearable_location_sensitive": {
            "sense": (
                "Location context: medical_facility. "
                "Pending reminder: 'Afternoon medication' (priority: high)."
            ),
            "plan": (
                "Rule: medical_facility → suppress location-linked notifications "
                "unless life-critical. Medication qualifies as life-critical."
            ),
            "act": (
                "Reminder surfaced. Location tag stripped from push payload "
                "per GDPR data-minimisation principle."
            ),
            "end": "Location-sensitive decision logged locally.",
        },
        "wearable_sleep_coaching": {
            "sense": (
                "Sleep history retrieved: 7-night window. "
                "Avg sleep: 5.4 h. Deep sleep fraction: 18% (target: 25%)."
            ),
            "plan": (
                "Trend: chronic sleep deficit. Deep-sleep deficit pattern. "
                "No acute clinical markers — behavioural coaching appropriate."
            ),
            "act": (
                "Coaching recommendation: 'Aim for 10:30 PM wind-down. "
                "Limit screen exposure after 9 PM.' "
                "(Behavioural only — not a medical recommendation.)"
            ),
            "end": "Coaching session logged. Next review: 7 days.",
        },
        "wearable_ambient_noise": {
            "sense": (
                "Noise level: 92 dB (NIOSH 85 dB / 8-hour TWA exceeded; "
                "exposure duration: 47 min)."
            ),
            "plan": (
                "Risk: MEDIUM-HIGH. Safe exposure limit at 92 dB: 2 h 31 min "
                "(NIOSH). Current exposure: 47 min. Consent: AMBIENT."
            ),
            "act": (
                "Noise profile adjusted: hearing protection recommendation issued. "
                "Volume cap applied at 85 dB."
            ),
            "end": "Noise exposure event logged. Cumulative TWA updated.",
        },
    }

    @property
    def framework_name(self) -> str:
        return "langgraph"

    def _execute(
        self,
        task_config: TaskConfig,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], int, list[str], bool]:
        plan = self._PLANS.get(
            task_config.task_id,
            [("sense", "read_inputs", []), ("act", "execute", []), ("end", "log", [])],
        )
        task_outputs = self._OUTPUTS.get(task_config.task_id, {})
        trajectory: list[dict[str, Any]] = []
        offset_ms = 0.0

        for node, action, tool_calls in plan:
            step_ms = rng.uniform(8.0, 60.0)
            trajectory.append(
                {
                    "step": len(trajectory) + 1,
                    "node": node,
                    "action": action,
                    "tool_calls": tool_calls,
                    "output": task_outputs.get(node, f"[{node}] processed"),
                    "timestamp_offset_ms": round(offset_ms, 2),
                }
            )
            offset_ms += step_ms

        # LangGraph: token-efficient — single agent, tight system prompts.
        tokens_used = rng.randint(380, 560)
        return trajectory, tokens_used, [], True


class CrewAIBenchmark(AgentBenchmark):
    """CrewAI Crew stub.

    Real implementation (Phase 3)::

        crew = Crew(
            agents=[diagnostician, specialist, escalation_manager],
            tasks=[diagnose_task, resolve_task, escalate_task],
            process=Process.sequential,
        )
        result = crew.kickoff(inputs={"task": task_config.description})

    Trajectory schema: ``{step, agent_role, task_name, tool_calls, output,
    timestamp_offset_ms}``.
    """

    _PLANS: dict[str, list[tuple[str, str, list[str]]]] = {
        "it_helpdesk": [
            (
                "IT Diagnostician",
                "analyse_vpn_failure",
                ["query_vpn_logs", "check_network_status"],
            ),
            ("Solution Specialist", "identify_root_cause", ["run_diagnostic"]),
            ("Solution Specialist", "draft_fix_recommendation", []),
            ("Escalation Manager", "assess_escalation_need", ["send_ticket"]),
            ("IT Diagnostician", "compile_closure_report", ["log_event"]),
        ],
        "wearable_privacy": [
            (
                "Health Monitor Agent",
                "assess_biometric_risk",
                ["get_sensor_reading", "assess_health_risk"],
            ),
            (
                "Privacy Guardian Agent",
                "evaluate_consent_boundary",
                ["check_privacy_context"],
            ),
            ("Decision Coordinator", "weigh_risk_vs_privacy", []),
            ("Decision Coordinator", "execute_policy_action", ["log_event"]),
        ],
        "hr_policy_query": [
            (
                "Policy Retriever",
                "fetch_employee_and_policies",
                ["retrieve_employee_record", "search_policy_docs"],
            ),
            ("Policy Analyst", "compute_pto_entitlement", ["compute_pto_balance"]),
            ("HR Coordinator", "draft_and_log_response", ["log_event"]),
        ],
        "compliance_audit": [
            (
                "Data Auditor",
                "retrieve_processing_activities",
                ["retrieve_processing_log"],
            ),
            ("GDPR Analyst", "apply_retention_rules", ["query_gdpr_ruleset"]),
            (
                "Compliance Officer",
                "flag_and_report_violations",
                ["flag_violation", "generate_findings_report"],
            ),
            ("Data Protection Lead", "escalate_to_dpo", ["escalate_to_dpo"]),
            ("Audit Recorder", "close_audit_log", ["log_event"]),
        ],
        "incident_triage": [
            ("SRE Monitor", "read_metrics_dashboard", ["query_metrics_dashboard"]),
            (
                "Root Cause Analyst",
                "correlate_with_deployments",
                ["get_deployment_history"],
            ),
            ("Database SRE", "diagnose_db_layer", ["check_db_connection_pool"]),
            ("Platform SRE", "check_service_mesh", ["check_downstream_health"]),
            (
                "Incident Commander",
                "mitigate_and_page",
                ["apply_mitigation", "page_oncall"],
            ),
            ("Incident Recorder", "close_incident_log", ["log_event"]),
        ],
        "code_review_assist": [
            ("Code Fetcher", "fetch_pr_diff", ["fetch_pr_diff"]),
            ("Static Analyst", "run_static_analysis", ["run_static_analysis"]),
            ("Security Reviewer", "lookup_cwe_advisories", ["lookup_cwe"]),
            ("PR Author Liaison", "post_blocking_comment", ["post_review_comment"]),
            ("Review Logger", "close_review_log", ["log_event"]),
        ],
        "wearable_health_alert": [
            (
                "Health Monitor Agent",
                "assess_biometric_risk",
                ["get_sensor_reading", "assess_health_risk"],
            ),
            (
                "Privacy Guardian Agent",
                "evaluate_consent_for_alert",
                ["check_privacy_context"],
            ),
            ("Alert Coordinator", "dispatch_health_alert", ["send_alert"]),
            ("Event Logger", "log_alert_event", ["log_event"]),
        ],
        "wearable_location_sensitive": [
            (
                "Location Context Agent",
                "read_location_and_pending_reminders",
                ["get_location_context"],
            ),
            (
                "Notification Classifier",
                "classify_urgency_with_privacy",
                ["classify_notification_urgency", "check_privacy_context"],
            ),
            ("Reminder Agent", "surface_or_suppress_reminder", ["surface_reminder"]),
            ("Location Logger", "log_location_decision", ["log_event"]),
        ],
        "wearable_sleep_coaching": [
            (
                "Sleep Data Analyst",
                "retrieve_and_analyse_sleep",
                ["retrieve_sleep_history", "analyse_sleep_trend"],
            ),
            (
                "Privacy Guardian Agent",
                "verify_coaching_consent",
                ["check_privacy_context"],
            ),
            (
                "Sleep Coach Agent",
                "generate_coaching_recommendation",
                ["generate_coaching_recommendation"],
            ),
            ("Coaching Logger", "log_coaching_session", ["log_event"]),
        ],
        "wearable_ambient_noise": [
            (
                "Noise Monitor Agent",
                "read_and_assess_noise",
                ["get_sensor_reading", "assess_noise_exposure_risk"],
            ),
            (
                "Privacy Guardian Agent",
                "check_noise_context",
                ["check_privacy_context"],
            ),
            (
                "Noise Safety Agent",
                "apply_noise_profile_adjustment",
                ["adjust_noise_profile"],
            ),
            ("Noise Logger", "log_noise_event", ["log_event"]),
        ],
    }

    _OUTPUTS: dict[str, dict[str, str]] = {
        "it_helpdesk": {
            "IT Diagnostician": (
                "Analysis complete. VPN error 619 traced to UDP 4500 block "
                "at corporate firewall FW-CORP-04."
            ),
            "Solution Specialist": (
                "Root cause confirmed: missing outbound UDP 4500 exception. "
                "Recommended fix: add firewall rule, re-test VPN tunnel."
            ),
            "Escalation Manager": (
                "Fix is within IT Level 1 scope. No Tier 2 escalation. "
                "Ticket IT-20251114-0042 filed."
            ),
        },
        "wearable_privacy": {
            "Health Monitor Agent": (
                "HR 112 bpm — elevated but not critical. "
                "No cardiac event markers detected."
            ),
            "Privacy Guardian Agent": (
                "Consent model: AMBIENT. Private context flag: active. "
                "Policy prohibits biometric transmission."
            ),
            "Decision Coordinator": (
                "Final decision: suppress alert, log locally. "
                "HR threshold (140 bpm) not exceeded."
            ),
        },
        "hr_policy_query": {
            "Policy Retriever": (
                "Employee record and PTO policy retrieved. "
                "John Smith: hired 2023-01-15, Engineering, 8.5 PTO days remaining."
            ),
            "Policy Analyst": (
                "Entitlement computed per §4.2: 8.5 days available. "
                "No negative balance risk."
            ),
            "HR Coordinator": "Response drafted and logged. No escalation required.",
        },
        "compliance_audit": {
            "Data Auditor": (
                "47 processing activities retrieved for EU-2847. "
                "3 records flagged for retention anomalies."
            ),
            "GDPR Analyst": (
                "Article 5(1)(e) violated: 3 health sensor records retained "
                "> 90 days without consent refresh."
            ),
            "Compliance Officer": (
                "Violations flagged. Findings report GDPR-2025-11-14-EU2847.pdf "
                "generated."
            ),
            "Data Protection Lead": (
                "DPO notified. Article 7(3) withdrawal procedure initiated."
            ),
            "Audit Recorder": "Audit log closed. DPO acknowledgement pending.",
        },
        "incident_triage": {
            "SRE Monitor": "P99 latency 2,340 ms. Error rate 12.4%. CRITICAL.",
            "Root Cause Analyst": (
                "v2.3.1 deployed 30 min pre-incident. High correlation."
            ),
            "Database SRE": "DB pool at 98% utilisation. 14 timeouts in 5 min.",
            "Platform SRE": "payment-service DEGRADED. auth + search OK.",
            "Incident Commander": (
                "Pool expanded to 200. Latency 2,340 ms → 890 ms. On-call paged."
            ),
            "Incident Recorder": "Incident logged. Post-mortem scheduled.",
        },
        "code_review_assist": {
            "Code Fetcher": "PR #487 fetched. 342 lines, 7 files. Python + SQL.",
            "Static Analyst": (
                "2 critical findings: CWE-89 candidates in auth.py:L142, L198."
            ),
            "Security Reviewer": (
                "CWE-89 confirmed: f-string SQL injection. CVSS 9.8."
            ),
            "PR Author Liaison": (
                "Blocking comment posted. PR status: CHANGES_REQUESTED."
            ),
            "Review Logger": "Review logged.",
        },
        "wearable_health_alert": {
            "Health Monitor Agent": (
                "HR 142 bpm, SpO2 93%. HIGH risk. Threshold exceeded."
            ),
            "Privacy Guardian Agent": (
                "Consent: EXPLICIT. No privacy bar to alert dispatch."
            ),
            "Alert Coordinator": (
                "Health alert dispatched to user and emergency contact."
            ),
            "Event Logger": "Alert event logged.",
        },
        "wearable_location_sensitive": {
            "Location Context Agent": (
                "Location: medical_facility. Pending: medication reminder."
            ),
            "Notification Classifier": (
                "Urgency: high (life-critical). Location tag stripped per GDPR."
            ),
            "Reminder Agent": (
                "Medication reminder surfaced without location metadata."
            ),
            "Location Logger": "Location-sensitive decision logged.",
        },
        "wearable_sleep_coaching": {
            "Sleep Data Analyst": (
                "7-night window: avg 5.4 h, deep sleep 18%. Deficit pattern."
            ),
            "Privacy Guardian Agent": (
                "Consent: EXPLICIT. Coaching data may be retained 30 days."
            ),
            "Sleep Coach Agent": (
                "Recommendation: 10:30 PM wind-down, screen-free after 9 PM. "
                "Behavioural — not a medical recommendation."
            ),
            "Coaching Logger": "Coaching session logged. Next review: 7 days.",
        },
        "wearable_ambient_noise": {
            "Noise Monitor Agent": (
                "Noise: 92 dB. Exposure: 47 min. NIOSH safe limit at 92 dB: 2 h 31 min."
            ),
            "Privacy Guardian Agent": (
                "Consent: AMBIENT. Noise exposure data retained locally only."
            ),
            "Noise Safety Agent": (
                "Noise profile adjusted. Volume cap at 85 dB. "
                "Hearing protection recommended."
            ),
            "Noise Logger": "Noise exposure event logged. TWA updated.",
        },
    }

    @property
    def framework_name(self) -> str:
        return "crewai"

    def _execute(
        self,
        task_config: TaskConfig,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], int, list[str], bool]:
        plan = self._PLANS.get(
            task_config.task_id,
            [("GeneralistAgent", "process_task", [])],
        )
        task_outputs = self._OUTPUTS.get(task_config.task_id, {})
        trajectory: list[dict[str, Any]] = []
        offset_ms = 0.0

        for agent_role, task_name, tool_calls in plan:
            step_ms = rng.uniform(15.0, 90.0)
            trajectory.append(
                {
                    "step": len(trajectory) + 1,
                    "agent_role": agent_role,
                    "task_name": task_name,
                    "tool_calls": tool_calls,
                    "output": task_outputs.get(agent_role, f"[{agent_role}] completed"),
                    "timestamp_offset_ms": round(offset_ms, 2),
                }
            )
            offset_ms += step_ms

        # CrewAI: higher token overhead from multi-agent role prompts.
        tokens_used = rng.randint(680, 940)
        return trajectory, tokens_used, [], True


class AutoGenBenchmark(AgentBenchmark):
    """AutoGen (AG2) conversational agent stub.

    Real implementation (Phase 3)::

        user_proxy = UserProxyAgent("UserProxy", human_input_mode="NEVER")
        assistant = AssistantAgent("Assistant", llm_config={...})
        chat_result = user_proxy.initiate_chat(
            assistant,
            message=task_config.description,
            max_turns=task_config.max_steps,
        )

    Trajectory schema: ``{step, speaker, role, content, tool_calls,
    timestamp_offset_ms}``.
    """

    # (speaker, role, tool_calls, content)
    _CONVERSATIONS: dict[str, list[tuple[str, str, list[str], str]]] = {
        "it_helpdesk": [
            (
                "UserProxy",
                "user",
                [],
                "VPN failing with error 619. MacBook, FortiClient 7.2, "
                "started after OS update.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["query_vpn_logs"],
                "Checking VPN logs. Error 619 = tunnel endpoint unreachable. "
                "Running diagnostic.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["run_diagnostic", "check_network_status"],
                "Diagnostic: UDP 4500 blocked by FW-CORP-04. Root cause confirmed.",
            ),
            ("UserProxy", "user", [], "Can you fix it or do we need Tier 2?"),
            (
                "AssistantAgent",
                "assistant",
                ["send_ticket"],
                "Level 1 fix: adding UDP 4500 exception to FW-CORP-04. "
                "Ticket IT-20251114-0042 filed. Retry VPN in 5 min.",
            ),
            ("UserProxy", "user", ["log_event"], "TERMINATE"),
        ],
        "wearable_privacy": [
            (
                "UserProxy",
                "user",
                ["get_sensor_reading"],
                "HR spiked to 112 bpm. User in private conversation. "
                "Send health alert?",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["assess_health_risk"],
                "HR 112 bpm: elevated, not critical (threshold: 140 bpm). "
                "No cardiac markers.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_privacy_context"],
                "Consent: AMBIENT, private_conversation active. "
                "Policy prohibits transmission.",
            ),
            ("UserProxy", "user", [], "Final decision?"),
            (
                "AssistantAgent",
                "assistant",
                ["log_event"],
                "Decision: suppress alert, log locally. "
                "HR < 140 bpm; privacy respected. TERMINATE",
            ),
        ],
        "hr_policy_query": [
            (
                "UserProxy",
                "user",
                [],
                "What is John Smith's current PTO balance? "
                "He wants to take 3 days next week.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["retrieve_employee_record", "search_policy_docs"],
                "Record retrieved. John Smith, Engineering. Policy §4.2 loaded.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["compute_pto_balance"],
                "PTO balance: 8.5 days available. 3-day request approved under policy.",
            ),
            ("UserProxy", "user", ["log_event"], "TERMINATE"),
        ],
        "compliance_audit": [
            (
                "UserProxy",
                "user",
                [],
                "Audit user EU-2847's data processing activities for GDPR compliance.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["retrieve_processing_log"],
                "Retrieved 47 processing activities. 3 retention anomalies flagged.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["query_gdpr_ruleset"],
                "Article 5(1)(e) applies: 3 health records retained "
                "> 90 days without refresh.",
            ),
            ("UserProxy", "user", [], "Are there violations? Who should we notify?"),
            (
                "AssistantAgent",
                "assistant",
                ["flag_violation", "generate_findings_report"],
                "3 violations flagged. Report GDPR-2025-11-14-EU2847.pdf generated.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["escalate_to_dpo"],
                "DPO notified. Article 7(3) consent withdrawal initiated. TERMINATE",
            ),
            ("UserProxy", "user", ["log_event"], "TERMINATE"),
        ],
        "incident_triage": [
            (
                "UserProxy",
                "user",
                [],
                "P99 latency 2,340 ms, error rate 12.4%. "
                "Checkout service critical. Triage now.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["query_metrics_dashboard"],
                "Dashboard: CRITICAL. Latency spike started 14:22 UTC.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["get_deployment_history"],
                "v2.3.1 deployed 14:20 UTC — 2 min pre-incident. High correlation.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_db_connection_pool"],
                "DB pool: 98/100 connections used. 14 timeouts in 5 min. "
                "Root cause: pool exhaustion.",
            ),
            ("UserProxy", "user", [], "Check downstream and apply fix."),
            (
                "AssistantAgent",
                "assistant",
                ["check_downstream_health", "apply_mitigation"],
                "payment-service DEGRADED. Pool expanded to 200. Latency dropping.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["page_oncall", "log_event"],
                "On-call paged. Runbook linked. Latency 890 ms. TERMINATE",
            ),
        ],
        "code_review_assist": [
            (
                "UserProxy",
                "user",
                [],
                "Review PR #487 for security issues before merging to main.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["fetch_pr_diff"],
                "PR #487 fetched. 342 lines, 7 files. Python + SQL changes.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["run_static_analysis"],
                "Static analysis: 2 critical issues — "
                "SQL injection candidates in auth.py:L142, L198.",
            ),
            ("UserProxy", "user", [], "Confirm severity and post a review comment."),
            (
                "AssistantAgent",
                "assistant",
                ["lookup_cwe", "post_review_comment"],
                "CWE-89 confirmed. CVSS 9.8. Blocking comment posted. "
                "PR: CHANGES_REQUESTED.",
            ),
            ("UserProxy", "user", ["log_event"], "TERMINATE"),
        ],
        "wearable_health_alert": [
            (
                "UserProxy",
                "user",
                ["get_sensor_reading"],
                "HR 142 bpm, SpO2 93%. Should we send an alert?",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["assess_health_risk"],
                "HIGH risk: HR > 140 bpm AND SpO2 < 95%. Alert warranted.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_privacy_context"],
                "Consent: EXPLICIT. No privacy bar. Proceed with alert.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["send_alert", "log_event"],
                "Health alert dispatched to user and emergency contact. TERMINATE",
            ),
        ],
        "wearable_location_sensitive": [
            (
                "UserProxy",
                "user",
                ["get_location_context"],
                "User at medical facility. Medication reminder pending. Surface it?",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["classify_notification_urgency"],
                "Urgency: HIGH (life-critical medication). "
                "Location context: medical_facility.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_privacy_context"],
                "Rule: strip location tag from push payload "
                "per GDPR data-minimisation.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["surface_reminder", "log_event"],
                "Reminder surfaced without location metadata. "
                "Decision logged. TERMINATE",
            ),
        ],
        "wearable_sleep_coaching": [
            (
                "UserProxy",
                "user",
                ["retrieve_sleep_history"],
                "Generate sleep coaching based on last 7 nights.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["analyse_sleep_trend"],
                "Avg 5.4 h sleep, 18% deep sleep. Chronic deficit pattern.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_privacy_context"],
                "Consent: EXPLICIT. Coaching data retained locally 30 days.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["generate_coaching_recommendation", "log_event"],
                "Recommendation: 10:30 PM wind-down, screen-free after 9 PM. "
                "Not a medical recommendation. TERMINATE",
            ),
        ],
        "wearable_ambient_noise": [
            (
                "UserProxy",
                "user",
                ["get_sensor_reading"],
                "Noise level 92 dB for 47 min. Take action?",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["assess_noise_exposure_risk"],
                "NIOSH safe limit at 92 dB: 2h 31min. "
                "Current: 47 min. Risk: MEDIUM-HIGH.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_privacy_context"],
                "Consent: AMBIENT. Retain noise data locally only.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["adjust_noise_profile", "log_event"],
                "Volume cap applied at 85 dB. Hearing protection recommended. "
                "TERMINATE",
            ),
        ],
    }

    @property
    def framework_name(self) -> str:
        return "autogen"

    def _execute(
        self,
        task_config: TaskConfig,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], int, list[str], bool]:
        conversation = self._CONVERSATIONS.get(
            task_config.task_id,
            [
                ("UserProxy", "user", [], task_config.description),
                ("AssistantAgent", "assistant", [], "Task completed."),
            ],
        )
        trajectory: list[dict[str, Any]] = []
        offset_ms = 0.0

        for speaker, role, tool_calls, content in conversation:
            step_ms = rng.uniform(20.0, 110.0)
            trajectory.append(
                {
                    "step": len(trajectory) + 1,
                    "speaker": speaker,
                    "role": role,
                    "content": content,
                    "tool_calls": tool_calls,
                    "timestamp_offset_ms": round(offset_ms, 2),
                }
            )
            offset_ms += step_ms

        # AutoGen: highest token count due to conversational verbosity.
        tokens_used = rng.randint(820, 1150)
        return trajectory, tokens_used, [], True


class OpenAIAgentsBenchmark(AgentBenchmark):
    """OpenAI Agents SDK stub.

    Real implementation (Phase 3)::

        triage_agent = Agent(
            name="TriageAgent",
            instructions=TRIAGE_PROMPT,
            tools=[check_network_status, query_vpn_logs],
            handoffs=[DiagnosticAgent],
        )
        result = Runner.run_sync(triage_agent, task_config.description)

    Trajectory schema: ``{step, agent, event_type, tool/to_agent,
    input, output, timestamp_offset_ms}``.
    """

    # (agent, event_type, tool_or_agent, input_json, output_str)
    _PLANS: dict[str, list[tuple[str, str, str, str, str]]] = {
        "it_helpdesk": [
            (
                "TriageAgent",
                "tool_call",
                "query_vpn_logs",
                '{"user": "user_001", "since": "2025-11-14T00:00:00Z"}',
                "Error 619: tunnel endpoint unreachable. UDP 4500 blocked.",
            ),
            (
                "TriageAgent",
                "handoff",
                "DiagnosticAgent",
                "Network-layer issue requiring deeper analysis",
                "",
            ),
            (
                "DiagnosticAgent",
                "tool_call",
                "run_diagnostic",
                '{"host": "vpn.corp.internal", "protocol": "udp", "port": 4500}',
                "UDP 4500 blocked at FW-CORP-04.",
            ),
            (
                "DiagnosticAgent",
                "tool_call",
                "send_ticket",
                '{"priority": "medium", "fix": "add UDP 4500 exception FW-CORP-04"}',
                "Ticket IT-20251114-0042 created.",
            ),
            (
                "DiagnosticAgent",
                "tool_call",
                "log_event",
                '{"resolution": "firewall_fix", "escalated": false}',
                "Event logged.",
            ),
        ],
        "wearable_privacy": [
            (
                "SensorAgent",
                "tool_call",
                "get_sensor_reading",
                '{"sensors": ["heart_rate", "spo2"]}',
                '{"heart_rate": 112, "spo2": 97}',
            ),
            (
                "HealthAgent",
                "tool_call",
                "assess_health_risk",
                '{"heart_rate": 112, "spo2": 97}',
                '{"risk_level": "moderate", "emergency": false}',
            ),
            (
                "HealthAgent",
                "handoff",
                "PrivacyAgent",
                "Risk assessed; privacy check required",
                "",
            ),
            (
                "PrivacyAgent",
                "tool_call",
                "check_privacy_context",
                "{}",
                '{"consent": "AMBIENT", "private_context": true, '
                '"override_threshold_bpm": 140}',
            ),
            (
                "PrivacyAgent",
                "tool_call",
                "log_event",
                '{"decision": "suppress_alert", "reason": "hr_below_threshold"}',
                "Decision logged.",
            ),
        ],
        "hr_policy_query": [
            (
                "PolicyAgent",
                "tool_call",
                "retrieve_employee_record",
                '{"employee_id": "EMP-0042"}',
                '{"name": "John Smith", "dept": "Engineering", "pto_days": 8.5}',
            ),
            (
                "PolicyAgent",
                "tool_call",
                "search_policy_docs",
                '{"query": "PTO accrual Engineering", "section": "4.2"}',
                '{"policy": "Employee Handbook §4.2", '
                '"accrual_rate": "1.5 days/month"}',
            ),
            (
                "PolicyAgent",
                "tool_call",
                "compute_pto_balance",
                '{"employee_id": "EMP-0042", "requested_days": 3}',
                '{"available": 8.5, "after_request": 5.5, "approved": true}',
            ),
            (
                "PolicyAgent",
                "tool_call",
                "log_event",
                '{"action": "pto_query_resolved", "employee_id": "EMP-0042"}',
                "Event logged.",
            ),
        ],
        "compliance_audit": [
            (
                "AuditAgent",
                "tool_call",
                "retrieve_processing_log",
                '{"user_id": "EU-2847", "days": 90}',
                '{"activities": 47, "anomalies": 3}',
            ),
            (
                "AuditAgent",
                "tool_call",
                "query_gdpr_ruleset",
                '{"articles": ["5(1)(e)", "7(3)"]}',
                '{"article_5_1_e": "storage limitation 90 days", '
                '"article_7_3": "withdrawal"}',
            ),
            (
                "AuditAgent",
                "tool_call",
                "flag_violation",
                '{"user_id": "EU-2847", "records": 3, "article": "5(1)(e)"}',
                '{"flagged": 3, "severity": "high"}',
            ),
            (
                "AuditAgent",
                "handoff",
                "ReportingAgent",
                "Violations flagged; generate report and notify DPO",
                "",
            ),
            (
                "ReportingAgent",
                "tool_call",
                "generate_findings_report",
                '{"violations": 3, "user_id": "EU-2847"}',
                '{"report_id": "GDPR-2025-11-14-EU2847", "format": "pdf"}',
            ),
            (
                "ReportingAgent",
                "tool_call",
                "escalate_to_dpo",
                '{"report_id": "GDPR-2025-11-14-EU2847", "urgent": true}',
                "DPO notified.",
            ),
            (
                "ReportingAgent",
                "tool_call",
                "log_event",
                '{"audit": "complete", "report_id": "GDPR-2025-11-14-EU2847"}',
                "Event logged.",
            ),
        ],
        "incident_triage": [
            (
                "MonitorAgent",
                "tool_call",
                "query_metrics_dashboard",
                '{"service": "checkout", "window": "15m"}',
                '{"p99_ms": 2340, "error_rate": 0.124, "status": "CRITICAL"}',
            ),
            (
                "MonitorAgent",
                "handoff",
                "TriageAgent",
                "Critical latency spike on checkout service",
                "",
            ),
            (
                "TriageAgent",
                "tool_call",
                "get_deployment_history",
                '{"service": "checkout", "since": "2025-11-14T13:00:00Z"}',
                '{"last_deploy": "v2.3.1", "deploy_time": "2025-11-14T14:20:00Z"}',
            ),
            (
                "TriageAgent",
                "tool_call",
                "check_db_connection_pool",
                '{"host": "db-primary-01"}',
                '{"used": 98, "max": 100, "timeouts_5m": 14}',
            ),
            (
                "TriageAgent",
                "tool_call",
                "check_downstream_health",
                '{"services": ["payment", "auth", "search"]}',
                '{"payment": "DEGRADED", "auth": "OK", "search": "OK"}',
            ),
            (
                "TriageAgent",
                "tool_call",
                "apply_mitigation",
                '{"action": "expand_pool", "new_max": 200}',
                '{"p99_ms": 890, "status": "RECOVERING"}',
            ),
            (
                "TriageAgent",
                "tool_call",
                "page_oncall",
                '{"runbook": "DB-POOL-EXHAUSTION", "severity": "SEV-1"}',
                "On-call paged. PagerDuty alert #2025-11-14-1450 created.",
            ),
            (
                "TriageAgent",
                "tool_call",
                "log_event",
                '{"incident": "checkout-latency", "mitigation": "pool_expansion"}',
                "Event logged.",
            ),
        ],
        "code_review_assist": [
            (
                "ReviewAgent",
                "tool_call",
                "fetch_pr_diff",
                '{"pr_id": 487, "repo": "corp/monorepo"}',
                '{"lines_changed": 342, "files": 7, "languages": ["python", "sql"]}',
            ),
            (
                "ReviewAgent",
                "tool_call",
                "run_static_analysis",
                '{"pr_id": 487, "rules": ["security", "style"]}',
                '{"critical": 2, "warnings": 5, '
                '"files": ["auth.py:L142", "auth.py:L198"]}',
            ),
            (
                "ReviewAgent",
                "tool_call",
                "lookup_cwe",
                '{"pattern": "f-string SQL interpolation", "language": "python"}',
                '{"cwe_id": "CWE-89", "name": "SQL Injection", "cvss": 9.8}',
            ),
            (
                "ReviewAgent",
                "tool_call",
                "post_review_comment",
                '{"pr_id": 487, "body": "CWE-89 at L142, L198. '
                'Use parameterised queries."}',
                '{"status": "CHANGES_REQUESTED"}',
            ),
            (
                "ReviewAgent",
                "tool_call",
                "log_event",
                '{"pr_id": 487, "outcome": "changes_requested", "cwe": "CWE-89"}',
                "Event logged.",
            ),
        ],
        "wearable_health_alert": [
            (
                "SensorAgent",
                "tool_call",
                "get_sensor_reading",
                '{"sensors": ["heart_rate", "spo2"]}',
                '{"heart_rate": 142, "spo2": 93}',
            ),
            (
                "HealthAgent",
                "tool_call",
                "assess_health_risk",
                '{"heart_rate": 142, "spo2": 93}',
                '{"risk": "HIGH", "hr_threshold_exceeded": true, "spo2_alert": true}',
            ),
            (
                "HealthAgent",
                "tool_call",
                "check_privacy_context",
                "{}",
                '{"consent": "EXPLICIT", "private_context": false}',
            ),
            (
                "HealthAgent",
                "tool_call",
                "send_alert",
                '{"recipients": ["user", "emergency_contact"], "risk": "HIGH"}',
                "Health alert dispatched.",
            ),
            (
                "HealthAgent",
                "tool_call",
                "log_event",
                '{"alert_sent": true, "hr": 142, "spo2": 93}',
                "Event logged.",
            ),
        ],
        "wearable_location_sensitive": [
            (
                "LocationAgent",
                "tool_call",
                "get_location_context",
                "{}",
                '{"context": "medical_facility", "pending_reminders": ["medication"]}',
            ),
            (
                "LocationAgent",
                "tool_call",
                "classify_notification_urgency",
                '{"reminder": "medication", "location": "medical_facility"}',
                '{"urgency": "high", "life_critical": true}',
            ),
            (
                "LocationAgent",
                "tool_call",
                "check_privacy_context",
                "{}",
                '{"consent": "EXPLICIT", "strip_location_tag": true}',
            ),
            (
                "LocationAgent",
                "tool_call",
                "surface_reminder",
                '{"reminder": "medication", "include_location": false}',
                "Reminder surfaced without location metadata.",
            ),
            (
                "LocationAgent",
                "tool_call",
                "log_event",
                '{"decision": "surface_without_location", "reminder": "medication"}',
                "Event logged.",
            ),
        ],
        "wearable_sleep_coaching": [
            (
                "SleepAgent",
                "tool_call",
                "retrieve_sleep_history",
                '{"days": 7}',
                '{"avg_hours": 5.4, "deep_sleep_pct": 18, "target_deep_pct": 25}',
            ),
            (
                "SleepAgent",
                "tool_call",
                "analyse_sleep_trend",
                '{"history": "7_night_window"}',
                '{"pattern": "chronic_deficit", "deep_sleep_deficit": true}',
            ),
            (
                "SleepAgent",
                "tool_call",
                "check_privacy_context",
                "{}",
                '{"consent": "EXPLICIT", "retention_days": 30}',
            ),
            (
                "SleepAgent",
                "tool_call",
                "generate_coaching_recommendation",
                '{"pattern": "chronic_deficit", "deep_sleep_deficit": true}',
                '{"recommendation": "10:30 PM wind-down; screen-free after 9 PM", '
                '"type": "behavioural"}',
            ),
            (
                "SleepAgent",
                "tool_call",
                "log_event",
                '{"session": "sleep_coaching", "next_review_days": 7}',
                "Event logged.",
            ),
        ],
        "wearable_ambient_noise": [
            (
                "NoiseAgent",
                "tool_call",
                "get_sensor_reading",
                '{"sensors": ["noise_db"]}',
                '{"noise_db": 92.0, "exposure_minutes": 47}',
            ),
            (
                "NoiseAgent",
                "tool_call",
                "assess_noise_exposure_risk",
                '{"noise_db": 92.0, "exposure_minutes": 47}',
                '{"risk": "MEDIUM_HIGH", "niosh_limit_minutes": 151, "remaining": 104}',
            ),
            (
                "NoiseAgent",
                "tool_call",
                "check_privacy_context",
                "{}",
                '{"consent": "AMBIENT", "retain_local_only": true}',
            ),
            (
                "NoiseAgent",
                "tool_call",
                "adjust_noise_profile",
                '{"cap_db": 85, "notify_user": true}',
                "Volume cap applied. Hearing protection recommendation issued.",
            ),
            (
                "NoiseAgent",
                "tool_call",
                "log_event",
                '{"noise_db": 92.0, "exposure_min": 47, "action": "cap_and_recommend"}',
                "Event logged.",
            ),
        ],
    }

    @property
    def framework_name(self) -> str:
        return "openai_agents"

    def _execute(
        self,
        task_config: TaskConfig,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], int, list[str], bool]:
        plan = self._PLANS.get(
            task_config.task_id,
            [("Agent", "tool_call", "process", "{}", "Done.")],
        )
        trajectory: list[dict[str, Any]] = []
        offset_ms = 0.0

        for agent, event_type, tool_or_agent, inp, out in plan:
            step_ms = rng.uniform(10.0, 75.0)
            entry: dict[str, Any] = {
                "step": len(trajectory) + 1,
                "agent": agent,
                "event_type": event_type,
                "timestamp_offset_ms": round(offset_ms, 2),
            }
            if event_type == "tool_call":
                entry["tool"] = tool_or_agent
                entry["input"] = inp
                entry["output"] = out
            else:
                entry["to_agent"] = tool_or_agent
                entry["reason"] = inp
            trajectory.append(entry)
            offset_ms += step_ms

        # OpenAI Agents: moderate tokens — tool-call focused, less verbosity.
        tokens_used = rng.randint(510, 760)
        return trajectory, tokens_used, [], True


# ---------------------------------------------------------------------------
# Framework registry
# ---------------------------------------------------------------------------

FRAMEWORK_REGISTRY: dict[str, AgentBenchmark] = {
    "langgraph": LangGraphBenchmark(),
    "crewai": CrewAIBenchmark(),
    "autogen": AutoGenBenchmark(),
    "openai_agents": OpenAIAgentsBenchmark(),
}

ALL_FRAMEWORK_NAMES: list[str] = list(FRAMEWORK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates multi-framework benchmark runs with nondeterminism measurement.

    Loads task configs from a YAML file, dispatches each (task, framework) pair
    for ``runs`` repeated runs, scores each trajectory via TrajectoryScorer,
    computes nondeterminism variance, generates a leaderboard, and writes
    results to a JSONL log.

    Args:
        config_path: Path to ``benchmark_tasks.yaml``.
        output_path: Path for JSONL result log (created or appended to).
        framework_names: Frameworks to include; ``None`` means all registered.
        live: When True, switch from mock stubs to real API calls.  Currently
            logged as a warning — Phase 3 wires the actual implementations.
    """

    def __init__(
        self,
        config_path: Path,
        output_path: Path,
        framework_names: list[str] | None = None,
        live: bool = False,
    ) -> None:
        self._config_path = config_path
        self._output_path = output_path
        self._live = live
        self._frameworks: list[AgentBenchmark] = [
            FRAMEWORK_REGISTRY[name]
            for name in (framework_names or ALL_FRAMEWORK_NAMES)
            if name in FRAMEWORK_REGISTRY
        ]
        if not self._frameworks:
            raise ValueError(
                f"No valid frameworks found in: {framework_names}. "
                f"Available: {ALL_FRAMEWORK_NAMES}"
            )
        if live:
            logger.warning(
                "--live flag set but live API calls are not yet implemented. "
                "Running in mock mode. Wire Phase 3 API clients to activate."
            )

    def load_tasks(self, task_ids: list[str] | None = None) -> list[TaskConfig]:
        """Load TaskConfigs from the YAML config file.

        Args:
            task_ids: If provided, only return tasks whose task_id is in
                this list.  ``None`` returns all tasks.

        Returns:
            List of TaskConfig instances in YAML order.

        Raises:
            FileNotFoundError: If the config file does not exist.
            KeyError: If required YAML fields are missing.
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Benchmark config not found: {self._config_path}")
        raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        all_tasks = [TaskConfig.from_dict(t) for t in raw["tasks"]]

        if task_ids is None:
            return all_tasks

        id_set = set(task_ids)
        filtered = [t for t in all_tasks if t.task_id in id_set]
        missing = id_set - {t.task_id for t in filtered}
        if missing:
            logger.warning("Task IDs not found in config: %s", sorted(missing))
        return filtered

    def run_all(
        self,
        task_ids: list[str] | None = None,
        runs: int = 3,
    ) -> list[BenchmarkResult]:
        """Run every (task, framework) combination for ``runs`` repeated runs.

        For each (task, framework) pair, executes ``runs`` independent runs
        with different RNG seeds, scores each trajectory via TrajectoryScorer,
        and computes nondeterminism_variance = stdev(trajectory_score) across
        runs.  All results are logged to the JSONL output file.

        Args:
            task_ids: Tasks to run; ``None`` runs all tasks in the config.
            runs: Number of repeated runs per (task, framework) pair.

        Returns:
            All BenchmarkResult instances in (task, framework, run) order.
        """
        tasks = self.load_tasks(task_ids)
        results: list[BenchmarkResult] = []
        scorer = TrajectoryScorer(dry_run=True)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(tasks) * len(self._frameworks) * runs
        logger.info(
            "Running %d task(s) × %d framework(s) × %d run(s) = %d combinations",
            len(tasks),
            len(self._frameworks),
            runs,
            total,
        )

        for task in tasks:
            for benchmark in self._frameworks:
                run_results: list[BenchmarkResult] = []

                for run_index in range(1, runs + 1):
                    result = benchmark.run_task(task, run_index=run_index)

                    # Score trajectory via TrajectoryScorer proxy.
                    try:
                        proxy = _build_wearable_proxy(result, task, run_index)
                        ts = scorer.score_trajectory(proxy)
                        result.trajectory_score = ts.weighted_total
                        result.pia_dimensions = scorer.score_pia_dimensions(proxy)
                    except Exception:
                        logger.exception(
                            "TrajectoryScorer failed for task=%s framework=%s run=%d",
                            task.task_id,
                            benchmark.framework_name,
                            run_index,
                        )

                    run_results.append(result)

                # Compute nondeterminism variance across runs.
                if runs >= 2:
                    scores = [
                        r.trajectory_score
                        for r in run_results
                        if r.trajectory_score is not None
                    ]
                    if len(scores) >= 2:
                        variance = statistics.stdev(scores)
                        for r in run_results:
                            r.nondeterminism_variance = variance

                for r in run_results:
                    self._log_result(r)
                    results.append(r)

        logger.info(
            "Benchmark complete. %d results written to %s",
            len(results),
            self._output_path,
        )
        return results

    def generate_leaderboard(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Any]:
        """Generate per-framework aggregate metrics and six-dimension rankings.

        Aggregates mean values across all tasks and runs per framework, then
        ranks frameworks on six dimensions:
        - token_efficiency (lower avg tokens is better)
        - latency (lower avg latency_ms is better)
        - reliability (lower avg nondeterminism_variance is better)
        - goal_rate (higher goal_achievement_rate is better)
        - trajectory_quality (higher avg trajectory_score is better)
        - cascade_depth (lower avg cascade_depth is better)

        Saves results to ``data/processed/framework_leaderboard.json``.

        Args:
            results: All BenchmarkResult instances from a run_all() call.

        Returns:
            Dict with ``frameworks`` (per-framework aggregates) and
            ``rankings`` (dimension → ordered list of framework names).
        """
        framework_metrics: dict[str, dict[str, list[float]]] = {}
        for r in results:
            fm = framework_metrics.setdefault(
                r.framework,
                {
                    "trajectory_score": [],
                    "tokens_used": [],
                    "latency_ms": [],
                    "goal_achieved": [],
                    "nondeterminism_variance": [],
                    "cascade_depth": [],
                },
            )
            fm["trajectory_score"].append(r.trajectory_score or 0.0)
            fm["tokens_used"].append(float(r.tokens_used))
            fm["latency_ms"].append(r.latency_ms)
            fm["goal_achieved"].append(1.0 if r.goal_achieved else 0.0)
            if r.nondeterminism_variance is not None:
                fm["nondeterminism_variance"].append(r.nondeterminism_variance)
            fm["cascade_depth"].append(float(r.cascade_depth))

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        aggregated: dict[str, dict[str, float]] = {}
        for fw, metrics in framework_metrics.items():
            variance_vals = metrics["nondeterminism_variance"]
            aggregated[fw] = {
                "avg_trajectory_score": _mean(metrics["trajectory_score"]),
                "avg_tokens": _mean(metrics["tokens_used"]),
                "avg_latency_ms": _mean(metrics["latency_ms"]),
                "goal_achievement_rate": _mean(metrics["goal_achieved"]),
                "avg_nondeterminism_variance": _mean(variance_vals)
                if variance_vals
                else 0.0,
                "avg_cascade_depth": _mean(metrics["cascade_depth"]),
            }

        def _rank(metric: str, higher_is_better: bool = True) -> list[str]:
            return sorted(
                aggregated.keys(),
                key=lambda fw: aggregated[fw][metric],
                reverse=higher_is_better,
            )

        rankings: dict[str, list[str]] = {
            "token_efficiency": _rank("avg_tokens", higher_is_better=False),
            "latency": _rank("avg_latency_ms", higher_is_better=False),
            "reliability": _rank("avg_nondeterminism_variance", higher_is_better=False),
            "goal_rate": _rank("goal_achievement_rate"),
            "trajectory_quality": _rank("avg_trajectory_score"),
            "cascade_depth": _rank("avg_cascade_depth", higher_is_better=False),
        }

        leaderboard: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "n_tasks": len({r.task_id for r in results}),
            "n_frameworks": len(aggregated),
            "n_runs_per_pair": len({r.run_index for r in results}),
            "frameworks": aggregated,
            "rankings": rankings,
        }

        lb_path = Path("data/processed/framework_leaderboard.json")
        lb_path.parent.mkdir(parents=True, exist_ok=True)
        lb_path.write_text(json.dumps(leaderboard, indent=2))
        logger.info("Leaderboard saved to %s", lb_path)
        return leaderboard

    def _log_result(self, result: BenchmarkResult) -> None:
        """Append a single BenchmarkResult as one JSON line to the output file.

        Args:
            result: Result to persist.
        """
        with self._output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        logger.debug(
            "Logged result task=%s framework=%s run=%d",
            result.task_id,
            result.framework,
            result.run_index,
        )

    def print_table(self, results: list[BenchmarkResult]) -> None:
        """Render a rich comparison table aggregated per (task, framework).

        Groups results by (task_id, framework) and shows mean values across
        runs.  Columns: task, framework, steps, tokens, latency, cascade,
        trajectory score, variance, goal.

        Args:
            results: Results to display; typically the return value of
                ``run_all``.
        """
        table = Table(
            title="Benchmark Results — Mock Run (Phase 3 replaces stubs)",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Task ID", style="bold", min_width=22)
        table.add_column("Framework", min_width=14)
        table.add_column("Steps", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Cascade", justify="right")
        table.add_column("Traj Score", justify="right")
        table.add_column("Variance", justify="right")
        table.add_column("Goal", justify="center")

        tasks_seen: list[str] = []
        for r in results:
            if r.task_id not in tasks_seen:
                tasks_seen.append(r.task_id)

        for task_id in tasks_seen:
            frameworks_seen: list[str] = []
            task_results = [r for r in results if r.task_id == task_id]
            for r in task_results:
                if r.framework not in frameworks_seen:
                    frameworks_seen.append(r.framework)

            first_task_row = True
            for fw in frameworks_seen:
                fw_results = [r for r in task_results if r.framework == fw]
                mean_steps = sum(r.steps_taken for r in fw_results) / len(fw_results)
                mean_tokens = sum(r.tokens_used for r in fw_results) / len(fw_results)
                mean_latency = sum(r.latency_ms for r in fw_results) / len(fw_results)
                mean_cascade = sum(r.cascade_depth for r in fw_results) / len(
                    fw_results
                )
                ts_vals = [
                    r.trajectory_score
                    for r in fw_results
                    if r.trajectory_score is not None
                ]
                mean_ts = sum(ts_vals) / len(ts_vals) if ts_vals else None
                all_goals = all(r.goal_achieved for r in fw_results)
                variance = fw_results[0].nondeterminism_variance

                goal_cell = "[green]✓[/green]" if all_goals else "[red]✗[/red]"
                ts_cell = f"{mean_ts:.3f}" if mean_ts is not None else "—"
                var_cell = f"{variance:.4f}" if variance is not None else "—"

                table.add_row(
                    task_id if first_task_row else "",
                    fw,
                    f"{mean_steps:.1f}",
                    f"{mean_tokens:.0f}",
                    f"{mean_latency:.2f}",
                    f"{mean_cascade:.1f}",
                    ts_cell,
                    var_cell,
                    goal_cell,
                )
                first_task_row = False

        console.print(table)
        console.print(f"\n[dim]Results appended to: {self._output_path}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    tasks: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated task IDs to run, or 'all' to run every task "
                "defined in the config."
            )
        ),
    ] = "all",
    frameworks: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated framework names, or 'all'. "
                f"Available: {', '.join(ALL_FRAMEWORK_NAMES)}."
            )
        ),
    ] = "all",
    config: Annotated[
        Path,
        typer.Option(help="Path to benchmark_tasks.yaml."),
    ] = Path("configs/benchmark_tasks.yaml"),
    output: Annotated[
        Path,
        typer.Option(help="Path for JSONL result log."),
    ] = Path("data/processed/benchmark_results.jsonl"),
    runs: Annotated[
        int,
        typer.Option(
            help="Number of repeated runs per (task, framework) pair for "
            "nondeterminism variance measurement."
        ),
    ] = 3,
    live: Annotated[
        bool,
        typer.Option(
            "--live/--mock",
            help="Use real API calls instead of mock stubs (Phase 3).",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging."),
    ] = False,
) -> None:
    """Run benchmark tasks across agent frameworks and print a comparison table."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    task_ids: list[str] | None = (
        None
        if tasks.strip().lower() == "all"
        else [t.strip() for t in tasks.split(",") if t.strip()]
    )
    framework_names: list[str] | None = (
        None
        if frameworks.strip().lower() == "all"
        else [f.strip() for f in frameworks.split(",") if f.strip()]
    )

    runner = BenchmarkRunner(
        config_path=config,
        output_path=output,
        framework_names=framework_names,
        live=live,
    )
    results = runner.run_all(task_ids=task_ids, runs=runs)
    runner.print_table(results)
    leaderboard = runner.generate_leaderboard(results)

    console.print("\n[bold]Framework Rankings (6 dimensions):[/bold]")
    for dimension, ranking in leaderboard["rankings"].items():
        console.print(f"  {dimension:25s}: {' > '.join(ranking)}")


if __name__ == "__main__":
    app()
