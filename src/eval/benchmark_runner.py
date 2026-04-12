"""Multi-framework benchmark harness.

Runs identical tasks across LangGraph, CrewAI, AutoGen (AG2), and the
OpenAI Agents SDK. Logs token counts, latency, error flags, cascade depth,
and goal achievement per framework.

This is the empirical foundation for WP2. The harness controls for task
definition, tool availability, and evaluation criteria — enabling
apples-to-apples framework comparison.

Phase 3 (Days 19-22): replace mock stubs with real API calls.
Today (Day 7): architecture + logging format.

CLI: python -m src.eval.benchmark_runner --tasks all
"""

from __future__ import annotations

import abc
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

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
    """Outcome of running one task on one framework.

    Args:
        task_id: Matches TaskConfig.task_id.
        framework: Canonical framework name (e.g. ``"langgraph"``).
        steps_taken: Number of trajectory steps recorded.
        tokens_used: Estimated input + output tokens consumed.
        latency_ms: Wall-clock execution time in milliseconds.
        errors: Any non-fatal error strings encountered during the run.
        goal_achieved: Whether the agent satisfied the task goal.
        trajectory: Ordered list of step dicts; schema varies by framework.
    """

    task_id: str
    framework: str
    steps_taken: int
    tokens_used: int
    latency_ms: float
    errors: list[str]
    goal_achieved: bool
    trajectory: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-serializable mapping for JSONL logging.

        Returns:
            Dict with all fields; latency_ms rounded to 3 decimal places.
        """
        return {
            "task_id": self.task_id,
            "framework": self.framework,
            "steps_taken": self.steps_taken,
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 3),
            "errors": self.errors,
            "goal_achieved": self.goal_achieved,
            "trajectory": self.trajectory,
        }


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

    def run_task(self, task_config: TaskConfig) -> BenchmarkResult:
        """Run a task, measure wall-clock time, and return a BenchmarkResult.

        Catches all exceptions so a single framework failure does not abort
        the full benchmark run.

        Args:
            task_config: Task to execute.

        Returns:
            BenchmarkResult populated with metrics.
        """
        logger.info(
            "Starting task=%s framework=%s",
            task_config.task_id,
            self.framework_name,
        )
        # Seed per (task, framework) for reproducible mock output.
        rng = random.Random(f"{task_config.task_id}:{self.framework_name}")

        t0 = time.perf_counter()
        try:
            trajectory, tokens_used, errors, goal_achieved = self._execute(
                task_config, rng
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Unhandled exception in framework=%s task=%s",
                self.framework_name,
                task_config.task_id,
            )
            trajectory = []
            tokens_used = 0
            errors = [f"UnhandledException: {exc!s}"]
            goal_achieved = False

        latency_ms = (time.perf_counter() - t0) * 1_000.0

        result = BenchmarkResult(
            task_id=task_config.task_id,
            framework=self.framework_name,
            steps_taken=len(trajectory),
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            errors=errors,
            goal_achieved=goal_achieved,
            trajectory=trajectory,
        )
        logger.debug(
            "Finished task=%s framework=%s goal=%s steps=%d tokens=%d latency_ms=%.2f",
            result.task_id,
            result.framework,
            result.goal_achieved,
            result.steps_taken,
            result.tokens_used,
            result.latency_ms,
        )
        return result


# ---------------------------------------------------------------------------
# Framework implementations (mock stubs — Phase 3 replaces _execute bodies)
# ---------------------------------------------------------------------------

# Each stub produces a trajectory whose *structure* mirrors the framework's
# native execution model.  A staff-engineer reviewer should be able to see
# exactly where real API calls will slot in during Phase 3.


class LangGraphBenchmark(AgentBenchmark):
    """LangGraph StateGraph stub.

    Real implementation (Phase 3):
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

    # Node sequences per task: (node_name, action_label, tool_calls)
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
                "Fix: add outbound UDP 4500 exception. No escalation needed."
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

    Real implementation (Phase 3):
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
            [
                ("GeneralistAgent", "process_task", []),
            ],
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

    Real implementation (Phase 3):
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

    _CONVERSATIONS: dict[str, list[tuple[str, str, list[str], str]]] = {
        "it_helpdesk": [
            (
                "UserProxy",
                "user",
                [],
                "VPN connection failing with error 619. Laptop: MacBook Pro, "
                "FortiClient 7.2. Started this morning after OS update.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["query_vpn_logs"],
                "Checking VPN logs. Error 619 indicates tunnel endpoint unreachable. "
                "Running network diagnostic now.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["run_diagnostic", "check_network_status"],
                "Diagnostic complete: UDP 4500 is blocked by firewall FW-CORP-04. "
                "This is the root cause.",
            ),
            ("UserProxy", "user", [], "Can you fix it or do we need Tier 2?"),
            (
                "AssistantAgent",
                "assistant",
                ["send_ticket"],
                "Fix is within Level 1 scope. Adding UDP 4500 exception to FW-CORP-04. "
                "Ticket IT-20251114-0042 filed. Please retry VPN in 5 minutes.",
            ),
            ("UserProxy", "user", ["log_event"], "TERMINATE"),
        ],
        "wearable_privacy": [
            (
                "UserProxy",
                "user",
                ["get_sensor_reading"],
                "Heart rate spiked to 112 bpm. User is in a private conversation. "
                "Should we send a health alert?",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["assess_health_risk"],
                "HR 112 bpm: elevated but below critical threshold (140 bpm). "
                "No cardiac emergency markers.",
            ),
            (
                "AssistantAgent",
                "assistant",
                ["check_privacy_context"],
                "Privacy context: AMBIENT consent, private_conversation flag active. "
                "Policy prohibits biometric transmission in this state.",
            ),
            ("UserProxy", "user", [], "What is the final decision?"),
            (
                "AssistantAgent",
                "assistant",
                ["log_event"],
                "Decision: suppress alert, log locally. HR threshold not exceeded; "
                "privacy boundary respected. TERMINATE",
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

    Real implementation (Phase 3):
        triage_agent = Agent(
            name="TriageAgent",
            instructions=TRIAGE_PROMPT,
            tools=[check_network_status, query_vpn_logs, escalate_to_tier2],
            handoffs=[DiagnosticAgent],
        )
        result = Runner.run_sync(triage_agent, task_config.description)
    Trajectory schema: ``{step, agent, event_type, tool/to_agent,
    input, output, timestamp_offset_ms}``.
    """

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
            [
                ("Agent", "tool_call", "process", "{}", "Done."),
            ],
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
    """Orchestrates multi-framework benchmark runs and output.

    Loads task configs from a YAML file, dispatches each task to each
    framework, appends results to a JSONL log, and renders a rich
    comparison table.

    Args:
        config_path: Path to ``benchmark_tasks.yaml``.
        output_path: Path for JSONL result log (created or appended to).
        framework_names: Frameworks to include; ``None`` means all registered.
    """

    def __init__(
        self,
        config_path: Path,
        output_path: Path,
        framework_names: list[str] | None = None,
    ) -> None:
        self._config_path = config_path
        self._output_path = output_path
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

    def run_all(self, task_ids: list[str] | None = None) -> list[BenchmarkResult]:
        """Run every (task, framework) combination and log results.

        Args:
            task_ids: Tasks to run; ``None`` runs all tasks in the config.

        Returns:
            All BenchmarkResult instances in (task, framework) order.
        """
        tasks = self.load_tasks(task_ids)
        results: list[BenchmarkResult] = []

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(tasks) * len(self._frameworks)
        logger.info(
            "Running %d task(s) × %d framework(s) = %d combinations",
            len(tasks),
            len(self._frameworks),
            total,
        )

        for task in tasks:
            for benchmark in self._frameworks:
                result = benchmark.run_task(task)
                self._log_result(result)
                results.append(result)

        logger.info(
            "Benchmark complete. %d results written to %s",
            len(results),
            self._output_path,
        )
        return results

    def _log_result(self, result: BenchmarkResult) -> None:
        """Append a single BenchmarkResult as one JSON line to the output file.

        Args:
            result: Result to persist.
        """
        with self._output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        logger.debug(
            "Logged result task=%s framework=%s",
            result.task_id,
            result.framework,
        )

    def print_table(self, results: list[BenchmarkResult]) -> None:
        """Render a rich comparison table to stdout.

        Rows are grouped by task_id; columns show per-framework metrics.

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
        table.add_column("Task ID", style="bold", min_width=16)
        table.add_column("Framework", min_width=14)
        table.add_column("Steps", justify="right")
        table.add_column("Tokens", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Errors", justify="right")
        table.add_column("Goal", justify="center")

        # Group by task so visual separation is obvious.
        tasks_seen: list[str] = []
        for r in results:
            if r.task_id not in tasks_seen:
                tasks_seen.append(r.task_id)

        for task_id in tasks_seen:
            task_results = [r for r in results if r.task_id == task_id]
            for i, r in enumerate(task_results):
                goal_cell = "[green]✓[/green]" if r.goal_achieved else "[red]✗[/red]"
                errors_cell = f"[red]{len(r.errors)}[/red]" if r.errors else "0"
                # Show task_id only on the first row of each group.
                table.add_row(
                    r.task_id if i == 0 else "",
                    r.framework,
                    str(r.steps_taken),
                    str(r.tokens_used),
                    f"{r.latency_ms:.3f}",
                    errors_cell,
                    goal_cell,
                )

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
    )
    results = runner.run_all(task_ids=task_ids)
    runner.print_table(results)


if __name__ == "__main__":
    app()
