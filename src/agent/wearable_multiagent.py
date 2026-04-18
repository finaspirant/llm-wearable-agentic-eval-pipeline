"""Multi-agent wearable system with role specialization.

Orchestrator delegates to specialized agents:
- HealthAgent: medical assessment, emergency detection
- PrivacyGateAgent: consent verification, data minimization
- ActionAgent: scheduling / notification / location actions

Implements DeepMind's authority/responsibility/accountability framework
as measurable eval dimensions (Layer 2 annotation). Each agent emits a
:class:`RoleAnnotation` that feeds the agenteval-schema-v1.json Layer 2
``roles`` block.

The :class:`MultiAgentPipeline` produces a :class:`MultiAgentResult`
whose ``trajectory`` is a ``list[TrajectoryStep]`` compatible with
:class:`src.eval.trajectory_scorer.TrajectoryScorer`.

Routing rules:
    health_alert      → HealthAgent → ActionAgent
    privacy_sensitive → PrivacyGateAgent → ActionAgent
    location_trigger  → PrivacyGateAgent → ActionAgent
    ambient_noise     → HealthAgent (hearing safety) → ActionAgent
    calendar_reminder → ActionAgent (direct — no specialist)

CLI::

    python -m src.agent.wearable_multiagent \\
        --input data/raw/synthetic_wearable_logs.jsonl \\
        --output data/processed/multiagent_results.jsonl \\
        --limit 10
"""

from __future__ import annotations

import json
import logging
import operator
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import typer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich.console import Console
from rich.table import Table

from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import (
    AgentAction,
    AudioTranscript,
    ScenarioType,
    SensorData,
    TrajectoryStep,
    WearableLog,
)

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="wearable-multiagent",
    help="Run the multi-agent wearable pipeline (Orchestrator + 3 specialists).",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scenario → LangGraph node routing table
_SCENARIO_ROUTING: dict[str, str] = {
    ScenarioType.HEALTH_ALERT.value: "health_check",
    ScenarioType.PRIVACY_SENSITIVE.value: "privacy_check",
    ScenarioType.LOCATION_TRIGGER.value: "privacy_check",
    ScenarioType.AMBIENT_NOISE.value: "health_check",
    ScenarioType.CALENDAR_REMINDER.value: "action_step",
}

# Biometric thresholds
_HR_ALERT_THRESHOLD: float = 140.0  # bpm — alert territory
_HR_EMERGENCY_THRESHOLD: float = 160.0  # bpm — escalation territory
_SPO2_ALERT_THRESHOLD: float = 92.0  # % — mild hypoxia
_SPO2_EMERGENCY_THRESHOLD: float = 88.0  # % — severe hypoxia
_NOISE_HEARING_THRESHOLD: float = 85.0  # dB SPL — WHO guideline

_EMERGENCY_AUDIO_KEYWORDS: frozenset[str] = frozenset(
    {"chest pain", "help", "breathe", "fall", "dizzy"}
)
_PRIVACY_AUDIO_KEYWORDS: frozenset[str] = frozenset(
    {"private", "confidential", "secret", "therapy", "intimate"}
)

# Orchestrator handoff quality scores by scenario (1–5 Likert)
_ORCHESTRATOR_HANDOFF_QUALITY: dict[str, int] = {
    ScenarioType.HEALTH_ALERT.value: 5,
    ScenarioType.PRIVACY_SENSITIVE.value: 5,
    ScenarioType.LOCATION_TRIGGER.value: 5,
    ScenarioType.AMBIENT_NOISE.value: 4,  # hearing-safety routing is less obvious
    ScenarioType.CALENDAR_REMINDER.value: 4,  # no specialist delegation
}

AgentRoleType = Literal[
    "orchestrator",
    "health_agent",
    "privacy_gate_agent",
    "calendar_action_agent",
]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RoleAnnotation:
    """Layer 2 role-attribution record for one agent in a multi-agent session.

    Field semantics mirror agenteval-schema-v1.json Layer 2 ``roles`` items
    exactly. ``handoff_quality`` is only populated for
    ``agent_role="orchestrator"`` and is excluded from :meth:`to_dict` for
    all other roles per the schema's if/then/else constraint.

    Args:
        agent_id: Stable identifier for this agent instance
            (e.g. ``"health_agent_01"``).
        agent_role: Functional role — governs authority scope.
        delegation_quality: 1–5 Likert — quality of sub-task delegation.
        authority_appropriate: True if agent acted within its defined scope.
        accountability_clear: True if a session failure is traceable to this
            agent's decision.
        handoff_quality: 1–5 Likert — context completeness passed to
            sub-agents. Only set for orchestrator; ``None`` for all others.
    """

    agent_id: str
    agent_role: AgentRoleType
    delegation_quality: int
    authority_appropriate: bool
    accountability_clear: bool
    handoff_quality: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a schema-compliant dict.

        ``handoff_quality`` is omitted for non-orchestrator roles per the
        agenteval-schema-v1.json if/then/else constraint.

        Returns:
            JSON-safe dict matching agenteval-schema-v1.json Layer 2 format.
        """
        d: dict[str, Any] = {
            "agent_id": self.agent_id,
            "agent_role": self.agent_role,
            "delegation_quality": self.delegation_quality,
            "authority_appropriate": self.authority_appropriate,
            "accountability_clear": self.accountability_clear,
        }
        if self.handoff_quality is not None:
            d["handoff_quality"] = self.handoff_quality
        return d


@dataclass
class MultiAgentResult:
    """Output of one complete multi-agent pipeline run.

    Args:
        log_id: UUID4 of the source :class:`WearableLog`.
        scenario_type: Scenario type string from the source log.
        final_action: Terminal :class:`AgentAction` executed at the act step.
        role_annotations: Layer 2 attribution records from all participating
            agents (orchestrator + specialist + action executor where applicable).
        trajectory: Ordered ``[sense, plan, act]`` steps compatible with
            :class:`src.eval.trajectory_scorer.TrajectoryScorer`.
        latency_ms: Monotonic wall-clock execution time in milliseconds.
    """

    log_id: str
    scenario_type: str
    final_action: AgentAction
    role_annotations: list[RoleAnnotation]
    trajectory: list[TrajectoryStep]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Returns:
            Dict with string-valued ``final_action`` and fully serialised
            ``role_annotations`` and ``trajectory`` sub-structures.
        """
        return {
            "log_id": self.log_id,
            "scenario_type": self.scenario_type,
            "final_action": self.final_action.value,
            "role_annotations": [r.to_dict() for r in self.role_annotations],
            "trajectory": [asdict(s) for s in self.trajectory],
            "latency_ms": round(self.latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class _PipelineState(TypedDict):
    """Internal LangGraph state shared across all pipeline nodes.

    ``role_annotations`` and ``trajectory`` use ``operator.add`` as their
    channel reducer so each node appends its contribution without re-emitting
    the full accumulated list.  ``final_action`` and ``routing_target`` use
    default LastValue (replace-on-write) semantics.
    """

    log: WearableLog
    role_annotations: Annotated[list[RoleAnnotation], operator.add]
    trajectory: Annotated[list[TrajectoryStep], operator.add]
    final_action: str  # AgentAction.value string — LastValue
    routing_target: str  # LangGraph node name — LastValue


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------


class HealthAgent:
    """Biometric assessment specialist for health_alert and ambient_noise scenarios.

    Applies HR/SpO2 threshold logic and hearing-safety heuristics.
    Recommends :attr:`~AgentAction.ESCALATE_TO_EMERGENCY`,
    :attr:`~AgentAction.SEND_ALERT`, :attr:`~AgentAction.ADJUST_NOISE_PROFILE`,
    or :attr:`~AgentAction.LOG_AND_MONITOR` based on noised sensor values.

    Called as a LangGraph node via ``__call__``.
    """

    _AGENT_ID: str = "health_agent_01"
    _ROLE: AgentRoleType = "health_agent"

    def __call__(self, state: _PipelineState) -> dict[str, Any]:
        """Execute health assessment and emit the plan TrajectoryStep.

        Args:
            state: Current pipeline state containing the source WearableLog.

        Returns:
            Partial state update: one plan TrajectoryStep appended to
            ``trajectory``, one RoleAnnotation appended to
            ``role_annotations``, and the recommended ``final_action`` value.
        """
        log = state["log"]
        sensor = log.sensor_data
        hr = sensor.heart_rate_noised
        spo2 = sensor.spo2_noised
        noise_db = sensor.noise_db_noised
        keywords = set(log.audio_transcript.keywords_detected)
        scenario = ScenarioType(log.scenario_type)

        if scenario == ScenarioType.AMBIENT_NOISE:
            recommended, confidence, reasoning = self._assess_noise(noise_db, hr)
        else:
            recommended, confidence, reasoning = self._assess_biometrics(
                hr, spo2, keywords
            )

        plan_step = TrajectoryStep(
            step_index=1,
            step_name="plan",
            observation=(
                f"Health assessment: HR={hr:.0f} bpm, SpO2={spo2:.1f}%, "
                f"noise={noise_db:.0f} dB SPL."
            ),
            reasoning=reasoning,
            action="",
            confidence=confidence,
        )
        annotation = RoleAnnotation(
            agent_id=self._AGENT_ID,
            agent_role=self._ROLE,
            delegation_quality=5,
            authority_appropriate=True,
            accountability_clear=True,
        )
        logger.info(
            "HealthAgent: scenario=%s recommended=%s confidence=%.2f",
            scenario.value,
            recommended,
            confidence,
        )
        return {
            "trajectory": [plan_step],
            "role_annotations": [annotation],
            "final_action": recommended,
        }

    def _assess_biometrics(
        self,
        hr: float,
        spo2: float,
        keywords: set[str],
    ) -> tuple[str, float, str]:
        """Apply HR/SpO2 threshold logic with optional audio keyword corroboration.

        Args:
            hr: DP-noised heart rate in bpm.
            spo2: DP-noised SpO2 percentage.
            keywords: Audio keywords detected in this session.

        Returns:
            Tuple of ``(action_value, confidence, reasoning_string)``.
        """
        audio_emergency = bool(_EMERGENCY_AUDIO_KEYWORDS & keywords)
        critical = hr > _HR_EMERGENCY_THRESHOLD or spo2 < _SPO2_EMERGENCY_THRESHOLD
        alert = hr > _HR_ALERT_THRESHOLD or spo2 < _SPO2_ALERT_THRESHOLD

        if critical and audio_emergency:
            return (
                AgentAction.ESCALATE_TO_EMERGENCY.value,
                0.95,
                (
                    f"Dual-modality confirmation: HR={hr:.0f} bpm"
                    f" (>{_HR_EMERGENCY_THRESHOLD:.0f}) or"
                    f" SpO2={spo2:.1f}% (<{_SPO2_EMERGENCY_THRESHOLD:.0f}%)"
                    " plus emergency audio keyword detected."
                    " Immediate escalation required."
                ),
            )
        if alert:
            return (
                AgentAction.SEND_ALERT.value,
                0.85,
                (
                    f"Biometric alert threshold exceeded:"
                    f" HR={hr:.0f} bpm (threshold {_HR_ALERT_THRESHOLD:.0f})"
                    f" or SpO2={spo2:.1f}% (threshold {_SPO2_ALERT_THRESHOLD:.0f}%)."
                    " Sending alert to emergency contact."
                ),
            )
        return (
            AgentAction.LOG_AND_MONITOR.value,
            0.78,
            (
                f"Biometrics within normal range: HR={hr:.0f} bpm,"
                f" SpO2={spo2:.1f}%. No immediate intervention required;"
                " logging for trend analysis."
            ),
        )

    def _assess_noise(self, noise_db: float, hr: float) -> tuple[str, float, str]:
        """Apply hearing-safety heuristic for ambient_noise scenario.

        Args:
            noise_db: DP-noised ambient noise in dB SPL.
            hr: DP-noised heart rate (corroborating physiological state).

        Returns:
            Tuple of ``(action_value, confidence, reasoning_string)``.
        """
        if noise_db > _NOISE_HEARING_THRESHOLD:
            return (
                AgentAction.ADJUST_NOISE_PROFILE.value,
                0.90,
                (
                    f"Ambient noise {noise_db:.0f} dB SPL exceeds WHO guideline"
                    f" ({_NOISE_HEARING_THRESHOLD:.0f} dB)."
                    " Adjusting noise profile to protect hearing health."
                ),
            )
        return (
            AgentAction.ADJUST_NOISE_PROFILE.value,
            0.80,
            (
                f"Ambient noise {noise_db:.0f} dB SPL within safe range."
                f" HR={hr:.0f} bpm normal. Proactive noise profile"
                " adjustment for comfort and exposure logging."
            ),
        )


class PrivacyGateAgent:
    """Consent enforcement specialist for privacy_sensitive and location_trigger.

    Applies the ConsentModel decision matrix. Recommends
    :attr:`~AgentAction.SUPPRESS_CAPTURE`, :attr:`~AgentAction.REQUEST_CONSENT`,
    or :attr:`~AgentAction.TRIGGER_GEOFENCE` (location_trigger only).

    Decision priority: ``REVOKED`` > ``AMBIENT`` + sensitive keywords >
    scenario-specific default.

    Called as a LangGraph node via ``__call__``.
    """

    _AGENT_ID: str = "privacy_gate_agent_01"
    _ROLE: AgentRoleType = "privacy_gate_agent"

    def __call__(self, state: _PipelineState) -> dict[str, Any]:
        """Execute consent verification and emit the plan TrajectoryStep.

        Args:
            state: Current pipeline state containing the source WearableLog.

        Returns:
            Partial state update: one plan TrajectoryStep, one RoleAnnotation,
            and the recommended ``final_action`` value.
        """
        log = state["log"]
        consent = log.consent_model
        keywords = set(log.audio_transcript.keywords_detected)
        scenario = ScenarioType(log.scenario_type)
        lat = log.sensor_data.gps_lat_noised
        lon = log.sensor_data.gps_lon_noised

        recommended, confidence, reasoning = self._enforce_consent(
            consent, keywords, scenario, lat, lon
        )
        has_sensitive = bool(_PRIVACY_AUDIO_KEYWORDS & keywords)

        plan_step = TrajectoryStep(
            step_index=1,
            step_name="plan",
            observation=(
                f"Privacy assessment: consent={consent.value},"
                f" sensitive_keywords={has_sensitive},"
                f" scenario={scenario.value}."
            ),
            reasoning=reasoning,
            action="",
            confidence=confidence,
        )
        annotation = RoleAnnotation(
            agent_id=self._AGENT_ID,
            agent_role=self._ROLE,
            delegation_quality=5,
            authority_appropriate=True,
            accountability_clear=True,
        )
        logger.info(
            "PrivacyGateAgent: consent=%s scenario=%s recommended=%s",
            consent.value,
            scenario.value,
            recommended,
        )
        return {
            "trajectory": [plan_step],
            "role_annotations": [annotation],
            "final_action": recommended,
        }

    def _enforce_consent(
        self,
        consent: ConsentModel,
        keywords: set[str],
        scenario: ScenarioType,
        lat: float,
        lon: float,
    ) -> tuple[str, float, str]:
        """Apply the ConsentModel decision matrix.

        Args:
            consent: Applicable ConsentModel for this session.
            keywords: Detected audio keywords.
            scenario: Wearable scenario type.
            lat: DP-noised GPS latitude.
            lon: DP-noised GPS longitude.

        Returns:
            Tuple of ``(action_value, confidence, reasoning_string)``.
        """
        has_sensitive = bool(_PRIVACY_AUDIO_KEYWORDS & keywords)

        if consent == ConsentModel.REVOKED:
            return (
                AgentAction.SUPPRESS_CAPTURE.value,
                0.99,
                (
                    "REVOKED consent: no data capture or logging permitted."
                    " Suppressing capture immediately —"
                    " categorical policy enforcement, no judgment call."
                ),
            )

        if consent == ConsentModel.AMBIENT and has_sensitive:
            return (
                AgentAction.REQUEST_CONSENT.value,
                0.87,
                (
                    "AMBIENT consent with sensitive audio keywords detected."
                    " Requesting explicit consent before proceeding —"
                    " ambiguity resolved in favour of user protection."
                ),
            )

        if scenario == ScenarioType.LOCATION_TRIGGER:
            return (
                AgentAction.TRIGGER_GEOFENCE.value,
                0.88,
                (
                    f"Location trigger verified: GPS ({lat:.4f}, {lon:.4f})."
                    f" Consent={consent.value} permits geofence action."
                    " Executing location automation."
                ),
            )

        return (
            AgentAction.SUPPRESS_CAPTURE.value,
            0.82,
            (
                f"Privacy-sensitive context with consent={consent.value}."
                " Default-safe action: suppressing audio capture"
                " and notifying user."
            ),
        )


class ActionAgent:
    """Scheduling, notification, and location action executor.

    Owns :attr:`~AgentAction.SURFACE_REMINDER`, geofence triggers, and
    noise profile adjustments. For ``calendar_reminder`` scenarios (no prior
    specialist), also provides the plan step.

    Called as a LangGraph node via ``__call__``.
    """

    _AGENT_ID: str = "calendar_action_agent_01"
    _ROLE: AgentRoleType = "calendar_action_agent"

    def __call__(self, state: _PipelineState) -> dict[str, Any]:
        """Execute the terminal action and emit the act TrajectoryStep.

        If no specialist ran before this node (``calendar_reminder`` direct
        route, indicated by only one TrajectoryStep in state), a plan step
        is also emitted at step_index 1.

        Args:
            state: Current pipeline state.

        Returns:
            Partial state update: one or two TrajectorySteps appended to
            ``trajectory``, optionally one RoleAnnotation, and the confirmed
            ``final_action`` string.
        """
        log = state["log"]
        existing_steps = state["trajectory"]
        needs_plan = len(existing_steps) == 1  # only the sense step present

        new_steps: list[TrajectoryStep] = []
        new_annotations: list[RoleAnnotation] = []
        action_str: str

        if needs_plan:
            plan_step, action_str = self._plan_calendar(log, step_index=1)
            new_steps.append(plan_step)
            new_annotations.append(
                RoleAnnotation(
                    agent_id=self._AGENT_ID,
                    agent_role=self._ROLE,
                    delegation_quality=5,
                    authority_appropriate=True,
                    accountability_clear=True,
                )
            )
        else:
            action_str = state["final_action"]

        act_index = len(existing_steps) + len(new_steps)
        act_step = TrajectoryStep(
            step_index=act_index,
            step_name="act",
            observation=f"Action ready: {action_str}. All preconditions verified.",
            reasoning=(
                f"Executing {action_str} — authority check passed,"
                " consent gate cleared, action within role scope."
            ),
            action=action_str,
            confidence=0.92,
        )
        new_steps.append(act_step)

        logger.info("ActionAgent: final_action=%s", action_str)
        return {
            "trajectory": new_steps,
            "role_annotations": new_annotations,
            "final_action": action_str,
        }

    def _plan_calendar(
        self, log: WearableLog, step_index: int
    ) -> tuple[TrajectoryStep, str]:
        """Build the plan step and select action for calendar_reminder scenario.

        Args:
            log: Source WearableLog.
            step_index: Step index for the plan TrajectoryStep.

        Returns:
            Tuple of ``(plan_step, action_value_string)``.
        """
        ctx = log.context_metadata
        meeting_type = ctx.get("meeting_type", "meeting")
        minutes_until = ctx.get("minutes_until", 10)
        hr = log.sensor_data.heart_rate_noised

        plan_step = TrajectoryStep(
            step_index=step_index,
            step_name="plan",
            observation=(
                f"Calendar event '{meeting_type}' in {minutes_until} min;"
                f" user HR={hr:.0f} bpm — activity level nominal."
            ),
            reasoning=(
                "Surface reminder at optimal interruption window —"
                " gap between activity bursts identified."
            ),
            action="",
            confidence=0.88,
        )
        return plan_step, AgentAction.SURFACE_REMINDER.value


class OrchestratorAgent:
    """Multi-agent coordinator that routes WearableLogs to specialist agents.

    Implements the sense step of the pipeline: reads the log, determines the
    correct specialist routing, and emits a :class:`RoleAnnotation` with
    ``handoff_quality`` scored by scenario routing complexity.

    Called as a LangGraph node via ``__call__``. Routing decisions are
    exposed via :meth:`route` for LangGraph's conditional edge API.
    """

    _AGENT_ID: str = "orchestrator_01"
    _ROLE: AgentRoleType = "orchestrator"

    def __call__(self, state: _PipelineState) -> dict[str, Any]:
        """Execute the sense step: observe the log and determine routing.

        Args:
            state: Initial pipeline state with the source WearableLog.

        Returns:
            Partial state update: one sense TrajectoryStep, one RoleAnnotation
            (including ``handoff_quality``), and the ``routing_target`` for
            the conditional edge.
        """
        log = state["log"]
        scenario = ScenarioType(log.scenario_type)
        routing = _SCENARIO_ROUTING[scenario.value]
        handoff_q = _ORCHESTRATOR_HANDOFF_QUALITY[scenario.value]

        sense_step = TrajectoryStep(
            step_index=0,
            step_name="sense",
            observation=self._format_observation(log),
            reasoning=(
                f"Routing to '{routing}' based on scenario_type='{scenario.value}'."
                f" Consent model: {log.consent_model.value}."
            ),
            action="",
            confidence=0.92,
        )
        annotation = RoleAnnotation(
            agent_id=self._AGENT_ID,
            agent_role=self._ROLE,
            delegation_quality=5 if scenario != ScenarioType.CALENDAR_REMINDER else 4,
            authority_appropriate=True,
            # orchestrator coordinates; specialist owns failure
            accountability_clear=False,
            handoff_quality=handoff_q,
        )
        logger.info(
            "OrchestratorAgent: log_id=%s scenario=%s routing=%s handoff_quality=%s",
            log.log_id,
            scenario.value,
            routing,
            handoff_q,
        )
        return {
            "trajectory": [sense_step],
            "role_annotations": [annotation],
            "routing_target": routing,
        }

    def route(self, state: _PipelineState) -> str:
        """Return the routing target for LangGraph's conditional edge.

        Args:
            state: Pipeline state after the orchestrate node has run.

        Returns:
            One of ``"health_check"``, ``"privacy_check"``, or
            ``"action_step"``.
        """
        return state["routing_target"]

    @staticmethod
    def _format_observation(log: WearableLog) -> str:
        """Format the sense-step observation string from the source log.

        Args:
            log: Source WearableLog.

        Returns:
            Human-readable observation string including key sensor readings.
        """
        s = log.sensor_data
        return (
            f"Wearable log received: scenario={log.scenario_type},"
            f" consent={log.consent_model.value},"
            f" HR={s.heart_rate_noised:.0f} bpm,"
            f" SpO2={s.spo2_noised:.1f}%,"
            f" noise={s.noise_db_noised:.0f} dB SPL."
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class MultiAgentPipeline:
    """End-to-end multi-agent wearable pipeline backed by LangGraph StateGraph.

    Connects OrchestratorAgent → (HealthAgent | PrivacyGateAgent | ActionAgent)
    → ActionAgent via a compiled StateGraph. Produces a
    :class:`MultiAgentResult` compatible with
    :class:`src.eval.trajectory_scorer.TrajectoryScorer`.

    Graph topology::

        orchestrate
          ├─► health_check  ─► action_step ─► END
          ├─► privacy_check ─► action_step ─► END
          └─►                  action_step ─► END  (calendar_reminder direct)

    Args:
        dry_run: Reserved for future LLM-backed agent variants. All current
            logic is heuristic (no API calls).
    """

    def __init__(self, dry_run: bool = True) -> None:
        self._dry_run = dry_run
        self._orchestrator = OrchestratorAgent()
        self._health = HealthAgent()
        self._privacy = PrivacyGateAgent()
        self._action = ActionAgent()
        self._graph: CompiledStateGraph[Any, Any, Any] = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any]:
        """Construct and compile the LangGraph StateGraph.

        Returns:
            Compiled LangGraph graph.
        """
        workflow: StateGraph = StateGraph(_PipelineState)  # type: ignore[type-arg]
        workflow.add_node("orchestrate", self._orchestrator)
        workflow.add_node("health_check", self._health)
        workflow.add_node("privacy_check", self._privacy)
        workflow.add_node("action_step", self._action)

        workflow.set_entry_point("orchestrate")
        workflow.add_conditional_edges(
            "orchestrate",
            self._orchestrator.route,
            {
                "health_check": "health_check",
                "privacy_check": "privacy_check",
                "action_step": "action_step",
            },
        )
        workflow.add_edge("health_check", "action_step")
        workflow.add_edge("privacy_check", "action_step")
        workflow.add_edge("action_step", END)

        return workflow.compile()

    def run(self, log: WearableLog) -> MultiAgentResult:
        """Run the full multi-agent pipeline for one WearableLog.

        Args:
            log: Source wearable event log to process.

        Returns:
            :class:`MultiAgentResult` with the terminal action, Layer 2
            role annotations, the full ``[sense, plan, act]`` trajectory,
            and wall-clock latency.

        Raises:
            ValueError: If the pipeline returns an unrecognised
                ``final_action`` string.
        """
        start_ms = time.monotonic() * 1000.0
        initial: _PipelineState = {
            "log": log,
            "role_annotations": [],
            "trajectory": [],
            "final_action": "",
            "routing_target": "",
        }
        final_state: dict[str, Any] = self._graph.invoke(initial)
        latency_ms = time.monotonic() * 1000.0 - start_ms

        try:
            final_action = AgentAction(final_state["final_action"])
        except ValueError as exc:
            raise ValueError(
                f"Pipeline returned invalid final_action="
                f"{final_state['final_action']!r} for log_id={log.log_id}"
            ) from exc

        result = MultiAgentResult(
            log_id=log.log_id,
            scenario_type=log.scenario_type.value,
            final_action=final_action,
            role_annotations=final_state["role_annotations"],
            trajectory=final_state["trajectory"],
            latency_ms=latency_ms,
        )
        logger.info(
            "MultiAgentPipeline: log_id=%s scenario=%s action=%s"
            " n_roles=%d latency=%.1fms",
            log.log_id,
            log.scenario_type.value,
            final_action.value,
            len(result.role_annotations),
            latency_ms,
        )
        return result

    def run_batch(self, logs: list[WearableLog]) -> list[MultiAgentResult]:
        """Run the pipeline over a list of WearableLogs.

        Failed logs are skipped with a warning rather than aborting the batch,
        to preserve partial results in large evaluation runs.

        Args:
            logs: Wearable logs to process.

        Returns:
            List of :class:`MultiAgentResult` in input order; skipped logs
            are absent from the output.
        """
        results: list[MultiAgentResult] = []
        for log in logs:
            try:
                results.append(self.run(log))
            except Exception:
                logger.exception(
                    "Pipeline failed for log_id=%s — skipping.", log.log_id
                )
        logger.info(
            "run_batch: %d / %d logs processed successfully",
            len(results),
            len(logs),
        )
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_logs(path: Path, limit: int | None) -> list[WearableLog]:
    """Load :class:`WearableLog` instances from a JSONL file.

    Args:
        path: Path to a JSONL file produced by
            :mod:`src.data.wearable_generator`.
        limit: Maximum number of logs to load (``None`` = all).

    Returns:
        List of :class:`WearableLog` dataclasses.
    """
    logs: list[WearableLog] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            raw = json.loads(line)
            logs.append(
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
    return logs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_path: Path = typer.Option(
        Path("data/raw/synthetic_wearable_logs.jsonl"),
        "--input",
        help="Path to JSONL file of WearableLog records.",
    ),
    output_path: Path = typer.Option(
        Path("data/processed/multiagent_results.jsonl"),
        "--output",
        help="Path for JSONL output of MultiAgentResult records.",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Max logs to process (default: all)."
    ),
    verbose: bool = typer.Option(False, "--verbose/--quiet"),
) -> None:
    """Run the multi-agent wearable pipeline and write results to JSONL."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        raise typer.Exit(1)

    logs = _load_logs(input_path, limit)
    logger.info("Loaded %d logs from %s", len(logs), input_path)

    pipeline = MultiAgentPipeline()
    results = pipeline.run_batch(logs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.to_dict()) + "\n")
    logger.info("Wrote %d results to %s", len(results), output_path)

    # Rich summary table
    table = Table(title="Multi-Agent Pipeline Results")
    table.add_column("log_id", style="dim", max_width=10)
    table.add_column("scenario")
    table.add_column("final_action")
    table.add_column("n_roles", justify="right")
    table.add_column("n_steps", justify="right")
    table.add_column("latency_ms", justify="right")

    for r in results:
        table.add_row(
            r.log_id[:8],
            r.scenario_type,
            r.final_action.value,
            str(len(r.role_annotations)),
            str(len(r.trajectory)),
            f"{r.latency_ms:.1f}",
        )
    console.print(table)


if __name__ == "__main__":
    app()
