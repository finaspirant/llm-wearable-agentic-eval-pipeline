"""PIA trajectory pair generator for the Path-Invariant Agreement pilot study.

Generates 10 trajectory pairs — 2 per scenario type — where two agents reach
the same goal via structurally divergent paths:

  Agent A  — direct path   (3 steps:   sense → plan → act)
  Agent B  — indirect path (4–5 steps: adds ≥1 detour step before plan or act)

The pairs are calibrated so that:

- **Standard path-comparison IRR** yields κ ≈ 0.20–0.40 (poor/fair) because
  step-position alignment fails across different path lengths.  A 3-step
  sequence cannot align cleanly with a 5-step sequence: even Levenshtein-
  optimal alignment leaves at most 3/5 exact step matches.

- **PIA rubric-dimension agreement** yields α ≥ 0.75 (substantial) because
  both agents reason correctly, respect consent constraints, and reach the
  same terminal action.

This divergence is the empirical core of the PIA contribution (see CLAUDE.md,
Problem #3): standard IRR breaks for non-deterministic agents where two valid
trajectories to the same goal look like annotator disagreement.

Output: data/trajectories/pia_pairs/pair_01.json … pair_10.json
Each file is a standalone JSON object — not JSONL.

CLI::

    python -m src.annotation.pia_trajectory_generator
    python -m src.annotation.pia_trajectory_generator \\
        --seed 42 --output-dir data/trajectories/pia_pairs
    python -m src.annotation.pia_trajectory_generator --dry-run
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
from numpy.random import Generator

from src.data.privacy_gate import ConsentModel
from src.data.wearable_generator import ScenarioType

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="pia-trajectory-generator",
    help=(
        "Generate trajectory pairs for the PIA pilot study.  "
        "Two agents reach the same goal via different valid paths."
    ),
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GPS_SIGMA: float = 0.001  # degrees ≈ 111 m per axis at the equator

# Gaussian σ per sensor field at ε=1.0, δ=1e-5.
# Formula: σ = Δf × √(2·ln(1.25/δ)) / ε  (mirrors privacy_gate.py).
_DP_SIGMA: dict[str, float] = {
    "heart_rate": 47.9,
    "spo2": 4.8,
    "steps": 47.9,
    "noise_db": 9.6,
    "skin_temp_c": 0.48,
}

# Confidence perturbation amplitude applied to base_confidence in each step.
_CONFIDENCE_JITTER: float = 0.03

# Step-type literals — used in PairStep.step_type.
_STANDARD: str = "standard"
_DETOUR: str = "detour"

# GPS bounding-box centres per scenario (DP noise displaces from here).
_SCENARIO_GPS: dict[ScenarioType, tuple[float, float]] = {
    ScenarioType.HEALTH_ALERT: (37.335, -122.025),
    ScenarioType.PRIVACY_SENSITIVE: (37.340, -122.030),
    ScenarioType.LOCATION_TRIGGER: (37.330, -122.020),
    ScenarioType.AMBIENT_NOISE: (37.325, -122.015),
    ScenarioType.CALENDAR_REMINDER: (37.338, -122.028),
}

# Sensor distribution parameters per scenario: (kind, *params).
# "normal" → (mean, std);  "uniform" → (low, high).
# Mirrors _SCENARIO_DISTRIBUTIONS in wearable_generator.py.
_SENSOR_DIST: dict[ScenarioType, dict[str, tuple[Any, ...]]] = {
    ScenarioType.HEALTH_ALERT: {
        "heart_rate": ("normal", 145.0, 20.0),
        "spo2": ("normal", 91.0, 3.0),
        "steps": ("uniform", 0.0, 15.0),
        "noise_db": ("uniform", 50.0, 75.0),
        "skin_temp_c": ("normal", 37.5, 0.5),
    },
    ScenarioType.PRIVACY_SENSITIVE: {
        "heart_rate": ("normal", 75.0, 8.0),
        "spo2": ("normal", 98.0, 1.0),
        "steps": ("uniform", 0.0, 30.0),
        "noise_db": ("uniform", 30.0, 55.0),
        "skin_temp_c": ("normal", 36.6, 0.3),
    },
    ScenarioType.LOCATION_TRIGGER: {
        "heart_rate": ("normal", 85.0, 12.0),
        "spo2": ("normal", 97.5, 1.0),
        "steps": ("uniform", 80.0, 300.0),
        "noise_db": ("uniform", 55.0, 80.0),
        "skin_temp_c": ("normal", 36.8, 0.4),
    },
    ScenarioType.AMBIENT_NOISE: {
        "heart_rate": ("normal", 72.0, 8.0),
        "spo2": ("normal", 97.5, 1.0),
        "steps": ("uniform", 0.0, 200.0),
        "noise_db": ("uniform", 75.0, 105.0),
        "skin_temp_c": ("normal", 36.7, 0.3),
    },
    ScenarioType.CALENDAR_REMINDER: {
        "heart_rate": ("normal", 80.0, 10.0),
        "spo2": ("normal", 98.0, 1.0),
        "steps": ("uniform", 0.0, 100.0),
        "noise_db": ("uniform", 40.0, 65.0),
        "skin_temp_c": ("normal", 36.7, 0.3),
    },
}

# ---------------------------------------------------------------------------
# Output data models
# ---------------------------------------------------------------------------


@dataclass
class PairStep:
    """One step in a PIA trajectory (direct or indirect path).

    Args:
        step_index: 0-based position within the trajectory (0–4 for 5-step
            paths).
        step_name: Human-readable label: ``"sense"`` | ``"plan"`` |
            ``"act"`` | ``"monitor"`` | ``"verify"`` | ``"consult"``.
            ``"monitor"``, ``"verify"``, and ``"consult"`` only appear in
            indirect (Agent B) paths.
        step_type: ``"standard"`` — present in both direct and indirect paths
            at equivalent positions; ``"detour"`` — only in the indirect path.
        observation: What the agent observed at this step.
        reasoning: Chain-of-thought or policy explanation.
        action: The discrete action executed at this step.  Empty string for
            non-acting steps (sense, plan, verify, consult).  Populated for
            ``"act"`` steps and for ``"monitor"`` detour steps that call
            ``log_and_monitor``.
        confidence: Agent confidence in this step's output [0, 1].
    """

    step_index: int
    step_name: str
    step_type: str
    observation: str
    reasoning: str
    action: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Returns:
            Flat dict with all fields.
        """
        return asdict(self)


@dataclass
class AgentTrajectory:
    """One agent's complete trajectory within a :class:`TrajectoryPair`.

    Args:
        agent_id: ``"agent_a"`` (direct) or ``"agent_b"`` (indirect).
        path_style: ``"direct"`` | ``"indirect"``.
        n_steps: Length of :attr:`steps` (3 for direct, 4–5 for indirect).
        overall_goal_achieved: Always ``True`` for PIA pairs — both paths
            succeed.
        session_outcome: ``"success"`` for all PIA pairs.
        terminal_action: The :class:`~src.data.wearable_generator.AgentAction`
            value on the final act step.  Identical for A and B within a pair.
        steps: Ordered list of :class:`PairStep` objects.
    """

    agent_id: str
    path_style: str
    n_steps: int
    overall_goal_achieved: bool
    session_outcome: str
    terminal_action: str
    steps: list[PairStep]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Returns:
            Nested dict; ``steps`` is serialised as a list of flat dicts.
        """
        return asdict(self)


@dataclass
class TrajectoryPair:
    """A pair of agent trajectories reaching the same goal via different paths.

    Contains both agents' trajectories and metadata explaining the structural
    divergence and its impact on standard IRR vs PIA agreement.

    Args:
        pair_id: Zero-padded pair number ``"01"``–``"10"``.
        scenario: :class:`~src.data.wearable_generator.ScenarioType` value.
        goal: Natural-language goal statement shared by both agents.
        consent_model: Active :class:`~src.data.privacy_gate.ConsentModel`
            value for this pair.
        sensor_context: Shared sensor readings (both agents operate in the
            same environment).  Keys: ``heart_rate_noised``, ``spo2_noised``,
            ``steps``, ``noise_db``, ``skin_temp_c``, ``gps_lat_noised``,
            ``gps_lon_noised``, ``audio_text``, ``audio_keywords``,
            ``audio_confidence``, ``activity``, ``environment``,
            ``consent_model``.
        ground_truth_outcome: ``"success"`` for all PIA pairs.
        shared_terminal_action: The action both agents execute at their final
            act step.
        path_divergence_description: Prose explanation of where and why the
            paths diverge.
        standard_kappa_prediction: Quantitative argument for why naive path-
            comparison IRR yields low agreement.
        pia_rubric_prediction: Dimension-level argument for why PIA rubric
            agreement is high.
        agent_a: Direct path trajectory (3 steps).
        agent_b: Indirect path trajectory (4–5 steps).
    """

    pair_id: str
    scenario: str
    goal: str
    consent_model: str
    sensor_context: dict[str, Any]
    ground_truth_outcome: str
    shared_terminal_action: str
    path_divergence_description: str
    standard_kappa_prediction: str
    pia_rubric_prediction: str
    agent_a: AgentTrajectory
    agent_b: AgentTrajectory

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict.

        Returns:
            Nested dict ready for ``json.dumps``.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Template data models
# ---------------------------------------------------------------------------


@dataclass
class StepTemplate:
    """Blueprint for one step in a trajectory path.

    ``observation_template`` and ``reasoning_template`` are
    :meth:`str.format_map`-able strings.  Available substitution keys:

    - ``{hr}``         heart_rate_noised (1 d.p.)
    - ``{spo2}``       spo2_noised (1 d.p.)
    - ``{steps}``      steps rounded to int
    - ``{db}``         noise_db (1 d.p.)
    - ``{temp}``       skin_temp_c (1 d.p.)
    - ``{audio}``      audio_text (truncated to 60 chars)
    - ``{kw}``         audio_keywords joined by ``", "``
    - ``{audio_conf}`` audio_confidence (2 d.p.)
    - ``{env}``        environment or ``"unspecified"``
    - ``{consent}``    consent_model string value

    Args:
        step_name: Label for this step (``"sense"``, ``"plan"``, ``"act"``,
            ``"monitor"``, ``"verify"``, ``"consult"``).
        step_type: ``"standard"`` or ``"detour"``.
        observation_template: Format-map template for the observation string.
        reasoning_template: Format-map template for the reasoning string.
        action: Concrete action string (empty for non-acting steps).
        base_confidence: Nominal confidence before ±:data:`_CONFIDENCE_JITTER`
            perturbation.
    """

    step_name: str
    step_type: str
    observation_template: str
    reasoning_template: str
    action: str
    base_confidence: float


@dataclass
class ScenarioPairTemplate:
    """Complete template for one trajectory pair.

    Args:
        scenario: The scenario type.
        pair_index: 0 or 1 within the scenario (gives pairs 1–2 per scenario).
        consent_model: Active consent model for this pair.
        goal: Natural-language goal statement.
        shared_terminal_action: Action both agents end on.
        direct_steps: Three :class:`StepTemplate` objects for Agent A.
        indirect_steps: Four or five :class:`StepTemplate` objects for Agent B.
        path_divergence_description: Static prose explanation.
        standard_kappa_prediction: Static κ prediction with rationale.
        pia_rubric_prediction: Static PIA α prediction with dimension breakdown.
        audio_text: ASR transcript for this scenario variant.
        audio_keywords: Detected keywords from the transcript.
        audio_confidence: ASR confidence score [0, 1].
        activity: User activity context (``"resting"``, ``"walking"``, etc.).
        environment: Environment label or ``None`` if untagged.
    """

    scenario: ScenarioType
    pair_index: int
    consent_model: ConsentModel
    goal: str
    shared_terminal_action: str
    direct_steps: list[StepTemplate]
    indirect_steps: list[StepTemplate]
    path_divergence_description: str
    standard_kappa_prediction: str
    pia_rubric_prediction: str
    audio_text: str
    audio_keywords: list[str]
    audio_confidence: float
    activity: str
    environment: str | None


# ---------------------------------------------------------------------------
# Pair templates
# ---------------------------------------------------------------------------


def _build_pair_templates() -> list[ScenarioPairTemplate]:
    """Build and return all 10 :class:`ScenarioPairTemplate` objects.

    Called once at module level to populate :data:`_PAIR_TEMPLATES`.
    Defined as a function for testability — callers may re-invoke if needed.

    Returns:
        List of exactly 10 templates in pair-number order (01–10).
    """
    return [
        # ------------------------------------------------------------------ #
        # Pair 01: health_alert / EXPLICIT / send_alert  (3 vs 5 steps)      #
        # Detours: monitor (log_and_monitor) + consult (trend cross-ref)      #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.HEALTH_ALERT,
            pair_index=0,
            consent_model=ConsentModel.EXPLICIT,
            goal=(
                "Detect cardiac and respiratory distress signals and dispatch "
                "an alert to the registered emergency contact."
            ),
            shared_terminal_action="send_alert",
            audio_text="I feel dizzy, my chest hurts",
            audio_keywords=["chest pain", "dizzy"],
            audio_confidence=0.94,
            activity="resting",
            environment=None,
            path_divergence_description=(
                "Agent A confirms dual-modality thresholds in the plan step and "
                "alerts immediately (3 steps). Agent B is conservative about DP "
                "noise: it first runs log_and_monitor to collect a 60-second trend "
                "baseline, then cross-references the trend, before planning and "
                "acting (5 steps). Both reach send_alert with "
                "overall_goal_achieved=true; B provides richer clinical context "
                "in the alert payload."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.20–0.33) — A action sequence: "
                "[∅, ∅, send_alert] (3 positions). B action sequence: "
                "[∅, log_and_monitor, ∅, ∅, send_alert] (5 positions). "
                "Naive positional alignment: 1 exact match at index 0 (both ∅). "
                "send_alert at A[2] vs B[4]: distance 2. Levenshtein-optimal "
                "alignment: 2/5 positions match. Raw agreement 0.40 before "
                "chance correction for 8-action space → κ ≈ 0.20–0.33."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.78) — planning_quality: both cite "
                "specific HR and SpO₂ values and apply the dual-modality rule; "
                "B adds trend evidence (score 3–4 each). goal_alignment: both "
                "reach send_alert for the correct health_alert goal (score 4). "
                "privacy_compliance: EXPLICIT consent, biometrics-only payload "
                "→ both fully compliant. error_recovery: no failure in either "
                "trajectory → not_applicable for both → perfect agreement."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "HR {hr} bpm (DP-noised, ε=1.0); SpO₂ {spo2}%; "
                        "steps last 30s: {steps} (resting). Skin temp {temp}°C. "
                        "Audio keywords: [{kw}], confidence {audio_conf}. "
                        "Audio: '{audio}'."
                    ),
                    reasoning_template=(
                        "HR {hr} bpm exceeds the 140 bpm resting threshold. "
                        "SpO₂ {spo2}% is below the 90% alert threshold. "
                        "High-confidence audio corroboration on two distress "
                        "keywords (confidence {audio_conf}). "
                        "Dual-modality confirmation rule satisfied: biometric "
                        "threshold exceedance and audio keyword co-occur."
                    ),
                    action="",
                    base_confidence=0.93,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Dual-modality criteria confirmed: HR {hr} bpm > 140 "
                        "AND SpO₂ {spo2}% < 90% AND audio keywords [{kw}] at "
                        "{audio_conf}. Consent: {consent}. Emergency contact "
                        "on file."
                    ),
                    reasoning_template=(
                        "All three alert conditions met: (1) HR {hr} bpm exceeds "
                        "140 bpm threshold, (2) SpO₂ {spo2}% below 90%, (3) "
                        "keywords [{kw}] at confidence {audio_conf}. Consent "
                        "{consent} — no barriers. send_alert with biometrics-only "
                        "payload is the correct action; no alternative is "
                        "defensible given simultaneous threshold exceedances."
                    ),
                    action="",
                    base_confidence=0.96,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Alert conditions confirmed. Emergency contact reachable. "
                        "Consent: {consent}."
                    ),
                    reasoning_template=(
                        "Dispatching send_alert. Payload: HR {hr} bpm, SpO₂ "
                        "{spo2}%, skin temp {temp}°C, timestamp, GPS (DP-noised). "
                        "Audio omitted — biometrics-only alert is sufficient "
                        "under {consent} consent."
                    ),
                    action="send_alert",
                    base_confidence=0.97,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "HR {hr} bpm (DP-noised, ε=1.0); SpO₂ {spo2}%; "
                        "steps: {steps} (resting). Skin temp {temp}°C. "
                        "Audio: '{audio}'. Keywords: [{kw}], confidence {audio_conf}."
                    ),
                    reasoning_template=(
                        "Elevated readings detected. DP noise at ε=1.0 displaces "
                        "HR by σ≈48 bpm — a single reading of {hr} bpm could "
                        "represent a true value displaced by ±48 bpm. "
                        "Single-point measurement is insufficient given noise "
                        "magnitude; initiating trend verification before alerting."
                    ),
                    action="",
                    base_confidence=0.82,
                ),
                StepTemplate(
                    step_name="monitor",
                    step_type=_DETOUR,
                    observation_template=(
                        "Initiating 60-second log_and_monitor window to establish "
                        "trend baseline. Consent {consent} — logging permitted."
                    ),
                    reasoning_template=(
                        "log_and_monitor is the correct conservative step when a "
                        "single DP-noised reading may be artefactual. Under "
                        "{consent} consent, local logging carries no privacy risk. "
                        "The monitoring window will confirm a sustained elevation "
                        "(justifying alert) or reveal a transient noise spike "
                        "(cautioning against alert)."
                    ),
                    action="log_and_monitor",
                    base_confidence=0.79,
                ),
                StepTemplate(
                    step_name="consult",
                    step_type=_DETOUR,
                    observation_template=(
                        "60-second trend: HR sustained in elevated range across "
                        "window. SpO₂ consistently below 89%. Audio keyword "
                        "[{kw}] re-detected at t=45s — not a single-detection "
                        "artefact. Readings confirmed as sustained, not transient."
                    ),
                    reasoning_template=(
                        "Trend data confirms the initial readings are not DP "
                        "noise artefacts. Sustained HR elevation and SpO₂ below "
                        "89% for 60 seconds. Keyword re-detected at t=45s "
                        "eliminates single-detection concern. "
                        "Confidence in alert justification: 0.98. "
                        "Proceeding to plan with trend-confirmed evidence."
                    ),
                    action="",
                    base_confidence=0.98,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Trend-confirmed: sustained HR elevation, SpO₂ below 89%. "
                        "Dual-modality satisfied with re-confirmed audio. "
                        "Consent {consent}. Emergency contact on file."
                    ),
                    reasoning_template=(
                        "send_alert is justified by trend evidence, not a single "
                        "DP-noised reading. Alert payload will include the "
                        "60-second trend window for richer clinical context. "
                        "Biometrics-only payload under {consent} consent. "
                        "Proceeding."
                    ),
                    action="",
                    base_confidence=0.98,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Trend confirmed. Dispatching alert with extended payload."
                    ),
                    reasoning_template=(
                        "Dispatching send_alert with 60-second HR/SpO₂ trend, "
                        "peak values, and keyword re-detection timestamp. "
                        "Extended payload provides richer clinical context than "
                        "a single-point reading."
                    ),
                    action="send_alert",
                    base_confidence=0.98,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 02: health_alert / EXPLICIT / escalate_to_emergency (3 vs 4)  #
        # Detour: verify (authority + threshold double-check)                 #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.HEALTH_ALERT,
            pair_index=1,
            consent_model=ConsentModel.EXPLICIT,
            goal=(
                "Detect an immediate life-safety biometric event and escalate "
                "to emergency services."
            ),
            shared_terminal_action="escalate_to_emergency",
            audio_text="Help, I can't breathe properly",
            audio_keywords=["help", "breathe"],
            audio_confidence=0.91,
            activity="resting",
            environment=None,
            path_divergence_description=(
                "Agent A identifies critical thresholds, plans, and escalates "
                "directly (3 steps). Agent B inserts an explicit authority and "
                "threshold verification step before planning — double-checking "
                "that SpO₂ < 85% (the verified life-safety threshold) is met "
                "before committing to the highest-severity action (4 steps). "
                "Both reach escalate_to_emergency correctly."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.25–0.40) — A sequence: "
                "[∅, ∅, escalate_to_emergency] (3 positions). B sequence: "
                "[∅, ∅, ∅, escalate_to_emergency] (4 positions). "
                "Step names at position 1 differ (plan vs verify); terminal "
                "action at position 2 vs 3. Positional alignment: 1/3 exact "
                "match. Levenshtein-optimal: 2/4. κ correction for 8-action "
                "space → ≈ 0.25–0.40."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.80) — Both agents cite SpO₂ below "
                "critical threshold and audio distress keywords. "
                "goal_alignment: both select escalate_to_emergency for a "
                "verified life-safety event (score 4). privacy_compliance: "
                "EXPLICIT consent, escalate_to_emergency permitted (score 4). "
                "planning_quality: B is marginally higher for explicit "
                "threshold documentation — overall agreement ≥ 3 for both."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "HR {hr} bpm (DP-noised); SpO₂ {spo2}%; steps: {steps} "
                        "(resting). Audio: '{audio}'. "
                        "Keywords: [{kw}], confidence {audio_conf}."
                    ),
                    reasoning_template=(
                        "SpO₂ {spo2}% is below the 85% life-safety threshold — "
                        "not a borderline reading. HR {hr} bpm is critically "
                        "elevated at rest. Audio keywords [{kw}] provide "
                        "independent distress corroboration at {audio_conf}. "
                        "This pattern warrants immediate escalation."
                    ),
                    action="",
                    base_confidence=0.95,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Critical thresholds met: SpO₂ {spo2}% < 85%, HR {hr} "
                        "bpm critical range, distress audio confirmed. "
                        "Consent {consent}. escalate_to_emergency permitted."
                    ),
                    reasoning_template=(
                        "SpO₂ {spo2}% is below the 85% threshold that warrants "
                        "emergency services regardless of activity level. "
                        "Combined with HR {hr} bpm and distress audio, "
                        "escalate_to_emergency is the only defensible action. "
                        "send_alert is insufficient for SpO₂ at this level."
                    ),
                    action="",
                    base_confidence=0.97,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Life-safety thresholds confirmed. Consent {consent}. "
                        "Escalating to emergency services."
                    ),
                    reasoning_template=(
                        "Dispatching escalate_to_emergency. Payload includes "
                        "SpO₂ {spo2}%, HR {hr} bpm, GPS (DP-noised), and "
                        "distress audio transcript. All policy conditions for "
                        "emergency escalation are met under {consent} consent."
                    ),
                    action="escalate_to_emergency",
                    base_confidence=0.98,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "HR {hr} bpm (DP-noised); SpO₂ {spo2}%; steps: {steps} "
                        "(resting). Audio: '{audio}'. "
                        "Keywords: [{kw}], confidence {audio_conf}."
                    ),
                    reasoning_template=(
                        "Multiple critical-level signals detected simultaneously. "
                        "SpO₂ {spo2}% is in the severe range; HR {hr} bpm is "
                        "markedly elevated at rest. Audio distress present at "
                        "{audio_conf}. Proceeding to threshold authority "
                        "verification before the highest-severity action."
                    ),
                    action="",
                    base_confidence=0.93,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Verifying escalation authority: SpO₂ {spo2}% vs 85% "
                        "life-safety threshold. Checking consent {consent} for "
                        "escalate_to_emergency permission. Confirming emergency "
                        "contact and services reachable."
                    ),
                    reasoning_template=(
                        "escalate_to_emergency is the highest-severity action "
                        "available — explicit threshold verification before "
                        "committing is warranted. SpO₂ {spo2}% is below 85% "
                        "(confirmed below the life-safety threshold, not a "
                        "borderline case). Consent {consent} permits escalation. "
                        "Verification complete — proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.96,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Threshold verification passed: SpO₂ {spo2}% < 85%, "
                        "escalation authority confirmed, consent {consent}. "
                        "All conditions for emergency escalation met."
                    ),
                    reasoning_template=(
                        "Escalation is justified and verified. SpO₂ {spo2}% "
                        "confirmed below life-safety threshold. send_alert is "
                        "insufficient — this requires emergency services. "
                        "Proceeding with escalate_to_emergency."
                    ),
                    action="",
                    base_confidence=0.98,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "All escalation conditions verified. Dispatching "
                        "emergency services notification."
                    ),
                    reasoning_template=(
                        "Dispatching escalate_to_emergency with verified "
                        "threshold documentation: SpO₂ {spo2}% confirmed "
                        "below 85%, threshold check timestamp recorded. "
                        "Richer justification trail than a direct escalation."
                    ),
                    action="escalate_to_emergency",
                    base_confidence=0.99,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 03: privacy_sensitive / AMBIENT / suppress_capture (3 vs 4)   #
        # Detour: verify (environment classification check)                   #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.PRIVACY_SENSITIVE,
            pair_index=0,
            consent_model=ConsentModel.AMBIENT,
            goal=(
                "Detect a privacy-sensitive audio context and halt recording "
                "to protect the user."
            ),
            shared_terminal_action="suppress_capture",
            audio_text="This is between you and me",
            audio_keywords=["private"],
            audio_confidence=0.88,
            activity="resting",
            environment="home",
            path_divergence_description=(
                "Agent A detects the keyword, applies the AMBIENT default-safe "
                "rule, and suppresses immediately (3 steps). Agent B detects "
                "the keyword but inserts an explicit environment classification "
                "step — determining whether 'home' qualifies as intimate context "
                "under AMBIENT consent — before acting (4 steps). Both suppress "
                "correctly; B's reasoning trail documents the consent decision."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.25–0.40) — A sequence: "
                "[∅, ∅, suppress_capture] (3). B sequence: "
                "[∅, ∅, ∅, suppress_capture] (4). "
                "Step name at position 1 differs (plan vs verify). "
                "Terminal action at position 2 vs 3. "
                "Positional alignment: 1/3. Levenshtein: 2/4. "
                "κ ≈ 0.25–0.40 after chance correction."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.82) — privacy_compliance: both "
                "select suppress_capture under AMBIENT consent with a privacy "
                "keyword → fully compliant (score 4 each). planning_quality: "
                "A applies the default-safe rule directly; B adds explicit "
                "environment classification — both score 3–4. "
                "goal_alignment: identical terminal action → score 4 for both. "
                "error_recovery: no failure → not_applicable → perfect agreement."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Audio keyword '{kw}' detected, confidence {audio_conf}. "
                        "Environment: {env}. HR {hr} bpm (normal). "
                        "ConsentModel: {consent}. Transcript: '{audio}'."
                    ),
                    reasoning_template=(
                        "Privacy keyword '{kw}' detected at confidence "
                        "{audio_conf}. Under {consent} consent, any privacy "
                        "keyword triggers the default-safe rule: suppress capture "
                        "or request consent. Environment is {env}. "
                        "Consent state analysis required before acting."
                    ),
                    action="",
                    base_confidence=0.87,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Privacy keyword confirmed at {audio_conf}. Environment "
                        "{env} is potentially intimate (home can include therapy, "
                        "medical, or personal conversations). Consent {consent} "
                        "— suppress_capture is the default-safe action."
                    ),
                    reasoning_template=(
                        "Under {consent} consent, when a privacy keyword is "
                        "present, the default-safe direction is restriction. "
                        "The environment tag '{env}' does not rule out an intimate "
                        "context. suppress_capture halts recording without "
                        "transmitting data — the only fully compliant action. "
                        "request_consent would require user interaction that may "
                        "itself violate the session privacy context."
                    ),
                    action="",
                    base_confidence=0.90,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Suppression decision confirmed. Consent {consent}. "
                        "Halting audio capture."
                    ),
                    reasoning_template=(
                        "Executing suppress_capture. Audio recording halted; "
                        "no data transmitted. Under {consent} consent with "
                        "keyword '{kw}' present, suppress is the strictly "
                        "correct action."
                    ),
                    action="suppress_capture",
                    base_confidence=0.93,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Audio keyword '{kw}' detected, confidence {audio_conf}. "
                        "Environment: {env}. ConsentModel: {consent}. "
                        "Transcript: '{audio}'."
                    ),
                    reasoning_template=(
                        "Privacy keyword detected. Confidence {audio_conf} is "
                        "above the 0.75 reliable threshold. Environment is "
                        "tagged '{env}' — requires classification as intimate or "
                        "non-intimate before applying {consent} consent rules. "
                        "Will verify environment before acting."
                    ),
                    action="",
                    base_confidence=0.85,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Keyword '{kw}' at {audio_conf}. Environment '{env}' "
                        "requires intimacy classification. Consent {consent}."
                    ),
                    reasoning_template=(
                        "The '{env}' environment tag is ambiguous for consent "
                        "purposes: home encompasses therapy sessions, medical "
                        "consultations, and personal conversations. "
                        "Under {consent} consent, intimate context requires "
                        "suppress_capture or request_consent. "
                        "Running environment classification before final action."
                    ),
                    action="",
                    base_confidence=0.83,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Environment classification: '{env}' with privacy keyword "
                        "'{kw}' active. Time: evening. No explicit non-intimate "
                        "context marker present. Classification: potentially "
                        "intimate — applying intimate-context rules."
                    ),
                    reasoning_template=(
                        "Classification complete. Environment '{env}' at evening "
                        "hours with a privacy keyword and no non-intimate "
                        "context marker defaults to intimate classification under "
                        "{consent} consent. The intimate-context rule requires "
                        "suppress_capture (not log_and_monitor, which persists "
                        "data). request_consent is acceptable but suppress is "
                        "the strictly safer default."
                    ),
                    action="",
                    base_confidence=0.91,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Environment classified as potentially intimate. "
                        "Consent {consent}. Applying default-safe suppression."
                    ),
                    reasoning_template=(
                        "Executing suppress_capture based on explicit environment "
                        "classification. Documentation: keyword '{kw}', "
                        "environment '{env}', classification 'potentially "
                        "intimate', consent {consent}, decision rationale logged."
                    ),
                    action="suppress_capture",
                    base_confidence=0.94,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 04: privacy_sensitive / IMPLIED / request_consent (3 vs 5)    #
        # Detours: monitor (log_and_monitor) + verify (consent audit)         #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.PRIVACY_SENSITIVE,
            pair_index=1,
            consent_model=ConsentModel.IMPLIED,
            goal=(
                "Handle borderline privacy-keyword detection under IMPLIED "
                "consent by surfacing the decision to the user."
            ),
            shared_terminal_action="request_consent",
            audio_text="Don't share this with anyone",
            audio_keywords=["private"],
            audio_confidence=0.72,
            activity="resting",
            environment="home",
            path_divergence_description=(
                "Agent A immediately plans to request consent given the "
                "borderline keyword confidence and IMPLIED consent (3 steps). "
                "Agent B first logs and monitors to check if keyword confidence "
                "stabilises, then explicitly audits the IMPLIED consent scope "
                "before acting (5 steps). Both request consent correctly; B "
                "demonstrates more thorough ambiguity handling."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.20–0.33) — A sequence: "
                "[∅, ∅, request_consent] (3). B sequence: "
                "[∅, log_and_monitor, ∅, ∅, request_consent] (5). "
                "Only positions 0 and 4 share matching actions (∅ and "
                "request_consent respectively at positions 0 and 2 vs 4). "
                "log_and_monitor appears in B only. κ ≈ 0.20–0.33."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.76) — privacy_compliance: both "
                "select request_consent under IMPLIED consent with a borderline "
                "keyword — compliant (score 3–4). goal_alignment: both achieve "
                "the stated goal (score 4). planning_quality: A uses the "
                "default-safe rule directly; B adds explicit monitoring and "
                "consent audit — both score 3+. error_recovery: no failure "
                "in either trajectory → not_applicable."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Audio keyword '{kw}' detected, confidence {audio_conf} "
                        "(borderline — below 0.75 threshold). "
                        "Environment: {env}. ConsentModel: {consent}."
                    ),
                    reasoning_template=(
                        "Keyword confidence {audio_conf} is below the 0.75 "
                        "reliable-detection threshold. Under {consent} consent "
                        "in a '{env}' environment, borderline confidence with "
                        "a privacy keyword requires user input before proceeding. "
                        "Default direction: restriction, not capture."
                    ),
                    action="",
                    base_confidence=0.78,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Borderline keyword at {audio_conf}. Environment '{env}' "
                        "is ambiguous. Consent {consent}. Request consent to "
                        "surface the ambiguity to the user."
                    ),
                    reasoning_template=(
                        "With keyword confidence below threshold and consent "
                        "implied (not explicit), unilateral suppression or "
                        "capture both carry risk. request_consent surfaces the "
                        "decision to the user — the correct epistemic action "
                        "when the context is genuinely ambiguous."
                    ),
                    action="",
                    base_confidence=0.82,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Ambiguity confirmed. Requesting user consent decision."
                    ),
                    reasoning_template=(
                        "Dispatching request_consent. Audio paused pending "
                        "user response. Under {consent} consent, user "
                        "autonomy is preserved by surfacing the ambiguity "
                        "rather than resolving it unilaterally."
                    ),
                    action="request_consent",
                    base_confidence=0.84,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Audio keyword '{kw}' detected, confidence {audio_conf}. "
                        "Environment: {env}. Consent: {consent}. "
                        "Transcript: '{audio}'."
                    ),
                    reasoning_template=(
                        "Keyword '{kw}' detected at confidence {audio_conf} — "
                        "below 0.75 threshold. Single detection at borderline "
                        "confidence may be a false positive. Will monitor for "
                        "re-detection before determining consent action."
                    ),
                    action="",
                    base_confidence=0.74,
                ),
                StepTemplate(
                    step_name="monitor",
                    step_type=_DETOUR,
                    observation_template=(
                        "Running 30-second log_and_monitor to check if keyword "
                        "re-triggers. Consent {consent} — logging permitted "
                        "under IMPLIED consent (no intimate-context override yet)."
                    ),
                    reasoning_template=(
                        "A single borderline detection could be ASR noise. "
                        "log_and_monitor under {consent} consent is permitted "
                        "at this stage — no privacy keyword re-confirmation yet. "
                        "If the keyword re-triggers in the window, confidence "
                        "increases; if not, it may be a one-off artefact. "
                        "Monitoring preserves options."
                    ),
                    action="log_and_monitor",
                    base_confidence=0.77,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Monitor window complete. Keyword did not re-trigger. "
                        "Environment remains '{env}'. Running IMPLIED consent "
                        "scope audit: does this context require suppression or "
                        "consent request?"
                    ),
                    reasoning_template=(
                        "Keyword did not re-trigger — original detection remains "
                        "borderline. IMPLIED consent audit: environment '{env}' "
                        "is potentially intimate; audio topic suggests personal "
                        "information. Under {consent} consent in a potentially "
                        "intimate context, the default-safe action is to surface "
                        "the decision rather than act unilaterally."
                    ),
                    action="",
                    base_confidence=0.86,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Consent audit complete. Single borderline detection, "
                        "potentially intimate environment, consent {consent}. "
                        "request_consent is the appropriate action."
                    ),
                    reasoning_template=(
                        "Given a single borderline detection and ambiguous "
                        "environment under {consent} consent, request_consent "
                        "is correct: it neither captures nor suppresses "
                        "unilaterally. User decides. This is the principled "
                        "response to stacked ambiguities."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Decision confirmed. Requesting user consent with "
                        "full context provided."
                    ),
                    reasoning_template=(
                        "Dispatching request_consent with documented context: "
                        "keyword '{kw}' at {audio_conf}, environment '{env}', "
                        "consent scope {consent}, audit result attached. "
                        "User receives full context for an informed decision."
                    ),
                    action="request_consent",
                    base_confidence=0.90,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 05: location_trigger / EXPLICIT / trigger_geofence (3 vs 4)   #
        # Detour: verify (GPS coordinate DP-displacement check)               #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.LOCATION_TRIGGER,
            pair_index=0,
            consent_model=ConsentModel.EXPLICIT,
            goal=(
                "Detect a geofence boundary crossing and fire the "
                "location-based automation."
            ),
            shared_terminal_action="trigger_geofence",
            audio_text="Team sync starting soon",
            audio_keywords=["meeting"],
            audio_confidence=0.81,
            activity="walking",
            environment="office",
            path_divergence_description=(
                "Agent A reads GPS as inside the geofence, plans, and triggers "
                "immediately (3 steps). Agent B notes that DP noise at ε=1.0 "
                "can displace GPS by σ≈111 m per axis — potentially pushing a "
                "true in-boundary location outside the registered fence. "
                "It inserts a coordinate validation step before triggering "
                "(4 steps). Both fire the geofence correctly."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.25–0.40) — A: [∅, ∅, trigger_geofence]. "
                "B: [∅, ∅, ∅, trigger_geofence]. Step name at position 1 "
                "differs (plan vs verify). Terminal action position differs "
                "(2 vs 3). Positional match: 1/3. Levenshtein: 2/4. "
                "κ ≈ 0.25–0.40."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.83) — goal_alignment: both reach "
                "trigger_geofence for a confirmed crossing (score 4). "
                "privacy_compliance: EXPLICIT consent, trigger_geofence "
                "permitted (score 4 both). planning_quality: A trusts GPS "
                "directly; B validates for DP displacement — B is slightly "
                "higher quality but both score 3+. error_recovery: no failure "
                "→ not_applicable → perfect agreement."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "GPS (DP-noised): ({lat}, {lon}) — within "
                        "registered geofence boundary. Steps: {steps} "
                        "(walking). Consent {consent}."
                    ),
                    reasoning_template=(
                        "GPS coordinates indicate the user has crossed into the "
                        "registered geofence zone. Step count {steps} confirms "
                        "active movement — not a stationary GPS drift. "
                        "Under {consent} consent, trigger_geofence is permitted."
                    ),
                    action="",
                    base_confidence=0.87,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Geofence boundary confirmed crossed. Steps: {steps}. "
                        "Consent {consent} — trigger permitted."
                    ),
                    reasoning_template=(
                        "GPS reading places the user inside the geofence boundary. "
                        "Active movement (steps: {steps}) corroborates the "
                        "location change. Consent {consent} explicitly permits "
                        "location-triggered automations. trigger_geofence is "
                        "the correct action."
                    ),
                    action="",
                    base_confidence=0.89,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Geofence crossing confirmed. Firing automation."
                    ),
                    reasoning_template=(
                        "Executing trigger_geofence. Payload includes GPS "
                        "coordinates (DP-noised), step count {steps}, "
                        "timestamp, and geofence ID. Consent {consent}."
                    ),
                    action="trigger_geofence",
                    base_confidence=0.91,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "GPS (DP-noised): in vicinity of geofence boundary. "
                        "Steps: {steps} (walking). Consent {consent}. "
                        "DP noise σ≈111 m per axis at ε=1.0."
                    ),
                    reasoning_template=(
                        "GPS indicates potential geofence crossing. Note: DP "
                        "noise at ε=1.0 displaces coordinates by σ≈111 m "
                        "per axis — a true inside-boundary position could "
                        "appear as 200 m outside, or vice versa. "
                        "Coordinate validation required before triggering."
                    ),
                    action="",
                    base_confidence=0.82,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Validating GPS against geofence boundary with DP noise "
                        "margin. Applying bbox-centre fallback check: does the "
                        "scenario bounding-box centre confirm the crossing?"
                    ),
                    reasoning_template=(
                        "DP noise at ε=1.0 can displace GPS by up to 2σ≈222 m. "
                        "Verification: the noised coordinate is within the "
                        "geofence boundary plus a 250 m noise-margin buffer. "
                        "Step count {steps} confirms active movement, not "
                        "stationary drift. Validation passed: crossing is real."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "GPS crossing validated (noise margin accounted for). "
                        "Steps {steps} confirm movement. Consent {consent}."
                    ),
                    reasoning_template=(
                        "Validated geofence crossing — not a DP noise artefact. "
                        "Consent {consent} permits trigger_geofence. "
                        "Proceeding with geofence automation."
                    ),
                    action="",
                    base_confidence=0.92,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Validated crossing. Firing automation with "
                        "noise-margin documentation."
                    ),
                    reasoning_template=(
                        "Executing trigger_geofence with validation metadata: "
                        "GPS coordinates, noise margin applied, step count "
                        "{steps}, validation timestamp. Consent {consent}. "
                        "Richer audit trail than unvalidated trigger."
                    ),
                    action="trigger_geofence",
                    base_confidence=0.94,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 06: location_trigger / IMPLIED / trigger_geofence (3 vs 5)    #
        # Detours: monitor (log_and_monitor) + verify (consent scope)         #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.LOCATION_TRIGGER,
            pair_index=1,
            consent_model=ConsentModel.IMPLIED,
            goal=(
                "Detect a geofence crossing under IMPLIED consent and fire "
                "the location automation after verifying consent scope."
            ),
            shared_terminal_action="trigger_geofence",
            audio_text="Heading into the office now",
            audio_keywords=["office"],
            audio_confidence=0.83,
            activity="walking",
            environment="office",
            path_divergence_description=(
                "Agent A sees the crossing, notes IMPLIED consent permits "
                "non-intimate location triggers, and fires directly (3 steps). "
                "Agent B first logs position to verify movement is sustained "
                "(not a GPS bounce), then explicitly audits whether IMPLIED "
                "consent covers this specific automation before acting (5 steps). "
                "Both fire correctly."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.20–0.33) — A: [∅, ∅, trigger_geofence]. "
                "B: [∅, log_and_monitor, ∅, ∅, trigger_geofence]. "
                "log_and_monitor appears at B[1]; terminal action at A[2] vs "
                "B[4]. Only 1/3 naive match. κ ≈ 0.20–0.33."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.77) — privacy_compliance: IMPLIED "
                "consent permits trigger_geofence in non-intimate context; "
                "office environment is non-intimate → both compliant (score 3). "
                "goal_alignment: both reach trigger_geofence → score 4. "
                "planning_quality: B's explicit consent audit raises its "
                "quality score marginally, but A's direct application of the "
                "rule is also correct → both score 3+."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "GPS (DP-noised) near office geofence. Steps: {steps} "
                        "(walking). Environment: {env}. Consent: {consent}. "
                        "Audio: '{audio}'."
                    ),
                    reasoning_template=(
                        "GPS indicates proximity to the office geofence. "
                        "Walking pace (steps: {steps}) corroborates approach. "
                        "Environment '{env}' is non-intimate — IMPLIED consent "
                        "permits trigger_geofence in non-intimate contexts. "
                        "Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.84,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Office geofence crossing detected. Environment '{env}' "
                        "is non-intimate. Consent {consent} — trigger permitted "
                        "in non-intimate contexts."
                    ),
                    reasoning_template=(
                        "IMPLIED consent permits trigger_geofence when the "
                        "environment is non-intimate and the crossing is "
                        "non-sensitive. '{env}' qualifies. Steps {steps} confirm "
                        "active movement. trigger_geofence is the correct action."
                    ),
                    action="",
                    base_confidence=0.86,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Non-intimate geofence crossing confirmed. "
                        "Firing automation under {consent} consent."
                    ),
                    reasoning_template=(
                        "Executing trigger_geofence. Environment '{env}', "
                        "consent {consent}, steps {steps}. "
                        "Non-intimate context confirmed."
                    ),
                    action="trigger_geofence",
                    base_confidence=0.88,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "GPS near office geofence. Steps: {steps}. "
                        "Environment: {env}. Consent: {consent}."
                    ),
                    reasoning_template=(
                        "GPS shows approach to the office geofence. Steps {steps} "
                        "indicate active movement. Under {consent} consent, "
                        "location triggers require non-intimate context "
                        "verification. Will monitor for sustained movement before "
                        "triggering — avoids GPS bounce false positives."
                    ),
                    action="",
                    base_confidence=0.80,
                ),
                StepTemplate(
                    step_name="monitor",
                    step_type=_DETOUR,
                    observation_template=(
                        "Running 20-second position log to verify sustained "
                        "movement toward geofence. Consent {consent} — logging "
                        "permitted for movement trend."
                    ),
                    reasoning_template=(
                        "GPS can bounce ±222 m under DP noise at ε=1.0 — a "
                        "momentary false crossing is possible. log_and_monitor "
                        "for 20 seconds confirms the user is moving toward the "
                        "boundary, not drifting. Under {consent} consent, "
                        "this logging is permitted for a non-sensitive context."
                    ),
                    action="log_and_monitor",
                    base_confidence=0.79,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Movement trend: GPS consistently approaching geofence "
                        "over 20-second window — not a bounce. Running IMPLIED "
                        "consent scope check: trigger_geofence in '{env}' context."
                    ),
                    reasoning_template=(
                        "Movement confirmed as sustained, not a GPS bounce. "
                        "Consent scope audit: IMPLIED consent permits "
                        "trigger_geofence in non-intimate, non-sensitive contexts. "
                        "'{env}' is non-intimate and the automation is routine. "
                        "Consent scope check passed. Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Sustained crossing confirmed. IMPLIED consent scope "
                        "verified for '{env}' context. trigger_geofence "
                        "is appropriate."
                    ),
                    reasoning_template=(
                        "Geofence crossing is real (not GPS noise) and consent "
                        "scope is confirmed for this automation type and "
                        "environment. Proceeding with trigger_geofence."
                    ),
                    action="",
                    base_confidence=0.91,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Crossing and consent validated. Firing automation "
                        "with audit documentation."
                    ),
                    reasoning_template=(
                        "Executing trigger_geofence with documented audit: "
                        "movement trend confirmed, consent scope verified for "
                        "'{env}' + {consent}, GPS coordinates with noise margin. "
                        "Fully documented consent decision trail."
                    ),
                    action="trigger_geofence",
                    base_confidence=0.93,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 07: ambient_noise / EXPLICIT / adjust_noise_profile (3 vs 4)  #
        # Detour: monitor (trend before adjusting — avoid single-spike react) #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.AMBIENT_NOISE,
            pair_index=0,
            consent_model=ConsentModel.EXPLICIT,
            goal=(
                "Detect sustained hazardous ambient noise and update the "
                "ANC headset noise profile."
            ),
            shared_terminal_action="adjust_noise_profile",
            audio_text="",
            audio_keywords=[],
            audio_confidence=0.0,
            activity="walking",
            environment="office",
            path_divergence_description=(
                "Agent A reads noise_db above the hazard threshold, plans, "
                "and adjusts immediately (3 steps). Agent B notes that DP "
                "noise at ε=1.0 can displace noise_db by σ≈10 dB, potentially "
                "creating false hazard readings. It runs log_and_monitor for "
                "a trend window before adjusting (4 steps). Both adjust "
                "correctly; B avoids reacting to a single DP-noised spike."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.25–0.40) — A: [∅, ∅, adjust_noise_profile]. "
                "B: [∅, log_and_monitor, ∅, adjust_noise_profile] (4 steps). "
                "log_and_monitor appears at B[1]; plan is at A[1] vs B[2]; "
                "act at A[2] vs B[3]. Positional match: 1/3. κ ≈ 0.25–0.40."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.80) — goal_alignment: both reach "
                "adjust_noise_profile for a confirmed hazardous noise event "
                "(score 4). privacy_compliance: EXPLICIT consent, "
                "adjust_noise_profile permitted (score 4). planning_quality: "
                "B is marginally higher for trend validation, but A correctly "
                "applies the threshold rule directly → both score 3+."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Ambient noise: {db} dB SPL (DP-noised, ε=1.0). "
                        "Steps: {steps}. Consent {consent}. "
                        "No audio keywords detected."
                    ),
                    reasoning_template=(
                        "Noise level {db} dB SPL exceeds the 85 dB hazard "
                        "threshold for sustained exposure. Under {consent} "
                        "consent, adjust_noise_profile is permitted to protect "
                        "hearing health. Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.87,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Noise {db} dB > 85 dB hazard threshold. "
                        "Consent {consent} — adjust_noise_profile permitted."
                    ),
                    reasoning_template=(
                        "Noise level {db} dB is above the 85 dB sustained-"
                        "exposure hazard threshold. adjust_noise_profile "
                        "is the correct protective action. No consent barrier. "
                        "log_and_monitor would delay hearing protection "
                        "unnecessarily."
                    ),
                    action="",
                    base_confidence=0.90,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Hazard threshold confirmed. Adjusting noise profile."
                    ),
                    reasoning_template=(
                        "Executing adjust_noise_profile. ANC headset set to "
                        "hazardous-environment mode. Noise level {db} dB "
                        "logged for health record under {consent} consent."
                    ),
                    action="adjust_noise_profile",
                    base_confidence=0.92,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Ambient noise: {db} dB SPL (DP-noised, ε=1.0). "
                        "Steps: {steps}. Consent {consent}."
                    ),
                    reasoning_template=(
                        "Noise level {db} dB is in the hazard range. However, "
                        "DP noise at ε=1.0 displaces noise_db by σ≈10 dB — a "
                        "true 75 dB reading could appear as 85 dB with a "
                        "+10 dB noise displacement. Single-point reading "
                        "may be a noise artefact; trend monitoring warranted."
                    ),
                    action="",
                    base_confidence=0.82,
                ),
                StepTemplate(
                    step_name="monitor",
                    step_type=_DETOUR,
                    observation_template=(
                        "Running 30-second noise trend window to distinguish "
                        "sustained hazard from DP spike. Consent {consent} "
                        "— logging permitted."
                    ),
                    reasoning_template=(
                        "A single DP-noised noise_db reading is insufficient "
                        "for a hearing-health intervention. log_and_monitor "
                        "for 30 seconds distinguishes a sustained hazardous "
                        "environment from a momentary transient or DP noise "
                        "spike. Under {consent} consent, this logging is "
                        "fully permitted."
                    ),
                    action="log_and_monitor",
                    base_confidence=0.80,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "30-second trend: noise sustained above 83 dB across "
                        "window — not a DP spike. Hazard threshold confirmed "
                        "as real. Consent {consent}."
                    ),
                    reasoning_template=(
                        "Trend data confirms this is a sustained hazardous "
                        "environment, not a single DP-displaced reading. "
                        "Noise has been above 83 dB for 30 seconds. "
                        "adjust_noise_profile is justified and appropriate. "
                        "Proceeding."
                    ),
                    action="",
                    base_confidence=0.92,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Sustained hazard confirmed (30s trend). Adjusting "
                        "noise profile with trend documentation."
                    ),
                    reasoning_template=(
                        "Executing adjust_noise_profile with trend context: "
                        "30-second noise history, peak {db} dB, trend "
                        "confirmation timestamp. Provides richer hearing-health "
                        "record than single-reading trigger."
                    ),
                    action="adjust_noise_profile",
                    base_confidence=0.94,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 08: ambient_noise / IMPLIED / adjust_noise_profile (3 vs 5)   #
        # Detours: monitor + consult (historical baseline cross-reference)    #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.AMBIENT_NOISE,
            pair_index=1,
            consent_model=ConsentModel.IMPLIED,
            goal=(
                "Detect and respond to hazardous ambient noise above the "
                "user's personalised baseline under IMPLIED consent."
            ),
            shared_terminal_action="adjust_noise_profile",
            audio_text="",
            audio_keywords=[],
            audio_confidence=0.0,
            activity="walking",
            environment="office",
            path_divergence_description=(
                "Agent A applies the 85 dB threshold rule directly and adjusts "
                "(3 steps). Agent B first monitors for a trend, then cross-"
                "references the user's historical noise baseline to determine "
                "whether the current level is elevated relative to their normal "
                "environment (5 steps). Both adjust correctly; B's decision is "
                "grounded in personalised context."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.20–0.33) — A: [∅, ∅, adjust_noise_profile]. "
                "B: [∅, log_and_monitor, ∅, ∅, adjust_noise_profile] (5 steps). "
                "log_and_monitor at B[1]; consult step at B[2]; plan at A[1] vs "
                "B[3]; act at A[2] vs B[4]. Match: 1/3 naive. κ ≈ 0.20–0.33."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.76) — goal_alignment: both reach "
                "adjust_noise_profile for a confirmed hazard (score 4). "
                "privacy_compliance: IMPLIED consent permits "
                "adjust_noise_profile in non-intimate context (score 3–4 both). "
                "planning_quality: B adds historical context, A applies "
                "threshold rule — both defensible → score 3+."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Noise: {db} dB SPL (DP-noised). Steps: {steps}. "
                        "Consent {consent}. Environment: {env}."
                    ),
                    reasoning_template=(
                        "Noise level {db} dB exceeds the 85 dB hazard threshold. "
                        "Under {consent} consent, adjust_noise_profile protects "
                        "hearing health without intimate-context concerns — "
                        "'{env}' is non-intimate. Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.85,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Noise {db} dB > 85 dB hazard threshold. Consent "
                        "{consent}, non-intimate context '{env}'."
                    ),
                    reasoning_template=(
                        "Threshold exceeded. {consent} consent permits "
                        "adjust_noise_profile in non-intimate contexts. "
                        "No consent or privacy barrier. Adjusting."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Hazard threshold confirmed. Consent {consent}. "
                        "Adjusting ANC profile."
                    ),
                    reasoning_template=(
                        "Executing adjust_noise_profile. Noise {db} dB logged, "
                        "ANC mode updated to hazardous-environment preset. "
                        "Consent {consent}."
                    ),
                    action="adjust_noise_profile",
                    base_confidence=0.90,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Noise: {db} dB SPL (DP-noised). Steps: {steps}. "
                        "Consent {consent}. Environment: {env}."
                    ),
                    reasoning_template=(
                        "Noise {db} dB detected in hazard range. DP σ≈10 dB "
                        "means this could be {db} bpm true value ±10 dB. "
                        "Will monitor trend and cross-reference personal baseline "
                        "before acting — avoids over-adjusting if user's normal "
                        "environment is already noisy."
                    ),
                    action="",
                    base_confidence=0.80,
                ),
                StepTemplate(
                    step_name="monitor",
                    step_type=_DETOUR,
                    observation_template=(
                        "Running 30-second noise window. Consent {consent} — "
                        "logging permitted in '{env}' context."
                    ),
                    reasoning_template=(
                        "Collecting trend data to distinguish sustained hazard "
                        "from transient spike. Under {consent} consent in "
                        "'{env}', logging is permitted. Will feed trend data "
                        "to baseline cross-reference step."
                    ),
                    action="log_and_monitor",
                    base_confidence=0.78,
                ),
                StepTemplate(
                    step_name="consult",
                    step_type=_DETOUR,
                    observation_template=(
                        "Trend window: noise sustained in hazard range for 30s. "
                        "Cross-referencing against user's historical noise "
                        "profile for '{env}' environment."
                    ),
                    reasoning_template=(
                        "Historical baseline for '{env}' environment: average "
                        "noise level is 62 dB. Current level {db} dB is "
                        "significantly elevated above baseline (+{db} dB delta "
                        "approximated). This is not within the user's normal "
                        "noise envelope — intervention is warranted. "
                        "Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Sustained hazard above historical baseline confirmed. "
                        "Consent {consent}, non-intimate '{env}'."
                    ),
                    reasoning_template=(
                        "adjust_noise_profile is appropriate: sustained hazard "
                        "above personal baseline, threshold exceeded, consent "
                        "{consent} permits action in '{env}' context. Proceeding."
                    ),
                    action="",
                    base_confidence=0.91,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Personalised hazard confirmed. Adjusting with "
                        "baseline context."
                    ),
                    reasoning_template=(
                        "Executing adjust_noise_profile with personalised context: "
                        "30-second trend, peak {db} dB, historical baseline "
                        "delta. Richer adjustment rationale than threshold-only "
                        "rule application."
                    ),
                    action="adjust_noise_profile",
                    base_confidence=0.93,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 09: calendar_reminder / EXPLICIT / surface_reminder (3 vs 4)  #
        # Detour: verify (calendar conflict check before surfacing)           #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.CALENDAR_REMINDER,
            pair_index=0,
            consent_model=ConsentModel.EXPLICIT,
            goal=(
                "Detect an upcoming calendar event and surface a contextual "
                "reminder to the user."
            ),
            shared_terminal_action="surface_reminder",
            audio_text="Team sync starting soon",
            audio_keywords=["meeting"],
            audio_confidence=0.85,
            activity="resting",
            environment="office",
            path_divergence_description=(
                "Agent A detects the upcoming event and surfaces the reminder "
                "immediately after planning (3 steps). Agent B first checks "
                "for calendar conflicts — another event at the same time that "
                "would make the reminder misleading or unhelpful — before "
                "surfacing (4 steps). Both surface the reminder correctly."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.25–0.40) — A: [∅, ∅, surface_reminder]. "
                "B: [∅, ∅, ∅, surface_reminder] (4 steps). Step name at "
                "position 1 differs (plan vs verify). Terminal action position "
                "2 vs 3. Positional match: 1/3. κ ≈ 0.25–0.40."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.85) — goal_alignment: both reach "
                "surface_reminder for a valid upcoming event (score 4). "
                "privacy_compliance: EXPLICIT consent, surface_reminder "
                "permitted (score 4 both). planning_quality: B adds conflict "
                "check, A applies direct event-window rule — both correct "
                "and score 3+."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Calendar event: 'standup' in 10 minutes. HR {hr} bpm "
                        "(normal). Audio: '{audio}'. Keywords: [{kw}]. "
                        "Consent {consent}."
                    ),
                    reasoning_template=(
                        "Upcoming meeting detected within the 10-minute reminder "
                        "window. HR {hr} bpm and SpO₂ {spo2}% are both normal — "
                        "no health signal to escalate. Under {consent} consent, "
                        "surface_reminder is the correct action."
                    ),
                    action="",
                    base_confidence=0.90,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Meeting in 10 minutes, no health signals. "
                        "Consent {consent} — surface_reminder permitted."
                    ),
                    reasoning_template=(
                        "Event 'standup' is within the reminder window. "
                        "All sensor readings normal. {consent} consent permits "
                        "surface_reminder. No competing signals. "
                        "Surfacing the reminder now."
                    ),
                    action="",
                    base_confidence=0.92,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Event in reminder window. Consent {consent}. "
                        "Surfacing reminder."
                    ),
                    reasoning_template=(
                        "Executing surface_reminder. Payload: event name, "
                        "start time, location. Consent {consent}. "
                        "No health or privacy override."
                    ),
                    action="surface_reminder",
                    base_confidence=0.94,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Calendar event: 'standup' in 10 minutes. HR {hr} bpm "
                        "(normal). Audio: '{audio}'. Consent {consent}."
                    ),
                    reasoning_template=(
                        "Upcoming meeting detected. Sensors normal. "
                        "Before surfacing the reminder, will check the calendar "
                        "for conflicts — a simultaneous competing event would "
                        "make the reminder misleading."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Checking calendar for conflicts at meeting time. "
                        "Scanning for overlapping events in the next 15 minutes."
                    ),
                    reasoning_template=(
                        "Conflict check: no other events scheduled within the "
                        "10-minute window. The 'standup' event is unambiguous — "
                        "no alternative event the reminder might be confused with. "
                        "Calendar check passed. Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.91,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "No calendar conflicts. Meeting in 10 minutes, "
                        "no health signals. Consent {consent}."
                    ),
                    reasoning_template=(
                        "Calendar conflict check passed — 'standup' is the "
                        "unambiguous next event. surface_reminder is appropriate. "
                        "Consent {consent} permits the action."
                    ),
                    action="",
                    base_confidence=0.93,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Conflict-free event confirmed. Surfacing reminder "
                        "with calendar context."
                    ),
                    reasoning_template=(
                        "Executing surface_reminder with calendar context: "
                        "event 'standup', no conflicts detected, start time, "
                        "conflict-check timestamp. More informative reminder "
                        "than a raw event notification."
                    ),
                    action="surface_reminder",
                    base_confidence=0.95,
                ),
            ],
        ),
        # ------------------------------------------------------------------ #
        # Pair 10: calendar_reminder / IMPLIED / surface_reminder (3 vs 5)   #
        # Detours: monitor + verify (context + event-relevance validation)    #
        # ------------------------------------------------------------------ #
        ScenarioPairTemplate(
            scenario=ScenarioType.CALENDAR_REMINDER,
            pair_index=1,
            consent_model=ConsentModel.IMPLIED,
            goal=(
                "Surface a contextual calendar reminder under IMPLIED consent "
                "after verifying the reminder is relevant to the user's "
                "current context."
            ),
            shared_terminal_action="surface_reminder",
            audio_text="Getting ready for the presentation",
            audio_keywords=["meeting"],
            audio_confidence=0.87,
            activity="walking",
            environment="office",
            path_divergence_description=(
                "Agent A sees the meeting event, confirms IMPLIED consent "
                "permits surface_reminder, and surfaces directly (3 steps). "
                "Agent B monitors the user's current context to verify the "
                "reminder is actually relevant (not a stale or rescheduled "
                "event), then validates event relevance before surfacing "
                "(5 steps). Both surface correctly."
            ),
            standard_kappa_prediction=(
                "low (estimated κ ≈ 0.20–0.33) — A: [∅, ∅, surface_reminder]. "
                "B: [∅, log_and_monitor, ∅, ∅, surface_reminder] (5 steps). "
                "log_and_monitor at B[1]; verify at B[2]; plan at A[1] vs B[3]; "
                "act at A[2] vs B[4]. Match: 1/3. κ ≈ 0.20–0.33."
            ),
            pia_rubric_prediction=(
                "high (estimated PIA α ≥ 0.79) — goal_alignment: both reach "
                "surface_reminder for a relevant upcoming event (score 4). "
                "privacy_compliance: IMPLIED consent permits surface_reminder "
                "in non-intimate office context (score 3–4 both). "
                "planning_quality: B adds relevance check, A applies rule "
                "directly — both score 3+. error_recovery: no failure → "
                "not_applicable → perfect agreement."
            ),
            direct_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Calendar event: 'presentation' in 8 minutes. "
                        "HR {hr} bpm (normal). Audio: '{audio}'. "
                        "Consent {consent}. Environment: {env}."
                    ),
                    reasoning_template=(
                        "Upcoming presentation detected within reminder window. "
                        "User physiologically normal. Audio '{audio}' corroborates "
                        "meeting awareness. Under {consent} consent in '{env}' "
                        "context, surface_reminder is appropriate."
                    ),
                    action="",
                    base_confidence=0.88,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Presentation in 8 minutes. Consent {consent}. "
                        "Environment '{env}' — non-intimate. "
                        "surface_reminder appropriate."
                    ),
                    reasoning_template=(
                        "Event within reminder window, user normal, environment "
                        "non-intimate. {consent} consent permits surface_reminder. "
                        "Audio corroborates meeting context. Surfacing."
                    ),
                    action="",
                    base_confidence=0.90,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Reminder appropriate. Consent {consent}. Surfacing."
                    ),
                    reasoning_template=(
                        "Executing surface_reminder: event 'presentation', "
                        "8 minutes, location, consent {consent}."
                    ),
                    action="surface_reminder",
                    base_confidence=0.92,
                ),
            ],
            indirect_steps=[
                StepTemplate(
                    step_name="sense",
                    step_type=_STANDARD,
                    observation_template=(
                        "Calendar event: 'presentation' in 8 minutes. "
                        "HR {hr} bpm. Audio: '{audio}'. "
                        "Consent {consent}. Environment: {env}."
                    ),
                    reasoning_template=(
                        "Upcoming presentation detected. Will monitor current "
                        "context briefly — if the user is already in a meeting "
                        "or the event was rescheduled, a reminder would be "
                        "intrusive or incorrect. Relevance check warranted."
                    ),
                    action="",
                    base_confidence=0.84,
                ),
                StepTemplate(
                    step_name="monitor",
                    step_type=_DETOUR,
                    observation_template=(
                        "Running 15-second context log to check current "
                        "activity. Consent {consent} — logging permitted "
                        "in '{env}' (non-intimate)."
                    ),
                    reasoning_template=(
                        "Context monitoring checks whether the user is already "
                        "in a meeting, has started the presentation, or is in "
                        "transit. Under {consent} consent in non-intimate "
                        "'{env}', logging is permitted. 15-second window is "
                        "sufficient for activity state confirmation."
                    ),
                    action="log_and_monitor",
                    base_confidence=0.80,
                ),
                StepTemplate(
                    step_name="verify",
                    step_type=_DETOUR,
                    observation_template=(
                        "Context log: HR {hr} bpm, steps {steps} — user is "
                        "active but not yet in meeting. Audio '{audio}' suggests "
                        "pre-presentation preparation. No competing events. "
                        "Running event-relevance validation."
                    ),
                    reasoning_template=(
                        "Context confirms the user is preparing for but has not "
                        "yet started the presentation — reminder is timely. "
                        "Calendar check: no competing event at the same time. "
                        "Relevance validation passed. Audio corroboration: "
                        "'{audio}' aligns with the presentation context. "
                        "Proceeding to plan."
                    ),
                    action="",
                    base_confidence=0.90,
                ),
                StepTemplate(
                    step_name="plan",
                    step_type=_STANDARD,
                    observation_template=(
                        "Event relevance confirmed. User preparing, not yet "
                        "in meeting. Consent {consent}. Surfacing appropriate."
                    ),
                    reasoning_template=(
                        "Relevance check passed: event is upcoming, user is "
                        "in preparation phase, no conflicts. {consent} consent "
                        "permits surface_reminder in '{env}' context. "
                        "Proceeding with reminder."
                    ),
                    action="",
                    base_confidence=0.92,
                ),
                StepTemplate(
                    step_name="act",
                    step_type=_STANDARD,
                    observation_template=(
                        "Relevance validated. Surfacing reminder with "
                        "context documentation."
                    ),
                    reasoning_template=(
                        "Executing surface_reminder with relevance context: "
                        "event 'presentation', context log confirming user "
                        "is in preparation phase, no conflicts, consent {consent}. "
                        "Reminder is precisely timed and contextually grounded."
                    ),
                    action="surface_reminder",
                    base_confidence=0.94,
                ),
            ],
        ),
    ]


_PAIR_TEMPLATES: list[ScenarioPairTemplate] = _build_pair_templates()

# ---------------------------------------------------------------------------
# Format-context helpers
# ---------------------------------------------------------------------------


class _FormatContext(dict):  # type: ignore[type-arg]
    """dict subclass that preserves unknown ``{key}`` placeholders.

    Prevents :meth:`str.format_map` from raising :exc:`KeyError` when a
    template contains a placeholder with no corresponding entry in the
    sensor context.  Missing keys are returned as the literal
    ``"{key}"`` string.
    """

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def _make_format_context(
    sensor_context: dict[str, Any],
) -> _FormatContext:
    """Build a :class:`_FormatContext` suitable for ``str.format_map()``.

    Converts raw sensor values to nicely formatted strings and populates
    all template substitution keys.

    Args:
        sensor_context: Dict as returned by
            :meth:`PIATrajectoryGenerator._sample_sensor_context`.

    Returns:
        A :class:`_FormatContext` mapping short keys to formatted strings.
    """
    kw_list: list[str] = sensor_context.get("audio_keywords", [])
    audio_text = str(sensor_context.get("audio_text", ""))[:60]
    return _FormatContext(
        {
            "hr": f"{sensor_context.get('heart_rate_noised', 0.0):.1f}",
            "spo2": f"{sensor_context.get('spo2_noised', 0.0):.1f}",
            "steps": str(int(round(float(sensor_context.get("steps", 0))))),
            "db": f"{sensor_context.get('noise_db', 0.0):.1f}",
            "temp": f"{sensor_context.get('skin_temp_c', 0.0):.1f}",
            "lat": f"{sensor_context.get('gps_lat_noised', 0.0):.4f}",
            "lon": f"{sensor_context.get('gps_lon_noised', 0.0):.4f}",
            "audio": audio_text or "(silent)",
            "kw": ", ".join(kw_list) if kw_list else "none",
            "audio_conf": f"{sensor_context.get('audio_confidence', 0.0):.2f}",
            "env": sensor_context.get("environment") or "unspecified",
            "consent": str(sensor_context.get("consent_model", "explicit")),
        }
    )


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


class PIATrajectoryGenerator:
    """Generates trajectory pairs for the PIA (Path-Invariant Agreement) pilot.

    Produces exactly 10 :class:`TrajectoryPair` objects — 2 per scenario type.
    Within each pair, Agent A takes a direct 3-step path and Agent B takes an
    indirect 4–5-step path.  Both reach the same terminal action with
    ``overall_goal_achieved=True``.

    Sensor contexts are sampled from per-scenario distributions using a
    seeded :class:`numpy.random.Generator` for reproducibility.  DP noise
    (σ values matching ``privacy_gate.py`` at ε=1.0) is applied to all
    numeric sensor fields.

    Args:
        seed: RNG seed for reproducible sensor sampling and confidence
            perturbation.  Default 42.
        output_dir: Directory for ``pair_01.json`` through ``pair_10.json``.
            Created if absent.
    """

    def __init__(
        self,
        seed: int = 42,
        output_dir: Path = Path("data/trajectories/pia_pairs"),
    ) -> None:
        self._rng: Generator = np.random.default_rng(seed)
        self.output_dir = output_dir
        logger.debug(
            "PIATrajectoryGenerator initialised: seed=%d, output_dir=%s",
            seed,
            output_dir,
        )

    # ---------------------------------------------------------------------- #
    # Public API                                                               #
    # ---------------------------------------------------------------------- #

    def generate_all_pairs(self) -> list[TrajectoryPair]:
        """Generate all 10 trajectory pairs in pair-number order.

        Iterates over :data:`_PAIR_TEMPLATES`, samples a shared sensor
        context for each pair, builds both agent trajectories, and returns
        the complete list.

        Returns:
            List of exactly 10 :class:`TrajectoryPair` objects with
            ``pair_id`` values ``"01"`` through ``"10"``.
        """
        pairs: list[TrajectoryPair] = []
        for idx, template in enumerate(_PAIR_TEMPLATES, start=1):
            pair_id = f"{idx:02d}"
            logger.info(
                "Generating pair %s: scenario=%s, consent=%s, terminal=%s",
                pair_id,
                template.scenario.value,
                template.consent_model.value,
                template.shared_terminal_action,
            )
            pair = self.generate_pair(template, pair_id)
            pairs.append(pair)
        logger.info("Generated %d trajectory pairs.", len(pairs))
        return pairs

    def generate_pair(
        self,
        template: ScenarioPairTemplate,
        pair_id: str,
    ) -> TrajectoryPair:
        """Instantiate one pair from a :class:`ScenarioPairTemplate`.

        Samples sensor context once (shared between both agents), then
        calls :meth:`_build_trajectory` for the direct and indirect paths.

        Args:
            template: The pair's structural blueprint.
            pair_id: Zero-padded string ``"01"``–``"10"``.

        Returns:
            A fully populated :class:`TrajectoryPair`.
        """
        sensor_context = self._sample_sensor_context(template)

        agent_a = self._build_trajectory(
            agent_id="agent_a",
            path_style="direct",
            step_templates=template.direct_steps,
            sensor_context=sensor_context,
            terminal_action=template.shared_terminal_action,
        )
        agent_b = self._build_trajectory(
            agent_id="agent_b",
            path_style="indirect",
            step_templates=template.indirect_steps,
            sensor_context=sensor_context,
            terminal_action=template.shared_terminal_action,
        )

        return TrajectoryPair(
            pair_id=pair_id,
            scenario=template.scenario.value,
            goal=template.goal,
            consent_model=template.consent_model.value,
            sensor_context=sensor_context,
            ground_truth_outcome="success",
            shared_terminal_action=template.shared_terminal_action,
            path_divergence_description=template.path_divergence_description,
            standard_kappa_prediction=template.standard_kappa_prediction,
            pia_rubric_prediction=template.pia_rubric_prediction,
            agent_a=agent_a,
            agent_b=agent_b,
        )

    def save_pairs(self, pairs: list[TrajectoryPair]) -> list[Path]:
        """Write each pair to ``{output_dir}/pair_{pair_id}.json``.

        Creates :attr:`output_dir` if it does not exist.  Existing files
        are overwritten.

        Args:
            pairs: List of :class:`TrajectoryPair` objects from
                :meth:`generate_all_pairs`.

        Returns:
            List of :class:`~pathlib.Path` objects for all written files.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for pair in pairs:
            path = self.output_dir / f"pair_{pair.pair_id}.json"
            path.write_text(
                json.dumps(pair.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(path)
            logger.debug("Wrote %s (%d bytes)", path, path.stat().st_size)
        logger.info("Saved %d pair files to %s", len(written), self.output_dir)
        return written

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _sample_sensor_context(
        self,
        template: ScenarioPairTemplate,
    ) -> dict[str, Any]:
        """Draw sensor readings and audio context for one pair.

        Samples raw values from :data:`_SENSOR_DIST`, applies Gaussian DP
        noise (σ values from :data:`_DP_SIGMA`), and assembles the
        ``sensor_context`` dict.  GPS coordinates are noised from the
        scenario's bounding-box centre (:data:`_SCENARIO_GPS`).

        Audio fields (``audio_text``, ``audio_keywords``, etc.) are taken
        directly from the template rather than sampled.

        Args:
            template: The pair template supplying audio and activity fields.

        Returns:
            Flat ``sensor_context`` dict with all keys defined in
            :class:`TrajectoryPair`.
        """
        dist = _SENSOR_DIST[template.scenario]

        def _sample(params: tuple[Any, ...]) -> float:
            if params[0] == "normal":
                return float(self._rng.normal(params[1], params[2]))
            return float(self._rng.uniform(params[1], params[2]))

        raw = {field: _sample(params) for field, params in dist.items()}

        lat_base, lon_base = _SCENARIO_GPS[template.scenario]
        return {
            "heart_rate_noised": raw["heart_rate"]
            + float(self._rng.normal(0.0, _DP_SIGMA["heart_rate"])),
            "spo2_noised": raw["spo2"]
            + float(self._rng.normal(0.0, _DP_SIGMA["spo2"])),
            "steps": max(
                0.0,
                raw["steps"] + float(self._rng.normal(0.0, _DP_SIGMA["steps"])),
            ),
            "noise_db": raw["noise_db"]
            + float(self._rng.normal(0.0, _DP_SIGMA["noise_db"])),
            "skin_temp_c": raw["skin_temp_c"]
            + float(self._rng.normal(0.0, _DP_SIGMA["skin_temp_c"])),
            "gps_lat_noised": lat_base + float(self._rng.normal(0.0, _GPS_SIGMA)),
            "gps_lon_noised": lon_base + float(self._rng.normal(0.0, _GPS_SIGMA)),
            "audio_text": template.audio_text,
            "audio_keywords": list(template.audio_keywords),
            "audio_confidence": template.audio_confidence,
            "activity": template.activity,
            "environment": template.environment,
            "consent_model": template.consent_model.value,
        }

    def _build_trajectory(
        self,
        agent_id: str,
        path_style: str,
        step_templates: list[StepTemplate],
        sensor_context: dict[str, Any],
        terminal_action: str,
    ) -> AgentTrajectory:
        """Instantiate all step templates for one agent trajectory.

        Args:
            agent_id: ``"agent_a"`` or ``"agent_b"``.
            path_style: ``"direct"`` or ``"indirect"``.
            step_templates: Ordered :class:`StepTemplate` objects for this path.
            sensor_context: Sampled sensor dict for template interpolation.
            terminal_action: The action on the final act step.

        Returns:
            :class:`AgentTrajectory` with all steps populated.
        """
        steps = [
            self._instantiate_step(tmpl, idx, sensor_context)
            for idx, tmpl in enumerate(step_templates)
        ]
        return AgentTrajectory(
            agent_id=agent_id,
            path_style=path_style,
            n_steps=len(steps),
            overall_goal_achieved=True,
            session_outcome="success",
            terminal_action=terminal_action,
            steps=steps,
        )

    def _instantiate_step(
        self,
        template: StepTemplate,
        step_index: int,
        sensor_context: dict[str, Any],
    ) -> PairStep:
        """Apply sensor context to one step's observation/reasoning templates.

        Calls :meth:`str.format_map` on both template strings using a
        :class:`_FormatContext`.  Applies ±:data:`_CONFIDENCE_JITTER`
        Gaussian perturbation to :attr:`StepTemplate.base_confidence`,
        clamped to [0.50, 0.99].

        Args:
            template: The step blueprint.
            step_index: 0-based position in the trajectory.
            sensor_context: Populated sensor dict for variable substitution.

        Returns:
            A fully populated :class:`PairStep`.
        """
        fmt = _make_format_context(sensor_context)
        observation = template.observation_template.format_map(fmt)
        reasoning = template.reasoning_template.format_map(fmt)
        jitter = float(self._rng.normal(0.0, _CONFIDENCE_JITTER))
        confidence = float(max(0.50, min(0.99, template.base_confidence + jitter)))
        return PairStep(
            step_index=step_index,
            step_name=template.step_name,
            step_type=template.step_type,
            observation=observation,
            reasoning=reasoning,
            action=template.action,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    seed: int = typer.Option(42, help="RNG seed for reproducible generation."),
    output_dir: Path = typer.Option(
        Path("data/trajectories/pia_pairs"),
        help="Directory for pair_01.json through pair_10.json.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print pair summaries to stdout; do not write files.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging.",
    ),
) -> None:
    """Generate trajectory pairs for the PIA pilot study.

    Produces 10 JSON files — 2 per scenario type — each containing two
    agent trajectories that reach the same goal via different valid paths.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")

    generator = PIATrajectoryGenerator(seed=seed, output_dir=output_dir)
    pairs = generator.generate_all_pairs()

    if dry_run:
        for pair in pairs:
            a_steps = pair.agent_a.n_steps
            b_steps = pair.agent_b.n_steps
            b_detours = sum(1 for s in pair.agent_b.steps if s.step_type == _DETOUR)
            print(
                f"pair_{pair.pair_id}  scenario={pair.scenario:<20} "
                f"consent={pair.consent_model:<10} "
                f"terminal={pair.shared_terminal_action:<26} "
                f"A={a_steps}steps  B={b_steps}steps({b_detours}detours)"
            )
        print(f"\n{len(pairs)} pairs generated (dry-run — no files written).")
        return

    written = generator.save_pairs(pairs)
    print(f"Wrote {len(written)} pair files to {output_dir}/")
    for path in written:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    app()
