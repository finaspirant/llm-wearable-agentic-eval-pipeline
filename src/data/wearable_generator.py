"""Synthetic wearable sensor/audio log generator.

Generates realistic wearable device data across 5 scenario types:
- health_alert: elevated heart rate, fall detection, SpO2 drops
- privacy_sensitive: private conversation detected, intimate context
- location_trigger: geofence entry/exit, commute pattern change
- ambient_noise: background audio classification, noise levels
- calendar_reminder: schedule conflicts, meeting preparation

Each log includes sensor_data, audio_transcript, context_metadata,
and a 3-step agent trajectory (sense → plan → act).

Differential privacy is applied via privacy_gate.py before output.

CLI: python -m src.data.wearable_generator --count 100
"""

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import typer
from numpy.random import Generator

from src.data.privacy_gate import ConsentModel, PrivacyGate

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="wearable-generator",
    help="Generate synthetic wearable sensor/audio logs with differential privacy.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ScenarioType(StrEnum):
    """Five canonical wearable/ambient AI scenario types."""

    HEALTH_ALERT = "health_alert"
    PRIVACY_SENSITIVE = "privacy_sensitive"
    LOCATION_TRIGGER = "location_trigger"
    AMBIENT_NOISE = "ambient_noise"
    CALENDAR_REMINDER = "calendar_reminder"


class AgentAction(StrEnum):
    """Discrete action space for the wearable agent's act step."""

    SEND_ALERT = "send_alert"
    SUPPRESS_CAPTURE = "suppress_capture"
    TRIGGER_GEOFENCE = "trigger_geofence"
    ADJUST_NOISE_PROFILE = "adjust_noise_profile"
    SURFACE_REMINDER = "surface_reminder"
    LOG_AND_MONITOR = "log_and_monitor"
    REQUEST_CONSENT = "request_consent"
    ESCALATE_TO_EMERGENCY = "escalate_to_emergency"


# ---------------------------------------------------------------------------
# Data models (dataclasses for zero external dependency at import time)
# ---------------------------------------------------------------------------


@dataclass
class SensorData:
    """Raw sensor readings from a wearable device.

    All numeric fields represent values *before* differential privacy
    is applied.  The ``noised`` variants (populated by the generator)
    carry the ε-noised versions that flow downstream.

    Args:
        heart_rate: Beats per minute (BPM), range 30–220.
        spo2: Blood oxygen saturation percentage, range 70–100.
        steps: Step count in the current 30-second epoch.
        gps_lat: WGS-84 latitude, degrees (raw — not for downstream use).
        gps_lon: WGS-84 longitude, degrees (raw — not for downstream use).
        noise_db: Ambient sound level in dB SPL.
        skin_temp_c: Skin-surface temperature in Celsius.
        heart_rate_noised: DP-noised heart rate (populated post-privacy-gate).
        spo2_noised: DP-noised SpO2 (populated post-privacy-gate).
        steps_noised: DP-noised step count (populated post-privacy-gate).
        noise_db_noised: DP-noised ambient dB (populated post-privacy-gate).
        gps_lat_noised: DP-noised latitude — use this downstream, not gps_lat.
        gps_lon_noised: DP-noised longitude — use this downstream, not gps_lon.
    """

    heart_rate: float
    spo2: float
    steps: float
    gps_lat: float
    gps_lon: float
    noise_db: float
    skin_temp_c: float
    heart_rate_noised: float = 0.0
    spo2_noised: float = 0.0
    steps_noised: float = 0.0
    noise_db_noised: float = 0.0
    gps_lat_noised: float = 0.0
    gps_lon_noised: float = 0.0


@dataclass
class AudioTranscript:
    """Short audio segment metadata captured by the wearable microphone.

    Args:
        text: ASR transcript of the audio segment (may be empty).
        language: BCP-47 language tag (e.g. ``"en-US"``).
        confidence: ASR confidence score in [0, 1].
        duration_s: Segment duration in seconds.
        keywords_detected: High-signal keywords flagged by the on-device
            classifier (e.g. ``["chest pain", "help"]``).
    """

    text: str
    language: str
    confidence: float
    duration_s: float
    keywords_detected: list[str]


@dataclass
class TrajectoryStep:
    """One step in the sense → plan → act agent trajectory.

    Args:
        step_index: 0 = sense, 1 = plan, 2 = act.
        step_name: Human-readable label (``"sense"``, ``"plan"``, ``"act"``).
        observation: What the agent observed at this step.
        reasoning: Chain-of-thought or policy explanation.
        action: The discrete action taken (only populated for act step).
        confidence: Agent confidence in this step's output, [0, 1].
    """

    step_index: int
    step_name: str
    observation: str
    reasoning: str
    action: str
    confidence: float


@dataclass
class WearableLog:
    """One complete synthetic wearable event log.

    Combines sensor data, audio transcript, context metadata, and a
    3-step agent trajectory.  Serialises to JSON via :meth:`to_dict`.

    Args:
        log_id: UUID4 string uniquely identifying this log.
        timestamp: ISO-8601 UTC timestamp.
        scenario_type: One of the five :class:`ScenarioType` values.
        consent_model: Applicable :class:`ConsentModel` for this session.
        sensor_data: Raw + noised sensor readings.
        audio_transcript: ASR transcript metadata.
        context_metadata: Freeform key-value context (device model,
            user demographics bracket, etc.).
        trajectory: Three-step agent trajectory.
        ground_truth_action: The "gold" action label for evaluation.
    """

    log_id: str
    timestamp: str
    scenario_type: ScenarioType
    consent_model: ConsentModel
    sensor_data: SensorData
    audio_transcript: AudioTranscript
    context_metadata: dict[str, Any]
    trajectory: list[TrajectoryStep]
    ground_truth_action: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise the log to a JSON-safe dict."""
        d = asdict(self)
        # asdict converts Enum members to their values via str — but
        # since ScenarioType / ConsentModel inherit str, .value is the str.
        d["scenario_type"] = self.scenario_type.value
        d["consent_model"] = self.consent_model.value
        return d

    def to_json(self) -> str:
        """Serialise the log to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Per-scenario distribution tables
# ---------------------------------------------------------------------------

# Each entry: (low, high, distribution)  — all floats
# Distribution tag: "uniform" | "normal(mean, std)"
_SCENARIO_DISTRIBUTIONS: dict[ScenarioType, dict[str, Any]] = {
    ScenarioType.HEALTH_ALERT: {
        "heart_rate": ("normal", 145.0, 20.0),  # elevated
        "spo2": ("normal", 91.0, 3.0),  # hypoxic range
        "steps": ("uniform", 0.0, 150.0),
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
        "steps": ("uniform", 80.0, 600.0),
        "noise_db": ("uniform", 55.0, 80.0),
        "skin_temp_c": ("normal", 36.8, 0.4),
    },
    ScenarioType.AMBIENT_NOISE: {
        "heart_rate": ("normal", 72.0, 8.0),
        "spo2": ("normal", 97.5, 1.0),
        "steps": ("uniform", 0.0, 200.0),
        "noise_db": ("uniform", 70.0, 105.0),  # loud environment
        "skin_temp_c": ("normal", 36.7, 0.3),
    },
    ScenarioType.CALENDAR_REMINDER: {
        "heart_rate": ("normal", 80.0, 10.0),
        "spo2": ("normal", 98.0, 1.0),
        "steps": ("uniform", 0.0, 300.0),
        "noise_db": ("uniform", 40.0, 65.0),
        "skin_temp_c": ("normal", 36.7, 0.3),
    },
}

_SCENARIO_AUDIO: dict[ScenarioType, dict[str, Any]] = {
    ScenarioType.HEALTH_ALERT: {
        "texts": [
            "I feel dizzy, my chest hurts",
            "Help, I can't breathe properly",
            "Something doesn't feel right",
            "",  # silent — fall detection only
        ],
        "keywords": ["chest pain", "dizzy", "help", "breathe", "fall"],
    },
    ScenarioType.PRIVACY_SENSITIVE: {
        "texts": [
            "This is between you and me",
            "Don't share this with anyone",
            "That was a private moment",
            "I'm in a therapy session",
        ],
        "keywords": ["private", "confidential", "secret", "therapy", "intimate"],
    },
    ScenarioType.LOCATION_TRIGGER: {
        "texts": [
            "Just arrived at the office",
            "Heading home now",
            "I'm near the gym",
            "Almost at the coffee shop",
        ],
        "keywords": ["arrived", "office", "home", "gym", "near"],
    },
    ScenarioType.AMBIENT_NOISE: {
        "texts": [
            "Traffic is really loud today",
            "The concert is great",
            "Construction noise all morning",
            "",  # purely sensor-based
        ],
        "keywords": ["loud", "noise", "traffic", "music", "construction"],
    },
    ScenarioType.CALENDAR_REMINDER: {
        "texts": [
            "I have a meeting in ten minutes",
            "Don't forget the standup",
            "Reminder: dentist appointment at 3",
            "Team sync starting soon",
        ],
        "keywords": ["meeting", "reminder", "appointment", "standup", "calendar"],
    },
}

_SCENARIO_CONTEXT: dict[ScenarioType, dict[str, list[Any]]] = {
    ScenarioType.HEALTH_ALERT: {
        "device_model": ["Garmin Venu 3", "Apple Watch Ultra 2", "Polar H10"],
        "activity": ["resting", "light_walk", "unknown"],
        "alert_severity": ["moderate", "high", "critical"],
    },
    ScenarioType.PRIVACY_SENSITIVE: {
        "device_model": ["Meta Ray-Ban 2", "Humane AI Pin", "Plaud Note"],
        "environment": ["home", "therapy_office", "vehicle"],
        "privacy_flag": [
            "conversation_detected",
            "intimate_context",
            "healthcare_setting",
        ],
    },
    ScenarioType.LOCATION_TRIGGER: {
        "device_model": ["Apple Watch Series 10", "Garmin Venu 3", "Fitbit Sense 3"],
        "geofence_type": ["work", "home", "gym", "poi"],
        "transition": ["entry", "exit", "dwell"],
    },
    ScenarioType.AMBIENT_NOISE: {
        "device_model": [
            "Bose QuietComfort Ultra",
            "Sony WH-1000XM6",
            "Apple AirPods Pro 3",
        ],
        "noise_class": ["traffic", "music_live", "construction", "crowd", "nature"],
        "anc_active": [True, False],
    },
    ScenarioType.CALENDAR_REMINDER: {
        "device_model": [
            "Apple Watch Series 10",
            "Samsung Galaxy Watch 7",
            "Garmin Venu 3",
        ],
        "meeting_type": ["standup", "1on1", "team_sync", "external", "medical"],
        "minutes_until": [5, 10, 15, 30],
    },
}

_SCENARIO_TRAJECTORY: dict[ScenarioType, list[dict[str, str]]] = {
    ScenarioType.HEALTH_ALERT: [
        {
            "observation": (
                "Heart rate {hr:.0f} bpm; SpO2 {spo2:.1f}%; no motion detected."
            ),
            "reasoning": (
                "Elevated HR + depressed SpO2 pattern consistent with"
                " cardiac or respiratory event."
            ),
            "action": "",
        },
        {
            "observation": (
                "Cross-referenced 5-min HR trend: persistent elevation."
                " Audio keyword 'chest pain' detected."
            ),
            "reasoning": (
                "Dual-modality confirmation raises alert confidence"
                " above 0.85 threshold."
            ),
            "action": "",
        },
        {
            "observation": "Alert threshold exceeded; emergency contact on file.",
            "reasoning": (
                "Escalation policy: sustained critical signal"
                " → send alert + log for clinical review."
            ),
            "action": AgentAction.SEND_ALERT.value,
        },
    ],
    ScenarioType.PRIVACY_SENSITIVE: [
        {
            "observation": "Audio keyword 'private' detected; location: {environment}.",
            "reasoning": "On-device classifier flags potential intimate context.",
            "action": "",
        },
        {
            "observation": (
                "Consent model is {consent}; no explicit recording permission"
                " for this context."
            ),
            "reasoning": (
                "Privacy policy requires suppressing capture when intimate"
                " context is detected without explicit consent."
            ),
            "action": "",
        },
        {
            "observation": (
                "Recording window flagged; user has not granted EXPLICIT consent."
            ),
            "reasoning": (
                "Default-safe action: suppress further audio capture and notify user."
            ),
            "action": AgentAction.SUPPRESS_CAPTURE.value,
        },
    ],
    ScenarioType.LOCATION_TRIGGER: [
        {
            "observation": (
                "GPS ({lat:.4f}, {lon:.4f}); geofence"
                " '{geofence_type}' transition: {transition}."
            ),
            "reasoning": (
                "Geofence engine detected boundary crossing with high confidence."
            ),
            "action": "",
        },
        {
            "observation": (
                "Commute pattern: {transition} at {geofence_type}"
                " consistent with 7-day baseline."
            ),
            "reasoning": (
                "Context switch warranted; check for pending"
                " location-triggered automations."
            ),
            "action": "",
        },
        {
            "observation": (
                "Location automation queue: 1 pending task"
                " for '{geofence_type}' {transition}."
            ),
            "reasoning": (
                "Execute scheduled location trigger and log event"
                " for trajectory analysis."
            ),
            "action": AgentAction.TRIGGER_GEOFENCE.value,
        },
    ],
    ScenarioType.AMBIENT_NOISE: [
        {
            "observation": (
                "Ambient noise: {noise_db:.0f} dB SPL; noise class: '{noise_class}'."
            ),
            "reasoning": (
                "Sound level exceeds conversational threshold;"
                " ANC status: {anc_active}."
            ),
            "action": "",
        },
        {
            "observation": (
                "Prolonged exposure at {noise_db:.0f} dB SPL above WHO 70 dB guideline."
            ),
            "reasoning": (
                "Hearing protection heuristic: >85 dB sustained"
                " → recommend noise profile adjustment."
            ),
            "action": "",
        },
        {
            "observation": "Noise profile update available; ANC headset connected.",
            "reasoning": (
                "Adjust ANC profile to match '{noise_class}' environment;"
                " log exposure for health report."
            ),
            "action": AgentAction.ADJUST_NOISE_PROFILE.value,
        },
    ],
    ScenarioType.CALENDAR_REMINDER: [
        {
            "observation": (
                "Calendar event '{meeting_type}' in {minutes_until} minutes;"
                " current HR {hr:.0f} bpm."
            ),
            "reasoning": "Upcoming commitment detected; assess user readiness.",
            "action": "",
        },
        {
            "observation": (
                "User is active (steps last 30 s: {steps:.0f});"
                " meeting prep window opening."
            ),
            "reasoning": (
                "Surface reminder at optimal interruption window"
                " — gap between activity bursts."
            ),
            "action": "",
        },
        {
            "observation": (
                "Reminder threshold reached; {minutes_until} min to '{meeting_type}'."
            ),
            "reasoning": (
                "Surface contextual reminder with meeting details;"
                " log delivery for IRR evaluation."
            ),
            "action": AgentAction.SURFACE_REMINDER.value,
        },
    ],
}

_SCENARIO_GROUND_TRUTH: dict[ScenarioType, str] = {
    ScenarioType.HEALTH_ALERT: AgentAction.SEND_ALERT.value,
    ScenarioType.PRIVACY_SENSITIVE: AgentAction.SUPPRESS_CAPTURE.value,
    ScenarioType.LOCATION_TRIGGER: AgentAction.TRIGGER_GEOFENCE.value,
    ScenarioType.AMBIENT_NOISE: AgentAction.ADJUST_NOISE_PROFILE.value,
    ScenarioType.CALENDAR_REMINDER: AgentAction.SURFACE_REMINDER.value,
}

# GPS bounding boxes per scenario (lat_min, lat_max, lon_min, lon_max)
_GPS_BOXES: dict[ScenarioType, tuple[float, float, float, float]] = {
    ScenarioType.HEALTH_ALERT: (37.77, 37.79, -122.42, -122.40),  # SF residential
    ScenarioType.PRIVACY_SENSITIVE: (37.78, 37.80, -122.43, -122.41),
    ScenarioType.LOCATION_TRIGGER: (37.33, 37.34, -122.03, -122.02),  # Cupertino
    ScenarioType.AMBIENT_NOISE: (40.75, 40.76, -73.99, -73.98),  # Midtown NYC
    ScenarioType.CALENDAR_REMINDER: (37.38, 37.40, -122.08, -122.07),  # Mountain View
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class WearableLogGenerator:
    """Generates batches of synthetic wearable event logs.

    Each log covers one of five scenario types with realistic sensor
    distributions, an ASR transcript, context metadata, a 3-step agent
    trajectory, and a ground-truth action label.

    Differential privacy (Gaussian, ε=1.0) is applied to numeric sensor
    fields via :class:`~src.data.privacy_gate.PrivacyGate`.

    Args:
        seed: Optional integer seed for reproducible generation.
        epsilon: Privacy budget forwarded to :class:`PrivacyGate`.
    """

    def __init__(
        self,
        seed: int | None = None,
        epsilon: float = 1.0,
    ) -> None:
        self._rng: Generator = np.random.default_rng(seed)
        self._privacy_gate = PrivacyGate(
            epsilon=epsilon,
            rng=np.random.default_rng(seed + 1 if seed is not None else None),
        )
        logger.info(
            "WearableLogGenerator initialised: seed=%s epsilon=%.2f",
            seed,
            epsilon,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        count: int,
        scenario_filter: list[ScenarioType] | None = None,
    ) -> list[WearableLog]:
        """Generate a batch of synthetic wearable logs.

        Args:
            count: Number of logs to generate.  Must be positive.
            scenario_filter: If provided, only generate logs for the
                listed scenario types (sampled uniformly within the
                filter).  Defaults to all five types.  Must not be
                an empty list.

        Returns:
            List of :class:`WearableLog` instances.

        Raises:
            ValueError: If ``count`` ≤ 0 or ``scenario_filter`` is
                an empty list.
        """
        if count <= 0:
            raise ValueError(f"count must be a positive integer, got {count}")
        if scenario_filter is not None and len(scenario_filter) == 0:
            raise ValueError(
                "scenario_filter must not be empty; pass None to use all scenarios"
            )
        scenarios = scenario_filter or list(ScenarioType)
        logs: list[WearableLog] = []
        for i in range(count):
            scenario = scenarios[i % len(scenarios)]
            log = self._generate_one(scenario)
            logs.append(log)
            if (i + 1) % 10 == 0:
                logger.debug("Generated %d / %d logs", i + 1, count)
        logger.info("Generated %d logs total", len(logs))
        return logs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample(self, spec: tuple[Any, ...]) -> float:
        """Sample a float from a distribution spec tuple."""
        dist, *params = spec
        if dist == "normal":
            mean, std = params
            return float(self._rng.normal(mean, std))
        if dist == "uniform":
            low, high = params
            return float(self._rng.uniform(low, high))
        raise ValueError(f"Unknown distribution: {dist!r}")

    @staticmethod
    def _validate_gps(
        lat: float,
        lon: float,
        fallback_lat: float,
        fallback_lon: float,
    ) -> tuple[float, float]:
        """Return ``(lat, lon)`` if valid WGS-84, otherwise the fallback centre.

        A coordinate is considered missing/invalid when it is NaN, ±inf, or
        exactly (0.0, 0.0) — the "Null Island" sentinel that GPS receivers
        emit when they have no fix.

        Args:
            lat: Candidate latitude in degrees.
            lon: Candidate longitude in degrees.
            fallback_lat: Latitude of the scenario's bounding-box centre.
            fallback_lon: Longitude of the scenario's bounding-box centre.

        Returns:
            The original pair if valid, else ``(fallback_lat, fallback_lon)``.
        """
        null_island = lat == 0.0 and lon == 0.0
        invalid = (
            math.isnan(lat)
            or math.isnan(lon)
            or math.isinf(lat)
            or math.isinf(lon)
            or not (-90.0 <= lat <= 90.0)
            or not (-180.0 <= lon <= 180.0)
            or null_island
        )
        if invalid:
            logger.warning(
                "GPS coordinates (%.6f, %.6f) invalid or missing — "
                "falling back to bbox centre (%.6f, %.6f)",
                lat,
                lon,
                fallback_lat,
                fallback_lon,
            )
            return fallback_lat, fallback_lon
        return lat, lon

    def _generate_sensor_data(self, scenario: ScenarioType) -> SensorData:
        """Sample raw sensor readings for the given scenario."""
        dist = _SCENARIO_DISTRIBUTIONS[scenario]
        lat_min, lat_max, lon_min, lon_max = _GPS_BOXES[scenario]

        raw_lat = float(self._rng.uniform(lat_min, lat_max))
        raw_lon = float(self._rng.uniform(lon_min, lon_max))
        centre_lat = (lat_min + lat_max) / 2.0
        centre_lon = (lon_min + lon_max) / 2.0
        gps_lat, gps_lon = self._validate_gps(raw_lat, raw_lon, centre_lat, centre_lon)

        raw = SensorData(
            heart_rate=max(30.0, min(220.0, self._sample(dist["heart_rate"]))),
            spo2=max(70.0, min(100.0, self._sample(dist["spo2"]))),
            steps=max(0.0, self._sample(dist["steps"])),
            gps_lat=gps_lat,
            gps_lon=gps_lon,
            noise_db=max(20.0, self._sample(dist["noise_db"])),
            skin_temp_c=float(
                self._rng.normal(dist["skin_temp_c"][1], dist["skin_temp_c"][2])
            ),
        )

        # Apply differential privacy to all numeric sensor fields.
        # Downstream code must use the *_noised variants — raw values are
        # retained only for ground-truth evaluation and are never exported.
        raw.heart_rate_noised = self._privacy_gate.apply_noise_to_sensor(
            "heart_rate", raw.heart_rate
        )
        raw.spo2_noised = max(
            70.0,
            min(
                100.0,
                self._privacy_gate.apply_noise_to_sensor("spo2", raw.spo2),
            ),
        )
        raw.steps_noised = max(
            0.0,
            self._privacy_gate.apply_noise_to_sensor("steps", raw.steps),
        )
        raw.noise_db_noised = max(
            20.0,
            self._privacy_gate.apply_noise_to_sensor("noise_db", raw.noise_db),
        )
        # GPS is the highest-risk field for re-identification; always noise it.
        raw.gps_lat_noised = self._privacy_gate.apply_noise_to_sensor(
            "gps_lat", raw.gps_lat
        )
        raw.gps_lon_noised = self._privacy_gate.apply_noise_to_sensor(
            "gps_lon", raw.gps_lon
        )
        return raw

    def _generate_audio_transcript(self, scenario: ScenarioType) -> AudioTranscript:
        """Sample a realistic audio transcript for the scenario.

        Handles the empty-transcript edge case explicitly: when the sampled
        text is empty or None (e.g. silent / fall-detection-only events),
        confidence is forced to 0.0 and keyword detection is skipped so that
        downstream annotators never receive a spurious non-zero confidence on
        a blank transcript.
        """
        audio_cfg = _SCENARIO_AUDIO[scenario]
        text_idx = int(self._rng.integers(0, len(audio_cfg["texts"])))
        raw_text: str | None = audio_cfg["texts"][text_idx]

        # Normalise: treat None and whitespace-only as empty
        text = (raw_text or "").strip()

        if not text:
            logger.debug(
                "Empty audio transcript for scenario %s — zeroing confidence",
                scenario,
            )
            return AudioTranscript(
                text="",
                language="en-US",
                confidence=0.0,
                duration_s=float(self._rng.uniform(2.0, 30.0)),
                keywords_detected=[],
            )

        detected = [kw for kw in audio_cfg["keywords"] if kw in text.lower()]
        return AudioTranscript(
            text=text,
            language="en-US",
            confidence=float(self._rng.uniform(0.72, 0.99)),
            duration_s=float(self._rng.uniform(2.0, 30.0)),
            keywords_detected=detected,
        )

    def _generate_context(
        self, scenario: ScenarioType, consent: ConsentModel
    ) -> dict[str, Any]:
        """Sample context metadata dict for the scenario."""
        ctx_cfg = _SCENARIO_CONTEXT[scenario]
        ctx: dict[str, Any] = {"consent_model": consent.value}
        for key, choices in ctx_cfg.items():
            idx = int(self._rng.integers(0, len(choices)))
            ctx[key] = choices[idx]
        return ctx

    def _render_trajectory(
        self,
        scenario: ScenarioType,
        sensor: SensorData,
        context: dict[str, Any],
        consent: ConsentModel,
    ) -> list[TrajectoryStep]:
        """Instantiate the scenario trajectory template with sampled values."""
        tmpl_list = _SCENARIO_TRAJECTORY[scenario]
        steps: list[TrajectoryStep] = []

        fmt_vars: dict[str, Any] = {
            "hr": sensor.heart_rate_noised,
            "spo2": sensor.spo2_noised,
            "steps": sensor.steps_noised,
            "noise_db": sensor.noise_db_noised,
            "lat": sensor.gps_lat_noised,
            "lon": sensor.gps_lon_noised,
            "consent": consent.value,
        }
        # Merge context keys (device_model, geofence_type, etc.)
        fmt_vars.update(context)

        step_names = ["sense", "plan", "act"]
        for i, tmpl in enumerate(tmpl_list):
            obs = tmpl["observation"].format_map(fmt_vars)
            rsn = tmpl["reasoning"].format_map(fmt_vars)
            action = tmpl["action"]

            steps.append(
                TrajectoryStep(
                    step_index=i,
                    step_name=step_names[i],
                    observation=obs,
                    reasoning=rsn,
                    action=action,
                    confidence=float(self._rng.uniform(0.70, 0.97)),
                )
            )
        return steps

    def _generate_one(self, scenario: ScenarioType) -> WearableLog:
        """Generate a single :class:`WearableLog` for the given scenario."""
        consent_choices = [
            ConsentModel.EXPLICIT,
            ConsentModel.IMPLIED,
            ConsentModel.AMBIENT,
        ]
        consent_idx = int(self._rng.integers(0, len(consent_choices)))
        consent = consent_choices[consent_idx]

        sensor = self._generate_sensor_data(scenario)
        audio = self._generate_audio_transcript(scenario)
        context = self._generate_context(scenario, consent)
        trajectory = self._render_trajectory(scenario, sensor, context, consent)

        return WearableLog(
            log_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=UTC).isoformat(),
            scenario_type=scenario,
            consent_model=consent,
            sensor_data=sensor,
            audio_transcript=audio,
            context_metadata=context,
            trajectory=trajectory,
            ground_truth_action=_SCENARIO_GROUND_TRUTH[scenario],
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def generate(
    count: int = typer.Option(100, "--count", "-n", help="Number of logs to generate."),
    output: Path = typer.Option(
        Path("data/raw/wearable_logs.jsonl"),
        "--output",
        "-o",
        help="Output JSONL file path.",
    ),
    seed: int | None = typer.Option(
        None, "--seed", "-s", help="Random seed for reproducibility."
    ),
    scenario_filter: list[str] | None = typer.Option(
        None,
        "--scenario",
        help="Restrict generation to specific scenario types (repeatable).",
    ),
    epsilon: float = typer.Option(
        1.0, "--epsilon", help="Differential privacy budget ε."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable debug logging."
    ),
) -> None:
    """Generate synthetic wearable sensor/audio logs with differential privacy."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    scenario_types: list[ScenarioType] | None = None
    if scenario_filter:
        try:
            scenario_types = [ScenarioType(s) for s in scenario_filter]
        except ValueError as exc:
            valid = [e.value for e in ScenarioType]
            typer.echo(f"Invalid scenario type: {exc}. Valid values: {valid}", err=True)
            raise typer.Exit(code=1) from exc

    generator = WearableLogGenerator(seed=seed, epsilon=epsilon)
    logs = generator.generate_batch(count, scenario_filter=scenario_types)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for log in logs:
            fh.write(log.to_json() + "\n")

    typer.echo(f"Wrote {len(logs)} logs to {output}")


if __name__ == "__main__":
    app()
