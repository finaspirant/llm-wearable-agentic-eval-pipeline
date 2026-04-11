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
import logging

logger = logging.getLogger(__name__)

# TODO(Day 6): Implement WearableLogGenerator class
#   - ScenarioType enum with 5 types
#   - SensorData dataclass (heart_rate, steps, gps, spo2)
#   - AudioTranscript dataclass (text, language, confidence, duration)
#   - WearableLog dataclass combining all fields
#   - generate_batch(count: int) → list[WearableLog]
#   - Realistic distributions per scenario type
#   - CLI via typer: --count, --output, --seed, --scenario-filter
