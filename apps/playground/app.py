"""
AgentTrace Playground — Streamlit UI
Paste a JSON agent trajectory → get PIA score + per-dimension breakdown.
Imports real PIAScorer and TrajectoryScorer from src/. No logic duplicated.
"""

import sys
import json
from pathlib import Path

# Ensure repo root is on the path so src/ imports resolve
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.annotation.pia_scorer import PIAScorer
from src.eval.trajectory_scorer import TrajectoryScorer
from src.data.wearable_generator import (
    AgentAction,
    AudioTranscript,
    SensorData,
    TrajectoryStep,
    WearableLog,
    ScenarioType,
)
from src.data.privacy_gate import ConsentModel

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentTrace Playground",
    page_icon="🔍",
    layout="wide",
)

# ── example fixture (mirrors minimal_trajectory from tests/) ─────────────────
def _build_example_log() -> WearableLog:
    """Builds the same minimal_trajectory fixture used in the test suite."""
    sensor = SensorData(
        heart_rate=102.0,
        spo2=94.0,
        steps=3200.0,
        gps_lat=37.7749,
        gps_lon=-122.4194,
        noise_db=55.0,
        skin_temp_c=36.8,
    )
    transcript = AudioTranscript(
        text="I feel dizzy and my chest hurts",
        language="en-US",
        confidence=0.95,
        duration_s=3.5,
        keywords_detected=["dizzy", "chest hurts"],
    )
    steps = [
        TrajectoryStep(
            step_index=0,
            step_name="sense",
            observation="Heart rate 102bpm, SpO2 94%",
            reasoning="Heart rate elevated at 102bpm, SpO2 at 94%",
            action="",
            confidence=0.9,
        ),
        TrajectoryStep(
            step_index=1,
            step_name="plan",
            observation="User reported dizziness and chest pain",
            reasoning="Combined sensor + audio signals indicate health alert",
            action="",
            confidence=0.85,
        ),
        TrajectoryStep(
            step_index=2,
            step_name="act",
            observation="Sending alert to emergency contact",
            reasoning="Health alert criteria met — escalating to send_alert",
            action=AgentAction.SEND_ALERT,
            confidence=0.92,
        ),
    ]
    return WearableLog(
        log_id="playground-example-001",
        timestamp="2026-01-15T14:30:00Z",
        scenario_type=ScenarioType.HEALTH_ALERT,
        consent_model=ConsentModel.EXPLICIT,
        sensor_data=sensor,
        audio_transcript=transcript,
        context_metadata={},
        trajectory=steps,
        ground_truth_action=AgentAction.SEND_ALERT,
    )


def _log_from_dict(raw: dict) -> WearableLog:
    """Reconstruct a WearableLog from a to_dict()-compatible dict."""
    sensor = SensorData(**raw["sensor_data"])
    transcript = AudioTranscript(**raw["audio_transcript"])
    steps = [TrajectoryStep(**s) for s in raw["trajectory"]]
    return WearableLog(
        log_id=raw["log_id"],
        timestamp=raw["timestamp"],
        scenario_type=ScenarioType(raw["scenario_type"]),
        consent_model=ConsentModel(raw["consent_model"]),
        sensor_data=sensor,
        audio_transcript=transcript,
        context_metadata=raw.get("context_metadata", {}),
        trajectory=steps,
        ground_truth_action=raw["ground_truth_action"],
    )


EXAMPLE_LOG = _build_example_log()
EXAMPLE_JSON = json.dumps(EXAMPLE_LOG.to_dict(), indent=2)


# ── scoring helper ───────────────────────────────────────────────────────────
def score_log(log: WearableLog) -> dict:
    pia_scorer = PIAScorer()
    traj_scorer = TrajectoryScorer()

    dim_scores = pia_scorer.score_trajectory(log)
    traj_score = traj_scorer.score_trajectory(log)

    return {
        "pia_dimensions": dim_scores,
        "trajectory": traj_score,
    }


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🔍 AgentTrace Playground")
st.caption(
    "Paste a JSON agent trajectory → get PIA score + per-dimension breakdown. "
    "Powered by real PIAScorer and TrajectoryScorer from `src/`."
)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("Trajectory Input")

    if st.button("Load example trajectory", use_container_width=True):
        st.session_state["trajectory_text"] = EXAMPLE_JSON

    trajectory_text = st.text_area(
        label="Paste JSON trajectory here",
        value=st.session_state.get("trajectory_text", ""),
        height=420,
        placeholder='{"log_id": "...", "scenario": "...", "trajectory": [...]}',
        key="trajectory_text",
    )

    run = st.button("▶  Score trajectory", type="primary", use_container_width=True)

with col_output:
    st.subheader("Results")

    if run:
        if not trajectory_text.strip():
            st.error("Paste a JSON trajectory first.")
        else:
            try:
                raw = json.loads(trajectory_text)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON — {e}")
                st.stop()

            try:
                log = _log_from_dict(raw)
            except Exception as e:
                st.error(f"JSON parsed but trajectory schema invalid — {e}")
                st.stop()

            with st.spinner("Scoring…"):
                try:
                    results = score_log(log)
                except Exception as e:
                    st.error(f"Scoring failed — {e}")
                    st.stop()

            dim = results["pia_dimensions"]
            traj = results["trajectory"]

            # ── PIA Dimensions ──
            st.markdown("#### PIA Dimension Scores")
            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.metric("Planning Quality", f"{dim.planning_quality:.2f}")
            d_col2.metric(
                "Error Recovery",
                f"{dim.error_recovery:.2f}" if dim.error_recovery is not None else "N/A",
            )
            d_col3.metric("Goal Alignment", f"{dim.goal_alignment:.2f}")

            st.divider()

            # ── TrajectoryScore breakdown ──
            st.markdown("#### Trajectory Scores")

            # Robustly display whatever fields TrajectoryScore exposes
            traj_dict = traj.to_dict()

            score_cols = st.columns(min(len(traj_dict), 4))
            for i, (k, v) in enumerate(traj_dict.items()):
                col = score_cols[i % len(score_cols)]
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    col.metric(label, f"{v:.2f}")
                elif isinstance(v, bool):
                    col.metric(label, "✓" if v else "✗")
                elif v is not None:
                    col.metric(label, str(v))

            # ── Failure flags ──
            flags = [k for k, v in traj_dict.items() if isinstance(v, bool) and not v]
            if flags:
                st.divider()
                st.markdown("#### ⚠ Failure Flags")
                for f in flags:
                    st.warning(f.replace("_", " ").title())
            else:
                st.success("No failure flags detected.")
    else:
        st.info("Load the example or paste a trajectory, then click **Score trajectory**.")
