"""Streamlit demo — Wearable Agentic Eval Pipeline.

Runs the full real pipeline on button click:
  Step 1 — generate synthetic wearable logs (WearableLogGenerator)
  Step 2 — privacy gate (PrivacyGate + ConsentModel)
  Step 3 — HITL trigger evaluation (HITLTriggerEvaluator)
  Step 4 — FACTS grounding score (FACTSGroundingScorer)
  Step 5 — 5-layer TrajectoryScorer + PIA dimensions (AgenticEvaluator)

If any pipeline module is unavailable, mock scores are shown with a
yellow warning banner rather than crashing the app.

Usage::

    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow imports from project root when launched via `streamlit run demo/app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

# ---------------------------------------------------------------------------
# Pipeline module imports — graceful fallback if any module is missing
# ---------------------------------------------------------------------------

_PIPELINE_AVAILABLE = True
_PIPELINE_IMPORT_ERROR: str | None = None

try:
    from demo.pipeline import run_eval_pipeline
    from src.data.wearable_generator import ScenarioType
except Exception as _import_exc:  # noqa: BLE001
    _PIPELINE_AVAILABLE = False
    _PIPELINE_IMPORT_ERROR = str(_import_exc)

    # Minimal stubs so the rest of the module-level code parses cleanly
    class ScenarioType:  # type: ignore[no-redef]
        _values = [
            "health_alert",
            "privacy_sensitive",
            "location_trigger",
            "ambient_noise",
            "calendar_reminder",
        ]

        @classmethod
        def values(cls) -> list[str]:
            return cls._values

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Wearable Agentic Eval Pipeline — Live Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

st.sidebar.title("Pipeline Controls")

_scenario_options = (
    [s.value for s in ScenarioType]
    if _PIPELINE_AVAILABLE
    else ScenarioType.values()  # type: ignore[attr-defined]
)

scenario_label: str = st.sidebar.selectbox(
    "Scenario type",
    options=_scenario_options,
    index=0,
)
num_trajectories: int = st.sidebar.slider(
    "Num trajectories to simulate",
    min_value=1,
    max_value=10,
    value=3,
)
enable_privacy_gate: bool = st.sidebar.checkbox("Enable privacy gate", value=True)
run_button: bool = st.sidebar.button("Run eval pipeline", type="primary")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Wearable Agentic Eval Pipeline — Live Demo")
st.caption(
    "Generates synthetic wearable logs → privacy gate → HITL triggers → "
    "FACTS grounding → 5-layer trajectory scoring."
)
st.divider()

# Show import warning banner once, at the top, if pipeline is unavailable
if not _PIPELINE_AVAILABLE:
    st.warning(
        f"⚠️ Module not yet available — showing mock scores.\n\n"
        f"Import error: `{_PIPELINE_IMPORT_ERROR}`"
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "results" not in st.session_state:
    st.session_state["results"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = None
if "score_history" not in st.session_state:
    st.session_state["score_history"] = []


# ---------------------------------------------------------------------------
# Mock fallback (used when _PIPELINE_AVAILABLE is False)
# ---------------------------------------------------------------------------


def _mock_results(n: int, scenario: str) -> tuple[list[Any], list[dict[str, Any]]]:
    """Return deterministic mock logs and results when pipeline imports fail."""
    mock_logs: list[Any] = [None] * n
    mock_results: list[dict[str, Any]] = []
    for i in range(n):
        mock_results.append(
            {
                "log": None,
                "log_id": f"mock-{i + 1:03d}",
                "scenario": scenario,
                "consent_model": "explicit",
                "privacy_blocked": False,
                "privacy_reason": "mock — pipeline unavailable",
                "eval": {
                    "weighted_total": 0.72,
                    "kore_trajectory_success": 0.67,
                    "kore_tool_invocation_accuracy": 1.0,
                    "kore_privacy_leak_detected": False,
                    "layer_intent": 0.75,
                    "layer_planning": 0.80,
                    "layer_tool_calls": 1.0,
                    "layer_recovery": None,
                    "layer_outcome": 1.0,
                    "pia_planning_quality": 0.70,
                    "pia_error_recovery": 0.65,
                    "pia_goal_alignment": 0.80,
                    "pia_tool_precision": 1.0,
                },
                "facts": {
                    "parametric_score": 0.70,
                    "search_score": 0.60,
                    "grounding_score": 0.75,
                    "overall_facts_score": 0.68,
                },
                "triggers": [],
                "trajectory_steps": [
                    {
                        "step_index": 0,
                        "step_name": "sense",
                        "observation": "(mock) Sensor reading captured.",
                        "reasoning": "(mock) Baseline established.",
                        "action": "",
                        "confidence": 0.91,
                    },
                    {
                        "step_index": 1,
                        "step_name": "plan",
                        "observation": "(mock) Policy check passed.",
                        "reasoning": "(mock) Action selected.",
                        "action": "",
                        "confidence": 0.85,
                    },
                    {
                        "step_index": 2,
                        "step_name": "act",
                        "observation": "(mock) Action dispatched.",
                        "reasoning": "(mock) Goal achieved.",
                        "action": "send_alert",
                        "confidence": 0.93,
                    },
                ],
                "privacy_gate_enabled": True,
                "is_mock": True,
            }
        )
    return mock_logs, mock_results


# ---------------------------------------------------------------------------
# Real pipeline runner (delegates to demo/pipeline.py)
# ---------------------------------------------------------------------------


def _run_real_pipeline(
    scenario: str,
    n: int,
    gate_enabled: bool,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Delegate to :func:`demo.pipeline.run_eval_pipeline` and unpack logs."""
    results: list[dict[str, Any]] = run_eval_pipeline(
        scenario=scenario,
        num_trajectories=n,
        privacy_gate_enabled=gate_enabled,
        seed=42,
    )
    logs: list[Any] = [r["log"] for r in results]
    return logs, results


# ---------------------------------------------------------------------------
# Run on button click
# ---------------------------------------------------------------------------

if run_button:
    with st.spinner("Running pipeline…"):
        if _PIPELINE_AVAILABLE:
            try:
                logs, results = _run_real_pipeline(
                    scenario_label, num_trajectories, enable_privacy_gate
                )
            except Exception as run_exc:  # noqa: BLE001
                st.warning(
                    f"⚠️ Pipeline raised an error — showing mock scores.\n\n`{run_exc}`"
                )
                logs, results = _mock_results(num_trajectories, scenario_label)
        else:
            logs, results = _mock_results(num_trajectories, scenario_label)

    st.session_state["logs"] = logs
    st.session_state["results"] = results

    # Accumulate score history across runs
    history: list[dict[str, Any]] = st.session_state["score_history"]
    for r in results:
        history.append(
            {
                "scenario_type": str(r["scenario"]),
                "facts_score": round(r["facts"]["overall_facts_score"], 3),
                "traj_quality": round(r["eval"]["weighted_total"], 3),
                "tool_accuracy": round(r["eval"]["kore_tool_invocation_accuracy"], 3),
                "hitl_triggered": len(r["triggers"]),
                "privacy_blocked": r["privacy_blocked"],
            }
        )
    st.session_state["score_history"] = history

if st.session_state["results"] is None:
    st.info(
        "Configure options in the sidebar and click **Run eval pipeline** to start."
    )
    st.stop()

results: list[dict[str, Any]] = st.session_state["results"]  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _severity_icon(severity: str) -> str:
    return {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
        severity, "⚪"
    )


def _score_status(value: float | None, hi: float = 0.7, lo: float = 0.5) -> str:
    """Return 'success', 'warning', or 'error' based on thresholds."""
    if value is None:
        return "warning"
    if value >= hi:
        return "success"
    if value >= lo:
        return "warning"
    return "error"


def _score_callout(
    label: str, value: float | None, hi: float = 0.7, lo: float = 0.5
) -> None:  # noqa: E501
    """Render a one-line score callout using st.success/warning/error."""
    txt = f"**{label}:** {value:.3f}" if value is not None else f"**{label}:** N/A"
    status = _score_status(value, hi, lo)
    if status == "success":
        st.success(txt)
    elif status == "warning":
        st.warning(txt)
    else:
        st.error(txt)


def _threshold_color(score: float) -> str:
    """Return hex color based on score thresholds: teal / amber / coral."""
    if score >= 0.7:
        return "#1D9E75"
    if score >= 0.5:
        return "#BA7517"
    return "#D85A30"


def _make_radar_fig(run_results: list[dict[str, Any]]) -> Any:  # noqa: ANN401
    """Build an overlaid radar chart for all trajectories in a run."""
    import plotly.graph_objects as go  # noqa: PLC0415

    _AXES = [
        "FACTS Grounding",
        "Trajectory Quality",
        "Tool Accuracy",
        "Privacy Compliance",
        "HITL Sensitivity",
    ]

    fig = go.Figure()
    n = len(run_results)
    fill_opacity = 0.30 if n > 1 else 0.50

    for i, r in enumerate(run_results):
        ev = r["eval"]
        facts_score = r["facts"]["overall_facts_score"]
        privacy_score = (
            0.0 if (r["privacy_blocked"] or ev["kore_privacy_leak_detected"]) else 1.0
        )  # noqa: E501
        hitl_score = max(0.0, 1.0 - len(r["triggers"]) / 4.0)

        scores = [
            facts_score,
            ev["weighted_total"],
            ev["kore_tool_invocation_accuracy"],
            privacy_score,
            hitl_score,
        ]
        mean_score = sum(scores) / len(scores)
        color = _threshold_color(mean_score)

        # Close the polygon by repeating the first point
        r_vals = scores + [scores[0]]
        theta_vals = _AXES + [_AXES[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                fillcolor=color,
                line=dict(color=color, width=2),
                opacity=fill_opacity,
                name=f"Traj {i + 1}  ({mean_score:.2f})",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat=".1f",
                gridcolor="rgba(150,150,150,0.3)",
            ),
            angularaxis=dict(gridcolor="rgba(150,150,150,0.3)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20, x=0.5, xanchor="center"
        ),  # noqa: E501
        height=420,
        margin=dict(t=30, b=60, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _make_score_bar_fig(run_results: list[dict[str, Any]]) -> Any:  # noqa: ANN401
    """Horizontal bar chart of per-trajectory quality scores."""
    import plotly.graph_objects as go  # noqa: PLC0415

    labels = [f"Traj {i + 1}" for i in range(len(run_results))]
    scores = [r["eval"]["weighted_total"] for r in run_results]
    colors = [_threshold_color(s) for s in scores]

    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.3f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], tickformat=".2f", title="Trajectory Quality"),
        yaxis=dict(autorange="reversed"),
        height=max(160, len(run_results) * 44 + 60),
        margin=dict(t=10, b=40, l=80, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# Per-trajectory display
# ---------------------------------------------------------------------------

for idx, result in enumerate(results):
    ev: dict[str, Any] = result["eval"]
    facts: dict[str, float] = result["facts"]
    triggers = result["triggers"]
    steps = result["trajectory_steps"]
    is_mock: bool = result["is_mock"]
    blocked: bool = result["privacy_blocked"]

    # Expander title summarises the most important signals at a glance
    privacy_tag = "🔒 BLOCKED" if blocked else "✅ PASSED"
    facts_val = facts["overall_facts_score"]
    quality_val = ev["weighted_total"]
    expander_title = (
        f"Trajectory {idx + 1} — {result['scenario']}  |  "
        f"quality: {quality_val:.2f}  |  "
        f"FACTS: {facts_val:.2f}  |  "
        f"privacy: {privacy_tag}  |  "
        f"HITL: {len(triggers)}"
    )

    with st.expander(expander_title, expanded=(idx == 0)):
        if is_mock:
            st.caption("⚠️ Mock scores — pipeline module unavailable.")

        col1, col2, col3 = st.columns([1.2, 2.2, 1.4])

        # ----------------------------------------------------------------
        # Col 1 — Input Log
        # ----------------------------------------------------------------
        with col1:
            st.subheader("Input Log")

            # Step 2 — privacy gate result
            if blocked:
                st.error(f"🔒 **Privacy gate: BLOCKED**\n\n{result['privacy_reason']}")
            else:
                gate_label = (
                    "✅ Privacy gate: PASSED"
                    if result["privacy_gate_enabled"]
                    else "⚠️ Privacy gate: disabled"
                )  # noqa: E501
                st.success(f"{gate_label}\n\n{result['privacy_reason']}")

            log_obj = result["log"]
            if log_obj is not None:
                log_dict = log_obj.to_dict()
                preview = {
                    "log_id": result["log_id"][:16] + "…",
                    "scenario_type": str(result["scenario"]),
                    "consent_model": result["consent_model"],
                    "sensor_data": {
                        k: round(v, 2) if isinstance(v, float) else v
                        for k, v in log_dict["sensor_data"].items()
                        if k
                        in (
                            "heart_rate",
                            "spo2",
                            "steps",
                            "noise_db",
                            "heart_rate_noised",
                            "spo2_noised",
                        )
                    },
                    "audio_transcript": {
                        "text": log_dict["audio_transcript"]["text"],
                        "confidence": round(
                            log_dict["audio_transcript"]["confidence"], 3
                        ),  # noqa: E501
                        "keywords_detected": log_dict["audio_transcript"][
                            "keywords_detected"
                        ],  # noqa: E501
                    },
                    "ground_truth_action": log_dict["ground_truth_action"],
                }
                st.json(preview, expanded=False)
            else:
                st.json(
                    {
                        "log_id": result["log_id"],
                        "scenario": result["scenario"],
                        "consent_model": result["consent_model"],
                        "note": "mock log",
                    },
                    expanded=False,
                )

        # ----------------------------------------------------------------
        # Col 2 — Eval Scores
        # ----------------------------------------------------------------
        with col2:
            st.subheader("Eval Scores")

            # Row 1 — six st.metric cards
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Trajectory Quality",
                f"{quality_val:.2f}",
                help="5-layer weighted composite (TrajectoryScorer)",
            )
            m2.metric(
                "Tool Accuracy",
                f"{ev['kore_tool_invocation_accuracy']:.2f}",
                help="Kore.ai tool_invocation_accuracy",
            )
            m3.metric(
                "FACTS Grounding",
                f"{facts_val:.2f}",
                help="DeepMind FACTS: parametric + search + grounding mean",
            )

            m4, m5, m6 = st.columns(3)
            privacy_metric_ok = (
                result["privacy_gate_enabled"] and not ev["kore_privacy_leak_detected"]
            )
            m4.metric(
                "Privacy Gate",
                "PASS" if privacy_metric_ok else "FAIL",
                delta="no PII" if not ev["kore_privacy_leak_detected"] else "PII LEAK",
                delta_color="normal" if privacy_metric_ok else "inverse",
                help="PII pattern scan (email, US phone, SSN) across tool outputs",
            )
            m5.metric(
                "HITL Triggers",
                len(triggers),
                help="Steps that fired a human-review trigger",
            )
            m6.metric(
                "Trajectory Success",
                f"{ev['kore_trajectory_success']:.2f}",
                help="Kore.ai trajectory_success_rate",
            )

            # Step 4 — threshold callouts for FACTS and quality
            st.caption("Threshold assessment")
            _score_callout("FACTS grounding", facts_val, hi=0.70, lo=0.50)
            _score_callout("Trajectory quality", quality_val, hi=0.70, lo=0.50)

            # 5-layer breakdown bar chart
            st.caption("5-layer score breakdown")
            layer_scores: dict[str, float | None] = {
                "intent": ev["layer_intent"],
                "planning": ev["layer_planning"],
                "tool_calls": ev["layer_tool_calls"],
                "recovery": ev["layer_recovery"],
                "outcome": ev["layer_outcome"],
            }
            try:
                import plotly.graph_objects as go  # noqa: PLC0415

                fig_bar = go.Figure(
                    go.Bar(
                        x=list(layer_scores.keys()),
                        y=[v if v is not None else 0.0 for v in layer_scores.values()],
                        marker_color=[
                            "#2ecc71"
                            if (v or 0) >= 0.7
                            else "#f39c12"
                            if (v or 0) >= 0.4
                            else "#e74c3c"
                            for v in layer_scores.values()
                        ],
                        text=[
                            f"{v:.2f}" if v is not None else "N/A"
                            for v in layer_scores.values()
                        ],
                        textposition="outside",
                    )
                )
                fig_bar.update_layout(
                    yaxis=dict(range=[0, 1.2], tickformat=".2f"),
                    height=220,
                    margin=dict(t=10, b=10, l=10, r=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{idx}")
            except ImportError:
                st.json(
                    {
                        k: round(v, 3) if v is not None else None
                        for k, v in layer_scores.items()
                    }
                )

            # Step 5 — PIA dimensions
            st.caption("PIA rubric dimensions")
            pia_cols = st.columns(4)
            pia_fields = [
                ("planning_quality", "Planning"),
                ("error_recovery", "Recovery"),
                ("goal_alignment", "Goal align"),
                ("tool_precision", "Tool prec."),
            ]
            for col, (key, label) in zip(pia_cols, pia_fields):
                val = ev.get(f"pia_{key}")
                col.metric(label, f"{val:.2f}" if val is not None else "N/A")

            # Step 3 — HITL trigger details
            if triggers:
                st.caption("HITL trigger details")
                for t in triggers:
                    t_type = (
                        t.trigger_type.value if hasattr(t, "trigger_type") else str(t)
                    )  # noqa: E501
                    t_sev = t.severity if hasattr(t, "severity") else "unknown"
                    t_step = t.step_index if hasattr(t, "step_index") else "?"
                    t_desc = (
                        t.action_description if hasattr(t, "action_description") else ""
                    )  # noqa: E501
                    t_rec = (
                        t.recommended_action if hasattr(t, "recommended_action") else ""
                    )  # noqa: E501
                    icon = _severity_icon(t_sev)
                    st.markdown(
                        f"{icon} **{t_type}** (step {t_step})  \n{t_desc}  \n*{t_rec}*"
                    )
            else:
                st.caption("No HITL triggers fired.")

        # ----------------------------------------------------------------
        # Col 3 — Trajectory
        # ----------------------------------------------------------------
        with col3:
            st.subheader("Trajectory")
            for step in steps:
                st.markdown(
                    f"**Step {step['step_index']}: {step['step_name'].upper()}**"
                )
                st.caption(f"Observation: {step['observation']}")
                st.caption(f"Reasoning: {step['reasoning']}")
                if step["action"]:
                    st.success(f"Action: `{step['action']}`")
                conf = step["confidence"]
                conf_icon = "🟢" if conf >= 0.85 else "🟡" if conf >= 0.70 else "🔴"
                st.caption(f"Confidence: {conf_icon} {conf:.3f}")
                st.divider()


# ---------------------------------------------------------------------------
# Eval Overview — radar chart + per-scenario bar chart
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Eval Overview")

try:
    import plotly.graph_objects as go  # noqa: PLC0415, F401

    _radar_col, _bar_col = st.columns([1, 1])

    with _radar_col:
        st.caption(
            "Radar: FACTS Grounding · Trajectory Quality · Tool Accuracy · "
            "Privacy Compliance · HITL Sensitivity  "
            "(teal ≥ 0.7 · amber 0.5–0.69 · coral < 0.5)"
        )
        radar_fig = _make_radar_fig(results)
        st.plotly_chart(radar_fig, use_container_width=True, key="radar_overview")

        # PNG download (requires kaleido); fall back to HTML
        try:
            _img_bytes: bytes = radar_fig.to_image(format="png", width=700, height=500)
            st.download_button(
                "⬇ Download radar chart (PNG)",
                _img_bytes,
                file_name="radar_chart.png",
                mime="image/png",
            )
        except Exception:  # noqa: BLE001
            _html_bytes: bytes = radar_fig.to_html().encode("utf-8")
            st.download_button(
                "⬇ Download radar chart (HTML — kaleido not installed)",
                _html_bytes,
                file_name="radar_chart.html",
                mime="text/html",
            )

    with _bar_col:
        if len(results) > 1:
            st.caption("Per-trajectory quality score distribution")
            bar_fig = _make_score_bar_fig(results)
            st.plotly_chart(bar_fig, use_container_width=True, key="bar_overview")
        else:
            st.info("Run ≥ 2 trajectories to see per-trajectory distribution.")

except ImportError:
    st.warning("Install plotly to enable radar and bar overview charts.")


# ---------------------------------------------------------------------------
# Aggregate summary table
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Aggregate Summary")
summary_rows = []
for idx, result in enumerate(results):
    ev = result["eval"]
    facts = result["facts"]
    triggers = result["triggers"]
    summary_rows.append(
        {
            "#": idx + 1,
            "scenario": str(result["scenario"]),
            "consent": result["consent_model"],
            "privacy": "BLOCKED" if result["privacy_blocked"] else "PASSED",
            "quality": round(ev["weighted_total"], 3),
            "tool_acc": round(ev["kore_tool_invocation_accuracy"], 3),
            "facts": round(facts["overall_facts_score"], 3),
            "HITL": len(triggers),
            "mock": result["is_mock"],
        }
    )

try:
    import pandas as pd  # noqa: PLC0415

    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
except ImportError:
    st.json(summary_rows)


# ---------------------------------------------------------------------------
# Score History — accumulates across multiple "Run" clicks
# ---------------------------------------------------------------------------

history_rows: list[dict[str, Any]] = st.session_state["score_history"]
if history_rows:
    st.divider()
    st.subheader("Score History")
    st.caption(
        "Accumulates across all pipeline runs in this session. "
        "Refresh the page to clear."
    )
    try:
        import pandas as pd  # noqa: PLC0415

        df_history = pd.DataFrame(
            history_rows,
            columns=[
                "scenario_type",
                "facts_score",
                "traj_quality",
                "tool_accuracy",
                "hitl_triggered",
                "privacy_blocked",
            ],
        )
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    except ImportError:
        st.json(history_rows)
