# Live Eval Dashboard — Wearable Agentic Pipeline

Upload a wearable sensor scenario → agent runs → 5 eval metrics displayed in real time

---

## Quick Start

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
Run from the **project root** — all demo dependencies are already declared in
`pyproject.toml`:

```bash
# From project root
uv sync
streamlit run demo/app.py
```

If you prefer a plain pip install (e.g. in a fresh venv):

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

The app opens at `http://localhost:8501`.

📺 Demo: https://www.loom.com/share/5bec5764428b4e48aa868134e54a894e

---

## Screenshot

![Dashboard screenshot](screenshot.png)

*Screenshot placeholder — will be added after Loom recording.*

---

## What the Dashboard Shows

| Metric | Source | Threshold |
|--------|--------|-----------|
| **Trajectory Quality** | 5-layer `TrajectoryScorer` (intent 0.15 · planning 0.25 · tools 0.25 · recovery 0.15 · outcome 0.20) | ≥ 0.70 green |
| **Tool Accuracy** | Kore.ai `tool_invocation_accuracy` — fraction of steps calling a known, valid tool | ≥ 0.70 green |
| **FACTS Grounding** | DeepMind FACTS mean of parametric · search · grounding sub-scores (RAGAS Faithfulness backend) | ≥ 0.70 green |
| **Privacy Compliance** | Consent model check (REVOKED → BLOCKED) + PII leak scan (email / US phone / SSN patterns) | PASS / FAIL |
| **HITL Sensitivity** | `HITLTriggerEvaluator` — four trigger conditions: confidence < 0.70, safety-adjacent action, novel tool, domain expertise required | Count displayed |

### Eval Overview Charts

- **Radar chart** — overlays all trajectories on 5 axes; color-coded teal (≥ 0.70) / amber (0.50–0.69) / coral (< 0.50) by mean score. Downloadable as PNG (requires `kaleido`) or HTML.
- **Per-trajectory bar chart** — horizontal quality-score bars (visible when ≥ 2 trajectories are run).
- **Score History table** — accumulates every run within the session so you can compare scenario types across multiple clicks.

---

## Pipeline Steps

```
1. WearableLogGenerator  →  synthetic sensor + audio log (5 scenario types, ε=1.0 DP)
2. PrivacyGate           →  consent check; REVOKED consent blocks the trajectory
3. HITLTriggerEvaluator  →  flags steps requiring human review
4. FACTSGroundingScorer  →  parametric + search + grounding sub-scores
5. AgenticEvaluator      →  5-layer TrajectoryScorer + 4 PIA rubric dimensions
```

---

## Related

- [Project README](../README.md) — full architecture, papers implemented, setup guide
- [`src/eval/agentic_eval.py`](../src/eval/agentic_eval.py) — AgenticEvaluator + FACTSGroundingScorer
- [`src/eval/hitl_trigger.py`](../src/eval/hitl_trigger.py) — HITLTriggerEvaluator
- [`src/eval/trajectory_scorer.py`](../src/eval/trajectory_scorer.py) — 5-layer TrajectoryScorer + PIA
