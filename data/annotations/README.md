---
license: mit
task_categories:
- text-classification
- question-answering
language:
- en
tags:
- annotation
- inter-rater-reliability
- wearable-agents
- agentic-evaluation
- trajectory-annotation
pretty_name: Wearable Agent Trajectory Annotation Dataset
size_categories:
- 100<n<1K
---

# Wearable Agent Trajectory Annotation Dataset

## Dataset Summary

50 wearable agent trajectories annotated by 5 LLM-simulated annotator personas
using the `agenteval-schema-v1` JSON schema, across two calibration phases
(500 annotation records total). Designed to benchmark annotation-quality pipelines
for agentic AI systems.

Each trajectory captures a wearable AI agent responding to a real-time sensor event
(health alert, privacy-sensitive context, location trigger, ambient noise, calendar
reminder). Five LLM annotator personas — each with a systematic scoring bias — rate
four rubric dimensions on a 1–4 ordinal scale, before and after a calibration round
built on 5 anchor examples.

> **Correction (2026-09-06).** Earlier versions of this card carried a "Blog Post"
> banner citing "κ = −0.065 → +0.743 · 100 trajectories," reported a calibration
> lift of "Cohen's κ 0.55 → 0.82," and described agreement as moving "from poor to
> substantial." Those figures were not empirically measured and have been removed.
> The only agreement figure this dataset supports is the pre-calibration Fleiss' κ
> ≈ −0.036 (poor) from the deterministic dry-run annotation mode; the
> post-calibration κ = 1.00 is a mathematical artifact of that mode (see *IAA
> Results*) and is not an empirical result. A separate 5-annotator human study
> (Prolific, 10 trajectory pairs) is summarised under *Related: Path-Invariant
> Agreement*.

**Key numbers at a glance:**

| | Value |
|---|---|
| Trajectories | 50 |
| Annotator personas | 5 (LLM-simulated, dry-run) |
| Calibration phases | 2 (pre + post) |
| Total annotation records | 500 |
| Rubric dimensions | 4 (step_quality, privacy_compliance, goal_alignment, error_recovery) |
| Scenario types | 5 (health_alert, privacy_sensitive, location_trigger, ambient_noise, calendar_reminder) |
| Pre-calibration Fleiss' κ (overall) | ≈ −0.036 (poor) |

---

## Dataset Structure

Annotations follow a 3-layer schema defined in `agenteval-schema-v1.json`:

**Layer 1 — Session-level outcome**
One record per trajectory capturing the end-to-end evaluation:
`overall_goal_achieved`, `session_outcome` (success / partial / failure),
`privacy_compliance_overall`, `user_trust_maintained`, `latency_acceptable`.
This layer corresponds to outcome-reward (ORM) signal.

**Layer 2 — Role-level attribution**
Per-agent-role records for multi-agent trajectories. Captures which agent
contributed what to the outcome. The orchestrator role has an additional required
field `handoff_quality`; non-orchestrator roles explicitly exclude it. This layer
enables attribution in cascade-error analysis.

**Layer 3 — Step-level PRM feed**
One record per trajectory step: `process_reward_score` (float, −1.0 to +1.0),
`partial_credit` (float, 0.0 to 1.0), `annotator_rationale` (minimum 20 characters
for BERTScore quality gate), `tool_called` (enum of 8 actions). This is the
primary input for process-supervised reward model (PRM) training.

### Parquet file layout (`../processed/wearable_annotated_50.parquet`)

The consolidated parquet joins pre- and post-calibration annotations with trajectory
metadata from the raw wearable logs. Load with:

```python
from datasets import Dataset
import pandas as pd

df = pd.read_parquet("wearable_annotated_50.parquet")
ds = Dataset.from_pandas(df)
```

Column reference:

| Column | Type | Description |
|---|---|---|
| `annotation_id` | string | UUID per annotation record |
| `log_id` | string | UUID linking back to source trajectory |
| `calibration_phase` | string | `"pre"` or `"post"` |
| `persona_name` | string | One of 5 annotator persona names |
| `scenario_type` | string | One of 5 wearable scenario types |
| `consent_model` | string | `explicit`, `implied`, `ambient`, or `revoked` |
| `ground_truth_action` | string | Gold-label agent action for this trajectory |
| `step_quality` | int8 | Rubric score 1–4 |
| `privacy_compliance` | int8 | Rubric score 1–4 |
| `goal_alignment` | int8 | Rubric score 1–4 |
| `error_recovery` | int8 | Rubric score 1–4 |
| `n_trajectory_steps` | int64 | Number of steps in agent trajectory |
| `final_action` | string | Terminal action taken by agent |
| `mean_step_confidence` | float64 | Mean confidence across trajectory steps |
| `heart_rate` | float64 | Sensor reading (bpm) |
| `spo2` | float64 | Sensor reading (%) |
| `noise_db` | float64 | Ambient noise level (dB) |
| `device_model` | string | Wearable device identifier |
| `activity` | string | User activity at annotation time |
| `alert_severity` | string | `high`, `medium`, or `low` |
| `rationale` | string | Annotator free-text justification |
| `created_at` | string | ISO 8601 annotation timestamp |

---

## IAA Results

Agreement is measured with three complementary metrics across all 5 annotator
personas (C(5,2) = 10 pairwise combinations for Cohen's κ; Fleiss' κ computed
jointly). All numbers below come from the deterministic **dry-run** annotation
mode — a live-API annotation run has not been performed for this dataset.

### Overall agreement

| Metric | Pre-calibration | Post-calibration |
|---|---|---|
| Fleiss' κ | −0.036 | 1.00* |
| Krippendorff's α | −0.11 | 1.00* |

Pre-calibration values reflect the genuine, deliberately-designed disagreement
between the five persona biases. Post-calibration values are a dry-run artifact —
see the note below.

### Per-dimension (pre-calibration, dry-run personas)

| Dimension | Fleiss' κ | Cohen's κ (mean) | Krippendorff's α |
|---|---|---|---|
| step_quality | −0.04 | −0.01 | −0.09 |
| privacy_compliance | −0.06 | 0.01 | −0.16 |
| goal_alignment | 0.00 | 0.05 | −0.04 |
| error_recovery | −0.04 | 0.05 | −0.16 |

> **\* Dry-run artifact note.** Post-calibration Fleiss' κ = 1.00 and
> Krippendorff's α = 1.00 are produced by the dry-run annotation mode, which uses
> SHA-256-seeded deterministic scores blended at weight 0.82 toward gold anchor
> targets. With a blending window of ±0.72 and rounded integer gold means
> ({step_quality: 2, privacy_compliance: 3, goal_alignment: 3, error_recovery: 2}),
> all personas collapse to identical integer scores on non-anchor trajectories —
> making perfect agreement mathematically inevitable rather than empirically achieved.
> **Do not cite the post-calibration κ/α = 1.00 as an empirical annotation result.**

---

## Related: Path-Invariant Agreement (separate study)

A separate small human study — 5 annotators recruited via Prolific, each scoring the
same 10 wearable agent trajectory pairs — tested whether scoring rubric *dimensions*
rather than path-specific *steps* recovers usable agreement on non-deterministic
trajectories:

| Instrument | Fleiss' κ | Interpretation |
|---|---|---|
| Standard path-comparison | ≈ −0.208 | poor (−0.14 to −0.23 across step-count assumptions) |
| PIA rubric (planning quality, error recovery, goal alignment) | ≈ +0.219 | fair (Δ ≈ +0.427) |

Goal alignment on its own shows no agreement above chance (κ = −0.003), and the lift
is sensitive to one internally-inconsistent participant. That study's raw export,
analysis script, and results are in `data/human_study/` in the code repository — not
in this dataset. Full write-up: `white_papers/NeurIPS/neurips2026_main.pdf`, §3.

---

## Annotation Schema

Schema file: `agenteval-schema-v1.json` (included in this directory)

The schema is a JSON Schema (draft-07) document with three top-level `$defs`:
`SessionAnnotation`, `RoleAnnotation`, and `StepAnnotation`. It enforces:

- Integer scores in [1, 4] for all rubric dimensions
- `process_reward_score` in [−1.0, +1.0] for PRM training signal
- Minimum 20-character `annotator_rationale` (enables BERTScore quality gate)
- Conditional `handoff_quality` field — required for orchestrator roles, forbidden
  for non-orchestrator roles (JSON Schema `if`/`then`/`else`)
- `rubric_anchors` block: per-dimension good/bad scored examples grounded in
  wearable scenario types

Human-readable annotator rubric: `wearable_annotation_rubric.md`

---

## Calibration Protocol

Five anchor trajectories were selected from the pre-calibration annotation set
using a threshold-based disagreement criterion: any trajectory where the per-persona
score variance exceeded 1.5 or Fleiss' κ fell below −0.10 was nominated as a
calibration anchor. For each anchor, gold-label scores were assigned by the
rubric author and supplemented with `IF/THEN` clarification rules for the three
most-contested dimensions (step_quality, goal_alignment, privacy_compliance).
Annotator persona scores were then re-weighted using the formula
`calibrated_score = 0.82 × gold_target + 0.18 × persona_base_score`. In the
deterministic dry-run mode this collapses non-anchor scores to identical integers
(see the artifact note above); the effect of this protocol under live-API annotation
has not been measured. Calibration configuration, anchor examples, and rubric
updates are recorded in `calibration_round_01.json`.

---

## Annotator Personas

Five LLM-simulated annotators with systematic scoring biases, designed to mirror
real inter-annotator disagreement patterns observed in clinical and agentic AI
annotation studies:

| Persona | Primary bias |
|---|---|
| PrivacyMaximalist | Scores `privacy_compliance` strictly; penalises any data disclosure under ambient or revoked consent |
| OutcomeOptimist | Scores `goal_alignment` high; deprioritises process over outcome |
| ProcessPurist | Scores `step_quality` strictly; rewards chain-of-thought evidence |
| ClinicalSafetyFirst | Scores `goal_alignment` high for health_alert scenarios; low for all others |
| RecoverySkeptic | Scores `error_recovery` low; requires explicit recovery actions, not passive fallbacks |

---

## Citation

If you use this dataset, please cite:

```
@misc{wearable-agent-trajectory-annotation,
  title   = {Wearable Agent Trajectory Annotation Dataset},
  author  = {Bade, Shailendra},
  year    = {2026},
  url     = {https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline},
  note    = {50 trajectories, 5 LLM annotator personas, agenteval-schema-v1}
}
```

---

## Related Work

- **Kore.ai Agent Evaluation Blog (October 2025)** — Documents that 89% of
  enterprises have agent observability, but only 52% have real evaluation;
  identifies annotation methodology as the primary gap.
- **Verga et al. 2024, "Replacing Judges with Juries"** — Multi-LLM evaluation
  panels reduce single-model bias; this dataset extends that insight to ordinal
  rubric annotation for agentic trajectories.
- **Cohere Command A (arXiv 2504.00698)** — 800-prompt annotation study with
  65 annotators on a 5-point scale. Notably reports no inter-rater agreement
  statistics (no κ, no α). This dataset directly addresses that gap by providing
  reproducible IAA methodology for LLM output annotation.
- **OpenAI HealthBench** — Rubric-based clinical evaluation showing
  physician-physician agreement of 0.55–0.75 even in controlled conditions.
- **ReasonRAG (NeurIPS 2025, arXiv 2505.14069)** — Process-supervised DPO
  outperforms outcome-supervised RL with 18× fewer training queries. The step-level
  PRM annotation in Layer 3 of this dataset is designed as input for that training
  regime.
