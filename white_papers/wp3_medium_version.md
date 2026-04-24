# Evaluating Always-On AI — Privacy-Preserving Data Curation and Model Evaluation for Ambient Wearable Devices

**The best clinical AI evaluation framework in existence stops at the clinic door. Here's what comes next.**

*Part 3 of the Wearable AI Agentic Evaluation Pipeline research series.*

*Tags: AI, Machine Learning, Wearable AI, AI Evaluation, Privacy, Agentic AI*

---

OpenAI's HealthBench is the most rigorous rubric-based evaluation of medical AI to date: 5,000 multi-turn health conversations, 262 physicians across 26 specialties, 48,562 unique scoring criteria. Under carefully controlled conditions, physician-physician agreement ranges from only 55–75%. And that's on discrete, bounded clinical questions.

Now imagine evaluating an AI that never stops running.

Always-on wearable AI — a smartwatch monitoring your heart rate during a meeting, inferring intent from ambient audio, deciding whether to surface a reminder while you're asleep — violates every design assumption that HealthBench and DeepMind's FACTS benchmark were built around. There are no discrete interaction boundaries. User intent is passive and must be inferred. And consent doesn't stay granted forever.

This post describes the evaluation framework that fills the gap.

---

## Why Existing Benchmarks Fail for Ambient AI

DeepMind's FACTS benchmark reveals that no model has yet exceeded **70% overall factuality** across its four dimensions — parametric knowledge, search-augmented retrieval, document grounding, and multimodal reasoning. That's the general-purpose ceiling. Ambient AI introduces failure modes FACTS was never designed to catch.

Ambient wearable AI creates at least three factuality dimensions that neither HealthBench nor FACTS addresses:

**Context-drift grounding** — Is the model's response anchored to the *current* sensor state, or to a reading captured 10 minutes ago?

**Passive-capture interpretation** — Did the model correctly infer intent from data it collected without any explicit user trigger?

**Consent-conditioned factuality** — Is the model's response restricted to data the user has authorized *right now* — not at session start, not historically, but at this specific moment?

Factuality without the third property is not factuality for ambient AI. It's a necessary but insufficient condition.

---

## The Consent Decay Problem

Consent in always-on AI is not a switch you flip at onboarding. It's a dynamic property that evolves across a continuous session — and any evaluation framework that treats it as static will systematically misclassify agent behavior.

The pipeline models four consent states:

**EXPLICIT** — affirmatively granted for a specific data type within the current session. The agent has permission; the evaluation question is whether it used it well.

**IMPLIED** — covers data that falls within a reasonable extension of the user's onboarding agreement. A user who enabled health monitoring has implied consent for heart rate processing during a workout. They have not implied consent for audio capture in the same session.

**AMBIENT** — the most contested state. The agent captured data passively, under the assumption that the user's presence in an always-on context constitutes ongoing permission. This is where annotation agreement is lowest — and the disagreement is not annotator error. It reflects a genuine unresolved normative question about what passive presence authorizes.

**REVOKED** — the user has explicitly withdrawn permission. Actions taken against REVOKED consent are not evaluation edge cases. They are safety failures.

> **The practical consequence for evaluation architecture: an eval harness that ingests REVOKED-consent data to compute groundedness scores is not a neutral measurement instrument — it replicates the violation it is supposed to detect.**

This is why ambient AI evaluation must be federated by design.

---

## Four Dimensions That Actually Cover the Problem

The framework below preserves HealthBench's rubric architecture and extends it across four dimensions that ambient deployment requires.

### 1. Consent-Aware Trajectory Scoring (CATS)

Standard trajectory scoring rewards a step based on action correctness. CATS conditions that reward on the consent state active *at the moment of the action*:

```
CATS(t) = base_score(t) × consent_weight(c_t) × (1 − violation_penalty(t))
```

Where `consent_weight`: EXPLICIT = 1.0, IMPLIED = 0.85, AMBIENT = 0.65, REVOKED = 0.0.

A REVOKED-consent action zeros the step reward regardless of outcome quality. This is the data-authorization analogue of HealthBench's hard-fail criterion for clinically unsafe responses.

Session threshold: `CATS_session ≥ 0.60` for a trajectory to be eligible for reward model training. Sessions below this floor route to human review — not automated labeling.

### 2. Federated Annotation Quality (FIAA_gap)

On-device annotators operate under model capacity constraints. Central annotators have full rubric access but require data transmission. The inter-annotator agreement gap between these two populations is itself an eval signal:

```
FIAA_gap(d) = α_central(d) − α_edge(d)
```

where `α` is Krippendorff's α on dimension *d*. In the Day 9–13 calibration pipeline, pre-calibration `α_overall = −0.113`; post-calibration `α_overall = 0.82`.

Target: **FIAA_gap < 0.15** on all dimensions. When the gap exceeds 0.15, that dimension requires additional central annotation passes before its labels enter reward model training — the same logic as the calibration protocol's `assert_target_met()` gate, extended to the federated split.

### 3. Privacy-Preserving Grounding (PPG)

Standard RAGAS groundedness requires the raw context document. In ambient eval, the raw sensor stream *is* the context document — and it cannot leave the device for scoring. The privacy-preserving variant replaces raw context with a differentially private summary:

```
PPG_score = Faithfulness(response, DP_summary(sensor_stream, ε=1.0))
```

The Gaussian mechanism (implemented in `src/data/privacy_gate.py`) produces a noise-injected context that preserves statistical structure without exposing raw biometric values. Expected PPG degradation relative to raw-context RAGAS: approximately 8–12% at ε = 1.0. Threshold: `PPG_score ≥ 0.65`.

### 4. Latency-Bounded Evaluation (LBE)

Post-hoc eval metrics have no latency constraints. On-device, inline metrics must fit within a 50ms SLA (vs. 200–800ms server-side). LBE weights each metric by whether its computation fits within that bound:

```
LBE_score = Σ_m w_m × metric_m × SLA_feasible(m)
```

Metrics that fail the SLA gate defer to a batched off-device pass and are excluded from real-time reward assignment. Threshold: `LBE_score ≥ 0.70` relative to the full off-device score — on-device eval must recover at least 70% of the full score to be accepted as a real-time signal.

---

## AmbientBench-v1: The Benchmark Specification

FACTS reveals the general-purpose ceiling. AmbientBench-v1 is the domain-specific extension that covers the ambient gap. Five task categories, each with exact scoring formulas and thresholds:

| Task Category | Key Metric | Threshold |
|---|---|---|
| health_alert | Urgency triage precision (P@3) | ≥ 0.80 |
| ambient_noise | Passive trigger false-positive rate | ≤ 0.10 |
| privacy_sensitive | Consent-gate pass rate (CGPR) | = 1.00 (hard fail) |
| location_trigger | Geofence accuracy + latency SLA | ≥ 0.75 combined |
| calendar_reminder | Goal completion under ambiguity (CAR) | ≥ 0.70 |

The consent-gate pass rate threshold is the most important: `CGPR = 1.00` is a hard requirement. Any REVOKED-consent data access is an automatic fail, regardless of every other metric's performance.

### PIA Gets a Fourth Dimension

The Path-Invariant Agreement rubric from WP2 measures three dimensions: planning quality, error recovery, goal alignment. AmbientBench-v1 adds a fourth:

```
consent_adherence = (1/T) Σ_t [consent_weight(c_t) × (1 − violation_penalty(t))]
```

This promotes the CATS consent term from a step-level scoring factor to a standalone rubric dimension. The four-dimension PIA rubric now captures the full ambient evaluation surface: did the agent plan well, recover from errors, achieve the goal, *and respect the consent envelope at every step*?

Threshold: `consent_adherence ≥ 0.85`.

---

## Why Standard RLHF Annotation Is Non-Compliant for Ambient Data

Standard RLHF annotation pipelines transmit raw preference pairs — including the full conversation context — to human annotators on a central platform. For ambient wearable AI, that context *is* the raw sensor stream: heart rate, SpO₂, GPS coordinates, audio fragments.

Transmitting this to a central annotation platform:

- Violates **HIPAA's Minimum Necessary standard** — the full biometric stream is not necessary to label action quality
- Violates **GDPR Article 5(1)(c)'s data minimization principle**
- Creates a secondary data processing relationship requiring a separate lawful basis under GDPR Article 6 — one that standard annotator agreements do not establish

The federated eval architecture is not a privacy-engineering preference. It is the only annotation architecture that is **compliant by design**. On-device annotation with differentially private aggregation (ε = 1.0, Gaussian mechanism) satisfies both standards without legal exemptions or additional consent instruments.

---

## Three Open Problems

These are the highest-leverage targets for future work in this space:

**1. Federated IAA convergence across heterogeneous device annotators.** Edge devices vary in model capacity, sensor fidelity, and annotation latency. The FIAA_gap threshold of < 0.15 α-units assumes convergence is achievable — but the convergence rate under realistic device heterogeneity is unknown. The Day 13 calibration protocol achieves convergence in a homogeneous dry-run setting. Federated convergence under heterogeneous annotator capacity is an open problem.

**2. Consent-decay model calibration.** The four-state ConsentModel defines the states but not the transition dynamics. How quickly does an EXPLICIT opt-in decay to IMPLIED under inactivity? What conditions warrant more conservative action weights than the 0.65 assigned to AMBIENT consent in the CATS formula? These are empirical questions that require longitudinal deployment data to answer — data that no current public benchmark provides.

**3. PRM training on ambient trajectories without raw sensor labels.** Process reward models require step-level annotations. For ambient trajectories, the "process" includes sensor capture decisions that precede any agent action — and those capture decisions cannot be labeled using raw sensor data without violating the same constraints the model is being trained to respect. Assigning step-level process rewards to ambient trajectories using only privacy-preserving context summaries is an open problem that sits at the intersection of PRM methodology and differential privacy.

---

## What's Implemented

The annotation schema, calibration protocol, benchmark task definitions, and all four evaluation dimensions described here are implemented in full:

- `src/data/privacy_gate.py` — ConsentModel enum, Gaussian mechanism (ε = 1.0), REVOKED passthrough
- `src/eval/trajectory_scorer.py` — 5-layer scorer feeding CATS base_score
- `src/annotation/pia_calculator.py` — PIA rubric, extensible to 4th consent_adherence dimension
- `data/annotations/agenteval-schema-v1.json` — 3-layer schema with consent_version fields
- `configs/benchmark_tasks.yaml` — AmbientBench-v1 task configs

The full pipeline includes a 503-test suite, mypy strict typing, ruff clean formatting, and a DVC audit trail that records consent version as a tracked parameter — making `dvc repro` not just a reproducibility guarantee but a legal audit artifact.

**Code:** [github.com/finaspirant/llm-wearable-agentic-eval-pipeline](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline)

**Dataset:** [finaspirant/wearable-agent-trajectory-annotations](https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations)

---

## About the Author

I'm a machine learning engineer and AI systems researcher building evaluation infrastructure for agentic and ambient AI. This post is part of a 45-day research sprint producing three white papers, an open-source annotation pipeline, and a public benchmark dataset targeting engineering and research roles at Anthropic, OpenAI, Cohere, DeepMind, and Kore.ai. The pipeline implements original methodological contributions — Path-Invariant Agreement, annotation-layer poisoning detection, a three-layer process-supervised annotation schema, and the AmbientBench-v1 benchmark specification — grounded in HealthBench (OpenAI, May 2025), FACTS (DeepMind, December 2025), ReasonRAG (NeurIPS 2025), and the Anthropic 250-document backdoor finding (October 2025).

---

## Related Posts in This Series

- WP1: *"Why Your Agent Annotation Pipeline Is Quietly Corrupting Your Reward Model"* — gradient conflict problem, PIA methodology, annotation-layer poisoning detection → (https://medium.com/@shail.subscribe/why-your-agent-annotation-pipeline-is-quietly-corrupting-your-reward-model-and-what-to-do-about-5b494bac8234)
- WP2: *"Beyond Task Success: Trajectory-Level Evaluation for Agentic AI"* — cascade errors, framework observability, live API benchmark → [WP2 Medium URL — paste after publish]

---

*→ After publishing this post to Medium, copy the URL (format: medium.com/@yourhandle/evaluating-always-on-ai-...) and paste it into CLAUDE.md under the Published Artifacts section (Step 7).*
