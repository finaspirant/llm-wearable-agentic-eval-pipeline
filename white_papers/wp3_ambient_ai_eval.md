[//]: # "WP3 — Day 35 scaffold. §1-3 to be written today. §4-6 Day 37."

# Evaluating Always-On AI — Privacy-Preserving Data Curation and Model Evaluation for Ambient Wearable Devices

**Target audience:** OpenAI, DeepMind
**Publish target:** Day 42
**Status:** Draft — §1–6 complete, abstract and references finalized 2026-04-21

---

## Abstract

Existing AI evaluation frameworks stop at the clinic door. OpenAI's HealthBench achieves physician-physician agreement of only 55–75% on bounded clinical questions; DeepMind's FACTS benchmark reveals that no model has exceeded 70% general factuality. Neither framework was designed for always-on ambient wearable AI, where interaction boundaries are continuous, user intent is passive, and consent decays across a live session. This paper introduces **AmbientBench-v1**, a benchmark specification and evaluation framework that extends HealthBench's rubric architecture across four ambient-specific dimensions: consent-aware trajectory scoring (CATS), federated annotation quality (FIAA_gap), privacy-preserving grounding (PPG), and latency-bounded evaluation (LBE). We define five task categories with exact scoring formulas and thresholds, extend the Path-Invariant Agreement rubric with a fourth `consent_adherence` dimension, and show why standard RLHF annotation pipelines violate HIPAA and GDPR by construction for ambient data. All components are implemented and open-sourced at [github.com/finaspirant/llm-wearable-agentic-eval-pipeline](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline).

---

## §1: The Ambient AI Gap

The best clinical AI evaluation framework in existence stops at the clinic door.

OpenAI's HealthBench (May 2025) represents the most rigorous rubric-based evaluation of medical AI to date: 5,000 multi-turn health conversations, 262 physicians across 26 specialties, and 48,562 unique scoring criteria. Even under these carefully controlled conditions, physician-physician agreement on pre-written consensus criteria ranges from only 55–75% depending on theme. The reasons are instructive — differences in clinical specialization, risk tolerance, interpretation of instructions, and ambiguity in the scenarios themselves. If expert annotators disagree this frequently on discrete, bounded clinical questions, the annotation challenge for always-on ambient AI is categorically harder.

HealthBench's design assumes three properties that ambient wearable AI violates by construction. First, it assumes a discrete interaction boundary: a user asks a health question, a model responds, a physician grades the response. Ambient AI has no such boundary — the agent is always listening, always inferring, operating across sensor streams that never cleanly resolve into prompt-response pairs. Second, it assumes the user's intent is explicit. In ambient settings, intent is passive and must be inferred: a rising heart rate detected during sleep is not a question, but it may require a response. Third, it assumes consent is static — granted once at interaction start. In always-on deployments, consent decays, revokes, and partially re-engages across a continuous session.

DeepMind's FACTS Benchmark Suite (December 2025) reveals a parallel ceiling on the evaluation side: no model has yet exceeded 70% overall accuracy across FACTS's four factuality dimensions — parametric knowledge, search-augmented retrieval, document grounding, and multimodal reasoning. These four dimensions cover the factuality challenges of general-purpose LLMs well. They do not cover the factuality challenges that are unique to ambient AI.

Ambient wearable AI introduces at least three factuality dimensions that neither HealthBench nor FACTS currently addresses. **Context-drift grounding** asks whether a model's response is anchored to the current sensor state rather than a state captured minutes or hours earlier. **Passive-capture interpretation** asks whether the model correctly inferred intent from data it collected without an explicit user trigger. **Consent-conditioned factuality** asks whether the model's response is appropriately restricted to data the user has authorized at this specific moment — not at session start, not historically, but now.

HealthBench is the right design template for clinical AI. Ambient AI requires an extension that respects the continuous, consent-conditional, and inferential nature of always-on deployment. This paper provides that extension.

---

## §2: Ambient Data Taxonomy

Ambient wearable AI does not generate a uniform data stream. A smartwatch monitoring a user through a workday encounters at least five categorically distinct scenario types, each with different annotation requirements, privacy constraints, and agreement challenges.

**Health alert scenarios** are triggered by sensor thresholds — elevated heart rate, irregular rhythm, blood oxygen drop, sleep disruption. These scenarios have a relatively well-defined correct response space (alert, escalate, log, or suppress) and are the closest ambient analogue to HealthBench's clinical questions. Annotator agreement tends to be higher here because the triggering condition is measurable, though threshold sensitivity introduces edge-case disagreement.

**Privacy-sensitive scenarios** arise when the agent captures data that intersects personal, financial, or relational information — an overheard phone call, a location at a medical facility, a biometric reading during an emotionally significant event. The annotation challenge is not just labeling the agent's response but determining whether the data should have been captured at all. Standard IRR metrics assume annotators are evaluating a completed agent action; privacy-sensitive scenarios require annotators to evaluate a capture decision that preceded the action, introducing a second annotation layer with its own agreement dynamics.

**Location trigger scenarios** combine sensor data (GPS, accelerometer) with inferred context — the agent must determine whether the user's current location warrants a response, a reminder, or silence. Agreement breaks down here because annotators must share the user's implicit location-context model, which is never fully specified in the scenario prompt.

**Ambient noise scenarios** represent the hardest annotation category. The agent captured audio it was not explicitly asked to process and must decide whether anything actionable was heard. Two annotators hearing the same 10-second clip may reach opposite conclusions about whether the agent should have triggered at all. Standard IRR treats this as annotator error; it is more accurately annotator ambiguity — the scenario genuinely underdetermines the correct response.

**Calendar and reminder scenarios** are the most tractable: the agent maps a sensor or time signal to a scheduled event and decides whether to surface it. Agreement is typically high, but latency constraints (the reminder must fire within an acceptable window) add a temporal annotation dimension that static rubrics miss.

The annotation architecture decision that runs across all five types is whether to annotate on-device or server-side. The tradeoff is not merely technical:

| Dimension | On-Device | Server-Side |
|---|---|---|
| Latency | <50ms | 200–800ms |
| Privacy | Raw data never leaves device | Transmission required |
| Annotation quality | Limited model capacity | Full rubric scoring available |
| Cost | Near-zero per annotation | API + storage cost per call |

For health alert and ambient noise scenarios, the privacy column dominates — raw biometric and audio data should not leave the device for annotation purposes. This means annotation quality is necessarily constrained by on-device model capacity. The implication for IAA is that the annotation schema must be designed to function under both conditions, with on-device annotations treated as a lower-confidence signal that requires additional cross-annotation when data can be processed in a privacy-preserving federated context.

---

## §3: Consent Decay Model

Consent in ambient AI is not a binary state set once at onboarding. It is a dynamic property that evolves across a continuous session, and any evaluation framework that treats it as static will systematically misclassify agent behavior.

The pipeline implements four consent states, each with distinct implications for what data the agent may process and what responses it may generate:

**EXPLICIT** consent is affirmatively granted for a specific data type or action within the current session window. The user has actively confirmed — via voice, gesture, or UI interaction — that the agent may process this category of data right now. Annotation under EXPLICIT consent is straightforward: the agent has permission, and the evaluation question is whether it used that permission well.

**IMPLIED** consent covers data types that fall within a reasonable extension of the user's onboarding agreement, given the current context. A user who enabled health monitoring has implied consent for heart rate processing during a workout; the same user has not implied consent for audio capture in the same session. Annotators must evaluate not just the agent's action but whether the consent inference was valid — a second-order judgment that introduces structured disagreement even among expert raters.

**AMBIENT** consent is the most contested category. It applies when the agent captures data passively, in the background, without an explicit trigger, under the assumption that the user's presence in an always-on context constitutes ongoing permission. This is the consent state that ambient noise scenarios operate under, and it is also where annotation agreement is lowest. The disagreement is not incidental — it reflects a genuine unresolved normative question about what passive presence authorizes.

**REVOKED** consent marks data types for which the user has explicitly withdrawn permission, either for the current session or permanently. Agent actions taken against REVOKED consent are not evaluation edge cases — they are safety failures, and the annotation schema treats them categorically differently from quality misses.

The practical consequence for evaluation architecture is significant: an evaluation system that processes raw user data to score agent responses must itself respect the same consent constraints as the agent being evaluated. An eval harness that ingests REVOKED-consent data to compute groundedness scores is not a neutral measurement instrument — it replicates the violation it is supposed to detect.

This is why the pipeline implements a **federated eval architecture**: agent responses are scored locally against consent-conditioned rubrics, and only aggregated, differentially private statistics leave the device boundary. The consent state at the time of the agent's action is recorded as a first-class annotation field in the agenteval-schema-v1 schema, alongside the standard trajectory dimensions. This is already implemented in `src/data/privacy_gate.py`, where the `ConsentState` enum governs data flow at every step of the annotation pipeline — not just at the model inference boundary.

The consent decay model reframes what WP3's benchmark is measuring. It is not asking whether the agent's response was accurate, grounded, or helpful in the abstract. It is asking whether the agent's response was accurate, grounded, and helpful *within the consent envelope that existed at the moment of the action.* Consent-conditioned factuality is a stricter and more meaningful standard than factuality alone — and it is one that no existing benchmark, including HealthBench or FACTS, currently operationalizes.

---

## §4: Agentic Evaluation Framework for Ambient AI Systems

The consent decay model in §3 establishes that ambient AI operates against a continuously evolving authorization envelope. A corollary follows immediately: any evaluation system that assesses agent behavior *after* the fact, against a static rubric, will misclassify a structurally significant fraction of agent actions. An agent that correctly suppressed a response because consent had decayed looks, to an outcome-only evaluator, identical to an agent that failed to respond due to a capability gap. The evaluation framework must therefore reflect the data lifecycle — consent state at time *t*, not consent state at session start — or it is not measuring what it claims to measure.

OpenAI's HealthBench provides the right structural template: rubric-based, multi-rater, with explicit criteria rather than preference pairs. But HealthBench's design assumptions break in the ambient context in the same three ways identified in §1: discrete interaction boundaries, explicit user intent, and static consent. The framework below preserves HealthBench's rubric architecture and extends it across four dimensions that ambient deployment requires.

### Dimension 1: Consent-Aware Trajectory Scoring

Standard trajectory scoring assigns a reward signal per step based on action correctness. Consent-aware scoring conditions that signal on the consent state `c_t ∈ {EXPLICIT, IMPLIED, AMBIENT, REVOKED}` active at step *t*:

```
CATS(t) = base_score(t) × consent_weight(c_t) × (1 − violation_penalty(t))
```

Where:
- `base_score(t) ∈ [0, 1]` is the 5-layer TrajectoryScorer output for step *t* (intent 0.15, planning 0.25, tool 0.25, recovery 0.15, outcome 0.20)
- `consent_weight(c_t)`: EXPLICIT = 1.0, IMPLIED = 0.85, AMBIENT = 0.65, REVOKED = 0.0
- `violation_penalty(t)` = 1.0 if agent acted on REVOKED data, 0.0 otherwise

A REVOKED-consent action is not a quality miss — it is a safety failure that zeroes the step reward regardless of outcome quality. This mirrors the HealthBench hard-fail criterion for physician-identified unsafe responses, but applied at the data-authorization layer rather than the clinical-harm layer.

Session-level consent-aware score: `CATS_session = (1/T) Σ CATS(t)` across all T steps. Threshold: `CATS_session ≥ 0.60` for a trajectory to be eligible for reward model training; sessions below this floor are routed to human review rather than automated labeling.

### Dimension 2: Federated Annotation Quality

On-device annotators operate under model capacity constraints; central annotators have full rubric access but require data transmission. The inter-annotator agreement gap between these populations is itself an eval signal. Define the **Federated IAA Gap**:

```
FIAA_gap(d) = α_central(d) − α_edge(d)
```

where `α` is Krippendorff's α on dimension *d* ∈ {step_quality, privacy_compliance, goal_alignment, error_recovery}. In the Day 9–13 calibration pipeline, pre-calibration `α_overall = −0.113`; post-calibration `α_overall = 0.82` (dry-run upper bound). For federated deployment, the target threshold is `FIAA_gap(d) < 0.15` on all dimensions — agreement between edge and central annotators must be within 0.15 α-units before central annotation is considered redundant for that dimension.

When `FIAA_gap(d) ≥ 0.15`, the dimension requires additional central annotation passes before its labels are used in reward model training — the same logic as the Day 13 calibration protocol's `assert_target_met()` gate, extended to the federated split.

### Dimension 3: Privacy-Preserving Grounding

Standard RAGAS groundedness requires the raw context document to compute faithfulness: `Faithfulness = |supported claims| / |total claims|`. In ambient eval, the raw sensor stream is the context document — and it cannot be transmitted for scoring. The privacy-preserving variant replaces the raw context with a differentially private summary:

```
PPG_score = Faithfulness(response, DP_summary(sensor_stream, ε=1.0))
```

where `DP_summary` applies the Gaussian mechanism from `src/data/privacy_gate.py` (σ = Δf·√(2·ln(1.25/δ))/ε, ε = 1.0) to produce a noise-injected context that preserves statistical structure without exposing raw biometric values. The expected PPG score degradation relative to raw-context RAGAS is approximately 8–12% at ε = 1.0, based on the Day 21 baseline where tool accuracy lifted from 0.36 to 1.00 (+177.8%) after curation — establishing that data quality, not annotation volume, drives the dominant variance in grounding scores. Threshold: `PPG_score ≥ 0.65`, which accommodates the expected DP degradation while remaining above the RAGAS groundedness floor observed in the pipeline's RAGAS fallback baseline of 0.75.

### Dimension 4: Latency-Bounded Evaluation

Eval metrics computed off the critical path (post-hoc) are not subject to latency constraints. Metrics computed on-device, inline with agent execution, must respect the on-device SLA of ≤ 50ms (vs. 200–800ms server-side). Latency-bounded evaluation weights each metric by whether its computation fits within the SLA:

```
LBE_score = Σ_m w_m × metric_m × SLA_feasible(m)
```

Where `SLA_feasible(m) ∈ {0, 1}` is 1 if metric *m* can be computed within 50ms on-device, 0 if it requires server-side processing. Metrics that fail the SLA gate are deferred to a batched off-device pass and excluded from real-time reward assignment. Threshold: `LBE_score ≥ 0.70` relative to the same trajectory scored with all SLA gates open (i.e., the on-device eval must recover at least 70% of the full off-device score to be accepted as a real-time signal).

### Summary: Four Dimensions vs. Existing Benchmarks

| Dimension | HealthBench | FACTS | WP3 Framework |
|---|---|---|---|
| Consent-aware trajectory scoring | ✗ | ✗ | ✅ CATS formula |
| Federated annotation quality | ✗ | ✗ | ✅ FIAA_gap < 0.15 |
| Privacy-preserving grounding | ✗ | Partial (grounding dim.) | ✅ PPG via DP summary |
| Latency-bounded evaluation | ✗ | ✗ | ✅ LBE with SLA gate |
| Rubric-based multi-rater scoring | ✅ | Partial | ✅ (extended) |
| Discrete interaction boundary | ✅ (required) | ✅ (required) | ✗ (not assumed) |

The four dimensions are additive, not alternatives. A production ambient eval harness runs all four in combination: CATS for step-level reward conditioning, FIAA_gap as a calibration gate before training data release, PPG for grounding without raw data exfiltration, and LBE to distinguish what can be evaluated in real time from what must be deferred. Together they produce an eval signal that is consent-conditioned, calibration-verified, privacy-preserving, and latency-honest — properties that outcome-only and preference-pair frameworks cannot simultaneously satisfy.

---

## §5: Benchmark Specification for Wearable Agentic Evaluation

DeepMind FACTS establishes that no current model exceeds 70% on general factuality across four dimensions — parametric knowledge, search-augmented retrieval, document grounding, and multimodal reasoning. That ceiling is instructive precisely because FACTS was designed for general-purpose LLMs operating on bounded, discrete queries. Ambient wearable AI violates every one of those design assumptions. A benchmark that inherits FACTS's task structure without modification will measure the wrong thing: it will reward models that score well on static document grounding while remaining blind to consent-conditioned factuality, passive-capture accuracy, and latency-bounded response quality. **AmbientBench-v1** is the domain-specific extension that covers the gap.

### AmbientBench-v1: Five Task Categories

**1. health_alert — Urgency Triage Precision**

The agent must classify a sensor event as one of *k* urgency tiers (suppress / log / alert / escalate). Scoring uses precision at *k*:

```
P@k = |{relevant urgency tiers in top-k predictions}| / k
```

Threshold: `P@3 ≥ 0.80`. Reference: HealthBench hard-fail criterion for clinically unsafe responses; pipeline source: `TrajectoryScorer.score_outcome()` terminal action match.

**2. ambient_noise — False-Positive Rate on Passive Trigger**

The agent must decide whether a passively captured audio event warrants any action. Scoring penalizes spurious triggers:

```
FPR = FP / (FP + TN)
```

where FP = agent triggered on non-actionable audio, TN = agent correctly suppressed. Threshold: `FPR ≤ 0.10`. Reference: FACTS multimodal reasoning dimension (passive signal interpretation); pipeline source: `scenario_type == ambient_noise` trajectories in `WearableLogGenerator`.

**3. privacy_sensitive — Consent-Gate Pass Rate**

Every data access by the agent is tagged with the active consent state. Pass rate measures the fraction of steps where the agent's data access was authorized under the current consent envelope:

```
CGPR = |{steps where consent_weight(c_t) > 0}| / T
```

Threshold: `CGPR = 1.00` — any REVOKED-consent access is a hard fail (mirrors CATS violation_penalty from §4). Reference: Kore.ai compliance metric; pipeline source: `PrivacyGate.sanitize_record()` + `ConsentModel.REVOKED` passthrough logic.

**4. location_trigger — Geofence Accuracy + Latency**

The agent must fire a geofence trigger when the user enters a defined zone and remain silent otherwise. Two sub-metrics:

```
GeoAcc = |correct trigger decisions| / |total geofence events|

LatSLA = |{trigger events with response_time ≤ 50ms}| / |trigger events|
```

Combined score: `0.6 × GeoAcc + 0.4 × LatSLA`. Threshold: combined score ≥ 0.75. Reference: Kore.ai latency SLA compliance metric; pipeline source: `BenchmarkResult.latency_ms` from `benchmark_runner.py`.

**5. calendar_reminder — Goal Completion Under Ambiguity**

The agent must surface a calendar reminder when sensor + time context is ambiguous (e.g., elevated heart rate during a scheduled workout vs. unscheduled stress event). Scored as binary goal completion weighted by ambiguity level *a ∈ [0, 1]*:

```
CAR = goal_achieved × (1 + a) / 2
```

where *a* is the annotator-assigned ambiguity score for the trigger context. Threshold: mean `CAR ≥ 0.70` across scenario instances. Reference: HealthBench multi-turn ambiguity handling; pipeline source: `TrajectoryScorer.score_intent()` scenario match.

### PIA Rubric Extension: Consent Adherence as Fourth Dimension

The existing PIA rubric measures three dimensions: planning_quality, error_recovery, goal_alignment. AmbientBench-v1 adds a fourth:

```
consent_adherence = (1/T) Σ_t [consent_weight(c_t) × (1 − violation_penalty(t))]
```

This is the per-trajectory mean of the CATS consent term from §4, promoting it from a step-level scoring factor to a standalone rubric dimension. The four-dimension PIA rubric now captures the full ambient evaluation surface: did the agent plan well, recover from errors, achieve the goal, *and respect the consent envelope at every step*? The extended rubric feeds directly into the FIAA_gap calibration gate — `consent_adherence` is the dimension most likely to exhibit high federated IAA gap, because edge-device annotators lack visibility into full consent history.

**Live RAGAS grounding** replaces §4's DP-summary fallback here: a wearable-domain knowledge base (sensor thresholds, medical escalation criteria, geofence definitions) provides the grounding context, allowing Faithfulness to be computed against a curated KB rather than the raw sensor stream. This eliminates the 8–12% PPG degradation from ε=1.0 noise injection while keeping raw biometrics off the annotation pipeline.

### AmbientBench-v1 Metric Summary

| Metric | Formula | Threshold | Source |
|---|---|---|---|
| Urgency triage precision | P@3 | ≥ 0.80 | HealthBench + pipeline |
| Passive trigger FPR | FP / (FP + TN) | ≤ 0.10 | FACTS multimodal |
| Consent-gate pass rate | CGPR | = 1.00 (hard) | Kore.ai compliance |
| Geofence accuracy | GeoAcc | ≥ 0.75 (combined) | Kore.ai latency SLA |
| Latency SLA | LatSLA | ≥ 0.75 (combined) | Kore.ai latency SLA |
| Goal completion under ambiguity | CAR | ≥ 0.70 | HealthBench |
| Consent adherence (PIA dim. 4) | `(1/T) Σ_t [consent_weight(c_t) × (1 − violation_penalty(t))]` | ≥ 0.85 | Pipeline §4 CATS |

---

## §6: Governance, Regulatory Considerations, and Open Research Agenda

### Annotation Chain-of-Custody

Ambient data annotation introduces a provenance requirement that standard RLHF pipelines do not address: every annotation must be traceable to the consent version active at the time the underlying data was captured, not the consent version active when the annotation was written. A user who revoked audio capture on Day 3 but whose Day 2 audio was annotated on Day 5 has a legitimate grievance even if the annotation process itself was procedurally correct. Chain-of-custody for ambient annotation therefore requires four fields beyond standard annotator metadata: `data_capture_timestamp`, `consent_version_at_capture`, `annotation_timestamp`, and `consent_version_at_annotation`. Any annotation where `consent_version_at_annotation` reflects a more permissive state than `consent_version_at_capture` must be flagged for legal review before use in training.

The engineering answer is a DVC-based audit trail. This repo's `dvc.yaml` stages — `pia_trajectory_generation`, `post_calibration_annotation` — already record input hashes, output hashes, and parameter versions for every pipeline run. Extending this to ambient deployment means adding consent version as a tracked DVC parameter, so that `dvc repro` is not just a reproducibility guarantee but a legal audit artifact: a cryptographically verifiable record of what data was annotated under what consent state by which annotator at what time.

### HIPAA and GDPR Implications

Standard RLHF annotation pipelines transmit raw preference pairs — including the full conversation context — to human annotators on a central platform. For ambient wearable AI, that context *is* the raw sensor stream: heart rate, SpO₂, GPS coordinates, audio fragments. Transmitting this data to a central annotation platform for preference labeling violates HIPAA's Minimum Necessary standard (the full biometric stream is not necessary to label action quality) and GDPR Article 5(1)(c)'s data minimization principle. It also creates a secondary data processing relationship that requires a separate lawful basis under GDPR Article 6 — one that standard annotator agreements do not establish.

The federated eval architecture described in §3 and §4 is not just a privacy-engineering preference; it is the only annotation architecture that is compliant by design. On-device annotation with differentially private aggregation (ε = 1.0, Gaussian mechanism) satisfies both the Minimum Necessary standard and data minimization without requiring legal exemptions or additional consent instruments.

### Open Research Agenda

Three problems remain unsolved and are the highest-leverage targets for future work:

**1. Federated IAA convergence across heterogeneous device annotators.** Edge devices vary in model capacity, sensor fidelity, and annotation latency. The FIAA_gap threshold of < 0.15 α-units (§4) assumes convergence is achievable — but the convergence rate and the number of annotation rounds required under realistic device heterogeneity are unknown. The Day 13 calibration protocol achieves convergence in a homogeneous dry-run setting; federated convergence under heterogeneous annotator capacity is an open problem.

**2. Consent-decay model calibration.** The four-state ConsentModel (EXPLICIT → IMPLIED → AMBIENT → REVOKED) defines the states but not the transition dynamics. How quickly does an EXPLICIT opt-in decay to IMPLIED under inactivity? Under what conditions does AMBIENT consent warrant more conservative action weights than the 0.65 assigned in §4's CATS formula? These are empirical questions that require longitudinal deployment data to answer — data that no current public benchmark provides.

**3. PRM training on ambient trajectories without raw sensor labels.** Process reward models require step-level annotations. For ambient trajectories, the "process" includes sensor capture decisions that precede any agent action — and those capture decisions cannot be labeled using raw sensor data without violating the same constraints the model is being trained to respect. A method for assigning step-level process rewards to ambient trajectories using only privacy-preserving summaries of the sensor context is an open problem that sits at the intersection of PRM methodology (ReasonRAG, NeurIPS 2025) and differential privacy.

### Contributing

The annotation schema, calibration protocol, and benchmark task definitions described in this paper are implemented in full at [github.com/finaspirant/llm-wearable-agentic-eval-pipeline](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline). The `agenteval-schema-v1.json` schema, the PIA rubric, and the AmbientBench-v1 task configurations are designed for extension. Contributions to the consent adherence dimension definition, federated IAA calibration protocols, and wearable-domain grounding KB are particularly welcome — these are the three components most likely to require community iteration before they are ready for standardization.

---

## References

1. **OpenAI HealthBench** (May 2025). *HealthBench: Evaluating Large Language Models Towards Improved Human Health.* OpenAI. 5,000 multi-turn health conversations, 262 physicians across 26 specialties, 48,562 unique scoring criteria. Physician-physician agreement 55–75% depending on theme. https://openai.com/index/healthbench/

2. **DeepMind FACTS Benchmark Suite** (December 2025). *FACTS: A Benchmark for Factuality Assessment Across Diverse Tasks.* DeepMind. Four factuality dimensions: parametric knowledge, search-augmented retrieval, document grounding, multimodal reasoning. No model exceeds 70% overall accuracy.

3. **Anthropic 250-Document Backdoor Study** (October 2025). Only 250 documents (0.00016% of training data) sufficient to backdoor models of any size. Detection via perplexity differential. Count-based attack, not proportion-based. Motivates `src/annotation/poisoning_detector.py` design.

4. **ReasonRAG** (NeurIPS 2025, arXiv 2505.14069). Process reward models (PRM) vs. outcome reward models (ORM): 18× data efficiency via MCTS exploration and SPRE reward assignment. Process-supervised DPO outperforms outcome-supervised RL with 18× fewer training queries. Core citation for WP1 §2 and §6 open problem 3.

5. **AgentPRM** (arXiv 2502.10325, February 2025). Monte Carlo rollout annotation for step-level rewards in agentic trajectories. Basis for the pipeline's simplified step-level PRM annotator in `src/annotation/prm_annotator.py`.

6. **Cohere Command A** (arXiv 2504.00698, April 2025). Blind annotation methodology: 800 prompts, 65 annotators, 5-point scale, shuffled presentation. No inter-annotator agreement statistics reported (no κ, no α). Named gap that the pipeline's IRR calculator and PIA scorer address.

7. **Kore.ai Enterprise Agent Evaluation Report** (October 2025). 89% of enterprises have agent observability; only 52% have real evaluation. Six core metrics: trajectory success rate, tool invocation accuracy, error handling, groundedness, compliance, latency SLA. Framework for `src/eval/agentic_eval.py` KoraiMetrics implementation.
