[//]: # "WP3 — Day 35 scaffold. §1-3 to be written today. §4-6 Day 37."

# Evaluating Always-On AI — Privacy-Preserving Data Curation and Model Evaluation for Ambient Wearable Devices

**Target audience:** OpenAI, DeepMind
**Publish target:** Day 42
**Status:** Scaffold only — prose pending

---

## Abstract

<!-- Placeholder: 150-word summary of core claim, headline numbers, and open-source pointer. -->

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

## §4: Agentic Eval for Ambient AI

<!-- Placeholder: trajectory scoring adapted for always-on context — how 5-layer
     TrajectoryScorer and PIA rubric extend to continuous (non-episodic) agent runs.
     Day 37. -->

---

## §5: Benchmark Specification

<!-- Placeholder: concrete metric formulas for ambient eval — consent compliance rate,
     context drift score, passive capture sensitivity, latency SLA under DP noise.
     Day 37. -->

---

## §6: Governance and Regulatory Considerations

<!-- Placeholder: HIPAA, GDPR, and emerging ambient AI regulation implications
     for eval dataset design and model deployment. Day 37. -->

---

## References

<!-- Placeholder: OpenAI HealthBench, DeepMind FACTS, Anthropic 250-doc backdoor,
     ReasonRAG NeurIPS 2025, AgentPRM arXiv 2502.10325, Cohere Command A arXiv 2504.00698,
     Kore.ai enterprise agent report Oct 2025. -->
