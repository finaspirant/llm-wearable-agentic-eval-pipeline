# Beyond Task Success: A Trajectory-Level Evaluation Framework for Multi-Agent Enterprise AI

**Why binary pass/fail evaluation is the proximate cause of the enterprise agent methodology gap — and a three-part framework that closes it.**

*Part 2 of the Wearable AI Agentic Evaluation Pipeline research series.*

*Tags: AI, Machine Learning, Wearable AI, AI Evaluation, Privacy, Agentic AI*

---

89% of enterprises have deployed agent observability tooling. Only 52% have real evaluation coverage.

That 37-point gap is not a tooling problem. It is a methodology problem. Enterprises are generating rich execution traces and discarding the diagnostic signal they contain — because the dominant evaluation paradigm reduces every multi-step agent execution to a single binary: the task succeeded or it did not.

This post explains why that reduction fails, and presents three contributions that fix it.

---

## The Input-Output Evaluation Failure

Consider two agents completing the same support ticket task. Agent A takes four well-sequenced steps, calls the right tools, and resolves it cleanly. Agent B takes twelve steps, invokes the wrong tool three times, recovers from two errors by accident, and arrives at the same terminal state.

Under pass/fail evaluation: both agents score 1.0. Indistinguishable.

But from the perspective of a team generating fine-tuning data, they are entirely different artifacts. Agent A's trajectory is a clean training example. Agent B's trajectory is noise — and if it enters the reward pipeline, it actively degrades the model it was meant to improve.

The DeepMind FACTS benchmark makes the same point from the factuality direction. No frontier model currently exceeds 70% on FACTS across all four dimensions. This is not primarily a capability gap. It is what happens when you optimize for endpoints while ignoring paths — training on outcome signals that reward correct final answers without penalizing hallucinated intermediate steps.

For wearable and ambient AI, the stakes are higher. An always-on agent processing heart rate, blood oxygen, GPS, and ambient audio is making implicit decisions at every step: what to log, what to suppress, when to alert, when to defer. A session-level success metric cannot see these decisions. Automated trajectory-level evaluation is not optional; it is the only viable methodology.

---

## The Trajectory Decomposition Model

A trajectory **T = { s₀, a₁, s₁, a₂, s₂, …, aₙ, sₙ }** is a sequence of alternating agent states and actions. Under pass/fail evaluation, only **(s₀, sₙ)** is evaluated. The trajectory decomposition model treats the full execution path as the unit of evaluation.

This enables three things that endpoint scoring cannot do:

**Partial credit.** A trajectory that executes four of five steps correctly before failing the terminal action receives a score reflecting its intermediate quality. An outcome reward model (ORM) gives it a zero. A process reward model (PRM) trained on trajectory-decomposed data learns to distinguish a nearly-correct chain from a fundamentally broken one. ReasonRAG (NeurIPS 2025) shows this signal difference translates to 18× data efficiency: PRM achieves equivalent task performance with 18× fewer training queries than ORM.

**Error attribution.** A tool call precision failure at step 3 is distinguishable from a planning failure at step 2. When failures can be localized, retraining can be directed at the responsible layer rather than applied globally.

**Non-determinism measurement.** Running the same task 20 times and computing the per-layer score standard deviation reveals where execution variance concentrates — intent, planning, tool calls, recovery, or outcome. You cannot report per-layer variance under pass/fail evaluation.

The annotation schema that populates this framework operates at three granularity levels. Session-level judgments (overall goal achieved, consent respected) feed outcome dashboards. Role-level annotations (delegation quality, handoff integrity, accountability coverage) feed multi-agent debugging workflows. Step-level PRM scores (process reward −1.0 to +1.0, partial credit 0.0 to 1.0) feed the fine-tuning pipeline directly.

---

## Path-Invariant Agreement: A Fix for IRR Collapse

Building a trajectory decomposition pipeline requires human annotators. Human annotators require inter-rater reliability measurement. And here is where standard evaluation methodology breaks.

Standard IRR statistics — Fleiss' κ, Cohen's κ — assume the raters are scoring the same item. For non-deterministic agents, they are not.

Two annotators assigned to the same task may each score a different valid trajectory: Agent A took three steps to reach the goal; Agent B took five. Both trajectories are correct. When the annotators assign different step-level labels, standard IRR treats this as disagreement. It isn't — it's a description of two structurally different paths. The measurement instrument is asking them to agree on something that was never the same object.

We call this the **path-comparison fallacy**.

> **Mode A (standard step-comparison IRR): κ = −0.065. Below-chance agreement. A responsible team would discard these annotations.**

The annotations weren't bad. The measurement was.

---

## Path-Invariant Agreement: A Measurement Fix That Works

Path-Invariant Agreement (PIA) relocates the unit of measurement. Rather than scoring path-specific step sequences across trajectories of different lengths, PIA asks each annotator to score each trajectory against a shared rubric of outcome dimensions:

- **Planning Quality** — does the action sequence reflect coherent decomposition of the session goal?
- **Error Recovery** — when a tool call fails or sensor readings are anomalous, does the agent recover without propagating the error?
- **Goal Alignment** — does each step move the session toward the declared goal, given the current consent model?

These dimensions are path-invariant. An annotator can score them by reading a single trajectory in isolation, with no knowledge of the alternative paths other agents took. The comparison that generated apparent disagreement disappears, because the rubric anchors the judgment to properties of the trajectory as an artifact — not to its step sequence as a path.

The pilot results confirm the hypothesis. Ten trajectory pairs across five wearable scenario types, five annotator personas, same annotations, two measurement modes:

| Method | Fleiss' κ | Interpretation |
|---|---|---|
| Standard IRR (path-comparison) | −0.065 | Poor (below chance) |
| PIA (rubric-dimension scoring) | +0.743 | Substantial |
| **Δ** | **+0.808** | **Poor → Substantial** |

Per-dimension: Planning Quality = 0.705, Error Recovery = 0.827, Goal Alignment = 0.697.

> **The decision to defer HITL evaluation of agentic systems — which a majority of enterprises have made, per the Kore.ai finding — may be based on a measurement artifact, not a genuine annotator capability limitation.**

The data and methodology are publicly available:
[finaspirant/wearable-agent-trajectory-annotations](https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations)

---

## What the Curation Pipeline Actually Does to Evaluation Metrics

We ran an A/B experiment on 100 wearable trajectory runs: top-50 by weighted trajectory score (curated group) versus bottom-50 with 50% terminal-step corruption applied (raw group). Both groups scored on the six Kore.ai enterprise evaluation metrics.

| Kore.ai Metric | Raw | Curated | Δ (%) |
|---|---|---|---|
| Trajectory Success Rate | 0.120 | 0.333 | +177.8% |
| Tool Invocation Accuracy | 0.360 | 1.000 | +177.8% |
| Groundedness (RAGAS) | 0.750 | 0.750 | — |
| Privacy Leak Detection | 0.000 | 0.000 | — |
| Orchestrator Correctness | 1.000 | 1.000 | — |
| Latency SLA Compliance | 1.000 | 1.000 | — |

The two training-relevant metrics lift by +177.8% each. The uncurated pipeline sends the reward model a dataset where 64% of tool calls are incorrect and 88% of trajectories fail the success criterion. The groundedness score shows no lift because the current pipeline lacks a live retrieval layer — both conditions fall back to the RAGAS constant of 0.75. That limitation is the WP3 target.

---

## Framework Benchmark: Task Success Rate Produces a Four-Way Tie

We benchmarked four frameworks — LangGraph, CrewAI, AutoGen, OpenAI Agents SDK — across 10 wearable AI tasks, 3 runs each (120 total trajectories).

Task success rate: **100% / 100% / 100% / 100%.** A four-way tie. Zero signal for framework selection.

The trajectory decomposition layer resolves it:

| Framework | Avg Tokens | Trajectory Score | Annotation Compatibility |
|---|---|---|---|
| LangGraph | 491 | 0.8686 | **High** |
| CrewAI | 808 | 0.8686 | Medium |
| AutoGen (AG2) | 1,012 | 0.8591 | Low |
| OpenAI Agents SDK | 634 | 0.8306 | Medium |

LangGraph consumes 2.1× fewer tokens than AutoGen at equal trajectory quality. More importantly, it scores highest on **annotation compatibility** — the dimension no existing benchmark measures.

Annotation compatibility measures how readily a framework's execution trace maps to the three-layer annotation schema without post-hoc log transformation. LangGraph's typed StateGraph produces node-per-step artifacts that map directly to PIA rubric dimensions. AutoGen's speaker-turn conversational model requires significant parsing to reconstruct step-level records at all.

> **Framework selection has downstream annotation cost consequences that task-success benchmarks make completely invisible. A framework that ties on goal achievement may require 2–3× the annotation infrastructure if its execution model doesn't expose intermediate state as scorable artifacts.**

Live API anchors (claude-sonnet-4-6, wearable_privacy task): OpenAI Agents SDK leads on latency (10.0s mean), CrewAI is slowest (15.0s mean) and highest-token (1,463 avg). Token efficiency rankings from the dry-run benchmark hold in production conditions.

---

## Three Contributions, One Pipeline

**1. Five-Layer Trajectory Decomposition.** Intent (0.15) → Planning (0.25) → Tool Calls (0.25) → Recovery (0.15) → Outcome (0.20), with weight renormalization when the recovery layer is absent. Enables partial credit, error attribution, and per-layer nondeterminism measurement.

**2. Path-Invariant Agreement (PIA).** The first IRR methodology designed for non-deterministic agents. Δκ = +0.808 without new labels, without additional annotators, on identical data.

**3. Role-Level Multi-Agent Attribution.** Four Layer 2 metrics: authority compliance rate, delegation quality, accountability coverage, orchestrator handoff score. Sets cascade_risk=True when a trajectory fails and no agent has accountability_clear=True — the leading indicator that aggregate success rate cannot provide.

---

**Coming next in this series:**

WP3: *"Evaluating Always-On AI: Privacy-Preserving Assessment for Ambient Wearable Agents"* — consent decay, passive capture, AmbientBench-v1 benchmark specification, and the governance framework for continuous ambient AI evaluation.

---

**Code:** https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline

**Dataset:** https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations


---

## About the Author

I'm a machine learning engineer and AI systems researcher building evaluation infrastructure for agentic and ambient AI. This post is part of a 45-day research sprint producing three white papers, an open-source annotation pipeline, and a public benchmark dataset targeting engineering and research roles at Anthropic, OpenAI, Cohere, DeepMind, and Kore.ai. The pipeline implements original methodological contributions — Path-Invariant Agreement, annotation-layer poisoning detection, and a three-layer process-supervised annotation schema — grounded in recent work including ReasonRAG (NeurIPS 2025), AgentPRM (arXiv 2502.10325), and the DeepMind FACTS benchmark (Dec 2025). All code is open-source at [github.com/finaspirant/llm-wearable-agentic-eval-pipeline](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline). The annotated benchmark dataset is available at [huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations](https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations).

## Related Posts in This Series

- WP1: *"Why Your Agent Annotation Pipeline Is Quietly Corrupting Your Reward Model"* — gradient conflict problem, PIA methodology, annotation-layer poisoning detection → (https://medium.com/@shail.subscribe/why-your-agent-annotation-pipeline-is-quietly-corrupting-your-reward-model-and-what-to-do-about-5b494bac8234)