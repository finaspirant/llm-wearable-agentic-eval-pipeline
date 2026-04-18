# Why Your Agent Annotation Pipeline Is Quietly Corrupting Your Reward Model — and What to Do About It

**How standard RLHF preference pairs break for multi-step agents, why inter-annotator agreement collapses, and a three-layer fix that works.**

*Published as part of the Wearable AI Agentic Evaluation Pipeline research series.*

---

The assumption embedded in every RLHF preference pair is deceptively simple: one label is enough to characterize the quality of an interaction.

For single-turn dialogue, that's fine. For a 15-step agentic trajectory where an AI navigates medical sensor data, negotiates consent, selects tools, and escalates to emergency services — it's catastrophically wrong.

This post explains why, and presents a methodology fix.

---

## The Gradient Conflict Problem Nobody Talks About

Consider an agent that takes 14 correct steps and then fails on step 15.

Under outcome reward modeling (ORM) — the backbone of standard RLHF — the reward assigned to every step is:

> **r(s₁) = r(s₂) = ... = r(s₁₄) = 0. r(s₁₅) = −1.**

The gradient signal propagated backward through the policy is uniformly negative across a trajectory that was correct for 93% of its length. The model learns to suppress the reasoning patterns it should amplify.

> **We measured this in 20 synthetic wearable-AI trajectories. The gradient conflict rate was 100%. Every single outcome-failed trajectory contained a majority of positively-rewarded intermediate steps.**

Process reward modeling (PRM) assigns an independent reward r_PRM ∈ [−1, +1] to each step, with a continuous partial credit signal r_partial ∈ [0, 1] for steps that are directionally correct but not optimal. ReasonRAG (NeurIPS 2025) shows PRM achieves equivalent performance with **18× fewer training queries** than ORM — because it doesn't throw away the intermediate signal.

The problem is that PRM requires step-level annotation. And step-level annotation for agents has a measurement problem nobody has fixed yet.

---

## The IAA Collapse: When Agreement Metrics Lie

Standard inter-annotator agreement (IAA) — Cohen's κ, Fleiss' κ — measures whether raters agreed on the same label for the same artifact.

For agents, this breaks in a specific way.

Two annotators can both believe: *"this agent understood the goal, selected appropriate tools, and achieved the right outcome."* But if one annotator saw the direct 3-step path and the other saw the exploratory 5-step path, they'll register as disagreeing on every step-specific label — even though their underlying quality judgment is identical.

We call this the **path-comparison fallacy**.

> **Mode A (standard step-comparison IAA): κ = −0.065. "Poor" agreement. A responsible team would discard these annotations.**

The annotations weren't bad. The measurement was.

---

## Path-Invariant Agreement: A Fix That Works

We ran the same annotation data through a different measurement: **Path-Invariant Agreement (PIA)**.

Instead of measuring agreement on step-specific choices, PIA measures agreement on three rubric dimensions:

- **Planning quality** — did the agent select an appropriate action sequence given its goal and context?
- **Error recovery** — when the agent encountered ambiguity or failure, did it recover appropriately?
- **Goal alignment** — did the agent's terminal action correctly address the user's stated intent?

These dimensions are invariant to path. An agent that takes the direct route and one that takes the exploratory route can both score 4/5 on planning quality if they both chose well given what they knew.

> **Mode B (PIA rubric IAA): κ = +0.743. "Substantial" agreement. Same annotations. No new labels.**

**Δκ = +0.808 on identical data.**

Per-dimension: planning_quality = 0.705, error_recovery = 0.826, goal_alignment = 0.697.

The pilot dataset — 500 annotated records across 50 trajectories and 5 annotator personas — is publicly available on HuggingFace:
[finaspirant/wearable-agent-trajectory-annotations](https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations)

---

## The Poisoning Problem Nobody Has Solved at the Right Layer

The Anthropic 250-document backdoor finding (October 2025) established that only **250 documents — 0.00016% of training data** — suffice to backdoor a model of any size. The attack is count-based, not proportion-based.

This means the annotation layer is the attack surface.

A malicious annotator who suppresses `privacy_compliance` scores while inflating `step_quality` scores creates training examples that reward privacy-violating agent behaviors — while the aggregate annotation statistics look clean.

We built a two-layer detection architecture:

**Layer 1: MAD outlier scoring.** Each annotator gets a suspicion score [0, 1] based on deviation from the per-dimension consensus. Effective against independent bad actors.

**Known failure mode (documented honestly):** Three or more colluding annotators who submit identical biased labels each score 0.0. The MAD detector is completely blind to coordinated collusion. F1 = 0.0 at every threshold.

**Layer 2: Confident Learning (cleanlab).** Measures deviation from a model-estimated label distribution rather than annotator consensus. Provides partial recovery in exactly the collusion scenario where Layer 1 fails.

> **This is, to our knowledge, the only published detection methodology targeting the annotation layer — before poisoned labels enter the reward model gradient.**

Model-layer detection (perplexity differentials, activation analysis) is useful post-hoc. It cannot prevent gradient contamination from occurring. Annotation-layer detection is a necessary complement.

---

## What the HH-RLHF Data Tells Us

We ran 200 pairs from Anthropic's HH-RLHF dataset through our pipeline's IAA measurement.

| Dimension | Fleiss' κ | Krippendorff's α |
|---|---|---|
| helpfulness | −0.121 | −0.245 |
| harmlessness | −0.093 | −0.253 |
| coherence | +0.001 | −0.002 |
| **Overall** | **−0.071** | — |

HH-RLHF annotators are not poor raters. This is what IAA looks like when you measure multi-dimensional preference-pair annotation with standard metrics.

The Cohere Command A paper (arXiv 2504.00698) annotated 800 prompts with 65 raters on a 5-point scale — and reported no κ, no α. Our results suggest this wasn't an oversight. Measured preference-pair IAA is low enough that accurate reporting invites questions the methodology is not equipped to answer.

> **HH-RLHF was designed for single-turn dialogue alignment and has driven substantial progress there. The gaps become acute when you extend the paradigm to multi-step agentic tasks.**

What our framework adds: step-level PRM labels, PIA calibration, and gradient conflict detection — the three things preference pairs don't provide.

---

## Framework Benchmark: Annotation Amenability Varies More Than You'd Expect

We benchmarked four frameworks — LangGraph, CrewAI, AutoGen, OpenAI Agents SDK — across 10 tasks and 120 trajectory runs.

All four frameworks achieved the same average trajectory score: **0.8235**.

What varies is how annotation-friendly each framework's execution model is:

| Framework | Avg tokens | Annotation amenability | Key finding |
|---|---|---|---|
| LangGraph | 519 | Highest | Named nodes, typed state, clean step separation |
| OpenAI Agents SDK | ~620 | High | Clean event log, fast (10.3s), missing HITL hook |
| CrewAI | ~730 | Medium | Delegation cascades inflate step counts |
| AutoGen | ~950 | Lowest | Turn-segmentation required, attribution ambiguous |

> **Framework selection has downstream consequences for annotation cost that outcome-only benchmarks make invisible.**

A framework that ties on task success may require 2–3× the annotation infrastructure if its execution model doesn't expose intermediate state as observable artifacts.

---

## Three Contributions, One Pipeline

**1. Path-Invariant Agreement (PIA).** The first IAA methodology designed for non-deterministic agents. Δκ = +0.808 without new labels. Methodology, rubric, dataset, and code released open-source.

**2. Annotation-layer poisoning detection.** Two-layer architecture (MAD + Confident Learning) targeting the annotation layer before gradient contamination occurs. First published methodology at this layer. Blind spot documented.

**3. Open-source annotation pipeline.** Three-layer schema, calibration protocol, PRM annotator, poisoning detector, 503-test suite, HuggingFace dataset. mypy strict. ruff clean.

---

**Coming next in this series:**

WP2: *"Beyond Task Success: Trajectory-Level Evaluation for Agentic AI"* — cascade errors, framework observability metrics, live API benchmark. (Day 30)

WP3: *"Evaluating Always-On AI"* — consent decay, passive capture, privacy-preserving eval for ambient wearable systems. (Day 42)

---

**Code:** https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline

**Dataset:** https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations

---

## About the Author

I'm a machine learning engineer and AI systems researcher building evaluation infrastructure for agentic and ambient AI. This post is part of a 45-day research sprint producing three white papers, an open-source annotation pipeline, and a public benchmark dataset targeting engineering and research roles at Anthropic, OpenAI, Cohere, DeepMind, and Kore.ai. The pipeline implements original methodological contributions — Path-Invariant Agreement, annotation-layer poisoning detection, and a three-layer process-supervised annotation schema — grounded in recent work including ReasonRAG (NeurIPS 2025), AgentPRM (arXiv 2502.10325), and the Anthropic 250-document backdoor finding (October 2025). All code is open-source at [github.com/finaspirant/llm-wearable-agentic-eval-pipeline](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline). The annotated benchmark dataset is available at [huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations](https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations).
