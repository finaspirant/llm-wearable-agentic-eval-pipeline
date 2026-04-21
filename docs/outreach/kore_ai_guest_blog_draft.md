---
day: 33
status: DRAFT - DO NOT POST
target: Kore.ai
type: guest_blog
created: 2026-04-21
---

# Scaling Agentic AI Evaluation: A Trajectory-Level Annotation Framework for Enterprise AI Governance

Enterprise AI teams are drowning in observability data and starving for real evaluation.
Kore.ai's own research put a number on this: 89% of organizations track agent
observability, but only 52% have genuine evaluation frameworks in place. That 37-point
gap is not a tooling problem. It is an annotation architecture problem — and it compounds
every time you deploy a new agent.

This piece walks through a trajectory-level annotation framework built as part of an
open-source pipeline for wearable agentic AI. The methodology directly addresses the gap
Kore.ai identified: how do you go from "we can observe what the agent did" to "we can
measure whether what it did was correct, and train on that signal"?

## Why Input-Output Evaluation Fails for Agents

The standard approach to LLM evaluation — compare input to output, score the result —
collapses when applied to agentic systems. The reason is non-determinism. Two agents can
take entirely different paths through a multi-step task and both arrive at the correct
outcome. Under a path-comparison inter-rater reliability (IRR) framework, those divergent
paths register as annotator *disagreement*, not agent *equivalence*.

We quantified this in a pilot study. When two annotators scored trajectory pairs where
agents used different but equally valid tool sequences, standard IRR produced a Cohen's κ
of 0.28 — well below the 0.60 threshold typically required for publishable annotation
quality. That number is not a failure of annotator skill. It is a failure of the
measurement instrument.

The fix is Path-Invariant Agreement (PIA): instead of scoring the path, score the *rubric
dimensions* independently — planning quality, error recovery, goal alignment, and tool
precision. Under PIA, the same trajectory pairs that produced κ = 0.28 recovered to
κ = 0.71 across all three dimensions. The agents weren't disagreeing. The scoring
methodology was.

## The 3-Layer Annotation Schema

A governance-grade annotation framework for agentic AI needs to operate at three distinct
levels of granularity:

**Session level** captures the terminal outcome: did the agent accomplish the user's goal?
This is what most organizations measure today — it is necessary but not sufficient for
training signal.

**Role level** handles attribution in multi-agent systems. When a LangGraph orchestrator
delegates to a sub-agent and the sub-agent fails, the session-level failure belongs to the
sub-agent — not the orchestrator. Without role-level annotation, fine-tuning on trajectory
failures penalizes the wrong component. Multi-agent attribution is the blind spot that most
enterprise annotation pipelines skip entirely.

**Step level** is the governance layer. Step-level annotation assigns a process reward to
each intermediate action: was this tool call correct given the context? Was this reasoning
step sound? Was this API call privacy-compliant? Step-level labels feed Process Reward
Models (PRMs) rather than Outcome Reward Models (ORMs).

The distinction matters because of gradient conflict. Our pipeline quantified this on 20
wearable agent trajectories: a trajectory that fails at step 15 has 14 preceding steps
that may each be correct. Under ORM training, all 14 are penalized identically to a model
that failed from step 1. Under PRM training with step-level labels, each correct
intermediate step contributes positive signal. Research from ReasonRAG (NeurIPS 2025)
found process-supervised training to be 18× more data-efficient than outcome-supervised
RL for exactly this reason.

## Connecting to Kore.ai's Six Core Metrics

Step-level annotation does not exist in isolation — it needs to connect to the eval
metrics your organization already tracks. The pipeline integrates directly with Kore.ai's
six-metric framework: trajectory success rate, tool invocation accuracy, groundedness
score, privacy leak detection, orchestrator correctness, and latency SLA compliance.

The A/B experiment results show why annotation quality upstream determines metric quality
downstream. Comparing 50 raw trajectories against 50 curated trajectories (passed through
IAA thresholding, PRM labeling, and poisoning detection):
Metric                     Raw (n=50)   Curated (n=50)   Delta
Trajectory Success Rate       0.120          0.333       +177.8%
Tool Invocation Accuracy      0.360          1.000       +177.8%
Groundedness (RAGAS)          0.750          0.750           —
Privacy Leak Detection        0.000          0.000           —
Orchestrator Correctness      1.000          1.000           —
Latency SLA Compliance        1.000          1.000           —

The two training-relevant metrics — trajectory success rate and tool invocation accuracy —
lifted by 177.8% each after curation. The stable metrics (privacy, orchestrator
correctness, latency) were stable by design in synthetic data. In production deployments,
those three are where annotation-layer poisoning would manifest first.

## IAA as a CI Quality Gate

The operational question for an engineering manager deploying this at scale is: where does
annotation quality enforcement live in the pipeline? The answer is continuous integration.

After each annotation batch, compute IAA across all active annotators. If Cohen's κ falls
below your threshold (we use 0.75 as the gate), the batch is flagged before it enters the
reward model training queue. DVC tracks annotation versions so that when a model degrades
post-deployment, you can trace the regression back to a specific annotation batch and
annotator cohort — not just to a vague "data quality issue."

This is the eval flywheel Kore.ai's 52% are missing: annotate → compute IAA → gate on κ
threshold → label process rewards → train → evaluate → repeat. Each iteration is
auditable. Each gate is a governance checkpoint.

## Annotator Poisoning: The Silent Failure Mode

One failure mode that receives insufficient attention in enterprise AI annotation is
coordinated annotator bias. Anthropic's backdoor poisoning research demonstrated that only
250 malicious documents — 0.00016% of a training corpus — are sufficient to backdoor
models of any size. The same count-based dynamic applies to annotators.

Our poisoning detection module exposed a named blind spot in Median Absolute Deviation
(MAD) scoring: three colluding annotators with identical bias collapse to panel consensus
under MAD, producing a suspicion score of 0.0 — indistinguishable from your most reliable
annotator. Step-level IAA monitoring catches what session-level agreement misses, because
coordinated bias at the step level produces a characteristic disagreement pattern with the
non-colluding annotators that MAD scoring over the full session obscures.

## What This Enables

The framework described here — 3-layer annotation schema, PIA rubric scoring, CI-gated
IAA, DVC label provenance, PRM-ready step rewards — is designed to scale from a 5-person
annotation team to a 50-person data organization without losing auditability.

The full pipeline, including the annotation schema, PIA calculator, poisoning detector,
and A/B experiment harness, is open-source: [Repo](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline)

For enterprises already running Kore.ai's evaluation framework, the step-level annotation
layer slots in upstream of your existing metric computation — it enriches the training data
that determines the metrics, rather than replacing the metrics themselves.

The 89%/52% gap Kore.ai identified is closeable. The methodology is here.