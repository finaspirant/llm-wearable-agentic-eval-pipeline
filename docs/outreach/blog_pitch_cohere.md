---
type: blog_pitch
target: Cohere Labs editorial / engineering blog
status: DRAFT — review before sending
date: 2026-04-21
---

**Subject:** Guest post pitch — extending Command A's blind annotation methodology into agentic RAG pipelines

---

Hi [Name],

Command A's annotation methodology is one of the most rigorous blind evaluation designs published by any lab: 800 prompts, 65 annotators, 5-point scale, shuffled presentation order to eliminate anchoring effects. What it doesn't report — by design, because it's an LLM capability paper — is inter-annotator agreement statistics. No κ, no α, no IAA breakdown by prompt category. For a single-turn win-rate study that's a reasonable omission. For enterprise RAG pipelines where the "document" is a live retrieval context and the "response" is a multi-step agentic trajectory, that gap is the methodology problem that practitioners hit first.

**What the post covers**

The post extends Command A's blind annotation design into three-layer agentic annotation and shows what happens to IAA when you move from win-rate to trajectory-level grounding quality. Three concrete sections:

1. **The 3-layer annotation schema.** Session-level outcome (overall goal achieved, compliance), role-level attribution (multi-agent handoff quality, authority accountability), and step-level process rewards (PRM score −1.0 to +1.0, partial credit, annotator rationale). The schema is already open-sourced as `agenteval-schema-v1.json`. The point for Cohere's readers: win-rate annotation collapses the step-level signal that reranker and retrieval quality eval actually requires.

2. **IAA lift: 0.55 → 0.82 (Cohen's κ) after calibration.** Pre-calibration κ across 5 annotator personas on agentic trajectories was −0.035 — worse than chance, for the same structural reason Command A avoided reporting it: two valid retrieval paths to the same answer look like annotator disagreement under path-comparison IRR. A Path-Invariant Agreement (PIA) rubric that scores on *outcome dimensions* rather than *path choices* lifted Krippendorff's α from −0.113 to 0.82. The post shows the rubric design and why it's directly applicable to RAG reranker annotation.

3. **The RAGAS grounding fallback problem.** RAGAS groundedness requires a live retrieval context at annotation time. In practice, most annotation pipelines run post-hoc against a static KB snapshot — the retrieval context the model actually used is gone. The post proposes a differential privacy variant (PPG: Privacy-Preserving Grounding) that scores faithfulness against a DP-summarized context rather than the raw retrieved document, keeping annotation pipelines compliant when the KB contains sensitive enterprise data. Expected degradation at ε=1.0: 8–12% relative to raw-context RAGAS, with a minimum acceptable threshold of PPG ≥ 0.65.

**Why this fits Cohere Labs**

Cohere's engineering readers are building enterprise RAG on proprietary corpora — legal, finance, compliance, HR. The annotation methodology question is one they hit the moment they move beyond benchmark eval into production annotation for fine-tuning or RLHF on domain-specific retrievals. Command A's blind annotation design is the right starting point; the post shows the three extensions needed to make it work at the trajectory and grounding layer. Two white papers providing the full empirical backing (WP1: data curation and poisoning detection; WP2: trajectory-level evaluation and framework benchmarks) are already published — this post is the RAG-specific angle that connects directly to Cohere's product surface.

**What I'm offering**

A ~1,500-word post with two figures:
- Figure 1: IAA before/after calibration (pre κ = −0.035, post κ = 0.82), broken out by annotation dimension
- Figure 2: 3-layer annotation schema diagram — session / role / step with field-level detail

Open dataset reference: 500 annotated agentic trajectories (50 trajectories × 5 annotator personas × 2 calibration phases) published at [huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations](https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations). Full pipeline at [github.com/finaspirant/llm-wearable-agentic-eval-pipeline](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline).

I can weight the post more heavily toward the RAGAS/grounding angle or the IAA calibration angle depending on what's more relevant to your current editorial calendar. Does Cohere Labs accept external contributions, and if so, what's your submission process?

Thanks for considering it.

Shail Bade
shail.finaspirant@gmail.com
https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline
