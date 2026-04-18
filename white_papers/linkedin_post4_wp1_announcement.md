# LinkedIn Post #4 — WP1 Announcement

🟡 DO NOT POST — PENDING REVIEW

---

## LinkedIn Post (1,222 characters — within 1,300 limit)

Standard inter-annotator agreement on agentic trajectories: κ = −0.065.

Same annotations, different measurement: κ = +0.743.

No new labels collected. No annotators replaced. Just measuring agreement on the right thing.

That's the headline result from White Paper 1 — published today after 28 days of building this pipeline in public.

Three things worth your time:

• Path-Invariant Agreement (PIA) — why Cohen's κ collapses for non-deterministic agents, and a rubric-dimension fix that recovers Δκ = +0.808 on identical annotation data

• Annotation-layer poisoning detection — the only methodology we know of that catches coordinated label injection before poisoned labels enter the reward model gradient

• Gradient conflict rate: 100% — every outcome-failed trajectory contained a majority of positively-rewarded intermediate steps. ORM silently misattributes all of it as failure signal.

Still building. Blind spots documented honestly.

→ Medium: [link]
→ GitHub: github.com/finaspirant/llm-wearable-agentic-eval-pipeline
→ HuggingFace: huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations

#AIAlignment #LLMEvaluation #DataAnnotation #MachineLearning #OpenSource

---

## Twitter/X Thread Starter (272 characters — within 280 limit)

🟡 DO NOT POST — PENDING REVIEW

Standard IAA on agentic trajectories: κ = −0.065 (poor).

Same annotations, rubric-dimension scoring: κ = +0.743 (substantial).

Δ = +0.808. No new labels.

WP1 on annotation methodology for process-supervised agent training — published today 🧵
