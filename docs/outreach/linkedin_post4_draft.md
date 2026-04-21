---
day: 33
status: DRAFT - DO NOT POST
target: AI21 Labs + Kore.ai
type: linkedin_post
created: 2026-04-21
closing_line_choice: TBD (Option A or Option B — decide before posting)
---

# LinkedIn Post #4

---

89% of enterprise AI teams have agent observability.

Only 52% have real evaluation.

That gap — from Kore.ai's own research — is not a tooling problem. It's an annotation
architecture problem. And it silently corrupts your training data every sprint.

AI21 put it plainly in "Enterprise AI after the hype curve": benchmarks don't map to
production reality. They're right. Here's why, and what the eval flywheel actually looks
like as an EM.

---

The annotation-to-eval flywheel:

1. Annotate agent trajectories at step level (not just session outcome)
2. Compute IAA across annotators — gate on k >= 0.75 before any batch enters training
3. Label process rewards per step (PRM, not ORM) — so failed trajectories with correct
   intermediate steps don't penalize good reasoning
4. Evaluate with your full metric harness
5. Repeat — with DVC tracking every annotation version so regressions are traceable

This is building in public. Here's what the pipeline shows so far:

-> IAA: 0.55 -> 0.82 after annotator calibration
-> PIA k: 0.28 -> 0.71 (standard IRR breaks for non-deterministic agents —
   rubric-dimension scoring fixes it)
-> Curation impact on downstream metrics:

Metric                   Raw     Curated    Delta
Trajectory Success       0.12    0.33      +177.8%
Tool Invocation Acc.     0.36    1.00      +177.8%

Same agent. Same tasks. Different annotation quality upstream.

As an EM, this is where I'd focus at a frontier lab: not on the model, not on the eval
harness — on the annotation layer that determines what both of them see.

[Repo](https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline) — open source, all experiments reproducible.

Day 33 of 45.

---

<!-- CHOOSE ONE CLOSING LINE BEFORE POSTING — delete the other -->

<!-- Option A (analytical): -->
The annotation layer is the only place in the ML pipeline where an EM's process
discipline directly determines model quality. That's where I'd spend the first 90 days.

<!-- Option B (provocative): -->
Most orgs are running evals that measure label noise, not model capability. The fix isn't
a better benchmark. It's governance upstream.

---

<!-- POSTING CHECKLIST -->
<!-- [ ] Replace [REPO_URL] with actual repo link -->
<!-- [ ] Choose and delete one closing line option -->
<!-- [ ] Tag @AI21Labs @Kore_ai in the post body -->
<!-- [ ] Comment separately on AI21's most recent LinkedIn post with a specific insight + repo link -->