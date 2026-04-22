# CFP Abstract — BOTS Summit 2026 (Practitioner Track)

**Status:** 🟡 DRAFT — DO NOT SUBMIT without reviewing deadline at botssummit.com

**Title:** Why Your Agentic AI Eval is Lying to You — and How to Fix It

---

## Abstract (~150 words)

You shipped an agentic AI system. Your eval says 85% task success. Your users say it feels
unreliable. Both are right — and the gap is your evaluation methodology.

Standard inter-annotator agreement (IAA) metrics break for agents because two valid paths to
the same goal look like annotator disagreement. Outcome-only reward signals penalize 14 correct
steps because step 15 failed. And your annotation pipeline has no way to detect when annotators
are subtly colluding to bias training data.

This talk walks through three fixes — Path-Invariant Agreement (PIA), process reward scoring
with partial credit, and annotator poisoning detection — each validated on an open-source
wearable AI evaluation pipeline. Results: IAA lifted from κ = −0.07 to κ = +0.74. Tool
invocation accuracy improved 177% after curation. Code is on GitHub and you can fork it today.

---

## Three Takeaways

1. **Why IAA breaks for agents** — path diversity is misclassified as annotator disagreement;
   standard κ scores are meaningless on non-deterministic trajectories
2. **How PIA fixes it** — rate rubric dimensions (planning quality, error recovery, goal
   alignment) instead of path-specific actions; κ lifts from −0.07 to +0.74
3. **Step-level monitoring as governance** — process reward scores flag gradient conflicts
   before they corrupt your reward model; same signal catches annotator poisoning

---

## Speaker Bio Placeholder

[Insert 2-sentence bio here before submitting]

---

## Submission Checklist

- [ ] Verify BOTS Summit 2026 deadline at botssummit.com
- [ ] Repo is public before submission
- [ ] Add demo link (Streamlit app: `demo/app.py`) to abstract
- [ ] Add ArXiv preprint URL once live
