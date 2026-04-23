# Alignment Forum Hook — WP3 Ambient AI Eval

**Status:** DRAFT — DO NOT POST until WP3 Medium URL is live
**Target:** LessWrong / Alignment Forum
**Word count:** ~200 words

---

No eval framework exists for always-on ambient AI agents — I built the spec.

HealthBench is the best clinical AI eval in existence (physician-physician agreement 55–75%). But ambient/wearable AI breaks all three of its assumptions: no discrete prompt/response boundary, passive capture creates intent ambiguity, and context drifts continuously.

I propose three new FACTS dimensions that no existing benchmark covers:

- **Context-drift grounding** (is the response grounded in the CURRENT sensor state?)
- **Passive-capture interpretation** (did the model correctly infer intent from ambient data?)
- **Consent-conditioned factuality** (is the response restricted to data the user authorised at this moment?)

Full framework, benchmark spec with formula-level metrics, and governance/regulatory considerations in WP3: [WP3 MEDIUM URL — paste after publish]

Repo: github.com/finaspirant/llm-wearable-agentic-eval-pipeline

---

*→ Replace [WP3 MEDIUM URL] before posting. Cross-post to LessWrong and Alignment Forum simultaneously.*
