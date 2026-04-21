## AgentTrace v1.0 — Trajectory-Level Evaluation for Multi-Agent AI

### What is AgentTrace?

AgentTrace is an open-source evaluation harness for multi-agent AI trajectories. It moves beyond binary task-success evaluation by decomposing each agent execution into five scored layers — intent parsing, planning quality, tool call precision, error recovery, and outcome — and computing per-layer scores that can feed directly into a process reward model (PRM) training pipeline. AgentTrace is designed for production agentic systems where a single pass/fail signal is insufficient for debugging, regression detection, or training data curation.

AgentTrace implements the Path-Invariant Agreement (PIA) methodology introduced in White Paper 2: "Beyond Task Success — Trajectory-Level Evaluation Framework for Multi-Agent Enterprise AI." PIA resolves the inter-rater reliability (IRR) paradox that emerges when evaluating non-deterministic agents: two valid trajectories reaching the same goal by different paths should not register as annotator disagreement. By relocating the unit of measurement from path-specific step sequences to rubric dimensions (planning quality, error recovery, goal alignment), PIA produces reliable annotator agreement in conditions where standard Fleiss' κ collapses to below-chance levels.

---

### Key Components Released in v1.0

| Module | File | Description |
|---|---|---|
| **TrajectoryScorer** | `src/eval/trajectory_scorer.py` | PIA rubric scoring across 4 dimensions: planning quality, error recovery, goal alignment, tool precision. Also computes 5-layer decomposition (intent 0.15, planning 0.25, tool_calls 0.25, recovery 0.15, outcome 0.20) with weight renormalization when recovery layer is absent. |
| **AgenticEval** | `src/eval/agentic_eval.py` | 6 Kore.ai-inspired enterprise metrics: trajectory success rate, tool invocation accuracy, groundedness score (RAGAS), privacy leak detection, orchestrator correctness, latency SLA compliance. Includes DeepEval LLM-as-judge ensemble and FACTSGroundingScorer stub. |
| **BenchmarkRunner** | `src/eval/benchmark_runner.py` | Multi-framework evaluation harness. Runs identical wearable AI tasks across LangGraph, CrewAI, AutoGen (AG2), and OpenAI Agents SDK. Each task × framework combination is run 3× to compute nondeterminism variance. Produces a leaderboard across 6 dimensions (goal rate, tokens, error rate, trajectory score, PIA score, tool precision). |
| **NondeterminismVariance** | `src/eval/trajectory_scorer.py` (`compute_nondeterminism_variance`) | Accepts ≥2 trajectory runs of the same task and returns per-layer standard deviation. Surfaces whether execution variance concentrates at the intent, planning, tool-call, recovery, or outcome layer — enabling targeted stability improvements. |
| **ABExperiment** | `src/eval/ab_experiment.py` | Curated-versus-raw A/B evaluation. Splits trajectories by weighted_total score, applies configurable terminal-step corruption to the raw group, scores both groups across all 6 Kore.ai metrics, and reports absolute and percentage deltas. |
| **HITLTrigger** | `src/eval/hitl_trigger.py` | Four-type Human-in-the-Loop escalation detector: CONFIDENCE_BELOW_THRESHOLD, SAFETY_ADJACENT_ACTION, NOVEL_TOOL_PATTERN, DOMAIN_EXPERTISE_REQUIRED. KNOWN_TOOLS registry enforces an approved tool surface; any call outside it fires a NOVEL_TOOL_PATTERN trigger. |
| **RoleAttributionScorer** | `src/eval/role_attribution.py` | Layer 2 multi-agent attribution metrics: authority compliance rate, average delegation quality, accountability coverage, orchestrator handoff score. Sets cascade_risk=True when a trajectory fails and no agent holds accountability_clear=True. |
| **CascadeError** | `src/eval/cascade_error.py` | Error propagation taxonomy classifying which layer failures propagate downstream versus self-contain. |
| **FACTSIntegration** | `src/eval/facts_integration.py` | DeepMind FACTS factuality integration stub targeting the Kaggle FACTS benchmark (WP3 target, Day 41). |

---

### Key Empirical Results (from WP2)

- **Standard IRR breaks for non-deterministic agents:** Fleiss' κ = **−0.065** (below-chance agreement) when scoring non-deterministic trajectory pairs by path-specific step comparison. Ten trajectory pairs across five wearable scenario types; five annotator personas.
- **PIA rubric-dimension scoring recovers reliability:** Fleiss' κ = **+0.743** (substantial agreement) when the same annotators score the same trajectories against three rubric dimensions (planning quality, error recovery, goal alignment) rather than step sequences. Δ = **+0.808** κ points.
- **Curation pipeline lifts tool invocation accuracy:** **+177.8%** (0.36 → 1.00) across 100 wearable trajectories (A/B experiment, seed=42, n=50 per group).
- **Curation pipeline lifts trajectory success rate:** **+177.8%** (0.12 → 0.33) in the same experiment.
- **Framework benchmark:** All four frameworks achieve 100% goal rate on wearable tasks — task-success rate produces a four-way tie. Trajectory decomposition surfaces differentiation: LangGraph leads on token efficiency (491 tokens, 2.1× fewer than AutoGen); CrewAI ties LangGraph on trajectory score (0.8686) at 1.6× the token cost.
- **Multi-agent lift:** Orchestrator + specialist routing yields a mean Δ of +0.071 over single-agent baseline on the 3/10 scenarios where specialist routing applies (privacy_sensitive, ambient_noise). No lift on health_alert, location_trigger, or calendar_reminder scenarios.

*All results derived from the synthetic wearable benchmark in dry-run mode. Live API annotation expected to yield Cohen's κ ≈ 0.78–0.85 post-calibration.*

---

### White Paper 2

[White Paper 2 — Beyond Task Success: A Trajectory-Level Evaluation Framework for Multi-Agent Enterprise AI](https://medium.com/@shailendrabade) *(Medium URL — to be updated after publish)*

Full draft: [`docs/white_papers/wp2_beyond_task_success_DRAFT.md`](docs/white_papers/wp2_beyond_task_success_DRAFT.md)

Companion notebook: [`notebooks/agentic_eval_flywheel.ipynb`](notebooks/agentic_eval_flywheel.ipynb) ([HTML](notebooks/agentic_eval_flywheel.html))

---

### Citation

```
Bade, S. (2026). AgentTrace v1.0: Trajectory-Level Evaluation for Multi-Agent AI.
GitHub. https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline
```

Or in BibTeX:

```bibtex
@software{bade2026agenttrace,
  author  = {Bade, Shailendra},
  title   = {{AgentTrace v1.0}: Trajectory-Level Evaluation for Multi-Agent {AI}},
  year    = {2026},
  url     = {https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline},
  version = {1.0.0}
}
```

---

### Roadmap (v1.1 and Beyond)

- **Federated eval support:** Run AgentTrace evaluation harnesses across distributed agent deployments without centralizing raw trajectory data — enabling privacy-preserving cross-organization benchmarking.
- **Live RAGAS with wearable knowledge base:** Replace the current RAGAS fallback (constant 0.75) with a live retrieval layer backed by medical device specifications, consent policy documents, and ambient sensor calibration data. This unlocks the groundedness lift expected from curation (WP3).
- **FACTS grounding integration (WP3):** Complete the `FACTSGroundingScorer` stub to run full DeepMind FACTS factuality evaluation across all four dimensions (parametric, search, context, multimodal) on wearable AI trajectory outputs. Target: Day 41 Kaggle submission.
