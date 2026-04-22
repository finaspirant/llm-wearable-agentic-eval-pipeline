# CFP Abstract — NeurIPS 2026 / COLM 2026

**Status:** 🟡 DRAFT — DO NOT SUBMIT without reviewing deadlines and ArXiv preprint link

**Title:** Path-Invariant Agreement: A Step-Level Annotation Methodology for Non-Deterministic Agentic AI Systems

---

## Abstract (~280 words)

Inter-annotator agreement (IAA) metrics are foundational to annotation quality assurance,
yet standard measures such as Cohen's κ and Krippendorff's α assume that two annotators
rating the same item will observe the same item. This assumption breaks for non-deterministic
agentic AI systems, where multiple valid execution trajectories can achieve the same goal via
different action sequences. When annotators observe divergent but equally correct paths, path
diversity is misclassified as annotator disagreement, artificially depressing IAA scores and
corrupting the training signal fed to reward models.

We introduce **Path-Invariant Agreement (PIA)**, a step-level annotation methodology that
decouples IAA measurement from path identity. Rather than requiring annotators to agree on
which actions were taken, PIA asks annotators to rate outcome-invariant rubric dimensions —
planning quality, error recovery, and goal alignment — that remain stable across valid path
variants. This reframing converts a path-comparison problem into a dimension-scoring problem,
recovering meaningful agreement signal even when no two agents follow the same trajectory.

We validate PIA on a pilot study using a wearable/ambient AI evaluation pipeline: 10 trajectory
pairs (20 agents, 5 scenario types), each pair sharing a terminal goal but differing in path
length and action sequence. Standard step-comparison IAA yields κ = −0.065 (poor). PIA rubric
scoring yields κ = +0.743 (substantial), a lift of Δ = +0.808. Per-dimension results:
planning_quality κ = 0.705, error_recovery κ = 0.826, goal_alignment κ = 0.697.

We release the full annotation pipeline, trajectory dataset, and PIA scorer as open-source
(Python, MIT license) to support reproducibility and adoption. We discuss implications for
process reward model (PRM) training data quality, annotator poisoning detection, and the
gradient conflict problem in outcome-supervised reinforcement learning from human feedback.

**Keywords:** inter-annotator agreement, agentic AI evaluation, process reward models,
annotation methodology, wearable AI, RLHF

---

## Supporting References

- ReasonRAG (NeurIPS 2025, arXiv 2505.14069) — PRM vs ORM, 18× data efficiency
- AgentPRM (arXiv 2502.10325) — MC rollout annotation for step-level rewards
- Cohere Command A (arXiv 2504.00698) — blind annotation methodology, no κ/α reported
- DeepMind FACTS (Dec 2025) — factuality benchmark, no model > 70%
- Anthropic 250-doc backdoor (Oct 2025) — count-based poisoning attack

---

## Submission Checklist

- [ ] Verify NeurIPS 2026 deadline at neurips.cc
- [ ] Verify COLM 2026 deadline at colmweb.org
- [ ] Repo is public before submission
- [ ] Add ArXiv preprint URL to abstract (PENDING — endorsement still outstanding)
- [ ] Confirm pilot study data committed and versioned
- [ ] WP1 Medium post live as supporting reference
