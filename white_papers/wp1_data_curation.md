# Beyond Preference Pairs: A Process-Supervised Annotation Framework for Agentic AI Training Data

**Subtitle:** A Process-Supervised Approach to Training Data Curation for Agentic Systems

**Target audience:** Anthropic, Cohere, AI21 Labs
**Publish target:** Day 28

---

## Abstract

The paradigm shift from outcome-supervised to process-supervised reward modeling requires a corresponding shift in annotation methodology. Standard RLHF preference pairs — designed for single-turn dialogue — embed a silent assumption: that a single label can characterize the quality of an entire exchange. This assumption breaks catastrophically for multi-step agentic tasks, where correct intermediate reasoning and failed terminal outcomes routinely co-occur. We present a three-layer annotation framework operating upstream of the reward model gradient, at the annotation layer itself. Across 20 synthetic wearable-AI trajectories, we find that 100% of outcome-failed trajectories contained a majority of positively-rewarded intermediate steps — the gradient conflict rate that outcome-only reward modeling silently misattributes as failure signal. Our framework introduces Path-Invariant Agreement (PIA), a novel inter-rater reliability measure that decouples annotator agreement from path-specific step choices, raising measured κ from −0.065 (poor) to +0.743 (substantial) on identical annotation data — a Δ of +0.808.

---

## Status

- [x] §1: The paradigm shift (ORM → PRM)
- [x] §2: The gradient conflict problem
- [x] §3: Path-Invariant Agreement (PIA)
- [x] §4: Annotation-layer poisoning detection
- [x] §5: HH-RLHF empirical analysis
- [x] §6: Framework benchmark summary
- [x] §7: Conclusion + publication roadmap

---

## §1 — The Paradigm Shift: From RLHF to Agentic Process Supervision

Reinforcement Learning from Human Feedback was developed to align single-turn language models with human preference. Its foundational annotation unit — the preference pair — asks a human rater to select between two completions of a prompt. This mechanism was fit for purpose when the entire interaction was contained within a single exchange. Multi-step agentic tasks break this assumption in three compounding ways.

First, the temporal structure of agency collapses under pairwise comparison. An agent navigating a 15-step task accumulates planning decisions, tool invocations, and error recovery actions across a trajectory that cannot be captured in a single preference label without discarding nearly all of the intermediate signal. Second, the causal structure of failure is non-local. A terminal step failure may trace to an annotation-layer error 12 steps upstream, a tool routing error at step 3, or a sensor ambiguity at the first observation — yet outcome-based reward modeling attributes the failure uniformly backward across every preceding step, training the model to suppress reasoning patterns that were in fact correct. Third, and most consequentially for annotation methodology, inter-rater reliability breaks when raters compare trajectories rather than outcomes: two annotators who agree that an agent achieved its goal while disagreeing on whether it took the direct or exploratory path will register as disagreeing under standard path-comparison IRR metrics, even though their underlying judgment about goal quality is identical.

These three failure modes are not theoretical. ReasonRAG (NeurIPS 2025, arXiv 2505.14069) demonstrates that process-supervised reward modeling achieves equivalent performance with 18× fewer training queries than outcome-supervised RL, precisely because process supervision preserves the intermediate step signal that outcome-only labels discard. The Anthropic 250-document backdoor finding (October 2025) establishes that model-level attacks are not the primary threat vector: only 250 documents — 0.00016% of training data — suffice to backdoor models of any scale, which means poisoning enters at the annotation layer, not the model layer. This reframes data curation as a security problem, not merely a quality problem. Cohere Command A (arXiv 2504.00698) provides the clearest articulation of the methodology gap: 800 prompts annotated by 65 raters on a 5-point scale with shuffled presentation — and no inter-annotator agreement statistics reported. No κ. No α. No calibration protocol. This is not an oversight; it reflects the absence of an annotation framework designed for the agentic setting.

We propose the first annotation framework operating upstream of the reward model gradient, at the annotation layer itself. Rather than measuring the quality of model outputs after reward modeling, our framework instruments the annotation process — calibrating rater behavior, detecting coordinated bias injection, and assigning step-level partial credit — so that the training signal entering the reward model is structurally sound before any gradient is computed.

## §2 — The Gradient Conflict Problem: Why Outcome Labels Corrupt Good Steps

The gradient conflict problem is the central failure mode of outcome-only reward modeling applied to multi-step agents. Consider a trajectory of $T$ steps, where each step $s_t$ represents the agent's action given its current state. Under outcome reward modeling (ORM), the reward for step $s_t$ is defined as:

$$r_{\text{ORM}}(s_t) = \begin{cases} +1 & \text{if } t = T \text{ and outcome\_success} = \texttt{True} \\ -1 & \text{if } t = T \text{ and outcome\_success} = \texttt{False} \\ 0 & \text{if } t < T \end{cases}$$

This formulation is exact for single-step tasks and a reasonable approximation for short sequences. For trajectories of length 5 or greater, it has a structural defect: terminal failure sets $r_{\text{ORM}}(s_T) = -1$ and zeros all $r_{\text{ORM}}(s_t)$ for $t < T$. The gradient signal backpropagated through the policy is uniformly negative across a trajectory in which every step except the last may have been correct. The model learns to suppress the very reasoning patterns it should amplify.

Process reward modeling (PRM) assigns an independent reward to each step:

$$r_{\text{PRM}}(s_t) \in [-1, +1], \quad r_{\text{partial}}(s_t) \in [0, 1]$$

where $r_{\text{partial}}$ provides a continuous partial credit signal for steps that are directionally correct but not optimal. AgentPRM (arXiv 2502.10325) implements this via Monte Carlo rollout annotation, sampling multiple completions from each intermediate state to estimate the probability of eventual success. Our pipeline implements a simplified version that does not require live rollout sampling, using a heuristic cascade over tool-match signals, step quality ratings, and positional priors to assign $r_{\text{PRM}}$ and $r_{\text{partial}}$ from existing annotation dimensions.

Our empirical finding from 20 synthetic wearable-AI trajectories is unambiguous: the gradient conflict rate in our dataset is **100%**. Every single outcome-failed trajectory contained a majority of non-terminal steps with positive process reward scores ($r_{\text{PRM}} > 0$). The mean process reward score across all non-terminal steps is **0.175**, confirming that intermediate steps carry positive signal that ORM discards. The statistic is synthetic — the trajectories were generated with controlled step quality distributions — but the mechanism it illustrates is real and well-documented in the ReasonRAG results.

This finding motivates the three-layer annotation schema at the core of our framework. Layer 1 captures session-level outcome: whether the overall goal was achieved and whether privacy constraints were respected throughout the session. Layer 2 assigns responsibility in multi-agent settings, attributing errors to the specific role (orchestrator, specialist, action executor) where the failure originated. Layer 3 is the PRM feed: step-level $r_{\text{PRM}}$ and $r_{\text{partial}}$ scores with a minimum 20-character rationale per step, providing the annotator's causal reasoning as an auditable artifact. The three layers are not redundant; each operates at a different temporal grain, and the combination allows a downstream reward model to distinguish terminal failure from systematic failure — a distinction ORM cannot make.

---

## §3 — Path-Invariant Agreement: Fixing IAA for Non-Deterministic Agents

Standard inter-annotator agreement (IAA) metrics assume that raters are evaluating the same observable artifact. For single-turn dialogue, this assumption holds: both annotators read the same response and assign a quality label. For multi-step agentic trajectories, it breaks in a precise and damaging way.

Two annotators can reach identical conclusions about the quality of an agent's planning — "this agent understood the goal and selected appropriate tools" — while disagreeing entirely about whether the agent should have taken a direct 3-step path or an exploratory 5-step path with intermediate verification. Under Cohen's κ, these annotators register as disagreeing. Under Fleiss' κ computed on step-level labels, the disagreement compounds across every additional step in the longer trajectory. The measured agreement collapses toward zero — not because annotators disagree on quality, but because standard IAA metrics conflate path choices with quality judgments.

This is what we call the **path-comparison fallacy**: the practice of computing IAA on path-specific step labels for agents whose optimal path is intentionally non-deterministic.

**Pilot study design.** We generated 20 matched trajectory pairs across five wearable-AI scenario types (health_alert, privacy_sensitive, location_trigger, ambient_noise, calendar_reminder). Each pair contains one direct agent (3-step path: sense → plan → act) and one exploratory agent (4–5 steps, including intermediate verification steps labeled as "detour" steps). Both agents in every pair achieve the same terminal goal with `overall_goal_achieved = True`. Five simulated annotator personas — each with systematic scoring biases across four quality dimensions — labeled all trajectories.

**Mode A (Standard Path-Comparison IAA):** Fleiss' κ computed on 75 step-level labels across all 20 agents. Result: **κ = −0.065** (poor). Under the Landis & Koch scale, this is the conclusion that annotators are effectively guessing — a finding that would cause any responsible research team to discard the annotations.

**Mode B (Path-Invariant Agreement):** Fleiss' κ computed on three rubric dimensions rather than step-specific choices:
- *Planning quality*: did the agent select an appropriate action sequence given the goal and sensor context?
- *Error recovery*: when the agent encountered ambiguity or failure, did it recover appropriately?
- *Goal alignment*: did the agent's terminal action correctly address the user's stated intent?

Result: **κ = +0.743** (substantial). Per-dimension: planning_quality = 0.705, error_recovery = 0.826, goal_alignment = 0.697.

**The delta is Δ = +0.808 on identical annotation data.** No new labels were collected. No annotator calibration was applied. The lift is entirely attributable to measuring agreement on the right unit: rubric dimensions invariant to path, rather than step choices that vary by design.

PIA operationalizes a single principle: for non-deterministic agents, IAA should measure whether annotators agree on *what the agent accomplished* along each quality dimension — not *how it chose to accomplish it*. The three dimensions were selected because they are (a) directly measurable from trajectory data without reference to a ground-truth path, (b) each independently predictive of downstream training signal quality, and (c) sufficiently orthogonal that agreement on one does not mechanically imply agreement on another.

The pilot dataset — 500 annotated records across 50 trajectories and five annotator personas — is publicly available on HuggingFace (finaspirant/wearable-agent-trajectory-annotations). We are not aware of any prior work that explicitly names and addresses the path-comparison IAA fallacy for agentic systems. The methodology, scoring rubric, and replication code are released as open-source artifacts.

---

## §4 — Annotation-Layer Poisoning Detection

The Anthropic 250-document backdoor finding (October 2025) established a counterintuitive result: model-level poisoning attacks require far less data than previously assumed. Only 250 documents — 0.00016% of total training data — suffice to implant a backdoor in models of any scale, with the attack being count-based rather than proportion-based. The implication for annotation pipelines is direct. If the effective attack surface is this small, then the economic cost of a targeted annotation-layer attack is negligible, and the first line of defense must operate at the annotation layer — before poisoned labels enter the reward model gradient.

**Threat model.** We focus on coordinated annotator bias injection: a small coalition of malicious raters who systematically inflate or suppress specific annotation dimensions. In the wearable-AI context, a privacy-adversarial attacker might suppress `privacy_compliance` scores while inflating `step_quality` scores, creating training examples that reward privacy-violating agent behaviors while appearing to maintain overall annotation quality in aggregate. This attack is invisible to outcome-level evaluation and to naïve score averaging across annotators.

**Detection layer 1: MAD outlier scoring.** Each annotator receives a suspicion score in [0, 1] based on their median absolute deviation from the per-dimension consensus distribution, normalized across the annotator pool. A threshold of 0.6 flags annotators for human review. This layer is effective against independent bad actors whose labels deviate visibly from the pool median.

**Documented blind spot.** Three or more colluding annotators who submit identical biased labels collapse to the consensus of the malicious group. Each poisoner receives a suspicion score of 0.0 — the MAD detector is entirely blind to coordinated collusion at this scale. In our pilot evaluation, a coalition of 3 injected poisoners achieved F1 = 0.0 at every detection threshold. We report this failure mode explicitly because honest documentation of detector limitations is a prerequisite for trustworthy safety methodology.

**Detection layer 2: Confident Learning.** The second layer addresses the collusion blind spot via cleanlab's Confident Learning implementation (`find_label_issues`, `get_label_quality_scores`). Rather than measuring deviation from annotator consensus, Confident Learning measures deviation from a model-estimated label distribution, using Laplace-smoothed vote distributions as predicted probabilities and majority vote as the given label. This layer provides partial recovery of detection signal in the colluding-annotator scenario — the exact regime where MAD provides no signal at all.

The combined two-layer architecture is complementary by design, analogous to input sanitization (MAD, effective against independent actors) and anomaly detection (Confident Learning, effective against coordinated actors) in security engineering.

The core architectural claim: this is the only published methodology we are aware of that explicitly targets poisoning detection at the annotation layer, before contaminated labels enter the reward model gradient. Existing detection work focuses on model-level indicators — perplexity differentials, activation pattern analysis — which are useful post-hoc but cannot prevent gradient contamination from occurring. Annotation-layer detection is a necessary complement: if 250 documents suffice to backdoor any model, the expected loss from a single successful poisoning incident vastly exceeds the engineering cost of annotation-layer instrumentation.

Implementation: `src/annotation/poisoning_detector.py`. Empirical results: `data/processed/day17_detection_results.json`.

---

## §5 — HH-RLHF Through a Process-Supervision Lens

The Anthropic HH-RLHF dataset is the most widely studied preference annotation corpus in alignment research, containing approximately 170,000 conversation pairs labeled for helpfulness and harmlessness. It represents the state of the art in outcome-level preference annotation — carefully collected, thoughtfully structured, and extensively benchmarked against downstream model quality. It is, by design, a single-label annotation dataset: raters select the preferred response with no step-level decomposition, no per-dimension rubric, and no reported inter-annotator agreement statistics.

We analyzed a 200-pair sample through our pipeline's annotation lens, applying three IAA metrics across three dimensions — helpfulness, harmlessness, and coherence — using five annotator personas with distinct scoring biases.

| Dimension | Fleiss' κ | Krippendorff's α | Interpretation |
|---|---|---|---|
| helpfulness | −0.121 | −0.245 | poor |
| harmlessness | −0.093 | −0.253 | poor |
| coherence | +0.001 | −0.002 | slight |
| **Overall** | **−0.071** | — | **poor** |

These results do not reflect poorly on HH-RLHF annotators. They reflect the inherent difficulty of achieving multi-dimensional IAA on preference-pair data — a difficulty that compounds in agentic settings where path variance adds a second source of apparent disagreement on top of genuine quality disagreement. The Cohere Command A paper (arXiv 2504.00698) annotated 800 prompts with 65 raters on a 5-point scale with shuffled presentation order and reported no κ or α statistics. Our results provide one plausible explanation for that omission: measured agreement in multi-dimensional preference annotation is low enough that accurate reporting would invite questions the methodology is not equipped to answer. Naming this gap is not a criticism of Cohere's work; it is an argument for why an annotation framework specifically designed for the agentic setting is needed.

**What our framework adds.** Applied to HH-RLHF-style data, the three-layer schema contributes three things that preference pairs do not provide. First, step-level PRM labels decompose each trajectory into per-step rewards, preserving gradient signal that outcome labels discard. For HH-RLHF's multi-turn dialogues, this means each assistant turn receives an independent `r_PRM` and `r_partial` score rather than having the terminal preference label propagated backward over all turns. Second, PIA calibration separates disagreement arising from path variance (annotators preferring different conversational strategies) from disagreement arising from genuine quality differences. Third, gradient conflict detection identifies the specific dialogues where outcome-level labeling actively corrupts training signal — the conversations where every intermediate turn was correct but the final response failed, a pattern that ORM treats as uniformly negative signal.

The HH-RLHF dataset was designed for single-turn and short-horizon dialogue alignment and has driven substantial progress in that domain. The annotation methodology gaps we identify become acute when the same paradigm is extended to multi-step agentic tasks — a context the original dataset was not designed for, and one that requires the methodology extensions this paper describes.

---

## §6 — Framework Benchmark: Annotation Amenability Across Four Architectures

Our Day 22 benchmark evaluated four agentic frameworks across 10 tasks (five enterprise, five wearable-AI) with three runs per task-framework combination, yielding 120 trajectory observations. All four frameworks achieved an average trajectory score of 0.8235 on our 5-layer evaluation rubric (intent, planning, tool execution, recovery, outcome), confirming that framework choice does not significantly affect outcome quality on standardized tasks. What varies substantially is the framework's **annotation amenability**: the degree to which its native execution model exposes intermediate reasoning as annotatable artifacts.

**LangGraph** is the most annotation-friendly architecture. Node transitions with explicit state dictionaries at each step provide clean, observable intermediate state. The sense → plan → act separation maps directly to our Layer 3 step-level PRM schema, and token overhead is lowest (519 tokens per task on average). Every annotatable step is a named graph node with a typed input/output contract — exactly the structure our annotation schema expects.

**OpenAI Agents SDK** achieves the lowest latency in live API runs (10.3 seconds versus 14.8 for CrewAI) and produces clean tool-call and handoff event logs that are straightforward to segment into annotatable steps. The primary limitation for wearable-AI deployment is the absence of a native human-in-the-loop escalation mechanism: HITL must be instrumented externally, adding architectural complexity that the annotation pipeline must account for.

**CrewAI** exhibits a verification spiral in our benchmark: the Specialist agent delegates to the Escalation Manager on a higher fraction of tasks than task complexity warrants, inflating token count (~730) and step count without improving outcome quality. Our cascade error taxonomy classifies this as a depth-2 cascade — a pattern invisible to outcome-only evaluation but clearly identifiable at the trajectory level. Annotation of CrewAI trajectories requires a delegation-depth normalization step to prevent cascade patterns from artificially deflating per-step quality scores.

**AutoGen** produces the richest inter-agent dialogue but the most annotation-resistant trajectories. Speaker/message turns do not cleanly map to annotatable steps, error attribution across UserProxy/AssistantAgent exchanges is ambiguous, and token overhead is highest (~950). Annotating AutoGen trajectories at the step level requires a custom turn-segmentation layer that identifies atomic action units within the multi-turn exchange — a non-trivial preprocessing step that the other three frameworks do not require.

The implication for research teams building annotation pipelines for multi-agent systems: framework selection has downstream consequences for annotation cost and IAA quality that are not visible in outcome-only benchmarks. A framework that achieves identical task success rates as its competitors may require 2–3× the annotation infrastructure investment if its execution model does not expose intermediate state as observable artifacts. Full benchmark results are available in `data/processed/framework_leaderboard.json`. WP2 (Day 30) extends this to full trajectory observability metrics.

---

## §7 — Conclusion and Publication Roadmap

This paper presents three contributions to the methodology of agentic AI training data curation.

**Path-Invariant Agreement (PIA)** addresses the most direct failure of applying standard RLHF methodology to multi-step agents: IAA metrics that conflate path choices with quality judgments, producing near-zero agreement scores on annotation data that is in fact highly consistent. PIA decouples these by measuring agreement on rubric dimensions — planning quality, error recovery, goal alignment — that are invariant to the specific path an agent takes to reach its goal. The pilot demonstration yields Δκ = +0.808 on identical annotation data, the largest reported IAA lift we are aware of that does not require collecting new labels. We release the methodology, rubric, pilot dataset, and replication code as open-source artifacts and are not aware of prior work that explicitly names and addresses the path-comparison IAA fallacy.

**Annotation-layer poisoning detection** addresses a threat vector that model-layer detection cannot reach. Motivated by the Anthropic 250-document backdoor finding, we implement and evaluate a two-layer detection architecture (MAD outlier scoring + Confident Learning) that operates before poisoned labels enter the reward model gradient. We document honestly that the MAD layer is blind to coordinated collusion at coalition sizes of three or more, and that the Confident Learning layer provides partial but incomplete recovery. The methodology is released as `src/annotation/poisoning_detector.py`, with empirical results in `data/processed/day17_detection_results.json`. To our knowledge, this is the first published detection methodology explicitly targeting the annotation layer.

**An open-source annotation pipeline** — three-layer annotation schema (`agenteval-schema-v1.json`), annotator calibration protocol, PRM annotator with gradient conflict detection, poisoning detector, and HuggingFace benchmark dataset — released as a unified, typed, and tested codebase. The pipeline passes mypy strict, ruff lint, and a 503-test suite covering all annotation layers, IAA calculators, and evaluation harnesses.

**Publication roadmap.** This is the first of three white papers in this series. WP2, *"Beyond Task Success: Trajectory-Level Evaluation for Agentic AI"* (target: Day 30), extends the framework to full trajectory observability, cascade error taxonomy, and framework benchmark comparisons across real API calls. WP3, *"Evaluating Always-On AI: An Annotation Framework for Ambient Wearable Systems"* (target: Day 42), addresses consent decay, passive data capture, and privacy-preserving evaluation — the open problems in ambient AI that HealthBench's clinical framing does not cover and that no existing benchmark addresses.

**For Alignment Forum readers.** The gradient conflict problem described in §2 connects directly to the robustness of Constitutional AI training pipelines (Bai et al., 2022): if a model's RLHF phase trains on preference pairs containing corrupted step-level signal, constitutional principles applied at inference time cannot repair the underlying training data damage — the gradient has already incorporated the poisoned signal. The annotation methodology contributions in this paper operate upstream of that gradient, at the layer where the signal is actually constructed. Researchers working on the data foundations of alignment — on what reward models are trained to value, and how reliably that value signal is measured — are the intended audience for these methodological extensions.

**Code and data.** Full pipeline: https://github.com/finaspirant/llm-wearable-agentic-eval-pipeline. Annotated benchmark dataset (500 records, 50 trajectories, 5 annotator personas, 2 calibration phases): https://huggingface.co/datasets/finaspirant/wearable-agent-trajectory-annotations.
