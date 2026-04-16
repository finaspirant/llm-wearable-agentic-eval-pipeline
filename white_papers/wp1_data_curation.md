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
- [x] §2: Why existing datasets fail step-level quality checks
- [ ] §3: Gradient conflict reframing
- [ ] §4: IAA calibration methodology
- [ ] §5: Poisoning detection at the annotation layer
- [ ] §6: Empirical results (HH-RLHF analysis)
- [ ] §7: Open-source toolkit

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

*§3–§7 pending live-API annotation run and HH-RLHF analysis.*
