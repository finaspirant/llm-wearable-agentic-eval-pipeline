"""Step-level Process Reward Model annotator with partial credit.

Implements simplified version of AgentPRM (2502.10325) MC rollout
annotation for step-level rewards. Addresses the gradient conflict
problem identified in ReasonRAG (NeurIPS 2025, 2505.14069):
outcome-only reward penalizes all correct intermediate steps when
the final step fails.

Key insight from ReasonRAG: PRM achieves 18× data efficiency over
ORM using MCTS exploration + SPRE reward assignment.
"""

import logging

logger = logging.getLogger(__name__)

# TODO(Day 12): Implement PRMAnnotator class
#   - annotate_step(step, context) → StepReward (correct/neutral/incorrect + score)
#   - partial_credit(trajectory, failed_step_idx) → list[StepReward]
#   - gradient_conflict_detector(trajectory) → bool
#   - Output format compatible with DPO training pipelines
