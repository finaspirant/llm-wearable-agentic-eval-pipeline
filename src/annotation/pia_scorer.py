"""Path-Invariant Agreement (PIA) scorer for non-deterministic agents.

Original methodological contribution. Standard IRR assumes identical
stimuli — invalid for agents that produce legitimately different but
equally correct trajectories. PIA measures agreement on rubric
dimensions (planning quality, error recovery, goal alignment) rather
than path-specific choices.

No existing paper addresses this problem by name. This is a publishable
contribution targeting WP2.
"""
import logging

logger = logging.getLogger(__name__)

# TODO(Day 15): Implement PIAScorer class
#   - define rubric dimensions (planning, recovery, goal, tool_use)
#   - score_trajectory(trajectory, rubric) → DimensionScores
#   - compute_pia(scores_rater1, scores_rater2) → float
#   - Compare PIA vs standard κ on same annotation set
