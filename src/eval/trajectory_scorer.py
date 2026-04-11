"""Trajectory-level scorer with 5-layer decomposition.

Decomposes agent trajectories into 5 evaluation layers:
1. Intent parsing — did the agent correctly understand the request?
2. Planning quality — was the plan reasonable and efficient?
3. Tool call precision — were tools called correctly with right args?
4. Recovery behavior — how did the agent handle errors/failures?
5. Outcome — did the agent achieve the goal?

Each layer can be scored independently, enabling fine-grained
diagnosis of where agent performance breaks down.

Integrates with DeepMind FACTS for factuality assessment.
"""
import logging

logger = logging.getLogger(__name__)

# TODO(Day 21): Implement TrajectoryScorer class
#   - score_intent(trajectory) → IntentScore
#   - score_planning(trajectory) → PlanningScore
#   - score_tool_calls(trajectory) → ToolCallScore
#   - score_recovery(trajectory) → RecoveryScore
#   - score_outcome(trajectory) → OutcomeScore
#   - aggregate(scores, weights) → TrajectoryScore
