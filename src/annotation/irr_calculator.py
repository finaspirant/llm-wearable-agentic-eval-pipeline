"""Inter-Rater Reliability calculator for annotation quality assessment.

Computes multiple agreement metrics across annotator pools:
- Cohen's κ (2 raters, nominal)
- Fleiss' κ (3+ raters, nominal)
- Krippendorff's α (any number of raters, any measurement level)
- BERTScore-based semantic agreement (for free-text annotations)

Designed to expose the gap identified in Cohere Command A (2504.00698):
800 prompts, 65 annotators, 5-point scale — but NO agreement statistics
reported. This tool fills that gap.

CLI: python -m src.annotation.irr_calculator --dataset hh_rlhf --metric all
"""
import logging

logger = logging.getLogger(__name__)

# TODO(Day 9): Implement IRRCalculator class
#   - cohens_kappa(rater1, rater2) → float
#   - fleiss_kappa(ratings_matrix) → float
#   - krippendorffs_alpha(reliability_data, level) → float
#   - bertscore_agreement(texts1, texts2) → float
#   - disagreement_heatmap(ratings) → matplotlib figure
#   - CLI via typer with --dataset, --metric, --output flags
