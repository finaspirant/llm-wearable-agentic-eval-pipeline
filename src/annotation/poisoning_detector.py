"""Annotator poisoning and outlier detection module.

Implements detection heuristics inspired by Anthropic's 250-doc
backdoor finding (Oct 2025): only 250 malicious documents needed to
backdoor models of any size. Key insight: attack success is count-based,
not proportion-based.

For annotation pipelines, this means a small number of compromised
annotators can systematically bias training data. This module uses
cleanlab confident learning to identify label quality issues.

Detection signal: perplexity differential between triggered and
non-triggered annotation patterns (adapted from Anthropic's method).
"""

import logging

logger = logging.getLogger(__name__)

# TODO(Day 17): Implement PoisoningDetector class
#   - detect_outlier_annotators(annotations, annotator_ids) → list[str]
#   - confidence_score(annotation, context) → float (via cleanlab)
#   - inject_synthetic_poisoner(pool, n_malicious) → pool (for testing)
#   - detection_roc(predictions, ground_truth) → ROC curve
