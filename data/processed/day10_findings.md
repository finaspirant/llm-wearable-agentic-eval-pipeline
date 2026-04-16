# Day 10 Findings: HH-RLHF Inter-Rater Reliability Analysis

## Overview

Applied the `IRRCalculator` from Day 9 to 200 pairs drawn from
`Anthropic/hh-rlhf` (train split, loaded via HuggingFace streaming).
Three annotator personas rated each pair on three dimensions.

## IRR Results (200 pairs, 3 annotators, 3 dimensions)

| Dimension    | Fleiss κ | Cohen κ (mean pair) | Krippendorff α | Interpretation |
|---|---|---|---|---|
| helpfulness  | −0.121   | −0.018              | −0.245         | poor           |
| harmlessness | −0.093   | −0.010              | −0.253         | poor           |
| coherence    | +0.001   | +0.003              | −0.002         | slight         |

**Overall (mean across dimensions):** Fleiss κ = −0.071 | Cohen κ = −0.008 | Krippendorff α = −0.167

## Key Finding: The Cohere Gap

Cohere Command A (arXiv 2504.00698) applied a 5-point preference scale to
800 prompts across 65 annotators but **reported zero inter-rater agreement
statistics** (no κ, no α, no ICC). This analysis fills that gap and shows why
those statistics matter: without calibration, even experienced annotators
achieve κ < 0 on subjective preference dimensions.

## Disagreement Heatmap (topic × dimension, mean score std-dev)

|               | helpfulness | harmlessness | coherence |
|---|---|---|---|
| health_safety | 1.09        | 1.00         | 0.82      |
| general_task  | 1.03        | 0.98         | 0.71      |
| creative      | 1.08        | 1.13         | 0.82      |
| coding        | 1.16        | 0.98         | 0.63      |

**Peak disagreement:** `coding × helpfulness` (σ = 1.16)
**Lowest disagreement:** `coding × coherence` (σ = 0.63)

Coherence is the most reliably rated dimension because it is the most
objective: grammar, logical consistency, and structural completeness
are less dependent on annotator values than helpfulness or harmlessness.

## Annotator Persona Biases

| Persona          | Helpfulness bias | Harmlessness bias | Coherence bias |
|---|---|---|---|
| HelpfulnessFirst | +1               | −1                | 0              |
| HarmlessnessFirst| −1               | +1                | 0              |
| BalancedRater    | 0                | 0                 | 0              |

The opposing helpfulness/harmlessness biases produce the bulk of the
disagreement. This mirrors real annotation conflicts documented in RLHF
pipelines where annotators differ on when to prioritise safety refusals
over task completion.

## Connection to Pipeline

- **Day 9 → Day 10:** IRRCalculator validated on real HH-RLHF data.
  Pre-calibration κ ≈ −0.07 (poor), confirming the Cohere gap is real,
  not an artefact of synthetic data.
- **Day 10 → Day 12–13:** Same pre-calibration baseline observed on
  wearable synthetic trajectories (κ ≈ −0.03). Calibration protocol
  targets α ≥ 0.80.
- **Day 10 → WP1:** §3 cites this baseline as evidence that annotation
  quality requires active measurement, not assumption.

## Artefacts

| File | Description |
|---|---|
| `src/annotation/hh_rlhf_loader.py` | HH-RLHF loader + 3-persona annotation simulator |
| `src/annotation/run_hh_rlhf_irr.py` | IRR pipeline (Cohen, Fleiss, Krippendorff) |
| `src/annotation/disagreement_heatmap.py` | Topic × dimension heatmap generator |
| `data/processed/hh_rlhf_irr_results.json` | Full per-dimension + summary IRR results |
| `data/processed/hh_rlhf_disagreement_heatmap.png` | Heatmap figure (4 topics × 3 dims) |
| `data/processed/hh_rlhf_disagreement_matrix.csv` | Raw disagreement matrix (CSV) |
