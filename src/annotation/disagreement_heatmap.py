"""Annotator disagreement heatmap generator for HH-RLHF IRR analysis.

Reads IRR results from :mod:`src.annotation.run_hh_rlhf_irr` and raw
annotations from :class:`~src.annotation.hh_rlhf_loader.HHRLHFLoader`,
then produces two artefacts:

1. **PNG heatmap** — seaborn annotated heatmap with axes:
     - X: annotation dimension (helpfulness, harmlessness, coherence)
     - Y: topic bucket (health_safety, general_task, creative, coding)
     - Cell: mean pairwise score std-dev (higher = more disagreement)

2. **CSV matrix** — the same values as a tidy CSV for downstream analysis.

Key finding visualised:
  ``health_safety × harmlessness`` consistently shows the highest
  disagreement — the HelpfulnessFirst vs HarmlessnessFirst persona split
  is widest when both safety and task completion are at stake.
  This motivates the calibration protocol in Day 13 (calibration_protocol.py).

CLI:
  python -m src.annotation.disagreement_heatmap \\
      --n-samples 200 \\
      --output-png data/processed/hh_rlhf_disagreement_heatmap.png \\
      --output-csv data/processed/hh_rlhf_disagreement_matrix.csv
"""

from __future__ import annotations

import csv
import logging
import statistics
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import typer

from src.annotation.hh_rlhf_loader import (
    _DIMENSIONS,
    _TOPICS,
    HHRLHFAnnotation,
    HHRLHFLoader,
)

matplotlib.use("Agg")  # non-interactive backend for CI / headless runs

logger = logging.getLogger(__name__)

_app = typer.Typer(add_completion=False)

# ---------------------------------------------------------------------------
# Disagreement computation
# ---------------------------------------------------------------------------


def compute_disagreement_matrix(
    annotations: list[HHRLHFAnnotation],
) -> dict[str, dict[str, float]]:
    """Compute per-(topic, dimension) mean pairwise score standard deviation.

    For each (topic, dimension) cell, collects all item-level score
    vectors (one score per annotator persona) and computes the population
    standard deviation across personas.  The cell value is then averaged
    across all items in that cell.

    High std-dev → high annotator disagreement on that dimension/topic.
    Low std-dev → annotators largely agree, calibration not critical there.

    Args:
        annotations: Full flat annotation list from
            :meth:`HHRLHFLoader.simulate_annotations`.

    Returns:
        Nested dict ``{topic: {dimension: mean_std_dev}}``.  Topics are
        rows; dimensions are columns — matching the heatmap layout.
    """
    # Group scores by (topic, dimension, sample_id).
    buckets: dict[tuple[str, str, str], list[int]] = {}
    for ann in annotations:
        key = (ann.topic, ann.dimension, ann.sample_id)
        buckets.setdefault(key, []).append(ann.score)

    # Aggregate to (topic, dimension) mean std-dev.
    cell_stdevs: dict[tuple[str, str], list[float]] = {}
    for (topic, dim, _sid), scores in buckets.items():
        if len(scores) >= 2:
            cell_stdevs.setdefault((topic, dim), []).append(
                statistics.stdev(scores)
            )

    result: dict[str, dict[str, float]] = {t: {} for t in _TOPICS}
    for topic in _TOPICS:
        for dim in _DIMENSIONS:
            vals = cell_stdevs.get((topic, dim), [0.0])
            result[topic][dim] = statistics.mean(vals) if vals else 0.0

    return result


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------


def render_heatmap(
    matrix: dict[str, dict[str, float]],
    output_png: Path,
    output_csv: Path,
) -> None:
    """Render the disagreement matrix as a PNG heatmap and CSV file.

    Args:
        matrix: Nested dict ``{topic: {dimension: mean_std_dev}}`` from
            :func:`compute_disagreement_matrix`.
        output_png: Destination path for the PNG figure.
        output_csv: Destination path for the CSV export.
    """
    # Build ordered 2-D list for seaborn (rows=topics, cols=dimensions).
    data: list[list[float]] = [
        [matrix[topic].get(dim, 0.0) for dim in _DIMENSIONS]
        for topic in _TOPICS
    ]

    # ---- CSV export -------------------------------------------------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["topic"] + list(_DIMENSIONS))
        for topic, row in zip(_TOPICS, data):
            writer.writerow([topic] + [f"{v:.4f}" for v in row])
    logger.info("Disagreement matrix → %s", output_csv)

    # ---- PNG heatmap ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=list(_DIMENSIONS),
        yticklabels=list(_TOPICS),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean score std-dev (higher = more disagreement)"},
        ax=ax,
        vmin=0.0,
        vmax=1.5,
    )

    ax.set_title(
        "Annotator Disagreement: HH-RLHF (topic × dimension)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("Annotation Dimension", fontsize=10)
    ax.set_ylabel("Topic Bucket", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Annotate the highest-disagreement cell in the title footnote.
    max_val = max(v for row in data for v in row)
    max_topic = _TOPICS[next(i for i, row in enumerate(data) if max(row) == max_val)]
    max_topic_idx = _TOPICS.index(max_topic)
    max_dim = _DIMENSIONS[
        next(j for j, v in enumerate(data[max_topic_idx]) if v == max_val)
    ]
    ax.text(
        0.5,
        -0.12,
        f"Peak disagreement: {max_topic} × {max_dim} (σ={max_val:.2f}) "
        "— motivates calibration protocol (Day 13)",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="dimgray",
    )

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Disagreement heatmap → %s", output_png)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@_app.command()
def main(
    n_samples: int = typer.Option(200, help="Number of HH-RLHF pairs to load."),
    output_png: Path = typer.Option(
        Path("data/processed/hh_rlhf_disagreement_heatmap.png"),
        help="Output PNG path.",
    ),
    output_csv: Path = typer.Option(
        Path("data/processed/hh_rlhf_disagreement_matrix.csv"),
        help="Output CSV path.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    verbose: bool = typer.Option(False, help="Enable DEBUG logging."),
) -> None:
    """Generate topic × dimension annotator disagreement heatmap."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    loader = HHRLHFLoader(seed=seed)
    pairs = loader.load(n_samples=n_samples)
    annotations = loader.simulate_annotations(pairs)

    matrix = compute_disagreement_matrix(annotations)
    render_heatmap(matrix, output_png, output_csv)

    console_rows = "\n".join(
        f"  {topic:<18} " + "  ".join(f"{matrix[topic][d]:.2f}" for d in _DIMENSIONS)
        for topic in _TOPICS
    )
    logger.info(
        "Disagreement matrix (rows=topic, cols=%s):\n%s", _DIMENSIONS, console_rows
    )


if __name__ == "__main__":
    _app()
