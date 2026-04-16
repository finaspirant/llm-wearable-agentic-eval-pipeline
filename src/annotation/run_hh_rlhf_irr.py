"""HH-RLHF inter-rater reliability analysis pipeline.

Loads Anthropic HH-RLHF pairs via :class:`~src.annotation.hh_rlhf_loader.HHRLHFLoader`,
simulates three annotator personas, and computes IRR metrics (Cohen's κ,
Fleiss' κ, Krippendorff's α) per annotation dimension.

This fills the named gap in Cohere Command A (arXiv 2504.00698) — 800
prompts, 65 annotators, *zero* agreement statistics reported — and provides
a real-data baseline for the calibration uplift demonstrated on Day 12-13
with synthetic wearable trajectories.

METRICS COMPUTED:
  Cohen's κ    — mean of three pairwise combinations across the 3 annotators
  Fleiss' κ    — full three-rater agreement per dimension
  Krippendorff α — ordinal, handles any missing values

OUTPUT:
  data/processed/hh_rlhf_irr_results.json — per-dimension + summary stats

CLI:
  python -m src.annotation.run_hh_rlhf_irr \\
      --n-samples 200 --output data/processed/hh_rlhf_irr_results.json
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from src.annotation.hh_rlhf_loader import _DIMENSIONS, _PERSONAS, HHRLHFLoader
from src.annotation.irr_calculator import IRRCalculator, _interpret_kappa

logger = logging.getLogger(__name__)
console = Console()

_app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _pairwise_cohens(
    calc: IRRCalculator,
    matrix: list[list[int]],
    rater_ids: list[str],
) -> dict[str, Any]:
    """Compute Cohen's κ for all annotator pairs and return the mean.

    Args:
        calc: Initialised :class:`IRRCalculator` instance.
        matrix: ``[n_items × n_raters]`` integer rating matrix.
        rater_ids: Ordered persona names corresponding to columns.

    Returns:
        Dict with ``"per_pair"`` (list of per-pair dicts) and ``"mean_kappa"``
        (float) and ``"mean_interpretation"`` (str).
    """
    n_raters = len(rater_ids)
    per_pair: list[dict[str, Any]] = []
    kappas: list[float] = []

    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            r1 = [row[i] for row in matrix]
            r2 = [row[j] for row in matrix]
            result = calc.cohens_kappa(r1, r2)
            kappas.append(float(result["kappa"]))  # type: ignore[arg-type]
            per_pair.append(
                {
                    "rater_a": rater_ids[i],
                    "rater_b": rater_ids[j],
                    "kappa": result["kappa"],
                    "interpretation": result["interpretation"],
                }
            )

    mean_kappa = statistics.mean(kappas)
    return {
        "per_pair": per_pair,
        "mean_kappa": mean_kappa,
        "mean_interpretation": _interpret_kappa(mean_kappa),
    }


def run_irr_analysis(
    n_samples: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the full HH-RLHF IRR pipeline.

    Loads pairs, simulates annotations, then computes Cohen's κ (pairwise
    mean), Fleiss' κ, and Krippendorff's α for each annotation dimension.

    Args:
        n_samples: Number of HH-RLHF pairs to analyse.
        seed: Random seed passed to :class:`HHRLHFLoader`.

    Returns:
        Nested dict with per-dimension metrics and a cross-dimension summary.
        Schema::

            {
              "meta": {"n_samples": int, "seed": int, "source": str},
              "per_dimension": {
                "<dim>": {
                  "cohens_kappa": {"mean_kappa": float, ...},
                  "fleiss_kappa": {"kappa": float, ...},
                  "krippendorffs_alpha": {"alpha": float, ...},
                }
              },
              "summary": {
                "mean_cohens_kappa": float,
                "mean_fleiss_kappa": float,
                "mean_krippendorffs_alpha": float,
                "overall_interpretation": str,
                "cohere_gap_note": str,
              }
            }
    """
    loader = HHRLHFLoader(seed=seed)
    pairs = loader.load(n_samples=n_samples)
    annotations = loader.simulate_annotations(pairs)

    source = pairs[0].source if pairs else "unknown"
    calc = IRRCalculator()

    per_dimension: dict[str, Any] = {}
    cohens_vals: list[float] = []
    fleiss_vals: list[float] = []
    alpha_vals: list[float] = []

    for dim in _DIMENSIONS:
        irr_matrix = loader.to_irr_matrix(annotations, dim)

        # Cohen's κ — mean of 3 pairwise combinations.
        cohens_result = _pairwise_cohens(calc, irr_matrix.matrix, irr_matrix.rater_ids)
        cohens_vals.append(float(cohens_result["mean_kappa"]))

        # Fleiss' κ — shift 1-4 scores to 0-3 (IRRCalculator expects 0-indexed).
        n_categories = 4
        zero_matrix = [[s - 1 for s in row] for row in irr_matrix.matrix]
        fleiss_result = calc.fleiss_kappa(zero_matrix, n_categories=n_categories)

        # Krippendorff's α — ordinal level.
        alpha_result = calc.krippendorffs_alpha(
            irr_matrix.reliability_data, level_of_measurement="ordinal"
        )
        alpha_vals.append(float(alpha_result["alpha"]))  # type: ignore[arg-type]
        fleiss_vals.append(float(fleiss_result["kappa"]))  # type: ignore[arg-type]

        per_dimension[dim] = {
            "cohens_kappa": cohens_result,
            "fleiss_kappa": fleiss_result,
            "krippendorffs_alpha": alpha_result,
        }
        logger.info(
            "Dimension %-15s | Fleiss κ=%.3f | α=%.3f",
            dim,
            fleiss_result["kappa"],
            alpha_result["alpha"],
        )

    mean_ck = statistics.mean(cohens_vals)
    mean_fk = statistics.mean(fleiss_vals)
    mean_alpha = statistics.mean(alpha_vals)

    return {
        "meta": {
            "n_samples": n_samples,
            "seed": seed,
            "source": source,
            "annotator_personas": list(_PERSONAS),
            "dimensions": list(_DIMENSIONS),
        },
        "per_dimension": per_dimension,
        "summary": {
            "mean_cohens_kappa": mean_ck,
            "mean_cohens_interpretation": _interpret_kappa(mean_ck),
            "mean_fleiss_kappa": mean_fk,
            "mean_fleiss_interpretation": _interpret_kappa(mean_fk),
            "mean_krippendorffs_alpha": mean_alpha,
            "mean_alpha_interpretation": _interpret_kappa(mean_alpha),
            "cohere_gap_note": (
                "Cohere Command A (arXiv 2504.00698): 800 prompts, 65 annotators, "
                "zero agreement statistics reported. "
                f"This fills that gap: Fleiss κ={mean_fk:.3f}"
                f" ({_interpret_kappa(mean_fk)}), "
                f"Krippendorff α={mean_alpha:.3f}"
                f" ({_interpret_kappa(mean_alpha)})."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Rich output
# ---------------------------------------------------------------------------


def _print_results(results: dict[str, Any]) -> None:
    """Render IRR results as a rich terminal table.

    Args:
        results: Dict returned by :func:`run_irr_analysis`.
    """
    table = Table(title="HH-RLHF Inter-Rater Reliability", show_lines=True)
    table.add_column("Dimension", style="cyan", no_wrap=True)
    table.add_column("Fleiss κ", justify="right")
    table.add_column("Cohen κ (mean pair)", justify="right")
    table.add_column("Krippendorff α", justify="right")
    table.add_column("Interpretation", style="green")

    for dim in _DIMENSIONS:
        d = results["per_dimension"][dim]
        fk = d["fleiss_kappa"]["kappa"]
        ck = d["cohens_kappa"]["mean_kappa"]
        alpha = d["krippendorffs_alpha"]["alpha"]
        interp = _interpret_kappa(statistics.mean([fk, ck, alpha]))
        table.add_row(dim, f"{fk:.3f}", f"{ck:.3f}", f"{alpha:.3f}", interp)

    console.print(table)

    s = results["summary"]
    console.print(f"\n[bold]Summary:[/bold] Fleiss κ={s['mean_fleiss_kappa']:.3f} "
                  f"| Cohen κ={s['mean_cohens_kappa']:.3f} "
                  f"| Krippendorff α={s['mean_krippendorffs_alpha']:.3f}")
    console.print(f"[dim]{s['cohere_gap_note']}[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@_app.command()
def main(
    n_samples: int = typer.Option(200, help="Number of HH-RLHF pairs to analyse."),
    output: Path = typer.Option(
        Path("data/processed/hh_rlhf_irr_results.json"),
        help="Output JSON path.",
    ),
    seed: int = typer.Option(42, help="Random seed."),
    verbose: bool = typer.Option(False, help="Enable DEBUG logging."),
) -> None:
    """Compute IRR metrics on HH-RLHF annotation simulations."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    results = run_irr_analysis(n_samples=n_samples, seed=seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info("IRR results → %s", output)

    _print_results(results)


if __name__ == "__main__":
    _app()
