"""Eval quality gate — asserts benchmark metrics against configured thresholds.

Reads benchmark_results.jsonl, computes mean values for each tracked metric,
then exits with code 1 if any metric falls below its threshold.  Designed to
run as the final step in the eval-quality-gate CI job.

Metrics available:
    trajectory_quality  — mean of ``trajectory_score`` across all results
    tool_accuracy       — mean of ``pia_dimensions.tool_precision``

Usage::

    python scripts/check_eval_gate.py \\
        --results data/processed/benchmark_results.jsonl \\
        --thresholds trajectory_quality=0.70,tool_accuracy=0.75
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="check-eval-gate",
    help="Assert benchmark metrics against thresholds. Exits 1 if any breach.",
    add_completion=False,
)

# Canonical metric name → extractor function.
# Each extractor receives a single parsed benchmark result dict and returns
# a float (or None to exclude the record from the mean).
_METRIC_EXTRACTORS: dict[str, object] = {
    "trajectory_quality": lambda r: float(r["trajectory_score"])
    if "trajectory_score" in r
    else None,
    "tool_accuracy": lambda r: float(r["pia_dimensions"]["tool_precision"])
    if "pia_dimensions" in r and "tool_precision" in r.get("pia_dimensions", {})
    else None,
}


def _parse_thresholds(raw: str) -> dict[str, float]:
    """Parse ``key=value,key=value`` threshold string into a dict.

    Args:
        raw: Comma-separated ``metric=threshold`` pairs, e.g.
            ``"trajectory_quality=0.70,tool_accuracy=0.75"``.

    Returns:
        Dict mapping metric name to float threshold.

    Raises:
        typer.BadParameter: If a pair cannot be parsed or the metric is unknown.
    """
    thresholds: dict[str, float] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise typer.BadParameter(f"Expected key=value, got: {pair!r}")
        key, _, val = pair.partition("=")
        key = key.strip()
        if key not in _METRIC_EXTRACTORS:
            known = ", ".join(_METRIC_EXTRACTORS)
            raise typer.BadParameter(
                f"Unknown metric {key!r}. Known metrics: {known}"
            )
        try:
            thresholds[key] = float(val.strip())
        except ValueError:
            raise typer.BadParameter(
                f"Threshold for {key!r} must be a float, got: {val!r}"
            )
    return thresholds


def _load_results(path: Path) -> list[dict[str, object]]:
    """Read and parse all non-empty lines from a JSONL benchmark results file.

    Args:
        path: Path to the ``.jsonl`` file produced by ``benchmark_runner``.

    Returns:
        List of parsed result dicts.
    """
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _compute_means(
    records: list[dict[str, object]],
    metrics: list[str],
) -> dict[str, float]:
    """Compute per-metric means across all records.

    Records missing a metric are excluded from that metric's mean.

    Args:
        records: Parsed benchmark result dicts.
        metrics: Metric names to compute (must be keys of ``_METRIC_EXTRACTORS``).

    Returns:
        Dict mapping metric name to its mean value.  Value is 0.0 if no
        record yielded a valid reading for that metric.
    """
    sums: dict[str, float] = {m: 0.0 for m in metrics}
    counts: dict[str, int] = {m: 0 for m in metrics}

    for record in records:
        for metric in metrics:
            extractor = _METRIC_EXTRACTORS[metric]
            value = extractor(record)  # type: ignore[operator]
            if value is not None:
                sums[metric] += value
                counts[metric] += 1

    return {
        m: (sums[m] / counts[m] if counts[m] > 0 else 0.0) for m in metrics
    }


@app.command()
def main(
    results_path: Annotated[
        Path,
        typer.Option(
            "--results",
            "-r",
            help="JSONL file of benchmark results produced by benchmark_runner.",
        ),
    ] = Path("data/processed/benchmark_results.jsonl"),
    thresholds_raw: Annotated[
        str,
        typer.Option(
            "--thresholds",
            "-t",
            help=(
                "Comma-separated metric=threshold pairs, e.g. "
                "'trajectory_quality=0.70,tool_accuracy=0.75'."
            ),
        ),
    ] = "trajectory_quality=0.70,tool_accuracy=0.75",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging."),
    ] = False,
) -> None:
    """Assert benchmark eval metrics against thresholds.  Exits 1 on breach."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        raise typer.Exit(1)

    thresholds = _parse_thresholds(thresholds_raw)
    records = _load_results(results_path)
    logger.info("Loaded %d benchmark results from %s", len(records), results_path)

    if not records:
        console.print("[bold red]FAIL[/bold red] — results file is empty.")
        raise typer.Exit(1)

    means = _compute_means(records, list(thresholds.keys()))

    # Build rich results table.
    table = Table(
        title=f"Eval Gate — {len(records)} benchmark results",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold", min_width=22)
    table.add_column("Mean", justify="right", min_width=8)
    table.add_column("Threshold", justify="right", min_width=10)
    table.add_column("Status", justify="center", min_width=8)

    any_breach = False
    for metric, threshold in thresholds.items():
        mean_val = means[metric]
        passed = mean_val >= threshold
        if not passed:
            any_breach = True
        status_str = (
            "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        )
        table.add_row(
            metric,
            f"{mean_val:.4f}",
            f"{threshold:.4f}",
            status_str,
        )

    console.print(table)

    if any_breach:
        console.print(
            "\n[bold red]Eval gate FAILED.[/bold red] "
            "One or more metrics are below threshold — PR is blocked."
        )
        raise typer.Exit(1)

    console.print(
        "\n[bold green]Eval gate PASSED.[/bold green] "
        "All metrics meet or exceed thresholds."
    )


if __name__ == "__main__":
    app()
