"""Day 17 detection pipeline script.

Runs the full PoisoningDetector pipeline on day12_annotations.jsonl and
saves results to data/processed/day17_detection_results.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Ensure project root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.annotation.poisoning_detector import PoisoningDetector  # noqa: E402

logging.basicConfig(level=logging.WARNING)
console = Console()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ANNOTATIONS_PATH = _ROOT / "data" / "annotations" / "day12_annotations.jsonl"
OUTPUT_PATH = _ROOT / "data" / "processed" / "day17_detection_results.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INJECTED_NAMES = ["Poisoner_A", "Poisoner_B", "Poisoner_C"]


def _score_table(title: str, scores: dict[str, float]) -> Table:
    table = Table(title=title, show_lines=True)
    table.add_column("Persona", style="cyan", no_wrap=True)
    table.add_column("Suspicion Score", justify="right", style="magenta")
    table.add_column("Flag", justify="center")

    for persona, score in sorted(scores.items(), key=lambda x: -x[1]):
        flag = "[bold red]FLAGGED[/bold red]" if score >= 0.6 else ""
        table.add_row(persona, f"{score:.4f}", flag)

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    results: dict = {}

    # ------------------------------------------------------------------
    # Step 1: Load annotations
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 1 — Load Annotations")
    records: list[dict] = []
    with ANNOTATIONS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    console.print(f"Loaded [bold]{len(records)}[/bold] records from {ANNOTATIONS_PATH.name}")
    results["input_path"] = str(ANNOTATIONS_PATH)
    results["n_records_loaded"] = len(records)

    # ------------------------------------------------------------------
    # Step 2: Instantiate detector
    # ------------------------------------------------------------------
    detector = PoisoningDetector()

    # ------------------------------------------------------------------
    # Step 3: Detect on clean pool (5 personas)
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 3 — Detect Outliers (Clean Pool)")
    clean_scores = detector.detect_outlier_annotators(records)
    console.print(_score_table("Clean Pool — Suspicion Scores", clean_scores))
    results["clean_pool_scores"] = clean_scores

    # ------------------------------------------------------------------
    # Step 4: Inject synthetic poisoners
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 4 — Inject Synthetic Poisoners")
    augmented = detector.inject_synthetic_poisoners(records, n_malicious=3, seed=42)
    console.print(
        f"Augmented pool: [bold]{len(augmented)}[/bold] total records "
        f"(original {len(records)} + {len(augmented) - len(records)} injected)"
    )
    results["n_records_augmented"] = len(augmented)

    # ------------------------------------------------------------------
    # Step 5: Detect on augmented pool (8 annotators)
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 5 — Detect Outliers (Augmented Pool)")
    aug_scores = detector.detect_outlier_annotators(augmented)
    console.print(_score_table("Augmented Pool — Suspicion Scores", aug_scores))

    detected_count = sum(
        1 for name in INJECTED_NAMES if aug_scores.get(name, 0.0) >= 0.6
    )
    color = "green" if detected_count > 0 else "red"
    console.print(
        f"\nInjected poisoners detected above 0.6 threshold: "
        f"[bold {color}]{detected_count}/3[/bold {color}]"
    )
    results["augmented_pool_scores"] = aug_scores
    results["poisoners_detected_above_0_6"] = detected_count

    # ------------------------------------------------------------------
    # Step 6: Evaluate detection (precision / recall / F1)
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 6 — Evaluate Detection")
    eval_report = detector.evaluate_detection(
        augmented,
        injected_names=INJECTED_NAMES,
        threshold=0.6,
    )

    eval_table = Table(title="Detection Evaluation (threshold=0.6)", show_lines=True)
    eval_table.add_column("Metric", style="cyan")
    eval_table.add_column("Value", justify="right", style="green")
    key_labels = [
        ("threshold", "Threshold"),
        ("true_positives", "True Positives"),
        ("false_positives", "False Positives"),
        ("false_negatives", "False Negatives"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]
    for key, label in key_labels:
        val = eval_report[key]
        eval_table.add_row(label, f"{val:.4f}" if isinstance(val, float) else str(val))

    console.print(eval_table)
    results["evaluation"] = {
        k: v for k, v in eval_report.items() if k != "per_annotator_scores"
    }
    results["evaluation"]["per_annotator_scores"] = eval_report["per_annotator_scores"]

    # ------------------------------------------------------------------
    # Step 7: Cleanlab label quality on privacy_compliance
    # ------------------------------------------------------------------
    console.rule("[bold blue]Step 7 — Cleanlab Label Quality (privacy_compliance)")
    cl_report = detector.cleanlab_label_quality(augmented, dimension="privacy_compliance")

    cl_table = Table(title="Cleanlab Label Quality — privacy_compliance", show_lines=True)
    cl_table.add_column("Field", style="cyan")
    cl_table.add_column("Value", style="yellow")
    cl_table.add_row("dimension", cl_report["dimension"])
    cl_table.add_row("n_logs", str(cl_report["n_logs"]))
    cl_table.add_row("n_issues_found", str(cl_report["n_issues_found"]))
    flagged_preview = cl_report["flagged_log_ids"][:5]
    cl_table.add_row("flagged_log_ids (first 5)", ", ".join(flagged_preview) if flagged_preview else "—")

    console.print(cl_table)
    results["cleanlab"] = {
        "dimension": cl_report["dimension"],
        "n_logs": cl_report["n_logs"],
        "n_issues_found": cl_report["n_issues_found"],
        "flagged_log_ids": cl_report["flagged_log_ids"],
        "quality_scores": cl_report["quality_scores"],
    }

    # ------------------------------------------------------------------
    # Save full results
    # ------------------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as fh:
        json.dump(results, fh, indent=2)

    console.rule("[bold green]Complete")
    console.print(f"Full results saved to [bold]{OUTPUT_PATH}[/bold]")


if __name__ == "__main__":
    main()
