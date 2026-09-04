"""Analyse the Prolific human PIA annotation study (data/human_study/).

Cleans the raw Prolific export, then computes two inter-rater agreement
numbers using the *same* Fleiss' kappa machinery as
``src.annotation.pia_calculator`` (``IRRCalculator.fleiss_kappa``):

* **Standard path-comparison kappa (Mode A)** -- step-level agreement on
  "which steps are wrong". The human instrument records a set of flagged
  step indices per (rater, trajectory); each step is one binary item
  (flagged / not flagged), Fleiss' kappa with ``n_categories=2``.
  NOTE: this differs from ``pia_calculator``'s Mode A, which rates every
  step 1-4. The study collected binary "wrong step" flags instead, so the
  step-count per trajectory is not known exactly -- see the sensitivity
  block for how the number moves with that assumption.

* **PIA rubric kappa (Mode B)** -- trajectory-level agreement on the rubric
  dimensions. 10 trajectories x 5 raters, scores 1-4, Fleiss' kappa with
  ``n_categories=4`` per dimension; overall = mean of per-dimension kappa
  (matching ``PIACalculator.compare``'s ``pia_overall_kappa``).
  Reported both 3-dimension (planning, recovery, goal -- the paper's
  current definition) and 4-dimension (adding privacy).

Usage::

    uv run python scripts/analyze_human_pia_study.py
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.annotation.irr_calculator import IRRCalculator

logger = logging.getLogger(__name__)

_CSV_PATH = Path("data/human_study/annotation-responses.csv")
_OUT_PATH = Path("data/human_study/human_pia_results.json")

_N_TRAJ = 10
_DIMENSIONS = ("planning", "recovery", "goal", "privacy")
_PIA_3DIM = ("planning", "recovery", "goal")
_RUBRIC_SCALE = 4  # scores are 1-4
_STEP_FLOOR = 3  # every study trajectory appears to have >= 3 steps


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def load_and_clean(csv_path: Path) -> list[dict[str, str]]:
    """Load the Prolific export, drop the test row, dedupe by participant.

    Dedup keeps the row with the latest ``submitted_at`` per ``prolific_pid``.

    Args:
        csv_path: Path to the raw Prolific CSV export.

    Returns:
        One cleaned row dict per unique real participant.
    """
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    real = [r for r in rows if r["prolific_pid"].strip().lower() != "test"]
    dropped_test = len(rows) - len(real)

    by_pid: dict[str, dict[str, str]] = {}
    for r in real:
        pid = r["prolific_pid"]
        if pid not in by_pid or r["submitted_at"] > by_pid[pid]["submitted_at"]:
            by_pid[pid] = r
    deduped = list(by_pid.values())

    logger.info(
        "Cleaned: %d raw rows -> dropped %d test -> %d real submissions -> "
        "%d unique participants",
        len(rows),
        dropped_test,
        len(real),
        len(deduped),
    )
    for r in deduped:
        logger.info(
            "  participant %s (submitted %s)", r["prolific_pid"], r["submitted_at"]
        )
    return deduped


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_wrong_steps(cell: str) -> set[int]:
    """Parse a ``"1,2,3"`` style cell into a set of 1-based step indices."""
    cell = cell.strip()
    if not cell:
        return set()
    return {int(tok) for tok in cell.split(",") if tok.strip()}


def parse_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Extract rubric scores and flagged-step sets from cleaned rows.

    Args:
        rows: Cleaned participant rows.

    Returns:
        Dict with ``participants`` (ordered pids), ``scores``
        (``dim -> traj_idx -> [score per participant]``), and ``flags``
        (``traj_idx -> [set(step) per participant]``).
    """
    pids = [r["prolific_pid"] for r in rows]
    scores: dict[str, dict[int, list[int]]] = {
        d: {t: [] for t in range(1, _N_TRAJ + 1)} for d in _DIMENSIONS
    }
    flags: dict[int, list[set[int]]] = {t: [] for t in range(1, _N_TRAJ + 1)}

    for r in rows:
        for t in range(1, _N_TRAJ + 1):
            for d in _DIMENSIONS:
                raw = r[f"t{t}_{d}"].strip()
                if raw == "":
                    raise ValueError(
                        f"participant {r['prolific_pid']} has empty {d} for t{t}"
                    )
                scores[d][t].append(int(raw))
            flags[t].append(_parse_wrong_steps(r[f"t{t}_wrong_steps"]))

    return {"participants": pids, "scores": scores, "flags": flags}


# ---------------------------------------------------------------------------
# Mode B -- PIA rubric kappa
# ---------------------------------------------------------------------------


def pia_rubric_kappa(
    irr: IRRCalculator, scores: dict[str, dict[int, list[int]]]
) -> dict[str, Any]:
    """Fleiss' kappa per rubric dimension + 3-dim and 4-dim overall means.

    Args:
        irr: Shared IRR calculator.
        scores: ``dim -> traj_idx -> [score per participant]`` (1-4).

    Returns:
        Dict with ``per_dimension`` kappa, ``overall_3dim``, ``overall_4dim``.
    """
    per_dim: dict[str, float] = {}
    for d in _DIMENSIONS:
        matrix = [[s - 1 for s in scores[d][t]] for t in range(1, _N_TRAJ + 1)]
        res = irr.fleiss_kappa(matrix, n_categories=_RUBRIC_SCALE)
        per_dim[d] = float(res["kappa"])  # type: ignore[arg-type]

    overall_3 = sum(per_dim[d] for d in _PIA_3DIM) / len(_PIA_3DIM)
    overall_4 = sum(per_dim[d] for d in _DIMENSIONS) / len(_DIMENSIONS)
    return {
        "per_dimension": per_dim,
        "overall_3dim": overall_3,
        "overall_4dim": overall_4,
    }


# ---------------------------------------------------------------------------
# Mode A -- standard path-comparison kappa
# ---------------------------------------------------------------------------


def _step_kappa(
    irr: IRRCalculator, flags: dict[int, list[set[int]]], n_steps: dict[int, int]
) -> tuple[float, int]:
    """Fleiss' kappa over binary per-step "flagged wrong" labels.

    Args:
        irr: Shared IRR calculator.
        flags: ``traj_idx -> [set(flagged step) per participant]``.
        n_steps: ``traj_idx -> assumed step count``.

    Returns:
        ``(kappa, n_step_items)``.
    """
    matrix: list[list[int]] = []
    for t in range(1, _N_TRAJ + 1):
        for step in range(1, n_steps[t] + 1):
            matrix.append(
                [1 if step in flags[t][p] else 0 for p in range(len(flags[t]))]
            )
    res = irr.fleiss_kappa(matrix, n_categories=2)
    return float(res["kappa"]), len(matrix)  # type: ignore[arg-type]


def standard_path_kappa(
    irr: IRRCalculator, flags: dict[int, list[set[int]]]
) -> dict[str, Any]:
    """Mode A kappa under several step-count assumptions.

    Primary: per-trajectory step count = max flagged index (floor
    ``_STEP_FLOOR``). Sensitivity: uniform 3 / 4 / 5 steps for every
    trajectory, and "only steps flagged by >= 1 rater".

    Args:
        irr: Shared IRR calculator.
        flags: ``traj_idx -> [set(flagged step) per participant]``.

    Returns:
        Dict of assumption label -> {kappa, n_items, n_steps_per_traj}.
    """
    max_flag = {
        t: max((max(s) for s in flags[t] if s), default=0)
        for t in range(1, _N_TRAJ + 1)
    }

    out: dict[str, Any] = {"max_flagged_index_per_traj": max_flag, "variants": {}}

    primary = {t: max(max_flag[t], _STEP_FLOOR) for t in range(1, _N_TRAJ + 1)}
    k, n = _step_kappa(irr, flags, primary)
    out["variants"]["primary_maxflag_floor3"] = {
        "kappa": k,
        "n_step_items": n,
        "n_steps_per_traj": primary,
    }

    # If the 10 study trajectories are the 10 pia_pairs' indirect (agent_b)
    # paths, their step counts are [5,4,4,5,4,5,4,5,4,5].
    agent_b = dict(
        zip(range(1, _N_TRAJ + 1), (5, 4, 4, 5, 4, 5, 4, 5, 4, 5), strict=True)
    )
    k, n = _step_kappa(irr, flags, agent_b)
    out["variants"]["pia_pairs_agent_b_stepcounts"] = {
        "kappa": k,
        "n_step_items": n,
        "n_steps_per_traj": agent_b,
    }

    for uniform in (3, 4, 5):
        u = {t: uniform for t in range(1, _N_TRAJ + 1)}
        k, n = _step_kappa(irr, flags, u)
        out["variants"][f"uniform_{uniform}_steps"] = {
            "kappa": k,
            "n_step_items": n,
            "n_steps_per_traj": u,
        }

    # Only steps flagged by at least one rater become items.
    flagged_only = {t: max(max_flag[t], 0) for t in range(1, _N_TRAJ + 1)}
    if all(v == 0 for v in flagged_only.values()):
        out["variants"]["flagged_steps_only"] = {"kappa": None, "note": "no flags"}
    else:
        # steps 1..max_flag; drop trajectories with zero flags entirely
        fmatrix: list[list[int]] = []
        for t in range(1, _N_TRAJ + 1):
            for step in range(1, max_flag[t] + 1):
                fmatrix.append(
                    [1 if step in flags[t][p] else 0 for p in range(len(flags[t]))]
                )
        res = irr.fleiss_kappa(fmatrix, n_categories=2)
        out["variants"]["flagged_steps_only"] = {
            "kappa": float(res["kappa"]),  # type: ignore[arg-type]
            "n_step_items": len(fmatrix),
            "n_steps_per_traj": flagged_only,
        }

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full clean + compute pipeline and write ``human_pia_results.json``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    irr = IRRCalculator()

    rows = load_and_clean(_CSV_PATH)
    assert len(rows) == 5, f"expected 5 unique participants, got {len(rows)}"

    parsed = parse_rows(rows)
    scores = parsed["scores"]
    flags = parsed["flags"]

    mode_b = pia_rubric_kappa(irr, scores)
    mode_a = standard_path_kappa(irr, flags)

    # Flag participants who flagged an implausible number of steps (>=6),
    # a low-effort-response signal that materially shifts Mode A.
    outliers = [
        pid
        for i, pid in enumerate(parsed["participants"])
        if any(len(flags[t][i]) >= 6 for t in range(1, _N_TRAJ + 1))
    ]

    std_primary = mode_a["variants"]["primary_maxflag_floor3"]["kappa"]

    result = {
        "n_participants": len(rows),
        "participants": parsed["participants"],
        "low_effort_participants": outliers,
        "n_trajectories": _N_TRAJ,
        "raw_dimension_scores": {
            d: {f"t{t}": scores[d][t] for t in range(1, _N_TRAJ + 1)}
            for d in _DIMENSIONS
        },
        "flagged_steps": {
            f"t{t}": [sorted(s) for s in flags[t]] for t in range(1, _N_TRAJ + 1)
        },
        "mode_a_standard_path_comparison": mode_a,
        "mode_b_pia_rubric": mode_b,
        "headline": {
            "standard_kappa_primary": std_primary,
            "pia_kappa_3dim": mode_b["overall_3dim"],
            "pia_kappa_4dim": mode_b["overall_4dim"],
            "delta_3dim": mode_b["overall_3dim"] - std_primary,
            "delta_4dim": mode_b["overall_4dim"] - std_primary,
        },
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(result, indent=2, default=str))

    # ---- console report ----
    print("\n=== CLEANED DATASET ===")
    print(f"participants ({len(rows)}): {', '.join(parsed['participants'])}")
    print(f"low-effort (flagged >=6 steps on some trajectory): {outliers or 'none'}")

    print(
        "\n=== MODE B: PIA RUBRIC Fleiss' kappa (10 trajectories x 5 raters, 1-4) ==="
    )
    for d, k in mode_b["per_dimension"].items():
        print(f"  {d:<10} kappa = {k:+.4f}")
    o3 = mode_b["overall_3dim"]
    o4 = mode_b["overall_4dim"]
    print(f"  overall 3-dim (planning+recovery+goal) = {o3:+.4f}")
    print(f"  overall 4-dim (+privacy)               = {o4:+.4f}")

    print(
        "\n=== MODE A: STANDARD PATH-COMPARISON Fleiss' kappa (binary wrong-step) ==="
    )
    print(f"  max flagged step index per traj: {mode_a['max_flagged_index_per_traj']}")
    for label, v in mode_a["variants"].items():
        if v.get("kappa") is None:
            print(f"  {label:<28} -> {v.get('note')}")
        else:
            n = v["n_step_items"]
            print(f"  {label:<28} kappa = {v['kappa']:+.4f}  ({n} step-items)")

    print("\n=== HEADLINE ===")
    h = result["headline"]
    k3, d3 = h["pia_kappa_3dim"], h["delta_3dim"]
    k4, d4 = h["pia_kappa_4dim"], h["delta_4dim"]
    print(f"  standard kappa (primary) = {h['standard_kappa_primary']:+.4f}")
    print(f"  PIA kappa 3-dim          = {k3:+.4f}  (delta {d3:+.4f})")
    print(f"  PIA kappa 4-dim          = {k4:+.4f}  (delta {d4:+.4f})")
    print(f"\nWrote {_OUT_PATH}")


if __name__ == "__main__":
    main()
