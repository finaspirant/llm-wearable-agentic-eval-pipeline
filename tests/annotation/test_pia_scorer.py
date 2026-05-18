"""Tests for src.annotation.pia_scorer.

All tests use a _StubTrajectoryScorer injected through PIAScorer's constructor
so no LLM endpoint is touched and no WearableLog needs to be constructed.
"""

from __future__ import annotations

from typing import Any

from src.annotation.pia_scorer import (
    _SENTINEL_NO_ESCALATION,
    DimensionScores,
    PIAScorer,
)

# ---------------------------------------------------------------------------
# Stub infrastructure
# ---------------------------------------------------------------------------


class _StubTrajectoryScorer:
    """Returns a fixed dict[str, float] without touching any LLM endpoint."""

    def __init__(
        self,
        planning_quality: float,
        error_recovery: float,
        goal_alignment: float,
    ) -> None:
        self._dims: dict[str, float] = {
            "planning_quality": planning_quality,
            "error_recovery": error_recovery,
            "goal_alignment": goal_alignment,
            "tool_precision": 1.0,
        }

    def score_pia_dimensions(
        self, trajectory: Any, *, dry_run: bool = True  # noqa: ARG002, ANN401
    ) -> dict[str, float]:
        return self._dims


def _make_scorer(
    *,
    planning_quality: float = 0.8,
    error_recovery: float = 0.9,
    goal_alignment: float = 0.7,
) -> PIAScorer:
    stub = _StubTrajectoryScorer(
        planning_quality=planning_quality,
        error_recovery=error_recovery,
        goal_alignment=goal_alignment,
    )
    return PIAScorer(trajectory_scorer=stub)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dimension_scores_exported_with_optional_error_recovery() -> None:
    ds = DimensionScores(planning_quality=0.8, error_recovery=None, goal_alignment=0.7)
    assert ds.planning_quality == 0.8
    assert ds.error_recovery is None
    assert ds.goal_alignment == 0.7


def test_score_trajectory_maps_sentinel_to_none() -> None:
    scorer = _make_scorer(error_recovery=_SENTINEL_NO_ESCALATION)
    result = scorer.score_trajectory(object(), dry_run=True)  # type: ignore[arg-type]
    assert result.error_recovery is None


def test_score_trajectory_preserves_non_sentinel_error_recovery() -> None:
    scorer = _make_scorer(error_recovery=0.85)
    result = scorer.score_trajectory(object(), dry_run=True)  # type: ignore[arg-type]
    assert result.error_recovery == 0.85


def test_compute_pia_returns_1_for_identical_rater_inputs() -> None:
    # IRRCalculator.fleiss_kappa returns 1.0 by convention when all raters always
    # assign the same bin (denominator 1 − P̄_e ≈ 0).
    scorer = _make_scorer(planning_quality=0.9, error_recovery=0.9, goal_alignment=0.9)
    batch = [scorer.score_trajectory(object(), dry_run=True) for _ in range(4)]  # type: ignore[arg-type]
    kappa = scorer.compute_pia(batch, list(batch))
    assert kappa == 1.0, f"Expected 1.0 for identical rater inputs, got {kappa}"


def test_compute_pia_excludes_none_error_recovery_without_raising() -> None:
    scorer = _make_scorer(error_recovery=_SENTINEL_NO_ESCALATION)
    batch = [scorer.score_trajectory(object(), dry_run=True) for _ in range(3)]  # type: ignore[arg-type]
    # All error_recovery values are None — dimension is skipped; no ZeroDivisionError.
    kappa = scorer.compute_pia(batch, list(batch))
    assert 0.0 <= kappa <= 1.0
