"""Path-Invariant Agreement (PIA) scorer for non-deterministic agents.

Original methodological contribution. Standard IRR assumes identical
stimuli — invalid for agents that produce legitimately different but
equally correct trajectories. PIA measures agreement on rubric
dimensions (planning quality, error recovery, goal alignment) rather
than path-specific choices.

No existing paper addresses this problem by name. This is a publishable
contribution targeting WP2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, cast

from src.annotation.irr_calculator import IRRCalculator
from src.data.wearable_generator import WearableLog
from src.eval.trajectory_scorer import TrajectoryScorer

logger = logging.getLogger(__name__)

# TrajectoryScorer.score_pia_dimensions() emits this sentinel value for
# error_recovery when the trajectory had no mid-step escalation event.
_SENTINEL_NO_ESCALATION: float = 0.5

# Bin continuous [0.0, 1.0] dimension scores into low / medium / high for
# the Fleiss' kappa rating matrix.
_N_BINS: int = 3


# ---------------------------------------------------------------------------
# Internal protocol — allows test injection without subclassing TrajectoryScorer
# ---------------------------------------------------------------------------


class _PIAScorable(Protocol):
    """Structural interface: any object with score_pia_dimensions qualifies."""

    def score_pia_dimensions(self, trajectory: WearableLog) -> dict[str, float]:
        ...


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class DimensionScores:
    """Per-trajectory PIA dimension scores produced by PIAScorer.score_trajectory.

    Attributes:
        planning_quality: Score in [0, 1]; higher = more efficient plan.
        error_recovery: Score in [0, 1] when mid-trajectory escalation was
            detected; ``None`` when no escalation occurred.
        goal_alignment: Score in [0, 1]; mean of intent and outcome scores.
    """

    planning_quality: float
    error_recovery: float | None
    goal_alignment: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _bin(score: float) -> int:
    """Map a [0.0, 1.0] score to a discrete bin index in {0, 1, 2}."""
    return min(int(score * _N_BINS), _N_BINS - 1)


def _rating_matrix(r1: list[float], r2: list[float]) -> list[list[int]]:
    """Build an (n_items × 2) labels matrix for IRRCalculator.fleiss_kappa.

    Each row contains the bin index assigned by rater-1 and rater-2 to the
    same item.  The row format matches the ``[n_items × n_raters]`` layout
    that IRRCalculator.fleiss_kappa expects.

    Args:
        r1: Scores from rater 1; values in [0, 1].
        r2: Scores from rater 2; values in [0, 1].

    Returns:
        2-D list of shape [len(r1) × 2] with 0-indexed bin labels.
    """
    return [[_bin(s1), _bin(s2)] for s1, s2 in zip(r1, r2)]


# ---------------------------------------------------------------------------
# PIAScorer
# ---------------------------------------------------------------------------


class PIAScorer:
    """Entry point for Path-Invariant Agreement scoring.

    Wraps :class:`~src.eval.trajectory_scorer.TrajectoryScorer` to expose:

    * :meth:`score_trajectory` — map one trajectory into PIA dimension scores.
    * :meth:`compute_pia` — aggregate Fleiss' κ over a paired-rater batch.

    Args:
        trajectory_scorer: Optional pre-constructed scorer.  Defaults to
            ``TrajectoryScorer()`` (dry-run mode) when omitted.
    """

    def __init__(self, trajectory_scorer: _PIAScorable | None = None) -> None:
        self._scorer: _PIAScorable = trajectory_scorer or TrajectoryScorer()

    def score_trajectory(
        self, trajectory: WearableLog, *, dry_run: bool = True  # noqa: ARG002
    ) -> DimensionScores:
        """Score a single trajectory on the three PIA dimensions.

        ``dry_run`` is accepted for API consistency but has no effect here —
        dryness is governed by the :class:`~src.eval.trajectory_scorer.TrajectoryScorer`
        passed at construction time.

        Args:
            trajectory: Wearable log to evaluate.
            dry_run: Accepted but unused; controls scorer at construction time.

        Returns:
            :class:`DimensionScores` with ``error_recovery=None`` when no
            mid-trajectory escalation event was detected.
        """
        raw = self._scorer.score_pia_dimensions(trajectory)
        return DimensionScores(
            planning_quality=raw["planning_quality"],
            error_recovery=(
                None
                if raw["error_recovery"] == _SENTINEL_NO_ESCALATION
                else raw["error_recovery"]
            ),
            goal_alignment=raw["goal_alignment"],
        )

    def compute_pia(
        self,
        rater1_scores: list[DimensionScores],
        rater2_scores: list[DimensionScores],
    ) -> float:
        """Compute mean Fleiss' κ across PIA dimensions for a paired-rater batch.

        Each dimension is scored independently.  ``error_recovery`` pairs where
        either rater's value is ``None`` are excluded from that dimension; if no
        valid pairs exist the dimension is dropped from the mean.

        Args:
            rater1_scores: Dimension scores from the first rater.
            rater2_scores: Dimension scores from the second rater; must be the
                same length as *rater1_scores*.

        Returns:
            Mean Fleiss' κ in [-1, 1] across active dimensions.

        Raises:
            ValueError: If the two lists differ in length.
        """
        if len(rater1_scores) != len(rater2_scores):
            raise ValueError(
                f"rater lists must be the same length; "
                f"got {len(rater1_scores)} vs {len(rater2_scores)}"
            )
        if not rater1_scores:
            return 0.0

        irr = IRRCalculator()
        kappas: list[float] = []

        def _kappa(r1: list[float], r2: list[float]) -> float:
            result = irr.fleiss_kappa(_rating_matrix(r1, r2), n_categories=_N_BINS)
            return cast(float, result["kappa"])

        kappas.append(
            _kappa(
                [s.planning_quality for s in rater1_scores],
                [s.planning_quality for s in rater2_scores],
            )
        )

        kappas.append(
            _kappa(
                [s.goal_alignment for s in rater1_scores],
                [s.goal_alignment for s in rater2_scores],
            )
        )

        er_pairs: list[tuple[float, float]] = [
            (s1.error_recovery, s2.error_recovery)
            for s1, s2 in zip(rater1_scores, rater2_scores)
            if s1.error_recovery is not None and s2.error_recovery is not None
        ]
        if er_pairs:
            kappas.append(
                _kappa(
                    [p[0] for p in er_pairs],
                    [p[1] for p in er_pairs],
                )
            )

        return sum(kappas) / len(kappas)
