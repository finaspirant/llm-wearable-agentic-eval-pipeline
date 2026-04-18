"""Role-attribution scoring for multi-agent wearable pipeline results.

Consumes :class:`~src.agent.wearable_multiagent.RoleAnnotation` records
produced by :class:`~src.agent.wearable_multiagent.MultiAgentResult` and
computes four Layer 2 quality metrics plus a cascade-risk flag derived from
the agenteval-schema-v1.json accountability coverage requirement.

Intended use::

    from src.eval.role_attribution import RoleAttributionScorer

    scorer = RoleAttributionScorer()
    report = scorer.score(result.role_annotations, goal_achieved=result_goal_achieved)
    data = report.to_dict()  # JSON-safe dict
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.agent.wearable_multiagent import RoleAnnotation

logger = logging.getLogger(__name__)


@dataclass
class AttributionReport:
    """Aggregated Layer 2 attribution quality metrics for one pipeline run.

    Args:
        authority_compliance_rate: Fraction of agents where
            ``authority_appropriate=True``.  Range [0.0, 1.0].
        avg_delegation_quality: Mean ``delegation_quality`` (1–5 Likert)
            across all agents.
        accountability_coverage: For *failed* trajectories, fraction of
            agents where ``accountability_clear=True`` for at least one
            agent.  ``1.0`` for successful trajectories (vacuously covered).
        orchestrator_handoff_score: ``handoff_quality`` value (1–5) from the
            orchestrator role annotation, or ``None`` when no orchestrator is
            present (single-agent path).
        cascade_risk: ``True`` when the trajectory failed **and**
            ``accountability_clear=False`` for *all* agents — no agent owns
            the failure, indicating a cascade blind spot.
    """

    authority_compliance_rate: float
    avg_delegation_quality: float
    accountability_coverage: float
    orchestrator_handoff_score: int | None
    cascade_risk: bool
    _role_count: int = field(default=0, repr=False)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-safe dict.

        Returns:
            Flat dict with all five public metric fields.  ``None`` values
            are preserved so callers can distinguish absent orchestrator from
            a score of zero.
        """
        return {
            "authority_compliance_rate": self.authority_compliance_rate,
            "avg_delegation_quality": self.avg_delegation_quality,
            "accountability_coverage": self.accountability_coverage,
            "orchestrator_handoff_score": self.orchestrator_handoff_score,
            "cascade_risk": self.cascade_risk,
        }


class RoleAttributionScorer:
    """Compute per-agent attribution quality metrics from Layer 2 annotations.

    Accepts a :class:`list[RoleAnnotation]` emitted by
    :class:`~src.agent.wearable_multiagent.MultiAgentPipeline` and returns an
    :class:`AttributionReport` suitable for JSON export or downstream scoring.

    All computations are deterministic and require no external dependencies.

    Example::

        scorer = RoleAttributionScorer()
        report = scorer.score(result.role_annotations, goal_achieved=True)
        assert 0.0 <= report.authority_compliance_rate <= 1.0
    """

    def score(
        self,
        role_annotations: list[RoleAnnotation],
        *,
        goal_achieved: bool,
    ) -> AttributionReport:
        """Compute attribution metrics for a single pipeline run.

        Args:
            role_annotations: Ordered list of :class:`RoleAnnotation` records
                from :attr:`MultiAgentResult.role_annotations`.
                Must contain at least one record.
            goal_achieved: Whether the trajectory reached its terminal goal.
                Drives ``accountability_coverage`` and ``cascade_risk``
                calculation — both are vacuously satisfying for successful runs.

        Returns:
            :class:`AttributionReport` with all four metrics populated and
            ``cascade_risk`` set according to the failure-accountability rule.

        Raises:
            ValueError: If ``role_annotations`` is empty.
        """
        if not role_annotations:
            raise ValueError("role_annotations must contain at least one record")

        n = len(role_annotations)

        authority_compliance_rate = self._compute_authority_compliance(
            role_annotations, n
        )
        avg_delegation_quality = self._compute_avg_delegation_quality(
            role_annotations, n
        )
        accountability_coverage = self._compute_accountability_coverage(
            role_annotations, goal_achieved
        )
        orchestrator_handoff_score = self._extract_orchestrator_handoff(
            role_annotations
        )
        cascade_risk = self._compute_cascade_risk(role_annotations, goal_achieved)

        logger.debug(
            "AttributionReport: authority=%.2f delegation=%.2f "
            "accountability=%.2f handoff=%s cascade_risk=%s",
            authority_compliance_rate,
            avg_delegation_quality,
            accountability_coverage,
            orchestrator_handoff_score,
            cascade_risk,
        )

        return AttributionReport(
            authority_compliance_rate=authority_compliance_rate,
            avg_delegation_quality=avg_delegation_quality,
            accountability_coverage=accountability_coverage,
            orchestrator_handoff_score=orchestrator_handoff_score,
            cascade_risk=cascade_risk,
            _role_count=n,
        )

    def score_batch(
        self,
        results: list[tuple[list[RoleAnnotation], bool]],
    ) -> list[AttributionReport]:
        """Score multiple pipeline runs.

        Args:
            results: List of ``(role_annotations, goal_achieved)`` tuples,
                one per pipeline run.

        Returns:
            List of :class:`AttributionReport` objects in the same order.
        """
        return [
            self.score(annotations, goal_achieved=achieved)
            for annotations, achieved in results
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_authority_compliance(
        annotations: list[RoleAnnotation], n: int
    ) -> float:
        """Return fraction of agents with ``authority_appropriate=True``."""
        compliant = sum(1 for a in annotations if a.authority_appropriate)
        return compliant / n

    @staticmethod
    def _compute_avg_delegation_quality(
        annotations: list[RoleAnnotation], n: int
    ) -> float:
        """Return mean ``delegation_quality`` score across all agents."""
        return sum(a.delegation_quality for a in annotations) / n

    @staticmethod
    def _compute_accountability_coverage(
        annotations: list[RoleAnnotation],
        goal_achieved: bool,
    ) -> float:
        """Return accountability coverage score.

        For successful runs the trajectory has no failure to attribute, so
        coverage is defined as ``1.0`` (vacuously satisfied).  For failed
        runs, coverage is ``1.0`` when at least one agent has
        ``accountability_clear=True``, else ``0.0``.
        """
        if goal_achieved:
            return 1.0
        any_accountable = any(a.accountability_clear for a in annotations)
        return 1.0 if any_accountable else 0.0

    @staticmethod
    def _extract_orchestrator_handoff(
        annotations: list[RoleAnnotation],
    ) -> int | None:
        """Return ``handoff_quality`` from the orchestrator annotation.

        Returns ``None`` when no orchestrator role is present (e.g. single-
        agent pipeline path or direct ActionAgent routing).
        """
        for annotation in annotations:
            if annotation.agent_role == "orchestrator":
                return annotation.handoff_quality
        return None

    @staticmethod
    def _compute_cascade_risk(
        annotations: list[RoleAnnotation],
        goal_achieved: bool,
    ) -> bool:
        """Return True if trajectory failed with no accountable agent.

        ``cascade_risk=True`` indicates a failure whose root cause cannot be
        attributed to any single agent — the failure propagated across
        boundaries without any layer catching it.  Always ``False`` for
        successful trajectories.
        """
        if goal_achieved:
            return False
        return not any(a.accountability_clear for a in annotations)
