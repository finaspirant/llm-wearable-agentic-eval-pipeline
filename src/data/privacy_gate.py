"""Differential privacy module for wearable sensor data.

Applies calibrated Gaussian noise to sensor readings before they
leave the device simulation. Implements the local DP model where
noise is added at the data source, not at the aggregator.

Default epsilon=1.0 provides moderate privacy protection while
preserving utility for downstream agent decision-making.

References:
- Dwork & Roth, "The Algorithmic Foundations of Differential Privacy"
- Apple's local DP implementation for iOS health data
- Gaussian mechanism: σ = Δf · √(2 ln(1.25/δ)) / ε
"""

import logging
import math
from enum import Enum

import numpy as np
from numpy.random import Generator

logger = logging.getLogger(__name__)

# Sensitivities calibrated per sensor type (max plausible change in one step)
_DEFAULT_SENSITIVITIES: dict[str, float] = {
    "heart_rate": 10.0,   # bpm — one inter-beat interval outlier
    "spo2": 2.0,          # % — SpO2 clamped 0-100
    "steps": 50.0,        # steps per epoch
    "gps_lat": 0.001,     # ~111 m per 0.001 degree
    "gps_lon": 0.001,
    "noise_db": 5.0,      # dB SPL
    "temperature": 0.5,   # °C skin-surface
    "confidence": 0.05,   # transcript confidence fraction
}


class ConsentModel(Enum):
    """Consent states for ambient/wearable data capture.

    Tracks whether the user has explicitly granted consent for a
    given data modality, so the pipeline can gate noise budgets
    and downstream annotation accordingly.
    """

    EXPLICIT = "explicit"    # User actively opted in, full pipeline allowed
    IMPLIED = "implied"      # Inferred from product ToS; limited annotation
    AMBIENT = "ambient"      # Background capture; strict privacy budget
    REVOKED = "revoked"      # User opted out; no capture or annotation


class PrivacyGate:
    """Local differential privacy gate for wearable sensor readings.

    Each call to :meth:`apply_gaussian_noise` consumes epsilon budget
    proportional to the operation's sensitivity.  Call
    :meth:`validate_epsilon_budget` before a batch of operations to
    ensure the cumulative cost stays within the configured budget.

    Args:
        epsilon: Global privacy budget (ε).  Smaller → more privacy,
            less utility.  Default 1.0 follows CLAUDE.md spec.
        delta: Failure probability for the (ε, δ)-DP guarantee.
            Default 1e-5 is standard for Gaussian mechanism.
        rng: Optional seeded NumPy random generator for reproducibility.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        rng: Generator | None = None,
    ) -> None:
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.epsilon = epsilon
        self.delta = delta
        self._rng: Generator = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Core DP primitives
    # ------------------------------------------------------------------

    def calibrate_noise(
        self,
        epsilon: float,
        delta: float,
        sensitivity: float,
    ) -> float:
        """Compute the Gaussian noise standard deviation σ.

        Uses the analytic Gaussian mechanism formula from Dwork & Roth:
            σ = Δf · √(2 · ln(1.25 / δ)) / ε

        Args:
            epsilon: Per-operation privacy budget.
            delta: Failure probability.
            sensitivity: L2 sensitivity of the query (Δf).

        Returns:
            σ — the standard deviation to use for the Gaussian perturbation.
        """
        sigma = sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
        logger.debug(
            "calibrate_noise: ε=%.4f δ=%.2e Δf=%.4f → σ=%.4f",
            epsilon,
            delta,
            sensitivity,
            sigma,
        )
        return sigma

    def apply_gaussian_noise(
        self,
        value: float,
        sensitivity: float,
        epsilon: float | None = None,
        delta: float | None = None,
    ) -> float:
        """Add calibrated Gaussian noise to a scalar sensor reading.

        Args:
            value: The raw sensor value to perturb.
            sensitivity: L2 sensitivity of the query (Δf).
            epsilon: Per-operation budget override; defaults to ``self.epsilon``.
            delta: Per-operation delta override; defaults to ``self.delta``.

        Returns:
            The noised scalar value.
        """
        eps = epsilon if epsilon is not None else self.epsilon
        dlt = delta if delta is not None else self.delta
        sigma = self.calibrate_noise(eps, dlt, sensitivity)
        noise = float(self._rng.normal(0.0, sigma))
        noised = value + noise
        logger.debug(
            "apply_gaussian_noise: raw=%.4f noise=%.4f → noised=%.4f",
            value,
            noise,
            noised,
        )
        return noised

    def apply_noise_to_sensor(
        self,
        sensor_name: str,
        value: float,
        epsilon: float | None = None,
    ) -> float:
        """Apply noise using a built-in per-sensor sensitivity lookup.

        Convenience wrapper around :meth:`apply_gaussian_noise` that
        uses :data:`_DEFAULT_SENSITIVITIES` to choose sensitivity
        automatically.

        Args:
            sensor_name: Key in ``_DEFAULT_SENSITIVITIES`` (e.g. ``"heart_rate"``).
            value: Raw sensor value.
            epsilon: Per-operation budget override.

        Returns:
            The noised value.

        Raises:
            KeyError: If ``sensor_name`` is not in the sensitivity table.
        """
        sensitivity = _DEFAULT_SENSITIVITIES[sensor_name]
        return self.apply_gaussian_noise(value, sensitivity, epsilon=epsilon)

    def validate_epsilon_budget(self, operations: list[float]) -> bool:
        """Check whether a list of per-operation epsilons stays within budget.

        Uses basic composition: total ε = Σ εᵢ.  Advanced composition
        (e.g. Rényi DP) is not implemented here.

        Args:
            operations: List of per-operation epsilon costs.

        Returns:
            ``True`` if the sum does not exceed ``self.epsilon``.
        """
        total = sum(operations)
        within_budget = total <= self.epsilon
        logger.debug(
            "validate_epsilon_budget: Σε=%.4f budget=%.4f within=%s",
            total,
            self.epsilon,
            within_budget,
        )
        return within_budget

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def sanitize_record(
        self,
        record: dict[str, float],
        consent: ConsentModel = ConsentModel.EXPLICIT,
    ) -> dict[str, float]:
        """Apply per-field Gaussian noise to a sensor record dict.

        Fields not present in ``_DEFAULT_SENSITIVITIES`` are passed
        through unchanged.  If consent is REVOKED the original record
        is returned without any processing (caller is responsible for
        not persisting it).

        Args:
            record: Dict mapping sensor name → raw float value.
            consent: Consent state governing this record.

        Returns:
            Dict with the same keys, values noised according to the
            configured (ε, δ) budget.
        """
        if consent == ConsentModel.REVOKED:
            logger.warning(
                "sanitize_record called with REVOKED consent — returning as-is"
            )
            return dict(record)

        sanitized: dict[str, float] = {}
        for key, val in record.items():
            if key in _DEFAULT_SENSITIVITIES:
                sanitized[key] = self.apply_noise_to_sensor(key, val)
            else:
                sanitized[key] = val
        return sanitized
