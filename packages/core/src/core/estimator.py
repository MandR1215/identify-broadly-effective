from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from .arrays import FloatArray, IntArray, as_float_array, as_mask


class CVaREstimator(Protocol):
    def estimate_cvar(
        self,
        values: np.ndarray,
        observed: np.ndarray,
    ) -> FloatArray:
        ...

    def estimate_best_candidate(
        self,
        values: np.ndarray,
        observed: np.ndarray,
    ) -> IntArray:
        ...


@dataclass(frozen=True)
class EmpiricalEstimator:
    """Empirical estimator.

    Shape convention:
        values[..., i, j] = value for sample i and candidate j
        observed[..., i, j] = whether values[..., i, j] is observed

    The last axis is the candidate axis.
    The second-to-last axis is the sample or participant axis.
    All leading axes are batch axes.
    """

    gamma: float

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be in (0, 1).")

    def estimate_cvar(
        self,
        values: np.ndarray,
        observed: np.ndarray,
    ) -> FloatArray:
        """Estimate lower-tail empirical CVaR along the sample axis.

        Args:
            values:
                Array with shape (..., n_samples, n_candidates).
            observed:
                Observation mask with the same shape as values.

        Returns:
            Array with shape (..., n_candidates).
        """
        x = as_float_array(values)

        if x.ndim < 2:
            raise ValueError("values must have at least two dimensions.")

        obs = as_mask(observed, x.shape)

        if np.any(~np.isfinite(x[obs])):
            raise ValueError("Observed values must be finite.")

        n_observed = obs.sum(axis=-2)

        if np.any(n_observed == 0):
            raise ValueError("Each candidate must have at least one observation.")

        masked_x = np.where(obs, x, np.inf)
        sorted_x = np.sort(masked_x, axis=-2)

        n_samples = x.shape[-2]

        rank_shape = (1,) * (x.ndim - 2) + (n_samples, 1)
        rank = np.arange(n_samples, dtype=np.int64).reshape(rank_shape)

        n_gamma = n_observed * self.gamma

        r = np.floor(n_gamma).astype(np.int64)
        lam = n_gamma - r

        r_expanded = np.expand_dims(r, axis=-2)
        lam_expanded = np.expand_dims(lam, axis=-2)

        weights = (rank < r_expanded).astype(np.float64)
        weights += (rank == r_expanded).astype(np.float64) * lam_expanded

        sorted_x_safe = np.where(np.isfinite(sorted_x), sorted_x, 0.0)

        numerator = np.sum(sorted_x_safe * weights, axis=-2)

        return (numerator / n_gamma).astype(np.float64)

    def estimate_best_candidate(
        self,
        values: np.ndarray,
        observed: np.ndarray,
    ) -> IntArray:
        cvar = self.estimate_cvar(values=values, observed=observed)
        return np.asarray(np.argmax(cvar, axis=-1), dtype=np.int64)

    def estimate_cvar_from_power_law(
        self,
        score: np.ndarray,
        candidate: np.ndarray,
        reward_coef: float,
        max_score: float,
        beta: float,
    ) -> FloatArray:
        """
        Estimate CVaR after recovering theta from power-law scores.

        This method is valid only under the non-binding condition.

        Formula:
            theta = clip_{[0, max_score]} { (tau * beta / |S_i|)^(-beta) * q^(1 - beta) }

        Args:
            score:
                Observed scores.
            candidate:
                Presented candidate mask.
            reward_coef:
                Reward coefficient tau.
            max_score:
                Maximum value of score.
            beta:
                Score exponent.

        Returns:
            Estimated CVaR for each candidate.
        """
        recovered = self._recover_power_law_ability(
            score=score,
            candidate=candidate,
            reward_coef=reward_coef,
            max_score=max_score,
            beta=beta
        )

        return self.estimate_cvar(
            values=recovered,
            observed=candidate,
        )

    def estimate_best_candidate_from_power_law(
        self,
        score: np.ndarray,
        candidate: np.ndarray,
        reward_coef: float,
        max_score: float,
        beta: float,
    ) -> IntArray:
        cvar = self.estimate_cvar_from_power_law(
            score=score,
            candidate=candidate,
            reward_coef=reward_coef,
            max_score=max_score,
            beta=beta
        )

        return np.asarray(np.argmax(cvar, axis=-1), dtype=np.int64)

    def _recover_power_law_ability(
        self,
        score: np.ndarray,
        candidate: np.ndarray,
        reward_coef: float,
        max_score: float,
        beta: float,
    ) -> FloatArray:
        """Recover theta from power-law scores under the non-binding condition.

        Shape convention:
            score[..., i, j] = observed score q_{i,j}
            candidate[..., i, j] = whether candidate j is presented to i

        Assumes:
            Each participant receives an average additive reward over the
            presented set S_i, and the effort constraint is non-binding.
        """
        if reward_coef <= 0:
            raise ValueError("reward_coef must be positive.")

        if max_score <= 0:
            raise ValueError("max_score must be positive.")

        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1).")

        q = as_float_array(score)

        if q.ndim < 2:
            raise ValueError("score must have at least two dimensions.")

        observed = as_mask(candidate, q.shape)

        if np.any(~np.isfinite(q[observed])):
            raise ValueError("Observed scores must be finite.")

        if np.any(q[observed] < 0):
            raise ValueError("Observed scores must be non-negative.")

        num_presented = observed.sum(axis=-1, keepdims=True)

        if np.any(num_presented == 0):
            raise ValueError("Each participant must be presented with at least one candidate.")

        coef = reward_coef * beta / num_presented

        recovered = np.power(coef, -beta) * np.power(q, 1.0 - beta)
        recovered = np.where(observed, recovered, 0.0)
        recovered = np.clip(recovered, 0.0, max_score)

        return recovered.astype(np.float64)