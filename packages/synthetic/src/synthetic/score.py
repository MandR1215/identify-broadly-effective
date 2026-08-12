from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from core.arrays import FloatArray, as_mask, as_float_array


class ScoreGenerationFunction(Protocol):
    def generate_score(
        self,
        reward_coef: float,
        ability: np.ndarray,
        candidate: np.ndarray,
    ) -> FloatArray:
        ...

    def generate_score_given_effort(
        self,
        ability: np.ndarray | float,
        effort: np.ndarray | float,
    ) -> FloatArray:
        ...


@dataclass(frozen=True)
class LinearScoreGeneration:
    """
    Batched score generation function:

        g(ability, effort) = ability * effort

    Shape convention:
        ability[..., i, j]   = ability of i for candidate j
        candidate[..., i, j] = whether candidate j is presented to i

    The last axis is always the candidate axis.
    All leading axes are batch axes.
    """

    tie_break: Literal["split", "first"] = "split"

    def generate_score(
        self,
        reward_coef: float,
        ability: np.ndarray,
        candidate: np.ndarray,
    ) -> FloatArray:
        ability = as_float_array(ability)
        presented = as_mask(candidate, ability.shape)

        if reward_coef < 0:
            raise ValueError("reward_coef must be non-negative.")

        if np.any(ability < 0):
            raise ValueError("ability must be non-negative.")

        effort = self.generate_effort(
            reward_coef=reward_coef,
            ability=ability,
            candidate=presented,
        )

        return (ability * effort).astype(np.float64)

    def generate_score_given_effort(
        self,
        ability: np.ndarray | float,
        effort: np.ndarray | float,
    ) -> FloatArray:
        ability = np.asarray(ability, dtype=np.float64)
        effort = np.asarray(effort, dtype=np.float64)

        if np.any(ability < 0):
            raise ValueError("ability must be non-negative.")

        if np.any(effort < 0):
            raise ValueError("effort must be non-negative.")

        return np.asarray(ability * effort, dtype=np.float64)

    def generate_effort(
        self,
        reward_coef: float,
        ability: np.ndarray,
        candidate: np.ndarray,
    ) -> FloatArray:
        ability = as_float_array(ability)
        presented = as_mask(candidate, ability.shape)

        if reward_coef < 0:
            raise ValueError("reward_coef must be non-negative.")

        if np.any(ability < 0):
            raise ValueError("ability must be non-negative.")

        num_presented = presented.sum(axis=-1, keepdims=True)
        safe_num_presented = np.maximum(num_presented, 1)

        marginal_gain = reward_coef * ability / safe_num_presented - 1.0
        marginal_gain = np.where(presented, marginal_gain, -np.inf)

        best_gain = np.max(marginal_gain, axis=-1, keepdims=True)
        has_positive_best = best_gain > 0

        is_best = (marginal_gain == best_gain) & has_positive_best

        if self.tie_break == "split":
            num_best = np.maximum(is_best.sum(axis=-1, keepdims=True), 1)
            effort = is_best / num_best

        elif self.tie_break == "first":
            cumulative_best = np.cumsum(is_best, axis=-1)
            is_first_best = is_best & (cumulative_best == 1)
            effort = is_first_best.astype(np.float64)

        else:
            raise ValueError(f"Unknown tie_break: {self.tie_break}")

        effort = np.where(presented, effort, 0.0)

        return effort.astype(np.float64)
    

@dataclass(frozen=True)
class PowerLawScoreGeneration:
    """
    Batched power-law score generation function:

        g(ability, effort) = ability * effort^beta

    Shape convention:
        ability[..., i, j]   = ability of i for candidate j
        candidate[..., i, j] = whether candidate j is presented to i

    The last axis is always the candidate axis.
    All leading axes are batch axes.

    The effort is computed internally as the best response
    under the total effort constraint sum_j effort_j <= 1.
    """

    beta: float

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1).")

    def generate_score(
        self,
        reward_coef: float,
        ability: np.ndarray,
        candidate: np.ndarray,
    ) -> FloatArray:
        ability = as_float_array(ability)
        presented = as_mask(candidate, ability.shape)

        if reward_coef < 0:
            raise ValueError("reward_coef must be non-negative.")

        if np.any(ability < 0):
            raise ValueError("ability must be non-negative.")

        effort = self.generate_effort(
            reward_coef=reward_coef,
            ability=ability,
            candidate=presented,
        )

        score = ability * np.power(effort, self.beta)
        score = np.where(presented, score, 0.0)

        return score.astype(np.float64)

    def generate_score_given_effort(
        self,
        ability: np.ndarray | float,
        effort: np.ndarray | float,
    ) -> FloatArray:
        ability = np.asarray(ability, dtype=np.float64)
        effort = np.asarray(effort, dtype=np.float64)

        if np.any(ability < 0):
            raise ValueError("ability must be non-negative.")

        if np.any(effort < 0):
            raise ValueError("effort must be non-negative.")

        return np.asarray(ability * np.power(effort, self.beta), dtype=np.float64)

    def generate_effort(
        self,
        reward_coef: float,
        ability: np.ndarray,
        candidate: np.ndarray,
    ) -> FloatArray:
        ability = as_float_array(ability)
        presented = as_mask(candidate, ability.shape)

        if reward_coef < 0:
            raise ValueError("reward_coef must be non-negative.")

        if np.any(ability < 0):
            raise ValueError("ability must be non-negative.")

        num_presented = presented.sum(axis=-1, keepdims=True)
        safe_num_presented = np.maximum(num_presented, 1)

        # Unconstrained optimum:
        # effort_{i,j}^{unc} = (tau * beta * ability_{i,j} / |S_i|)^{1 / (1 - beta)}
        exponent = 1.0 / (1.0 - self.beta)

        base = reward_coef * self.beta * ability / safe_num_presented
        base = np.where(presented, base, 0.0)

        unconstrained_effort = np.power(base, exponent)

        # If sum_j effort_j^{unc} <= 1, the unconstrained optimum is feasible.
        # If not, the total effort constraint binds.
        total_unconstrained_effort = unconstrained_effort.sum(axis=-1, keepdims=True)

        safe_total = np.maximum(total_unconstrained_effort, 1.0)

        constrained_effort = unconstrained_effort / safe_total

        effort = np.where(
            total_unconstrained_effort <= 1.0,
            unconstrained_effort,
            constrained_effort,
        )

        effort = np.where(presented, effort, 0.0)

        return effort.astype(np.float64)
