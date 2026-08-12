from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .allocator import Allocator, Balanced, AllAssignment
from .arrays import FloatArray, IntArray
from .estimator import EmpiricalEstimator


class Mechanism(Protocol):
    def reward_coef(
        self,
        num_participants: int,
        num_methods: int,
        budget: float,
        max_score: float,
    ) -> float:
        ...

    def assignment(self, size: tuple[int, ...]) -> FloatArray:
        ...

    def estimate_cvar(self, score: np.ndarray, gamma: float) -> FloatArray:
        ...

    def estimate_best_candidate(self, score: np.ndarray, gamma: float) -> IntArray:
        ...


def _validate_design_inputs(
    num_participants: int,
    num_methods: int,
    budget: float,
    max_score: float,
) -> None:
    if num_participants <= 0:
        raise ValueError("num_participants must be positive.")
    if num_methods <= 0:
        raise ValueError("num_methods must be positive.")
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if max_score <= 0:
        raise ValueError("max_score must be positive.")


def _c_m(num_methods: int) -> float:
    if num_methods <= 0:
        raise ValueError("num_methods must be positive.")
    if num_methods == 1:
        return 1.0
    if num_methods == 2:
        return 2.0
    return float(np.e * np.log(num_methods))


@dataclass
class SingleMethodMechanism:
    allocator: Allocator = field(default_factory=Balanced)

    def reward_coef(
        self,
        num_participants: int,
        num_methods: int,
        budget: float,
        max_score: float,
    ) -> float:
        _validate_design_inputs(num_participants, num_methods, budget, max_score)
        return budget / (num_participants * max_score)

    def assignment(self, size: tuple[int, ...]) -> FloatArray:
        return self.allocator.allocate(size=size)

    def estimate_cvar(self, score: np.ndarray, gamma: float) -> FloatArray:
        return EmpiricalEstimator(gamma=gamma).estimate_cvar(
            values=score,
            observed=self.assignment(size=score.shape),
        )

    def estimate_best_candidate(self, score: np.ndarray, gamma: float) -> IntArray:
        return EmpiricalEstimator(gamma=gamma).estimate_best_candidate(
            values=score,
            observed=self.assignment(size=score.shape),
        )


@dataclass
class PowerLawMechanism:
    beta: float = 1 / 2
    allocator: Allocator = field(default_factory=AllAssignment)
    tau: float | None = field(default=None, init=False)
    max_score: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError("beta must be in (0, 1).")

    def reward_coef(
        self,
        num_participants: int,
        num_methods: int,
        budget: float,
        max_score: float,
    ) -> float:
        _validate_design_inputs(num_participants, num_methods, budget, max_score)

        c = _c_m(num_methods)
        budget_bound = budget / (num_participants * max_score)
        nonbinding_bound = c / max_score

        self.tau = min(budget_bound, nonbinding_bound)
        self.max_score = max_score

        return self.tau

    def assignment(self, size: tuple[int, ...]) -> FloatArray:
        return self.allocator.allocate(size=size)

    def _state_or_raise(self) -> tuple[float, float]:
        if self.tau is None or self.max_score is None:
            raise RuntimeError("reward_coef must be called before estimation.")
        return self.tau, self.max_score

    def estimate_cvar(self, score: np.ndarray, gamma: float) -> FloatArray:
        tau, max_score = self._state_or_raise()
        return EmpiricalEstimator(gamma=gamma).estimate_cvar_from_power_law(
            score=score,
            candidate=self.assignment(size=score.shape),
            reward_coef=tau,
            max_score=max_score,
            beta=self.beta,
        )

    def estimate_best_candidate(self, score: np.ndarray, gamma: float) -> IntArray:
        tau, max_score = self._state_or_raise()
        return EmpiricalEstimator(gamma=gamma).estimate_best_candidate_from_power_law(
            score=score,
            candidate=self.assignment(size=score.shape),
            reward_coef=tau,
            max_score=max_score,
            beta=self.beta,
        )
