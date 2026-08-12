from __future__ import annotations

import numpy as np

from .create_data import build_distribution
from .score import (
    LinearScoreGeneration,
    PowerLawScoreGeneration,
    ScoreGenerationFunction,
)


def compute_reference_score_oracles(
    distribution_configs: list[dict],
    score_generation: ScoreGenerationFunction,
    gammas: tuple[float, ...],
    reference_effort: float = 1.0,
) -> dict[float, dict]:
    if not isinstance(score_generation, (LinearScoreGeneration, PowerLawScoreGeneration)):
        raise TypeError(
            "compute_reference_score_oracles currently supports "
            "LinearScoreGeneration and PowerLawScoreGeneration."
        )

    distributions = [build_distribution(config) for config in distribution_configs]
    ability_means = np.asarray([dist.mean() for dist in distributions], dtype=np.float64)
    oracle_means = np.asarray(
        score_generation.generate_score_given_effort(
            ability=ability_means,
            effort=reference_effort,
        ),
        dtype=np.float64,
    )
    oracles = {}

    for gamma in gammas:
        ability_cvars = np.asarray(
            [dist.cvar(gamma) for dist in distributions],
            dtype=np.float64,
        )
        oracle_cvars = np.asarray(
            score_generation.generate_score_given_effort(
                ability=ability_cvars,
                effort=reference_effort,
            ),
            dtype=np.float64,
        )
        oracles[gamma] = {
            "oracle_cvar": oracle_cvars,
            "oracle_mean": oracle_means,
            "oracle_best": int(np.argmax(oracle_cvars)),
            "oracle_mean_best": int(np.argmax(oracle_means)),
            "reference_effort": reference_effort,
        }

    return oracles


def build_oracle_rows(oracles: dict[float, dict]) -> list[dict]:
    rows = []

    for gamma, oracle in oracles.items():
        for candidate, (cvar, mean) in enumerate(
            zip(oracle["oracle_cvar"], oracle["oracle_mean"])
        ):
            rows.append(
                {
                    "gamma": gamma,
                    "candidate": candidate,
                    "reference_effort": oracle.get("reference_effort", ""),
                    "oracle_cvar": float(cvar),
                    "oracle_mean": float(mean),
                    "is_oracle_best": int(candidate == oracle["oracle_best"]),
                    "is_oracle_mean_best": int(candidate == oracle["oracle_mean_best"]),
                }
            )

    return rows
