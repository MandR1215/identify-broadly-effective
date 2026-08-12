from __future__ import annotations

from pathlib import Path
from typing import Literal

from .sweep_experiment import (
    ExperimentConfig,
    run_sweep_conditions,
)
from .settings import (
    BETA,
    BETA_TRUE,
    BETA_TRUE_VALUES,
    BUDGET_SCALES,
    DISTRIBUTIONS,
    EXPERIMENT_SEED,
    FIXED_GAMMA,
    GAMMAS,
    LINEAR_GAMMA_BUDGET_SCALE,
    M,
    N,
    N_TRIALS,
    POWER_BUDGET_SCALE,
    QMAX,
)


SweepName = Literal[
    "linear-budget",
    "linear-gamma",
    "power-beta-true",
    "power-gamma",
]


def linear_config(
    budget: float,
    gammas: tuple[float, ...],
    metadata: dict,
) -> ExperimentConfig:
    return ExperimentConfig(
        score_gen="linear",
        distribution_configs=tuple(DISTRIBUTIONS),
        gammas=gammas,
        n=N,
        m=M,
        qmax=QMAX,
        budget=budget,
        seed=EXPERIMENT_SEED,
        n_trials=N_TRIALS,
        beta=BETA,
        metadata=metadata,
    )


def power_config(
    beta_true: float,
    budget: float,
    gammas: tuple[float, ...],
    metadata: dict,
) -> ExperimentConfig:
    return ExperimentConfig(
        score_gen="power_law",
        distribution_configs=tuple(DISTRIBUTIONS),
        gammas=gammas,
        n=N,
        m=M,
        qmax=QMAX,
        budget=budget,
        seed=EXPERIMENT_SEED,
        n_trials=N_TRIALS,
        beta_true=beta_true,
        beta=BETA,
        metadata=metadata,
    )


def build_sweep_configs(
    sweep: SweepName,
) -> list[ExperimentConfig]:
    if sweep == "linear-budget":
        return [
            linear_config(
                budget=k * N * QMAX,
                gammas=(FIXED_GAMMA,),
                metadata={"sweep": sweep, "budget_scale": k, "fixed_gamma": FIXED_GAMMA},
            )
            for k in BUDGET_SCALES
        ]

    if sweep == "linear-gamma":
        return [
            linear_config(
                budget=LINEAR_GAMMA_BUDGET_SCALE * N * QMAX,
                gammas=GAMMAS,
                metadata={"sweep": sweep, "budget_scale": LINEAR_GAMMA_BUDGET_SCALE},
            )
        ]

    if sweep == "power-beta-true":
        return [
            power_config(
                beta_true=beta_true,
                budget=POWER_BUDGET_SCALE * N * QMAX,
                gammas=(FIXED_GAMMA,),
                metadata={
                    "sweep": sweep,
                    "beta_true": beta_true,
                    "beta": BETA,
                    "fixed_gamma": FIXED_GAMMA,
                    "budget_scale": POWER_BUDGET_SCALE,
                },
            )
            for beta_true in BETA_TRUE_VALUES
        ]

    if sweep == "power-gamma":
        return [
            power_config(
                beta_true=BETA_TRUE,
                budget=POWER_BUDGET_SCALE * N * QMAX,
                gammas=GAMMAS,
                metadata={
                    "sweep": sweep,
                    "beta_true": BETA_TRUE,
                    "beta": BETA,
                    "budget_scale": POWER_BUDGET_SCALE,
                },
            )
        ]

    raise ValueError(f"Unknown sweep: {sweep}")


def run_sweep(
    sweep: SweepName,
    data_dir: Path,
    output_dir: Path,
) -> Path:
    configs = build_sweep_configs(
        sweep=sweep,
    )

    return run_sweep_conditions(configs=configs, data_dir=data_dir, output_dir=output_dir)
