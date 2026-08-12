from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from core.estimator import EmpiricalEstimator
from core.mechanism import PowerLawMechanism, SingleMethodMechanism
from .oracle import build_oracle_rows, compute_reference_score_oracles
from .save import write_config, write_csv
from .score import LinearScoreGeneration, PowerLawScoreGeneration, ScoreGenerationFunction
from .settings import EXPERIMENT_SEED


ScoreGen = Literal["linear", "power_law"]


@dataclass(frozen=True)
class ExperimentConfig:
    score_gen: ScoreGen
    distribution_configs: tuple[dict, ...]
    gammas: tuple[float, ...]
    n: int
    m: int
    qmax: float
    budget: float
    seed: int
    n_trials: int
    beta: float
    beta_true: float | None = None
    metadata: dict = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        config = {
            "score_gen": self.score_gen,
            "gammas": list(self.gammas),
            "n": self.n,
            "m": self.m,
            "qmax": self.qmax,
            "budget": self.budget,
            "seed": self.seed,
            "n_trials": self.n_trials,
            "beta": self.beta,
            "metadata": self.metadata,
        }

        if self.beta_true is not None:
            config["beta_true"] = self.beta_true

        return config


def load_data(data_dir: Path) -> tuple[np.ndarray, dict]:
    with (data_dir / "config.json").open() as f:
        config = json.load(f)

    data_file = config.get("data_file", "ability.npz")

    with np.load(data_dir / data_file) as data:
        ability = np.asarray(data["ability"], dtype=np.float64)

    if ability.ndim != 3:
        raise ValueError(f"ability must have shape (n_trials, n, m), got {ability.shape}")

    return ability, config


def build_score_generation(config: ExperimentConfig) -> ScoreGenerationFunction:
    if config.score_gen == "linear":
        return LinearScoreGeneration()

    if config.beta_true is None:
        raise ValueError("beta_true is required when score_gen is power_law.")

    return PowerLawScoreGeneration(beta=config.beta_true)


def condition_rows(
    ability: np.ndarray,
    config: ExperimentConfig,
    data_config: dict,
    random_selected: np.ndarray,
    oracles: dict[float, dict],
) -> list[dict]:
    n_trials, n, m = ability.shape
    score_generation = build_score_generation(config)
    rows = []
    single_mechanism = SingleMethodMechanism()
    power_law_mechanism = PowerLawMechanism(beta=config.beta)

    single_tau = single_mechanism.reward_coef(
        num_participants=n,
        num_methods=m,
        budget=config.budget,
        max_score=config.qmax,
    )
    single_candidate = single_mechanism.assignment(size=ability.shape)
    single_score = score_generation.generate_score(
        reward_coef=single_tau,
        ability=ability,
        candidate=single_candidate,
    )

    power_law_tau = power_law_mechanism.reward_coef(
        num_participants=n,
        num_methods=m,
        budget=config.budget,
        max_score=config.qmax,
    )
    power_law_candidate = power_law_mechanism.assignment(size=ability.shape)
    power_law_score = score_generation.generate_score(
        reward_coef=power_law_tau,
        ability=ability,
        candidate=power_law_candidate,
    )

    for gamma in config.gammas:
        selected_by_method = {
            "random": random_selected,
            "single_assignment_score_cvar": EmpiricalEstimator(
                gamma=gamma,
            ).estimate_best_candidate(
                values=single_score,
                observed=single_candidate,
            ),
            "power_law_cvar": power_law_mechanism.estimate_best_candidate(
                score=power_law_score,
                gamma=gamma,
            ),
        }

        oracle = oracles[gamma]
        oracle_best = oracle["oracle_best"]
        oracle_cvar = oracle["oracle_cvar"]

        for method, selected in selected_by_method.items():
            rows.extend(
                {
                    **config.metadata,
                    "trial_id": trial_id,
                    "data_dir": data_config.get("data_dir", ""),
                    "score_gen": config.score_gen,
                    "budget": config.budget,
                    "beta": config.beta,
                    "beta_true": config.beta_true,
                    "gamma": gamma,
                    "method": method,
                    "selected": int(selected_j),
                    "oracle_best": oracle_best,
                    "correct": int(selected_j == oracle_best),
                    "regret": float(oracle_cvar[oracle_best] - oracle_cvar[selected_j]),
                }
                for trial_id, selected_j in enumerate(selected)
            )

    if n_trials != len(random_selected):
        raise ValueError("random_selected length must match n_trials.")

    return rows


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for row in rows:
        key = (
            row.get("sweep", ""),
            row.get("budget_scale", ""),
            row.get("fixed_gamma", ""),
            row.get("beta_true", ""),
            row["score_gen"],
            row["budget"],
            row["beta"],
            row["gamma"],
            row["method"],
        )
        groups[key].append(row)

    summary_rows = []

    for key, group in groups.items():
        regrets = np.asarray([row["regret"] for row in group], dtype=np.float64)
        correct = np.asarray([row["correct"] for row in group], dtype=np.float64)
        (
            sweep,
            budget_scale,
            fixed_gamma,
            beta_true,
            score_gen,
            budget,
            beta,
            gamma,
            method,
        ) = key
        summary_rows.append(
            {
                "sweep": sweep,
                "budget_scale": budget_scale,
                "fixed_gamma": fixed_gamma,
                "score_gen": score_gen,
                "budget": budget,
                "beta": beta,
                "beta_true": beta_true,
                "gamma": gamma,
                "method": method,
                "n_trials": len(group),
                "mean_regret": float(np.mean(regrets)),
                "std_regret": float(np.std(regrets)) if len(regrets) > 1 else 0.0,
                "correct_rate": float(np.mean(correct)),
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: (
            row["sweep"],
            row["budget_scale"],
            row["beta_true"],
            row["gamma"],
            row["method"],
        ),
    )


def run_sweep_conditions(
    configs: list[ExperimentConfig],
    data_dir: Path,
    output_dir: Path,
) -> Path:
    if not configs:
        raise ValueError("configs must not be empty.")

    ability, loaded_data_config = load_data(data_dir)
    data_config = {**loaded_data_config, "data_dir": str(data_dir)}
    n_trials, n, m = ability.shape
    rng = np.random.default_rng(EXPERIMENT_SEED)
    random_selected = rng.integers(low=0, high=m, size=n_trials)
    gammas = tuple(sorted({gamma for config in configs for gamma in config.gammas}))
    oracle_key_by_config = {
        index: (config.score_gen, config.beta_true)
        for index, config in enumerate(configs)
    }
    oracles_by_key = {
        key: compute_reference_score_oracles(
            distribution_configs=data_config["distributions"],
            score_generation=build_score_generation(config),
            gammas=gammas,
        )
        for index, config in enumerate(configs)
        for key in [oracle_key_by_config[index]]
    }
    rows = [
        row
        for index, config in enumerate(configs)
        for row in condition_rows(
            ability=ability,
            config=config,
            data_config=data_config,
            random_selected=random_selected,
            oracles=oracles_by_key[oracle_key_by_config[index]],
        )
    ]
    oracle_rows = [
        {**row, "score_gen": score_gen, "beta_true": beta_true}
        for (score_gen, beta_true), oracles in oracles_by_key.items()
        for row in build_oracle_rows(oracles)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trials.csv", rows)
    write_csv(output_dir / "summary.csv", summarize(rows))
    write_csv(output_dir / "sweep_summary.csv", summarize(rows))
    write_csv(output_dir / "oracle.csv", oracle_rows)
    write_config(
        output_dir / "config.json",
        {
            "sweep": configs[0].metadata.get("sweep", ""),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "n_trials": n_trials,
            "n": n,
            "m": m,
            "conditions": [config.to_json_dict() for config in configs],
            "data_config": data_config,
        },
    )
    write_config(
        output_dir / "sweep_config.json",
        {
            "sweep": configs[0].metadata.get("sweep", ""),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
        },
    )

    return output_dir
