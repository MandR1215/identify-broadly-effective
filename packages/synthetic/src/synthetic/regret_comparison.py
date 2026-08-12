from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Literal

import numpy as np
from core.mechanism import PowerLawMechanism, SingleMethodMechanism
from core.utils import current_timestamp

from .oracle import build_oracle_rows, compute_reference_score_oracles
from .save import write_config, write_csv
from .score import LinearScoreGeneration, PowerLawScoreGeneration
from .sweep_experiment import load_data
from .settings import (
    BETA,
    BETA_TRUE,
    BUDGET_SCALES,
    C_CVAR,
    DELTAS,
    FIXED_GAMMA,
    N,
    QMAX,
    REGRET_COMPARISON_BUDGET_SCALE,
)


ScoreGen = Literal["linear", "power_law"]


def imputation_error(
    beta_true: float,
    beta: float,
    tau: float,
    qmax: float,
    m: int,
    grid_size: int = 20_001,
) -> float:
    theta = np.linspace(0.0, qmax, grid_size)
    true_score = (
        (tau * beta_true / m) ** (beta_true / (1.0 - beta_true))
        * np.power(theta, 1.0 / (1.0 - beta_true))
    )
    recovered = (
        (tau * beta / m) ** (-beta)
        * np.power(true_score, 1.0 - beta)
    )

    return float(np.max(np.abs(np.clip(recovered, 0.0, qmax) - theta)))


def linear_theory_bound(
    n: int,
    m: int,
    budget: float,
    qmax: float,
    delta: float,
    gamma: float,
) -> float:
    return (
        2.0 * n * qmax / budget
        + 2.0
        * C_CVAR
        * qmax
        * gamma
        * math.sqrt(math.log(2.0 * m / delta) / math.floor(n / m))
    )


def power_law_theory_bound(
    beta_true: float,
    beta: float,
    tau: float,
    qmax: float,
    gamma: float,
    n: int,
    m: int,
    delta: float,
) -> float:
    return (
        2.0 * imputation_error(
            beta_true=beta_true,
            beta=beta,
            tau=tau,
            qmax=qmax,
            m=m,
        )
        + 2.0
        * C_CVAR
        * qmax
        / gamma
        * math.sqrt(math.log(2.0 * m / delta) / n)
    )


def run_trials(
    ability: np.ndarray,
    score_gen: ScoreGen,
    budget: float,
    gamma: float,
    oracles: dict[float, dict],
) -> tuple[list[dict], dict]:
    n_trials, n, m = ability.shape

    if score_gen == "linear":
        mechanism = SingleMethodMechanism()
        score_function = LinearScoreGeneration()
        beta = None
        beta_true = None
    elif score_gen == "power_law":
        mechanism = PowerLawMechanism(beta=BETA)
        score_function = PowerLawScoreGeneration(beta=BETA_TRUE)
        beta = BETA
        beta_true = BETA_TRUE
    else:
        raise ValueError(f"Unknown score_gen: {score_gen}")

    tau = mechanism.reward_coef(
        num_participants=n,
        num_methods=m,
        budget=budget,
        max_score=QMAX,
    )
    candidate = mechanism.assignment(size=ability.shape)
    score = score_function.generate_score(
        reward_coef=tau,
        ability=ability,
        candidate=candidate,
    )
    rows = []

    selected = mechanism.estimate_best_candidate(score=score, gamma=gamma)
    oracle = oracles[gamma]
    oracle_best = oracle["oracle_best"]
    oracle_cvar = oracle["oracle_cvar"]
    rows.extend(
        {
            "trial_id": trial_id,
            "gamma": gamma,
            "selected": int(selected_j),
            "oracle_best": oracle_best,
            "regret": float(oracle_cvar[oracle_best] - oracle_cvar[selected_j]),
        }
        for trial_id, selected_j in enumerate(selected)
    )

    return rows, {
        "score_gen": score_gen,
        "budget": budget,
        "budget_scale": budget / (n * QMAX),
        "tau": tau,
        "beta": beta,
        "beta_true": beta_true,
        "n_trials": n_trials,
        "n": n,
        "m": m,
    }


def build_score_function(score_gen: ScoreGen) -> LinearScoreGeneration | PowerLawScoreGeneration:
    if score_gen == "linear":
        return LinearScoreGeneration()

    if score_gen == "power_law":
        return PowerLawScoreGeneration(beta=BETA_TRUE)

    raise ValueError(f"Unknown score_gen: {score_gen}")


def theory_bound(score_gen: ScoreGen, gamma: float, delta: float, metadata: dict) -> float:
    if score_gen == "linear":
        return linear_theory_bound(
            n=metadata["n"],
            m=metadata["m"],
            budget=metadata["budget"],
            qmax=QMAX,
            delta=delta,
            gamma=gamma,
        )

    return power_law_theory_bound(
        beta_true=metadata["beta_true"],
        beta=metadata["beta"],
        tau=metadata["tau"],
        qmax=QMAX,
        gamma=gamma,
        n=metadata["n"],
        m=metadata["m"],
        delta=delta,
    )


def summarize(
    trial_rows: list[dict],
    metadata: dict,
    x_name: str,
    x_value: float,
    delta: float,
) -> list[dict]:
    rows = []

    gamma = trial_rows[0]["gamma"]
    regrets = np.asarray([row["regret"] for row in trial_rows], dtype=np.float64)
    bound = theory_bound(
        score_gen=metadata["score_gen"],
        gamma=gamma,
        delta=delta,
        metadata=metadata,
    )
    actual_quantile = float(np.quantile(regrets, 1.0 - delta))

    rows.append(
        {
            "score_gen": metadata["score_gen"],
            "sweep": x_name,
            x_name: x_value,
            "gamma": gamma,
            "delta": delta,
            "n": metadata["n"],
            "budget": metadata["budget"],
            "budget_scale": metadata["budget_scale"],
            "tau": metadata["tau"],
            "beta_true": metadata["beta_true"],
            "beta": metadata["beta"],
            "n_trials": metadata["n_trials"],
            "actual_quantile_regret": actual_quantile,
            "theory_bound": bound,
        }
    )

    return rows


def add_context(
    rows: list[dict],
    metadata: dict,
    x_name: str,
    x_value: float,
) -> list[dict]:
    return [
        {
            "score_gen": metadata["score_gen"],
            "sweep": x_name,
            x_name: x_value,
            "n": metadata["n"],
            "budget": metadata["budget"],
            "budget_scale": metadata["budget_scale"],
            "tau": metadata["tau"],
            "beta_true": metadata["beta_true"],
            "beta": metadata["beta"],
            **row,
        }
        for row in rows
    ]


def run_one_point(
    data_dir: Path,
    score_gen: ScoreGen,
    budget: float,
    x_name: str,
    x_value: float,
    delta: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    ability, data_config = load_data(data_dir)
    gamma = FIXED_GAMMA
    oracles = compute_reference_score_oracles(
        distribution_configs=data_config["distributions"],
        score_generation=build_score_function(score_gen),
        gammas=(gamma,),
    )
    trial_rows, metadata = run_trials(
        ability=ability,
        score_gen=score_gen,
        budget=budget,
        gamma=gamma,
        oracles=oracles,
    )

    return (
        add_context(
            rows=trial_rows,
            metadata=metadata,
            x_name=x_name,
            x_value=x_value,
        ),
        summarize(
            trial_rows=trial_rows,
            metadata=metadata,
            x_name=x_name,
            x_value=x_value,
            delta=delta,
        ),
        build_oracle_rows(oracles),
    )


def run_regret_comparison(
    score_gen: ScoreGen,
    data_dirs: list[Path],
    output_dir: Path,
) -> Path:
    if not data_dirs:
        raise ValueError("data_dirs must not be empty.")

    delta = DELTAS[0]
    trial_rows = []
    summary_rows = []
    oracle_rows = []

    if len(data_dirs) > 1:
        sweep = "n"
        fixed_budget = REGRET_COMPARISON_BUDGET_SCALE * N * QMAX

        for data_dir in data_dirs:
            ability, _ = load_data(data_dir)
            n = ability.shape[1]
            trials, summary, oracle = run_one_point(
                data_dir=data_dir,
                score_gen=score_gen,
                budget=fixed_budget,
                x_name=sweep,
                x_value=n,
                delta=delta,
            )
            trial_rows.extend(trials)
            summary_rows.extend(summary)
            oracle_rows = oracle
    else:
        sweep = "budget_scale"
        data_dir = data_dirs[0]
        ability, _ = load_data(data_dir)
        n = ability.shape[1]

        for budget_scale in BUDGET_SCALES:
            trials, summary, oracle = run_one_point(
                data_dir=data_dir,
                score_gen=score_gen,
                budget=budget_scale * n * QMAX,
                x_name=sweep,
                x_value=budget_scale,
                delta=delta,
            )
            trial_rows.extend(trials)
            summary_rows.extend(summary)
            oracle_rows = oracle

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trials.csv", trial_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "oracle.csv", oracle_rows)
    write_config(
        output_dir / "config.json",
        {
            "comparison": "regret_bound",
            "score_gen": score_gen,
            "sweep": sweep,
            "data_dirs": [str(path) for path in data_dirs],
            "output_dir": str(output_dir),
            "gamma": FIXED_GAMMA,
            "delta": delta,
            "qmax": QMAX
        },
    )

    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare empirical regret quantiles and theory bounds.")
    parser.add_argument("score_gen", choices=["linear", "power_law"])
    parser.add_argument(
        "--data-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One data directory for budget-scale sweep, or multiple data "
            "directories for n sweep with fixed budget."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults under outputs/synthetic/regret-comparison/.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = (
            Path("outputs")
            / "synthetic"
            / "regret-comparison"
            / args.score_gen
            / current_timestamp()
        )

    output_dir = run_regret_comparison(
        score_gen=args.score_gen,
        data_dirs=args.data_dir,
        output_dir=output_dir,
    )
    print(f"Saved regret comparison to: {output_dir}")


if __name__ == "__main__":
    main()
