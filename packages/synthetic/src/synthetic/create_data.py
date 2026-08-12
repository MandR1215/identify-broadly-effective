from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from core.utils import current_timestamp

from .ability import AbilityDistribution, TruncatedNormal, TruncatedNormalMixture
from .save import write_config
from .settings import DISTRIBUTIONS, N, N_TRIALS, DATA_SEED

def build_distribution(config: dict) -> AbilityDistribution:
    if config["type"] == "truncated_normal":
        return TruncatedNormal(
            mu=config["mu"],
            sigma=config["sigma"],
            trunc_left=config["trunc_left"],
            trunc_right=config["trunc_right"],
        )

    if config["type"] == "truncated_normal_mixture":
        return TruncatedNormalMixture(
            weights=np.array(config["weights"], dtype=np.float64),
            mus=np.array(config["mus"], dtype=np.float64),
            sigmas=np.array(config["sigmas"], dtype=np.float64),
            trunc_lefts=np.array(config["trunc_lefts"], dtype=np.float64),
            trunc_rights=np.array(config["trunc_rights"], dtype=np.float64),
        )

    raise ValueError(f"Unknown distribution type: {config['type']}")


def sample_ability(
    distributions: list[AbilityDistribution],
    n: int,
    n_trials: int,
    random_state: np.random.Generator | int | None,
) -> np.ndarray:
    """Sample ability arrays with shape (n_trials, n, m=len(distributions))."""
    rng = np.random.default_rng(random_state)
    theta_columns = []

    for dist in distributions:
        theta_j = dist.sample(size=(n_trials, n), random_state=rng)
        theta_columns.append(theta_j)

    return np.stack(theta_columns, axis=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic ability data.")
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help=f"Number of participants per trial. Defaults to {N}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: outputs/synthetic/data/current_timestamp/.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n <= 0:
        raise ValueError("n must be positive.")

    distributions = [build_distribution(config) for config in DISTRIBUTIONS]
    ability = sample_ability(
        distributions=distributions,
        n=args.n,
        n_trials=N_TRIALS,
        random_state=DATA_SEED,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs") / "synthetic" / "data" / current_timestamp()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ability.npz"

    np.savez(output_path, ability=ability)

    write_config(
        output_dir / "config.json",
        {
            "seed": DATA_SEED,
            "n": args.n,
            "n_trials": N_TRIALS,
            "m": int(ability.shape[-1]),
            "shape": list(ability.shape),
            "distributions": DISTRIBUTIONS,
            "data_file": "ability.npz",
        },
    )

    print(f"Saved {output_path} with ability shape {ability.shape}")


if __name__ == "__main__":
    main()
