from __future__ import annotations

import argparse
from pathlib import Path

from core.utils import current_timestamp

from .sweep_config import run_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a preset sweep.")
    parser.add_argument(
        "sweep",
        choices=["linear-budget", "linear-gamma", "power-beta-true", "power-gamma"],
        default="linear-gamma",
        help="Sweep preset to run.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Data directory containing ability.npz and config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sweep output directory. Defaults under outputs/synthetic/sweep/.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir

    if output_dir is None:
        output_dir = Path("outputs") / "synthetic" / "sweep" / args.sweep / current_timestamp()

    output_dir = run_sweep(
        sweep=args.sweep,
        data_dir=args.data_dir,
        output_dir=output_dir,
    )
    print(f"Saved sweep results to: {output_dir}")


if __name__ == "__main__":
    main()
