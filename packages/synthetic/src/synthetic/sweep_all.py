from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from core.utils import current_timestamp

from .plot import _configure_matplotlib_fonts, plot_sweep
from .sweep_config import SweepName, run_sweep


SWEEPS: tuple[SweepName, ...] = (
    "linear-budget",
    "linear-gamma",
    "power-beta-true",
    "power-gamma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all synthetic sweeps and plots.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory containing ability.npz and config.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "synthetic" / "sweep",
        help="Root directory for sweep outputs.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Save with timestamp.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Replot existing latest sweep outputs without running sweeps.",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        choices=SWEEPS,
        default=list(SWEEPS),
        help="Sweep presets to run.",
    )

    return parser.parse_args()


def plot_latest_outputs(output_root: Path, sweeps: list[str]) -> None:
    latest_dirs = [
        output_root / cast(SweepName, sweep_name) / "latest"
        for sweep_name in sweeps
        if (output_root / cast(SweepName, sweep_name) / "latest" / "sweep_config.json").exists()
    ]

    if not latest_dirs:
        print(f"No latest sweep outputs found under {output_root}. Nothing to plot.")
        return

    _configure_matplotlib_fonts()

    for sweep_dir in latest_dirs:
        plot_dir = plot_sweep(sweep_dir)
        print(f"Saved {sweep_dir.parent.name} plots to: {plot_dir}")


def main() -> None:
    args = parse_args()

    if args.plot_only:
        plot_latest_outputs(
            output_root=args.output_root,
            sweeps=args.sweeps,
        )
        return

    if args.data_dir is None:
        raise SystemExit("--data-dir is required unless --plot-only is specified.")

    timestamp = current_timestamp() if args.timestamp else "latest"

    _configure_matplotlib_fonts()

    for sweep_name in args.sweeps:
        sweep = cast(SweepName, sweep_name)
        output_dir = args.output_root / sweep / timestamp
        print(f"Running {sweep}: {output_dir}")
        run_dir = run_sweep(
            sweep=sweep,
            data_dir=args.data_dir,
            output_dir=output_dir,
        )
        plot_dir = plot_sweep(run_dir)
        print(f"Saved {sweep} plots to: {plot_dir}")


if __name__ == "__main__":
    main()
