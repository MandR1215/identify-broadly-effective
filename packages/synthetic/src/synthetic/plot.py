from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.container import BarContainer


class PlotStyle(TypedDict):
    color: Any
    marker: str
    linestyle: Any


LEGEND_KWARGS = {
    "loc": "upper left",
    "bbox_to_anchor": (1.02, 1.0),
    "borderaxespad": 0.0,
    "frameon": False,
}

SERIF_FONT_FALLBACKS = [
    "Times New Roman",
    "Times",
    "STIXGeneral",
    "DejaVu Serif",
]


FIGURE_SIZE = (6.4, 3.8)
AXES_RECT = (0.12, 0.16, 0.56, 0.76)
SAVE_PADDING = 0.08

FONT_SIZE = 20
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20
LEGEND_FONT_SIZE = 20

LINE_WIDTH = 1.8
MARKER_SIZE = 5.0
FILL_ALPHA = 0.18

DENSITY_GRID_SIZE = 600
HIST_BINS = 50


METHOD_ORDER = [
    "single_assignment_score_cvar",
    "power_law_cvar",
    "random",
]

METHOD_LABELS = {
    "single_assignment_score_cvar": r"Single-Method Assignment",
    "power_law_cvar": "All-Methods Assignment",
    "random": "Random",
}

METHOD_STYLES: dict[str, PlotStyle] = {
    "random": {
        "color": "#7f7f7f",
        "marker": "x",
        "linestyle": (0, (3, 1, 1, 1)),
    },
    "single_assignment_score_cvar": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "power_law_cvar": {
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
    },
}

EXCLUDED_METHODS = {"all_assignment_score_cvar"}

CANDIDATE_STYLES: list[PlotStyle] = [
    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    {"color": "#d62728", "marker": "D", "linestyle": ":"},
    {"color": "#9467bd", "marker": "v", "linestyle": (0, (3, 1, 1, 1))},
    {"color": "#8c564b", "marker": "P", "linestyle": (0, (5, 2))},
]

SAMPLE_ALPHA = 0.22
SAMPLE_HATCHES = ["///", "\\\\\\", "xxx", "...", "+++", "ooo"]

REGRET_BOUND_STYLES: dict[str, PlotStyle] = {
    "actual_quantile_regret": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "theory_bound": {
        "color": "#d62728",
        "marker": "^",
        "linestyle": "--",
    },
}

REGRET_BOUND_LABELS = {
    "actual_quantile_regret": r"Empirical $(1-\delta)$-quantile",
    "theory_bound": "Theory bound",
}


def _configure_matplotlib_fonts() -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = next(
        (font for font in SERIF_FONT_FALLBACKS if font in available_fonts),
        "DejaVu Serif",
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [font_family, *SERIF_FONT_FALLBACKS],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.figsize": FIGURE_SIZE,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _new_figure() -> tuple[Figure, Axes]:
    fig = plt.figure(figsize=FIGURE_SIZE)
    ax = fig.add_axes(AXES_RECT)
    return fig, ax


def _save_pdf(
    fig: Figure,
    path: Path,
    **kwargs: Any,
) -> None:
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", SAVE_PADDING)
    fig.savefig(path, **kwargs)


def _save_legend_only(
    ax: Axes,
    output_dir: Path,
    stem: str,
    legend_label_order: list[str] | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _order_legend_handles(
        handles=handles,
        labels=labels,
        label_order=legend_label_order,
    )

    if not handles:
        return

    legend_layouts = [
        ("vertical", 1),
        ("horizontal", len(labels)),
    ]

    for suffix, ncol in legend_layouts:
        n_rows = math.ceil(len(labels) / ncol)
        height = max(0.8, 0.26 * n_rows)
        fig = plt.figure(figsize=(2.8 * ncol, height))
        fig.legend(
            handles,
            labels,
            loc="center",
            frameon=False,
            ncol=ncol,
        )
        _save_pdf(
            fig,
            output_dir / f"{stem}_legend_{suffix}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def _save_with_legend_variants(
    fig: Figure,
    ax: Axes,
    output_dir: Path,
    stem: str,
    legend_label_order: list[str] | None = None,
    legend_ncol: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_legend_only(
        ax=ax,
        output_dir=output_dir,
        stem=stem,
        legend_label_order=legend_label_order,
    )

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _order_legend_handles(
        handles=handles,
        labels=labels,
        label_order=legend_label_order,
    )
    legend = ax.legend(handles, labels, ncol=legend_ncol, **LEGEND_KWARGS)
    _save_pdf(fig, output_dir / f"{stem}_with_legend.pdf")
    legend.remove()

    _save_pdf(fig, output_dir / f"{stem}_no_legend.pdf")
    plt.close(fig)


def _order_legend_handles(
    handles: list,
    labels: list[str],
    label_order: list[str] | None,
) -> tuple[list, list[str]]:
    if label_order is None:
        return handles, labels

    rank = {label: index for index, label in enumerate(label_order)}
    ordered = sorted(
        zip(handles, labels),
        key=lambda item: (rank.get(item[1], len(rank)), labels.index(item[1])),
    )

    return [handle for handle, _ in ordered], [label for _, label in ordered]


def _read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _method_label_order(methods: list[str]) -> list[str]:
    return [_method_label(method) for method in methods]


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _method_style(method: str) -> PlotStyle:
    style = METHOD_STYLES.get(method)
    if style is not None:
        return style

    return METHOD_STYLES.get(
        method,
        {
            "color": None,
            "marker": "o",
            "linestyle": "-",
        },
    )


def _available_method_order(methods: list[str]) -> list[str]:
    ordered = [method for method in METHOD_ORDER if method in methods]
    ordered.extend(sorted(set(methods) - set(ordered)))
    return ordered


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def _truncated_normal_pdf(
    x: np.ndarray,
    mu: float,
    sigma: float,
    trunc_left: float,
    trunc_right: float,
) -> np.ndarray:
    a = (trunc_left - mu) / sigma
    b = (trunc_right - mu) / sigma
    z = (x - mu) / sigma

    normalizer = _normal_cdf(np.array([b]))[0] - _normal_cdf(np.array([a]))[0]
    density = _normal_pdf(z) / sigma / normalizer

    return np.where((x >= trunc_left) & (x <= trunc_right), density, 0.0)


def _distribution_pdf(x: np.ndarray, config: dict) -> np.ndarray:
    if config["type"] == "truncated_normal":
        return _truncated_normal_pdf(
            x=x,
            mu=config["mu"],
            sigma=config["sigma"],
            trunc_left=config["trunc_left"],
            trunc_right=config["trunc_right"],
        )
    elif config["type"] == "truncated_normal_mixture":
        weights = np.asarray(config["weights"], dtype=np.float64)
        mus = np.asarray(config["mus"], dtype=np.float64)
        sigmas = np.asarray(config["sigmas"], dtype=np.float64)
        trunc_lefts = np.asarray(config["trunc_lefts"], dtype=np.float64)
        trunc_rights = np.asarray(config["trunc_rights"], dtype=np.float64)

        density = np.zeros_like(x, dtype=np.float64)

        for weight, mu, sigma, trunc_left, trunc_right in zip(
            weights,
            mus,
            sigmas,
            trunc_lefts,
            trunc_rights,
        ):
            density += weight * _truncated_normal_pdf(
                x=x,
                mu=float(mu),
                sigma=float(sigma),
                trunc_left=float(trunc_left),
                trunc_right=float(trunc_right),
            )

        return density
    else:
        raise ValueError(f"Unknown distribution type: {config['type']}")


def _x_range_from_distributions(
    distribution_configs: list[dict],
) -> tuple[float, float]:
    lefts = []
    rights = []

    for config in distribution_configs:
        if config["type"] == "truncated_normal":
            lefts.append(config["trunc_left"])
            rights.append(config["trunc_right"])
        elif config["type"] == "truncated_normal_mixture":
            lefts.extend(config["trunc_lefts"])
            rights.extend(config["trunc_rights"])
        else:
            raise ValueError(f"Unknown distribution type: {config['type']}")

    return float(min(lefts)), float(max(rights))


def _read_ability_npz(data_dir: Path, data_file: str | None = None) -> np.ndarray:
    data_path = data_dir / (data_file or "ability.npz")

    if not data_path.exists():
        raise FileNotFoundError(f"Ability npz not found: {data_path}")

    with np.load(data_path) as data:
        if "ability" not in data:
            raise KeyError(f"'ability' array not found in {data_path}")

        ability = np.asarray(data["ability"], dtype=np.float64)

    if ability.ndim != 3:
        raise ValueError(
            f"ability must have shape (n_trials, n, m), got {ability.shape}"
        )

    return ability


def plot_data_distribution(data_dir: Path) -> Path:
    config = _read_json(data_dir / "config.json")
    ability = _read_ability_npz(
        data_dir=data_dir,
        data_file=config.get("data_file"),
    )
    distribution_configs = config.get("distributions")

    if distribution_configs is None:
        raise KeyError(f"'distributions' not found in {data_dir / 'config.json'}")

    x_min, x_max = _x_range_from_distributions(distribution_configs)
    x = np.linspace(x_min, x_max, DENSITY_GRID_SIZE)
    plot_dir = data_dir / "plots"

    fig, ax = _new_figure()

    for j, config_j in enumerate(distribution_configs):
        style = CANDIDATE_STYLES[j % len(CANDIDATE_STYLES)]
        samples = ability[:, :, j].reshape(-1)
        density = _distribution_pdf(x, config_j)

        ax.plot(
            x,
            density,
            label=f"Candidate {j}",
            linewidth=LINE_WIDTH,
            color=style["color"],
            linestyle=style["linestyle"],
        )

        _, _, patches = ax.hist(
            samples,
            bins=HIST_BINS,
            density=True,
            histtype="bar",
            linewidth=0.6,
            color=style["color"],
            edgecolor=style["color"],
            alpha=SAMPLE_ALPHA,
        )

        hatch = SAMPLE_HATCHES[j % len(SAMPLE_HATCHES)]
        bar_container = cast(BarContainer, patches)

        for patch in bar_container.patches:
            patch.set_hatch(hatch)

    ax.set_xlabel(r"Ability $\theta$")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)

    _save_with_legend_variants(
        fig=fig,
        ax=ax,
        output_dir=plot_dir,
        stem="ability_distribution",
    )

    return plot_dir


def _sweep_axis(sweep_config: dict, rows: list[dict]) -> tuple[str, str]:
    sweep = sweep_config["sweep"]

    if sweep == "linear-budget":
        return "budget_scale", r"Budget scale $k$"

    if sweep == "power-beta-true":
        return "beta_true", r"True exponent $\beta_0$"

    if sweep in {"linear-gamma", "power-gamma"}:
        return "gamma", r"CVaR level $\gamma$"

    if rows and "gamma" in rows[0]:
        return "gamma", r"CVaR level $\gamma$"

    raise ValueError(f"Unknown sweep axis for sweep: {sweep}")


def _regret_comparison_axis(config: dict, rows: list[dict]) -> tuple[str, str]:
    sweep = config["sweep"]

    if sweep == "n":
        return "n", r"Number of participants $n$"

    if sweep == "budget_scale":
        return "budget_scale", r"Budget scale $k$"

    if rows and sweep in rows[0]:
        return sweep, sweep

    raise ValueError(f"Unknown regret comparison axis for sweep: {sweep}")


def plot_sweep_metric(
    rows: list[dict],
    x_key: str,
    xlabel: str,
    metric: str,
    ylabel: str,
    output_dir: Path,
    stem: str,
    clip: tuple[float, float] | None = None,
    show_std: bool = False,
    legend_ncol: int = 1,
) -> None:
    grouped = defaultdict(list)

    for row in rows:
        if row["method"] in EXCLUDED_METHODS:
            continue

        std_key = f"std_{metric.removeprefix('mean_')}"
        std = float(row[std_key]) if std_key in row else 0.0
        group_key = row["method"]
        grouped[group_key].append((float(row[x_key]), float(row[metric]), std))

    legend_method_order = _available_method_order(list(set(grouped.keys())))

    fig, ax = _new_figure()

    for method in legend_method_order:
        points = sorted(grouped[method], key=lambda item: item[0])
        x = np.asarray([point[0] for point in points], dtype=np.float64)
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        std = np.asarray([point[2] for point in points], dtype=np.float64)
        style = _method_style(method)

        ax.plot(
            x,
            y,
            label=_method_label(method),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

        if show_std:
            lower = y - std
            upper = y + std

            if clip is not None:
                lower = np.clip(lower, clip[0], clip[1])
                upper = np.clip(upper, clip[0], clip[1])

            ax.fill_between(
                x,
                lower,
                upper,
                color=style["color"],
                alpha=FILL_ALPHA,
                linewidth=0.0,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_key == "budget_scale":
        ax.set_xticks([2.5, 5.0, 7.5])
    elif x_key == "gamma":
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
    elif x_key == "beta":
        ax.set_xticks([0.25, 0.50, 0.75])
    if metric == "mean_regret":
        ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.grid(True, alpha=0.3)

    if clip is not None:
        ax.set_ylim(*clip)

    _save_with_legend_variants(
        fig=fig,
        ax=ax,
        output_dir=output_dir,
        stem=stem,
        legend_label_order=_method_label_order(legend_method_order),
        legend_ncol=legend_ncol,
    )


def plot_sweep(sweep_dir: Path) -> Path:
    sweep_config = _read_json(sweep_dir / "sweep_config.json")
    rows = _read_csv(sweep_dir / "sweep_summary.csv")
    x_key, xlabel = _sweep_axis(sweep_config=sweep_config, rows=rows)
    plot_dir = sweep_dir / "plots"

    plot_sweep_metric(
        rows=rows,
        x_key=x_key,
        xlabel=xlabel,
        metric="correct_rate",
        ylabel=r"Correct Selection Rate $\rightarrow$",
        output_dir=plot_dir,
        stem=f"correct_rate_{sweep_config['sweep']}",
        clip=(0.0, 1.0),
    )
    plot_sweep_metric(
        rows=rows,
        x_key=x_key,
        xlabel=xlabel,
        metric="mean_regret",
        ylabel=r"$\leftarrow$ Performance Gap",
        output_dir=plot_dir,
        stem=f"performance_gap_{sweep_config['sweep']}",
        clip=(-0.01, 0.11),
        show_std=True,
    )

    return plot_dir


def plot_regret_comparison(comparison_dir: Path) -> Path:
    config = _read_json(comparison_dir / "config.json")
    rows = _read_csv(comparison_dir / "summary.csv")
    x_key, xlabel = _regret_comparison_axis(config=config, rows=rows)
    plot_dir = comparison_dir / "plots"

    fig, ax = _new_figure()

    for metric in ("actual_quantile_regret", "theory_bound"):
        points = sorted(
            ((float(row[x_key]), float(row[metric])) for row in rows),
            key=lambda item: item[0],
        )
        x = np.asarray([point[0] for point in points], dtype=np.float64)
        y = np.asarray([point[1] for point in points], dtype=np.float64)
        style = REGRET_BOUND_STYLES[metric]

        ax.plot(
            x,
            y,
            label=REGRET_BOUND_LABELS[metric],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\leftarrow$ Performance Gap")
    ax.set_xticks([10, 100, 200])
    ax.grid(True, alpha=0.3)

    _save_with_legend_variants(
        fig=fig,
        ax=ax,
        output_dir=plot_dir,
        stem=f"performance_gap_{config['score_gen']}",
        legend_label_order=[
            REGRET_BOUND_LABELS["actual_quantile_regret"],
            REGRET_BOUND_LABELS["theory_bound"],
        ],
    )

    return plot_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Sweep directory or data date directory to plot.",
    )

    return parser.parse_args()


def plot_input_dir(input_dir: Path) -> Path:
    if (input_dir / "sweep_config.json").exists():
        return plot_sweep(sweep_dir=input_dir)

    if (input_dir / "config.json").exists():
        config = _read_json(input_dir / "config.json")

        if config.get("comparison") == "regret_bound":
            return plot_regret_comparison(comparison_dir=input_dir)

        return plot_data_distribution(data_dir=input_dir)

    raise FileNotFoundError(
        "Input directory must contain either sweep_config.json or "
        f"config.json for data plots: {input_dir}"
    )


def main() -> None:
    args = parse_args()
    _configure_matplotlib_fonts()

    plot_dir = plot_input_dir(input_dir=args.input_dir)
    print(f"Saved plots to: {plot_dir}")


if __name__ == "__main__":
    main()
