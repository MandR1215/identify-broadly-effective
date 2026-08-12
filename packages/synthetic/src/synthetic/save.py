from pathlib import Path
import csv

import numpy as np

from core.utils import format_json_compact_lists


def write_sample_abilities_csv(path: Path, samples: np.ndarray) -> None:
    """Write sampled plot abilities in long CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "candidate", "ability"],
        )
        writer.writeheader()

        for sample_id in range(samples.shape[0]):
            for candidate in range(samples.shape[1]):
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "candidate": candidate,
                        "ability": float(samples[sample_id, candidate]),
                    }
                )


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""
    if len(rows) == 0:
        raise ValueError("rows must not be empty.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_config(path: Path, config: dict) -> None:
    """Write experiment configuration to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(format_json_compact_lists(config))
        f.write("\n")
