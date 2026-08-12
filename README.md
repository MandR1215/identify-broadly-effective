# Setup

```bash
uv sync
```

# Create Data

```bash
uv run python -m synthetic.create_data --output-dir outputs/synthetic/data/main
```

# Run Experiments

## Run All Experiments

```bash
uv run python -m synthetic.sweep_all --data-dir outputs/synthetic/data/main
```

## Run Selected Experiments

```bash
uv run python -m synthetic.sweep_all \
  --data-dir outputs/synthetic/data/main \
  --sweeps linear-budget linear-gamma power-beta-true power-gamma
```

# Generate Plots

## Generate All Plots

```bash
uv run python -m synthetic.sweep_all --plot-only
```

## Plot One Output Directory

```bash
uv run python -m synthetic.plot outputs/synthetic/sweep/linear-gamma/latest
```
