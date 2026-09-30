# GDL-Quantum-Finance

Geometric deep learning model for quantum-inspired financial market dynamics.

This project combines three modelling components:

- A geometric layer that maps financial features onto a hypersphere.
- A quantum-inspired regime layer that models market states with density matrices.
- A graph layer that captures cross-asset dependencies with graph neural networks.

## Repository Structure

- `config.yaml`: Project configuration for data paths, features, model settings, training, and evaluation.
- `src/`: Data pipeline, model layers, training, evaluation, and diagnostics code.
- `scripts/`: Plotting and analysis scripts.
- `notebooks/`: Exploratory notebooks retained for manual inspection.
- `figures/`: Selected figures for reporting.
- `data/`: Local data directory. Raw and generated data are not committed.

## Setup

Install dependencies with `uv` from the repository root:

```bash
uv sync
```

## Configuration

Set API keys locally before running data ingestion. Do not commit credentials.
The committed `config.yaml` keeps the API key fields empty.

## Data Pipeline

Run the pipeline stages in order:

```bash
uv run python src/data_pipeline/stage1_consolidate_equities.py
uv run python src/data_pipeline/stage1_consolidate_macro.py
uv run python src/data_pipeline/stage1_consolidate_commodities.py
uv run python src/data_pipeline/stage2_merge_master.py
uv run python src/data_pipeline/stage3_engineer_features.py
uv run python src/data_pipeline/stage4_export_parquet.py
```

## Training And Evaluation

Random seeds for ablation runs are read from `config.yaml` under
`evaluation.ablation_seeds`, or given with `--seeds`. The training protocol
(inner-validation early stopping, stopping metric, common random numbers) is
read from `training.protocol`; `--original-protocol` reproduces the
dissertation's single-seed protocol.

### Full ablation study (all configurations)

Runs every ablation configuration defined in `src/run_ablation.py`, using
your processed fact table and the walk-forward settings in `config.yaml`.

```bash
uv run python src/run_ablation.py --no-wandb
```

### Run a subset of configurations

Useful when you want a smaller sweep (for example, baselines versus the full
model).

```bash
uv run python src/run_ablation.py --configs A0 A8 --no-wandb
```

### Limit walk-forward folds or epochs (smoke or partial runs)

`--max-folds` caps how many walk-forward folds are executed.
`--max-epochs` overrides `training.max_epochs` for a shorter run.

```bash
uv run python src/run_ablation.py --configs A8 --max-folds 1 --max-epochs 1 --device cpu --no-wandb
```

### Synthetic smoke test (no Parquet required)

Uses built-in synthetic data so you can validate wiring, losses, and device
behaviour without `data/processed/fact_table.parquet`.

```bash
uv run python src/run_ablation.py --smoke-test --configs A0 --max-folds 1 --device cpu --no-wandb
```

### Weights & Biases logging

Enable online logging when your `config.yaml` `wandb.mode` allows it and you
are authenticated (`wandb login`).

```bash
uv run python src/run_ablation.py --wandb
```

### Model verification (layer constraints and small optimisation checks)

Runs progressive checks in `src/verify_model.py` (not the same as a full
ablation).

```bash
uv run python src/verify_model.py --stage 0
```

Run the full test suite:

```bash
uv run --group dev python -m pytest tests/
```
