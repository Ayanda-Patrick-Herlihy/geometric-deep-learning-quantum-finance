#!/usr/bin/env bash
# Regenerates every experiment and statistic behind the paper from pinned seeds.
#
# Stages follow the experiment decision gates: stop after a gate whose
# criterion fails. Every run_ablation call writes to its own directory under
# experiments/ (checkpoints, logs, result tables, prediction dumps), so no run
# overwrites another and the restored dissertation checkpoints in
# output/checkpoints/ablation are never touched. Status: dry-run on synthetic
# data only; not yet run on the real fact table.
#
# Usage: bash reproduce.sh [stage ...]   (default: all stages 0-5)
set -euo pipefail
cd "$(dirname "$0")"

INNER=126                 # purged inner-validation window for early stopping (trading days)
SEEDS_CHEAP="0 1 2 3 4"   # non-graph configurations
SEEDS_GRAPH="0 1 2"       # graph configurations (about 30-40 GPU-min per fold each)
PROTOCOL="--inner-val-days $INNER --stop-metric supervised --crn"
TAG="_iv${INNER}_sup_crn" # file-name tag that run_ablation derives from PROTOCOL
RUN="uv run python src/run_ablation.py --no-wandb --deterministic --dump-predictions $PROTOCOL"
POST="uv run python src/posthoc_eval.py"
STAGES="${*:-0 1 2 3 4 5}"

stage() { [[ " $STAGES " == *" $1 "* ]]; }

if stage 0; then  # environment and data provenance
  uv sync --frozen --group dev
  uv run --group dev python -m pytest tests/ -q
  test -f data/processed/fact_table.parquet || {
    echo "fact_table.parquet missing: run src/data_pipeline/stage1..4 (needs EODHD/FRED keys)"; exit 1; }
  mkdir -p experiments
  sha256sum data/processed/fact_table.parquet config.yaml uv.lock | tee experiments/manifest.sha256
  git rev-parse HEAD | tee experiments/code_commit.txt
fi

if stage 1; then  # Gate 0: reconcile saved checkpoints with the dissertation (inference only)
  # Requires the OneDrive bundle restored to output/checkpoints/ablation and
  # experiments/master_ablation_results.csv.
  uv run python src/reevaluate_checkpoints.py --configs A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 \
    --results experiments/master_ablation_results.csv --out experiments/gate0
  uv run python src/reevaluate_checkpoints.py --configs A0 A1 A2 A6 A7 A8 --split train --out experiments/gate0
  $POST --pred-dir experiments/gate0/predictions --configs A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 \
    --baseline A0 --out experiments/gate0/posthoc
  $POST --pred-dir experiments/gate0/predictions --tag _train --configs A0 A1 A2 A6 A7 A8 \
    --baseline A0 --out experiments/gate0/posthoc_train
fi

if stage 2; then  # Gate 1: negative and positive controls under the corrected protocol
  $RUN --experiments-dir experiments/gate1/inject --seeds 0 --configs A0 --control inject_target
  $RUN --experiments-dir experiments/gate1/shuffle --seeds 0 1 --configs A0 A6 --control shuffle_labels
  $POST --pred-dir experiments/gate1/shuffle --configs A0 A6 --tag _shuffle_labels$TAG --out experiments/gate1/posthoc
  $POST --pred-dir experiments/gate1/inject --configs A0 --tag _inject_target$TAG --out experiments/gate1/posthoc
fi

if stage 3; then  # Gates 2 and 3 plus the sphere term-level ablation (non-graph, cheap)
  $RUN --experiments-dir experiments/main/cheap --seeds $SEEDS_CHEAP \
    --configs A0 A1 A2 A2e A2u A2m A6 A6e A6u A6c A6s A9t
fi

if stage 4; then  # graph configurations (expensive)
  $RUN --experiments-dir experiments/main/graph --seeds $SEEDS_GRAPH --configs A8 A5
  $RUN --experiments-dir experiments/graph_controls/permute --seeds 0 --configs A8 --control permute_graph_nodes
  $RUN --experiments-dir experiments/graph_controls/stale --seeds 0 --configs A8 --control stale_graph
fi

if stage 5; then  # statistics for every table (prediction dumps are found recursively)
  $POST --pred-dir experiments/main --tag $TAG --baseline A0 --out experiments/posthoc_vs_A0 \
    --configs A0 A1 A2 A2e A2u A2m A6 A6e A6u A6c A6s A9t A5 A8
  $POST --pred-dir experiments/main --tag $TAG --baseline A2 --out experiments/posthoc_vs_A2 \
    --configs A2 A2e A2u A2m A6 A5
  $POST --pred-dir experiments/main --tag $TAG --baseline A6 --out experiments/posthoc_vs_A6 \
    --configs A6 A6e A6u A6c A6s A8
  $POST --pred-dir experiments/graph_controls --tag _permute_graph_nodes$TAG --configs A8 --out experiments/graph_controls/posthoc
  $POST --pred-dir experiments/graph_controls --tag _stale_graph$TAG --configs A8 --out experiments/graph_controls/posthoc
fi
