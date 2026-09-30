"""Re-evaluate saved ablation checkpoints without retraining.

For each (config, fold) checkpoint written by ``run_ablation.train_ablation_fold``
this script rebuilds the same walk-forward validation window, loads the
weights with ``strict=True`` (a failure means the current code no longer
matches the code that produced the checkpoint), re-runs
``evaluate_model_on_dataset`` with a per-stock prediction dump, and compares
the recomputed metrics with the per-fold rows of a results table such as the
OneDrive ``master_ablation_results.csv``.

The dumps feed ``posthoc_eval.py`` (turnover costs, skip-day returns,
same-universe IC, tail composition) at inference cost only.

Usage:
    uv run python src/reevaluate_checkpoints.py \
        --checkpoints output/checkpoints/ablation --configs A0 A6 A8 \
        --results experiments/master_ablation_results.csv --out experiments/reeval
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.data_loader import FinancialGraphDataset, WalkForwardSplitter, dataset_view  # noqa: E402
from src.evaluation import evaluate_model_on_dataset  # noqa: E402
from src.run_ablation import VARIANT_DATASET_KWARGS, build_ablation_model  # noqa: E402
from src.train import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_CKPT_RE = re.compile(r"(?P<cid>A\d[a-z]?)_.*_fold_(?P<fold>\d+)_seed_(?P<seed>\d+)\.pt$")
_METRICS = ["ic", "sharpe_annual", "directional_accuracy", "max_drawdown", "mse"]


def find_checkpoints(root: Path, config_id: str) -> list[tuple[int, int, Path]]:
    """Finds checkpoints by config_id prefix, ignoring the (formerly mislabelled) model name."""
    found = []
    for path in sorted((root / config_id).glob(f"{config_id}_*_fold_*_seed_*.pt")):
        match = _CKPT_RE.search(path.name)
        if match and match["cid"] == config_id:
            found.append((int(match["fold"]), int(match["seed"]), path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=_PROJECT_ROOT / "output" / "checkpoints" / "ablation")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--results", type=Path, default=None, help="Per-fold results CSV to reconcile against.")
    parser.add_argument("--out", type=Path, default=_PROJECT_ROOT / "experiments" / "reeval")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="'train' evaluates each checkpoint on its own training window, giving the "
        "in-sample vs out-of-sample gap needed to test the overfitting/regularisation claim.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config = load_config(args.config)
    device = torch.device(args.device)
    dataframe = pd.read_parquet(_PROJECT_ROOT / config["data"]["parquet_path"])
    full_ds = FinancialGraphDataset(parquet_path="unused", config=config, dataframe=dataframe)
    wf = config["training"]["walk_forward"]
    folds = list(
        WalkForwardSplitter(
            dates=full_ds.valid_dates,
            train_window=wf["train_window"],
            validation_window=wf["validation_window"],
            step_size=wf["step_size"],
            purge_gap=int(wf.get("purge_gap", 20)),
        )
    )
    reference = pd.read_csv(args.results) if args.results else None

    rows = []
    for cid in args.configs:
        found = find_checkpoints(args.checkpoints, cid)
        keys = [(fold, seed) for fold, seed, _ in found]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            # e.g. an old mislabelled A6_Graph_Only_* file next to a re-run A6_Geometry_Quantum_*.
            raise SystemExit(
                f"{cid}: several checkpoints for (fold, seed) {duplicates} in {args.checkpoints / cid}; "
                "keep one set per directory."
            )
        for fold, seed, path in found:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if checkpoint.get("config_id") != cid:
                logger.error("%s: stored config_id %r != %r", path.name, checkpoint.get("config_id"), cid)
            model = build_ablation_model(cid, config, device)
            load_error = ""
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            except RuntimeError as exc:  # architecture drift between runs and current code
                load_error = str(exc).splitlines()[0]
                logger.error("%s: %s", path.name, load_error)
                rows.append({"config_id": cid, "fold": fold, "seed": seed, "load_error": load_error})
                continue
            torch.manual_seed(seed)
            eval_dates = folds[fold][1] if args.split == "val" else folds[fold][0]
            val_ds = dataset_view(full_ds, eval_dates, None, seed, **VARIANT_DATASET_KWARGS.get(cid, {}))
            metrics = evaluate_model_on_dataset(
                model,
                val_ds,
                device,
                dump_predictions_path=args.out
                / "predictions"
                / f"pred_{cid}_fold_{fold:02d}_seed_{seed}{'' if args.split == 'val' else '_train'}.parquet",
            )
            row = {"config_id": cid, "fold": fold, "seed": seed, "split": args.split, "best_epoch": checkpoint.get("epoch"), "load_error": ""}
            row.update({f"re_{m}": metrics.get(m, np.nan) for m in _METRICS})
            if reference is not None and args.split == "val":
                match = reference[(reference["config_id"] == cid) & (reference["fold"] == fold)]
                if "seed" in reference.columns:
                    match = match[match["seed"] == seed]
                if len(match) == 1:
                    for m in _METRICS:
                        if m in match.columns:
                            row[f"reported_{m}"] = float(match[m].iloc[0])
                            row[f"absdiff_{m}"] = abs(row[f"re_{m}"] - row[f"reported_{m}"])
                else:
                    row["reconcile_note"] = f"{len(match)} matching rows in results table"
            rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / f"reconciliation_{args.split}.csv", index=False)
    diff_cols = [c for c in table.columns if c.startswith("absdiff_")]
    if diff_cols:
        logger.info("Max absolute differences vs reported:\n%s", table.groupby("config_id")[diff_cols].max().to_string())
    logger.info("Wrote %s", args.out / f"reconciliation_{args.split}.csv")


if __name__ == "__main__":
    main()
