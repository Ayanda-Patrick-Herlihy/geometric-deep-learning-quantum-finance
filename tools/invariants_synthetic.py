"""Shared helpers for the invariant audit (scratch only; never writes into the repo).

* ``REPO``: repository root (read-only).
* ``SCRATCH``: this directory; all artefacts (checkpoints, pnl dumps, CSVs) go here.
* ``load_small_config``: config.yaml loaded, then overridden IN MEMORY for tiny CPU runs.
* ``make_synthetic_df``: ``build_synthetic_dataframe`` + optional factor-structured
  returns (so the |R|>0.25 graph actually has edges) + optional heavy-tailed features.
* ``train_fold_in_scratch``: calls ``run_ablation.train_ablation_fold`` with the
  module-level ``_PROJECT_ROOT`` monkeypatched to a scratch directory, so the
  checkpoint and debug-PnL files it writes land under SCRATCH, not the repo.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parent.parent
SCRATCH = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.data_loader import build_synthetic_dataframe  # noqa: E402


def load_small_config(
    max_epochs: int = 3,
    batch_size: int = 8,
    workers: int = 0,
    train_window: int = 200,
    val_window: int = 60,
    step: int = 60,
) -> dict:
    with open(REPO / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    cfg["wandb"]["mode"] = "disabled"
    t = cfg["training"]
    t["max_epochs"] = max_epochs
    t["batch_size"] = batch_size
    t["dataloader_workers"] = workers
    t["walk_forward"] = {
        "train_window": train_window,
        "validation_window": val_window,
        "step_size": step,
    }
    # evaluation writes debug pnl relative to (monkeypatched) project root
    cfg["evaluation"]["experiments_dir"] = "experiments"
    return cfg


def make_synthetic_df(
    n_tickers: int = 40,
    n_dates: int = 420,
    seed: int = 42,
    factor: bool = True,
    heavy: bool = False,
) -> pd.DataFrame:
    """Synthetic fact table.

    factor=True : replace log_return_1d with a 4-sector one-factor model so that
                  within-sector Pearson |R| ~ 0.5 (> 0.25 threshold) -> non-trivial graph.
    heavy=True  : replace the 12 node features with Student-t(df=2) draws, inject
                  +-50 sigma outliers in ~0.5% of cells and all-zero feature rows in
                  ~1% of rows, plus one constant (MAD=0) feature column on some dates.
    """
    df = build_synthetic_dataframe(n_tickers=n_tickers, n_dates=n_dates, seed=seed)
    rng = np.random.default_rng(seed + 1)
    eq = df["AssetClass"] == "equity"
    if factor:
        dates = sorted(df.loc[eq, "Date"].unique())
        n_sec = 4
        f = rng.normal(0, 0.015, size=(len(dates), n_sec))
        d_idx = {d: i for i, d in enumerate(dates)}
        tick_id = df.loc[eq, "Ticker"].str[-4:].astype(int).to_numpy()
        di = df.loc[eq, "Date"].map(d_idx).to_numpy()
        sec = tick_id % n_sec
        r = f[di, sec] + rng.normal(0, 0.015, size=len(di))
        df.loc[eq, "log_return_1d"] = r
    if heavy:
        feat_cols = [
            "log_return_5d", "log_return_20d", "vol_10d", "vol_22d", "vol_60d",
            "mom_3m", "mom_6m", "mom_12m", "volume_ratio", "amihud_illiquidity",
            "high_low_spread_22d", "price_to_52w_high",
        ]
        n = int(eq.sum())
        X = rng.standard_t(df=2, size=(n, len(feat_cols)))
        out = rng.random(size=X.shape) < 0.005
        X[out] = 50.0 * np.sign(rng.normal(size=out.sum()))
        zero_rows = rng.random(n) < 0.01
        X[zero_rows] = 0.0
        # A column that is constant on ~10% of dates (MAD = 0 on those cross-sections)
        dts = df.loc[eq, "Date"].to_numpy()
        uniq = np.unique(dts)
        const_dates = set(rng.choice(uniq, size=max(1, len(uniq) // 10), replace=False))
        const_mask = np.array([d in const_dates for d in dts])
        X[const_mask, 10] = 0.0
        df.loc[eq, feat_cols] = X
    return df


def train_fold_in_scratch(
    config_id: str,
    fold_idx: int,
    train_dates,
    val_dates,
    config: dict,
    df: pd.DataFrame,
    seed: int = 42,
    root: Path | None = None,
) -> tuple[dict, Path]:
    """Runs run_ablation.train_ablation_fold with _PROJECT_ROOT -> scratch root."""
    import src.run_ablation as ra

    root = Path(root or (SCRATCH / "runs" / "default"))
    root.mkdir(parents=True, exist_ok=True)
    orig = ra._PROJECT_ROOT
    ra._PROJECT_ROOT = root
    try:
        res = ra.train_ablation_fold(
            config_id=config_id,
            fold_idx=fold_idx,
            train_dates=train_dates,
            val_dates=val_dates,
            config=config,
            device=torch.device("cpu"),
            seed=seed,
            dataframe=df,
            parquet_path=None,
            use_wandb=False,
        )
    finally:
        ra._PROJECT_ROOT = orig
    ckpt = (
        root / "output" / "checkpoints" / "ablation" / config_id
        / f"{config_id}_{ra._get_config_name(config_id)}_fold_{fold_idx:02d}_seed_{seed}.pt"
    )
    return res, ckpt
