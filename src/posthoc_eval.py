"""Offline evaluation from dumped per-stock predictions.

``run_ablation.py --dump-predictions`` writes one parquet per
(config, fold, seed) with columns date, ticker_id, ticker, pred, y, y_skip1,
close, volume. This module recomputes every reported metric from those files
and adds the diagnostics a reviewer will ask for, without retraining:

    * IC on the training universe (identical to the logged ``ic``) and on the
      liquid universe that the Sharpe portfolio actually trades.
    * Top/bottom-k long-short PnL with the repository's flat 2 x bps cost
      (reproduces ``sharpe_annual``) and with turnover-based costs over a
      grid of per-side costs.
    * PnL on simple returns. The in-loop PnL averages log returns, which
      credits a short leg of volatile names with their volatility drag
      (about sigma^2/2 per day) even when their simple returns average zero.
    * Skip-day variants (returns from t+1 to t+2) that separate predictive
      signal from next-day bid-ask bounce.
    * Tail composition (price, dollar volume, realised |return|) of the
      traded names.
    * Pairwise comparisons with dependence-robust tests (see stats_tests.py).

Usage:
    uv run python src/posthoc_eval.py --pred-dir experiments/predictions \
        --configs A0 A6 A8 --out experiments/posthoc
"""

import argparse
import json
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

from src import stats_tests  # noqa: E402

logger = logging.getLogger(__name__)

_COLUMNS = ["date", "ticker_id", "pred", "y", "y_skip1", "close", "volume"]
_FILE_RE = re.compile(r"pred_(?P<cid>[A-Za-z0-9_+-]+)_fold_(?P<fold>\d+)_seed_(?P<seed>\d+)(?P<tag>_[a-z0-9_]+)?\.parquet$")


def prediction_files(pred_dir: Path, config_id: str, tag: str = "") -> list[tuple[int, int, Path]]:
    """(seed, fold, path) for every dump of one configuration with exactly this tag."""
    found = []
    for path in sorted(Path(pred_dir).rglob(f"pred_{config_id}_fold_*_seed_*{tag}.parquet")):
        match = _FILE_RE.search(path.name)
        if match is None or match["cid"] != config_id or (match["tag"] or "") != tag:
            continue
        found.append((int(match["seed"]), int(match["fold"]), path))
    return sorted(found)


def load_predictions(pred_dir: Path, config_id: str, seed: int | None = None, tag: str = "") -> pd.DataFrame:
    """Concatenates the fold files for one configuration (and seed), keeping row order within dates."""
    frames = []
    for file_seed, fold, path in prediction_files(pred_dir, config_id, tag):
        if seed is not None and file_seed != seed:
            continue
        frame = pd.read_parquet(path, columns=_COLUMNS)
        frame["fold"] = fold
        frame["seed"] = file_seed
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No prediction files for {config_id} (seed={seed}, tag={tag!r}) in {pred_dir}.")
    return pd.concat(frames, ignore_index=True).sort_values(["seed", "fold", "date"], kind="stable")


def daily_metrics_from_files(pred_dir: Path, config_id: str, tag: str = "", **kwargs) -> pd.DataFrame:
    """daily_metrics computed one fold file at a time, so memory stays at one fold."""
    frames = []
    for file_seed, fold, path in prediction_files(pred_dir, config_id, tag):
        frame = pd.read_parquet(path, columns=_COLUMNS)
        frame["fold"] = fold
        frame["seed"] = file_seed
        frames.append(daily_metrics(frame.sort_values("date", kind="stable"), **kwargs))
    if not frames:
        raise FileNotFoundError(f"No prediction files for {config_id} (tag={tag!r}) in {pred_dir}.")
    return pd.concat(frames, ignore_index=True)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return 0.0  # matches evaluation.information_coefficient for constant input
    return float(np.corrcoef(ra, rb)[0, 1])


def daily_metrics(
    preds: pd.DataFrame,
    top_k: int = 50,
    min_price: float = 5.0,
    min_dollar_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """One row per (seed, fold, date) with IC, gross PnL, turnover and tail stats."""
    rows = []
    for (seed, fold), fold_frame in preds.groupby(["seed", "fold"], sort=True):
        prev_long: set = set()
        prev_short: set = set()
        for date, day in fold_frame.groupby("date", sort=True):
            pred = day["pred"].to_numpy(np.float64)
            y = day["y"].to_numpy(np.float64)
            y2 = day["y_skip1"].to_numpy(np.float64)
            liquid = (
                np.isfinite(day["close"].to_numpy())
                & (day["close"].to_numpy() >= min_price)
                & (day["close"].to_numpy() * day["volume"].to_numpy() >= min_dollar_volume)
                & np.isfinite(pred)
                & np.isfinite(y)
            )
            row = {
                "seed": seed,
                "fold": fold,
                "date": date,
                "n_all": len(day),
                "n_liquid": int(liquid.sum()),
                "ic": _spearman(pred, y),
                "ic_liquid": _spearman(pred[liquid], y[liquid]),
                "ic_skip1_liquid": _spearman(pred[liquid], y2[liquid]),
            }
            if liquid.sum() >= 2 * top_k:
                sub = day.loc[liquid]
                # Same selection as evaluation.compute_daily_pnl (torch.topk on
                # float32 in dataset row order), so ties resolve identically.
                scores = torch.from_numpy(sub["pred"].to_numpy(np.float32))
                long_rows = sub.iloc[torch.topk(scores, top_k).indices.numpy()]
                short_rows = sub.iloc[torch.topk(scores, top_k, largest=False).indices.numpy()]
                long_ids = set(long_rows["ticker_id"].tolist())
                short_ids = set(short_rows["ticker_id"].tolist())
                # Fraction of each leg replaced. Replacing a name trades 2/k of the
                # leg (sell + buy); the first day of a fold opens the leg from
                # cash, which trades 1 unit.
                opening = not prev_long
                turn_long = 1.0 if opening else len(long_ids - prev_long) / top_k
                turn_short = 1.0 if opening else len(short_ids - prev_short) / top_k
                traded = 2.0 if opening else 2.0 * (turn_long + turn_short)
                prev_long, prev_short = long_ids, short_ids
                row.update(
                    {
                        "gross": long_rows["y"].mean() - short_rows["y"].mean(),
                        "gross_skip1": long_rows["y_skip1"].mean() - short_rows["y_skip1"].mean(),
                        "gross_simple": np.expm1(long_rows["y"]).mean() - np.expm1(short_rows["y"]).mean(),
                        "gross_skip1_simple": np.expm1(long_rows["y_skip1"]).mean()
                        - np.expm1(short_rows["y_skip1"]).mean(),
                        "turnover_long": turn_long,
                        "turnover_short": turn_short,
                        "traded_notional": traded,
                        "tail_price": pd.concat([long_rows["close"], short_rows["close"]]).median(),
                        "tail_log_dollar_volume": np.log10(
                            pd.concat([long_rows["close"] * long_rows["volume"], short_rows["close"] * short_rows["volume"]])
                        ).median(),
                        "tail_abs_return": pd.concat([long_rows["y"], short_rows["y"]]).abs().mean(),
                        "universe_abs_return": np.abs(y[liquid]).mean(),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def net_pnl(
    daily: pd.DataFrame, cost_bps: float, flat: bool = False, skip: bool = False, simple: bool = False
) -> pd.Series:
    """Daily PnL after costs.

    ``flat=True`` reproduces ``evaluation.compute_daily_pnl`` (2 x cost every
    day). Otherwise each replaced name costs ``cost_bps`` on the sale and on
    the purchase: traded notional per leg is 2 x turnover. ``simple=True``
    uses simple rather than log returns.
    """
    gross = daily["gross" + ("_skip1" if skip else "") + ("_simple" if simple else "")]
    if flat:
        return gross - 2.0 * cost_bps / 1e4
    return gross - cost_bps / 1e4 * daily["traded_notional"]


def _sharpe_both(traded: pd.DataFrame, pnl: pd.Series) -> tuple[float, float]:
    """(mean of per-fold Sharpe, as in the dissertation; Sharpe of all days pooled)."""
    by_fold = pnl.groupby(traded["fold"]).apply(stats_tests.annualised_sharpe)
    return float(by_fold.mean()), stats_tests.annualised_sharpe(pnl)


def summarise(daily: pd.DataFrame, cost_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 40.0)) -> dict:
    """Headline numbers for one (config, seed) from its daily metrics.

    Every Sharpe is reported twice: ``*_foldmean`` (mean of per-fold Sharpe,
    the dissertation's aggregation) and ``*_pooled`` (all days together).
    """
    traded = daily.dropna(subset=["gross"])
    fold_ic = daily.groupby("fold")["ic"].mean()
    out = {
        "n_days": int(len(daily)),
        "ic_mean_over_folds": float(fold_ic.mean()),
        "ic_liquid_mean": float(daily["ic_liquid"].mean()),
        "ic_skip1_liquid_mean": float(daily["ic_skip1_liquid"].mean()),
        "turnover_mean": float((traded["turnover_long"] + traded["turnover_short"]).mean() / 2.0),
        "gross_daily_mean_bps": float(traded["gross"].mean() * 1e4),
        "gross_skip1_daily_mean_bps": float(traded["gross_skip1"].mean() * 1e4),
        "gross_simple_daily_mean_bps": float(traded["gross_simple"].mean() * 1e4),
        "tail_price_median": float(traded["tail_price"].median()),
        "tail_log10_dollar_volume_median": float(traded["tail_log_dollar_volume"].median()),
        "tail_abs_return_over_universe": float((traded["tail_abs_return"] / traded["universe_abs_return"]).mean()),
    }
    variants = {
        "log_flat20": net_pnl(traded, 20.0, flat=True),  # the dissertation's definition
        "simple_flat20": net_pnl(traded, 20.0, flat=True, simple=True),
    }
    for cost in cost_grid:
        variants[f"log_turnover_{cost:g}bps"] = net_pnl(traded, cost)
        variants[f"simple_turnover_{cost:g}bps"] = net_pnl(traded, cost, simple=True)
        variants[f"simple_skip1_turnover_{cost:g}bps"] = net_pnl(traded, cost, skip=True, simple=True)
    for name, pnl in variants.items():
        out[f"sharpe_{name}_foldmean"], out[f"sharpe_{name}_pooled"] = _sharpe_both(traded, pnl)
    return out


def compare(daily_a: pd.DataFrame, daily_b: pd.DataFrame, cost_bps: float = 20.0) -> dict:
    """Dependence-robust comparison of two configurations on aligned dates.

    Metrics are averaged over seeds per date first, so each configuration is
    judged as a training procedure rather than as one lucky initialisation.
    """
    def per_date(daily: pd.DataFrame) -> pd.DataFrame:
        pnl = net_pnl(daily, cost_bps, simple=True).rename("pnl")
        return pd.concat([daily[["fold", "date", "ic", "ic_liquid"]], pnl], axis=1).groupby(["fold", "date"]).mean()

    a, b = per_date(daily_a), per_date(daily_b)
    joined = a.join(b, lsuffix="_a", rsuffix="_b", how="inner").dropna(subset=["ic_a", "ic_b"])
    diff = (joined["ic_a"] - joined["ic_b"]).to_numpy()
    fold_diff = (joined["ic_a"] - joined["ic_b"]).groupby(level="fold").mean().to_numpy()
    pnl_ok = joined.dropna(subset=["pnl_a", "pnl_b"])
    return {
        "n_days": int(len(joined)),
        "ic_diff_mean": float(diff.mean()),
        "ic_diff_dm_hac": stats_tests.hac_mean_test(diff),
        "ic_diff_block_bootstrap": stats_tests.stationary_bootstrap(diff, n_boot=1000),
        "ic_diff_fold_t_lag0": stats_tests.fold_cluster_test(fold_diff, lags=0),
        "ic_diff_fold_hac_lag1": stats_tests.fold_cluster_test(fold_diff, lags=1),
        "ic_liquid_diff_dm_hac": stats_tests.hac_mean_test((joined["ic_liquid_a"] - joined["ic_liquid_b"]).to_numpy()),
        f"sharpe_diff_simple_turnover_{cost_bps:g}bps_bootstrap": stats_tests.sharpe_difference_bootstrap(
            pnl_ok["pnl_a"].to_numpy(), pnl_ok["pnl_b"].to_numpy(), n_boot=1000
        ),
    }


def seed_fold_table(daily: pd.DataFrame, metric: str = "ic") -> pd.DataFrame:
    """(fold x seed) table of per-fold mean metric, for variance_components."""
    return daily.groupby(["fold", "seed"])[metric].mean().unstack("seed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument(
        "--tag", default="", help="File-name tag of the runs to load, e.g. _iv126 or _shuffle_labels_iv126."
    )
    parser.add_argument("--baseline", default=None, help="Config every other config is compared against.")
    parser.add_argument("--out", type=Path, default=Path("experiments/posthoc"))
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    daily_by_config: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict] = {}
    for cid in args.configs:
        try:
            daily = daily_metrics_from_files(args.pred_dir, cid, tag=args.tag, top_k=args.top_k)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", cid, exc)
            continue
        daily.to_csv(args.out / f"daily_{cid}{args.tag}.csv", index=False)
        daily_by_config[cid] = daily
        summary[cid] = {f"seed_{s}": summarise(d) for s, d in daily.groupby("seed")}
        table = seed_fold_table(daily)
        if table.shape[1] >= 2:
            summary[cid]["ic_variance_components"] = stats_tests.variance_components(table.to_numpy())

    baseline = args.baseline or args.configs[0]
    comparisons = {}
    raw_p = []
    for cid in args.configs:
        if cid == baseline or cid not in daily_by_config or baseline not in daily_by_config:
            continue
        result = compare(daily_by_config[cid], daily_by_config[baseline])
        comparisons[f"{cid}_vs_{baseline}"] = result
        raw_p.append(result["ic_diff_dm_hac"]["p"])
    if raw_p:
        for key, adj in zip(comparisons, stats_tests.bh_adjust(raw_p)):
            comparisons[key]["ic_diff_dm_hac"]["p_bh"] = float(adj)

    with open(args.out / f"posthoc_summary{args.tag}.json", "w") as fh:
        json.dump({"summary": summary, "comparisons": comparisons}, fh, indent=2, default=float)
    logger.info("Wrote %s", args.out / f"posthoc_summary{args.tag}.json")


if __name__ == "__main__":
    main()
