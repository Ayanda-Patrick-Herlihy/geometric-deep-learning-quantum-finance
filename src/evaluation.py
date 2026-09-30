"""Evaluation utilities: metrics and statistical significance.

Provides prediction metrics (MSE, MAE, directional accuracy, IC,
Sharpe, max drawdown), statistical comparison helpers (paired t-test,
bootstrap CI), and a convenience function for evaluating the full
model on a dataset.

Theoretical Context:
    We evaluate the model using standard quantitative finance metrics:
    1. Information Coefficient (IC): The correlation between predicted and
       actual returns (Grinold & Kahn, 1999).
    2. Sharpe Ratio: Risk-adjusted return (mean/std).
    3. Directional Accuracy: The "Hit Ratio" for binary up/down prediction.
    Standard MSE is often insufficient because a model can have low MSE but
    negative correlation with the market.

References:
    Grinold and Kahn (1999) "Active Portfolio Management."
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def mean_squared_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Computes mean squared error between predictions and targets.

    Args:
        predictions: Predicted values (flattened 1-D tensor).
        targets: Actual values (flattened 1-D tensor).

    Returns:
        MSE as a Python float.
    """
    return float(torch.mean((predictions - targets) ** 2).item())


def mean_absolute_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Computes mean absolute error between predictions and targets.

    Args:
        predictions: Predicted values (flattened 1-D tensor).
        targets: Actual values (flattened 1-D tensor).

    Returns:
        MAE as a Python float.
    """
    return float(torch.mean(torch.abs(predictions - targets)).item())


def directional_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Percentage of predictions with correct return sign.

    A prediction is counted as correct when sign(pred) == sign(target).

    Ref: "Hit Ratio" in Quantitative Finance.

    Args:
        predictions: Predicted values (flattened 1-D tensor).
        targets: Actual values (flattened 1-D tensor).

    Returns:
        Directional accuracy as a percentage (0--100).
    """
    correct = (torch.sign(predictions) == torch.sign(targets)).float()
    return float(correct.mean().item()) * 100.0


def information_coefficient(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Spearman rank correlation between predictions and targets.

    The Information Coefficient (IC) measures how well the ranking
    of predicted returns agrees with the ranking of realised returns.

    Ref: Grinold & Kahn (1999), Fundamental Law of Active Management.

    Args:
        predictions: Predicted values (flattened 1-D tensor).
        targets: Actual values (flattened 1-D tensor).

    Returns:
        Spearman rank correlation coefficient.
    """
    predictions = predictions.detach().cpu().flatten()
    targets = targets.detach().cpu().flatten()
    valid_mask = torch.isfinite(predictions) & torch.isfinite(targets)
    if int(valid_mask.sum().item()) < 3:
        return 0.0

    corr = scipy_stats.spearmanr(
        predictions[valid_mask].numpy(),
        targets[valid_mask].numpy(),
    ).correlation
    if corr is None or not np.isfinite(corr):
        return 0.0
    return float(corr)


def compute_daily_pnl(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    close_prices: torch.Tensor | None,
    volumes: torch.Tensor | None,
    top_k: int,
    min_price: float,
    min_dollar_volume: float,
    transaction_cost_bps: float,
    date_label: str,
) -> float | None:
    """Computes daily long-short PnL after evaluation-time filtering.

    Args:
        predictions: Predicted returns for one date.
        targets: Realised returns for one date.
        close_prices: Close prices for one date, when available.
        volumes: Traded volumes for one date, when available.
        top_k: Number of names in each portfolio leg.
        min_price: Minimum close price to include.
        min_dollar_volume: Minimum daily dollar volume to include.
        transaction_cost_bps: Cost per side in basis points.
        date_label: Date string for warning messages.

    Returns:
        Daily PnL after transaction costs, or None when the date is skipped.
    """
    predictions = predictions.detach().cpu().flatten()
    targets = targets.detach().cpu().flatten()
    valid_mask = torch.isfinite(predictions) & torch.isfinite(targets)

    if close_prices is None or volumes is None:
        logger.warning(
            "Date %s is missing close price or volume at evaluation time. "
            "Skipping the evaluation-time universe filter for this date.",
            date_label,
        )
    else:
        close_prices = close_prices.detach().cpu().flatten()
        volumes = volumes.detach().cpu().flatten()

        if not (
            len(predictions)
            == len(targets)
            == len(close_prices)
            == len(volumes)
        ):
            raise ValueError("Evaluation tensors must have matching lengths per date.")

        dollar_volume = close_prices * volumes
        valid_mask = (
            valid_mask
            & torch.isfinite(close_prices)
            & torch.isfinite(volumes)
            & (close_prices >= min_price)
            & (dollar_volume >= min_dollar_volume)
        )

    eligible_count = int(valid_mask.sum().item())
    if eligible_count < 2 * top_k:
        logger.warning(
            "Skipping date %s for daily PnL: only %d stocks pass the "
            "evaluation universe filter, need at least %d.",
            date_label,
            eligible_count,
            2 * top_k,
        )
        return None

    filtered_predictions = predictions[valid_mask]
    filtered_targets = targets[valid_mask]
    long_indices = torch.topk(filtered_predictions, top_k).indices
    short_indices = torch.topk(filtered_predictions, top_k, largest=False).indices

    gross_pnl = (
        filtered_targets[long_indices].mean().item()
        - filtered_targets[short_indices].mean().item()
    )
    transaction_cost = 2.0 * float(transaction_cost_bps) / 10_000.0
    return float(gross_pnl - transaction_cost)


def sharpe_daily(
    daily_pnl: list[float] | np.ndarray,
) -> float:
    """Computes the non-annualised Sharpe ratio from daily PnL.

    Args:
        daily_pnl: Sequence of daily profit-and-loss values.

    Returns:
        Daily Sharpe ratio, or 0.0 if insufficient data.
    """
    pnl = np.asarray(daily_pnl, dtype=np.float64)
    if len(pnl) < 2:
        return 0.0

    mean_pnl = pnl.mean()
    std_pnl = pnl.std(ddof=1)

    if std_pnl < 1e-12:
        return 0.0

    return float(mean_pnl / std_pnl)


def max_drawdown(daily_pnl: list[float] | np.ndarray) -> float:
    """Worst peak-to-trough decline of cumulative PnL.

    Computes the maximum drawdown of the cumulative PnL path in raw
    return units. Returns 0.0 when there is no drawdown or the series
    is too short.

    Args:
        daily_pnl: Sequence of daily profit-and-loss values.

    Returns:
        Maximum drawdown (non-negative float).
    """
    pnl = np.asarray(daily_pnl, dtype=np.float64)
    if len(pnl) < 2:
        return 0.0

    cum_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum_pnl)

    drawdowns = running_max - cum_pnl
    return float(drawdowns.max())


def compute_all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    datewise_predictions: list[torch.Tensor],
    datewise_targets: list[torch.Tensor],
    daily_pnl: list[float] | np.ndarray,
) -> dict[str, float]:
    """Computes the full evaluation metric suite.

    Args:
        predictions: Predicted values (flattened 1-D tensor).
        targets: Actual values (flattened 1-D tensor).
        datewise_predictions: Per-date prediction tensors before portfolio
            filtering.
        datewise_targets: Per-date realised return tensors before portfolio
            filtering.
        daily_pnl: Daily Top-K long-short PnL after costs.

    Returns:
        Dictionary mapping metric names to float values.
    """
    if len(predictions) == 0:
        logger.warning("Empty predictions passed to compute_all_metrics.")
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "ic": float("nan"),
            "sharpe_daily": float("nan"),
            "sharpe_annual": float("nan"),
            "max_drawdown": float("nan"),
        }

    # Ensure tensors are on CPU and detached
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()

    valid_mask = torch.isfinite(predictions) & torch.isfinite(targets)
    valid_predictions = predictions[valid_mask]
    valid_targets = targets[valid_mask]
    if len(valid_predictions) == 0:
        logger.warning("No finite prediction-target pairs available for metrics.")
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "ic": float("nan"),
            "sharpe_daily": float("nan"),
            "sharpe_annual": float("nan"),
            "max_drawdown": float("nan"),
        }

    mse_val = mean_squared_error(valid_predictions, valid_targets)
    mae_val = mean_absolute_error(valid_predictions, valid_targets)
    da_val = directional_accuracy(valid_predictions, valid_targets)

    date_ics: list[float] = []
    for date_predictions, date_targets in zip(datewise_predictions, datewise_targets):
        date_predictions = date_predictions.detach().cpu().flatten()
        date_targets = date_targets.detach().cpu().flatten()
        date_mask = torch.isfinite(date_predictions) & torch.isfinite(date_targets)
        if int(date_mask.sum().item()) < 3:
            continue
        date_ics.append(
            information_coefficient(
                date_predictions[date_mask],
                date_targets[date_mask],
            )
        )
    ic_val = float(np.mean(date_ics)) if date_ics else float("nan")

    pnl_arr = np.asarray(daily_pnl, dtype=np.float64)
    finite_pnl = pnl_arr[np.isfinite(pnl_arr)]
    if finite_pnl.size >= 2:
        sr_daily = sharpe_daily(finite_pnl)
        sr_annual = float(sr_daily * np.sqrt(252.0))
        mdd_val = max_drawdown(finite_pnl)
    else:
        sr_daily = float("nan")
        sr_annual = float("nan")
        mdd_val = float("nan")

    if np.isfinite(sr_daily) and np.isfinite(sr_annual):
        if sr_annual > 3.0 and sr_daily < 0.20:
            logger.warning(
                "Annualised Sharpe %.2f is high while daily Sharpe is only %.2f; "
                "the signal may be too weak to be economically meaningful.",
                sr_annual,
                sr_daily,
            )

    return {
        "mse": mse_val,
        "mae": mae_val,
        "directional_accuracy": da_val,
        "ic": ic_val,
        "sharpe_daily": sr_daily,
        "sharpe_annual": sr_annual,
        "max_drawdown": mdd_val,
    }


def paired_t_test(
    series_a: list[float] | np.ndarray,
    series_b: list[float] | np.ndarray,
) -> dict[str, float]:
    """Paired t-test between two per-fold metric series.

    Tests H0: mean(A) == mean(B) against H1: mean(A) != mean(B).

    Args:
        series_a: Metric values from configuration A (one per fold).
        series_b: Metric values from configuration B (one per fold).

    Returns:
        Dictionary with 't_statistic' and 'p_value'.

    Raises:
        ValueError: If the series have different lengths or fewer than
            two observations.
    """
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Series lengths differ: {len(a)} vs {len(b)}.")
    if len(a) < 2:
        raise ValueError(f"Need at least 2 observations for t-test, got {len(a)}.")

    # Filter out NaN pairs
    valid = np.isfinite(a) & np.isfinite(b)
    a_clean = a[valid]
    b_clean = b[valid]

    if len(a_clean) < 2:
        logger.warning("Fewer than 2 finite observation pairs for t-test.")
        return {"t_statistic": float("nan"), "p_value": float("nan")}

    t_stat, p_val = scipy_stats.ttest_rel(a_clean, b_clean)
    return {"t_statistic": float(t_stat), "p_value": float(p_val)}


def bootstrap_confidence_interval(
    series_a: list[float] | np.ndarray,
    series_b: list[float] | np.ndarray,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for the difference mean(A) - mean(B).

    Useful for comparing e.g. Sharpe ratio of A8 vs an ablation.

    Args:
        series_a: Metric values from configuration A (one per fold/seed).
        series_b: Metric values from configuration B.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'mean_diff', 'ci_lower', 'ci_upper'.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)

    # Filter NaN pairs
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        logger.warning("Fewer than 2 finite observations for bootstrap CI.")
        return {
            "mean_diff": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    diffs = a - b
    observed_diff = float(diffs.mean())

    # Bootstrap resampling
    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    n = len(diffs)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = diffs[idx].mean()

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    return {
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


@torch.no_grad()
def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    top_k: int | None = None,
    return_debug: bool = False,
    debug_pnl: bool = False,
    dump_pnl_path: str | Path | None = None,
    dump_predictions_path: str | Path | None = None,
) -> dict[str, float] | tuple[dict[str, float], dict[str, Any]]:
    """Evaluates a model across all dates in a dataset.

    Collects predictions and targets, then computes the full metric
    suite. Sharpe and max drawdown are derived only from the per-day
    Top-K long/short PnL series; days with fewer than 2*top_k assets
    are skipped for PnL so the series remains consistent.

    Args:
        model: Trained model accepting the flat PyG batch format.
        dataset: Iterable of PyG Data objects.
        device: Compute device for inference.
        top_k: Number of assets to go long/short (default: 50).
        return_debug: If True, also return a dict with predictions, targets,
            and daily_pnl for offline debugging.
        debug_pnl: If True, log summary statistics of daily_pnl.
        dump_pnl_path: If provided, save daily_pnl array to this path (npy format).
        dump_predictions_path: If provided, save one row per (date, supervised
            stock) with prediction, targets, price and volume to this parquet
            file so every metric can be recomputed offline (see posthoc_eval.py).

    Returns:
        Dictionary of named evaluation metrics. If return_debug is True,
        returns (metrics_dict, debug_dict) where debug_dict has keys
        "predictions", "targets", "daily_pnl".
    """
    from src.layers.graph_utils import build_batched_graphs_gpu

    model.eval()
    eval_cfg = getattr(dataset, "_config", {}).get("evaluation", {})
    graph_cfg = getattr(dataset, "_config", {}).get("model", {}).get(
        "graph_construction",
        {},
    )
    portfolio_top_k = int(top_k if top_k is not None else eval_cfg.get("top_k", 50))
    min_price = float(eval_cfg.get("eval_min_price", 5.0))
    min_dollar_volume = float(eval_cfg.get("eval_min_dollar_volume", 1_000_000.0))
    transaction_cost_bps = float(eval_cfg.get("transaction_cost_bps", 20.0))
    graph_threshold = float(graph_cfg["correlation_threshold"])
    max_neighbours_cfg = graph_cfg.get("max_neighbours")
    max_neighbours = int(max_neighbours_cfg) if max_neighbours_cfg is not None else None

    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    datewise_preds: list[torch.Tensor] = []
    datewise_targets: list[torch.Tensor] = []
    daily_pnl: list[float] = []
    dump_frames: list[dict[str, np.ndarray]] = []

    for idx in range(len(dataset)):
        data = dataset[idx]
        date_label = str(dataset.valid_dates[idx].date())

        if data.target_mask.sum() == 0:
            continue

        data = data.to(device)
        x = data.x                     # (N, D)
        edge_index = data.edge_index   # (2, E)
        edge_weight = data.edge_attr   # (E,)
        context = data.context         # (1, C)
        targets = data.y               # (N,)
        target_mask = data.target_mask  # (N,)
        close_prices = getattr(data, "close_price", None)
        volumes = getattr(data, "volume", None)
        batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Build correlation graph on GPU for this date
        past_returns = getattr(data, "past_returns", None)
        if past_returns is not None:
            edge_index, edge_weight = build_batched_graphs_gpu(
                past_returns,
                batch_vec,
                threshold=graph_threshold,
                max_neighbours=max_neighbours,
            )

        x_seq = getattr(data, "x_seq", None)

        output = model(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            context=context,
            batch=batch_vec,
            x_seq=x_seq,
        )
        preds = output["predictions"].squeeze(-1)  # (N,)

        masked_preds = preds[target_mask]
        masked_targets = targets[target_mask]
        masked_close = close_prices[target_mask] if close_prices is not None else None
        masked_volume = volumes[target_mask] if volumes is not None else None

        if dump_predictions_path is not None:
            n_masked = int(target_mask.sum().item())
            ticker_id = getattr(data, "ticker_id", None)
            y_skip1 = getattr(data, "y_skip1", None)
            nan_col = np.full(n_masked, np.nan, dtype=np.float32)
            dump_frames.append(
                {
                    "date": np.full(n_masked, np.datetime64(dataset.valid_dates[idx]), dtype="datetime64[ns]"),
                    "ticker_id": ticker_id[target_mask].cpu().numpy() if ticker_id is not None else np.full(n_masked, -1),
                    "pred": masked_preds.float().cpu().numpy(),
                    "y": masked_targets.float().cpu().numpy(),
                    "y_skip1": y_skip1[target_mask].cpu().numpy() if y_skip1 is not None else nan_col,
                    "close": masked_close.cpu().numpy() if masked_close is not None else nan_col,
                    "volume": masked_volume.cpu().numpy() if masked_volume is not None else nan_col,
                }
            )

        # Store for pointwise metrics and date-wise IC.
        all_preds.append(masked_preds.cpu())
        all_targets.append(masked_targets.cpu())
        datewise_preds.append(masked_preds.cpu())
        datewise_targets.append(masked_targets.cpu())

        pnl_value = compute_daily_pnl(
            predictions=masked_preds,
            targets=masked_targets,
            close_prices=masked_close,
            volumes=masked_volume,
            top_k=portfolio_top_k,
            min_price=min_price,
            min_dollar_volume=min_dollar_volume,
            transaction_cost_bps=transaction_cost_bps,
            date_label=date_label,
        )
        if pnl_value is not None:
            daily_pnl.append(pnl_value)

    # Diagnostic logging and dumping
    if debug_pnl or dump_pnl_path:
        pnl_arr = np.array(daily_pnl)
        n_days = len(pnl_arr)
        if n_days > 0:
            mean_pnl = float(pnl_arr.mean())
            std_pnl = float(pnl_arr.std(ddof=1)) if n_days > 1 else 0.0
            min_pnl = float(pnl_arr.min())
            max_pnl = float(pnl_arr.max())
            pos_frac = float((pnl_arr > 0).mean())
            logger.info(
                "Eval PnL stats: n_days=%d, mean=%.6f, std=%.6f, min=%.6f, max=%.6f, pos_frac=%.2f",
                n_days,
                mean_pnl,
                std_pnl,
                min_pnl,
                max_pnl,
                pos_frac,
            )
        else:
            logger.info("Eval PnL stats: n_days=0")

    if dump_pnl_path and daily_pnl:
        try:
            output_path = Path(dump_pnl_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, np.array(daily_pnl))
            logger.info("Dumped daily_pnl to %s", output_path)
        except Exception as e:
            logger.error("Failed to dump daily_pnl to %s: %s", dump_pnl_path, e)

    if dump_predictions_path is not None and dump_frames:
        import pandas as pd

        frame = pd.DataFrame({k: np.concatenate([f[k] for f in dump_frames]) for k in dump_frames[0]})
        pivot_cols = getattr(dataset, "return_pivot", None)
        if pivot_cols is not None:
            tickers = np.asarray(pivot_cols.columns, dtype=object)
            valid = frame["ticker_id"].to_numpy() >= 0
            frame["ticker"] = None
            frame.loc[valid, "ticker"] = tickers[frame.loc[valid, "ticker_id"].to_numpy()]
        output_path = Path(dump_predictions_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        logger.info("Dumped %d prediction rows to %s", len(frame), output_path)

    if not all_preds:
        logger.warning("No valid predictions collected during evaluation.")
        out = {
            "mse": float("nan"),
            "mae": float("nan"),
            "directional_accuracy": float("nan"),
            "ic": float("nan"),
            "sharpe_daily": float("nan"),
            "sharpe_annual": float("nan"),
            "max_drawdown": float("nan"),
        }
        if return_debug:
            return out, {"predictions": None, "targets": None, "daily_pnl": []}
        return out

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    out = compute_all_metrics(
        predictions=preds_cat,
        targets=targets_cat,
        datewise_predictions=datewise_preds,
        datewise_targets=datewise_targets,
        daily_pnl=daily_pnl,
    )

    if return_debug:
        debug = {
            "predictions": preds_cat.numpy(),
            "targets": targets_cat.numpy(),
            "daily_pnl": daily_pnl,
        }
        return out, debug
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.manual_seed(7)
    toy_predictions = torch.randn(10)
    toy_targets = torch.randn(10)
    toy_daily_pnl = [0.01, -0.005, 0.0, 0.02]
    toy_metrics = compute_all_metrics(
        toy_predictions,
        toy_targets,
        [toy_predictions],
        [toy_targets],
        toy_daily_pnl,
    )
    logger.info("Manual check daily Sharpe: %.6f", toy_metrics["sharpe_daily"])
    logger.info("Manual check max drawdown: %.6f", toy_metrics["max_drawdown"])
