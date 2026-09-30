"""Tests for point-in-time tradability (stage 3) and the paper-protocol defaults."""

import argparse

import numpy as np
import pandas as pd
from src.data_loader import FinancialGraphDataset, build_synthetic_dataframe
from src.data_pipeline.stage1_fetch_splits import eodhd_code
from src.data_pipeline.stage3_engineer_features import (
    FEATURE_CONFIG,
    compute_point_in_time_prices,
    compute_tradability_flags,
    load_split_history,
)
from src.run_ablation import _resolve_protocol
from src.train import load_config


def _split_adjusted_panel() -> pd.DataFrame:
    """One stock trading at $4 until a 4:1 split, then $1; vendor data back-adjusted."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    split_date = dates[30]
    as_traded = np.where(dates < split_date, 4.0, 1.0)
    shares = np.where(dates < split_date, 100_000.0, 400_000.0)
    return pd.DataFrame(
        {
            "Date": dates,
            "Ticker": "XYZ",
            "Close": as_traded / np.where(dates < split_date, 4.0, 1.0),  # vendor: $1 throughout
            "Adjusted_close": as_traded / np.where(dates < split_date, 4.0, 1.0),
            "Volume": shares * np.where(dates < split_date, 4.0, 1.0),
        }
    ), split_date


def test_load_split_history_parses_ratios(tmp_path) -> None:
    pd.DataFrame({"date": ["2020-08-31"], "split": ["4.000000/1.000000"]}).to_csv(tmp_path / "AAPL.csv", index=False)
    pd.DataFrame(columns=["date", "split"]).to_csv(tmp_path / "NOSPLIT.csv", index=False)
    splits = load_split_history(tmp_path)
    assert list(splits["Ticker"]) == ["AAPL"]
    assert splits["ratio"].iloc[0] == 4.0
    assert load_split_history(tmp_path / "missing").empty


def test_point_in_time_prices_undo_later_splits() -> None:
    df, split_date = _split_adjusted_panel()
    splits = pd.DataFrame({"Ticker": ["XYZ"], "split_date": [split_date], "ratio": [4.0]})
    out = compute_point_in_time_prices(df, "Ticker", splits)
    before = out["Date"] < split_date
    np.testing.assert_allclose(out.loc[before, "close_pit"], 4.0)
    np.testing.assert_allclose(out.loc[~before, "close_pit"], 1.0)
    np.testing.assert_allclose(out.loc[before, "volume_pit"], 100_000.0)
    # Dollar volume is invariant to split adjustment.
    np.testing.assert_allclose(out["close_pit"] * out["volume_pit"], out["Close"] * out["Volume"])


def test_tradability_uses_as_traded_price_only_when_enabled() -> None:
    df, split_date = _split_adjusted_panel()
    splits = pd.DataFrame({"Ticker": ["XYZ"], "split_date": [split_date], "ratio": [4.0]})
    df = compute_point_in_time_prices(df, "Ticker", splits)
    config = {**FEATURE_CONFIG, "tradability_min_price": 2.0, "tradability_min_dollar_volume": 0}
    pit = compute_tradability_flags(df, "Ticker", {**config, "tradability_point_in_time": True})
    legacy = compute_tradability_flags(df, "Ticker", {**config, "tradability_point_in_time": False})
    early = df["Date"] < split_date - pd.Timedelta(days=45)
    # It traded at $4 before the split, so it passed a $2 floor at the time...
    assert pit.loc[early, "is_tradable"].all()
    # ...but the back-adjusted price ($1) wrongly failed it in the dissertation's gate.
    assert not legacy.loc[early, "is_tradable"].any()


def test_evaluation_filter_prefers_as_traded_columns() -> None:
    config = load_config(None)
    df = build_synthetic_dataframe(n_tickers=10, n_dates=90, seed=1)
    df["close_pit"] = np.where(df["AssetClass"] == "equity", 3.0, np.nan)
    df["volume_pit"] = np.where(df["AssetClass"] == "equity", 7.0, np.nan)
    item = FinancialGraphDataset("unused", config, dataframe=df)[5]
    assert float(item.close_price.max()) == 3.0 and float(item.volume.max()) == 7.0


def test_eodhd_code_adds_exchange_suffix() -> None:
    assert eodhd_code("AAPL") == "AAPL.US"
    assert eodhd_code("BRK-B.US") == "BRK-B.US"


def _args(**overrides) -> argparse.Namespace:
    base = {"original_protocol": False, "inner_val_days": None, "stop_metric": None, "crn": None, "seeds": None}
    return argparse.Namespace(**{**base, **overrides})


def test_protocol_defaults_come_from_config() -> None:
    config = load_config(None)
    assert _resolve_protocol(_args(), config) == (126, "supervised", True, None)
    assert config["evaluation"]["ablation_seeds"] == [0, 1, 2]


def test_cli_overrides_and_original_protocol() -> None:
    config = load_config(None)
    assert _resolve_protocol(_args(inner_val_days=60, crn=False), config) == (60, "supervised", False, None)
    assert _resolve_protocol(_args(original_protocol=True), config) == (None, "total", False, [42])
