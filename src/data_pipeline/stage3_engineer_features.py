"""Transforms the merged master dataset into model-ready features.
Computes log returns, multi-scale volatility, skip-month momentum,
liquidity signals, and price-level features for the equity universe.
Yield curve decomposition features are computed for the quantum layer.

Theoretical Context:
    Raw financial prices are non-stationary and unsuitable for deep learning.
    We transform data into stationary signals using Log Returns (Campbell et al., 1997)
    for additivity, Realised Volatility (Andersen et al., 2003) for risk scaling,
    skip-month Momentum (Jegadeesh & Titman, 1993) for behavioural trend capture,
    the Amihud (2002) illiquidity ratio for price impact, the Corwin-Schultz (2012)
    high-low spread estimator for transaction cost proxying, the 52-week high ratio
    (George & Hwang, 2004) for bounded price-level context, and Yield Curve
    decomposition (Diebold & Li, 2006) to capture macro-regime context, ensuring
    all inputs are numerically stable for geometric projection.

    Higher-moment features (skewness, kurtosis, tail_risk) are intentionally
    excluded. Empirical analysis of the full dataset revealed kurtosis of 2042
    at the universe level, producing extreme rolling estimates from 60-day windows
    that dominate feature vector L2 norms and collapse the hypersphere projection.
    The liquidity and price-level feature group replaces this with bounded,
    stable signals that are architecturally cleanly separated from the quantum
    layer's regime detection role.

Model Feature Set (MODEL_FEATURE_COLS — 12 features):
    Returns:    log_return_5d, log_return_20d
    Volatility: vol_10d, vol_22d, vol_60d
    Momentum:   mom_3m, mom_6m, mom_12m
    Liquidity:  volume_ratio, amihud_illiquidity
    Price Level: high_low_spread_22d, price_to_52w_high

    Note: log_return_1d is computed and retained as the supervised prediction
    target but is not included in MODEL_FEATURE_COLS.

References:
    Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and
        Selling Losers. Journal of Finance, 48(1), 65-91.
    Andersen, T. G. et al. (2003). Modeling and Forecasting Realized
        Volatility. Econometrica, 71(2), 579-625.
    Amihud, Y. (2002). Illiquidity and stock returns: cross-section and
        time-series effects. Journal of Financial Markets, 5(1), 31-56.
    Corwin, S. A. & Schultz, P. (2012). A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices. Journal of
        Finance, 67(2), 719-760.
    George, T. J. & Hwang, C.-Y. (2004). The 52-Week High and Momentum
        Investing. Journal of Finance, 59(5), 2145-2176.
    Nelson, C. R. & Siegel, A. F. (1987). Parsimonious Modeling of Yield
        Curves. Journal of Business, 60(4), 473-489.
    Campbell, Lo, & MacKinlay (1997). The Econometrics of Financial Markets.
    Huber (1981). Robust Statistics.
    Diebold & Li (2006). Forecasting the term structure of government bond yields.
"""

from pathlib import Path
from typing import Optional

import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Number of equity tickers per batch to limit peak memory (avoids OOM on large universes)
EQUITY_TICKER_CHUNK_SIZE = 2000

# The 12 features the model reads from the fact table.
# All other columns are retained in the parquet for diagnostics and evaluation
# but are not passed to any model layer. This is the single source of truth
# for the feature mask used in data_loader.py and any evaluation scripts.
MODEL_FEATURE_COLS: list[str] = [
    # Returns — where has this stock been recently
    "log_return_5d",
    "log_return_20d",
    # Volatility — how risky is it at multiple time scales
    "vol_10d",
    "vol_22d",
    "vol_60d",
    # Momentum — is the trend persistent (Jegadeesh & Titman, 1993)
    "mom_3m",
    "mom_6m",
    "mom_12m",
    # Liquidity — how easily can it be traded
    "volume_ratio",
    "amihud_illiquidity",
    # Price level — where is it relative to its own history
    "high_low_spread_22d",
    "price_to_52w_high",
]

# Feature engineering parameters
FEATURE_CONFIG = {
    # log_return_1d is always computed (it is the supervised target).
    # log_return_5d and log_return_20d are model features.
    "return_horizons": [1, 5, 20],
    "volatility_windows": [10, 22, 60],
    # Momentum: 21 days ~ 1 month, 63 ~ 3 months, 126 ~ 6 months, 252 ~ 12 months
    # Ref: Jegadeesh & Titman (1993), skip-month formulation
    "momentum_horizons": {
        "3m": {"formation": 63, "skip": 21},
        "6m": {"formation": 126, "skip": 21},
        "12m": {"formation": 252, "skip": 21},
    },
    # Volume ratio: today's volume vs rolling N-day average
    "volume_ratio_window": 20,
    # High-low spread: rolling window for Corwin-Schultz estimator
    "spread_window": 22,
    # 52-week high: rolling maximum price window (252 trading days)
    "high_window": 252,
    "yield_series_10y": "US_10Y",
    "yield_series_2y": "US_02Y",
    "zscore_window": 252,
    # Winsorisation (applied immediately after log return computation) 
    # Hard absolute caps on log returns to catch data errors.
    # ln(3) ≈ 1.099; no legitimate equity triples or zeros in a single day.
    "winsorise_caps": {
        1: 1.0,   # 1-day: ±100% log return (hard cap)
        5: 1.5,   # 5-day: ±150%
        20: 2.0,  # 20-day: ±200%
    },
    # Universe quality gating 
    "price_cap": 900000.0,                    # Accommodates BRK.A but catches obvious data errors
    "tradability_min_price": 1.0,            # rolling median price floor ($)
    "tradability_min_dollar_volume": 50_000,  # rolling median dollar volume floor ($)
    "tradability_max_zero_vol_frac": 0.50,    # rolling zero-volume fraction ceiling
    "tradability_price_window": 22,           # rolling window for median price
    "tradability_dollar_vol_window": 22,      # rolling window for median dollar volume
    "tradability_zero_vol_window": 63,        # rolling window for zero-volume fraction
    # Build the price and dollar-volume gates from point-in-time (as-traded)
    # prices. EODHD's Close/Volume are back-adjusted for every later split and
    # Adjusted_close also for later dividends, so gating on them uses future
    # corporate actions. False reproduces the dissertation's fact table.
    "tradability_point_in_time": True,
    # Directory of per-ticker split histories from stage1_fetch_splits.py.
    "splits_dir": None,
    # Foreign yield series for Group B/C 
    "foreign_yields": {
        "GER_10Y": {"ticker": "GER_10Y", "label": "DE"},
        "JPN_10Y": {"ticker": "JPN_10Y", "label": "JP"},
        "UK_10Y":  {"ticker": "UK_10Y",  "label": "UK"},
    },
}


def create_daily_date_spine(
    start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DatetimeIndex:
    """Create a business-day date spine for frequency alignment.

    Args:
        start_date: First date in spine.
        end_date: Last date in spine.

    Returns:
        DatetimeIndex of business days.
    """
    return pd.bdate_range(start=start_date, end=end_date, freq="B")


def align_to_daily(
    df: pd.DataFrame, date_spine: pd.DatetimeIndex, value_columns: list[str]
) -> pd.DataFrame:
    """Align data to daily frequency using forward-fill.

    Preserves the "information as known at time t" principle for
    data that arrives at lower frequencies (e.g. monthly yields).

    Args:
        df: DataFrame with Date column (may be monthly).
        date_spine: Target daily DatetimeIndex.
        value_columns: Columns to align.

    Returns:
        DataFrame aligned to daily frequency.
    """
    df_working = df.copy()
    if "Date" in df_working.columns:
        df_working = df_working.set_index("Date")

    if not isinstance(df_working.index, pd.DatetimeIndex):
        df_working.index = pd.to_datetime(df_working.index)

    df_daily = df_working.reindex(date_spine)

    # Forward-fill preserves last-known value
    df_daily = df_daily.ffill()

    df_daily = df_daily.reset_index().rename(columns={"index": "Date"})
    return df_daily


def compute_log_returns(
    df: pd.DataFrame, price_col: str, ticker_col: str, horizons: list[int]
) -> pd.DataFrame:
    """Compute log returns at multiple horizons using grouped shifts.

    Ref: Campbell, Lo & MacKinlay (1997), Chapter 1, Section 1.4.1.

    Args:
        df: DataFrame sorted by [ticker, Date].
        price_col: Column containing prices.
        ticker_col: Column identifying each security.
        horizons: List of look-back periods in trading days.

    Returns:
        DataFrame with added log_return_*d columns.
    """
    df = df.copy()
    grouped = df.groupby(ticker_col)[price_col]

    for h in horizons:
        col_name = f"log_return_{h}d"
        prices = df[price_col].replace(0, np.nan)
        shifted_prices = grouped.shift(h).replace(0, np.nan)

        # Intent: Stationarity (Campbell et al., 1997).
        # Raw prices are non-stationary (they drift), which breaks neural networks.
        # Log returns are additive and statistically stationary, making them
        # suitable for geometric projection.
        df[col_name] = np.log(prices / shifted_prices)

    return df


def winsorise_returns(
    df: pd.DataFrame,
    ticker_col: str,
    caps: dict[int, float],
) -> pd.DataFrame:
    """Clip log returns to hard absolute caps per horizon.

    Raw log returns can contain data errors (e.g. a max single-day return
    of +2.5 billion percent found in the EDA). These propagate into every
    downstream feature — volatility, momentum, and the training target.
    This function applies symmetric hard caps immediately after return
    computation and before any rolling window features are derived.

    The raw (unclipped) returns are preserved in separate columns for
    diagnostic purposes.

    Ref: Huber (1981), Robust Statistics — rationale for hard capping.

    Args:
        df: DataFrame containing log_return_*d columns.
        ticker_col: Column identifying each security (kept for interface
            consistency with other feature functions).
        caps: Mapping of horizon (int days) to absolute cap (float).
            E.g. {1: 1.0, 5: 1.5, 20: 2.0}.

    Returns:
        DataFrame with clipped log_return_*d columns and new
        log_return_*d_raw columns preserving originals.
    """
    df = df.copy()

    for horizon, cap in caps.items():
        col = f"log_return_{horizon}d"
        if col not in df.columns:
            logger.warning("Column %s not found — skipping winsorisation.", col)
            continue

        raw_col = f"{col}_raw"
        df[raw_col] = df[col].copy()

        n_clipped = int(((df[col] < -cap) | (df[col] > cap)).sum())
        df[col] = df[col].clip(lower=-cap, upper=cap)

        if n_clipped > 0:
            logger.info(
                "  Winsorised %s: %d values clipped to [%.2f, +%.2f].",
                col,
                n_clipped,
                -cap,
                cap,
            )

    return df


def load_split_history(splits_dir: Path | None) -> pd.DataFrame:
    """Loads per-ticker split histories written by stage1_fetch_splits.py.

    Each ``<TICKER>.csv`` has columns ``date`` (first as-traded date after the
    split) and ``split`` ("new/old", e.g. "4.000000/1.000000").

    Returns:
        DataFrame with columns Ticker, split_date and ratio (new/old shares).
        Empty if the directory is missing.
    """
    if splits_dir is None or not Path(splits_dir).exists():
        return pd.DataFrame(columns=["Ticker", "split_date", "ratio"])
    frames = []
    for path in sorted(Path(splits_dir).glob("*.csv")):
        raw = pd.read_csv(path)
        if raw.empty:
            continue
        parts = raw["split"].astype(str).str.split("/", expand=True).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "Ticker": path.stem,
                    "split_date": pd.to_datetime(raw["date"]),
                    "ratio": parts[0] / parts[1],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["Ticker", "split_date", "ratio"])
    splits = pd.concat(frames, ignore_index=True)
    return splits[np.isfinite(splits["ratio"]) & (splits["ratio"] > 0)]


def compute_point_in_time_prices(
    df: pd.DataFrame,
    ticker_col: str,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Adds as-traded close and volume by undoing later split adjustments.

    EODHD's ``Close`` and ``Volume`` are adjusted for every split that happens
    after the row's date. The as-traded values multiply (divide) them by the
    product of the ratios of all splits dated after the row:
    ``close_pit = Close * F(t)``, ``volume_pit = Volume / F(t)``. Dollar
    volume ``Close * Volume`` is unaffected by splits.

    Args:
        df: DataFrame with Date, Close and Volume columns.
        ticker_col: Column identifying each security.
        splits: Output of ``load_split_history``.

    Returns:
        DataFrame with ``close_pit``, ``volume_pit`` and ``split_factor_after``.
    """
    df = df.copy()
    factor = np.ones(len(df))
    if not splits.empty:
        dates = df["Date"].to_numpy()
        tickers = df[ticker_col].astype(str).to_numpy()
        for ticker, group in splits.groupby("Ticker"):
            rows = np.flatnonzero(tickers == ticker)
            if rows.size == 0:
                continue
            for split_date, ratio in zip(group["split_date"], group["ratio"]):
                before = rows[dates[rows] < np.datetime64(split_date)]
                factor[before] *= ratio
    df["split_factor_after"] = factor
    df["close_pit"] = df["Close"] * factor
    df["volume_pit"] = df["Volume"] / factor
    return df


def compute_tradability_flags(
    df: pd.DataFrame,
    ticker_col: str,
    config: dict,
) -> pd.DataFrame:
    """Compute per-row tradability flags for universe quality gating.

    Generates a boolean ``is_tradable`` column that downstream consumers
    (data_loader.py) use to mask targets and PnL. Tickers that fail
    tradability remain in the dataset as graph nodes — the GNN can still
    learn from their feature patterns and correlations — but they are
    excluded from the supervised loss and any portfolio construction.

    Criteria (all must be True simultaneously for is_tradable=True):
        1. Rolling median price >= tradability_min_price. With
           tradability_point_in_time the price is the as-traded close_pit;
           otherwise (dissertation) the back-adjusted Adjusted_close.
        2. Rolling median dollar volume >= tradability_min_dollar_volume,
           from Close * Volume (split-invariant, no dividend adjustment) when
           point-in-time, otherwise Adjusted_close * Volume.
        3. Rolling zero-volume fraction < tradability_max_zero_vol_frac.
        4. Adjusted_close is not NaN on this row.

    Ref: Amihud (2002), Illiquidity and stock returns — motivation for
        dollar volume gating.

    Args:
        df: DataFrame sorted by [ticker_col, Date] with Adjusted_close
            and Volume columns present.
        ticker_col: Column identifying each security.
        config: FEATURE_CONFIG dictionary.

    Returns:
        DataFrame with added columns:
            rolling_median_price (float): diagnostic rolling price.
            rolling_median_dollar_vol (float): diagnostic rolling dollar vol.
            rolling_zero_vol_frac (float): diagnostic zero-volume fraction.
            is_tradable (bool): final gating flag.
    """
    df = df.copy()
    grouped = df.groupby(ticker_col, observed=True)

    price_window = config.get("tradability_price_window", 22)
    dv_window = config.get("tradability_dollar_vol_window", 22)
    zv_window = config.get("tradability_zero_vol_window", 63)
    min_price = config.get("tradability_min_price", 1.0)
    min_dv = config.get("tradability_min_dollar_volume", 50_000)
    max_zv = config.get("tradability_max_zero_vol_frac", 0.50)

    point_in_time = config.get("tradability_point_in_time", False) and "close_pit" in df.columns
    price_source = "close_pit" if point_in_time else "Adjusted_close"
    dv_price_source = "Close" if point_in_time else "Adjusted_close"

    # Rolling median price
    df["rolling_median_price"] = grouped[price_source].transform(
        lambda s: s.rolling(window=price_window, min_periods=1).median()
    )

    # Rolling median dollar volume
    df["_dollar_vol"] = df[dv_price_source].abs() * df["Volume"]
    df["rolling_median_dollar_vol"] = grouped["_dollar_vol"].transform(
        lambda s: s.rolling(window=dv_window, min_periods=1).median()
    )
    df.drop(columns=["_dollar_vol"], inplace=True)

    # Rolling zero-volume fraction
    df["_is_zero_vol"] = (df["Volume"] == 0).astype(float)
    df["rolling_zero_vol_frac"] = grouped["_is_zero_vol"].transform(
        lambda s: s.rolling(window=zv_window, min_periods=1).mean()
    )
    df.drop(columns=["_is_zero_vol"], inplace=True)

    # Adjusted_close availability
    has_adj_close = df["Adjusted_close"].notna()

    df["is_tradable"] = (
        (df["rolling_median_price"] >= min_price)
        & (df["rolling_median_dollar_vol"] >= min_dv)
        & (df["rolling_zero_vol_frac"] < max_zv)
        & has_adj_close
    )

    n_total = len(df)
    n_tradable = int(df["is_tradable"].sum())
    logger.info(
        "  Tradability: %d / %d rows (%.1f%%) flagged as tradable.",
        n_tradable,
        n_total,
        100.0 * n_tradable / max(n_total, 1),
    )

    return df


def compute_forward_target(
    df: pd.DataFrame,
    ticker_col: str,
    return_col: str = "log_return_1d",
) -> pd.DataFrame:
    """Compute the 1-step-ahead supervised prediction target.

    Shifts ``return_col`` backward by 1 within each ticker group to
    create ``target_fwd``: the return realised on the next trading day.
    This is the value the model is trained to predict.

    Previously computed at runtime in data_loader.py (line 278). Moved
    here so the Parquet file contains the target natively, eliminating a
    large groupby().shift() at dataset initialisation time.

    Args:
        df: DataFrame sorted by [ticker_col, Date].
        ticker_col: Column identifying each security.
        return_col: Column to shift (default: ``log_return_1d``).

    Returns:
        DataFrame with added ``target_fwd`` column. The last row per
        ticker is always NaN by construction (no known next-day return).
    """
    df = df.copy()
    df["target_fwd"] = (
        df.groupby(ticker_col, observed=True)[return_col].shift(-1)
    )

    n_valid = int(df["target_fwd"].notna().sum())
    logger.info(
        "  Forward target: %d valid values (last day per ticker is NaN by design).",
        n_valid,
    )

    return df


def compute_realised_volatility(
    df: pd.DataFrame, return_col: str, ticker_col: str, windows: list[int]
) -> pd.DataFrame:
    """Compute rolling standard deviation of returns at multiple scales.

    Ref: Andersen et al. (2003), Section 2, Eq 1 & 2.

    Args:
        df: DataFrame sorted by [ticker, Date].
        return_col: Column containing log returns.
        ticker_col: Column identifying each security.
        windows: List of rolling window sizes in trading days.

    Returns:
        DataFrame with added vol_*d columns.
    """
    df = df.copy()
    grouped = df.groupby(ticker_col)[return_col]

    for w in windows:
        col_name = f"vol_{w}d"
        # Intent: Volatility Contagion (Andersen et al., 2003).
        # We compute volatility at multiple scales (short, monthly, quarterly)
        # to allow the Graph Layer to detect when risk "spills over" from
        # one asset to another.
        rolling_std = grouped.rolling(window=w).std()
        # Drop the groupby level to align with the original index
        rolling_std_aligned = rolling_std.reset_index(level=0, drop=True)
        df[col_name] = rolling_std_aligned

    return df


def compute_momentum(
    df: pd.DataFrame, price_col: str, ticker_col: str, horizons: dict
) -> pd.DataFrame:
    """Compute skip-month momentum at multiple horizons.

    Ref: Jegadeesh & Titman (1993), Section I ("Data and Methodology").

    Args:
        df: DataFrame sorted by [ticker, Date].
        price_col: Column containing prices.
        ticker_col: Column identifying each security.
        horizons: Dict mapping horizon name to formation/skip params.

    Returns:
        DataFrame with added mom_* columns.
    """
    df = df.copy()
    grouped = df.groupby(ticker_col)[price_col]

    for name, params in horizons.items():
        col_name = f"mom_{name}"
        formation = params["formation"]
        skip = params["skip"]

        # Price at t-skip divided by price at t-formation, minus 1
        p_skip = grouped.shift(skip)
        p_formation = grouped.shift(formation)

        # Intent: Behavioural Momentum (Jegadeesh & Titman, 1993).
        # We skip the most recent month (21 days) to avoid short-term
        # mean reversion noise, capturing the true underlying trend.
        df[col_name] = (p_skip / p_formation) - 1

    return df


def compute_volume_features(
    df: pd.DataFrame,
    ticker_col: str,
    return_col: str,
    volume_ratio_window: int = 20,
) -> pd.DataFrame:
    """Compute volume ratio and Amihud illiquidity measure.

    Volume ratio captures short-term volume surges relative to the
    recent average. The Amihud (2002) illiquidity ratio captures the
    price impact per unit of trading volume — the average absolute
    return per unit of dollar volume. Higher values indicate a stock
    moves more per unit of volume, reflecting illiquidity.

    Both signals are bounded and numerically stable alternatives to
    rolling skewness and kurtosis for capturing distributional extremes
    in the cross-section.

    Ref: Amihud, Y. (2002). Illiquidity and stock returns: cross-section
        and time-series effects. Journal of Financial Markets, 5(1), 31-56.

    Args:
        df: DataFrame sorted by [ticker, Date] with Volume and return_col.
        ticker_col: Column identifying each security.
        return_col: Column containing log returns (log_return_1d).
        volume_ratio_window: Rolling window for average volume baseline.

    Returns:
        DataFrame with added volume_ratio and amihud_illiquidity columns.
    """
    df = df.copy()

    # Replace zero volume with NaN to avoid division errors
    volume = df["Volume"].replace(0, np.nan)

    # Volume ratio: today's volume divided by rolling N-day average volume.
    # A ratio > 1 indicates above-average activity; > 2 flags a volume surge.
    rolling_avg_vol = (
        df.groupby(ticker_col, observed=True)["Volume"]
        .transform(lambda x: x.replace(0, np.nan).rolling(window=volume_ratio_window).mean())
    )
    df["volume_ratio"] = volume / rolling_avg_vol

    # Amihud illiquidity: |return| / volume.
    # Multiply by 1e6 to bring to a numerically reasonable scale
    # (raw values are very small for liquid large-cap stocks).
    # Intent: Price Impact (Amihud, 2002).
    # A stock that moves 1% on 1 million shares of volume is more liquid
    # than one that moves 1% on 10,000 shares. This ratio captures that
    # difference and feeds the graph layer's liquidity-based edge weighting.
    abs_return = df[return_col].abs()
    df["amihud_illiquidity"] = (abs_return / volume) * 1e6

    return df


def compute_price_level_features(
    df: pd.DataFrame,
    price_col: str,
    ticker_col: str,
    high_window: int = 252,
) -> pd.DataFrame:
    """Compute the 52-week high ratio for price-level context.

    The ratio of today's price to the rolling 252-day maximum captures
    how far a stock is trading from its recent peak. This is bounded
    between 0 and 1 by construction, making it naturally scaled for
    hypersphere projection without further normalisation.

    Critically, this signal separates from momentum: a stock can show
    strong 12-month momentum whilst still being far from its 52-week
    high if it crashed and partially recovered. The two features
    complement rather than duplicate each other.

    Ref: George, T. J. & Hwang, C.-Y. (2004). The 52-Week High and
        Momentum Investing. Journal of Finance, 59(5), 2145-2176.

    Args:
        df: DataFrame sorted by [ticker, Date].
        price_col: Column containing prices.
        ticker_col: Column identifying each security.
        high_window: Rolling window in trading days (252 = 1 year).

    Returns:
        DataFrame with added price_to_52w_high column.
    """
    df = df.copy()

    # Rolling maximum price over the window, grouped by ticker
    rolling_max = (
        df.groupby(ticker_col, observed=True)[price_col]
        .transform(lambda x: x.rolling(window=high_window, min_periods=1).max())
    )

    # Ratio: today's price / rolling max. Replace zero max with NaN to avoid division.
    rolling_max = rolling_max.replace(0, np.nan)

    # Intent: Price-Level Anchoring (George & Hwang, 2004).
    # Investors use the 52-week high as a psychological anchor. Stocks near
    # their high face resistance; stocks far below attract mean-reversion
    # attention. This ratio provides the geometric layer with bounded
    # price-level context absent from any return-based feature.
    df["price_to_52w_high"] = df[price_col] / rolling_max

    return df


def compute_spread_features(
    df: pd.DataFrame,
    ticker_col: str,
    spread_window: int = 22,
) -> pd.DataFrame:
    """Compute the rolling high-low spread as a bid-ask proxy.

    The Corwin-Schultz (2012) estimator uses the daily high-low range
    relative to the mid-price as a proxy for the effective bid-ask spread
    and market maker uncertainty. A higher spread indicates wider
    transaction costs and greater adverse selection risk.

    This provides the graph layer with a second liquidity dimension
    orthogonal to the Amihud measure: Amihud captures price impact
    per unit volume whilst the spread captures intraday price range
    as a direct proxy for transaction costs.

    Ref: Corwin, S. A. & Schultz, P. (2012). A Simple Way to Estimate
        Bid-Ask Spreads from Daily High and Low Prices. Journal of
        Finance, 67(2), 719-760.

    Args:
        df: DataFrame sorted by [ticker, Date] with High and Low columns.
        ticker_col: Column identifying each security.
        spread_window: Rolling window in trading days for the average.

    Returns:
        DataFrame with added high_low_spread_22d column.
    """
    df = df.copy()

    # Guard: if High or Low columns are missing, skip this feature
    if "High" not in df.columns or "Low" not in df.columns:
        logger.warning(
            "High or Low columns not found — skipping high_low_spread computation"
        )
        df["high_low_spread_22d"] = np.nan
        return df

    high = df["High"].replace(0, np.nan)
    low = df["Low"].replace(0, np.nan)

    # Daily proportional spread: (High - Low) / mid-price
    # Mid-price = (High + Low) / 2, following Corwin-Schultz (2012)
    mid = (high + low) / 2
    daily_spread = (high - low) / mid

    # Intent: Transaction Cost Proxy (Corwin & Schultz, 2012).
    # A consistently wide high-low range signals high effective transaction
    # costs and adverse selection risk. Ghost tickers and illiquid stocks
    # are expected to show systematically elevated spread values here.
    col_name = f"high_low_spread_{spread_window}d"
    rolling_spread = (
        pd.Series(daily_spread.values, index=df.index)
        .groupby(df[ticker_col], observed=True)
        .transform(lambda x: x.rolling(window=spread_window).mean())
    )
    df[col_name] = rolling_spread

    return df


def compute_yield_curve_features(
    yields_df: pd.DataFrame,
    series_10y: str = "US_10Y",
    series_2y: str = "US_02Y",
    zscore_window: int = 252,
) -> pd.DataFrame:
    """Compute yield curve level, slope, and slope z-score.

    Ref: Diebold & Li (2006), Section 2.1 ("The Nelson-Siegel Model").

    Args:
        yields_df: Long-format yield data with Date, Ticker, Close.
        series_10y: Ticker for the 10-year yield.
        series_2y: Ticker for the 2-year yield.
        zscore_window: Window for rolling slope z-score.

    Returns:
        DataFrame with yc_level, yc_slope, and yc_slope_zscore per date.
    """
    if "Ticker" not in yields_df.columns and "Series" in yields_df.columns:
        yields_df = yields_df.rename(columns={"Series": "Ticker"})

    yields_wide = yields_df.pivot(index="Date", columns="Ticker", values="Close")
    result = pd.DataFrame(index=yields_wide.index)

    has_10y = series_10y in yields_wide.columns
    has_2y = series_2y in yields_wide.columns

    if not (has_10y and has_2y):
        logger.warning(
            f"Missing required yield series {series_10y} or {series_2y} for curve computation"
        )
        return result

    y10 = yields_wide[series_10y]
    y2 = yields_wide[series_2y]

    # Intent: Macro-Regime Context (Diebold & Li, 2006).
    # Level and Slope are the two most powerful predictors of economic transitions.
    # These feed into the Quantum Layer to condition the density matrix on the macro state.
    result["yc_level"] = (y10 + y2) / 2
    result["yc_slope"] = y10 - y2

    # How unusual is the current slope relative to recent history?
    slope_mean = result["yc_slope"].rolling(window=zscore_window).mean()
    slope_std = result["yc_slope"].rolling(window=zscore_window).std()
    result["yc_slope_zscore"] = (result["yc_slope"] - slope_mean) / (slope_std + 1e-8)

    return result.reset_index()


def compute_foreign_yield_features(
    yields_wide_daily: pd.DataFrame,
    us_slope: Optional[pd.Series],
    foreign_config: dict[str, dict],
) -> pd.DataFrame:
    """Compute Group B (slow-moving context) and Group C (divergence) yield features.

    Group B captures global macro regime context from forward-filled monthly
    foreign yields. Group C captures cross-country monetary policy divergence
    by computing spreads between the US daily slope and each foreign proxy.

    Foreign yields update monthly (~323 rows over 26 years) and are forward-
    filled to daily frequency, so most values are stale by design. The divergence
    spreads are dominated by the US daily signal; the foreign component adds the
    structural monetary policy divergence dimension, not daily noise.

    Ref: Diebold & Li (2006), Section 2.1 for yield decomposition framework.

    Args:
        yields_wide_daily: Wide-format daily-frequency yields with ticker
            columns. Must already be forward-filled to daily frequency.
        us_slope: US yield curve slope series (yc_slope), daily frequency,
            aligned to the same index as yields_wide_daily.
        foreign_config: Dict mapping config keys to sub-dicts containing
            ``ticker`` and ``label`` string fields.

    Returns:
        DataFrame with Date column and new feature columns:
            global_yc_level, {label}_slope_proxy, spread_us_{label}_slope.
    """
    result = pd.DataFrame(index=yields_wide_daily.index)

    available_foreign: list[tuple[str, str]] = []
    for key, spec in foreign_config.items():
        ticker = spec["ticker"]
        label = spec["label"]
        if ticker not in yields_wide_daily.columns:
            logger.warning(
                "Foreign yield %s not found in data — skipping %s features.",
                ticker,
                label,
            )
            continue
        available_foreign.append((ticker, label))

    if not available_foreign:
        logger.warning("No foreign yield series available. Skipping Group B/C features.")
        return result.reset_index().rename(columns={"index": "Date"})

    # Group B: Global YC Level — mean of all available foreign 10Y yields
    foreign_cols = [t for t, _ in available_foreign]
    result["global_yc_level"] = yields_wide_daily[foreign_cols].mean(axis=1)

    # Group B and C: per-country slope proxy (63-day change, ~quarterly)
    for ticker, label in available_foreign:
        slope_col = f"{label.lower()}_slope_proxy"
        result[slope_col] = (
            yields_wide_daily[ticker] - yields_wide_daily[ticker].shift(63)
        )

        # Group C: divergence spread (US daily slope minus foreign slope proxy)
        spread_col = f"spread_us_{label.lower()}_slope"
        if us_slope is not None and len(us_slope) == len(result):
            result[spread_col] = us_slope.values - result[slope_col].values
        else:
            logger.warning(
                "US slope length mismatch — skipping %s divergence spread.", label
            )

    return result.reset_index().rename(columns={"index": "Date"})


def robust_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Compute robust z-score using rolling median and MAD.

    Uses the consistency constant 1.4826 to make MAD comparable
    to standard deviation under a Gaussian distribution.

    Ref: Huber (1981), Section on Median Absolute Deviation.

    Args:
        series: Input time series.
        window: Rolling window size.

    Returns:
        Robust z-scored series.
    """
    median = series.rolling(window=window).median()

    def calc_mad(x: np.ndarray) -> float:
        return np.nanmedian(np.abs(x - np.nanmedian(x)))

    mad = series.rolling(window=window).apply(calc_mad, raw=True)

    # Intent: Robust Statistics (Huber, 1981).
    # Financial data contains "Black Swan" outliers. We use Median and MAD
    # (scaled by 1.4826) instead of Mean and StdDev to prevent single extreme
    # events from distorting the geometric embedding.
    return (series - median) / (1.4826 * mad + 1e-8)


def cap_prices(df: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Replace extreme corrupted price values with NaN across all OHLCV columns.

    Return winsorisation catches anomalous returns but cannot catch the case where
    two consecutive corrupted prices produce a normal-looking return. A single
    $1,000,000 value will keep the price_to_52w_high rolling maximum inflated
    for 252 trading days and create large spikes in skip-month momentum. Setting
    the value to NaN (rather than clamping to the cap) avoids creating artificial
    returns at the boundary; downstream NaN handling zeroes these rows safely.

    Args:
        df: DataFrame with any subset of OHLCV columns present.
        cap: Absolute price ceiling above which values are set to NaN.

    Returns:
        DataFrame with extreme price values replaced by NaN.
    """
    df = df.copy()
    price_cols = ["Adjusted_close", "Close", "Open", "High", "Low"]

    for col in price_cols:
        if col not in df.columns:
            continue
        mask = df[col] > cap
        n_capped = int(mask.sum())
        if n_capped > 0:
            df.loc[mask, col] = np.nan
            logger.info(
                "  Capped %d extreme values in %s to NaN (threshold: %.0f).",
                n_capped,
                col,
                cap,
            )

    return df


def engineer_equity_features(equity_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply all feature engineering steps to equity data.

    Computes the full 12-feature set defined in MODEL_FEATURE_COLS plus
    log_return_1d as the supervised prediction target and the new quality-
    gating columns. All raw OHLCV columns are retained for diagnostics.

    Order of operations:
        1. Sort by [Ticker, Date].
        2. cap_prices            — set extreme corrupted prices to NaN.
        3. compute_log_returns   — raw log returns at 1d, 5d, 20d.
        4. winsorise_returns     — cap data errors before any rolling features.
        5. compute_realised_volatility — vol_10d, vol_22d, vol_60d.
        6. compute_momentum      — mom_3m, mom_6m, mom_12m.
        7. compute_volume_features    — volume_ratio, amihud_illiquidity.
        8. compute_price_level_features — price_to_52w_high.
        9. compute_spread_features    — high_low_spread_22d.
        10. compute_tradability_flags  — is_tradable + diagnostic cols.
        11. compute_forward_target    — target_fwd (shifted from data_loader).

    Args:
        equity_df: Equity rows from the master dataset.
        config: Feature configuration dictionary (FEATURE_CONFIG).

    Returns:
        Equity DataFrame with all engineered feature columns appended.
    """
    logger.info(
        f"Processing features for {equity_df['Ticker'].nunique()} equity tickers..."
    )

    equity_df = equity_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Use adjusted close to account for corporate actions (splits, dividends)
    if "Adjusted_close" in equity_df.columns:
        price_col = "Adjusted_close"
    else:
        logger.warning("Adjusted_close not found, falling back to Close")
        price_col = "Close"

    # Cap extreme corrupted prices before any return or price-level calculations
    if "price_cap" in config:
        equity_df = cap_prices(equity_df, cap=config["price_cap"])

    # Log returns at 1, 5 and 20 day horizons
    # log_return_1d = supervised target; log_return_5d and 20d = model features
    equity_df = compute_log_returns(
        equity_df, price_col, "Ticker", config["return_horizons"]
    )

    # Winsorise returns to cap data errors before downstream features
    if "winsorise_caps" in config:
        equity_df = winsorise_returns(
            equity_df, ticker_col="Ticker", caps=config["winsorise_caps"]
        )

    # Step 4: Realised volatility at short, monthly and quarterly scales
    if "log_return_1d" in equity_df.columns:
        equity_df = compute_realised_volatility(
            equity_df, "log_return_1d", "Ticker", config["volatility_windows"]
        )

    # Skip-month momentum at 3, 6 and 12 month horizons
    equity_df = compute_momentum(
        equity_df, price_col, "Ticker", config["momentum_horizons"]
    )

    # Liquidity signals: volume ratio and Amihud illiquidity
    if "log_return_1d" in equity_df.columns:
        equity_df = compute_volume_features(
            equity_df,
            ticker_col="Ticker",
            return_col="log_return_1d",
            volume_ratio_window=config["volume_ratio_window"],
        )

    # Price level: proximity to 52-week high
    equity_df = compute_price_level_features(
        equity_df,
        price_col=price_col,
        ticker_col="Ticker",
        high_window=config["high_window"],
    )

    # Transaction cost proxy: rolling high-low spread
    equity_df = compute_spread_features(
        equity_df,
        ticker_col="Ticker",
        spread_window=config["spread_window"],
    )

    # As-traded prices for universe gating (no look-ahead through later splits)
    if config.get("tradability_point_in_time", False) and "Close" in equity_df.columns:
        splits = config.get("_split_history")
        if splits is None:
            splits = load_split_history(config.get("splits_dir"))
        if splits.empty:
            logger.warning(
                "  No split history found (splits_dir=%s): close_pit equals the "
                "split-adjusted Close, so the price gate still uses later splits. "
                "Run stage1_fetch_splits.py first.",
                config.get("splits_dir"),
            )
        equity_df = compute_point_in_time_prices(equity_df, "Ticker", splits)

    # Universe quality gating — flag tradable rows
    equity_df = compute_tradability_flags(
        equity_df, ticker_col="Ticker", config=config
    )

    # Forward prediction target (pre-computed here, not in data_loader)
    if "log_return_1d" in equity_df.columns:
        equity_df = compute_forward_target(
            equity_df, ticker_col="Ticker", return_col="log_return_1d"
        )

    return equity_df


def engineer_yield_features(
    yield_df: pd.DataFrame, date_spine: pd.DatetimeIndex, config: dict
) -> pd.DataFrame:
    """Apply feature engineering to yield data.

    Aligns yields to a daily spine via forward-fill before computing
    curve decomposition features.

    Args:
        yield_df: Yield rows from the master dataset.
        date_spine: Business-day date index for alignment.
        config: Feature configuration dictionary.

    Returns:
        Daily-frequency yield DataFrame with curve features.
    """
    # Pivot to wide for reindexing, then back to long
    yield_wide = yield_df.pivot(index="Date", columns="Ticker", values="Close")
    yield_daily_wide = yield_wide.reindex(date_spine).ffill()

    yield_daily_long = yield_daily_wide.stack().reset_index()
    yield_daily_long.columns = ["Date", "Ticker", "Close"]
    yield_daily_long["AssetClass"] = "yield"

    yc_features = compute_yield_curve_features(
        yield_daily_long,
        config["yield_series_10y"],
        config["yield_series_2y"],
        config["zscore_window"],
    )

    yield_daily_features = yield_daily_long.merge(yc_features, on="Date", how="left")

    # Group B/C: foreign yield features and cross-country divergence spreads
    foreign_config = config.get("foreign_yields", {})
    if foreign_config:
        us_slope: Optional[pd.Series] = (
            yc_features["yc_slope"] if "yc_slope" in yc_features.columns else None
        )
        foreign_feats = compute_foreign_yield_features(
            yield_daily_wide, us_slope, foreign_config
        )
        if not foreign_feats.empty:
            yield_daily_features = yield_daily_features.merge(
                foreign_feats, on="Date", how="left"
            )
            logger.info(
                "  Added foreign yield features: %s",
                [c for c in foreign_feats.columns if c != "Date"],
            )

    return yield_daily_features


def engineer_commodity_features(
    commodity_df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Apply feature engineering to commodity data.

    Commodities receive a minimal feature set (1d returns, 22d volatility)
    because they serve as regime-broadcasting hub nodes in the graph.
    Only VIX showed meaningful correlation with the equity universe
    (0.31 mean absolute correlation) at the 0.3 graph threshold; the
    remaining commodities are retained as structural hub nodes with
    potential for lower-threshold connectivity.

    Args:
        commodity_df: Commodity rows from the master dataset.
        config: Feature configuration dictionary.

    Returns:
        Commodity DataFrame with engineered feature columns appended.
    """
    logger.info(
        f"Processing features for {commodity_df['Ticker'].nunique()} commodities..."
    )

    commodity_df = commodity_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    commodity_df = compute_log_returns(commodity_df, "Close", "Ticker", [1])

    if "log_return_1d" in commodity_df.columns:
        commodity_df = compute_realised_volatility(
            commodity_df, "log_return_1d", "Ticker", [22]
        )

    return commodity_df


def _canonical_columns(
    yields_df: pd.DataFrame,
    commodities_df: pd.DataFrame,
    equity_columns: list[str],
) -> list[str]:
    """Build canonical column order for combined export (key cols first, then rest)."""
    key_cols = ["Date", "Ticker", "AssetClass"]
    all_cols = set(yields_df.columns) | set(commodities_df.columns) | set(equity_columns)
    other = sorted(all_cols - set(key_cols))
    return [c for c in key_cols if c in all_cols] + other


def export_features(df: pd.DataFrame, output_path: Path) -> None:
    """Export feature-engineered dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported to: {output_path}")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")


def export_features_streamed(
    yields_featured: pd.DataFrame,
    commodities_featured: pd.DataFrame,
    equity_temp_path: Path,
    output_path: Path,
    full_columns: list[str],
) -> None:
    """Write combined CSV from yields, commodities, and chunked equity temp file.

    Avoids holding the full combined dataframe in memory. Order: yields, then
    commodities, then equities (each block sorted by Date, Ticker).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        wrote_header = False
        for _, df in [
            ("yields", yields_featured),
            ("commodities", commodities_featured),
        ]:
            if df.empty:
                continue
            reindexed = df.reindex(columns=full_columns)
            reindexed.to_csv(f, index=False, header=not wrote_header)
            if not wrote_header:
                wrote_header = True
            total_rows += len(reindexed)

        if equity_temp_path.exists():
            for chunk in pd.read_csv(
                equity_temp_path, parse_dates=["Date"], chunksize=500_000
            ):
                reindexed = chunk.reindex(columns=full_columns)
                reindexed.to_csv(f, index=False, header=False)
                total_rows += len(reindexed)

    if equity_temp_path.exists():
        equity_temp_path.unlink()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported to: {output_path}")
    logger.info(f"  Total rows: {total_rows:,}")
    logger.info(f"  Columns: {full_columns}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")


def main(master_path: Path, output_path: Path, config: Optional[dict] = None) -> None:
    """Orchestrate the complete feature engineering pipeline."""
    if config is None:
        config = FEATURE_CONFIG
    config = dict(config)
    if config.get("tradability_point_in_time", False):
        # Load split histories once, not once per ticker batch.
        config["_split_history"] = load_split_history(config.get("splits_dir"))
        logger.info(
            "Point-in-time tradability: %d splits loaded from %s.",
            len(config["_split_history"]),
            config.get("splits_dir"),
        )


    logger.info("STAGE 3: Feature Engineering")
    logger.info(f"Model feature set ({len(MODEL_FEATURE_COLS)} features): {MODEL_FEATURE_COLS}")

    logger.info(f"Loading master data from: {master_path}")
    if not master_path.exists():
        logger.error(f"File not found: {master_path}")
        return

    master = pd.read_csv(master_path, parse_dates=["Date"])
    logger.info(f"  Loaded {len(master):,} rows")

    # Drop null-Ticker rows to prevent silent downstream errors
    null_ticker_count = master["Ticker"].isnull().sum()
    if null_ticker_count > 0:
        master = master.dropna(subset=["Ticker"])
        logger.info(f"  Dropped {null_ticker_count:,} rows where Ticker is null")

    min_date = master["Date"].min()
    max_date = master["Date"].max()
    date_spine = pd.bdate_range(start=min_date, end=max_date)
    logger.info(
        f"  Date spine: {min_date.date()} to {max_date.date()} ({len(date_spine)} days)"
    )

    # Split by asset class and free master to reduce peak memory
    equities = master[master["AssetClass"] == "equity"].copy()
    yields = master[master["AssetClass"] == "yield"].copy()
    commodities = master[master["AssetClass"] == "commodity"].copy()
    del master

    logger.info(
        f"  Equities: {equities['Ticker'].nunique()} tickers, {len(equities):,} rows"
    )
    logger.info(f"  Yields: {yields['Ticker'].nunique()} series, {len(yields):,} rows")
    logger.info(
        f"  Commodities: {commodities['Ticker'].nunique()} assets, {len(commodities):,} rows"
    )

    logger.info("Engineering yield features...")
    if not yields.empty:
        yields_featured = engineer_yield_features(yields, date_spine, config)
    else:
        yields_featured = pd.DataFrame()
        logger.warning("No yield data found, skipping yield engineering")

    logger.info("Engineering commodity features...")
    if not commodities.empty:
        commodities_featured = engineer_commodity_features(commodities, config)
    else:
        commodities_featured = pd.DataFrame()

    equity_temp_path = output_path.parent / (output_path.stem + "_equity_temp.csv")
    equity_columns: list[str] = []

    if not equities.empty:
        logger.info("Engineering equity features (chunked by ticker)...")
        tickers = equities["Ticker"].unique()
        n_batches = (len(tickers) + EQUITY_TICKER_CHUNK_SIZE - 1) // EQUITY_TICKER_CHUNK_SIZE
        for i in range(0, len(tickers), EQUITY_TICKER_CHUNK_SIZE):
            batch_tickers = tickers[i: i + EQUITY_TICKER_CHUNK_SIZE]
            batch_num = i // EQUITY_TICKER_CHUNK_SIZE + 1
            batch_df = equities[equities["Ticker"].isin(batch_tickers)].copy()
            batch_featured = engineer_equity_features(batch_df, config)
            del batch_df
            if not equity_columns:
                equity_columns = list(batch_featured.columns)
            write_header = not equity_temp_path.exists()
            batch_featured.to_csv(
                equity_temp_path,
                index=False,
                mode="w" if write_header else "a",
                header=write_header,
            )
            del batch_featured
            logger.info(
                f"  Equity batch {batch_num}/{n_batches} ({len(batch_tickers)} tickers) written"
            )
    else:
        logger.warning("No equity data found!")

    full_columns = _canonical_columns(
        yields_featured, commodities_featured, equity_columns or []
    )
    logger.info("Combining all asset classes (streamed)...")
    export_features_streamed(
        yields_featured,
        commodities_featured,
        equity_temp_path,
        output_path,
        full_columns,
    )


    logger.info("STAGE 3 COMPLETE")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    MASTER_FILE = PROJECT_ROOT / "data" / "interim" / "master_consolidated.csv"
    OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "master_features.csv"

    main(
        master_path=MASTER_FILE,
        output_path=OUTPUT_FILE,
        config={**FEATURE_CONFIG, "splits_dir": PROJECT_ROOT / "data" / "raw" / "splits"},
    )