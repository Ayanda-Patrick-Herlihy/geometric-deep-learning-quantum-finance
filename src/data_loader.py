"""Financial graph dataset and walk-forward data utilities.

Loads the processed parquet file, builds per-date cross-sectional
graph data, and provides temporal walk-forward splitting for
training and evaluation.

Graph construction is deferred to the GPU at training time via
build_dynamic_graph_gpu in graph_utils. The dataset prepares the
raw trailing return window so the correlation matrix can be computed
on-device in milliseconds rather than on CPU.

References:
    Sandhu et al. (2021) "Network Geometry and Market Instability."
        Nature Physics.
"""

import copy
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Node feature columns (must match stage3 output schema).
DEFAULT_FEATURE_COLUMNS: list[str] = [
    # Returns (model features)
    "log_return_5d",
    "log_return_20d",
    # Volatility
    "vol_10d",
    "vol_22d",
    "vol_60d",
    # Momentum
    "mom_3m",
    "mom_6m",
    "mom_12m",
    # Liquidity and price level
    "volume_ratio",
    "amihud_illiquidity",
    "high_low_spread_22d",
    "price_to_52w_high",
]

# Yield curve context columns (global macro context).
DEFAULT_CONTEXT_COLUMNS: list[str] = [
    "yc_level",
    "yc_slope",
    "yc_slope_zscore",
    # Group B: global macro context (forward-filled monthly foreign yields)
    "global_yc_level",
    "de_slope_proxy",
    "jp_slope_proxy",
    "uk_slope_proxy",
    # Group C: cross-country divergence spreads
    "spread_us_de_slope",
    "spread_us_jp_slope",
    "spread_us_uk_slope",
]

# Column used for rolling correlation and graph construction.
RETURN_COLUMN: str = "log_return_1d"

# Negative controls for FinancialGraphDataset(control=...). None = original data.
#   shuffle_labels      : targets permuted across stocks within each date (IC -> 0 expected)
#   permute_graph_nodes : correlation graph built from other stocks' histories
#   stale_graph         : correlation graph from the preceding non-overlapping window
#                         (ending rolling_window+1 days before t); empty graph if none exists
#   inject_target       : POSITIVE control; first feature replaced by the target (IC -> ~1)
NEGATIVE_CONTROLS: tuple[str | None, ...] = (
    None,
    "shuffle_labels",
    "permute_graph_nodes",
    "stale_graph",
    "inject_target",
)

# MAD scaling constant: 1.4826 * MAD approximates standard deviation
# for normally distributed data, making this robust to outliers.
_MAD_SCALE: float = 1.4826


def robust_zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Robust z-score normalisation using median and MAD.

    More resilient to outliers than standard z-score because it uses
    the median and median absolute deviation instead of mean and std.

    Ref: Huber (1981), Section on Median Absolute Deviation.

    Args:
        x: Feature tensor of shape (N, D).
        eps: Stability constant to prevent division by zero.

    Returns:
        Normalised tensor of the same shape.
    """
    median = x.median(dim=0).values
    mad = (x - median).abs().median(dim=0).values
    scale = _MAD_SCALE * mad + eps
    return (x - median) / scale


class FinancialGraphDataset(TorchDataset):
    """Dataset that packages each trading date as a graph.

    Each item is a single date: the active stocks become nodes, their
    trailing returns are packaged for on-device graph construction,
    and yield curve data provides global context.

    Graph edges are NOT built here. Instead, past_returns is attached
    to each Data object so that build_dynamic_graph_gpu can compute
    the correlation graph on the GPU during the forward pass.

    Args:
        parquet_path: Path to data/processed/fact_table.parquet.
        config: Configuration dictionary (from config.yaml) or a
            Path to config.yaml.
        feature_columns: Node feature column names (overrides config).
        context_columns: Yield curve context column names.
        dates: Restrict dataset to these dates only (used by
            WalkForwardSplitter for train/val subsets).
        normalise: Apply robust z-score normalisation per cross-section.
        dataframe: Pre-loaded DataFrame. If provided, parquet_path
            is ignored. Useful for testing without real data.
        sequence_length: Number of past return steps for LSTM baselines.
        control: Optional negative control (see NEGATIVE_CONTROLS). None
            reproduces the original pipeline exactly.
        control_seed: Seed for the control's per-date randomisation.
        sequence_includes_current: If True, the LSTM sequence ends at the
            current date's return r_t (known at the close of t, like every
            engineered feature). False keeps the original r_{t-T}..r_{t-1}.
    """

    valid_dates: list[pd.Timestamp]

    def __init__(
        self,
        parquet_path: Path | str,
        config: dict | Path | str,
        feature_columns: list[str] | None = None,
        context_columns: list[str] | None = None,
        dates: list[pd.Timestamp] | None = None,
        normalise: bool = True,
        dataframe: pd.DataFrame | None = None,
        sequence_length: int = 60,
        control: str | None = None,
        control_seed: int = 0,
        sequence_includes_current: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.normalise = normalise
        self.sequence_length = sequence_length
        if control not in NEGATIVE_CONTROLS:
            raise ValueError(f"Unknown control {control!r}; expected one of {NEGATIVE_CONTROLS}.")
        self.control = control
        self.control_seed = control_seed
        self.sequence_includes_current = sequence_includes_current

        # Resolve configuration
        if isinstance(config, dict):
            self._config = config
        else:
            self._config = self._load_config(Path(config))

        # Resolve column lists: explicit args > config > defaults
        data_cfg = self._config.get("data", {})
        self.feature_columns = (
            feature_columns
            or data_cfg.get("feature_columns", None)
            or DEFAULT_FEATURE_COLUMNS
        )
        self.context_columns = (
            context_columns
            or data_cfg.get("context_columns", None)
            or DEFAULT_CONTEXT_COLUMNS
        )

        # Graph construction parameters
        gc = self._config.get("model", {}).get("graph_construction", {})
        self.rolling_window: int = gc.get("rolling_window", 252)
        self.correlation_threshold: float = gc.get(
            "correlation_threshold",
            0.3,
        )

        # Load and preprocess
        self._load_data(dataframe)

        # Optional date restriction (walk-forward subsets)
        if dates is not None:
            restricted = sorted(set(dates) & set(self.valid_dates))
            self.valid_dates = restricted
            logger.info(
                "Restricted to %d of %d requested dates.",
                len(self.valid_dates),
                len(dates),
            )

        logger.info(
            "FinancialGraphDataset ready: %d dates, %d features, "
            "%d tickers in the equity pool.",
            len(self.valid_dates),
            len(self.available_features),
            self.return_pivot.shape[1],
        )

    @staticmethod
    def _load_config(path: Path) -> dict:
        """Loads YAML configuration file.

        Args:
            path: Path to config.yaml.

        Returns:
            Parsed configuration dictionary.

        Raises:
            FileNotFoundError: If the file is absent.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r") as fh:
            return yaml.safe_load(fh)

    # Data loading and preprocessing

    def _load_data(self, dataframe: pd.DataFrame | None) -> None:
        """Loads and preprocesses the fact table.

        Steps performed:
            1. Read parquet (or accept pre-loaded DataFrame).
            2. Separate equity and yield rows.
            3. Identify available feature columns.
            4. Compute forward return targets.
            5. Build return pivot for correlation.
            6. Build yield curve context lookup.
            7. Determine valid dates.

        Args:
            dataframe: Pre-loaded DataFrame, or None to read parquet.

        Raises:
            FileNotFoundError: If parquet is missing and no DataFrame given.
        """
        if dataframe is not None:
            df = dataframe.copy()
            logger.info(
                "Using pre-loaded DataFrame: %d rows, %d columns.",
                len(df),
                len(df.columns),
            )
        else:
            if not self.parquet_path.exists():
                raise FileNotFoundError(
                    f"Fact table not found: {self.parquet_path}. "
                    "Run the data pipeline (stages 1-4) first."
                )
            logger.info("Loading %s", self.parquet_path)
            df = pd.read_parquet(self.parquet_path)

        df["Date"] = pd.to_datetime(df["Date"])
        logger.info(
            "Data: %d rows, %d columns, range %s to %s.",
            len(df),
            len(df.columns),
            df["Date"].min().strftime("%Y-%m-%d"),
            df["Date"].max().strftime("%Y-%m-%d"),
        )

        # Separate asset classes
        equity_df = df[df["AssetClass"] == "equity"].copy()
        yield_df = df[df["AssetClass"] == "yield"].copy()
        logger.info(
            "Equity rows: %d.  Yield rows: %d.",
            len(equity_df),
            len(yield_df),
        )

        required_eval_columns = {"Close", "Volume"}
        missing_eval_columns = sorted(required_eval_columns - set(equity_df.columns))
        if missing_eval_columns:
            raise KeyError(
                "Evaluation requires the following equity columns in the fact table: "
                f"{missing_eval_columns}."
            )

        # Identify available feature columns
        available = [c for c in self.feature_columns if c in equity_df.columns]
        missing = sorted(set(self.feature_columns) - set(available))
        if missing:
            logger.warning(
                "Feature columns absent from data (excluded): %s",
                missing,
            )
        self.available_features: list[str] = available
        self.feature_dim: int = len(available)

        # Warn on config mismatch
        cfg_dim = self._config.get("model", {}).get("geometric", {}).get("input_dim")
        if cfg_dim and cfg_dim != self.feature_dim:
            logger.warning(
                "config model.geometric.input_dim=%d but data yields %d "
                "features.  Consider updating config.yaml.",
                cfg_dim,
                self.feature_dim,
            )

        # Sort equities. Forward targets are pre-computed in Stage 3
        # (see compute_forward_target in stage3_engineer_features.py).
        equity_df.sort_values(["Ticker", "Date"], inplace=True)

        if "target_fwd" not in equity_df.columns:
            logger.warning(
                "target_fwd not found in data — computing at load time. "
                "Re-run Stage 3 + Stage 4 to embed targets in Parquet."
            )
            g = equity_df.groupby("Ticker", observed=True)[RETURN_COLUMN]
            equity_df["target_fwd"] = g.shift(-1)

        # Return realised one day later (t+1 -> t+2) for skip-day evaluation,
        # which separates predictive signal from next-day bid-ask bounce.
        equity_df["target_fwd_skip1"] = equity_df.groupby("Ticker", observed=True)[
            "target_fwd"
        ].shift(-1)

        # Date-grouped lookup for fast cross-section access
        self._date_groups: dict[pd.Timestamp, pd.DataFrame] = dict(
            list(equity_df.groupby("Date"))
        )

        # Return pivot table: (dates, tickers) for rolling correlation
        self.return_pivot: pd.DataFrame = equity_df.pivot_table(
            index="Date",
            columns="Ticker",
            values=RETURN_COLUMN,
            aggfunc="first",
            observed=True,
        ).astype(np.float32)
        self.return_pivot.sort_index(inplace=True)

        self._all_dates: list[pd.Timestamp] = self.return_pivot.index.tolist()
        self._date_to_idx: dict[pd.Timestamp, int] = {
            d: i for i, d in enumerate(self._all_dates)
        }

        # Yield curve context lookup (date -> float32 array)
        self._yield_context = self._build_yield_context(yield_df)

        # Compute valid dates
        self.valid_dates: list[pd.Timestamp] = self._compute_valid_dates()

    def _build_yield_context(
        self,
        yield_df: pd.DataFrame,
    ) -> dict[pd.Timestamp, np.ndarray]:
        """Builds a date-keyed lookup for yield curve context.

        Aggregates yc_level, yc_slope, yc_slope_zscore across yield
        tickers per date (mean) and forward-fills to cover gaps.

        Args:
            yield_df: Yield-class rows from the fact table.

        Returns:
            Mapping of date to float32 array of shape (context_dim,).
        """
        ctx_dim = len(self.context_columns)
        lookup: dict[pd.Timestamp, np.ndarray] = {}

        if yield_df.empty:
            logger.warning("No yield data found. Context will default to zeros.")
            return lookup

        avail_ctx = [c for c in self.context_columns if c in yield_df.columns]
        if not avail_ctx:
            logger.warning("Yield context columns absent. Context defaults to zeros.")
            return lookup

        # Mean across yield tickers per date, then forward-fill
        ctx_df = yield_df.groupby("Date")[avail_ctx].mean()
        ctx_df.sort_index(inplace=True)
        ctx_df = ctx_df.reindex(self._all_dates, method="ffill")

        for date in self._all_dates:
            if date not in ctx_df.index:
                continue
            row = ctx_df.loc[date]
            vals = np.zeros(ctx_dim, dtype=np.float32)
            for i, col in enumerate(self.context_columns):
                if col in avail_ctx:
                    v = row[col]
                    vals[i] = v if np.isfinite(v) else 0.0
            lookup[date] = vals

        logger.info(
            "Yield context built: %d dates, %d features.",
            len(lookup),
            len(avail_ctx),
        )
        return lookup

    def _compute_valid_dates(self) -> list[pd.Timestamp]:
        """Identifies dates suitable for training samples.

        A date is valid when:
            1. At least 2 equity tickers are present (graph needs edges).
            2. At least one ticker has a valid forward target.
            3. Sufficient return history for correlation (>= 60 days).

        Returns:
            Sorted list of valid trading dates.
        """
        min_history = 60
        valid: list[pd.Timestamp] = []

        for date in self._all_dates:
            if date not in self._date_groups:
                continue
            cs = self._date_groups[date]

            if len(cs) < 2:
                continue
            if cs["target_fwd"].notna().sum() < 1:
                continue
            if self._date_to_idx.get(date, 0) < min_history:
                continue

            valid.append(date)

        logger.info(
            "Valid dates: %d of %d total.",
            len(valid),
            len(self._all_dates),
        )
        return valid

    # Dataset interface

    def __len__(self) -> int:
        return len(self.valid_dates)

    def __getitem__(self, idx: int) -> Data:
        """Produces a PyG Data object for a single trading date.

        Graph edges are placeholder self-loops; the real correlation
        graph is built on-device by build_dynamic_graph_gpu using the
        past_returns tensor attached to the Data object.

        Args:
            idx: Index into valid_dates.

        Returns:
            Data with attributes:
                x: Node features (N_t, D).
                edge_index: Placeholder self-loops (2, N_t).
                edge_attr: Placeholder weights (N_t,).
                y: Forward return targets (N_t,).
                context: Yield curve context (C,).
                target_mask: Valid-target boolean mask (N_t,).
                past_returns: Trailing returns (N_t, T_window) for
                    GPU graph construction.
                num_nodes: Scalar N_t.
        """
        date = self.valid_dates[idx]
        cs = self._date_groups[date].copy()
        candidates = cs["Ticker"].tolist()

        # Enforce invariant: only use tickers that exist in the return history
        # table. Tickers in the daily cross-section but missing from the pivot
        # (e.g. no valid log returns, or data errors like GSEN) would cause
        # KeyError when building past_returns. Filter once here so the rest
        # of the pipeline never sees a mismatch.
        valid_tickers = [t for t in candidates if t in self.return_pivot.columns]
        if len(valid_tickers) < len(candidates):
            dropped = set(candidates) - set(valid_tickers)
            logger.debug(
                "Date %s: dropped %d ticker(s) missing from return history: %s",
                date.strftime("%Y-%m-%d"),
                len(dropped),
                sorted(dropped)[:10],
            )
            cs = cs[cs["Ticker"].isin(valid_tickers)]
        tickers = valid_tickers
        n = len(tickers)

        # Need at least 2 nodes for a meaningful graph; otherwise skip this date
        if n < 2:
            dummy = Data(
                x=torch.zeros(1, self.feature_dim, dtype=torch.float32),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, dtype=torch.float32),
                y=torch.zeros(1, dtype=torch.float32),
                num_nodes=1,
            )
            dummy.context = torch.zeros(1, len(self.context_columns), dtype=torch.float32)
            dummy.target_mask = torch.tensor([False], dtype=torch.bool)
            dummy.past_returns = torch.zeros(1, self.rolling_window, dtype=torch.float32)
            dummy.x_seq = torch.zeros(1, self.sequence_length, 1, dtype=torch.float32)
            dummy.close_price = torch.zeros(1, dtype=torch.float32)
            dummy.volume = torch.zeros(1, dtype=torch.float32)
            dummy.y_skip1 = torch.full((1,), float("nan"), dtype=torch.float32)
            dummy.ticker_id = torch.full((1,), -1, dtype=torch.long)
            return dummy

        # Node features — replace NaN *and* inf (e.g. log(0) = -inf
        # from delisted stocks) so they do not poison normalisation.
        feats = cs[self.available_features].values.astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if self.control == "inject_target":
            # Deliberate leak: shows the harness detects one when it exists.
            feats[:, 0] = np.nan_to_num(cs["target_fwd"].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(feats, dtype=torch.float32)

        # Robust normalisation to handle fat-tailed financial distributions
        if self.normalise and n > 1:
            x = robust_zscore(x)

        # Clamp to prevent numerical overflow in downstream linear layers
        x = x.clamp(-10.0, 10.0)

        # Forward targets — mark inf as invalid (same as NaN)
        raw_targets = cs["target_fwd"].values.astype(np.float32)
        raw_skip = cs["target_fwd_skip1"].values.astype(np.float32)
        rng = np.random.default_rng([self.control_seed, self._date_to_idx[date]])
        mask = np.isfinite(raw_targets)

        # If is_tradable exists (from Stage 3 quality gating), further
        # restrict the target mask so untradable rows do not contribute
        # to the supervised loss. They still exist as graph nodes.
        if "is_tradable" in cs.columns:
            tradable = cs["is_tradable"].values.astype(bool)
            mask = mask & tradable

        if self.control == "shuffle_labels":
            # Permute targets among the supervised stocks of this date: breaks
            # any feature-target link but keeps the universe and distribution.
            idx = np.flatnonzero(mask)
            perm = rng.permutation(idx)
            raw_targets = raw_targets.copy()
            raw_skip = raw_skip.copy()
            raw_targets[idx] = raw_targets[perm]
            raw_skip[idx] = raw_skip[perm]

        targets = np.nan_to_num(raw_targets, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.tensor(targets, dtype=torch.float32)
        y_skip1 = torch.tensor(raw_skip, dtype=torch.float32)  # NaN kept: marks missing
        target_mask = torch.tensor(mask, dtype=torch.bool)
        close_prices = torch.tensor(
            np.nan_to_num(
                cs["Close"].values.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dtype=torch.float32,
        )
        volumes = torch.tensor(
            np.nan_to_num(
                cs["Volume"].values.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            dtype=torch.float32,
        )

        # Yield curve context
        ctx_arr = self._yield_context.get(
            date,
            np.zeros(len(self.context_columns), dtype=np.float32),
        )
        # Shape (1, C) so PyG Batch concatenation produces (B, C)
        context = torch.tensor(ctx_arr, dtype=torch.float32).unsqueeze(0)

        # Trailing return window for GPU graph construction.
        # Pad to fixed length (rolling_window) so PyG Batch collate can
        # concatenate along node dimension; variable T would break batching.
        date_idx = self._date_to_idx[date]
        graph_idx = date_idx
        if self.control == "stale_graph":
            # Correlations from the window that ends just before today's window
            # starts: same length and coverage, strictly past, no overlap.
            graph_idx = date_idx - self.rolling_window - 1
        if graph_idx < 0:
            window_df = self.return_pivot.iloc[0:0][tickers]  # no stale history: empty graph
        else:
            start = max(0, graph_idx - self.rolling_window)
            window_dates = self._all_dates[start : graph_idx + 1]
            window_df = self.return_pivot.loc[window_dates, tickers]
        returns_arr = window_df.values  # (T_actual, N)
        past_returns = torch.tensor(returns_arr.T, dtype=torch.float32)
        past_returns = torch.nan_to_num(past_returns, nan=0.0, posinf=0.0, neginf=0.0)
        T_actual = past_returns.size(1)
        if T_actual < self.rolling_window:
            pad_left = self.rolling_window - T_actual
            past_returns = torch.nn.functional.pad(
                past_returns, (pad_left, 0), mode="constant", value=0.0
            )
        elif T_actual > self.rolling_window:
            past_returns = past_returns[:, -self.rolling_window :]
        if self.control == "permute_graph_nodes":
            # Randomly reassign return histories to stocks: same degree and
            # weight distribution, but neighbours are unrelated stocks.
            past_returns = past_returns[torch.as_tensor(rng.permutation(n))]

        # Sequence data for LSTM baselines (N, T, 1)
        x_seq = self._get_sequence_data(date, tickers)

        # Placeholder edges (self-loops); replaced on GPU by _forward_step
        loop_idx = torch.arange(n, dtype=torch.long)
        edge_index = torch.stack([loop_idx, loop_idx])
        edge_weight = torch.ones(n, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=y,
            num_nodes=n,
        )
        data.context = context
        data.target_mask = target_mask
        data.x_seq = x_seq
        data.past_returns = past_returns
        data.close_price = close_prices
        data.volume = volumes
        data.y_skip1 = y_skip1
        # Column position in return_pivot; lets evaluation track turnover and
        # join predictions back to tickers. Not named *index* so PyG collation
        # does not offset it.
        data.ticker_id = torch.tensor(
            self.return_pivot.columns.get_indexer(tickers), dtype=torch.long
        )

        return data

    def _get_sequence_data(
        self,
        date: pd.Timestamp,
        tickers: list[str],
    ) -> torch.Tensor:
        """Extracts historical return sequences for active tickers.

        The sequence contains only returns strictly before the label date (last
        timestep is date_idx-1). This keeps x_seq and the 1-day forward target
        temporally disjoint and avoids lookahead.

        Args:
            date: Current cross-section date (label date).
            tickers: List of active tickers.

        Returns:
            Tensor of shape (N, sequence_length, 1).
        """
        if self.sequence_length <= 0:
            return torch.empty(len(tickers), 0, 1)

        date_idx = self._date_to_idx[date]
        end_idx = date_idx + 1 if self.sequence_includes_current else date_idx
        start_idx = max(0, end_idx - self.sequence_length)

        # The target is r_{t+1}, so r_t (row date_idx) is already disjoint from it;
        # the original code excluded it, which hides any day-t signal from A9.
        window_df = self.return_pivot.iloc[start_idx:end_idx]
        valid_cols = [t for t in tickers if t in self.return_pivot.columns]
        seq_data = window_df[valid_cols].values  # (T_actual, N_with_history)

        T_actual, _ = seq_data.shape
        out = np.zeros((len(tickers), self.sequence_length), dtype=np.float32)

        if T_actual > 0:
            col_to_idx = {t: i for i, t in enumerate(valid_cols)}
            fill_len = min(T_actual, self.sequence_length)

            for i, ticker in enumerate(tickers):
                if ticker in col_to_idx:
                    src_col = col_to_idx[ticker]
                    out[i, -fill_len:] = seq_data[-fill_len:, src_col]

        out_tensor = torch.tensor(out, dtype=torch.float32).unsqueeze(-1)
        out_tensor = torch.nan_to_num(out_tensor, nan=0.0)

        return out_tensor

    # Convenience properties

    @property
    def dates(self) -> list[pd.Timestamp]:
        """Sorted list of valid trading dates."""
        return self.valid_dates

    @property
    def num_features(self) -> int:
        """Dimension of the node feature vector."""
        return self.feature_dim

    @property
    def num_context_features(self) -> int:
        """Dimension of the yield curve context vector."""
        return len(self.context_columns)


def dataset_view(
    base: "FinancialGraphDataset",
    dates: list[pd.Timestamp] | None = None,
    control: str | None = None,
    control_seed: int = 0,
    **attributes: object,
) -> "FinancialGraphDataset":
    """Date-restricted view of an already loaded dataset.

    Equivalent to ``FinancialGraphDataset(dates=dates, control=..., ...)`` on
    the same data and config (``dates`` only filters ``valid_dates``, and
    controls act only in ``__getitem__``), but shares the loaded tables instead
    of copying the full panel again.
    """
    if control not in NEGATIVE_CONTROLS:
        raise ValueError(f"Unknown control {control!r}; expected one of {NEGATIVE_CONTROLS}.")
    view = copy.copy(base)
    if dates is not None:
        view.valid_dates = sorted(set(dates) & set(base.valid_dates))
    view.control = control
    view.control_seed = control_seed
    for name, value in attributes.items():
        setattr(view, name, value)
    return view


class WalkForwardSplitter:
    """Walk-forward temporal cross-validation splitter.

    Generates non-overlapping train/validation date windows that
    advance through time to prevent future information leakage.

    A purge gap is inserted between the end of training and the start of
    validation to prevent leakage from overlapping return windows. The feature
    set includes log_return_20d (a 20-day rolling window), so the last training
    sample and first validation sample share 20 days of price history. The
    purge gap removes those contaminated dates from the boundary. The default
    purge_gap=20 matches the longest return horizon in MODEL_FEATURE_COLS.

    Pattern per fold:
        train:  dates[start : start + W]
        purge:  dates[start + W : start + W + P]  (skipped)
        val:    dates[start + W + P : start + W + P + V]
        Next fold advances start by step_size S.

    Ref: Aronson (2006) / Purvey (2018), standard backtesting methodology.
    Ref: De Prado (2018), Advances in Financial Machine Learning, Chapter 7.

    Args:
        dates: Sorted list of all available dates.
        train_window: Number of dates in the training window.
        validation_window: Number of dates in the validation window.
        step_size: Window advance per fold (number of dates).
        purge_gap: Dates skipped between train end and val start to prevent
            leakage from overlapping return features. Default 20 matches
            the longest return horizon (log_return_20d).
    """

    def __init__(
        self,
        dates: list[pd.Timestamp],
        train_window: int = 1260,
        validation_window: int = 252,
        step_size: int = 63,
        purge_gap: int = 20,
    ) -> None:
        self.dates = sorted(dates)
        self.train_window = train_window
        self.validation_window = validation_window
        self.step_size = step_size
        self.purge_gap = purge_gap
        self.n_folds = self._count_folds()

        logger.info(
            "WalkForwardSplitter: %d dates, train=%d, purge=%d, val=%d, step=%d "
            "-> %d folds.",
            len(self.dates),
            train_window,
            purge_gap,
            validation_window,
            step_size,
            self.n_folds,
        )

    def _count_folds(self) -> int:
        """Counts the number of complete train+purge+val folds."""
        total = len(self.dates)
        required = self.train_window + self.purge_gap + self.validation_window
        if total < required:
            return 0
        count = 0
        start = 0
        while start + required <= total:
            count += 1
            start += self.step_size
        return count

    def __len__(self) -> int:
        return self.n_folds

    def __iter__(
        self,
    ) -> Iterator[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
        """Yields (train_dates, val_dates) tuples per fold.

        The purge_gap dates between training end and validation start are
        excluded from both splits to eliminate return-window leakage.

        Yields:
            Tuple of (train_dates, val_dates) lists.
        """
        total = len(self.dates)
        required = self.train_window + self.purge_gap + self.validation_window
        start = 0
        fold = 0

        while start + required <= total:
            t_end = start + self.train_window
            v_start = t_end + self.purge_gap
            v_end = v_start + self.validation_window
            train_dates = self.dates[start:t_end]
            val_dates = self.dates[v_start:v_end]

            logger.info(
                "Fold %d: train %s..%s (%d dates), purge=%d, "
                "val %s..%s (%d dates).",
                fold,
                train_dates[0].strftime("%Y-%m-%d"),
                train_dates[-1].strftime("%Y-%m-%d"),
                len(train_dates),
                self.purge_gap,
                val_dates[0].strftime("%Y-%m-%d"),
                val_dates[-1].strftime("%Y-%m-%d"),
                len(val_dates),
            )

            yield train_dates, val_dates

            start += self.step_size
            fold += 1

    def get_fold(
        self,
        fold_idx: int,
    ) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
        """Returns the date split for a specific fold.

        Args:
            fold_idx: Zero-based fold index.

        Returns:
            Tuple of (train_dates, val_dates).

        Raises:
            IndexError: If fold_idx is out of range.
        """
        if fold_idx < 0 or fold_idx >= self.n_folds:
            raise IndexError(f"Fold {fold_idx} out of range [0, {self.n_folds}).")
        start = fold_idx * self.step_size
        t_end = start + self.train_window
        v_start = t_end + self.purge_gap
        v_end = v_start + self.validation_window
        return self.dates[start:t_end], self.dates[v_start:v_end]


def get_device() -> torch.device:
    """Selects the best available compute device.

    Priority order: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device for model and data placement.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device: CUDA (%s).", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: MPS (Apple Silicon).")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU.")
    return device


def build_synthetic_dataframe(
    n_tickers: int = 20,
    n_dates: int = 400,
    start_date: str = "2020-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    """Creates a synthetic fact table for smoke-testing.

    Generates a DataFrame mimicking the schema of fact_table.parquet
    with random but structurally valid data. Useful for testing the
    data loader and training loop without running the full pipeline.

    Args:
        n_tickers: Number of synthetic equity tickers.
        n_dates: Number of trading dates to generate.
        start_date: First trading date (string).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with the fact_table schema.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_dates)
    tickers = [f"SYNTH_{i:04d}" for i in range(n_tickers)]

    # Equity rows
    rows: list[dict] = []
    for date in dates:
        for ticker in tickers:
            # Simulate occasional delistings (survivorship)
            if rng.random() < 0.02:
                continue
            row = {
                "Date": date,
                "Ticker": ticker,
                "AssetClass": "equity",
                "Close": float(100.0 + rng.normal(0, 10)),
                "Volume": float(abs(rng.normal(2_000_000, 500_000))),
                "log_return_1d": float(rng.normal(0, 0.02)),
                "log_return_5d": float(rng.normal(0, 0.04)),
                "log_return_20d": float(rng.normal(0, 0.08)),
                "vol_10d": float(abs(rng.normal(0.02, 0.005))),
                "vol_22d": float(abs(rng.normal(0.02, 0.005))),
                "vol_60d": float(abs(rng.normal(0.02, 0.005))),
                "mom_3m": float(rng.normal(0, 0.10)),
                "mom_6m": float(rng.normal(0, 0.15)),
                "mom_12m": float(rng.normal(0, 0.20)),
                # Liquidity and price-level features (synthetic but structurally aligned
                # with Stage 3 output).
                "volume_ratio": float(abs(rng.normal(1.0, 0.5))),
                "amihud_illiquidity": float(abs(rng.normal(1.0, 0.5))),
                "high_low_spread_22d": float(abs(rng.normal(0.02, 0.01))),
                "price_to_52w_high": float(min(max(rng.normal(0.7, 0.2), 0.0), 1.2)),
            }
            rows.append(row)

    # Yield rows (one "ticker" for US yield curve context)
    for date in dates:
        row = {
            "Date": date,
            "Ticker": "US_YC",
            "AssetClass": "yield",
            "Close": float(rng.normal(3.0, 0.5)),
            "Volume": 0.0,
            "yc_level": float(rng.normal(3.0, 0.5)),
            "yc_slope": float(rng.normal(1.0, 0.5)),
            "yc_slope_zscore": float(rng.normal(0, 1.0)),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    logger.info(
        "Synthetic DataFrame: %d rows (%d equity tickers, %d dates).",
        len(df),
        n_tickers,
        n_dates,
    )
    return df
