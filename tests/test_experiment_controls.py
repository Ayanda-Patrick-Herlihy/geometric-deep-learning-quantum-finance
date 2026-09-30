"""Tests for the negative controls, ablation variants, post-hoc evaluation and
dependence-robust statistics added for the paper experiments."""

import numpy as np
import pandas as pd
import pytest
import torch

from src import posthoc_eval, stats_tests
from src.data_loader import FinancialGraphDataset, build_synthetic_dataframe
from src.evaluation import compute_daily_pnl
from src.layers.geometric import GeometricEmbedding
from src.layers.quantum import QuantumRegimeDetector, build_regime_detector
from src.run_ablation import (
    ABLATION_VARIANTS,
    _build_criterion_for_config,
    _get_config_name,
    build_ablation_model,
)
from src.train import load_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(None)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return build_synthetic_dataframe(n_tickers=30, n_dates=120, seed=3)


# Statistics


def test_hac_test_detects_mean_and_widens_under_autocorrelation() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(size=3000)
    assert stats_tests.hac_mean_test(noise)["p"] > 0.01
    assert stats_tests.hac_mean_test(noise + 0.2)["p"] < 1e-6

    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.8 * ar[t - 1] + noise[t]
    hac_se = stats_tests.hac_mean_test(ar)["se"]
    naive_se = ar.std(ddof=1) / np.sqrt(len(ar))
    assert hac_se > 2.0 * naive_se


def test_stationary_bootstrap_interval_and_p_value() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.5, size=500)
    result = stats_tests.stationary_bootstrap(x, n_boot=500, seed=2)
    assert result["ci_lower"] < 0.5 < result["ci_upper"]
    assert result["p"] < 0.01


def test_bh_adjust_known_values() -> None:
    adjusted = stats_tests.bh_adjust([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.02])


def test_deflated_sharpe_decreases_with_trials() -> None:
    rng = np.random.default_rng(4)
    pnl = rng.normal(loc=0.001, scale=0.01, size=1000)
    one = stats_tests.deflated_sharpe_ratio(pnl, n_trials=1, sr_variance=0.002)
    assert one == pytest.approx(stats_tests.probabilistic_sharpe_ratio(pnl, 0.0))
    assert stats_tests.deflated_sharpe_ratio(pnl, n_trials=100, sr_variance=0.002) < one


def test_variance_components_recovers_known_split() -> None:
    rng = np.random.default_rng(5)
    table = rng.normal(scale=1.0, size=(400, 1)) + rng.normal(scale=0.5, size=(400, 5))
    parts = stats_tests.variance_components(table)
    assert parts["sigma2_fold"] == pytest.approx(1.0, rel=0.2)
    assert parts["sigma2_seed"] == pytest.approx(0.25, rel=0.2)
    assert np.isinf(stats_tests.seeds_needed(1.0, 0.25, delta=0.01, n_folds=22))
    assert np.isfinite(stats_tests.seeds_needed(1e-8, 1e-5, delta=0.002, n_folds=22))


# Negative controls and dataset additions


def test_shuffle_labels_permutes_targets_within_supervised_set(config: dict, synthetic_df: pd.DataFrame) -> None:
    plain = FinancialGraphDataset("unused", config, dataframe=synthetic_df)
    shuffled = FinancialGraphDataset("unused", config, dataframe=synthetic_df, control="shuffle_labels", control_seed=7)
    a, b = plain[10], shuffled[10]
    assert torch.equal(a.target_mask, b.target_mask)
    assert torch.equal(a.x, b.x)
    assert torch.equal(torch.sort(a.y[a.target_mask]).values, torch.sort(b.y[b.target_mask]).values)
    assert not torch.equal(a.y, b.y)


def test_skip_day_target_is_next_target_of_same_ticker(config: dict, synthetic_df: pd.DataFrame) -> None:
    ds = FinancialGraphDataset("unused", config, dataframe=synthetic_df)
    today, tomorrow = ds[20], ds[21]
    common = np.intersect1d(today.ticker_id.numpy(), tomorrow.ticker_id.numpy())
    for tid in common[:5]:
        i = int(np.flatnonzero(today.ticker_id.numpy() == tid)[0])
        j = int(np.flatnonzero(tomorrow.ticker_id.numpy() == tid)[0])
        if torch.isfinite(today.y_skip1[i]) and tomorrow.target_mask[j]:
            assert float(today.y_skip1[i]) == pytest.approx(float(tomorrow.y[j]))


def test_permute_graph_nodes_permutes_return_histories(config: dict, synthetic_df: pd.DataFrame) -> None:
    plain = FinancialGraphDataset("unused", config, dataframe=synthetic_df)[30]
    permuted = FinancialGraphDataset(
        "unused", config, dataframe=synthetic_df, control="permute_graph_nodes", control_seed=1
    )[30]
    assert not torch.equal(plain.past_returns, permuted.past_returns)
    assert torch.equal(
        torch.sort(plain.past_returns.sum(dim=1)).values, torch.sort(permuted.past_returns.sum(dim=1)).values
    )


def test_unknown_control_rejected(config: dict, synthetic_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        FinancialGraphDataset("unused", config, dataframe=synthetic_df, control="bogus")


# Ablation variants


def test_classical_regime_head_is_parameter_matched() -> None:
    quantum = QuantumRegimeDetector(input_dim=64, context_dim=3, num_regimes=4)
    classical = build_regime_detector("softmax_mlp", 64, 3, 4)
    n_q = sum(p.numel() for p in quantum.parameters())
    n_c = sum(p.numel() for p in classical.parameters())
    assert abs(n_c - n_q) / n_q < 0.05
    probs, rho = classical(torch.randn(5, 64), torch.randn(5, 3))
    torch.testing.assert_close(probs.sum(-1), torch.ones(5))
    torch.testing.assert_close(torch.diagonal(rho, dim1=-2, dim2=-1), probs)


def test_euclidean_twin_skips_normalisation() -> None:
    torch.manual_seed(0)
    layer = GeometricEmbedding(12, 64, normalize=False)
    norms = layer(torch.randn(50, 12)).norm(dim=-1)
    assert not torch.allclose(norms, torch.ones_like(norms))


@pytest.mark.parametrize("variant", sorted(ABLATION_VARIANTS))
def test_variants_differ_from_base_only_in_named_term(config: dict, variant: str) -> None:
    base_id, flags, loss_overrides = ABLATION_VARIANTS[variant]
    model = build_ablation_model(variant, config, torch.device("cpu"))
    base = build_ablation_model(base_id, config, torch.device("cpu"))
    if "sphere" in flags:
        assert model.geometric.normalize is False and base.geometric.normalize is True
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in base.parameters())
    if flags.get("magnitude"):
        extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters())
        assert extra == model.head[0].out_features  # one extra input column
        out = model(torch.randn(7, 12), None, None, torch.randn(1, 3))
        assert out["predictions"].shape == (7, 1)
    if "regime_head" in flags:
        assert type(model.quantum).__name__ == "ClassicalRegimeDetector"
    criterion = _build_criterion_for_config(variant, config)
    base_criterion = _build_criterion_for_config(base_id, config)
    expected_uniform = loss_overrides.get("uniformity", base_criterion.w_uniform)
    assert criterion.w_uniform == expected_uniform
    assert criterion.w_dir == base_criterion.w_dir and criterion.w_dm == base_criterion.w_dm


def test_config_names_match_built_models() -> None:
    assert _get_config_name("A3") == "Graph_Only"
    assert _get_config_name("A6") == "Geometry_Quantum"
    assert _get_config_name("A8e").endswith("A8e")


# Post-hoc evaluation reproduces the in-loop PnL


def test_posthoc_flat_cost_matches_compute_daily_pnl() -> None:
    rng = np.random.default_rng(8)
    rows = []
    for d, date in enumerate(pd.bdate_range("2021-01-04", periods=4)):
        n = 150
        rows.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker_id": rng.permutation(400)[:n],
                    "pred": rng.normal(size=n),
                    "y": rng.normal(scale=0.02, size=n),
                    "y_skip1": rng.normal(scale=0.02, size=n),
                    "close": rng.uniform(1, 50, size=n),
                    "volume": rng.uniform(1e4, 1e6, size=n),
                    "fold": 0,
                    "seed": 42,
                }
            )
        )
    preds = pd.concat(rows, ignore_index=True)
    daily = posthoc_eval.daily_metrics(preds, top_k=20)
    ours = posthoc_eval.net_pnl(daily, 20.0, flat=True).to_numpy()
    for i, (_, day) in enumerate(preds.groupby("date")):
        repo = compute_daily_pnl(
            torch.tensor(day["pred"].to_numpy()),
            torch.tensor(day["y"].to_numpy()),
            torch.tensor(day["close"].to_numpy()),
            torch.tensor(day["volume"].to_numpy()),
            top_k=20,
            min_price=5.0,
            min_dollar_volume=1_000_000.0,
            transaction_cost_bps=20.0,
            date_label="t",
        )
        if repo is None:
            assert np.isnan(ours[i])
        else:
            assert ours[i] == pytest.approx(repo, abs=1e-9)


# Regression tests for the code-review fixes


def test_stale_graph_applies_to_every_date_with_history(config: dict, synthetic_df: pd.DataFrame) -> None:
    import copy

    small = copy.deepcopy(config)
    small["model"]["graph_construction"]["rolling_window"] = 20
    plain = FinancialGraphDataset("unused", small, dataframe=synthetic_df)
    stale = FinancialGraphDataset("unused", small, dataframe=synthetic_df, control="stale_graph")
    changed = 0
    for i, date in enumerate(plain.valid_dates):
        date_idx = plain._date_to_idx[date]
        a, b = plain[i].past_returns, stale[i].past_returns
        if date_idx > 20:
            # Stale window = plain window of the date rolling_window + 1 days earlier.
            earlier = plain._all_dates[date_idx - 21]
            if earlier in plain.valid_dates:
                ref = plain[plain.valid_dates.index(earlier)]
                if torch.equal(ref.ticker_id, plain[i].ticker_id):
                    assert torch.equal(b, ref.past_returns)
            changed += int(not torch.equal(a, b))
        else:
            assert torch.count_nonzero(b) == 0
    assert changed > 0.9 * sum(plain._date_to_idx[d] > 20 for d in plain.valid_dates)


def test_dataset_view_matches_fresh_construction(config: dict, synthetic_df: pd.DataFrame) -> None:
    from src.data_loader import dataset_view

    full = FinancialGraphDataset("unused", config, dataframe=synthetic_df)
    dates = full.valid_dates[5:25]
    fresh = FinancialGraphDataset("unused", config, dataframe=synthetic_df, dates=dates, control="shuffle_labels", control_seed=3)
    view = dataset_view(full, dates, "shuffle_labels", 3)
    assert view.valid_dates == fresh.valid_dates
    for i in range(len(view)):
        a, b = view[i], fresh[i]
        for key in ("x", "y", "target_mask", "past_returns", "ticker_id"):
            assert torch.equal(getattr(a, key), getattr(b, key))
    assert full.control is None  # the base dataset is untouched


def test_bh_adjust_ignores_nan() -> None:
    adjusted = stats_tests.bh_adjust([0.001, float("nan"), 0.03])
    assert np.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.002, 0.03])


def test_posthoc_matches_topk_with_ties() -> None:
    rng = np.random.default_rng(9)
    frames = []
    for date in pd.bdate_range("2021-01-04", periods=5):
        n = 150
        frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker_id": np.arange(n),
                    "pred": rng.integers(0, 4, size=n).astype(np.float32),  # heavy ties
                    "y": rng.normal(scale=0.02, size=n).astype(np.float32),
                    "y_skip1": rng.normal(scale=0.02, size=n).astype(np.float32),
                    "close": np.full(n, 20.0, dtype=np.float32),
                    "volume": np.full(n, 1e6, dtype=np.float32),
                    "fold": 0,
                    "seed": 1,
                }
            )
        )
    preds = pd.concat(frames, ignore_index=True)
    ours = posthoc_eval.net_pnl(posthoc_eval.daily_metrics(preds, top_k=20), 20.0, flat=True).to_numpy()
    for i, (_, day) in enumerate(preds.groupby("date")):
        repo = compute_daily_pnl(
            torch.tensor(day["pred"].to_numpy()),
            torch.tensor(day["y"].to_numpy()),
            torch.tensor(day["close"].to_numpy()),
            torch.tensor(day["volume"].to_numpy()),
            top_k=20,
            min_price=5.0,
            min_dollar_volume=1_000_000.0,
            transaction_cost_bps=20.0,
            date_label="t",
        )
        assert ours[i] == pytest.approx(repo, abs=1e-6)


def test_first_day_of_fold_trades_one_unit_per_leg() -> None:
    rng = np.random.default_rng(10)
    day = pd.DataFrame(
        {
            "date": pd.Timestamp("2021-01-04"),
            "ticker_id": np.arange(120),
            "pred": rng.normal(size=120),
            "y": 0.0,
            "y_skip1": 0.0,
            "close": 20.0,
            "volume": 1e6,
            "fold": 0,
            "seed": 1,
        }
    )
    daily = posthoc_eval.daily_metrics(day, top_k=20)
    assert daily["traded_notional"].iloc[0] == 2.0
    assert posthoc_eval.net_pnl(daily, 20.0).iloc[0] == pytest.approx(-2 * 20 / 1e4)


def test_defaults_still_run_only_a0_to_a9() -> None:
    from src.run_ablation import ABLATION_IDS, ALL_IDS

    assert ABLATION_IDS == [f"A{i}" for i in range(10)]
    assert set(ABLATION_VARIANTS) <= set(ALL_IDS)


def test_a9t_sequence_ends_at_current_return(config: dict, synthetic_df: pd.DataFrame) -> None:
    plain = FinancialGraphDataset("unused", config, dataframe=synthetic_df)
    current = FinancialGraphDataset("unused", config, dataframe=synthetic_df, sequence_includes_current=True)
    item_plain, item_current = plain[40], current[40]
    date = plain.valid_dates[40]
    ticker = plain.return_pivot.columns[int(item_current.ticker_id[0])]
    r_t = float(np.nan_to_num(plain.return_pivot.loc[date, ticker]))
    assert float(item_current.x_seq[0, -1, 0]) == pytest.approx(r_t)
    assert torch.equal(item_current.x_seq[0, :-1, 0], item_plain.x_seq[0, 1:, 0])


def test_inner_val_days_must_be_positive(config: dict, synthetic_df: pd.DataFrame) -> None:
    from src.run_ablation import train_ablation_fold

    dates = FinancialGraphDataset("unused", config, dataframe=synthetic_df).valid_dates
    with pytest.raises(ValueError):
        train_ablation_fold("A0", 0, dates[:40], dates[45:60], config, torch.device("cpu"), 0,
                            dataframe=synthetic_df, inner_val_days=0)
