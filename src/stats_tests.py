"""Dependence-robust statistics for comparing walk-forward forecasts.

The fold-level paired t-test in ``evaluation.paired_t_test`` treats the
walk-forward folds as independent draws and ignores both within-fold serial
dependence and initialisation (seed) noise. The functions here provide the
standard alternatives:

    * ``hac_mean_test``: Newey-West t-test on a daily series. Applied to the
      daily difference of two models' ICs (loss = -IC) this is the
      Diebold-Mariano test.
    * ``stationary_bootstrap``: Politis-Romano block bootstrap for any
      statistic of a (possibly paired) daily series, e.g. a Sharpe difference.
    * ``fold_cluster_test``: inference with folds as the sampling unit, the
      conservative choice when the claim is about future years.
    * ``bh_adjust``: Benjamini-Hochberg false discovery rate adjustment.
    * ``deflated_sharpe_ratio``: Sharpe ratio corrected for the number of
      configurations tried and for non-normal returns.
    * ``variance_components`` / ``seeds_needed``: split a metric's variance
      into fold (market) and seed (initialisation) parts and size a
      multi-seed experiment.

References:
    Newey and West (1987) Econometrica 55(3).
    Diebold and Mariano (1995) JBES 13(3); Harvey, Leybourne and Newbold
        (1997) IJF 13(2).
    Politis and Romano (1994) JASA 89(428).
    Benjamini and Hochberg (1995) JRSS-B 57(1).
    Bailey and Lopez de Prado (2012) J. Risk 15(2) (PSR); (2014) JPM 40(5) (DSR).
"""

import math

import numpy as np
from scipy import stats

_EULER_GAMMA: float = 0.5772156649015329


def _finite(x: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return arr[np.isfinite(arr)]


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) plug-in bandwidth floor(4 (n/100)^(2/9))."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def hac_variance_of_mean(x: np.ndarray, lags: int | None = None) -> tuple[float, int]:
    """Bartlett-kernel long-run variance of the sample mean.

    Returns:
        Tuple of (variance of the mean, lags used).
    """
    x = _finite(x)
    n = len(x)
    if lags is None:
        lags = newey_west_lag(n)
    u = x - x.mean()
    long_run = float(u @ u) / n
    for k in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - k / (lags + 1.0)
        long_run += 2.0 * weight * float(u[k:] @ u[:-k]) / n
    return long_run / n, lags


def hac_mean_test(x: np.ndarray | list[float], lags: int | None = None) -> dict[str, float]:
    """Two-sided test of H0: E[x] = 0 with a Newey-West standard error.

    With x = IC_A - IC_B per day this is the Diebold-Mariano test; the
    Harvey-Leybourne-Newbold small-sample factor and a t(n-1) reference
    distribution are applied.
    """
    arr = _finite(x)
    n = len(arr)
    if n < 3:
        return {"mean": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan"), "lags": 0, "n": n}
    var_mean, used = hac_variance_of_mean(arr, lags)
    se = math.sqrt(max(var_mean, 0.0))
    t_raw = arr.mean() / se if se > 0 else float("nan")
    t_stat = t_raw * math.sqrt((n - 1.0) / n)  # HLN correction for 1-step forecasts
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    return {"mean": float(arr.mean()), "se": se, "t": float(t_stat), "p": float(p_value), "lags": used, "n": n}


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano resampling indices, shape (n_boot, n)."""
    p_new = 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    new_block = rng.random((n_boot, n)) < p_new
    new_block[:, 0] = True
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def stationary_bootstrap(
    series: np.ndarray | list[np.ndarray],
    statistic=None,
    mean_block: float | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Block-bootstrap confidence interval and p-value for H0: statistic = 0.

    Args:
        series: A 1-D array, or a list of equally long aligned arrays that are
            resampled jointly (e.g. two strategies' daily PnL).
        statistic: Function of the array(s); default is the mean of a 1-D series.
        mean_block: Expected block length; default n^(1/3).
        n_boot: Number of bootstrap replications.
        alpha: Two-sided level of the percentile interval.
        seed: Random seed.
    """
    arrays = [np.asarray(s, dtype=np.float64) for s in (series if isinstance(series, list) else [series])]
    keep = np.all([np.isfinite(a) for a in arrays], axis=0)
    arrays = [a[keep] for a in arrays]
    n = len(arrays[0])
    if statistic is None:
        statistic = np.mean
    if mean_block is None:
        mean_block = max(1.0, n ** (1.0 / 3.0))
    rng = np.random.default_rng(seed)
    estimate = float(statistic(*arrays))
    idx = stationary_bootstrap_indices(n, mean_block, n_boot, rng)
    boot = np.array([statistic(*[a[row] for a in arrays]) for row in idx], dtype=np.float64)
    lower, upper = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Centred bootstrap p-value: how often |boot - estimate| >= |estimate|.
    p_value = float(np.mean(np.abs(boot - estimate) >= abs(estimate)))
    return {"estimate": estimate, "ci_lower": float(lower), "ci_upper": float(upper), "p": p_value, "mean_block": float(mean_block), "n": n}


def annualised_sharpe(pnl: np.ndarray, periods: int = 252) -> float:
    pnl = _finite(pnl)
    if len(pnl) < 2:
        return float("nan")
    sd = pnl.std(ddof=1)
    return float(pnl.mean() / sd * math.sqrt(periods)) if sd > 0 else float("nan")


def sharpe_difference_bootstrap(pnl_a: np.ndarray, pnl_b: np.ndarray, **kwargs) -> dict[str, float]:
    """Stationary-bootstrap test of equal annualised Sharpe for two aligned daily PnL series."""
    return stationary_bootstrap([pnl_a, pnl_b], statistic=lambda a, b: annualised_sharpe(a) - annualised_sharpe(b), **kwargs)


def fold_cluster_test(fold_values: np.ndarray | list[float], lags: int = 1) -> dict[str, float]:
    """Inference with folds as the unit: HAC t-test over the per-fold means.

    ``lags=0`` reduces to the one-sample t-test used in the dissertation;
    ``lags=1`` allows adjacent years to be correlated.
    """
    return hac_mean_test(fold_values, lags=lags)


def bh_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone)."""
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))  # NaN p-values stay NaN and do not count in m
    m = len(finite)
    if m == 0:
        return adjusted
    order = finite[np.argsort(p[finite])]
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def probabilistic_sharpe_ratio(pnl: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark) allowing for skew and kurtosis."""
    pnl = _finite(pnl)
    n = len(pnl)
    sr = pnl.mean() / pnl.std(ddof=1)
    skew = stats.skew(pnl)
    kurt = stats.kurtosis(pnl, fisher=False)
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1.0) / denom))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among n_trials zero-skill trials."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe_ratio(pnl: np.ndarray, n_trials: int, sr_variance: float) -> float:
    """Deflated Sharpe Ratio of Bailey and Lopez de Prado (2014).

    Args:
        pnl: Daily PnL of the selected strategy.
        n_trials: Number of configurations/hyperparameter settings tried,
            including discarded ones (e.g. every autoresearch mutation).
        sr_variance: Variance of the per-period Sharpe ratios across trials.
    """
    return probabilistic_sharpe_ratio(pnl, expected_max_sharpe(n_trials, sr_variance))


def variance_components(table: np.ndarray) -> dict[str, float]:
    """One-way random-effects split of a (n_folds, n_seeds) metric table.

    Pass per-(fold, seed) values of one configuration, or of the paired
    difference between two configurations trained with the same seeds.
    """
    table = np.asarray(table, dtype=np.float64)
    n_folds, n_seeds = table.shape
    if n_seeds < 2:
        raise ValueError("Need at least two seeds per fold to separate seed variance.")
    ms_within = float(table.var(axis=1, ddof=1).mean())
    ms_between = float(n_seeds * table.mean(axis=1).var(ddof=1))
    sigma2_seed = ms_within
    sigma2_fold = max((ms_between - ms_within) / n_seeds, 0.0)
    total = sigma2_fold + sigma2_seed
    return {
        "sigma2_fold": sigma2_fold,
        "sigma2_seed": sigma2_seed,
        "share_fold": sigma2_fold / total if total > 0 else float("nan"),
        "n_folds": n_folds,
        "n_seeds": n_seeds,
    }


def min_detectable_effect(
    sigma2_fold: float, sigma2_seed: float, n_folds: int, n_seeds: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest mean difference detectable with fold-level inference."""
    se = math.sqrt((sigma2_fold + sigma2_seed / n_seeds) / n_folds)
    return (stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power)) * se


def seeds_needed(
    sigma2_fold: float, sigma2_seed: float, delta: float, n_folds: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Seeds per fold needed to detect ``delta``; inf if fold variance alone forbids it."""
    target_var = (delta / (stats.norm.isf(alpha / 2) + stats.norm.isf(1 - power))) ** 2
    remaining = target_var * n_folds - sigma2_fold
    if remaining <= 0:
        return float("inf")
    return float(max(1, math.ceil(sigma2_seed / remaining)))
