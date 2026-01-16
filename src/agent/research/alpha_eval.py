"""agent.research.alpha_eval

Lightweight, standard factor evaluation for the P0 loop.

Per-date (cross-sectional) metrics:
- IC: Pearson correlation between factor values and forward returns
- RankIC: Pearson correlation between cross-sectional ranks
- Long-short spread: mean(fwd | top quantile) - mean(fwd | bottom quantile)
- Turnover: 1 - overlap(prev, curr) for top/bottom sets (simple proxy)

Aggregate metrics:
- information_ratio: mean(long_short) / std(long_short) * sqrt(252)
- annualized_return, max_drawdown: from the long-short equity curve
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_forward_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Compute forward returns aligned on the same index.

    For MultiIndex(datetime, instrument), uses groupby(instrument).shift(-horizon).
    """
    if close is None or close.empty:
        return close

    h = int(horizon)
    if h <= 0:
        raise ValueError("horizon must be >= 1")

    if isinstance(close.index, pd.MultiIndex) and "instrument" in close.index.names:
        future = close.groupby(level="instrument").shift(-h)
    else:
        future = close.shift(-h)

    with np.errstate(divide="ignore", invalid="ignore"):
        return (future / close) - 1.0


def _max_drawdown(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(mu / (sd / np.sqrt(n)))


def _ir(x: np.ndarray, trading_days: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(mu / sd * np.sqrt(float(trading_days)))


def _rank_series(x: pd.Series) -> pd.Series:
    return x.rank(method="average")


def _split_quantiles(x: pd.Series, n_quantiles: int) -> Optional[pd.Series]:
    # Use ranks to reduce qcut issues with ties.
    r = x.rank(method="first")
    try:
        q = pd.qcut(r, q=int(n_quantiles), labels=False, duplicates="drop")
    except Exception:
        return None
    if q is None:
        return None
    if q.nunique(dropna=True) < 2:
        return None
    return q.astype("Int64")


def _overlap_turnover(prev: Optional[set], curr: set) -> float:
    if not prev:
        return 0.0
    if not curr:
        return 0.0
    inter = len(prev.intersection(curr))
    denom = float(len(prev))
    if denom == 0.0:
        return 0.0
    return float(1.0 - inter / denom)


def evaluate_alpha(
    factor: pd.Series,
    fwd_returns: pd.Series,
    *,
    n_quantiles: int = 5,
    trading_days: int = 252,
    cost_bps: float = 0.0,
    min_obs_per_day: int = 20,
    universe_size_by_date: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Evaluate a factor with a simple daily cross-sectional long-short proxy."""
    if factor is None or fwd_returns is None:
        return {"error": "Empty inputs"}

    df = pd.DataFrame({"factor": factor, "fwd": fwd_returns}).dropna()
    if df.empty:
        return {"error": "No overlapping non-NaN rows"}

    if not isinstance(df.index, pd.MultiIndex) or "datetime" not in df.index.names or "instrument" not in df.index.names:
        return {"error": "Expected MultiIndex(datetime, instrument) for evaluation"}

    n_quantiles = int(n_quantiles)
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")

    cost = float(cost_bps) / 10000.0

    daily_rows: List[Dict[str, Any]] = []
    ics: List[float] = []
    rics: List[float] = []
    spreads: List[float] = []
    ls_rets: List[float] = []
    turnovers: List[float] = []
    coverages: List[float] = []

    prev_long: Optional[set] = None
    prev_short: Optional[set] = None

    for dt, g in df.groupby(level="datetime", sort=True):
        x = g["factor"]
        y = g["fwd"]
        n = int(len(g))
        if n < int(min_obs_per_day):
            continue

        ic = float(x.corr(y))
        rx = _rank_series(x)
        ry = _rank_series(y)
        ric = float(rx.corr(ry))

        q = _split_quantiles(x, n_quantiles=n_quantiles)
        if q is None:
            continue

        top_q = int(q.max())
        bot_q = int(q.min())

        long_mask = (q == top_q).to_numpy()
        short_mask = (q == bot_q).to_numpy()

        if long_mask.sum() == 0 or short_mask.sum() == 0:
            continue

        long_mean = float(y.to_numpy()[long_mask].mean())
        short_mean = float(y.to_numpy()[short_mask].mean())
        spread = long_mean - short_mean

        # Turnover proxy: membership changes in top/bottom sets.
        inst = g.index.get_level_values("instrument").astype(str)
        long_set = set(inst[long_mask])
        short_set = set(inst[short_mask])
        t_long = _overlap_turnover(prev_long, long_set)
        t_short = _overlap_turnover(prev_short, short_set)
        turnover = 0.5 * (t_long + t_short)

        prev_long = long_set
        prev_short = short_set

        ls = spread - turnover * cost

        if universe_size_by_date is not None:
            denom = float(universe_size_by_date.get(dt, np.nan))
            cov = float(n / denom) if np.isfinite(denom) and denom > 0 else float("nan")
        else:
            cov = float("nan")

        daily_rows.append(
            {
                "datetime": dt.isoformat(),
                "n": n,
                "ic": ic,
                "rank_ic": ric,
                "spread": spread,
                "turnover": turnover,
                "ls_return": ls,
                "coverage": cov,
            }
        )

        ics.append(ic)
        rics.append(ric)
        spreads.append(spread)
        ls_rets.append(ls)
        turnovers.append(turnover)
        coverages.append(cov)

    if not daily_rows:
        return {"error": "Not enough valid daily observations"}

    ic_arr = np.asarray(ics, dtype=float)
    ric_arr = np.asarray(rics, dtype=float)
    spread_arr = np.asarray(spreads, dtype=float)
    ls_arr = np.asarray(ls_rets, dtype=float)
    to_arr = np.asarray(turnovers, dtype=float)

    info_ratio = _ir(ls_arr, trading_days=trading_days)
    annualized_return = float((1.0 + np.nanmean(ls_arr)) ** float(trading_days) - 1.0)
    max_dd = _max_drawdown(pd.Series(ls_arr, index=pd.to_datetime([r["datetime"] for r in daily_rows])))

    coverage_mean = float(np.nanmean(np.asarray(coverages, dtype=float))) if coverages else float("nan")

    return {
        "ic": float(np.nanmean(ic_arr)),
        "ic_std": float(np.nanstd(ic_arr, ddof=1)) if ic_arr.size > 1 else 0.0,
        "ic_tstat": _tstat(ic_arr),
        "rank_ic_mean": float(np.nanmean(ric_arr)),
        "rank_ic_std": float(np.nanstd(ric_arr, ddof=1)) if ric_arr.size > 1 else 0.0,
        "rank_ic_tstat": _tstat(ric_arr),
        "spread_mean": float(np.nanmean(spread_arr)),
        "spread_std": float(np.nanstd(spread_arr, ddof=1)) if spread_arr.size > 1 else 0.0,
        "spread_tstat": _tstat(spread_arr),
        "turnover_mean": float(np.nanmean(to_arr)),
        "coverage_mean": coverage_mean,
        "information_ratio": info_ratio,
        "annualized_return": annualized_return,
        "max_drawdown": max_dd,
        "n_obs": int(len(daily_rows)),
        "cost_bps": float(cost_bps),
        "daily": daily_rows,
    }
