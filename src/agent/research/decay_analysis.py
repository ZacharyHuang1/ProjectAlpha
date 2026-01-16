"""agent.research.decay_analysis

P2.15: Multi-horizon decay analysis.

This module answers two practical research questions:

1) Predictive decay: how does IC / RankIC / spread change as the
   forward-return horizon increases (1d, 2d, 5d, 10d, ...)?
2) Signal persistence: how stable are the top/bottom names over longer lags
   (a simple proxy for signal half-life).

All computations are research diagnostics (not a trade simulation).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.research.alpha_eval import compute_forward_returns


def _tstat(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(mu / (sd / np.sqrt(float(n))))


def _ir_ann(x: np.ndarray, trading_days: int, horizon: int) -> float:
    """Annualized IR proxy for horizon returns.

    Forward returns of length `horizon` overlap across dates, so this is only a
    rough diagnostic. We scale by sqrt(trading_days/horizon) to make horizons
    more comparable.
    """

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0
    scale = np.sqrt(float(trading_days) / float(max(1, int(horizon))))
    return float(mu / sd * scale)


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


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return float("nan")
    u = a.union(b)
    if not u:
        return float("nan")
    return float(len(a.intersection(b)) / float(len(u)))


def _top_bottom_sets(row: pd.Series, *, n_quantiles: int) -> Tuple[set, set]:
    s = row.dropna()
    if s.empty:
        return set(), set()
    q = 1.0 / float(max(2, int(n_quantiles)))
    lo = float(s.quantile(q))
    hi = float(s.quantile(1.0 - q))
    top = set([str(i) for i in s.index[s >= hi]])
    bot = set([str(i) for i in s.index[s <= lo]])
    return top, bot


def _signal_overlap_by_horizon(
    factor_wide: pd.DataFrame,
    horizons: Iterable[int],
    *,
    n_quantiles: int,
    min_obs_per_day: int,
) -> Dict[int, Dict[str, Any]]:
    """Compute Jaccard overlap for top/bottom sets across lag=horizon days."""

    out: Dict[int, Dict[str, Any]] = {}
    if factor_wide is None or factor_wide.empty:
        return out

    wide = factor_wide.sort_index()
    dates = list(wide.index)
    n_dates = int(len(dates))

    for h in sorted({int(x) for x in horizons if int(x) >= 1}):
        if n_dates <= h:
            out[h] = {"signal_overlap_mean": float("nan"), "signal_pairs": 0}
            continue

        vals: List[float] = []
        for i in range(0, n_dates - h):
            r0 = wide.iloc[i]
            r1 = wide.iloc[i + h]
            if int(r0.notna().sum()) < int(min_obs_per_day) or int(r1.notna().sum()) < int(min_obs_per_day):
                continue
            top0, bot0 = _top_bottom_sets(r0, n_quantiles=n_quantiles)
            top1, bot1 = _top_bottom_sets(r1, n_quantiles=n_quantiles)
            ov_top = _jaccard(top0, top1)
            ov_bot = _jaccard(bot0, bot1)
            if np.isfinite(ov_top) and np.isfinite(ov_bot):
                vals.append(0.5 * (float(ov_top) + float(ov_bot)))

        arr = np.asarray(vals, dtype=float)
        out[h] = {
            "signal_overlap_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
            "signal_overlap_std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else float("nan"),
            "signal_pairs": int(arr.size),
        }

    return out


def compute_horizon_decay(
    factor: pd.Series,
    close: pd.Series,
    horizons: Iterable[int],
    *,
    n_quantiles: int = 5,
    trading_days: int = 252,
    min_obs_per_day: int = 20,
    universe_size_by_date: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Compute predictive and signal decay metrics across multiple horizons."""

    if factor is None or close is None or factor.empty or close.empty:
        return {"enabled": False, "error": "Empty inputs"}

    if not isinstance(factor.index, pd.MultiIndex) or "datetime" not in factor.index.names or "instrument" not in factor.index.names:
        return {"enabled": False, "error": "Expected MultiIndex(datetime, instrument) for factor"}
    if not isinstance(close.index, pd.MultiIndex) or "datetime" not in close.index.names or "instrument" not in close.index.names:
        return {"enabled": False, "error": "Expected MultiIndex(datetime, instrument) for close"}

    hs = sorted({int(h) for h in horizons if int(h) >= 1})
    if not hs:
        hs = [1, 2, 5, 10, 20]

    # Pre-compute signal persistence using the factor itself.
    try:
        factor_wide = factor.unstack("instrument")
    except Exception:
        factor_wide = pd.DataFrame()
    sig = _signal_overlap_by_horizon(
        factor_wide,
        hs,
        n_quantiles=int(n_quantiles),
        min_obs_per_day=int(min_obs_per_day),
    )

    rows: List[Dict[str, Any]] = []
    for h in hs:
        fwd = compute_forward_returns(close, horizon=int(h))
        df = pd.DataFrame({"factor": factor, "fwd": fwd}).dropna()
        if df.empty:
            continue

        ics: List[float] = []
        rics: List[float] = []
        spreads: List[float] = []
        coverages: List[float] = []

        for dt, g in df.groupby(level="datetime", sort=True):
            x = g["factor"]
            y = g["fwd"]
            n = int(len(g))
            if n < int(min_obs_per_day):
                continue

            ic = float(x.corr(y))
            rx = x.rank(method="average")
            ry = y.rank(method="average")
            ric = float(rx.corr(ry))

            q = _split_quantiles(x, n_quantiles=int(n_quantiles))
            if q is None:
                continue
            top_q = int(q.max())
            bot_q = int(q.min())
            long_mask = (q == top_q).to_numpy()
            short_mask = (q == bot_q).to_numpy()
            if long_mask.sum() == 0 or short_mask.sum() == 0:
                continue
            spread = float(y.to_numpy()[long_mask].mean()) - float(y.to_numpy()[short_mask].mean())

            cov = float("nan")
            if universe_size_by_date is not None:
                try:
                    denom = float(universe_size_by_date.get(dt, np.nan))
                    cov = float(n / denom) if np.isfinite(denom) and denom > 0.0 else float("nan")
                except Exception:
                    cov = float("nan")

            ics.append(ic)
            rics.append(ric)
            spreads.append(spread)
            coverages.append(cov)

        ic_arr = np.asarray(ics, dtype=float)
        ric_arr = np.asarray(rics, dtype=float)
        spr_arr = np.asarray(spreads, dtype=float)
        cov_arr = np.asarray(coverages, dtype=float)
        n_days = int(np.isfinite(ic_arr).sum())

        row = {
            "horizon": int(h),
            "n_days": int(n_days),
            "ic_mean": float(np.nanmean(ic_arr)) if ic_arr.size else float("nan"),
            "ic_tstat": float(_tstat(ic_arr)) if ic_arr.size else 0.0,
            "rank_ic_mean": float(np.nanmean(ric_arr)) if ric_arr.size else float("nan"),
            "rank_ic_tstat": float(_tstat(ric_arr)) if ric_arr.size else 0.0,
            "spread_mean": float(np.nanmean(spr_arr)) if spr_arr.size else float("nan"),
            "spread_tstat": float(_tstat(spr_arr)) if spr_arr.size else 0.0,
            "spread_ir_ann_proxy": float(_ir_ann(spr_arr, int(trading_days), int(h))) if spr_arr.size else 0.0,
            "coverage_mean": float(np.nanmean(cov_arr)) if cov_arr.size else float("nan"),
        }

        row.update(sig.get(int(h), {}))
        rows.append(row)

    # Pick a default "best" horizon for quick iteration.
    best: Dict[str, Any] = {}
    if rows:
        def _score(r: Dict[str, Any]) -> float:
            # Prefer statistically strong rank IC, then spread IR.
            a = abs(float(r.get("rank_ic_tstat") or 0.0))
            b = abs(float(r.get("spread_ir_ann_proxy") or 0.0))
            return float(a * 10.0 + b)

        best_row = max(rows, key=_score)
        best = {
            "criterion": "abs_rank_ic_tstat_then_spread_ir",
            "horizon": int(best_row.get("horizon") or 1),
            "rank_ic_tstat": float(best_row.get("rank_ic_tstat") or 0.0),
            "spread_ir_ann_proxy": float(best_row.get("spread_ir_ann_proxy") or 0.0),
            "signal_overlap_mean": float(best_row.get("signal_overlap_mean") or float("nan")),
        }

    return {
        "enabled": True,
        "horizons": hs,
        "n_quantiles": int(n_quantiles),
        "min_obs_per_day": int(min_obs_per_day),
        "trading_days": int(trading_days),
        "metrics": rows,
        "best": best,
    }
