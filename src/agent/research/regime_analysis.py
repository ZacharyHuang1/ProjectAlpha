"""agent.research.regime_analysis

P2.13: simple regime breakdown utilities.

The goal is a lightweight diagnostic: "does this strategy behave differently
in different market conditions?" This is intentionally minimal and should not
be used as a full risk model.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.research.risk_exposures import close_to_returns, market_return, to_wide_close_volume


def _max_drawdown(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _summarize_returns(returns: pd.Series, trading_days: int) -> Dict[str, Any]:
    if returns is None or returns.empty:
        return {"information_ratio": 0.0, "annualized_return": 0.0, "max_drawdown": 0.0, "n_obs": 0}
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if r.size < 2:
        return {"information_ratio": 0.0, "annualized_return": 0.0, "max_drawdown": float(_max_drawdown(returns)), "n_obs": int(r.size)}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    ir = float(mu / sd * np.sqrt(float(trading_days))) if sd > 0 else 0.0
    ann = float((1.0 + mu) ** float(trading_days) - 1.0)
    return {"information_ratio": ir, "annualized_return": ann, "max_drawdown": float(_max_drawdown(returns)), "n_obs": int(r.size)}


def market_volatility_regime(
    ohlcv: pd.DataFrame,
    *,
    window: int = 20,
    min_obs: int = 20,
    buckets: int = 3,
) -> pd.Series:
    """Quantile-bucketed market volatility regime (lookahead-safe)."""

    close_wide, _ = to_wide_close_volume(ohlcv)
    ret_wide = close_to_returns(close_wide)
    mkt = market_return(ret_wide)
    vol = mkt.rolling(int(window), min_periods=int(min_obs)).std(ddof=1).shift(1)
    vol = vol.replace([np.inf, -np.inf], np.nan)

    # Use qcut on the non-nan portion; map back to full index.
    lbl = pd.Series(index=vol.index, dtype=float)
    vv = vol.dropna()
    if vv.empty:
        return lbl
    try:
        q = pd.qcut(vv, q=max(1, int(buckets)), labels=False, duplicates="drop")
        lbl.loc[q.index] = q.astype(float)
    except Exception:
        # Fallback: median split.
        med = float(vv.median())
        lbl.loc[vv.index] = (vv > med).astype(float)
    return lbl


def market_liquidity_regime(
    ohlcv: pd.DataFrame,
    *,
    window: int = 20,
    min_obs: int = 20,
    buckets: int = 3,
) -> Optional[pd.Series]:
    """Quantile-bucketed market liquidity regime from ADV (lookahead-safe)."""

    if "volume" not in ohlcv.columns:
        return None
    close_wide, vol_wide = to_wide_close_volume(ohlcv)
    if vol_wide is None:
        return None

    adv = (close_wide * vol_wide).rolling(int(window), min_periods=int(min_obs)).mean().shift(1)
    mkt_adv = np.log1p(adv.mean(axis=1, skipna=True))
    mkt_adv = mkt_adv.replace([np.inf, -np.inf], np.nan)

    lbl = pd.Series(index=mkt_adv.index, dtype=float)
    vv = mkt_adv.dropna()
    if vv.empty:
        return lbl
    try:
        q = pd.qcut(vv, q=max(1, int(buckets)), labels=False, duplicates="drop")
        lbl.loc[q.index] = q.astype(float)
    except Exception:
        med = float(vv.median())
        lbl.loc[vv.index] = (vv > med).astype(float)
    return lbl


def regime_performance(
    returns: pd.Series,
    regime: pd.Series,
    *,
    trading_days: int = 252,
    name: str = "regime",
) -> List[Dict[str, Any]]:
    """Compute performance metrics conditioned on a discrete regime label."""

    if returns is None or returns.empty or regime is None or regime.empty:
        return []

    r = returns.copy()
    g = regime.reindex(r.index)
    m = pd.DataFrame({"r": r, "g": g}).dropna()
    if m.empty:
        return []

    out: List[Dict[str, Any]] = []
    for k, grp in m.groupby("g"):
        s = _summarize_returns(grp["r"], trading_days=int(trading_days))
        out.append(
            {
                "name": str(name),
                "bucket": int(k) if float(k).is_integer() else float(k),
                "information_ratio": float(s.get("information_ratio") or 0.0),
                "annualized_return": float(s.get("annualized_return") or 0.0),
                "max_drawdown": float(s.get("max_drawdown") or 0.0),
                "n_obs": int(s.get("n_obs") or 0),
            }
        )

    out.sort(key=lambda x: float(x.get("bucket") or 0.0))
    return out
