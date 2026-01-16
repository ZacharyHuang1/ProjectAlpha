"""agent.research.risk_exposures

Lightweight risk exposure estimators used by the research backtests.

All exposures are *lookahead-safe* by construction: they are shifted by 1 day
so that values at date t only use information available up to t-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExposureConfig:
    adv_window: int = 20
    beta_window: int = 60
    vol_window: int = 20
    min_obs: int = 20


def to_wide_close_volume(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Return (close_wide, volume_wide) with index=date and columns=instrument."""

    close_wide = ohlcv["close"].unstack("instrument").sort_index()
    volume_wide = None
    if "volume" in ohlcv.columns:
        volume_wide = ohlcv["volume"].unstack("instrument").sort_index()
    return close_wide, volume_wide


def close_to_returns(close_wide: pd.DataFrame) -> pd.DataFrame:
    r = close_wide.pct_change().replace([np.inf, -np.inf], np.nan)
    return r


def market_return(ret_wide: pd.DataFrame) -> pd.Series:
    """Equal-weighted market return."""

    return ret_wide.mean(axis=1, skipna=True).fillna(0.0)


def rolling_volatility(ret_wide: pd.DataFrame, *, window: int, min_obs: int) -> pd.DataFrame:
    vol = ret_wide.rolling(int(window), min_periods=int(min_obs)).std(ddof=1).shift(1)
    return vol


def rolling_log_adv(
    close_wide: pd.DataFrame,
    volume_wide: pd.DataFrame,
    *,
    window: int,
    min_obs: int,
) -> pd.DataFrame:
    adv = (close_wide * volume_wide).rolling(int(window), min_periods=int(min_obs)).mean().shift(1)
    return np.log1p(adv)


def rolling_beta(
    ret_wide: pd.DataFrame,
    mkt_ret: pd.Series,
    *,
    window: int,
    min_obs: int,
) -> pd.DataFrame:
    """Rolling beta of each instrument vs the equal-weight market return."""

    window = int(window)
    min_obs = int(min_obs)

    rm = pd.Series(mkt_ret, index=ret_wide.index, dtype=float)

    mean_rm = rm.rolling(window, min_periods=min_obs).mean()
    mean_rm2 = (rm * rm).rolling(window, min_periods=min_obs).mean()
    var_rm = (mean_rm2 - mean_rm * mean_rm).replace(0.0, np.nan)

    mean_ri = ret_wide.rolling(window, min_periods=min_obs).mean()
    mean_rirm = (ret_wide.mul(rm, axis=0)).rolling(window, min_periods=min_obs).mean()
    cov = mean_rirm - mean_ri.mul(mean_rm, axis=0)

    beta = cov.div(var_rm, axis=0).shift(1)
    return beta
