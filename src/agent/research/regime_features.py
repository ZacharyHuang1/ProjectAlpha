"""agent.research.regime_features

P2.21: Regime-aware allocation needs a simple, lookahead-safe notion of
"market state".

This module produces a small set of daily market features derived from the
OHLCV panel. All features are shifted by 1 day where appropriate so that the
value at date *t* only uses information up to *t-1*.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from agent.research.risk_exposures import close_to_returns, market_return, to_wide_close_volume


def compute_market_feature_frame(
    ohlcv: pd.DataFrame,
    *,
    window: int = 20,
    min_obs: int = 20,
) -> pd.DataFrame:
    """Compute daily market features used for regime labeling.

    Returns a DataFrame indexed by datetime with columns:
    - mkt_ret: equal-weight market return
    - mkt_vol: rolling std of mkt_ret (shifted by 1)
    - mkt_trend: rolling mean of mkt_ret (shifted by 1)
    - mkt_liq: rolling mean log(ADV) (shifted by 1) if volume is present
    - xs_disp: cross-sectional dispersion of single-name returns (shifted by 1)
    """

    close_wide, vol_wide = to_wide_close_volume(ohlcv)
    ret_wide = close_to_returns(close_wide)

    mkt_ret = market_return(ret_wide).astype(float)
    mkt_vol = mkt_ret.rolling(int(window), min_periods=int(min_obs)).std(ddof=1).shift(1)
    mkt_trend = mkt_ret.rolling(int(window), min_periods=int(min_obs)).mean().shift(1)

    xs_disp = ret_wide.std(axis=1, skipna=True).rolling(int(window), min_periods=int(min_obs)).mean().shift(1)

    mkt_liq: Optional[pd.Series] = None
    if vol_wide is not None:
        adv = (close_wide * vol_wide).rolling(int(window), min_periods=int(min_obs)).mean().shift(1)
        mkt_liq = np.log1p(adv.mean(axis=1, skipna=True)).astype(float)

    out = pd.DataFrame(
        {
            "mkt_ret": mkt_ret,
            "mkt_vol": mkt_vol,
            "mkt_trend": mkt_trend,
            "xs_disp": xs_disp,
        }
    )
    if mkt_liq is not None:
        out["mkt_liq"] = mkt_liq
    else:
        out["mkt_liq"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out
