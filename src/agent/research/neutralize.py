"""agent.research.neutralize

Small utilities to neutralize portfolio weights against exposures.

This is a research-grade implementation: deterministic, dependency-light, and
safe to run on medium-size universes.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


def make_sector_dummies(
    instruments: Sequence[str],
    sector_map: Dict[str, str],
    *,
    drop_first: bool = True,
) -> pd.DataFrame:
    """Create one-hot sector dummies with index=instruments."""

    s = pd.Series({i: sector_map.get(i) for i in instruments}, name="sector")
    d = pd.get_dummies(s, prefix="sector", dummy_na=False)
    if drop_first and d.shape[1] > 1:
        d = d.iloc[:, 1:]
    d.index = pd.Index(instruments)
    return d.astype(float)


def neutralize_weights(
    w: pd.Series,
    exposures: pd.DataFrame,
    *,
    add_intercept: bool = True,
    ridge: float = 1e-8,
) -> pd.Series:
    """Project weights to be orthogonal to the exposure columns."""

    if w is None or w.empty:
        return w

    active = w.index[w != 0.0]
    if len(active) < 3:
        return w

    X = exposures.reindex(active)
    if X is None or X.empty:
        return w

    # Fill missing exposures within the active set.
    X = X.copy()
    for c in X.columns:
        med = float(X[c].median(skipna=True)) if X[c].notna().any() else 0.0
        X[c] = X[c].astype(float).fillna(med)

    if add_intercept:
        X.insert(0, "intercept", 1.0)

    y = w.reindex(active).astype(float).to_numpy()
    A = X.to_numpy(dtype=float)

    if A.shape[0] <= A.shape[1]:
        return w

    XtX = A.T @ A
    XtX = XtX + float(ridge) * np.eye(XtX.shape[0])
    Xty = A.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
    except Exception:
        coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    y_hat = A @ coef
    w_out = w.copy()
    w_out.loc[active] = (y - y_hat)
    return w_out


def rescale_long_short(
    w: pd.Series,
    *,
    gross_long: float = 0.5,
    gross_short: float = 0.5,
    scale_up: bool = True,
) -> Optional[pd.Series]:
    """Rescale positive/negative legs to target gross exposures."""

    if w is None or w.empty:
        return None

    w = w.astype(float).copy()
    pos = w[w > 0.0]
    neg = w[w < 0.0]
    if pos.empty or neg.empty:
        return None

    s_pos = float(pos.sum())
    s_neg = float((-neg).sum())
    if s_pos <= 0.0 or s_neg <= 0.0:
        return None

    a = float(gross_long) / s_pos
    b = float(gross_short) / s_neg
    if not scale_up:
        a = min(1.0, a)
        b = min(1.0, b)

    w.loc[pos.index] = w.loc[pos.index] * a
    w.loc[neg.index] = w.loc[neg.index] * b
    return w


def clip_weights(w: pd.Series, *, max_abs_weight: float) -> pd.Series:
    if not np.isfinite(max_abs_weight) or max_abs_weight <= 0.0:
        return w
    return w.clip(lower=-float(max_abs_weight), upper=float(max_abs_weight))


def drop_small_weights(w: pd.Series, *, eps: float = 0.0) -> pd.Series:
    if eps <= 0.0:
        return w
    out = w.copy()
    out.loc[out.abs() < float(eps)] = 0.0
    return out
