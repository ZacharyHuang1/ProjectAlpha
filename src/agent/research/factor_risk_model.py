"""agent.research.factor_risk_model

A lightweight factor risk model estimator for research backtests.

We estimate a simple model:
  Sigma = B F B^T + D

- B: cross-sectional factor loadings (exposures)
- F: factor covariance (estimated from trailing factor returns)
- D: diagonal idiosyncratic variance (estimated from trailing residuals)

The estimator is lookahead-safe when the input returns are trailing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRiskModel:
    loadings: pd.DataFrame  # index=instrument, columns=factors (standardized)
    factor_cov: np.ndarray  # k x k, annualized
    idio_var: pd.Series  # index=instrument, annualized
    meta: Dict[str, Any]


def _standardize_loadings(B: pd.DataFrame) -> pd.DataFrame:
    out = B.copy()
    for c in out.columns:
        col = out[c].astype(float).replace([np.inf, -np.inf], np.nan)
        med = float(col.median(skipna=True)) if col.notna().any() else 0.0
        col = col.fillna(med)
        mu = float(col.mean())
        sd = float(col.std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            sd = 1.0
        out[c] = (col - mu) / sd
    return out


def _make_psd(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (M + M.T)
    try:
        w, V = np.linalg.eigh(A)
        w = np.clip(w, eps, None)
        return (V * w) @ V.T
    except Exception:
        return A + eps * np.eye(A.shape[0], dtype=float)


def _winsorize_matrix(X: np.ndarray, clip_sigma: float) -> np.ndarray:
    if clip_sigma <= 0.0:
        return X
    out = np.asarray(X, dtype=float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        sd = float(np.std(col, ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        lo = -clip_sigma * sd
        hi = clip_sigma * sd
        out[:, j] = np.clip(col, lo, hi)
    return out


def _effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def _ewm_cov(X: np.ndarray, halflife: float) -> Tuple[np.ndarray, float]:
    T = int(X.shape[0])
    hl = float(max(1e-6, halflife))
    decay = float(np.exp(np.log(0.5) / hl))
    ages = (T - 1) - np.arange(T, dtype=float)
    w = decay ** ages  # most recent has weight 1.0
    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return np.zeros((X.shape[1], X.shape[1]), dtype=float), 0.0
    mu = (w[:, None] * X).sum(axis=0) / w_sum
    Xc = X - mu[None, :]
    cov = (Xc * w[:, None]).T @ Xc / w_sum
    return cov, _effective_sample_size(w)


def _oas_shrink_to_identity(S: np.ndarray, n_samples: float) -> Tuple[np.ndarray, float]:
    p = int(S.shape[0])
    if p <= 0:
        return S, 0.0
    n = float(max(1.0, n_samples))
    tr = float(np.trace(S))
    mu = tr / float(p)
    tr2 = float(np.sum(S * S))
    denom = (n + 1.0 - 2.0 / p) * (tr2 - (tr * tr) / float(p))
    if denom <= 1e-18:
        return S, 0.0
    num = (1.0 - 2.0 / p) * tr2 + tr * tr
    alpha = float(np.clip(num / denom, 0.0, 1.0))
    F = mu * np.eye(p, dtype=float)
    out = (1.0 - alpha) * S + alpha * F
    return out, alpha


def estimate_factor_risk_model(
    returns_wide: pd.DataFrame,
    loadings: pd.DataFrame,
    *,
    window: int,
    min_obs: int,
    ridge: float = 1e-3,
    cov_shrink: float = 0.2,
    cov_shrink_method: str = "fixed",
    cov_estimator: str = "sample",
    ewm_halflife: float = 20.0,
    factor_return_clip_sigma: float = 6.0,
    idio_shrink: float = 0.2,
    idio_clip_q: float = 0.99,
    trading_days: int = 252,
) -> Optional[FactorRiskModel]:
    """Estimate a (B, F, D) factor risk model from trailing returns.

    returns_wide:
      date x instrument returns (daily). Should be trailing (no lookahead).

    loadings:
      instrument x factor exposures for the same instrument set.

    Notes:
      - cov_shrink_method='fixed' uses cov_shrink to shrink to diagonal.
      - cov_shrink_method='oas' uses an automatic OAS shrinkage to identity.
      - cov_estimator='ewm' uses exponential weights and an n_eff proxy.
    """

    if returns_wide is None or returns_wide.empty or loadings is None or loadings.empty:
        return None

    window = max(5, int(window))
    min_obs = max(2, int(min_obs))
    ridge = float(max(1e-12, ridge))
    cov_shrink = float(np.clip(cov_shrink, 0.0, 1.0))
    idio_shrink = float(np.clip(idio_shrink, 0.0, 1.0))
    idio_clip_q = float(np.clip(idio_clip_q, 0.0, 1.0))

    names = pd.Index([str(x) for x in loadings.index])
    R = returns_wide.copy()
    R.columns = pd.Index([str(x) for x in R.columns])
    R = R.reindex(columns=names)

    if R.shape[0] > window:
        R = R.iloc[-window:]

    T = int(R.shape[0])
    if T < min_obs:
        return None

    Rm = R.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    B0 = loadings.copy()
    B0.index = pd.Index([str(x) for x in B0.index])
    B0 = B0.reindex(index=names)
    B = _standardize_loadings(B0)

    if B.shape[1] == 0 or B.shape[0] < 4:
        return None

    X = B.to_numpy(dtype=float)  # n x k
    n, k = X.shape

    XtX = X.T @ X
    XtX = XtX + ridge * np.eye(k)
    try:
        inv = np.linalg.inv(XtX)
    except Exception:
        inv = np.linalg.pinv(XtX)

    # Cross-sectional regression each day: f_t = (X'X + ridge I)^{-1} X' r_t
    Fret = (Rm @ X) @ inv  # T x k
    if Fret.shape[0] < 2:
        return None

    Fret = _winsorize_matrix(Fret, float(factor_return_clip_sigma))

    cov_est = str(cov_estimator or "sample").lower().strip()
    if cov_est == "ewm":
        Fcov, n_eff = _ewm_cov(Fret, float(ewm_halflife))
    else:
        Fcov = np.cov(Fret, rowvar=False, ddof=1)
        n_eff = float(Fret.shape[0])

    if np.ndim(Fcov) == 0:
        Fcov = np.array([[float(Fcov)]], dtype=float)

    Fcov = 0.5 * (Fcov + Fcov.T)

    shrink_method = str(cov_shrink_method or "fixed").lower().strip()
    shrink_intensity = None
    if shrink_method in {"oas", "oas_identity"}:
        Fcov, alpha = _oas_shrink_to_identity(Fcov, n_eff)
        shrink_intensity = float(alpha)
    else:
        if cov_shrink > 0.0:
            Fcov = (1.0 - cov_shrink) * Fcov + cov_shrink * np.diag(np.diag(Fcov))
            shrink_intensity = float(cov_shrink)

    Fcov = _make_psd(Fcov, eps=1e-12) * float(trading_days)

    # Residuals and idiosyncratic variance.
    pred = Fret @ X.T  # T x n
    resid = Rm - pred
    idv = np.var(resid, axis=0, ddof=1)
    idv = np.where(np.isfinite(idv), idv, 0.0)
    idv = np.clip(idv, 0.0, None) * float(trading_days)

    if idio_clip_q > 0.0 and idio_clip_q < 1.0 and idv.size:
        hi = float(np.quantile(idv, idio_clip_q))
        if np.isfinite(hi) and hi > 0.0:
            idv = np.clip(idv, 0.0, hi)

    if idio_shrink > 0.0 and idv.size:
        med = float(np.median(idv))
        idv = (1.0 - idio_shrink) * idv + idio_shrink * med

    idio_var = pd.Series(idv, index=names, dtype=float)

    meta: Dict[str, Any] = {
        "n_days": int(T),
        "n_names": int(n),
        "n_factors": int(k),
        "ridge": float(ridge),
        "cov_estimator": str(cov_est),
        "cov_shrink_method": str(shrink_method),
        "cov_shrink": float(cov_shrink),
        "shrink_intensity": shrink_intensity,
        "ewm_halflife": float(ewm_halflife),
        "n_eff": float(n_eff),
        "factor_return_clip_sigma": float(factor_return_clip_sigma),
        "idio_shrink": float(idio_shrink),
        "idio_clip_q": float(idio_clip_q),
        "annualized": True,
    }

    return FactorRiskModel(loadings=B, factor_cov=Fcov, idio_var=idio_var, meta=meta)
