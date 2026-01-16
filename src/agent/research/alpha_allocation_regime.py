"""agent.research.alpha_allocation_regime

P2.21: Regime-aware alpha allocation.

This extends the P2.19/ P2.20 alpha allocation layer by learning *separate*
alpha weights for different market regimes (e.g., low/med/high volatility).

Design goals:
- No test leakage: regime thresholds are fit on the fit segment only.
- Robust fallbacks: if a regime has too few samples, fall back to the global
  allocation for that split.
- Minimal dependencies: reuses the existing allocation solver.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from agent.research.alpha_allocation import fit_alpha_allocation
from agent.research.alpha_allocation import project_to_bounded_simplex


def fit_regime_allocations(
    returns: pd.DataFrame,
    regime_labels: pd.Series,
    *,
    trading_days: int = 252,
    score_metric: str = "information_ratio",
    lambda_corr: float = 0.5,
    l2: float = 1e-6,
    turnover_lambda: float = 0.0,
    prev_global: Optional[pd.Series] = None,
    prev_by_regime: Optional[Dict[str, pd.Series]] = None,
    max_weight: float = 1.0,
    use_abs_corr: bool = True,
    backend: str = "auto",
    solver: str = "",
    min_days: int = 30,
) -> Dict[str, Any]:
    """Fit a global allocation and per-regime allocations.

    Returns a dict with:
    - global_weights: pd.Series
    - weights_by_regime: Dict[str, pd.Series]
    - diagnostics: List[dict]
    - prev_by_regime: updated mapping for turnover anchoring
    """

    prev_by_regime = dict(prev_by_regime or {})
    diag_rows: list[dict] = []

    if returns is None or returns.empty or returns.shape[1] < 2:
        return {
            "global_weights": pd.Series(dtype=float),
            "weights_by_regime": {},
            "diagnostics": [{"error": "empty returns"}],
            "prev_by_regime": prev_by_regime,
        }

    r = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    g = pd.Series(regime_labels, index=r.index).astype(object)

    # Always fit a global allocation as the fallback.
    global_fit = fit_alpha_allocation(
        r,
        trading_days=int(trading_days),
        score_metric=str(score_metric),
        lambda_corr=float(lambda_corr),
        l2=float(l2),
        turnover_lambda=float(turnover_lambda),
        prev_weights=prev_global,
        max_weight=float(max_weight),
        use_abs_corr=bool(use_abs_corr),
        backend=str(backend),
        solver=str(solver),
    )
    w_global = global_fit.get("weights")
    if not isinstance(w_global, pd.Series) or w_global.empty:
        w_global = pd.Series(1.0 / float(r.shape[1]), index=r.columns, dtype=float)
        global_diag = {"method": "fallback_equal", "reason": "global_allocation_failed"}
    else:
        global_diag = dict(global_fit.get("diagnostics") or {})
    global_diag.update({"regime": "__global__", "n_days": int(len(r.index))})
    diag_rows.append(global_diag)

    weights_by_regime: Dict[str, pd.Series] = {}
    for reg, idx in g.dropna().groupby(g.dropna()).groups.items():
        try:
            reg = str(reg)
        except Exception:
            continue
        sub = r.loc[list(idx)]
        if sub.empty or int(len(sub.index)) < int(min_days):
            diag_rows.append({"regime": reg, "method": "skipped", "n_days": int(len(sub.index))})
            continue

        prev = prev_by_regime.get(reg)
        fit = fit_alpha_allocation(
            sub,
            trading_days=int(trading_days),
            score_metric=str(score_metric),
            lambda_corr=float(lambda_corr),
            l2=float(l2),
            turnover_lambda=float(turnover_lambda),
            prev_weights=prev,
            max_weight=float(max_weight),
            use_abs_corr=bool(use_abs_corr),
            backend=str(backend),
            solver=str(solver),
        )
        w = fit.get("weights")
        if not isinstance(w, pd.Series) or w.empty:
            diag_rows.append({"regime": reg, "method": "failed", "n_days": int(len(sub.index))})
            continue

        weights_by_regime[reg] = w
        prev_by_regime[reg] = w

        d = dict(fit.get("diagnostics") or {})
        d.update({"regime": reg, "n_days": int(len(sub.index))})
        diag_rows.append(d)

    return {
        "global_weights": w_global,
        "weights_by_regime": weights_by_regime,
        "diagnostics": diag_rows,
        "prev_by_regime": prev_by_regime,
    }


def build_daily_alpha_weights(
    *,
    labels: pd.Series,
    weights_by_regime: Dict[str, pd.Series],
    fallback: pd.Series,
    smoothing: float = 0.0,
    max_weight: float = 1.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert regime labels into a daily alpha weight matrix.

    The output is a DataFrame indexed by date with columns=alpha_id.
    Each row is projected back to the simplex after optional smoothing.
    """

    if labels is None or labels.empty:
        return pd.DataFrame(), {"error": "empty labels"}

    fb = pd.Series(fallback).astype(float)
    fb = fb.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    if float(fb.sum()) <= 0.0:
        fb = pd.Series(1.0 / float(len(fb.index)), index=fb.index, dtype=float)
    fb = fb / float(fb.sum())

    cols = list(fb.index.astype(str))
    idx = pd.to_datetime(pd.Index(labels.index)).sort_values()
    out = pd.DataFrame(0.0, index=idx, columns=cols)

    s = float(max(0.0, min(1.0, smoothing)))
    cap = float(max_weight) if (float(max_weight) > 0.0 and float(max_weight) < 1.0) else 1.0

    prev = fb.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    prev = project_to_bounded_simplex(prev, cap=cap)

    used_fallback = 0
    for dt in idx:
        reg = labels.get(dt)
        target = None
        if reg is not None and reg == reg:
            w = weights_by_regime.get(str(reg))
            if isinstance(w, pd.Series) and not w.empty:
                target = w.reindex(cols).fillna(0.0).to_numpy(dtype=float)
        if target is None:
            used_fallback += 1
            target = fb.reindex(cols).fillna(0.0).to_numpy(dtype=float)

        if s > 0.0:
            cur = (1.0 - s) * target + s * prev
        else:
            cur = target

        cur = project_to_bounded_simplex(cur, cap=cap)
        out.loc[dt, :] = cur
        prev = cur

    diag = {
        "n_days": int(len(out.index)),
        "n_alphas": int(len(out.columns)),
        "fallback_days": int(used_fallback),
        "fallback_frac": float(used_fallback) / float(max(len(out.index), 1)),
        "smoothing": float(s),
        "max_weight": float(max_weight),
    }
    return out, diag
