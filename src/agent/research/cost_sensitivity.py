"""agent.research.cost_sensitivity

P2.14: Cost sensitivity curves and break-even estimates.

This module answers a practical question:
"How much transaction cost / spread / impact / borrow can this strategy tolerate?"

We operate in an execution-only setting: the realized trading path is fixed.
Only cost assumptions change.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig


def _max_drawdown(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _summarize_returns(r: pd.Series, trading_days: int) -> Dict[str, float]:
    if r is None or r.empty:
        return {"information_ratio": 0.0, "annualized_return": 0.0, "max_drawdown": 0.0, "n_obs": 0.0}
    arr = r.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "information_ratio": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": float(_max_drawdown(r)),
            "n_obs": int(r.size),
        }
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    ir = float(mu / sd * np.sqrt(float(trading_days))) if sd > 0 else 0.0
    ann = float((1.0 + mu) ** float(trading_days) - 1.0)
    return {
        "information_ratio": ir,
        "annualized_return": ann,
        "max_drawdown": float(_max_drawdown(r)),
        "n_obs": int(r.size),
    }


def _estimate_break_even(values: List[float], mus: List[float]) -> Dict[str, Any]:
    """Estimate the parameter value where mean daily return crosses zero."""

    if not values or not mus or len(values) != len(mus):
        return {"break_even": None, "within_grid": False, "status": "invalid"}

    # Sort by values (defensive).
    pairs = sorted([(float(x), float(y)) for x, y in zip(values, mus)], key=lambda t: t[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    if ys[0] <= 0.0:
        return {"break_even": float(xs[0]), "within_grid": False, "status": "below_grid"}
    if ys[-1] >= 0.0:
        return {"break_even": float(xs[-1]), "within_grid": False, "status": "above_grid"}

    # Find the first index where we go negative.
    j = None
    for i in range(1, len(xs)):
        if ys[i] <= 0.0:
            j = i
            break
    if j is None:
        return {"break_even": None, "within_grid": False, "status": "no_cross"}

    x0, y0 = xs[j - 1], ys[j - 1]
    x1, y1 = xs[j], ys[j]
    if y1 == y0:
        return {"break_even": float(x0), "within_grid": True, "status": "flat"}

    be = x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0)
    return {"break_even": float(be), "within_grid": True, "status": "ok"}


def _returns_from_daily(
    daily: List[Dict[str, Any]],
    *,
    trading_days: int,
    base: BacktestConfig,
    borrow_rates_present: bool,
    cost_bps: float,
    half_spread_bps: float,
    impact_bps: float,
    borrow_bps: Optional[float] = None,
    borrow_multiplier: Optional[float] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Compute gross and net returns for a given cost parameter set."""

    if not daily:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    idx = pd.to_datetime([r.get("datetime") for r in daily])
    gross = np.asarray([float(r.get("gross_return") or 0.0) for r in daily], dtype=float)
    turnover = np.asarray([float(r.get("turnover") or 0.0) for r in daily], dtype=float)
    impact_unit = np.asarray([float(r.get("impact_unit") or 0.0) for r in daily], dtype=float)
    gross_short = np.asarray([float(r.get("gross_short") or 0.0) for r in daily], dtype=float)
    borrow_realized = np.asarray([float(r.get("borrow") or 0.0) for r in daily], dtype=float)

    linear = turnover * (float(cost_bps) / 10000.0)
    spread = (2.0 * turnover) * (float(half_spread_bps) / 10000.0)
    impact = impact_unit * (float(impact_bps) / 10000.0)

    if borrow_rates_present:
        base_mult = float(getattr(base, "borrow_cost_multiplier", 1.0) or 1.0)
        new_mult = float(borrow_multiplier) if borrow_multiplier is not None else base_mult
        scale = (new_mult / base_mult) if base_mult > 0 else 0.0
        borrow = borrow_realized * float(scale)
    else:
        bbps = float(borrow_bps) if borrow_bps is not None else float(getattr(base, "borrow_bps", 0.0) or 0.0)
        mult = float(getattr(base, "borrow_cost_multiplier", 1.0) or 1.0)
        borrow = gross_short * (bbps / 10000.0 / float(trading_days)) * mult

    net = gross - linear - spread - impact - borrow
    return pd.Series(gross, index=idx).sort_index(), pd.Series(net, index=idx).sort_index()


def compute_cost_sensitivity(
    daily: List[Dict[str, Any]],
    *,
    base_cfg: BacktestConfig,
    borrow_rates_present: bool,
    linear_bps_grid: List[float],
    half_spread_bps_grid: List[float],
    impact_bps_grid: List[float],
    borrow_bps_grid: List[float],
    borrow_mult_grid: List[float],
) -> Dict[str, Any]:
    """Compute cost sensitivity curves and break-even points for one realized path."""

    td = int(getattr(base_cfg, "trading_days", 252) or 252)

    base_cost_bps = float(getattr(base_cfg, "commission_bps", 0.0) + getattr(base_cfg, "slippage_bps", 0.0))
    base_half_spread_bps = float(getattr(base_cfg, "half_spread_bps", 0.0) or 0.0)
    base_impact_bps = float(getattr(base_cfg, "impact_bps", 0.0) or 0.0)
    base_borrow_bps = float(getattr(base_cfg, "borrow_bps", 0.0) or 0.0)
    base_borrow_mult = float(getattr(base_cfg, "borrow_cost_multiplier", 1.0) or 1.0)

    def _ensure(grid: List[float], base_val: float) -> List[float]:
        g = [float(x) for x in (grid or []) if np.isfinite(float(x))]
        if not any(abs(float(x) - float(base_val)) < 1e-12 for x in g):
            g.append(float(base_val))
        g = sorted(set([round(float(x), 10) for x in g]))
        return [float(x) for x in g]

    linear_bps_grid = _ensure(linear_bps_grid, base_cost_bps)
    half_spread_bps_grid = _ensure(half_spread_bps_grid, base_half_spread_bps)
    impact_bps_grid = _ensure(impact_bps_grid, base_impact_bps)

    # Borrow has two modes: absolute bps (when rates are constant) or a multiplier (when a borrow curve is provided).
    if borrow_rates_present:
        borrow_mult_grid = _ensure(borrow_mult_grid, base_borrow_mult)
        borrow_bps_grid = []
    else:
        borrow_bps_grid = _ensure(borrow_bps_grid, base_borrow_bps)
        borrow_mult_grid = []

    curves: List[Dict[str, Any]] = []
    break_even: List[Dict[str, Any]] = []

    # Baseline gross series for drag calculations.
    gross_base, _ = _returns_from_daily(
        daily,
        trading_days=td,
        base=base_cfg,
        borrow_rates_present=borrow_rates_present,
        cost_bps=base_cost_bps,
        half_spread_bps=base_half_spread_bps,
        impact_bps=base_impact_bps,
        borrow_bps=base_borrow_bps,
        borrow_multiplier=base_borrow_mult,
    )
    gross_mu = float(gross_base.mean()) if not gross_base.empty else 0.0

    def _run_param(parameter: str, values: List[float]) -> None:
        mus: List[float] = []
        xs: List[float] = []
        for v in values:
            kw = {
                "cost_bps": base_cost_bps,
                "half_spread_bps": base_half_spread_bps,
                "impact_bps": base_impact_bps,
                "borrow_bps": base_borrow_bps,
                "borrow_multiplier": base_borrow_mult,
            }
            if parameter == "linear_cost_bps":
                kw["cost_bps"] = float(v)
            elif parameter == "half_spread_bps":
                kw["half_spread_bps"] = float(v)
            elif parameter == "impact_bps":
                kw["impact_bps"] = float(v)
            elif parameter == "borrow_bps":
                kw["borrow_bps"] = float(v)
            elif parameter == "borrow_multiplier":
                kw["borrow_multiplier"] = float(v)

            g, net = _returns_from_daily(
                daily,
                trading_days=td,
                base=base_cfg,
                borrow_rates_present=borrow_rates_present,
                **kw,
            )
            s = _summarize_returns(net, trading_days=td)
            mu = float(net.mean()) if not net.empty else 0.0
            mus.append(mu)
            xs.append(float(v))

            drag_bps = float((gross_mu - mu) * 10000.0)
            curves.append(
                {
                    "parameter": parameter,
                    "value": float(v),
                    "information_ratio": float(s.get("information_ratio") or 0.0),
                    "annualized_return": float(s.get("annualized_return") or 0.0),
                    "max_drawdown": float(s.get("max_drawdown") or 0.0),
                    "n_obs": int(s.get("n_obs") or 0),
                    "mean_cost_drag_bps": float(drag_bps),
                }
            )

        be = _estimate_break_even(xs, mus)
        break_even.append(
            {
                "parameter": parameter,
                "break_even": be.get("break_even"),
                "within_grid": bool(be.get("within_grid")),
                "status": be.get("status"),
                "min": float(min(xs)) if xs else None,
                "max": float(max(xs)) if xs else None,
            }
        )

    _run_param("linear_cost_bps", linear_bps_grid)
    _run_param("half_spread_bps", half_spread_bps_grid)
    _run_param("impact_bps", impact_bps_grid)
    if borrow_rates_present:
        _run_param("borrow_multiplier", borrow_mult_grid)
    else:
        _run_param("borrow_bps", borrow_bps_grid)

    curves.sort(key=lambda r: (str(r.get("parameter")), float(r.get("value") or 0.0)))
    break_even.sort(key=lambda r: str(r.get("parameter")))

    return {
        "enabled": True,
        "borrow_mode": "multiplier" if borrow_rates_present else "bps",
        "base": {
            "backtest": asdict(base_cfg),
            "base_linear_cost_bps": base_cost_bps,
            "base_half_spread_bps": base_half_spread_bps,
            "base_impact_bps": base_impact_bps,
            "base_borrow_bps": base_borrow_bps,
            "base_borrow_multiplier": base_borrow_mult,
        },
        "break_even": break_even,
        "curves": curves,
    }
