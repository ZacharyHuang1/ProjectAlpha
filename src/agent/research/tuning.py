"""agent.research.tuning

P2.12: lightweight parameter tuning utilities.

This module provides:
- deterministic grid generation for BacktestConfig parameters
- a small helper to downsample large grids

The goal is research productivity (quick sweeps) without introducing
external dependencies.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from agent.research.portfolio_backtest import BacktestConfig


def _as_float_list(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append(float(fv))
        except Exception:
            continue
    # Deduplicate while preserving order.
    seen = set()
    uniq: List[float] = []
    for v in out:
        key = round(float(v), 12)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


def _fmt_id(x: float) -> str:
    # Keep ids filesystem-friendly.
    s = f"{float(x):.4g}"
    s = s.replace("-", "m").replace(".", "p")
    return s


def default_sweep_param_lists(base: BacktestConfig) -> Dict[str, List[float]]:
    """Reasonable small defaults for a sweep.

    These defaults are intentionally small to keep runtime manageable.
    Users can override each list via CLI.
    """

    toc = float(getattr(base, "optimizer_turnover_cap", 0.0) or 0.0)
    cap = float(getattr(base, "max_abs_weight", 0.0) or 0.0)
    ra = float(getattr(base, "optimizer_risk_aversion", 0.0) or 0.0)
    ca = float(getattr(base, "optimizer_cost_aversion", 1.0) or 0.0)

    turnover_caps = [0.0, toc] if toc > 0 else [0.0, 0.10, 0.20]
    max_abs_weights = [0.0, cap] if cap > 0 else [0.0, 0.02]
    risk_aversions = [0.0, ra] if ra > 0 else [0.0, 5.0]
    cost_aversions = [0.0, ca] if ca > 0 else [0.0, 1.0]

    return {
        "optimizer_turnover_cap": _as_float_list(turnover_caps),
        "max_abs_weight": _as_float_list(max_abs_weights),
        "optimizer_risk_aversion": _as_float_list(risk_aversions),
        "optimizer_cost_aversion": _as_float_list(cost_aversions),
    }


def downsample_grid(grid: List[Tuple[Any, ...]], max_points: int) -> List[Tuple[Any, ...]]:
    """Deterministically downsample a cartesian product."""

    if max_points <= 0 or len(grid) <= max_points:
        return grid
    idx = np.linspace(0, len(grid) - 1, int(max_points)).astype(int)
    idx = np.unique(idx)
    return [grid[int(i)] for i in idx]


def generate_sweep_configs(
    base: BacktestConfig,
    *,
    param_lists: Dict[str, List[float]],
    max_combos: int = 24,
) -> List[Dict[str, Any]]:
    """Create a deterministic list of BacktestConfig variants."""

    keys = list(param_lists.keys())
    values = [list(param_lists.get(k) or []) for k in keys]
    if not keys or any(len(v) == 0 for v in values):
        return [{"config_id": "base", "params": {}, "bt_config": base}]

    combos = list(product(*values))
    combos = downsample_grid(combos, max_points=int(max_combos))

    out: List[Dict[str, Any]] = []
    for tpl in combos:
        params = {k: float(v) for k, v in zip(keys, tpl)}
        cfg = replace(base, **params)
        cid = "_".join([f"{k[:2]}{_fmt_id(float(params[k]))}" for k in keys])
        out.append({"config_id": cid, "params": params, "bt_config": cfg})

    # Ensure the base config is evaluated as a reference.
    out.insert(0, {"config_id": "base", "params": {}, "bt_config": base})
    return out
