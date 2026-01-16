"""agent.research.utility_weight_calibration

P2.29: Utility-weight auto calibration.

The Pareto "utility" selector works in *normalized goodness space* (0..1) where
higher is better for every objective. That means weights are *unitless* and
should primarily reflect *preference strength*.

This module provides a deterministic heuristic that maps a small set of
configuration knobs (turnover cost bps + selection constraints) to a
reasonable default weight map.

Design goals:
  - deterministic: same inputs -> same weights
  - safe: no extra dependencies
  - conservative: never creates negative weights
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def calibrate_utility_weights(
    *,
    turnover_cost_bps: float,
    constraints: Optional[Dict[str, Any]] = None,
    include_stability: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Return (weights, meta) for the Pareto utility selector.

    The output is a *superset* of possible objective keys. Selectors will ignore
    weights for keys not present in the objective list.
    """

    cons = dict(constraints or {})
    cost_bps = float(max(0.0, _sf(turnover_cost_bps, default=0.0)))

    # A simple cost-aware scale around the common default (0.2bps/unit turnover).
    # This only affects the *relative* weighting of execution realism terms.
    cost_scale = float(np.clip(cost_bps / 0.2 if cost_bps > 0.0 else 1.0, 0.5, 2.5))

    def _has(k: str) -> bool:
        return cons.get(k) is not None

    # Base: always care about objective.
    w: Dict[str, float] = {
        "objective": 1.0,
        "holdings_objective": 1.0,
    }

    # Execution realism knobs.
    w["alpha_weight_turnover_mean"] = float(0.25 * cost_scale + (0.25 if _has("max_alpha_weight_turnover_mean") else 0.0))
    w["turnover_cost_drag_bps_mean"] = float(0.25 * cost_scale + (0.25 if _has("max_turnover_cost_drag_bps_mean") else 0.0))
    w["regime_switch_rate_mean"] = float(0.15 + (0.15 if _has("max_regime_switch_rate_mean") else 0.0))
    w["fallback_frac_mean"] = float(0.15 + (0.15 if _has("max_fallback_frac_mean") else 0.0))

    # Holdings-only execution diagnostics (used when present).
    w["ensemble_turnover_mean"] = float(0.20 * cost_scale)
    w["ensemble_cost_mean"] = float(0.20 * cost_scale)
    w["ensemble_borrow_mean"] = float(0.10 * cost_scale)

    # Stability objectives (optional).
    if include_stability:
        w["objective_split_std"] = 0.20
        w["objective_split_min"] = 0.20
        w["holdings_objective_split_std"] = 0.20
        w["holdings_objective_split_min"] = 0.20

    # Ensure all weights are finite and non-negative.
    for k in list(w.keys()):
        v = _sf(w.get(k), default=0.0)
        if not np.isfinite(v) or v < 0.0:
            w[k] = 0.0
        else:
            w[k] = float(v)

    meta: Dict[str, Any] = {
        "enabled": True,
        "method": "heuristic_v1",
        "turnover_cost_bps": float(cost_bps),
        "cost_scale": float(cost_scale),
        "constraints": dict(cons),
        "include_stability": bool(include_stability),
    }
    return w, meta


def weights_to_kv_string(weights: Dict[str, float]) -> str:
    """Format weights as a stable comma-separated key=value string."""

    items = []
    for k in sorted((weights or {}).keys()):
        try:
            v = float(weights[k])
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        items.append(f"{k}={float(v):.6g}")
    return ",".join(items)
