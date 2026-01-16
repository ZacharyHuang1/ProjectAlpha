"""agent.research.alpha_allocation_tuning

P2.20: Meta-tuning utilities for alpha allocation.

Goal
----
Alpha allocation has a small set of hyperparameters (e.g., correlation penalty,
max single-alpha weight, and a smoothing/turnover penalty vs the previous split).

This module implements a deterministic, dependency-light meta-tuning routine that:
- fits allocation weights on each split's *train* segment
- scores the resulting alpha-ensemble on that split's *valid* segment
- aggregates the valid performance across splits to pick a single best config

No future leakage: test segments are never used for tuning.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.research.alpha_allocation import fit_alpha_allocation


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


def downsample_grid(grid: List[Tuple[Any, ...]], max_points: int) -> List[Tuple[Any, ...]]:
    if max_points <= 0 or len(grid) <= max_points:
        return grid
    idx = np.linspace(0, len(grid) - 1, int(max_points)).astype(int)
    idx = np.unique(idx)
    return [grid[int(i)] for i in idx]


def _information_ratio(arr: np.ndarray, trading_days: int) -> float:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(mu / sd * np.sqrt(float(trading_days)))


def _annualized_return(mu_daily: float, trading_days: int) -> float:
    try:
        return float((1.0 + float(mu_daily)) ** float(trading_days) - 1.0)
    except Exception:
        return 0.0


def default_allocation_sweep_param_lists(
    *,
    base_lambda_corr: float,
    base_max_weight: float,
    base_turnover_lambda: float,
) -> Dict[str, List[float]]:
    """Small default grids for allocation meta-tuning."""

    lam = float(base_lambda_corr)
    mw = float(base_max_weight)
    tl = float(base_turnover_lambda)

    lambda_corr = [0.0, lam] if lam > 0 else [0.0, 0.2, 0.5, 0.8]
    max_weight = [mw, 1.0] if (mw > 0 and mw < 1.0) else [0.5, 0.8, 1.0]
    turnover_lambda = [0.0, tl] if tl > 0 else [0.0, 0.5, 2.0]

    return {
        "lambda_corr": _as_float_list(lambda_corr),
        "max_weight": _as_float_list(max_weight),
        "turnover_lambda": _as_float_list(turnover_lambda),
    }


def generate_allocation_sweep_configs(
    *,
    param_lists: Dict[str, List[float]],
    max_combos: int = 24,
) -> List[Dict[str, Any]]:
    """Deterministically generate a small grid of allocation configs."""

    keys = ["lambda_corr", "max_weight", "turnover_lambda"]
    values = [list(param_lists.get(k) or []) for k in keys]
    if any(len(v) == 0 for v in values):
        return [{"config_id": "base", "params": {}}]

    combos = list(product(*values))
    combos = downsample_grid(combos, max_points=int(max_combos))

    out: List[Dict[str, Any]] = []
    for lam, mw, tl in combos:
        params = {
            "lambda_corr": float(lam),
            "max_weight": float(mw),
            "turnover_lambda": float(tl),
        }
        cid = f"lam{lam:.3g}_mw{mw:.3g}_tl{tl:.3g}".replace("-", "m").replace(".", "p")
        out.append({"config_id": cid, "params": params})

    out.insert(0, {"config_id": "base", "params": {}})
    return out


def meta_tune_alpha_allocation(
    *,
    train_by_split: Dict[int, pd.DataFrame],
    valid_by_split: Dict[int, pd.DataFrame],
    trading_days: int,
    score_metric: str,
    use_abs_corr: bool,
    backend: str,
    solver: str,
    base_params: Dict[str, Any],
    param_lists: Dict[str, List[float]],
    max_combos: int = 24,
    min_days: int = 30,
    tune_metric: str = "information_ratio",
) -> Dict[str, Any]:
    """Pick a single allocation config by aggregating valid performance across splits."""

    configs = generate_allocation_sweep_configs(param_lists=param_lists, max_combos=int(max_combos))

    rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_v = float("-inf")

    split_ids = sorted(set(train_by_split.keys()).intersection(set(valid_by_split.keys())))

    for cfg in configs:
        params = dict(base_params)
        params.update(cfg.get("params") or {})
        lam = float(params.get("lambda_corr") or 0.0)
        mw = float(params.get("max_weight") or 1.0)
        tl = float(params.get("turnover_lambda") or 0.0)

        w_prev: Optional[pd.Series] = None
        valid_series: List[pd.Series] = []
        n_used = 0
        turnover_list: List[float] = []

        for sid in split_ids:
            tr = train_by_split.get(int(sid))
            va = valid_by_split.get(int(sid))
            if tr is None or va is None or tr.empty or va.empty:
                continue

            tr2 = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            va2 = va.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Keep only alphas present in both segments.
            cols = [c for c in tr2.columns if c in va2.columns]
            if len(cols) < 2:
                continue
            tr2 = tr2[cols]
            va2 = va2[cols]
            if int(len(tr2.index)) < int(min_days):
                continue

            alloc = fit_alpha_allocation(
                tr2,
                trading_days=int(trading_days),
                score_metric=str(score_metric),
                lambda_corr=float(lam),
                l2=float(params.get("l2") or 0.0),
                turnover_lambda=float(tl),
                prev_weights=w_prev,
                max_weight=float(mw),
                use_abs_corr=bool(use_abs_corr),
                backend=str(backend),
                solver=str(solver),
            )
            w = alloc.get("weights")
            if not isinstance(w, pd.Series) or w.empty:
                continue

            # Score on the valid segment.
            rr = va2.to_numpy(dtype=float) @ w.reindex(cols).fillna(0.0).to_numpy(dtype=float)
            valid_series.append(pd.Series(rr, index=va2.index))
            n_used += 1

            # Track split-to-split allocation turnover.
            if w_prev is not None:
                w_prev_aligned = w_prev.reindex(cols).fillna(0.0)
                if float(w_prev_aligned.sum()) > 0.0:
                    w_prev_aligned = w_prev_aligned / float(w_prev_aligned.sum())
                    turnover_list.append(float(0.5 * np.sum(np.abs(w.to_numpy() - w_prev_aligned.to_numpy()))))

            w_prev = w

        if not valid_series:
            rows.append(
                {
                    "config_id": cfg.get("config_id"),
                    "lambda_corr": float(lam),
                    "max_weight": float(mw),
                    "turnover_lambda": float(tl),
                    "tune_metric": str(tune_metric),
                    "valid_metric": float("-inf"),
                    "n_splits_used": int(n_used),
                    "n_valid_days": 0,
                    "allocation_turnover_mean": float(np.mean(turnover_list)) if turnover_list else 0.0,
                    "error": "no_valid_series",
                }
            )
            continue

        vv = pd.concat(valid_series).sort_index()
        vv = vv[~vv.index.duplicated(keep="first")]
        arr = vv.to_numpy(dtype=float)
        mu = float(np.mean(arr)) if arr.size else 0.0

        if str(tune_metric).lower() in {"annualized_return", "ann"}:
            metric_v = _annualized_return(mu, trading_days=int(trading_days))
        else:
            metric_v = _information_ratio(arr, trading_days=int(trading_days))

        row = {
            "config_id": cfg.get("config_id"),
            "lambda_corr": float(lam),
            "max_weight": float(mw),
            "turnover_lambda": float(tl),
            "tune_metric": str(tune_metric),
            "valid_metric": float(metric_v),
            "n_splits_used": int(n_used),
            "n_valid_days": int(arr.size),
            "allocation_turnover_mean": float(np.mean(turnover_list)) if turnover_list else 0.0,
        }
        rows.append(row)

        if np.isfinite(metric_v) and float(metric_v) > best_v:
            best_v = float(metric_v)
            best_row = dict(row)

    # Sort for reporting.
    rows_sorted = sorted(rows, key=lambda r: float(r.get("valid_metric") or float("-inf")), reverse=True)

    chosen = best_row or (rows_sorted[0] if rows_sorted else None)
    out_params = dict(base_params)
    if chosen:
        out_params.update(
            {
                "lambda_corr": float(chosen.get("lambda_corr") or 0.0),
                "max_weight": float(chosen.get("max_weight") or 1.0),
                "turnover_lambda": float(chosen.get("turnover_lambda") or 0.0),
            }
        )

    return {
        "enabled": True,
        "tune_metric": str(tune_metric),
        "chosen": chosen or {},
        "best_params": out_params,
        "results": rows_sorted,
    }
