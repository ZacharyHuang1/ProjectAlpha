"""agent.research.walk_forward

P1: walk-forward evaluation.

This module runs sequential train/valid/test splits and reports out-of-sample
metrics on the concatenated test segments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig, backtest_long_short


@dataclass(frozen=True)
class WalkForwardConfig:
    train_days: int = 252
    valid_days: int = 63
    test_days: int = 63
    step_days: int = 63
    expanding_train: bool = True


def make_walk_forward_splits(
    dates: Sequence[pd.Timestamp],
    *,
    train_days: int,
    valid_days: int,
    test_days: int,
    step_days: int,
    expanding_train: bool,
) -> List[Dict[str, Any]]:
    """Create sequential train/valid/test splits over a date index."""

    dts = pd.to_datetime(pd.Index(list(dates))).sort_values()
    n = int(len(dts))
    if n < (train_days + valid_days + test_days + 2):
        return []

    splits: List[Dict[str, Any]] = []
    cursor = 0
    split_id = 0

    train_days = max(1, int(train_days))
    valid_days = max(0, int(valid_days))
    test_days = max(1, int(test_days))
    step_days = max(1, int(step_days))

    while True:
        if expanding_train:
            train_start = 0
            train_end = cursor + train_days - 1
        else:
            train_start = cursor
            train_end = cursor + train_days - 1

        valid_start = train_end + 1
        valid_end = valid_start + valid_days - 1

        test_start = valid_end + 1
        test_end = test_start + test_days - 1

        if test_end >= n:
            break

        splits.append(
            {
                "split_id": int(split_id),
                "train": (dts[train_start], dts[train_end]),
                "valid": (dts[valid_start], dts[valid_end]) if valid_days > 0 else None,
                "test": (dts[test_start], dts[test_end]),
            }
        )

        split_id += 1
        cursor += step_days
        if cursor + train_days + valid_days + test_days >= n:
            break

    return splits


def _summarize_bt(bt: Dict[str, Any]) -> Dict[str, Any]:
    if not bt or "error" in bt:
        return {"error": bt.get("error") if isinstance(bt, dict) else "Unknown error"}
    keys = [
        "information_ratio",
        "annualized_return",
        "max_drawdown",
        "turnover_mean",
        "cost_mean",
        "linear_cost_mean",
        "spread_cost_mean",
        "impact_cost_mean",
        "borrow_mean",
        "n_obs",
        "rebalance_days",
        "holding_days",
        "cost_bps",
        "borrow_bps",
    ]
    out = {k: bt.get(k) for k in keys if k in bt}

    # Keep only a small, stable construction summary for reporting.
    c = bt.get("construction")
    if isinstance(c, dict):
        opt = c.get("optimizer") if isinstance(c.get("optimizer"), dict) else {}
        out["construction"] = {
            "method": c.get("method"),
            "optimizer": {
                "backend_requested": (opt or {}).get("backend_requested"),
                "backend_used": (opt or {}).get("backend_used"),
                "last": (opt or {}).get("last"),
            },
        }
    return out


def _returns_from_bt(bt: Dict[str, Any]) -> pd.Series:
    daily = (bt or {}).get("daily") or []
    if not daily:
        return pd.Series(dtype=float)
    idx = pd.to_datetime([r["datetime"] for r in daily])
    vals = [float(r.get("net_return") or 0.0) for r in daily]
    return pd.Series(vals, index=idx).sort_index()


def _scenario_returns_from_daily(daily: List[Dict[str, Any]], scenario: str) -> pd.Series:
    """Recompute returns from a realized path under different cost assumptions.

    This is an execution-only ablation: the trading path (weights/trades)
    is fixed; only the cost deduction changes.
    """

    if not daily:
        return pd.Series(dtype=float)

    idx = pd.to_datetime([r["datetime"] for r in daily])
    gross = np.asarray([float(r.get("gross_return") or 0.0) for r in daily], dtype=float)
    linear = np.asarray([float(r.get("linear_cost") or 0.0) for r in daily], dtype=float)
    spread = np.asarray([float(r.get("spread_cost") or 0.0) for r in daily], dtype=float)
    impact = np.asarray([float(r.get("impact_cost") or 0.0) for r in daily], dtype=float)
    borrow = np.asarray([float(r.get("borrow") or 0.0) for r in daily], dtype=float)

    s = str(scenario or "").strip().lower()
    if s == "no_costs":
        net = gross
    elif s == "linear_only":
        net = gross - linear
    elif s == "linear_spread":
        net = gross - linear - spread
    elif s == "linear_spread_impact":
        net = gross - linear - spread - impact
    elif s == "full":
        net = gross - linear - spread - impact - borrow
    else:
        net = np.asarray([float(r.get("net_return") or 0.0) for r in daily], dtype=float)
    return pd.Series(net, index=idx).sort_index()


def _execution_only_ablation_summary(
    *,
    segments: Dict[str, List[pd.Series]],
    trading_days: int,
) -> List[Dict[str, Any]]:
    """Summarize execution-only scenarios from per-split series segments."""

    if not segments:
        return []

    def _concat(parts: List[pd.Series]) -> pd.Series:
        if not parts:
            return pd.Series(dtype=float)
        r = pd.concat(parts).sort_index()
        return r[~r.index.duplicated(keep="first")]

    gross = _concat(segments.get("no_costs") or [])
    gross_mu = float(gross.mean()) if not gross.empty else 0.0

    out: List[Dict[str, Any]] = []
    for name, parts in segments.items():
        r = _concat(parts)
        s = _summarize_returns(r, trading_days=trading_days)
        net_mu = float(r.mean()) if not r.empty else 0.0
        drag_bps = float((gross_mu - net_mu) * 10000.0)
        out.append(
            {
                "scenario": str(name),
                "information_ratio": float(s.get("information_ratio") or 0.0),
                "annualized_return": float(s.get("annualized_return") or 0.0),
                "max_drawdown": float(s.get("max_drawdown") or 0.0),
                "n_obs": int(s.get("n_obs") or 0),
                "mean_cost_drag_bps": float(drag_bps),
            }
        )

    out.sort(key=lambda x: str(x.get("scenario")))
    return out


def _summarize_returns(r: pd.Series, trading_days: int) -> Dict[str, float]:
    if r is None or r.empty:
        return {"information_ratio": 0.0, "annualized_return": 0.0, "max_drawdown": 0.0, "n_obs": 0.0}
    arr = r.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {"information_ratio": 0.0, "annualized_return": 0.0, "max_drawdown": float(_max_drawdown(r)), "n_obs": int(r.size)}
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


def _max_drawdown(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _yearly_breakdown(returns: pd.Series, trading_days: int) -> Dict[str, Dict[str, float]]:
    if returns is None or returns.empty:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for year, r in returns.groupby(returns.index.year):
        s = _summarize_returns(r, trading_days=trading_days)
        out[str(int(year))] = {
            "information_ratio": float(s.get("information_ratio") or 0.0),
            "annualized_return": float(s.get("annualized_return") or 0.0),
            "max_drawdown": float(s.get("max_drawdown") or 0.0),
            "n_obs": int(s.get("n_obs") or 0),
        }
    return out


def walk_forward_evaluate_factor(
    factor: pd.Series,
    ohlcv_or_close: pd.DataFrame | pd.Series,
    *,
    wf_config: WalkForwardConfig,
    bt_config: BacktestConfig,
    splits: Optional[List[Dict[str, Any]]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
    execution_only_ablation: bool = False,
    return_oos_daily_decomp: bool = False,
    return_valid_daily: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single factor via walk-forward splits."""

    if isinstance(ohlcv_or_close, pd.DataFrame):
        close = ohlcv_or_close["close"]
    else:
        close = ohlcv_or_close

    dates = pd.to_datetime(close.index.get_level_values("datetime").unique()).sort_values()
    if splits is None:
        splits = make_walk_forward_splits(
            dates,
            train_days=wf_config.train_days,
            valid_days=wf_config.valid_days,
            test_days=wf_config.test_days,
            step_days=wf_config.step_days,
            expanding_train=wf_config.expanding_train,
        )
    if not splits:
        return {"error": "Not enough data for walk-forward splits"}

    split_rows: List[Dict[str, Any]] = []
    test_returns_all: List[pd.Series] = []
    valid_returns_all: List[pd.Series] = []
    test_irs: List[float] = []
    test_nobs_list: List[int] = []
    train_irs: List[float] = []
    test_turnover_wsum: float = 0.0
    test_turnover_nsum: int = 0

    # Extra OOS attribution (weighted by test n_obs).
    attr_keys = ["cost_mean", "linear_cost_mean", "spread_cost_mean", "impact_cost_mean", "borrow_mean"]
    attr_wsum: Dict[str, float] = {k: 0.0 for k in attr_keys}

    exec_only_segments: Dict[str, List[pd.Series]] = {}
    if bool(execution_only_ablation):
        for name in ["no_costs", "linear_only", "linear_spread", "linear_spread_impact", "full"]:
            exec_only_segments[name] = []

    oos_daily_decomp: List[Dict[str, Any]] = []

    optimizer_backend_used: Dict[str, int] = {"qp": 0, "ridge": 0}
    optimizer_fallback_counts: Dict[str, int] = {}
    latest_optimizer_last: Optional[Dict[str, Any]] = None

    for s in splits:
        tr = s["train"]
        va = s.get("valid")
        te = s["test"]

        bt_pos = backtest_long_short(
            factor,
            ohlcv_or_close,
            config=bt_config,
            start=tr[0],
            end=tr[1],
            include_daily=False,
            sector_map=sector_map,
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )
        bt_neg = backtest_long_short(
            -factor,
            ohlcv_or_close,
            config=bt_config,
            start=tr[0],
            end=tr[1],
            include_daily=False,
            sector_map=sector_map,
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )

        ir_pos = float((bt_pos or {}).get("information_ratio") or 0.0)
        ir_neg = float((bt_neg or {}).get("information_ratio") or 0.0)
        sign = 1.0 if ir_pos >= ir_neg else -1.0

        train_bt = bt_pos if sign > 0 else bt_neg
        try:
            train_irs.append(float((train_bt or {}).get("information_ratio") or 0.0))
        except Exception:
            pass

        valid_bt = None
        if va is not None:
            valid_bt = backtest_long_short(
                sign * factor,
                ohlcv_or_close,
                config=bt_config,
                start=va[0],
                end=va[1],
                include_daily=bool(return_valid_daily),
                sector_map=sector_map,
                borrow_rates=borrow_rates,
                hard_to_borrow=hard_to_borrow,
            )
        if bool(return_valid_daily) and isinstance(valid_bt, dict) and isinstance(valid_bt.get("daily"), list):
            try:
                vrows = valid_bt.get("daily") or []
                vidx = pd.to_datetime([r.get("datetime") for r in vrows])
                vvals = [float(r.get("net_return") or 0.0) for r in vrows]
                vs = pd.Series(vvals, index=vidx).sort_index()
                valid_returns_all.append(vs)
            except Exception:
                pass
            # Drop the heavy daily payload; split-level metrics stay intact.
            valid_bt["daily"] = None

        test_bt_full = backtest_long_short(
            sign * factor,
            ohlcv_or_close,
            config=bt_config,
            start=te[0],
            end=te[1],
            include_daily=True,
            sector_map=sector_map,
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )

        if bool(return_oos_daily_decomp):
            daily_full = (test_bt_full or {}).get("daily") or []
            if isinstance(daily_full, list) and daily_full:
                oos_daily_decomp.extend([dict(r) for r in daily_full if isinstance(r, dict)])

        if bool(execution_only_ablation):
            daily = (test_bt_full or {}).get("daily") or []
            if isinstance(daily, list) and daily:
                for name in exec_only_segments.keys():
                    exec_only_segments[name].append(_scenario_returns_from_daily(daily, name))
        try:
            tmean = float((test_bt_full or {}).get("turnover_mean") or 0.0)
            nobs = int((test_bt_full or {}).get("n_obs") or 0)
            test_nobs_list.append(int(nobs))
            if nobs > 0:
                test_turnover_wsum += tmean * float(nobs)
                test_turnover_nsum += nobs

                # Weighted cost attribution.
                for k in attr_keys:
                    try:
                        attr_wsum[k] += float((test_bt_full or {}).get(k) or 0.0) * float(nobs)
                    except Exception:
                        pass
        except Exception:
            pass

        # Optimizer usage attribution (best-effort).
        cmeta = (test_bt_full or {}).get("construction") or {}
        if isinstance(cmeta, dict):
            opt = cmeta.get("optimizer") if isinstance(cmeta.get("optimizer"), dict) else {}
            bu = (opt or {}).get("backend_used") or {}
            if isinstance(bu, dict):
                for k in ["qp", "ridge"]:
                    try:
                        optimizer_backend_used[k] += int(bu.get(k) or 0)
                    except Exception:
                        pass
            last = (opt or {}).get("last")
            if isinstance(last, dict):
                latest_optimizer_last = last
                fb = str(last.get("fallback") or "").strip()
                if fb:
                    key = fb.split(":", 1)[0]
                    optimizer_fallback_counts[key] = int(optimizer_fallback_counts.get(key, 0)) + 1
        test_r = _returns_from_bt(test_bt_full)
        if not test_r.empty:
            test_returns_all.append(test_r)
            test_irs.append(float((test_bt_full or {}).get("information_ratio") or 0.0))

        split_rows.append(
            {
                "split_id": int(s["split_id"]),
                "sign": float(sign),
                "train": {"start": tr[0].isoformat(), "end": tr[1].isoformat(), "metrics": _summarize_bt(train_bt)},
                "valid": None
                if va is None
                else {"start": va[0].isoformat(), "end": va[1].isoformat(), "metrics": _summarize_bt(valid_bt or {})},
                "test": {"start": te[0].isoformat(), "end": te[1].isoformat(), "metrics": _summarize_bt(test_bt_full)},
            }
        )

    # Concatenate test segments (out-of-sample)
    if test_returns_all:
        oos_returns = pd.concat(test_returns_all).sort_index()
        # Remove duplicates (can happen at boundaries with include_daily alignment)
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="first")]
    else:
        oos_returns = pd.Series(dtype=float)

    oos_daily = (
        [{"datetime": d.isoformat(), "net_return": float(v)} for d, v in oos_returns.items()]
        if not oos_returns.empty
        else []
    )
    # Optionally concatenate validation segments for selection meta-tuning.
    if bool(return_valid_daily) and valid_returns_all:
        valid_returns = pd.concat(valid_returns_all).sort_index()
        valid_returns = valid_returns[~valid_returns.index.duplicated(keep="first")]
    else:
        valid_returns = pd.Series(dtype=float)

    valid_daily = (
        [{"datetime": d.isoformat(), "net_return": float(v)} for d, v in valid_returns.items()]
        if bool(return_valid_daily) and not valid_returns.empty
        else []
    )

    if bool(return_oos_daily_decomp) and oos_daily_decomp:
        # Keep a single, de-duplicated OOS path (datetime is the join key).
        tmp = pd.DataFrame(oos_daily_decomp)
        if "datetime" in tmp.columns:
            tmp = tmp.drop_duplicates(subset=["datetime"], keep="first")
            tmp = tmp.sort_values(by=["datetime"], ascending=True)
            oos_daily_decomp = tmp.to_dict(orient="records")

    oos = _summarize_returns(oos_returns, trading_days=bt_config.trading_days)
    oos_yearly = _yearly_breakdown(oos_returns, trading_days=bt_config.trading_days)

    finite_test_irs = [float(x) for x in test_irs if np.isfinite(float(x))]
    finite_train_irs = [float(x) for x in train_irs if np.isfinite(float(x))]

    stability = {
        "n_splits": int(len(splits)),
        "train_ir_mean": float(np.mean(finite_train_irs)) if finite_train_irs else 0.0,
        "test_ir_mean": float(np.mean(finite_test_irs)) if finite_test_irs else 0.0,
        "test_ir_std": float(np.std(finite_test_irs, ddof=1)) if len(finite_test_irs) > 1 else 0.0,
        "test_ir_positive_frac": float(np.mean([1.0 if x > 0 else 0.0 for x in finite_test_irs])) if finite_test_irs else 0.0,
        "test_ir_n": int(len(finite_test_irs)),
        "test_ir_min": float(np.min(finite_test_irs)) if finite_test_irs else 0.0,
        "test_ir_median": float(np.median(finite_test_irs)) if finite_test_irs else 0.0,
        "test_ir_max": float(np.max(finite_test_irs)) if finite_test_irs else 0.0,
        "test_n_obs_mean": float(np.mean(test_nobs_list)) if test_nobs_list else 0.0,
        "test_n_obs_min": int(np.min(test_nobs_list)) if test_nobs_list else 0,
        "test_n_obs_zero_splits": int(sum(1 for n in test_nobs_list if int(n) <= 0)),
        "generalization_gap": float((np.mean(finite_train_irs) - np.mean(finite_test_irs)))
        if finite_train_irs and finite_test_irs
        else 0.0,
    }

    oos_turnover_mean = float(test_turnover_wsum / float(test_turnover_nsum)) if test_turnover_nsum > 0 else 0.0

    oos_cost_attr = (
        {k: float(attr_wsum[k] / float(test_turnover_nsum)) for k in attr_keys} if test_turnover_nsum > 0 else {}
    )

    exec_only = None
    if bool(execution_only_ablation):
        exec_only = {
            "mode": "execution_only",
            "scenarios": _execution_only_ablation_summary(segments=exec_only_segments, trading_days=int(bt_config.trading_days)),
        }

    return {
        "mode": "p1",
        "information_ratio": float(oos.get("information_ratio") or 0.0),
        "annualized_return": float(oos.get("annualized_return") or 0.0),
        "max_drawdown": float(oos.get("max_drawdown") or 0.0),
        "turnover_mean": float(oos_turnover_mean),
        "oos_cost_attribution": oos_cost_attr,
        "walk_forward": {
            "config": {"walk_forward": asdict(wf_config), "backtest": asdict(bt_config)},
            "splits": split_rows,
            "oos": oos,
            "oos_yearly": oos_yearly,
            "oos_daily": oos_daily,
            "valid_daily": valid_daily if bool(return_valid_daily) else None,
            "oos_daily_decomp": oos_daily_decomp if bool(return_oos_daily_decomp) else None,
            "stability": stability,
            "optimizer_usage": {
                "backend_used": optimizer_backend_used,
                "fallback_reasons": optimizer_fallback_counts,
                "latest": latest_optimizer_last,
            },
        },
        "execution_only_ablation": exec_only,
    }