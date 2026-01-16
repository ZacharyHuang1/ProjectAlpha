"""agent.research.holdings_ensemble

P2.18 / P2.19: Holdings-level ensemble.

P2.17 blended *strategy return streams* (mean of net returns). This module
blends at the *portfolio holdings* level and then re-prices the combined
portfolio with the same cost / borrow model. This captures trade netting.

P2.19 adds walk-forward alpha allocation (learned weights across alphas).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig, backtest_from_weights, backtest_long_short
from agent.research.walk_forward import WalkForwardConfig, make_walk_forward_splits
from agent.research.alpha_allocation import fit_alpha_allocation
from agent.research.alpha_allocation_regime import build_daily_alpha_weights, fit_regime_allocations
from agent.research.alpha_allocation_tuning import (
    default_allocation_sweep_param_lists,
    meta_tune_alpha_allocation,
)
from agent.research.alpha_allocation_regime_tuning import (
    default_regime_sweep_param_lists,
    meta_tune_regime_aware_allocation,
)
from agent.research.regime_features import compute_market_feature_frame
from agent.research.regime_labels import make_regime_labels, regime_stats
from agent.research.constraint_selection import annotate_pareto, select_best_row


def positions_to_weight_matrix(
    *,
    positions: List[Dict[str, Any]],
    position_dates: Sequence[str],
    instruments: Sequence[str],
) -> pd.DataFrame:
    """Convert a sparse long-form positions payload into a dense weight matrix."""

    idx = pd.to_datetime(pd.Index(list(position_dates))).sort_values()
    cols = list(instruments)
    if not positions:
        return pd.DataFrame(0.0, index=idx, columns=cols)

    df = pd.DataFrame(positions)
    if df.empty or "datetime" not in df.columns or "instrument" not in df.columns or "weight" not in df.columns:
        return pd.DataFrame(0.0, index=idx, columns=cols)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["instrument"] = df["instrument"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    wide = df.pivot_table(index="datetime", columns="instrument", values="weight", aggfunc="last")
    wide = wide.reindex(index=idx).reindex(columns=cols).fillna(0.0)
    wide.index.name = "datetime"
    return wide


def _summarize_returns(r: pd.Series, trading_days: int) -> Dict[str, Any]:
    rr = pd.Series(r).dropna()
    arr = rr.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {"error": "Not enough observations"}

    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    ir = float(mu / sd * np.sqrt(float(trading_days))) if sd > 0.0 else 0.0
    ann = float((1.0 + mu) ** float(trading_days) - 1.0)

    equity = (1.0 + pd.Series(arr)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    mdd = float(dd.min()) if not dd.empty else 0.0

    return {
        "information_ratio": float(ir),
        "annualized_return": float(ann),
        "max_drawdown": float(mdd),
        "mean_daily_return": float(mu),
        "std_daily_return": float(sd),
        "n_obs": int(arr.size),
    }


def _extract_split_signs(alpha: Dict[str, Any]) -> Dict[int, float]:
    """Extract per-split sign decisions from an evaluated alpha payload."""

    out: Dict[int, float] = {}
    m = alpha.get("backtest_results") or {}
    wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
    splits = (wf.get("splits") or []) if isinstance(wf, dict) else []
    if not isinstance(splits, list):
        return out
    for s in splits:
        if not isinstance(s, dict):
            continue
        try:
            sid = int(s.get("split_id"))
            sign = float(s.get("sign") or 1.0)
            out[sid] = float(sign)
        except Exception:
            continue
    return out


def walk_forward_holdings_ensemble(
    *,
    selected_alphas: List[Dict[str, Any]],
    factor_cache: Dict[str, pd.Series],
    ohlcv: pd.DataFrame,
    wf_config: WalkForwardConfig,
    bt_config: BacktestConfig,
    splits: Optional[List[Dict[str, Any]]] = None,
    bt_config_by_alpha: Optional[Dict[str, BacktestConfig]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
    alpha_test_cache: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    apply_turnover_cap: bool = False,
    constraints: Optional[Dict[str, Any]] = None,
    prefer_pareto: bool = False,
) -> Dict[str, Any]:
    """Build a holdings-level equal-weight ensemble over walk-forward OOS tests."""

    if not selected_alphas:
        return {"enabled": False, "error": "Empty selection"}

    alpha_ids: List[str] = []
    for a in selected_alphas:
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if aid is not None:
            alpha_ids.append(str(aid))

    alpha_ids = list(dict.fromkeys(alpha_ids))
    if len(alpha_ids) < 1:
        return {"enabled": False, "error": "No valid alpha ids"}

    if splits is None:
        close = ohlcv["close"]
        dates = pd.to_datetime(close.index.get_level_values("datetime").unique()).sort_values()
        splits = make_walk_forward_splits(
            dates,
            train_days=wf_config.train_days,
            valid_days=wf_config.valid_days,
            test_days=wf_config.test_days,
            step_days=wf_config.step_days,
            expanding_train=wf_config.expanding_train,
        )
    if not splits:
        return {"enabled": False, "error": "Not enough data for walk-forward splits"}

    # Infer the instrument universe from the OHLCV input.
    instruments = list(pd.Index(ohlcv.index.get_level_values("instrument")).unique())

    sign_map: Dict[str, Dict[int, float]] = {aid: {} for aid in alpha_ids}
    for a in selected_alphas:
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if aid is None:
            continue
        sign_map[str(aid)] = _extract_split_signs(a)

    oos_daily_all: List[Dict[str, Any]] = []
    oos_positions_all: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    for sp in splits:
        split_id = int(sp.get("split_id") or 0)
        te = sp.get("test")
        if not te:
            continue
        t_start, t_end = te

        w_list: List[pd.DataFrame] = []
        for aid in alpha_ids:
            fac = factor_cache.get(aid)
            if fac is None or fac.empty:
                continue
            sign = float(sign_map.get(aid, {}).get(split_id, 1.0))
            cfg = bt_config
            if bt_config_by_alpha is not None and aid in bt_config_by_alpha:
                cfg = bt_config_by_alpha[aid]

            cache_key = (aid, split_id)
            cached = None if alpha_test_cache is None else alpha_test_cache.get(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("weights"), pd.DataFrame):
                w_mat = cached["weights"]
            else:
                bt = backtest_long_short(
                    sign * fac,
                    ohlcv,
                    config=cfg,
                    start=t_start,
                    end=t_end,
                    include_daily=False,
                    include_positions=True,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                )
                if bt.get("error"):
                    continue

                pos_rows = list(bt.get("positions") or [])
                pos_dates = list(bt.get("position_dates") or [])
                w_mat = positions_to_weight_matrix(
                    positions=pos_rows,
                    position_dates=pos_dates,
                    instruments=instruments,
                )
                if alpha_test_cache is not None:
                    alpha_test_cache[cache_key] = {"weights": w_mat, "positions": pos_rows, "position_dates": pos_dates}
            w_list.append(w_mat)

        if not w_list:
            continue

        # Equal weight across alpha portfolios (cash when a strategy is flat).
        w_sum = None
        for w in w_list:
            if w_sum is None:
                w_sum = w.copy()
            else:
                w_sum = w_sum.add(w, fill_value=0.0)
        w_ens = (w_sum / float(len(w_list))) if w_sum is not None else pd.DataFrame()
        if w_ens is None or w_ens.empty:
            continue

        bt_ens = backtest_from_weights(
            w_ens,
            ohlcv,
            config=bt_config,
            start=t_start,
            end=t_end,
            include_daily=True,
            include_positions=True,
            apply_turnover_cap=bool(apply_turnover_cap),
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )
        if bt_ens.get("error"):
            continue

        daily = list(bt_ens.get("daily") or [])
        if daily:
            oos_daily_all.extend(daily)
        pos = list(bt_ens.get("positions") or [])
        if pos:
            oos_positions_all.extend(pos)

        split_rows.append(
            {
                "split_id": split_id,
                "test": {"start": pd.to_datetime(t_start).isoformat(), "end": pd.to_datetime(t_end).isoformat()},
                "n_alphas": int(len(w_list)),
                "metrics": {
                    "information_ratio": bt_ens.get("information_ratio"),
                    "annualized_return": bt_ens.get("annualized_return"),
                    "max_drawdown": bt_ens.get("max_drawdown"),
                    "turnover_mean": bt_ens.get("turnover_mean"),
                    "cost_mean": bt_ens.get("cost_mean"),
                    "borrow_mean": bt_ens.get("borrow_mean"),
                    "n_obs": bt_ens.get("n_obs"),
                },
            }
        )

    if not oos_daily_all:
        return {"enabled": False, "error": "No OOS daily rows produced"}

    # De-duplicate by datetime and sort.
    df_daily = pd.DataFrame(oos_daily_all)
    if "datetime" in df_daily.columns:
        df_daily["datetime"] = pd.to_datetime(df_daily["datetime"])
        df_daily = df_daily.drop_duplicates(subset=["datetime"], keep="first")
        df_daily = df_daily.sort_values(by=["datetime"], ascending=True)

    net = pd.Series(df_daily["net_return"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["datetime"]))

    perf = _summarize_returns(net, trading_days=int(bt_config.trading_days))
    perf["n_alphas"] = int(len(alpha_ids))
    perf["apply_turnover_cap"] = bool(apply_turnover_cap)

    # Extra cost attribution on the realized path.
    for k in ["turnover", "cost", "linear_cost", "spread_cost", "impact_cost", "borrow"]:
        if k in df_daily.columns:
            try:
                perf[f"{k}_mean"] = float(pd.to_numeric(df_daily[k], errors="coerce").fillna(0.0).mean())
            except Exception:
                pass

    if oos_positions_all:
        try:
            df_pos = pd.DataFrame(oos_positions_all)
            if {"datetime", "instrument", "weight"}.issubset(set(df_pos.columns)):
                df_pos["datetime"] = pd.to_datetime(df_pos["datetime"])
                df_pos["instrument"] = df_pos["instrument"].astype(str)
                df_pos["weight"] = pd.to_numeric(df_pos["weight"], errors="coerce").fillna(0.0)
                df_pos = df_pos.drop_duplicates(subset=["datetime", "instrument"], keep="first")
                df_pos = df_pos.sort_values(by=["datetime", "instrument"], ascending=True)
                oos_positions_all = df_pos.assign(datetime=df_pos["datetime"].astype(str)).to_dict(orient="records")
        except Exception:
            pass

    out = {
        "enabled": True,
        "method": "equal_weight_holdings",
        "selected_alpha_ids": alpha_ids,
        "metrics": perf,
        "daily": df_daily.assign(datetime=df_daily["datetime"].astype(str)).to_dict(orient="records"),
        "positions": oos_positions_all,
        "walk_forward": {
            "config": {"walk_forward": asdict(wf_config), "backtest": asdict(bt_config)},
            "splits": split_rows,
        },
    }
    return out
def _daily_series_from_bt(bt: Dict[str, Any], key: str = "net_return") -> pd.Series:
    rows = bt.get("daily") or []
    if not isinstance(rows, list) or not rows:
        return pd.Series(dtype=float)
    try:
        idx = pd.to_datetime([r.get("datetime") for r in rows])
        vals = [float(r.get(key) or 0.0) for r in rows]
        s = pd.Series(vals, index=idx).sort_index()
        s = s[~s.index.duplicated(keep="first")]
        return s
    except Exception:
        return pd.Series(dtype=float)


def _build_split_return_matrices(
    *,
    alpha_ids: List[str],
    factor_cache: Dict[str, pd.Series],
    sign_map: Dict[str, Dict[int, float]],
    ohlcv: pd.DataFrame,
    splits: List[Dict[str, Any]],
    bt_config: BacktestConfig,
    bt_config_by_alpha: Optional[Dict[str, BacktestConfig]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[int, pd.DataFrame], Dict[int, pd.DataFrame]]:
    """Compute per-split (train, valid) alpha return matrices.

    This is the shared data needed for:
    - allocation meta-tuning (train->valid)
    - per-split allocation fitting (train or train+valid)
    """

    split_info: List[Dict[str, Any]] = []
    train_by_split: Dict[int, pd.DataFrame] = {}
    valid_by_split: Dict[int, pd.DataFrame] = {}

    for sp in splits:
        split_id = int(sp.get("split_id") or 0)
        tr = sp.get("train")
        va = sp.get("valid")
        te = sp.get("test")
        if not tr or not te:
            continue

        split_info.append({"split_id": split_id, "train": tr, "valid": va, "test": te})

        tr_cols: Dict[str, pd.Series] = {}
        va_cols: Dict[str, pd.Series] = {}

        for aid in alpha_ids:
            fac = factor_cache.get(aid)
            if fac is None:
                continue
            sign = float(sign_map.get(aid, {}).get(split_id, 1.0))
            cfg = bt_config_by_alpha.get(aid, bt_config) if bt_config_by_alpha else bt_config

            bt_tr = backtest_long_short(
                sign * fac,
                ohlcv,
                config=cfg,
                start=tr[0],
                end=tr[1],
                include_daily=True,
                include_positions=False,
                sector_map=sector_map,
                borrow_rates=borrow_rates,
                hard_to_borrow=hard_to_borrow,
            )
            if not bt_tr.get("error"):
                s_tr = _daily_series_from_bt(bt_tr, key="net_return")
                if not s_tr.empty:
                    tr_cols[aid] = s_tr

            if va:
                bt_va = backtest_long_short(
                    sign * fac,
                    ohlcv,
                    config=cfg,
                    start=va[0],
                    end=va[1],
                    include_daily=True,
                    include_positions=False,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                )
                if not bt_va.get("error"):
                    s_va = _daily_series_from_bt(bt_va, key="net_return")
                    if not s_va.empty:
                        va_cols[aid] = s_va

        if len(tr_cols) >= 2:
            df_tr = pd.concat(tr_cols, axis=1).sort_index()
            df_tr = df_tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            train_by_split[int(split_id)] = df_tr
        if va and len(va_cols) >= 2:
            df_va = pd.concat(va_cols, axis=1).sort_index()
            df_va = df_va.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            valid_by_split[int(split_id)] = df_va

    split_info = sorted(split_info, key=lambda d: int(d.get("split_id") or 0))
    return split_info, train_by_split, valid_by_split


def walk_forward_holdings_ensemble_allocated(
    *,
    selected_alphas: List[Dict[str, Any]],
    factor_cache: Dict[str, pd.Series],
    ohlcv: pd.DataFrame,
    wf_config: WalkForwardConfig,
    bt_config: BacktestConfig,
    splits: Optional[List[Dict[str, Any]]] = None,
    bt_config_by_alpha: Optional[Dict[str, BacktestConfig]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
    apply_turnover_cap: bool = False,
    alpha_test_cache: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    # Allocation config.
    allocation_fit: str = "train_valid",  # "train" | "train_valid"
    allocation_backend: str = "auto",  # "auto" | "qp" | "pgd"
    allocation_score_metric: str = "information_ratio",
    allocation_lambda: float = 0.5,
    allocation_l2: float = 1e-6,
    allocation_turnover_lambda: float = 0.0,
    allocation_max_weight: float = 0.8,
    allocation_use_abs_corr: bool = True,
    allocation_min_days: int = 30,
    allocation_solver: str = "",
    # Meta-tuning (single config chosen on aggregate valid performance).
    allocation_tune: bool = False,
    allocation_tune_metric: str = "information_ratio",
    allocation_tune_max_combos: int = 24,
    allocation_tune_lambda_grid: Optional[Sequence[float]] = None,
    allocation_tune_max_weight_grid: Optional[Sequence[float]] = None,
    allocation_tune_turnover_lambda_grid: Optional[Sequence[float]] = None,
    allocation_tune_save_top: int = 10,
    # Optional cache to avoid recomputing per-split train/valid return matrices.
    split_returns_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Holdings-level ensemble with walk-forward-learned alpha weights."""

    if not selected_alphas:
        return {"enabled": False, "error": "Empty selection"}

    alpha_ids: List[str] = []
    for a in selected_alphas:
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if aid is not None:
            alpha_ids.append(str(aid))
    alpha_ids = list(dict.fromkeys(alpha_ids))
    if len(alpha_ids) < 2:
        return {"enabled": False, "error": "Need at least 2 alphas for allocation"}

    if splits is None:
        splits = make_walk_forward_splits(
            ohlcv.index.get_level_values("datetime").unique(),
            train_days=wf_config.train_days,
            valid_days=wf_config.valid_days,
            test_days=wf_config.test_days,
            step_days=wf_config.step_days,
            expanding_train=wf_config.expanding_train,
        )
    if not splits:
        return {"enabled": False, "error": "Not enough data for walk-forward splits"}

    instruments = list(pd.Index(ohlcv.index.get_level_values("instrument")).unique())

    sign_map: Dict[str, Dict[int, float]] = {aid: {} for aid in alpha_ids}
    for a in selected_alphas:
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if aid is None:
            continue
        sign_map[str(aid)] = _extract_split_signs(a)

    oos_daily_all: List[Dict[str, Any]] = []
    oos_positions_all: List[Dict[str, Any]] = []
    allocations_all: List[Dict[str, Any]] = []
    allocation_diagnostics_all: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    fit_mode = str(allocation_fit).lower().strip()
    td = int(getattr(bt_config, "trading_days", 252) or 252)

    # Precompute per-split return matrices for meta-tuning (train -> valid).
    cached = None
    if isinstance(split_returns_cache, dict):
        cached = split_returns_cache.get("split_returns")
        if isinstance(cached, dict) and cached.get("alpha_ids") != list(alpha_ids):
            cached = None

    if isinstance(cached, dict):
        split_info = list(cached.get("split_info") or [])
        train_by_split = dict(cached.get("train_by_split") or {})
        valid_by_split = dict(cached.get("valid_by_split") or {})
    else:
        split_info, train_by_split, valid_by_split = _build_split_return_matrices(
            alpha_ids=alpha_ids,
            factor_cache=factor_cache,
            sign_map=sign_map,
            ohlcv=ohlcv,
            splits=list(splits),
            bt_config=bt_config,
            bt_config_by_alpha=bt_config_by_alpha,
            sector_map=sector_map,
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )
        if isinstance(split_returns_cache, dict):
            split_returns_cache["split_returns"] = {
                "alpha_ids": list(alpha_ids),
                "split_info": split_info,
                "train_by_split": train_by_split,
                "valid_by_split": valid_by_split,
            }

    # Optional meta-tuning: choose a single allocation config on aggregate valid performance.
    tuned = False
    tuning_payload: Dict[str, Any] = {}
    if bool(allocation_tune) and valid_by_split:
        try:
            base_params = {
                "lambda_corr": float(allocation_lambda),
                "max_weight": float(allocation_max_weight),
                "turnover_lambda": float(allocation_turnover_lambda),
                "l2": float(allocation_l2),
            }

            param_lists = default_allocation_sweep_param_lists(
                base_lambda_corr=float(allocation_lambda),
                base_max_weight=float(allocation_max_weight),
                base_turnover_lambda=float(allocation_turnover_lambda),
            )
            if allocation_tune_lambda_grid:
                param_lists["lambda_corr"] = [float(x) for x in allocation_tune_lambda_grid]
            if allocation_tune_max_weight_grid:
                param_lists["max_weight"] = [float(x) for x in allocation_tune_max_weight_grid]
            if allocation_tune_turnover_lambda_grid:
                param_lists["turnover_lambda"] = [float(x) for x in allocation_tune_turnover_lambda_grid]

            tuning_payload = meta_tune_alpha_allocation(
                train_by_split=train_by_split,
                valid_by_split=valid_by_split,
                trading_days=int(td),
                score_metric=str(allocation_score_metric),
                use_abs_corr=bool(allocation_use_abs_corr),
                backend=str(allocation_backend),
                solver=str(allocation_solver),
                base_params=base_params,
                param_lists=param_lists,
                max_combos=int(allocation_tune_max_combos),
                min_days=int(allocation_min_days),
                tune_metric=str(allocation_tune_metric),
            )

            best = dict(tuning_payload.get("best_params") or {})
            allocation_lambda = float(best.get("lambda_corr") or allocation_lambda)
            allocation_max_weight = float(best.get("max_weight") or allocation_max_weight)
            allocation_turnover_lambda = float(best.get("turnover_lambda") or allocation_turnover_lambda)
            tuned = True
        except Exception as e:
            tuning_payload = {"enabled": False, "error": str(e)}

    # Now produce test OOS results using the (possibly tuned) allocation config.
    w_prev_global: Optional[pd.Series] = None

    for sp in split_info:
        split_id = int(sp.get("split_id") or 0)
        tr = sp.get("train")
        va = sp.get("valid")
        te = sp.get("test")
        if not tr or not te:
            continue

        fit_start = tr[0]
        fit_end = tr[1] if (fit_mode == "train" or not va) else va[1]

        df_tr = train_by_split.get(int(split_id))
        df_va = valid_by_split.get(int(split_id)) if va else None
        if df_tr is None or df_tr.empty or df_tr.shape[1] < 2:
            continue

        # Build the fit return matrix based on allocation_fit.
        if fit_mode == "train" or df_va is None or df_va.empty:
            fit_df = df_tr
        else:
            fit_df = pd.concat([df_tr, df_va], axis=0).sort_index()
            fit_df = fit_df[~fit_df.index.duplicated(keep="first")]

        fit_df = fit_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if int(len(fit_df.index)) < int(allocation_min_days):
            w_alpha = pd.Series(1.0 / float(fit_df.shape[1]), index=fit_df.columns, dtype=float)
            alloc_diag = {"method": "fallback_equal", "reason": "not_enough_fit_days", "n_fit_days": int(len(fit_df.index))}
        else:
            alloc = fit_alpha_allocation(
                fit_df,
                trading_days=int(td),
                score_metric=str(allocation_score_metric),
                lambda_corr=float(allocation_lambda),
                l2=float(allocation_l2),
                turnover_lambda=float(allocation_turnover_lambda),
                prev_weights=w_prev_global,
                max_weight=float(allocation_max_weight),
                use_abs_corr=bool(allocation_use_abs_corr),
                backend=str(allocation_backend),
                solver=str(allocation_solver),
            )
            w_alpha = alloc.get("weights")
            if not isinstance(w_alpha, pd.Series) or w_alpha.empty:
                w_alpha = pd.Series(1.0 / float(fit_df.shape[1]), index=fit_df.columns, dtype=float)
                alloc_diag = {"method": "fallback_equal", "reason": "allocation_failed"}
            else:
                alloc_diag = dict(alloc.get("diagnostics") or {})
                alloc_diag["n_fit_days"] = int(len(fit_df.index))

        w_prev_global = w_alpha.reindex(alpha_ids).fillna(0.0)
        if float(w_prev_global.sum()) > 0.0:
            w_prev_global = w_prev_global / float(w_prev_global.sum())

        for aid in list(w_alpha.index):
            allocations_all.append(
                {
                    "split_id": int(split_id),
                    "alpha_id": str(aid),
                    "weight": float(w_alpha.get(aid) or 0.0),
                    "fit_start": pd.to_datetime(fit_start).isoformat(),
                    "fit_end": pd.to_datetime(fit_end).isoformat(),
                }
            )
        allocation_diagnostics_all.append({"split_id": int(split_id), **alloc_diag})

        # Build the test ensemble by blending holdings with the learned alpha weights.
        t_start, t_end = te
        w_list: List[Tuple[float, pd.DataFrame]] = []
        for aid, wa in w_alpha.items():
            wa = float(wa)
            if wa <= 0.0:
                continue

            fac = factor_cache.get(str(aid))
            if fac is None:
                continue

            sign = float(sign_map.get(str(aid), {}).get(split_id, 1.0))
            cfg = bt_config_by_alpha.get(str(aid), bt_config) if bt_config_by_alpha else bt_config

            cache_key = (str(aid), int(split_id))
            cached = None if alpha_test_cache is None else alpha_test_cache.get(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("weights"), pd.DataFrame):
                w_mat = cached["weights"]
            else:
                bt = backtest_long_short(
                    sign * fac,
                    ohlcv,
                    config=cfg,
                    start=t_start,
                    end=t_end,
                    include_daily=False,
                    include_positions=True,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                )
                if bt.get("error"):
                    continue
                pos_rows = list(bt.get("positions") or [])
                pos_dates = list(bt.get("position_dates") or [])
                w_mat = positions_to_weight_matrix(positions=pos_rows, position_dates=pos_dates, instruments=instruments)
                if alpha_test_cache is not None:
                    alpha_test_cache[cache_key] = {"weights": w_mat, "positions": pos_rows, "position_dates": pos_dates}

            if w_mat is None or w_mat.empty:
                continue
            w_list.append((wa, w_mat))

        if not w_list:
            continue

        all_idx = pd.Index(sorted({d for _, w in w_list for d in w.index}))
        w_sum = None
        for wa, w in w_list:
            ww = w.reindex(index=all_idx, columns=instruments).fillna(0.0) * float(wa)
            if w_sum is None:
                w_sum = ww
            else:
                w_sum = w_sum.add(ww, fill_value=0.0)

        w_ens = w_sum if w_sum is not None else pd.DataFrame()
        if w_ens.empty:
            continue

        bt_ens = backtest_from_weights(
            w_ens,
            ohlcv,
            config=bt_config,
            start=t_start,
            end=t_end,
            include_daily=True,
            include_positions=True,
            apply_turnover_cap=bool(apply_turnover_cap),
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )
        if bt_ens.get("error"):
            continue

        daily = list(bt_ens.get("daily") or [])
        if daily:
            oos_daily_all.extend(daily)
        pos = list(bt_ens.get("positions") or [])
        if pos:
            oos_positions_all.extend(pos)

        split_rows.append(
            {
                "split_id": int(split_id),
                "fit": {"start": pd.to_datetime(fit_start).isoformat(), "end": pd.to_datetime(fit_end).isoformat()},
                "test": {"start": pd.to_datetime(t_start).isoformat(), "end": pd.to_datetime(t_end).isoformat()},
                "n_alphas": int(len(w_list)),
                "allocation": {str(aid): float(w) for aid, w in w_alpha.items()},
                "metrics": {
                    "information_ratio": bt_ens.get("information_ratio"),
                    "annualized_return": bt_ens.get("annualized_return"),
                    "max_drawdown": bt_ens.get("max_drawdown"),
                },
            }
        )

    if not oos_daily_all:
        return {"enabled": False, "error": "No OOS daily results for allocated ensemble"}

    df_daily = pd.DataFrame(oos_daily_all)
    if "datetime" in df_daily.columns:
        df_daily["datetime"] = pd.to_datetime(df_daily["datetime"])
        df_daily = df_daily.drop_duplicates(subset=["datetime"], keep="first")
        df_daily = df_daily.sort_values(by=["datetime"], ascending=True)

    net = pd.Series(df_daily["net_return"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["datetime"]))
    perf = _summarize_returns(net, trading_days=int(td))
    perf["n_alphas"] = int(len(alpha_ids))
    perf["apply_turnover_cap"] = bool(apply_turnover_cap)

    for k in ["turnover", "cost", "linear_cost", "spread_cost", "impact_cost", "borrow"]:
        if k in df_daily.columns:
            try:
                perf[f"{k}_mean"] = float(pd.to_numeric(df_daily[k], errors="coerce").fillna(0.0).mean())
            except Exception:
                pass

    out = {
        "enabled": True,
        "method": "allocated_holdings",
        "selected_alpha_ids": alpha_ids,
        "allocation": {
            "fit": str(allocation_fit),
            "backend": str(allocation_backend),
            "score_metric": str(allocation_score_metric),
            "lambda_corr": float(allocation_lambda),
            "l2": float(allocation_l2),
            "turnover_lambda": float(allocation_turnover_lambda),
            "max_weight": float(allocation_max_weight),
            "use_abs_corr": bool(allocation_use_abs_corr),
            "min_days": int(allocation_min_days),
            "tuned": bool(tuned),
        },
        "metrics": perf,
        "daily": df_daily.assign(datetime=df_daily["datetime"].astype(str)).to_dict(orient="records"),
        "positions": oos_positions_all,
        "allocations": allocations_all,
        "allocation_diagnostics": allocation_diagnostics_all,
        "allocation_tuning": (
            {
                "enabled": bool(tuned),
                "tune_metric": str(allocation_tune_metric),
                "chosen": dict(tuning_payload.get("chosen") or {}),
                # Keep the JSON payload small; the full table is also exported as a CSV artifact.
                "results": (
                    list(tuning_payload.get("results") or [])
                    if int(allocation_tune_save_top) == 0
                    else list((tuning_payload.get("results") or [])[: int(max(0, allocation_tune_save_top))])
                ),
                **(
                    {"error": str(tuning_payload.get("error"))}
                    if isinstance(tuning_payload, dict) and tuning_payload.get("error")
                    else {}
                ),
            }
            if tuning_payload
            else {"enabled": False}
        ),
        "allocation_tuning_results": list(tuning_payload.get("results") or []) if tuning_payload else [],
        "walk_forward": {
            "config": {"walk_forward": asdict(wf_config), "backtest": asdict(bt_config)},
            "splits": split_rows,
        },
    }
    return out




def _revalidate_regime_configs_holdings(
    *,
    candidate_rows: List[Dict[str, Any]],
    top_n: int,
    ohlcv: pd.DataFrame,
    instruments: List[str],
    alpha_ids: List[str],
    factor_cache: Dict[str, pd.Series],
    sign_map: Dict[str, Dict[int, float]],
    split_info: List[Dict[str, Any]],
    train_by_split: Dict[int, pd.DataFrame],
    valid_by_split: Dict[int, pd.DataFrame],
    bt_config: BacktestConfig,
    bt_config_by_alpha: Optional[Dict[str, BacktestConfig]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
    # Allocation params (kept fixed during revalidation).
    allocation_score_metric: str = "information_ratio",
    allocation_lambda: float = 0.5,
    allocation_l2: float = 1e-6,
    allocation_turnover_lambda: float = 0.0,
    allocation_max_weight: float = 0.8,
    allocation_use_abs_corr: bool = True,
    allocation_backend: str = "auto",
    allocation_solver: str = "",
    allocation_min_days: int = 30,
    regime_min_days: int = 30,
    # Revalidation objective.
    tune_metric: str = "information_ratio",
    turnover_penalty: float = 0.0,
    turnover_cost_bps: Optional[float] = None,
    apply_turnover_cap: bool = False,
    # P2.26/P2.27: constraints + Pareto selection.
    constraints: Optional[Dict[str, Any]] = None,
    prefer_pareto: bool = False,
    pareto_metrics: Optional[Sequence[str]] = None,
    selection_method: str = "best_objective",
    utility_weights: Optional[Dict[str, Any]] = None,
    include_stability_objectives: bool = True,
) -> Dict[str, Any]:
    """Re-rank regime configs using holdings-level validation.

    P2.24 used a proxy objective (alpha-return matrix) to pick top configs.
    P2.25 refines the final choice by evaluating those configs with a true
    holdings-level backtest on the validation segments.

    The optional alpha-weight turnover penalty is applied as an additional bps
    cost on the alpha-weight turnover series:

        r_adj[t] = r[t] - alpha_turnover[t] * turnover_cost_bps / 10000

    No leakage: for each split we fit on train, evaluate on valid.
    """

    def _sf(x: Any, default: float = float("nan")) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    cost_bps = float(turnover_penalty if turnover_cost_bps is None else turnover_cost_bps)
    if not np.isfinite(cost_bps):
        cost_bps = 0.0
    cost_bps = float(max(0.0, cost_bps))

    def _alpha_turnover_series(w: pd.DataFrame) -> pd.Series:
        if w is None or w.empty or w.shape[0] < 2:
            return pd.Series(0.0, index=pd.Index([], dtype="datetime64[ns]"))
        try:
            d = w.diff().abs().sum(axis=1).fillna(0.0)
            return 0.5 * d
        except Exception:
            return pd.Series(0.0, index=pd.to_datetime(pd.Index(w.index)) if w is not None else None)

    def _pareto_objectives(extra: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
        """Build a list of Pareto objectives for holdings validation rows."""

        base: List[Tuple[str, str]] = [
            ("holdings_objective", "max"),
            ("alpha_weight_turnover_mean", "min"),
        ]

        extra_map = {
            "turnover_cost_drag_bps_mean": ("turnover_cost_drag_bps_mean", "min"),
            "regime_switch_rate_mean": ("regime_switch_rate_mean", "min"),
            "fallback_frac_mean": ("fallback_frac_mean", "min"),
            "ensemble_turnover_mean": ("ensemble_turnover_mean", "min"),
            "ensemble_cost_mean": ("ensemble_cost_mean", "min"),
            "ensemble_borrow_mean": ("ensemble_borrow_mean", "min"),
            # Stability (optional).
            "holdings_objective_split_std": ("holdings_objective_split_std", "min"),
            "holdings_objective_split_min": ("holdings_objective_split_min", "max"),
            "holdings_objective_split_negative_frac": ("holdings_objective_split_negative_frac", "min"),
        }

        out = list(base)
        for m in list(extra or []):
            spec = extra_map.get(str(m).strip())
            if not spec:
                continue
            k, d = spec
            if any(x[0] == k for x in out):
                continue
            out.append((k, str(d)))
        return out

    top_n = int(max(0, top_n))
    if top_n <= 0:
        return {"enabled": False, "error": "top_n_disabled"}

    usable = []
    for r in candidate_rows or []:
        if not isinstance(r, dict):
            continue
        obj = _sf(r.get("objective"), default=float("-inf"))
        win = int(float(r.get("window") or 0) or 0)
        if not np.isfinite(obj) or win <= 0:
            continue
        usable.append(dict(r))

    usable = sorted(usable, key=lambda d: _sf(d.get("objective"), default=float("-inf")), reverse=True)
    if not usable:
        return {"enabled": False, "error": "no_usable_proxy_rows"}

    keep: List[Dict[str, Any]] = []
    seen = set()
    for r in usable:
        key = (
            str(r.get("mode") or "").lower(),
            int(float(r.get("window") or 0) or 0),
            int(float(r.get("buckets") or 0) or 0),
            float(_sf(r.get("smoothing"), default=0.0)),
        )
        if key in seen:
            continue
        seen.add(key)
        keep.append(r)
        if len(keep) >= top_n:
            break

    if not keep:
        return {"enabled": False, "error": "no_configs_after_dedup"}

    feat_cache: Dict[int, pd.DataFrame] = {}
    for r in keep:
        w = int(float(r.get("window") or 0) or 0)
        if w <= 0 or w in feat_cache:
            continue
        try:
            feat_cache[w] = compute_market_feature_frame(ohlcv, window=int(w), min_obs=max(5, int(w)))
        except Exception:
            feat_cache[w] = pd.DataFrame()

    # Cache alpha holdings on validation segments (independent of regime config).
    holdings_cache: Dict[Tuple[str, int], pd.DataFrame] = {}

    rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_obj = float("-inf")

    td = int(getattr(bt_config, "trading_days", 252) or 252)
    tm = str(tune_metric).lower().strip()

    for r in keep:
        mode = str(r.get("mode") or "vol")
        window = int(float(r.get("window") or 0) or 0)
        buckets = int(float(r.get("buckets") or 3) or 3)
        smoothing = float(_sf(r.get("smoothing"), default=0.0))
        config_id = str(r.get("config_id") or "")

        feats = feat_cache.get(int(window))
        if feats is None or feats.empty:
            rows.append(
                {
                    "config_id": config_id,
                    "mode": mode,
                    "window": int(window),
                    "buckets": int(buckets),
                    "smoothing": float(smoothing),
                    "tune_metric": str(tune_metric),
                    "turnover_cost_bps": float(cost_bps),
                    "holdings_valid_metric": float("-inf"),
                    "holdings_valid_metric_after_turnover_cost": float("-inf"),
                    "holdings_objective": float("-inf"),
                    "n_splits_used": 0,
                    "n_valid_days": 0,
                    "alpha_weight_turnover_mean": 0.0,
                    "turnover_cost_drag_bps_mean": 0.0,
                    "ensemble_turnover_mean": 0.0,
                    "ensemble_cost_mean": 0.0,
                    "ensemble_borrow_mean": 0.0,
                    "holdings_objective_split_std": 0.0,
                    "holdings_objective_split_min": 0.0,
                    "holdings_objective_split_negative_frac": 0.0,
                    "error": "missing_features",
                }
            )
            continue

        w_prev_global: Optional[pd.Series] = None
        w_prev_by_regime: Dict[str, pd.Series] = {}

        valid_series_raw: List[pd.Series] = []
        valid_series_adj: List[pd.Series] = []
        split_obj_adj: List[float] = []

        alpha_to_wsum = 0.0
        alpha_to_nsum = 0.0
        ens_turn_wsum = 0.0
        ens_cost_wsum = 0.0
        ens_borrow_wsum = 0.0
        ens_nsum = 0.0
        n_used = 0
        switch_rates: List[float] = []
        fallback_fracs: List[float] = []

        for sp in split_info:
            split_id = int(sp.get("split_id") or 0)
            tr = sp.get("train")
            va = sp.get("valid")
            if not tr or not va:
                continue

            df_tr = train_by_split.get(int(split_id))
            df_va = valid_by_split.get(int(split_id))
            if df_tr is None or df_va is None or df_tr.empty or df_va.empty:
                continue

            cols = [c for c in alpha_ids if c in df_tr.columns and c in df_va.columns]
            if len(cols) < 2:
                continue

            tr2 = df_tr.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            va2 = df_va.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)

            train_idx = pd.to_datetime(pd.Index(tr2.index)).sort_values()
            valid_idx = pd.to_datetime(pd.Index(va2.index)).sort_values()
            if int(len(train_idx)) < int(max(5, allocation_min_days)):
                continue

            labels_train, _ = make_regime_labels(
                feats,
                target_index=train_idx,
                fit_index=train_idx,
                mode=str(mode),
                buckets=int(buckets),
            )
            labels_valid, _ = make_regime_labels(
                feats,
                target_index=valid_idx,
                fit_index=train_idx,
                mode=str(mode),
                buckets=int(buckets),
            )

            fit_idx = pd.to_datetime(pd.Index(labels_train.index)).intersection(train_idx)
            tgt_idx = pd.to_datetime(pd.Index(labels_valid.index)).intersection(valid_idx)
            if fit_idx.empty or tgt_idx.empty:
                continue

            fit_df = tr2.reindex(index=fit_idx).fillna(0.0)
            if int(len(fit_df.index)) < int(max(5, allocation_min_days)):
                w_global = pd.Series(1.0 / float(len(cols)), index=cols, dtype=float)
                weights_by_regime = {}
                daily_w, daily_diag = build_daily_alpha_weights(
                    labels=labels_valid.reindex(tgt_idx),
                    weights_by_regime=weights_by_regime,
                    fallback=w_global,
                    smoothing=float(smoothing),
                    max_weight=float(allocation_max_weight),
                )
            else:
                fit_out = fit_regime_allocations(
                    fit_df,
                    labels_train.reindex(fit_df.index),
                    trading_days=int(td),
                    score_metric=str(allocation_score_metric),
                    lambda_corr=float(allocation_lambda),
                    l2=float(allocation_l2),
                    turnover_lambda=float(allocation_turnover_lambda),
                    prev_global=w_prev_global,
                    prev_by_regime=w_prev_by_regime,
                    max_weight=float(allocation_max_weight),
                    use_abs_corr=bool(allocation_use_abs_corr),
                    backend=str(allocation_backend),
                    solver=str(allocation_solver),
                    min_days=int(regime_min_days),
                )
                w_global = fit_out.get("global_weights")
                if not isinstance(w_global, pd.Series) or w_global.empty:
                    w_global = pd.Series(1.0 / float(len(cols)), index=cols, dtype=float)
                weights_by_regime = dict(fit_out.get("weights_by_regime") or {})
                w_prev_by_regime = dict(fit_out.get("prev_by_regime") or {})
                w_prev_global = w_global

                daily_w, daily_diag = build_daily_alpha_weights(
                    labels=labels_valid.reindex(tgt_idx),
                    weights_by_regime=weights_by_regime,
                    fallback=w_global,
                    smoothing=float(smoothing),
                    max_weight=float(allocation_max_weight),
                )

            # Build alpha holdings on the validation segment (cache).
            w_hold: Dict[str, pd.DataFrame] = {}
            for aid in cols:
                cache_key = (str(aid), int(split_id))
                w_mat = holdings_cache.get(cache_key)
                if w_mat is None:
                    fac = factor_cache.get(str(aid))
                    if fac is None or fac.empty:
                        continue
                    sign = float(sign_map.get(str(aid), {}).get(int(split_id), 1.0))
                    cfg = bt_config_by_alpha.get(str(aid), bt_config) if bt_config_by_alpha else bt_config
                    bt = backtest_long_short(
                        sign * fac,
                        ohlcv,
                        config=cfg,
                        start=va[0],
                        end=va[1],
                        include_daily=False,
                        include_positions=True,
                        sector_map=sector_map,
                        borrow_rates=borrow_rates,
                        hard_to_borrow=hard_to_borrow,
                    )
                    if bt.get("error"):
                        continue
                    w_mat = positions_to_weight_matrix(
                        positions=list(bt.get("positions") or []),
                        position_dates=list(bt.get("position_dates") or []),
                        instruments=instruments,
                    )
                    holdings_cache[cache_key] = w_mat
                if isinstance(w_mat, pd.DataFrame) and not w_mat.empty:
                    w_hold[str(aid)] = w_mat

            if len(w_hold) < 2:
                continue

            common = pd.to_datetime(pd.Index(daily_w.index)).intersection(valid_idx)
            for w_mat in w_hold.values():
                common = common.intersection(pd.to_datetime(pd.Index(w_mat.index)))
            if common.empty or int(len(common)) < 3:
                continue

            # Diagnostics for constraint-based selection (P2.26).
            try:
                sr = float(regime_stats(labels_valid.reindex(common)).get("switch_rate") or 0.0)
                switch_rates.append(sr)
            except Exception:
                pass
            try:
                fallback_fracs.append(float(daily_diag.get("fallback_frac") or 0.0))
            except Exception:
                pass

            a_w = daily_w.reindex(index=common, columns=cols).fillna(0.0)

            w_ens = pd.DataFrame(0.0, index=common, columns=instruments)
            for aid, w_mat in w_hold.items():
                if aid not in a_w.columns:
                    continue
                try:
                    w_i = w_mat.reindex(index=common, columns=instruments).fillna(0.0)
                except Exception:
                    continue
                w_ens = w_ens.add(w_i.mul(a_w[aid], axis=0), fill_value=0.0)
            if w_ens.abs().sum().sum() <= 0.0:
                continue

            bt_ens = backtest_from_weights(
                w_ens,
                ohlcv,
                config=bt_config,
                start=va[0],
                end=va[1],
                include_daily=True,
                include_positions=False,
                apply_turnover_cap=bool(apply_turnover_cap),
                borrow_rates=borrow_rates,
                hard_to_borrow=hard_to_borrow,
            )
            if bt_ens.get("error"):
                continue

            daily = list(bt_ens.get("daily") or [])
            if not daily:
                continue

            try:
                idx = pd.to_datetime([d.get("datetime") for d in daily])
                vals = [float(d.get("net_return") or 0.0) for d in daily]
                s_raw = pd.Series(vals, index=idx).sort_index()
                s_raw = s_raw[~s_raw.index.duplicated(keep="first")]
            except Exception:
                continue

            to_s = _alpha_turnover_series(a_w).reindex(index=s_raw.index).fillna(0.0)
            if cost_bps > 0.0:
                s_adj = s_raw - to_s * float(cost_bps) / 10000.0
            else:
                s_adj = s_raw

            valid_series_raw.append(s_raw)
            valid_series_adj.append(s_adj)

            # Split-level stability diagnostics (objective after turnover cost).
            try:
                perf_sp = _summarize_returns(s_adj, trading_days=int(td))
                if not perf_sp.get("error"):
                    if tm in {"annualized_return", "ann"}:
                        split_obj_adj.append(float(perf_sp.get("annualized_return") or 0.0))
                    else:
                        split_obj_adj.append(float(perf_sp.get("information_ratio") or 0.0))
            except Exception:
                pass

            nobs = int(bt_ens.get("n_obs") or len(daily))
            if nobs > 0:
                ens_turn_wsum += float(bt_ens.get("turnover_mean") or 0.0) * float(nobs)
                ens_cost_wsum += float(bt_ens.get("cost_mean") or 0.0) * float(nobs)
                ens_borrow_wsum += float(bt_ens.get("borrow_mean") or 0.0) * float(nobs)
                ens_nsum += float(nobs)

            ato = float(to_s.mean()) if len(to_s.index) else 0.0
            alpha_to_wsum += float(ato) * float(len(common))
            alpha_to_nsum += float(len(common))
            n_used += 1

        if not valid_series_raw:
            ato_mean = float(alpha_to_wsum / max(alpha_to_nsum, 1.0))
            rows.append(
                {
                    "config_id": config_id,
                    "mode": mode,
                    "window": int(window),
                    "buckets": int(buckets),
                    "smoothing": float(smoothing),
                    "tune_metric": str(tune_metric),
                    "turnover_cost_bps": float(cost_bps),
                    "holdings_valid_metric": float("-inf"),
                    "holdings_valid_metric_after_turnover_cost": float("-inf"),
                    "holdings_objective": float("-inf"),
                    "n_splits_used": int(n_used),
                    "n_valid_days": 0,
                    "alpha_weight_turnover_mean": float(ato_mean),
                    "turnover_cost_drag_bps_mean": float(ato_mean * float(cost_bps)),
                    "ensemble_turnover_mean": float(ens_turn_wsum / max(ens_nsum, 1.0)),
                    "ensemble_cost_mean": float(ens_cost_wsum / max(ens_nsum, 1.0)),
                    "ensemble_borrow_mean": float(ens_borrow_wsum / max(ens_nsum, 1.0)),
                    "holdings_objective_split_std": 0.0,
                    "holdings_objective_split_min": 0.0,
                    "holdings_objective_split_negative_frac": 0.0,
                    "error": "no_valid_series",
                }
            )
            continue

        vv_raw = pd.concat(valid_series_raw).sort_index()
        vv_raw = vv_raw[~vv_raw.index.duplicated(keep="first")]
        vv_adj = pd.concat(valid_series_adj).sort_index()
        vv_adj = vv_adj[~vv_adj.index.duplicated(keep="first")]

        perf_raw = _summarize_returns(vv_raw, trading_days=int(td))
        perf_adj = _summarize_returns(vv_adj, trading_days=int(td))

        def _pick(perf: Dict[str, Any]) -> float:
            if not isinstance(perf, dict) or perf.get("error"):
                return 0.0
            if tm in {"annualized_return", "ann"}:
                return float(perf.get("annualized_return") or 0.0)
            return float(perf.get("information_ratio") or 0.0)

        valid_metric = _pick(perf_raw)
        obj_metric = _pick(perf_adj)

        ato_mean = float(alpha_to_wsum / max(alpha_to_nsum, 1.0))
        drag_bps = float(ato_mean) * float(cost_bps)

        # Stability: variability across splits.
        h_obj_std = float(np.std(split_obj_adj, ddof=1)) if len(split_obj_adj) >= 2 else 0.0
        h_obj_min = float(np.min(split_obj_adj)) if split_obj_adj else 0.0
        h_obj_neg = float(np.mean([1.0 if float(x) < 0.0 else 0.0 for x in split_obj_adj])) if split_obj_adj else 0.0

        row = {
            "config_id": config_id,
            "mode": mode,
            "window": int(window),
            "buckets": int(buckets),
            "smoothing": float(smoothing),
            "tune_metric": str(tune_metric),
            "turnover_cost_bps": float(cost_bps),
            "holdings_valid_metric": float(valid_metric),
            "holdings_valid_metric_after_turnover_cost": float(obj_metric),
            "holdings_objective": float(obj_metric),
            "n_splits_used": int(n_used),
            "n_valid_days": int(len(vv_raw.index)),
            "alpha_weight_turnover_mean": float(ato_mean),
            "turnover_cost_drag_bps_mean": float(drag_bps),
            "regime_switch_rate_mean": float(np.mean(switch_rates)) if switch_rates else 0.0,
            "fallback_frac_mean": float(np.mean(fallback_fracs)) if fallback_fracs else 0.0,
            "ensemble_turnover_mean": float(ens_turn_wsum / max(ens_nsum, 1.0)),
            "ensemble_cost_mean": float(ens_cost_wsum / max(ens_nsum, 1.0)),
            "ensemble_borrow_mean": float(ens_borrow_wsum / max(ens_nsum, 1.0)),
            "holdings_objective_split_std": float(h_obj_std),
            "holdings_objective_split_min": float(h_obj_min),
            "holdings_objective_split_negative_frac": float(h_obj_neg),
        }
        rows.append(row)

        if np.isfinite(obj_metric) and float(obj_metric) > best_obj:
            best_obj = float(obj_metric)
            best_row = dict(row)

    rows_sorted = sorted(rows, key=lambda d: float(d.get("holdings_objective") or float("-inf")), reverse=True)

    extra = list(pareto_metrics or [])
    if include_stability_objectives and str(selection_method).strip().lower() in {"knee", "utility"}:
        extra.extend(["holdings_objective_split_std", "holdings_objective_split_min"])

    pareto_obj = _pareto_objectives(extra)
    pareto_meta = annotate_pareto(
        rows_sorted,
        objectives=pareto_obj,
        pareto_key="is_pareto",
        rank_key="pareto_rank",
    )

    chosen, sel_meta = select_best_row(
        rows_sorted,
        objective_key="holdings_objective",
        constraints=constraints,
        prefer_pareto=bool(prefer_pareto),
        pareto_key="is_pareto",
        selection_method=str(selection_method),
        objectives=pareto_obj,
        utility_weights=utility_weights,
    )

    best_params: Dict[str, Any] = {}
    if chosen and np.isfinite(float(chosen.get("holdings_objective") or float("-inf"))):
        best_params = {
            "mode": str(chosen.get("mode") or "vol"),
            "window": int(chosen.get("window") or 20),
            "buckets": int(chosen.get("buckets") or 3),
            "smoothing": float(chosen.get("smoothing") or 0.0),
        }

    return {
        "enabled": bool(best_params),
        "top_n": int(top_n),
        "tune_metric": str(tune_metric),
        "turnover_penalty": float(cost_bps),
        "turnover_cost_bps": float(cost_bps),
        "constraints": dict(sel_meta.get("constraints") or {}),
        "selection": dict(sel_meta or {}),
        "feasible_count": int(sel_meta.get("feasible_count") or 0),
        "selected_by": str(sel_meta.get("selected_by") or ""),
        "chosen": chosen or {},
        "best_params": best_params,
        "pareto_objectives": list(pareto_meta.get("objectives") or []),
        "pareto_count": int(pareto_meta.get("pareto_count") or 0),
        "results": rows_sorted,
    }

def walk_forward_holdings_ensemble_allocated_regime(
    *,
    selected_alphas: List[Dict[str, Any]],
    factor_cache: Dict[str, pd.Series],
    ohlcv: pd.DataFrame,
    wf_config: WalkForwardConfig,
    bt_config: BacktestConfig,
    splits: Optional[List[Dict[str, Any]]] = None,
    bt_config_by_alpha: Optional[Dict[str, BacktestConfig]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
    apply_turnover_cap: bool = False,
    alpha_test_cache: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
    # Allocation config (shared with P2.19/P2.20).
    allocation_fit: str = "train_valid",  # "train" | "train_valid"
    allocation_backend: str = "auto",  # "auto" | "qp" | "pgd"
    allocation_score_metric: str = "information_ratio",
    allocation_lambda: float = 0.5,
    allocation_l2: float = 1e-6,
    allocation_turnover_lambda: float = 0.0,
    allocation_max_weight: float = 0.8,
    allocation_use_abs_corr: bool = True,
    allocation_min_days: int = 30,
    allocation_solver: str = "",
    # Optional meta-tuning.
    allocation_tune: bool = False,
    allocation_tune_metric: str = "information_ratio",
    allocation_tune_max_combos: int = 24,
    allocation_tune_lambda_grid: Optional[Sequence[float]] = None,
    allocation_tune_max_weight_grid: Optional[Sequence[float]] = None,
    allocation_tune_turnover_lambda_grid: Optional[Sequence[float]] = None,
    allocation_tune_save_top: int = 10,
    # Regime config.
    regime_mode: str = "vol",
    regime_window: int = 20,
    regime_buckets: int = 3,
    regime_min_days: int = 30,
    regime_smoothing: float = 0.0,
    # P2.23: regime meta-tuning (train->valid across splits).
    regime_tune: bool = False,
    regime_tune_metric: str = "information_ratio",
    regime_tune_max_combos: int = 24,
    regime_tune_mode_grid: Optional[Sequence[str]] = None,
    regime_tune_window_grid: Optional[Sequence[int]] = None,
    regime_tune_buckets_grid: Optional[Sequence[int]] = None,
    regime_tune_smoothing_grid: Optional[Sequence[float]] = None,
    regime_tune_turnover_penalty: float = 0.0,
    regime_tune_save_top: int = 10,
    # P2.24: holdings-level revalidation for the top proxy configs.
    regime_tune_holdings_top: int = 3,
    regime_tune_holdings_metric: str = "",
    regime_tune_holdings_save_top: int = 10,
    # P2.26: constraint-based selection for regime configs.
    regime_tune_max_alpha_turnover: Optional[float] = None,
    regime_tune_max_turnover_cost_drag_bps: Optional[float] = None,
    regime_tune_max_regime_switch_rate: Optional[float] = None,
    regime_tune_max_fallback_frac: Optional[float] = None,
    regime_tune_prefer_pareto: bool = False,
    regime_tune_pareto_metrics: Optional[Sequence[str]] = None,
    regime_tune_selection_method: str = "best_objective",
    regime_tune_utility_weights: Optional[Dict[str, Any]] = None,
    regime_tune_include_stability_objectives: bool = True,
    # Optional cache to avoid recomputing per-split train/valid return matrices.
    split_returns_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Holdings-level ensemble with regime-aware alpha allocation.

    For each walk-forward split we:
    1) fit global alpha weights on the fit segment (train or train+valid)
    2) fit per-regime alpha weights on subsets of the fit segment
    3) map each test day to a regime using thresholds from the fit segment
    4) blend holdings using the regime-specific weights (with optional smoothing)
    """

    if not selected_alphas:
        return {"enabled": False, "error": "Empty selection"}

    alpha_ids: List[str] = []
    for a in selected_alphas:
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if aid is not None:
            alpha_ids.append(str(aid))
    alpha_ids = list(dict.fromkeys(alpha_ids))
    if len(alpha_ids) < 2:
        return {"enabled": False, "error": "Need at least 2 alphas for allocation"}

    if splits is None:
        splits = make_walk_forward_splits(
            ohlcv.index.get_level_values("datetime").unique(),
            train_days=wf_config.train_days,
            valid_days=wf_config.valid_days,
            test_days=wf_config.test_days,
            step_days=wf_config.step_days,
            expanding_train=wf_config.expanding_train,
        )
    if not splits:
        return {"enabled": False, "error": "Not enough data for walk-forward splits"}

    instruments = list(pd.Index(ohlcv.index.get_level_values("instrument")).unique())

    sign_map: Dict[str, Dict[int, float]] = {aid: {} for aid in alpha_ids}
    for a in selected_alphas:
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if aid is None:
            continue
        sign_map[str(aid)] = _extract_split_signs(a)

    fit_mode = str(allocation_fit).lower().strip()
    td = int(getattr(bt_config, "trading_days", 252) or 252)

    # Market features are computed after optional regime meta-tuning (P2.23).
    feats: Optional[pd.DataFrame] = None

    # Reuse (train, valid) return matrices if a cache is provided.
    cached = None
    if isinstance(split_returns_cache, dict):
        cached = split_returns_cache.get("split_returns")
        if isinstance(cached, dict) and cached.get("alpha_ids") != list(alpha_ids):
            cached = None

    if isinstance(cached, dict):
        split_info = list(cached.get("split_info") or [])
        train_by_split = dict(cached.get("train_by_split") or {})
        valid_by_split = dict(cached.get("valid_by_split") or {})
    else:
        split_info, train_by_split, valid_by_split = _build_split_return_matrices(
            alpha_ids=alpha_ids,
            factor_cache=factor_cache,
            sign_map=sign_map,
            ohlcv=ohlcv,
            splits=list(splits),
            bt_config=bt_config,
            bt_config_by_alpha=bt_config_by_alpha,
            sector_map=sector_map,
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )
        if isinstance(split_returns_cache, dict):
            split_returns_cache["split_returns"] = {
                "alpha_ids": list(alpha_ids),
                "split_info": split_info,
                "train_by_split": train_by_split,
                "valid_by_split": valid_by_split,
            }

    # Optional meta-tuning (same as P2.20).
    tuned = False
    tuning_payload: Dict[str, Any] = {}
    if bool(allocation_tune) and valid_by_split:
        try:
            base_params = {
                "lambda_corr": float(allocation_lambda),
                "max_weight": float(allocation_max_weight),
                "turnover_lambda": float(allocation_turnover_lambda),
                "l2": float(allocation_l2),
            }

            param_lists = default_allocation_sweep_param_lists(
                base_lambda_corr=float(allocation_lambda),
                base_max_weight=float(allocation_max_weight),
                base_turnover_lambda=float(allocation_turnover_lambda),
            )
            if allocation_tune_lambda_grid:
                param_lists["lambda_corr"] = [float(x) for x in allocation_tune_lambda_grid]
            if allocation_tune_max_weight_grid:
                param_lists["max_weight"] = [float(x) for x in allocation_tune_max_weight_grid]
            if allocation_tune_turnover_lambda_grid:
                param_lists["turnover_lambda"] = [float(x) for x in allocation_tune_turnover_lambda_grid]

            tuning_payload = meta_tune_alpha_allocation(
                train_by_split=train_by_split,
                valid_by_split=valid_by_split,
                trading_days=int(td),
                score_metric=str(allocation_score_metric),
                use_abs_corr=bool(allocation_use_abs_corr),
                backend=str(allocation_backend),
                solver=str(allocation_solver),
                base_params=base_params,
                param_lists=param_lists,
                max_combos=int(allocation_tune_max_combos),
                min_days=int(allocation_min_days),
                tune_metric=str(allocation_tune_metric),
            )

            best = dict(tuning_payload.get("best_params") or {})
            allocation_lambda = float(best.get("lambda_corr") or allocation_lambda)
            allocation_max_weight = float(best.get("max_weight") or allocation_max_weight)
            allocation_turnover_lambda = float(best.get("turnover_lambda") or allocation_turnover_lambda)
            tuned = True
        except Exception as e:
            tuning_payload = {"enabled": False, "error": str(e)}

    # P2.23: meta-tune regime hyperparams (mode/window/buckets/smoothing) on
    # train->valid splits, with an optional alpha-weight turnover cost (bps).
    # P2.26: regime tuning constraints (optional).
    regime_constraints: Dict[str, Any] = {}
    if regime_tune_max_alpha_turnover is not None:
        regime_constraints["max_alpha_weight_turnover_mean"] = float(regime_tune_max_alpha_turnover)
    if regime_tune_max_turnover_cost_drag_bps is not None:
        regime_constraints["max_turnover_cost_drag_bps_mean"] = float(regime_tune_max_turnover_cost_drag_bps)
    if regime_tune_max_regime_switch_rate is not None:
        regime_constraints["max_regime_switch_rate_mean"] = float(regime_tune_max_regime_switch_rate)
    if regime_tune_max_fallback_frac is not None:
        regime_constraints["max_fallback_frac_mean"] = float(regime_tune_max_fallback_frac)

    regime_tuned = False
    regime_tuned_method = "fixed"
    regime_tuning_payload: Dict[str, Any] = {}
    if bool(regime_tune) and valid_by_split:
        try:
            reg_param_lists = default_regime_sweep_param_lists(
                base_mode=str(regime_mode),
                base_window=int(regime_window),
                base_buckets=int(regime_buckets),
                base_smoothing=float(regime_smoothing),
            )
            if regime_tune_mode_grid:
                reg_param_lists["mode"] = [str(x) for x in regime_tune_mode_grid]
            if regime_tune_window_grid:
                reg_param_lists["window"] = [int(x) for x in regime_tune_window_grid]
            if regime_tune_buckets_grid:
                reg_param_lists["buckets"] = [int(x) for x in regime_tune_buckets_grid]
            if regime_tune_smoothing_grid:
                reg_param_lists["smoothing"] = [float(x) for x in regime_tune_smoothing_grid]

            regime_tuning_payload = meta_tune_regime_aware_allocation(
                ohlcv=ohlcv,
                train_by_split=train_by_split,
                valid_by_split=valid_by_split,
                trading_days=int(td),
                score_metric=str(allocation_score_metric),
                lambda_corr=float(allocation_lambda),
                l2=float(allocation_l2),
                turnover_lambda=float(allocation_turnover_lambda),
                max_weight=float(allocation_max_weight),
                use_abs_corr=bool(allocation_use_abs_corr),
                backend=str(allocation_backend),
                solver=str(allocation_solver),
                regime_min_days=int(regime_min_days),
                param_lists=reg_param_lists,
                max_combos=int(regime_tune_max_combos),
                tune_metric=str(regime_tune_metric),
                turnover_cost_bps=float(regime_tune_turnover_penalty),
                constraints=regime_constraints,
                prefer_pareto=bool(regime_tune_prefer_pareto),
                pareto_metrics=list(regime_tune_pareto_metrics or []),
                selection_method=str(regime_tune_selection_method),
                utility_weights=dict(regime_tune_utility_weights or {}),
                include_stability_objectives=bool(regime_tune_include_stability_objectives),
            )

            best_reg = dict(regime_tuning_payload.get("best_params") or {})
            if best_reg:
                regime_mode = str(best_reg.get("mode") or regime_mode)
                regime_window = int(best_reg.get("window") or regime_window)
                regime_buckets = int(best_reg.get("buckets") or regime_buckets)
                regime_smoothing = float(best_reg.get("smoothing") or regime_smoothing)
                regime_tuned = True
                regime_tuned_method = "proxy"
        except Exception as e:
            regime_tuning_payload = {"enabled": False, "error": str(e)}

    # P2.24: holdings-level revalidation for the top proxy configs.
    regime_holdings_validation_payload: Dict[str, Any] = {}
    if bool(regime_tune) and int(regime_tune_holdings_top) > 0 and isinstance(regime_tuning_payload, dict):
        try:
            metric_h = str(regime_tune_holdings_metric or regime_tune_metric)
            regime_holdings_validation_payload = _revalidate_regime_configs_holdings(
                candidate_rows=list(regime_tuning_payload.get("results") or []),
                top_n=int(regime_tune_holdings_top),
                ohlcv=ohlcv,
                instruments=instruments,
                alpha_ids=alpha_ids,
                factor_cache=factor_cache,
                sign_map=sign_map,
                split_info=split_info,
                train_by_split=train_by_split,
                valid_by_split=valid_by_split,
                bt_config=bt_config,
                bt_config_by_alpha=bt_config_by_alpha,
                sector_map=sector_map,
                borrow_rates=borrow_rates,
                hard_to_borrow=hard_to_borrow,
                allocation_score_metric=str(allocation_score_metric),
                allocation_lambda=float(allocation_lambda),
                allocation_l2=float(allocation_l2),
                allocation_turnover_lambda=float(allocation_turnover_lambda),
                allocation_max_weight=float(allocation_max_weight),
                allocation_use_abs_corr=bool(allocation_use_abs_corr),
                allocation_backend=str(allocation_backend),
                allocation_solver=str(allocation_solver),
                allocation_min_days=int(allocation_min_days),
                regime_min_days=int(regime_min_days),
                tune_metric=str(metric_h),
                turnover_cost_bps=float(regime_tune_turnover_penalty),
                apply_turnover_cap=bool(apply_turnover_cap),
                constraints=regime_constraints,
                prefer_pareto=bool(regime_tune_prefer_pareto),
                pareto_metrics=list(regime_tune_pareto_metrics or []),
                selection_method=str(regime_tune_selection_method),
                utility_weights=dict(regime_tune_utility_weights or {}),
                include_stability_objectives=bool(regime_tune_include_stability_objectives),
            )

            best_h = dict(regime_holdings_validation_payload.get("best_params") or {})
            if best_h:
                regime_mode = str(best_h.get("mode") or regime_mode)
                regime_window = int(best_h.get("window") or regime_window)
                regime_buckets = int(best_h.get("buckets") or regime_buckets)
                regime_smoothing = float(best_h.get("smoothing") or regime_smoothing)
                regime_tuned = True
                regime_tuned_method = "holdings_valid"
        except Exception as e:
            regime_holdings_validation_payload = {"enabled": False, "error": str(e)}

    # Compute regime features once, using the (possibly tuned) window.
    if feats is None:
        feats = compute_market_feature_frame(ohlcv, window=int(regime_window), min_obs=max(5, int(regime_window)))

    # Walk-forward test evaluation.
    oos_daily_all: List[Dict[str, Any]] = []
    oos_positions_all: List[Dict[str, Any]] = []
    allocations_regime_all: List[Dict[str, Any]] = []
    allocation_diagnostics_all: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []

    w_prev_global: Optional[pd.Series] = None
    w_prev_by_regime: Dict[str, pd.Series] = {}

    for sp in split_info:
        split_id = int(sp.get("split_id") or 0)
        tr = sp.get("train")
        va = sp.get("valid")
        te = sp.get("test")
        if not tr or not te:
            continue

        fit_start = tr[0]
        fit_end = tr[1] if (fit_mode == "train" or not va) else va[1]

        df_tr = train_by_split.get(int(split_id))
        df_va = valid_by_split.get(int(split_id)) if va else None
        if df_tr is None or df_tr.empty or df_tr.shape[1] < 2:
            continue

        if fit_mode == "train" or df_va is None or df_va.empty:
            fit_df = df_tr
        else:
            fit_df = pd.concat([df_tr, df_va], axis=0).sort_index()
            fit_df = fit_df[~fit_df.index.duplicated(keep="first")]
        fit_df = fit_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        fit_idx = pd.to_datetime(pd.Index(fit_df.index)).sort_values()
        labels_fit, label_meta = make_regime_labels(
            feats,
            target_index=fit_idx,
            fit_index=fit_idx,
            mode=str(regime_mode),
            buckets=int(regime_buckets),
        )

        if int(len(fit_df.index)) < int(allocation_min_days):
            w_global = pd.Series(1.0 / float(fit_df.shape[1]), index=fit_df.columns, dtype=float)
            weights_by_regime: Dict[str, pd.Series] = {}
            diag = [{"method": "fallback_equal", "reason": "not_enough_fit_days", "n_fit_days": int(len(fit_df.index))}]
        else:
            fit_out = fit_regime_allocations(
                fit_df,
                labels_fit.reindex(fit_df.index),
                trading_days=int(td),
                score_metric=str(allocation_score_metric),
                lambda_corr=float(allocation_lambda),
                l2=float(allocation_l2),
                turnover_lambda=float(allocation_turnover_lambda),
                prev_global=w_prev_global,
                prev_by_regime=w_prev_by_regime,
                max_weight=float(allocation_max_weight),
                use_abs_corr=bool(allocation_use_abs_corr),
                backend=str(allocation_backend),
                solver=str(allocation_solver),
                min_days=int(regime_min_days),
            )
            w_global = fit_out.get("global_weights")
            if not isinstance(w_global, pd.Series) or w_global.empty:
                w_global = pd.Series(1.0 / float(fit_df.shape[1]), index=fit_df.columns, dtype=float)
            weights_by_regime = dict(fit_out.get("weights_by_regime") or {})
            w_prev_by_regime = dict(fit_out.get("prev_by_regime") or {})
            diag = list(fit_out.get("diagnostics") or [])

        w_prev_global = w_global.reindex(alpha_ids).fillna(0.0)
        if float(w_prev_global.sum()) > 0.0:
            w_prev_global = w_prev_global / float(w_prev_global.sum())

        # Persist weights for tracking.
        for aid, wa in w_global.items():
            allocations_regime_all.append(
                {
                    "split_id": int(split_id),
                    "regime": "__global__",
                    "alpha_id": str(aid),
                    "weight": float(wa or 0.0),
                    "fit_start": pd.to_datetime(fit_start).isoformat(),
                    "fit_end": pd.to_datetime(fit_end).isoformat(),
                }
            )
        for reg, w in weights_by_regime.items():
            for aid, wa in w.items():
                allocations_regime_all.append(
                    {
                        "split_id": int(split_id),
                        "regime": str(reg),
                        "alpha_id": str(aid),
                        "weight": float(wa or 0.0),
                        "fit_start": pd.to_datetime(fit_start).isoformat(),
                        "fit_end": pd.to_datetime(fit_end).isoformat(),
                    }
                )

        for d in diag:
            if isinstance(d, dict):
                allocation_diagnostics_all.append({"split_id": int(split_id), **d})

        # Build per-alpha test holdings.
        t_start, t_end = te
        w_by_alpha: Dict[str, pd.DataFrame] = {}
        for aid in alpha_ids:
            fac = factor_cache.get(str(aid))
            if fac is None:
                continue
            sign = float(sign_map.get(str(aid), {}).get(split_id, 1.0))
            cfg = bt_config_by_alpha.get(str(aid), bt_config) if bt_config_by_alpha else bt_config

            cache_key = (str(aid), int(split_id))
            cached = None if alpha_test_cache is None else alpha_test_cache.get(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("weights"), pd.DataFrame):
                w_mat = cached["weights"]
            else:
                bt = backtest_long_short(
                    sign * fac,
                    ohlcv,
                    config=cfg,
                    start=t_start,
                    end=t_end,
                    include_daily=False,
                    include_positions=True,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                )
                if bt.get("error"):
                    continue
                pos_rows = list(bt.get("positions") or [])
                pos_dates = list(bt.get("position_dates") or [])
                w_mat = positions_to_weight_matrix(positions=pos_rows, position_dates=pos_dates, instruments=instruments)
                if alpha_test_cache is not None:
                    alpha_test_cache[cache_key] = {"weights": w_mat, "positions": pos_rows, "position_dates": pos_dates}

            if w_mat is None or w_mat.empty:
                continue
            w_by_alpha[str(aid)] = w_mat

        if not w_by_alpha:
            continue

        all_idx = pd.Index(sorted({d for w in w_by_alpha.values() for d in w.index}))
        if all_idx.empty:
            continue

        labels_test, _ = make_regime_labels(
            feats,
            target_index=pd.to_datetime(all_idx),
            fit_index=fit_idx,
            mode=str(regime_mode),
            buckets=int(regime_buckets),
        )

        daily_w, daily_diag = build_daily_alpha_weights(
            labels=labels_test,
            weights_by_regime=weights_by_regime,
            fallback=w_global,
            smoothing=float(regime_smoothing),
            max_weight=float(allocation_max_weight),
        )

        # Combine holdings with daily alpha weights.
        w_sum = None
        for aid, w_mat in w_by_alpha.items():
            if aid not in daily_w.columns:
                continue
            ww = w_mat.reindex(index=all_idx, columns=instruments).fillna(0.0)
            ww = ww.mul(daily_w[aid].reindex(all_idx).fillna(0.0), axis=0)
            w_sum = ww if w_sum is None else w_sum.add(ww, fill_value=0.0)

        w_ens = w_sum if w_sum is not None else pd.DataFrame()
        if w_ens.empty:
            continue

        bt_ens = backtest_from_weights(
            w_ens,
            ohlcv,
            config=bt_config,
            start=t_start,
            end=t_end,
            include_daily=True,
            include_positions=True,
            apply_turnover_cap=bool(apply_turnover_cap),
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )
        if bt_ens.get("error"):
            continue

        daily = list(bt_ens.get("daily") or [])
        if daily:
            oos_daily_all.extend(daily)
        pos = list(bt_ens.get("positions") or [])
        if pos:
            oos_positions_all.extend(pos)

        test_stats = regime_stats(labels_test)
        split_rows.append(
            {
                "split_id": int(split_id),
                "fit": {"start": pd.to_datetime(fit_start).isoformat(), "end": pd.to_datetime(fit_end).isoformat()},
                "test": {"start": pd.to_datetime(t_start).isoformat(), "end": pd.to_datetime(t_end).isoformat()},
                "regime": {
                    "mode": str(label_meta.mode),
                    "buckets": int(label_meta.buckets),
                    "effective_vol_buckets": int(label_meta.effective_vol_buckets),
                    "effective_liq_buckets": int(label_meta.effective_liq_buckets),
                },
                "n_regimes_fitted": int(len(weights_by_regime)),
                "daily_weighting": {**daily_diag, **test_stats},
                "metrics": {
                    "information_ratio": bt_ens.get("information_ratio"),
                    "annualized_return": bt_ens.get("annualized_return"),
                    "max_drawdown": bt_ens.get("max_drawdown"),
                },
            }
        )

    if not oos_daily_all:
        return {"enabled": False, "error": "No OOS daily results for regime-aware allocated ensemble"}

    df_daily = pd.DataFrame(oos_daily_all)
    if "datetime" in df_daily.columns:
        df_daily["datetime"] = pd.to_datetime(df_daily["datetime"])
        df_daily = df_daily.drop_duplicates(subset=["datetime"], keep="first")
        df_daily = df_daily.sort_values(by=["datetime"], ascending=True)

    net = pd.Series(df_daily["net_return"].to_numpy(dtype=float), index=pd.to_datetime(df_daily["datetime"]))
    perf = _summarize_returns(net, trading_days=int(td))
    perf["n_alphas"] = int(len(alpha_ids))
    perf["apply_turnover_cap"] = bool(apply_turnover_cap)

    for k in ["turnover", "cost", "linear_cost", "spread_cost", "impact_cost", "borrow"]:
        if k in df_daily.columns:
            try:
                perf[f"{k}_mean"] = float(pd.to_numeric(df_daily[k], errors="coerce").fillna(0.0).mean())
            except Exception:
                pass

    out = {
        "enabled": True,
        "method": "allocated_holdings_regime",
        "selected_alpha_ids": alpha_ids,
        "allocation": {
            "fit": str(allocation_fit),
            "backend": str(allocation_backend),
            "score_metric": str(allocation_score_metric),
            "lambda_corr": float(allocation_lambda),
            "l2": float(allocation_l2),
            "turnover_lambda": float(allocation_turnover_lambda),
            "max_weight": float(allocation_max_weight),
            "use_abs_corr": bool(allocation_use_abs_corr),
            "min_days": int(allocation_min_days),
            "tuned": bool(tuned),
            "regime": {
                "mode": str(regime_mode),
                "window": int(regime_window),
                "buckets": int(regime_buckets),
                "min_days": int(regime_min_days),
                "smoothing": float(regime_smoothing),
                "tuned": bool(regime_tuned),
                "tuned_method": str(regime_tuned_method),
            },
        },
        "metrics": perf,
        "daily": df_daily.assign(datetime=df_daily["datetime"].astype(str)).to_dict(orient="records"),
        "positions": oos_positions_all,
        "allocations_regime": allocations_regime_all,
        "allocation_regime_diagnostics": allocation_diagnostics_all,
        "allocation_tuning": (
            {
                "enabled": bool(tuned),
                "tune_metric": str(allocation_tune_metric),
                "chosen": dict(tuning_payload.get("chosen") or {}),
                "results": (
                    list(tuning_payload.get("results") or [])
                    if int(allocation_tune_save_top) == 0
                    else list((tuning_payload.get("results") or [])[: int(max(0, allocation_tune_save_top))])
                ),
                **(
                    {"error": str(tuning_payload.get("error"))}
                    if isinstance(tuning_payload, dict) and tuning_payload.get("error")
                    else {}
                ),
            }
            if tuning_payload
            else {"enabled": False}
        ),
        "allocation_tuning_results": list(tuning_payload.get("results") or []) if tuning_payload else [],
        "regime_tuning": (
            {
                "enabled": bool(regime_tuned),
                "tune_metric": str(regime_tune_metric),
                "turnover_penalty": float(regime_tune_turnover_penalty),
                # Richer selection metadata for debugging and reporting.
                "turnover_cost_bps": float(regime_tuning_payload.get("turnover_cost_bps") or regime_tune_turnover_penalty),
                "constraints": dict(regime_tuning_payload.get("constraints") or {}),
                "selection": dict(regime_tuning_payload.get("selection") or {}),
                "selected_by": str(regime_tuning_payload.get("selected_by") or ""),
                "feasible_count": int(regime_tuning_payload.get("feasible_count") or 0),
                "pareto_objectives": list(regime_tuning_payload.get("pareto_objectives") or []),
                "pareto_count": int(regime_tuning_payload.get("pareto_count") or 0),
                "chosen": dict(regime_tuning_payload.get("chosen") or {}),
                "results": (
                    list(regime_tuning_payload.get("results") or [])
                    if int(regime_tune_save_top) == 0
                    else list((regime_tuning_payload.get("results") or [])[: int(max(0, regime_tune_save_top))])
                ),
                **(
                    {"error": str(regime_tuning_payload.get("error"))}
                    if isinstance(regime_tuning_payload, dict) and regime_tuning_payload.get("error")
                    else {}
                ),
            }
            if regime_tuning_payload
            else {"enabled": False}
        ),
        "regime_tuning_results": list(regime_tuning_payload.get("results") or []) if regime_tuning_payload else [],

        "regime_tuning_holdings_validation": (
            {
                "enabled": bool(regime_holdings_validation_payload.get("enabled")),
                "tune_metric": str(regime_tune_holdings_metric or regime_tune_metric),
                "top_n": int(regime_tune_holdings_top),
                "turnover_penalty": float(regime_tune_turnover_penalty),
                # Mirror the proxy-stage selection metadata.
                "turnover_cost_bps": float(regime_holdings_validation_payload.get("turnover_cost_bps") or regime_tune_turnover_penalty),
                "constraints": dict(regime_holdings_validation_payload.get("constraints") or {}),
                "selection": dict(regime_holdings_validation_payload.get("selection") or {}),
                "selected_by": str(regime_holdings_validation_payload.get("selected_by") or ""),
                "feasible_count": int(regime_holdings_validation_payload.get("feasible_count") or 0),
                "pareto_objectives": list(regime_holdings_validation_payload.get("pareto_objectives") or []),
                "pareto_count": int(regime_holdings_validation_payload.get("pareto_count") or 0),
                "chosen": dict(regime_holdings_validation_payload.get("chosen") or {}),
                "results": (
                    list(regime_holdings_validation_payload.get("results") or [])
                    if int(regime_tune_holdings_save_top) == 0
                    else list(
                        (regime_holdings_validation_payload.get("results") or [])[
                            : int(max(0, regime_tune_holdings_save_top))
                        ]
                    )
                ),
                **(
                    {"error": str(regime_holdings_validation_payload.get("error"))}
                    if isinstance(regime_holdings_validation_payload, dict) and regime_holdings_validation_payload.get("error")
                    else {}
                ),
            }
            if regime_holdings_validation_payload
            else {"enabled": False}
        ),
        "regime_tuning_holdings_validation_results": (
            list(regime_holdings_validation_payload.get("results") or []) if regime_holdings_validation_payload else []
        ),
        "walk_forward": {
            "config": {"walk_forward": asdict(wf_config), "backtest": asdict(bt_config)},
            "splits": split_rows,
        },
    }
    return out
