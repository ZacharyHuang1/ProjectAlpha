"""agent.research.alpha_allocation_regime_tuning

P2.23: Meta-tuning for regime-aware alpha allocation.

We tune *regime labeling* hyperparameters (mode/window/buckets) and the
*daily alpha-weight smoothing* coefficient using walk-forward validation
segments.

No leakage:
- For each split we fit regime thresholds and regime-specific allocations on
  the split's train segment.
- We score the resulting dynamic alpha weights on the split's valid segment.
- Test segments are never used for tuning.

The tuning objective can optionally include an *interpretable bps cost*
applied to the day-to-day turnover of the *alpha-weight vector*.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent.research.alpha_allocation_regime import build_daily_alpha_weights, fit_regime_allocations
from agent.research.regime_features import compute_market_feature_frame
from agent.research.regime_labels import make_regime_labels, regime_stats
from agent.research.constraint_selection import annotate_pareto, select_best_row


def _as_int_list(values: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for v in values:
        try:
            iv = int(float(v))
            if iv >= 1:
                out.append(int(iv))
        except Exception:
            continue
    # Deduplicate while preserving order.
    seen = set()
    uniq: List[int] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(int(v))
    return uniq


def _as_float_list(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
            if np.isfinite(fv):
                out.append(float(fv))
        except Exception:
            continue
    seen = set()
    uniq: List[float] = []
    for v in out:
        key = round(float(v), 12)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


def _as_str_list(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if s:
            out.append(s)
    seen = set()
    uniq: List[str] = []
    for s in out:
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
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


def default_regime_sweep_param_lists(
    *,
    base_mode: str,
    base_window: int,
    base_buckets: int,
    base_smoothing: float,
) -> Dict[str, List[Any]]:
    """Small default grids for regime-aware allocation tuning."""

    mode0 = str(base_mode).strip() or "vol"
    win0 = int(base_window) if int(base_window) > 0 else 20
    buc0 = int(base_buckets) if int(base_buckets) > 0 else 3
    sm0 = float(base_smoothing)

    modes = [mode0]
    if mode0 != "vol_liq":
        modes.append("vol_liq")
    if mode0 != "vol":
        modes.append("vol")

    # Keep the window grid small; window impacts feature computation cost.
    windows = [win0]
    if win0 not in {10, 20, 40}:
        windows.extend([10, 20, 40])
    else:
        # Add a neighbor window for small local exploration.
        windows.append(int(max(5, win0 * 2)))

    buckets = [buc0]
    if buc0 != 3:
        buckets.append(3)
    buckets.extend([2, 4])

    smooth = [0.0, sm0, 0.1, 0.2]

    return {
        "mode": _as_str_list(modes),
        "window": _as_int_list(windows),
        "buckets": _as_int_list(buckets),
        "smoothing": _as_float_list(smooth),
    }


def generate_regime_sweep_configs(
    *,
    param_lists: Dict[str, List[Any]],
    max_combos: int = 24,
) -> List[Dict[str, Any]]:
    """Deterministically generate a small grid of regime configs."""

    modes = list(param_lists.get("mode") or [])
    windows = list(param_lists.get("window") or [])
    buckets = list(param_lists.get("buckets") or [])
    smooth = list(param_lists.get("smoothing") or [])
    if not modes or not windows or not buckets or not smooth:
        return [{"config_id": "base", "params": {}}]

    combos = list(product(modes, windows, buckets, smooth))
    combos = downsample_grid(combos, max_points=int(max_combos))

    out: List[Dict[str, Any]] = []
    for mode, window, buc, sm in combos:
        params = {
            "mode": str(mode),
            "window": int(window),
            "buckets": int(buc),
            "smoothing": float(sm),
        }
        cid = f"m{str(mode)}_w{int(window)}_b{int(buc)}_s{float(sm):.3g}"
        cid = cid.replace("-", "m").replace(".", "p")
        out.append({"config_id": cid, "params": params})

    out.insert(0, {"config_id": "base", "params": {}})
    return out


def _alpha_weight_turnover_mean(w: pd.DataFrame) -> float:
    """Mean day-to-day turnover of an alpha-weight matrix (0..1)."""

    if w is None or w.empty or w.shape[0] < 2:
        return 0.0
    try:
        d = w.diff().abs().sum(axis=1).fillna(0.0)
        return float(0.5 * float(d.mean()))
    except Exception:
        return 0.0


def meta_tune_regime_aware_allocation(
    *,
    ohlcv: Optional[pd.DataFrame],
    train_by_split: Dict[int, pd.DataFrame],
    valid_by_split: Dict[int, pd.DataFrame],
    trading_days: int,
    score_metric: str,
    # Allocation params (kept fixed during regime tuning).
    lambda_corr: float,
    l2: float,
    turnover_lambda: float,
    max_weight: float,
    use_abs_corr: bool,
    backend: str,
    solver: str,
    regime_min_days: int,
    # Regime grid.
    param_lists: Dict[str, List[Any]],
    max_combos: int = 24,
    tune_metric: str = "information_ratio",
    turnover_penalty: float = 0.0,
    turnover_cost_bps: Optional[float] = None,
    # Optional cache for deterministic tests / re-use.
    features_by_window: Optional[Dict[int, pd.DataFrame]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    prefer_pareto: bool = False,
    pareto_metrics: Optional[Sequence[str]] = None,
    selection_method: str = "best_objective",
    utility_weights: Optional[Dict[str, Any]] = None,
    include_stability_objectives: bool = True,
) -> Dict[str, Any]:
    """Pick a single regime config by aggregating valid performance across splits.

    P2.25: Make the optional turnover penalty interpretable by applying it as a
    *bps cost* on the day-to-day turnover of the alpha-weight vector:

        r_adj[t] = r[t] - alpha_turnover[t] * turnover_cost_bps / 10000

    This keeps the objective in the same units as the chosen performance metric
    (IR / annualized return), while giving the penalty a clear meaning.
    """

    # Backward compatible alias: older callers pass turnover_penalty.
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
        """Build a list of Pareto objectives for proxy tuning rows."""

        base: List[Tuple[str, str]] = [
            ("objective", "max"),
            ("alpha_weight_turnover_mean", "min"),
        ]

        # Supported extra metrics. Missing keys are simply ignored by the Pareto annotator.
        extra_map = {
            "turnover_cost_drag_bps_mean": ("turnover_cost_drag_bps_mean", "min"),
            "regime_switch_rate_mean": ("regime_switch_rate_mean", "min"),
            "fallback_frac_mean": ("fallback_frac_mean", "min"),
            # Stability (optional).
            "objective_split_std": ("objective_split_std", "min"),
            "objective_split_min": ("objective_split_min", "max"),
            "objective_split_negative_frac": ("objective_split_negative_frac", "min"),
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

    configs = generate_regime_sweep_configs(param_lists=param_lists, max_combos=int(max_combos))

    split_ids = sorted(set(train_by_split.keys()).intersection(set(valid_by_split.keys())))
    if not split_ids:
        return {"enabled": False, "error": "no_common_splits"}

    # Pre-compute market features for each candidate window.
    windows = sorted({int((c.get("params") or {}).get("window") or 0) for c in configs if isinstance(c, dict)})
    windows = [w for w in windows if w > 0]
    feat_cache: Dict[int, pd.DataFrame] = {}
    if isinstance(features_by_window, dict):
        for w in windows:
            f = features_by_window.get(int(w))
            if isinstance(f, pd.DataFrame) and not f.empty:
                feat_cache[int(w)] = f

    for w in windows:
        if int(w) in feat_cache:
            continue
        if ohlcv is None or not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
            continue
        try:
            feat_cache[int(w)] = compute_market_feature_frame(ohlcv, window=int(w), min_obs=max(5, int(w)))
        except Exception:
            continue

    rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_obj = float("-inf")

    tm = str(tune_metric).lower().strip()

    for cfg in configs:
        params = dict(cfg.get("params") or {})
        mode = str(params.get("mode") or "vol")
        window = int(params.get("window") or 0)
        buckets = int(params.get("buckets") or 3)
        smoothing = float(params.get("smoothing") or 0.0)

        feats = feat_cache.get(int(window))
        if feats is None or feats.empty:
            rows.append(
                {
                    "config_id": cfg.get("config_id"),
                    "mode": mode,
                    "window": int(window),
                    "buckets": int(buckets),
                    "smoothing": float(smoothing),
                    "tune_metric": str(tune_metric),
                    "turnover_cost_bps": float(cost_bps),
                    "valid_metric": float("-inf"),
                    "valid_metric_after_turnover_cost": float("-inf"),
                    "objective": float("-inf"),
                    "n_splits_used": 0,
                    "n_valid_days": 0,
                    "alpha_weight_turnover_mean": 0.0,
                    "turnover_cost_drag_bps_mean": 0.0,
                    "regime_switch_rate_mean": 0.0,
                    "fallback_frac_mean": 0.0,
                    "objective_split_std": 0.0,
                    "objective_split_min": 0.0,
                    "objective_split_negative_frac": 0.0,
                    "error": "missing_features",
                }
            )
            continue

        w_prev_global: Optional[pd.Series] = None
        w_prev_by_regime: Dict[str, pd.Series] = {}

        valid_series_raw: List[pd.Series] = []
        valid_series_adj: List[pd.Series] = []
        split_obj_raw: List[float] = []
        split_obj_adj: List[float] = []
        turnover_days: List[float] = []
        switch_rates: List[float] = []
        fallback_fracs: List[float] = []
        n_used = 0

        for sid in split_ids:
            tr = train_by_split.get(int(sid))
            va = valid_by_split.get(int(sid))
            if tr is None or va is None or tr.empty or va.empty:
                continue

            tr2 = tr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            va2 = va.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cols = [c for c in tr2.columns if c in va2.columns]
            if len(cols) < 2:
                continue
            tr2 = tr2[cols]
            va2 = va2[cols]

            train_idx = pd.to_datetime(pd.Index(tr2.index)).sort_values()
            valid_idx = pd.to_datetime(pd.Index(va2.index)).sort_values()
            if int(len(train_idx)) < int(regime_min_days):
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
            if int(len(fit_df.index)) < int(regime_min_days):
                continue

            fit_out = fit_regime_allocations(
                fit_df,
                labels_train.reindex(fit_df.index),
                trading_days=int(trading_days),
                score_metric=str(score_metric),
                lambda_corr=float(lambda_corr),
                l2=float(l2),
                turnover_lambda=float(turnover_lambda),
                prev_global=w_prev_global,
                prev_by_regime=w_prev_by_regime,
                max_weight=float(max_weight),
                use_abs_corr=bool(use_abs_corr),
                backend=str(backend),
                solver=str(solver),
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
                max_weight=float(max_weight),
            )

            common = pd.to_datetime(pd.Index(daily_w.index)).intersection(pd.to_datetime(pd.Index(va2.index)))
            if common.empty:
                continue

            w_mat = daily_w.reindex(index=common, columns=cols).fillna(0.0)
            r_mat = va2.reindex(index=common, columns=cols).fillna(0.0)

            rr = (r_mat.to_numpy(dtype=float) * w_mat.to_numpy(dtype=float)).sum(axis=1)
            r_raw = pd.Series(rr, index=common)
            valid_series_raw.append(r_raw)

            to_s = _alpha_turnover_series(w_mat).reindex(index=common).fillna(0.0)
            turnover_days.append(float(to_s.mean()) if len(to_s.index) else 0.0)

            if cost_bps > 0.0:
                r_adj = r_raw - to_s * float(cost_bps) / 10000.0
            else:
                r_adj = r_raw
            valid_series_adj.append(r_adj)

            # Split-level stability diagnostics (used for optional knee/utility selection).
            try:
                if tm in {"annualized_return", "ann"}:
                    split_obj_raw.append(_annualized_return(float(r_raw.mean()), trading_days=int(trading_days)))
                    split_obj_adj.append(_annualized_return(float(r_adj.mean()), trading_days=int(trading_days)))
                else:
                    split_obj_raw.append(_information_ratio(r_raw.to_numpy(dtype=float), trading_days=int(trading_days)))
                    split_obj_adj.append(_information_ratio(r_adj.to_numpy(dtype=float), trading_days=int(trading_days)))
            except Exception:
                pass

            n_used += 1
            switch_rates.append(float(regime_stats(labels_valid.reindex(common)).get("switch_rate") or 0.0))
            fallback_fracs.append(float(daily_diag.get("fallback_frac") or 0.0))

        if not valid_series_raw:
            rows.append(
                {
                    "config_id": cfg.get("config_id"),
                    "mode": mode,
                    "window": int(window),
                    "buckets": int(buckets),
                    "smoothing": float(smoothing),
                    "tune_metric": str(tune_metric),
                    "turnover_cost_bps": float(cost_bps),
                    "valid_metric": float("-inf"),
                    "valid_metric_after_turnover_cost": float("-inf"),
                    "objective": float("-inf"),
                    "n_splits_used": int(n_used),
                    "n_valid_days": 0,
                    "alpha_weight_turnover_mean": float(np.mean(turnover_days)) if turnover_days else 0.0,
                    "turnover_cost_drag_bps_mean": float(np.mean(turnover_days) * float(cost_bps)) if turnover_days else 0.0,
                    "regime_switch_rate_mean": float(np.mean(switch_rates)) if switch_rates else 0.0,
                    "fallback_frac_mean": float(np.mean(fallback_fracs)) if fallback_fracs else 0.0,
                    "objective_split_std": 0.0,
                    "objective_split_min": 0.0,
                    "objective_split_negative_frac": 0.0,
                    "error": "no_valid_series",
                }
            )
            continue

        vv_raw = pd.concat(valid_series_raw).sort_index()
        vv_raw = vv_raw[~vv_raw.index.duplicated(keep="first")]
        arr_raw = vv_raw.to_numpy(dtype=float)
        mu_raw = float(np.mean(arr_raw)) if arr_raw.size else 0.0

        vv_adj = pd.concat(valid_series_adj).sort_index()
        vv_adj = vv_adj[~vv_adj.index.duplicated(keep="first")]
        arr_adj = vv_adj.to_numpy(dtype=float)
        mu_adj = float(np.mean(arr_adj)) if arr_adj.size else 0.0

        if tm in {"annualized_return", "ann"}:
            valid_metric = _annualized_return(mu_raw, trading_days=int(trading_days))
            obj_metric = _annualized_return(mu_adj, trading_days=int(trading_days))
        else:
            valid_metric = _information_ratio(arr_raw, trading_days=int(trading_days))
            obj_metric = _information_ratio(arr_adj, trading_days=int(trading_days))

        to_mean = float(np.mean(turnover_days)) if turnover_days else 0.0
        drag_bps = float(to_mean) * float(cost_bps)

        # Stability: variability across splits.
        obj_std = float(np.std(split_obj_adj, ddof=1)) if len(split_obj_adj) >= 2 else 0.0
        obj_min = float(np.min(split_obj_adj)) if split_obj_adj else 0.0
        obj_neg = float(np.mean([1.0 if float(x) < 0.0 else 0.0 for x in split_obj_adj])) if split_obj_adj else 0.0

        row = {
            "config_id": cfg.get("config_id"),
            "mode": mode,
            "window": int(window),
            "buckets": int(buckets),
            "smoothing": float(smoothing),
            "tune_metric": str(tune_metric),
            "turnover_cost_bps": float(cost_bps),
            "valid_metric": float(valid_metric),
            "valid_metric_after_turnover_cost": float(obj_metric),
            "objective": float(obj_metric),
            "n_splits_used": int(n_used),
            "n_valid_days": int(arr_raw.size),
            "alpha_weight_turnover_mean": float(to_mean),
            "turnover_cost_drag_bps_mean": float(drag_bps),
            "regime_switch_rate_mean": float(np.mean(switch_rates)) if switch_rates else 0.0,
            "fallback_frac_mean": float(np.mean(fallback_fracs)) if fallback_fracs else 0.0,
            "objective_split_std": float(obj_std),
            "objective_split_min": float(obj_min),
            "objective_split_negative_frac": float(obj_neg),
        }
        rows.append(row)

        if np.isfinite(obj_metric) and float(obj_metric) > best_obj:
            best_obj = float(obj_metric)
            best_row = dict(row)

    rows_sorted = sorted(rows, key=lambda r: float(r.get("objective") or float("-inf")), reverse=True)

    extra = list(pareto_metrics or [])
    if include_stability_objectives and str(selection_method).strip().lower() in {"knee", "utility"}:
        extra.extend(["objective_split_std", "objective_split_min"])

    pareto_meta = annotate_pareto(
        rows_sorted,
        objectives=_pareto_objectives(extra),
        pareto_key="is_pareto",
        rank_key="pareto_rank",
    )

    chosen, sel_meta = select_best_row(
        rows_sorted,
        objective_key="objective",
        constraints=constraints,
        prefer_pareto=bool(prefer_pareto),
        pareto_key="is_pareto",
        selection_method=str(selection_method),
        objectives=_pareto_objectives(extra),
        utility_weights=utility_weights,
    )
    best_params: Dict[str, Any] = {}
    if chosen:
        best_params = {
            "mode": str(chosen.get("mode") or "vol"),
            "window": int(chosen.get("window") or 20),
            "buckets": int(chosen.get("buckets") or 3),
            "smoothing": float(chosen.get("smoothing") or 0.0),
        }

    return {
        "enabled": True,
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
