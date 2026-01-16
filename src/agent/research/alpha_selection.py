"""agent.research.alpha_selection

Alpha selection utilities (consolidated).

This module centralizes:
- building return matrices (validation / OOS)
- correlation estimation
- diversified greedy selection (with optional hard constraints)
- validation meta-tuning of selection hyperparameters
- explainable selection reports

The project evolved across many P-stages; this file intentionally merges
previously scattered selection code to keep the repo compact and easier
to navigate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Presets / constraints
# ----------------------------


_ALPHA_SELECTION_PRESETS: Dict[str, Dict[str, Any]] = {
    # Keep defaults conservative so synthetic demos still select K names.
    "aggressive": {
        "max_pairwise_corr": None,
        "min_valid_ir": None,
        "min_valid_coverage": None,
        "max_total_cost_bps": None,
        "min_wf_test_ir_positive_frac": None,
    },
    "low_redundancy": {
        "max_pairwise_corr": 0.35,
    },
    "low_cost": {
        "max_total_cost_bps": 8.0,
    },
    "stable_generalization": {
        "min_valid_ir": 0.10,
        "min_valid_coverage": 0.60,
    },
}


def get_alpha_selection_preset(name: str) -> Dict[str, Any]:
    """Return a preset constraints dict (best-effort)."""

    key = str(name or "").strip().lower()
    return dict(_ALPHA_SELECTION_PRESETS.get(key) or {})


def merge_selection_constraints(*parts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge constraint dicts with right-most priority for explicit values."""

    out: Dict[str, Any] = {}
    for d in parts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            if v is None:
                continue
            out[str(k)] = v
    return out


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _alpha_id(alpha: Dict[str, Any]) -> str:
    return str(alpha.get("alpha_id") or alpha.get("alphaID") or alpha.get("id") or "")


def _total_cost_bps(metrics: Dict[str, Any]) -> float:
    """Best-effort total cost drag in bps (cost + borrow)."""

    try:
        attr = metrics.get("oos_cost_attribution") or {}
        if isinstance(attr, dict) and attr:
            cost = float(attr.get("cost_mean") or 0.0) + float(attr.get("borrow_mean") or 0.0)
        else:
            cost = float(metrics.get("cost_mean") or 0.0) + float(metrics.get("borrow_mean") or 0.0)
        return float(cost) * 10000.0
    except Exception:
        return float("nan")


# ----------------------------
# Return matrices
# ----------------------------


def build_return_matrix(
    alphas: List[Dict[str, Any]],
    *,
    wf_key: str,
    value_key: str = "net_return",
) -> pd.DataFrame:
    """Build a daily return matrix from per-alpha payloads.

    Expected structure:
      alpha['backtest_results']['walk_forward'][wf_key] = [{'datetime': ..., value_key: ...}, ...]

    Args:
      wf_key: "oos_daily" (test) or "valid_daily" (validation)

    Returns:
      DataFrame indexed by datetime, columns=alpha_id, values=value_key.
    """

    series_by_id: Dict[str, pd.Series] = {}
    for a in alphas or []:
        if not isinstance(a, dict):
            continue
        aid = _alpha_id(a)
        if not aid:
            continue

        m = a.get("backtest_results") or {}
        wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
        rows = (wf.get(str(wf_key)) or []) if isinstance(wf, dict) else []
        if not isinstance(rows, list) or not rows:
            continue

        try:
            idx = pd.to_datetime([r.get("datetime") for r in rows])
            vals = [float(r.get(value_key) or 0.0) for r in rows]
            s = pd.Series(vals, index=idx).sort_index()
            series_by_id[str(aid)] = s
        except Exception:
            continue

    if not series_by_id:
        return pd.DataFrame()

    df = pd.concat(series_by_id, axis=1)
    df.index.name = "datetime"
    return df.sort_index()


def build_oos_return_matrix(alphas: List[Dict[str, Any]], *, value_key: str = "net_return") -> pd.DataFrame:
    return build_return_matrix(alphas, wf_key="oos_daily", value_key=value_key)


def build_valid_return_matrix(alphas: List[Dict[str, Any]], *, value_key: str = "net_return") -> pd.DataFrame:
    return build_return_matrix(alphas, wf_key="valid_daily", value_key=value_key)


# ----------------------------
# Correlation and ensemble metrics
# ----------------------------


def compute_return_correlation(
    oos_returns: pd.DataFrame, *, min_periods: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlation matrix and pairwise overlap counts."""

    if oos_returns is None or oos_returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    mask = oos_returns.notna().astype(int)
    nobs = pd.DataFrame(mask.values.T @ mask.values, index=oos_returns.columns, columns=oos_returns.columns)

    corr = oos_returns.corr(min_periods=int(min_periods))
    return corr, nobs


def summarize_return_stream(returns: pd.Series, *, trading_days: int = 252) -> Dict[str, Any]:
    """Compute basic performance metrics from a daily return stream."""

    r = pd.Series(returns).dropna()
    arr = r.values.astype(float)
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
        "information_ratio": ir,
        "annualized_return": ann,
        "max_drawdown": mdd,
        "mean_daily_return": mu,
        "std_daily_return": sd,
        "n_obs": int(arr.size),
    }


def make_equal_weight_ensemble(
    oos_returns: pd.DataFrame,
    selected_ids: List[str],
    *,
    trading_days: int = 252,
) -> Dict[str, Any]:
    """Build an equal-weight ensemble of strategy return streams."""

    if oos_returns is None or oos_returns.empty:
        return {"enabled": False, "error": "Empty return matrix"}

    ids = [x for x in (selected_ids or []) if x in oos_returns.columns]
    if len(ids) == 0:
        return {"enabled": False, "error": "No selected ids available in return matrix"}

    sub = oos_returns[ids].copy()
    ens = sub.mean(axis=1, skipna=True)
    coverage = float(sub.notna().any(axis=1).mean()) if len(sub) else 0.0

    met = summarize_return_stream(ens, trading_days=int(trading_days))
    met["coverage"] = coverage
    met["n_alphas"] = int(len(ids))

    daily = pd.DataFrame({"datetime": ens.index.astype(str), "net_return": ens.values})
    return {
        "enabled": True,
        "method": "equal_weight_strategy_returns",
        "selected_alpha_ids": ids,
        "metrics": met,
        "daily": daily.to_dict(orient="records"),
    }


def correlation_summary(corr: pd.DataFrame, selected: List[str]) -> Dict[str, Any]:
    """Simple correlation summary for the selected set."""

    ids = [x for x in (selected or []) if x in corr.index]
    if len(ids) < 2:
        return {"n": int(len(ids)), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}

    sub = corr.loc[ids, ids].copy()
    arr = sub.values.astype(float)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    vals = np.abs(arr[mask])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"n": int(len(ids)), "avg_abs_corr": 0.0, "max_abs_corr": 0.0}

    return {
        "n": int(len(ids)),
        "avg_abs_corr": float(np.mean(vals)),
        "max_abs_corr": float(np.max(vals)),
    }


# ----------------------------
# Selection (diversified greedy)
# ----------------------------


def greedy_diversified_selection(
    *,
    scores: Dict[str, float],
    corr: pd.DataFrame,
    k: int,
    diversity_lambda: float = 0.0,
    use_abs_corr: bool = True,
    max_pairwise_corr: Optional[float] = None,
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Greedy diversified selector with an optional hard correlation cap.

    Step objective:
      pick argmax_i (score_i - lambda * avg_corr(i, selected))

    If max_pairwise_corr is set, candidates with max(abs corr to selected) above the cap
    are skipped.
    """

    if not scores or k <= 0:
        return [], [], []

    ordered = sorted(scores.keys(), key=lambda x: float(scores.get(x) or -np.inf), reverse=True)

    selected: List[str] = []
    table: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    # Pick the first by base score.
    first = ordered[0]
    selected.append(first)
    table.append(
        {
            "step": 1,
            "alpha_id": first,
            "base_score": float(scores.get(first) or 0.0),
            "avg_corr_to_selected": 0.0,
            "max_corr_to_selected": 0.0,
            "diversity_lambda": float(diversity_lambda),
            "diversity_score": float(scores.get(first) or 0.0),
        }
    )

    cap = None
    if max_pairwise_corr is not None:
        try:
            cap = float(max_pairwise_corr)
            if not np.isfinite(cap) or cap <= 0.0:
                cap = None
        except Exception:
            cap = None

    while len(selected) < int(k) and len(selected) < len(ordered):
        best_id = None
        best_div_score = float("-inf")
        best_avg_corr = 0.0
        best_max_corr = 0.0

        for cid in ordered:
            if cid in selected:
                continue

            corrs: List[float] = []
            for sid in selected:
                v = np.nan
                try:
                    if cid in corr.index and sid in corr.columns:
                        v = float(corr.loc[cid, sid])
                    elif sid in corr.index and cid in corr.columns:
                        v = float(corr.loc[sid, cid])
                except Exception:
                    v = np.nan

                if np.isfinite(v):
                    corrs.append(abs(v) if use_abs_corr else v)

            avg_corr = float(np.mean(corrs)) if corrs else 0.0
            max_corr = float(np.max(corrs)) if corrs else 0.0

            if cap is not None and max_corr > cap:
                rejected.append(
                    {
                        "alpha_id": str(cid),
                        "reason": "max_pairwise_corr",
                        "max_corr_to_selected": float(max_corr),
                        "cap": float(cap),
                    }
                )
                continue

            div_score = float(scores.get(cid) or 0.0) - float(diversity_lambda) * avg_corr
            if div_score > best_div_score:
                best_id = cid
                best_div_score = div_score
                best_avg_corr = avg_corr
                best_max_corr = max_corr

        if best_id is None:
            break

        selected.append(best_id)
        table.append(
            {
                "step": int(len(selected)),
                "alpha_id": best_id,
                "base_score": float(scores.get(best_id) or 0.0),
                "avg_corr_to_selected": float(best_avg_corr),
                "max_corr_to_selected": float(best_max_corr),
                "diversity_lambda": float(diversity_lambda),
                "diversity_score": float(best_div_score),
            }
        )

    return selected, table, rejected


# ----------------------------
# Selection meta-tuning (validation)
# ----------------------------


def _grid(values: Sequence[Any], default: Sequence[Any]) -> List[Any]:
    out = [v for v in values if v is not None]
    return list(out) if out else list(default)


def tune_diverse_selection(
    alphas: List[Dict[str, Any]],
    *,
    top_k: int,
    candidate_pool_grid: Sequence[int] = (10, 20),
    lambda_grid: Sequence[float] = (0.0, 0.2, 0.5, 0.8),
    top_k_grid: Optional[Sequence[int]] = None,
    use_abs_corr: bool = True,
    min_periods: int = 20,
    metric: str = "information_ratio",
    trading_days: int = 252,
    max_combos: int = 24,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Tune diversified selection hyperparams on validation return streams.

    constraints (validation-domain):
      - min_valid_ir
      - min_valid_coverage
      - max_pairwise_corr

    Returns a dict with:
      - enabled: bool
      - best: best row (dict)
      - results: list of rows (dict)
      - selected_alpha_ids: list[str]
    """

    top_k = max(1, int(top_k))
    td = max(1, int(trading_days))
    min_periods = max(2, int(min_periods))

    cons = dict(constraints or {})
    min_valid_ir = cons.get("min_valid_ir")
    min_valid_cov = cons.get("min_valid_coverage")
    max_pairwise_corr = cons.get("max_pairwise_corr")

    valid_mat = build_valid_return_matrix(alphas, value_key="net_return")
    if valid_mat is None or valid_mat.empty or valid_mat.shape[1] < 2:
        return {"enabled": False, "error": "Not enough validation return streams"}

    # Per-alpha base score computed on validation returns (no test leakage).
    base_scores: Dict[str, float] = {}
    per_alpha: Dict[str, Dict[str, Any]] = {}
    for cid in list(valid_mat.columns):
        s = valid_mat[cid]
        met = summarize_return_stream(s, trading_days=td)
        v = met.get("information_ratio")
        try:
            base_scores[str(cid)] = float(v) if v is not None else float("-inf")
        except Exception:
            base_scores[str(cid)] = float("-inf")

        cov = float(pd.Series(s).notna().mean())
        per_alpha[str(cid)] = {
            "valid_information_ratio": float(met.get("information_ratio") or 0.0) if isinstance(met, dict) else 0.0,
            "valid_annualized_return": float(met.get("annualized_return") or 0.0) if isinstance(met, dict) else 0.0,
            "valid_max_drawdown": float(met.get("max_drawdown") or 0.0) if isinstance(met, dict) else 0.0,
            "valid_coverage": float(cov),
            "valid_n_obs": int(met.get("n_obs") or 0) if isinstance(met, dict) else 0,
        }

    ranked_ids = sorted(base_scores.keys(), key=lambda k: float(base_scores.get(k) or float("-inf")), reverse=True)
    ranked_ids = [x for x in ranked_ids if np.isfinite(float(base_scores.get(x) or float("-inf")))]

    if min_valid_ir is not None:
        try:
            thr = float(min_valid_ir)
            ranked_ids = [i for i in ranked_ids if float(base_scores.get(i) or float("-inf")) >= thr]
        except Exception:
            pass

    if min_valid_cov is not None:
        try:
            thr = float(min_valid_cov)
            ranked_ids = [i for i in ranked_ids if float(per_alpha.get(i, {}).get("valid_coverage") or 0.0) >= thr]
        except Exception:
            pass

    if len(ranked_ids) < 2:
        return {"enabled": False, "error": "No valid candidates after validation constraints"}

    cand_grid = _grid(candidate_pool_grid, default=(min(10, len(ranked_ids)),))
    lam_grid = _grid(lambda_grid, default=(0.0,))
    k_grid = list(top_k_grid) if top_k_grid else [top_k]

    rows: List[Dict[str, Any]] = []
    combos: List[Tuple[int, float, int]] = []
    for p in cand_grid:
        for lam in lam_grid:
            for k in k_grid:
                combos.append((int(p), float(lam), int(k)))

    combos = combos[: max(1, int(max_combos))]

    for pool_n, lam, k in combos:
        k = max(1, int(k))
        pool_n = max(k, int(pool_n))
        pool_ids = ranked_ids[: min(pool_n, len(ranked_ids))]
        if len(pool_ids) < 2:
            continue

        sub = valid_mat[pool_ids].copy()
        corr, _nobs = compute_return_correlation(sub, min_periods=min_periods)

        scores = {i: float(base_scores.get(i) or 0.0) for i in pool_ids}
        selected_ids, table, rejected = greedy_diversified_selection(
            scores=scores,
            corr=corr,
            k=int(k),
            diversity_lambda=float(lam),
            use_abs_corr=bool(use_abs_corr),
            max_pairwise_corr=(_safe_float(max_pairwise_corr) if max_pairwise_corr is not None else None),
        )

        if not selected_ids:
            continue

        ens = make_equal_weight_ensemble(valid_mat, selected_ids, trading_days=td)
        met = (ens.get("metrics") or {}) if isinstance(ens, dict) else {}
        tune_val = met.get(metric)

        try:
            tune_val_f = float(tune_val)
        except Exception:
            tune_val_f = float("-inf")

        cs = correlation_summary(corr, selected_ids)

        rows.append(
            {
                "candidate_pool": int(pool_n),
                "diversity_lambda": float(lam),
                "top_k": int(k),
                "metric": str(metric),
                "valid_metric": tune_val_f,
                "valid_information_ratio": float(met.get("information_ratio") or 0.0),
                "valid_annualized_return": float(met.get("annualized_return") or 0.0),
                "valid_max_drawdown": float(met.get("max_drawdown") or 0.0),
                "valid_coverage": float(met.get("coverage") or 0.0),
                "selected_alpha_ids": list(selected_ids),
                "avg_abs_corr": float(cs.get("avg_abs_corr") or 0.0),
                "max_abs_corr": float(cs.get("max_abs_corr") or 0.0),
                "selection_table": table,
                "rejected": rejected,
            }
        )

    if not rows:
        return {"enabled": False, "error": "No valid selection configs evaluated"}

    rows_sorted = sorted(rows, key=lambda r: float(r.get("valid_metric") or float("-inf")), reverse=True)
    best = rows_sorted[0]

    return {
        "enabled": True,
        "method": "validation_meta_tune_diverse_greedy",
        "metric": str(metric),
        "n_candidates": int(valid_mat.shape[1]),
        "n_days": int(valid_mat.shape[0]),
        "min_periods": int(min_periods),
        "use_abs_corr": bool(use_abs_corr),
        "constraints": {
            "min_valid_ir": min_valid_ir,
            "min_valid_coverage": min_valid_cov,
            "max_pairwise_corr": max_pairwise_corr,
        },
        "per_alpha_valid": per_alpha,
        "best": best,
        "results": rows_sorted,
        "selected_alpha_ids": list(best.get("selected_alpha_ids") or []),
    }


# ----------------------------
# Explainable selection report
# ----------------------------


def _extract_candidate_rows(
    coded_alphas: List[Dict[str, Any]],
    *,
    per_alpha_valid: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pav = dict(per_alpha_valid or {})

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = _alpha_id(a)
        if not aid:
            continue

        m = a.get("backtest_results") or {}
        if not isinstance(m, dict):
            m = {}
        qg = m.get("quality_gate") if isinstance(m, dict) else None
        passed = True
        reasons: List[str] = []
        if isinstance(qg, dict):
            passed = bool(qg.get("passed", True))
            reasons = list(qg.get("reasons") or [])

        wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
        stab = (wf.get("stability") or {}) if isinstance(wf, dict) else {}

        v = pav.get(aid, {})

        rows.append(
            {
                "alpha_id": aid,
                "information_ratio": _safe_float(m.get("information_ratio")),
                "annualized_return": _safe_float(m.get("annualized_return")),
                "max_drawdown": _safe_float(m.get("max_drawdown")),
                "turnover_mean": _safe_float(m.get("turnover_mean")),
                "coverage_mean": _safe_float(m.get("coverage_mean")),
                "total_cost_bps": _safe_float(_total_cost_bps(m)),
                "wf_n_splits": _safe_float(stab.get("n_splits")),
                "wf_test_ir_mean": _safe_float(stab.get("test_ir_mean")),
                "wf_test_ir_std": _safe_float(stab.get("test_ir_std")),
                "wf_test_ir_positive_frac": _safe_float(stab.get("test_ir_positive_frac")),
                "valid_information_ratio": _safe_float(v.get("valid_information_ratio")),
                "valid_max_drawdown": _safe_float(v.get("valid_max_drawdown")),
                "valid_coverage": _safe_float(v.get("valid_coverage")),
                "gate_passed": bool(passed),
                "gate_reasons": ",".join([str(r) for r in reasons]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["information_ratio"] = pd.to_numeric(df["information_ratio"], errors="coerce")
    df = df.sort_values(by=["information_ratio"], ascending=False, na_position="last")
    return df


def _selected_ids_from_result(result: Dict[str, Any]) -> List[str]:
    sel = result.get("selection") if isinstance(result, dict) else None
    if isinstance(sel, dict):
        ids = sel.get("selected_alpha_ids")
        if isinstance(ids, list) and ids:
            return [str(x) for x in ids if str(x)]

    sota = result.get("sota_alphas") if isinstance(result, dict) else None
    out: List[str] = []
    if isinstance(sota, list):
        for a in sota:
            if isinstance(a, dict):
                aid = _alpha_id(a)
                if aid:
                    out.append(aid)
    return out


def _compute_corr_to_selected(
    *,
    coded_by_id: Dict[str, Dict[str, Any]],
    candidate_ids: List[str],
    selected_ids: List[str],
    min_periods: int,
) -> Tuple[Optional[pd.DataFrame], Dict[str, float], Dict[str, float]]:
    """Compute avg/max abs corr from each candidate to the selected set (test OOS streams)."""

    ids = [i for i in candidate_ids if i in coded_by_id]
    if not ids:
        return None, {}, {}

    alphas = [coded_by_id[i] for i in ids]
    mat = build_oos_return_matrix(alphas, value_key="net_return")
    if mat is None or mat.empty or mat.shape[1] < 2:
        return None, {}, {}

    corr, _nobs = compute_return_correlation(mat, min_periods=int(min_periods))
    sel = [s for s in selected_ids if s in corr.index]
    if not sel:
        return corr, {}, {}

    avg_map: Dict[str, float] = {}
    max_map: Dict[str, float] = {}
    for cid in list(corr.index):
        if cid not in sel:
            vals: List[float] = []
            for sid in sel:
                if cid == sid:
                    continue
                v = corr.loc[cid, sid] if sid in corr.columns else np.nan
                try:
                    v = float(v)
                except Exception:
                    v = np.nan
                if np.isfinite(v):
                    vals.append(abs(v))
            if vals:
                avg_map[str(cid)] = float(np.mean(vals))
                max_map[str(cid)] = float(np.max(vals))

    return corr, avg_map, max_map


def build_alpha_selection_report(
    *,
    config: Dict[str, Any],
    result: Dict[str, Any],
    top_n: int = 15,
    corr_pool_max: int = 40,
) -> Dict[str, Any]:
    """Build a structured report explaining the alpha selection decision."""

    cfg = dict(config or {})
    res = dict(result or {})

    coded = res.get("coded_alphas") if isinstance(res, dict) else None
    coded_alphas = [a for a in (coded or []) if isinstance(a, dict)]
    if not coded_alphas:
        return {"enabled": False, "error": "No coded_alphas"}

    selection = res.get("selection") if isinstance(res, dict) else None
    selection_tuning = res.get("selection_tuning") if isinstance(res, dict) else None

    pav = None
    if isinstance(selection_tuning, dict):
        pav = selection_tuning.get("per_alpha_valid")

    df = _extract_candidate_rows(coded_alphas, per_alpha_valid=(pav if isinstance(pav, dict) else None))
    coded_by_id = {str(_alpha_id(a)): a for a in coded_alphas if _alpha_id(a)}

    selected_ids = _selected_ids_from_result(res)

    method = str((selection or {}).get("method") or "score_rank") if isinstance(selection, dict) else "score_rank"
    domain = str((selection or {}).get("domain") or "test") if isinstance(selection, dict) else "test"

    # Infer the candidate pool size used for correlation diagnostics.
    pool_n = None
    if isinstance(selection, dict) and selection.get("candidate_pool") is not None:
        pool_n = int(selection.get("candidate_pool") or 0)
    if pool_n is None and isinstance(selection_tuning, dict):
        best = selection_tuning.get("best") if isinstance(selection_tuning, dict) else None
        if isinstance(best, dict) and best.get("candidate_pool") is not None:
            pool_n = int(best.get("candidate_pool") or 0)

    passed_df = df[df["gate_passed"] == True].copy() if (not df.empty and "gate_passed" in df.columns) else df
    passed_ids = [str(x) for x in passed_df["alpha_id"].astype(str).tolist()] if (not passed_df.empty) else []

    if pool_n is None or pool_n <= 0:
        pool_n = min(int(corr_pool_max), len(passed_ids))
    pool_n = max(len(selected_ids), min(int(pool_n), int(corr_pool_max)))
    corr_candidate_ids = list(dict.fromkeys(list(passed_ids[:pool_n]) + list(selected_ids)))

    min_periods = int(cfg.get("diverse_min_periods", 20) or 20)
    corr, avg_corr_map, max_corr_map = _compute_corr_to_selected(
        coded_by_id=coded_by_id,
        candidate_ids=corr_candidate_ids,
        selected_ids=selected_ids,
        min_periods=min_periods,
    )

    if not df.empty:
        df["avg_abs_corr_to_selected"] = df["alpha_id"].map(avg_corr_map)
        df["max_abs_corr_to_selected"] = df["alpha_id"].map(max_corr_map)

    diversity_lambda = 0.0
    if isinstance(selection, dict) and selection.get("diversity_lambda") is not None:
        diversity_lambda = float(selection.get("diversity_lambda") or 0.0)
    if diversity_lambda == 0.0 and isinstance(selection_tuning, dict):
        best = selection_tuning.get("best") if isinstance(selection_tuning, dict) else None
        if isinstance(best, dict) and best.get("diversity_lambda") is not None:
            diversity_lambda = float(best.get("diversity_lambda") or 0.0)

    if not df.empty:
        df["diversified_score"] = df["information_ratio"] - float(diversity_lambda) * df[
            "avg_abs_corr_to_selected"
        ].fillna(0.0)
        df["selected"] = df["alpha_id"].astype(str).isin(set(selected_ids))

    # Constraints (best-effort).
    preset = str(cfg.get("alpha_selection_preset") or "")
    constraints = {
        "max_pairwise_corr": cfg.get("alpha_selection_max_pairwise_corr"),
        "min_valid_ir": cfg.get("alpha_selection_min_valid_ir"),
        "min_valid_coverage": cfg.get("alpha_selection_min_valid_coverage"),
        "max_total_cost_bps": cfg.get("alpha_selection_max_total_cost_bps"),
        "min_wf_test_ir_positive_frac": cfg.get("alpha_selection_min_wf_test_ir_positive_frac"),
    }

    if not df.empty:
        viol: List[str] = []
        for _idx, r in df.iterrows():
            reasons: List[str] = []
            try:
                cap = constraints.get("max_pairwise_corr")
                if cap is not None and np.isfinite(float(cap)):
                    if _safe_float(r.get("max_abs_corr_to_selected")) > float(cap):
                        reasons.append("max_pairwise_corr")
            except Exception:
                pass

            try:
                cap = constraints.get("max_total_cost_bps")
                if cap is not None and np.isfinite(float(cap)):
                    if _safe_float(r.get("total_cost_bps")) > float(cap):
                        reasons.append("max_total_cost_bps")
            except Exception:
                pass

            try:
                thr = constraints.get("min_wf_test_ir_positive_frac")
                if thr is not None and np.isfinite(float(thr)):
                    if _safe_float(r.get("wf_test_ir_positive_frac"), 0.0) < float(thr):
                        reasons.append("min_wf_test_ir_positive_frac")
            except Exception:
                pass

            try:
                thr = constraints.get("min_valid_ir")
                if thr is not None and np.isfinite(float(thr)):
                    if _safe_float(r.get("valid_information_ratio"), float("-inf")) < float(thr):
                        reasons.append("min_valid_ir")
            except Exception:
                pass

            try:
                thr = constraints.get("min_valid_coverage")
                if thr is not None and np.isfinite(float(thr)):
                    if _safe_float(r.get("valid_coverage"), 0.0) < float(thr):
                        reasons.append("min_valid_coverage")
            except Exception:
                pass

            viol.append(",".join(reasons))

        df["constraint_violations"] = viol
        df["violates_constraints"] = df["constraint_violations"].astype(str).str.len() > 0

    top_candidates = df.head(int(top_n)).to_dict(orient="records") if not df.empty else []

    selected_rows = []
    if not df.empty and selected_ids:
        sel_df = df[df.get("selected") == True].copy()
        order = {sid: i for i, sid in enumerate(selected_ids)}
        sel_df["_ord"] = sel_df["alpha_id"].map(lambda x: order.get(str(x), 1_000_000))
        sel_df = sel_df.sort_values(by=["_ord"], ascending=True)
        selected_rows = sel_df.drop(columns=["_ord"]).to_dict(orient="records")

    gate_failed = []
    if not df.empty:
        gf = df[df.get("gate_passed") == False].copy()
        if not gf.empty:
            gate_failed = gf.head(10).to_dict(orient="records")

    excluded_by_constraints = []
    if not df.empty:
        ex = df[(df.get("selected") == False) & (df.get("violates_constraints") == True)].copy()
        if not ex.empty:
            ex = ex.sort_values(by=["information_ratio"], ascending=False, na_position="last")
            excluded_by_constraints = ex.head(10).to_dict(orient="records")

    redundant = []
    if not df.empty and selected_ids:
        rn = df[(df.get("gate_passed") == True) & (df.get("selected") == False)].copy()
        rn = rn.sort_values(by=["information_ratio"], ascending=False, na_position="last")
        redundant = rn.head(10).to_dict(orient="records")

    selection_table = None
    if isinstance(selection, dict) and isinstance(selection.get("selection_table"), list):
        selection_table = selection.get("selection_table")
    if selection_table is None and isinstance(selection_tuning, dict):
        best = selection_tuning.get("best") if isinstance(selection_tuning, dict) else None
        if isinstance(best, dict) and isinstance(best.get("selection_table"), list):
            selection_table = best.get("selection_table")

    rejected = None
    if isinstance(selection, dict) and isinstance(selection.get("rejected"), list):
        rejected = selection.get("rejected")
    if rejected is None and isinstance(selection_tuning, dict):
        best = selection_tuning.get("best") if isinstance(selection_tuning, dict) else None
        if isinstance(best, dict) and isinstance(best.get("rejected"), list):
            rejected = best.get("rejected")

    tuning_top = []
    if isinstance(selection_tuning, dict) and isinstance(selection_tuning.get("results"), list):
        tuning_top = list(selection_tuning.get("results") or [])[:5]

    gates_cfg = {
        "min_coverage": cfg.get("min_coverage"),
        "max_turnover": cfg.get("max_turnover"),
        "min_ir": cfg.get("min_ir"),
        "max_drawdown": cfg.get("max_dd"),
        "max_total_cost_bps": cfg.get("max_total_cost_bps"),
        "min_wf_splits": cfg.get("min_wf_splits"),
    }

    gate_passed_n = int(df["gate_passed"].sum()) if (df is not None and not df.empty and "gate_passed" in df.columns) else 0
    gate_failed_n = int((~df["gate_passed"]).sum()) if (df is not None and not df.empty and "gate_passed" in df.columns) else 0

    report: Dict[str, Any] = {
        "enabled": True,
        "summary": {
            "n_coded_alphas": int(len(coded_alphas)),
            "gate_passed": int(gate_passed_n),
            "gate_failed": int(gate_failed_n),
            "top_k_requested": int(cfg.get("top_k") or len(selected_ids) or 0),
            "selected_count": int(len(selected_ids)),
            "selection_domain": domain,
        },
        "quality_gates": {"config": gates_cfg},
        "selection_constraints": {"preset": preset, "constraints": constraints},
        "selection": {
            "method": method,
            "domain": domain,
            "metric": (selection or {}).get("metric") if isinstance(selection, dict) else "information_ratio",
            "diversity_lambda": float(diversity_lambda),
            "min_periods": int(min_periods),
            "selected_alpha_ids": list(selected_ids),
            "selection_table": selection_table,
            "rejected": rejected,
        },
        "selection_tuning": {
            "enabled": bool((selection_tuning or {}).get("enabled")) if isinstance(selection_tuning, dict) else False,
            "best": (selection_tuning or {}).get("best") if isinstance(selection_tuning, dict) else None,
            "top_results": tuning_top,
        },
        "diagnostics": {
            "top_candidates": top_candidates,
            "selected": selected_rows,
            "top_gate_failed": gate_failed,
            "top_excluded_by_constraints": excluded_by_constraints,
            "top_not_selected": redundant,
        },
        "ensemble": {
            "return_stream": res.get("ensemble"),
            "holdings": res.get("ensemble_holdings"),
            "holdings_allocated": res.get("ensemble_holdings_allocated"),
            "holdings_allocated_regime": res.get("ensemble_holdings_allocated_regime"),
        },
    }

    if isinstance(selection, dict) and isinstance(selection.get("correlation_summary"), dict):
        report["selection"]["correlation_summary"] = selection.get("correlation_summary")

    return report


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        v = float(x)
        if np.isfinite(v):
            return f"{v:.{nd}f}"
    except Exception:
        pass
    return ""


def render_alpha_selection_report_md(report: Dict[str, Any]) -> str:
    """Render a short markdown report."""

    if not isinstance(report, dict) or not bool(report.get("enabled")):
        return "# Alpha selection report\n\nNo selection report available.\n"

    summ = report.get("summary") or {}
    sel = report.get("selection") or {}
    diag = report.get("diagnostics") or {}
    tuning = report.get("selection_tuning") or {}
    cons = report.get("selection_constraints") or {}

    lines: List[str] = []
    lines.append("# Alpha selection report")
    lines.append("")
    lines.append(
        f"- coded_alphas: {int(summ.get('n_coded_alphas') or 0)} (gate_passed={int(summ.get('gate_passed') or 0)}, gate_failed={int(summ.get('gate_failed') or 0)})"
    )
    lines.append(
        f"- selected: {int(summ.get('selected_count') or 0)} (top_k_requested={int(summ.get('top_k_requested') or 0)})"
    )
    lines.append(f"- selection_domain: `{str(summ.get('selection_domain') or 'test')}`")
    lines.append(f"- method: `{sel.get('method')}`")
    lines.append(
        f"- diversity_lambda: {_fmt(sel.get('diversity_lambda'), 3)} (min_periods={int(sel.get('min_periods') or 0)})"
    )

    cs = sel.get("correlation_summary")
    if isinstance(cs, dict):
        lines.append(
            f"- selected corr: avg_abs={_fmt(cs.get('avg_abs_corr'), 3)}, max_abs={_fmt(cs.get('max_abs_corr'), 3)}"
        )

    preset = str(cons.get("preset") or "")
    constraints = cons.get("constraints") if isinstance(cons, dict) else None
    if preset or isinstance(constraints, dict):
        lines.append("")
        lines.append("## Selection constraints")
        if preset:
            lines.append(f"- preset: `{preset}`")
        if isinstance(constraints, dict):
            # Render only non-empty constraints.
            for k, v in constraints.items():
                if v is None or v == "":
                    continue
                lines.append(f"- {k}: {v}")

    lines.append("")

    # Selected table.
    lines.append("## Selected alphas")
    selected_rows = diag.get("selected") or []
    if isinstance(selected_rows, list) and selected_rows:
        lines.append("| alpha_id | IR | valid_IR | turnover | cost_bps | max_abs_corr_to_selected |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in selected_rows[:15]:
            lines.append(
                "| {aid} | {ir} | {vir} | {to} | {cb} | {c} |".format(
                    aid=str(r.get("alpha_id") or ""),
                    ir=_fmt(r.get("information_ratio"), 3),
                    vir=_fmt(r.get("valid_information_ratio"), 3),
                    to=_fmt(r.get("turnover_mean"), 3),
                    cb=_fmt(r.get("total_cost_bps"), 2),
                    c=_fmt(r.get("max_abs_corr_to_selected"), 3),
                )
            )
    else:
        lines.append("No selected alphas found in the result payload.")

    lines.append("")

    # Meta-tuning summary.
    if bool(tuning.get("enabled")):
        lines.append("## Selection meta-tuning (validation)")
        best = tuning.get("best")
        if isinstance(best, dict):
            lines.append(
                "- best: pool={pool}, lambda={lam}, top_k={k}, valid_metric={m:.3f}".format(
                    pool=int(best.get("candidate_pool") or 0),
                    lam=float(best.get("diversity_lambda") or 0.0),
                    k=int(best.get("top_k") or 0),
                    m=float(best.get("valid_metric") or float("nan")),
                )
            )
        top = tuning.get("top_results") or []
        if isinstance(top, list) and top:
            lines.append("")
            lines.append("Top configs:")
            lines.append("| rank | pool | lambda | top_k | valid_metric | avg_abs_corr |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for i, r in enumerate(top[:5], start=1):
                lines.append(
                    "| {i} | {p} | {lam:.2f} | {k} | {m:.3f} | {c:.3f} |".format(
                        i=int(i),
                        p=int(r.get("candidate_pool") or 0),
                        lam=float(r.get("diversity_lambda") or 0.0),
                        k=int(r.get("top_k") or 0),
                        m=float(r.get("valid_metric") or float("nan")),
                        c=float(r.get("avg_abs_corr") or 0.0),
                    )
                )
        lines.append("")

    # Greedy selection trace.
    table = sel.get("selection_table")
    if isinstance(table, list) and table:
        lines.append("## Greedy selection trace")
        lines.append("| step | alpha_id | base_score | max_corr_to_selected | diversity_score |")
        lines.append("|---:|---|---:|---:|---:|")
        for r in table[:20]:
            lines.append(
                "| {s} | {aid} | {b} | {c} | {d} |".format(
                    s=int(r.get("step") or 0),
                    aid=str(r.get("alpha_id") or ""),
                    b=_fmt(r.get("base_score"), 3),
                    c=_fmt(r.get("max_corr_to_selected"), 3),
                    d=_fmt(r.get("diversity_score"), 3),
                )
            )
        lines.append("")

    lines.append("## Why some candidates were not selected")

    ex = diag.get("top_excluded_by_constraints") or []
    if isinstance(ex, list) and ex:
        lines.append("### Excluded by selection constraints (examples)")
        lines.append("| alpha_id | IR | violations | max_abs_corr | cost_bps | valid_IR |")
        lines.append("|---|---:|---|---:|---:|---:|")
        for r in ex[:10]:
            lines.append(
                "| {aid} | {ir} | {v} | {c} | {cb} | {vir} |".format(
                    aid=str(r.get("alpha_id") or ""),
                    ir=_fmt(r.get("information_ratio"), 3),
                    v=str(r.get("constraint_violations") or ""),
                    c=_fmt(r.get("max_abs_corr_to_selected"), 3),
                    cb=_fmt(r.get("total_cost_bps"), 2),
                    vir=_fmt(r.get("valid_information_ratio"), 3),
                )
            )
        lines.append("")

    nf = diag.get("top_gate_failed") or []
    if isinstance(nf, list) and nf:
        lines.append("### Failed quality gates (examples)")
        lines.append("| alpha_id | IR | reasons |")
        lines.append("|---|---:|---|")
        for r in nf[:10]:
            lines.append(
                "| {aid} | {ir} | {reasons} |".format(
                    aid=str(r.get("alpha_id") or ""),
                    ir=_fmt(r.get("information_ratio"), 3),
                    reasons=str(r.get("gate_reasons") or ""),
                )
            )
        lines.append("")

    nn = diag.get("top_not_selected") or []
    if isinstance(nn, list) and nn:
        lines.append("### Passed gates but excluded (often redundancy)")
        lines.append("| alpha_id | IR | max_abs_corr_to_selected | diversified_score |")
        lines.append("|---|---:|---:|---:|")
        for r in nn[:10]:
            lines.append(
                "| {aid} | {ir} | {c} | {ds} |".format(
                    aid=str(r.get("alpha_id") or ""),
                    ir=_fmt(r.get("information_ratio"), 3),
                    c=_fmt(r.get("max_abs_corr_to_selected"), 3),
                    ds=_fmt(r.get("diversified_score"), 3),
                )
            )
        lines.append("")

    lines.append("## Artifacts")
    lines.append("- `alpha_selection_report.json`")
    lines.append("- `ALPHA_SELECTION_REPORT.md`")
    lines.append("- `alpha_selection_top_candidates.csv`")
    lines.append("")

    return "\n".join(lines)
