"""agent.research.optimizer

Dependency-light portfolio construction utilities.

This module provides:
- a heuristic candidate selector (top/bottom)
- a ridge-style "optimizer" construction (no extra deps)
- an optional constrained QP backend (cvxpy) with hard constraints

The constrained backend is intentionally optional and will gracefully
fallback to the ridge optimizer when cvxpy is not installed or when the
problem is infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent.research.neutralize import clip_weights, rescale_long_short


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer hyperparameters.

    Lambdas are dimensionless and should be tuned for your universe.
    """

    l2_lambda: float = 1.0
    turnover_lambda: float = 10.0
    exposure_lambda: float = 0.0
    max_iter: int = 2


@dataclass(frozen=True)
class OptimizerCostModel:
    """Optional real-world cost terms used inside constrained optimization.

    All costs are expressed in return units per rebalance step.
    """

    cost_aversion: float = 0.0
    trade_cost: Optional[pd.Series] = None  # per |Δw|
    borrow_cost: Optional[pd.Series] = None  # per |short_w|
    impact_coeff: Optional[pd.Series] = None  # coeff for |Δw|^(1+impact_exponent)
    impact_exponent: float = 0.5
    max_trade_abs: Optional[pd.Series] = None  # per-name bound on |Δw|
    exposure_slack_lambda: float = 0.0

    # Risk model terms used inside the optimizer objective.
    risk_aversion: float = 0.0
    risk_var: Optional[pd.Series] = None  # diag variance proxy (annualized)
    risk_model: str = "diag"  # diag | factor
    factor_loadings: Optional[pd.DataFrame] = None  # instrument x factor
    factor_cov: Optional[np.ndarray] = None  # k x k (annualized)
    idio_var: Optional[pd.Series] = None  # diag idiosyncratic variance (annualized)
    risk_meta: Optional[Dict[str, Any]] = None


def qp_feasibility_precheck(
    *,
    long_names: Sequence[str],
    short_names: Sequence[str],
    w_target: pd.Series,
    gross_long: float,
    gross_short: float,
    max_abs_weight: float,
    turnover_cap: float,
    exposures: Optional[pd.DataFrame],
    enforce_exposure_neutrality: bool,
    cost_model: Optional[OptimizerCostModel] = None,
) -> Dict[str, Any]:
    """Fast feasibility checks for the constrained optimizer.

    This is intentionally conservative: it can prove some problems are infeasible,
    but it won't prove feasibility.
    """

    long_set = set([str(x) for x in long_names])
    short_set = set([str(x) for x in short_names])
    cand = pd.Index(sorted(set(long_set) | set(short_set)))

    out: Dict[str, Any] = {
        "passed": True,
        "reasons": [],
        "warnings": [],
        "suggestions": [],
        "n_candidates": int(len(cand)),
        "n_long": int(len(long_set)),
        "n_short": int(len(short_set)),
    }

    if len(cand) < 4:
        out["reasons"].append("too_few_names")
        out["suggestions"].append(
            {
                "parameter": "candidate_set",
                "message": "Increase candidate count (e.g., larger max_names_per_side or smaller n_quantiles).",
            }
        )

    cap = float(max_abs_weight)
    if cap > 0.0:
        need_cap_long = float(gross_long) / max(1.0, float(len(long_set)))
        need_cap_short = float(gross_short) / max(1.0, float(len(short_set)))
        need_cap = float(max(need_cap_long, need_cap_short))

        if (len(long_set) * cap) + 1e-12 < float(gross_long):
            out["reasons"].append("cap_long_too_tight")
        if (len(short_set) * cap) + 1e-12 < float(gross_short):
            out["reasons"].append("cap_short_too_tight")

        if "cap_long_too_tight" in out["reasons"] or "cap_short_too_tight" in out["reasons"]:
            out["suggestions"].append(
                {
                    "parameter": "max_abs_weight",
                    "suggested_min": float(need_cap),
                    "message": "Increase max_abs_weight or expand candidate count to satisfy gross constraints.",
                }
            )

    wt = w_target.reindex(cand).astype(float).fillna(0.0)
    long_mask = pd.Index([str(x) for x in cand]).isin(long_set)
    short_mask = pd.Index([str(x) for x in cand]).isin(short_set)

    wt_long = wt[long_mask]
    wt_short = wt[short_mask]

    # Lower bound on the required L1 change to satisfy signs + gross constraints.
    wt_long_pos_sum = float(wt_long.clip(lower=0.0).sum())
    wt_short_abs_sum = float((-wt_short.clip(upper=0.0)).sum())
    sign_violation = float((-wt_long.clip(upper=0.0)).sum() + (wt_short.clip(lower=0.0)).sum())
    lb_l1 = sign_violation + abs(float(gross_long) - wt_long_pos_sum) + abs(float(gross_short) - wt_short_abs_sum)
    turnover_lb = 0.5 * float(lb_l1)
    out["turnover_lb_to_meet_gross"] = float(turnover_lb)

    tcap = float(turnover_cap)
    if tcap > 0.0 and turnover_lb > tcap + 1e-9:
        out["reasons"].append("turnover_cap_too_small")
        out["suggestions"].append(
            {
                "parameter": "optimizer_turnover_cap",
                "suggested_min": float(turnover_lb),
                "message": "Increase turnover_cap or relax constraints (caps/neutrality) to make the problem reachable.",
            }
        )

    cm = cost_model or OptimizerCostModel()
    max_u = getattr(cm, "max_trade_abs", None)
    if max_u is not None:
        mu = max_u.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
        mu = mu.fillna(np.inf).clip(lower=0.0)
        mv = mu.to_numpy(dtype=float)
        if np.isfinite(mv).any():
            sum_max_u = float(np.sum(np.where(np.isfinite(mv), mv, 1e9)))
            if turnover_lb > 0.5 * sum_max_u + 1e-9:
                out["reasons"].append("trade_bounds_too_tight_for_required_change")
                out["suggestions"].append(
                    {
                        "parameter": "optimizer_enforce_participation",
                        "suggested": False,
                        "message": "Trade bounds are too tight; disable participation bounds or increase impact_max_participation.",
                    }
                )

        # Gross reachability under per-name trade bounds.
        def _reachable_sum_range(side: str) -> Tuple[float, float]:
            if side == "long":
                names = [x for x in cand.astype(str) if x in long_set]
                lo_cap, hi_cap = 0.0, (cap if cap > 0.0 else np.inf)
            else:
                names = [x for x in cand.astype(str) if x in short_set]
                lo_cap, hi_cap = (-(cap if cap > 0.0 else np.inf)), 0.0

            if not names:
                return 0.0, 0.0

            wt_s = wt.reindex(names).astype(float).fillna(0.0)
            mu_s = mu.reindex(names).astype(float).fillna(np.inf)

            lo = np.maximum(lo_cap, (wt_s - mu_s).to_numpy(dtype=float))
            hi = np.minimum(hi_cap, (wt_s + mu_s).to_numpy(dtype=float))
            return float(np.sum(lo)), float(np.sum(hi))

        loL, hiL = _reachable_sum_range("long")
        loS, hiS = _reachable_sum_range("short")
        out["reachable_long_sum"] = [float(loL), float(hiL)]
        out["reachable_short_sum"] = [float(loS), float(hiS)]

        if float(gross_long) < loL - 1e-9 or float(gross_long) > hiL + 1e-9:
            out["reasons"].append("gross_long_unreachable_under_trade_bounds")
            out["suggestions"].append(
                {
                    "parameter": "impact_max_participation",
                    "message": "Increase impact_max_participation (or reduce portfolio_notional) so long gross becomes reachable.",
                }
            )
        target_short_sum = -float(gross_short)
        if target_short_sum < loS - 1e-9 or target_short_sum > hiS + 1e-9:
            out["reasons"].append("gross_short_unreachable_under_trade_bounds")
            out["suggestions"].append(
                {
                    "parameter": "impact_max_participation",
                    "message": "Increase impact_max_participation (or reduce portfolio_notional) so short gross becomes reachable.",
                }
            )

    # Degrees-of-freedom sanity check for strict exposure neutrality.
    if bool(enforce_exposure_neutrality) and exposures is not None and not exposures.empty:
        X = _prepare_exposures(exposures, index=cand)
        k = int(X.shape[1])
        dof = int(len(cand) - (2 + k))
        out["neutrality_k"] = k
        out["neutrality_dof"] = dof
        if dof <= 0:
            out["warnings"].append("very_low_dof_for_hard_neutrality")
            out["suggestions"].append(
                {
                    "parameter": "optimizer_exposure_slack_lambda",
                    "suggested_min": 1.0,
                    "message": "Hard neutrality has low degrees-of-freedom; enable exposure slack or relax neutrality flags.",
                }
            )

    out["passed"] = len(out["reasons"]) == 0
    return out


def _build_constraint_summary(
    *,
    w: pd.Series,
    w_target: pd.Series,
    gross_long: float,
    gross_short: float,
    max_abs_weight: float,
    turnover_cap: float,
    max_trade_abs: Optional[pd.Series],
    exposures: Optional[pd.DataFrame],
    slack: Optional[np.ndarray] = None,
    precheck: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wt = w_target.reindex(w.index).astype(float).fillna(0.0)
    wv = w.reindex(w.index).astype(float).fillna(0.0)
    delta = (wv - wt)
    abs_delta = delta.abs()

    t = 0.5 * float(abs_delta.sum())

    out: Dict[str, Any] = {
        "gross": {
            "gross_long": float(wv.clip(lower=0.0).sum()),
            "gross_short": float((-wv.clip(upper=0.0)).sum()),
            "gross_long_resid": float(wv.clip(lower=0.0).sum() - float(gross_long)),
            "gross_short_resid": float((-wv.clip(upper=0.0)).sum() - float(gross_short)),
        },
        "turnover": {
            "turnover_to_target": float(t),
            "turnover_cap": float(turnover_cap),
            "binding": bool(float(turnover_cap) > 0.0 and t >= 0.99 * float(turnover_cap)),
        },
        "max_abs_weight": {
            "cap": float(max_abs_weight),
            "enabled": bool(float(max_abs_weight) > 0.0),
        },
        "participation": {
            "enabled": bool(max_trade_abs is not None),
        },
        "exposure": {
            "enabled": bool(exposures is not None and not exposures.empty),
        },
        "warnings": list((precheck or {}).get("warnings") or []),
    }

    cap = float(max_abs_weight)
    if cap > 0.0:
        aw = np.abs(wv.to_numpy(dtype=float))
        out["max_abs_weight"].update(
            {
                "max_used": float(np.nanmax(aw)) if aw.size else 0.0,
                "at_cap_frac": float(np.mean(aw >= (0.999 * cap))) if aw.size else 0.0,
                "binding": bool(aw.size and float(np.nanmax(aw)) >= 0.99 * cap),
            }
        )

    if max_trade_abs is not None:
        mu = max_trade_abs.reindex(w.index).astype(float).replace([np.inf, -np.inf], np.nan)
        mu = mu.fillna(np.inf).clip(lower=0.0)
        muv = mu.to_numpy(dtype=float)
        dv = abs_delta.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(muv > 0.0, dv / muv, 0.0)
        out["participation"].update(
            {
                "max_ratio": float(np.nanmax(ratio)) if ratio.size else 0.0,
                "binding_frac": float(np.mean(ratio >= 0.999)) if ratio.size else 0.0,
                "violation_max": float(np.nanmax(dv - muv)) if dv.size else 0.0,
                "binding": bool(ratio.size and float(np.nanmax(ratio)) >= 0.99),
            }
        )

    if exposures is not None and not exposures.empty:
        X = _prepare_exposures(exposures, index=w.index)
        if not X.empty:
            ev = pd.Series((X.to_numpy(dtype=float).T @ wv.to_numpy(dtype=float)), index=X.columns, dtype=float)
            ev = ev.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            top = ev.abs().sort_values(ascending=False).head(10)
            out["exposure"].update(
                {
                    "max_abs": float(ev.abs().max()),
                    "l2": float(np.sqrt(float((ev * ev).sum()))),
                    "top_abs": [(str(k), float(ev.loc[k])) for k in top.index],
                }
            )
    else:
        out["exposure"].update({"enabled": False})

    if slack is not None:
        sv = np.asarray(slack, dtype=float).reshape(-1)
        if sv.size:
            out["exposure"].update(
                {
                    "slack_max_abs": float(np.nanmax(np.abs(sv))),
                    "slack_l2": float(np.sqrt(float(np.sum(sv * sv)))),
                }
            )

    bindings = []
    if out.get("turnover", {}).get("binding"):
        bindings.append("turnover_cap")
    if out.get("max_abs_weight", {}).get("binding"):
        bindings.append("max_abs_weight")
    if out.get("participation", {}).get("binding"):
        bindings.append("participation_cap")
    out["binding_constraints"] = bindings
    return out


def _split_quantiles(x: pd.Series, n_quantiles: int) -> Optional[pd.Series]:
    r = x.rank(method="first")
    try:
        q = pd.qcut(r, q=int(n_quantiles), labels=False, duplicates="drop")
    except Exception:
        return None
    if q is None:
        return None
    if q.nunique(dropna=True) < 2:
        return None
    return q.astype("Int64")


def select_long_short_candidates(
    scores: pd.Series,
    *,
    n_quantiles: int,
    max_names_per_side: int = 0,
    shortable: Optional[pd.Series] = None,
) -> Optional[Tuple[List[str], List[str]]]:
    """Select long/short candidate sets from a score series."""

    x = scores.dropna()
    if x.empty:
        return None

    max_names_per_side = int(max_names_per_side)

    if max_names_per_side > 0:
        xs = x.sort_values(kind="mergesort")
        if xs.size < (2 * max_names_per_side):
            return None

        if shortable is not None:
            m = shortable.reindex(xs.index).fillna(False).astype(bool)
            xs_short = xs[m]
            if xs_short.size < max_names_per_side:
                return None
            short = xs_short.iloc[:max_names_per_side]
        else:
            short = xs.iloc[:max_names_per_side]

        long = xs.iloc[-max_names_per_side:]
    else:
        q = _split_quantiles(x, n_quantiles=n_quantiles)
        if q is None:
            return None
        top_q = int(q.max())
        bot_q = int(q.min())
        long = x[q == top_q]
        short = x[q == bot_q]
        if shortable is not None:
            m = shortable.reindex(short.index).fillna(False).astype(bool)
            short = short[m]

    if long.empty or short.empty:
        return None

    return (list(long.index.astype(str)), list(short.index.astype(str)))


def _prepare_exposures(exposures: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    X = exposures.reindex(index)
    if X is None or X.empty:
        return pd.DataFrame(index=index)
    X = X.copy()
    for c in X.columns:
        col = X[c].astype(float)
        med = float(col.median(skipna=True)) if col.notna().any() else 0.0
        col = col.replace([np.inf, -np.inf], np.nan).fillna(med)
        mu = float(col.mean())
        sd = float(col.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0.0:
            sd = 1.0
        X[c] = (col - mu) / sd
    return X


def _solve_ridge(
    scores: pd.Series,
    *,
    w_target: pd.Series,
    exposures: Optional[pd.DataFrame],
    cfg: OptimizerConfig,
    diag_penalty: Optional[pd.Series] = None,
    low_rank_extra: Optional[np.ndarray] = None,
) -> pd.Series:
    """Solve a ridge problem with optional diagonal and low-rank quadratic penalties.

    We solve for w in:
      (D + A A^T) w = b
    using a Woodbury-style formula, where D is diagonal and A is low-rank.
    """

    idx = scores.index
    s = scores.reindex(idx).astype(float).fillna(0.0)
    wt = w_target.reindex(idx).astype(float).fillna(0.0)

    l2 = float(cfg.l2_lambda)
    tlam = float(cfg.turnover_lambda)
    base = float(max(1e-12, l2 + tlam))

    b = (s + tlam * wt).to_numpy(dtype=float)

    d = np.full(int(len(idx)), base, dtype=float)
    if diag_penalty is not None:
        dp = diag_penalty.reindex(idx).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dp = dp.clip(lower=0.0).to_numpy(dtype=float)
        d = d + dp

    d = np.maximum(d, 1e-12)
    dinv = 1.0 / d
    dinv_b = b * dinv

    A_parts = []

    lam = float(cfg.exposure_lambda)
    if exposures is not None and not exposures.empty and lam > 0.0:
        Z = _prepare_exposures(exposures, index=idx)
        if not Z.empty:
            A_parts.append(np.sqrt(lam) * Z.to_numpy(dtype=float))

    if low_rank_extra is not None:
        A2 = np.asarray(low_rank_extra, dtype=float)
        if A2.ndim == 2 and A2.shape[0] == len(idx) and A2.shape[1] > 0:
            A_parts.append(A2)

    if not A_parts:
        return pd.Series(dinv_b, index=idx, dtype=float)

    A = np.concatenate(A_parts, axis=1)
    k = int(A.shape[1])
    if k <= 0:
        return pd.Series(dinv_b, index=idx, dtype=float)

    # Woodbury: (D + A A^T)^{-1} b
    dinv_A = A * dinv.reshape(-1, 1)
    M = np.eye(k, dtype=float) + (A.T @ dinv_A)
    rhs = A.T @ dinv_b

    try:
        y = np.linalg.solve(M, rhs)
    except Exception:
        y = np.linalg.lstsq(M, rhs, rcond=None)[0]

    w = dinv_b - (dinv_A @ y)
    return pd.Series(w, index=idx, dtype=float)



def optimize_long_short_weights(
    scores: pd.Series,
    *,
    long_names: Sequence[str],
    short_names: Sequence[str],
    w_target: pd.Series,
    exposures: Optional[pd.DataFrame],
    cfg: OptimizerConfig,
    gross_long: float = 0.5,
    gross_short: float = 0.5,
    max_abs_weight: float = 0.0,
    cost_model: Optional[OptimizerCostModel] = None,
) -> Optional[pd.Series]:
    """Construct a long/short portfolio using a ridge-style optimizer."""

    long_names = list(dict.fromkeys([str(x) for x in long_names]))
    short_names = list(dict.fromkeys([str(x) for x in short_names]))
    if not long_names or not short_names:
        return None

    cand = pd.Index(sorted(set(long_names) | set(short_names)))
    s = scores.reindex(cand).astype(float)
    if s.isna().all():
        return None

    wt = w_target.reindex(cand).astype(float).fillna(0.0)
    cm = cost_model or OptimizerCostModel()
    rav = float(getattr(cm, "risk_aversion", 0.0) or 0.0)

    diag_penalty = None
    low_rank_extra = None
    used_factor_risk = False

    # Prefer a factor risk model when provided (Sigma = BFB^T + D).
    if rav > 0.0 and getattr(cm, "factor_cov", None) is not None and getattr(cm, "factor_loadings", None) is not None:
        Bdf = cm.factor_loadings.reindex(cand)
        if Bdf is not None and not Bdf.empty:
            Bz = _prepare_exposures(Bdf, index=cand)
            F = np.asarray(cm.factor_cov, dtype=float)
            if F.ndim == 0:
                F = np.array([[float(F)]], dtype=float)
            if F.ndim == 1:
                F = np.diag(F.astype(float))
            F = 0.5 * (F + F.T)

            # Robust PSD square-root for the low-rank factor term.
            try:
                vals, vecs = np.linalg.eigh(F)
                vals = np.clip(vals, 0.0, None)
                sqrtF = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            except Exception:
                sqrtF = np.eye(int(Bz.shape[1]), dtype=float)

            low_rank_extra = np.sqrt(rav) * (Bz.to_numpy(dtype=float) @ sqrtF)

            idv = getattr(cm, "idio_var", None)
            if idv is not None:
                d = idv.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
                med = float(np.nanmedian(d.to_numpy(dtype=float))) if np.isfinite(d.to_numpy(dtype=float)).any() else 0.0
                d = d.fillna(med).clip(lower=0.0)
                diag_penalty = rav * d

            used_factor_risk = True

    # Fallback: simple diagonal risk penalty sum(var_i * w_i^2).
    if (not used_factor_risk) and rav > 0.0 and getattr(cm, "risk_var", None) is not None:
        rv = cm.risk_var.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
        med = float(np.nanmedian(rv.to_numpy(dtype=float))) if np.isfinite(rv.to_numpy(dtype=float)).any() else 0.0
        rv = rv.fillna(med).clip(lower=0.0)
        diag_penalty = rav * rv

    w = _solve_ridge(
        s.fillna(0.0),
        w_target=wt,
        exposures=exposures,
        cfg=cfg,
        diag_penalty=diag_penalty,
        low_rank_extra=low_rank_extra,
    )

    w_out = pd.Series(0.0, index=cand, dtype=float)
    w_out.loc[long_names] = np.maximum(0.0, w.reindex(long_names).to_numpy(dtype=float))
    w_out.loc[short_names] = np.minimum(0.0, w.reindex(short_names).to_numpy(dtype=float))

    # A couple of simple projection steps to keep outputs well-formed.
    iters = max(1, int(cfg.max_iter))
    for _ in range(iters):
        w2 = rescale_long_short(w_out, gross_long=gross_long, gross_short=gross_short, scale_up=True)
        if w2 is None:
            return None
        w_out = w2
        if float(max_abs_weight) > 0.0:
            w_out = clip_weights(w_out, max_abs_weight=float(max_abs_weight))
            w3 = rescale_long_short(w_out, gross_long=gross_long, gross_short=gross_short, scale_up=False)
            if w3 is None:
                break
            w_out = w3
        # Re-enforce side signs.
        w_out.loc[long_names] = np.maximum(0.0, w_out.reindex(long_names).to_numpy(dtype=float))
        w_out.loc[short_names] = np.minimum(0.0, w_out.reindex(short_names).to_numpy(dtype=float))

    if (w_out > 0.0).sum() == 0 or (w_out < 0.0).sum() == 0:
        return None
    return w_out


def _cost_aware_adjust_scores(
    scores: pd.Series,
    *,
    cand: pd.Index,
    short_set: set,
    cost_model: Optional[OptimizerCostModel],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Heuristic score adjustments used by the ridge fallback."""

    cm = cost_model or OptimizerCostModel()
    av = float(getattr(cm, "cost_aversion", 0.0) or 0.0)
    s = scores.reindex(cand).astype(float).fillna(0.0).copy()
    meta: Dict[str, Any] = {}
    if av <= 0.0:
        return s, meta

    if getattr(cm, "trade_cost", None) is not None:
        tc = cm.trade_cost.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        tc = tc.clip(lower=0.0)
        shrink = 1.0 / (1.0 + av * tc)
        s = s * shrink
        meta["trade_score_shrink_mean"] = float(np.mean(shrink.to_numpy(dtype=float)))

    if getattr(cm, "borrow_cost", None) is not None and short_set:
        bc = cm.borrow_cost.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        m = pd.Index([str(x) for x in cand]).isin(set([str(x) for x in short_set]))
        shift = bc.where(m, 0.0)
        s = s + (av * shift)
        meta["borrow_score_shift_mean"] = float(np.mean(shift.to_numpy(dtype=float)))

    meta["cost_aversion"] = float(av)
    return s, meta


def _apply_ridge_trade_limits(
    w: pd.Series,
    *,
    w_target: pd.Series,
    long_names: Sequence[str],
    short_names: Sequence[str],
    gross_long: float,
    gross_short: float,
    max_abs_weight: float,
    turnover_cap: float,
    cost_model: Optional[OptimizerCostModel],
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """Best-effort post-step limits for the ridge fallback."""

    meta: Dict[str, Any] = {}
    cm = cost_model or OptimizerCostModel()

    cand = w.index
    wt = w_target.reindex(cand).astype(float).fillna(0.0)
    w_out = w.astype(float).copy()

    mu = getattr(cm, "max_trade_abs", None)
    if mu is not None:
        m = mu.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
        m = m.fillna(np.inf).clip(lower=0.0)
        mv = m.to_numpy(dtype=float)
        dv = (w_out - wt).to_numpy(dtype=float)
        if np.isfinite(mv).any():
            dv = np.sign(dv) * np.minimum(np.abs(dv), np.where(np.isfinite(mv), mv, 1e18))
            w_out = wt + pd.Series(dv, index=cand, dtype=float)
            meta["trade_bounds_applied"] = True

    tcap = float(turnover_cap)
    if tcap > 0.0:
        dv = (w_out - wt).to_numpy(dtype=float)
        t = 0.5 * float(np.abs(dv).sum())
        if t > tcap + 1e-12:
            k = tcap / t
            w_out = wt + (w_out - wt) * float(k)
            meta["turnover_cap_applied"] = True
            meta["turnover_scale"] = float(k)

    # Keep the output well-formed for the downstream pipeline.
    long_set = set([str(x) for x in long_names])
    short_set = set([str(x) for x in short_names])
    long_idx = [x for x in cand.astype(str) if x in long_set]
    short_idx = [x for x in cand.astype(str) if x in short_set]
    w_out.loc[long_idx] = np.maximum(0.0, w_out.reindex(long_idx).to_numpy(dtype=float))
    w_out.loc[short_idx] = np.minimum(0.0, w_out.reindex(short_idx).to_numpy(dtype=float))

    w2 = rescale_long_short(w_out, gross_long=float(gross_long), gross_short=float(gross_short), scale_up=False)
    if w2 is None:
        return None, meta
    w_out = w2

    cap = float(max_abs_weight)
    if cap > 0.0:
        w_out = clip_weights(w_out, max_abs_weight=cap)
        w3 = rescale_long_short(w_out, gross_long=float(gross_long), gross_short=float(gross_short), scale_up=False)
        w_out = w3 if w3 is not None else w_out

    meta["turnover_to_target"] = float(0.5 * float(np.abs((w_out - wt)).sum()))
    meta["gross_long"] = float(w_out[w_out > 0.0].sum())
    meta["gross_short"] = float((-w_out[w_out < 0.0]).sum())
    return w_out, meta


def _cvxpy_available() -> bool:
    try:
        import cvxpy as _  # noqa: F401

        return True
    except Exception:
        return False


def _solve_qp_cvxpy(
    scores: pd.Series,
    *,
    long_names: Sequence[str],
    short_names: Sequence[str],
    w_target: pd.Series,
    exposures: Optional[pd.DataFrame],
    cfg: OptimizerConfig,
    gross_long: float,
    gross_short: float,
    max_abs_weight: float,
    turnover_cap: float,
    enforce_exposure_neutrality: bool,
    solver: Optional[str] = None,
    cost_model: Optional[OptimizerCostModel] = None,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """Solve a constrained long/short construction using cvxpy."""

    meta: Dict[str, Any] = {"cvxpy": True}

    long_set = set([str(x) for x in long_names])
    short_set = set([str(x) for x in short_names])
    if not long_set or not short_set:
        meta["status"] = "invalid_sides"
        return None, meta

    cand = pd.Index(sorted(set(long_set) | set(short_set)))

    pre = qp_feasibility_precheck(
        long_names=long_names,
        short_names=short_names,
        w_target=w_target,
        gross_long=gross_long,
        gross_short=gross_short,
        max_abs_weight=max_abs_weight,
        turnover_cap=turnover_cap,
        exposures=exposures,
        enforce_exposure_neutrality=enforce_exposure_neutrality,
        cost_model=cost_model,
    )
    meta["precheck"] = pre
    if not bool(pre.get("passed")):
        meta["status"] = "precheck_failed"
        return None, meta

    import cvxpy as cp

    n = int(len(cand))
    cap = float(max_abs_weight)

    s = scores.reindex(cand).astype(float).fillna(0.0).to_numpy(dtype=float)
    wt = w_target.reindex(cand).astype(float).fillna(0.0).to_numpy(dtype=float)

    cm = cost_model or OptimizerCostModel()
    av = float(getattr(cm, 'cost_aversion', 0.0) or 0.0)

    # Optional cost vectors aligned to the candidate index.
    trade_c = None
    if av > 0.0 and getattr(cm, 'trade_cost', None) is not None:
        tc = cm.trade_cost.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        trade_c = tc.to_numpy(dtype=float)

    borrow_c = None
    if av > 0.0 and getattr(cm, 'borrow_cost', None) is not None:
        bc = cm.borrow_cost.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        borrow_c = bc.to_numpy(dtype=float)

    impact_c = None
    impact_p = float(getattr(cm, 'impact_exponent', 0.5) or 0.5)
    impact_pow = 1.0 + float(impact_p)
    if av > 0.0 and getattr(cm, 'impact_coeff', None) is not None and impact_pow > 1.0:
        ic = cm.impact_coeff.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        impact_c = ic.to_numpy(dtype=float)

    # Risk model terms (diag or factor).
    risk_av = float(getattr(cm, 'risk_aversion', 0.0) or 0.0)
    risk_model = str(getattr(cm, 'risk_model', 'diag') or 'diag').strip().lower()

    use_factor_risk = False
    risk_v = None  # diag variance proxy

    risk_B = None  # n x k loadings (standardized)
    risk_F = None  # k x k factor covariance (PSD)
    risk_factor_names = None
    idio_v = None  # n idiosyncratic variance

    # Prefer factor risk model when provided: Sigma = BFB^T + D.
    if risk_av > 0.0 and getattr(cm, 'factor_cov', None) is not None and getattr(cm, 'factor_loadings', None) is not None:
        Bdf = cm.factor_loadings.reindex(cand)
        if Bdf is not None and not Bdf.empty:
            Bz = _prepare_exposures(Bdf, index=cand)
            if not Bz.empty:
                risk_B = Bz.to_numpy(dtype=float)
                risk_factor_names = list(Bz.columns)

                F = np.asarray(cm.factor_cov, dtype=float)
                if F.ndim == 0:
                    F = np.array([[float(F)]], dtype=float)
                if F.ndim == 1:
                    F = np.diag(F.astype(float))
                F = 0.5 * (F + F.T)

                # Robust PSD projection.
                try:
                    vals, vecs = np.linalg.eigh(F)
                    vals = np.clip(vals, 0.0, None)
                    F = vecs @ np.diag(vals) @ vecs.T
                except Exception:
                    F = np.diag(np.clip(np.diag(F), 0.0, None))

                risk_F = 0.5 * (F + F.T)

                idv_src = getattr(cm, 'idio_var', None)
                if idv_src is None and getattr(cm, 'risk_var', None) is not None:
                    idv_src = cm.risk_var

                if idv_src is not None:
                    idv = idv_src.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
                    med = float(np.nanmedian(idv.to_numpy(dtype=float))) if np.isfinite(idv.to_numpy(dtype=float)).any() else 0.0
                    idv = idv.fillna(med).clip(lower=0.0)
                    idio_v = idv.to_numpy(dtype=float)
                else:
                    idio_v = np.zeros(int(n), dtype=float)

                use_factor_risk = True
                risk_model = 'factor'

    # Fallback: diagonal risk penalty sum(var_i * w_i^2).
    if (not use_factor_risk) and risk_av > 0.0 and getattr(cm, 'risk_var', None) is not None:
        rv = cm.risk_var.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
        med = float(np.nanmedian(rv.to_numpy(dtype=float))) if np.isfinite(rv.to_numpy(dtype=float)).any() else 0.0
        rv = rv.fillna(med).clip(lower=0.0)
        risk_v = rv.to_numpy(dtype=float)
        risk_model = 'diag'
    max_u = None
    if getattr(cm, 'max_trade_abs', None) is not None:
        mu = cm.max_trade_abs.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
        # Missing bounds are treated as unconstrained.
        mu = mu.fillna(np.inf).clip(lower=0.0)
        mv = mu.to_numpy(dtype=float)
        if np.isfinite(mv).any():
            mv = np.where(np.isfinite(mv), mv, 1e9)
            max_u = mv

    slack_lambda = float(getattr(cm, 'exposure_slack_lambda', 0.0) or 0.0)
    slack_var = None
    w = cp.Variable(n)
    u = cp.Variable(n, nonneg=True)

    constraints = []

    # Sign / box constraints.
    for i, name in enumerate(list(cand.astype(str))):
        if name in long_set:
            constraints.append(w[i] >= 0.0)
            if cap > 0.0:
                constraints.append(w[i] <= cap)
        if name in short_set:
            constraints.append(w[i] <= 0.0)
            if cap > 0.0:
                constraints.append(w[i] >= -cap)

    # Dollar-neutral gross constraints.
    long_mask = np.asarray([1.0 if str(x) in long_set else 0.0 for x in cand], dtype=float)
    short_mask = np.asarray([1.0 if str(x) in short_set else 0.0 for x in cand], dtype=float)
    constraints.append(cp.sum(cp.multiply(long_mask, w)) == float(gross_long))
    constraints.append(cp.sum(cp.multiply(short_mask, w)) == -float(gross_short))

    # Turnover modeling vs the no-trade baseline (w_target).
    constraints.append(u >= w - wt)
    constraints.append(u >= -(w - wt))

    if max_u is not None:
        constraints.append(u <= max_u)
        meta['participation_cap'] = True


    tcap = float(turnover_cap)
    if tcap > 0.0:
        constraints.append(0.5 * cp.sum(u) <= tcap)

    # Exposure penalty / neutrality.
    A = None
    if exposures is not None and not exposures.empty:
        X = _prepare_exposures(exposures, index=cand)
        if not X.empty:
            A = X.to_numpy(dtype=float)
            if bool(enforce_exposure_neutrality):
                if slack_lambda > 0.0:
                    slack_var = cp.Variable(int(A.shape[1]))
                    constraints.append(A.T @ w == slack_var)
                    meta['exposure_slack'] = True
                else:
                    constraints.append(A.T @ w == 0.0)

    # Objective: maximize score - (l2 + turnover + optional exposure penalty).
    obj = -s @ w

    l2 = float(cfg.l2_lambda)
    if l2 > 0.0:
        obj = obj + l2 * cp.sum_squares(w)

    if risk_av > 0.0:
        if use_factor_risk and risk_B is not None and risk_F is not None:
            y = risk_B.T @ w
            obj = obj + risk_av * cp.quad_form(y, risk_F)
            if idio_v is not None:
                obj = obj + risk_av * cp.sum(cp.multiply(idio_v, cp.square(w)))
            meta['risk_model'] = 'factor'
            if getattr(cm, 'risk_meta', None) is not None:
                meta['risk_model_meta'] = cm.risk_meta
        elif risk_v is not None:
            obj = obj + risk_av * cp.sum(cp.multiply(risk_v, cp.square(w)))
            meta['risk_model'] = 'diag'

    tlam = float(cfg.turnover_lambda)
    if tlam > 0.0:
        obj = obj + tlam * cp.sum(u)

    lam = float(cfg.exposure_lambda)
    if lam > 0.0 and A is not None:
        obj = obj + lam * cp.sum_squares(A.T @ w)

    # Optional slack penalty when strict exposure neutrality is requested.
    if slack_var is not None and slack_lambda > 0.0:
        obj = obj + slack_lambda * cp.sum_squares(slack_var)

    # Real-world cost terms (convex) inside the optimizer objective.
    if av > 0.0 and trade_c is not None:
        obj = obj + av * cp.sum(cp.multiply(trade_c, u))
    if av > 0.0 and borrow_c is not None:
        obj = obj + av * cp.sum(cp.multiply(borrow_c, cp.pos(-w)))
    if av > 0.0 and impact_c is not None and impact_pow > 1.0:
        obj = obj + av * cp.sum(cp.multiply(impact_c, cp.power(u, impact_pow)))

    prob = cp.Problem(cp.Minimize(obj), constraints)

    used_solver = solver or "OSQP"
    try:
        prob.solve(solver=used_solver, warm_start=True, verbose=False)
        meta["solver"] = used_solver
    except Exception:
        try:
            prob.solve(warm_start=True, verbose=False)
            meta["solver"] = "default"
        except Exception as e:
            meta["status"] = f"solve_error:{e.__class__.__name__}"
            return None, meta

    meta["status"] = str(prob.status)
    if prob.status not in {"optimal", "optimal_inaccurate"}:
        return None, meta
    if w.value is None:
        return None, meta

    wv = np.asarray(w.value, dtype=float).reshape(-1)
    if wv.size != n or not np.isfinite(wv).any():
        return None, meta

    w_out = pd.Series(wv, index=cand, dtype=float)

    # Diagnostics are intentionally lightweight and JSON-friendly.
    delta = (w_out.to_numpy(dtype=float) - wt)
    abs_delta = np.abs(delta)

    turnover_to_target = float(0.5 * abs_delta.sum())
    meta["turnover_to_target"] = float(turnover_to_target)
    meta["objective_value"] = float(prob.value) if prob.value is not None else float("nan")

    diag: Dict[str, Any] = {
        "n_candidates": int(n),
        "n_long": int(len(long_set)),
        "n_short": int(len(short_set)),
        "gross_long": float(w_out.reindex(list(long_set)).fillna(0.0).sum()),
        "gross_short": float((-w_out.reindex(list(short_set)).fillna(0.0).sum())),
        "gross_long_resid": float(w_out.reindex(list(long_set)).fillna(0.0).sum() - float(gross_long)),
        "gross_short_resid": float(w_out.reindex(list(short_set)).fillna(0.0).sum() + float(gross_short)),
        "turnover_to_target": float(turnover_to_target),
        "turnover_cap": float(tcap),
    }

    if cap > 0.0:
        aw = np.abs(w_out.to_numpy(dtype=float))
        diag["max_abs_weight"] = float(cap)
        diag["max_abs_weight_used"] = float(np.nanmax(aw)) if aw.size else 0.0
        diag["at_cap_frac"] = float(np.mean(aw >= (0.999 * cap))) if aw.size else 0.0

    if max_u is not None:
        mu = np.asarray(max_u, dtype=float)
        mu = np.where(mu <= 0.0, 0.0, mu)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mu > 0.0, abs_delta / mu, 0.0)
        diag["participation_cap"] = True
        diag["participation_max_ratio"] = float(np.nanmax(ratio)) if ratio.size else 0.0
        diag["participation_binding_frac"] = float(np.mean(ratio >= 0.999)) if ratio.size else 0.0
        diag["participation_violation_max"] = float(np.nanmax(abs_delta - mu)) if abs_delta.size else 0.0

    # Exposure summary (useful when neutrality or penalties are enabled).
    if exposures is not None and not exposures.empty:
        X = _prepare_exposures(exposures, index=cand)
        if not X.empty:
            ev = pd.Series((X.to_numpy(dtype=float).T @ w_out.to_numpy(dtype=float)), index=X.columns, dtype=float)
            ev = ev.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            top = ev.abs().sort_values(ascending=False).head(10)
            diag["exposure_max_abs"] = float(ev.abs().max())
            diag["exposure_l2"] = float(np.sqrt(float((ev * ev).sum())))
            diag["exposure_top_abs"] = [(str(k), float(ev.loc[k])) for k in top.index]

    if slack_var is not None and getattr(slack_var, "value", None) is not None:
        sv = np.asarray(slack_var.value, dtype=float).reshape(-1)
        if sv.size:
            diag["exposure_slack_max_abs"] = float(np.nanmax(np.abs(sv)))
            diag["exposure_slack_l2"] = float(np.sqrt(float(np.sum(sv * sv))))

    # Objective breakdown for quick debugging.
    terms: Dict[str, Any] = {
        "alpha_score": float(np.dot(s, w_out.to_numpy(dtype=float))),
        "l2": float(float(cfg.l2_lambda) * float(np.sum(w_out.to_numpy(dtype=float) ** 2))),
        "turnover_l1": float(float(cfg.turnover_lambda) * float(abs_delta.sum())),
    }
    if risk_av > 0.0:
        wv2 = w_out.to_numpy(dtype=float) ** 2
        if use_factor_risk and risk_B is not None and risk_F is not None:
            y = risk_B.T @ w_out.to_numpy(dtype=float)
            factor_proxy = float(y.T @ risk_F @ y)
            idio_proxy = float(np.dot(idio_v, wv2)) if idio_v is not None else 0.0
            proxy = float(factor_proxy + idio_proxy)

            terms["factor_risk"] = float(risk_av * factor_proxy)
            terms["idio_risk"] = float(risk_av * idio_proxy)

            diag["risk_model"] = "factor"
            diag["risk_factor_proxy"] = float(factor_proxy)
            diag["risk_idio_proxy"] = float(idio_proxy)
            diag["risk_proxy"] = float(proxy)
            diag["risk_vol_annual_proxy"] = float(np.sqrt(max(proxy, 0.0)))
            diag["risk_aversion"] = float(risk_av)

            # Factor-level contributions: y_i * (F y)_i
            try:
                mvec = risk_F @ y
                cvec = y * mvec
                if risk_factor_names:
                    order = np.argsort(-np.abs(cvec))[: min(10, int(cvec.size))]
                    diag["risk_factor_top_contributors"] = [
                        {
                            "factor": str(risk_factor_names[j]),
                            "contrib": float(cvec[j]),
                            "exposure": float(y[j]),
                        }
                        for j in order
                        if np.isfinite(cvec[j])
                    ]
            except Exception:
                pass

            # Name-level idiosyncratic contributions.
            if idio_v is not None:
                ic = np.asarray(idio_v, dtype=float).reshape(-1) * wv2
                if np.isfinite(ic).any():
                    order = np.argsort(-ic)[:5]
                    diag["risk_idio_top_contributors"] = [
                        {
                            "instrument": str(w_out.index[j]),
                            "contrib": float(ic[j]),
                            "idio_var": float(idio_v[j]),
                            "weight": float(w_out.iloc[j]),
                        }
                        for j in order
                        if np.isfinite(ic[j])
                    ]

        elif risk_v is not None:
            proxy = float(np.dot(risk_v, wv2))
            terms["diag_risk"] = float(risk_av * proxy)
            diag["risk_model"] = "diag"
            diag["risk_proxy"] = float(proxy)
            diag["risk_vol_annual_proxy"] = float(np.sqrt(max(proxy, 0.0)))
            diag["risk_aversion"] = float(risk_av)
            contrib = risk_v * wv2
            if np.isfinite(contrib).any():
                top_idx = np.argsort(-contrib)[:5]
                diag["risk_top_contributors"] = [
                    {
                        "instrument": str(cand[j]),
                        "contrib": float(contrib[j]),
                        "risk_var": float(risk_v[j]),
                        "weight": float(w_out.iloc[j]),
                    }
                    for j in top_idx
                    if np.isfinite(contrib[j])
                ]

    if A is not None:

        exv = (A.T @ w_out.to_numpy(dtype=float)).reshape(-1)
        terms["exposure_penalty"] = float(float(cfg.exposure_lambda) * float(np.sum(exv * exv)))
    if slack_var is not None and slack_lambda > 0.0 and getattr(slack_var, "value", None) is not None:
        sv = np.asarray(slack_var.value, dtype=float).reshape(-1)
        terms["slack_penalty"] = float(float(slack_lambda) * float(np.sum(sv * sv)))
    if av > 0.0 and trade_c is not None:
        terms["trade_cost"] = float(av * float(np.dot(trade_c, abs_delta)))
        meta["predicted_trade_cost"] = float(np.dot(trade_c, abs_delta))
    if av > 0.0 and borrow_c is not None:
        terms["borrow_cost"] = float(av * float(np.dot(borrow_c, np.maximum(-w_out.to_numpy(dtype=float), 0.0))))
        meta["predicted_borrow_cost"] = float(np.dot(borrow_c, np.maximum(-w_out.to_numpy(dtype=float), 0.0)))
    if av > 0.0 and impact_c is not None and impact_pow > 1.0:
        terms["impact_cost"] = float(av * float(np.sum(impact_c * (abs_delta ** impact_pow))))
        meta["predicted_impact_cost"] = float(np.sum(impact_c * (abs_delta ** impact_pow)))

    diag["objective_terms"] = terms

    mu_s = None
    if getattr(cm, "max_trade_abs", None) is not None:
        mu_s = cm.max_trade_abs.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)

    slack_v = None
    if slack_var is not None and getattr(slack_var, "value", None) is not None:
        slack_v = np.asarray(slack_var.value, dtype=float)

    diag["constraint_summary"] = _build_constraint_summary(
        w=w_out,
        w_target=pd.Series(wt, index=cand, dtype=float),
        gross_long=float(gross_long),
        gross_short=float(gross_short),
        max_abs_weight=float(max_abs_weight),
        turnover_cap=float(tcap),
        max_trade_abs=mu_s,
        exposures=exposures,
        slack=slack_v,
        precheck=pre,
    )

    meta["diagnostics"] = diag

    if (w_out > 0.0).sum() == 0 or (w_out < 0.0).sum() == 0:
        meta["status"] = "degenerate_solution"
        return None, meta

    return w_out, meta


def optimize_long_short_weights_with_meta(
    scores: pd.Series,
    *,
    long_names: Sequence[str],
    short_names: Sequence[str],
    w_target: pd.Series,
    exposures: Optional[pd.DataFrame],
    cfg: OptimizerConfig,
    gross_long: float = 0.5,
    gross_short: float = 0.5,
    max_abs_weight: float = 0.0,
    backend: str = "auto",
    turnover_cap: float = 0.0,
    enforce_exposure_neutrality: bool = False,
    solver: Optional[str] = None,
    cost_model: Optional[OptimizerCostModel] = None,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """Optimize with an optional constrained backend.

    backend:
      - auto: try QP if available, else ridge
      - qp: force QP (fallback to ridge on failure)
      - ridge: force ridge
    """

    meta: Dict[str, Any] = {
        "backend_requested": str(backend),
        "backend_used": "ridge",
        "fallback": None,
    }

    b = str(backend or "auto").strip().lower()
    if b not in {"auto", "qp", "ridge"}:
        b = "auto"

    tried_qp = False
    if b in {"auto", "qp"}:
        tried_qp = True
        pre = qp_feasibility_precheck(
            long_names=long_names,
            short_names=short_names,
            w_target=w_target,
            gross_long=gross_long,
            gross_short=gross_short,
            max_abs_weight=max_abs_weight,
            turnover_cap=turnover_cap,
            exposures=exposures,
            enforce_exposure_neutrality=enforce_exposure_neutrality,
            cost_model=cost_model,
        )
        meta["qp_precheck"] = pre

        if not bool(pre.get("passed")):
            reasons = list(pre.get("reasons") or [])
            meta["fallback"] = "qp_precheck_failed" + (":" + ",".join(reasons[:2]) if reasons else "")
        elif not _cvxpy_available():
            meta["fallback"] = "cvxpy_not_installed"
        else:
            w_qp, m = _solve_qp_cvxpy(
                scores,
                long_names=long_names,
                short_names=short_names,
                w_target=w_target,
                exposures=exposures,
                cfg=cfg,
                gross_long=gross_long,
                gross_short=gross_short,
                max_abs_weight=max_abs_weight,
                turnover_cap=turnover_cap,
                enforce_exposure_neutrality=enforce_exposure_neutrality,
                solver=solver,
                cost_model=cost_model,
            )
            meta.update({"qp_meta": m})
            if w_qp is not None:
                meta["backend_used"] = "qp"
                # Promote a compact diagnostics view to a stable top-level field.
                if isinstance(m, dict) and "diagnostics" in m:
                    meta["diagnostics"] = m.get("diagnostics")
                return w_qp, meta
            meta["fallback"] = f"qp_failed:{m.get('status', 'unknown')}"

    # Ridge fallback (or forced ridge).
    long_set = set([str(x) for x in long_names])
    short_set = set([str(x) for x in short_names])
    cand = pd.Index(sorted(set(long_set) | set(short_set)))

    scores_adj, adj_meta = _cost_aware_adjust_scores(scores, cand=cand, short_set=short_set, cost_model=cost_model)
    if adj_meta:
        meta["ridge_adjustments"] = adj_meta

    w = optimize_long_short_weights(
        scores_adj,
        long_names=long_names,
        short_names=short_names,
        w_target=w_target,
        exposures=exposures,
        cfg=cfg,
        gross_long=gross_long,
        gross_short=gross_short,
        max_abs_weight=max_abs_weight,
        cost_model=cost_model,
    )
    if w is None:
        meta["backend_used"] = "ridge"
        if tried_qp and meta.get("fallback") is None:
            meta["fallback"] = "ridge_failed"
        return None, meta

    w2, limit_meta = _apply_ridge_trade_limits(
        w,
        w_target=w_target,
        long_names=long_names,
        short_names=short_names,
        gross_long=gross_long,
        gross_short=gross_short,
        max_abs_weight=max_abs_weight,
        turnover_cap=turnover_cap,
        cost_model=cost_model,
    )
    if limit_meta:
        meta["ridge_limits"] = limit_meta
    if w2 is None:
        meta["backend_used"] = "ridge"
        if tried_qp and meta.get("fallback") is None:
            meta["fallback"] = "ridge_failed"
        return None, meta

    meta["backend_used"] = "ridge"
    # Unify diagnostics schema across backends for run artifacts and reporting.
    try:
        cm = cost_model or OptimizerCostModel()
        cand = w2.index
        wt = w_target.reindex(cand).astype(float).fillna(0.0)
        svec = scores_adj.reindex(cand).astype(float).fillna(0.0).to_numpy(dtype=float)
        wv = w2.reindex(cand).astype(float).fillna(0.0).to_numpy(dtype=float)
        dv = np.abs((w2 - wt).to_numpy(dtype=float))

        diag: Dict[str, Any] = {
            "n_candidates": int(len(cand)),
            "n_long": int(len(long_set)),
            "n_short": int(len(short_set)),
            "gross_long": float(w2.clip(lower=0.0).sum()),
            "gross_short": float((-w2.clip(upper=0.0)).sum()),
            "turnover_to_target": float(0.5 * float(dv.sum())),
            "turnover_cap": float(turnover_cap),
        }

        mu_s = None
        if getattr(cm, "max_trade_abs", None) is not None:
            mu_s = cm.max_trade_abs.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)

        diag["constraint_summary"] = _build_constraint_summary(
            w=w2,
            w_target=wt,
            gross_long=float(gross_long),
            gross_short=float(gross_short),
            max_abs_weight=float(max_abs_weight),
            turnover_cap=float(turnover_cap),
            max_trade_abs=mu_s,
            exposures=exposures,
            slack=None,
            precheck=(meta.get("qp_precheck") if isinstance(meta.get("qp_precheck"), dict) else None),
        )

        terms: Dict[str, Any] = {
            "alpha_score": float(np.dot(svec, wv)),
            "l2": float(float(cfg.l2_lambda) * float(np.sum(wv * wv))),
            "turnover_l1": float(float(cfg.turnover_lambda) * float(dv.sum())),
        }

        # Exposure penalty proxy.
        if exposures is not None and not exposures.empty and float(cfg.exposure_lambda) > 0.0:
            X = _prepare_exposures(exposures, index=cand)
            if not X.empty:
                exv = (X.to_numpy(dtype=float).T @ wv)
                terms["exposure_penalty"] = float(float(cfg.exposure_lambda) * float(np.sum(exv * exv)))

        # Risk proxy.
        rav = float(getattr(cm, "risk_aversion", 0.0) or 0.0)
        if rav > 0.0:
            used_factor = False
            if getattr(cm, "factor_cov", None) is not None and getattr(cm, "factor_loadings", None) is not None:
                Bdf = cm.factor_loadings.reindex(cand)
                if Bdf is not None and not Bdf.empty:
                    Bz = _prepare_exposures(Bdf, index=cand)
                    if not Bz.empty:
                        B = Bz.to_numpy(dtype=float)
                        F = np.asarray(cm.factor_cov, dtype=float)
                        if F.ndim == 0:
                            F = np.array([[float(F)]], dtype=float)
                        if F.ndim == 1:
                            F = np.diag(F.astype(float))
                        F = 0.5 * (F + F.T)
                        try:
                            vals, vecs = np.linalg.eigh(F)
                            vals = np.clip(vals, 0.0, None)
                            F = vecs @ np.diag(vals) @ vecs.T
                        except Exception:
                            F = np.diag(np.clip(np.diag(F), 0.0, None))
                        y = B.T @ wv
                        factor_proxy = float(y.T @ F @ y)

                        idv_src = getattr(cm, "idio_var", None)
                        if idv_src is None and getattr(cm, "risk_var", None) is not None:
                            idv_src = cm.risk_var
                        if idv_src is not None:
                            idv = idv_src.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
                            med = float(np.nanmedian(idv.to_numpy(dtype=float))) if np.isfinite(idv.to_numpy(dtype=float)).any() else 0.0
                            idv = idv.fillna(med).clip(lower=0.0)
                            idio_v = idv.to_numpy(dtype=float)
                        else:
                            idio_v = np.zeros(int(len(cand)), dtype=float)
                        idio_proxy = float(np.dot(idio_v, wv * wv))
                        proxy = float(factor_proxy + idio_proxy)
                        diag["risk_model"] = "factor"
                        diag["risk_factor_proxy"] = float(factor_proxy)
                        diag["risk_idio_proxy"] = float(idio_proxy)
                        diag["risk_proxy"] = float(proxy)
                        diag["risk_vol_annual_proxy"] = float(np.sqrt(max(proxy, 0.0)))
                        diag["risk_aversion"] = float(rav)
                        if getattr(cm, "risk_meta", None) is not None:
                            diag["risk_model_meta"] = cm.risk_meta

                        terms["factor_risk"] = float(rav * factor_proxy)
                        terms["idio_risk"] = float(rav * idio_proxy)

                        try:
                            mvec = F @ y
                            cvec = y * mvec
                            names = list(Bz.columns)
                            order = np.argsort(-np.abs(cvec))[: min(10, int(cvec.size))]
                            diag["risk_factor_top_contributors"] = [
                                {"factor": str(names[j]), "contrib": float(cvec[j]), "exposure": float(y[j])}
                                for j in order
                                if np.isfinite(cvec[j])
                            ]
                        except Exception:
                            pass

                        ic = idio_v * (wv * wv)
                        if np.isfinite(ic).any():
                            order = np.argsort(-ic)[:5]
                            diag["risk_idio_top_contributors"] = [
                                {
                                    "instrument": str(cand[j]),
                                    "contrib": float(ic[j]),
                                    "idio_var": float(idio_v[j]),
                                    "weight": float(wv[j]),
                                }
                                for j in order
                                if np.isfinite(ic[j])
                            ]

                        used_factor = True

            if not used_factor and getattr(cm, "risk_var", None) is not None:
                rv = cm.risk_var.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan)
                med = float(np.nanmedian(rv.to_numpy(dtype=float))) if np.isfinite(rv.to_numpy(dtype=float)).any() else 0.0
                rv = rv.fillna(med).clip(lower=0.0)
                rva = rv.to_numpy(dtype=float)
                proxy = float(np.dot(rva, wv * wv))
                diag["risk_model"] = "diag"
                diag["risk_proxy"] = float(proxy)
                diag["risk_vol_annual_proxy"] = float(np.sqrt(max(proxy, 0.0)))
                diag["risk_aversion"] = float(rav)
                terms["diag_risk"] = float(rav * proxy)
                contrib = rva * (wv * wv)
                if np.isfinite(contrib).any():
                    top_idx = np.argsort(-contrib)[:5]
                    diag["risk_top_contributors"] = [
                        {
                            "instrument": str(cand[j]),
                            "contrib": float(contrib[j]),
                            "risk_var": float(rva[j]),
                            "weight": float(wv[j]),
                        }
                        for j in top_idx
                        if np.isfinite(contrib[j])
                    ]

        # Cost-aware proxy terms.
        av = float(getattr(cm, "cost_aversion", 0.0) or 0.0)
        if av > 0.0 and getattr(cm, "trade_cost", None) is not None:
            tc = cm.trade_cost.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            terms["trade_cost"] = float(av * float(np.dot(tc.to_numpy(dtype=float), dv)))
        if av > 0.0 and getattr(cm, "borrow_cost", None) is not None:
            bc = cm.borrow_cost.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            terms["borrow_cost"] = float(av * float(np.dot(bc.to_numpy(dtype=float), np.maximum(-wv, 0.0))))
        if av > 0.0 and getattr(cm, "impact_coeff", None) is not None:
            ic = cm.impact_coeff.reindex(cand).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            p = 1.0 + float(getattr(cm, "impact_exponent", 0.5) or 0.5)
            if p > 1.0:
                terms["impact_cost"] = float(av * float(np.sum(ic.to_numpy(dtype=float) * (dv ** p))))

        diag["objective_terms"] = terms
        meta["diagnostics"] = diag
    except Exception:
        pass
    return w2, meta
