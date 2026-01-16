"""agent.research.alpha_allocation

P2.19: Alpha-level allocation (portfolio of alphas).

This module learns non-negative weights over a set of alpha strategies using
only information available before each OOS test segment (walk-forward).

The default objective is a small convex trade-off:
- reward: in-sample score (e.g., IR)
- penalty: correlation (redundancy) across alphas

The backend is dependency-light. If `cvxpy` is available, a QP solution is used.
Otherwise, a projected-gradient approximation is used.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd



def project_to_bounded_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Project a vector onto the capped simplex.

    Constraint set: w >= 0, sum(w) = 1, and w_i <= cap for all i.
    """

    x = np.asarray(v, dtype=float).copy()
    x = np.where(np.isfinite(x), x, 0.0)
    n = int(x.size)
    if n == 0:
        return x

    cap = float(cap)
    if cap <= 0.0:
        # Degenerate: everything forced to 0; keep shape.
        return np.zeros_like(x)

    # If the cap is too small to sum to 1, fall back to uniform.
    if cap * n < 1.0:
        return np.ones_like(x) / float(n)

    # Active-set projection: iteratively fix capped coordinates, then project the rest
    # onto the simplex.
    free = np.ones(n, dtype=bool)
    w = np.zeros(n, dtype=float)

    def _proj_simplex(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        y = np.where(np.isfinite(y), y, 0.0)
        u = np.sort(y)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - 1.0))[0]
        if rho.size == 0:
            return np.zeros_like(y)
        rho = int(rho[-1])
        theta = (cssv[rho] - 1.0) / float(rho + 1)
        return np.maximum(y - theta, 0.0)

    remaining_mass = 1.0
    for _ in range(n + 1):
        idx = np.where(free)[0]
        if idx.size == 0:
            break
        y = x[idx]
        y_proj = _proj_simplex(y)
        # Rescale to the remaining mass.
        if y_proj.sum() > 0:
            y_proj = y_proj / float(y_proj.sum()) * float(remaining_mass)
        w[idx] = y_proj

        over = w > cap + 1e-12
        if not np.any(over & free):
            break

        # Fix the capped coords and iterate.
        fix = over & free
        w[fix] = cap
        free[fix] = False
        remaining_mass = 1.0 - float(w[~free].sum())
        if remaining_mass <= 0.0:
            break

    # Final clean-up.
    w = np.clip(w, 0.0, cap)
    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def _information_ratio(r: np.ndarray, trading_days: int) -> float:
    arr = np.asarray(r, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd <= 0.0:
        return 0.0
    return float(mu / sd * np.sqrt(float(trading_days)))


def _annualized_return(mu_daily: float, trading_days: int) -> float:
    try:
        return float((1.0 + float(mu_daily)) ** float(trading_days) - 1.0)
    except Exception:
        return 0.0


def _make_psd(mat: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """Eigenvalue-clamp a symmetric matrix into PSD form."""

    m = np.asarray(mat, dtype=float)
    m = (m + m.T) / 2.0
    try:
        w, v = np.linalg.eigh(m)
        w = np.maximum(w, float(min_eig))
        return (v * w) @ v.T
    except Exception:
        # Fallback: diagonal jitter.
        eps = float(min_eig)
        return m + np.eye(m.shape[0], dtype=float) * eps


def _project_to_bounded_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """Backward-compatible wrapper (kept for internal callers)."""

    return project_to_bounded_simplex(v, cap=float(cap))


def _align_prev_weights(prev: Optional[pd.Series], cols: list[str], cap: float) -> Optional[np.ndarray]:
    """Align previous weights to the current alpha universe.

    The optimizer expects a feasible reference point on the same simplex.
    """

    if prev is None or not isinstance(prev, pd.Series) or prev.empty:
        return None

    s = prev.reindex(cols).fillna(0.0).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Allocation weights are constrained to be non-negative.
    s = s.clip(lower=0.0)
    tot = float(s.sum())
    if tot <= 0.0:
        return None
    w = (s / tot).to_numpy(dtype=float)
    if cap > 0.0 and cap < 1.0:
        w = _project_to_bounded_simplex(w, cap=float(cap))
    return w


def _scores_from_returns(
    r: pd.DataFrame, *, trading_days: int, score_metric: str
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute per-alpha scores from an in-sample return matrix."""

    scores: Dict[str, float] = {}
    out: list[float] = []
    for c in r.columns:
        arr = pd.to_numeric(r[c], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        mu = float(np.mean(arr)) if arr.size else 0.0
        sd = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        ir = float(mu / sd * np.sqrt(float(trading_days))) if sd > 0.0 else 0.0
        ann = _annualized_return(mu, trading_days=int(trading_days))
        s = float(ir) if str(score_metric).lower() in {"ir", "information_ratio"} else float(ann)
        scores[str(c)] = float(s)
        out.append(float(s))
    return np.asarray(out, dtype=float), scores


def fit_alpha_allocation(
    returns: pd.DataFrame,
    *,
    trading_days: int = 252,
    score_metric: str = "information_ratio",
    lambda_corr: float = 0.5,
    l2: float = 1e-6,
    turnover_lambda: float = 0.0,
    prev_weights: Optional[pd.Series] = None,
    max_weight: float = 1.0,
    use_abs_corr: bool = True,
    backend: str = "auto",  # "auto" | "qp" | "pgd"
    solver: str = "",
    max_iter: int = 200,
    step_size: float = 0.1,
) -> Dict[str, Any]:
    """Fit non-negative alpha weights using in-sample strategy returns.

    Returns a dict with:
    - weights: pd.Series (index=alpha_id)
    - diagnostics: dict
    """

    if returns is None or returns.empty or returns.shape[1] < 1:
        return {"weights": pd.Series(dtype=float), "diagnostics": {"error": "empty returns"}}

    r = returns.copy()
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = list(r.columns)

    score_vec, score_map = _scores_from_returns(r, trading_days=int(trading_days), score_metric=str(score_metric))
    # Use non-negative scores to keep the allocation simple and stable.
    score_vec = np.maximum(score_vec, 0.0)
    if float(score_vec.sum()) <= 0.0:
        score_vec = np.ones_like(score_vec, dtype=float)

    corr = r.corr().fillna(0.0).to_numpy(dtype=float)
    np.fill_diagonal(corr, 1.0)
    if bool(use_abs_corr):
        corr = np.abs(corr)
    corr_psd = _make_psd(corr, min_eig=1e-8)

    k = int(len(cols))
    lam = float(max(lambda_corr, 0.0))
    l2 = float(max(l2, 0.0))
    tlam = float(max(turnover_lambda, 0.0))
    max_weight = float(max_weight)

    cap = float(max_weight) if (max_weight > 0.0 and max_weight < 1.0) else 1.0
    w_prev = _align_prev_weights(prev_weights, cols=cols, cap=cap)

    method_used = "pgd"
    w = (w_prev.copy() if w_prev is not None else np.ones(k, dtype=float) / float(k))
    qp_error: Optional[str] = None

    want_qp = str(backend).lower() in {"auto", "qp"}
    if want_qp:
        try:
            import cvxpy as cp  # type: ignore

            w_var = cp.Variable(k, nonneg=True)
            prev_const = (w_prev if w_prev is not None else (np.ones(k, dtype=float) / float(k)))
            objective = cp.Maximize(
                score_vec @ w_var
                - lam * cp.quad_form(w_var, corr_psd)
                - l2 * cp.sum_squares(w_var)
                - tlam * cp.sum_squares(w_var - prev_const)
            )
            constraints = [cp.sum(w_var) == 1.0]
            if max_weight > 0.0 and max_weight < 1.0:
                constraints.append(w_var <= float(max_weight))
            prob = cp.Problem(objective, constraints)

            chosen_solver = None
            if solver:
                chosen_solver = str(solver)
            else:
                # OSQP is a good default for small QPs when available.
                chosen_solver = "OSQP"

            if w_prev is not None:
                # Hint the solver to warm-start near the previous allocation.
                try:
                    w_var.value = np.asarray(w_prev, dtype=float)
                except Exception:
                    pass

            prob.solve(solver=chosen_solver, warm_start=True, verbose=False)

            if w_var.value is not None and prob.status in {"optimal", "optimal_inaccurate"}:
                w = np.asarray(w_var.value, dtype=float).reshape(-1)
                if not np.isfinite(w).all() or float(w.sum()) <= 0.0:
                    raise ValueError("Invalid QP solution")
                w = np.maximum(w, 0.0)
                w = w / float(w.sum())
                if max_weight > 0.0 and max_weight < 1.0:
                    w = _project_to_bounded_simplex(w, cap=float(max_weight))
                method_used = "qp"
            else:
                qp_error = str(prob.status)
        except Exception as e:
            qp_error = str(e)

    if method_used != "qp":
        # Projected gradient ascent.
        method_used = "pgd"
        for _ in range(int(max_iter)):
            prev_term = (w_prev if w_prev is not None else w)
            grad = score_vec - 2.0 * lam * (corr_psd @ w) - 2.0 * l2 * w - 2.0 * tlam * (w - prev_term)
            w_new = w + float(step_size) * grad
            w_new = _project_to_bounded_simplex(w_new, cap=cap)
            if float(np.linalg.norm(w_new - w)) <= 1e-8:
                w = w_new
                break
            w = w_new

    weights = pd.Series(w, index=pd.Index(cols, name="alpha_id"), dtype=float)

    diagnostics = {
        "method": method_used,
        "backend_requested": str(backend),
        "qp_error": qp_error,
        "lambda_corr": float(lam),
        "l2": float(l2),
        "turnover_lambda": float(tlam),
        "max_weight": float(max_weight),
        "use_abs_corr": bool(use_abs_corr),
        "score_metric": str(score_metric),
        "scores": score_map,
        "corr_mean_offdiag": float((np.sum(corr) - float(k)) / max(float(k * (k - 1)), 1.0)),
    }

    if w_prev is not None:
        try:
            diagnostics["turnover_to_prev"] = float(0.5 * np.sum(np.abs(w - w_prev)))
        except Exception:
            pass

    return {"weights": weights, "diagnostics": diagnostics}
