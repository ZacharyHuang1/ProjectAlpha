"""agent.research.constraint_selection

P2.26: Deterministic, reportable selection of tuning configs under constraints.

P2.28: Add optional Pareto-based *auto selection* strategies:

- "knee": pick the point closest to the "ideal" corner in normalized space.
- "utility": pick the point with the best weighted goodness score.

This is intentionally dependency-light.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _is_max(direction: str) -> bool:
    d = str(direction).strip().lower()
    return d in {"max", "+", "up", "higher", "hi", "high"}


def _normalize_goodness(
    rows: List[Dict[str, Any]],
    objectives: List[Tuple[str, str]],
    *,
    eps: float = 1e-12,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Return (valid_rows, goodness_matrix) in [0, 1] where higher is better."""

    if not rows or not objectives:
        return [], np.zeros((0, 0), dtype=float)

    keys = [str(k).strip() for k, _ in objectives]
    dirs = [str(d).strip() for _, d in objectives]

    mat: List[List[float]] = []
    ok_rows: List[Dict[str, Any]] = []
    for r in rows:
        vec: List[float] = []
        ok = True
        for k in keys:
            v = _sf(r.get(k), default=float("nan"))
            if not np.isfinite(v):
                ok = False
                break
            vec.append(float(v))
        if ok:
            ok_rows.append(r)
            mat.append(vec)

    if not mat:
        return [], np.zeros((0, len(keys)), dtype=float)

    arr = np.asarray(mat, dtype=float)
    goods = np.zeros_like(arr)
    for j, d in enumerate(dirs):
        col = arr[:, j]
        lo = float(np.min(col))
        hi = float(np.max(col))
        rng = float(hi - lo)
        if not np.isfinite(rng) or rng <= eps:
            goods[:, j] = 1.0
            continue
        if _is_max(d):
            goods[:, j] = (col - lo) / rng
        else:
            goods[:, j] = (hi - col) / rng
    goods = np.clip(goods, 0.0, 1.0)
    return ok_rows, goods


def _weights_for_objectives(
    objectives: List[Tuple[str, str]],
    *,
    objective_key: str,
    weights: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Build a non-negative weight vector aligned with objective keys."""

    w: List[float] = []
    for k, _ in objectives:
        kk = str(k).strip()
        if isinstance(weights, dict) and kk in weights:
            ww = _sf(weights.get(kk), default=0.0)
        else:
            ww = 1.0 if kk == str(objective_key) else 0.25
        w.append(float(max(0.0, ww)))
    arr = np.asarray(w, dtype=float)
    s = float(np.sum(arr))
    if not np.isfinite(s) or s <= 0.0:
        return np.ones((len(objectives),), dtype=float) / float(max(1, len(objectives)))
    return arr / s


def compute_pareto_front(
    rows: List[Dict[str, Any]],
    objectives: List[Tuple[str, str]],
    *,
    eps: float = 1e-12,
) -> set[int]:
    """Return indices of Pareto-efficient rows.

    The objective list is expressed as (key, direction) where direction is one
    of {"max", "min"}. A row A dominates row B if A is at least as good as B
    across all objectives and strictly better on at least one objective.

    This implementation is O(N^2 * D) which is fine for small tuning grids.
    """

    if not rows or not objectives:
        return set()

    # Normalize objective specs.
    obj: List[Tuple[str, int]] = []
    for k, d in objectives:
        kk = str(k).strip()
        if not kk:
            continue
        dd = str(d).strip().lower()
        sign = 1 if dd in {"max", "+", "up", "higher"} else -1
        obj.append((kk, int(sign)))
    if not obj:
        return set()

    # Extract numeric matrix (with sign flip for minimization).
    vals: List[Tuple[int, np.ndarray]] = []
    for i, r in enumerate(rows):
        v = []
        ok = True
        for k, s in obj:
            x = r.get(k)
            try:
                f = float(x)
            except Exception:
                ok = False
                break
            if not np.isfinite(f):
                ok = False
                break
            v.append(float(f) * float(s))
        if ok:
            vals.append((int(i), np.asarray(v, dtype=float)))

    if not vals:
        return set()

    idx = [i for i, _ in vals]
    mat = np.vstack([v for _, v in vals])

    keep: set[int] = set(idx)

    # A dominates B if all dims >= and at least one >.
    n = mat.shape[0]
    for a in range(n):
        if idx[a] not in keep:
            continue
        for b in range(n):
            if a == b or idx[b] not in keep:
                continue
            va = mat[a]
            vb = mat[b]
            if np.all(va >= vb - eps) and np.any(va > vb + eps):
                # a dominates b
                keep.discard(idx[b])

    return set(keep)


def annotate_pareto(
    rows: List[Dict[str, Any]],
    *,
    objectives: List[Tuple[str, str]],
    pareto_key: str = "is_pareto",
    rank_key: str = "pareto_rank",
) -> Dict[str, Any]:
    """Annotate rows with Pareto metadata.

    Currently we only compute the first front (rank=1). Non-front rows get
    rank=2.
    """

    front = compute_pareto_front(rows, objectives)
    for i, r in enumerate(rows):
        is_front = bool(int(i) in front)
        r[pareto_key] = is_front
        r[rank_key] = 1 if is_front else 2

    return {"objectives": list(objectives), "pareto_count": int(len(front))}


def _normalize_objective_specs(objectives: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
    """Normalize (key, direction) to (key, sign) where sign=+1 for max, -1 for min."""

    obj: List[Tuple[str, int]] = []
    for k, d in objectives:
        kk = str(k).strip()
        if not kk:
            continue
        dd = str(d).strip().lower()
        sign = 1 if dd in {"max", "+", "up", "higher"} else -1
        obj.append((kk, int(sign)))
    return obj


def _extract_objective_matrix(
    rows: List[Dict[str, Any]],
    objectives: List[Tuple[str, str]],
) -> Tuple[List[int], np.ndarray, List[Tuple[str, str]]]:
    """Extract a numeric matrix for objectives.

    Returns:
        idx: indices into `rows` for valid numeric rows
        mat: shape (n, d) raw objective values
        kept_objectives: objective specs actually used (non-empty)
    """

    obj = _normalize_objective_specs(objectives)
    if not obj:
        return [], np.zeros((0, 0), dtype=float), []

    idx: List[int] = []
    vals: List[List[float]] = []
    for i, r in enumerate(rows):
        v: List[float] = []
        ok = True
        for k, _ in obj:
            x = r.get(k)
            try:
                f = float(x)
            except Exception:
                ok = False
                break
            if not np.isfinite(f):
                ok = False
                break
            v.append(float(f))
        if ok:
            idx.append(int(i))
            vals.append(v)

    if not idx:
        return [], np.zeros((0, 0), dtype=float), []

    return idx, np.asarray(vals, dtype=float), [(k, "max" if s > 0 else "min") for k, s in obj]


def _objective_goodness(
    mat: np.ndarray,
    objectives: List[Tuple[str, str]],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Convert raw objective matrix to [0,1] goodness where higher is better."""

    if mat.size == 0 or not objectives:
        return np.zeros((0, 0), dtype=float)

    obj = _normalize_objective_specs(objectives)
    d = mat.shape[1]
    if len(obj) != d:
        obj = obj[:d]

    g = np.zeros_like(mat, dtype=float)
    for j in range(d):
        col = mat[:, j]
        lo = float(np.nanmin(col))
        hi = float(np.nanmax(col))
        rng = float(hi - lo)
        if not np.isfinite(rng) or rng < eps:
            g[:, j] = 1.0
            continue
        if int(obj[j][1]) > 0:
            # Maximize.
            g[:, j] = (col - lo) / rng
        else:
            # Minimize: lower is better.
            g[:, j] = (hi - col) / rng
        g[:, j] = np.clip(g[:, j], 0.0, 1.0)
    return g


def select_knee_row(
    rows: List[Dict[str, Any]],
    *,
    objectives: List[Tuple[str, str]],
    weights: Optional[Dict[str, float]] = None,
    annotate_key: str = "knee_distance",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Pick the row closest to the ideal point in normalized objective space."""

    idx, mat, kept_obj = _extract_objective_matrix(rows, objectives)
    if mat.size == 0:
        return None, {"enabled": False, "error": "no_numeric_rows"}

    g = _objective_goodness(mat, kept_obj)

    # Objective weights (positive). Missing -> 1.0.
    w = np.ones((g.shape[1],), dtype=float)
    if isinstance(weights, dict):
        for j, (k, _d) in enumerate(kept_obj):
            ww = weights.get(str(k))
            if ww is None:
                continue
            try:
                fv = float(ww)
                if np.isfinite(fv) and fv > 0.0:
                    w[j] = float(fv)
            except Exception:
                continue

    # Weighted distance to the ideal (1,1,...).
    dist = np.sqrt(((1.0 - g) ** 2 * w.reshape(1, -1)).sum(axis=1))
    best_i = int(np.argmin(dist))
    chosen = rows[int(idx[best_i])] if 0 <= best_i < len(idx) else None

    # Annotate rows for report/debug.
    for jj, ii in enumerate(idx):
        try:
            rows[int(ii)][annotate_key] = float(dist[jj])
        except Exception:
            continue

    meta = {
        "enabled": True,
        "method": "knee",
        "objectives": list(kept_obj),
        "weights": dict(weights or {}),
        "best_distance": float(dist[best_i]) if dist.size else float("nan"),
    }
    return chosen, meta


def select_utility_row(
    rows: List[Dict[str, Any]],
    *,
    objectives: List[Tuple[str, str]],
    weights: Optional[Dict[str, float]] = None,
    annotate_key: str = "utility_score",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Pick the row with the best weighted goodness score."""

    idx, mat, kept_obj = _extract_objective_matrix(rows, objectives)
    if mat.size == 0:
        return None, {"enabled": False, "error": "no_numeric_rows"}

    g = _objective_goodness(mat, kept_obj)

    w = np.ones((g.shape[1],), dtype=float)
    if isinstance(weights, dict) and weights:
        for j, (k, _d) in enumerate(kept_obj):
            ww = weights.get(str(k))
            if ww is None:
                continue
            try:
                fv = float(ww)
                if np.isfinite(fv) and fv >= 0.0:
                    w[j] = float(fv)
            except Exception:
                continue
    else:
        # Default: emphasize the first objective key (usually "objective").
        if w.size:
            w[0] = 1.0
            for j in range(1, int(w.size)):
                w[j] = 0.5

    denom = float(w.sum()) if float(w.sum()) > 0.0 else 1.0
    util = (g * w.reshape(1, -1)).sum(axis=1) / denom
    best_i = int(np.argmax(util))
    chosen = rows[int(idx[best_i])] if 0 <= best_i < len(idx) else None

    for jj, ii in enumerate(idx):
        try:
            rows[int(ii)][annotate_key] = float(util[jj])
        except Exception:
            continue

    meta = {
        "enabled": True,
        "method": "utility",
        "objectives": list(kept_obj),
        "weights": dict(weights or {}),
        "best_utility": float(util[best_i]) if util.size else float("nan"),
    }
    return chosen, meta


def _sf(x: Any, default: float) -> float:
    """Safe float."""
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _si(x: Any, default: int) -> int:
    """Safe int."""
    try:
        return int(float(x))
    except Exception:
        return default


def normalize_constraints(constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop None values and normalize simple numeric types."""
    out: Dict[str, Any] = {}
    if not isinstance(constraints, dict):
        return out

    for k, v in constraints.items():
        if v is None:
            continue
        if isinstance(v, (int, float, np.floating, np.integer)):
            out[str(k)] = float(v)
        else:
            out[str(k)] = v
    return out


def evaluate_constraints(row: Dict[str, Any], constraints: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (is_feasible, violation_reasons)."""
    reasons: List[str] = []
    if not constraints:
        return True, reasons

    # Max constraints
    m_to = constraints.get("max_alpha_weight_turnover_mean")
    if m_to is not None:
        v = _sf(row.get("alpha_weight_turnover_mean"), default=float("inf"))
        if not np.isfinite(v) or v > float(m_to):
            reasons.append(f"alpha_weight_turnover_mean>{float(m_to)}")

    m_drag = constraints.get("max_turnover_cost_drag_bps_mean")
    if m_drag is not None:
        v = _sf(row.get("turnover_cost_drag_bps_mean"), default=float("inf"))
        if not np.isfinite(v) or v > float(m_drag):
            reasons.append(f"turnover_cost_drag_bps_mean>{float(m_drag)}")

    m_sw = constraints.get("max_regime_switch_rate_mean")
    if m_sw is not None:
        v = _sf(row.get("regime_switch_rate_mean"), default=float("inf"))
        if not np.isfinite(v) or v > float(m_sw):
            reasons.append(f"regime_switch_rate_mean>{float(m_sw)}")

    m_fb = constraints.get("max_fallback_frac_mean")
    if m_fb is not None:
        v = _sf(row.get("fallback_frac_mean"), default=float("inf"))
        if not np.isfinite(v) or v > float(m_fb):
            reasons.append(f"fallback_frac_mean>{float(m_fb)}")

    # Min constraints
    m_splits = constraints.get("min_splits_used")
    if m_splits is not None:
        n = _si(row.get("n_splits_used"), default=0)
        if n < int(float(m_splits)):
            reasons.append(f"n_splits_used<{int(float(m_splits))}")

    return (len(reasons) == 0), reasons


def annotate_constraints(rows: List[Dict[str, Any]], constraints: Dict[str, Any]) -> None:
    """Annotate each row with constraint feasibility metadata."""
    if not constraints:
        for r in rows:
            r["is_feasible"] = True
            r["constraint_violations"] = ""
        return

    for r in rows:
        ok, reasons = evaluate_constraints(r, constraints)
        r["is_feasible"] = bool(ok)
        r["constraint_violations"] = ";".join(reasons)


def select_best_row(
    rows: List[Dict[str, Any]],
    *,
    objective_key: str,
    constraints: Optional[Dict[str, Any]] = None,
    prefer_pareto: bool = False,
    pareto_key: str = "is_pareto",
    selection_method: str = "best_objective",
    objectives: Optional[List[Tuple[str, str]]] = None,
    utility_weights: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Choose a row under constraints.

    selection_method:
        - "best_objective" (default): maximize objective_key (P2.26)
        - "knee": choose point closest to the ideal corner (P2.28)
        - "utility": choose point with best weighted goodness score (P2.28)

    If selection_method is knee/utility, we prefer selecting among Pareto rows
    when available.
    """
    cons = normalize_constraints(constraints)
    annotate_constraints(rows, cons)

    pareto_rows = [r for r in rows if bool(r.get(pareto_key))]
    candidates = pareto_rows if (prefer_pareto and pareto_rows) else list(rows)

    feasible = [r for r in candidates if bool(r.get("is_feasible"))] if cons else list(candidates)

    def _best(xx: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not xx:
            return None
        return max(xx, key=lambda r: _sf(r.get(objective_key), default=float("-inf")))

    method = str(selection_method or "best_objective").strip().lower()
    if method in {"objective", "best", "best_objective", ""}:
        method = "best_objective"
    elif method in {"knee", "knee_point", "closest", "closest_to_ideal"}:
        method = "knee"
    elif method in {"utility", "weighted", "weighted_utility"}:
        method = "utility"
    else:
        method = "best_objective"

    chosen: Optional[Dict[str, Any]] = None
    selected_by = "best_objective"

    # Keep the exact objective list / weights used (useful for reporting).
    obj_list_used: List[Tuple[str, str]] = list(objectives or [])
    if not obj_list_used:
        obj_list_used = [(str(objective_key), "max")]
    weights_used: Dict[str, float] = {}

    # Determine the base candidate set under constraints.
    base_set: List[Dict[str, Any]]
    if cons:
        if feasible:
            base_set = list(feasible)
            selected_by = "constrained_best" + ("_pareto" if (prefer_pareto and pareto_rows) else "")
        else:
            base_set = list(candidates)
            selected_by = "fallback_unconstrained_no_feasible"
    else:
        base_set = list(candidates)
        selected_by = "best_objective_pareto" if (prefer_pareto and pareto_rows) else "best_objective"

    # If using auto selection, prefer selecting among Pareto rows if present.
    if method in {"knee", "utility"}:
        pareto_base = [r for r in base_set if bool(r.get(pareto_key))]
        if pareto_base:
            base_set = pareto_base

    if method == "best_objective" or not base_set:
        chosen = _best(base_set)
    else:
        # Knee/utility selection require an objective list.
        obj_list = list(obj_list_used)

        ok_rows, goods = _normalize_goodness(base_set, obj_list)
        if goods.size == 0 or not ok_rows:
            chosen = _best(base_set)
        else:
            weights_map = utility_weights
            if method == "knee" and not weights_map:
                # Knee is intended as a balanced trade-off, so default to equal weights.
                weights_map = {k: 1.0 for k, _ in obj_list}
            w = _weights_for_objectives(obj_list, objective_key=str(objective_key), weights=weights_map)
            try:
                weights_used = {str(k): float(wi) for (k, _), wi in zip(obj_list, w)}
            except Exception:
                weights_used = {}
            if method == "knee":
                # Closest to ideal (1..1) in normalized goodness space.
                dist = np.sqrt(((1.0 - goods) ** 2 * w.reshape(1, -1)).sum(axis=1))
                j = int(np.argmin(dist))
                chosen = ok_rows[j]
                try:
                    chosen["knee_distance"] = float(dist[j])
                except Exception:
                    pass
            else:
                # Utility = weighted goodness.
                util = goods.dot(w)
                j = int(np.argmax(util))
                chosen = ok_rows[j]
                try:
                    chosen["utility_score"] = float(util[j])
                except Exception:
                    pass
            selected_by = f"{selected_by}_{method}"

    meta = {
        "constraints": cons,
        "prefer_pareto": bool(prefer_pareto),
        "candidate_count": int(len(candidates)),
        "pareto_count": int(len(pareto_rows)),
        "feasible_count": int(len(feasible)) if cons else int(len(candidates)),
        "selected_by": str(selected_by),
        "selection_method": str(method),
        "objectives": list(obj_list_used),
        "utility_weights_used": dict(weights_used),
    }
    return chosen, meta
