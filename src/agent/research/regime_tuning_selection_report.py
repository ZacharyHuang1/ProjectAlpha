"""agent.research.regime_tuning_selection_report

P2.29: Selection report for regime tuning.

Goal: produce a small, human-readable one-pager + a structured JSON payload
explaining *how* the final regime config was selected (constraints, Pareto,
method, and the key trade-offs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agent.research.constraint_selection import annotate_constraints, normalize_constraints, select_knee_row, select_utility_row


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return float(v) if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _parse_kv_float_map(s: str) -> Dict[str, float]:
    if not s:
        return {}
    out: Dict[str, float] = {}
    for part in str(s).split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out[str(k)] = float(fv)
    return out


def _constraint_slack(row: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, float]:
    """Compute constraint slack for a row.

    Slack is defined so that positive means "inside" the constraint.
    max_*: slack = max_value - actual
    min_*: slack = actual - min_value
    """

    out: Dict[str, float] = {}
    for k, thr in (constraints or {}).items():
        if thr is None:
            continue
        thr_f = _sf(thr, default=float("nan"))
        if not np.isfinite(thr_f):
            continue

        metric_key = str(k)
        sign = "max"
        if metric_key.startswith("max_"):
            metric_key = metric_key[len("max_") :]
            sign = "max"
        elif metric_key.startswith("min_"):
            metric_key = metric_key[len("min_") :]
            sign = "min"

        val_f = _sf(row.get(metric_key), default=float("nan"))
        if not np.isfinite(val_f):
            continue
        slack = (thr_f - val_f) if sign == "max" else (val_f - thr_f)
        out[str(k)] = float(slack)
    return out


def _select_base_set(
    rows: List[Dict[str, Any]],
    *,
    constraints: Optional[Dict[str, Any]],
    prefer_pareto: bool,
    pareto_key: str,
    selection_method: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Replicate the candidate-set logic used by select_best_row (for reporting)."""

    cons = normalize_constraints(constraints)
    annotate_constraints(rows, cons)

    pareto_rows = [r for r in rows if bool(r.get(pareto_key))]
    candidates = pareto_rows if bool(prefer_pareto and pareto_rows) else rows

    feasible = [r for r in candidates if bool(r.get("is_feasible", True))] if cons else list(candidates)
    base = feasible if (cons and feasible) else list(candidates)

    # For knee/utility we strongly prefer operating on the Pareto set when present.
    m = str(selection_method or "best_objective").strip().lower()
    if m in {"knee", "utility"}:
        pareto_base = [r for r in base if bool(r.get(pareto_key))]
        if pareto_base:
            base = pareto_base

    meta = {
        "constraints": cons,
        "prefer_pareto": bool(prefer_pareto),
        "candidate_count": int(len(candidates)),
        "feasible_count": int(len(feasible)) if cons else int(len(candidates)),
        "pareto_count": int(len(pareto_rows)),
        "base_set_size": int(len(base)),
    }
    return base, meta


def build_stage_selection_report(
    *,
    stage: str,
    rows: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    objective_key: str,
    pareto_key: str = "is_pareto",
    top_n: int = 10,
) -> Dict[str, Any]:
    """Build a compact selection report for a single stage."""

    top_n = int(max(1, top_n))
    work = [dict(r or {}) for r in (rows or []) if isinstance(r, dict)]

    constraints = dict(summary.get("constraints") or {})
    selection = dict(summary.get("selection") or {})
    prefer_pareto = bool(selection.get("prefer_pareto") or summary.get("prefer_pareto") or False)
    selection_method = str(selection.get("selection_method") or summary.get("selection_method") or "best_objective")
    objectives = list(summary.get("pareto_objectives") or selection.get("objectives") or [])
    if not objectives:
        objectives = [(str(objective_key), "max")]

    base_set, base_meta = _select_base_set(
        work,
        constraints=constraints,
        prefer_pareto=prefer_pareto,
        pareto_key=str(pareto_key),
        selection_method=selection_method,
    )

    m = str(selection_method or "best_objective").strip().lower()
    weights = dict(selection.get("utility_weights") or selection.get("utility_weights_used") or {})
    chosen = dict(summary.get("chosen") or {})

    # Compute selection scores for reporting.
    if m == "knee" and base_set:
        _ = select_knee_row(base_set, objectives=objectives, weights=weights or None)
        for r in base_set:
            r["selection_score"] = -_sf(r.get("knee_distance"), default=float("inf"))
    elif m == "utility" and base_set:
        _ = select_utility_row(base_set, objectives=objectives, weights=weights or None)
        for r in base_set:
            r["selection_score"] = _sf(r.get("utility_score"), default=float("-inf"))
    else:
        for r in base_set:
            r["selection_score"] = _sf(r.get(objective_key), default=float("-inf"))

    base_sorted = sorted(base_set, key=lambda d: _sf(d.get("selection_score"), default=float("-inf")), reverse=True)
    for i, r in enumerate(base_sorted):
        r["selection_rank"] = int(i + 1)

    def _row_id(r: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            str(r.get("config_id") or ""),
            str(r.get("mode") or ""),
            int(_sf(r.get("window"), default=0.0)),
            int(_sf(r.get("buckets"), default=0.0)),
            float(_sf(r.get("smoothing"), default=0.0)),
        )

    chosen_id = _row_id(chosen)

    top = []
    for r in base_sorted[:top_n]:
        rr = dict(r)
        rr["is_selected"] = bool(_row_id(rr) == chosen_id)
        top.append(rr)

    # Chosen constraint slack.
    slack = _constraint_slack(chosen, base_meta.get("constraints") or {}) if chosen else {}

    return {
        "stage": str(stage),
        "tune_metric": str(summary.get("tune_metric") or ""),
        "turnover_cost_bps": _sf(summary.get("turnover_cost_bps"), default=_sf(summary.get("turnover_penalty"))),
        "objective_key": str(objective_key),
        "selection_method": str(selection_method),
        "prefer_pareto": bool(prefer_pareto),
        "constraints": dict(base_meta.get("constraints") or {}),
        "counts": {
            "total_rows": int(len(work)),
            "candidate_count": int(base_meta.get("candidate_count") or 0),
            "feasible_count": int(base_meta.get("feasible_count") or 0),
            "pareto_count": int(base_meta.get("pareto_count") or 0),
            "base_set_size": int(base_meta.get("base_set_size") or 0),
        },
        "pareto_objectives": list(objectives),
        "weights": dict(weights),
        "chosen": dict(chosen),
        "chosen_constraint_slack": dict(slack),
        "top_candidates": top,
    }


def build_regime_tuning_selection_report(
    *,
    config: Dict[str, Any],
    proxy_rows: Sequence[Dict[str, Any]],
    proxy_summary: Dict[str, Any],
    holdings_rows: Optional[Sequence[Dict[str, Any]]] = None,
    holdings_summary: Optional[Dict[str, Any]] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Build the combined (proxy + holdings) selection report."""

    cfg = dict(config.get("configurable") or {}) if isinstance(config, dict) else {}
    preset = str(cfg.get("alpha_allocation_regime_tune_preset") or "")
    weight_src = str(cfg.get("alpha_allocation_regime_tune_utility_weights_source") or "")
    weight_str = str(cfg.get("alpha_allocation_regime_tune_utility_weights") or "")
    weight_meta = dict(cfg.get("alpha_allocation_regime_tune_utility_calibration_meta") or {})

    weights = _parse_kv_float_map(weight_str)

    proxy_stage = build_stage_selection_report(
        stage="proxy",
        rows=list(proxy_rows or []),
        summary=dict(proxy_summary or {}),
        objective_key="objective",
        top_n=int(top_n),
    )
    proxy_stage["weights"] = dict(proxy_stage.get("weights") or weights)

    holdings_stage: Optional[Dict[str, Any]] = None
    if holdings_rows is not None and holdings_summary is not None and list(holdings_rows or []):
        holdings_stage = build_stage_selection_report(
            stage="holdings_valid",
            rows=list(holdings_rows or []),
            summary=dict(holdings_summary or {}),
            objective_key="holdings_objective",
            top_n=int(top_n),
        )
        holdings_stage["weights"] = dict(holdings_stage.get("weights") or weights)

    final_stage = "holdings_valid" if holdings_stage and bool((holdings_summary or {}).get("enabled")) else "proxy"

    return {
        "preset": preset,
        "utility_weights_source": weight_src,
        "utility_weights": dict(weights),
        "utility_calibration_meta": dict(weight_meta),
        "final_stage": str(final_stage),
        "proxy": proxy_stage,
        "holdings_valid": holdings_stage,
    }


def _fmt(x: Any, nd: int = 4) -> str:
    v = _sf(x, default=float("nan"))
    if not np.isfinite(v):
        return ""
    try:
        return f"{v:.{int(max(0, nd))}f}"
    except Exception:
        return str(v)


def _md_table(rows: Sequence[Dict[str, Any]], cols: Sequence[Tuple[str, str]]) -> str:
    if not rows:
        return ""

    headers = [c[1] for c in cols]
    keys = [c[0] for c in cols]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        vals = []
        for k in keys:
            if k in {"mode", "config_id"}:
                vals.append(str(r.get(k) or ""))
            elif k in {"is_pareto", "is_feasible", "is_selected"}:
                vals.append("yes" if bool(r.get(k)) else "no")
            elif k in {"window", "buckets"}:
                vals.append(str(int(_sf(r.get(k), default=0.0)) or ""))
            else:
                vals.append(_fmt(r.get(k)))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def render_regime_tuning_report_md(report: Dict[str, Any]) -> str:
    """Render a one-page Markdown report."""

    if not isinstance(report, dict):
        return ""

    preset = str(report.get("preset") or "")
    final_stage = str(report.get("final_stage") or "proxy")
    w_src = str(report.get("utility_weights_source") or "")
    w = dict(report.get("utility_weights") or {})

    lines: List[str] = []
    lines.append("# Regime Tuning Report")
    lines.append("")
    if preset:
        lines.append(f"- Preset: `{preset}`")
    lines.append(f"- Final selection stage: `{final_stage}`")
    if w:
        lines.append(f"- Utility weights source: `{w_src or 'unknown'}`")
        lines.append(f"- Utility weights: `{', '.join([f'{k}={w[k]:.3g}' for k in sorted(w.keys())])}`")
    lines.append("")

    def _stage_block(stage_key: str, title: str) -> None:
        st = report.get(stage_key)
        if not isinstance(st, dict):
            return
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"- Selection method: `{st.get('selection_method')}`")
        lines.append(f"- Prefer Pareto: `{bool(st.get('prefer_pareto'))}`")
        if str(st.get("tune_metric") or ""):
            lines.append(f"- Tune metric: `{st.get('tune_metric')}`")
        tc = _sf(st.get("turnover_cost_bps"), default=float("nan"))
        if np.isfinite(tc):
            lines.append(f"- Turnover cost (bps): {_fmt(tc, nd=4)}")
        cnt = st.get("counts") or {}
        if isinstance(cnt, dict):
            lines.append(
                "- Counts: "
                f"total={int(cnt.get('total_rows') or 0)}, "
                f"pareto={int(cnt.get('pareto_count') or 0)}, "
                f"feasible={int(cnt.get('feasible_count') or 0)}, "
                f"base_set={int(cnt.get('base_set_size') or 0)}"
            )

        chosen = st.get("chosen") or {}
        if isinstance(chosen, dict) and chosen:
            lines.append("")
            lines.append("**Chosen config**")
            lines.append("")
            lines.append(
                f"- mode={chosen.get('mode')}, window={chosen.get('window')}, buckets={chosen.get('buckets')}, smoothing={chosen.get('smoothing')}"
            )
            lines.append(
                f"- {st.get('objective_key')}: {_fmt(chosen.get(st.get('objective_key')), nd=4)}"
            )
            if "alpha_weight_turnover_mean" in chosen:
                lines.append(f"- alpha_weight_turnover_mean: {_fmt(chosen.get('alpha_weight_turnover_mean'))}")
            if "turnover_cost_drag_bps_mean" in chosen:
                lines.append(f"- turnover_cost_drag_bps_mean: {_fmt(chosen.get('turnover_cost_drag_bps_mean'))}")
            if "regime_switch_rate_mean" in chosen:
                lines.append(f"- regime_switch_rate_mean: {_fmt(chosen.get('regime_switch_rate_mean'))}")
            if "fallback_frac_mean" in chosen:
                lines.append(f"- fallback_frac_mean: {_fmt(chosen.get('fallback_frac_mean'))}")

        slack = st.get("chosen_constraint_slack") or {}
        if isinstance(slack, dict) and slack:
            lines.append("")
            lines.append("**Constraint slack (positive is good)**")
            lines.append("")
            for k in sorted(slack.keys()):
                lines.append(f"- {k}: {_fmt(slack[k])}")

        top = st.get("top_candidates") or []
        if isinstance(top, list) and top:
            lines.append("")
            lines.append("**Top candidates (selection set)**")
            lines.append("")
            cols = [
                ("is_selected", "selected"),
                ("config_id", "config"),
                ("mode", "mode"),
                ("window", "win"),
                ("buckets", "bkt"),
                ("smoothing", "smooth"),
                (st.get("objective_key") or "objective", "obj"),
                ("alpha_weight_turnover_mean", "alpha_to"),
                ("turnover_cost_drag_bps_mean", "drag_bps"),
                ("regime_switch_rate_mean", "switch"),
                ("fallback_frac_mean", "fallback"),
                ("selection_rank", "rank"),
            ]
            lines.append(_md_table(top[:8], cols))

        lines.append("")

    _stage_block("proxy", "Proxy-stage selection")
    _stage_block("holdings_valid", "Holdings-validation selection")

    return "\n".join(lines).strip() + "\n"
