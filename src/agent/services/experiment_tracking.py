"""agent.services.experiment_tracking

P2.3: Local experiment tracking.

This module writes run artifacts to disk so research runs are:
- reproducible (config + outputs are saved)
- comparable (a stable metrics table is exported)

The tracking is intentionally filesystem-first (no external service required).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent.research.reporting import make_run_report_md
from agent.research.pareto_plotting import try_make_pareto_scatter
from agent.research.regime_tuning_selection_report import (
    build_regime_tuning_selection_report,
    render_regime_tuning_report_md,
)
from agent.research.alpha_selection import (
    build_alpha_selection_report,
    render_alpha_selection_report_md,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_dumps_stable(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def make_run_id(thread_id: str, config: Dict[str, Any]) -> str:
    """Create a readable run_id with a short config hash."""

    ts = _utc_timestamp()
    safe_thread = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in (thread_id or "thread"))
    safe_thread = safe_thread[:32] if safe_thread else "thread"

    cfg_str = _json_dumps_stable(config)
    h = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:10]
    return f"{ts}_{safe_thread}_{h}"


def _to_plain(obj: Any) -> Any:
    """Convert nested structures to JSON-friendly python types."""

    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}

    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]

    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)

    return obj


def extract_alpha_metrics_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha metrics into a stable table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
        stab = (wf.get("stability") or {}) if isinstance(wf, dict) else {}

        row: Dict[str, Any] = {
            "alpha_id": aid,
            "mode": m.get("mode"),
            "information_ratio": m.get("information_ratio"),
            "annualized_return": m.get("annualized_return"),
            "max_drawdown": m.get("max_drawdown"),
            "turnover_mean": m.get("turnover_mean"),
            "coverage_mean": m.get("coverage_mean"),
            "ic": m.get("ic"),
            "rank_ic": m.get("rank_ic"),
            "spread_mean": m.get("spread_mean"),
            "spread_tstat": m.get("spread_tstat"),
            "ic_tstat": m.get("ic_tstat"),
            "rank_ic_tstat": m.get("rank_ic_tstat"),
            "wf_n_splits": stab.get("n_splits"),
            "wf_test_ir_mean": stab.get("test_ir_mean"),
            "wf_test_ir_std": stab.get("test_ir_std"),
            "wf_test_ir_positive_frac": stab.get("test_ir_positive_frac"),
            "wf_generalization_gap": stab.get("generalization_gap"),
        }

        qg = m.get("quality_gate") if isinstance(m, dict) else None
        if isinstance(qg, dict):
            row["quality_gate_passed"] = bool(qg.get("passed"))
            row["quality_gate_reasons"] = ",".join(list(qg.get("reasons") or []))
        else:
            row["quality_gate_passed"] = True
            row["quality_gate_reasons"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Sort with NaNs at the bottom.
    df["information_ratio"] = pd.to_numeric(df["information_ratio"], errors="coerce")
    df = df.sort_values(by=["information_ratio"], ascending=False, na_position="last")
    return df




def extract_sweep_results_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha tuning sweep results into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        tuning = (m.get("tuning") or {}) if isinstance(m, dict) else {}
        sweep = (tuning.get("sweep") or {}) if isinstance(tuning, dict) else {}
        results = sweep.get("results") or []
        if not isinstance(results, list):
            continue

        for r in results:
            if not isinstance(r, dict):
                continue
            params = r.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            row = {
                "alpha_id": aid,
                "config_id": r.get("config_id"),
                "passed": r.get("passed"),
                "score": r.get("score"),
                "raw_score": r.get("raw_score"),
                "information_ratio": r.get("information_ratio"),
                "annualized_return": r.get("annualized_return"),
                "max_drawdown": r.get("max_drawdown"),
                "turnover_mean": r.get("turnover_mean"),
                "total_cost_bps": r.get("total_cost_bps"),
                "optimizer_turnover_cap": params.get("optimizer_turnover_cap"),
                "max_abs_weight": params.get("max_abs_weight"),
                "optimizer_risk_aversion": params.get("optimizer_risk_aversion"),
                "optimizer_cost_aversion": params.get("optimizer_cost_aversion"),
            }
            # Keep errors visible if present.
            if r.get("error"):
                row["error"] = r.get("error")
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.sort_values(by=["alpha_id", "score"], ascending=[True, False], na_position="last")
    return df


def extract_ablation_results_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha cost ablation scenarios into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        tuning = (m.get("tuning") or {}) if isinstance(m, dict) else {}
        ab = (tuning.get("ablation") or {}) if isinstance(tuning, dict) else {}
        if not isinstance(ab, dict) or not ab:
            continue

        def _add_rows(mode: str, scenarios: Any) -> None:
            if not isinstance(scenarios, list):
                return
            for sc in scenarios:
                if not isinstance(sc, dict):
                    continue
                rows.append(
                    {
                        "alpha_id": aid,
                        "ablation_mode": str(mode),
                        "scenario": sc.get("scenario"),
                        "information_ratio": sc.get("information_ratio"),
                        "annualized_return": sc.get("annualized_return"),
                        "max_drawdown": sc.get("max_drawdown"),
                        "turnover_mean": sc.get("turnover_mean"),
                        "total_cost_bps": sc.get("total_cost_bps"),
                        "mean_cost_drag_bps": sc.get("mean_cost_drag_bps"),
                        "error": sc.get("error"),
                    }
                )

        # Backward compatibility: P2.12 stored scenarios at the top level.
        if isinstance(ab.get("scenarios"), list):
            _add_rows("end_to_end", ab.get("scenarios"))
            continue

        e2e = ab.get("end_to_end") or {}
        exe = ab.get("execution_only") or {}
        _add_rows("end_to_end", (e2e.get("scenarios") if isinstance(e2e, dict) else None) or [])
        _add_rows("execution_only", (exe.get("scenarios") if isinstance(exe, dict) else None) or [])

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["information_ratio"] = pd.to_numeric(df["information_ratio"], errors="coerce")
    df = df.sort_values(by=["alpha_id", "ablation_mode", "scenario"], ascending=[True, True, True])
    return df

def _best_alpha_from_table(df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
    if df is None or df.empty:
        return None, {}
    row = df.iloc[0].to_dict()
    return str(row.get("alpha_id")), row


def extract_regime_analysis_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha regime analysis buckets into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
        regimes = (analysis.get("regime") or {}) if isinstance(analysis, dict) else {}
        if not isinstance(regimes, dict) or not regimes:
            continue

        for name, buckets in regimes.items():
            if not isinstance(buckets, list):
                continue
            for b in buckets:
                if not isinstance(b, dict):
                    continue
                rows.append(
                    {
                        "alpha_id": aid,
                        "regime": str(name),
                        "bucket": b.get("bucket"),
                        "information_ratio": b.get("information_ratio"),
                        "annualized_return": b.get("annualized_return"),
                        "max_drawdown": b.get("max_drawdown"),
                        "n_obs": b.get("n_obs"),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["information_ratio"] = pd.to_numeric(df["information_ratio"], errors="coerce")
    return df.sort_values(by=["alpha_id", "regime", "bucket"], ascending=[True, True, True])


def extract_cost_sensitivity_curves_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha cost sensitivity curves into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
        cs = (analysis.get("cost_sensitivity") or {}) if isinstance(analysis, dict) else {}
        curves = cs.get("curves") or []
        if not isinstance(curves, list):
            continue
        for r in curves:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "alpha_id": aid,
                    "parameter": r.get("parameter"),
                    "value": r.get("value"),
                    "information_ratio": r.get("information_ratio"),
                    "annualized_return": r.get("annualized_return"),
                    "max_drawdown": r.get("max_drawdown"),
                    "n_obs": r.get("n_obs"),
                    "mean_cost_drag_bps": r.get("mean_cost_drag_bps"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["information_ratio"] = pd.to_numeric(df["information_ratio"], errors="coerce")
    return df.sort_values(by=["alpha_id", "parameter", "value"], ascending=[True, True, True])


def extract_cost_sensitivity_break_even_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha break-even summaries into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
        cs = (analysis.get("cost_sensitivity") or {}) if isinstance(analysis, dict) else {}
        be = cs.get("break_even") or []
        if not isinstance(be, list):
            continue
        borrow_mode = cs.get("borrow_mode")
        for r in be:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "alpha_id": aid,
                    "parameter": r.get("parameter"),
                    "break_even": r.get("break_even"),
                    "within_grid": r.get("within_grid"),
                    "status": r.get("status"),
                    "min": r.get("min"),
                    "max": r.get("max"),
                    "borrow_mode": borrow_mode,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(by=["alpha_id", "parameter"], ascending=[True, True])


def extract_decay_analysis_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha horizon decay metrics into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
        decay = (analysis.get("decay") or {}) if isinstance(analysis, dict) else {}
        metrics = decay.get("metrics") or []
        if not isinstance(metrics, list):
            continue

        for r in metrics:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "alpha_id": aid,
                    "horizon": r.get("horizon"),
                    "n_days": r.get("n_days"),
                    "ic_mean": r.get("ic_mean"),
                    "ic_tstat": r.get("ic_tstat"),
                    "rank_ic_mean": r.get("rank_ic_mean"),
                    "rank_ic_tstat": r.get("rank_ic_tstat"),
                    "spread_mean": r.get("spread_mean"),
                    "spread_tstat": r.get("spread_tstat"),
                    "spread_ir_ann_proxy": r.get("spread_ir_ann_proxy"),
                    "coverage_mean": r.get("coverage_mean"),
                    "signal_overlap_mean": r.get("signal_overlap_mean"),
                    "signal_pairs": r.get("signal_pairs"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    return df.sort_values(by=["alpha_id", "horizon"], ascending=[True, True], na_position="last")


def extract_schedule_sweep_table(result: Any) -> pd.DataFrame:
    """Flatten per-alpha holding/rebalance schedule sweep results into a single table."""

    def _get(o: Any, k: str, default: Any = None) -> Any:
        return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)

    coded_alphas = _get(result, "coded_alphas", []) or []
    rows: List[Dict[str, Any]] = []

    for a in coded_alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        m = a.get("backtest_results") or {}
        analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
        ss = (analysis.get("schedule_sweep") or {}) if isinstance(analysis, dict) else {}
        results = ss.get("results") or []
        if not isinstance(results, list) or not results:
            continue

        for r in results:
            if not isinstance(r, dict):
                continue
            rows.append(
                {
                    "alpha_id": aid,
                    "rebalance_days": r.get("rebalance_days"),
                    "holding_days": r.get("holding_days"),
                    "max_active": r.get("max_active"),
                    "overlap_ratio": r.get("overlap_ratio"),
                    "information_ratio": r.get("information_ratio"),
                    "annualized_return": r.get("annualized_return"),
                    "max_drawdown": r.get("max_drawdown"),
                    "turnover_mean": r.get("turnover_mean"),
                    "total_cost_bps": r.get("total_cost_bps"),
                    "passed": r.get("passed"),
                    "reasons": ",".join(list(r.get("reasons") or [])) if isinstance(r.get("reasons"), list) else r.get("reasons"),
                    "error": r.get("error"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["information_ratio"] = pd.to_numeric(df["information_ratio"], errors="coerce")
    df = df.sort_values(by=["alpha_id", "information_ratio"], ascending=[True, False], na_position="last")
    return df


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_plain(obj), f, ensure_ascii=False, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_summary_md(
    *,
    run_id: str,
    thread_id: str,
    config: Dict[str, Any],
    metrics: pd.DataFrame,
    sota_alphas: List[Dict[str, Any]],
) -> str:
    eval_mode = str(config.get("eval_mode") or config.get("eval-mode") or "")
    data_path = str(config.get("data_path") or "")

    best_id, best_row = _best_alpha_from_table(metrics)

    def _fmt(v: Any) -> str:
        try:
            fv = float(v)
            return f"{fv:.4f}" if np.isfinite(fv) else "na"
        except Exception:
            return "na"

    lines: List[str] = []
    lines.append(f"# Run summary: `{run_id}`\n")
    lines.append("## Metadata\n")
    lines.append(f"- thread_id: `{thread_id}`\n")
    lines.append(f"- eval_mode: `{eval_mode}`\n")
    lines.append(f"- data_path: `{data_path or 'synthetic'}`\n")

    if best_id:
        lines.append("\n## Best alpha\n")
        lines.append(f"- alpha_id: `{best_id}`\n")
        lines.append(f"- information_ratio: `{_fmt(best_row.get('information_ratio'))}`\n")
        lines.append(f"- annualized_return: `{_fmt(best_row.get('annualized_return'))}`\n")
        lines.append(f"- max_drawdown: `{_fmt(best_row.get('max_drawdown'))}`\n")
        lines.append(f"- turnover_mean: `{_fmt(best_row.get('turnover_mean'))}`\n")

    # Small table for top factors.
    if metrics is not None and not metrics.empty:
        top = metrics.head(min(10, len(metrics))).copy()
        cols = [
            "alpha_id",
            "mode",
            "information_ratio",
            "annualized_return",
            "max_drawdown",
            "turnover_mean",
            "coverage_mean",
        ]
        top = top[[c for c in cols if c in top.columns]]
        lines.append("\n## Top metrics (first 10)\n")
        lines.append(top.to_markdown(index=False))
        lines.append("\n")

    if sota_alphas:
        lines.append("\n## Selected SOTA alphas\n")
        for a in sota_alphas:
            aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
            dsl = a.get("dsl") or a.get("code") or ""
            lines.append(f"- `{aid}`: `{dsl[:160]}{'...' if len(dsl) > 160 else ''}`\n")

    return "".join(lines)


def _append_index(runs_root: Path, record: Dict[str, Any]) -> None:
    idx = runs_root / "_index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    with idx.open("a", encoding="utf-8") as f:
        f.write(_json_dumps_stable(record))
        f.write("\n")


def _append_factor_registry(runs_root: Path, run_id: str, coded_alphas: List[Dict[str, Any]]) -> None:
    """Append coded alphas to a local registry.

    The registry is an append-only JSONL file for quick lookup and dedup later.
    """

    reg = runs_root / "factor_registry.jsonl"
    reg.parent.mkdir(parents=True, exist_ok=True)

    ts = _utc_timestamp()
    with reg.open("a", encoding="utf-8") as f:
        for a in coded_alphas or []:
            if not isinstance(a, dict):
                continue
            aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
            dsl = a.get("dsl") or a.get("code") or ""
            desc = a.get("description") or a.get("desc") or ""
            m = a.get("backtest_results") or {}
            dsl_hash = hashlib.sha256(str(dsl).encode("utf-8")).hexdigest()[:16] if dsl else ""

            rec = {
                "timestamp_utc": ts,
                "run_id": run_id,
                "alpha_id": aid,
                "dsl_hash": dsl_hash,
                "dsl": dsl,
                "description": desc,
                "mode": m.get("mode") if isinstance(m, dict) else None,
                "information_ratio": (m.get("information_ratio") if isinstance(m, dict) else None),
                "annualized_return": (m.get("annualized_return") if isinstance(m, dict) else None),
                "turnover_mean": (m.get("turnover_mean") if isinstance(m, dict) else None),
                "coverage_mean": (m.get("coverage_mean") if isinstance(m, dict) else None),
            }
            f.write(_json_dumps_stable(rec))
            f.write("\n")


def save_run_artifacts(
    *,
    runs_root: str | Path,
    run_id: str,
    thread_id: str,
    config: Dict[str, Any],
    result: Any,
    save_daily_top: int = 1,
) -> Path:
    """Save artifacts for a single run.

    Returns the run directory path.
    """

    root = Path(runs_root).expanduser().resolve()
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Stable config and full payload
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "result.json", result)

    # 2) Metrics table for all coded alphas
    metrics = extract_alpha_metrics_table(result)
    if not metrics.empty:
        metrics.to_csv(run_dir / "alpha_metrics.csv", index=False)

    # 2b) Optional tuning artifacts (P2.12)
    sweep_df = extract_sweep_results_table(result)
    if not sweep_df.empty:
        sweep_df.to_csv(run_dir / "sweep_results.csv", index=False)

    ablation_df = extract_ablation_results_table(result)
    if not ablation_df.empty:
        ablation_df.to_csv(run_dir / "ablation_results.csv", index=False)

    regime_df = extract_regime_analysis_table(result)
    if not regime_df.empty:
        regime_df.to_csv(run_dir / "regime_analysis.csv", index=False)

    cs_curve_df = extract_cost_sensitivity_curves_table(result)
    if not cs_curve_df.empty:
        cs_curve_df.to_csv(run_dir / "cost_sensitivity.csv", index=False)

    cs_be_df = extract_cost_sensitivity_break_even_table(result)
    if not cs_be_df.empty:
        cs_be_df.to_csv(run_dir / "cost_sensitivity_break_even.csv", index=False)

    decay_df = extract_decay_analysis_table(result)
    if not decay_df.empty:
        decay_df.to_csv(run_dir / "decay_analysis.csv", index=False)

    schedule_sweep_df = extract_schedule_sweep_table(result)
    if not schedule_sweep_df.empty:
        schedule_sweep_df.to_csv(run_dir / "schedule_sweep.csv", index=False)

    # 3) SOTA-only payload
    sota_alphas = []
    if isinstance(result, dict):
        sota_alphas = list(result.get("sota_alphas") or [])
    else:
        sota_alphas = list(getattr(result, "sota_alphas", []) or [])
    _write_json(run_dir / "sota_alphas.json", sota_alphas)

    # P2.30: a one-page report explaining alpha selection.
    try:
        sel_rep = build_alpha_selection_report(config=config, result=result, top_n=20)
        _write_json(run_dir / "alpha_selection_report.json", sel_rep)
        _write_text(run_dir / "ALPHA_SELECTION_REPORT.md", render_alpha_selection_report_md(sel_rep))

        top = ((sel_rep.get("diagnostics") or {}).get("top_candidates") or []) if isinstance(sel_rep, dict) else []
        if isinstance(top, list) and top:
            pd.DataFrame(top).to_csv(run_dir / "alpha_selection_top_candidates.csv", index=False)
    except Exception:
        # Best-effort: report generation should never fail the run.
        pass

    # 3b) P2.17: correlation matrix and ensemble artifacts (if available)
    try:
        if isinstance(result, dict):
            ac = result.get("alpha_correlation")
        else:
            ac = getattr(result, "alpha_correlation", None)
        if isinstance(ac, dict) and isinstance(ac.get("matrix"), dict):
            m = ac.get("matrix") or {}
            if isinstance(m, dict) and {"data", "index", "columns"}.issubset(set(m.keys())):
                dfc = pd.DataFrame(data=m.get("data"), index=m.get("index"), columns=m.get("columns"))
                dfc.to_csv(run_dir / "alpha_correlation.csv")
            n = ac.get("nobs") or {}
            if isinstance(n, dict) and {"data", "index", "columns"}.issubset(set(n.keys())):
                dfn = pd.DataFrame(data=n.get("data"), index=n.get("index"), columns=n.get("columns"))
                dfn.to_csv(run_dir / "alpha_correlation_nobs.csv")
    except Exception:
        pass

    try:
        if isinstance(result, dict):
            ens = result.get("ensemble")
        else:
            ens = getattr(result, "ensemble", None)
        if isinstance(ens, dict) and bool(ens.get("enabled")):
            _write_json(run_dir / "ensemble_metrics.json", ens.get("metrics") or {})
            daily = ens.get("daily") or []
            if isinstance(daily, list) and daily:
                pd.DataFrame(daily).to_csv(run_dir / "ensemble_oos_daily.csv", index=False)
    except Exception:
        pass

    # P2.18: holdings-level ensemble artifacts.
    try:
        if isinstance(result, dict):
            eh = result.get("ensemble_holdings")
        else:
            eh = getattr(result, "ensemble_holdings", None)
        if isinstance(eh, dict) and bool(eh.get("enabled")):
            _write_json(run_dir / "ensemble_holdings_metrics.json", eh.get("metrics") or {})
            daily = eh.get("daily") or []
            if isinstance(daily, list) and daily:
                pd.DataFrame(daily).to_csv(run_dir / "ensemble_holdings_oos_daily.csv", index=False)
            pos = eh.get("positions") or []
            if isinstance(pos, list) and pos:
                pd.DataFrame(pos).to_csv(run_dir / "ensemble_holdings_positions.csv", index=False)
    except Exception:
        pass

    
    # P2.22: selection meta-tuning artifacts.
    try:
        if isinstance(result, dict):
            st = result.get("selection_tuning")
        else:
            st = getattr(result, "selection_tuning", None)
        if isinstance(st, dict) and bool(st.get("enabled")):
            _write_json(run_dir / "selection_tuning_summary.json", st)
            rows = st.get("results") or []
            if isinstance(rows, list) and rows:
                dfst = pd.DataFrame(rows)
                # Stabilize list/dict columns for CSV export.
                for c in list(dfst.columns):
                    try:
                        if dfst[c].apply(lambda x: isinstance(x, (list, dict))).any():
                            dfst[c] = dfst[c].apply(
                                lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
                                if isinstance(x, (list, dict))
                                else x
                            )
                    except Exception:
                        continue
                dfst.to_csv(run_dir / "selection_tuning_results.csv", index=False)
    except Exception:
        pass

    # P2.19: allocated holdings ensemble artifacts.
    try:
        if isinstance(result, dict):
            eha = result.get("ensemble_holdings_allocated")
        else:
            eha = getattr(result, "ensemble_holdings_allocated", None)
        if isinstance(eha, dict) and bool(eha.get("enabled")):
            _write_json(run_dir / "ensemble_holdings_allocated_metrics.json", eha.get("metrics") or {})
            daily = eha.get("daily") or []
            if isinstance(daily, list) and daily:
                pd.DataFrame(daily).to_csv(run_dir / "ensemble_holdings_allocated_oos_daily.csv", index=False)
            pos = eha.get("positions") or []
            if isinstance(pos, list) and pos:
                pd.DataFrame(pos).to_csv(run_dir / "ensemble_holdings_allocated_positions.csv", index=False)
            alloc = eha.get("allocations") or []
            if isinstance(alloc, list) and alloc:
                pd.DataFrame(alloc).to_csv(run_dir / "alpha_allocations.csv", index=False)
            diag = eha.get("allocation_diagnostics") or []
            if isinstance(diag, list) and diag:
                pd.DataFrame(diag).to_csv(run_dir / "alpha_allocation_diagnostics.csv", index=False)

            # P2.20: meta-tuning table (valid performance across splits).
            tune_rows = eha.get("allocation_tuning_results") or []
            if isinstance(tune_rows, list) and tune_rows:
                pd.DataFrame(tune_rows).to_csv(run_dir / "alpha_allocation_tuning_results.csv", index=False)
            tune_summary = eha.get("allocation_tuning") or {}
            if isinstance(tune_summary, dict) and tune_summary:
                _write_json(run_dir / "alpha_allocation_tuning_summary.json", tune_summary)
    except Exception:
        pass

    # P2.21: regime-aware allocated holdings ensemble artifacts.
    try:
        if isinstance(result, dict):
            ehr = result.get("ensemble_holdings_allocated_regime")
        else:
            ehr = getattr(result, "ensemble_holdings_allocated_regime", None)
        if isinstance(ehr, dict) and bool(ehr.get("enabled")):
            _write_json(run_dir / "ensemble_holdings_allocated_regime_metrics.json", ehr.get("metrics") or {})
            daily = ehr.get("daily") or []
            if isinstance(daily, list) and daily:
                pd.DataFrame(daily).to_csv(run_dir / "ensemble_holdings_allocated_regime_oos_daily.csv", index=False)
            pos = ehr.get("positions") or []
            if isinstance(pos, list) and pos:
                pd.DataFrame(pos).to_csv(run_dir / "ensemble_holdings_allocated_regime_positions.csv", index=False)
            alloc = ehr.get("allocations_regime") or []
            if isinstance(alloc, list) and alloc:
                pd.DataFrame(alloc).to_csv(run_dir / "alpha_allocations_regime.csv", index=False)
            diag = ehr.get("allocation_regime_diagnostics") or []
            if isinstance(diag, list) and diag:
                pd.DataFrame(diag).to_csv(run_dir / "alpha_allocation_regime_diagnostics.csv", index=False)

            tune_rows = ehr.get("allocation_tuning_results") or []
            if isinstance(tune_rows, list) and tune_rows:
                pd.DataFrame(tune_rows).to_csv(run_dir / "alpha_allocation_regime_tuning_results.csv", index=False)
            tune_summary = ehr.get("allocation_tuning") or {}
            if isinstance(tune_summary, dict) and tune_summary:
                _write_json(run_dir / "alpha_allocation_regime_tuning_summary.json", tune_summary)

            # P2.23: regime hyperparam tuning table.
            reg_rows = ehr.get("regime_tuning_results") or []
            if isinstance(reg_rows, list) and reg_rows:
                pd.DataFrame(reg_rows).to_csv(run_dir / "alpha_allocation_regime_param_tuning_results.csv", index=False)
            reg_summary = ehr.get("regime_tuning") or {}
            if isinstance(reg_summary, dict) and reg_summary:
                _write_json(run_dir / "alpha_allocation_regime_param_tuning_summary.json", reg_summary)

            # P2.27: Pareto subsets + optional plots.
            try:
                cfg = config.get("configurable") if isinstance(config, dict) else {}
                plots_enabled = bool((cfg or {}).get("alpha_allocation_regime_tune_plots", True))
            except Exception:
                plots_enabled = True

            if isinstance(reg_rows, list) and reg_rows:
                try:
                    df_reg = pd.DataFrame(reg_rows)
                    if "is_pareto" in df_reg.columns:
                        df_p = df_reg[df_reg["is_pareto"] == True]
                        if not df_p.empty:
                            df_p.to_csv(run_dir / "alpha_allocation_regime_param_tuning_pareto.csv", index=False)
                except Exception:
                    pass

                if plots_enabled:
                    chosen = reg_summary.get("chosen") if isinstance(reg_summary, dict) else None
                    plot_meta: Dict[str, Any] = {}
                    try:
                        plot_meta["turnover"] = try_make_pareto_scatter(
                            rows=list(reg_rows),
                            x_key="alpha_weight_turnover_mean",
                            y_key="objective",
                            chosen=chosen if isinstance(chosen, dict) else None,
                            out_path=run_dir / "alpha_allocation_regime_param_tuning_pareto_turnover.png",
                            title="Regime tuning (proxy): objective vs alpha-weight turnover",
                        )
                    except Exception as e:
                        plot_meta["turnover"] = {"enabled": False, "error": str(e)}
                    try:
                        if any("turnover_cost_drag_bps_mean" in (r or {}) for r in reg_rows):
                            plot_meta["drag"] = try_make_pareto_scatter(
                                rows=list(reg_rows),
                                x_key="turnover_cost_drag_bps_mean",
                                y_key="objective",
                                chosen=chosen if isinstance(chosen, dict) else None,
                                out_path=run_dir / "alpha_allocation_regime_param_tuning_pareto_drag.png",
                                title="Regime tuning (proxy): objective vs turnover-cost drag (bps)",
                            )
                    except Exception as e:
                        plot_meta["drag"] = {"enabled": False, "error": str(e)}
                    if plot_meta:
                        _write_json(run_dir / "alpha_allocation_regime_param_tuning_pareto_plots.json", plot_meta)

            # P2.24: holdings-level revalidation for top regime configs.
            hold_rows = ehr.get("regime_tuning_holdings_validation_results") or []
            if isinstance(hold_rows, list) and hold_rows:
                pd.DataFrame(hold_rows).to_csv(
                    run_dir / "alpha_allocation_regime_holdings_validation_results.csv", index=False
                )
            hold_summary = ehr.get("regime_tuning_holdings_validation") or {}
            if isinstance(hold_summary, dict) and hold_summary:
                _write_json(run_dir / "alpha_allocation_regime_holdings_validation_summary.json", hold_summary)

            # P2.27: Pareto subsets + optional plots (holdings validation).
            if isinstance(hold_rows, list) and hold_rows:
                try:
                    df_hold = pd.DataFrame(hold_rows)
                    if "is_pareto" in df_hold.columns:
                        df_p = df_hold[df_hold["is_pareto"] == True]
                        if not df_p.empty:
                            df_p.to_csv(
                                run_dir / "alpha_allocation_regime_holdings_validation_pareto.csv", index=False
                            )
                except Exception:
                    pass

                if plots_enabled:
                    chosen_h = hold_summary.get("chosen") if isinstance(hold_summary, dict) else None
                    plot_meta_h: Dict[str, Any] = {}
                    try:
                        plot_meta_h["turnover"] = try_make_pareto_scatter(
                            rows=list(hold_rows),
                            x_key="alpha_weight_turnover_mean",
                            y_key="holdings_objective",
                            chosen=chosen_h if isinstance(chosen_h, dict) else None,
                            out_path=run_dir / "alpha_allocation_regime_holdings_validation_pareto_turnover.png",
                            title="Regime tuning (holdings): objective vs alpha-weight turnover",
                        )
                    except Exception as e:
                        plot_meta_h["turnover"] = {"enabled": False, "error": str(e)}

                    try:
                        if any("ensemble_cost_mean" in (r or {}) for r in hold_rows):
                            plot_meta_h["ensemble_cost"] = try_make_pareto_scatter(
                                rows=list(hold_rows),
                                x_key="ensemble_cost_mean",
                                y_key="holdings_objective",
                                chosen=chosen_h if isinstance(chosen_h, dict) else None,
                                out_path=run_dir / "alpha_allocation_regime_holdings_validation_pareto_cost.png",
                                title="Regime tuning (holdings): objective vs ensemble cost",
                            )
                    except Exception as e:
                        plot_meta_h["ensemble_cost"] = {"enabled": False, "error": str(e)}

                    if plot_meta_h:
                        _write_json(
                            run_dir / "alpha_allocation_regime_holdings_validation_pareto_plots.json",
                            plot_meta_h,
                        )

            # P2.29: a one-page selection report explaining the regime choice.
            try:
                sel_rep = build_regime_tuning_selection_report(
                    config=config,
                    proxy_rows=list(reg_rows or []),
                    proxy_summary=dict(reg_summary or {}),
                    holdings_rows=list(hold_rows or []) if isinstance(hold_rows, list) else None,
                    holdings_summary=dict(hold_summary or {}) if isinstance(hold_summary, dict) else None,
                    top_n=10,
                )
                _write_json(run_dir / "regime_tuning_selection_report.json", sel_rep)
                _write_text(run_dir / "REGIME_TUNING_REPORT.md", render_regime_tuning_report_md(sel_rep))

                proxy_top = ((sel_rep.get("proxy") or {}).get("top_candidates") or []) if isinstance(sel_rep, dict) else []
                if isinstance(proxy_top, list) and proxy_top:
                    pd.DataFrame(proxy_top).to_csv(run_dir / "regime_tuning_proxy_top_candidates.csv", index=False)

                hold_top = ((sel_rep.get("holdings_valid") or {}).get("top_candidates") or []) if isinstance(sel_rep, dict) else []
                if isinstance(hold_top, list) and hold_top:
                    pd.DataFrame(hold_top).to_csv(run_dir / "regime_tuning_holdings_valid_top_candidates.csv", index=False)
            except Exception:
                # Best-effort: report generation should never fail the run.
                pass
    except Exception:
        pass

# 4) Daily series for the top alphas (if available)
    if save_daily_top > 0 and sota_alphas:
        ddir = run_dir / "daily"
        ddir.mkdir(parents=True, exist_ok=True)
        for a in sota_alphas[: int(save_daily_top)]:
            aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
            m = a.get("backtest_results") or {}
            wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
            oos_daily = (wf.get("oos_daily") or []) if isinstance(wf, dict) else []
            if oos_daily:
                df = pd.DataFrame(oos_daily)
                df.to_csv(ddir / f"{aid}_oos_daily.csv", index=False)

    # 5) Human-readable summary
    summary = _run_summary_md(run_id=run_id, thread_id=thread_id, config=config, metrics=metrics, sota_alphas=sota_alphas)
    _write_text(run_dir / "SUMMARY.md", summary)

    # 5b) A deeper debugging report (P2.11)
    try:
        report = make_run_report_md(run_id=run_id, thread_id=thread_id, config=config, metrics=metrics, result=result)
        _write_text(run_dir / "REPORT.md", report)
    except Exception:
        # Keep the tracker robust: report generation is best-effort.
        pass

    # 6) Index record for quick listing
    best_id, best_row = _best_alpha_from_table(metrics)
    record = {
        "run_id": run_id,
        "timestamp_utc": _utc_timestamp(),
        "thread_id": thread_id,
        "eval_mode": config.get("eval_mode"),
        "data_path": config.get("data_path") or "",
        "best_alpha_id": best_id,
        "best_information_ratio": best_row.get("information_ratio") if best_row else None,
        "run_dir": str(run_dir),
    }
    _append_index(root, record)

    # 7) Append to a lightweight global factor registry
    coded_alphas = []
    if isinstance(result, dict):
        coded_alphas = list(result.get("coded_alphas") or [])
    else:
        coded_alphas = list(getattr(result, "coded_alphas", []) or [])
    _append_factor_registry(root, run_id, coded_alphas)

    # Helpful pointer to the latest run.
    _write_text(root / "LATEST", run_id + "\n")

    return run_dir
