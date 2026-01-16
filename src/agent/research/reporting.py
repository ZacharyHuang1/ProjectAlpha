"""agent.research.reporting

P2.11: A human-readable REPORT.md generator.

The goal is fast debugging and iteration: a single markdown file that explains
what happened in a run (performance, costs, optimizer usage, and constraints).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _fmt(x: Any, *, ndigits: int = 4) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "na"
        return f"{v:.{ndigits}f}"
    except Exception:
        return "na"


def _find_alpha_payload(result: Any, alpha_id: str) -> Optional[Dict[str, Any]]:
    if not alpha_id:
        return None
    if isinstance(result, dict):
        alphas = list(result.get("coded_alphas") or [])
    else:
        alphas = list(getattr(result, "coded_alphas", []) or [])
    for a in alphas:
        if not isinstance(a, dict):
            continue
        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
        if str(aid) == str(alpha_id):
            return a
    return None


def _best_alpha_from_metrics(metrics: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
    if metrics is None or metrics.empty:
        return None, {}
    row = metrics.iloc[0].to_dict()
    aid = row.get("alpha_id")
    return (str(aid) if aid is not None else None), row


def _constraint_lines(diag: Dict[str, Any]) -> List[str]:
    cs = (diag or {}).get("constraint_summary") or {}
    if not isinstance(cs, dict):
        return []
    lines: List[str] = []
    bindings = cs.get("binding_constraints") or []
    if bindings:
        lines.append(f"- binding_constraints: `{','.join([str(x) for x in bindings])}`\n")

    t = cs.get("turnover") or {}
    if isinstance(t, dict) and float(t.get("turnover_cap") or 0.0) > 0.0:
        lines.append(
            f"- turnover_cap: used `{_fmt(t.get('turnover_to_target'))}` / cap `{_fmt(t.get('turnover_cap'))}` (binding={bool(t.get('binding'))})\n"
        )

    mw = cs.get("max_abs_weight") or {}
    if isinstance(mw, dict) and bool(mw.get("enabled")):
        lines.append(
            f"- max_abs_weight: max_used `{_fmt(mw.get('max_used'))}` / cap `{_fmt(mw.get('cap'))}` (binding={bool(mw.get('binding'))}, at_cap_frac={_fmt(mw.get('at_cap_frac'))})\n"
        )

    p = cs.get("participation") or {}
    if isinstance(p, dict) and bool(p.get("enabled")):
        lines.append(
            f"- participation: max_ratio `{_fmt(p.get('max_ratio'))}` (binding={bool(p.get('binding'))}, binding_frac={_fmt(p.get('binding_frac'))})\n"
        )

    ex = cs.get("exposure") or {}
    if isinstance(ex, dict) and bool(ex.get("enabled")):
        lines.append(f"- exposure: max_abs `{_fmt(ex.get('max_abs'))}`, l2 `{_fmt(ex.get('l2'))}`\n")
        if ex.get("slack_max_abs") is not None:
            lines.append(
                f"- exposure_slack: max_abs `{_fmt(ex.get('slack_max_abs'))}`, l2 `{_fmt(ex.get('slack_l2'))}`\n"
            )

    warn = cs.get("warnings") or []
    if warn:
        lines.append(f"- warnings: `{','.join([str(x) for x in warn])}`\n")
    return lines


def make_run_report_md(
    *,
    run_id: str,
    thread_id: str,
    config: Dict[str, Any],
    metrics: pd.DataFrame,
    result: Any,
) -> str:
    """Create a single markdown report for a run."""

    eval_mode = str(config.get("eval_mode") or config.get("eval-mode") or "")
    data_path = str(config.get("data_path") or "")
    best_id, best_row = _best_alpha_from_metrics(metrics)

    alpha = _find_alpha_payload(result, best_id or "")
    dsl = (alpha or {}).get("dsl") or (alpha or {}).get("code") or ""
    desc = (alpha or {}).get("description") or ""

    selection = {}
    selection_tuning = {}
    ensemble = {}
    ensemble_holdings = {}
    ensemble_holdings_allocated = {}
    ensemble_holdings_allocated_regime = {}
    try:
        if isinstance(result, dict):
            selection = result.get("selection") or {}
            selection_tuning = result.get("selection_tuning") or {}
            ensemble = result.get("ensemble") or {}
            ensemble_holdings = result.get("ensemble_holdings") or {}
            ensemble_holdings_allocated = result.get("ensemble_holdings_allocated") or {}
            ensemble_holdings_allocated_regime = result.get("ensemble_holdings_allocated_regime") or {}
        else:
            selection = getattr(result, "selection", {}) or {}
            selection_tuning = getattr(result, "selection_tuning", {}) or {}
            ensemble = getattr(result, "ensemble", {}) or {}
            ensemble_holdings = getattr(result, "ensemble_holdings", {}) or {}
            ensemble_holdings_allocated = getattr(result, "ensemble_holdings_allocated", {}) or {}
            ensemble_holdings_allocated_regime = getattr(result, "ensemble_holdings_allocated_regime", {}) or {}
    except Exception:
        selection = {}
        selection_tuning = {}
        ensemble = {}
        ensemble_holdings = {}
        ensemble_holdings_allocated = {}
        ensemble_holdings_allocated_regime = {}

    m = (alpha or {}).get("backtest_results") or {}
    wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
    stab = (wf.get("stability") or {}) if isinstance(wf, dict) else {}
    yearly = (wf.get("oos_yearly") or {}) if isinstance(wf, dict) else {}
    cost_attr = (m.get("oos_cost_attribution") or {}) if isinstance(m, dict) else {}
    tuning = (m.get("tuning") or {}) if isinstance(m, dict) else {}
    ablation = (tuning.get("ablation") or {}) if isinstance(tuning, dict) else {}
    analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
    regimes = (analysis.get("regime") or {}) if isinstance(analysis, dict) else {}
    cost_sens = (analysis.get("cost_sensitivity") or {}) if isinstance(analysis, dict) else {}
    decay = (analysis.get("decay") or {}) if isinstance(analysis, dict) else {}
    schedule_sweep = (analysis.get("schedule_sweep") or {}) if isinstance(analysis, dict) else {}
    opt_usage = (wf.get("optimizer_usage") or {}) if isinstance(wf, dict) else {}
    latest_opt = (opt_usage.get("latest") or {}) if isinstance(opt_usage, dict) else {}
    diag = (latest_opt.get("diagnostics") or {}) if isinstance(latest_opt, dict) else {}

    lines: List[str] = []
    lines.append(f"# Run report: `{run_id}`\n\n")
    lines.append("## Metadata\n")
    lines.append(f"- thread_id: `{thread_id}`\n")
    lines.append(f"- eval_mode: `{eval_mode}`\n")
    lines.append(f"- data_path: `{data_path or 'synthetic'}`\n\n")

    if best_id:
        lines.append("## Best alpha\n")
        lines.append(f"- alpha_id: `{best_id}`\n")
        if dsl:
            snippet = str(dsl)[:220]
            lines.append(f"- dsl: `{snippet}{'...' if len(str(dsl)) > 220 else ''}`\n")
        if desc:
            lines.append(f"- description: {str(desc)[:240]}\n")
        lines.append(f"- information_ratio: `{_fmt(best_row.get('information_ratio'))}`\n")
        lines.append(f"- annualized_return: `{_fmt(best_row.get('annualized_return'))}`\n")
        lines.append(f"- max_drawdown: `{_fmt(best_row.get('max_drawdown'))}`\n")
        lines.append(f"- turnover_mean: `{_fmt(best_row.get('turnover_mean'))}`\n\n")

        # DSL sanity: show auto-fixes/warnings for fast debugging.
        try:
            fx = (alpha or {}).get("dsl_fixes") or []
            wa = (alpha or {}).get("dsl_warnings") or []
            if isinstance(fx, list) and fx:
                lines.append(f"- dsl_fixes: `{ ' ; '.join([str(x) for x in fx]) }`\n")
            if isinstance(wa, list) and wa:
                lines.append(f"- dsl_warnings: `{ ' ; '.join([str(x) for x in wa]) }`\n")
            if (isinstance(fx, list) and fx) or (isinstance(wa, list) and wa):
                lines.append("\n")
        except Exception:
            pass


        # P2.22: selection meta-tuning summary (validation only).
        if isinstance(selection_tuning, dict) and bool(selection_tuning.get("enabled")):
            lines.append("## Selection meta-tuning (validation)\n")
            lines.append(f"- metric: `{selection_tuning.get('metric')}`\n")
            best_sel = selection_tuning.get("best") or {}
            if isinstance(best_sel, dict):
                lines.append(f"- candidate_pool: `{int(best_sel.get('candidate_pool') or 0)}`\n")
                lines.append(f"- diversity_lambda: `{_fmt(best_sel.get('diversity_lambda'))}`\n")
                lines.append(f"- top_k: `{int(best_sel.get('top_k') or 0)}`\n")
                lines.append(f"- validation {selection_tuning.get('metric')}: `{_fmt(best_sel.get('valid_metric'))}`\n")
                ids = best_sel.get("selected_alpha_ids") or []
                if isinstance(ids, list) and ids:
                    lines.append(f"- selected_alpha_ids: `{', '.join([str(x) for x in ids])}`\n")
            lines.append("- See `selection_tuning_results.csv` for all evaluated configs.\n\n")

        # P2.17: report diversified selection if enabled.
        if isinstance(selection, dict) and selection.get("method") in {"diverse_greedy", "meta_tuned_diverse_greedy"}:
            lines.append("## Diversified top-K selection\n")
            sel_best = selection.get("best") if isinstance(selection.get("best"), dict) else {}
            lines.append(f"- diversity_lambda: `{_fmt(sel_best.get('diversity_lambda') or selection.get('diversity_lambda'))}`\n")
            lines.append(f"- candidate_pool: `{int(sel_best.get('candidate_pool') or selection.get('candidate_pool') or 0)}`\n")
            sel_ids = selection.get("selected_alpha_ids") or []
            if isinstance(sel_ids, list) and sel_ids:
                lines.append(f"- selected_alpha_ids: `{','.join([str(x) for x in sel_ids])}`\n")
            cs = selection.get("correlation_summary") or {}
            if isinstance(cs, dict):
                lines.append(f"- avg_abs_corr: `{_fmt(cs.get('avg_abs_corr'))}`\n")
                lines.append(f"- max_abs_corr: `{_fmt(cs.get('max_abs_corr'))}`\n")
            tbl = (sel_best.get("selection_table") if isinstance(sel_best, dict) else None) or selection.get("selection_table") or []
            if isinstance(tbl, list) and tbl:
                try:
                    dft = pd.DataFrame(tbl)
                    cols = [c for c in ["step", "alpha_id", "base_score", "avg_corr_to_selected", "diversity_score"] if c in dft.columns]
                    if cols:
                        lines.append("\n")
                        lines.append(dft[cols].to_markdown(index=False))
                        lines.append("\n\n")
                except Exception:
                    lines.append("\n")
            lines.append("- See `alpha_correlation.csv` for the full correlation matrix.\n\n")

        # P2.17: report ensemble metrics.
        if isinstance(ensemble, dict) and bool(ensemble.get("enabled")):
            lines.append("## Ensemble (equal-weight)\n")
            met = ensemble.get("metrics") or {}
            if isinstance(met, dict) and met:
                try:
                    dfm = pd.DataFrame([met])
                    cols = [c for c in ["information_ratio", "annualized_return", "max_drawdown", "coverage", "n_alphas"] if c in dfm.columns]
                    if cols:
                        lines.append(dfm[cols].to_markdown(index=False))
                        lines.append("\n\n")
                except Exception:
                    pass
            lines.append("- See `ensemble_oos_daily.csv` for the ensemble OOS daily returns.\n\n")

        # P2.18: report holdings-level ensemble metrics.
        if isinstance(ensemble_holdings, dict) and bool(ensemble_holdings.get("enabled")):
            lines.append("## Holdings-level ensemble (equal-weight)\n")
            met = ensemble_holdings.get("metrics") or {}
            if isinstance(met, dict) and met:
                try:
                    dfm = pd.DataFrame([met])
                    cols = [
                        c
                        for c in [
                            "information_ratio",
                            "annualized_return",
                            "max_drawdown",
                            "n_alphas",
                            "turnover_mean",
                            "cost_mean",
                            "borrow_mean",
                        ]
                        if c in dfm.columns
                    ]
                    if cols:
                        lines.append(dfm[cols].to_markdown(index=False))
                        lines.append("\n\n")
                except Exception:
                    pass

            # Small, high-signal comparison vs return-stream ensemble (if both exist).
            try:
                if isinstance(ensemble, dict) and bool(ensemble.get("enabled")):
                    m1 = ensemble.get("metrics") or {}
                    if isinstance(m1, dict):
                        d_ir = float(met.get("information_ratio") or 0.0) - float(m1.get("information_ratio") or 0.0)
                        lines.append(f"- ΔIR vs return-stream ensemble: `{_fmt(d_ir)}`\n")
            except Exception:
                pass

            # Cost netting snapshot (if computed by the agent).
            try:
                cmp = ensemble_holdings.get("comparison") or {}
                if isinstance(cmp, dict) and "cost_savings_bps" in cmp:
                    lines.append(f"- estimated cost netting vs avg single alpha: `{_fmt(cmp.get('cost_savings_bps'))}` bps\n")
            except Exception:
                pass

            lines.append("- See `ensemble_holdings_oos_daily.csv` for realized daily returns.\n")
            lines.append("- See `ensemble_holdings_positions.csv` for realized non-zero holdings.\n\n")

    # P2.19: report holdings-level allocated ensemble metrics.
    if isinstance(ensemble_holdings_allocated, dict) and bool(ensemble_holdings_allocated.get("enabled")):
        lines.append("## Holdings-level ensemble (allocated)\n")
        alloc_cfg = ensemble_holdings_allocated.get("allocation") or {}
        if isinstance(alloc_cfg, dict) and alloc_cfg:
            try:
                lines.append(
                    "- allocation: "
                    f"backend=`{alloc_cfg.get('backend')}` | "
                    f"fit=`{alloc_cfg.get('fit')}` | "
                    f"score=`{alloc_cfg.get('score_metric')}` | "
                    f"lambda=`{alloc_cfg.get('lambda_corr')}` | "
                    f"turnover_lambda=`{alloc_cfg.get('turnover_lambda')}` | "
                    f"max_weight=`{alloc_cfg.get('max_weight')}` | "
                    f"tuned=`{alloc_cfg.get('tuned')}`\n\n"
                )
            except Exception:
                pass

        met = ensemble_holdings_allocated.get("metrics") or {}
        if isinstance(met, dict) and met:
            try:
                dfm = pd.DataFrame([met])
                cols = [
                    c
                    for c in [
                        "information_ratio",
                        "annualized_return",
                        "max_drawdown",
                        "n_alphas",
                        "turnover_mean",
                        "cost_mean",
                        "borrow_mean",
                    ]
                    if c in dfm.columns
                ]
                if cols:
                    lines.append(dfm[cols].to_markdown(index=False))
                    lines.append("\n\n")
            except Exception:
                pass

        alloc_rows = ensemble_holdings_allocated.get("allocations") or []
        if isinstance(alloc_rows, list) and alloc_rows:
            try:
                dfw = pd.DataFrame(alloc_rows)
                if {"alpha_id", "weight"}.issubset(set(dfw.columns)):
                    w_mean = dfw.groupby("alpha_id")["weight"].mean().sort_values(ascending=False).head(10)
                    lines.append("Top mean alpha weights:\n\n")
                    lines.append(w_mean.to_frame("mean_weight").to_markdown())
                    lines.append("\n\n")
            except Exception:
                pass

        # P2.20: allocation meta-tuning summary (if enabled).
        tune = ensemble_holdings_allocated.get("allocation_tuning") or {}
        if isinstance(tune, dict) and bool(tune.get("enabled")):
            try:
                chosen = tune.get("chosen") or {}
                lines.append("Allocation meta-tuning (train→valid):\n\n")
                if isinstance(chosen, dict) and chosen:
                    lines.append(
                        "- chosen: "
                        f"config_id=`{chosen.get('config_id')}` | "
                        f"valid_metric=`{_fmt(chosen.get('valid_metric'))}` | "
                        f"lambda_corr=`{chosen.get('lambda_corr')}` | "
                        f"max_weight=`{chosen.get('max_weight')}` | "
                        f"turnover_lambda=`{chosen.get('turnover_lambda')}`\n\n"
                    )

                rows = tune.get("results") or []
                if isinstance(rows, list) and rows:
                    dft = pd.DataFrame(rows)
                    show_cols = [
                        c
                        for c in [
                            "config_id",
                            "valid_metric",
                            "lambda_corr",
                            "max_weight",
                            "turnover_lambda",
                            "allocation_turnover_mean",
                            "n_splits_used",
                            "n_valid_days",
                        ]
                        if c in dft.columns
                    ]
                    if show_cols:
                        lines.append(dft[show_cols].head(8).to_markdown(index=False))
                        lines.append("\n\n")
                lines.append("- See `alpha_allocation_tuning_results.csv` for the full tuning table.\n\n")
            except Exception:
                pass

        try:
            if isinstance(ensemble_holdings, dict) and bool(ensemble_holdings.get("enabled")):
                m0 = ensemble_holdings.get("metrics") or {}
                ir0 = float((m0 or {}).get("information_ratio") or 0.0)
                ir1 = float((met or {}).get("information_ratio") or 0.0)
                lines.append(f"- ΔIR vs equal-weight holdings ensemble: `{_fmt(ir1 - ir0)}`\n")
        except Exception:
            pass

        lines.append("- See `ensemble_holdings_allocated_oos_daily.csv` for realized daily returns.\n")
        lines.append("- See `ensemble_holdings_allocated_positions.csv` for realized non-zero holdings.\n")
        lines.append("- See `alpha_allocations.csv` for per-split alpha weights.\n\n")

    # P2.21: report holdings-level regime-aware allocated ensemble metrics.
    if isinstance(ensemble_holdings_allocated_regime, dict) and bool(ensemble_holdings_allocated_regime.get("enabled")):
        lines.append("## Holdings-level ensemble (allocated, regime-aware)\n")

        alloc_cfg = ensemble_holdings_allocated_regime.get("allocation") or {}
        reg_cfg = (alloc_cfg.get("regime") or {}) if isinstance(alloc_cfg, dict) else {}
        if isinstance(alloc_cfg, dict) and alloc_cfg:
            try:
                lines.append(
                    "- allocation: "
                    f"backend=`{alloc_cfg.get('backend')}` | "
                    f"fit=`{alloc_cfg.get('fit')}` | "
                    f"score=`{alloc_cfg.get('score_metric')}` | "
                    f"lambda=`{alloc_cfg.get('lambda_corr')}` | "
                    f"turnover_lambda=`{alloc_cfg.get('turnover_lambda')}` | "
                    f"max_weight=`{alloc_cfg.get('max_weight')}` | "
                    f"tuned=`{alloc_cfg.get('tuned')}`\n"
                )
            except Exception:
                pass
        if isinstance(reg_cfg, dict) and reg_cfg:
            try:
                lines.append(
                    "- regime: "
                    f"mode=`{reg_cfg.get('mode')}` | "
                    f"window=`{reg_cfg.get('window')}` | "
                    f"buckets=`{reg_cfg.get('buckets')}` | "
                    f"min_days=`{reg_cfg.get('min_days')}` | "
                    f"smoothing=`{reg_cfg.get('smoothing')}` | "
                    f"tuned_method=`{reg_cfg.get('tuned_method')}`\n\n"
                )
            except Exception:
                pass

        # Optional meta-tuning summaries (allocation + regime hyperparams).
        try:
            at = ensemble_holdings_allocated_regime.get("allocation_tuning") or {}
            if isinstance(at, dict) and (at.get("enabled") or at.get("error")):
                lines.append("### Allocation tuning (train->valid)\n")
                ch = at.get("chosen") or {}
                if isinstance(ch, dict) and ch:
                    lines.append(
                        "- chosen: "
                        f"lambda=`{ch.get('lambda_corr')}` | "
                        f"max_weight=`{ch.get('max_weight')}` | "
                        f"turnover_lambda=`{ch.get('turnover_lambda')}` | "
                        f"valid_metric=`{_fmt(ch.get('valid_metric'))}`\n"
                    )
                if at.get("error"):
                    lines.append(f"- error: `{str(at.get('error'))}`\n")
                lines.append(
                    "- Artifacts: `alpha_allocation_regime_tuning_results.csv`, "
                    "`alpha_allocation_regime_tuning_summary.json`\n\n"
                )
        except Exception:
            pass

        try:
            rt = ensemble_holdings_allocated_regime.get("regime_tuning") or {}
            if isinstance(rt, dict) and (rt.get("enabled") or rt.get("error")):
                lines.append("### Regime hyperparam tuning (train->valid)\n")
                preset = str(config.get("alpha_allocation_regime_tune_preset") or "").strip()
                if preset:
                    lines.append(f"- preset: `{preset}`\n")
                ch = rt.get("chosen") or {}
                if isinstance(ch, dict) and ch:
                    lines.append(
                        "- chosen: "
                        f"mode=`{ch.get('mode')}` | "
                        f"window=`{ch.get('window')}` | "
                        f"buckets=`{ch.get('buckets')}` | "
                        f"smoothing=`{ch.get('smoothing')}` | "
                        f"objective=`{_fmt(ch.get('objective'))}` | "
                        f"raw_valid_metric=`{_fmt(ch.get('valid_metric'))}` | "
                        f"adj_valid_metric=`{_fmt(ch.get('valid_metric_after_turnover_cost'))}` | "
                        f"alpha_weight_turnover=`{_fmt(ch.get('alpha_weight_turnover_mean'))}`\n"
                    )
                if rt.get("turnover_cost_bps") is not None:
                    lines.append(f"- turnover_cost_bps: `{rt.get('turnover_cost_bps')}`\n")
                elif rt.get("turnover_penalty") is not None:
                    lines.append(f"- turnover_penalty: `{rt.get('turnover_penalty')}`\n")
                cons = rt.get("constraints") or {}
                if isinstance(cons, dict) and cons:
                    cons_s = ", ".join([f"{k}={cons.get(k)}" for k in sorted(cons.keys())])
                    lines.append(f"- constraints: `{cons_s}`\n")
                if rt.get("selected_by"):
                    lines.append(
                        f"- selected_by: `{rt.get('selected_by')}` | feasible_count: `{rt.get('feasible_count')}`\n"
                    )

                pobjs = rt.get("pareto_objectives") or []
                if pobjs:
                    try:
                        pobjs_s = ", ".join([f"{k}:{d}" for k, d in list(pobjs)])
                    except Exception:
                        pobjs_s = str(pobjs)
                    lines.append(f"- pareto_objectives: `{pobjs_s}`\n")

                if rt.get("error"):
                    lines.append(f"- error: `{str(rt.get('error'))}`\n")

                # Show a small table when available (already trimmed in JSON by save_top).
                rows = rt.get("results") or []
                if isinstance(rows, list) and rows:
                    try:
                        dfr = pd.DataFrame(rows)
                        cols = [
                            c
                            for c in [
                                "mode",
                                "window",
                                "buckets",
                                "smoothing",
                                "valid_metric",
                                "valid_metric_after_turnover_cost",
                                "alpha_weight_turnover_mean",
                                "turnover_cost_drag_bps_mean",
                                "objective",
                                "is_pareto",
                            ]
                            if c in dfr.columns
                        ]
                        if cols:
                            lines.append(dfr[cols].head(10).to_markdown(index=False))
                            lines.append("\n\n")
                    except Exception:
                        pass

                plots_enabled = bool(config.get("alpha_allocation_regime_tune_plots", True))
                art = [
                    "alpha_allocation_regime_param_tuning_results.csv",
                    "alpha_allocation_regime_param_tuning_summary.json",
                    "alpha_allocation_regime_param_tuning_pareto.csv",
                ]
                if plots_enabled:
                    art += [
                        "alpha_allocation_regime_param_tuning_pareto_turnover.png",
                        "alpha_allocation_regime_param_tuning_pareto_drag.png",
                        "alpha_allocation_regime_param_tuning_pareto_plots.json",
                    ]
                lines.append("- Artifacts: " + ", ".join([f"`{a}`" for a in art]) + "\n\n")

        except Exception:
            pass

        # P2.24: holdings-level revalidation for the top proxy regime configs.
        try:
            hv = ensemble_holdings_allocated_regime.get("regime_tuning_holdings_validation") or {}
            if isinstance(hv, dict) and (hv.get("enabled") or hv.get("error")):
                lines.append("### Regime holdings-level revalidation (train->valid)\n")
                ch = hv.get("chosen") or {}
                if isinstance(ch, dict) and ch:
                    lines.append(
                        "- chosen: "
                        f"mode=`{ch.get('mode')}` | "
                        f"window=`{ch.get('window')}` | "
                        f"buckets=`{ch.get('buckets')}` | "
                        f"smoothing=`{ch.get('smoothing')}` | "
                        f"objective=`{_fmt(ch.get('holdings_objective'))}` | "
                        f"raw_valid_metric=`{_fmt(ch.get('holdings_valid_metric'))}` | "
                        f"adj_valid_metric=`{_fmt(ch.get('holdings_valid_metric_after_turnover_cost'))}` | "
                        f"alpha_weight_turnover=`{_fmt(ch.get('alpha_weight_turnover_mean'))}`\n"
                    )
                if hv.get("top_n") is not None:
                    lines.append(f"- top_n: `{hv.get('top_n')}`\n")
                if hv.get("turnover_cost_bps") is not None:
                    lines.append(f"- turnover_cost_bps: `{hv.get('turnover_cost_bps')}`\n")
                elif hv.get("turnover_penalty") is not None:
                    lines.append(f"- turnover_penalty: `{hv.get('turnover_penalty')}`\n")
                cons = hv.get("constraints") or {}
                if isinstance(cons, dict) and cons:
                    cons_s = ", ".join([f"{k}={cons.get(k)}" for k in sorted(cons.keys())])
                    lines.append(f"- constraints: `{cons_s}`\n")
                if hv.get("selected_by"):
                    lines.append(
                        f"- selected_by: `{hv.get('selected_by')}` | feasible_count: `{hv.get('feasible_count')}`\n"
                    )

                pobjs = hv.get("pareto_objectives") or []
                if pobjs:
                    try:
                        pobjs_s = ", ".join([f"{k}:{d}" for k, d in list(pobjs)])
                    except Exception:
                        pobjs_s = str(pobjs)
                    lines.append(f"- pareto_objectives: `{pobjs_s}`\n")

                if hv.get("error"):
                    lines.append(f"- error: `{str(hv.get('error'))}`\n")

                rows = hv.get("results") or []
                if isinstance(rows, list) and rows:
                    try:
                        dfh = pd.DataFrame(rows)
                        cols = [
                            c
                            for c in [
                                "mode",
                                "window",
                                "buckets",
                                "smoothing",
                                "holdings_valid_metric",
                                "holdings_valid_metric_after_turnover_cost",
                                "alpha_weight_turnover_mean",
                                "turnover_cost_drag_bps_mean",
                                "holdings_objective",
                                "is_pareto",
                                "ensemble_cost_mean",
                            ]
                            if c in dfh.columns
                        ]
                        if cols:
                            lines.append(dfh[cols].head(10).to_markdown(index=False))
                            lines.append("\n\n")
                    except Exception:
                        pass

                plots_enabled = bool(config.get("alpha_allocation_regime_tune_plots", True))
                art = [
                    "alpha_allocation_regime_holdings_validation_results.csv",
                    "alpha_allocation_regime_holdings_validation_summary.json",
                    "alpha_allocation_regime_holdings_validation_pareto.csv",
                ]
                if plots_enabled:
                    art += [
                        "alpha_allocation_regime_holdings_validation_pareto_turnover.png",
                        "alpha_allocation_regime_holdings_validation_pareto_cost.png",
                        "alpha_allocation_regime_holdings_validation_pareto_plots.json",
                    ]
                lines.append("- Artifacts: " + ", ".join([f"`{a}`" for a in art]) + "\n\n")
        except Exception:
            pass

        met = ensemble_holdings_allocated_regime.get("metrics") or {}
        if isinstance(met, dict) and met:
            try:
                dfm = pd.DataFrame([met])
                cols = [
                    c
                    for c in [
                        "information_ratio",
                        "annualized_return",
                        "max_drawdown",
                        "n_alphas",
                        "turnover_mean",
                        "cost_mean",
                        "borrow_mean",
                    ]
                    if c in dfm.columns
                ]
                if cols:
                    lines.append(dfm[cols].to_markdown(index=False))
                    lines.append("\n\n")
            except Exception:
                pass

        alloc_rows = ensemble_holdings_allocated_regime.get("allocations_regime") or []
        if isinstance(alloc_rows, list) and alloc_rows:
            try:
                dfw = pd.DataFrame(alloc_rows)
                if {"regime", "alpha_id", "weight"}.issubset(set(dfw.columns)):
                    # Show the top weights for the most frequent regimes.
                    top_regimes = (
                        dfw[dfw["regime"] != "__global__"].groupby("regime")["weight"].size().sort_values(ascending=False).head(3)
                    )
                    if not top_regimes.empty:
                        lines.append("Top alpha weights by regime (mean across splits):\n\n")
                        for reg in list(top_regimes.index):
                            sub = dfw[dfw["regime"] == reg].groupby("alpha_id")["weight"].mean().sort_values(ascending=False).head(8)
                            lines.append(f"**{reg}**\n\n")
                            lines.append(sub.to_frame("mean_weight").to_markdown())
                            lines.append("\n\n")
            except Exception:
                pass

        try:
            if isinstance(ensemble_holdings_allocated, dict) and bool(ensemble_holdings_allocated.get("enabled")):
                m0 = ensemble_holdings_allocated.get("metrics") or {}
                ir0 = float((m0 or {}).get("information_ratio") or 0.0)
                ir1 = float((met or {}).get("information_ratio") or 0.0)
                lines.append(f"- ΔIR vs static allocated holdings ensemble: `{_fmt(ir1 - ir0)}`\n")
        except Exception:
            pass

        lines.append("- See `ensemble_holdings_allocated_regime_oos_daily.csv` for realized daily returns.\n")
        lines.append("- See `ensemble_holdings_allocated_regime_positions.csv` for realized non-zero holdings.\n")
        lines.append("- See `alpha_allocations_regime.csv` for regime-specific alpha weights.\n\n")

    lines.append("## Walk-forward stability\n")
    lines.append(f"- n_splits: `{int(stab.get('n_splits') or 0)}`\n")
    lines.append(f"- test_ir_mean: `{_fmt(stab.get('test_ir_mean'))}`\n")
    lines.append(f"- test_ir_std: `{_fmt(stab.get('test_ir_std'))}`\n")
    lines.append(f"- test_ir_positive_frac: `{_fmt(stab.get('test_ir_positive_frac'))}`\n")
    # Extra diagnostics: helpful when test_ir_positive_frac is 0.
    if stab.get("test_ir_n") is not None:
        lines.append(f"- test_ir_n: `{int(stab.get('test_ir_n') or 0)}`\n")
    if stab.get("test_ir_min") is not None or stab.get("test_ir_max") is not None:
        lines.append(f"- test_ir_min/median/max: `{_fmt(stab.get('test_ir_min'))}` / `{_fmt(stab.get('test_ir_median'))}` / `{_fmt(stab.get('test_ir_max'))}`\n")
    if stab.get("test_n_obs_mean") is not None:
        lines.append(f"- test_n_obs_mean: `{_fmt(stab.get('test_n_obs_mean'))}`\n")
    if stab.get("test_n_obs_zero_splits") is not None:
        lines.append(f"- test_n_obs_zero_splits: `{int(stab.get('test_n_obs_zero_splits') or 0)}`\n")
    lines.append(f"- generalization_gap: `{_fmt(stab.get('generalization_gap'))}`\n\n")

    if cost_attr:
        lines.append("## OOS cost attribution\n")
        try:
            df = pd.DataFrame(
                {
                    "metric": list(cost_attr.keys()),
                    "mean_return_units": [float(cost_attr[k]) for k in cost_attr.keys()],
                    "mean_bps": [float(cost_attr[k]) * 10000.0 for k in cost_attr.keys()],
                }
            )
            lines.append(df.to_markdown(index=False))
            lines.append("\n\n")
        except Exception:
            for k, v in cost_attr.items():
                lines.append(f"- {k}: `{_fmt(v)}`\n")
            lines.append("\n")

    # P2.13: show both ablation modes when available.
    if isinstance(ablation, dict) and ablation:
        end_to_end = None
        exec_only = None

        # Backward compatibility: older runs stored scenarios at the top level.
        if isinstance(ablation.get("scenarios"), list):
            end_to_end = list(ablation.get("scenarios") or [])
        else:
            e2e = ablation.get("end_to_end") or {}
            if isinstance(e2e, dict) and isinstance(e2e.get("scenarios"), list):
                end_to_end = list(e2e.get("scenarios") or [])

        exe = ablation.get("execution_only") or {}
        if isinstance(exe, dict) and isinstance(exe.get("scenarios"), list):
            exec_only = list(exe.get("scenarios") or [])

        if end_to_end:
            lines.append("## Cost ablation (end_to_end)\n")
            try:
                rows = []
                for sc in end_to_end:
                    if not isinstance(sc, dict):
                        continue
                    rows.append(
                        {
                            "scenario": sc.get("scenario"),
                            "information_ratio": sc.get("information_ratio"),
                            "annualized_return": sc.get("annualized_return"),
                            "max_drawdown": sc.get("max_drawdown"),
                            "turnover_mean": sc.get("turnover_mean"),
                            "total_cost_bps": sc.get("total_cost_bps"),
                            "error": sc.get("error"),
                        }
                    )
                if rows:
                    df = pd.DataFrame(rows)
                    lines.append(df.to_markdown(index=False))
                    lines.append("\n\n")
            except Exception:
                pass

        if exec_only:
            lines.append("## Cost ablation (execution_only)\n")
            try:
                rows = []
                for sc in exec_only:
                    if not isinstance(sc, dict):
                        continue
                    rows.append(
                        {
                            "scenario": sc.get("scenario"),
                            "information_ratio": sc.get("information_ratio"),
                            "annualized_return": sc.get("annualized_return"),
                            "max_drawdown": sc.get("max_drawdown"),
                            "mean_cost_drag_bps": sc.get("mean_cost_drag_bps"),
                            "error": sc.get("error"),
                        }
                    )
                if rows:
                    df = pd.DataFrame(rows)
                    lines.append(df.to_markdown(index=False))
                    lines.append("\n\n")
            except Exception:
                pass

    if isinstance(regimes, dict) and regimes:
        mv = regimes.get("market_volatility")
        if isinstance(mv, list) and mv:
            lines.append("## Regime analysis (market volatility)\n")
            try:
                df = pd.DataFrame(mv)
                lines.append(df.to_markdown(index=False))
                lines.append("\n\n")
            except Exception:
                pass

        ml = regimes.get("market_liquidity")
        if isinstance(ml, list) and ml:
            lines.append("## Regime analysis (market liquidity)\n")
            try:
                df = pd.DataFrame(ml)
                lines.append(df.to_markdown(index=False))
                lines.append("\n\n")
            except Exception:
                pass

    if isinstance(cost_sens, dict) and bool(cost_sens.get("enabled")):
        lines.append("## Cost sensitivity (execution_only)\n")
        lines.append(f"- borrow_mode: `{cost_sens.get('borrow_mode')}`\n")
        base = cost_sens.get("base") or {}
        if isinstance(base, dict):
            lines.append(
                f"- base_linear_cost_bps: `{_fmt(base.get('base_linear_cost_bps'))}`, "
                f"base_half_spread_bps: `{_fmt(base.get('base_half_spread_bps'))}`, "
                f"base_impact_bps: `{_fmt(base.get('base_impact_bps'))}`\n"
            )
            if base.get("base_borrow_bps") is not None:
                lines.append(f"- base_borrow_bps: `{_fmt(base.get('base_borrow_bps'))}`\n")
            if base.get("base_borrow_multiplier") is not None:
                lines.append(f"- base_borrow_multiplier: `{_fmt(base.get('base_borrow_multiplier'))}`\n")
        be = cost_sens.get("break_even") or []
        if isinstance(be, list) and be:
            try:
                dfbe = pd.DataFrame(be)
                cols = [c for c in ["parameter", "break_even", "within_grid", "status", "min", "max"] if c in dfbe.columns]
                if cols:
                    lines.append("\n")
                    lines.append(dfbe[cols].to_markdown(index=False))
                    lines.append("\n\n")
            except Exception:
                pass
        lines.append("- See `cost_sensitivity.csv` in the run folder for full curves.\n\n")

    # P2.15: multi-horizon decay analysis (predictive decay + signal persistence).
    if isinstance(decay, dict) and bool(decay.get("enabled")):
        lines.append("## Horizon decay analysis\n")
        best = decay.get("best") or {}
        if isinstance(best, dict) and best.get("horizon") is not None:
            lines.append(f"- best_horizon: `{best.get('horizon')}` (criterion={best.get('criterion')})\n")
            if best.get("rank_ic_tstat") is not None:
                lines.append(f"- best_rank_ic_tstat: `{_fmt(best.get('rank_ic_tstat'))}`\n")
            if best.get("spread_ir_ann_proxy") is not None:
                lines.append(f"- best_spread_ir_ann_proxy: `{_fmt(best.get('spread_ir_ann_proxy'))}`\n")
            if best.get("signal_overlap_mean") is not None:
                lines.append(f"- best_signal_overlap_mean: `{_fmt(best.get('signal_overlap_mean'))}`\n")

        metrics_rows = decay.get("metrics") or []
        if isinstance(metrics_rows, list) and metrics_rows:
            try:
                df = pd.DataFrame(metrics_rows)
                cols = [
                    c
                    for c in [
                        "horizon",
                        "rank_ic_mean",
                        "rank_ic_tstat",
                        "ic_mean",
                        "spread_mean",
                        "spread_ir_ann_proxy",
                        "signal_overlap_mean",
                        "signal_pairs",
                    ]
                    if c in df.columns
                ]
                if cols:
                    d2 = df[cols].copy()
                    for c in cols:
                        if c in {"horizon", "signal_pairs"}:
                            continue
                        d2[c] = pd.to_numeric(d2[c], errors="coerce").round(4)
                    lines.append("\n")
                    lines.append(d2.to_markdown(index=False))
                    lines.append("\n\n")
            except Exception:
                pass

        lines.append("- See `decay_analysis.csv` in the run folder for the full table.\n\n")

    # P2.16: holding/rebalance schedule sweep (strategy-level). Show a compact
    # view of the trade-off surface and a recommended schedule.
    if isinstance(schedule_sweep, dict) and bool(schedule_sweep.get("enabled")):
        lines.append("## Holding / rebalance schedule sweep\n")

        metric = str(schedule_sweep.get("metric") or "information_ratio")
        base = schedule_sweep.get("base") or {}
        best = schedule_sweep.get("best") or {}
        best_raw = schedule_sweep.get("best_raw") or {}
        delta = schedule_sweep.get("delta") or {}

        lines.append(f"- metric: `{metric}`\n")
        if isinstance(base, dict) and base.get("rebalance_days") is not None and base.get("holding_days") is not None:
            lines.append(
                f"- base_schedule: rebalance_days=`{base.get('rebalance_days')}`, holding_days=`{base.get('holding_days')}`\n"
            )

        chosen = best if isinstance(best, dict) and best else best_raw
        if isinstance(chosen, dict) and chosen.get("rebalance_days") is not None and chosen.get("holding_days") is not None:
            lines.append(
                f"- recommended_schedule: rebalance_days=`{chosen.get('rebalance_days')}`, holding_days=`{chosen.get('holding_days')}` "
                f"(passed={bool(chosen.get('passed'))})\n"
            )
            if chosen.get("information_ratio") is not None:
                lines.append(f"- recommended_ir: `{_fmt(chosen.get('information_ratio'))}`\n")
            if chosen.get("annualized_return") is not None:
                lines.append(f"- recommended_annualized_return: `{_fmt(chosen.get('annualized_return'))}`\n")
            if chosen.get("turnover_mean") is not None:
                lines.append(f"- recommended_turnover_mean: `{_fmt(chosen.get('turnover_mean'))}`\n")
            if chosen.get("total_cost_bps") is not None:
                lines.append(f"- recommended_total_cost_bps: `{_fmt(chosen.get('total_cost_bps'))}`\n")

        if isinstance(delta, dict) and delta.get("improvement") is not None:
            lines.append(
                f"- delta_{str(delta.get('metric') or metric)}: `{_fmt(delta.get('improvement'))}` "
                f"(base={_fmt(delta.get('base'))}, best={_fmt(delta.get('best'))})\n"
            )

        rows = schedule_sweep.get("results") or []
        try:
            df = pd.DataFrame([r for r in rows if isinstance(r, dict) and not r.get("error")])
            if not df.empty and metric in df.columns:
                df[metric] = pd.to_numeric(df[metric], errors="coerce")
                df = df.sort_values(by=[metric], ascending=False, na_position="last")

                # 1) A compact heatmap-style table (when the grid is small).
                u_h = sorted(set([int(x) for x in df.get("holding_days", []).dropna().tolist()]))
                u_r = sorted(set([int(x) for x in df.get("rebalance_days", []).dropna().tolist()]))
                if len(u_h) <= 8 and len(u_r) <= 8:
                    piv = df.pivot_table(index="holding_days", columns="rebalance_days", values=metric, aggfunc="first")
                    piv = piv.round(3)
                    lines.append("\n")
                    lines.append(piv.to_markdown())
                    lines.append("\n\n")

                # 2) Top rows summary.
                cols = [
                    c
                    for c in [
                        "rebalance_days",
                        "holding_days",
                        metric,
                        "annualized_return",
                        "max_drawdown",
                        "turnover_mean",
                        "total_cost_bps",
                        "passed",
                    ]
                    if c in df.columns
                ]
                if cols:
                    lines.append(df[cols].head(12).to_markdown(index=False))
                    lines.append("\n\n")
        except Exception:
            pass

        lines.append("- See `schedule_sweep.csv` in the run folder for the full table.\n\n")

    if isinstance(opt_usage, dict) and opt_usage.get("backend_used"):
        lines.append("## Optimizer usage\n")
        lines.append(f"- backend_used: `{opt_usage.get('backend_used')}`\n")
        if opt_usage.get("fallback_reasons"):
            lines.append(f"- fallback_reasons: `{opt_usage.get('fallback_reasons')}`\n")
        lines.append("\n")

    if latest_opt:
        lines.append("## Latest optimizer diagnostics (last test segment)\n")
        lines.append(f"- backend_used: `{latest_opt.get('backend_used')}`\n")
        if latest_opt.get("fallback"):
            lines.append(f"- fallback: `{latest_opt.get('fallback')}`\n")

        pre = latest_opt.get("qp_precheck") or {}
        if isinstance(pre, dict) and pre.get("reasons"):
            lines.append(f"- qp_precheck_failed_reasons: `{','.join([str(x) for x in pre.get('reasons') or []])}`\n")
            if pre.get("suggestions"):
                lines.append("- qp_precheck_suggestions:\n")
                for s in list(pre.get("suggestions") or [])[:6]:
                    if isinstance(s, dict):
                        msg = s.get("message") or ""
                        par = s.get("parameter")
                        sm = s.get("suggested")
                        mn = s.get("suggested_min")
                        parts = [f"parameter={par}"]
                        if sm is not None:
                            parts.append(f"suggested={sm}")
                        if mn is not None:
                            parts.append(f"suggested_min={mn}")
                        lines.append(f"  - {', '.join(parts)}: {msg}\n")

        lines.append("\n### Constraint summary\n")
        c_lines = _constraint_lines(diag)
        lines.extend(c_lines if c_lines else ["- (no diagnostics available)\n"])

        # Risk attribution (if available).
        if diag.get("risk_model"):
            lines.append("\n### Risk attribution\n")
            lines.append(f"- risk_model: `{diag.get('risk_model')}`\n")
            lines.append(f"- risk_vol_annual_proxy: `{_fmt(diag.get('risk_vol_annual_proxy'))}`\n")
            if diag.get("risk_factor_top_contributors"):
                lines.append("- top_factor_contributors:\n")
                for r in list(diag.get("risk_factor_top_contributors") or [])[:6]:
                    if isinstance(r, dict):
                        lines.append(
                            f"  - {r.get('factor')}: contrib={_fmt(r.get('contrib'))}, exposure={_fmt(r.get('exposure'))}\n"
                        )
            if diag.get("risk_idio_top_contributors"):
                lines.append("- top_idio_contributors:\n")
                for r in list(diag.get("risk_idio_top_contributors") or [])[:6]:
                    if isinstance(r, dict):
                        lines.append(
                            f"  - {r.get('instrument')}: contrib={_fmt(r.get('contrib'))}, weight={_fmt(r.get('weight'))}\n"
                        )
            if diag.get("risk_top_contributors"):
                lines.append("- top_name_contributors:\n")
                for r in list(diag.get("risk_top_contributors") or [])[:6]:
                    if isinstance(r, dict):
                        lines.append(
                            f"  - {r.get('instrument')}: contrib={_fmt(r.get('contrib'))}, weight={_fmt(r.get('weight'))}\n"
                        )

        # Objective terms (if available).
        ot = diag.get("objective_terms") if isinstance(diag, dict) else None
        if isinstance(ot, dict) and ot:
            lines.append("\n### Objective term breakdown (proxy)\n")
            for k in sorted(ot.keys()):
                lines.append(f"- {k}: `{_fmt(ot.get(k))}`\n")

        lines.append("\n")

    if yearly:
        lines.append("## OOS yearly breakdown\n")
        try:
            yd = []
            for y, v in yearly.items():
                if not isinstance(v, dict):
                    continue
                yd.append(
                    {
                        "year": str(y),
                        "information_ratio": v.get("information_ratio"),
                        "annualized_return": v.get("annualized_return"),
                        "max_drawdown": v.get("max_drawdown"),
                        "n_obs": v.get("n_obs"),
                    }
                )
            if yd:
                dfy = pd.DataFrame(yd)
                lines.append(dfy.to_markdown(index=False))
                lines.append("\n")
        except Exception:
            pass

    return "".join(lines)
