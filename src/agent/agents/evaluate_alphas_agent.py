# src/agent/agents/evaluate_alphas_agent.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import replace

import numpy as np
import pandas as pd

from langchain_core.runnables import RunnableConfig

from agent.state import State
from agent.services.market_data import MarketDataSpec, get_market_data
from agent.research.factor_runner import run_factors
from agent.research.alpha_eval import compute_forward_returns, evaluate_alpha
from agent.research.portfolio_backtest import BacktestConfig
from agent.research.regime_analysis import market_liquidity_regime, market_volatility_regime, regime_performance
from agent.research.cost_sensitivity import compute_cost_sensitivity
from agent.research.decay_analysis import compute_horizon_decay
from agent.research.schedule_sweep import compute_holding_rebalance_sweep
from agent.research.alpha_selection import (
    get_alpha_selection_preset,
    merge_selection_constraints,
    build_oos_return_matrix,
    compute_return_correlation,
    greedy_diversified_selection,
    correlation_summary,
    make_equal_weight_ensemble,
    tune_diverse_selection,
)
from agent.research.holdings_ensemble import (
    walk_forward_holdings_ensemble,
    walk_forward_holdings_ensemble_allocated,
    walk_forward_holdings_ensemble_allocated_regime,
)
from agent.research.tuning import default_sweep_param_lists, generate_sweep_configs
from agent.research.walk_forward import WalkForwardConfig, make_walk_forward_splits, walk_forward_evaluate_factor
from agent.services.metadata import load_borrow_rates, load_hard_to_borrow, load_sector_map


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_config_value(config: RunnableConfig, key: str, default: Any = None) -> Any:
    return (config.get("configurable") or {}).get(key, default)


def _normalize_alpha(alpha: Dict[str, Any]) -> Dict[str, Any]:
    alpha_id = alpha.get("alpha_id") or alpha.get("alphaID") or alpha.get("id")
    out = dict(alpha)
    out["alpha_id"] = alpha_id
    out["alphaID"] = alpha_id
    out["expression"] = alpha.get("expression") or alpha.get("expr") or ""
    out["description"] = alpha.get("description") or alpha.get("desc") or ""
    return out

def _parse_float_list(s: str) -> List[float]:
    """Parse a comma-separated list of floats."""

    if not s:
        return []
    out: List[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = float(part)
            if np.isfinite(v):
                out.append(float(v))
        except Exception:
            continue
    # Deduplicate while preserving order.
    seen = set()
    uniq: List[float] = []
    for v in out:
        key = round(float(v), 12)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(float(v))
    return uniq


def _parse_int_list(s: str) -> List[int]:
    """Parse a comma-separated list of ints."""

    if not s:
        return []
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(float(part))
            if v >= 1:
                out.append(int(v))
        except Exception:
            continue

    seen = set()
    uniq: List[int] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(int(v))
    return uniq


def _parse_str_list(s: str) -> List[str]:
    """Parse a comma-separated list of strings."""

    if not s:
        return []
    out: List[str] = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(part)

    seen = set()
    uniq: List[str] = []
    for v in out:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    return uniq


def _parse_kv_float_map(s: str) -> Dict[str, float]:
    """Parse a comma-separated key=value float map.

    Example:
        "objective=1,turnover_cost_drag_bps_mean=0.5,regime_switch_rate_mean=0.2"
    """

    if not s:
        return {}
    out: Dict[str, float] = {}
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
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


def _opt_float(x: Any) -> Optional[float]:
    """Parse an optional float (returns None on invalid)."""

    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return float(v) if np.isfinite(v) else None


def _opt_int(x: Any) -> Optional[int]:
    """Parse an optional int (returns None on invalid)."""

    if x is None:
        return None
    try:
        v = int(float(x))
    except Exception:
        return None
    return int(v)



def _best_row_by_metric(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    """Pick the best row by a numeric metric (descending)."""

    best: Optional[Dict[str, Any]] = None
    best_v = float("-inf")
    for r in rows:
        if not isinstance(r, dict) or r.get("error"):
            continue
        try:
            v = float(r.get(metric) or float("-inf"))
        except Exception:
            continue
        if np.isfinite(v) and v > best_v:
            best_v = v
            best = r
    return best


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


def _apply_quality_gates(
    metrics: Dict[str, Any],
    *,
    coverage_mean: float,
    min_coverage: float,
    max_turnover: float,
    min_ir: float | None,
    max_drawdown: float | None,
    max_total_cost_bps: float | None,
    min_wf_splits: int | None,
) -> Dict[str, Any]:
    """Compute a consistent quality gate decision."""

    passed = True
    reasons: List[str] = []

    if np.isfinite(coverage_mean) and float(min_coverage) > 0.0 and float(coverage_mean) < float(min_coverage):
        passed = False
        reasons.append("coverage")

    to = metrics.get("turnover_mean")
    if isinstance(to, (int, float)) and np.isfinite(float(to)) and float(max_turnover) > 0.0 and float(to) > float(max_turnover):
        passed = False
        reasons.append("turnover")

    ir = metrics.get("information_ratio")
    if min_ir is not None:
        try:
            if float(ir) < float(min_ir):
                passed = False
                reasons.append("min_ir")
        except Exception:
            pass

    mdd = metrics.get("max_drawdown")
    if max_drawdown is not None:
        try:
            # max_drawdown is <= 0. A threshold of 0.2 means "no worse than -0.2".
            thr = -abs(float(max_drawdown))
            if float(mdd) < thr:
                passed = False
                reasons.append("drawdown")
        except Exception:
            pass

    if max_total_cost_bps is not None:
        tcb = _total_cost_bps(metrics)
        try:
            if np.isfinite(tcb) and float(tcb) > float(max_total_cost_bps):
                passed = False
                reasons.append("cost")
        except Exception:
            pass

    if min_wf_splits is not None:
        try:
            wf = metrics.get("walk_forward") or {}
            stab = (wf.get("stability") or {}) if isinstance(wf, dict) else {}
            n_splits = int(stab.get("n_splits") or 0)
            if n_splits < int(min_wf_splits):
                passed = False
                reasons.append("wf_splits")
        except Exception:
            pass

    return {"passed": bool(passed), "reasons": reasons}



async def evaluate_alphas_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Run coded factors on data and compute metrics.

    Outputs:
    - coded_alphas: each alpha enriched with `backtest_results`
    - sota_alphas: top-K alphas selected by information_ratio
    """

    coded_alphas: List[Dict[str, Any]] = _get_attr_or_key(state, "coded_alphas", []) or []
    if not coded_alphas:
        return {}

    data_path = str(_get_config_value(config, "data_path", "") or "")
    eval_mode = str(_get_config_value(config, "eval_mode", "p2") or "p2").strip().lower()
    horizon = int(_get_config_value(config, "horizon", 1))
    top_k = int(_get_config_value(config, "top_k", 3))
    n_quantiles = int(_get_config_value(config, "n_quantiles", 5))

    cost_bps = float(_get_config_value(config, "cost_bps", 0.0))
    min_obs_per_day = int(_get_config_value(config, "min_obs_per_day", 20))
    min_coverage = float(_get_config_value(config, "min_coverage", 0.0))
    max_turnover = float(_get_config_value(config, "max_turnover", 1.0))

    # P2.12 tuning (grid search) + ablation + extra quality gates
    tune_enabled = bool(_get_config_value(config, "tune", False))
    tune_metric = str(_get_config_value(config, "tune_metric", "information_ratio") or "information_ratio").strip()
    tune_max_combos = int(_get_config_value(config, "tune_max_combos", 24))
    tune_save_top = int(_get_config_value(config, "tune_save_top", 10))
    tune_turnover_cap = str(_get_config_value(config, "tune_turnover_cap", "") or "")
    tune_max_abs_weight = str(_get_config_value(config, "tune_max_abs_weight", "") or "")
    tune_risk_aversion = str(_get_config_value(config, "tune_risk_aversion", "") or "")
    tune_cost_aversion = str(_get_config_value(config, "tune_cost_aversion", "") or "")
    ablation_top = int(_get_config_value(config, "ablation_top", 1))
    ablation_mode = str(_get_config_value(config, "ablation_mode", "both") or "both").strip().lower()

    regime_analysis = bool(_get_config_value(config, "regime_analysis", True))
    regime_window = int(_get_config_value(config, "regime_window", 20))
    regime_buckets = int(_get_config_value(config, "regime_buckets", 3))

    # P2.14: execution-only cost sensitivity curves (top-N).
    cost_sensitivity = bool(_get_config_value(config, "cost_sensitivity", True))
    cost_sensitivity_top = int(_get_config_value(config, "cost_sensitivity_top", 1))
    cs_linear_bps = str(_get_config_value(config, "cost_sensitivity_linear_bps", "") or "")
    cs_half_spread_bps = str(_get_config_value(config, "cost_sensitivity_half_spread_bps", "") or "")
    cs_impact_bps = str(_get_config_value(config, "cost_sensitivity_impact_bps", "") or "")
    cs_borrow_bps = str(_get_config_value(config, "cost_sensitivity_borrow_bps", "") or "")
    cs_borrow_mult = str(_get_config_value(config, "cost_sensitivity_borrow_mult", "") or "")

    # P2.15: multi-horizon decay analysis (IC/spread + signal persistence).
    decay_analysis = bool(_get_config_value(config, "decay_analysis", True))
    decay_analysis_top = int(_get_config_value(config, "decay_analysis_top", 1))
    decay_horizons_s = str(_get_config_value(config, "decay_horizons", "") or "")

    # P2.16: strategy-level schedule sweep (rebalance_days x holding_days).
    schedule_sweep = bool(_get_config_value(config, "schedule_sweep", True))
    schedule_sweep_top = int(_get_config_value(config, "schedule_sweep_top", 1))
    schedule_sweep_metric = str(_get_config_value(config, "schedule_sweep_metric", "information_ratio") or "information_ratio")
    schedule_sweep_max_combos = int(_get_config_value(config, "schedule_sweep_max_combos", 25))
    schedule_sweep_rebalance_s = str(_get_config_value(config, "schedule_sweep_rebalance_days", "") or "")
    schedule_sweep_holding_s = str(_get_config_value(config, "schedule_sweep_holding_days", "") or "")

    # P2.17: diversity-aware top-K selection and a simple ensemble.
    diverse_selection = bool(_get_config_value(config, "diverse_selection", True))
    diverse_lambda = float(_get_config_value(config, "diverse_lambda", 0.5))
    diverse_use_abs_corr = bool(_get_config_value(config, "diverse_use_abs_corr", True))
    diverse_candidate_pool = int(_get_config_value(config, "diverse_candidate_pool", 20))
    diverse_min_periods = int(_get_config_value(config, "diverse_min_periods", 20))
    ensemble_enabled = bool(_get_config_value(config, "ensemble", True))

    # P2.22: selection meta-tuning on validation returns (no test leakage).
    selection_tune = bool(_get_config_value(config, "selection_tune", True))
    selection_tune_metric = str(_get_config_value(config, "selection_tune_metric", "information_ratio") or "information_ratio")
    selection_tune_max_combos = int(_get_config_value(config, "selection_tune_max_combos", 24))
    selection_tune_min_periods = int(_get_config_value(config, "selection_tune_min_periods", diverse_min_periods))

    selection_tune_lambda_grid = _parse_float_list(str(_get_config_value(config, "selection_tune_lambda_grid", "") or ""))
    selection_tune_candidate_pool_grid = _parse_int_list(
    str(_get_config_value(config, "selection_tune_candidate_pool_grid", "") or "")
    )
    selection_tune_topk_grid = _parse_int_list(str(_get_config_value(config, "selection_tune_topk_grid", "") or ""))


    # P2.31: alpha selection constraints / presets.
    alpha_selection_preset = str(_get_config_value(config, "alpha_selection_preset", "") or "").strip()
    alpha_selection_constraints = merge_selection_constraints(
        get_alpha_selection_preset(alpha_selection_preset) if alpha_selection_preset else {},
        {
            "max_pairwise_corr": _opt_float(_get_config_value(config, "alpha_selection_max_pairwise_corr", None)),
            "min_valid_ir": _opt_float(_get_config_value(config, "alpha_selection_min_valid_ir", None)),
            "min_valid_coverage": _opt_float(_get_config_value(config, "alpha_selection_min_valid_coverage", None)),
            "max_total_cost_bps": _opt_float(_get_config_value(config, "alpha_selection_max_total_cost_bps", None)),
            "min_wf_test_ir_positive_frac": _opt_float(
                _get_config_value(config, "alpha_selection_min_wf_test_ir_positive_frac", None)
            ),
        },
    )


    # P2.31: enforce max_pairwise_corr even when diversity selection is disabled.
    if (not selection_done) and eval_mode in {"p1", "p2"} and int(top_k) > 1 and scored_pool:
        try:
            max_pairwise_corr = None
            try:
                max_pairwise_corr = float((alpha_selection_constraints or {}).get("max_pairwise_corr"))
                if (not np.isfinite(max_pairwise_corr)) or max_pairwise_corr <= 0.0:
                    max_pairwise_corr = None
            except Exception:
                max_pairwise_corr = None

            if max_pairwise_corr is not None:
                pool = scored_pool
                pool_n = int(diverse_candidate_pool) if int(diverse_candidate_pool) > 0 else len(pool)
                pool = pool[: max(pool_n, int(top_k))]
                pool_alphas = [p["alpha"] for p in pool]

                oos_mat = build_oos_return_matrix(pool_alphas, value_key="net_return")
                if oos_mat is not None and not oos_mat.empty and oos_mat.shape[1] >= 2:
                    corr, nobs = compute_return_correlation(oos_mat, min_periods=int(diverse_min_periods))

                    scores = {}
                    for p in pool:
                        a = p.get("alpha") or {}
                        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
                        if aid:
                            scores[str(aid)] = float(p.get("score") or 0.0)

                    selected_ids, table, rejected = greedy_diversified_selection(
                        scores=scores,
                        corr=corr,
                        k=int(top_k),
                        diversity_lambda=0.0,
                        use_abs_corr=bool(diverse_use_abs_corr),
                        max_pairwise_corr=max_pairwise_corr,
                    )

                    by_id = {str((a.get("alpha_id") or a.get("alphaID") or a.get("id"))): a for a in pool_alphas}
                    selected_alphas = [by_id.get(i) for i in selected_ids if i in by_id]
                    selected_alphas = [a for a in selected_alphas if isinstance(a, dict)]

                    if selected_alphas:
                        sota_alphas = selected_alphas
                        selection_meta.update(
                            {
                                "method": "score_rank_constrained",
                                "domain": "test",
                                "selected_alpha_ids": selected_ids,
                                "selection_table": table,
                                "rejected": rejected,
                                "correlation_summary": correlation_summary(corr, selected_ids),
                                "min_periods": int(diverse_min_periods),
                                "correlation_domain": "test",
                            }
                        )

                        alpha_correlation = {
                            "method": "pearson",
                            "min_periods": int(diverse_min_periods),
                            "matrix": corr.to_dict(orient="split"),
                            "nobs": nobs.to_dict(orient="split"),
                            "alpha_ids": list(oos_mat.columns),
                        }

                        if bool(ensemble_enabled):
                            td = int(getattr(bt_cfg_base, "trading_days", 252)) if bt_cfg_base is not None else 252
                            ensemble = make_equal_weight_ensemble(oos_mat, selected_ids, trading_days=td)
        except Exception:
            pass



    # P2.18: holdings-level ensemble (combine weights then re-price with costs).
    holdings_ensemble_enabled = bool(_get_config_value(config, "holdings_ensemble", True))
    holdings_ensemble_apply_turnover_cap = bool(
        _get_config_value(config, "holdings_ensemble_apply_turnover_cap", False)
    )

    # P2.19: alpha allocation (learn weights across selected alphas).
    alpha_allocation_enabled = bool(_get_config_value(config, "alpha_allocation", True))
    alpha_allocation_backend = str(_get_config_value(config, "alpha_allocation_backend", "auto") or "auto")
    alpha_allocation_fit = str(_get_config_value(config, "alpha_allocation_fit", "train_valid") or "train_valid")
    alpha_allocation_score_metric = str(
        _get_config_value(config, "alpha_allocation_score_metric", "information_ratio") or "information_ratio"
    )
    alpha_allocation_lambda = float(_get_config_value(config, "alpha_allocation_lambda", 0.5))
    alpha_allocation_l2 = float(_get_config_value(config, "alpha_allocation_l2", 1e-6))
    alpha_allocation_turnover_lambda = float(_get_config_value(config, "alpha_allocation_turnover_lambda", 0.0))
    alpha_allocation_max_weight = float(_get_config_value(config, "alpha_allocation_max_weight", 0.8))
    alpha_allocation_use_abs_corr = bool(_get_config_value(config, "alpha_allocation_use_abs_corr", True))
    alpha_allocation_min_days = int(_get_config_value(config, "alpha_allocation_min_days", 30))
    alpha_allocation_solver = str(_get_config_value(config, "alpha_allocation_solver", "") or "")

    # P2.20: meta-tune allocation hyperparams.
    alpha_allocation_tune = bool(_get_config_value(config, "alpha_allocation_tune", False))
    alpha_allocation_tune_metric = str(
        _get_config_value(config, "alpha_allocation_tune_metric", "information_ratio") or "information_ratio"
    )
    alpha_allocation_tune_max_combos = int(_get_config_value(config, "alpha_allocation_tune_max_combos", 24))
    alpha_allocation_tune_lambda_grid = _parse_float_list(
        str(_get_config_value(config, "alpha_allocation_tune_lambda_grid", "") or "")
    )
    alpha_allocation_tune_max_weight_grid = _parse_float_list(
        str(_get_config_value(config, "alpha_allocation_tune_max_weight_grid", "") or "")
    )
    alpha_allocation_tune_turnover_lambda_grid = _parse_float_list(
        str(_get_config_value(config, "alpha_allocation_tune_turnover_lambda_grid", "") or "")
    )
    alpha_allocation_tune_save_top = int(_get_config_value(config, "alpha_allocation_tune_save_top", 10))

    # P2.21: regime-aware allocation (optional).
    alpha_allocation_regime_aware = bool(_get_config_value(config, "alpha_allocation_regime_aware", False))
    alpha_allocation_regime_mode = str(_get_config_value(config, "alpha_allocation_regime_mode", "vol") or "vol")
    alpha_allocation_regime_window = int(_get_config_value(config, "alpha_allocation_regime_window", 20))
    alpha_allocation_regime_buckets = int(_get_config_value(config, "alpha_allocation_regime_buckets", 3))
    alpha_allocation_regime_min_days = int(_get_config_value(config, "alpha_allocation_regime_min_days", 30))
    alpha_allocation_regime_smoothing = float(_get_config_value(config, "alpha_allocation_regime_smoothing", 0.0))

    # P2.23: meta-tune regime hyperparams on aggregate validation performance.
    alpha_allocation_regime_tune = bool(_get_config_value(config, "alpha_allocation_regime_tune", False))
    alpha_allocation_regime_tune_metric = str(
        _get_config_value(config, "alpha_allocation_regime_tune_metric", "information_ratio") or "information_ratio"
    )
    alpha_allocation_regime_tune_max_combos = int(_get_config_value(config, "alpha_allocation_regime_tune_max_combos", 24))
    alpha_allocation_regime_tune_mode_grid = _parse_str_list(
        str(_get_config_value(config, "alpha_allocation_regime_tune_mode_grid", "") or "")
    )
    alpha_allocation_regime_tune_window_grid = _parse_int_list(
        str(_get_config_value(config, "alpha_allocation_regime_tune_window_grid", "") or "")
    )
    alpha_allocation_regime_tune_buckets_grid = _parse_int_list(
        str(_get_config_value(config, "alpha_allocation_regime_tune_buckets_grid", "") or "")
    )
    alpha_allocation_regime_tune_smoothing_grid = _parse_float_list(
        str(_get_config_value(config, "alpha_allocation_regime_tune_smoothing_grid", "") or "")
    )
    alpha_allocation_regime_tune_turnover_penalty = float(
        _get_config_value(
            config,
            "alpha_allocation_regime_tune_turnover_cost_bps",
            _get_config_value(config, "alpha_allocation_regime_tune_turnover_penalty", 0.0),
        )
    )
    alpha_allocation_regime_tune_save_top = int(_get_config_value(config, "alpha_allocation_regime_tune_save_top", 10))

    # P2.24: holdings-level revalidation for top proxy regime configs.
    alpha_allocation_regime_tune_holdings_top = int(
        _get_config_value(config, "alpha_allocation_regime_tune_holdings_top", 3)
    )
    alpha_allocation_regime_tune_holdings_metric = str(
        _get_config_value(config, "alpha_allocation_regime_tune_holdings_metric", "") or ""
    )
    alpha_allocation_regime_tune_holdings_save_top = int(
        _get_config_value(config, "alpha_allocation_regime_tune_holdings_save_top", 10)
    )

    # P2.27: multi-objective Pareto settings.
    alpha_allocation_regime_tune_pareto_metrics = _parse_str_list(
        str(_get_config_value(config, "alpha_allocation_regime_tune_pareto_metrics", "") or "")
    )
    alpha_allocation_regime_tune_plots = bool(_get_config_value(config, "alpha_allocation_regime_tune_plots", True))

    # P2.26: constraint-based selection of regime configs.
    alpha_allocation_regime_tune_max_alpha_turnover = _opt_float(
        _get_config_value(config, "alpha_allocation_regime_tune_max_alpha_turnover", None)
    )
    alpha_allocation_regime_tune_max_turnover_cost_drag_bps = _opt_float(
        _get_config_value(config, "alpha_allocation_regime_tune_max_turnover_cost_drag_bps", None)
    )
    alpha_allocation_regime_tune_max_regime_switch_rate = _opt_float(
        _get_config_value(config, "alpha_allocation_regime_tune_max_regime_switch_rate", None)
    )
    alpha_allocation_regime_tune_max_fallback_frac = _opt_float(
        _get_config_value(config, "alpha_allocation_regime_tune_max_fallback_frac", None)
    )
    alpha_allocation_regime_tune_prefer_pareto = bool(
        _get_config_value(config, "alpha_allocation_regime_tune_prefer_pareto", False)
    )

    # P2.28: optional Pareto-based auto selection.
    alpha_allocation_regime_tune_selection_method = str(
        _get_config_value(config, "alpha_allocation_regime_tune_selection_method", "best_objective") or "best_objective"
    )
    alpha_allocation_regime_tune_utility_weights = _parse_kv_float_map(
        str(_get_config_value(config, "alpha_allocation_regime_tune_utility_weights", "") or "")
    )
    _tmp_stab = _get_config_value(config, "alpha_allocation_regime_tune_include_stability_objectives", None)
    alpha_allocation_regime_tune_include_stability_objectives = True if _tmp_stab is None else bool(_tmp_stab)


    min_ir = _get_config_value(config, "min_ir", None)
    max_dd = _get_config_value(config, "max_dd", None)
    max_total_cost_bps = _get_config_value(config, "max_total_cost_bps", None)
    min_wf_splits = _get_config_value(config, "min_wf_splits", None)


    max_nan_ratio = float(_get_config_value(config, "max_nan_ratio", 0.95))
    max_rows = int(_get_config_value(config, "max_rows", 2_000_000))
    enable_code_safety = bool(_get_config_value(config, "enable_code_safety", True))
    max_code_chars = int(_get_config_value(config, "max_code_chars", 20_000))
    prefer_dsl = bool(_get_config_value(config, "prefer_dsl", True))
    allow_python_exec = bool(_get_config_value(config, "allow_python_exec", False))
    max_dsl_chars = int(_get_config_value(config, "max_dsl_chars", 5_000))
    python_exec_timeout_sec = float(_get_config_value(config, "python_exec_timeout_sec", 2.0))

    # P1 (walk-forward + portfolio backtest)
    wf_train_days = int(_get_config_value(config, "wf_train_days", 126))
    wf_valid_days = int(_get_config_value(config, "wf_valid_days", 42))
    wf_test_days = int(_get_config_value(config, "wf_test_days", 42))
    wf_step_days = int(_get_config_value(config, "wf_step_days", wf_test_days))
    wf_expanding_train = bool(_get_config_value(config, "wf_expanding_train", True))

    rebalance_days = int(_get_config_value(config, "rebalance_days", 5))
    holding_days = int(_get_config_value(config, "holding_days", 5))
    commission_bps = float(_get_config_value(config, "commission_bps", 0.0))
    slippage_bps = float(_get_config_value(config, "slippage_bps", 0.0))
    borrow_bps = float(_get_config_value(config, "borrow_bps", 0.0))

    # P2.2 cost / borrow constraints
    half_spread_bps = float(_get_config_value(config, "half_spread_bps", 0.0))
    impact_bps = float(_get_config_value(config, "impact_bps", 0.0))
    impact_exponent = float(_get_config_value(config, "impact_exponent", 0.5))
    impact_max_participation = float(_get_config_value(config, "impact_max_participation", 0.2))
    portfolio_notional = float(_get_config_value(config, "portfolio_notional", 1_000_000.0))
    turnover_cap = float(_get_config_value(config, "turnover_cap", 0.0))
    max_borrow_bps = float(_get_config_value(config, "max_borrow_bps", 0.0))

    hard_to_borrow_path = str(_get_config_value(config, "hard_to_borrow_path", "") or "")
    borrow_rates_path = str(_get_config_value(config, "borrow_rates_path", "") or "")
    hard_to_borrow = None
    if hard_to_borrow_path:
        try:
            hard_to_borrow = load_hard_to_borrow(hard_to_borrow_path)
        except Exception:
            hard_to_borrow = None
    borrow_rates = None
    if borrow_rates_path:
        try:
            borrow_rates = load_borrow_rates(borrow_rates_path)
        except Exception:
            borrow_rates = None

    # P2 risk / constraints
    max_abs_weight = float(_get_config_value(config, "max_abs_weight", 0.0))
    max_names_per_side = int(_get_config_value(config, "max_names_per_side", 0))
    min_adv = float(_get_config_value(config, "min_adv", 0.0))
    adv_window = int(_get_config_value(config, "adv_window", 20))
    beta_window = int(_get_config_value(config, "beta_window", 60))
    vol_window = int(_get_config_value(config, "vol_window", 20))
    neutralize_beta = bool(_get_config_value(config, "neutralize_beta", False))
    neutralize_vol = bool(_get_config_value(config, "neutralize_vol", False))
    neutralize_liquidity = bool(_get_config_value(config, "neutralize_liquidity", False))
    neutralize_sector = bool(_get_config_value(config, "neutralize_sector", False))
    target_vol_annual = float(_get_config_value(config, "target_vol_annual", 0.0))
    vol_target_window = int(_get_config_value(config, "vol_target_window", 20))
    vol_target_max_leverage = float(_get_config_value(config, "vol_target_max_leverage", 3.0))
    sector_map_path = str(_get_config_value(config, "sector_map_path", "") or "")

    # P2.4 portfolio construction
    construction_method = str(_get_config_value(config, "construction_method", "heuristic") or "heuristic")
    optimizer_l2_lambda = float(_get_config_value(config, "optimizer_l2_lambda", 1.0))
    optimizer_turnover_lambda = float(_get_config_value(config, "optimizer_turnover_lambda", 10.0))
    optimizer_exposure_lambda = float(_get_config_value(config, "optimizer_exposure_lambda", 0.0))
    optimizer_max_iter = int(_get_config_value(config, "optimizer_max_iter", 2))

    # P2.5 constrained optimizer backend (optional)
    optimizer_backend = str(_get_config_value(config, "optimizer_backend", "auto") or "auto")
    optimizer_turnover_cap = float(_get_config_value(config, "optimizer_turnover_cap", 0.0))
    optimizer_solver = str(_get_config_value(config, "optimizer_solver", "") or "")

    # P2.6 cost-aware optimizer settings (used by QP backend)
    optimizer_cost_aversion = float(_get_config_value(config, "optimizer_cost_aversion", 1.0))
    optimizer_exposure_slack_lambda = float(_get_config_value(config, "optimizer_exposure_slack_lambda", 0.0))
    optimizer_enforce_participation = bool(_get_config_value(config, "optimizer_enforce_participation", True))

    # P2.8 diagonal risk term inside optimizer (variance proxy)
    optimizer_risk_aversion = float(_get_config_value(config, "optimizer_risk_aversion", 0.0))
    optimizer_risk_window = int(_get_config_value(config, "optimizer_risk_window", vol_window))

    # P2.9 factor risk model (optional)
    optimizer_risk_model = str(_get_config_value(config, "optimizer_risk_model", "diag") or "diag")
    optimizer_factor_risk_window = int(_get_config_value(config, "optimizer_factor_risk_window", 60))
    optimizer_factor_risk_shrink = float(_get_config_value(config, "optimizer_factor_risk_shrink", 0.2))
    optimizer_factor_risk_ridge = float(_get_config_value(config, "optimizer_factor_risk_ridge", 1e-3))

    # P2.10 robust factor risk estimation knobs
    optimizer_factor_risk_estimator = str(_get_config_value(config, "optimizer_factor_risk_estimator", "sample"))
    optimizer_factor_risk_shrink_method = str(_get_config_value(config, "optimizer_factor_risk_shrink_method", "fixed"))
    optimizer_factor_risk_ewm_halflife = float(_get_config_value(config, "optimizer_factor_risk_ewm_halflife", 20.0))
    optimizer_factor_return_clip_sigma = float(_get_config_value(config, "optimizer_factor_return_clip_sigma", 6.0))
    optimizer_idio_shrink = float(_get_config_value(config, "optimizer_idio_shrink", optimizer_factor_risk_shrink))
    optimizer_idio_clip_q = float(_get_config_value(config, "optimizer_idio_clip_q", 0.99))

    sector_map = None
    if sector_map_path:
        try:
            sector_map = load_sector_map(sector_map_path)
        except Exception:
            sector_map = None

    n_days = int(_get_config_value(config, "synthetic_n_days", 252))
    n_instruments = int(_get_config_value(config, "synthetic_n_instruments", 50))
    seed = int(_get_config_value(config, "synthetic_seed", 7))

    horizon = max(1, horizon)
    n_quantiles = max(2, n_quantiles)
    top_k = max(0, top_k)

    try:
        df = get_market_data(
            MarketDataSpec(
                path=data_path,
                n_days=n_days,
                n_instruments=n_instruments,
                seed=seed,
            )
        )
    except Exception as e:
        enriched = []
        for a in coded_alphas:
            alpha = _normalize_alpha(a)
            alpha["backtest_results"] = {"error": f"DataError: {e.__class__.__name__}: {e}"}
            enriched.append(alpha)
        return {"coded_alphas": enriched, "sota_alphas": []}

    universe_size_by_date = df["close"].groupby(level="datetime").size()

    fwd = compute_forward_returns(df["close"], horizon=horizon)

    # Optional metadata available to the DSL (e.g., for group neutralization).
    inst_index = df.index.get_level_values("instrument").astype(str)
    if sector_map:
        sector = pd.Series(inst_index, index=df.index).map(sector_map).fillna("UNKNOWN").astype(str)
        sector.name = "sector"
    else:
        sector = pd.Series("UNKNOWN", index=df.index, name="sector")
    dsl_extra_env = {"sector": sector}

    run_results = run_factors(
        df,
        coded_alphas,
        prefer_dsl=prefer_dsl,
        allow_python_exec=allow_python_exec,
        dsl_extra_env=dsl_extra_env,
        max_nan_ratio=max_nan_ratio,
        max_rows=max_rows,
        enable_code_safety=enable_code_safety,
        max_code_chars=max_code_chars,
        max_dsl_chars=max_dsl_chars,
        python_exec_timeout_sec=python_exec_timeout_sec,
    )

    enriched: List[Dict[str, Any]] = []
    scored: List[Dict[str, Any]] = []

    # Cache factor values in-memory for optional post-selection analysis (e.g., ablations).
    factor_cache: Dict[str, Any] = {}
    best_bt_cfg_by_alpha: Dict[str, BacktestConfig] = {}
    alpha_by_id: Dict[str, Dict[str, Any]] = {}

    # Build walk-forward configs once (and reuse splits across alphas and sweeps).
    wf_cfg = None
    bt_cfg_base = None
    wf_splits = None
    if eval_mode in {"p1", "p2"}:
        wf_cfg = WalkForwardConfig(
            train_days=wf_train_days,
            valid_days=wf_valid_days,
            test_days=wf_test_days,
            step_days=wf_step_days,
            expanding_train=wf_expanding_train,
        )
        bt_cfg_base = BacktestConfig(
            rebalance_days=rebalance_days,
            holding_days=holding_days,
            n_quantiles=n_quantiles,
            min_obs=min_obs_per_day,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            borrow_bps=borrow_bps,
            half_spread_bps=half_spread_bps,
            impact_bps=impact_bps,
            impact_exponent=impact_exponent,
            impact_max_participation=impact_max_participation,
            portfolio_notional=portfolio_notional,
            turnover_cap=turnover_cap,
            max_borrow_bps=max_borrow_bps,
            max_abs_weight=max_abs_weight,
            max_names_per_side=max_names_per_side,
            min_adv=min_adv,
            adv_window=adv_window,
            beta_window=beta_window,
            vol_window=vol_window,
            neutralize_beta=(neutralize_beta if eval_mode == "p2" else False),
            neutralize_vol=(neutralize_vol if eval_mode == "p2" else False),
            neutralize_liquidity=(neutralize_liquidity if eval_mode == "p2" else False),
            neutralize_sector=(neutralize_sector if eval_mode == "p2" else False),
            target_vol_annual=(target_vol_annual if eval_mode == "p2" else 0.0),
            vol_target_window=vol_target_window,
            vol_target_max_leverage=vol_target_max_leverage,
            construction_method=construction_method,
            optimizer_l2_lambda=optimizer_l2_lambda,
            optimizer_turnover_lambda=optimizer_turnover_lambda,
            optimizer_exposure_lambda=optimizer_exposure_lambda,
            optimizer_max_iter=optimizer_max_iter,
            optimizer_backend=optimizer_backend,
            optimizer_turnover_cap=optimizer_turnover_cap,
            optimizer_solver=optimizer_solver,
            optimizer_cost_aversion=optimizer_cost_aversion,
            optimizer_exposure_slack_lambda=optimizer_exposure_slack_lambda,
            optimizer_enforce_participation=optimizer_enforce_participation,
            optimizer_risk_aversion=optimizer_risk_aversion,
            optimizer_risk_window=optimizer_risk_window,
            optimizer_risk_model=optimizer_risk_model,
            optimizer_factor_risk_window=optimizer_factor_risk_window,
            optimizer_factor_risk_shrink=optimizer_factor_risk_shrink,
            optimizer_factor_risk_ridge=optimizer_factor_risk_ridge,
            optimizer_factor_risk_estimator=optimizer_factor_risk_estimator,
            optimizer_factor_risk_shrink_method=optimizer_factor_risk_shrink_method,
            optimizer_factor_risk_ewm_halflife=optimizer_factor_risk_ewm_halflife,
            optimizer_factor_return_clip_sigma=optimizer_factor_return_clip_sigma,
            optimizer_idio_shrink=optimizer_idio_shrink,
            optimizer_idio_clip_q=optimizer_idio_clip_q,
        )
        try:
            close = df["close"]
            dates = np.sort(pd.to_datetime(close.index.get_level_values("datetime").unique()))
            wf_splits = make_walk_forward_splits(
                dates,
                train_days=wf_train_days,
                valid_days=wf_valid_days,
                test_days=wf_test_days,
                step_days=wf_step_days,
                expanding_train=wf_expanding_train,
            )
        except Exception:
            wf_splits = None

    for rr in run_results:
        alpha = next((a for a in coded_alphas if (a.get("alpha_id") or a.get("alphaID")) == rr.alpha_id), None)
        alpha = _normalize_alpha(alpha or {"alpha_id": rr.alpha_id})
        alpha_by_id[str(alpha["alpha_id"])] = alpha

        if rr.error or rr.values is None:
            alpha["backtest_results"] = {"error": rr.error or "Unknown error"}
            enriched.append(alpha)
            continue

        factor_series = rr.values[rr.alpha_id]
        factor_cache[str(rr.alpha_id)] = factor_series

        # Coverage is a simple sanity check for both P0/P1 and also used for gates.
        cov_mean = float("nan")
        try:
            fac_wide = factor_series.unstack("instrument")
            cov = fac_wide.notna().sum(axis=1) / float(max(1, fac_wide.shape[1]))
            cov_mean = float(cov.mean())
        except Exception:
            pass

        metrics: Dict[str, Any] = {}

        if eval_mode in {"p1", "p2"} and wf_cfg is not None and bt_cfg_base is not None and wf_splits:
            # Optional tuning: grid-search a small set of BacktestConfig knobs.
            if tune_enabled and str(getattr(bt_cfg_base, "construction_method", "")).lower() == "optimizer":
                param_lists = default_sweep_param_lists(bt_cfg_base)
                if tune_turnover_cap:
                    vals = _parse_float_list(tune_turnover_cap)
                    if vals:
                        param_lists["optimizer_turnover_cap"] = vals
                if tune_max_abs_weight:
                    vals = _parse_float_list(tune_max_abs_weight)
                    if vals:
                        param_lists["max_abs_weight"] = vals
                if tune_risk_aversion:
                    vals = _parse_float_list(tune_risk_aversion)
                    if vals:
                        param_lists["optimizer_risk_aversion"] = vals
                if tune_cost_aversion:
                    vals = _parse_float_list(tune_cost_aversion)
                    if vals:
                        param_lists["optimizer_cost_aversion"] = vals

                sweep_cfgs = generate_sweep_configs(bt_cfg_base, param_lists=param_lists, max_combos=tune_max_combos)

                sweep_rows: List[Dict[str, Any]] = []
                best_metrics: Dict[str, Any] | None = None
                best_cfg: BacktestConfig | None = None
                best_score = float("-inf")
                best_raw = float("-inf")
                best_meta: Dict[str, Any] = {"config_id": "base", "params": {}}

                for sc in sweep_cfgs:
                    bt_cfg = sc.get("bt_config") or bt_cfg_base
                    met = walk_forward_evaluate_factor(
                        factor_series,
                        df,
                        wf_config=wf_cfg,
                        bt_config=bt_cfg,
                        splits=wf_splits,
                        sector_map=sector_map,
                        borrow_rates=borrow_rates,
                        hard_to_borrow=hard_to_borrow,
                    )
                    if "error" in met:
                        sweep_rows.append(
                            {
                                "config_id": sc.get("config_id") or "unknown",
                                "params": sc.get("params") or {},
                                "error": met.get("error"),
                            }
                        )
                        continue

                    met["mode"] = eval_mode
                    qg = _apply_quality_gates(
                        met,
                        coverage_mean=float(cov_mean),
                        min_coverage=float(min_coverage),
                        max_turnover=float(max_turnover),
                        min_ir=(float(min_ir) if min_ir is not None else None),
                        max_drawdown=(float(max_dd) if max_dd is not None else None),
                        max_total_cost_bps=(float(max_total_cost_bps) if max_total_cost_bps is not None else None),
                        min_wf_splits=(int(min_wf_splits) if min_wf_splits is not None else None),
                    )

                    raw = float(met.get(tune_metric) or 0.0) if tune_metric else float(met.get("information_ratio") or 0.0)
                    score = raw if bool(qg.get("passed")) else float("-inf")

                    row = {
                        "config_id": sc.get("config_id") or "unknown",
                        "params": sc.get("params") or {},
                        "passed": bool(qg.get("passed")),
                        "reasons": list(qg.get("reasons") or []),
                        "score": float(score),
                        "raw_score": float(raw),
                        "information_ratio": met.get("information_ratio"),
                        "annualized_return": met.get("annualized_return"),
                        "max_drawdown": met.get("max_drawdown"),
                        "turnover_mean": met.get("turnover_mean"),
                        "total_cost_bps": float(_total_cost_bps(met)),
                    }
                    sweep_rows.append(row)

                    if bool(qg.get("passed")) and float(score) > float(best_score):
                        best_score = float(score)
                        best_metrics = met
                        best_cfg = bt_cfg
                        best_meta = {"config_id": row["config_id"], "params": row["params"], "score": float(raw), "passed": True}

                    # Fallback choice: highest raw score (even if it failed gates).
                    if best_metrics is None and float(raw) > float(best_raw):
                        best_raw = float(raw)
                        best_metrics = met
                        best_cfg = bt_cfg
                        best_meta = {"config_id": row["config_id"], "params": row["params"], "score": float(raw), "passed": bool(qg.get("passed"))}

                if best_metrics is not None:
                    sweep_sorted = sorted(
                        sweep_rows,
                        key=lambda r: (
                            float(r.get("score") or float("-inf")),
                            float(r.get("raw_score") or float("-inf")),
                        ),
                        reverse=True,
                    )
                    if tune_save_top > 0:
                        sweep_sorted = sweep_sorted[: int(tune_save_top)]

                    best_metrics.setdefault("tuning", {})
                    best_metrics["tuning"]["sweep"] = {
                        "enabled": True,
                        "metric": tune_metric,
                        "n_evaluated": int(len(sweep_rows)),
                        "best": best_meta,
                        "results": sweep_sorted,
                    }
                    metrics = best_metrics

                    # Ensure validation daily returns are available for selection meta-tuning.
                    if bool(selection_tune) and eval_mode in {"p1", "p2"} and best_cfg is not None:
                        try:
                            met_v = walk_forward_evaluate_factor(
                                factor_series,
                                df,
                                wf_config=wf_cfg,
                                bt_config=best_cfg,
                                splits=wf_splits,
                                sector_map=sector_map,
                                borrow_rates=borrow_rates,
                                hard_to_borrow=hard_to_borrow,
                                return_valid_daily=True,
                            )
                            wf_v = (met_v.get("walk_forward") or {}).get("valid_daily") if isinstance(met_v, dict) else None
                            if isinstance(wf_v, list) and wf_v:
                                best_metrics.setdefault("walk_forward", {})
                                best_metrics["walk_forward"]["valid_daily"] = wf_v
                        except Exception:
                            pass

                    if best_cfg is not None:
                        best_bt_cfg_by_alpha[str(rr.alpha_id)] = best_cfg
                else:
                    metrics = {"error": "Tuning failed: no valid sweep results"}
            else:
                metrics = walk_forward_evaluate_factor(
                factor_series,
                df,
                wf_config=wf_cfg,
                bt_config=bt_cfg_base,
                splits=wf_splits,
                sector_map=sector_map,
                borrow_rates=borrow_rates,
                hard_to_borrow=hard_to_borrow,
                return_valid_daily=bool(selection_tune),
            )
                if "error" not in metrics:
                    metrics["mode"] = eval_mode
                    best_bt_cfg_by_alpha[str(rr.alpha_id)] = bt_cfg_base

            if "error" in metrics:
                # Graceful fallback when the dataset is too short for walk-forward or if a sweep failed.
                metrics = evaluate_alpha(
                    factor_series,
                    fwd,
                    n_quantiles=n_quantiles,
                    cost_bps=cost_bps,
                    min_obs_per_day=min_obs_per_day,
                    universe_size_by_date=universe_size_by_date,
                )
                metrics["mode"] = "p0_fallback"
        else:
            metrics = evaluate_alpha(
                factor_series,
                fwd,
                n_quantiles=n_quantiles,
                cost_bps=cost_bps,
                min_obs_per_day=min_obs_per_day,
                universe_size_by_date=universe_size_by_date,
            )

        metrics["coverage_mean"] = float(cov_mean)

        qg = _apply_quality_gates(
            metrics,
            coverage_mean=float(cov_mean),
            min_coverage=float(min_coverage),
            max_turnover=float(max_turnover),
            min_ir=(float(min_ir) if min_ir is not None else None),
            max_drawdown=(float(max_dd) if max_dd is not None else None),
            max_total_cost_bps=(float(max_total_cost_bps) if max_total_cost_bps is not None else None),
            min_wf_splits=(int(min_wf_splits) if min_wf_splits is not None else None),
        )
        metrics["quality_gate"] = qg

        alpha["backtest_results"] = metrics
        alpha["_run_meta"] = rr.meta or {}
        enriched.append(alpha)

        score = float(metrics.get("information_ratio") or 0.0) if bool(qg.get("passed")) else float("-inf")
        scored.append({"alpha": alpha, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    scored_valid = [s for s in scored if np.isfinite(s["score"])]

    # P2.31: apply test-domain selection constraints (best-effort).
    scored_pool = list(scored_valid)
    pool_excluded: Dict[str, int] = {}
    max_cost_bps = (alpha_selection_constraints or {}).get("max_total_cost_bps") if isinstance(alpha_selection_constraints, dict) else None
    min_pos_frac = (alpha_selection_constraints or {}).get("min_wf_test_ir_positive_frac") if isinstance(alpha_selection_constraints, dict) else None

    if (max_cost_bps is not None) or (min_pos_frac is not None):
        filtered = []
        for s in scored_pool:
            a = s.get("alpha") if isinstance(s, dict) else None
            m = (a.get("backtest_results") or {}) if isinstance(a, dict) else {}

            if max_cost_bps is not None:
                c = _total_cost_bps(m)  # bps
                try:
                    if np.isfinite(c) and c > float(max_cost_bps):
                        pool_excluded["max_total_cost_bps"] = int(pool_excluded.get("max_total_cost_bps", 0) + 1)
                        continue
                except Exception:
                    pass

            if min_pos_frac is not None:
                wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
                stab = (wf.get("stability") or {}) if isinstance(wf, dict) else {}
                try:
                    p = float(stab.get("test_ir_positive_frac") or float("nan"))
                except Exception:
                    p = float("nan")
                try:
                    if np.isfinite(p) and p < float(min_pos_frac):
                        pool_excluded["min_wf_test_ir_positive_frac"] = int(
                            pool_excluded.get("min_wf_test_ir_positive_frac", 0) + 1
                        )
                        continue
                except Exception:
                    pass

            filtered.append(s)
        scored_pool = filtered

    # Default selection: take the top-K by the primary score (post gates + selection constraints).
    sota_alphas = [s["alpha"] for s in scored_pool[:top_k]]

    selection_meta: Dict[str, Any] = {
        "method": "score_rank",
        "metric": "information_ratio",
        "top_k": int(top_k),
        "domain": "test",
        "preset": alpha_selection_preset or None,
        "constraints": dict(alpha_selection_constraints or {}),
        "pool_excluded_by_constraints": pool_excluded,
    }
    selection_tuning: Optional[Dict[str, Any]] = None
    alpha_correlation: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None
    ensemble_holdings: Optional[Dict[str, Any]] = None
    ensemble_holdings_allocated: Optional[Dict[str, Any]] = None
    ensemble_holdings_allocated_regime: Optional[Dict[str, Any]] = None


    selection_done = False

    # P2.22: meta-tune selection hyperparams on validation returns (no test leakage).
    if bool(selection_tune) and int(top_k) > 1 and eval_mode in {"p1", "p2"} and enriched:
        try:
            # Use all alphas with validation return streams (avoid filtering by test performance).
            pool_alphas: List[Dict[str, Any]] = []
            for a in enriched:
                m = a.get("backtest_results") or {}
                wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
                vd = wf.get("valid_daily")
                if isinstance(vd, list) and len(vd) > 0:
                    pool_alphas.append(a)

            td = int(getattr(bt_cfg_base, "trading_days", 252)) if bt_cfg_base is not None else 252

            selection_tuning = tune_diverse_selection(
                pool_alphas,
                top_k=int(top_k),
                candidate_pool_grid=selection_tune_candidate_pool_grid or [int(diverse_candidate_pool)],
                lambda_grid=selection_tune_lambda_grid or [float(diverse_lambda)],
                top_k_grid=selection_tune_topk_grid or None,
                use_abs_corr=bool(diverse_use_abs_corr),
                min_periods=int(selection_tune_min_periods),
                metric=str(selection_tune_metric),
                trading_days=int(td),
                max_combos=int(selection_tune_max_combos),
                constraints=dict(alpha_selection_constraints or {}),
            )

            sel_ids = selection_tuning.get("selected_alpha_ids") if isinstance(selection_tuning, dict) else None
            if isinstance(sel_ids, list) and sel_ids:
                by_id = {}
                for a in pool_alphas:
                    aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
                    if aid:
                        by_id[str(aid)] = a

                selected_alphas = [by_id.get(str(i)) for i in sel_ids if str(i) in by_id]
                selected_alphas = [a for a in selected_alphas if isinstance(a, dict)]

                if selected_alphas:
                    sota_alphas = selected_alphas
                    selection_done = True

                    best_row = selection_tuning.get("best") if isinstance(selection_tuning, dict) else None
                    selection_meta = {
                        "method": "meta_tuned_diverse_greedy",
                        "metric": str(selection_tune_metric),
                        "top_k": int(len(sel_ids)),
                        "domain": "valid",
                        "preset": alpha_selection_preset or None,
                        "constraints": dict(alpha_selection_constraints or {}),
                        "selected_alpha_ids": sel_ids,
                        "best": best_row,
                    }

                    # Report correlation + ensemble on test OOS returns (evaluation-only).
                    oos_mat = build_oos_return_matrix(sota_alphas, value_key="net_return")
                    if oos_mat is not None and not oos_mat.empty and oos_mat.shape[1] >= 2:
                        corr, nobs = compute_return_correlation(oos_mat, min_periods=int(diverse_min_periods))

                        selection_meta.update(
                            {
                                "correlation_summary": correlation_summary(corr, sel_ids),
                                "min_periods": int(diverse_min_periods),
                                "correlation_domain": "test",
                            }
                        )

                        alpha_correlation = {
                            "method": "pearson",
                            "min_periods": int(diverse_min_periods),
                            "matrix": corr.to_dict(orient="split"),
                            "nobs": nobs.to_dict(orient="split"),
                            "alpha_ids": list(oos_mat.columns),
                        }

                        if bool(ensemble_enabled):
                            ensemble = make_equal_weight_ensemble(oos_mat, sel_ids, trading_days=int(td))
        except Exception as e:
            selection_tuning = {"enabled": False, "error": str(e)}

    # P2.17: if walk-forward OOS returns exist, optionally pick a more diverse set.
    if (
        bool(diverse_selection)
        and (not selection_done)
        and int(top_k) > 1
        and eval_mode in {"p1", "p2"}
        and scored_pool
    ):
        pool = scored_pool
        if int(diverse_candidate_pool) > 0:
            pool = scored_pool[: int(diverse_candidate_pool)]
        pool_alphas = [p["alpha"] for p in pool]

        oos_mat = build_oos_return_matrix(pool_alphas, value_key="net_return")
        if oos_mat is not None and not oos_mat.empty and oos_mat.shape[1] >= 2:
            corr, nobs = compute_return_correlation(oos_mat, min_periods=int(diverse_min_periods))

            # Base scores are the same scores used for the initial ranking.
            scores: Dict[str, float] = {}
            for p in pool:
                a = p.get("alpha") or {}
                aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
                if aid:
                    scores[str(aid)] = float(p.get("score") or 0.0)

            max_pairwise_corr = None
            try:
                max_pairwise_corr = float((alpha_selection_constraints or {}).get("max_pairwise_corr"))
                if (not np.isfinite(max_pairwise_corr)) or max_pairwise_corr <= 0.0:
                    max_pairwise_corr = None
            except Exception:
                max_pairwise_corr = None

            selected_ids, table, rejected = greedy_diversified_selection(
                scores=scores,
                corr=corr,
                k=int(top_k),
                diversity_lambda=float(diverse_lambda),
                use_abs_corr=bool(diverse_use_abs_corr),
                max_pairwise_corr=max_pairwise_corr,
            )

            by_id = {str((a.get("alpha_id") or a.get("alphaID") or a.get("id"))): a for a in pool_alphas}
            selected_alphas = [by_id.get(i) for i in selected_ids if i in by_id]

            if selected_alphas:
                sota_alphas = selected_alphas
                selection_done = True

            selection_meta = {
                "method": "diverse_greedy",
                "metric": "information_ratio",
                "top_k": int(top_k),
                "domain": "test",
                "preset": alpha_selection_preset or None,
                "constraints": dict(alpha_selection_constraints or {}),
                "diversity_lambda": float(diverse_lambda),
                "use_abs_corr": bool(diverse_use_abs_corr),
                "candidate_pool": int(len(pool_alphas)),
                "selected_alpha_ids": selected_ids,
                "selection_table": table,
                "rejected": rejected,
                "correlation_summary": correlation_summary(corr, selected_ids),
                "min_periods": int(diverse_min_periods),
            }

            alpha_correlation = {
                "method": "pearson",
                "min_periods": int(diverse_min_periods),
                "matrix": corr.to_dict(orient="split"),
                "nobs": nobs.to_dict(orient="split"),
                "alpha_ids": list(oos_mat.columns),
            }

            if bool(ensemble_enabled):
                td = int(getattr(bt_cfg_base, "trading_days", 252)) if bt_cfg_base is not None else 252
                ensemble = make_equal_weight_ensemble(oos_mat, selected_ids, trading_days=td)


    # P2.31: enforce max_pairwise_corr even when diversity selection is disabled.
    if (not selection_done) and eval_mode in {"p1", "p2"} and int(top_k) > 1 and scored_pool:
        try:
            max_pairwise_corr = None
            try:
                max_pairwise_corr = float((alpha_selection_constraints or {}).get("max_pairwise_corr"))
                if (not np.isfinite(max_pairwise_corr)) or max_pairwise_corr <= 0.0:
                    max_pairwise_corr = None
            except Exception:
                max_pairwise_corr = None

            if max_pairwise_corr is not None:
                pool = scored_pool
                pool_n = int(diverse_candidate_pool) if int(diverse_candidate_pool) > 0 else len(pool)
                pool = pool[: max(pool_n, int(top_k))]
                pool_alphas = [p["alpha"] for p in pool]

                oos_mat = build_oos_return_matrix(pool_alphas, value_key="net_return")
                if oos_mat is not None and not oos_mat.empty and oos_mat.shape[1] >= 2:
                    corr, nobs = compute_return_correlation(oos_mat, min_periods=int(diverse_min_periods))

                    scores = {}
                    for p in pool:
                        a = p.get("alpha") or {}
                        aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
                        if aid:
                            scores[str(aid)] = float(p.get("score") or 0.0)

                    selected_ids, table, rejected = greedy_diversified_selection(
                        scores=scores,
                        corr=corr,
                        k=int(top_k),
                        diversity_lambda=0.0,
                        use_abs_corr=bool(diverse_use_abs_corr),
                        max_pairwise_corr=max_pairwise_corr,
                    )

                    by_id = {str((a.get("alpha_id") or a.get("alphaID") or a.get("id"))): a for a in pool_alphas}
                    selected_alphas = [by_id.get(i) for i in selected_ids if i in by_id]
                    selected_alphas = [a for a in selected_alphas if isinstance(a, dict)]

                    if selected_alphas:
                        sota_alphas = selected_alphas
                        selection_meta.update(
                            {
                                "method": "score_rank_constrained",
                                "domain": "test",
                                "selected_alpha_ids": selected_ids,
                                "selection_table": table,
                                "rejected": rejected,
                                "correlation_summary": correlation_summary(corr, selected_ids),
                                "min_periods": int(diverse_min_periods),
                                "correlation_domain": "test",
                            }
                        )

                        alpha_correlation = {
                            "method": "pearson",
                            "min_periods": int(diverse_min_periods),
                            "matrix": corr.to_dict(orient="split"),
                            "nobs": nobs.to_dict(orient="split"),
                            "alpha_ids": list(oos_mat.columns),
                        }

                        if bool(ensemble_enabled):
                            td = int(getattr(bt_cfg_base, "trading_days", 252)) if bt_cfg_base is not None else 252
                            ensemble = make_equal_weight_ensemble(oos_mat, selected_ids, trading_days=td)
        except Exception:
            pass



    # P2.18: holdings-level ensemble (combine weights then re-price with costs).
    if (
        bool(holdings_ensemble_enabled)
        and int(top_k) > 1
        and eval_mode in {"p1", "p2"}
        and wf_cfg is not None
        and bt_cfg_base is not None
        and wf_splits
        and sota_alphas
    ):
        alpha_test_cache = {}
        split_returns_cache: Dict[str, Any] = {}
        try:
            ensemble_holdings = walk_forward_holdings_ensemble(
                selected_alphas=sota_alphas,
                factor_cache=factor_cache,
                ohlcv=df,
                wf_config=wf_cfg,
                bt_config=bt_cfg_base,
                splits=wf_splits,
                bt_config_by_alpha=best_bt_cfg_by_alpha,
                sector_map=sector_map,
                borrow_rates=borrow_rates,
                hard_to_borrow=hard_to_borrow,
                apply_turnover_cap=bool(holdings_ensemble_apply_turnover_cap),
                alpha_test_cache=alpha_test_cache,
            )
        except Exception as e:
            ensemble_holdings = {"enabled": False, "error": str(e)}

        if bool(alpha_allocation_enabled):
            tune_for_regime = bool(alpha_allocation_tune)
            try:
                ensemble_holdings_allocated = walk_forward_holdings_ensemble_allocated(
                    selected_alphas=sota_alphas,
                    factor_cache=factor_cache,
                    ohlcv=df,
                    wf_config=wf_cfg,
                    bt_config=bt_cfg_base,
                    splits=wf_splits,
                    bt_config_by_alpha=best_bt_cfg_by_alpha,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                    apply_turnover_cap=bool(holdings_ensemble_apply_turnover_cap),
                    alpha_test_cache=alpha_test_cache,
                    split_returns_cache=split_returns_cache,
                    allocation_fit=str(alpha_allocation_fit),
                    allocation_backend=str(alpha_allocation_backend),
                    allocation_score_metric=str(alpha_allocation_score_metric),
                    allocation_lambda=float(alpha_allocation_lambda),
                    allocation_l2=float(alpha_allocation_l2),
                    allocation_turnover_lambda=float(alpha_allocation_turnover_lambda),
                    allocation_max_weight=float(alpha_allocation_max_weight),
                    allocation_use_abs_corr=bool(alpha_allocation_use_abs_corr),
                    allocation_min_days=int(alpha_allocation_min_days),
                    allocation_solver=str(alpha_allocation_solver),
                    allocation_tune=bool(tune_for_regime),
                    allocation_tune_metric=str(alpha_allocation_tune_metric),
                    allocation_tune_max_combos=int(alpha_allocation_tune_max_combos),
                    allocation_tune_lambda_grid=list(alpha_allocation_tune_lambda_grid or []),
                    allocation_tune_max_weight_grid=list(alpha_allocation_tune_max_weight_grid or []),
                    allocation_tune_turnover_lambda_grid=list(alpha_allocation_tune_turnover_lambda_grid or []),
                    allocation_tune_save_top=int(alpha_allocation_tune_save_top),
                )
            except Exception as e:
                ensemble_holdings_allocated = {"enabled": False, "error": str(e)}

            # If meta-tuning already ran, reuse the chosen params for the regime-aware variant
            # and skip re-tuning to keep runtime predictable.
            try:
                if isinstance(ensemble_holdings_allocated, dict):
                    tune = ensemble_holdings_allocated.get("allocation_tuning") or {}
                    chosen = (tune.get("chosen") or {}) if isinstance(tune, dict) else {}
                    if isinstance(chosen, dict) and chosen:
                        alpha_allocation_lambda = float(chosen.get("lambda_corr") or alpha_allocation_lambda)
                        alpha_allocation_max_weight = float(chosen.get("max_weight") or alpha_allocation_max_weight)
                        alpha_allocation_turnover_lambda = float(
                            chosen.get("turnover_lambda") or alpha_allocation_turnover_lambda
                        )
                        tune_for_regime = False
            except Exception:
                pass

            if bool(alpha_allocation_regime_aware):
                try:
                    ensemble_holdings_allocated_regime = walk_forward_holdings_ensemble_allocated_regime(
                        selected_alphas=sota_alphas,
                        factor_cache=factor_cache,
                        ohlcv=df,
                        wf_config=wf_cfg,
                        bt_config=bt_cfg_base,
                        splits=wf_splits,
                        bt_config_by_alpha=best_bt_cfg_by_alpha,
                        sector_map=sector_map,
                        borrow_rates=borrow_rates,
                        hard_to_borrow=hard_to_borrow,
                        apply_turnover_cap=bool(holdings_ensemble_apply_turnover_cap),
                        alpha_test_cache=alpha_test_cache,
                        split_returns_cache=split_returns_cache,
                        allocation_fit=str(alpha_allocation_fit),
                        allocation_backend=str(alpha_allocation_backend),
                        allocation_score_metric=str(alpha_allocation_score_metric),
                        allocation_lambda=float(alpha_allocation_lambda),
                        allocation_l2=float(alpha_allocation_l2),
                        allocation_turnover_lambda=float(alpha_allocation_turnover_lambda),
                        allocation_max_weight=float(alpha_allocation_max_weight),
                        allocation_use_abs_corr=bool(alpha_allocation_use_abs_corr),
                        allocation_min_days=int(alpha_allocation_min_days),
                        allocation_solver=str(alpha_allocation_solver),
                        allocation_tune=bool(tune_for_regime),
                        allocation_tune_metric=str(alpha_allocation_tune_metric),
                        allocation_tune_max_combos=int(alpha_allocation_tune_max_combos),
                        allocation_tune_lambda_grid=list(alpha_allocation_tune_lambda_grid or []),
                        allocation_tune_max_weight_grid=list(alpha_allocation_tune_max_weight_grid or []),
                        allocation_tune_turnover_lambda_grid=list(alpha_allocation_tune_turnover_lambda_grid or []),
                        allocation_tune_save_top=int(alpha_allocation_tune_save_top),
                        regime_mode=str(alpha_allocation_regime_mode),
                        regime_window=int(alpha_allocation_regime_window),
                        regime_buckets=int(alpha_allocation_regime_buckets),
                        regime_min_days=int(alpha_allocation_regime_min_days),
                        regime_smoothing=float(alpha_allocation_regime_smoothing),
                        regime_tune=bool(alpha_allocation_regime_tune),
                        regime_tune_metric=str(alpha_allocation_regime_tune_metric),
                        regime_tune_max_combos=int(alpha_allocation_regime_tune_max_combos),
                        regime_tune_mode_grid=list(alpha_allocation_regime_tune_mode_grid or []),
                        regime_tune_window_grid=list(alpha_allocation_regime_tune_window_grid or []),
                        regime_tune_buckets_grid=list(alpha_allocation_regime_tune_buckets_grid or []),
                        regime_tune_smoothing_grid=list(alpha_allocation_regime_tune_smoothing_grid or []),
                        regime_tune_turnover_penalty=float(alpha_allocation_regime_tune_turnover_penalty),
                        regime_tune_save_top=int(alpha_allocation_regime_tune_save_top),
                        regime_tune_holdings_top=int(alpha_allocation_regime_tune_holdings_top),
                        regime_tune_holdings_metric=str(alpha_allocation_regime_tune_holdings_metric),
                        regime_tune_holdings_save_top=int(alpha_allocation_regime_tune_holdings_save_top),
                        regime_tune_max_alpha_turnover=alpha_allocation_regime_tune_max_alpha_turnover,
                        regime_tune_max_turnover_cost_drag_bps=alpha_allocation_regime_tune_max_turnover_cost_drag_bps,
                        regime_tune_max_regime_switch_rate=alpha_allocation_regime_tune_max_regime_switch_rate,
                        regime_tune_max_fallback_frac=alpha_allocation_regime_tune_max_fallback_frac,
                        regime_tune_prefer_pareto=bool(alpha_allocation_regime_tune_prefer_pareto),
                        regime_tune_pareto_metrics=list(alpha_allocation_regime_tune_pareto_metrics or []),
                        regime_tune_selection_method=str(alpha_allocation_regime_tune_selection_method),
                        regime_tune_utility_weights=dict(alpha_allocation_regime_tune_utility_weights or {}),
                        regime_tune_include_stability_objectives=bool(alpha_allocation_regime_tune_include_stability_objectives),
                    )
                except Exception as e:
                    ensemble_holdings_allocated_regime = {"enabled": False, "error": str(e)}

    if isinstance(ensemble_holdings, dict) and bool(ensemble_holdings.get("enabled")):
        # Best-effort comparison: holdings-level netting vs average of single-alpha costs.
        comp: Dict[str, Any] = {}
        try:
            costs_bps: List[float] = []
            for a in list(sota_alphas) or []:
                met = a.get("backtest_results") or {}
                if not isinstance(met, dict):
                    continue
                c = float(_total_cost_bps(met))
                if np.isfinite(c):
                    costs_bps.append(float(c))
            if costs_bps:
                comp["avg_selected_alpha_total_cost_bps"] = float(np.mean(costs_bps))
                comp["min_selected_alpha_total_cost_bps"] = float(np.min(costs_bps))
                comp["max_selected_alpha_total_cost_bps"] = float(np.max(costs_bps))
        except Exception:
            pass

        try:
            eh_met = ensemble_holdings.get("metrics") or {}
            if isinstance(eh_met, dict):
                eh_cost_bps = float(_total_cost_bps(eh_met))
                if np.isfinite(eh_cost_bps):
                    comp["ensemble_holdings_total_cost_bps"] = float(eh_cost_bps)
                    if "avg_selected_alpha_total_cost_bps" in comp:
                        comp["cost_savings_bps"] = float(comp["avg_selected_alpha_total_cost_bps"] - eh_cost_bps)
        except Exception:
            pass

        # Compare realized daily returns vs the return-stream ensemble (if present).
        try:
            if isinstance(ensemble, dict) and bool(ensemble.get("enabled")):
                d1 = ensemble.get("daily") or []
                d2 = ensemble_holdings.get("daily") or []
                if isinstance(d1, list) and isinstance(d2, list) and d1 and d2:
                    s1 = pd.Series(
                        [float(r.get("net_return") or 0.0) for r in d1],
                        index=pd.to_datetime([r.get("datetime") for r in d1]),
                    )
                    s2 = pd.Series(
                        [float(r.get("net_return") or 0.0) for r in d2],
                        index=pd.to_datetime([r.get("datetime") for r in d2]),
                    )
                    common = s1.index.intersection(s2.index)
                    if len(common) >= 2:
                        diff = (s2.loc[common] - s1.loc[common]).astype(float)
                        td = int(getattr(bt_cfg_base, "trading_days", 252)) if bt_cfg_base is not None else 252
                        comp["vs_return_stream"] = {
                            "n_overlap": int(len(common)),
                            "mean_daily_delta_bps": float(diff.mean() * 10000.0),
                            "annualized_delta": float((1.0 + float(diff.mean())) ** float(td) - 1.0),
                        }
        except Exception:
            pass

        ensemble_holdings["comparison"] = comp


    # P2.13: cost ablation for the top-N alphas.
    # - end_to_end: re-run walk-forward with different cost settings (strategy adapts).
    # - execution_only: keep the realized trading path fixed and only change cost deduction.
    if (
        int(ablation_top) > 0
        and eval_mode in {"p1", "p2"}
        and wf_cfg is not None
        and bt_cfg_base is not None
        and wf_splits
        and scored_valid
    ):
        want_end_to_end = ablation_mode in {"both", "end_to_end"}
        want_exec_only = ablation_mode in {"both", "execution_only"}

        for s in scored_valid[: int(ablation_top)]:
            a = s["alpha"]
            aid = str(a.get("alpha_id") or a.get("alphaID") or a.get("id") or "")
            f = factor_cache.get(aid)
            base_cfg = best_bt_cfg_by_alpha.get(aid) or bt_cfg_base
            if f is None or base_cfg is None:
                continue

            scenario_defs = [
                (
                    "no_costs",
                    {
                        "commission_bps": 0.0,
                        "slippage_bps": 0.0,
                        "half_spread_bps": 0.0,
                        "impact_bps": 0.0,
                        "borrow_cost_multiplier": 0.0,
                        "optimizer_cost_aversion": 0.0,
                    },
                ),
                ("linear_only", {"half_spread_bps": 0.0, "impact_bps": 0.0, "borrow_cost_multiplier": 0.0}),
                ("linear_spread", {"impact_bps": 0.0, "borrow_cost_multiplier": 0.0}),
                ("linear_spread_impact", {"borrow_cost_multiplier": 0.0}),
                ("full", {"borrow_cost_multiplier": float(getattr(base_cfg, "borrow_cost_multiplier", 1.0) or 1.0)}),
            ]

            ablation_payload: Dict[str, Any] = {}
            exec_only = None

            if want_end_to_end:
                scenarios: List[Dict[str, Any]] = []
                for name, kw in scenario_defs:
                    bt_cfg = replace(base_cfg, **kw)
                    met = walk_forward_evaluate_factor(
                        f,
                        df,
                        wf_config=wf_cfg,
                        bt_config=bt_cfg,
                        splits=wf_splits,
                        sector_map=sector_map,
                        borrow_rates=borrow_rates,
                        hard_to_borrow=hard_to_borrow,
                        execution_only_ablation=bool(want_exec_only and str(name) == "full"),
                    )
                    if "error" in met:
                        scenarios.append({"scenario": name, "error": met.get("error")})
                        continue

                    if want_exec_only and str(name) == "full":
                        exec_only = (met or {}).get("execution_only_ablation")

                    met["mode"] = eval_mode
                    scenarios.append(
                        {
                            "scenario": name,
                            "information_ratio": met.get("information_ratio"),
                            "annualized_return": met.get("annualized_return"),
                            "max_drawdown": met.get("max_drawdown"),
                            "turnover_mean": met.get("turnover_mean"),
                            "total_cost_bps": float(_total_cost_bps(met)),
                        }
                    )

                ablation_payload["end_to_end"] = {"mode": "end_to_end", "scenarios": scenarios}

            if want_exec_only and exec_only is None:
                met_full = walk_forward_evaluate_factor(
                    f,
                    df,
                    wf_config=wf_cfg,
                    bt_config=base_cfg,
                    splits=wf_splits,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                    execution_only_ablation=True,
                )
                exec_only = (met_full or {}).get("execution_only_ablation")

            if want_exec_only and exec_only is not None:
                ablation_payload["execution_only"] = exec_only

            if ablation_payload:
                btres = a.get("backtest_results") or {}
                tuning = btres.get("tuning") or {}
                tuning["ablation"] = ablation_payload
                btres["tuning"] = tuning
                a["backtest_results"] = btres

    # P2.13: basic regime analysis for the top alpha (optional).
    if bool(regime_analysis) and sota_alphas and eval_mode in {"p1", "p2"}:
        try:
            td = int(getattr(bt_cfg_base, "trading_days", 252) or 252) if bt_cfg_base is not None else 252
            min_obs = max(10, int(regime_window))
            vol_reg = market_volatility_regime(df, window=int(regime_window), min_obs=min_obs, buckets=int(regime_buckets))
            liq_reg = market_liquidity_regime(df, window=int(regime_window), min_obs=min_obs, buckets=int(regime_buckets))

            for a in list(sota_alphas)[:1]:
                m = a.get("backtest_results") or {}
                wf = (m.get("walk_forward") or {}) if isinstance(m, dict) else {}
                oos_daily = (wf.get("oos_daily") or []) if isinstance(wf, dict) else []
                if not isinstance(oos_daily, list) or not oos_daily:
                    continue
                idx = pd.to_datetime([r.get("datetime") for r in oos_daily])
                vals = [float(r.get("net_return") or 0.0) for r in oos_daily]
                oos_r = pd.Series(vals, index=idx).sort_index()

                reg = {
                    "market_volatility": regime_performance(oos_r, vol_reg, trading_days=td, name="market_volatility"),
                }
                if liq_reg is not None:
                    reg["market_liquidity"] = regime_performance(oos_r, liq_reg, trading_days=td, name="market_liquidity")

                analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
                analysis["regime"] = reg
                m["analysis"] = analysis
                a["backtest_results"] = m
        except Exception:
            # Keep evaluation robust: regime analysis is best-effort.
            pass

    # P2.14: cost sensitivity curves for the top-N alphas (execution-only).
    if (
        bool(cost_sensitivity)
        and int(cost_sensitivity_top) > 0
        and sota_alphas
        and eval_mode in {"p1", "p2"}
        and wf_cfg is not None
        and bt_cfg_base is not None
        and wf_splits
    ):
        # Defaults are intentionally small and interpretable (bps units).
        linear_grid = _parse_float_list(cs_linear_bps) or [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
        spread_grid = _parse_float_list(cs_half_spread_bps) or [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
        impact_grid = _parse_float_list(cs_impact_bps) or [0.0, 25.0, 50.0, 75.0, 100.0]
        borrow_bps_grid = _parse_float_list(cs_borrow_bps) or [0.0, 50.0, 100.0, 200.0, 500.0]
        borrow_mult_grid = _parse_float_list(cs_borrow_mult) or [0.0, 0.5, 1.0, 2.0, 3.0]

        for a in list(sota_alphas)[: int(cost_sensitivity_top)]:
            try:
                aid = str(a.get("alpha_id") or a.get("alphaID") or a.get("id") or "")
                f = factor_cache.get(aid)
                base_cfg = best_bt_cfg_by_alpha.get(aid) or bt_cfg_base
                if f is None or base_cfg is None:
                    continue

                met_full = walk_forward_evaluate_factor(
                    f,
                    df,
                    wf_config=wf_cfg,
                    bt_config=base_cfg,
                    splits=wf_splits,
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                    return_oos_daily_decomp=True,
                )

                wf = (met_full.get("walk_forward") or {}) if isinstance(met_full, dict) else {}
                daily = (wf.get("oos_daily_decomp") or []) if isinstance(wf, dict) else []
                if not isinstance(daily, list) or not daily:
                    continue

                cs = compute_cost_sensitivity(
                    daily,
                    base_cfg=base_cfg,
                    borrow_rates_present=bool(borrow_rates is not None),
                    linear_bps_grid=linear_grid,
                    half_spread_bps_grid=spread_grid,
                    impact_bps_grid=impact_grid,
                    borrow_bps_grid=borrow_bps_grid,
                    borrow_mult_grid=borrow_mult_grid,
                )

                m = a.get("backtest_results") or {}
                analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
                analysis["cost_sensitivity"] = cs
                m["analysis"] = analysis
                a["backtest_results"] = m
            except Exception:
                # Best-effort: this should never break the core evaluation loop.
                continue

    # P2.15: horizon decay analysis for the top-N alphas.
    if bool(decay_analysis) and int(decay_analysis_top) > 0 and sota_alphas:
        horizons_list = _parse_int_list(decay_horizons_s) or [1, 2, 5, 10, 20]
        td = int(getattr(bt_cfg_base, "trading_days", 252) or 252) if bt_cfg_base is not None else 252
        for a in list(sota_alphas)[: int(decay_analysis_top)]:
            try:
                aid = str(a.get("alpha_id") or a.get("alphaID") or a.get("id") or "")
                f = factor_cache.get(aid)
                if f is None:
                    continue
                decay = compute_horizon_decay(
                    f,
                    df["close"],
                    horizons_list,
                    n_quantiles=int(n_quantiles),
                    min_obs_per_day=int(min_obs_per_day),
                    trading_days=int(td),
                    universe_size_by_date=universe_size_by_date,
                )
                m = a.get("backtest_results") or {}
                analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
                analysis["decay"] = decay
                if isinstance(m, dict):
                    m["analysis"] = analysis
                    a["backtest_results"] = m
            except Exception:
                continue

    # P2.16: holding / rebalance schedule sweep for the top-N alphas.
    # This is a strategy-level diagnostic: we keep the alpha fixed and only vary
    # (rebalance_days, holding_days) to see the turnover/cost/performance trade-off.
    if (
        bool(schedule_sweep)
        and int(schedule_sweep_top) > 0
        and sota_alphas
        and eval_mode in {"p1", "p2"}
        and wf_cfg is not None
        and bt_cfg_base is not None
        and wf_splits
    ):
        metric_key = str(schedule_sweep_metric or "information_ratio").strip()
        if metric_key not in {"information_ratio", "annualized_return"}:
            metric_key = "information_ratio"

        reb_grid = _parse_int_list(schedule_sweep_rebalance_s) or [1, 2, 5, 10]
        hold_grid = _parse_int_list(schedule_sweep_holding_s) or [1, 2, 5, 10, 20]

        for a in list(sota_alphas)[: int(schedule_sweep_top)]:
            try:
                aid = str(a.get("alpha_id") or a.get("alphaID") or a.get("id") or "")
                f = factor_cache.get(aid)
                base_cfg = best_bt_cfg_by_alpha.get(aid) or bt_cfg_base
                if f is None or base_cfg is None:
                    continue

                # Ensure the current base schedule is always in the grid.
                reb2 = list(dict.fromkeys(list(reb_grid) + [int(getattr(base_cfg, "rebalance_days", 5) or 5)]))
                hold2 = list(dict.fromkeys(list(hold_grid) + [int(getattr(base_cfg, "holding_days", 5) or 5)]))

                sweep = compute_holding_rebalance_sweep(
                    f,
                    df,
                    wf_config=wf_cfg,
                    base_bt_config=base_cfg,
                    splits=wf_splits,
                    rebalance_days_list=reb2,
                    holding_days_list=hold2,
                    max_combos=int(schedule_sweep_max_combos),
                    sector_map=sector_map,
                    borrow_rates=borrow_rates,
                    hard_to_borrow=hard_to_borrow,
                )

                rows = list(sweep.get("results") or []) if isinstance(sweep, dict) else []
                annotated: List[Dict[str, Any]] = []

                cov_mean = float((a.get("backtest_results") or {}).get("coverage_mean") or float("nan"))

                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    rr = dict(r)

                    if rr.get("error"):
                        rr["passed"] = False
                        rr["reasons"] = ["error"]
                        annotated.append(rr)
                        continue

                    # Build a minimal metrics dict so we can re-use the same gates.
                    met_small: Dict[str, Any] = {
                        "information_ratio": rr.get("information_ratio"),
                        "annualized_return": rr.get("annualized_return"),
                        "max_drawdown": rr.get("max_drawdown"),
                        "turnover_mean": rr.get("turnover_mean"),
                        "oos_cost_attribution": {
                            "cost_mean": rr.get("cost_mean"),
                            "borrow_mean": rr.get("borrow_mean"),
                        },
                        "walk_forward": {"stability": {"n_splits": rr.get("wf_n_splits")}},
                    }

                    qg_row = _apply_quality_gates(
                        met_small,
                        coverage_mean=float(cov_mean),
                        min_coverage=float(min_coverage),
                        max_turnover=float(max_turnover),
                        min_ir=(float(min_ir) if min_ir is not None else None),
                        max_drawdown=(float(max_dd) if max_dd is not None else None),
                        max_total_cost_bps=(float(max_total_cost_bps) if max_total_cost_bps is not None else None),
                        min_wf_splits=(int(min_wf_splits) if min_wf_splits is not None else None),
                    )
                    rr["passed"] = bool(qg_row.get("passed"))
                    rr["reasons"] = list(qg_row.get("reasons") or [])
                    annotated.append(rr)

                def _metric(r0: Dict[str, Any]) -> float:
                    try:
                        return float(r0.get(metric_key) or float("-inf"))
                    except Exception:
                        return float("-inf")

                non_error = [r for r in annotated if isinstance(r, dict) and not r.get("error")]
                passed_rows = [r for r in non_error if bool(r.get("passed"))]

                best = max(passed_rows, key=_metric) if passed_rows else None
                best_raw = max(non_error, key=_metric) if non_error else None

                base_pair = (int(getattr(base_cfg, "rebalance_days", 5) or 5), int(getattr(base_cfg, "holding_days", 5) or 5))
                base_row = next(
                    (r for r in non_error if int(r.get("rebalance_days") or 0) == base_pair[0] and int(r.get("holding_days") or 0) == base_pair[1]),
                    None,
                )

                def _row_meta(r0: Dict[str, Any] | None) -> Dict[str, Any]:
                    if not isinstance(r0, dict):
                        return {}
                    return {
                        "rebalance_days": r0.get("rebalance_days"),
                        "holding_days": r0.get("holding_days"),
                        "information_ratio": r0.get("information_ratio"),
                        "annualized_return": r0.get("annualized_return"),
                        "max_drawdown": r0.get("max_drawdown"),
                        "turnover_mean": r0.get("turnover_mean"),
                        "total_cost_bps": r0.get("total_cost_bps"),
                        "passed": bool(r0.get("passed")) if r0 is not None else None,
                        "reasons": list(r0.get("reasons") or []) if r0 is not None else [],
                    }

                sweep_out = dict(sweep) if isinstance(sweep, dict) else {"enabled": True}
                sweep_out["metric"] = metric_key
                sweep_out["results"] = annotated
                sweep_out["best"] = _row_meta(best)
                sweep_out["best_raw"] = _row_meta(best_raw)
                if base_row is not None:
                    sweep_out["base_row"] = _row_meta(base_row)

                if best is not None and base_row is not None:
                    try:
                        b0 = float(base_row.get(metric_key) or 0.0)
                        b1 = float(best.get(metric_key) or 0.0)
                        sweep_out["delta"] = {
                            "metric": metric_key,
                            "base": float(b0),
                            "best": float(b1),
                            "improvement": float(b1 - b0),
                        }
                    except Exception:
                        pass

                # Store under backtest_results.analysis.
                m = a.get("backtest_results") or {}
                analysis = (m.get("analysis") or {}) if isinstance(m, dict) else {}
                analysis["schedule_sweep"] = sweep_out
                if isinstance(m, dict):
                    m["analysis"] = analysis
                    a["backtest_results"] = m
            except Exception:
                continue

    return {
        "coded_alphas": enriched,
        "sota_alphas": sota_alphas,
        "selection": selection_meta,
        "selection_tuning": selection_tuning,
        "alpha_correlation": alpha_correlation,
        "ensemble": ensemble,
        "ensemble_holdings": ensemble_holdings,
        "ensemble_holdings_allocated": ensemble_holdings_allocated,
        "ensemble_holdings_allocated_regime": ensemble_holdings_allocated_regime,
    }
