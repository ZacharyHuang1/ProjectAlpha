"""agent.research.schedule_sweep

P2.16: Holding / rebalance schedule sweep.

This diagnostic evaluates a *fixed* alpha under different scheduling choices:
- rebalance frequency (days)
- holding period (days)

It helps answer a practical question: is the signal truly fast (needs frequent
rebalancing), or can you hold longer to reduce turnover and costs.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig
from agent.research.walk_forward import WalkForwardConfig, walk_forward_evaluate_factor


def _unique_positive_ints(xs: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for x in xs:
        try:
            v = int(x)
        except Exception:
            continue
        if v <= 0 or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _pick_pairs(
    pairs: List[Tuple[int, int]],
    *,
    base: Tuple[int, int],
    max_combos: int,
) -> List[Tuple[int, int]]:
    """Deterministically limit the number of evaluated pairs."""

    pairs = sorted(set(pairs))
    if base not in pairs:
        pairs = [base] + pairs

    max_combos = int(max(1, max_combos))
    if len(pairs) <= max_combos:
        return pairs

    picked: List[Tuple[int, int]] = [base]
    for p in pairs:
        if p == base:
            continue
        picked.append(p)
        if len(picked) >= max_combos:
            break
    return picked


def compute_holding_rebalance_sweep(
    factor: pd.Series,
    ohlcv_or_close: pd.DataFrame | pd.Series,
    *,
    wf_config: WalkForwardConfig,
    base_bt_config: BacktestConfig,
    splits: List[Dict[str, Any]],
    rebalance_days_list: Sequence[int],
    holding_days_list: Sequence[int],
    max_combos: int = 25,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Run a small grid sweep over (rebalance_days, holding_days) schedules.

    Notes:
    - This is a *strategy-level* sweep (not a factor transformation).
    - It re-runs walk-forward evaluation for each schedule.
    - The caller is expected to apply any quality gates.
    """

    base_pair = (int(base_bt_config.rebalance_days), int(base_bt_config.holding_days))

    rebs = _unique_positive_ints(list(rebalance_days_list) or [])
    holds = _unique_positive_ints(list(holding_days_list) or [])
    if not rebs:
        rebs = [1, 2, 5, 10]
    if not holds:
        holds = [1, 2, 5, 10, 20]

    pairs_all = list(product(rebs, holds))
    pairs = _pick_pairs(pairs_all, base=base_pair, max_combos=int(max_combos))

    results: List[Dict[str, Any]] = []
    n_errors = 0

    for reb, hold in pairs:
        bt_cfg = replace(base_bt_config, rebalance_days=int(reb), holding_days=int(hold))
        met = walk_forward_evaluate_factor(
            factor,
            ohlcv_or_close,
            wf_config=wf_config,
            bt_config=bt_cfg,
            splits=splits,
            sector_map=sector_map,
            borrow_rates=borrow_rates,
            hard_to_borrow=hard_to_borrow,
        )

        row: Dict[str, Any] = {
            "rebalance_days": int(reb),
            "holding_days": int(hold),
        }

        if not isinstance(met, dict) or met.get("error"):
            row["error"] = (met.get("error") if isinstance(met, dict) else None) or "Unknown error"
            results.append(row)
            n_errors += 1
            continue

        attr = met.get("oos_cost_attribution") or {}
        cost_mean = float(attr.get("cost_mean") or 0.0) if isinstance(attr, dict) else 0.0
        borrow_mean = float(attr.get("borrow_mean") or 0.0) if isinstance(attr, dict) else 0.0

        wf = met.get("walk_forward") or {}
        stab = (wf.get("stability") or {}) if isinstance(wf, dict) else {}
        oos = (wf.get("oos") or {}) if isinstance(wf, dict) else {}

        max_active = int(np.ceil(float(hold) / float(reb))) if int(hold) > int(reb) else 1
        overlap_ratio = float(hold) / float(reb) if float(reb) > 0 else float("nan")

        row.update(
            {
                "max_active": int(max_active),
                "overlap_ratio": float(overlap_ratio),
                "information_ratio": float(met.get("information_ratio") or 0.0),
                "annualized_return": float(met.get("annualized_return") or 0.0),
                "max_drawdown": float(met.get("max_drawdown") or 0.0),
                "turnover_mean": float(met.get("turnover_mean") or 0.0),
                "wf_n_splits": int(stab.get("n_splits") or 0),
                "n_obs": int(oos.get("n_obs") or 0),
                "cost_mean": float(cost_mean),
                "borrow_mean": float(borrow_mean),
                "total_cost_bps": float((cost_mean + borrow_mean) * 10000.0),
            }
        )
        results.append(row)

    return {
        "enabled": True,
        "base": {"rebalance_days": base_pair[0], "holding_days": base_pair[1]},
        "grid": {"rebalance_days": rebs, "holding_days": holds},
        "max_combos": int(max_combos),
        "n_requested": int(len(pairs_all) + (0 if base_pair in pairs_all else 1)),
        "n_evaluated": int(len(results)),
        "n_errors": int(n_errors),
        "results": results,
    }
