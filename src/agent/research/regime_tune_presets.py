"""agent.research.regime_tune_presets

P2.27: Simple preset bundles for regime-aware tuning.

The goal is to make tuning usable without requiring manual reasoning about
constraint values every run. Presets are *defaults*; CLI flags can override
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RegimeTunePreset:
    name: str
    description: str
    constraints: Dict[str, Any]
    prefer_pareto: bool
    pareto_metrics: List[str]
    turnover_cost_bps: float
    selection_method: str = "best_objective"
    utility_weights: Dict[str, float] = field(default_factory=dict)
    include_stability_objectives: bool = True


def list_presets() -> List[str]:
    return ["aggressive", "low_turnover", "execution_realistic"]


def get_preset(name: Optional[str]) -> Optional[RegimeTunePreset]:
    """Return a preset by name (case-insensitive)."""

    if not name:
        return None
    key = str(name).strip().lower()
    if not key:
        return None

    if key in {"aggressive", "fast"}:
        return RegimeTunePreset(
            name="aggressive",
            description="Maximize objective with minimal constraints.",
            constraints={},
            prefer_pareto=False,
            pareto_metrics=[],
            turnover_cost_bps=0.0,
        )

    if key in {"low_turnover", "low-turnover", "stable"}:
        return RegimeTunePreset(
            name="low_turnover",
            description="Prefer stable weights and low switching.",
            constraints={
                "max_alpha_weight_turnover_mean": 0.25,
                "max_regime_switch_rate_mean": 0.15,
                "max_turnover_cost_drag_bps_mean": 7.5,
            },
            prefer_pareto=True,
            pareto_metrics=["turnover_cost_drag_bps_mean", "regime_switch_rate_mean"],
            turnover_cost_bps=0.2,
            selection_method="knee",
            include_stability_objectives=True,
        )

    if key in {"execution_realistic", "execution", "realistic"}:
        return RegimeTunePreset(
            name="execution_realistic",
            description="Balance objective vs execution realism (turnover, switching, fallback).",
            constraints={
                "max_alpha_weight_turnover_mean": 0.35,
                "max_turnover_cost_drag_bps_mean": 10.0,
                "max_regime_switch_rate_mean": 0.20,
                "max_fallback_frac_mean": 0.25,
            },
            prefer_pareto=True,
            pareto_metrics=[
                "turnover_cost_drag_bps_mean",
                "regime_switch_rate_mean",
                "fallback_frac_mean",
                # Holdings-only metrics; ignored in proxy stage if missing.
                "ensemble_cost_mean",
                "ensemble_borrow_mean",
            ],
            turnover_cost_bps=0.2,
            selection_method="utility",
            # Weights apply in *normalized goodness space* (higher is better).
            utility_weights={
                "objective": 1.0,
                "holdings_objective": 1.0,
                "alpha_weight_turnover_mean": 0.4,
                "turnover_cost_drag_bps_mean": 0.4,
                "regime_switch_rate_mean": 0.2,
                "fallback_frac_mean": 0.2,
                # Stability.
                "objective_split_std": 0.2,
                "objective_split_min": 0.2,
                "holdings_objective_split_std": 0.2,
                "holdings_objective_split_min": 0.2,
                # Holdings-only execution metrics.
                "ensemble_cost_mean": 0.2,
                "ensemble_borrow_mean": 0.2,
            },
            include_stability_objectives=True,
        )

    return None
