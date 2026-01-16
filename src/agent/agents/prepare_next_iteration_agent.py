# src/agent/agents/prepare_next_iteration_agent.py
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from agent.state import State


def _best_alpha(sota_alphas: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sota_alphas:
        return {}
    return dict(sota_alphas[0])


async def prepare_next_iteration_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Prepare state for the next research iteration (optional loop)."""
    iteration = int(getattr(state, "iteration", 0) or 0)
    next_it = iteration + 1

    sota_alphas: List[Dict[str, Any]] = getattr(state, "sota_alphas", []) or []
    best = _best_alpha(sota_alphas)
    metrics = best.get("backtest_results") if isinstance(best, dict) else None

    feedback = {
        "iteration": iteration,
        "best_alpha_id": best.get("alpha_id") or best.get("alphaID") or best.get("id"),
        "metrics": metrics or {},
    }

    alpha_history: List[Dict[str, Any]] = list(getattr(state, "alpha_history", []) or [])
    alpha_history.append({"iteration": iteration, "best": feedback})
    alpha_history = alpha_history[-20:]

    # Keep sota_alphas, but clear large intermediate lists to keep state small.
    return {
        "iteration": next_it,
        "feedback": feedback,
        "alpha_history": alpha_history,
        "seed_alphas": [],
        "coded_alphas": [],
    }
