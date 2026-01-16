# src/agent/graph.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agent.state import State
from agent.agents.user_input_agent import user_input_agent
from agent.agents.hypothesis_agent import hypothesis_agent
from agent.agents.alpha_generator_agent import alpha_generator_agent
from agent.agents.alpha_coder_agent import alpha_coder_agent
from agent.agents.evaluate_alphas_agent import evaluate_alphas_agent
from agent.agents.persist_state_agent import persist_state_agent
from agent.agents.prepare_next_iteration_agent import prepare_next_iteration_agent
from agent.database.checkpointer_api import get_checkpoint_manager


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _best_information_ratio(sota_alphas: List[Dict[str, Any]]) -> float:
    best = 0.0
    for a in sota_alphas or []:
        m = (a or {}).get("backtest_results") or {}
        try:
            best = max(best, float(m.get("information_ratio") or 0.0))
        except Exception:
            continue
    return best


def create_graph() -> Any:
    workflow = StateGraph(State)

    # Nodes
    workflow.add_node("user_input", user_input_agent)
    workflow.add_node("hypothesis_generator", hypothesis_agent)
    workflow.add_node("alpha_generator", alpha_generator_agent)
    workflow.add_node("alpha_coder", alpha_coder_agent)
    workflow.add_node("evaluate_alphas", evaluate_alphas_agent)
    workflow.add_node("persist_state", persist_state_agent)
    workflow.add_node("prepare_next_iteration", prepare_next_iteration_agent)

    # Edges (P0 flow)
    workflow.set_entry_point("user_input")
    workflow.add_edge("user_input", "hypothesis_generator")
    workflow.add_edge("hypothesis_generator", "alpha_generator")
    workflow.add_edge("alpha_generator", "alpha_coder")
    workflow.add_edge("alpha_coder", "evaluate_alphas")
    workflow.add_edge("evaluate_alphas", "persist_state")

    # Optional research loop:
    # - stop if iteration >= max_iterations
    # - stop early if best IR reaches target_information_ratio
    def _route_after_persist(s: State) -> str:
        it = int(getattr(s, "iteration", 0) or 0)
        max_it = int(getattr(s, "max_iterations", 1) or 1)
        target_ir = float(getattr(s, "target_information_ratio", 0.0) or 0.0)
        best_ir = _best_information_ratio(getattr(s, "sota_alphas", []) or [])

        if it >= max_it:
            return "end"
        if target_ir > 0.0 and best_ir >= target_ir:
            return "end"
        return "continue"

    workflow.add_conditional_edges(
        "persist_state",
        _route_after_persist,
        {
            "continue": "prepare_next_iteration",
            "end": END,
        },
    )
    workflow.add_edge("prepare_next_iteration", "hypothesis_generator")

    # Checkpointer selection
    checkpointer = None
    if _truthy_env("USE_POSTGRES"):
        manager = get_checkpoint_manager()
        checkpointer = manager.get_langgraph_saver()

    if checkpointer is None:
        checkpointer = MemorySaver()

    g = workflow.compile(checkpointer=checkpointer)
    g.name = "Alpha Generation and Coding Workflow"
    return g


graph = create_graph()
