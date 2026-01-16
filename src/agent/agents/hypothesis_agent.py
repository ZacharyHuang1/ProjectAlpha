# src/agent/agents/hypothesis_agent.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import State
from agent.prompts.hypothesis_prompts import (
    HYPOTHESIS_SYSTEM_PROMPT,
    HYPOTHESIS_INITIAL_PROMPT,
    HYPOTHESIS_ITERATION_PROMPT,
    HYPOTHESIS_OUTPUT_FORMAT,
)
from agent.database.checkpointer_api import get_checkpoint_manager


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_json_loads(content: str) -> Dict[str, Any]:
    """Parse an LLM response that may contain extra text around a JSON object."""
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(content[json_start:json_end])
    return json.loads(content)


def _get_thread_id(config: RunnableConfig) -> str:
    return (config.get("configurable") or {}).get("thread_id", "default")


def _next_iteration(hypothesis_history: list[dict[str, Any]]) -> int:
    # hypothesis_history is returned in chronological order (oldest -> newest)
    if not hypothesis_history:
        return 1
    last = hypothesis_history[-1]
    try:
        return int(last.get("iteration", 0)) + 1
    except Exception:
        return 0


async def hypothesis_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate or refine a trading hypothesis."""

    thread_id = _get_thread_id(config)
    trading_idea = _get_attr_or_key(state, "trading_idea", "") or ""

    # Best-effort history lookup (should not crash if DB is unavailable)
    #
    # Why we do this:
    # - This project is meant to be iterative: later runs should be able to see earlier hypotheses/alphas.
    # - `thread_id` scopes the history (same thread_id = same research thread).
    hypothesis_history: list[dict[str, Any]] = []
    alpha_data: Optional[dict[str, Any]] = None
    backtest_data: Optional[list[dict[str, Any]]] = None

    try:
        checkpointer = get_checkpoint_manager()
        hypothesis_history = checkpointer.get_hypothesis_history(thread_id) or []
        latest_hypothesis = hypothesis_history[-1] if hypothesis_history else None

        if latest_hypothesis and latest_hypothesis.get("id"):
            alphas = checkpointer.get_alphas_for_hypothesis(latest_hypothesis["id"])
            if alphas:
                alpha_data = alphas[0]
                if alpha_data and alpha_data.get("id"):
                    backtest_data = checkpointer.get_backtest_results_for_alpha(
                        alpha_data["id"]
                    )
    except Exception:
        hypothesis_history = []
        alpha_data = None
        backtest_data = None

    if not hypothesis_history:
        hypothesis_history = list(getattr(state, "hypothesis_history", []) or [])

    if alpha_data is None and getattr(state, "alpha_history", None):
        alpha_data = (getattr(state, "alpha_history") or [])[-1]

    iteration = max(int(getattr(state, "iteration", 0) or 0), _next_iteration(hypothesis_history))
    is_first_iteration = (iteration <= 1) and (not hypothesis_history) and (not getattr(state, "hypothesis_history", []))

    # Dev/test stub (no API key)
    if not os.getenv("OPENAI_API_KEY"):
        trading_idea_stub = trading_idea or "Momentum-based strategy using volume and closing price"
        payload = {
            "trading_idea": trading_idea_stub,
            "iteration": iteration,
            "hypothesis": (
                "Short-horizon price momentum is more likely to persist when accompanied by rising "
                "trading volume (a proxy for conviction). Therefore, assets with positive recent returns "
                "and increasing volume should have higher next-period returns than those with weak/declining volume."
            ),
            "reason": (
                "Momentum effects are documented in empirical finance. Conditioning on volume can help distinguish "
                "informational price moves from noise; higher volume indicates stronger participation and potentially "
                "stronger continuation in the short term."
            ),
            "concise_reason": "Momentum + rising volume implies stronger continuation.",
            "concise_observation": "Volume can proxy for conviction/participation.",
            "concise_justification": "Higher participation may make trends more persistent.",
            "concise_knowledge": "If recent returns are positive and volume is rising, then near-term returns are more likely to remain positive.",
        }

        hist = list(getattr(state, "hypothesis_history", []) or [])
        hist.append(
            {
                "iteration": iteration,
                "trading_idea": trading_idea_stub,
                "hypothesis": payload["hypothesis"],
                "reason": payload["reason"],
            }
        )
        payload["hypothesis_history"] = hist[-20:]
        return payload

    # Build prompt
    if is_first_iteration:
        user_prompt = HYPOTHESIS_INITIAL_PROMPT.format(
            trading_idea=trading_idea,
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )
    else:
        history_payload: Dict[str, Any] = {
            "hypotheses": hypothesis_history[-3:],  # last few for brevity
            "alpha": alpha_data,
            "backtests": backtest_data,
            "feedback": getattr(state, "feedback", None),
            "sota_alphas": (getattr(state, "sota_alphas", []) or [])[:3],
        }
        user_prompt = HYPOTHESIS_ITERATION_PROMPT.format(
            hypothesis_history=json.dumps(history_payload, indent=2),
            output_format=HYPOTHESIS_OUTPUT_FORMAT,
        )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": HYPOTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    hypothesis_data = _safe_json_loads(response.content)
    hypothesis_data["trading_idea"] = trading_idea
    hypothesis_data["iteration"] = iteration

    hist = list(getattr(state, "hypothesis_history", []) or [])
    hist.append(
        {
            "iteration": iteration,
            "trading_idea": trading_idea,
            "hypothesis": hypothesis_data.get("hypothesis", ""),
            "reason": hypothesis_data.get("reason", ""),
        }
    )
    hypothesis_data["hypothesis_history"] = hist[-20:]
    return hypothesis_data
