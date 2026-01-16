# src/agent/agents/user_input_agent.py
from __future__ import annotations

from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from agent.state import State


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


async def user_input_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Capture user trading idea input.

    In a real application, this would get input from the user.
    For now, we accept an optional ``trading_idea`` in the incoming state.
    """
    trading_idea = _get_attr_or_key(state, "trading_idea", "") or "Momentum-based strategy using volume and closing price"
    return {"trading_idea": trading_idea}
