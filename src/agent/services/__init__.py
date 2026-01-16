"""agent.services package.

We keep imports lazy so that individual helpers can be imported without pulling
in optional heavy dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["invoke_graph_with_state", "get_state_history"]


def __getattr__(name: str) -> Any:
    if name in {"invoke_graph_with_state", "get_state_history"}:
        from agent.services.state_service import invoke_graph_with_state, get_state_history
        return {"invoke_graph_with_state": invoke_graph_with_state, "get_state_history": get_state_history}[name]
    raise AttributeError(name)
