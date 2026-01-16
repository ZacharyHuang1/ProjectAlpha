"""agent package.

This package contains the LangGraph workflow and supporting research utilities.

Note:
- We avoid importing the LangGraph graph at import time so that helper modules
  (data loader, DSL, evaluation) can be imported in isolation.
"""

from __future__ import annotations

from typing import Any

__all__ = ["graph"]


def __getattr__(name: str) -> Any:
    if name == "graph":
        from agent.graph import graph  # local import (may require langgraph dependency)
        return graph
    raise AttributeError(name)
