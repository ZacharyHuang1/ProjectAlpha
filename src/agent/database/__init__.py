"""agent.database package.

This package contains database models and persistence helpers.

We keep imports lazy so that the research utilities (DSL/eval) can be imported
without requiring a database driver at import time.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AlphaGPTCheckpointer", "get_checkpoint_manager"]


def __getattr__(name: str) -> Any:
    if name in {"AlphaGPTCheckpointer", "get_checkpoint_manager"}:
        from agent.database.checkpointer_api import AlphaGPTCheckpointer, get_checkpoint_manager
        return {"AlphaGPTCheckpointer": AlphaGPTCheckpointer, "get_checkpoint_manager": get_checkpoint_manager}[name]
    raise AttributeError(name)
