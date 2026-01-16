# src/agent/agents/persist_state_agent.py
from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from agent.state import State
from agent.database.checkpointer_api import get_checkpoint_manager


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


async def persist_state_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Persist the current state to the domain database (Hypothesis/Alpha/BacktestResult).

    This is intentionally a no-op unless USE_POSTGRES=true.
    """
    if not _truthy_env("USE_POSTGRES"):
        return {}

    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    checkpoint_id = (config.get("configurable") or {}).get("checkpoint_id")

    manager = get_checkpoint_manager()
    result = manager.save_state(thread_id=thread_id, checkpoint_id=checkpoint_id, state=state)

    # Avoid polluting state with too much persistence metadata; keep minimal
    return {"_persistence": result}
