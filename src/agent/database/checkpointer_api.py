"""AlphaGPT Checkpointer API.

This module provides a thin wrapper around LangGraph checkpointing plus our own
SQLAlchemy tables (Hypothesis / Alpha / BacktestResult).

Key design goals:
- Work even when Postgres is not available (degrade gracefully).
- Avoid crashing agents during local development when DB isn't running.
- Provide a single place to wire up persistence for both LangGraph checkpoints and
  domain-specific tables.
"""

from __future__ import annotations

import atexit
import os
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional

from agent.database.operations import (
    create_tables,
    get_db_engine,
    get_hypothesis_history,
    get_alphas_for_hypothesis,
    get_backtest_results_for_alpha,
    save_hypothesis,
    save_alphas,
    save_backtest_results,
)
from agent.database.operations.db_connection import get_db_url


# Optional LangGraph imports (so basic module import doesn't hard-fail in minimal envs)
try:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
except Exception:  # pragma: no cover
    BaseCheckpointSaver = object  # type: ignore
    PostgresSaver = None  # type: ignore
    AsyncPostgresSaver = None  # type: ignore


def _truthy_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """Convert a State dataclass (or dict-like) to a plain dict."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    if is_dataclass(state):
        return asdict(state)
    # Fallback: try attribute dict
    if hasattr(state, "__dict__"):
        return dict(state.__dict__)
    return {}


class AlphaGPTCheckpointer:
    """Manages optional LangGraph persistence + custom domain persistence."""

    def __init__(self) -> None:
        self._db_ready: bool = False

        # LangGraph PostgresSaver lifecycle (kept open for the life of the process)
        self._sync_cm = None
        self._sync_saver: Optional[BaseCheckpointSaver] = None
        self._async_cm = None
        self._async_saver = None

    # ---------------------------
    # Database (SQLAlchemy) setup
    # ---------------------------
    def ensure_db_ready(self) -> bool:
        """Create custom tables if possible. Returns whether DB is ready."""
        if self._db_ready:
            return True
        try:
            engine = get_db_engine()
            create_tables(engine)
            self._db_ready = True
        except Exception:
            # Postgres not running / bad env vars / etc.
            self._db_ready = False
        return self._db_ready

    # ---------------------------
    # LangGraph checkpointer setup
    # ---------------------------
    def _checkpoint_db_uri(self) -> str:
        """Connection string used by LangGraph PostgresSaver."""
        # Allow explicit override
        uri = (
            os.getenv("LANGGRAPH_POSTGRES_URI")
            or os.getenv("POSTGRES_URI")
            or os.getenv("DATABASE_URL")
            or get_db_url()
        )
        return uri

    def get_langgraph_saver(self) -> Optional[BaseCheckpointSaver]:
        """Return a LangGraph BaseCheckpointSaver (sync) if enabled and available."""
        if not _truthy_env("USE_POSTGRES"):
            return None
        if PostgresSaver is None:
            return None
        if self._sync_saver is not None:
            return self._sync_saver

        # Ensure our domain tables exist (not required for LangGraph tables, but convenient)
        self.ensure_db_ready()

        try:
            uri = self._checkpoint_db_uri()
            # from_conn_string returns a context manager (iterator); we keep it open
            self._sync_cm = PostgresSaver.from_conn_string(uri)
            self._sync_saver = self._sync_cm.__enter__()

            # One-time setup of LangGraph checkpoint tables (safe to call multiple times)
            try:
                self._sync_saver.setup()  # type: ignore[attr-defined]
            except Exception:
                pass

            atexit.register(self.close)
            return self._sync_saver
        except Exception:
            self._sync_cm = None
            self._sync_saver = None
            return None

    async def aget_langgraph_saver(self) -> Optional[Any]:
        """Return a LangGraph BaseCheckpointSaver (async) if enabled and available."""
        if not _truthy_env("USE_POSTGRES"):
            return None
        if AsyncPostgresSaver is None:
            return None
        if self._async_saver is not None:
            return self._async_saver

        self.ensure_db_ready()

        try:
            uri = self._checkpoint_db_uri()
            self._async_cm = AsyncPostgresSaver.from_conn_string(uri)
            self._async_saver = await self._async_cm.__aenter__()

            try:
                await self._async_saver.setup()
            except Exception:
                pass

            atexit.register(self.close)
            return self._async_saver
        except Exception:
            self._async_cm = None
            self._async_saver = None
            return None

    def close(self) -> None:
        """Close any open LangGraph checkpointer connections."""
        if self._sync_cm is not None:
            try:
                self._sync_cm.__exit__(None, None, None)
            except Exception:
                pass
            self._sync_cm = None
            self._sync_saver = None

        if self._async_cm is not None:
            try:
                # best-effort async close
                import asyncio

                async def _close():
                    try:
                        await self._async_cm.__aexit__(None, None, None)
                    except Exception:
                        pass

                asyncio.get_event_loop().create_task(_close())
            except Exception:
                pass
            self._async_cm = None
            self._async_saver = None

    # ---------------------------
    # Domain persistence helpers
    # ---------------------------
    def save_state(
        self,
        *,
        thread_id: str,
        checkpoint_id: Optional[str],
        state: Any,
    ) -> Dict[str, Any]:
        """Persist State into custom domain tables.

        Returns a small summary dict (safe to add to graph state if needed).
        """
        if not self.ensure_db_ready():
            return {"persisted": False, "reason": "db_not_ready"}

        state_values = _state_to_dict(state)
        cp_id = checkpoint_id or str(uuid.uuid4())

        try:
            hyp = save_hypothesis(thread_id=thread_id, checkpoint_id=cp_id, state_values=state_values)
            hyp_id = hyp.id if hyp is not None else None

            if hyp_id is not None:
                save_alphas(
                    thread_id=thread_id,
                    checkpoint_id=cp_id,
                    state_values=state_values,
                    hypothesis_id=hyp_id,
                )

            # If sota_alphas contain backtest results, persist them too (optional)
            save_backtest_results(
                thread_id=thread_id,
                checkpoint_id=cp_id,
                state_values=state_values,
            )

            return {"persisted": True, "checkpoint_id": cp_id, "hypothesis_id": hyp_id}
        except Exception as e:
            return {"persisted": False, "reason": f"exception: {e}"}

    # ---------------------------
    # Read APIs (used by agents)
    # ---------------------------
    def get_hypothesis_history(self, thread_id: str) -> List[Dict[str, Any]]:
        try:
            if not self.ensure_db_ready():
                return []
            return get_hypothesis_history(thread_id)
        except Exception:
            return []

    def get_alphas_for_hypothesis(self, hypothesis_id: int) -> List[Dict[str, Any]]:
        try:
            if not self.ensure_db_ready():
                return []
            return get_alphas_for_hypothesis(hypothesis_id)
        except Exception:
            return []

    def get_backtest_results_for_alpha(self, alpha_id: int) -> List[Dict[str, Any]]:
        try:
            if not self.ensure_db_ready():
                return []
            return get_backtest_results_for_alpha(alpha_id)
        except Exception:
            return []


_MANAGER: Optional[AlphaGPTCheckpointer] = None


def get_checkpoint_manager() -> AlphaGPTCheckpointer:
    """Singleton accessor for the AlphaGPTCheckpointer."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = AlphaGPTCheckpointer()
    return _MANAGER
