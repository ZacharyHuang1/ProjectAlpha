import pytest

pytest.importorskip("langsmith")
from langsmith import unit

from agent import graph


def _get(obj, key: str):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


@pytest.mark.asyncio
@unit
async def test_graph_runs_with_minimal_input() -> None:
    res = await graph.ainvoke({"trading_idea": "demo idea"}, {"configurable": {"thread_id": "test_thread"}})
    assert res is not None
    assert _get(res, "hypothesis") is not None
