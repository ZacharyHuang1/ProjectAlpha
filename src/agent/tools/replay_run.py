"""agent.tools.replay_run

Replay a saved run by reading its config + idea.

Usage:
  python -m agent.tools.replay_run runs/<run_id>

Notes:
- If OPENAI_API_KEY is set, LLM outputs may differ (non-deterministic).
- With stubs, replay is deterministic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from agent.graph import graph
from agent.state import State


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str, help="Path to a run directory containing config.json/result.json")
    p.add_argument("--thread-id", type=str, default="", help="Optional new thread_id (default: reuse saved).")
    p.add_argument("--async", dest="use_async", action="store_true", help="Use graph.ainvoke")
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_json(run_dir / "config.json")
    res = _load_json(run_dir / "result.json")

    thread_id = args.thread_id or str(cfg.get("thread_id") or "replay")
    idea = str(res.get("trading_idea") or "")

    state = State(
        trading_idea=idea,
        max_iterations=int(res.get("max_iterations") or 1),
        target_information_ratio=float(res.get("target_information_ratio") or 0.0),
    )

    config = {"configurable": dict(cfg)}
    config["configurable"]["thread_id"] = thread_id

    if args.use_async:
        import asyncio
        out = asyncio.run(graph.ainvoke(state, config))
    else:
        out = graph.invoke(state, config)

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
