"""agent.tools.list_runs

List local runs written by P2.3 experiment tracking.

Usage:
  python -m agent.tools.list_runs --runs-root runs --limit 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=str, default="runs", help="Runs root directory")
    p.add_argument("--limit", type=int, default=20, help="How many runs to show")
    args = p.parse_args()

    root = Path(args.runs_root).expanduser().resolve()
    idx = root / "_index.jsonl"
    rows = _read_jsonl(idx)

    if not rows:
        print("No runs found.")
        return

    rows = rows[-int(max(1, args.limit)) :]
    for r in rows:
        print(
            f"{r.get('run_id')} | mode={r.get('eval_mode')} | best={r.get('best_alpha_id')} "
            f"| IR={r.get('best_information_ratio')} | data={r.get('data_path') or 'synthetic'}"
        )


if __name__ == "__main__":
    main()
