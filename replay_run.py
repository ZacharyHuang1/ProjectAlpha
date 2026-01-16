"""Replay a saved run.

Wrapper around `agent.tools.replay_run`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from agent.tools.replay_run import main


if __name__ == "__main__":
    main()
