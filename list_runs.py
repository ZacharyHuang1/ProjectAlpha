"""List local runs.

Wrapper around `agent.tools.list_runs`.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from agent.tools.list_runs import main


if __name__ == "__main__":
    main()
