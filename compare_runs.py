"""Compare two local runs.

This is a thin wrapper around `agent.tools.compare_runs` so you can run:

  python compare_runs.py runs/<A> runs/<B>

without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from agent.tools.compare_runs import main


if __name__ == "__main__":
    main()
