"""Pytest configuration.

This makes the `src/` layout importable when running tests without installing
the package (e.g., `pytest` from the repo root).
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
