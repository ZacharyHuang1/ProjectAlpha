"""Deprecated module (compat).

Selection meta-tuning utilities live in `agent.research.alpha_selection`.

This wrapper keeps older import paths working.
"""

from __future__ import annotations

from agent.research.alpha_selection import build_valid_return_matrix, tune_diverse_selection

__all__ = [
    "build_valid_return_matrix",
    "tune_diverse_selection",
]
