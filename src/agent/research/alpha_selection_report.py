"""Deprecated module (compat).

The alpha selection report builder was consolidated into
`agent.research.alpha_selection`.

This wrapper keeps the old import path working.
"""

from __future__ import annotations

from agent.research.alpha_selection import build_alpha_selection_report, render_alpha_selection_report_md

__all__ = [
    "build_alpha_selection_report",
    "render_alpha_selection_report_md",
]
