"""agent.research.pareto_plotting

P2.27: Optional plotting helpers for tuning frontiers.

These helpers are deliberately optional: if matplotlib is not available,
plot generation is skipped and the caller receives a structured error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _sf(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def try_make_pareto_scatter(
    *,
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    chosen: Optional[Dict[str, Any]],
    out_path: Path,
    title: str,
    feasible_key: str = "is_feasible",
    pareto_key: str = "is_pareto",
) -> Dict[str, Any]:
    """Make a simple scatter plot highlighting feasible + Pareto points.

    We intentionally avoid styling choices; matplotlib default colors are used.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return {"enabled": False, "error": f"matplotlib_unavailable:{e}"}

    if not rows:
        return {"enabled": False, "error": "no_rows"}

    xs: List[float] = []
    ys: List[float] = []
    is_feasible: List[bool] = []
    is_pareto: List[bool] = []

    for r in rows:
        x = _sf(r.get(x_key))
        y = _sf(r.get(y_key))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(float(x))
        ys.append(float(y))
        is_feasible.append(bool(r.get(feasible_key, True)))
        is_pareto.append(bool(r.get(pareto_key, False)))

    if len(xs) < 2:
        return {"enabled": False, "error": "insufficient_points"}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(str(title))
    ax.set_xlabel(str(x_key))
    ax.set_ylabel(str(y_key))

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    feas = np.asarray(is_feasible, dtype=bool)
    pare = np.asarray(is_pareto, dtype=bool)

    # Base layer
    ax.scatter(xs_arr, ys_arr, alpha=0.35, marker="o", label="all")
    # Feasible
    if feas.any():
        ax.scatter(xs_arr[feas], ys_arr[feas], alpha=0.55, marker="o", label="feasible")
    # Pareto
    if pare.any():
        ax.scatter(xs_arr[pare], ys_arr[pare], alpha=0.9, marker="x", label="pareto")

    # Chosen point (best-effort match by coordinates)
    if isinstance(chosen, dict) and chosen:
        cx = _sf(chosen.get(x_key))
        cy = _sf(chosen.get(y_key))
        if np.isfinite(cx) and np.isfinite(cy):
            ax.scatter([cx], [cy], marker="*", s=120, label="chosen")

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return {"enabled": True, "path": str(out_path)}
