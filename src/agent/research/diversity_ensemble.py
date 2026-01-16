"""Deprecated module (compat).

The project consolidated selection/tuning/ensemble utilities into
`agent.research.alpha_selection` to reduce duplication.

This file keeps the old import path working.

Note: the old API for `greedy_diversified_selection` returned `(selected, table)`.
The consolidated implementation returns `(selected, table, rejected)`.
This wrapper preserves the 2-tuple return.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agent.research.alpha_selection import (
    compute_return_correlation,
    greedy_diversified_selection as _greedy_impl,
    make_equal_weight_ensemble,
)


def greedy_diversified_selection(
    *,
    scores: Dict[str, float],
    corr: pd.DataFrame,
    k: int,
    diversity_lambda: float = 0.0,
    use_abs_corr: bool = True,
    max_pairwise_corr: Optional[float] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Compatibility wrapper returning `(selected, selection_table)`."""

    selected, table, _rejected = _greedy_impl(
        scores=scores,
        corr=corr,
        k=k,
        diversity_lambda=diversity_lambda,
        use_abs_corr=use_abs_corr,
        max_pairwise_corr=max_pairwise_corr,
    )
    return selected, table


__all__ = [
    "compute_return_correlation",
    "greedy_diversified_selection",
    "make_equal_weight_ensemble",
]
