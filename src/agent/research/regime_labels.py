"""agent.research.regime_labels

P2.21: Map continuous market features into a small discrete regime label.

Key points:
- Thresholds are fit on the *fit* segment only (train or train+valid).
- Regime values are then assigned to any target date range using the same
  thresholds.
- Inputs are expected to be lookahead-safe already (e.g., rolling features
  shifted by 1 day).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeLabelMeta:
    mode: str
    buckets: int
    vol_edges: Tuple[float, ...]
    liq_edges: Tuple[float, ...]
    effective_vol_buckets: int
    effective_liq_buckets: int


def _fit_quantile_edges(values: pd.Series, buckets: int) -> np.ndarray:
    """Fit quantile edges on non-NaN values.

    The output is an array of edges with -inf/inf at the ends. If the data
    is degenerate, this gracefully falls back to a single bucket.
    """

    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if v.size < 2:
        return np.asarray([-np.inf, np.inf], dtype=float)

    q = np.linspace(0.0, 1.0, max(1, int(buckets)) + 1)
    try:
        edges = np.quantile(v, q)
    except Exception:
        edges = np.asarray([-np.inf, np.inf], dtype=float)

    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return np.asarray([-np.inf, np.inf], dtype=float)

    edges[0] = -np.inf
    edges[-1] = np.inf

    # Remove duplicates to avoid zero-width buckets.
    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    if edges.size < 2:
        return np.asarray([-np.inf, np.inf], dtype=float)
    return edges


def _assign_bucket(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Assign 0..K-1 bucket labels using pre-fit edges."""

    if edges is None or len(edges) < 2:
        return pd.Series(index=values.index, dtype=float)

    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    out = pd.Series(index=x.index, dtype=float)
    ok = x.notna()
    if not bool(ok.any()):
        return out

    # digitize uses the interior edges as bin cut points.
    cuts = np.asarray(edges[1:-1], dtype=float)
    b = np.digitize(x.loc[ok].to_numpy(dtype=float), cuts, right=True).astype(float)
    out.loc[ok] = b
    return out


def make_regime_labels(
    features: pd.DataFrame,
    *,
    target_index: pd.DatetimeIndex,
    fit_index: pd.DatetimeIndex,
    mode: str = "vol",
    buckets: int = 3,
) -> Tuple[pd.Series, RegimeLabelMeta]:
    """Create discrete regime labels for a target index.

    Parameters
    ----------
    features:
        Output from `compute_market_feature_frame`.
    target_index:
        Dates to label (e.g., test dates).
    fit_index:
        Dates used to fit thresholds (e.g., train or train+valid).
    mode:
        - "vol": volatility-only regimes
        - "vol_liq": volatility x liquidity regimes
    buckets:
        Requested number of quantile buckets per feature.
    """

    mode = str(mode).lower().strip()
    b = max(1, int(buckets))

    f = features.copy()
    f.index = pd.to_datetime(f.index)
    f = f.sort_index()

    fit_idx = pd.to_datetime(pd.Index(fit_index)).intersection(f.index)
    tgt_idx = pd.to_datetime(pd.Index(target_index)).intersection(f.index)

    vol_edges = _fit_quantile_edges(f.loc[fit_idx, "mkt_vol"], buckets=b)
    liq_edges = _fit_quantile_edges(f.loc[fit_idx, "mkt_liq"], buckets=b)

    vol_bucket = _assign_bucket(f.loc[tgt_idx, "mkt_vol"], vol_edges)
    liq_bucket = _assign_bucket(f.loc[tgt_idx, "mkt_liq"], liq_edges)

    eff_vol = int(max(1, len(vol_edges) - 1))
    eff_liq = int(max(1, len(liq_edges) - 1))

    if mode == "vol_liq":
        lbl = pd.Series(index=tgt_idx, dtype=object)
        for dt in tgt_idx:
            v = vol_bucket.get(dt)
            l = liq_bucket.get(dt)
            if pd.isna(v) or pd.isna(l):
                lbl.loc[dt] = np.nan
            else:
                lbl.loc[dt] = f"vol_{int(v)}_liq_{int(l)}"
    else:
        # Default: volatility-only.
        lbl = vol_bucket.apply(lambda x: (f"vol_{int(x)}" if pd.notna(x) else np.nan))
        lbl.index = tgt_idx

    meta = RegimeLabelMeta(
        mode=mode,
        buckets=int(b),
        vol_edges=tuple(float(x) for x in np.asarray(vol_edges, dtype=float).tolist()),
        liq_edges=tuple(float(x) for x in np.asarray(liq_edges, dtype=float).tolist()),
        effective_vol_buckets=int(eff_vol),
        effective_liq_buckets=int(eff_liq),
    )

    return lbl, meta


def regime_stats(labels: pd.Series) -> Dict[str, float]:
    """Compute light regime usage diagnostics."""

    if labels is None or labels.empty:
        return {"n_days": 0.0, "n_regimes": 0.0, "switch_rate": 0.0}

    s = pd.Series(labels).dropna()
    if s.empty:
        return {"n_days": float(len(labels)), "n_regimes": 0.0, "switch_rate": 0.0}

    n = int(s.size)
    uniq = int(s.nunique(dropna=True))
    switches = int((s != s.shift(1)).sum() - 1) if n >= 2 else 0
    switch_rate = float(switches) / float(max(n - 1, 1))
    return {"n_days": float(n), "n_regimes": float(uniq), "switch_rate": float(switch_rate)}
