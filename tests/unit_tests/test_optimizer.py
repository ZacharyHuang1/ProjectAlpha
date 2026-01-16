import numpy as np
import pandas as pd

from agent.research.optimizer import OptimizerConfig, optimize_long_short_weights, select_long_short_candidates


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    x = x.astype(float)
    y = y.astype(float)
    xr = x.rank(method="average").to_numpy(dtype=float)
    yr = y.rank(method="average").to_numpy(dtype=float)
    xr = xr - xr.mean()
    yr = yr - yr.mean()
    denom = float(np.sqrt((xr * xr).sum() * (yr * yr).sum()))
    if denom <= 0.0:
        return 0.0
    return float((xr * yr).sum() / denom)


def test_optimizer_constructs_valid_long_short_weights() -> None:
    scores = pd.Series({"A": -3.0, "B": -2.0, "C": -1.0, "D": 1.0, "E": 2.0, "F": 3.0})
    cand = select_long_short_candidates(scores, n_quantiles=2, max_names_per_side=2)
    assert cand is not None
    long_names, short_names = cand

    cfg = OptimizerConfig(l2_lambda=1.0, turnover_lambda=0.0, exposure_lambda=0.0, max_iter=2)
    w = optimize_long_short_weights(
        scores,
        long_names=long_names,
        short_names=short_names,
        w_target=pd.Series(0.0, index=scores.index),
        exposures=None,
        cfg=cfg,
        gross_long=0.5,
        gross_short=0.5,
    )
    assert w is not None

    # Signs and gross exposure.
    assert float(w.loc[long_names].min()) >= -1e-12
    assert float(w.loc[short_names].max()) <= 1e-12
    assert abs(float(w.clip(lower=0.0).sum()) - 0.5) < 1e-6
    assert abs(float((-w.clip(upper=0.0)).sum()) - 0.5) < 1e-6

    # Weights should reflect scores within each side (not equal-weight).
    long_corr = _rank_corr(scores.loc[long_names], w.loc[long_names])
    short_corr = _rank_corr((-scores.loc[short_names]), (-w.loc[short_names]))
    assert long_corr > 0.5
    assert short_corr > 0.5


def test_optimizer_respects_shortable_mask_in_selection() -> None:
    scores = pd.Series({"A": -3.0, "B": -2.0, "C": 1.0, "D": 2.0})
    shortable = pd.Series({"A": False, "B": True, "C": True, "D": True})
    cand = select_long_short_candidates(scores, n_quantiles=2, max_names_per_side=1, shortable=shortable)
    assert cand is not None
    long_names, short_names = cand
    assert "A" not in short_names
