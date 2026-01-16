import pandas as pd
import numpy as np

from agent.research.alpha_eval import evaluate_alpha


def _toy_series(n_days: int = 40, n_inst: int = 20):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    rng = np.random.default_rng(1)
    fwd = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    factor = fwd + rng.normal(0, 0.001, size=len(idx))
    universe = pd.Series(n_inst, index=dates)
    return factor, fwd, universe


def test_evaluate_alpha_returns_metrics() -> None:
    factor, fwd, universe = _toy_series()
    m = evaluate_alpha(factor, fwd, n_quantiles=5, universe_size_by_date=universe, min_obs_per_day=10)
    assert "information_ratio" in m
    assert m.get("n_obs", 0) > 0
    # factor is constructed to be positively correlated with fwd
    assert m.get("ic", 0.0) > 0.5
