import numpy as np
import pandas as pd

from agent.research.diversity_ensemble import (
    compute_return_correlation,
    greedy_diversified_selection,
    make_equal_weight_ensemble,
)


def test_greedy_diversified_selection_prefers_lower_corr_when_lambda_high():
    rng = np.random.default_rng(123)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")

    a = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    b = a + pd.Series(rng.normal(0, 0.001, size=len(idx)), index=idx)  # highly correlated with a
    c = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)  # low correlation

    df = pd.DataFrame({"A": a, "B": b, "C": c})
    corr, _ = compute_return_correlation(df, min_periods=50)

    scores = {"B": 2.0, "A": 1.9, "C": 1.5}

    sel, _ = greedy_diversified_selection(scores=scores, corr=corr, k=2, diversity_lambda=5.0, use_abs_corr=True)

    # Should select the best scorer first, then avoid the redundant highly-correlated one.
    assert sel[0] == "B"
    assert "C" in sel


def test_equal_weight_ensemble_shapes():
    idx = pd.date_range("2021-01-01", periods=50, freq="B")
    df = pd.DataFrame(
        {"x": np.linspace(-0.001, 0.001, len(idx)), "y": np.linspace(0.002, -0.002, len(idx))},
        index=idx,
    )

    ens = make_equal_weight_ensemble(df, ["x", "y"], trading_days=252)
    assert ens.get("enabled") is True
    assert ens.get("metrics", {}).get("n_alphas") == 2
    assert isinstance(ens.get("daily"), list)
    assert len(ens.get("daily")) == len(idx)
