import numpy as np
import pandas as pd

from agent.research.factor_risk_model import estimate_factor_risk_model


def test_factor_risk_model_estimator_psd() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    names = ["A", "B", "C", "D"]

    # One latent factor + idiosyncratic noise.
    f = rng.normal(0.0, 0.01, size=len(dates))
    eps = rng.normal(0.0, 0.02, size=(len(dates), len(names)))
    loadings = pd.DataFrame({"f1": [1.0, 0.5, -0.2, 0.0]}, index=names)

    R = np.outer(f, loadings["f1"].to_numpy()) + eps
    ret = pd.DataFrame(R, index=dates, columns=names)

    model = estimate_factor_risk_model(ret, loadings, window=30, min_obs=10, ridge=1e-3, cov_shrink=0.2, trading_days=252)
    assert model is not None
    assert model.factor_cov.shape == (1, 1)
    assert float(model.factor_cov[0, 0]) >= 0.0
    assert (model.idio_var >= 0.0).all()

    # Symmetry sanity check.
    assert np.allclose(model.factor_cov, model.factor_cov.T, atol=1e-10)
