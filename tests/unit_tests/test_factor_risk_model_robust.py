import numpy as np
import pandas as pd

from agent.research.factor_risk_model import estimate_factor_risk_model


def test_factor_risk_model_robust_options() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    names = ["A", "B", "C", "D", "E", "F"]

    loadings = pd.DataFrame(
        {
            "beta": [1.0, 0.8, 0.2, -0.1, -0.4, 0.0],
            "liq": [0.2, -0.1, 0.4, 0.3, -0.3, 0.0],
        },
        index=names,
    )

    f1 = rng.normal(0.0, 0.01, size=len(dates))
    f2 = rng.normal(0.0, 0.01, size=len(dates))
    eps = rng.normal(0.0, 0.02, size=(len(dates), len(names)))
    R = np.outer(f1, loadings["beta"].to_numpy()) + np.outer(f2, loadings["liq"].to_numpy()) + eps
    ret = pd.DataFrame(R, index=dates, columns=names)

    model = estimate_factor_risk_model(
        ret,
        loadings,
        window=60,
        min_obs=20,
        cov_estimator="ewm",
        ewm_halflife=15,
        cov_shrink_method="oas",
        factor_return_clip_sigma=4.0,
        idio_clip_q=0.95,
        idio_shrink=0.5,
        trading_days=252,
    )
    assert model is not None
    assert model.factor_cov.shape == (2, 2)
    assert np.allclose(model.factor_cov, model.factor_cov.T, atol=1e-10)
    eig = np.linalg.eigvalsh(model.factor_cov)
    assert float(eig.min()) >= -1e-8
    assert (model.idio_var >= 0.0).all()

    meta = model.meta
    assert meta.get("cov_estimator") in {"ewm", "sample"}
    assert meta.get("cov_shrink_method") in {"oas", "fixed", "oas_identity"}
    si = meta.get("shrink_intensity")
    if si is not None:
        assert 0.0 <= float(si) <= 1.0
