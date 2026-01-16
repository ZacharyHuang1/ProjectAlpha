import numpy as np
import pandas as pd

from agent.research.optimizer import OptimizerConfig, OptimizerCostModel, optimize_long_short_weights


def test_diag_risk_penalty_shifts_weights() -> None:
    # Two long, two short with identical scores.
    scores = pd.Series({"A": 1.0, "B": 1.0, "C": -1.0, "D": -1.0})

    long_names = ["A", "B"]
    short_names = ["C", "D"]

    cfg = OptimizerConfig(l2_lambda=1.0, turnover_lambda=0.0, exposure_lambda=0.0, max_iter=2)

    w0 = optimize_long_short_weights(
        scores,
        long_names=long_names,
        short_names=short_names,
        w_target=pd.Series(0.0, index=scores.index),
        exposures=None,
        cfg=cfg,
        gross_long=0.5,
        gross_short=0.5,
    )
    assert w0 is not None
    assert np.isclose(float(w0[w0 > 0.0].sum()), 0.5, atol=1e-8)
    assert np.isclose(float((-w0[w0 < 0.0]).sum()), 0.5, atol=1e-8)

    # Penalize A and C more than B and D.
    risk_var = pd.Series({"A": 0.20, "B": 0.02, "C": 0.20, "D": 0.02})
    cm = OptimizerCostModel(risk_aversion=10.0, risk_var=risk_var)

    w1 = optimize_long_short_weights(
        scores,
        long_names=long_names,
        short_names=short_names,
        w_target=pd.Series(0.0, index=scores.index),
        exposures=None,
        cfg=cfg,
        gross_long=0.5,
        gross_short=0.5,
        cost_model=cm,
    )
    assert w1 is not None

    # Lower variance names should carry more weight magnitude.
    assert float(w1["B"]) > float(w1["A"])
    assert abs(float(w1["D"])) > abs(float(w1["C"]))
