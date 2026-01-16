import pandas as pd

from agent.research.optimizer import OptimizerConfig, optimize_long_short_weights_with_meta, select_long_short_candidates


def test_optimizer_qp_backend_gracefully_falls_back() -> None:
    scores = pd.Series({"A": -3.0, "B": -2.0, "C": -1.0, "D": 1.0, "E": 2.0, "F": 3.0})
    cand = select_long_short_candidates(scores, n_quantiles=2, max_names_per_side=2)
    assert cand is not None
    long_names, short_names = cand

    exposures = pd.DataFrame({"beta": [0.5, 0.2, -0.1, 0.1, -0.2, -0.3]}, index=list(scores.index))

    cfg = OptimizerConfig(l2_lambda=1.0, turnover_lambda=1.0, exposure_lambda=0.1, max_iter=2)
    w, meta = optimize_long_short_weights_with_meta(
        scores,
        long_names=long_names,
        short_names=short_names,
        w_target=pd.Series(0.0, index=scores.index),
        exposures=exposures,
        cfg=cfg,
        backend="qp",
        turnover_cap=0.2,
        enforce_exposure_neutrality=True,
        max_abs_weight=0.5,
    )

    assert w is not None
    assert str(meta.get("backend_requested")) == "qp"
    assert str(meta.get("backend_used")) in {"ridge", "qp"}

    # In CI / minimal installs we expect no cvxpy, therefore a ridge fallback.
    if str(meta.get("backend_used")) == "ridge":
        assert meta.get("fallback") is not None
