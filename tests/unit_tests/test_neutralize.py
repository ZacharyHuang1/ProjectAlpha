import numpy as np
import pandas as pd

from agent.research.neutralize import neutralize_weights


def test_neutralize_weights_removes_exposure() -> None:
    rng = np.random.default_rng(0)
    inst = [f"S{i:03d}" for i in range(30)]
    w = pd.Series(rng.normal(0, 1, size=len(inst)), index=inst)
    w = w - float(w.mean())  # roughly dollar-neutral
    w = w / float(w.abs().sum())

    beta = pd.Series(rng.normal(0, 1, size=len(inst)), index=inst, name="beta")
    X = pd.DataFrame({"beta": beta})

    w2 = neutralize_weights(w, X, add_intercept=True)
    assert abs(float(w2.sum())) < 1e-8
    assert abs(float((w2 * beta).sum())) < 1e-6
