import numpy as np
import pandas as pd

from agent.research.alpha_eval import compute_forward_returns
from agent.research.decay_analysis import compute_horizon_decay


def test_compute_horizon_decay_basic() -> None:
    # Build a synthetic close series where the factor equals 1-day forward returns.
    n_days = 40
    n_inst = 8
    rng = np.random.default_rng(123)

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    inst = [f"I{i}" for i in range(n_inst)]

    # Generate daily returns (t -> t+1) and convert to a close price path.
    rets = rng.normal(loc=0.0, scale=0.01, size=(n_days - 1, n_inst))
    close = np.ones((n_days, n_inst), dtype=float) * 100.0
    for t in range(n_days - 1):
        close[t + 1] = close[t] * (1.0 + rets[t])

    close_df = pd.DataFrame(close, index=dates, columns=inst)
    close_s = close_df.stack()
    close_s.index = close_s.index.set_names(["datetime", "instrument"])

    # Factor = exact 1-day forward return.
    factor = compute_forward_returns(close_s, horizon=1)

    out = compute_horizon_decay(
        factor=factor,
        close=close_s,
        horizons=[1, 2],
        n_quantiles=5,
        min_obs_per_day=5,
        trading_days=252,
    )

    assert out.get("enabled") is True
    rows = out.get("metrics")
    assert isinstance(rows, list)
    assert len(rows) == 2

    by_h = {int(r.get("horizon")): r for r in rows}
    assert 1 in by_h and 2 in by_h

    ic1 = float(by_h[1].get("ic_mean"))
    ic2 = float(by_h[2].get("ic_mean"))

    # Horizon-1 should be extremely close to perfect correlation.
    assert ic1 > 0.99
    # Horizon-2 should be strictly less than perfect.
    assert ic2 < 0.999

    ov1 = by_h[1].get("signal_overlap_mean")
    ov2 = by_h[2].get("signal_overlap_mean")
    assert ov1 is None or (0.0 <= float(ov1) <= 1.0)
    assert ov2 is None or (0.0 <= float(ov2) <= 1.0)
