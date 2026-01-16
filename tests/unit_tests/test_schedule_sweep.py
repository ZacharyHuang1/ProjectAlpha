import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig
from agent.research.schedule_sweep import compute_holding_rebalance_sweep
from agent.research.walk_forward import WalkForwardConfig, make_walk_forward_splits


def _synthetic_factor_and_ohlcv(n_days: int = 140, n_inst: int = 12, seed: int = 7):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)

    f = rng.normal(0, 1, size=(n_days, n_inst))
    noise = rng.normal(0, 0.01, size=(n_days, n_inst))
    r = np.zeros_like(f)
    r[1:] = 0.002 * f[:-1] + noise[1:]

    px = np.full((n_days, n_inst), 50.0)
    for t in range(1, n_days):
        px[t] = px[t - 1] * (1.0 + r[t])

    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    factor = pd.Series(f.reshape(-1), index=idx)
    close = pd.Series(px.reshape(-1), index=idx)
    volume = pd.Series(rng.integers(500_000, 2_000_000, size=close.size), index=idx)
    ohlcv = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    return factor, ohlcv


def test_schedule_sweep_runs_and_returns_rows() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()

    wf = WalkForwardConfig(train_days=60, valid_days=0, test_days=20, step_days=20, expanding_train=True)
    dates = pd.to_datetime(ohlcv.index.get_level_values("datetime").unique()).sort_values()
    splits = make_walk_forward_splits(
        dates,
        train_days=wf.train_days,
        valid_days=wf.valid_days,
        test_days=wf.test_days,
        step_days=wf.step_days,
        expanding_train=wf.expanding_train,
    )
    assert splits, "Expected at least one walk-forward split"

    bt = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5, min_obs=10)
    out = compute_holding_rebalance_sweep(
        factor,
        ohlcv,
        wf_config=wf,
        base_bt_config=bt,
        splits=splits,
        rebalance_days_list=[5, 10],
        holding_days_list=[5],
        max_combos=3,
    )

    assert out.get("enabled") is True
    rows = out.get("results") or []
    assert isinstance(rows, list)
    assert len(rows) >= 1

    # At least one row should have numeric fields when the run succeeds.
    ok = [r for r in rows if isinstance(r, dict) and not r.get("error")]
    assert ok, "Expected at least one successful schedule evaluation"
    assert "information_ratio" in ok[0]
