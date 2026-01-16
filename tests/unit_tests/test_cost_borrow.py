import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig, _weights_from_factor, backtest_long_short


def _synthetic_factor_and_ohlcv(n_days: int = 80, n_inst: int = 12, seed: int = 11):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)

    f = rng.normal(0, 1, size=(n_days, n_inst))
    noise = rng.normal(0, 0.01, size=(n_days, n_inst))
    r = np.zeros_like(f)
    r[1:] = 0.002 * f[:-1] + noise[1:]

    px = np.ones((n_days, n_inst), dtype=float)
    for t in range(1, n_days):
        px[t] = px[t - 1] * (1.0 + r[t])

    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    factor = pd.Series(f.reshape(-1), index=idx)
    close = pd.Series(px.reshape(-1), index=idx)
    volume = pd.Series(rng.integers(500_000, 2_000_000, size=close.size), index=idx)
    ohlcv = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    return factor, ohlcv


def test_weights_from_factor_respects_shortable_mask() -> None:
    x = pd.Series({"A": -3.0, "B": -2.0, "C": 1.0, "D": 2.0})
    shortable = pd.Series({"A": False, "B": True, "C": True, "D": True})
    w = _weights_from_factor(x, n_quantiles=2, max_names_per_side=1, shortable=shortable)
    assert w is not None
    assert float(w.get("A", 0.0)) == 0.0
    assert float(w.get("B", 0.0)) < 0.0
    assert float(w.get("D", 0.0)) > 0.0


def test_impact_cost_reduces_net_returns() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()

    base = BacktestConfig(
        rebalance_days=1,
        holding_days=1,
        n_quantiles=3,
        commission_bps=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
        impact_bps=0.0,
    )
    with_impact = BacktestConfig(
        rebalance_days=1,
        holding_days=1,
        n_quantiles=3,
        commission_bps=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
        impact_bps=500.0,
        impact_exponent=0.5,
        portfolio_notional=1e9,
        impact_max_participation=0.2,
    )

    out0 = backtest_long_short(factor, ohlcv, config=base, include_daily=True)
    out1 = backtest_long_short(factor, ohlcv, config=with_impact, include_daily=True)

    d0 = out0.get("daily") or []
    d1 = out1.get("daily") or []
    m0 = float(np.mean([r["net_return"] for r in d0])) if d0 else 0.0
    m1 = float(np.mean([r["net_return"] for r in d1])) if d1 else 0.0

    assert float(out1.get("impact_cost_mean") or 0.0) >= 0.0
    assert m1 <= m0 + 1e-12
