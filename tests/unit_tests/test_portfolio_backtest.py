import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig, backtest_long_short


def _synthetic_factor_and_ohlcv(n_days: int = 160, n_inst: int = 30, seed: int = 123):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)

    # Factor at t predicts returns at t+1.
    f = rng.normal(0, 1, size=(n_days, n_inst))
    noise = rng.normal(0, 0.01, size=(n_days, n_inst))
    r = np.zeros_like(f)
    r[1:] = 0.002 * f[:-1] + noise[1:]

    px = np.full((n_days, n_inst), 100.0)
    for t in range(1, n_days):
        px[t] = px[t - 1] * (1.0 + r[t])

    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    factor = pd.Series(f.reshape(-1), index=idx)
    close = pd.Series(px.reshape(-1), index=idx)

    # Provide volume so P2 features (ADV, liquidity exposure) can run.
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, size=close.size), index=idx)
    ohlcv = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    return factor, ohlcv


def test_backtest_long_short_runs() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()
    cfg = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5, commission_bps=0.0, slippage_bps=0.0)
    out = backtest_long_short(factor, ohlcv, config=cfg, include_daily=False)
    assert "information_ratio" in out
    assert out.get("n_obs", 0) > 0


def test_backtest_p2_features_do_not_crash() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()
    cfg = BacktestConfig(
        rebalance_days=5,
        holding_days=5,
        n_quantiles=5,
        commission_bps=0.0,
        slippage_bps=0.0,
        neutralize_beta=True,
        neutralize_liquidity=True,
        target_vol_annual=0.10,
        vol_target_window=20,
    )
    out = backtest_long_short(factor, ohlcv, config=cfg, include_daily=False)
    assert "information_ratio" in out


def test_backtest_optimizer_construction_runs() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()
    cfg = BacktestConfig(
        rebalance_days=5,
        holding_days=5,
        n_quantiles=5,
        construction_method="optimizer",
        optimizer_turnover_lambda=5.0,
        neutralize_beta=True,
        neutralize_liquidity=True,
    )
    out = backtest_long_short(factor, ohlcv, config=cfg, include_daily=False)
    assert "information_ratio" in out