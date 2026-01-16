import numpy as np
import pandas as pd

from agent.research.portfolio_backtest import BacktestConfig
from agent.research.walk_forward import WalkForwardConfig, walk_forward_evaluate_factor


def _synthetic_factor_and_ohlcv(n_days: int = 260, n_inst: int = 20, seed: int = 7):
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


def test_walk_forward_evaluate_factor_returns_oos_metrics() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()
    wf = WalkForwardConfig(train_days=126, valid_days=42, test_days=42, step_days=42, expanding_train=True)
    bt = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5, commission_bps=0.0, slippage_bps=0.0)
    out = walk_forward_evaluate_factor(factor, ohlcv, wf_config=wf, bt_config=bt)
    assert out.get("mode") == "p1"
    assert "walk_forward" in out
    assert out.get("walk_forward", {}).get("splits")


def test_walk_forward_execution_only_ablation_has_all_scenarios() -> None:
    factor, ohlcv = _synthetic_factor_and_ohlcv()
    wf = WalkForwardConfig(train_days=126, valid_days=42, test_days=42, step_days=42, expanding_train=True)
    bt = BacktestConfig(
        rebalance_days=5,
        holding_days=5,
        n_quantiles=5,
        commission_bps=2.0,
        slippage_bps=1.0,
        half_spread_bps=5.0,
        impact_bps=50.0,
        borrow_bps=100.0,
        borrow_cost_multiplier=1.0,
    )

    out = walk_forward_evaluate_factor(factor, ohlcv, wf_config=wf, bt_config=bt, execution_only_ablation=True)
    eo = out.get("execution_only_ablation") or {}
    assert eo.get("mode") == "execution_only"

    scenarios = eo.get("scenarios") or []
    names = sorted([str(s.get("scenario")) for s in scenarios if isinstance(s, dict)])
    assert names == sorted(["full", "linear_only", "linear_spread", "linear_spread_impact", "no_costs"])

    drag = {str(s.get("scenario")): float(s.get("mean_cost_drag_bps") or 0.0) for s in scenarios if isinstance(s, dict)}
    assert abs(drag.get("no_costs", 0.0)) < 1e-9
    assert drag.get("full", 0.0) >= drag.get("linear_spread_impact", 0.0) >= drag.get("linear_spread", 0.0) >= drag.get("linear_only", 0.0)