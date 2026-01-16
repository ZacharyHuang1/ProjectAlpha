import numpy as np
import pandas as pd

from agent.research.holdings_ensemble import positions_to_weight_matrix, walk_forward_holdings_ensemble
from agent.research.portfolio_backtest import BacktestConfig, backtest_from_weights, backtest_long_short
from agent.research.walk_forward import WalkForwardConfig


def _synthetic_factor_and_ohlcv(n_days: int = 180, n_inst: int = 25, seed: int = 7):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)

    f = rng.normal(0, 1, size=(n_days, n_inst))
    f2 = rng.normal(0, 1, size=(n_days, n_inst))
    noise = rng.normal(0, 0.01, size=(n_days, n_inst))
    r = np.zeros_like(f)
    r[1:] = 0.002 * f[:-1] + 0.001 * f2[:-1] + noise[1:]

    px = np.full((n_days, n_inst), 100.0)
    for t in range(1, n_days):
        px[t] = px[t - 1] * (1.0 + r[t])

    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    factor1 = pd.Series(f.reshape(-1), index=idx)
    factor2 = pd.Series(f2.reshape(-1), index=idx)
    close = pd.Series(px.reshape(-1), index=idx)
    volume = pd.Series(rng.integers(500_000, 2_000_000, size=close.size), index=idx)
    ohlcv = pd.DataFrame({"close": close, "volume": volume}, index=idx)
    return factor1, factor2, ohlcv


def test_backtest_positions_and_backtest_from_weights_roundtrip() -> None:
    f1, _, ohlcv = _synthetic_factor_and_ohlcv()
    cfg = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5)

    bt = backtest_long_short(f1, ohlcv, config=cfg, include_daily=False, include_positions=True)
    assert bt.get("error") is None
    assert isinstance(bt.get("position_dates"), list)

    w = positions_to_weight_matrix(
        positions=list(bt.get("positions") or []),
        position_dates=list(bt.get("position_dates") or []),
        instruments=list(pd.Index(ohlcv.index.get_level_values("instrument")).unique()),
    )
    assert not w.empty

    bt2 = backtest_from_weights(w, ohlcv, config=cfg, include_daily=False)
    assert bt2.get("error") is None
    assert "information_ratio" in bt2


def test_walk_forward_holdings_ensemble_runs() -> None:
    f1, f2, ohlcv = _synthetic_factor_and_ohlcv()
    wf_cfg = WalkForwardConfig(train_days=80, valid_days=20, test_days=20, step_days=20, expanding_train=True)
    bt_cfg = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5)

    # Minimal alpha payloads with split sign metadata.
    alphas = [
        {"alpha_id": "A1", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
        {"alpha_id": "A2", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
    ]

    out = walk_forward_holdings_ensemble(
        selected_alphas=alphas,
        factor_cache={"A1": f1, "A2": f2},
        ohlcv=ohlcv,
        wf_config=wf_cfg,
        bt_config=bt_cfg,
        apply_turnover_cap=False,
    )
    assert bool(out.get("enabled")) is True
    met = out.get("metrics") or {}
    assert "information_ratio" in met
    assert isinstance(out.get("daily"), list)