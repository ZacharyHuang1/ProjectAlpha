import numpy as np
import pandas as pd

from agent.research.alpha_allocation import fit_alpha_allocation
from agent.research.holdings_ensemble import walk_forward_holdings_ensemble_allocated
from agent.research.portfolio_backtest import BacktestConfig
from agent.research.walk_forward import WalkForwardConfig


def test_fit_alpha_allocation_basic() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, size=len(idx)),
            "B": rng.normal(0.0002, 0.01, size=len(idx)),
            "C": rng.normal(0.0001, 0.01, size=len(idx)),
        },
        index=idx,
    )
    out = fit_alpha_allocation(r, trading_days=252, lambda_corr=0.5, max_weight=0.8, backend="pgd")
    w = out.get("weights")
    assert isinstance(w, pd.Series)
    assert abs(float(w.sum()) - 1.0) < 1e-6
    assert (w >= 0.0).all()
    assert float(w.max()) <= 0.800001


def _synthetic_factor_and_ohlcv(n_days: int = 200, n_inst: int = 25, seed: int = 7):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)

    f = rng.normal(0, 1, size=(n_days, n_inst))
    f2 = rng.normal(0, 1, size=(n_days, n_inst))
    noise = rng.normal(0, 0.01, size=(n_days, n_inst))
    rets = 0.001 * (f / (np.std(f) + 1e-9)) + noise

    px = 100 * np.cumprod(1 + rets, axis=0)
    vol = rng.integers(1000, 5000, size=(n_days, n_inst))

    midx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    df = pd.DataFrame(
        {
            "open": px.reshape(-1),
            "high": (px * 1.01).reshape(-1),
            "low": (px * 0.99).reshape(-1),
            "close": px.reshape(-1),
            "volume": vol.reshape(-1),
        },
        index=midx,
    )

    factor = pd.Series(f.reshape(-1), index=midx, name="factor")
    factor2 = pd.Series(f2.reshape(-1), index=midx, name="factor2")
    return factor, factor2, df


def test_walk_forward_holdings_ensemble_allocated_runs() -> None:
    f1, f2, ohlcv = _synthetic_factor_and_ohlcv()
    wf_cfg = WalkForwardConfig(train_days=80, valid_days=20, test_days=20, step_days=20, expanding_train=True)
    bt_cfg = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5)

    alphas = [
        {"alpha_id": "A1", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
        {"alpha_id": "A2", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
    ]

    out = walk_forward_holdings_ensemble_allocated(
        selected_alphas=alphas,
        factor_cache={"A1": f1, "A2": f2},
        ohlcv=ohlcv,
        wf_config=wf_cfg,
        bt_config=bt_cfg,
        allocation_backend="pgd",
        allocation_min_days=10,
    )
    assert bool(out.get("enabled")) is True
    met = out.get("metrics") or {}
    assert "information_ratio" in met
    assert isinstance(out.get("allocations"), list)
