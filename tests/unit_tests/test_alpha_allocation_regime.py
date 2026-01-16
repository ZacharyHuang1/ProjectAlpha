import numpy as np
import pandas as pd

from agent.research.alpha_allocation_regime import build_daily_alpha_weights, fit_regime_allocations
from agent.research.holdings_ensemble import walk_forward_holdings_ensemble_allocated_regime
from agent.research.portfolio_backtest import BacktestConfig
from agent.research.walk_forward import WalkForwardConfig


def test_fit_regime_allocations_prefers_correct_alpha_by_regime() -> None:
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.DataFrame(0.0, index=idx, columns=["A", "B", "C"])
    r.loc[idx[:30], "A"] = 0.01
    r.loc[idx[30:], "B"] = 0.01

    lbl = pd.Series(["vol_0"] * 30 + ["vol_1"] * 30, index=idx)

    out = fit_regime_allocations(
        r,
        lbl,
        trading_days=252,
        score_metric="information_ratio",
        lambda_corr=0.0,
        max_weight=1.0,
        backend="pgd",
        min_days=10,
    )

    w_by = out.get("weights_by_regime") or {}
    assert "vol_0" in w_by and "vol_1" in w_by
    assert w_by["vol_0"].idxmax() == "A"
    assert w_by["vol_1"].idxmax() == "B"

    # Build daily weights and ensure the majority weight follows the regime.
    w_daily, diag = build_daily_alpha_weights(labels=lbl, weights_by_regime=w_by, fallback=out["global_weights"])
    assert diag["fallback_days"] == 0
    assert (w_daily.loc[idx[:30]].mean().idxmax()) == "A"
    assert (w_daily.loc[idx[30:]].mean().idxmax()) == "B"


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


def test_walk_forward_holdings_ensemble_allocated_regime_runs() -> None:
    f1, f2, ohlcv = _synthetic_factor_and_ohlcv()
    wf_cfg = WalkForwardConfig(train_days=80, valid_days=20, test_days=20, step_days=20, expanding_train=True)
    bt_cfg = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5)

    alphas = [
        {"alpha_id": "A1", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
        {"alpha_id": "A2", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
    ]

    out = walk_forward_holdings_ensemble_allocated_regime(
        selected_alphas=alphas,
        factor_cache={"A1": f1, "A2": f2},
        ohlcv=ohlcv,
        wf_config=wf_cfg,
        bt_config=bt_cfg,
        allocation_backend="pgd",
        allocation_min_days=10,
        regime_mode="vol",
        regime_window=10,
        regime_buckets=3,
        regime_min_days=10,
        regime_smoothing=0.10,
    )
    assert bool(out.get("enabled")) is True
    met = out.get("metrics") or {}
    assert "information_ratio" in met
    assert isinstance(out.get("allocations_regime"), list)

def test_regime_holdings_revalidation_runs() -> None:
    """Smoke test for P2.24 holdings-level revalidation.

    We only assert that the pipeline runs and emits the expected payload fields.
    """
    f1, f2, ohlcv = _synthetic_factor_and_ohlcv(n_days=160, n_inst=15, seed=11)
    wf_cfg = WalkForwardConfig(train_days=60, valid_days=20, test_days=20, step_days=20, expanding_train=True)
    bt_cfg = BacktestConfig(rebalance_days=5, holding_days=5, n_quantiles=5)

    alphas = [
        {"alpha_id": "A1", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
        {"alpha_id": "A2", "backtest_results": {"walk_forward": {"splits": [{"split_id": 0, "sign": 1.0}]}}},
    ]

    out = walk_forward_holdings_ensemble_allocated_regime(
        selected_alphas=alphas,
        factor_cache={"A1": f1, "A2": f2},
        ohlcv=ohlcv,
        wf_config=wf_cfg,
        bt_config=bt_cfg,
        allocation_backend="pgd",
        allocation_min_days=10,
        regime_mode="vol",
        regime_window=10,
        regime_buckets=3,
        regime_min_days=10,
        regime_smoothing=0.10,
        regime_tune=True,
        regime_tune_metric="information_ratio",
        regime_tune_max_combos=6,
        regime_tune_mode_grid=["vol"],
        regime_tune_window_grid=[10],
        regime_tune_buckets_grid=[2, 3],
        regime_tune_smoothing_grid=[0.0, 0.1],
        regime_tune_turnover_penalty=0.0,
        regime_tune_holdings_top=2,
        regime_tune_holdings_metric="",
        regime_tune_holdings_save_top=5,
    )

    assert bool(out.get("enabled")) is True
    hv = out.get("regime_tuning_holdings_validation") or {}
    assert isinstance(hv, dict)
    assert int(hv.get("top_n") or 0) == 2

    reg_cfg = (out.get("allocation") or {}).get("regime") or {}
    assert str(reg_cfg.get("tuned_method")) in {"fixed", "proxy", "holdings_valid"}
