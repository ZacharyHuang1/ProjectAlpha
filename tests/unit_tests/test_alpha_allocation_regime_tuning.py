import numpy as np
import pandas as pd

from agent.research.alpha_allocation_regime_tuning import meta_tune_regime_aware_allocation


def test_regime_meta_tuning_picks_more_informative_buckets():
    # Synthetic walk-forward: one split with train then valid.
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    train_idx = dates[:40]
    valid_idx = dates[40:]

    # Two alphas:
    # - A works in low-vol, fails in high-vol
    # - B fails in low-vol, works in high-vol
    a_train = np.concatenate([np.full(20, 0.001), np.full(20, -0.001)])
    b_train = np.concatenate([np.full(20, -0.001), np.full(20, 0.001)])
    a_valid = np.full(20, -0.001)
    b_valid = np.full(20, 0.001)

    train_df = pd.DataFrame({"A": a_train, "B": b_train}, index=train_idx)
    valid_df = pd.DataFrame({"A": a_valid, "B": b_valid}, index=valid_idx)

    train_by_split = {0: train_df}
    valid_by_split = {0: valid_df}

    # Hand-crafted market features that cleanly separate low/high volatility.
    mkt_vol = pd.Series(np.concatenate([np.full(20, 0.1), np.full(40, 1.0)]), index=dates)
    feats = pd.DataFrame({"mkt_vol": mkt_vol, "mkt_liq": 1.0}, index=dates)

    payload = meta_tune_regime_aware_allocation(
        ohlcv=None,  # not used because we pass features_by_window
        train_by_split=train_by_split,
        valid_by_split=valid_by_split,
        trading_days=252,
        score_metric="information_ratio",
        lambda_corr=0.0,
        l2=1e-8,
        turnover_lambda=0.0,
        max_weight=1.0,
        use_abs_corr=True,
        backend="auto",
        solver="",
        regime_min_days=5,
        param_lists={"mode": ["vol"], "window": [10], "buckets": [1, 2], "smoothing": [0.0]},
        max_combos=10,
        tune_metric="information_ratio",
        turnover_penalty=0.0,
        features_by_window={10: feats},
    )

    assert payload.get("enabled") is True
    best = payload.get("best_params") or {}
    assert int(best.get("buckets")) == 2

    results = [r for r in (payload.get("results") or []) if str(r.get("config_id")) != "base"]
    assert len(results) == 2

    # P2.28: stability diagnostics are present.
    assert "objective_split_std" in results[0]
    assert "objective_split_min" in results[0]

    # Buckets=2 should dominate buckets=1 on valid (positive returns in high-vol).
    by_buckets = {int(r.get("buckets")): float(r.get("valid_metric") or 0.0) for r in results}
    assert by_buckets[2] > by_buckets[1]
