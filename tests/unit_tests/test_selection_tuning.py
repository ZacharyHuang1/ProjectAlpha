import numpy as np
import pandas as pd

from agent.research.selection_tuning import build_valid_return_matrix, tune_diverse_selection


def _alpha_payload(alpha_id: str, dates: pd.DatetimeIndex, returns: np.ndarray):
    rows = [{"datetime": d.isoformat(), "net_return": float(r)} for d, r in zip(dates, returns)]
    return {"alpha_id": alpha_id, "backtest_results": {"walk_forward": {"valid_daily": rows}}}


def test_build_valid_return_matrix_shape() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10)
    a1 = _alpha_payload("a1", dates, np.ones(len(dates)) * 0.001)
    a2 = _alpha_payload("a2", dates, np.ones(len(dates)) * -0.001)

    mat = build_valid_return_matrix([a1, a2])
    assert mat.shape == (10, 2)
    assert set(mat.columns) == {"a1", "a2"}


def test_tune_diverse_selection_prefers_diversification_when_it_improves_ir() -> None:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2021-01-01", periods=252)

    # A: decent IR
    a = rng.normal(0.001, 0.01, size=len(dates))

    # B: almost identical to A (high correlation)
    b = a + rng.normal(0.0, 0.0002, size=len(dates))

    # C: lower standalone IR, but uncorrelated and helps the ensemble IR
    c = rng.normal(0.0007, 0.008, size=len(dates))

    alphas = [
        _alpha_payload("A", dates, a),
        _alpha_payload("B", dates, b),
        _alpha_payload("C", dates, c),
    ]

    tuned = tune_diverse_selection(
        alphas,
        top_k=2,
        candidate_pool_grid=[3],
        lambda_grid=[0.0, 0.8],
        metric="information_ratio",
        min_periods=60,
        trading_days=252,
        max_combos=10,
    )
    assert tuned.get("enabled") is True
    best = tuned.get("best") or {}
    sel = best.get("selected_alpha_ids") or []
    assert "A" in sel
    # With a correlation penalty, the second pick should prefer C over B.
    assert "C" in sel
