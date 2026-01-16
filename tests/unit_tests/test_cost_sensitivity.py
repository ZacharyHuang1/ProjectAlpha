import pandas as pd

from agent.research.cost_sensitivity import compute_cost_sensitivity
from agent.research.portfolio_backtest import BacktestConfig


def test_cost_sensitivity_break_even_linear_cost() -> None:
    # A simple deterministic path:
    # - gross_return = 10 bps/day
    # - turnover = 10%/day
    # With no other costs, break-even linear_cost_bps should be 100 bps.
    dates = pd.bdate_range("2020-01-01", periods=30)
    daily = [
        {
            "datetime": d.isoformat(),
            "gross_return": 0.0010,
            "turnover": 0.10,
            "impact_unit": 0.0,
            "gross_short": 0.0,
            "borrow": 0.0,
        }
        for d in dates
    ]

    cfg = BacktestConfig(
        commission_bps=0.0,
        slippage_bps=0.0,
        half_spread_bps=0.0,
        impact_bps=0.0,
        borrow_bps=0.0,
        trading_days=252,
    )

    cs = compute_cost_sensitivity(
        daily,
        base_cfg=cfg,
        borrow_rates_present=False,
        linear_bps_grid=[0.0, 50.0, 100.0, 150.0],
        half_spread_bps_grid=[0.0],
        impact_bps_grid=[0.0],
        borrow_bps_grid=[0.0],
        borrow_mult_grid=[1.0],
    )

    assert cs.get("enabled") is True
    be = {r["parameter"]: r for r in cs.get("break_even") or []}
    assert "linear_cost_bps" in be
    assert abs(float(be["linear_cost_bps"]["break_even"]) - 100.0) < 1e-6
