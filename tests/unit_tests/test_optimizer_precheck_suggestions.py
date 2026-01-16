import pandas as pd

from agent.research.optimizer import qp_feasibility_precheck


def test_qp_precheck_suggests_relaxing_max_abs_weight() -> None:
    w_target = pd.Series(0.0, index=["A", "B", "C", "D"])
    out = qp_feasibility_precheck(
        long_names=["A", "B"],
        short_names=["C", "D"],
        w_target=w_target,
        gross_long=0.5,
        gross_short=0.5,
        max_abs_weight=0.10,
        turnover_cap=0.0,
        exposures=None,
        enforce_exposure_neutrality=False,
    )
    assert out.get("passed") is False
    assert "cap_long_too_tight" in (out.get("reasons") or [])
    sugg = list(out.get("suggestions") or [])
    m = [s for s in sugg if isinstance(s, dict) and s.get("parameter") == "max_abs_weight"]
    assert m
    assert float(m[0].get("suggested_min") or 0.0) >= 0.25


def test_qp_precheck_suggests_relaxing_turnover_cap() -> None:
    w_target = pd.Series(0.0, index=["A", "B", "C", "D"])
    out = qp_feasibility_precheck(
        long_names=["A", "B"],
        short_names=["C", "D"],
        w_target=w_target,
        gross_long=0.5,
        gross_short=0.5,
        max_abs_weight=1.0,
        turnover_cap=0.10,
        exposures=None,
        enforce_exposure_neutrality=False,
    )
    assert out.get("passed") is False
    assert "turnover_cap_too_small" in (out.get("reasons") or [])
    sugg = list(out.get("suggestions") or [])
    m = [s for s in sugg if isinstance(s, dict) and s.get("parameter") == "optimizer_turnover_cap"]
    assert m
    assert float(m[0].get("suggested_min") or 0.0) >= 0.5
