import pandas as pd
import numpy as np

from agent.research.factor_runner import run_factors


def _toy_df(n_days: int = 30, n_inst: int = 10) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    rng = np.random.default_rng(2)
    close = pd.Series(rng.normal(50, 1, size=len(idx)), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(10, 0.3, size=len(idx)),
        },
        index=idx,
    )


def test_run_factors_dsl() -> None:
    df = _toy_df()
    alphas = [
        {
            "alpha_id": "A001",
            "dsl": "zscore(ts_mean(returns(close, 1), 5))",
        }
    ]
    res = run_factors(df, alphas, prefer_dsl=True, allow_python_exec=False)
    assert len(res) == 1
    assert res[0].error is None
    assert res[0].values is not None
    assert "A001" in res[0].values.columns


def test_run_factors_dsl_with_sector_extras() -> None:
    df = _toy_df(n_days=25, n_inst=8)
    inst = df.index.get_level_values("instrument").astype(str)
    sector = pd.Series(["A" if int(x[1:]) % 2 == 0 else "B" for x in inst], index=df.index, name="sector")
    alphas = [{"alpha_id": "A002", "dsl": "cs_demean_group(returns(close, 1), sector)"}]
    res = run_factors(df, alphas, prefer_dsl=True, allow_python_exec=False, dsl_extra_env={"sector": sector})
    assert res[0].error is None
    assert res[0].values is not None
    assert "A002" in res[0].values.columns
