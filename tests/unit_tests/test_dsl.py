import pandas as pd
import numpy as np
import pytest

from agent.research.dsl import DSLValidationError, eval_dsl


def _toy_df(n_days: int = 10, n_inst: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    inst = [f"S{i:02d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([dates, inst], names=["datetime", "instrument"])
    rng = np.random.default_rng(0)
    close = pd.Series(rng.normal(100, 1, size=len(idx)), index=idx)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(10, 0.2, size=len(idx)),
        },
        index=idx,
    )
    return df


def test_eval_dsl_basic_shape() -> None:
    df = _toy_df()
    s = eval_dsl("zscore(ts_mean(returns(close, 1), 3))", df)
    assert isinstance(s, pd.Series)
    assert s.index.equals(df.index)


def test_eval_dsl_boolean_where() -> None:
    df = _toy_df()
    s = eval_dsl("where(close > delay(close, 1), 1, 0)", df)
    assert set(s.dropna().unique()).issubset({0, 1})


def test_eval_dsl_rejects_bad_name() -> None:
    df = _toy_df()
    with pytest.raises(DSLValidationError):
        eval_dsl("os.system('rm -rf /')", df)


def test_eval_dsl_new_ops() -> None:
    df = _toy_df()
    s = eval_dsl("zscore(safe_div(ts_mean(returns(close, 1), 5), ts_std(returns(close, 1), 5)))", df)
    assert isinstance(s, pd.Series)


def test_eval_dsl_group_neutralization_with_extras() -> None:
    df = _toy_df(n_days=12, n_inst=4)
    inst = df.index.get_level_values("instrument").astype(str)
    # Deterministic, two-group labeling.
    sector = pd.Series(["A" if int(x[1:]) % 2 == 0 else "B" for x in inst], index=df.index, name="sector")

    s1 = eval_dsl("cs_demean_group(close, sector)", df, extras={"sector": sector})
    tmp1 = pd.DataFrame({"x": s1, "sector": sector}).dropna()
    means1 = tmp1.groupby([tmp1.index.get_level_values("datetime"), "sector"])["x"].mean()
    assert np.allclose(means1.to_numpy(dtype=float), 0.0, atol=1e-10)

    s2 = eval_dsl("cs_neutralize_group(close, sector, log1p(volume))", df, extras={"sector": sector})
    tmp2 = pd.DataFrame({"x": s2, "sector": sector}).dropna()
    means2 = tmp2.groupby([tmp2.index.get_level_values("datetime"), "sector"])["x"].mean()
    assert np.allclose(means2.to_numpy(dtype=float), 0.0, atol=1e-10)
