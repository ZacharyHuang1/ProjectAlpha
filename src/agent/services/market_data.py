"""agent.services.market_data

Small helpers for loading or generating OHLCV data for P0 evaluation.

Data contract:
- MultiIndex: (datetime, instrument)
- Columns: open, high, low, close, volume
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class MarketDataSpec:
    path: str = ""
    n_days: int = 252
    n_instruments: int = 50
    seed: int = 7


def validate_ohlcv(df: pd.DataFrame, *, required_columns: Iterable[str] = _REQUIRED_COLUMNS) -> pd.DataFrame:
    """Validate and lightly normalize an OHLCV panel.

    Rules (P0):
    - enforce MultiIndex(datetime, instrument)
    - enforce required OHLCV columns
    - coerce numeric dtypes, replace inf with NaN
    - minimal NaN policy: drop missing close; fill others with simple defaults
    """

    if df is None or df.empty:
        raise ValueError("OHLCV dataframe is empty")

    if not isinstance(df.index, pd.MultiIndex):
        if {"datetime", "instrument"}.issubset(df.columns):
            tmp = df.copy()
            tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce", utc=False)
            tmp["instrument"] = tmp["instrument"].astype(str)
            df = tmp.set_index(["datetime", "instrument"]).sort_index()
        else:
            raise ValueError("OHLCV must use MultiIndex or include datetime/instrument columns")

    if df.index.nlevels != 2:
        raise ValueError("OHLCV index must have 2 levels: (datetime, instrument)")

    if list(df.index.names) != ["datetime", "instrument"]:
        df = df.copy()
        df.index = df.index.set_names(["datetime", "instrument"])

    dt = pd.to_datetime(df.index.get_level_values("datetime"), errors="coerce", utc=False)
    inst = df.index.get_level_values("instrument").astype(str)
    df = df.copy()
    df.index = pd.MultiIndex.from_arrays([dt, inst], names=["datetime", "instrument"])

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV is missing required columns: {missing}")

    for c in required_columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.dropna(subset=["close"])

    if df["open"].isna().any():
        prev_close = df["close"].groupby(level="instrument").shift(1)
        df["open"] = df["open"].fillna(prev_close).fillna(df["close"])

    df["volume"] = df["volume"].fillna(0.0)
    df.loc[df["volume"] < 0, "volume"] = df.loc[df["volume"] < 0, "volume"].abs()

    if df["high"].isna().any():
        df["high"] = df["high"].fillna(np.maximum(df["open"], df["close"]))
    if df["low"].isna().any():
        df["low"] = df["low"].fillna(np.minimum(df["open"], df["close"]))

    df["high"] = np.maximum(df["high"], np.maximum(df["open"], df["close"]))
    df["low"] = np.minimum(df["low"], np.minimum(df["open"], df["close"]))

    return df


def generate_synthetic_ohlcv(
    *,
    n_days: int = 252,
    n_instruments: int = 50,
    seed: int = 7,
) -> pd.DataFrame:
    """Generate a small, reproducible OHLCV panel for demos/tests."""

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    instruments = [f"S{str(i).zfill(4)}" for i in range(n_instruments)]

    rets = rng.normal(loc=0.0002, scale=0.02, size=(n_days, n_instruments))
    base_prices = rng.uniform(20.0, 80.0, size=(1, n_instruments))
    close = base_prices * np.exp(np.cumsum(rets, axis=0))

    open_ = np.vstack([close[0], close[:-1]])
    intraday = rng.normal(loc=0.0, scale=0.01, size=(n_days, n_instruments))
    high = np.maximum(open_, close) * (1.0 + np.abs(intraday))
    low = np.minimum(open_, close) * (1.0 - np.abs(intraday))
    volume = rng.lognormal(mean=12.0, sigma=0.4, size=(n_days, n_instruments))

    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    out = pd.DataFrame(
        {
            "open": open_.reshape(-1),
            "high": high.reshape(-1),
            "low": low.reshape(-1),
            "close": close.reshape(-1),
            "volume": volume.reshape(-1),
        },
        index=idx,
    )
    return validate_ohlcv(out)


def load_ohlcv_data(path: str) -> pd.DataFrame:
    """Load OHLCV data from a local file.

    Supported formats:
    - .csv: expects columns [datetime, instrument, open, high, low, close, volume]
    - .parquet
    - .h5 / .hdf
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data path not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        df["datetime"] = pd.to_datetime(df.get("datetime"), errors="coerce", utc=False)
        if "instrument" in df.columns:
            df["instrument"] = df["instrument"].astype(str)
        return validate_ohlcv(df)

    if p.suffix.lower() == ".parquet":
        return validate_ohlcv(pd.read_parquet(p))

    if p.suffix.lower() in {".h5", ".hdf"}:
        return validate_ohlcv(pd.read_hdf(p))

    raise ValueError(f"Unsupported data format: {p.suffix}")


def get_market_data(spec: Optional[MarketDataSpec] = None) -> pd.DataFrame:
    """Load data if a path is provided; otherwise generate synthetic data."""

    spec = spec or MarketDataSpec()
    if spec.path:
        return load_ohlcv_data(spec.path)
    return generate_synthetic_ohlcv(n_days=spec.n_days, n_instruments=spec.n_instruments, seed=spec.seed)
