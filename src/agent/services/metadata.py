"""agent.services.metadata

Small helpers to load optional metadata used by research backtests.
"""

from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def _read_table(path: str) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def load_sector_map(path: str) -> Dict[str, str]:
    """Load a simple instrument->sector mapping from CSV/Parquet.

    Expected columns: instrument, sector
    """

    df = _read_table(path)
    if "instrument" not in df.columns or "sector" not in df.columns:
        raise ValueError("sector map must contain columns: instrument, sector")

    out: Dict[str, str] = {}
    for inst, sec in zip(df["instrument"].astype(str), df["sector"].astype(str)):
        inst = inst.strip()
        sec = sec.strip()
        if inst:
            out[inst] = sec
    return out


def load_hard_to_borrow(path: str) -> Set[str]:
    """Load a hard-to-borrow list or a shortability map.

    Accepted formats:
      - instrument                      (treated as hard-to-borrow)
      - instrument, shortable/borrowable (False/0 => hard-to-borrow)
    """

    df = _read_table(path)
    if "instrument" not in df.columns:
        raise ValueError("hard-to-borrow file must contain column: instrument")

    inst = df["instrument"].astype(str).str.strip()
    flag_col = None
    for c in ("shortable", "borrowable", "is_shortable", "is_borrowable"):
        if c in df.columns:
            flag_col = c
            break

    if flag_col is None:
        htb = set(inst.tolist())
        return {x for x in htb if x}

    s = df[flag_col].astype(str).str.strip().str.lower()
    ok = s.isin({"1", "true", "t", "yes", "y"})
    htb = set(inst[~ok].tolist())
    return {x for x in htb if x}


def load_borrow_rates(path: str) -> pd.Series:
    """Load per-instrument borrow rates (annualized) in basis points.

    Expected columns:
      - datetime (or date)
      - instrument
      - borrow_bps (or borrow_rate_bps or rate_bps)
    """

    df = _read_table(path)
    dt_col = None
    for c in ("datetime", "date", "dt"):
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        raise ValueError("borrow rates file must contain a datetime/date column")

    if "instrument" not in df.columns:
        raise ValueError("borrow rates file must contain column: instrument")

    rate_col = None
    for c in ("borrow_bps", "borrow_rate_bps", "rate_bps"):
        if c in df.columns:
            rate_col = c
            break
    if rate_col is None:
        raise ValueError("borrow rates file must contain a borrow_bps column (in annualized bps)")

    df = df[[dt_col, "instrument", rate_col]].copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df["instrument"] = df["instrument"].astype(str).str.strip()
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")

    df = df.dropna(subset=[dt_col, "instrument", rate_col])
    idx = pd.MultiIndex.from_arrays([df[dt_col], df["instrument"]], names=["datetime", "instrument"])
    s = pd.Series(df[rate_col].astype(float).to_numpy(), index=idx, name="borrow_bps").sort_index()
    return s
