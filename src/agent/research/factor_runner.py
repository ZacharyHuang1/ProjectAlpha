"""agent.research.factor_runner

Run factor definitions against an OHLCV dataframe and return validated outputs.

Default path:
- Prefer a small operator DSL (see agent.research.dsl) to avoid exec().

Optional fallback:
- A restricted exec() sandbox can be enabled explicitly (for legacy code paths).
"""

from __future__ import annotations

from dataclasses import dataclass
import builtins
import math
import signal
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agent.research.code_safety import check_factor_code_safety
from agent.research.dsl import DSLValidationError, eval_dsl


_REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass
class FactorRunResult:
    """One factor run output + basic validation metadata."""

    alpha_id: str
    values: Optional[pd.DataFrame]
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


_ALLOWED_IMPORTS = {
    "pandas": pd,
    "numpy": np,
    "math": math,
}


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    if name in _ALLOWED_IMPORTS:
        return builtins.__import__(name, globals, locals, fromlist, level)
    raise ImportError(f"Import not allowed in sandbox: {name}")


class _PythonExecTimeout:
    def __init__(self, timeout_sec: float):
        self.timeout_sec = float(timeout_sec)
        self._old_handler = None

    def __enter__(self) -> None:
        # Only works on Unix main thread; best-effort guardrail.
        if self.timeout_sec <= 0:
            return
        if not hasattr(signal, "SIGALRM"):
            return
        self._old_handler = signal.getsignal(signal.SIGALRM)

        def _handler(signum, frame):  # type: ignore[no-untyped-def]
            raise TimeoutError("Python factor exec timed out")

        signal.signal(signal.SIGALRM, _handler)
        # Use setitimer for sub-second precision when available.
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, self.timeout_sec)
        else:
            signal.alarm(int(self.timeout_sec))

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if not hasattr(signal, "SIGALRM"):
            return
        try:
            if hasattr(signal, "setitimer"):
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            else:
                signal.alarm(0)
        finally:
            if self._old_handler is not None:
                signal.signal(signal.SIGALRM, self._old_handler)


def _safe_builtins() -> Dict[str, Any]:
    allowed = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "bool": bool,
        "print": print,
    }
    allowed["__import__"] = _safe_import
    return allowed


def _safe_identifier(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(s))


def _validate_input_df(df: pd.DataFrame, *, max_rows: int) -> Optional[str]:
    if not isinstance(df, pd.DataFrame):
        return "Input must be a pandas DataFrame"
    if not isinstance(df.index, pd.MultiIndex):
        return "Input index must be a MultiIndex (datetime, instrument)"
    if list(df.index.names) != ["datetime", "instrument"]:
        return "Input MultiIndex must be named ['datetime', 'instrument']"
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        return f"Missing required columns: {sorted(missing)}"
    if len(df) > max_rows:
        return f"Dataset too large: {len(df)} rows > max_rows={max_rows}"
    return None


def _ensure_single_column_df(x: Any, *, alpha_id: str, index: pd.Index) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        s = x
        s.name = alpha_id
        return s.reindex(index).to_frame()
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Factor output must be single-column")
        out = x.copy()
        out = out.reindex(index)
        out.columns = [alpha_id]
        return out
    raise TypeError(f"Factor output must be a Series or DataFrame, got {type(x).__name__}")


def _nan_ratio(df: pd.DataFrame) -> float:
    if df.empty:
        return 1.0
    total = float(df.shape[0] * df.shape[1])
    return float(df.isna().sum().sum()) / total


def _run_dsl(alpha_id: str, dsl_expr: str, df: pd.DataFrame, *, dsl_extra_env: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    s = eval_dsl(dsl_expr, df, extras=dsl_extra_env or None)
    return _ensure_single_column_df(s, alpha_id=alpha_id, index=df.index)


def _run_python(alpha_id: str, code: str, df: pd.DataFrame, *, timeout_sec: float = 0.0) -> pd.DataFrame:
    sandbox_globals: Dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "pd": pd,
        "np": np,
        "math": math,
        "__name__": "__factor__",
    }
    sandbox_locals: Dict[str, Any] = {}

    with _PythonExecTimeout(timeout_sec):
        exec(code, sandbox_globals, sandbox_locals)

    safe_id = _safe_identifier(alpha_id)
    fn_name_candidates = [f"calculate_{alpha_id}", f"calculate_{safe_id}"]

    namespace = {**sandbox_globals, **sandbox_locals}
    fn = None
    for n in fn_name_candidates:
        if n in namespace and callable(namespace[n]):
            fn = namespace[n]
            break

    if fn is None:
        # Fallback: pick the first calculate_* found
        for k, v in namespace.items():
            if isinstance(k, str) and k.startswith("calculate_") and callable(v):
                fn = v
                break

    if fn is None:
        raise ValueError("No calculate_* function found in factor code")

    out = fn(df)
    return _ensure_single_column_df(out, alpha_id=alpha_id, index=df.index)


def run_factors(
    df: pd.DataFrame,
    coded_alphas: List[Dict[str, Any]],
    *,
    prefer_dsl: bool = True,
    allow_python_exec: bool = False,
    dsl_extra_env: Optional[Dict[str, Any]] = None,
    max_nan_ratio: float = 0.95,
    max_rows: int = 2_000_000,
    enable_code_safety: bool = True,
    max_code_chars: int = 20_000,
    max_dsl_chars: int = 5_000,
    python_exec_timeout_sec: float = 2.0,
) -> List[FactorRunResult]:
    """Execute each factor definition against the input OHLCV dataframe."""

    err = _validate_input_df(df, max_rows=max_rows)
    if err:
        return [FactorRunResult(alpha_id="__all__", values=None, error=err)]

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    results: List[FactorRunResult] = []

    for raw in coded_alphas:
        alpha_id = raw.get("alpha_id") or raw.get("alphaID") or raw.get("id") or "unknown"
        dsl_expr = (raw.get("dsl") or "").strip()
        code = (raw.get("code") or "").strip()

        try:
            used = None
            out_df: Optional[pd.DataFrame] = None

            if prefer_dsl and dsl_expr:
                if len(dsl_expr) > max_dsl_chars:
                    raise ValueError(f"DSL too long: {len(dsl_expr)} chars > max_dsl_chars={max_dsl_chars}")
                out_df = _run_dsl(alpha_id, dsl_expr, df, dsl_extra_env=dsl_extra_env)
                used = "dsl"

            elif allow_python_exec and code:
                if len(code) > max_code_chars:
                    raise ValueError(f"Code too long: {len(code)} chars > max_code_chars={max_code_chars}")
                if enable_code_safety:
                    issue = check_factor_code_safety(code, allowed_imports=set(_ALLOWED_IMPORTS.keys()))
                    if issue:
                        raise ValueError(f"CodeSafetyError: {issue}")
                out_df = _run_python(alpha_id, code, df, timeout_sec=python_exec_timeout_sec)
                used = "python"

            else:
                raise ValueError("No executable factor definition (provide dsl, or enable python exec)")

            # Normalize to numeric and drop infinities
            out_df = out_df.apply(pd.to_numeric, errors="coerce")
            out_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            nan_ratio = _nan_ratio(out_df)
            if nan_ratio > max_nan_ratio:
                raise ValueError(f"Too many NaNs: nan_ratio={nan_ratio:.3f} > max_nan_ratio={max_nan_ratio}")

            results.append(
                FactorRunResult(
                    alpha_id=str(alpha_id),
                    values=out_df,
                    error=None,
                    meta={"nan_ratio": nan_ratio, "engine": used},
                )
            )

        except DSLValidationError as e:
            results.append(
                FactorRunResult(
                    alpha_id=str(alpha_id),
                    values=None,
                    error=f"DSLValidationError: {e}",
                )
            )
        except Exception as e:
            results.append(
                FactorRunResult(
                    alpha_id=str(alpha_id),
                    values=None,
                    error=f"ExecutionError: {e.__class__.__name__}: {e}",
                )
            )

    return results
