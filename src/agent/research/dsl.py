"""agent.research.dsl

A small, safe operator DSL for factor definitions.

Design goals:
- Avoid executing arbitrary Python via exec().
- Keep syntax simple: Python-like expressions + function calls.
- Restrict the AST surface area and operators via allow-lists.

Example:
    cs_rank(ts_mean(returns(close, 1), 20)) - cs_rank(ts_mean(returns(open, 1), 20))
"""

from __future__ import annotations

import ast
import math
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


class DSLValidationError(ValueError):
    """Raised when a DSL expression violates validation rules."""


# ----------------------------
# Limits (keep P0 predictable)
# ----------------------------
_MAX_AST_NODES = 300
_MAX_WINDOW = 1000
_MAX_LAG = 1000


# ----------------------------
# Allowed symbols
# ----------------------------
_BASE_COLUMNS = ("open", "high", "low", "close", "volume")
# Optional metadata fields that can be injected by the pipeline (e.g., sector maps).
_META_FIELDS = ("sector",)

_ALLOWED_COLUMNS = _BASE_COLUMNS + _META_FIELDS


def allowed_columns() -> Tuple[str, ...]:
    return _ALLOWED_COLUMNS


def _require_series(x: Any, ctx: str) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame) and x.shape[1] == 1:
        return x.iloc[:, 0]
    raise TypeError(f"{ctx} expects a pandas Series (or single-col DataFrame), got {type(x).__name__}")


def _groupby_datetime(s: pd.Series):
    if isinstance(s.index, pd.MultiIndex) and "datetime" in s.index.names:
        return s.groupby(level="datetime", group_keys=False)
    return s.groupby(s.index, group_keys=False)


def _groupby_instrument(s: pd.Series):
    if isinstance(s.index, pd.MultiIndex) and "instrument" in s.index.names:
        return s.groupby(level="instrument", group_keys=False)
    return s.groupby(s.index, group_keys=False)


def _check_window(w: int, ctx: str) -> int:
    w = int(w)
    if w <= 0:
        raise ValueError(f"{ctx}: window must be > 0")
    if w > _MAX_WINDOW:
        raise ValueError(f"{ctx}: window too large (>{_MAX_WINDOW})")
    return w


def _check_lag(lag: int, ctx: str) -> int:
    lag = int(lag)
    if lag < 0:
        raise ValueError(f"{ctx}: lag must be >= 0")
    if lag > _MAX_LAG:
        raise ValueError(f"{ctx}: lag too large (>{_MAX_LAG})")
    return lag


# ----------------------------
# Time-series operators (per instrument)
# ----------------------------
def delay(x: pd.Series, lag: int) -> pd.Series:
    s = _require_series(x, "delay(x)")
    k = _check_lag(lag, "delay")
    return _groupby_instrument(s).shift(k)


def returns(x: pd.Series, lag: int = 1) -> pd.Series:
    s = _require_series(x, "returns(x)")
    k = _check_lag(lag, "returns")
    if k == 0:
        return s * 0.0
    return (s / delay(s, k)) - 1.0


def ts_delta(x: pd.Series, lag: int = 1) -> pd.Series:
    s = _require_series(x, "ts_delta(x)")
    k = _check_lag(lag, "ts_delta")
    if k == 0:
        return s * 0.0
    return s - delay(s, k)


def ts_mean(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_mean(x)")
    w = _check_window(window, "ts_mean")
    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).mean())


def ts_sum(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_sum(x)")
    w = _check_window(window, "ts_sum")
    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).sum())


def ts_std(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_std(x)")
    w = _check_window(window, "ts_std")
    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).std(ddof=0))


def ts_zscore(x: pd.Series, window: int) -> pd.Series:
    """Time-series z-score per instrument.

    Returns (x - rolling_mean) / rolling_std.
    """

    s = _require_series(x, "ts_zscore(x)")
    w = _check_window(window, "ts_zscore")
    mu = ts_mean(s, w)
    sd = ts_std(s, w)
    return (s - mu) / sd.replace(0.0, np.nan)


def rel_volume(x: pd.Series, window: int) -> pd.Series:
    """Relative volume per instrument.

    This makes volume dimensionless via volume / rolling_mean(volume).
    """

    s = _require_series(x, "rel_volume(x)")
    w = _check_window(window, "rel_volume")
    return safe_div(s, ts_mean(s, w))


def ts_min(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_min(x)")
    w = _check_window(window, "ts_min")
    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).min())


def ts_max(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_max(x)")
    w = _check_window(window, "ts_max")
    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).max())


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_rank(x)")
    w = _check_window(window, "ts_rank")

    def _rank_last(arr: np.ndarray) -> float:
        a = pd.Series(arr)
        return float(a.rank(pct=True).iloc[-1])

    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).apply(_rank_last, raw=True))


def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    a = _require_series(x, "ts_corr(x)")
    b = _require_series(y, "ts_corr(y)")
    w = _check_window(window, "ts_corr")
    return _groupby_instrument(a).apply(lambda g: g.rolling(w, min_periods=w).corr(b.loc[g.index]))


def ts_argmax(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_argmax(x)")
    w = _check_window(window, "ts_argmax")

    def _argmax(arr: np.ndarray) -> float:
        return float(np.argmax(arr))

    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).apply(_argmax, raw=True))


def ts_argmin(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_argmin(x)")
    w = _check_window(window, "ts_argmin")

    def _argmin(arr: np.ndarray) -> float:
        return float(np.argmin(arr))

    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).apply(_argmin, raw=True))


def ts_decay_linear(x: pd.Series, window: int) -> pd.Series:
    s = _require_series(x, "ts_decay_linear(x)")
    w = _check_window(window, "ts_decay_linear")
    weights = np.arange(1.0, float(w) + 1.0, dtype=float)
    denom = float(weights.sum())

    def _wma(arr: np.ndarray) -> float:
        return float(np.dot(arr, weights) / denom)

    return _groupby_instrument(s).apply(lambda g: g.rolling(w, min_periods=w).apply(_wma, raw=True))


def ewm_mean(x: pd.Series, span: int) -> pd.Series:
    s = _require_series(x, "ewm_mean(x)")
    sp = _check_window(span, "ewm_mean")
    return _groupby_instrument(s).apply(lambda g: g.ewm(span=sp, adjust=False).mean())


def ewm_std(x: pd.Series, span: int) -> pd.Series:
    s = _require_series(x, "ewm_std(x)")
    sp = _check_window(span, "ewm_std")
    return _groupby_instrument(s).apply(lambda g: g.ewm(span=sp, adjust=False).std(bias=True))


# Aliases
def sma(x: pd.Series, window: int) -> pd.Series:
    return ts_mean(x, window)


# ----------------------------
# Cross-sectional operators (per datetime)
# ----------------------------
def cs_rank(x: pd.Series) -> pd.Series:
    s = _require_series(x, "cs_rank(x)")
    return _groupby_datetime(s).rank(pct=True)


def cs_demean(x: pd.Series) -> pd.Series:
    s = _require_series(x, "cs_demean(x)")
    g = _groupby_datetime(s)
    mu = g.transform("mean")
    return s - mu


def cs_demean_group(x: pd.Series, group: pd.Series) -> pd.Series:
    """Cross-sectional group de-meaning per datetime.

    Typical usage: remove sector effects before ranking/z-scoring.
        cs_demean_group(signal, sector)
    """

    s = _require_series(x, "cs_demean_group(x)")
    grp = _require_series(group, "cs_demean_group(group)")

    tmp = pd.DataFrame({"x": s, "g": grp})

    def _demean(d: pd.DataFrame) -> pd.Series:
        g = d["g"].astype(object)
        g = g.where(pd.notna(g), "UNKNOWN")
        mu = d["x"].groupby(g).transform("mean")
        return d["x"] - mu

    return tmp.groupby(level="datetime", group_keys=False).apply(_demean)


def zscore(x: pd.Series) -> pd.Series:
    s = _require_series(x, "zscore(x)")
    g = _groupby_datetime(s)
    mu = g.transform("mean")
    sd = g.transform(lambda v: float(v.std(ddof=0)))
    return (s - mu) / sd.replace(0.0, np.nan)


def cs_neutralize(x: pd.Series, *controls: pd.Series) -> pd.Series:
    """Cross-sectional residualization per datetime via OLS.

    Returns residuals of: x ~ 1 + controls (fit independently per date).
    """

    y = _require_series(x, "cs_neutralize(x)")
    ctrls = [_require_series(c, "cs_neutralize(control)") for c in controls if c is not None]
    if not ctrls:
        return cs_demean(y)

    tmp = pd.DataFrame({"y": y})
    for i, c in enumerate(ctrls):
        tmp[f"c{i}"] = c

    def _resid(g: pd.DataFrame) -> pd.Series:
        yv = g["y"].to_numpy(dtype=float)
        cols = [c for c in g.columns if c.startswith("c")]
        X = g[cols].to_numpy(dtype=float) if cols else np.empty((len(g), 0), dtype=float)

        mask = np.isfinite(yv)
        if X.size:
            mask = mask & np.isfinite(X).all(axis=1)

        n = int(mask.sum())
        p = int(X.shape[1]) if X.ndim == 2 else 0
        if n <= p + 1:
            return pd.Series(np.nan, index=g.index)

        Xd = np.column_stack([np.ones(n, dtype=float), X[mask]])
        beta, *_ = np.linalg.lstsq(Xd, yv[mask], rcond=None)
        resid = yv[mask] - Xd @ beta

        out = np.full(len(yv), np.nan, dtype=float)
        out[mask] = resid
        return pd.Series(out, index=g.index)

    return tmp.groupby(level="datetime", group_keys=False).apply(_resid)


def cs_neutralize_group(x: pd.Series, group: pd.Series, *controls: pd.Series) -> pd.Series:
    """Cross-sectional residualization per datetime with group fixed effects.

    This removes group means (e.g., sectors) and then neutralizes w.r.t. numeric controls.
    It is equivalent to a per-date fixed-effect regression.
    """

    y = _require_series(x, "cs_neutralize_group(x)")
    grp = _require_series(group, "cs_neutralize_group(group)")
    ctrls = [_require_series(c, "cs_neutralize_group(control)") for c in controls if c is not None]

    tmp = pd.DataFrame({"y": y, "g": grp})
    for i, c in enumerate(ctrls):
        tmp[f"c{i}"] = c

    def _resid(d: pd.DataFrame) -> pd.Series:
        yv = d["y"].to_numpy(dtype=float)
        g = d["g"].astype(object)
        g = g.where(pd.notna(g), "UNKNOWN").astype(str)

        cols = [c for c in d.columns if c.startswith("c")]
        X = d[cols].to_numpy(dtype=float) if cols else np.empty((len(d), 0), dtype=float)

        mask = np.isfinite(yv)
        if X.size:
            mask = mask & np.isfinite(X).all(axis=1)

        n = int(mask.sum())
        p = int(X.shape[1]) if X.ndim == 2 else 0
        if n <= p + 1:
            return pd.Series(np.nan, index=d.index)

        g_m = g.to_numpy()[mask]
        y_m = yv[mask]

        # Within-group de-meaning (fixed effects removal).
        y_dm = y_m - pd.Series(y_m).groupby(g_m).transform("mean").to_numpy()

        if p == 0:
            resid = y_dm
        else:
            X_m = X[mask]
            X_dm = np.empty_like(X_m)
            for j in range(p):
                col = X_m[:, j]
                mu = pd.Series(col).groupby(g_m).transform("mean").to_numpy()
                X_dm[:, j] = col - mu

            if n <= p:
                return pd.Series(np.nan, index=d.index)

            beta, *_ = np.linalg.lstsq(X_dm, y_dm, rcond=None)
            resid = y_dm - X_dm @ beta

        out = np.full(len(d), np.nan, dtype=float)
        out[mask] = resid
        return pd.Series(out, index=d.index)

    return tmp.groupby(level="datetime", group_keys=False).apply(_resid)


def winsorize(x: pd.Series, limit: float = 0.01) -> pd.Series:
    s = _require_series(x, "winsorize(x)")
    lim = float(limit)
    if lim < 0.0 or lim >= 0.5:
        raise ValueError("winsorize: limit must be in [0, 0.5)")
    g = _groupby_datetime(s)

    def _clip(v: pd.Series) -> pd.Series:
        lo = float(v.quantile(lim))
        hi = float(v.quantile(1.0 - lim))
        return v.clip(lo, hi)

    return g.apply(_clip)


def scale(x: pd.Series) -> pd.Series:
    s = _require_series(x, "scale(x)")
    g = _groupby_datetime(s)
    denom = g.transform(lambda v: float(v.abs().sum()))
    denom = denom.replace(0.0, np.nan)
    return s / denom


# Alias
def rank(x: pd.Series) -> pd.Series:
    return cs_rank(x)


# ----------------------------
# Elementwise helpers
# ----------------------------
def clip(x: pd.Series, lo: float, hi: float) -> pd.Series:
    s = _require_series(x, "clip(x)")
    return s.clip(float(lo), float(hi))


def log(x: pd.Series) -> pd.Series:
    s = _require_series(x, "log(x)")
    return np.log(s)


def log1p(x: pd.Series) -> pd.Series:
    s = _require_series(x, "log1p(x)")
    return np.log1p(s)


def sqrt(x: pd.Series) -> pd.Series:
    s = _require_series(x, "sqrt(x)")
    return np.sqrt(s)


def abs_(x: pd.Series) -> pd.Series:
    s = _require_series(x, "abs(x)")
    return s.abs()


def sign(x: pd.Series) -> pd.Series:
    s = _require_series(x, "sign(x)")
    return np.sign(s)


def where(cond: pd.Series, a: Any, b: Any) -> pd.Series:
    c = _require_series(cond, "where(cond)")
    # a/b can be scalars or Series
    return pd.Series(np.where(c, a, b), index=c.index)


def maximum(a: Any, b: Any) -> pd.Series:
    sa = _require_series(a, "maximum(a)") if not np.isscalar(a) else a
    sb = _require_series(b, "maximum(b)") if not np.isscalar(b) else b
    if isinstance(sa, pd.Series) and isinstance(sb, pd.Series):
        return np.maximum(sa, sb)
    if isinstance(sa, pd.Series):
        return np.maximum(sa, float(sb))
    if isinstance(sb, pd.Series):
        return np.maximum(float(sa), sb)
    return pd.Series(np.maximum(float(sa), float(sb)))


def minimum(a: Any, b: Any) -> pd.Series:
    sa = _require_series(a, "minimum(a)") if not np.isscalar(a) else a
    sb = _require_series(b, "minimum(b)") if not np.isscalar(b) else b
    if isinstance(sa, pd.Series) and isinstance(sb, pd.Series):
        return np.minimum(sa, sb)
    if isinstance(sa, pd.Series):
        return np.minimum(sa, float(sb))
    if isinstance(sb, pd.Series):
        return np.minimum(float(sa), sb)
    return pd.Series(np.minimum(float(sa), float(sb)))


def safe_div(a: Any, b: Any, eps: float = 1e-12) -> pd.Series:
    sa = _require_series(a, "safe_div(a)") if not np.isscalar(a) else a
    sb = _require_series(b, "safe_div(b)") if not np.isscalar(b) else b
    e = float(eps)
    if isinstance(sa, pd.Series) and isinstance(sb, pd.Series):
        return sa / (sb.abs() + e)
    if isinstance(sa, pd.Series):
        return sa / (abs(float(sb)) + e)
    if isinstance(sb, pd.Series):
        return float(sa) / (sb.abs() + e)
    return pd.Series(float(sa) / (abs(float(sb)) + e))


# ----------------------------
# DSL evaluation (AST allow-list)
# ----------------------------
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_UNARYOPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
    ast.Not: lambda a: ~a if isinstance(a, (pd.Series, pd.DataFrame)) else (not a),
}

_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


_ALLOWED_FUNCS: Dict[str, Callable[..., Any]] = {
    # time-series
    "delay": delay,
    "returns": returns,
    "ts_delta": ts_delta,
    "ts_mean": ts_mean,
    "ts_sum": ts_sum,
    "ts_std": ts_std,
    "ts_zscore": ts_zscore,
    "rel_volume": rel_volume,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_rank": ts_rank,
    "ts_corr": ts_corr,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "ts_decay_linear": ts_decay_linear,
    "ewm_mean": ewm_mean,
    "ewm_std": ewm_std,
    "sma": sma,
    # cross-sectional
    "cs_rank": cs_rank,
    "rank": rank,
    "cs_demean": cs_demean,
    "cs_demean_group": cs_demean_group,
    "zscore": zscore,
    "cs_neutralize": cs_neutralize,
    "cs_neutralize_group": cs_neutralize_group,
    "winsorize": winsorize,
    "scale": scale,
    # elementwise
    "clip": clip,
    "log": log,
    "log1p": log1p,
    "sqrt": sqrt,
    "abs": abs_,
    "abs_": abs_,
    "sign": sign,
    "where": where,
    "maximum": maximum,
    "minimum": minimum,
    "safe_div": safe_div,
}


def allowed_functions() -> Tuple[str, ...]:
    return tuple(sorted(_ALLOWED_FUNCS.keys()))


def _validate_ast(node: ast.AST) -> None:
    nodes = list(ast.walk(node))
    if len(nodes) > _MAX_AST_NODES:
        raise DSLValidationError("DSL expression too complex")

    for n in nodes:
        if isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Tuple, ast.Set)):
            raise DSLValidationError(f"DSL forbids node: {type(n).__name__}")
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.While, ast.For, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
            raise DSLValidationError(f"DSL forbids statement node: {type(n).__name__}")

        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise DSLValidationError("Only simple function calls are allowed")
            if n.func.id not in _ALLOWED_FUNCS:
                raise DSLValidationError(f"Function not allowed: {n.func.id}")

        if isinstance(n, ast.Name):
            if n.id not in _ALLOWED_COLUMNS and n.id not in _ALLOWED_FUNCS and n.id not in {"True", "False", "None"}:
                raise DSLValidationError(f"Name not allowed: {n.id}")

        if isinstance(n, ast.BinOp) and type(n.op) not in _BINOPS:
            raise DSLValidationError(f"Binary operator not allowed: {type(n.op).__name__}")
        if isinstance(n, ast.UnaryOp) and type(n.op) not in _UNARYOPS:
            raise DSLValidationError(f"Unary operator not allowed: {type(n.op).__name__}")

        if isinstance(n, ast.Compare):
            if len(n.ops) != 1 or len(n.comparators) != 1:
                raise DSLValidationError("Chained comparisons are not supported")
            for op in n.ops:
                if type(op) not in _CMPOPS:
                    raise DSLValidationError(f"Comparison operator not allowed: {type(op).__name__}")

        if isinstance(n, ast.BoolOp) and type(n.op) not in (ast.And, ast.Or):
            raise DSLValidationError(f"Boolean operator not allowed: {type(n.op).__name__}")

        if isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)):
            raise DSLValidationError("String constants are not allowed")


@lru_cache(maxsize=1024)
def _parse_and_validate(expr: str) -> ast.Expression:
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree)
    return tree


def _eval_node(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return env[node.id]

    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS[type(node.op)]
        return op(_eval_node(node.operand, env))

    if isinstance(node, ast.BinOp):
        op = _BINOPS[type(node.op)]
        return op(_eval_node(node.left, env), _eval_node(node.right, env))

    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            out = vals[0]
            for v in vals[1:]:
                out = out & v
            return out
        if isinstance(node.op, ast.Or):
            out = vals[0]
            for v in vals[1:]:
                out = out | v
            return out
        raise DSLValidationError("Unsupported BoolOp")

    if isinstance(node, ast.Compare):
        op = _CMPOPS[type(node.ops[0])]
        return op(_eval_node(node.left, env), _eval_node(node.comparators[0], env))

    if isinstance(node, ast.Call):
        fn = env[node.func.id]
        args = [_eval_node(a, env) for a in node.args]
        return fn(*args)

    raise DSLValidationError(f"Unsupported node: {type(node).__name__}")


def eval_dsl(expr: str, df: pd.DataFrame, *, extras: Dict[str, Any] | None = None) -> pd.Series:
    """Evaluate a DSL expression against an OHLCV dataframe.

    The runtime may inject optional metadata fields via `extras` (e.g., sector labels).
    """
    if not isinstance(expr, str) or not expr.strip():
        raise DSLValidationError("Empty DSL expression")

    tree = _parse_and_validate(expr)
    env: Dict[str, Any] = {k: df[k] for k in _BASE_COLUMNS}

    # Optional fields may live either in the dataframe or in `extras`.
    for k in _META_FIELDS:
        if k in df.columns:
            env[k] = df[k]

    if extras:
        for k, v in extras.items():
            if k in _BASE_COLUMNS or k in _ALLOWED_FUNCS:
                raise DSLValidationError(f"extras may not override reserved name: {k}")
            if k not in _ALLOWED_COLUMNS:
                raise DSLValidationError(f"extras contains unsupported name: {k}")
            env[k] = v
    env.update(_ALLOWED_FUNCS)

    try:
        out = _eval_node(tree, env)
    except KeyError as e:
        name = str(e.args[0]) if e.args else "<unknown>"
        raise DSLValidationError(f"Missing field: {name}") from None
    return _require_series(out, "dsl output")


# ----------------------------
# DSL linting / auto-fixes
# ----------------------------

def critique_dsl(expr: str) -> Dict[str, Any]:
    """Heuristic checks for common DSL pitfalls.

    This is intentionally lightweight: it does not attempt full symbolic analysis.
    """

    warnings: List[str] = []
    e = (expr or "").replace(" ", "")

    # Mixing raw volume and returns often creates a size/liquidity proxy.
    if "returns(" in e and "volume" in e and "*" in e:
        normalized = any(
            tok in e
            for tok in [
                "ts_rank(volume",
                "rel_volume(volume",
                "ts_zscore(",
                "zscore(log1p(volume",
                "log1p(volume)",
            ]
        )
        if ("ts_mean(volume" in e or "*volume" in e or "volume*" in e) and not normalized:
            warnings.append(
                "Potential scale issue: returns multiplied by raw volume. Prefer ts_rank(volume,w), "
                "rel_volume(volume,w), or ts_zscore(log1p(volume),w)."
            )

    if "log(volume" in e and "log1p(volume" not in e:
        warnings.append("log(volume) may produce -inf when volume==0. Prefer log1p(volume).")

    return {"warnings": warnings}


def autofix_dsl(expr: str) -> Tuple[str, List[str]]:
    """Best-effort auto-fix for a small set of known bad patterns."""

    import re

    out = (expr or "").strip()
    fixes: List[str] = []
    e = out.replace(" ", "")

    # Replace ts_mean(volume,w) inside a returns*volume product with a dimensionless proxy.
    if "returns(" in e and "ts_mean(volume" in e and "*" in e:
        if "ts_zscore" not in e and "ts_rank(volume" not in e and "rel_volume(volume" not in e:
            def _repl(m: re.Match) -> str:
                w = m.group(1)
                return f"ts_zscore(log1p(volume), {w})"

            new = re.sub(r"ts_mean\s*\(\s*volume\s*,\s*(\d+)\s*\)", _repl, out)
            if new != out:
                out = new
                fixes.append("Replaced ts_mean(volume,w) with ts_zscore(log1p(volume),w) to reduce scale dominance.")

    return out, fixes
