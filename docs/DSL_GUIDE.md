# DSL Guide

This repo prefers a **small operator DSL** for factor definitions instead of executing
arbitrary Python via `exec()`.

## Why a DSL?

- Easier to validate (AST allow-list + complexity limits).
- Avoids imports / file IO / networking by design.
- Keeps factors reproducible (same operator set, same data contract).

Implementation: `src/agent/research/dsl.py`

## Data contract

The DSL evaluates against an OHLCV panel:

- Index: `MultiIndex(datetime, instrument)`
- Columns: `open`, `high`, `low`, `close`, `volume`

## Syntax

The DSL is a Python-like expression made of:

- Numbers: `1`, `0.5`, `20`
- Field names: `close`, `volume`, ...
- Function calls: `ts_mean(...)`, `cs_rank(...)`
- Operators: `+ - * / ** %`
- Comparisons / booleans: `> < == and or`
- Parentheses

Forbidden:

- Attribute access (`x.y`)
- Indexing (`x[0]`)
- Imports, loops, lambdas, strings

## Operator overview

Time-series (per instrument):

- `delay(x, lag)`
- `returns(x, lag=1)`
- `ts_delta(x, lag=1)`
- `ts_mean / ts_sum / ts_std / ts_min / ts_max (x, window)`
- `ts_zscore(x, window)`
- `rel_volume(x, window)`
- `ts_rank(x, window)`
- `ts_corr(x, y, window)`
- `ts_argmax / ts_argmin (x, window)`
- `ts_decay_linear(x, window)`
- `ewm_mean / ewm_std (x, span)`
- `sma(x, window)` (alias for `ts_mean`)

Cross-sectional (per datetime):

- `cs_rank(x)` / `rank(x)`
- `cs_demean(x)`
- `zscore(x)`
- `cs_neutralize(x, *controls)` (OLS residualization per date)
- `winsorize(x, limit=0.01)`
- `scale(x)`

Elementwise:

- `clip(x, lo, hi)`
- `log(x)`, `log1p(x)`, `sqrt(x)`, `abs(x)`, `sign(x)`
- `where(cond, a, b)`
- `maximum(a, b)`, `minimum(a, b)`
- `safe_div(a, b, eps=1e-12)`

## Examples

Simple momentum:

```text
zscore(ts_mean(returns(close, 1), 20))
```

Volume-conditioned momentum (normalized volume; avoids raw scale dominance):

```text
zscore(ts_mean(returns(close, 1), 20) * ts_zscore(log1p(volume), 20))
```

Volume-conditioned momentum with liquidity neutralization:

```text
zscore(cs_neutralize(
  ts_mean(returns(close, 1), 20) * ts_zscore(log1p(volume), 20),
  log1p(ts_mean(close * volume, 20))
))
```

Mean-reversion with winsorization:

```text
-zscore(winsorize(ts_delta(close, 1), 0.02))
```

Risk-scaled momentum (avoid divide-by-zero):

```text
zscore(safe_div(ts_mean(returns(close, 1), 20), ts_std(returns(close, 1), 20)))
```

## Legacy Python exec

For debugging/legacy runs only:

```bash
python main.py --allow-python-exec --no-prefer-dsl
```

This is **not recommended**. The default path is DSL-first.
