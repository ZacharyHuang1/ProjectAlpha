# P2.32 — DSL formula sanity (volume normalization + neutralization)

## Why

During offline runs (no OpenAI API key), the deterministic `alpha_coder_agent` stub produced a volume-weighted momentum factor:

```
zscore(ts_mean(returns(close, 1), 20) * ts_mean(volume, 20))
```

This mixes a dimensionless return with a scale-dependent `volume` level, which can unintentionally turn the factor into a **liquidity/size proxy** (large-cap / high-volume names dominate the cross-sectional ranking).

## What changed

### 1) New DSL operators

Added a few small, reusable primitives to make “volume-aware” formulas more principled:

- `log1p(x)` — safer than `log(x)` for non-negative series (avoids `-inf` when `x==0`).
- `ts_zscore(x, window)` — per-instrument rolling z-score: `(x - mean) / std`.
- `rel_volume(x, window)` — per-instrument relative level: `x / ts_mean(x, window)`.
- `cs_neutralize(x, *controls)` — per-date OLS residualization: removes linear exposure to one or more cross-sectional controls.

These are available to both the runtime DSL engine and the alpha-coder prompt allow-list.

### 2) Deterministic offline stub upgraded

When a seed alpha mentions volume, the offline stub now generates a **dimensionless** and **liquidity-neutralized** signal:

```
zscore(cs_neutralize(
  ts_mean(returns(close, 1), 20) * ts_zscore(log1p(volume), 20),
  log1p(ts_mean(close * volume, 20))
))
```

Notes:

- `ts_zscore(log1p(volume), 20)` captures *relative* volume surprises rather than absolute volume.
- `log1p(ts_mean(close * volume, 20))` is a rough liquidity proxy (dollar ADV) used as a neutralization control.

### 3) DSL lint + auto-fix hooks

- `critique_dsl(expr)` adds lightweight warnings for common pitfalls (e.g. returns × raw volume).
- `autofix_dsl(expr)` applies a conservative rewrite for a known bad pattern:
  - `ts_mean(volume, w)` in a returns×volume product → `ts_zscore(log1p(volume), w)`

`alpha_coder_agent` now records:

- `dsl_original` (if changed)
- `dsl_fixes` (list of fix messages)
- `dsl_warnings` (list of warnings)

The run `REPORT.md` surfaces these fields for the best alpha to speed up debugging.

### 4) Prompt guidance

`alpha_coder_prompts.py` now explicitly warns the model to avoid multiplying returns by raw volume, and points to the normalized alternatives.

### 5) Walk-forward stability diagnostics

The walk-forward stability summary now also exposes additional split-level aggregates (min/median/max IR, IR count, and test n_obs stats).
This makes it easier to understand why `test_ir_positive_frac` (aka OOS positive fraction) is 0: it can be a true performance issue (all splits negative), or a data/coverage issue (zero-observation splits).

## Compatibility

- No breaking changes to the existing DSL syntax.
- Existing formulas continue to run.
- The new operators are optional.

## Next ideas

- Add optional sector-neutral demeaning if a sector map is available.
- Add a richer DSL “cookbook” section with do/don’t examples.
