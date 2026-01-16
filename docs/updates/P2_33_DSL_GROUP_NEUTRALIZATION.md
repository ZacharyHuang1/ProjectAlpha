# P2.33 — DSL Group Neutralization (Sector Fixed Effects)

## Summary

This update makes the factor DSL safer and more “industrial” for volume/scale-sensitive ideas by adding **group neutralization** primitives and wiring optional metadata (e.g., sectors) into the DSL runtime.

The goal is to prevent common failure modes where a factor unintentionally becomes a proxy for market cap / liquidity (e.g., multiplying returns by raw volume), and to enable straightforward **sector neutrality** directly in the DSL.

## What changed

### 1) New DSL operators

Added two cross-sectional operators:

- `cs_demean_group(x, group)`
  - Removes per-group mean cross-sectionally for each date.
  - Example: `cs_demean_group(signal, sector)`.

- `cs_neutralize_group(x, group, *controls)`
  - Per-date group fixed effects + optional numeric controls.
  - Implemented via “within transformation”: de-mean within group for both `x` and `controls`, then regress.
  - Example: `cs_neutralize_group(signal, sector, log1p(ts_mean(close*volume, 20)))`.

### 2) Optional metadata injection into the DSL runtime

`eval_dsl(..., extras=...)` now supports injecting optional fields (currently `sector`).

The evaluation pipeline passes a `sector` Series aligned to the OHLCV MultiIndex:

- If `--sector-map-path` is provided, `sector` comes from the instrument→sector mapping.
- Otherwise, `sector` defaults to `"UNKNOWN"`, which degrades group-neutralization to a plain cross-sectional demean.

### 3) Alpha coder prompt + offline stub upgraded

- The system prompt now explicitly recommends `cs_demean_group(...)` / `cs_neutralize_group(...)` when sector labels are available.
- The offline deterministic volume stub now uses:
  - standardized volume (`ts_zscore(log1p(volume), 20)`) and
  - group neutralization (`cs_neutralize_group(..., sector, log1p(ts_mean(close*volume,20)))`).

## Why this matters

Using absolute `volume` directly in a factor expression often introduces a strong scale bias and can dominate the signal. Group neutralization makes it easy to remove sector structure *before* ranking/z-scoring, which improves interpretability and reduces unintended exposures.

## How to use

If you have a sector map:

```bash
python main.py \
  --sector-map-path path/to/sector_map.csv \
  --idea "volume-conditioned momentum" \
  --eval-mode p2
```

Example DSL:

```text
zscore(cs_neutralize_group(
  ts_mean(returns(close, 1), 20) * ts_zscore(log1p(volume), 20),
  sector,
  log1p(ts_mean(close * volume, 20))
))
```
