# P2.24 — Regime tuning holdings-level revalidation

## Why

P2.23 introduced a fast **proxy** objective for regime hyperparameter tuning:
it ranks configs using an *alpha return matrix* (train→valid) and a turnover penalty on the *alpha weights*.

That proxy is useful and cheap, but it can disagree with the true holdings-level outcome because:

- trade netting happens at the blended holdings level
- costs/borrow are applied on the realized turnover of the blended portfolio
- small differences in daily alpha-weight paths can change trading activity

## What changed

After the P2.23 proxy sweep ranks regime configs, the system now optionally:

1. **takes the top-N proxy configs**
2. **re-prices the blended holdings on the valid segments**
3. **re-ranks configs using the holdings-level valid metric**
4. selects the final regime config

This keeps the search fast (still proxy-driven), while making the final pick more faithful to the real execution model.

Key safety property: **no test leakage** — for each split we fit regime thresholds + allocations on *train* and evaluate on *valid*.

## New CLI flags

These flags apply when `--alpha-allocation-regime-aware` and `--alpha-allocation-regime-tune` are enabled:

- `--alpha-allocation-regime-tune-holdings-top`  
  Revalidate top-N proxy configs with holdings-level pricing on valid.  
  Set to `0` to disable. Default: `3`.

- `--alpha-allocation-regime-tune-holdings-metric`  
  Override the metric for holdings revalidation.  
  Empty means “use `--alpha-allocation-regime-tune-metric`”.

- `--alpha-allocation-regime-tune-holdings-save-top`  
  Trim the JSON summary table to top-N rows (0 keeps all). Default: `10`.

## New artifacts

When holdings-level revalidation runs, the following run artifacts are produced:

- `alpha_allocation_regime_holdings_validation_results.csv`
- `alpha_allocation_regime_holdings_validation_summary.json`

The main allocation config now also reports:

- `allocation.regime.tuned_method` ∈ `{fixed, proxy, holdings_valid}`

## Notes / limitations

- Only the **top-N** proxy configs are revalidated for speed.
- Holdings-level revalidation runs additional alpha backtests on the valid segments (to rebuild positions), so it is slower than proxy-only tuning.
