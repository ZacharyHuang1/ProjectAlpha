# P2.13: Execution-only cost ablation + regime diagnostics

This update adds two small but high-leverage research diagnostics:

1) **Execution-only cost ablation** (fixed trading path)
2) **Simple regime breakdown** for the best alpha

The goal is to make cost analysis *interpretable* and to help answer:

- “Are costs killing performance because the *strategy trades too much*, or because the *optimizer changes its behavior* when costs change?”
- “Does the alpha behave differently in high-vol vs low-vol markets?”

---

## 1) Cost ablation now has two modes

### A. `end_to_end` (re-optimization)

This is the existing ablation: we rerun the full walk-forward pipeline with modified cost settings.
Because the optimizer sees different costs, it may change turnover, weights, and exposure.

Use this to answer:

- “If I changed my execution model (or added borrow), would the strategy adapt and remain profitable?”

### B. `execution_only` (fixed weights/trades)

This is new in P2.13.

We run the strategy once (normal full-cost configuration), keep the **realized trading path** fixed, and recompute daily returns under different cost deductions:

- `no_costs`: gross returns, no costs subtracted
- `linear_only`: subtract commission + slippage
- `linear_spread`: subtract linear + half-spread
- `linear_spread_impact`: subtract linear + spread + impact
- `full`: subtract costs + borrow

Use this to answer:

- “How much performance is lost to each cost component, holding trades constant?”

---

## 2) CLI changes

New flags:

- `--ablation-mode {both,end_to_end,execution_only}`
  - `both` is the default when ablation is enabled.
- `--regime-analysis / --no-regime-analysis` (default: enabled)
- `--regime-window` (default: 20)
- `--regime-buckets` (default: 3)

Example:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --ablation-top 1 \
  --ablation-mode both
```

---

## 3) Output / artifacts

### `runs/<run_id>/ablation_results.csv`

Now includes an `ablation_mode` column:

- `end_to_end`
- `execution_only`

### `runs/<run_id>/REPORT.md`

The run report now prints:

- End-to-end ablation table
- Execution-only ablation table (with `mean_cost_drag_bps`)
- Regime analysis tables (market volatility, and liquidity if volume is present)

### `runs/<run_id>/regime_analysis.csv`

If regime analysis is enabled, the per-bucket metrics are also exported as a flat CSV.

---

## 4) Implementation notes

Execution-only ablation is computed inside `walk_forward_evaluate_factor`:

- We already compute a daily decomposition in `backtest_long_short` (`gross_return`, `linear_cost`, `spread_cost`, `impact_cost`, `borrow`).
- For each test segment, we rebuild a net return series under each scenario by summing the appropriate components.
- We concatenate the segments into a single out-of-sample series per scenario and summarize metrics.

Regime analysis is intentionally simple:

- Market volatility regime uses rolling std of the equal-weight market return (shifted by 1 day).
- Market liquidity regime uses rolling log(ADV) (shifted by 1 day), if `volume` is available.
- Days are bucketed by quantiles (`--regime-buckets`).

This is a debugging tool, not a complete risk model.
