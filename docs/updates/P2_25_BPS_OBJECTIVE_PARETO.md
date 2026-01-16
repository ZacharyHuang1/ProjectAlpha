# P2.25 — Interpretable turnover costs (bps) + Pareto flags for regime tuning

This update improves the **regime-aware allocation meta-tuning** workflow (P2.23) and the
**holdings-level revalidation** step (P2.24).

The goal is to make the “turnover penalty” **interpretable** (in bps) and to make it easy to
inspect the trade-off between performance and allocation stability.

## What changed

### 1) Turnover penalty is now a *bps cost* applied in returns-space

Previously, regime tuning used:

- `objective = valid_metric - turnover_penalty * alpha_weight_turnover_mean`

That mixes units (IR / annualized return vs. a penalty weight), which makes it hard to
reason about the size of the penalty.

As of **P2.25**, `turnover_penalty` is treated as an **additional cost in bps** per unit of
alpha-weight turnover, applied directly to the validation return series:

- `r_adj[t] = r[t] - alpha_turnover[t] * turnover_cost_bps / 10000`

Then the objective is computed on the adjusted series:

- `objective = metric(r_adj)`

This keeps the objective in the same units as the chosen metric while giving the
penalty an intuitive interpretation.

### 2) Preferred CLI flag (old flag remains as an alias)

A clearer flag name was added:

- `--alpha-allocation-regime-tune-turnover-cost-bps <float>`

The old flag still works but is deprecated:

- `--alpha-allocation-regime-tune-turnover-penalty <float>`

Both map to the same internal config value.

### 3) New columns exported in tuning results

The regime meta-tuning CSV now includes extra fields:

- `valid_metric_after_turnover_cost`
- `turnover_cost_bps`
- `turnover_cost_drag_bps_mean` (≈ `alpha_weight_turnover_mean * turnover_cost_bps`)

The holdings-level revalidation CSV includes:

- `holdings_valid_metric_after_turnover_cost`
- `turnover_cost_bps`
- `turnover_cost_drag_bps_mean`

### 4) Pareto front flags (2D)

Both the proxy tuning results and the holdings revalidation results now include:

- `is_pareto`

This is a simple 2D Pareto flag (maximize objective, minimize alpha-weight turnover), so
you can quickly identify “efficient” trade-offs without committing to a single scalar.

### 5) Report improvements

`REPORT.md` now surfaces:

- raw vs. adjusted validation metrics for the chosen config
- `turnover_cost_bps`
- `turnover_cost_drag_bps_mean`
- `is_pareto` in the mini-tables

## How to use

### Regime tuning with a turnover cost

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-tune \
  --alpha-allocation-regime-tune-mode-grid "vol,vol_liq" \
  --alpha-allocation-regime-tune-window-grid "10,20,40" \
  --alpha-allocation-regime-tune-buckets-grid "2,3,4" \
  --alpha-allocation-regime-tune-smoothing-grid "0,0.05,0.10" \
  --alpha-allocation-regime-tune-turnover-cost-bps 5
```

### How to pick `turnover_cost_bps`

There is no single “right” number. Practical guidance:

- Start with `0` to understand the unconstrained performance.
- Then try a small cost like `1–10` bps to see how much performance you give up to
  reduce alpha-weight churn.
- Inspect `turnover_cost_drag_bps_mean` in the CSV to sanity-check the implied daily drag.

Note: this is an *extra* cost applied to the alpha-weight allocation layer. The holdings-level
backtest already includes trading costs from the blended holdings weights.

## Files changed

- `src/agent/research/alpha_allocation_regime_tuning.py`
- `src/agent/research/holdings_ensemble.py`
- `src/agent/research/reporting.py`
- `src/agent/agents/evaluate_alphas_agent.py`
- `main.py`
- `docs/RUNBOOK.md`
- `docs/updates/P2_23_REGIME_META_TUNING.md`

