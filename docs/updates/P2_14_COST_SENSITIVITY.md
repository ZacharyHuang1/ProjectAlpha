# P2.14: Execution-only cost sensitivity

This update adds **cost sensitivity curves** and **break-even estimates** for the top alphas.

The goal is to answer a practical question:

> “How much transaction cost / spread / impact / borrow can this strategy tolerate before it stops making money?”

This is different from the P2.13 ablation:

- **P2.13 ablation** can be `end_to_end` (strategy adapts / re-optimizes under different costs) or `execution_only` (same trades, different cost deductions).
- **P2.14 cost sensitivity** is always **execution-only** and sweeps cost levels on a grid to produce a curve (and a break-even estimate).

## What is computed

For each selected alpha we compute, on the out-of-sample (OOS) walk-forward path:

- A curve for each parameter:
  - `linear_cost_bps` (commission + slippage)
  - `half_spread_bps`
  - `impact_bps`
  - `borrow_bps` (only when no borrow curve is provided)
  - `borrow_multiplier` (when a borrow curve is provided)
- For each point on the curve we report:
  - information ratio (IR)
  - annualized return
  - max drawdown
  - mean cost drag (bps)
- A **break-even estimate** (parameter value where **mean daily net return crosses 0**) via linear interpolation.

## How to run

Cost sensitivity is enabled by default and runs for the top alpha:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --cost-sensitivity \
  --cost-sensitivity-top 1
```

Custom sweep grids (comma-separated):

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --cost-sensitivity \
  --cost-sensitivity-linear-bps "0,2,5,10,15,20" \
  --cost-sensitivity-half-spread-bps "0,1,2,5,10" \
  --cost-sensitivity-impact-bps "0,25,50,75,100" \
  --cost-sensitivity-borrow-bps "0,50,100,200,500"
```

If you provide a borrow curve (`--borrow-rates-path`), the borrow sweep uses a multiplier:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --borrow-rates-path data/borrow_rates.csv \
  --cost-sensitivity-borrow-mult "0,0.5,1,2,3"
```

## New artifacts

Saved under `runs/<run_id>/`:

- `cost_sensitivity.csv` (full curves)
- `cost_sensitivity_break_even.csv` (break-even summary)
- `REPORT.md` includes a “Cost sensitivity (execution_only)” section

## Implementation notes

- `backtest_long_short()` now records an additional daily field `impact_unit` so impact cost can be recomputed as `impact_unit * impact_bps / 10000` without re-running the strategy.
- `agent.research.cost_sensitivity.compute_cost_sensitivity()` converts the OOS daily path into curves and break-even points.
- `experiment_tracking.py` exports the curve and break-even tables into CSV files.

## Limitations

- This is **execution-only**: it assumes the same trades and only changes the cost deduction. If the strategy would materially change its trading behavior under higher costs, use `ablation_mode=end_to_end` (P2.13).
- The sweeps only vary **bps coefficients**. They do not change the impact exponent or participation caps.
- Break-even is defined using **mean daily net return = 0** (a simple and interpretable threshold).
