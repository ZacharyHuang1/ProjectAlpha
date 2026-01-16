# P2.16: Holding / Rebalance Schedule Sweep

This release adds a **strategy-level** diagnostic that sweeps over
`(rebalance_days, holding_days)` schedules for the top alpha(s).

Unlike tuning (which changes optimizer knobs), this sweep keeps the alpha *fixed*
and changes only **when you trade**.

The goal is to understand the trade-off between:

- **signal decay** (fast signals often need frequent refresh),
- **turnover and costs** (slower schedules reduce trading),
- and overall **out-of-sample performance**.

## What it does

For each selected alpha, the evaluator:

1. builds a small deterministic grid of `(rebalance_days, holding_days)` pairs
2. re-runs walk-forward evaluation for each pair
3. records IR / return / drawdown / turnover and mean total cost
4. selects a recommended schedule using the chosen metric (default: IR)

The sweep results are saved into both:

- the run payload (`result.json` → `backtest_results.analysis.schedule_sweep`)
- a flat CSV artifact: `runs/<run_id>/schedule_sweep.csv`

## How to run

Defaults (enabled for the top alpha):

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --schedule-sweep
```

Customize the grids (comma-separated):

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --schedule-sweep \
  --schedule-sweep-rebalance-days "1,2,5,10" \
  --schedule-sweep-holding-days "1,2,5,10,20" \
  --schedule-sweep-max-combos 25
```

Choose how schedules are ranked:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --schedule-sweep-metric information_ratio
```

## Output artifacts

The sweep adds:

- `runs/<run_id>/schedule_sweep.csv` (long-form table)
- a new section in `runs/<run_id>/REPORT.md` with:
  - a recommended schedule
  - a small pivot table (when the grid is small)

## Notes

- This sweep is intentionally **small** by default to keep runtime reasonable.
- The base schedule is always included in the evaluated set even if you do not
  include it in the CLI grids.
- The evaluator reuses the same walk-forward splits for all schedules.
