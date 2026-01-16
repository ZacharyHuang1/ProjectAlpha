# P2.18: Holdings-level ensemble (multi-alpha at the portfolio layer)

P2.17 introduced a *return-stream* ensemble (equal-weight blending of OOS daily returns). That is easy and useful, but it cannot capture **trade netting** (when two alphas trade opposite directions, portfolio-level turnover and impact can be lower than the average of each strategy).

P2.18 adds a **holdings-level ensemble**:

1. Re-run each selected alpha on each walk-forward test split, but request its *daily executed weights* (holdings).
2. Combine those holdings into a single portfolio weight path (equal-weight across alphas).
3. Re-price the combined portfolio with the same cost / borrow model to obtain realized OOS performance.

This makes the “multi-alpha” evaluation closer to how a real combined book behaves.

## What changed

### 1) Portfolio backtest: optional positions output

- `backtest_long_short(..., include_positions=True)` now returns:
  - `positions`: long-form rows `{datetime, instrument, weight}` for non-zero holdings
  - `position_dates`: all daily timestamps covered by the backtest (so fully-flat days are still represented)

### 2) Portfolio backtest: pricing from a weight path

- New helper: `backtest_from_weights(weights, ohlcv, config=..., ...)`.

It takes a daily weight matrix and computes:

- gross returns (`w · r`)
- turnover and linear/spread/impact costs
- borrow costs (optional `borrow_rates` input)

This is the engine used to evaluate the holdings-level ensemble.

### 3) New module: `agent.research.holdings_ensemble`

- `walk_forward_holdings_ensemble(...)` builds an equal-weight holdings ensemble over walk-forward OOS test splits.
- It uses each alpha’s per-split sign decision (`walk_forward.splits[*].sign`) so the ensemble matches the alpha’s chosen orientation.

### 4) Agent integration + run artifacts

`EvaluateAlphasAgent` now attaches `ensemble_holdings` to the result payload.

The experiment tracker writes:

- `ensemble_holdings_metrics.json`
- `ensemble_holdings_oos_daily.csv`
- `ensemble_holdings_positions.csv`

`REPORT.md` includes a short holdings-ensemble section and a delta vs the return-stream ensemble.

## CLI flags

New flags:

- `--holdings-ensemble / --no-holdings-ensemble` (default: enabled)
- `--holdings-ensemble-apply-turnover-cap / --no-holdings-ensemble-apply-turnover-cap`
  - Default is **False**. When enabled, the global `--turnover-cap` is applied when pricing the combined holdings path.

## How to run

Example:

```bash
python main.py \
  --eval-mode p1 \
  --data-path ./data/sample.parquet \
  --top-k 5 \
  --diverse-selection \
  --ensemble \
  --holdings-ensemble
```

Then inspect the newest run under `./runs/<run_id>/`.

## Notes / limitations

- The holdings-level ensemble is intentionally **equal-weight** to keep behavior simple and interpretable.
- The combined portfolio can have lower leverage than the average strategy due to offsetting positions. This is expected and is part of the trade-netting benefit.
- If you want to lever the ensemble back to a target gross, that should be added explicitly as a portfolio construction policy (future work).
