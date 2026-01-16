# P2.15: Multi-horizon decay analysis

This update adds a small but high-signal research diagnostic: **how predictive power and signal persistence change with the forward-return horizon**.

It is designed to answer:

1. **Predictive decay**: does the alpha work better at 1-day, 5-day, 10-day horizons?
2. **Signal persistence**: does the *rank / top names* stay stable for multiple days (useful to decide a reasonable holding/rebalance cadence)?

This is **not a trade simulation**. It is a factor diagnostic that complements walk-forward backtests.

---

## What was added

- `src/agent/research/decay_analysis.py`
  - `compute_horizon_decay(...)`: computes per-horizon metrics
    - `IC` / `RankIC` mean + t-stat
    - `spread_mean` (top quantile minus bottom quantile, using forward returns)
    - `spread_ir_ann_proxy` (annualized IR proxy scaled by `sqrt(252/horizon)`)
    - `signal_overlap_mean`: Jaccard overlap of top/bottom sets between `t` and `t+h` (signal persistence)

- `evaluate_alphas_agent` integration
  - Stores results under `backtest_results.analysis.decay` for the top alphas.

- Experiment tracking
  - Exports `runs/<run_id>/decay_analysis.csv`
  - Adds a new section in `runs/<run_id>/REPORT.md`

---

## How to run

Decay analysis is enabled by default.

```bash
python main.py --eval-mode p2 --decay-analysis --decay-analysis-top 1
```

Override horizons:

```bash
python main.py \
  --eval-mode p2 \
  --decay-analysis \
  --decay-horizons "1,2,5,10,20"
```

Disable:

```bash
python main.py --no-decay-analysis
```

---

## Outputs

In the run folder:

- `decay_analysis.csv`: full per-horizon table
- `REPORT.md`: includes a summary table and a suggested "best" horizon

Key columns:

- `rank_ic_mean`, `rank_ic_tstat`: cross-sectional rank correlation strength
- `spread_mean`: mean top-minus-bottom forward return
- `spread_ir_ann_proxy`: annualized proxy scaling by `sqrt(252/h)`
- `signal_overlap_mean`: stability of top/bottom names across the horizon lag

---

## Notes / interpretation

- If `RankIC` peaks at short horizons but `signal_overlap_mean` collapses quickly, the signal is likely **fast** (short holding periods, careful turnover control).
- If `RankIC` persists and `signal_overlap_mean` stays high for 5-10 days, the alpha is likely **slower** and may benefit from longer holding / lower turnover.
- `spread_ir_ann_proxy` is a diagnostic only. Forward returns overlap across dates, so do not treat it as a production-grade IR.
