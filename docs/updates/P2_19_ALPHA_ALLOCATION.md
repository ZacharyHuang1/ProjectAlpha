# P2.19 — Alpha allocation (walk-forward) + holdings-level ensemble

This update adds a **second-stage model** on top of individual alphas:

- Stage 1: each alpha is evaluated with walk-forward backtests (already implemented).
- Stage 2 (new): for each walk-forward split, we **learn non-negative weights across the selected alphas** using only the data *before* the test segment, and then **blend holdings** and re-price the combined portfolio with costs (trade netting).

## What was added

### 1) Alpha allocation solver

New module: `src/agent/research/alpha_allocation.py`

- Fits weights `w >= 0`, `sum(w)=1` over alpha strategies using an in-sample return matrix.
- Objective (conceptually):

`maximize  score(w)  - lambda_corr * (w^T Corr w) - l2 * ||w||^2`

- **Correlation penalty** discourages redundant alphas.
- Backend:
  - `cvxpy` QP if available (`--alpha-allocation-backend qp|auto`)
  - projected-gradient fallback (`--alpha-allocation-backend pgd`)

### 2) Walk-forward allocated holdings ensemble

Updated module: `src/agent/research/holdings_ensemble.py`

New function: `walk_forward_holdings_ensemble_allocated(...)`

For each split:

1. Fit alpha weights on the **train** or **train+valid** segment (`--alpha-allocation-fit`).
2. Run each alpha on the **test** segment to produce its holdings path.
3. Blend holdings by learned alpha weights.
4. Re-price the blended portfolio via `backtest_from_weights(...)` so costs/borrow are applied to the **netted trades**.

### 3) Pipeline integration + artifacts

- `evaluate_alphas_agent.py` now runs both:
  - equal-weight holdings ensemble (P2.18)
  - allocated holdings ensemble (P2.19, optional)
- New run artifacts (when enabled):
  - `ensemble_holdings_allocated_metrics.json`
  - `ensemble_holdings_allocated_oos_daily.csv`
  - `ensemble_holdings_allocated_positions.csv`
  - `alpha_allocations.csv`
  - `alpha_allocation_diagnostics.csv`
- `REPORT.md` now includes a section for the allocated ensemble and a simple **mean weight** summary.

## How to run

Example:

```bash
python main.py   --idea "mean reversion with liquidity filter"   --max-iterations 1   --eval-mode p2   --holdings-ensemble   --alpha-allocation   --alpha-allocation-backend auto   --alpha-allocation-fit train_valid   --alpha-allocation-score-metric information_ratio   --alpha-allocation-lambda 0.5   --alpha-allocation-max-weight 0.8
```

Notes:

- If the fit window is too short (`--alpha-allocation-min-days`), the split falls back to equal weights.
- If `cvxpy` is not installed, `auto` falls back to the PGD solver.

## Why this matters

Equal-weight ensembles are a good baseline, but allocation can improve:

- **robustness** (down-weight unstable alphas)
- **diversification** (penalize correlated alpha clusters)
- **netting-aware costs** (still priced at holdings-level, not return-level)
