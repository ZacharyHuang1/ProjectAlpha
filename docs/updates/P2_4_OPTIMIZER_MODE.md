# P2.4 – Optimizer Portfolio Construction

This update adds an **optimizer-based portfolio construction mode** as an alternative to the
original heuristic **equal-weight top/bottom** long/short portfolio.

The goal is to make portfolio construction more realistic (lower churn, smoother weights,
and soft risk control), while staying **dependency-light** (numpy/pandas only) and fully
deterministic.

## What changed

- New module: `src/agent/research/optimizer.py`
- Updated backtest: `src/agent/research/portfolio_backtest.py`
- New CLI flags in `main.py`

The walk-forward evaluation (`P1/P2`) automatically uses the selected construction method.

## Construction methods

### `heuristic` (existing)

- Select top and bottom names (by quantile or by top/bottom N)
- Assign equal weights (0.5 long gross, 0.5 short gross)
- Optional post-processing: neutralization, weight caps, volatility targeting

### `optimizer` (new)

The optimizer uses the same long/short candidate sets, but assigns **non-uniform weights**
by solving a small ridge-style problem.

It supports:

- **Turnover anchoring** to a target weight vector (`w_target`) to reduce churn
- **Soft exposure penalty** (low-rank) on the same exposures used by P2 neutralization
  (beta/vol/liquidity/sector)

Hard neutralization (exact orthogonality) is still applied in the existing post-processing
step when `--neutralize-*` flags are enabled.

## Objective (high-level)

At each rebalance date we solve a convex quadratic problem of the form:

```
maximize   scoreᵀ w
          - 0.5 * λ_l2       * ||w||²
          - 0.5 * λ_turnover * ||w - w_target||²
          - 0.5 * λ_exposure * ||Xᵀ w||²
```

Where:

- `score` is the factor value on the rebalance date
- `w_target` is computed from the previous portfolio weights (scaled) to discourage churn
- `X` are standardized exposures (beta/vol/logADV/sector dummies) when available

After the solve we apply a small number of deterministic projection steps:

- enforce long candidates to be non-negative and short candidates to be non-positive
- rescale to the target gross exposures (0.5 long / 0.5 short)
- apply weight cap (optional) and rescale (without violating the cap)

## How to run

### Default (heuristic)

```bash
python main.py --idea "Momentum + liquidity" --eval-mode p2
```

### Optimizer mode

```bash
python main.py \
  --idea "Momentum + liquidity" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-turnover-lambda 10 \
  --optimizer-l2-lambda 1 \
  --optimizer-exposure-lambda 0.5
```

Notes:

- `optimizer_turnover_lambda` is the most important control (higher = lower turnover).
- `optimizer_exposure_lambda` is a **soft** penalty. For strict control use the existing
  P2 neutralization flags (`--neutralize-beta`, `--neutralize-liquidity`, ...).

## Limitations

- This is **not** a production-grade optimizer: box constraints and strict turnover
  constraints are approximated via projection and/or existing backtest caps.
- The solve is intentionally lightweight (no external QP solver). For strict hard
  constraints, see **P2.5**, which adds an optional `cvxpy` QP backend.
