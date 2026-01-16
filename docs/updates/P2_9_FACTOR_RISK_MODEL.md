# P2.9 Factor Risk Model (B F B^T + D)

This update adds an optional **factor risk model** term inside the portfolio optimizer.

Previously (P2.8), the optimizer could penalize a **diagonal** variance proxy:

- **Diagonal risk**: `sum_i(var_i * w_i^2)`

That helps de-emphasize high-volatility names, but it ignores **correlations** between names.

## What was added

### 1) A lightweight factor risk model estimator

A new module was added:

- `src/agent/research/factor_risk_model.py`

It estimates a simple covariance decomposition:

```
Sigma = B F B^T + D
```

Where:

- `B` is the factor loading matrix (cross-sectional exposures)
- `F` is the factor covariance (estimated from trailing factor returns)
- `D` is diagonal idiosyncratic variance (estimated from trailing residuals)

The estimator is dependency-light (NumPy + Pandas) and designed for research backtests.

### 2) Optimizer support for Sigma = B F B^T + D

`src/agent/research/optimizer.py` now supports two risk modes:

- `diag`: diagonal variance proxy (P2.8 behavior)
- `factor`: factor model risk (P2.9)

When `factor` is available, the optimizer penalizes:

- **Factor risk**: `(B^T w)^T F (B^T w)`
- **Idiosyncratic risk**: `sum_i(d_i * w_i^2)`

Both the QP backend and the ridge fallback support this.

### 3) Backtest integration + new CLI flags

`src/agent/research/portfolio_backtest.py` can now estimate the factor risk model per rebalance step.

New CLI options were added in `main.py` (and threaded through the agent config):

- `--optimizer-risk-model {diag,factor}`
- `--optimizer-factor-risk-window` (default: 60)
- `--optimizer-factor-risk-shrink` (default: 0.2)
- `--optimizer-factor-risk-ridge` (default: 1e-3)

For robust estimation options (EWM/OAS/outlier control), see `P2_10_ROBUST_RISK_ESTIMATION.md`.

## How it works in practice

1. At each rebalance date, the backtest already builds an exposure matrix `X` (beta/vol/liquidity + optional sector dummies).
2. If `--optimizer-risk-model factor` is enabled, it takes a trailing window of daily returns and estimates `(B, F, D)`.
3. The optimizer uses the resulting covariance structure to penalize portfolios with large systematic factor exposure or large idiosyncratic concentration.

If there is not enough history, or exposures are empty, the backtest automatically falls back to the diagonal variance proxy.

## Quick usage examples

Diagonal risk (P2.8 behavior):

```bash
python main.py --eval-mode p2   --construction-method optimizer   --optimizer-risk-aversion 5   --optimizer-risk-model diag
```

Factor risk:

```bash
python main.py --eval-mode p2   --construction-method optimizer   --optimizer-risk-aversion 5   --optimizer-risk-model factor   --optimizer-factor-risk-window 60   --optimizer-factor-risk-shrink 0.2
```

## Notes / limitations

- This is a lightweight factor model meant for research iterations, not a production-grade risk system.
- Factor covariance is estimated from cross-sectional regressions of returns on exposures; it can be noisy on small universes.
- Sector dummies help, but a real model would usually add more robust style factors and stronger shrinkage / regularization.

## Tests added

- `tests/unit_tests/test_factor_risk_model_estimator.py`
- `tests/unit_tests/test_optimizer_factor_risk.py`
