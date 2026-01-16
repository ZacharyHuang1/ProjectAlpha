# P2.8 Diagonal Risk Penalty (Variance Proxy)

This update adds an optional **diagonal risk term** to the portfolio construction optimizer.

It improves realism by discouraging concentration in high-variance names, while keeping the project
dependency-light and runnable.

## What changed

- Added two new knobs in `BacktestConfig` (and CLI):
  - `optimizer_risk_aversion`
  - `optimizer_risk_window`

- Extended `OptimizerCostModel` to carry a per-name variance proxy (`risk_var`) and a coefficient
  (`risk_aversion`).

- QP backend (`cvxpy`) objective now includes:

  `risk_aversion * sum_i risk_var_i * w_i^2`

- Ridge fallback now supports the same diagonal penalty via a generalized Woodbury solve:
  it solves `(D + lam * A A^T)^{-1} b` where `D` is diagonal.

- Diagnostics for QP solutions include:
  - `risk_proxy` (annualized diagonal variance proxy)
  - `risk_vol_annual_proxy = sqrt(risk_proxy)`
  - `risk_top_contributors` (top names by `risk_var_i * w_i^2`)

## Risk proxy definition

We estimate a lookahead-safe volatility series and convert it into an **annualized variance proxy**:

- compute rolling volatility on close-to-close returns with window `optimizer_risk_window`
- shift by 1 day (lookahead-safe)
- `risk_var = vol^2 * trading_days`

This is not a full covariance risk model (no correlations). It is a robust, cheap first step.

## How to run

Enable the optimizer and add risk aversion:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend auto \
  --optimizer-risk-aversion 10 \
  --optimizer-risk-window 20
```

If you have `cvxpy` installed, you can force the QP backend:

```bash
pip install -r requirements-optimizer.txt

python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend qp \
  --optimizer-risk-aversion 10 \
  --optimizer-risk-window 20 \
  --optimizer-turnover-cap 0.15
```

## Tuning tips

- Start with `--optimizer-risk-aversion` in `[1, 50]`.
- If you see:
  - too much concentration in a few volatile names → increase risk aversion
  - over-diversification / muted signal → decrease risk aversion

This term interacts with:
- `--optimizer-l2-lambda` (overall shrink)
- `--optimizer-turnover-lambda` / `--optimizer-turnover-cap` (churn control)
- cost-aware terms (`--optimizer-cost-aversion`)

## Limitations and next step

Diagonal risk is only a proxy. The next “production-grade” step is a factor/covariance risk model:

- build `Sigma = B F B^T + D`
- add `w^T Sigma w` to the objective
- optionally enforce exposure bounds (not only neutrality)

That is the natural follow-up for P2.9 / P3.
