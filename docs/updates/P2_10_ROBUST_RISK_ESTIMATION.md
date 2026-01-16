# P2.10: Robust risk estimation and attribution

This update improves the stability and interpretability of the **factor risk model** used inside the optimizer objective.

## What changed

### 1) More robust factor covariance estimation

In `agent.research.factor_risk_model` the factor covariance estimator now supports:

- **sample** covariance (baseline)
- **EWM** covariance with an `n_eff` proxy for shrinkage routines

### 2) Automatic shrinkage (OAS)

You can now shrink the factor covariance using an automatic OAS-style shrinkage **to identity**:

- `--optimizer-factor-risk-shrink-method oas`

This is useful when the factor covariance is noisy (short windows, many factors, regime shifts).

### 3) Outlier control

- Estimated factor returns can be winsorized:
  - `--optimizer-factor-return-clip-sigma 6.0` (0 disables)
- Idiosyncratic variance can be clipped and shrunk:
  - `--optimizer-idio-clip-q 0.99`
  - `--optimizer-idio-shrink 0.2`

### 4) Risk attribution in diagnostics

Optimizer diagnostics now include:

- `risk_factor_top_contributors` (factor-level)
- `risk_idio_top_contributors` (name-level idiosyncratic)

This helps you understand whether risk is coming from factor exposures or concentrated idiosyncratic bets.

## How to run

### Factor risk model + robust estimation (recommended)

```bash
python main.py \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend auto \
  --optimizer-risk-model factor \
  --optimizer-risk-aversion 5 \
  --optimizer-factor-risk-window 60 \
  --optimizer-factor-risk-estimator ewm \
  --optimizer-factor-risk-ewm-halflife 20 \
  --optimizer-factor-risk-shrink-method oas \
  --optimizer-factor-return-clip-sigma 6 \
  --optimizer-idio-clip-q 0.99 \
  --optimizer-idio-shrink 0.2
```

### Where to find diagnostics

After the run:

- `runs/<run_id>/result.json`
  - `construction.optimizer.last.qp_meta.diagnostics`

Look for:
- `risk_model` and `risk_model_meta`
- `risk_factor_top_contributors`
- `risk_idio_top_contributors`
- `objective_terms`

## Notes

- OAS shrinkage is an approximation when used with EWM weights; we use `n_eff` as an effective sample size proxy.
- This is still a research-grade estimator; for production you would typically add a stronger regime-aware or Bayesian shrinkage model.
