# P2.20 — Alpha allocation smoothing + meta-tuning

This update improves the **walk-forward alpha allocation** (P2.19) in two ways:

1) **Smoothing / turnover control** at the *alpha-weight* level
2) **Meta-tuning** of allocation hyperparameters using *train → valid* segments (no test leakage)

## What changed

### 1) Alpha weight smoothing (turnover penalty)

- `fit_alpha_allocation(...)` now supports:
  - `prev_weights`: previous split's alpha weights
  - `turnover_lambda`: L2 penalty on weight changes, i.e. `||w - w_prev||^2`
- This reduces allocation churn across walk-forward splits.

Files:
- `src/agent/research/alpha_allocation.py`

### 2) Meta-tuning allocation hyperparameters

Added a deterministic meta-tuning routine that:

- Fits weights on each split's **train** segment
- Evaluates the resulting alpha-ensemble on that split's **valid** segment
- Aggregates valid performance across splits to choose **one** best config

Tuned knobs:
- `lambda_corr` (correlation/redundancy penalty)
- `max_weight` (single-alpha cap)
- `turnover_lambda` (alpha-weight smoothing)

Files:
- `src/agent/research/alpha_allocation_tuning.py`
- `src/agent/research/holdings_ensemble.py`

### 3) CLI + reporting + artifacts

New CLI flags (all optional):

- `--alpha-allocation-turnover-lambda`
- `--alpha-allocation-tune`
- `--alpha-allocation-tune-metric`
- `--alpha-allocation-tune-max-combos`
- `--alpha-allocation-tune-lambda-grid`
- `--alpha-allocation-tune-max-weight-grid`
- `--alpha-allocation-tune-turnover-lambda-grid`
- `--alpha-allocation-tune-save-top`

Artifacts:
- `alpha_allocation_tuning_results.csv`
- `alpha_allocation_tuning_summary.json`

Files:
- `main.py`
- `src/agent/agents/evaluate_alphas_agent.py`
- `src/agent/services/experiment_tracking.py`
- `src/agent/research/reporting.py`

## How to use

Run with meta-tuning enabled:

```bash
python main.py \
  --eval-mode p2 \
  --alpha-allocation --alpha-allocation-tune \
  --alpha-allocation-tune-lambda-grid "0,0.2,0.5,0.8" \
  --alpha-allocation-tune-max-weight-grid "0.5,0.8,1.0" \
  --alpha-allocation-tune-turnover-lambda-grid "0,0.5,2" \
  --top-k 5
```

If you prefer a fixed smoothing level (no tuning):

```bash
python main.py --eval-mode p2 --alpha-allocation --alpha-allocation-turnover-lambda 1.0
```

## Notes

- Meta-tuning uses only **train/valid**. Test segments are reserved for the final OOS evaluation.
- The tuning grid is intentionally small and deterministic to keep runtime manageable.
