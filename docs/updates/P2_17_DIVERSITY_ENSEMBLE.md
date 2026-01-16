# P2.17 – Diversity-aware selection and alpha ensemble

This update adds a **diversity-aware top-K selector** and a simple **equal-weight ensemble** based on **walk-forward OOS daily return streams**.

The goal is to avoid selecting multiple alphas that are effectively the same trade and to provide a quick “portfolio of alphas” view.

## What changed

### 1) Correlation matrix on OOS returns
For the candidate pool (by default, the top `--diverse-candidate-pool` alphas), the system builds an OOS return matrix and computes:

- Pearson correlation matrix
- pairwise overlap counts (number of common OOS days)

Artifacts:
- `runs/<run_id>/alpha_correlation.csv`
- `runs/<run_id>/alpha_correlation_nobs.csv`

### 2) Greedy diversified top-K selection
If enabled (`--diverse-selection`), selection becomes:

`pick argmax_i (base_score_i - lambda * avg_corr(i, selected))`

Where:
- `base_score` is the same score used for ranking (default: IR)
- `avg_corr` is the average correlation to already-selected alphas
- `lambda` is controlled by `--diverse-lambda`

The selection trace is written into:
- `result.json` under `selection.selection_table`
- `REPORT.md` under **Diversified top-K selection**

### 3) Equal-weight ensemble
If enabled (`--ensemble`), the system creates an equal-weight ensemble of the selected alphas’ OOS return streams:

- `ensemble_return[t] = mean_i r_i[t]` over available returns

Artifacts:
- `runs/<run_id>/ensemble_metrics.json`
- `runs/<run_id>/ensemble_oos_daily.csv`

## How to use

Typical run (P2 + diversified selection + ensemble on):
```bash
python main.py --eval-mode p2 --top-k 5 --diverse-selection --ensemble
```

Tune the diversity penalty:
```bash
python main.py --eval-mode p2 --top-k 5 --diverse-lambda 0.8
```

Disable diversity (pure ranking):
```bash
python main.py --eval-mode p2 --top-k 5 --no-diverse-selection
```

## Notes / limitations

- This ensemble is a **portfolio of strategies** (return stream blending), not a single unified holdings-level portfolio.
- Correlation estimates require enough overlapping OOS days (`--diverse-min-periods`). If overlaps are too small, the selector falls back to pure ranking.
