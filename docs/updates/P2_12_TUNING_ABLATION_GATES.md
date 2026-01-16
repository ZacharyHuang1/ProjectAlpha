# P2.12: Tuning, cost ablation, and stronger quality gates

This update adds a *research-friendly* tuning and diagnostics layer on top of the existing P2 pipeline.

## What changed

### 1) Deterministic parameter sweep (tuning)
- Added a small grid-search utility (`src/agent/research/tuning.py`) that generates `BacktestConfig` variants.
- `evaluate_alphas_agent` can optionally evaluate each alpha across this sweep and select the best configuration by a chosen metric (default: `information_ratio`).
- The sweep reuses a single set of walk-forward splits to keep runs comparable and to reduce duplicated work.

Artifacts:
- `runs/<run_id>/sweep_results.csv` (flattened sweep table)
- `runs/<run_id>/REPORT.md` includes a “P2.12: Tuning sweep” section for the best alpha.

### 2) End-to-end cost ablation for the top alphas
- After selecting the top alphas, the runner can optionally perform a small “cost ablation”:
  - `no_costs`
  - `linear_only`
  - `linear_spread`
  - `linear_spread_impact`
  - `full`
- This is an **end-to-end** ablation: the portfolio construction is re-run under each cost setting (i.e., the optimizer can change the weights when costs are removed).

Artifacts:
- `runs/<run_id>/ablation_results.csv`
- `runs/<run_id>/REPORT.md` includes a “P2.12: Cost ablation” section for the best alpha.

### 3) Extra quality gates
In addition to `min_coverage` and `max_turnover`, the following optional gates were added:
- `--min-ir`
- `--max-dd` (positive number, e.g. `0.25` means “no worse than -25%”)
- `--max-total-cost-bps` (mean daily total cost drag in bps, `cost + borrow`)
- `--min-wf-splits`

These are enforced consistently:
- during tuning selection (so the sweep does not “cheat” by picking obviously bad configs)
- for final SOTA selection

### 4) Borrow cost multiplier (ablation support)
- Added `borrow_cost_multiplier` to `BacktestConfig`.
- This allows turning borrow *cost* on/off while keeping borrow *constraints* (e.g., `max_borrow_bps`) intact.

## How to run

### Basic P2 run (no tuning)
```bash
python main.py --eval-mode p2 --top-k 3
```

### Enable tuning + ablation + gates
```bash
python main.py \
  --eval-mode p2 \
  --construction-method optimizer \
  --tune \
  --tune-metric information_ratio \
  --tune-max-combos 24 \
  --ablation-top 1 \
  --min-ir 0.0 \
  --max-dd 0.30 \
  --max-total-cost-bps 10.0 \
  --min-wf-splits 3
```

### Override sweep grids
All sweep lists are comma-separated:
```bash
python main.py \
  --eval-mode p2 \
  --construction-method optimizer \
  --tune \
  --tune-turnover-cap "0,0.05,0.10,0.20" \
  --tune-max-abs-weight "0,0.01,0.02" \
  --tune-risk-aversion "0,5,10" \
  --tune-cost-aversion "0,0.5,1"
```

## Notes / design choices
- The sweep grid is intentionally small by default to keep runtime reasonable.
- The ablation is end-to-end (re-optimization happens). If you want a “pure execution drag” ablation (fixed weights, varying costs), that would be a separate follow-up.
