# P2.22 — Selection meta-tuning (validation) for diversified top-K

This update adds **walk-forward meta-tuning** for *selection hyperparameters* (the `diverse_selection` step).

The goal is to avoid a common research pitfall:

- P2.17 diversified selection is helpful, but if you pick `diverse_lambda` / candidate pool size using **test** data, you can overfit.
- P2.22 tunes these knobs on the **validation** segments (train → valid), then evaluates the chosen selection on the **test** segments.

## What changed

### 1) Walk-forward now optionally exports validation daily returns

`walk_forward_evaluate_factor(...)` gained a new flag:

- `return_valid_daily: bool = False`

When enabled, it concatenates the per-split **validation** daily return streams into:

- `metrics['walk_forward']['valid_daily'] = [{'datetime': ..., 'net_return': ...}, ...]`

This is used only for selection tuning; the default remains `False` to keep outputs small.

### 2) New selection tuning module

Added:

- `src/agent/research/selection_tuning.py`

It provides:

- `build_valid_return_matrix(...)`
- `tune_diverse_selection(...)`

The tuning loop is deterministic:

1. Build a validation return matrix (columns = alpha ids).
2. Rank alphas by validation IR (base score).
3. For each `(candidate_pool, diversity_lambda[, top_k])` combo:
   - Run greedy diversified selection.
   - Evaluate an equal-weight ensemble on **validation** returns.
4. Pick the best configuration by `--selection-tune-metric`.

### 3) Pipeline integration in `evaluate_alphas_agent`

If enabled (`--selection-tune`, default **on**):

- run selection tuning using validation return streams
- set the final selected alphas to the tuned selection
- report correlation + ensemble on **test OOS** returns
- export tuning artifacts under `runs/<run_id>/`

### 4) New artifacts

When selection tuning is enabled and validation returns exist:

- `runs/<run_id>/selection_tuning_summary.json`
- `runs/<run_id>/selection_tuning_results.csv`

The standard P2.17 outputs are still produced (correlation matrix + test ensemble).

## How to run

Example:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --selection-tune \
  --selection-tune-metric information_ratio \
  --selection-tune-lambda-grid "0,0.2,0.5,0.8" \
  --selection-tune-candidate-pool-grid "10,20,40"
```

If you want to also tune `top_k`:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --selection-tune-topk-grid "2,3,5"
```

Disable tuning (falls back to P2.17 behavior):

```bash
python main.py --no-selection-tune
```

## Notes / limitations

- If `WalkForwardConfig.valid_days == 0`, there are no validation segments; selection tuning is skipped.
- This is a **simple deterministic** tuner; it does not do Bayesian optimization and it does not “peek” at the test set.

