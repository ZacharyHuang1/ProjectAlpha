# P2.28 — Pareto auto-selection (knee / utility) + stability objectives

This update adds an **automatic config chooser** for regime-aware allocation tuning.

Previously, the system selected a regime tuning config via:

1. apply constraints (turnover / drag / switching / fallback)
2. select the row with the **best objective** (e.g., highest IR)
3. optionally restrict the candidate set to the **Pareto-efficient rows** (`--alpha-allocation-regime-tune-prefer-pareto`)

P2.28 keeps the old behavior (`best_objective`) but adds two new selection methods:

- `knee`: pick the Pareto point closest to the “ideal” point (max objective, min penalties) after normalizing each objective
- `utility`: pick the Pareto point with the highest weighted sum of normalized “goodness” values

In addition, the tuning results now include split-level stability diagnostics and can optionally include those stability terms in the Pareto/selection objectives.

## How the selector works

Given a list of objectives like:

- `objective` (maximize)
- `alpha_weight_turnover_mean` (minimize)
- extra metrics (optional), e.g. `turnover_cost_drag_bps_mean`, `regime_switch_rate_mean`, `fallback_frac_mean`

We first normalize each metric across candidate rows to a `[0, 1]` “goodness” score where higher is always better:

- for a max objective: `(x - min) / (max - min)`
- for a min objective: `(max - x) / (max - min)`

If a metric has zero range, its goodness is treated as `1.0` for all rows.

Selection is then:

- knee: minimize Euclidean distance to the ideal vector of ones
- utility: maximize weighted sum of goodness values (weights are normalized to sum to 1)

The selector still respects constraints first. If no rows satisfy constraints, it falls back to selecting from the unconstrained candidate set.

## Stability objectives

We now compute split-level objective values and aggregate them into:

Proxy tuning:
- `objective_split_std`
- `objective_split_min`
- `objective_split_negative_frac`

Holdings revalidation:
- `holdings_objective_split_std`
- `holdings_objective_split_min`
- `holdings_objective_split_negative_frac`

When `--alpha-allocation-regime-tune-include-stability-objectives` is enabled and the selection method is `knee` or `utility`, the Pareto objectives automatically include:

- split std (min)
- split min (max)

## New CLI flags

- `--alpha-allocation-regime-tune-selection-method best_objective|knee|utility`
- `--alpha-allocation-regime-tune-utility-weights "k=v,k=v,..."` (weights are applied in normalized goodness space)
- `--alpha-allocation-regime-tune-include-stability-objectives` / `--no-alpha-allocation-regime-tune-include-stability-objectives`

Example:

```bash
python main.py --idea "Momentum" --eval-mode p2   --alpha-allocation-regime-aware   --alpha-allocation-regime-tune   --alpha-allocation-regime-tune-preset execution_realistic   --alpha-allocation-regime-tune-selection-method utility   --alpha-allocation-regime-tune-utility-weights "objective=1,alpha_weight_turnover_mean=0.4,turnover_cost_drag_bps_mean=0.4,regime_switch_rate_mean=0.2,fallback_frac_mean=0.2"
```

## Preset changes

- `low_turnover`: defaults to `selection_method="knee"` and includes stability objectives
- `execution_realistic`: defaults to `selection_method="utility"` and provides a reasonable default weight map

## Outputs

The regime tuning CSV/JSON artifacts now include stability columns and selection metadata:

- `runs/<run_id>/alpha_allocation_regime_param_tuning_results.csv`
- `runs/<run_id>/alpha_allocation_regime_param_tuning_summary.json`
- `runs/<run_id>/alpha_allocation_regime_holdings_validation_results.csv`
- `runs/<run_id>/alpha_allocation_regime_holdings_validation_summary.json`

If `matplotlib` is installed and plots are enabled, the chosen point is highlighted in the generated Pareto scatter plots.
