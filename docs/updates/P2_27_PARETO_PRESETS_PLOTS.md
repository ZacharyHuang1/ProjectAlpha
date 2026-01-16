# P2.27 – Multi-objective Pareto, tuning presets, and plots

This update tightens the **regime-aware tuning workflow** by making the selection step more interpretable and easier to run out-of-the-box.

## What changed

### 1) Multi-objective Pareto front (not just 2D)

Previously, the “Pareto” label was primarily based on a 2D tradeoff (`objective` vs `alpha_weight_turnover_mean`).

Now we compute a **generic Pareto front across multiple objectives**:

- **Proxy stage (train->valid, proxy score):**
  - maximize: `objective`
  - minimize: `alpha_weight_turnover_mean`
  - optionally minimize extra metrics:
    - `turnover_cost_drag_bps_mean`
    - `regime_switch_rate_mean`
    - `fallback_frac_mean`

- **Holdings revalidation stage (train->valid, holdings pricing):**
  - maximize: `holdings_objective`
  - minimize: `alpha_weight_turnover_mean`
  - optionally minimize extra metrics:
    - `turnover_cost_drag_bps_mean`
    - `regime_switch_rate_mean`
    - `fallback_frac_mean`
    - `ensemble_cost_mean`
    - `ensemble_borrow_mean`
    - `ensemble_turnover_mean`

Each row now carries:

- `is_pareto`: `True` if the row is Pareto-efficient under the chosen objective set
- `pareto_rank`: `1` for front points, `0` otherwise (kept simple on purpose)

### 2) Regime tuning presets

You can now set a single flag and get sensible defaults for:

- selection constraints (turnover, cost drag, switch rate, fallback)
- “prefer Pareto” selection
- turnover cost bps (when not explicitly provided)
- default Pareto metrics

CLI:

```bash
--alpha-allocation-regime-tune-preset low_turnover
--alpha-allocation-regime-tune-preset aggressive
--alpha-allocation-regime-tune-preset execution_realistic
```

Presets are **defaults only**: explicit CLI flags still override them.

### 3) Pareto artifacts and optional plots

When regime tuning is enabled, the run now writes additional artifacts:

**Proxy stage**

- `alpha_allocation_regime_param_tuning_pareto.csv`
- `alpha_allocation_regime_param_tuning_pareto_turnover.png`
- `alpha_allocation_regime_param_tuning_pareto_drag.png`
- `alpha_allocation_regime_param_tuning_pareto_plots.json`

**Holdings validation stage**

- `alpha_allocation_regime_holdings_validation_pareto.csv`
- `alpha_allocation_regime_holdings_validation_pareto_turnover.png`
- `alpha_allocation_regime_holdings_validation_pareto_cost.png`
- `alpha_allocation_regime_holdings_validation_pareto_plots.json`

Plot generation is controlled via:

```bash
--alpha-allocation-regime-tune-plots
--no-alpha-allocation-regime-tune-plots
```

Plots require `matplotlib`. If it is not installed, plot generation is skipped (the code still runs).

## New/updated code

- `src/agent/research/constraint_selection.py`
  - `compute_pareto_front()` + `annotate_pareto()` for multi-objective Pareto labeling
- `src/agent/research/regime_tune_presets.py`
  - preset bundles (`low_turnover`, `aggressive`, `execution_realistic`)
- `src/agent/research/pareto_plotting.py`
  - optional matplotlib plotting helpers
- `src/agent/research/alpha_allocation_regime_tuning.py`
  - accepts `pareto_metrics` and annotates multi-objective Pareto
- `src/agent/research/holdings_ensemble.py`
  - fixes missing `constraints/prefer_pareto` wiring in holdings revalidation
  - supports `pareto_metrics` and annotates multi-objective Pareto
- `src/agent/services/experiment_tracking.py`
  - writes Pareto CSV subsets + plot artifacts
- `src/agent/research/reporting.py`
  - surfaces preset, Pareto objectives, and new artifacts in `REPORT.md`

## How to use

Minimal example:

```bash
python main.py \
  --eval-mode p2 \
  --alpha-allocation --holdings-ensemble --alpha-allocation-regime-aware \
  --alpha-allocation-regime-tune \
  --alpha-allocation-regime-tune-preset execution_realistic \
  --save-run
```

Override the Pareto dimensions (adds extra minimization objectives):

```bash
--alpha-allocation-regime-tune-pareto-metrics turnover_cost_drag_bps_mean,regime_switch_rate_mean
```

## Why this matters

In practice, a single “best objective” regime config is rarely what you want.

Multi-objective Pareto + presets make it much easier to select a config that balances:

- out-of-sample performance
- weight stability / turnover
- realistic execution costs
- regime choppiness and fallback behavior
