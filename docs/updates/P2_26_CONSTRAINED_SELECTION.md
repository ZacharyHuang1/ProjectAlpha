# P2.26 — Constraint-based selection for regime tuning

## What changed

P2.25 added an interpretable **turnover cost in bps** for regime tuning and marked **Pareto-efficient** configs.

P2.26 turns that into a **controllable selection policy**:

- You can specify simple constraints (max turnover / max drag / max switch-rate / max fallback).
- The tuner picks the **best objective** *among feasible configs*.
- If **no config is feasible**, we fall back to the best objective overall and record the reason.

This applies to:

1) **Proxy regime tuning** (alpha return-stream proxy on `train -> valid`)
2) **Holdings-level revalidation** (Top-N proxy configs re-priced on `train -> valid`)

Both stages now share the same deterministic selection logic.

## New CLI flags

These flags only matter when `--alpha-allocation-regime-tune` is enabled.

Selection constraints:

- `--alpha-allocation-regime-tune-max-alpha-turnover <float>`
  - constraint on `alpha_weight_turnover_mean`
- `--alpha-allocation-regime-tune-max-turnover-cost-drag-bps <float>`
  - constraint on `turnover_cost_drag_bps_mean`
- `--alpha-allocation-regime-tune-max-switch-rate <float>`
  - constraint on `regime_switch_rate_mean`
- `--alpha-allocation-regime-tune-max-fallback-frac <float>`
  - constraint on `fallback_frac_mean`

Optional selection preference:

- `--alpha-allocation-regime-tune-prefer-pareto`
  - if set, we pick the best config **among Pareto-efficient rows** (still respecting constraints)

## Example commands

### 1) Proxy tuning + holdings revalidation + constraints

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-tune \
  --alpha-allocation-regime-tune-turnover-cost-bps 0.2 \
  --alpha-allocation-regime-tune-holdings-top 3 \
  --alpha-allocation-regime-tune-max-alpha-turnover 0.25 \
  --alpha-allocation-regime-tune-max-turnover-cost-drag-bps 5 \
  --alpha-allocation-regime-tune-max-switch-rate 0.15
```

### 2) Prefer Pareto-efficient rows (still constraint-filtered)

```bash
python main.py \
  --eval-mode p2 \
  --alpha-allocation \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-tune \
  --alpha-allocation-regime-tune-prefer-pareto
```

## Output changes

Tuning result tables now include:

- `is_feasible` (bool)
- `constraint_violations` (semicolon-separated reasons)

Tuning summaries include:

- `constraints` (normalized dict)
- `selected_by` (e.g. `constrained_best`, `fallback_unconstrained_no_feasible`)
- `feasible_count`

`REPORT.md` prints the constraints and selection method for both:
- “Regime hyperparam tuning (train->valid)”
- “Regime holdings-level revalidation (train->valid)”

## Implementation notes

- Selection logic lives in `src/agent/research/constraint_selection.py`.
- The same constraints are used for proxy tuning and holdings revalidation.
- No additional data leakage is introduced: all constraints are evaluated on `train -> valid` outputs.
