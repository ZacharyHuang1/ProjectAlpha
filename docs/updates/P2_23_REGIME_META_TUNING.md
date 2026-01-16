# P2.23 — Regime-aware allocation meta-tuning (validation + turnover cost)

This update extends the **regime-aware alpha allocation** layer (P2.21) with a
**walk-forward validation meta-tuning** step.

The goal is to remove the “pick regime params by hand” workflow and to make the
regime-aware ensemble more *tradeable* by optionally penalizing **alpha-weight turnover**
when selecting hyperparameters.

**Note (P2.25):** the "turnover penalty" is now interpreted as an **additional cost in bps**
applied to the validation return series (see `docs/updates/P2_25_BPS_OBJECTIVE_PARETO.md`).

## What changed

### 1) New meta-tuning loop for regime hyperparameters

A new module was added:

- `src/agent/research/alpha_allocation_regime_tuning.py`

It evaluates candidate configurations across **walk-forward splits** using:

- **fit segment**: `train`
- **score segment**: `valid`

For each split and each candidate config:

1. Compute market regime labels (fit thresholds on `train`, apply to `train` + `valid`).
2. Fit **global alpha weights** + **per-regime alpha weights** on `train`.
3. Build **daily alpha weights** on `valid` (regime lookup + optional smoothing).
4. Score the resulting **return-stream ensemble** on `valid`.
5. Track **alpha-weight turnover** (daily `0.5 * sum(|w_t - w_{t-1}|)`).

Finally we pick the config that maximizes:

- Apply a turnover cost in bps: `r_adj[t] = r[t] - alpha_turnover[t] * turnover_cost_bps / 10000`
- Then rank by: `objective = metric(r_adj)`

Notes:

- This is **leakage-safe**: `valid` is never used to fit thresholds or weights.
- The tuning objective uses a **return-stream proxy** (fast). The final evaluation
  still runs the **holdings-level** ensemble on the OOS test segments.

### 2) Regime-aware CLI flags are now fully wired

`main.py` now forwards the regime-aware configuration keys into the agent config.
(Previously the flags existed but were not passed into the run config.)

### 3) New artifacts

When regime tuning is enabled, runs now export:

- `runs/<run_id>/alpha_allocation_regime_param_tuning_results.csv`
- `runs/<run_id>/alpha_allocation_regime_param_tuning_summary.json`

These are separate from the **allocation hyperparam** tuning artifacts:

- `runs/<run_id>/alpha_allocation_regime_tuning_results.csv`
- `runs/<run_id>/alpha_allocation_regime_tuning_summary.json`

## How to use

### Enable regime-aware allocation

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-mode vol \
  --alpha-allocation-regime-window 20 \
  --alpha-allocation-regime-buckets 3 \
  --alpha-allocation-regime-min-days 30 \
  --alpha-allocation-regime-smoothing 0.10
```

### Meta-tune regime hyperparameters (train->valid)

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-tune \
  --alpha-allocation-regime-tune-mode-grid "vol,vol_liq" \
  --alpha-allocation-regime-tune-window-grid "10,20,40" \
  --alpha-allocation-regime-tune-buckets-grid "2,3,4" \
  --alpha-allocation-regime-tune-smoothing-grid "0,0.05,0.10" \
  --alpha-allocation-regime-tune-turnover-cost-bps 5
```

### Recommended “most realistic” setup

Run allocation hyperparam tuning once, reuse its chosen params, then tune regime params:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-tune \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-tune
```

## Implementation notes

- The tuning loop is intentionally small and deterministic (grid + max combos cap).
- The turnover cost is optional; keep it at `0.0` to rank purely by `valid_metric`.
- Smoothing is applied on **alpha weights** (not on holdings), which reduces unnecessary
  daily mixing between alphas.
