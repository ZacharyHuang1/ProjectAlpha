# P2.29 — Regime tuning selection report + utility-weight auto calibration

This update focuses on **making the regime-tuning decision explainable and reproducible**.

## What changed

### 1) One-page regime selection report

New artifacts are written to every run that uses regime tuning:

- `runs/<run_id>/REGIME_TUNING_REPORT.md` — a compact, human-readable summary
- `runs/<run_id>/regime_tuning_selection_report.json` — a structured payload for programmatic use
- `runs/<run_id>/regime_tuning_proxy_top_candidates.csv` — top candidates from the proxy-stage selection set
- `runs/<run_id>/regime_tuning_holdings_valid_top_candidates.csv` — top candidates from the holdings revalidation stage (if enabled)

The report explains:

- which stage was ultimately used (`proxy` vs `holdings_valid`)
- which selection method was used (`best_objective`, `knee`, or `utility`)
- how constraints and Pareto preference affected the candidate set
- the chosen regime config and its key trade-offs
- constraint slack for the chosen config (positive means inside the bound)

### 2) Richer selection metadata preserved in results

`walk_forward_holdings_ensemble_allocated_regime()` now keeps the following in:

- `result['ensemble_holdings_allocated_regime']['regime_tuning']`
- `result['ensemble_holdings_allocated_regime']['regime_tuning_holdings_validation']`

so it can be exported by the tracker:

- `constraints`
- `selection` (includes the objectives + normalized weights actually used)
- `selected_by`
- Pareto objective list and counts

### 3) Utility weights: `auto`

If you run with:

```bash
--alpha-allocation-regime-tune-selection-method utility \
--alpha-allocation-regime-tune-utility-weights auto
```

the CLI will generate a **deterministic, cost/constraint-aware** utility-weight map.

Two extra config keys are stored for traceability:

- `alpha_allocation_regime_tune_utility_weights_source`
- `alpha_allocation_regime_tune_utility_calibration_meta`

## Files added / updated

- `src/agent/research/utility_weight_calibration.py`
- `src/agent/research/regime_tuning_selection_report.py`
- `src/agent/research/constraint_selection.py` (selection meta now includes objectives + weights used)
- `src/agent/research/holdings_ensemble.py` (keeps selection metadata in the returned summaries)
- `src/agent/services/experiment_tracking.py` (writes the new artifacts)
- `docs/RUNBOOK.md` (documents the new flags and artifacts)
- `docs/TODO.md` (marks P2.29 complete)

## Next upgrade ideas

- Extend the same “selection report” pattern to alpha top-K selection (diversity gating) and ensemble construction.
- Add a small optional “weight sweep” for utility selection (coarse grid) to validate that the chosen config is stable vs weight choices.
- Include a tiny “why not” section: a few Pareto candidates that were rejected specifically due to constraint violations.
