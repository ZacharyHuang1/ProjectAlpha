# P2.31 — Alpha selection constraints + presets (and repo cleanup)

This update improves the **top-K alpha selection stage** by adding explicit, user-facing constraint knobs and a few practical presets, and then **consolidates redundant selection code** into a single module.

## What I added

### 1) Selection constraints (hard filters)

You can now enforce hard constraints during alpha selection:

- `max_pairwise_corr`: a hard cap on the maximum absolute correlation between any selected alpha and a new candidate.
- `min_valid_ir`: minimum validation IR (only used when `--selection-tune` is enabled).
- `min_valid_coverage`: minimum validation coverage ratio (only used when `--selection-tune` is enabled).
- `max_total_cost_bps`: maximum total cost drag in bps (test-domain best-effort filter).
- `min_wf_test_ir_positive_frac`: minimum walk-forward test IR positive fraction (test-domain stability filter).

The constraints are applied **inside the selector** (not just reported).

### 2) Selection presets

A preset is a convenience shortcut that sets recommended constraint defaults.

Supported presets:

- `low_redundancy` → focuses on correlation caps
- `low_cost` → focuses on cost drag caps
- `stable_generalization` → focuses on validation IR/coverage
- `aggressive` → no extra constraints

CLI-provided explicit constraint values override the preset defaults.

### 3) Better selection explainability

The alpha selection report now includes:

- the active preset + constraint settings
- a summary of which candidates violated which constraints
- correlation-to-selected diagnostics on the relevant domain (`valid` vs `test`)

Artifacts:

- `runs/<run_id>/alpha_selection_report.json`
- `runs/<run_id>/ALPHA_SELECTION_REPORT.md`
- `runs/<run_id>/alpha_selection_top_candidates.csv`

## Important behavior details

### Validation meta-tuning (`--selection-tune`)

When `--selection-tune` is enabled, selection is performed on **validation returns** (train → valid), and test data is only used for evaluation. In this mode:

- `max_pairwise_corr`, `min_valid_ir`, `min_valid_coverage` are the primary constraints used for selection.
- test-domain constraints (`max_total_cost_bps`, `min_wf_test_ir_positive_frac`) are reported, but they are **not used** to drive the validation-domain selection decision.

### Non-tuned selection

When `--selection-tune` is disabled, selection is performed on **test OOS returns** and test-domain constraints apply normally.

## Repo cleanup (less redundancy)

To make the project more maintainable, I consolidated selection-related utilities:

- **New**: `src/agent/research/alpha_selection.py`
  - return matrix helpers (valid + OOS)
  - correlation estimation
  - greedy diversified selection (now supports hard constraints)
  - validation meta-tuning of selection hyperparameters
  - alpha selection report generation

Removed redundant modules:

- `src/agent/research/diversity_ensemble.py`
- `src/agent/research/selection_tuning.py`
- `src/agent/research/alpha_selection_report.py`

I also removed a tiny standalone projection helper module and inlined it:

- deleted `src/agent/research/projections.py`
- moved `project_to_bounded_simplex(...)` into `src/agent/research/alpha_allocation.py`

## New CLI flags

- `--alpha-selection-preset <name>`
- `--alpha-selection-max-pairwise-corr <float>`
- `--alpha-selection-min-valid-ir <float>`
- `--alpha-selection-min-valid-coverage <float>`
- `--alpha-selection-max-total-cost-bps <float>`
- `--alpha-selection-min-wf-test-ir-positive-frac <float>`

Example:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --selection-tune \
  --alpha-selection-preset low_redundancy \
  --alpha-selection-max-pairwise-corr 0.35
```

## Next upgrade ideas

- Add validation-domain cost decomposition (so `max_total_cost_bps` can be applied without test leakage).
- Add selection-time multi-objective selection (Pareto/utility) similar to regime-tuning selection.
- Add an automatic “minimum effective K” policy: if constraints prevent selecting K, suggest relaxations.
