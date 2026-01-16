# P2.30 — Alpha selection report

This update extends the “selection report” pattern (introduced for regime tuning in P2.29)
to the **top-K alpha selection** stage.

The goal is to make the final alpha set explainable:

- which alphas were selected
- which candidates were excluded
- whether exclusions came from **quality gates** (coverage/turnover/cost/stability) or **redundancy** (high correlation)
- what selection method was used (score-rank vs diverse greedy vs validation meta-tune)

## What changed

### 1) New report builder

Added a new module:

- `src/agent/research/alpha_selection_report.py`

It builds a structured report (`dict`) and a one-page markdown summary.

The report is intentionally **best-effort** and will not fail a run if inputs are missing.

### 2) Experiment tracker now exports alpha selection artifacts

Updated `src/agent/services/experiment_tracking.py` to write:

- `runs/<run_id>/ALPHA_SELECTION_REPORT.md`
- `runs/<run_id>/alpha_selection_report.json`
- `runs/<run_id>/alpha_selection_top_candidates.csv`

The CSV is a compact “top candidates” view (includes gate status, correlation-to-selected diagnostics when available, and a `selected` flag).

### 3) Tests

- Added `tests/unit_tests/test_alpha_selection_report.py`
- Updated `tests/unit_tests/test_experiment_tracking.py` to assert the new artifacts are created

## How the report explains selection

The report uses the following logic:

1. **Candidate scoring**: uses `information_ratio` from each alpha’s backtest results.
2. **Quality gates**: uses the existing `quality_gate` payload (passed/reasons).
3. **Redundancy diagnostics**: when OOS daily returns are available, it computes correlation and annotates:
   - `avg_abs_corr_to_selected`
   - `max_abs_corr_to_selected`
   - `diversified_score = IR - lambda * avg_abs_corr_to_selected`
4. **Selection trace**: if the selection method produces a `selection_table` (diverse greedy or selection meta-tune), it is rendered in the markdown.

## Notes / limitations

- Correlation diagnostics are computed on a **bounded candidate pool** (default: top 40 passing-gate candidates) to keep report generation fast.
- The “diversified_score” is a diagnostic and may not exactly match the greedy stepwise objective if the selection was meta-tuned.
- The report does not introduce new gating knobs; it explains the gates that already exist.

## Suggested next upgrade

If you want selection to be more “production-like”, the natural next step is:

- add explicit selection constraints (max corr / min stability / max total cost) and expose them as CLI/config knobs
- add small presets (e.g., `low_redundancy`, `low_cost`, `stable_generalization`)
