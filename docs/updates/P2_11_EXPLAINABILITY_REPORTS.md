# P2.11: Explainability, diagnostics, and reports

This update focuses on two practical research pains:

1) **When the constrained optimizer fails or falls back, it should tell you _why_ and _how to fix it_.**
2) **Every run should emit a single markdown report (`REPORT.md`) that explains results in plain English.**

The end state is: you can open one file after a run and immediately understand:

- which alpha “won” and why
- what the **turnover / cost / borrow** profile looks like
- whether any constraints were **binding** (cap, turnover, participation)
- whether the QP failed and what parameter changes would make it feasible

## What changed

### 1) Stronger QP feasibility precheck

`agent.research.optimizer.qp_feasibility_precheck()` now returns:

- `reasons`: machine-readable infeasibility reasons
- `warnings`: soft warnings
- `suggestions`: best-effort “how to fix” hints (some with numeric minimums)

Examples of numeric suggestions:

- `max_abs_weight` minimum required to satisfy gross constraints
- `optimizer_turnover_cap` minimum required to reach the target gross from `w_target`

Examples of structural suggestions:

- disable participation bounds if per-name trade limits make the problem unreachable
- enable exposure slack if hard neutrality has very low degrees-of-freedom

### 2) Unified constraint diagnostics (QP and ridge)

Both backends now expose a shared schema in `optimizer_last_meta["diagnostics"]["constraint_summary"]`:

- `turnover` (cap, used, binding)
- `max_abs_weight` (cap, max_used, binding)
- `participation` (max_ratio, binding)
- `exposure` (max_abs, l2, and optional slack stats)
- `binding_constraints` (a short list for quick scanning)

This makes downstream reporting and debugging consistent across backends.

### 3) Walk-forward cost attribution + optimizer usage stats

`walk_forward_evaluate_factor()` now also computes weighted (by test `n_obs`) out-of-sample means for:

- `cost_mean`, `linear_cost_mean`, `spread_cost_mean`, `impact_cost_mean`
- `borrow_mean`

It also tracks optimizer usage across splits:

- how many times QP vs ridge was used
- fallback reason counts
- the latest optimizer meta from the last test segment

### 4) Automatic run report

`agent.services.experiment_tracking.save_run_artifacts()` now writes:

- `SUMMARY.md` (unchanged)
- `REPORT.md` (new; P2.11)

The report includes:

- best alpha snapshot (IR / return / drawdown / turnover)
- walk-forward stability metrics
- OOS cost attribution (also shown in bps)
- optimizer usage + latest diagnostics
- binding constraint summary and risk attribution (when available)

## Where to look

After a run is saved:

- `runs/<run_id>/SUMMARY.md` — short overview
- `runs/<run_id>/REPORT.md` — debugging-friendly detailed report
- `runs/<run_id>/result.json` — full payload

## Notes

- The report generator is **best-effort**. If report generation fails, run tracking will still succeed.
- Diagnostics are intentionally JSON-friendly (no large arrays) so artifacts stay readable.
