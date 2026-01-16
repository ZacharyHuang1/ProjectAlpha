# P2.7: Feasibility pre-checks and optimizer diagnostics

This update improves the **constrained optimizer workflow** (P2.5/P2.6) in three ways:

1) A fast **feasibility pre-check** that can explain why a constrained portfolio is impossible *before* we invoke a solver.
2) A lightweight **diagnostics payload** that helps you understand which constraints are binding and what the optimizer traded off.
3) A more consistent **ridge fallback** when QP is unavailable/infeasible (cost-aware score tweaks + best-effort trade limits).

The goal is to reduce “silent” QP failures and to make parameter tuning much more systematic.

---

## What changed

### A) QP feasibility pre-check

We added `qp_feasibility_precheck(...)` in:

- `src/agent/research/optimizer.py`

The pre-check is **conservative**:

- It can prove some setups are impossible.
- It cannot prove feasibility (the solver still determines that).

The pre-check is run **before** attempting QP in `optimize_long_short_weights_with_meta(...)`.

If it fails, we skip QP and go directly to the ridge fallback.

### B) Diagnostics for QP solutions

When QP solves, the meta now includes a `diagnostics` section that summarizes:

- gross exposures and residuals
- turnover to target and whether the turnover cap is binding
- max weight usage and cap binding fraction
- participation cap usage (if enabled)
- exposure summary and slack magnitude (if enabled)
- objective term breakdown (score, penalties, costs)

Diagnostics are **JSON-friendly** and intentionally lightweight.

### C) Better ridge fallback

When we fall back to ridge, we now apply:

- **cost-aware score adjustments** (trade-cost shrink, borrow-cost short penalty)
- best-effort **trade bounds** and **turnover cap** application vs `w_target`

This fallback is still heuristic (no strict guarantees), but it behaves closer to the constrained intent.

---

## Feasibility pre-check rules (high level)

### 1) Max weight cap feasibility

If `max_abs_weight = cap`:

- `n_long * cap >= gross_long` must hold
- `n_short * cap >= gross_short` must hold

Otherwise the gross constraints cannot be satisfied.

### 2) Turnover cap lower bound

We compute a lower bound on the L1 change required to satisfy:

- side signs (longs non-negative, shorts non-positive)
- gross sums (long = `gross_long`, short = `-gross_short`)

If the implied `turnover_lb > optimizer_turnover_cap`, QP is impossible.

### 3) Per-name trade bounds / participation cap reachability

If per-name trade bounds are enabled (via participation cap):

- each name has `|Δw_i| <= max_trade_abs_i`

We check if the target gross sums are within the **reachable sum ranges** implied by those bounds.

### 4) Hard neutrality degrees of freedom (warning)

If strict neutrality is requested (`X^T w = 0`) and the exposure matrix has many columns,
we emit a warning if the degrees of freedom are very low.

---

## Where to look for diagnostics

For each alpha backtest result, look in:

- `result.json` → `coded_alphas[*].backtest_results.walk_forward.construction.optimizer.last`

Key fields:

- `backend_requested` / `backend_used`
- `fallback` (if ridge was used)
- `qp_precheck` (always present when QP was requested)
- `qp_meta` (only when we actually attempted QP)
- `qp_meta.diagnostics` (only when QP solved)
- `ridge_adjustments`, `ridge_limits` (when ridge fallback is used)

---

## Practical tuning tips

If you see `fallback: qp_precheck_failed:turnover_cap_too_small`:

- increase `--optimizer-turnover-cap`
- reduce rebalance frequency
- increase holding period

If you see `cap_long_too_tight` / `cap_short_too_tight`:

- increase `--max-abs-weight`
- increase the candidate set size (`--max-names-per-side`)

If you see frequent `qp_failed:infeasible` and you are enforcing strict neutrality:

- enable slack: `--optimizer-exposure-slack-lambda 50.0` (example)
- relax neutrality (turn off some neutralization flags)

If participation caps bind heavily:

- increase `--impact-max-participation`
- reduce `--portfolio-notional`
- reduce rebalance frequency
