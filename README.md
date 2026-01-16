## Project status

This repository is under active development. The current codebase implements:

### P0: fast factor sanity checks

- generate -> code -> run factors -> evaluate -> select top-K
- metrics include: IC / RankIC / quantile spread / turnover (see `docs/updates/P0_IMPLEMENTATION.md`)

### P1: walk-forward research backtest

- walk-forward splits: train/valid/test (sequential)
- portfolio-style long/short backtest with rebalance + holding period
- cost model: linear fees + optional half-spread and nonlinear impact (ADV-based participation) + optional borrow cost
- out-of-sample aggregation across test segments + basic stability breakdown

See `docs/updates/P1_IMPLEMENTATION.md`.

### P2: risk controls (research-grade)

- liquidity filter (ADV)
- exposure neutralization: beta / volatility / liquidity / sector
- simple volatility targeting overlay

See `docs/updates/P2_IMPLEMENTATION.md`.

### P2.2: execution realism (costs + borrow)

- half-spread + nonlinear impact (ADV participation model)
- hard-to-borrow / per-instrument borrow rates + optional max borrow threshold
- optional turnover cap at rebalance/expiry transitions

See `docs/updates/P2_2_IMPLEMENTATION.md`.

### P2.3: experiment tracking (local)

- local run artifacts: `runs/<run_id>/` (config + outputs + metrics table + summary)
- lightweight factor registry: `runs/factor_registry.jsonl`
- simple CLI tools: list / compare / replay

See `docs/updates/P2_3_EXPERIMENT_TRACKING.md`.

### P2.4: optimizer portfolio construction

- optional optimizer-based weights (ridge + turnover anchoring)
- optional soft exposure penalty (uses the same exposure set as P2 neutralization)

See `docs/updates/P2_4_OPTIMIZER_MODE.md`.

### P2.5: constrained optimizer backend (optional)

- optional QP backend (cvxpy) with hard constraints (gross, bounds, turnover, neutrality)
- graceful fallback to ridge when cvxpy is unavailable or infeasible

See `docs/updates/P2_5_CONSTRAINED_OPTIMIZER.md`.


### P2.6: cost-aware constrained optimizer (optional)

- include linear trade costs, borrow drag, and a convex impact proxy directly in the QP objective
- optional per-name participation bounds (|Δw| <= max_participation * ADV / notional)
- optional exposure slack (to avoid infeasible problems while staying near-neutral)

See `docs/updates/P2_6_COST_AWARE_OPTIMIZER.md`.

### P2.7: feasibility pre-checks and diagnostics

- fast constrained optimizer pre-checks (cap / turnover / trade bounds)
- lightweight diagnostics payload for QP solutions
- more consistent ridge fallback (cost-aware + best-effort trade limits)

See `docs/updates/P2_7_FEASIBILITY_AND_DIAGNOSTICS.md`.

### P2.8: diagonal risk penalty (variance proxy)

- optional diagonal risk term inside optimizer construction (QP + ridge fallback)
- risk proxy estimated from rolling return variance (lookahead-safe, shifted by 1 day)
- diagnostics include risk proxy and top per-name risk contributors

See `docs/updates/P2_8_DIAGONAL_RISK.md`.



### P2.9: factor risk model (B F B^T + D)

- optional factor risk model term inside optimizer construction (QP + ridge fallback)
- estimates factor covariance from trailing cross-sectional regressions (lookahead-safe)
- falls back to the diagonal variance proxy when history is insufficient

See `docs/updates/P2_9_FACTOR_RISK_MODEL.md`.


P2.10: robust risk estimation + attribution

- supports EWM factor covariance estimation
- supports OAS shrinkage (automatic shrink to identity) for factor covariance
- winsorizes factor returns and clips/shrinks idiosyncratic variance
- adds factor + idiosyncratic risk attribution in optimizer diagnostics

See `docs/updates/P2_10_ROBUST_RISK_ESTIMATION.md`.


### P2.11: explainability + run reports

- stronger QP feasibility precheck with actionable suggestions
- unified constraint diagnostics schema across QP and ridge
- automatic `REPORT.md` artifact per run (cost attribution, constraint binding, optimizer usage)

See `docs/updates/P2_11_EXPLAINABILITY_REPORTS.md`.


### P2.13: execution-only cost ablation + regime diagnostics

- optional deterministic grid-search tuning for optimizer/backtest knobs
- cost ablation for top alphas with two modes:
  - `end_to_end`: re-optimizes
  - `execution_only`: fixed trades, cost components toggled
- optional regime breakdown for the best alpha
- additional quality gates (min IR / max drawdown / max total cost / min WF splits)
- new run artifacts: `sweep_results.csv`, `ablation_results.csv`, and improved `REPORT.md`

See `docs/updates/P2_13_EXECUTION_ONLY_ABLATION.md`.


### P2.14: cost sensitivity curves

- execution-only cost sensitivity curves (sweep bps levels)
- break-even estimates for key cost knobs (linear / spread / impact / borrow)
- new run artifacts: `cost_sensitivity.csv` and `cost_sensitivity_break_even.csv`

See `docs/updates/P2_14_COST_SENSITIVITY.md`.


### P2.15: multi-horizon decay analysis

- multi-horizon factor diagnostics for the top alphas (IC / RankIC / spread)
- signal persistence proxy via top/bottom set overlap (Jaccard) across horizon lags
- new run artifact: `decay_analysis.csv` and a new section in `REPORT.md`

See `docs/updates/P2_15_DECAY_ANALYSIS.md`.


### P2.16: holding / rebalance schedule sweep

- strategy-level sweep over `(rebalance_days, holding_days)` for the top alpha(s)
- helps choose a schedule that trades off turnover/cost vs signal decay
- new run artifact: `schedule_sweep.csv` + a new section in `REPORT.md`

See `docs/updates/P2_16_HOLDING_REBALANCE_SWEEP.md`.


P2.17: diversity-aware top-K selection + simple alpha ensemble

- compute OOS return correlations across candidate alphas
- greedy diversified selector to reduce redundancy in the top-K set
- equal-weight ensemble of selected alpha OOS return streams
- new run artifacts: `alpha_correlation.csv`, `ensemble_metrics.json`, `ensemble_oos_daily.csv`

See `docs/updates/P2_17_DIVERSITY_ENSEMBLE.md`.


### P2.18: holdings-level ensemble (portfolio-layer multi-alpha)

- builds a combined portfolio by averaging *holdings* (weights) from multiple selected alphas
- re-prices the combined portfolio with the same cost / borrow model to capture trade netting
- new run artifacts: `ensemble_holdings_metrics.json`, `ensemble_holdings_oos_daily.csv`, `ensemble_holdings_positions.csv`

See `docs/updates/P2_18_HOLDINGS_ENSEMBLE.md`.


### P2.19: alpha allocation (walk-forward) + allocated holdings ensemble

- learns non-negative alpha weights per split (correlation-penalized) using only pre-test data
- blends holdings using the learned weights and re-prices the combined portfolio (netting-aware costs)
- new run artifacts: `ensemble_holdings_allocated_metrics.json`, `ensemble_holdings_allocated_oos_daily.csv`, `ensemble_holdings_allocated_positions.csv`, `alpha_allocations.csv`

See `docs/updates/P2_19_ALPHA_ALLOCATION.md`.


### P2.20: allocation smoothing + meta-tuned allocation hyperparameters

- adds an alpha-weight smoothing penalty across splits (`turnover_lambda`)
- adds a deterministic meta-tuning step (train → valid) to pick one best allocation config
- new run artifacts: `alpha_allocation_tuning_results.csv`, `alpha_allocation_tuning_summary.json`

See `docs/updates/P2_20_ALPHA_ALLOCATION_META_TUNING.md`.


### P2.21: regime-aware alpha allocation (dynamic weights)

- optional dynamic alpha weights conditioned on market regime (e.g., volatility or volatility×liquidity)
- thresholds are fit on the fit segment only (train or train+valid) to avoid leakage
- optional daily smoothing for regime-driven weight switching
- new run artifacts: `ensemble_holdings_allocated_regime_oos_daily.csv`, `alpha_allocations_regime.csv`, `alpha_allocation_regime_diagnostics.csv`

See `docs/updates/P2_21_REGIME_AWARE_ALLOCATION.md`.




## DSL

This repo prefers a small operator DSL for factor definitions. See `docs/DSL_GUIDE.md`.

---

## Research acknowledgments

- [Alpha-GPT: Human-AI Interactive Alpha Mining for Quantitative Investment](https://arxiv.org/pdf/2308.00016)
- [Alpha-GPT 2.0: Human-in-the-Loop AI for Quantitative Investment](https://arxiv.org/pdf/2402.09746v1)

---

## How to run

1) Install dependencies

```bash
pip install -r requirements.txt
cp .env.example .env  # optional; add OPENAI_API_KEY to enable real LLM calls
```

Optional installs:

```bash
# Parquet support for --data-path
pip install -r requirements-parquet.txt

# Postgres checkpointing + domain persistence (USE_POSTGRES=true)
pip install -r requirements-postgres.txt

# Optional: constrained optimizer backend (cvxpy)
pip install -r requirements-optimizer.txt

# Dev/test tools
pip install -r requirements-dev.txt

```

2) Run (synthetic OHLCV by default)

```bash
python main.py --idea "Volume-conditioned momentum" --thread-id demo1
```

3) Run with your own dataset

```bash
python main.py \
  --idea "Volume-conditioned momentum" \
  --data-path /path/to/ohlcv.csv \
  --eval-mode p2 \
  --top-k 3
```

### Evaluation knobs

P0 (forward-return proxy):

- `--cost-bps`: apply a simple turnover-based transaction cost to the long-short proxy return.
- `--min-obs-per-day`: skip days with too few instruments.
- `--min-coverage` / `--max-turnover`: optional quality gates (set to 0 / 1 to disable).

### Iteration loop (optional)

By default the graph runs **one** iteration. You can enable a small loop:

- `--max-iterations 3`: run up to 3 iterations in one command
- `--target-ir 0.5`: stop early once the best factor reaches the target information ratio

### Output

- `--output-json /tmp/run.json`: save the full result payload (including daily metrics).
- By default, the runner also saves local artifacts to `runs/<run_id>/`.
  You can disable this with `--save-run false`.

CSV format expectation:

- columns: `datetime, instrument, open, high, low, close, volume`

---

## Documents

- `docs/updates/P0_IMPLEMENTATION.md`: what P0 does, how to run, output schema
- `docs/updates/P0_HARDENING.md`: guardrails (data validation + factor code safety checks)
- `docs/updates/P0_PLAN.md`: the original P0 MVP plan (now implemented)
- `docs/updates/P1_IMPLEMENTATION.md`: walk-forward research backtest (P1)
- `docs/updates/P2_IMPLEMENTATION.md`: risk controls + constraints (P2)
- `docs/updates/P2_2_IMPLEMENTATION.md`: costs + borrow constraints (P2.2)
- `docs/updates/P2_3_EXPERIMENT_TRACKING.md`: experiment tracking (local artifacts)
- `docs/updates/P2_4_OPTIMIZER_MODE.md`: optimizer portfolio construction
- `docs/updates/P2_5_CONSTRAINED_OPTIMIZER.md`: constrained optimizer backend (optional QP)
- `docs/updates/P2_17_DIVERSITY_ENSEMBLE.md`: diverse selection + return-stream ensemble
- `docs/updates/P2_18_HOLDINGS_ENSEMBLE.md`: holdings-level ensemble (trade netting)
- `docs/updates/P2_19_ALPHA_ALLOCATION.md`: walk-forward alpha allocation + allocated holdings ensemble
- `docs/updates/P2_20_ALPHA_ALLOCATION_META_TUNING.md`: allocation smoothing + meta-tuning
- `docs/TODO.md`: roadmap for P1/P2/P3
