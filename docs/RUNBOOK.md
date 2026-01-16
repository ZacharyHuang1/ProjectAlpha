# Runbook

This document describes how to run Alpha-GPT end-to-end (P0/P1/P2) with a minimal setup.

## 1) Create a virtual environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
```

## 2) Install dependencies

Core runtime:

```bash
pip install -r requirements.txt
```

Optional installs:

```bash
# Parquet support for --data-path
pip install -r requirements-parquet.txt

# Postgres checkpointing + domain persistence
pip install -r requirements-postgres.txt

# Optional: constrained optimizer backend (cvxpy)
pip install -r requirements-optimizer.txt

# Dev/test tools
pip install -r requirements-dev.txt
```

## 3) Configure environment variables (optional)

Copy the example file:

```bash
cp .env.example .env
```

- If `OPENAI_API_KEY` is not set, the project uses deterministic stubs for hypothesis and DSL generation.
- If `USE_POSTGRES=true`, the project attempts to connect to Postgres and persist checkpoints/results.

## 4) Run with synthetic OHLCV (no data required)

```bash
python main.py --idea "Volume-conditioned momentum" --thread-id demo1
```

By default, this runs P2 (walk-forward + risk controls). You can force P0:

```bash
python main.py --idea "demo" --thread-id demo1 --eval-mode p0
```

## 5) Run with your own dataset

Supported file formats:
- CSV (recommended)
- Parquet (requires `requirements-parquet.txt`)

CSV schema:
- columns: `datetime, instrument, open, high, low, close, volume`

Example:

```bash
python main.py \
  --idea "Volume-conditioned momentum" \
  --data-path /path/to/ohlcv.csv \
  --eval-mode p2
```

P2 supports optional risk controls:

- exposure neutralization: `--neutralize-beta`, `--neutralize-liquidity`, `--neutralize-vol`
- volatility targeting: `--target-vol-annual 0.10` (10% annualized)
- sector neutralization: `--neutralize-sector --sector-map-path /path/to/sector.csv`

Sector map file format:

- columns: `instrument, sector`

Note: when a sector map is provided, the DSL runtime exposes a `sector` field. This enables group-neutral DSL patterns like:

```text
cs_demean_group(signal, sector)
cs_neutralize_group(signal, sector, log1p(ts_mean(close*volume, 20)))
```

## 5b) Optional: constrained optimizer backend (P2.5)

By default, `--construction-method heuristic` uses equal-weight top/bottom portfolios.

You can enable the optimizer-based construction (P2.4) and optionally the constrained QP backend (P2.5).

Install cvxpy (optional):

```bash
pip install -r requirements-optimizer.txt
```

Run with QP constraints:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend qp \
  --optimizer-turnover-cap 0.15
```

Auto mode is recommended (tries QP when available, otherwise uses ridge):

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend auto
```

## 5c) P2.13: tuning + cost ablation + regime diagnostics

You can run a small deterministic grid search to tune a few optimizer/backtest knobs and then run a cost ablation for the top alphas.

Two ablation modes are supported:
- `end_to_end`: re-runs walk-forward with different cost settings (the strategy adapts/re-optimizes).
- `execution_only`: keeps the realized trading path fixed and only changes cost deduction.
- `both`: runs end-to-end and also computes the execution-only breakdown (default).

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --tune \
  --tune-turnover-cap "0,0.1,0.2" \
  --tune-max-abs-weight "0,0.02" \
  --ablation-top 1 \
  --ablation-mode both
```

Optional additional quality gates:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --tune \
  --min-ir 0.0 \
  --max-dd 0.30 \
  --max-total-cost-bps 10 \
  --min-wf-splits 3
```

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Optional P2.26 flags (constraint-based selection):

- `--alpha-allocation-regime-tune-max-alpha-turnover <float>` (max `alpha_weight_turnover_mean`)
- `--alpha-allocation-regime-tune-max-turnover-cost-drag-bps <float>` (max `turnover_cost_drag_bps_mean`)
- `--alpha-allocation-regime-tune-max-switch-rate <float>` (max `regime_switch_rate_mean`)
- `--alpha-allocation-regime-tune-max-fallback-frac <float>` (max `fallback_frac_mean`)
- `--alpha-allocation-regime-tune-prefer-pareto` / `--no-alpha-allocation-regime-tune-prefer-pareto` (pick the best among Pareto-efficient rows, still respecting constraints)

Optional P2.27 flags (presets + multi-objective Pareto + plots):

- `--alpha-allocation-regime-tune-preset low_turnover|aggressive|execution_realistic`
- `--alpha-allocation-regime-tune-pareto-metrics <comma-separated metrics>`
  - example: `turnover_cost_drag_bps_mean,regime_switch_rate_mean,fallback_frac_mean`
- `--alpha-allocation-regime-tune-plots` / `--no-alpha-allocation-regime-tune-plots` (requires `matplotlib`)

Optional P2.28 flags (Pareto auto selection):

- `--alpha-allocation-regime-tune-selection-method best_objective|knee|utility`
- `--alpha-allocation-regime-tune-utility-weights "k=v,k=v,..."` (weights are applied in normalized goodness space)
  - example: `objective=1,alpha_weight_turnover_mean=0.4,turnover_cost_drag_bps_mean=0.4,regime_switch_rate_mean=0.2,fallback_frac_mean=0.2`
  - or: `auto` (P2.29) to generate a conservative cost/constraint-aware default
- `--alpha-allocation-regime-tune-include-stability-objectives` / `--no-alpha-allocation-regime-tune-include-stability-objectives`

Additional P2.27 artifacts:

- `runs/<run_id>/alpha_allocation_regime_param_tuning_pareto.csv`
- `runs/<run_id>/alpha_allocation_regime_holdings_validation_pareto.csv`
- `runs/<run_id>/*pareto*.png` (if plots are enabled)

Note: the Pareto CSVs include stability columns (e.g., `objective_split_std`, `objective_split_min`, `holdings_objective_split_std`).

Additional P2.29 artifacts (selection explanation):

- `runs/<run_id>/REGIME_TUNING_REPORT.md` (one-page summary)
- `runs/<run_id>/regime_tuning_selection_report.json` (structured payload)
- `runs/<run_id>/regime_tuning_proxy_top_candidates.csv`
- `runs/<run_id>/regime_tuning_holdings_valid_top_candidates.csv`

Additional P2.30 artifacts (alpha selection explanation):

- `runs/<run_id>/ALPHA_SELECTION_REPORT.md` (one-page summary)
- `runs/<run_id>/alpha_selection_report.json` (structured payload)
- `runs/<run_id>/alpha_selection_top_candidates.csv`

Artifacts:
- `runs/<run_id>/sweep_results.csv`
- `runs/<run_id>/ablation_results.csv`
- `runs/<run_id>/regime_analysis.csv` (if enabled)
- `runs/<run_id>/cost_sensitivity.csv` (if enabled)
- `runs/<run_id>/cost_sensitivity_break_even.csv` (if enabled)
- `runs/<run_id>/REPORT.md`

Optional regime breakdown for the top alpha (enabled by default):

```bash
python main.py --idea "Momentum" --eval-mode p2 --regime-analysis
```

See `docs/updates/P2_13_EXECUTION_ONLY_ABLATION.md`.

## 5d) P2.14: cost sensitivity curves

This is an **execution-only** diagnostic that sweeps cost levels on a grid and reports IR/return/drawdown plus a break-even estimate.

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --cost-sensitivity \
  --cost-sensitivity-top 1
```

Custom sweep grids (comma-separated):

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --cost-sensitivity-linear-bps "0,2,5,10,15,20" \
  --cost-sensitivity-impact-bps "0,25,50,75,100"
```

See `docs/updates/P2_14_COST_SENSITIVITY.md`.

## 5e) P2.15: multi-horizon decay analysis

This diagnostic computes how predictive power and signal persistence change across forward horizons.

It reports, per horizon:

- IC / RankIC mean and t-stat
- top-minus-bottom spread (using forward returns)
- a simple annualized IR proxy scaled by `sqrt(252/horizon)`
- signal persistence: top/bottom set overlap between `t` and `t+h` (Jaccard)

Run:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --decay-analysis \
  --decay-analysis-top 1 \
  --decay-horizons "1,2,5,10,20"
```

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/decay_analysis.csv`
- `runs/<run_id>/REPORT.md` (includes a summary section)

See `docs/updates/P2_15_DECAY_ANALYSIS.md`.

## 5f) P2.16: holding / rebalance schedule sweep

This diagnostic sweeps over `(rebalance_days, holding_days)` for the top alpha(s)
to help choose a practical trading schedule.

Run:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --schedule-sweep \
  --schedule-sweep-rebalance-days "1,2,5,10" \
  --schedule-sweep-holding-days "1,2,5,10,20" \
  --schedule-sweep-max-combos 25
```

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/schedule_sweep.csv`
- `runs/<run_id>/REPORT.md` (includes a summary section)

See `docs/updates/P2_16_HOLDING_REBALANCE_SWEEP.md`.

## P2.17: Diverse selection + alpha ensemble

When `--top-k > 1`, you can optionally select a **less redundant** top-K set using
OOS return correlations, and compute an equal-weight ensemble of the selected
alpha return streams.

Run:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --diverse-selection \
  --diverse-lambda 0.5 \
  --ensemble
```

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/alpha_correlation.csv`
- `runs/<run_id>/ensemble_metrics.json`
- `runs/<run_id>/ensemble_oos_daily.csv`
- `runs/<run_id>/REPORT.md` (includes a summary section)

See `docs/updates/P2_17_DIVERSITY_ENSEMBLE.md`.

## P2.22: Selection meta-tuning (validation)

P2.22 tunes diversified selection hyperparameters using the **validation** segments
from walk-forward (train → valid), and then evaluates the chosen selection on the
**test** segments.

This helps reduce selection overfitting when you want a robust top-K set.

Run:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --selection-tune \
  --selection-tune-metric information_ratio \
  --selection-tune-lambda-grid "0,0.2,0.5,0.8" \
  --selection-tune-candidate-pool-grid "10,20,40"
```

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/selection_tuning_summary.json`
- `runs/<run_id>/selection_tuning_results.csv`
- `runs/<run_id>/REPORT.md` (includes a summary section)

See `docs/updates/P2_22_SELECTION_META_TUNING.md`.

## P2.31: Alpha selection constraints + presets

P2.31 adds explicit **selection constraint knobs** and a few presets. These
constraints are applied during selection (not just reported).

### Preset example

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --selection-tune \
  --alpha-selection-preset low_redundancy \
  --alpha-selection-max-pairwise-corr 0.35
```

### Individual knobs

- `--alpha-selection-max-pairwise-corr 0.4`
- `--alpha-selection-min-valid-ir 0.2` (only used with `--selection-tune`)
- `--alpha-selection-min-valid-coverage 0.7` (only used with `--selection-tune`)
- `--alpha-selection-max-total-cost-bps 10` (test-domain best-effort filter)
- `--alpha-selection-min-wf-test-ir-positive-frac 0.6` (test-domain stability filter)

Artifacts:

- `runs/<run_id>/alpha_selection_report.json`
- `runs/<run_id>/ALPHA_SELECTION_REPORT.md`
- `runs/<run_id>/alpha_selection_top_candidates.csv`

See `docs/updates/P2_31_ALPHA_SELECTION_CONSTRAINTS.md`.

## P2.18: Holdings-level ensemble (portfolio-layer multi-alpha)

P2.17 blends *return streams*. P2.18 blends *holdings*: it averages the executed
portfolio weights from multiple selected alphas and then re-prices the combined
portfolio with the same cost / borrow model. This captures trade netting.

Run:

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --diverse-selection \
  --ensemble \
  --holdings-ensemble
```

Optional (usually keep off for clean comparisons):

- `--holdings-ensemble-apply-turnover-cap`

Artifacts:

- `runs/<run_id>/ensemble_holdings_metrics.json`
- `runs/<run_id>/ensemble_holdings_oos_daily.csv`
- `runs/<run_id>/ensemble_holdings_positions.csv`
- `runs/<run_id>/REPORT.md` (includes a summary section)

See `docs/updates/P2_18_HOLDINGS_ENSEMBLE.md`.

## P2.19/P2.20: Walk-forward alpha allocation (allocated ensemble)

P2.19 learns **alpha weights** (across the selected alphas) on each walk-forward split
using in-sample strategy returns and a redundancy penalty.

P2.20 adds:

- a smoothing penalty on changes in alpha weights across splits (`turnover_lambda`)
- an optional meta-tuning step (train → valid) to pick one best allocation config

### Run (fixed allocation)

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-lambda 0.5 \
  --alpha-allocation-max-weight 0.8 \
  --alpha-allocation-turnover-lambda 1.0
```

### Run (meta-tune allocation hyperparams)

```bash
python main.py \
  --eval-mode p2 \
  --top-k 5 \
  --alpha-allocation \
  --alpha-allocation-tune \
  --alpha-allocation-tune-lambda-grid "0,0.2,0.5,0.8" \
  --alpha-allocation-tune-max-weight-grid "0.5,0.8,1.0" \
  --alpha-allocation-tune-turnover-lambda-grid "0,0.5,2"
```

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/alpha_allocations.csv` (per-split alpha weights)
- `runs/<run_id>/alpha_allocation_tuning_results.csv` (meta-tuning table)
- `runs/<run_id>/ensemble_holdings_allocated_oos_daily.csv`
- `runs/<run_id>/REPORT.md`

## P2.21: Regime-aware alpha allocation (dynamic weights)

P2.21 extends the alpha allocation layer by learning **separate alpha weights per market regime** (e.g., low/med/high volatility). During the OOS test segment, the system selects the corresponding weights based on the current regime.

This is opt-in.

### Run (regime-aware allocation)

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

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/ensemble_holdings_allocated_regime_oos_daily.csv`
- `runs/<run_id>/ensemble_holdings_allocated_regime_positions.csv`
- `runs/<run_id>/alpha_allocations_regime.csv`
- `runs/<run_id>/alpha_allocation_regime_diagnostics.csv`

See `docs/updates/P2_21_REGIME_AWARE_ALLOCATION.md`.

## P2.23: Regime-aware allocation meta-tuning (regime params)

P2.23 adds a **validation meta-tuning** loop for the regime-aware allocation layer.
It searches over regime labeling hyperparams (mode/window/buckets) and alpha-weight
smoothing, and can optionally apply an **interpretable bps cost** on **alpha-weight turnover** when ranking configs.

P2.24 adds an optional second pass: it revalidates the **top-N proxy configs** using the
**holdings-level** blended portfolio on the valid segments (capturing trade netting + costs),
and can override the final winner.


### Run (meta-tune regime hyperparams)

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

Note: `--alpha-allocation-regime-tune-turnover-penalty` still works as an alias but is deprecated.

Optional P2.24 flags:

- `--alpha-allocation-regime-tune-holdings-top 3` (0 disables)
- `--alpha-allocation-regime-tune-holdings-metric information_ratio|annualized_return` (empty uses the proxy metric)
- `--alpha-allocation-regime-tune-holdings-save-top 10` (0 keeps all)

Artifacts:

- `runs/<run_id>/alpha_allocation_regime_param_tuning_results.csv`
- `runs/<run_id>/alpha_allocation_regime_param_tuning_summary.json`
- `runs/<run_id>/alpha_allocation_regime_holdings_validation_results.csv` (P2.24, if enabled)
- `runs/<run_id>/alpha_allocation_regime_holdings_validation_summary.json` (P2.24, if enabled)

See `docs/updates/P2_23_REGIME_META_TUNING.md`.

## 5g) Cost-aware constrained optimizer (P2.6)

If you enable the P2.2 execution model (spread / impact / borrow), you can also include convex cost terms directly inside the QP objective via:

- `--optimizer-cost-aversion` (0 disables)
- `--optimizer-enforce-participation` / `--no-optimizer-enforce-participation`
- `--optimizer-exposure-slack-lambda` (recommended when constraints are tight)

Example:

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend qp \
  --optimizer-turnover-cap 0.15 \
  --optimizer-cost-aversion 1.0 \
  --half-spread-bps 5 \
  --impact-bps 50 \
  --impact-exponent 0.5 \
  --impact-max-participation 0.2
```

See `docs/updates/P2_6_COST_AWARE_OPTIMIZER.md` for details.

## 5h) Feasibility pre-checks and diagnostics (P2.7)

When you request the QP backend (`--optimizer-backend qp|auto`), the system runs a fast feasibility
pre-check (cap / turnover / trade bounds) before calling a solver. If it detects an impossible setup,
it will skip QP and fall back to the ridge backend.

To inspect *why* QP fell back or which constraints were binding, open the run artifact:

- `runs/<run_id>/result.json` → `construction.optimizer.last`

See `docs/updates/P2_7_FEASIBILITY_AND_DIAGNOSTICS.md` for details.


## 5i) Diagonal risk penalty (P2.8)

You can add a diagonal risk term inside the optimizer objective to reduce concentration in
high-variance names (works for both QP and ridge fallback):

- `--optimizer-risk-aversion` (0 disables)
- `--optimizer-risk-window` (rolling window in days for the variance proxy)

The optimizer penalizes `sum(var * w^2)` where `var` is an **annualized** variance proxy computed
from rolling close-to-close returns (shifted by 1 day).

Diagnostics are stored in `runs/<run_id>/result.json` under:
`construction.optimizer.last.qp_meta.diagnostics` → `risk_proxy`, `risk_vol_annual_proxy`,
and `risk_top_contributors`.

See `docs/updates/P2_8_DIAGONAL_RISK.md`.

## 5j) Factor risk model (P2.9)

You can replace the diagonal variance proxy with a simple factor risk model:

- `--optimizer-risk-model factor`
- `--optimizer-factor-risk-window` (lookback window, days)
- `--optimizer-factor-risk-shrink` (fixed diagonal shrink 0..1)
- `--optimizer-factor-risk-ridge` (ridge used in cross-sectional regressions)

Example:

```bash
python main.py \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-risk-model factor \
  --optimizer-risk-aversion 5 \
  --optimizer-factor-risk-window 60
```

See `docs/updates/P2_9_FACTOR_RISK_MODEL.md`.

## 5k) Robust risk estimation (P2.10)

P2.10 adds more robust covariance estimation and attribution:

- `--optimizer-factor-risk-estimator sample|ewm`
- `--optimizer-factor-risk-shrink-method fixed|oas`
- `--optimizer-factor-risk-ewm-halflife`
- `--optimizer-factor-return-clip-sigma`
- `--optimizer-idio-clip-q`
- `--optimizer-idio-shrink`

Example (recommended):

```bash
python main.py \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-risk-model factor \
  --optimizer-risk-aversion 5 \
  --optimizer-factor-risk-estimator ewm \
  --optimizer-factor-risk-ewm-halflife 20 \
  --optimizer-factor-risk-shrink-method oas
```

See `docs/updates/P2_10_ROBUST_RISK_ESTIMATION.md`.

## 6) Enable the optional research loop

```bash
python main.py \
  --idea "Momentum + volume" \
  --max-iterations 3 \
  --target-ir 0.5
```

## 7) Troubleshooting

### "No SOTA alphas selected"

This usually means:
- The factor output is mostly NaNs (fails the NaN guardrail), or
- The universe is too small for quantiles, or
- The dataset is too short for walk-forward splits (P1 fallback will handle short data, but extremely short data may still be uninformative).

Try:
- Increase `--synthetic-n-instruments` and `--synthetic-n-days`, or
- Lower `--min-obs-per-day`, or
- Use `--eval-mode p0` to sanity check first.

### Walk-forward fallback

If the dataset is too short for walk-forward (P1/P2), the evaluator falls back to P0 and sets `mode = p0_fallback`.

## 8) Local experiment tracking (P2.3)

By default, `main.py` writes artifacts to:

- `runs/<run_id>/`

You can disable saving with:

```bash
python main.py --save-run false
```

Useful utilities (wrappers around `agent.tools.*`):

```bash
python list_runs.py --runs-root runs --limit 20
python compare_runs.py runs/<runA> runs/<runB>
python replay_run.py runs/<run_id>
```

See `docs/updates/P2_3_EXPERIMENT_TRACKING.md`.

## 9) Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 10) Optional: enable costs + borrow constraints

```bash
python main.py \
  --idea "Momentum" \
  --eval-mode p2 \
  --commission-bps 1 \
  --slippage-bps 2 \
  --half-spread-bps 5 \
  --impact-bps 50 \
  --impact-exponent 0.5 \
  --portfolio-notional 1000000
```

Hard-to-borrow list and borrow rates:

```bash
python main.py \
  --idea "Value" \
  --eval-mode p2 \
  --hard-to-borrow-path ./hard_to_borrow.csv \
  --borrow-rates-path ./borrow_rates.csv \
  --max-borrow-bps 500
```

See `docs/updates/P2_2_IMPLEMENTATION.md` for file formats.
