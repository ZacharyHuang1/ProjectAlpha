# Alpha-GPT roadmap

This repo implements an **LLM-assisted alpha research loop**:

1) trading idea → 2) hypothesis → 3) seed alphas → 4) safe DSL factors → 5) run + evaluate → 6) select top-K (+ ensembles)

## Done

### P0 / P0.5 — foundations

- [x] Stable OHLCV data contract + validator (`MultiIndex(datetime, instrument)` + `open/high/low/close/volume`)
- [x] Deterministic synthetic data generator for local runs
- [x] Safe factor runner (DSL-first) with AST validation and output checks
- [x] Minimal evaluation (IC, RankIC, spread, turnover proxy, IR, drawdown)
- [x] Optional Postgres persistence (graceful no-op when disabled)
- [x] Guardrails: size limits, NaN checks, code safety checks for legacy Python exec
- [x] Basic tests (graph run, DSL, factor runner, evaluation)

### P1 — research quality

- [x] Train/valid/test splits + walk-forward evaluation
- [x] Portfolio-style long/short backtest with rebalance + holding period (overlap supported)
- [x] Transaction cost model: linear fees + half-spread + ADV impact + borrow constraints + turnover cap (P2.2)
- [x] Robustness checks (per-split stability + yearly breakdown)
- [x] Regime analysis (market volatility / liquidity buckets) (P2.13)
- [x] Decay curves across horizons (multi-horizon evaluation) (P2.15)
- [x] Holding/rebalance schedule sweep for top alphas (P2.16)

### P2 — usability & reproducibility

- [x] Experiment tracking: configs, artifacts, metrics (local JSON/CSV/MD in `runs/<run_id>/`)
- [x] Basic factor registry: append-only JSONL (`runs/factor_registry.jsonl`)
- [x] CLI commands: list / compare / replay
- [x] Optimizer portfolio construction mode (ridge + turnover anchoring)
- [x] Optional constrained optimizer backend (cvxpy QP) + feasibility diagnostics (P2.5–P2.7)
- [x] Risk-aware optimizer objective (diagonal + factor risk model) (P2.8–P2.10)
- [x] Deterministic tuning sweep + cost ablation + extra quality gates (P2.12)
- [x] Execution-only cost sensitivity curves + break-even (P2.14)
- [x] Diversity-aware top-K selection using OOS return correlations (P2.17)
- [x] Equal-weight holdings-level ensemble (trade netting) (P2.18)
- [x] Walk-forward alpha allocation (corr-penalized) + allocated holdings ensemble (P2.19)
- [x] Alpha allocation smoothing (turnover penalty) + meta-tuned allocation hyperparams (P2.20)
- [x] Regime-aware alpha allocation (dynamic weights by market state) (P2.21)
- [x] Selection meta-tuning on validation returns (diverse top-K hyperparams) (P2.22)
- [x] Regime-aware allocation meta-tuning (regime mode/window/buckets + smoothing + turnover penalty) (P2.23)
- [x] Regime tuning holdings-level revalidation for top proxy configs (P2.24)
- [x] Interpretable regime tuning turnover costs (bps) + Pareto flags (P2.25)
- [x] Constraint-based selection of regime configs (turnover/drag/switch/fallback constraints) (P2.26)
- [x] Multi-objective Pareto front + tuning presets + plots for regime tuning (P2.27)
- [x] Automatic knee/utility selection on the Pareto set + stability objectives (P2.28)
- [x] Regime tuning selection report (MD/JSON) + utility-weight auto calibration (P2.29)
- [x] Alpha top-K selection report (MD/JSON/CSV) explaining gating + redundancy (P2.30)
- [x] Alpha selection constraints + presets + selection module consolidation (P2.31)
- [x] DSL formula sanity: volume normalization helpers + cs_neutralize + lint/auto-fix (P2.32)
- [x] DSL group neutralization: sector fixed effects + metadata injection (P2.33)
- [x] Docs cleanup: update notes moved under `docs/updates/`

## Next

### P2.x — research/product improvements

- [ ] Enrich factor registry: tags, lineage, dataset requirements
- [ ] Add a simple “alpha cookbook”: reusable DSL snippets + best practices
- [ ] Add better “alpha diagnostics”: exposures, top contributors, turnover drivers, borrow hot-spots
- [ ] Add a lightweight “dimension/unit” DSL linter (tags + AST propagation) to detect scale-mixing beyond heuristics
- [ ] Add size/sector exposure diagnostics (e.g., corr to log(ADV), sector mean contributions) in `REPORT.md`

### Performance / engineering

- [ ] Faster execution: caching of intermediate series, vectorized ops, optional numba
- [ ] Parallelize independent alpha backtests (local multiprocessing)
- [ ] Add a tiny benchmark suite (runtime + memory budgets)

### P3 — production hardening

- [ ] Isolate factor execution in a true sandbox (separate process + resource limits)
- [ ] Add dataset connectors (Parquet partitions, SQL, object storage)
- [ ] Add monitoring + alerting for scheduled runs
