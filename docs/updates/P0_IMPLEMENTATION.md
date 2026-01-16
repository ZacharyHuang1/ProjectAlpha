# P0 Implementation Notes

This repository originally generated hypotheses and alpha factor *code*, but it did not run those factors on data or evaluate them.

This update implements **P0: a minimal research loop**:

1. Generate a hypothesis (LLM)
2. Generate seed alphas (LLM)
3. Generate factor DSL definitions (LLM)
4. **Run the factor definitions on OHLCV data (DSL-first)**
5. **Evaluate each factor using simple cross-sectional metrics**
6. Select top-K factors as `sota_alphas` and persist results (optional Postgres)

All changes are kept intentionally small so the project remains easy to extend in P1/P2.

See `DSL_GUIDE.md` for DSL syntax and operators.

---

## What was added / changed

### 1) Graph: insert a P0 evaluation node

File: `src/agent/graph.py`

The workflow is now:

`user_input -> hypothesis_generator -> alpha_generator -> alpha_coder -> evaluate_alphas -> persist_state -> END`

The new node (`evaluate_alphas`) runs coded factors on data and computes metrics.

### 2) Factor execution (sandbox)

File: `src/agent/research/factor_runner.py`

Implemented `run_factors(df, coded_alphas)`:

- Executes each `alpha["code"]` in a restricted environment.
- Only a small module allow-list is supported (`pandas`, `numpy`, `math`).
- Validates outputs strictly:
  - must return a `pd.Series` or `pd.DataFrame`
  - output is reindexed to the input index
  - must be single-column with the alpha id as the column name

> **Note:** sandboxing arbitrary Python is *not* perfect. A safer production direction is to switch to a small operator DSL and never `exec` LLM code.

### 3) Alpha evaluation metrics

File: `src/agent/research/alpha_eval.py`

Implemented:

- `compute_forward_returns(close, horizon)` using groupby(instrument).shift(-horizon)
- `evaluate_alpha(factor, forward_returns)` producing:
  - `ic` (mean daily Pearson correlation)
  - `rank_ic_mean` (rank correlation without scipy)
  - `spread_mean` / `spread_std` (top-minus-bottom quantile)
  - `turnover_mean` (top-quantile set turnover)
  - `information_ratio`, `annualized_return`, `max_drawdown`
  - `daily` (JSON-serializable per-day rows)

### 4) Data loading / synthetic data for demos

File: `src/agent/services/market_data.py`

- If `--data-path` is provided, load OHLCV from **csv/parquet/h5**.
- Otherwise generate a small, deterministic synthetic OHLCV panel.

### 5) New agent: evaluate_alphas_agent

File: `src/agent/agents/evaluate_alphas_agent.py`

This agent:

- loads data
- runs all coded factors
- computes P0 metrics
- enriches each `coded_alpha` with `backtest_results`
- selects top-K by `information_ratio` into `sota_alphas`

### 6) CLI updates

File: `main.py`

New flags for P0 evaluation:

- `--data-path` (optional)
- `--horizon`
- `--top-k`
- `--n-quantiles`
- synthetic data controls: `--synthetic-n-days`, `--synthetic-n-instruments`, `--synthetic-seed`

### 7) requirements.txt

File: `requirements.txt`

Added:

- `pandas`
- `numpy`
- `pyarrow` (only needed for parquet loading)

---

## How to run

### A) Minimal demo (no external data, uses synthetic OHLCV)

```bash
pip install -r requirements.txt
python main.py --idea "Volume-conditioned momentum" --thread-id demo1
```

### B) Use your own OHLCV file

```bash
python main.py \
  --idea "Volume-conditioned momentum" \
  --data-path /path/to/ohlcv.csv \
  --horizon 1 \
  --top-k 3
```

CSV format expectation:

- columns: `datetime, instrument, open, high, low, close, volume`

---

## Output you should see

The final graph state will include:

- `seed_alphas`: expressions + descriptions
- `coded_alphas`: each with `code` and `backtest_results`
- `sota_alphas`: top-K selected factors with metrics

If `USE_POSTGRES=true`, `persist_state_agent` also writes Hypothesis/Alpha/BacktestResult rows.

---

## Known limitations (intentional for P0)

- The sandbox is best-effort and is **not** a security boundary.
- The evaluation uses a simple top-minus-bottom spread and a simple turnover definition.
- There is no full portfolio backtest engine yet (that is P1).

---

## Suggested next steps (P1)

- Replace Python `exec` with a controlled DSL + operator library.
- Add a vectorized long/short backtest with costs and constraints.
- Add conditional edges in LangGraph to iterate automatically based on metric improvements.


---

## Hardening update

See `P0_HARDENING.md` for details on data validation and factor code safety checks.


## Evaluation details

The P0 evaluator computes, per date:

- IC (Pearson correlation)
- RankIC (rank-based correlation)
- Top-minus-bottom quantile spread
- A simple turnover proxy based on membership changes

It then aggregates into:

- `information_ratio` (annualized from daily long-short proxy)
- `annualized_return`, `max_drawdown`
- `coverage_mean`, `turnover_mean`

You can control a few knobs via CLI:

- `--cost-bps` (transaction cost applied to turnover)
- `--min-obs-per-day`
- `--min-coverage` / `--max-turnover` quality gates
