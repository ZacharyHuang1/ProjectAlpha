# P0 MVP: Factor Runner + Alpha Evaluation (Minimum Viable Research Loop)

Goal: turn "LLM-generated alpha code" into **computed factors** and produce a minimal set of **quantitative evaluation metrics**
(IC / RankIC / quantile spread / turnover). This closes the loop: **generate -> run -> evaluate -> select**.

Scope note: P0 is a **research loop** (factor -> eval). It is not a full portfolio backtest yet.

## Deliverables
- A single, stable input schema:
  - `pd.DataFrame` with `MultiIndex(datetime, instrument)`
  - columns include: `open, high, low, close, volume`
- A factor runner (safe subset):
  - input: `coded_alphas` (list of dicts with `alpha_id` + `code`)
  - output: factor series aligned to the input index
- An evaluation module:
  - forward returns (e.g., 1d / 5d)
  - IC / RankIC
  - quantile spread (top - bottom)
  - turnover (top bucket membership change)
- A graph node `evaluate_alphas_agent` that writes top-K into `state.sota_alphas`

## Recommended implementation order
### 1) Data + alignment
- Support at least one local dataset: CSV or Parquet
- Output must satisfy the schema above (MultiIndex + OHLCV columns)

### 2) Forward returns
Compute forward return aligned on the same index:

- 1-day: `fwd = close.shift(-1) / close - 1`
- Panel data requires `groupby(instrument).shift(-horizon)`

### 3) Factor execution (runner)
P0 options:
- Preferred: make the LLM output a small DSL you control (e.g., `ts_mean(returns(close, 1), 20)`), then interpret it.
- If executing Python code strings:
  - run in a restricted namespace
  - disallow IO/network/system usage
  - validate output shape and index strictly

### 4) Evaluation metrics (minimal set)
- Daily IC / RankIC (cross-sectional)
- Quantile spread (top bucket mean return - bottom bucket mean return)
- Turnover (top bucket membership change rate)

### 5) Selection
- Select top-K factors based on a simple score (e.g., information ratio of the daily spread)
- Store selected factors in `state.sota_alphas` with metrics attached
