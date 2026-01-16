# P1 Implementation: Walk-Forward Research Backtest

This document describes the **P1** upgrade in this repo.

P1 extends P0 (fast factor sanity checks) by adding a **portfolio-style backtest**
and **walk-forward evaluation** so that factor ranking is based on
**out-of-sample** performance.

---

## What changed

### New modules

- `src/agent/research/portfolio_backtest.py`
  - Deterministic long/short portfolio backtest.
  - Supports `rebalance_days`, `holding_days` (with overlap), and a simple cost model.

- `src/agent/research/walk_forward.py`
  - Sequential walk-forward splits: train/valid/test.
  - For each split:
    - choose factor **sign** on the train window (best IR of +factor vs -factor)
    - evaluate on valid (optional) and test
  - Aggregates all test segments into a single out-of-sample return stream.

### Updated agent

- `src/agent/agents/evaluate_alphas_agent.py`
  - Adds `eval_mode`:
    - `p0`: forward-return proxy metrics (IC / RankIC / spread / turnover)
    - `p1`: walk-forward + portfolio backtest
  - Graceful fallback: if the dataset is too short for P1 splits, it falls back to P0.

### Updated CLI

- `main.py`
  - Adds `--eval-mode p0|p1` (default: `p1`)
  - Adds walk-forward and portfolio backtest knobs.

---

## P1 backtest logic (high-level)

1. Build **daily returns** from close prices.
2. On each rebalance date:
   - rank instruments by factor value
   - long top quantile, short bottom quantile (equal-weight, dollar-neutral)
3. Hold positions for `holding_days`.
   - if `holding_days > rebalance_days`, holdings overlap via multiple active sub-portfolios.
4. Transaction costs:
   - `commission_bps + slippage_bps` applied to **portfolio turnover**
   - optional `borrow_bps` applied daily to short exposure
5. Walk-forward:
   - run sequential splits
   - choose factor sign on train
   - report aggregated **out-of-sample** metrics on test segments

---

## How to run

### Install

```bash
pip install -r requirements.txt
cp .env.example .env  # optional; add OPENAI_API_KEY for real LLM calls
```

### Run P1 (default)

```bash
python main.py --idea "Volume-conditioned momentum" --thread-id demo1
```

### Run P1 with your own data

```bash
python main.py \
  --idea "Volume-conditioned momentum" \
  --data-path /path/to/ohlcv.csv \
  --eval-mode p1 \
  --rebalance-days 5 \
  --holding-days 5 \
  --commission-bps 2 \
  --slippage-bps 3 \
  --wf-train-days 126 \
  --wf-valid-days 42 \
  --wf-test-days 42 \
  --wf-step-days 42
```

### Run P0 explicitly

```bash
python main.py --eval-mode p0 --horizon 1 --cost-bps 10
```

---

## Output schema (P1)

Each alpha gets a `backtest_results` dict like:

```json
{
  "mode": "p1",
  "information_ratio": 0.42,
  "annualized_return": 0.10,
  "max_drawdown": -0.08,
  "turnover_mean": 0.22,
  "coverage_mean": 0.95,
  "walk_forward": {
    "config": {"walk_forward": {...}, "backtest": {...}},
    "splits": [...],
    "oos": {...},
    "oos_daily": [...],
    "stability": {...}
  }
}
```

`sota_alphas` are selected by `information_ratio` (out-of-sample IR in P1).

---

## Limitations

- This is a **research backtest**, not production execution.
- The cost model is intentionally simple.
- For real trading, you will want:
  - a risk model / constraints (sector, beta, leverage, limits)
  - robust universe definitions and survivorship-bias controls
  - more detailed cost/slippage modeling
  - experiment tracking and reproducibility (configs/artifacts)
