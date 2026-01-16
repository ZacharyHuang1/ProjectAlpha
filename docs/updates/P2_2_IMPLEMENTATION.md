# P2.2 Implementation: Costs + Borrow Constraints

This update makes the research backtest **closer to a tradable long/short strategy** by adding:

- a more realistic transaction cost model (linear fees + half-spread + nonlinear impact)
- short availability / borrow constraints (hard-to-borrow list + per-instrument borrow rates)
- an optional turnover cap at rebalance/expiry transitions

The implementation remains deterministic and dependency-light.

---

## 1) Transaction cost model

Costs are paid at **close(t)** when the portfolio transitions from `w_prev` to `w_now`.
They reduce the return realized from **t -> t+1**.

Let:

- `Δw = w_now - w_prev`
- `turnover = 0.5 * Σ |Δw_i|`
- `abs_trade = Σ |Δw_i|` (note: `abs_trade = 2 * turnover`)

### Linear cost (commission + slippage)

A simple linear cost is applied to turnover:

`linear_cost = turnover * (commission_bps + slippage_bps) / 10000`

### Half-spread cost

A half-spread cost is applied to absolute traded weight:

`spread_cost = abs_trade * half_spread_bps / 10000`

### Nonlinear impact cost (ADV-based participation)

Impact uses a participation-style model based on **rolling ADV** (dollar volume).

- `trade_notional_i = |Δw_i| * portfolio_notional`
- `participation_i = trade_notional_i / ADV_i`
- clipped: `participation_i = min(participation_i, impact_max_participation)`

Impact is then:

`impact_cost = Σ |Δw_i| * (impact_bps / 10000) * (participation_i ** impact_exponent)`

All ADV inputs are lookahead-safe: the ADV used at date `t` is shifted by 1 day.

---

## 2) Borrow constraints

### Hard-to-borrow list

If an instrument is marked hard-to-borrow, it **cannot** be selected into the short leg.

File format (CSV/Parquet):

- `instrument` (required)
- optional: `shortable` / `borrowable` (False/0 => hard-to-borrow)

### Borrow rates

Borrow rates can be provided as a per-instrument time series (annualized, in bps).

File format (CSV/Parquet):

- `datetime` (or `date`)
- `instrument`
- `borrow_bps` (annualized bps)

Borrow cost is applied daily on the short exposure:

`borrow_cost = Σ short_abs_i * (borrow_bps_i / 10000 / trading_days)`

### Max borrow threshold

Optionally, shorts with `borrow_bps > max_borrow_bps` are excluded from the short leg.

---

## 3) Turnover cap

An optional cap can be applied to the executed transition:

If `turnover > turnover_cap` then we scale the trade:

`w_exec = w_prev + (w_now - w_prev) * (turnover_cap / turnover)`

This is a simple research heuristic and is not a full trade optimizer.

---

## Code locations

- Backtest + costs + borrow constraints:
  - `src/agent/research/portfolio_backtest.py`
- Metadata loaders:
  - `src/agent/services/metadata.py`
- Walk-forward wiring:
  - `src/agent/research/walk_forward.py`
- Pipeline wiring (config + selection):
  - `src/agent/agents/evaluate_alphas_agent.py`
  - `main.py`

---

## How to run

### Basic (P2 default, synthetic data)
```bash
python main.py --idea "Momentum + liquidity" --eval-mode p2
```

### Enable costs + impact
```bash
python main.py \
  --idea "Momentum + liquidity" \
  --eval-mode p2 \
  --commission-bps 1 \
  --slippage-bps 2 \
  --half-spread-bps 5 \
  --impact-bps 50 \
  --impact-exponent 0.5 \
  --portfolio-notional 1000000
```

### Add borrow constraints
```bash
python main.py \
  --idea "Value within sectors" \
  --eval-mode p2 \
  --hard-to-borrow-path ./hard_to_borrow.csv \
  --borrow-rates-path ./borrow_rates.csv \
  --max-borrow-bps 500
```

### Add turnover cap
```bash
python main.py \
  --idea "Low turnover variant" \
  --eval-mode p2 \
  --turnover-cap 0.3
```

---
