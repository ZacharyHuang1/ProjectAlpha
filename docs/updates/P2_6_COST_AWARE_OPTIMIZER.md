# P2.6: Cost-aware constrained optimizer

This update extends the optional constrained optimizer backend (P2.5) so that it can account for **real-world execution frictions inside the optimization objective**, not only in the backtest PnL.

The goal is to make the portfolio construction step more "trade-aware" and reduce the chance that the optimizer selects a theoretically strong but practically untradeable allocation.


## What's new

### 1) Convex cost terms inside the QP objective (cvxpy backend)
When the optimizer backend is **QP** (`--optimizer-backend qp` or `auto` with cvxpy installed), the optimizer can now include:

- **Linear trade costs** (convex, L1): proportional to `|Δw|`
- **Borrow drag** (convex, linear on short weights): proportional to `|short_w|`
- **Convex impact proxy** (convex): proportional to `|Δw|^(1+alpha)`

All of the above are scaled by `--optimizer-cost-aversion`.

Notes:
- Linear trade costs are derived from your backtest settings:
  - `commission_bps + slippage_bps` are applied to turnover (`0.5 * sum(|Δw|)`)
  - `half_spread_bps` is applied to `sum(|Δw|)`
  - The optimizer uses the equivalent per-name coefficient: `0.5 * (commission+slippage) + half_spread` (in bps, then converted to returns).
- Borrow drag uses the **current borrow bps** (either a constant `borrow_bps` or per-name time series) and approximates the expected drag over `holding_days`.


### 2) Optional per-name participation bounds
If ADV is available and `impact_max_participation > 0`, the QP can enforce:

`|Δw_i| <= impact_max_participation * ADV_i / portfolio_notional`

This mirrors the backtest participation clipping, but as a **hard constraint** (and therefore avoids non-convex min/clip logic inside the objective).

You can disable this with:

- `--no-optimizer-enforce-participation`


### 3) Optional exposure slack to reduce infeasibility
In P2.5, when neutrality constraints are enabled, the QP may become infeasible if you also impose tight caps (weights, turnover, participation).

If `--optimizer-exposure-slack-lambda > 0`, the optimizer uses a slack variable `s`:

`X^T w = s`

and adds a quadratic penalty `slack_lambda * ||s||^2`.

This keeps the solution near-neutral but makes the problem much more robust.


## How to use

### Install (optional)
The constrained optimizer remains optional.

```bash
pip install -r requirements-optimizer.txt
```


### Run with cost-aware QP construction

```bash
python main.py \
  --idea "Momentum + liquidity" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend qp \
  --optimizer-cost-aversion 1.0 \
  --optimizer-turnover-cap 0.15 \
  --impact-bps 50 \
  --impact-exponent 0.5 \
  --impact-max-participation 0.2 \
  --half-spread-bps 5 \
  --commission-bps 1 \
  --slippage-bps 2
```


### Use exposure slack (recommended when constraints are tight)

```bash
python main.py \
  --idea "Sector-neutral mean reversion" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend qp \
  --neutralize-beta true \
  --neutralize-liquidity true \
  --max-abs-weight 0.02 \
  --optimizer-turnover-cap 0.10 \
  --optimizer-exposure-slack-lambda 50.0
```


## Implementation notes

- Code:
  - `src/agent/research/optimizer.py`
    - adds `OptimizerCostModel`
    - QP objective now supports weighted `|Δw|`, borrow penalty, and a convex impact proxy
    - optional participation bounds via `u <= max_trade_abs`
    - optional exposure slack for neutrality constraints
  - `src/agent/research/portfolio_backtest.py`
    - builds a per-rebalance `OptimizerCostModel` using lookahead-safe inputs (ADV shifted by 1 day)
  - `main.py` / `evaluate_alphas_agent.py`
    - exposes and wires new CLI/config knobs


## Limitations

- The optimizer objective is still score-driven; costs are a penalty, not a fully calibrated expected return model.
- The impact proxy is a simplification (convex power penalty). The backtest still applies the exact P2.2 impact model.
- The ridge backend does not solve an L1/power cost problem; the full cost-aware objective is only used by the QP backend.
