# P2.5: Constrained Optimizer Backend (Optional QP)

This update adds an **optional constrained optimizer backend** for portfolio construction.

- Default installs remain dependency-light (NumPy/Pandas only).
- If you install **cvxpy**, you can enable a **QP-style solver** that enforces *hard constraints* (gross, bounds, neutrality, etc.).
- If cvxpy is missing or the problem is infeasible, the system **automatically falls back** to the ridge optimizer from P2.4.

---

## What changed

### New optional dependency file

- `requirements-optimizer.txt`

Install:

```bash
pip install -r requirements-optimizer.txt
```

### Code changes

- `src/agent/research/optimizer.py`
  - Added `optimize_long_short_weights_with_meta(...)`
  - Added a cvxpy QP solver path with strict constraints
  - Added graceful fallback to the ridge optimizer

- `src/agent/research/portfolio_backtest.py`
  - Added config fields to select the optimizer backend
  - Skips post-hoc neutralization/clipping when the QP backend is used (to avoid breaking hard constraints)
  - Exports `construction` metadata in the backtest output

- `main.py`
  - Added CLI flags to control the constrained backend

---

## How to use

### 1) Ridge optimizer (no extra deps)

```bash
python main.py \
  --idea "Momentum + liquidity" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend ridge
```

### 2) Constrained QP optimizer (requires cvxpy)

```bash
pip install -r requirements-optimizer.txt

python main.py \
  --idea "Momentum + liquidity" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend qp \
  --optimizer-turnover-cap 0.15
```

### 3) Auto mode (recommended)

Auto tries QP when available; otherwise it uses ridge:

```bash
python main.py \
  --idea "Momentum + liquidity" \
  --eval-mode p2 \
  --construction-method optimizer \
  --optimizer-backend auto
```

---

## What the QP backend solves

At each rebalance date, we construct a *single* long/short portfolio over the candidate sets.

### Hard constraints

- **Sign constraints**
  - longs: `w_i >= 0`
  - shorts: `w_i <= 0`

- **Gross constraints** (dollar-neutral)
  - `sum(w_longs) = gross_long` (default 0.5)
  - `sum(w_shorts) = -gross_short` (default -0.5)

- **Per-name bounds** (optional)
  - if `--max-abs-weight > 0`: `|w_i| <= max_abs_weight`

- **Turnover cap vs baseline** (optional)
  - baseline is `w_target` (the “no-trade” weights given overlap + vol targeting)
  - if `--optimizer-turnover-cap > 0`:

    `0.5 * sum(|w - w_target|) <= optimizer_turnover_cap`

- **Hard exposure neutrality** (optional)
  - if P2 neutralization flags are enabled, the same exposure columns are used as *equality constraints*:

    `X^T w = 0`

  Exposures may include: beta, vol, log(ADV), sector dummies.

### Objective

The solver maximizes score while penalizing risk and trading:

- maximize: `score^T w`
- penalize:
  - `l2_lambda * ||w||_2^2`
  - `turnover_lambda * ||w - w_target||_1`
  - `exposure_lambda * ||X^T w||_2^2` (soft, optional)

---

## Fallback behavior

Even in `--optimizer-backend qp`, the system is designed to stay runnable:

- If **cvxpy is not installed**, it falls back to the ridge optimizer.
- If the QP is **infeasible** (common with tight caps + sector neutrality on small candidate sets), it falls back to ridge.

The backtest output includes:

- `construction.optimizer.last`: last meta payload (backend_used, fallback reason, cvxpy status, etc.)

---

## Notes / limitations

- Sector neutrality can be infeasible if the long and short candidate sets do not cover the same sectors.
- Tight `--max-abs-weight` requires enough names per side to reach the gross targets.
- The portfolio-level `--turnover-cap` (P2.2) still applies after construction and acts as an additional safety limiter.

