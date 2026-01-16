# P2 Implementation


> Note: For the upgraded execution model (half-spread, ADV impact, borrow constraints), see `P2_2_IMPLEMENTATION.md`.

P2 upgrades the P1 walk-forward backtest with **research-grade risk controls**.

The main goals:

- make results more realistic (avoid a single style bet driving performance)
- make it easier to compare alphas fairly (common risk/constraint settings)
- keep everything deterministic + dependency-light

---

## What's new in P2

### 1) Liquidity filter (ADV)

At each rebalance date, the backtest can filter the tradable universe using
Average Daily Dollar Volume (ADV):

```
ADV[t] = mean_{window}(close * volume)  (shifted by 1 day)
```

If `--min-adv` is set (> 0), instruments below the threshold are excluded from
long/short selection.

### 2) Exposure neutralization

Weights can be neutralized against a set of exposures on the **active names only**.
This uses a simple linear projection (ridge-stabilized least squares):

- market beta (rolling)
- volatility (rolling)
- liquidity (log ADV)
- sector dummies (optional, requires a sector map)

All exposures are lookahead-safe: they are shifted by 1 day.

### 3) Volatility targeting

An optional volatility targeting overlay scales the portfolio weights so the
strategy targets a desired annualized volatility.

Implementation (simple overlay):

- estimate realized volatility from trailing strategy net returns
- compute a leverage multiplier
- clamp leverage to `--vol-target-max-leverage`

---

## How to run

Default (synthetic OHLCV):

```bash
python main.py --idea "Volume-conditioned momentum" --eval-mode p2
```

Enable/disable P2 knobs:

```bash
python main.py \
  --idea "Momentum + liquidity" \
  --data-path /path/to/ohlcv.csv \
  --eval-mode p2 \
  --neutralize-beta \
  --neutralize-liquidity \
  --no-neutralize-vol \
  --target-vol-annual 0.10 \
  --min-adv 2000000
```

### Sector neutralization

Provide a sector map (CSV/Parquet) with columns:

- `instrument`
- `sector`

Example:

```bash
python main.py \
  --idea "Value within sectors" \
  --data-path /path/to/ohlcv.csv \
  --eval-mode p2 \
  --neutralize-sector \
  --sector-map-path /path/to/sector_map.csv
```

---

## Code locations

- Backtest + P2 features: `src/agent/research/portfolio_backtest.py`
- Exposure estimators: `src/agent/research/risk_exposures.py`
- Neutralization utilities: `src/agent/research/neutralize.py`
- Optional metadata loader: `src/agent/services/metadata.py`

---

## Notes / limitations

- This is a research backtest (close-to-close, simplified execution).
- Neutralization is done on the active names; it does not attempt to build a full
  production risk model.
- Vol targeting uses trailing strategy returns and is applied as a daily leverage overlay.

If you want production-grade realism next, the natural follow-ups are:

- more detailed cost/impact models (spread + nonlinear impact)
- borrow availability and short constraints
- factor decay curves + multi-horizon validation
