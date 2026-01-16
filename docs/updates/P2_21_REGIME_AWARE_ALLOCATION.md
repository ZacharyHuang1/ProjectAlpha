# P2.21 – Regime-aware alpha allocation (dynamic weights)

## What changed

This update adds an **optional regime-aware alpha allocation** layer on top of the existing holdings-level ensemble (P2.18) and the static allocation (P2.19/P2.20).

When enabled, the system:
- computes a small set of **lookahead-safe market features** (shifted by 1 day)
- assigns each day to a **discrete regime** using thresholds fit on the *fit segment only* (train or train+valid)
- learns **separate alpha weights per regime** on the fit segment
- applies the corresponding regime weights during the OOS test segment (optionally smoothed)

The result is a single holdings-level portfolio where the **mixture of alphas adapts** to market state (e.g., low vs high volatility).

## Key files

- `src/agent/research/regime_features.py`
  - computes daily market features (`mkt_vol`, `mkt_liq`, etc.) from OHLCV
  - all rolling features are shifted by 1 day to avoid lookahead

- `src/agent/research/regime_labels.py`
  - fits quantile thresholds on the fit segment
  - assigns regime labels for a target date range
  - includes lightweight regime usage diagnostics (`switch_rate`, etc.)

- `src/agent/research/alpha_allocation_regime.py`
  - fits a global allocation (fallback) + per-regime allocations
  - converts regime labels into a **daily alpha weight matrix** with optional smoothing

- `src/agent/research/holdings_ensemble.py`
  - adds `walk_forward_holdings_ensemble_allocated_regime(...)`
  - reuses the same per-split return-matrix cache used by the static allocated ensemble

- `main.py`
  - adds CLI flags to enable and configure regime-aware allocation

- `src/agent/services/experiment_tracking.py`
  - exports new artifacts when the regime-aware ensemble is enabled

- `src/agent/research/reporting.py`
  - adds a new report section: *Holdings-level ensemble (allocated, regime-aware)*

## How to run

Enable regime-aware allocation via flags:

```bash
python main.py \
  --eval-mode p2 \
  --alpha-allocation \
  --alpha-allocation-regime-aware \
  --alpha-allocation-regime-mode vol \
  --alpha-allocation-regime-window 20 \
  --alpha-allocation-regime-buckets 3 \
  --alpha-allocation-regime-min-days 30 \
  --alpha-allocation-regime-smoothing 0.10
```

Notes:
- `vol` is the default mode (volatility-only). `vol_liq` uses volatility × liquidity.
- If a regime has too few fit days, the system falls back to the split’s global allocation.

## New artifacts

When enabled, the run directory will include:

- `ensemble_holdings_allocated_regime_metrics.json`
- `ensemble_holdings_allocated_regime_oos_daily.csv`
- `ensemble_holdings_allocated_regime_positions.csv`
- `alpha_allocations_regime.csv` (per-split, per-regime alpha weights)
- `alpha_allocation_regime_diagnostics.csv`

## Why this matters

Static alpha weights assume the relative edge of alphas is stable. In practice, many alphas are **regime dependent** (e.g., mean-reversion does better in high-vol chop; trend works in strong trends).

Regime-aware allocation is a lightweight way to capture that non-stationarity while keeping the core walk-forward discipline and avoiding leakage.
