"""agent.research.portfolio_backtest

Research backtest utilities.

P1 implements a small, deterministic long/short backtest.
P2 extends it with research-grade risk/constraint features:
- optional liquidity filter (ADV)
- optional exposure neutralization (beta / vol / liquidity / sector)
- optional volatility targeting (simple leverage overlay)

The goal is research validation, not production execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from agent.research.neutralize import (
    clip_weights,
    make_sector_dummies,
    neutralize_weights,
    rescale_long_short,
)
from agent.research.risk_exposures import (
    close_to_returns,
    market_return,
    rolling_beta,
    rolling_log_adv,
    rolling_volatility,
    to_wide_close_volume,
)

from agent.research.factor_risk_model import estimate_factor_risk_model
from agent.research.optimizer import OptimizerConfig, OptimizerCostModel, optimize_long_short_weights_with_meta, select_long_short_candidates


@dataclass(frozen=True)
class BacktestConfig:
    rebalance_days: int = 5
    holding_days: int = 5
    n_quantiles: int = 5
    min_obs: int = 20
    trading_days: int = 252
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    borrow_bps: float = 0.0

    borrow_cost_multiplier: float = 1.0
    # P2.2 cost model
    half_spread_bps: float = 0.0
    impact_bps: float = 0.0
    impact_exponent: float = 0.5
    impact_max_participation: float = 0.2
    portfolio_notional: float = 1_000_000.0
    turnover_cap: float = 0.0
    max_borrow_bps: float = 0.0

    # P2 risk/constraints
    max_abs_weight: float = 0.0
    max_names_per_side: int = 0
    min_adv: float = 0.0
    adv_window: int = 20
    beta_window: int = 60
    vol_window: int = 20
    neutralize_beta: bool = False
    neutralize_vol: bool = False
    neutralize_liquidity: bool = False
    neutralize_sector: bool = False
    target_vol_annual: float = 0.0
    vol_target_window: int = 20
    vol_target_max_leverage: float = 3.0

    # P2.4 portfolio construction
    construction_method: str = "heuristic"  # "heuristic" | "optimizer"
    optimizer_l2_lambda: float = 1.0
    optimizer_turnover_lambda: float = 10.0
    optimizer_exposure_lambda: float = 0.0
    optimizer_max_iter: int = 2

    # P2.5 constrained optimizer backend (optional)
    optimizer_backend: str = "auto"  # auto | ridge | qp
    optimizer_turnover_cap: float = 0.0
    optimizer_solver: str = ""

    # P2.6 cost-aware objective inside the constrained optimizer (QP backend)
    optimizer_cost_aversion: float = 1.0
    optimizer_exposure_slack_lambda: float = 0.0
    optimizer_enforce_participation: bool = True

    # P2.8 diagonal risk (annualized variance proxy)
    optimizer_risk_aversion: float = 0.0
    optimizer_risk_window: int = 20

    # P2.9 risk model (diag | factor)
    optimizer_risk_model: str = "diag"
    optimizer_factor_risk_window: int = 60
    optimizer_factor_risk_shrink: float = 0.2
    optimizer_factor_risk_ridge: float = 1e-3

    # P2.10 robust risk estimation for the factor risk model
    optimizer_factor_risk_estimator: str = "sample"
    optimizer_factor_risk_shrink_method: str = "fixed"
    optimizer_factor_risk_ewm_halflife: float = 20.0
    optimizer_factor_return_clip_sigma: float = 6.0
    optimizer_idio_shrink: float = 0.2
    optimizer_idio_clip_q: float = 0.99

def _ir(x: np.ndarray, trading_days: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 0.0
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(mu / sd * np.sqrt(float(trading_days)))


def _max_drawdown(returns: pd.Series) -> float:
    if returns is None or returns.empty:
        return 0.0
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _split_quantiles(x: pd.Series, n_quantiles: int) -> Optional[pd.Series]:
    r = x.rank(method="first")
    try:
        q = pd.qcut(r, q=int(n_quantiles), labels=False, duplicates="drop")
    except Exception:
        return None
    if q is None:
        return None
    if q.nunique(dropna=True) < 2:
        return None
    return q.astype("Int64")


def _weights_from_factor(
    x: pd.Series,
    *,
    n_quantiles: int,
    max_names_per_side: int = 0,
    shortable: Optional[pd.Series] = None,
) -> Optional[pd.Series]:
    x = x.dropna()
    if x.empty:
        return None
    max_names_per_side = int(max_names_per_side)

    if max_names_per_side > 0:
        xs = x.sort_values(kind="mergesort")
        if xs.size < (2 * max_names_per_side):
            return None

        if shortable is not None:
            m = shortable.reindex(xs.index).fillna(False).astype(bool)
            xs_short = xs[m]
            if xs_short.size < max_names_per_side:
                return None
            short = xs_short.iloc[:max_names_per_side]
        else:
            short = xs.iloc[:max_names_per_side]

        long = xs.iloc[-max_names_per_side:]
    else:
        q = _split_quantiles(x, n_quantiles=n_quantiles)
        if q is None:
            return None

        top_q = int(q.max())
        bot_q = int(q.min())
        long = x[q == top_q]
        short = x[q == bot_q]
        if shortable is not None:
            m = shortable.reindex(short.index).fillna(False).astype(bool)
            short = short[m]
    if long.empty or short.empty:
        return None

    w = pd.Series(0.0, index=x.index, dtype=float)
    w.loc[long.index] = 0.5 / float(len(long))
    w.loc[short.index] = -0.5 / float(len(short))
    return w


def _yearly_breakdown(returns: pd.Series, trading_days: int) -> Dict[str, Dict[str, float]]:
    if returns is None or returns.empty:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    by_year = returns.groupby(returns.index.year)
    for y, r in by_year:
        arr = r.to_numpy(dtype=float)
        ir = _ir(arr, trading_days=trading_days)
        ann = float((1.0 + np.nanmean(arr)) ** float(trading_days) - 1.0) if arr.size else 0.0
        mdd = _max_drawdown(r)
        out[str(int(y))] = {
            "information_ratio": float(ir),
            "annualized_return": float(ann),
            "max_drawdown": float(mdd),
            "n_days": int(r.size),
        }
    return out


def backtest_long_short(
    factor: pd.Series,
    ohlcv_or_close: pd.DataFrame | pd.Series,
    *,
    config: BacktestConfig,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    include_daily: bool = True,
    include_positions: bool = False,
    sector_map: Optional[Dict[str, str]] = None,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Run a simple long/short backtest driven by a factor.

    The factor is observed at date *t* and used to trade at close(t),
    therefore it impacts returns from *t -> t+1*.
    """

    if factor is None or ohlcv_or_close is None or factor.empty:
        return {"error": "Empty inputs"}

    if isinstance(ohlcv_or_close, pd.DataFrame):
        ohlcv = ohlcv_or_close
        if "close" not in ohlcv.columns:
            return {"error": "OHLCV is missing 'close'"}
    else:
        close = ohlcv_or_close
        ohlcv = pd.DataFrame(index=close.index)
        ohlcv["close"] = close

    if not isinstance(factor.index, pd.MultiIndex) or not isinstance(ohlcv.index, pd.MultiIndex):
        return {"error": "Expected MultiIndex(datetime, instrument)"}

    # Wide matrices: dates x instruments
    fac = factor.unstack("instrument").sort_index()
    px, vol = to_wide_close_volume(ohlcv)

    if start is not None:
        fac = fac.loc[fac.index >= pd.to_datetime(start)]
        px = px.loc[px.index >= pd.to_datetime(start)]
        if vol is not None:
            vol = vol.loc[vol.index >= pd.to_datetime(start)]
    if end is not None:
        fac = fac.loc[fac.index <= pd.to_datetime(end)]
        px = px.loc[px.index <= pd.to_datetime(end)]
        if vol is not None:
            vol = vol.loc[vol.index <= pd.to_datetime(end)]

    # Align on common dates/instruments
    common_dates = fac.index.intersection(px.index)
    fac = fac.loc[common_dates]
    px = px.loc[common_dates]
    if vol is not None:
        vol = vol.loc[common_dates]
    fac = fac.reindex(columns=px.columns)

    if len(common_dates) < 3:
        return {"error": "Not enough dates after alignment"}

    ret = close_to_returns(px).fillna(0.0)
    dates = list(common_dates)

    rebalance_days = max(1, int(config.rebalance_days))
    holding_days = max(1, int(config.holding_days))
    n_quantiles = max(2, int(config.n_quantiles))
    min_obs = max(1, int(config.min_obs))

    max_active = int(np.ceil(float(holding_days) / float(rebalance_days))) if holding_days > rebalance_days else 1
    capital_per = 1.0 / float(max_active)

    linear_cost_rate = float(config.commission_bps + config.slippage_bps) / 10000.0
    half_spread_rate = float(config.half_spread_bps) / 10000.0
    impact_rate = float(config.impact_bps) / 10000.0
    impact_exponent = float(config.impact_exponent)
    impact_max_part = float(config.impact_max_participation)
    portfolio_notional = float(config.portfolio_notional)
    turnover_cap = float(config.turnover_cap)

    borrow_mult = float(getattr(config, "borrow_cost_multiplier", 1.0) or 1.0)
    borrow_daily_const = float(config.borrow_bps) / 10000.0 / float(config.trading_days) * borrow_mult
    borrow_wide = None
    if borrow_rates is not None:
        try:
            br = borrow_rates
            if isinstance(br, pd.DataFrame) and br.shape[1] == 1:
                br = br.iloc[:, 0]
            if isinstance(br, pd.Series) and isinstance(br.index, pd.MultiIndex):
                borrow_wide = br.unstack("instrument").sort_index().ffill()
        except Exception:
            borrow_wide = None
    if borrow_wide is not None:
        borrow_wide = borrow_wide.reindex(index=px.index, columns=px.columns)

    base_shortable = pd.Series(True, index=px.columns, dtype=bool)
    if hard_to_borrow:
        for inst in hard_to_borrow:
            if inst in base_shortable.index:
                base_shortable.loc[inst] = False

    rebalance_idx = set(range(0, len(dates), rebalance_days))

    # Optional precomputed exposures (lookahead-safe by construction).
    adv = None
    log_adv = None
    if vol is not None:
        adv = (px * vol).rolling(int(config.adv_window), min_periods=min_obs).mean().shift(1)
        log_adv = np.log1p(adv)

    vol_x = rolling_volatility(ret, window=int(config.vol_window), min_obs=min_obs) if config.neutralize_vol else None
    beta_x = None
    if config.neutralize_beta:
        mkt = market_return(ret)
        beta_x = rolling_beta(ret, mkt, window=int(config.beta_window), min_obs=min_obs)

    liq_x = log_adv if config.neutralize_liquidity else None

    risk_var_x = None
    risk_av = float(getattr(config, "optimizer_risk_aversion", 0.0) or 0.0)
    if risk_av > 0.0:
        rwin = int(getattr(config, "optimizer_risk_window", int(config.vol_window)))
        risk_vol = rolling_volatility(ret, window=rwin, min_obs=min_obs)
        risk_var_x = (risk_vol * risk_vol) * float(config.trading_days)


    sector_d = None
    if config.neutralize_sector and sector_map:
        sector_d = make_sector_dummies(list(px.columns), sector_map, drop_first=True)

    # Active portfolios: list of (expiry_index, weight_series)
    active: List[Tuple[int, pd.Series]] = []
    w_prev = pd.Series(0.0, index=px.columns, dtype=float)

    # Track construction metadata for debugging and run artifacts.
    construction_meta: Dict[str, Any] = {
        "method": str(getattr(config, "construction_method", "heuristic") or "heuristic"),
        "optimizer": {
            "backend_requested": str(getattr(config, "optimizer_backend", "auto") or "auto"),
            "attempts": 0,
            "backend_used": {"qp": 0, "ridge": 0},
            "last": None,
        },
    }


    daily_rows: List[Dict[str, Any]] = []
    pos_rows: List[Dict[str, Any]] = []
    pos_dates: List[str] = []
    pnl: List[float] = []
    gross_pnl: List[float] = []

    def _vol_target_scale() -> float:
        if float(config.target_vol_annual) <= 0.0:
            return 1.0
        window = int(config.vol_target_window)
        if len(pnl) < window:
            return 1.0
        trailing = np.asarray(pnl[-window:], dtype=float)
        trailing = trailing[np.isfinite(trailing)]
        if trailing.size < 2:
            return 1.0
        sd = float(np.std(trailing, ddof=1))
        if sd <= 0.0:
            return 1.0
        target_daily = float(config.target_vol_annual) / np.sqrt(float(config.trading_days))
        scale = target_daily / sd
        return float(np.clip(scale, 0.0, float(config.vol_target_max_leverage)))

    # We generate weights for dates[0..N-2] and apply to returns at dates[1..N-1].
    for i in range(0, len(dates) - 1):
        # Drop expired portfolios
        active = [(exp, w) for (exp, w) in active if exp > i]

        scale = _vol_target_scale()

        if i in rebalance_idx:
            x = fac.iloc[i]
            if adv is not None and float(config.min_adv) > 0.0:
                adv_row = adv.iloc[i]
                x = x[adv_row >= float(config.min_adv)]
            obs = int(x.notna().sum())
            if obs >= min_obs:
                shortable_x = base_shortable.reindex(x.index).fillna(False).astype(bool)
                if float(config.max_borrow_bps) > 0.0:
                    if borrow_wide is not None:
                        br_row = borrow_wide.iloc[i].reindex(x.index)
                        br_row = br_row.fillna(float(config.borrow_bps))
                    else:
                        br_row = pd.Series(float(config.borrow_bps), index=x.index)
                    shortable_x = shortable_x & (br_row <= float(config.max_borrow_bps))

                w: Optional[pd.Series] = None
                opt_backend_used: Optional[str] = None

                method = str(getattr(config, "construction_method", "heuristic") or "heuristic").strip().lower()
                if method == "optimizer":
                    cand = select_long_short_candidates(
                        x,
                        n_quantiles=n_quantiles,
                        max_names_per_side=int(config.max_names_per_side),
                        shortable=shortable_x,
                    )
                    if cand is not None:
                        long_names, short_names = cand

                        # Current unscaled aggregate weights from existing active sub-portfolios.
                        if active:
                            w_curr = sum((capital_per * w0.reindex(px.columns).fillna(0.0)) for (_, w0) in active)
                        else:
                            w_curr = pd.Series(0.0, index=px.columns, dtype=float)

                        if scale > 0.0:
                            w_target_total = (w_prev / scale) - w_curr
                        else:
                            w_target_total = -w_curr
                        w_target_port = w_target_total / float(capital_per)

                        # Exposure set matches P2 neutralization flags.
                        cand_names = list(dict.fromkeys(list(long_names) + list(short_names)))
                        X_parts = []
                        if beta_x is not None:
                            X_parts.append(pd.Series(beta_x.iloc[i].reindex(cand_names), name="beta"))
                        if vol_x is not None:
                            X_parts.append(pd.Series(vol_x.iloc[i].reindex(cand_names), name="vol"))
                        if liq_x is not None:
                            X_parts.append(pd.Series(liq_x.iloc[i].reindex(cand_names), name="log_adv"))
                        X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=cand_names)
                        if sector_d is not None:
                            X = X.join(sector_d.reindex(cand_names), how="left")

                        opt_cfg = OptimizerConfig(
                            l2_lambda=float(config.optimizer_l2_lambda),
                            turnover_lambda=float(config.optimizer_turnover_lambda),
                            exposure_lambda=float(config.optimizer_exposure_lambda),
                            max_iter=int(config.optimizer_max_iter),
                        )

                        backend = str(getattr(config, "optimizer_backend", "auto") or "auto").strip().lower()
                        opt_turnover_cap = float(getattr(config, "optimizer_turnover_cap", 0.0) or 0.0)
                        solver = str(getattr(config, "optimizer_solver", "") or "").strip() or None

                        enforce_hard = bool(
                            (config.neutralize_beta or config.neutralize_vol or config.neutralize_liquidity or config.neutralize_sector)
                            and (not X.empty)
                        )

                        # Optional cost-aware terms for the constrained optimizer.
                        cost_aversion = float(getattr(config, "optimizer_cost_aversion", 1.0) or 0.0)
                        slack_lambda = float(getattr(config, "optimizer_exposure_slack_lambda", 0.0) or 0.0)
                        enforce_part = bool(getattr(config, "optimizer_enforce_participation", True))

                        trade_cost = None
                        trade_per_abs = (0.5 * float(linear_cost_rate)) + float(half_spread_rate)
                        if cost_aversion > 0.0 and trade_per_abs > 0.0:
                            trade_cost = pd.Series(trade_per_abs, index=cand_names, dtype=float)

                        borrow_cost = None
                        if cost_aversion > 0.0:
                            if borrow_wide is not None:
                                br_row = borrow_wide.iloc[i].reindex(cand_names).fillna(float(config.borrow_bps))
                            else:
                                br_row = pd.Series(float(config.borrow_bps), index=cand_names)
                            borrow_daily = br_row.astype(float) / 10000.0 / float(config.trading_days) * borrow_mult
                            borrow_cost = borrow_daily * float(holding_days)

                        impact_coeff = None
                        max_trade_abs = None
                        if adv is not None:
                            denom = adv.iloc[i].reindex(cand_names).replace([np.inf, -np.inf], np.nan)
                            denom = denom.where(denom > 0)
                            if denom.notna().any():
                                fill = float(np.nanmedian(denom.to_numpy(dtype=float)))
                                if not np.isfinite(fill) or fill <= 0.0:
                                    fill = 1.0
                            else:
                                fill = 1.0
                            denom = denom.fillna(fill)

                            if cost_aversion > 0.0 and impact_rate > 0.0:
                                impact_coeff = impact_rate * (float(portfolio_notional) ** float(impact_exponent)) / (denom ** float(impact_exponent))

                            if enforce_part and float(impact_max_part) > 0.0:
                                max_trade_abs = (float(impact_max_part) * denom) / float(portfolio_notional)
                                max_trade_abs = max_trade_abs.clip(lower=0.0, upper=1.0)
                        risk_var = None
                        factor_loadings = None
                        factor_cov = None
                        idio_var = None
                        risk_meta = None

                        risk_aversion = float(getattr(config, "optimizer_risk_aversion", 0.0) or 0.0)
                        risk_model = str(getattr(config, "optimizer_risk_model", "diag") or "diag").strip().lower()

                        if risk_aversion > 0.0:
                            # P2.9: try a simple factor risk model first.
                            if risk_model == "factor" and not X.empty:
                                fr_win = int(getattr(config, "optimizer_factor_risk_window", max(20, int(config.beta_window))))
                                fr_shrink = float(getattr(config, "optimizer_factor_risk_shrink", 0.2))
                                fr_ridge = float(getattr(config, "optimizer_factor_risk_ridge", 1e-3))
                                fr_est = str(getattr(config, "optimizer_factor_risk_estimator", "sample"))
                                fr_sm = str(getattr(config, "optimizer_factor_risk_shrink_method", "fixed"))
                                fr_hl = float(getattr(config, "optimizer_factor_risk_ewm_halflife", 20.0))
                                fr_clip = float(getattr(config, "optimizer_factor_return_clip_sigma", 6.0))
                                idio_shrink = float(getattr(config, "optimizer_idio_shrink", fr_shrink))
                                idio_clip_q = float(getattr(config, "optimizer_idio_clip_q", 0.99))

                                r_hist = ret.iloc[max(0, i - fr_win) : i].reindex(columns=cand_names)
                                model = estimate_factor_risk_model(
                                    r_hist,
                                    X.reindex(cand_names),
                                    window=fr_win,
                                    min_obs=min_obs,
                                    ridge=fr_ridge,
                                    cov_shrink=fr_shrink,
                                    cov_shrink_method=fr_sm,
                                    cov_estimator=fr_est,
                                    ewm_halflife=fr_hl,
                                    factor_return_clip_sigma=fr_clip,
                                    idio_shrink=idio_shrink,
                                    idio_clip_q=idio_clip_q,
                                    trading_days=int(config.trading_days),
                                )
                                if model is not None:
                                    factor_loadings = model.loadings
                                    factor_cov = model.factor_cov
                                    idio_var = model.idio_var
                                    risk_meta = model.meta
                                    risk_model = "factor"

                            # Fallback: diagonal variance proxy.
                            if risk_model != "factor" and risk_var_x is not None:
                                rv = pd.Series(risk_var_x.iloc[i].reindex(cand_names), index=cand_names, dtype=float)
                                rv = rv.replace([np.inf, -np.inf], np.nan)
                                vals = rv.to_numpy(dtype=float)
                                med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else 0.0
                                rv = rv.fillna(med).clip(lower=0.0)
                                risk_var = rv
                                risk_model = "diag"

                        cost_model = OptimizerCostModel(
                            cost_aversion=cost_aversion,
                            trade_cost=trade_cost,
                            borrow_cost=borrow_cost,
                            impact_coeff=impact_coeff,
                            impact_exponent=float(impact_exponent),
                            max_trade_abs=max_trade_abs,
                            exposure_slack_lambda=slack_lambda,
                            risk_aversion=risk_aversion,
                            risk_var=risk_var,
                            risk_model=risk_model,
                            factor_loadings=factor_loadings,
                            factor_cov=factor_cov,
                            idio_var=idio_var,
                            risk_meta=risk_meta,
                        )
                        w_opt, ometa = optimize_long_short_weights_with_meta(
                            x,
                            long_names=long_names,
                            short_names=short_names,
                            w_target=w_target_port,
                            exposures=(X if not X.empty else None),
                            cfg=opt_cfg,
                            max_abs_weight=float(config.max_abs_weight),
                            backend=backend,
                            turnover_cap=opt_turnover_cap,
                            enforce_exposure_neutrality=enforce_hard,
                            solver=solver,
                            cost_model=cost_model,
                        )

                        construction_meta["optimizer"]["attempts"] = int(construction_meta["optimizer"].get("attempts", 0)) + 1
                        construction_meta["optimizer"]["last"] = ometa
                        used = str(ometa.get("backend_used") or "ridge")
                        bu = construction_meta["optimizer"].get("backend_used") or {}
                        if used in {"qp", "ridge"}:
                            bu[used] = int(bu.get(used, 0)) + 1
                        construction_meta["optimizer"]["backend_used"] = bu
                        opt_backend_used = used

                        w = w_opt
                        if w is not None:
                            w = w.reindex(px.columns).fillna(0.0)
                else:
                    w = _weights_from_factor(
                        x,
                        n_quantiles=n_quantiles,
                        max_names_per_side=int(config.max_names_per_side),
                        shortable=shortable_x,
                    )

                if w is not None:
                    w = w.reindex(px.columns).fillna(0.0)
                    active_names = list(w.index[w != 0.0])
                    skip_post = bool(method == "optimizer" and opt_backend_used == "qp")

                    if active_names and not skip_post:
                        X_parts = []
                        if beta_x is not None:
                            X_parts.append(pd.Series(beta_x.iloc[i].reindex(active_names), name="beta"))
                        if vol_x is not None:
                            X_parts.append(pd.Series(vol_x.iloc[i].reindex(active_names), name="vol"))
                        if liq_x is not None:
                            X_parts.append(pd.Series(liq_x.iloc[i].reindex(active_names), name="log_adv"))

                        X = pd.concat(X_parts, axis=1) if X_parts else pd.DataFrame(index=active_names)
                        if sector_d is not None:
                            X = X.join(sector_d.reindex(active_names), how="left")

                        if not X.empty:
                            w.loc[active_names] = neutralize_weights(w.loc[active_names], X, add_intercept=True)
                            w2 = rescale_long_short(w, gross_long=0.5, gross_short=0.5, scale_up=True)
                            if w2 is None:
                                w = None
                            else:
                                w = w2

                    if w is not None and float(config.max_abs_weight) > 0.0 and not skip_post:
                        w = clip_weights(w, max_abs_weight=float(config.max_abs_weight))
                        w2 = rescale_long_short(w, gross_long=0.5, gross_short=0.5, scale_up=False)
                        w = w2 if w2 is not None else None
                if w is not None:
                    exp = min(len(dates) - 1, i + holding_days)
                    active.append((exp, w.reindex(px.columns).fillna(0.0)))

        if active:
            w_now = sum((capital_per * w) for (_, w) in active)
        else:
            w_now = pd.Series(0.0, index=px.columns, dtype=float)

        w_now = w_now.fillna(0.0)

        # Optional volatility targeting (simple leverage overlay).
        w_now = w_now * scale

        # Apply an optional turnover cap at rebalance/expiry transitions.
        if turnover_cap > 0.0:
            delta0 = w_now - w_prev
            t_raw = 0.5 * float(np.abs(delta0).sum())
            if t_raw > turnover_cap:
                k = turnover_cap / t_raw
                w_now = w_prev + delta0 * k

        delta = w_now - w_prev
        abs_trade = (delta.abs()).fillna(0.0)
        turnover = 0.5 * float(abs_trade.sum())

        linear_cost = turnover * linear_cost_rate
        spread_cost = float(abs_trade.sum()) * half_spread_rate

        # Impact cost is linear in the bps coefficient. We also keep a unit term
        # (independent of impact_bps) so we can run execution-only sensitivity sweeps.
        impact_unit = 0.0
        impact_cost = 0.0
        if adv is not None:
            adv_row_cost = adv.iloc[i].reindex(px.columns)
            denom = adv_row_cost.replace([np.inf, -np.inf], np.nan)
            denom = denom.where(denom > 0)
            if denom.notna().any():
                fill = float(np.nanmedian(denom.to_numpy(dtype=float)))
                if not np.isfinite(fill) or fill <= 0.0:
                    fill = 1.0
            else:
                fill = 1.0
            denom = denom.fillna(fill)
            part = (abs_trade * portfolio_notional) / denom
            part = part.clip(lower=0.0, upper=impact_max_part)
            impact_unit = float((abs_trade * (part ** impact_exponent)).sum())
            impact_cost = float(impact_unit * float(impact_rate))

        cost = float(linear_cost + spread_cost + impact_cost)

        r_next = ret.iloc[i + 1]
        g = float((w_now * r_next).sum())

        if borrow_wide is not None:
            br = borrow_wide.iloc[i].reindex(px.columns)
            br = br.fillna(float(config.borrow_bps))
            borrow_daily_row = br / 10000.0 / float(config.trading_days) * borrow_mult
            short_abs = (-w_now.clip(upper=0.0)).fillna(0.0)
            borrow = float((short_abs * borrow_daily_row).sum())
        else:
            short_gross = float((-w_now.clip(upper=0.0)).sum())
            borrow = short_gross * borrow_daily_const

        net = g - cost - borrow

        dt = dates[i + 1]
        dt_iso = pd.to_datetime(dt).isoformat()

        if bool(include_positions):
            pos_dates.append(dt_iso)
            nz = w_now[(w_now.abs() > 1e-12)].copy()
            for inst, wv in nz.items():
                pos_rows.append({"datetime": dt_iso, "instrument": str(inst), "weight": float(wv)})

        daily_rows.append(
            {
                "datetime": dt_iso,
                "gross_return": float(g),
                "net_return": float(net),
                "turnover": float(turnover),
                "linear_cost": float(linear_cost),
                "spread_cost": float(spread_cost),
                "impact_cost": float(impact_cost),
                "impact_unit": float(impact_unit),
                "cost": float(cost),
                "borrow": float(borrow),
                "gross_long": float(w_now.clip(lower=0.0).sum()),
                "gross_short": float((-w_now.clip(upper=0.0)).sum()),
                "n_long": int((w_now > 0.0).sum()),
                "n_short": int((w_now < 0.0).sum()),
                "active_ports": int(len(active)),
                "leverage": float(np.abs(w_now).sum()),
                "scale": float(scale),
            }
        )

        gross_pnl.append(g)
        pnl.append(net)
        w_prev = w_now

    r_net = pd.Series(pnl, index=pd.to_datetime([r["datetime"] for r in daily_rows]))
    r_gross = pd.Series(gross_pnl, index=r_net.index)

    ir = _ir(r_net.to_numpy(dtype=float), trading_days=config.trading_days)
    ann = float((1.0 + float(r_net.mean())) ** float(config.trading_days) - 1.0)
    mdd = _max_drawdown(r_net)

    out = {
        "information_ratio": float(ir),
        "annualized_return": float(ann),
        "max_drawdown": float(mdd),
        "turnover_mean": float(np.mean([r["turnover"] for r in daily_rows])) if daily_rows else 0.0,
        "cost_mean": float(np.mean([r["cost"] for r in daily_rows])) if daily_rows else 0.0,
        "linear_cost_mean": float(np.mean([r["linear_cost"] for r in daily_rows])) if daily_rows else 0.0,
        "spread_cost_mean": float(np.mean([r["spread_cost"] for r in daily_rows])) if daily_rows else 0.0,
        "impact_cost_mean": float(np.mean([r["impact_cost"] for r in daily_rows])) if daily_rows else 0.0,
        "borrow_mean": float(np.mean([r["borrow"] for r in daily_rows])) if daily_rows else 0.0,
        "cost_bps": float(config.commission_bps + config.slippage_bps),
        "half_spread_bps": float(config.half_spread_bps),
        "impact_bps": float(config.impact_bps),
        "impact_exponent": float(config.impact_exponent),
        "impact_max_participation": float(config.impact_max_participation),
        "portfolio_notional": float(config.portfolio_notional),
        "turnover_cap": float(config.turnover_cap),
        "max_borrow_bps": float(config.max_borrow_bps),
        "borrow_bps": float(config.borrow_bps),
        "borrow_cost_multiplier": float(borrow_mult),
        "rebalance_days": int(rebalance_days),
        "holding_days": int(holding_days),
        "n_obs": int(len(daily_rows)),
        "daily": daily_rows if include_daily else None,
        "yearly": _yearly_breakdown(r_net, trading_days=config.trading_days),
        "gross": {
            "information_ratio": float(_ir(r_gross.to_numpy(dtype=float), trading_days=config.trading_days)),
            "annualized_return": float((1.0 + float(r_gross.mean())) ** float(config.trading_days) - 1.0),
            "max_drawdown": float(_max_drawdown(r_gross)),
        },
    }
    out["construction"] = construction_meta

    if bool(include_positions):
        out["positions"] = pos_rows
        out["position_dates"] = pos_dates

    if not include_daily:
        out.pop("daily", None)
    return out


def backtest_from_weights(
    weights: pd.DataFrame,
    ohlcv_or_close: pd.DataFrame | pd.Series,
    *,
    config: BacktestConfig,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    include_daily: bool = True,
    include_positions: bool = False,
    apply_turnover_cap: bool = False,
    borrow_rates: Optional[pd.Series] = None,
    hard_to_borrow: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """Backtest a provided daily weight path with the same cost model.

    The input `weights` is interpreted as the *held* portfolio weights for the
    close-to-close return on that same date.

    This is useful for holdings-level ensembles: you can combine multiple
    alpha portfolios into a single weight path, and evaluate costs and netting
    effects at the portfolio level.
    """

    if weights is None or weights.empty:
        return {"error": "Empty weights"}

    if isinstance(ohlcv_or_close, pd.DataFrame):
        ohlcv = ohlcv_or_close
        if "close" not in ohlcv.columns:
            return {"error": "OHLCV is missing 'close'"}
    else:
        close = ohlcv_or_close
        ohlcv = pd.DataFrame(index=close.index)
        ohlcv["close"] = close

    if not isinstance(ohlcv.index, pd.MultiIndex):
        return {"error": "Expected MultiIndex(datetime, instrument) for OHLCV"}

    try:
        w = weights.copy()
        w.index = pd.to_datetime(w.index)
        w = w.sort_index()
    except Exception:
        return {"error": "Invalid weights index"}

    # Wide price/volume matrices.
    px, vol = to_wide_close_volume(ohlcv)

    if start is not None:
        px = px.loc[px.index >= pd.to_datetime(start)]
        if vol is not None:
            vol = vol.loc[vol.index >= pd.to_datetime(start)]
    if end is not None:
        px = px.loc[px.index <= pd.to_datetime(end)]
        if vol is not None:
            vol = vol.loc[vol.index <= pd.to_datetime(end)]

    if px is None or px.empty or len(px.index) < 3:
        return {"error": "Not enough dates after alignment"}

    # Align weights to the available market dates.
    common_dates = w.index.intersection(px.index)
    if common_dates.empty:
        return {"error": "No overlapping dates between weights and prices"}

    w = w.reindex(index=common_dates)
    w = w.reindex(columns=px.columns).fillna(0.0)

    ret = close_to_returns(px).fillna(0.0)
    ret = ret.reindex(index=common_dates, columns=px.columns).fillna(0.0)

    min_obs = max(1, int(config.min_obs))

    # Cost model params.
    linear_cost_rate = float(config.commission_bps + config.slippage_bps) / 10000.0
    half_spread_rate = float(config.half_spread_bps) / 10000.0
    impact_rate = float(config.impact_bps) / 10000.0
    impact_exponent = float(config.impact_exponent)
    impact_max_part = float(config.impact_max_participation)
    portfolio_notional = float(config.portfolio_notional)
    turnover_cap = float(config.turnover_cap)

    # Borrow setup.
    borrow_mult = float(getattr(config, "borrow_cost_multiplier", 1.0) or 1.0)
    borrow_daily_const = float(config.borrow_bps) / 10000.0 / float(config.trading_days) * borrow_mult

    borrow_wide = None
    if borrow_rates is not None:
        try:
            br = borrow_rates
            if isinstance(br, pd.DataFrame) and br.shape[1] == 1:
                br = br.iloc[:, 0]
            if isinstance(br, pd.Series) and isinstance(br.index, pd.MultiIndex):
                borrow_wide = br.unstack("instrument").sort_index().ffill()
        except Exception:
            borrow_wide = None
    if borrow_wide is not None:
        borrow_wide = borrow_wide.reindex(index=px.index, columns=px.columns)

    base_shortable = pd.Series(True, index=px.columns, dtype=bool)
    if hard_to_borrow:
        for inst in hard_to_borrow:
            if inst in base_shortable.index:
                base_shortable.loc[inst] = False

    # Impact needs ADV at the trade date (previous close). We compute ADV
    # lookahead-safe and then shift by 1 when using it for return dates.
    adv = None
    if vol is not None:
        adv = (px * vol).rolling(int(config.adv_window), min_periods=min_obs).mean().shift(1)

    # Map each date to its previous market date.
    px_dates = list(px.index)
    prev_date: Dict[pd.Timestamp, pd.Timestamp] = {}
    for j in range(1, len(px_dates)):
        prev_date[pd.to_datetime(px_dates[j])] = pd.to_datetime(px_dates[j - 1])

    daily_rows: List[Dict[str, Any]] = []
    pos_rows: List[Dict[str, Any]] = []
    pos_dates: List[str] = []
    pnl: List[float] = []
    gross_pnl: List[float] = []

    w_prev = pd.Series(0.0, index=px.columns, dtype=float)

    for dt in common_dates:
        dt = pd.to_datetime(dt)
        w_target = pd.Series(w.loc[dt], index=px.columns, dtype=float).fillna(0.0)

        # Enforce basic shortability (best-effort): negative weights on unshortable names are set to 0.
        if hard_to_borrow:
            bad = (~base_shortable) & (w_target < 0.0)
            if bool(bad.any()):
                w_target = w_target.copy()
                w_target.loc[bad] = 0.0

        w_now = w_target
        if bool(apply_turnover_cap) and turnover_cap > 0.0:
            delta0 = w_now - w_prev
            t_raw = 0.5 * float(np.abs(delta0).sum())
            if t_raw > turnover_cap:
                k = turnover_cap / t_raw
                w_now = w_prev + delta0 * k

        delta = w_now - w_prev
        abs_trade = (delta.abs()).fillna(0.0)
        turnover = 0.5 * float(abs_trade.sum())

        linear_cost = turnover * linear_cost_rate
        spread_cost = float(abs_trade.sum()) * half_spread_rate

        impact_unit = 0.0
        impact_cost = 0.0
        if adv is not None and impact_rate > 0.0:
            tdate = prev_date.get(dt)
            if tdate is None:
                tdate = dt
            adv_row_cost = adv.loc[tdate].reindex(px.columns)
            denom = adv_row_cost.replace([np.inf, -np.inf], np.nan)
            denom = denom.where(denom > 0)
            if denom.notna().any():
                fill = float(np.nanmedian(denom.to_numpy(dtype=float)))
                if not np.isfinite(fill) or fill <= 0.0:
                    fill = 1.0
            else:
                fill = 1.0
            denom = denom.fillna(fill)
            part = (abs_trade * portfolio_notional) / denom
            part = part.clip(lower=0.0, upper=impact_max_part)
            impact_unit = float((abs_trade * (part ** impact_exponent)).sum())
            impact_cost = float(impact_unit * float(impact_rate))

        cost = float(linear_cost + spread_cost + impact_cost)

        r_row = ret.loc[dt]
        g = float((w_now * r_row).sum())

        if borrow_wide is not None:
            tdate = prev_date.get(dt)
            if tdate is None:
                tdate = dt
            br = borrow_wide.loc[tdate].reindex(px.columns)
            br = br.fillna(float(config.borrow_bps))
            borrow_daily_row = br / 10000.0 / float(config.trading_days) * borrow_mult
            short_abs = (-w_now.clip(upper=0.0)).fillna(0.0)
            borrow = float((short_abs * borrow_daily_row).sum())
        else:
            short_gross = float((-w_now.clip(upper=0.0)).sum())
            borrow = short_gross * borrow_daily_const

        net = g - cost - borrow

        dt_iso = pd.to_datetime(dt).isoformat()
        if bool(include_positions):
            pos_dates.append(dt_iso)
            nz = w_now[(w_now.abs() > 1e-12)].copy()
            for inst, wv in nz.items():
                pos_rows.append({"datetime": dt_iso, "instrument": str(inst), "weight": float(wv)})

        daily_rows.append(
            {
                "datetime": dt_iso,
                "gross_return": float(g),
                "net_return": float(net),
                "turnover": float(turnover),
                "linear_cost": float(linear_cost),
                "spread_cost": float(spread_cost),
                "impact_cost": float(impact_cost),
                "impact_unit": float(impact_unit),
                "cost": float(cost),
                "borrow": float(borrow),
                "gross_long": float(w_now.clip(lower=0.0).sum()),
                "gross_short": float((-w_now.clip(upper=0.0)).sum()),
                "n_long": int((w_now > 0.0).sum()),
                "n_short": int((w_now < 0.0).sum()),
                "leverage": float(np.abs(w_now).sum()),
            }
        )

        gross_pnl.append(g)
        pnl.append(net)
        w_prev = w_now

    r_net = pd.Series(pnl, index=pd.to_datetime([r["datetime"] for r in daily_rows]))
    r_gross = pd.Series(gross_pnl, index=r_net.index)

    ir = _ir(r_net.to_numpy(dtype=float), trading_days=config.trading_days)
    ann = float((1.0 + float(r_net.mean())) ** float(config.trading_days) - 1.0)
    mdd = _max_drawdown(r_net)

    out = {
        "information_ratio": float(ir),
        "annualized_return": float(ann),
        "max_drawdown": float(mdd),
        "turnover_mean": float(np.mean([r["turnover"] for r in daily_rows])) if daily_rows else 0.0,
        "cost_mean": float(np.mean([r["cost"] for r in daily_rows])) if daily_rows else 0.0,
        "linear_cost_mean": float(np.mean([r["linear_cost"] for r in daily_rows])) if daily_rows else 0.0,
        "spread_cost_mean": float(np.mean([r["spread_cost"] for r in daily_rows])) if daily_rows else 0.0,
        "impact_cost_mean": float(np.mean([r["impact_cost"] for r in daily_rows])) if daily_rows else 0.0,
        "borrow_mean": float(np.mean([r["borrow"] for r in daily_rows])) if daily_rows else 0.0,
        "cost_bps": float(config.commission_bps + config.slippage_bps),
        "half_spread_bps": float(config.half_spread_bps),
        "impact_bps": float(config.impact_bps),
        "borrow_bps": float(config.borrow_bps),
        "apply_turnover_cap": bool(apply_turnover_cap),
        "turnover_cap": float(config.turnover_cap),
        "n_obs": int(len(daily_rows)),
        "daily": daily_rows if include_daily else None,
        "yearly": _yearly_breakdown(r_net, trading_days=config.trading_days),
        "gross": {
            "information_ratio": float(_ir(r_gross.to_numpy(dtype=float), trading_days=config.trading_days)),
            "annualized_return": float((1.0 + float(r_gross.mean())) ** float(config.trading_days) - 1.0),
            "max_drawdown": float(_max_drawdown(r_gross)),
        },
        "construction": {"method": "weights"},
    }

    if bool(include_positions):
        out["positions"] = pos_rows
        out["position_dates"] = pos_dates

    if not include_daily:
        out.pop("daily", None)

    return out
