"""Alpha-GPT demo runner.

Examples:
  python main.py --idea "Volume-conditioned momentum on US equities" --thread-id demo1
  USE_POSTGRES=true python main.py --idea "..." --thread-id demo1
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
import asyncio
import uuid

from dotenv import load_dotenv

# Allow running `python main.py` without installing the package.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from agent.graph import graph
from agent.state import State
from agent.services.experiment_tracking import make_run_id, save_run_artifacts
from agent.research.regime_tune_presets import get_preset
from agent.research.utility_weight_calibration import calibrate_utility_weights, weights_to_kv_string


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--idea", type=str, default="", help="Trading idea / research direction.")
    p.add_argument("--thread-id", type=str, default="", help="LangGraph thread_id (for persistence).")
    p.add_argument("--checkpoint-id", type=str, default="", help="Optional checkpoint_id override.")

    # Evaluation mode
    p.add_argument(
        "--eval-mode",
        type=str,
        default="p2",
        choices=["p0", "p1", "p2"],
        help="Evaluation mode: p0 (fast proxy), p1 (walk-forward), p2 (walk-forward + risk controls).",
    )

    # Data
    p.add_argument("--data-path", type=str, default="", help="Optional OHLCV data file (csv/parquet/h5).")
    p.add_argument("--horizon", type=int, default=1, help="Forward return horizon used in P0 evaluation.")
    p.add_argument("--cost-bps", type=float, default=0.0, help="Transaction cost in basis points applied to turnover (P0 proxy).")
    p.add_argument("--min-obs-per-day", type=int, default=20, help="Minimum cross-sectional observations per day for evaluation.")
    p.add_argument("--min-coverage", type=float, default=0.0, help="Quality gate: minimum average coverage (0 disables).")
    p.add_argument("--max-turnover", type=float, default=1.0, help="Quality gate: maximum average turnover (1 disables).")

    # P2.12 tuning / ablation / additional gates
    p.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a small deterministic grid search over optimizer/backtest knobs (P2.12).",
    )
    p.add_argument(
        "--tune-metric",
        type=str,
        default="information_ratio",
        choices=["information_ratio", "annualized_return"],
        help="Metric used to pick the best tuned configuration.",
    )
    p.add_argument("--tune-max-combos", type=int, default=24, help="Maximum configs evaluated per alpha during tuning.")
    p.add_argument("--tune-save-top", type=int, default=10, help="Keep only top-N sweep rows per alpha in JSON (0 keeps all).")
    p.add_argument(
        "--tune-turnover-cap",
        type=str,
        default="",
        help="Comma-separated sweep values for optimizer_turnover_cap, e.g. '0,0.1,0.2'.",
    )
    p.add_argument(
        "--tune-max-abs-weight",
        type=str,
        default="",
        help="Comma-separated sweep values for max_abs_weight, e.g. '0,0.02'.",
    )
    p.add_argument(
        "--tune-risk-aversion",
        type=str,
        default="",
        help="Comma-separated sweep values for optimizer_risk_aversion, e.g. '0,5'.",
    )
    p.add_argument(
        "--tune-cost-aversion",
        type=str,
        default="",
        help="Comma-separated sweep values for optimizer_cost_aversion, e.g. '0,1'.",
    )

    p.add_argument("--ablation-top", type=int, default=1, help="Run cost ablation for the top-N selected alphas (0 disables).")

    p.add_argument(
        "--ablation-mode",
        type=str,
        default="both",
        choices=["both", "end_to_end", "execution_only"],
        help="Cost ablation mode: end_to_end re-optimizes, execution_only keeps trades fixed.",
    )

    # P2.13: small regime diagnostics for the top alpha.
    p.add_argument(
        "--regime-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute a small regime breakdown (market volatility / liquidity) for the top alpha.",
    )
    p.add_argument("--regime-window", type=int, default=20, help="Rolling window used for regime estimators (trading days).")
    p.add_argument("--regime-buckets", type=int, default=3, help="Number of quantile buckets for regime splits.")

    # P2.14: execution-only cost sensitivity curves (cheap, uses a fixed trading path).
    p.add_argument(
        "--cost-sensitivity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute execution-only cost sensitivity curves for the top alphas.",
    )
    p.add_argument("--cost-sensitivity-top", type=int, default=1, help="How many top alphas to analyze (0 disables).")
    p.add_argument(
        "--cost-sensitivity-linear-bps",
        type=str,
        default="",
        help="Comma-separated sweep values for linear cost bps (commission+slippage).",
    )
    p.add_argument(
        "--cost-sensitivity-half-spread-bps",
        type=str,
        default="",
        help="Comma-separated sweep values for half-spread bps.",
    )
    p.add_argument(
        "--cost-sensitivity-impact-bps",
        type=str,
        default="",
        help="Comma-separated sweep values for impact bps.",
    )
    p.add_argument(
        "--cost-sensitivity-borrow-bps",
        type=str,
        default="",
        help="Comma-separated sweep values for borrow bps (used when no borrow curve is provided).",
    )
    p.add_argument(
        "--cost-sensitivity-borrow-mult",
        type=str,
        default="",
        help="Comma-separated sweep values for borrow multiplier (used when borrow curve is provided).",
    )

    # P2.15: multi-horizon decay analysis (IC/spread + signal persistence).
    p.add_argument(
        "--decay-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute multi-horizon decay metrics for the top alphas (IC / spread / signal overlap).",
    )
    p.add_argument("--decay-analysis-top", type=int, default=1, help="How many top alphas to analyze (0 disables).")
    p.add_argument(
        "--decay-horizons",
        type=str,
        default="",
        help="Comma-separated horizons for decay analysis, e.g. '1,2,5,10,20'. Empty uses defaults.",
    )

    # P2.16: strategy-level schedule sweep (rebalance_days x holding_days).
    p.add_argument(
        "--schedule-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute a holding/rebalance schedule sweep for the top alphas.",
    )
    p.add_argument("--schedule-sweep-top", type=int, default=1, help="How many top alphas to analyze (0 disables).")
    p.add_argument(
        "--schedule-sweep-metric",
        type=str,
        default="information_ratio",
        choices=["information_ratio", "annualized_return"],
        help="Metric used to rank schedules in the sweep.",
    )
    p.add_argument(
        "--schedule-sweep-max-combos",
        type=int,
        default=25,
        help="Maximum number of (rebalance_days, holding_days) combos to evaluate.",
    )
    p.add_argument(
        "--schedule-sweep-rebalance-days",
        type=str,
        default="",
        help="Comma-separated rebalance_days grid, e.g. '1,2,5,10'. Empty uses defaults.",
    )
    p.add_argument(
        "--schedule-sweep-holding-days",
        type=str,
        default="",
        help="Comma-separated holding_days grid, e.g. '1,2,5,10,20'. Empty uses defaults.",
    )

    # P2.17: diversity-aware selection and an alpha ensemble (OOS return blending).
    p.add_argument(
        "--diverse-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a greedy diversity-aware selector for top-K (uses OOS return correlations).",
    )
    p.add_argument("--diverse-lambda", type=float, default=0.5, help="Correlation penalty strength for diverse selection.")
    p.add_argument(
        "--diverse-use-abs-corr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use absolute correlation for the diversity penalty.",
    )
    p.add_argument("--diverse-candidate-pool", type=int, default=20, help="Candidate pool size for diverse selection (0 = all).")
    p.add_argument("--diverse-min-periods", type=int, default=20, help="Min overlapping days for correlation estimates.")

    # P2.22: selection meta-tuning on validation returns (no test leakage).
    p.add_argument(
        "--selection-tune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Meta-tune diverse selection hyperparams on validation returns (then evaluate on test).",
    )
    p.add_argument(
        "--selection-tune-metric",
        type=str,
        default="information_ratio",
        choices=["information_ratio", "annualized_return"],
        help="Metric used to pick the best selection configuration on validation.",
    )
    p.add_argument("--selection-tune-max-combos", type=int, default=24, help="Maximum selection configs evaluated.")
    p.add_argument(
        "--selection-tune-lambda-grid",
        type=str,
        default="0,0.2,0.5,0.8",
        help="Comma-separated diversity_lambda grid, e.g. '0,0.2,0.5,0.8'.",
    )
    p.add_argument(
        "--selection-tune-candidate-pool-grid",
        type=str,
        default="10,20,40",
        help="Comma-separated candidate pool grid, e.g. '10,20,40'.",
    )
    p.add_argument(
        "--selection-tune-topk-grid",
        type=str,
        default="",
        help="Optional comma-separated top-K grid. Empty keeps --top-k fixed.",
    )
    p.add_argument(
        "--selection-tune-min-periods",
        type=int,
        default=20,
        help="Min overlapping days for correlation estimates during selection tuning.",
    )


    # P2.31: alpha selection constraints / presets.
    p.add_argument(
        "--alpha-selection-preset",
        type=str,
        default="",
        help="Selection constraint preset (e.g., low_redundancy, low_cost, stable_generalization).",
    )
    p.add_argument(
        "--alpha-selection-max-pairwise-corr",
        type=float,
        default=None,
        help="Hard cap on max |corr| between any selected alpha and a new candidate during selection.",
    )
    p.add_argument(
        "--alpha-selection-min-valid-ir",
        type=float,
        default=None,
        help="Validation-domain minimum IR for candidates (used when --selection-tune is enabled).",
    )
    p.add_argument(
        "--alpha-selection-min-valid-coverage",
        type=float,
        default=None,
        help="Validation-domain minimum coverage ratio for candidates (used when --selection-tune is enabled).",
    )
    p.add_argument(
        "--alpha-selection-max-total-cost-bps",
        type=float,
        default=None,
        help="Test-domain maximum total cost drag in bps (best-effort filter).",
    )
    p.add_argument(
        "--alpha-selection-min-wf-test-ir-positive-frac",
        type=float,
        default=None,
        help="Test-domain minimum walk-forward test IR positive fraction (stability filter).",
    )


    p.add_argument(
        "--ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute an equal-weight ensemble of selected alpha OOS return streams.",
    )

    # P2.18: holdings-level ensemble (combine weights then re-price with costs).
    p.add_argument(
        "--holdings-ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute a holdings-level ensemble (trade netting) for the selected alphas.",
    )
    p.add_argument(
        "--holdings-ensemble-apply-turnover-cap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the global turnover_cap when pricing the combined holdings path.",
    )

    # P2.19: alpha allocation (learned weights across selected alphas).
    p.add_argument(
        "--alpha-allocation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Learn alpha-level allocation weights (walk-forward) for holdings-level ensemble.",
    )
    p.add_argument(
        "--alpha-allocation-backend",
        type=str,
        default="auto",
        help="Alpha allocation backend: auto|qp|pgd (qp uses cvxpy when installed).",
    )
    p.add_argument(
        "--alpha-allocation-fit",
        type=str,
        default="train_valid",
        help="Fit segment for alpha allocation: train|train_valid.",
    )
    p.add_argument(
        "--alpha-allocation-score-metric",
        type=str,
        default="information_ratio",
        help="Score for alpha allocation: information_ratio|annualized_return.",
    )
    p.add_argument("--alpha-allocation-lambda", type=float, default=0.5, help="Correlation penalty strength.")
    p.add_argument("--alpha-allocation-l2", type=float, default=1e-6, help="L2 regularization on alpha weights.")
    p.add_argument(
        "--alpha-allocation-turnover-lambda",
        type=float,
        default=0.0,
        help="Smoothing penalty vs previous split weights (L2). 0 disables.",
    )
    p.add_argument(
        "--alpha-allocation-max-weight",
        type=float,
        default=0.8,
        help="Per-alpha max weight cap (must be >= 1/K).",
    )
    p.add_argument(
        "--alpha-allocation-use-abs-corr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use absolute correlation in the redundancy penalty.",
    )
    p.add_argument("--alpha-allocation-min-days", type=int, default=30, help="Minimum in-sample days to fit weights.")
    p.add_argument(
        "--alpha-allocation-solver",
        type=str,
        default="",
        help="Optional cvxpy solver override for alpha allocation (e.g., OSQP, SCS).",
    )

    # P2.20: meta-tune allocation hyperparams on aggregate valid performance.
    p.add_argument(
        "--alpha-allocation-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Meta-tune allocation hyperparams on train->valid across splits (no leakage).",
    )
    p.add_argument(
        "--alpha-allocation-tune-metric",
        type=str,
        default="information_ratio",
        choices=["information_ratio", "annualized_return"],
        help="Metric used to rank allocation hyperparams on the valid segments.",
    )
    p.add_argument("--alpha-allocation-tune-max-combos", type=int, default=24, help="Max allocation configs evaluated.")
    p.add_argument(
        "--alpha-allocation-tune-lambda-grid",
        type=str,
        default="",
        help="Comma-separated grid for allocation lambda_corr, e.g. '0,0.2,0.5,0.8'.",
    )
    p.add_argument(
        "--alpha-allocation-tune-max-weight-grid",
        type=str,
        default="",
        help="Comma-separated grid for allocation max_weight, e.g. '0.5,0.8,1.0'.",
    )
    p.add_argument(
        "--alpha-allocation-tune-turnover-lambda-grid",
        type=str,
        default="",
        help="Comma-separated grid for allocation turnover_lambda, e.g. '0,0.5,2'.",
    )
    p.add_argument(
        "--alpha-allocation-tune-save-top",
        type=int,
        default=10,
        help="Keep only top-N allocation tuning rows in result JSON (0 keeps all).",
    )

    # P2.21: regime-aware allocation (optional).
    p.add_argument(
        "--alpha-allocation-regime-aware",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable regime-aware alpha allocation (dynamic weights by market regime).",
    )
    p.add_argument(
        "--alpha-allocation-regime-mode",
        type=str,
        default="vol",
        choices=["vol", "vol_liq"],
        help="Regime labeling mode used for regime-aware allocation.",
    )
    p.add_argument("--alpha-allocation-regime-window", type=int, default=20, help="Regime feature rolling window.")
    p.add_argument("--alpha-allocation-regime-buckets", type=int, default=3, help="Quantile buckets per regime feature.")
    p.add_argument(
        "--alpha-allocation-regime-min-days",
        type=int,
        default=30,
        help="Minimum fit days required to learn a regime-specific allocation.",
    )
    p.add_argument(
        "--alpha-allocation-regime-smoothing",
        type=float,
        default=0.0,
        help="Optional exponential smoothing for daily regime-aware alpha weights (0 disables).",
    )

    # P2.23: meta-tune regime hyperparams on aggregate valid performance.
    p.add_argument(
        "--alpha-allocation-regime-tune",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Meta-tune regime hyperparams on train->valid across splits (no test leakage).",
    )

    # P2.27: higher-level presets for constraints + Pareto settings.
    p.add_argument(
        "--alpha-allocation-regime-tune-preset",
        type=str,
        default="",
        help="Preset bundle for regime tuning defaults (e.g. 'low_turnover', 'aggressive', 'execution_realistic').",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-metric",
        type=str,
        default="information_ratio",
        choices=["information_ratio", "annualized_return"],
        help="Metric used to rank regime hyperparams on the valid segments.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-max-combos",
        type=int,
        default=24,
        help="Max regime configs evaluated.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-mode-grid",
        type=str,
        default="",
        help="Comma-separated grid for regime mode, e.g. 'vol,vol_liq'.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-window-grid",
        type=str,
        default="",
        help="Comma-separated grid for regime feature window, e.g. '10,20,40'.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-buckets-grid",
        type=str,
        default="",
        help="Comma-separated grid for regime buckets, e.g. '2,3,4'.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-smoothing-grid",
        type=str,
        default="",
        help="Comma-separated grid for regime smoothing, e.g. '0,0.1,0.2'.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-turnover-penalty",
        type=float,
        default=0.0,
        help="DEPRECATED: use --alpha-allocation-regime-tune-turnover-cost-bps. Interpreted as bps cost per unit alpha-weight turnover.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-turnover-cost-bps",
        type=float,
        default=None,
        help="Alpha-weight turnover cost (bps). Applied as: r_adj = r - turnover * cost_bps / 10000 when ranking configs.",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-save-top",
        type=int,
        default=10,
        help="Keep only top-N regime tuning rows in result JSON (0 keeps all).",
    )

    # P2.24: holdings-level revalidation for top proxy configs.
    p.add_argument(
        "--alpha-allocation-regime-tune-holdings-top",
        type=int,
        default=3,
        help="Revalidate the top-N proxy regime configs with holdings-level pricing on the valid segments (0 disables).",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-holdings-metric",
        type=str,
        default="",
        help="Override metric for holdings-level revalidation (empty uses --alpha-allocation-regime-tune-metric).",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-holdings-save-top",
        type=int,
        default=10,
        help="Keep only top-N holdings-validation rows in result JSON (0 keeps all).",
    )


    p.add_argument(
        "--alpha-allocation-regime-tune-max-alpha-turnover",
        type=float,
        default=None,
        help="Constraint for regime tuning selection: max alpha_weight_turnover_mean (None disables).",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-max-turnover-cost-drag-bps",
        type=float,
        default=None,
        help="Constraint for regime tuning selection: max turnover_cost_drag_bps_mean (None disables).",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-max-switch-rate",
        type=float,
        default=None,
        help="Constraint for regime tuning selection: max regime_switch_rate_mean (None disables).",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-max-fallback-frac",
        type=float,
        default=None,
        help="Constraint for regime tuning selection: max fallback_frac_mean (None disables).",
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-prefer-pareto",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, pick the best config among Pareto-efficient rows (still respecting constraints).",
    )

    p.add_argument(
        "--alpha-allocation-regime-tune-pareto-metrics",
        type=str,
        default="",
        help=(
            "Comma-separated extra metrics for the Pareto front. "
            "Proxy stage supports: turnover_cost_drag_bps_mean, regime_switch_rate_mean, fallback_frac_mean. "
            "Holdings stage additionally supports: ensemble_cost_mean, ensemble_borrow_mean, ensemble_turnover_mean."
        ),
    )

    # P2.28: Pareto-based auto selection.
    p.add_argument(
        "--alpha-allocation-regime-tune-selection-method",
        type=str,
        default="",
        help=(
            "Selection method for regime tuning: best_objective (default), knee, utility. "
            "If empty, presets may provide a default."
        ),
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-utility-weights",
        type=str,
        default="",
        help=(
            "Comma-separated key=value weights used by the utility/knee selector in normalized goodness space. "
            "Example: objective=1,alpha_weight_turnover_mean=0.4,turnover_cost_drag_bps_mean=0.4. "
            "Use 'auto' to generate a reasonable cost/constraint-aware default (P2.29)."
        ),
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-include-stability-objectives",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If true and selection-method is knee/utility, include split stability metrics in the Pareto/frontier objectives. "
            "If None, presets may provide a default."
        ),
    )
    p.add_argument(
        "--alpha-allocation-regime-tune-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Pareto scatter plots as run artifacts (requires matplotlib; auto-skips if unavailable).",
    )


    p.add_argument("--min-ir", type=float, default=None, help="Quality gate: minimum IR (None disables).")
    p.add_argument("--max-dd", type=float, default=None, help="Quality gate: max drawdown threshold (e.g. 0.25). None disables.")
    p.add_argument(
        "--max-total-cost-bps",
        type=float,
        default=None,
        help="Quality gate: max mean total cost (cost+borrow) in bps. None disables.",
    )
    p.add_argument("--min-wf-splits", type=int, default=None, help="Quality gate: minimum walk-forward splits (None disables).")

    # P1 walk-forward settings (only used when --eval-mode p1)
    p.add_argument("--wf-train-days", type=int, default=126, help="Walk-forward train window (trading days).")
    p.add_argument("--wf-valid-days", type=int, default=42, help="Walk-forward validation window (trading days).")
    p.add_argument("--wf-test-days", type=int, default=42, help="Walk-forward test window (trading days).")
    p.add_argument("--wf-step-days", type=int, default=42, help="Walk-forward step (how far the window moves each split).")
    p.add_argument(
        "--wf-expanding-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use expanding training window (recommended).",
    )

    # P1 portfolio backtest knobs
    p.add_argument("--rebalance-days", type=int, default=5, help="Rebalance frequency in trading days.")
    p.add_argument("--holding-days", type=int, default=5, help="Holding period in trading days (supports overlap).")
    p.add_argument("--commission-bps", type=float, default=0.0, help="Commission cost in basis points (applied to turnover).")
    p.add_argument("--slippage-bps", type=float, default=0.0, help="Slippage cost in basis points (applied to turnover).")
    p.add_argument("--borrow-bps", type=float, default=0.0, help="Short borrow cost in bps per year.")
    p.add_argument("--half-spread-bps", type=float, default=0.0, help="Half-spread cost in bps applied to |Δw|.")
    p.add_argument("--impact-bps", type=float, default=0.0, help="Nonlinear impact coefficient in bps (participation^exponent).")
    p.add_argument("--impact-exponent", type=float, default=0.5, help="Impact exponent alpha in (trade/ADV)^alpha.")
    p.add_argument("--impact-max-participation", type=float, default=0.2, help="Clip participation to this max.")
    p.add_argument("--portfolio-notional", type=float, default=1_000_000.0, help="Portfolio notional used for impact participation vs ADV.")
    p.add_argument("--turnover-cap", type=float, default=0.0, help="Turnover cap per rebalance step (0 disables).")
    p.add_argument("--max-borrow-bps", type=float, default=0.0, help="Exclude shorts with borrow rate above this threshold (0 disables).")
    p.add_argument("--hard-to-borrow-path", type=str, default="", help="CSV/Parquet with instruments that cannot be shorted.")
    p.add_argument("--borrow-rates-path", type=str, default="", help="CSV/Parquet with datetime,instrument,borrow_bps (annualized).")

    # P2 risk/constraint knobs (only used when --eval-mode p2)
    p.add_argument("--sector-map-path", type=str, default="", help="Optional CSV/Parquet: instrument, sector.")
    p.add_argument("--neutralize-beta", action=argparse.BooleanOptionalAction, default=True, help="Neutralize market beta exposure.")
    p.add_argument("--neutralize-vol", action=argparse.BooleanOptionalAction, default=False, help="Neutralize rolling volatility exposure.")
    p.add_argument("--neutralize-liquidity", action=argparse.BooleanOptionalAction, default=True, help="Neutralize log ADV exposure.")
    p.add_argument("--neutralize-sector", action=argparse.BooleanOptionalAction, default=False, help="Neutralize sector exposure (requires sector map).")
    p.add_argument("--beta-window", type=int, default=60, help="Rolling window for beta exposure (days).")
    p.add_argument("--vol-window", type=int, default=20, help="Rolling window for volatility exposure (days).")
    p.add_argument("--adv-window", type=int, default=20, help="Rolling window for ADV filter/exposure (days).")
    p.add_argument("--min-adv", type=float, default=0.0, help="Minimum average dollar volume (0 disables).")
    p.add_argument("--max-abs-weight", type=float, default=0.0, help="Max absolute weight per name (0 disables).")
    p.add_argument("--max-names-per-side", type=int, default=0, help="Select top/bottom N per side (0 uses quantiles).")

    # P2.4 portfolio construction
    p.add_argument(
        "--construction-method",
        type=str,
        default="heuristic",
        choices=["heuristic", "optimizer"],
        help="Portfolio construction: heuristic (equal-weight top/bottom) or optimizer (ridge + turnover anchor).",
    )
    p.add_argument("--optimizer-l2-lambda", type=float, default=1.0, help="Optimizer L2 penalty on weights.")
    p.add_argument("--optimizer-turnover-lambda", type=float, default=10.0, help="Optimizer penalty for deviating from w_target (lower turnover).")
    p.add_argument("--optimizer-exposure-lambda", type=float, default=0.0, help="Optimizer penalty on exposures (soft risk control).")
    p.add_argument("--optimizer-max-iter", type=int, default=2, help="Projection iterations in optimizer construction.")
    p.add_argument("--optimizer-backend", type=str, default="auto", choices=["auto","ridge","qp"], help="Optimizer backend: auto tries QP (cvxpy) when available.")
    p.add_argument("--optimizer-turnover-cap", type=float, default=0.0, help="Hard cap on 0.5*sum(|w-w_target|) inside the QP optimizer (0 disables).")
    p.add_argument("--optimizer-solver", type=str, default="", help="Optional cvxpy solver name (e.g., OSQP).")
    p.add_argument("--optimizer-cost-aversion", type=float, default=1.0, help="Scale real cost terms (spread/borrow/impact) inside the QP objective. 0 disables.")
    p.add_argument("--optimizer-exposure-slack-lambda", type=float, default=0.0, help="If >0, relax hard exposure neutrality constraints via slack with quadratic penalty.")
    p.add_argument(
        "--optimizer-enforce-participation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce per-name participation cap inside QP: |Δw| <= impact_max_participation*ADV/notional (requires ADV).",
    )

    # P2.8 diagonal risk term inside the optimizer objective
    p.add_argument("--optimizer-risk-aversion", type=float, default=0.0, help="Risk aversion inside optimizer: penalize sum(var * w^2). 0 disables.")
    p.add_argument("--optimizer-risk-window", type=int, default=20, help="Window (days) for diagonal risk variance estimate.")
    p.add_argument("--optimizer-risk-model", type=str, default="diag", choices=["diag", "factor"], help="Risk model inside optimizer objective: diag (per-name variance) or factor (Sigma=BFB^T+D).")
    p.add_argument("--optimizer-factor-risk-window", type=int, default=60, help="Lookback window (days) for factor risk model estimation.")
    p.add_argument("--optimizer-factor-risk-shrink", type=float, default=0.2, help="Diagonal shrinkage for factor covariance (0..1).")
    p.add_argument("--optimizer-factor-risk-ridge", type=float, default=1e-3, help="Ridge added to (B'B) when estimating factor returns.")

    p.add_argument(
        "--optimizer-factor-risk-estimator",
        type=str,
        default="sample",
        choices=["sample", "ewm"],
        help="Factor covariance estimator used in factor risk model: sample or ewm.",
    )
    p.add_argument(
        "--optimizer-factor-risk-shrink-method",
        type=str,
        default="fixed",
        choices=["fixed", "oas"],
        help="Shrinkage method for factor covariance: fixed (diagonal) or oas (automatic shrink to identity).",
    )
    p.add_argument(
        "--optimizer-factor-risk-ewm-halflife",
        type=float,
        default=20.0,
        help="EWM half-life (days) used when estimator=ewm.",
    )
    p.add_argument(
        "--optimizer-factor-return-clip-sigma",
        type=float,
        default=6.0,
        help="Winsorize estimated factor returns at +/- sigma*std (0 disables).",
    )
    p.add_argument(
        "--optimizer-idio-shrink",
        type=float,
        default=0.2,
        help="Idiosyncratic variance shrinkage toward median (0..1).",
    )
    p.add_argument(
        "--optimizer-idio-clip-q",
        type=float,
        default=0.99,
        help="Upper quantile for idiosyncratic variance clipping (0..1).",
    )
    p.add_argument("--target-vol-annual", type=float, default=0.0, help="Vol targeting (annualized). 0 disables.")
    p.add_argument("--vol-target-window", type=int, default=20, help="Window (days) for realized vol estimate.")
    p.add_argument("--vol-target-max-leverage", type=float, default=3.0, help="Cap leverage from vol targeting.")

    # Iteration loop (optional)
    p.add_argument("--max-iterations", type=int, default=1, help="Max research iterations in a single run (default: 1).")
    p.add_argument("--target-ir", type=float, default=0.0, help="Stop early if best information_ratio >= target (0 disables).")

    # Output
    p.add_argument("--output-json", type=str, default="", help="Optional path to save the full result JSON.")
    p.add_argument("--runs-root", type=str, default="runs", help="Directory for local run artifacts (P2.3).")
    p.add_argument("--run-id", type=str, default="", help="Optional run_id override (default is timestamp + config hash).")
    p.add_argument(
        "--save-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save local run artifacts to --runs-root (recommended).",
    )
    p.add_argument("--save-daily-top", type=int, default=1, help="Save OOS daily returns for the top-N SOTA alphas.")
    p.add_argument("--top-k", type=int, default=3, help="How many alphas to keep as SOTA (P0 selection).")
    p.add_argument("--n-quantiles", type=int, default=5, help="Quantiles for spread computation.")
    p.add_argument("--synthetic-n-days", type=int, default=252, help="Synthetic data length (used when --data-path is empty).")
    p.add_argument("--synthetic-n-instruments", type=int, default=50, help="Synthetic universe size.")
    p.add_argument("--synthetic-seed", type=int, default=7, help="Synthetic data RNG seed.")

    # Guardrails for P0 execution
    p.add_argument("--max-nan-ratio", type=float, default=0.95, help="Reject a factor if too many outputs are NaN.")
    p.add_argument("--max-rows", type=int, default=2_000_000, help="Reject datasets larger than this many rows.")
    p.add_argument("--max-code-chars", type=int, default=20_000, help="Reject factor code larger than this many chars.")
    p.add_argument("--max-dsl-chars", type=int, default=5_000, help="Reject factor DSL larger than this many chars.")
    p.add_argument("--prefer-dsl", action=argparse.BooleanOptionalAction, default=True, help="Prefer DSL execution when available (recommended).")
    p.add_argument("--allow-python-exec", action=argparse.BooleanOptionalAction, default=False, help="Allow legacy Python exec() execution (not recommended).")
    p.add_argument("--python-exec-timeout-sec", type=float, default=2.0, help="Timeout (seconds) for legacy Python exec() factors.")
    p.add_argument("--disable-code-safety", action="store_true", help="Disable static denylist scan (unsafe).")

    p.add_argument("--async", dest="use_async", action="store_true", help="Use graph.ainvoke (async).")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    thread_id = args.thread_id or f"thread_{uuid.uuid4().hex[:8]}"

    # Keep track of what the user explicitly passed before presets mutate defaults.
    _raw_cli_regime_util_weights = str(args.alpha_allocation_regime_tune_utility_weights or "").strip()
    _utility_weights_from_preset = False

    state = State(
        trading_idea=args.idea,
        max_iterations=max(1, int(args.max_iterations)),
        target_information_ratio=float(args.target_ir),
    )

    # P2.27: apply optional preset defaults for regime tuning.
    preset = get_preset(args.alpha_allocation_regime_tune_preset)
    if preset is not None:
        # Constraints (only fill when the user didn't specify a value).
        if args.alpha_allocation_regime_tune_max_alpha_turnover is None:
            args.alpha_allocation_regime_tune_max_alpha_turnover = preset.constraints.get(
                "max_alpha_weight_turnover_mean"
            )
        if args.alpha_allocation_regime_tune_max_turnover_cost_drag_bps is None:
            args.alpha_allocation_regime_tune_max_turnover_cost_drag_bps = preset.constraints.get(
                "max_turnover_cost_drag_bps_mean"
            )
        if args.alpha_allocation_regime_tune_max_switch_rate is None:
            args.alpha_allocation_regime_tune_max_switch_rate = preset.constraints.get(
                "max_regime_switch_rate_mean"
            )
        if args.alpha_allocation_regime_tune_max_fallback_frac is None:
            args.alpha_allocation_regime_tune_max_fallback_frac = preset.constraints.get("max_fallback_frac_mean")

        # Pareto preference (only fill when unspecified).
        if args.alpha_allocation_regime_tune_prefer_pareto is None:
            args.alpha_allocation_regime_tune_prefer_pareto = bool(preset.prefer_pareto)

        # Pareto metrics list (only fill when empty).
        if not str(args.alpha_allocation_regime_tune_pareto_metrics or "").strip() and preset.pareto_metrics:
            args.alpha_allocation_regime_tune_pareto_metrics = ",".join([str(x) for x in preset.pareto_metrics])

        # P2.28 selection method + utility weights.
        if not str(args.alpha_allocation_regime_tune_selection_method or "").strip() and preset.selection_method:
            args.alpha_allocation_regime_tune_selection_method = str(preset.selection_method)
        if not str(args.alpha_allocation_regime_tune_utility_weights or "").strip() and preset.utility_weights:
            items = [f"{k}={preset.utility_weights[k]}" for k in sorted(preset.utility_weights.keys())]
            args.alpha_allocation_regime_tune_utility_weights = ",".join(items)
            _utility_weights_from_preset = True
        if args.alpha_allocation_regime_tune_include_stability_objectives is None:
            args.alpha_allocation_regime_tune_include_stability_objectives = bool(preset.include_stability_objectives)

        # Turnover cost bps (only fill when user didn't set either flag).
        if (
            args.alpha_allocation_regime_tune_turnover_cost_bps is None
            and float(args.alpha_allocation_regime_tune_turnover_penalty or 0.0) == 0.0
        ):
            args.alpha_allocation_regime_tune_turnover_cost_bps = float(preset.turnover_cost_bps)


    # P2.25: prefer the explicit cost-bps flag but keep the legacy name as an alias.
    regime_turnover_cost_bps = args.alpha_allocation_regime_tune_turnover_cost_bps
    if regime_turnover_cost_bps is None:
        regime_turnover_cost_bps = args.alpha_allocation_regime_tune_turnover_penalty

    # P2.29: allow "auto" utility weights for regime tuning.
    utility_weights_source = "none"
    utility_calibration_meta = {}
    _cur_util_weights = str(args.alpha_allocation_regime_tune_utility_weights or "").strip()
    if _raw_cli_regime_util_weights.lower() == "auto":
        # Build the same constraint dict used by the selector.
        _cons = {}
        if args.alpha_allocation_regime_tune_max_alpha_turnover is not None:
            _cons["max_alpha_weight_turnover_mean"] = float(args.alpha_allocation_regime_tune_max_alpha_turnover)
        if args.alpha_allocation_regime_tune_max_turnover_cost_drag_bps is not None:
            _cons["max_turnover_cost_drag_bps_mean"] = float(args.alpha_allocation_regime_tune_max_turnover_cost_drag_bps)
        if args.alpha_allocation_regime_tune_max_switch_rate is not None:
            _cons["max_regime_switch_rate_mean"] = float(args.alpha_allocation_regime_tune_max_regime_switch_rate)
        if args.alpha_allocation_regime_tune_max_fallback_frac is not None:
            _cons["max_fallback_frac_mean"] = float(args.alpha_allocation_regime_tune_max_fallback_frac)

        _stab = args.alpha_allocation_regime_tune_include_stability_objectives
        include_stability = True if _stab is None else bool(_stab)

        w_map, w_meta = calibrate_utility_weights(
            turnover_cost_bps=float(regime_turnover_cost_bps or 0.0),
            constraints=_cons,
            include_stability=bool(include_stability),
        )
        args.alpha_allocation_regime_tune_utility_weights = weights_to_kv_string(w_map)
        utility_weights_source = "auto_calibrated"
        utility_calibration_meta = dict(w_meta or {})
    else:
        if _cur_util_weights:
            utility_weights_source = "preset" if _utility_weights_from_preset else "cli"
        else:
            utility_weights_source = "default"

    config = {
        "configurable": {
            "thread_id": thread_id,
            "eval_mode": args.eval_mode,
            "data_path": args.data_path,
            "horizon": args.horizon,
            "top_k": args.top_k,
            "n_quantiles": args.n_quantiles,
            "cost_bps": args.cost_bps,
            "min_obs_per_day": args.min_obs_per_day,
            "min_coverage": args.min_coverage,
            "max_turnover": args.max_turnover,

            "tune": args.tune,
            "tune_metric": args.tune_metric,
            "tune_max_combos": args.tune_max_combos,
            "tune_save_top": args.tune_save_top,
            "tune_turnover_cap": args.tune_turnover_cap,
            "tune_max_abs_weight": args.tune_max_abs_weight,
            "tune_risk_aversion": args.tune_risk_aversion,
            "tune_cost_aversion": args.tune_cost_aversion,
            "ablation_top": args.ablation_top,
            "ablation_mode": args.ablation_mode,
            "regime_analysis": args.regime_analysis,
            "regime_window": args.regime_window,
            "regime_buckets": args.regime_buckets,

            "cost_sensitivity": args.cost_sensitivity,
            "cost_sensitivity_top": args.cost_sensitivity_top,
            "cost_sensitivity_linear_bps": args.cost_sensitivity_linear_bps,
            "cost_sensitivity_half_spread_bps": args.cost_sensitivity_half_spread_bps,
            "cost_sensitivity_impact_bps": args.cost_sensitivity_impact_bps,
            "cost_sensitivity_borrow_bps": args.cost_sensitivity_borrow_bps,
            "cost_sensitivity_borrow_mult": args.cost_sensitivity_borrow_mult,
            "decay_analysis": args.decay_analysis,
            "decay_analysis_top": args.decay_analysis_top,
            "decay_horizons": args.decay_horizons,

            "schedule_sweep": args.schedule_sweep,
            "schedule_sweep_top": args.schedule_sweep_top,
            "schedule_sweep_metric": args.schedule_sweep_metric,
            "schedule_sweep_max_combos": args.schedule_sweep_max_combos,
            "schedule_sweep_rebalance_days": args.schedule_sweep_rebalance_days,
            "schedule_sweep_holding_days": args.schedule_sweep_holding_days,
            "diverse_selection": args.diverse_selection,
            "diverse_lambda": args.diverse_lambda,
            "diverse_use_abs_corr": args.diverse_use_abs_corr,
            "diverse_candidate_pool": args.diverse_candidate_pool,
            "diverse_min_periods": args.diverse_min_periods,
            "selection_tune": args.selection_tune,
            "selection_tune_metric": args.selection_tune_metric,
            "selection_tune_max_combos": args.selection_tune_max_combos,
            "selection_tune_lambda_grid": args.selection_tune_lambda_grid,
            "selection_tune_candidate_pool_grid": args.selection_tune_candidate_pool_grid,
            "selection_tune_topk_grid": args.selection_tune_topk_grid,
            "selection_tune_min_periods": args.selection_tune_min_periods,

            "alpha_selection_preset": args.alpha_selection_preset,
            "alpha_selection_max_pairwise_corr": args.alpha_selection_max_pairwise_corr,
            "alpha_selection_min_valid_ir": args.alpha_selection_min_valid_ir,
            "alpha_selection_min_valid_coverage": args.alpha_selection_min_valid_coverage,
            "alpha_selection_max_total_cost_bps": args.alpha_selection_max_total_cost_bps,
            "alpha_selection_min_wf_test_ir_positive_frac": args.alpha_selection_min_wf_test_ir_positive_frac,

            "ensemble": args.ensemble,

            "holdings_ensemble": args.holdings_ensemble,
            "holdings_ensemble_apply_turnover_cap": args.holdings_ensemble_apply_turnover_cap,

            "alpha_allocation": args.alpha_allocation,
            "alpha_allocation_backend": args.alpha_allocation_backend,
            "alpha_allocation_fit": args.alpha_allocation_fit,
            "alpha_allocation_score_metric": args.alpha_allocation_score_metric,
            "alpha_allocation_lambda": args.alpha_allocation_lambda,
            "alpha_allocation_l2": args.alpha_allocation_l2,
            "alpha_allocation_turnover_lambda": args.alpha_allocation_turnover_lambda,
            "alpha_allocation_max_weight": args.alpha_allocation_max_weight,
            "alpha_allocation_use_abs_corr": args.alpha_allocation_use_abs_corr,
            "alpha_allocation_min_days": args.alpha_allocation_min_days,
            "alpha_allocation_solver": args.alpha_allocation_solver,

            "alpha_allocation_tune": args.alpha_allocation_tune,
            "alpha_allocation_tune_metric": args.alpha_allocation_tune_metric,
            "alpha_allocation_tune_max_combos": args.alpha_allocation_tune_max_combos,
            "alpha_allocation_tune_lambda_grid": args.alpha_allocation_tune_lambda_grid,
            "alpha_allocation_tune_max_weight_grid": args.alpha_allocation_tune_max_weight_grid,
            "alpha_allocation_tune_turnover_lambda_grid": args.alpha_allocation_tune_turnover_lambda_grid,
            "alpha_allocation_tune_save_top": args.alpha_allocation_tune_save_top,

            # P2.21/P2.23: regime-aware allocation + regime meta-tuning.
            "alpha_allocation_regime_aware": args.alpha_allocation_regime_aware,
            "alpha_allocation_regime_mode": args.alpha_allocation_regime_mode,
            "alpha_allocation_regime_window": args.alpha_allocation_regime_window,
            "alpha_allocation_regime_buckets": args.alpha_allocation_regime_buckets,
            "alpha_allocation_regime_min_days": args.alpha_allocation_regime_min_days,
            "alpha_allocation_regime_smoothing": args.alpha_allocation_regime_smoothing,
            "alpha_allocation_regime_tune": args.alpha_allocation_regime_tune,
            "alpha_allocation_regime_tune_preset": args.alpha_allocation_regime_tune_preset,
            "alpha_allocation_regime_tune_metric": args.alpha_allocation_regime_tune_metric,
            "alpha_allocation_regime_tune_max_combos": args.alpha_allocation_regime_tune_max_combos,
            "alpha_allocation_regime_tune_mode_grid": args.alpha_allocation_regime_tune_mode_grid,
            "alpha_allocation_regime_tune_window_grid": args.alpha_allocation_regime_tune_window_grid,
            "alpha_allocation_regime_tune_buckets_grid": args.alpha_allocation_regime_tune_buckets_grid,
            "alpha_allocation_regime_tune_smoothing_grid": args.alpha_allocation_regime_tune_smoothing_grid,
            "alpha_allocation_regime_tune_turnover_penalty": regime_turnover_cost_bps,
            "alpha_allocation_regime_tune_turnover_cost_bps": regime_turnover_cost_bps,
            "alpha_allocation_regime_tune_save_top": args.alpha_allocation_regime_tune_save_top,
            "alpha_allocation_regime_tune_holdings_top": args.alpha_allocation_regime_tune_holdings_top,
            "alpha_allocation_regime_tune_holdings_metric": args.alpha_allocation_regime_tune_holdings_metric,
            "alpha_allocation_regime_tune_holdings_save_top": args.alpha_allocation_regime_tune_holdings_save_top,
            "alpha_allocation_regime_tune_max_alpha_turnover": args.alpha_allocation_regime_tune_max_alpha_turnover,
            "alpha_allocation_regime_tune_max_turnover_cost_drag_bps": args.alpha_allocation_regime_tune_max_turnover_cost_drag_bps,
            "alpha_allocation_regime_tune_max_regime_switch_rate": args.alpha_allocation_regime_tune_max_switch_rate,
            "alpha_allocation_regime_tune_max_fallback_frac": args.alpha_allocation_regime_tune_max_fallback_frac,
            "alpha_allocation_regime_tune_prefer_pareto": args.alpha_allocation_regime_tune_prefer_pareto,
            "alpha_allocation_regime_tune_pareto_metrics": args.alpha_allocation_regime_tune_pareto_metrics,
            "alpha_allocation_regime_tune_selection_method": args.alpha_allocation_regime_tune_selection_method,
            "alpha_allocation_regime_tune_utility_weights": args.alpha_allocation_regime_tune_utility_weights,
            "alpha_allocation_regime_tune_utility_weights_source": utility_weights_source,
            "alpha_allocation_regime_tune_utility_calibration_meta": utility_calibration_meta,
            "alpha_allocation_regime_tune_include_stability_objectives": args.alpha_allocation_regime_tune_include_stability_objectives,
            "alpha_allocation_regime_tune_plots": args.alpha_allocation_regime_tune_plots,

            "min_ir": args.min_ir,
            "max_dd": args.max_dd,
            "max_total_cost_bps": args.max_total_cost_bps,
            "min_wf_splits": args.min_wf_splits,

            # P1 walk-forward + portfolio backtest
            "wf_train_days": args.wf_train_days,
            "wf_valid_days": args.wf_valid_days,
            "wf_test_days": args.wf_test_days,
            "wf_step_days": args.wf_step_days,
            "wf_expanding_train": args.wf_expanding_train,
            "rebalance_days": args.rebalance_days,
            "holding_days": args.holding_days,
            "commission_bps": args.commission_bps,
            "slippage_bps": args.slippage_bps,
            "borrow_bps": args.borrow_bps,
            "half_spread_bps": args.half_spread_bps,
            "impact_bps": args.impact_bps,
            "impact_exponent": args.impact_exponent,
            "impact_max_participation": args.impact_max_participation,
            "portfolio_notional": args.portfolio_notional,
            "turnover_cap": args.turnover_cap,
            "max_borrow_bps": args.max_borrow_bps,
            "hard_to_borrow_path": args.hard_to_borrow_path,
            "borrow_rates_path": args.borrow_rates_path,

            # P2 risk/constraints
            "sector_map_path": args.sector_map_path,
            "neutralize_beta": args.neutralize_beta,
            "neutralize_vol": args.neutralize_vol,
            "neutralize_liquidity": args.neutralize_liquidity,
            "neutralize_sector": args.neutralize_sector,
            "beta_window": args.beta_window,
            "vol_window": args.vol_window,
            "adv_window": args.adv_window,
            "min_adv": args.min_adv,
            "max_abs_weight": args.max_abs_weight,
            "max_names_per_side": args.max_names_per_side,
            "construction_method": args.construction_method,
            "optimizer_l2_lambda": args.optimizer_l2_lambda,
            "optimizer_turnover_lambda": args.optimizer_turnover_lambda,
            "optimizer_exposure_lambda": args.optimizer_exposure_lambda,
            "optimizer_max_iter": args.optimizer_max_iter,
            "optimizer_backend": args.optimizer_backend,
            "optimizer_turnover_cap": args.optimizer_turnover_cap,
            "optimizer_solver": args.optimizer_solver,
            "optimizer_cost_aversion": args.optimizer_cost_aversion,
            "optimizer_exposure_slack_lambda": args.optimizer_exposure_slack_lambda,
            "optimizer_enforce_participation": args.optimizer_enforce_participation,
            "optimizer_risk_aversion": args.optimizer_risk_aversion,
            "optimizer_risk_window": args.optimizer_risk_window,
            "optimizer_risk_model": args.optimizer_risk_model,
            "optimizer_factor_risk_window": args.optimizer_factor_risk_window,
            "optimizer_factor_risk_shrink": args.optimizer_factor_risk_shrink,
            "optimizer_factor_risk_ridge": args.optimizer_factor_risk_ridge,
            "optimizer_factor_risk_estimator": args.optimizer_factor_risk_estimator,
            "optimizer_factor_risk_shrink_method": args.optimizer_factor_risk_shrink_method,
            "optimizer_factor_risk_ewm_halflife": args.optimizer_factor_risk_ewm_halflife,
            "optimizer_factor_return_clip_sigma": args.optimizer_factor_return_clip_sigma,
            "optimizer_idio_shrink": args.optimizer_idio_shrink,
            "optimizer_idio_clip_q": args.optimizer_idio_clip_q,
            "target_vol_annual": args.target_vol_annual,
            "vol_target_window": args.vol_target_window,
            "vol_target_max_leverage": args.vol_target_max_leverage,
            "synthetic_n_days": args.synthetic_n_days,
            "synthetic_n_instruments": args.synthetic_n_instruments,
            "synthetic_seed": args.synthetic_seed,
            "max_nan_ratio": args.max_nan_ratio,
            "max_rows": args.max_rows,
            "max_code_chars": args.max_code_chars,
            "max_dsl_chars": args.max_dsl_chars,
            "prefer_dsl": args.prefer_dsl,
            "allow_python_exec": args.allow_python_exec,
            "python_exec_timeout_sec": args.python_exec_timeout_sec,
            "enable_code_safety": (not args.disable_code_safety),
        }
    }
    if args.checkpoint_id:
        config["configurable"]["checkpoint_id"] = args.checkpoint_id

    run_id = args.run_id or make_run_id(thread_id=thread_id, config=config["configurable"])

    if args.use_async:
        result = asyncio.run(graph.ainvoke(state, config))
    else:
        result = graph.invoke(state, config)

    # Print a small human-readable summary
    print("==== SUMMARY ====")
    hyp = result.get("hypothesis") if isinstance(result, dict) else getattr(result, "hypothesis", None)
    if hyp:
        print("Hypothesis:", hyp)

    sota = result.get("sota_alphas") if isinstance(result, dict) else getattr(result, "sota_alphas", [])
    if sota:
        import math as _math

        def _fmt(v: object) -> str:
            try:
                fv = float(v)  # type: ignore[arg-type]
                return f"{fv:.3f}" if _math.isfinite(fv) else "na"
            except Exception:
                return "na"

        print(f"Top-{len(sota)} SOTA alphas:")
        for i, a in enumerate(sota, 1):
            aid = a.get("alpha_id") or a.get("alphaID") or a.get("id")
            dsl = a.get("dsl") or a.get("code") or ""
            m = a.get("backtest_results") or {}
            mode = m.get("mode") or "p0"
            ir = m.get("information_ratio")
            ic = m.get("ic")
            to = m.get("turnover_mean")
            cov = m.get("coverage_mean")
            msg = f"  {i}. {aid} | mode={mode} IR={_fmt(ir)} IC={_fmt(ic)} TO={_fmt(to)} COV={_fmt(cov)}"
            wf = m.get("walk_forward") or {}
            stab = wf.get("stability") or {}
            if stab:
                msg += f" splits={stab.get('n_splits')} oos_pos_frac={_fmt(stab.get('test_ir_positive_frac'))}"
            print(msg)
            if dsl:
                print("     DSL:", (dsl[:120] + "..." if len(dsl) > 120 else dsl))
    else:
        print("No SOTA alphas selected.")

    # Optionally save the full JSON result
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json as _json
        with out_path.open("w", encoding="utf-8") as f:
            _json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print("Saved JSON to:", str(out_path))

    # Local experiment tracking (P2.3)
    if args.save_run:
        try:
            run_dir = save_run_artifacts(
                runs_root=args.runs_root,
                run_id=run_id,
                thread_id=thread_id,
                config=config["configurable"],
                result=result,
                save_daily_top=int(max(0, args.save_daily_top)),
            )
            print("Saved run artifacts to:", str(run_dir))
        except Exception as e:
            print("[warn] Failed to save run artifacts:", f"{e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
