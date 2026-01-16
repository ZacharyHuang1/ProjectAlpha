# Update: "Perfect" P0/P0.5 polish

This document summarizes the additional improvements made after the initial P0 + DSL (P0.5) delivery.

## 1) Safer + richer DSL (v2)

File: `src/agent/research/dsl.py`

- Expanded the operator set (time-series, cross-sectional, elementwise):
  - TS: `ts_delta`, `ts_argmax`, `ts_argmin`, `ts_decay_linear`, `ewm_mean`, `ewm_std`
  - CS: `cs_demean`, `winsorize`, `scale`
  - Helpers: `maximum`, `minimum`, `safe_div`
- Added hard limits to keep runtime predictable:
  - AST node limit
  - max rolling window / lag
- Added an LRU cache for parsed + validated expressions (faster repeated evaluation).

Docs: `DSL_GUIDE.md` (updated with the new operator overview + examples).

## 2) Prompt and runtime stay in sync

File: `src/agent/prompts/alpha_coder_prompts.py`

- The alpha-coder system prompt now pulls the allow-list directly from the DSL runtime (`allowed_functions()` / `allowed_columns()`).
- This avoids “prompt drift” when the DSL operator set evolves.

## 3) Improved P0 evaluator (still lightweight)

File: `src/agent/research/alpha_eval.py`

- Per-date metrics:
  - IC, RankIC
  - Top-minus-bottom quantile spread
  - Turnover proxy (membership overlap for top/bottom legs)
  - Long-short return after a simple turnover cost: `--cost-bps`
- Aggregate metrics:
  - `information_ratio`, `annualized_return`, `max_drawdown`
  - `ic_tstat`, `rank_ic_tstat`, `spread_tstat`
  - `coverage_mean`, `turnover_mean`

## 4) Quality gates for factor selection

File: `src/agent/agents/evaluate_alphas_agent.py`

Optional gates (disabled by default):

- `--min-coverage`: reject factors with low average coverage
- `--max-turnover`: reject overly “flippy” factors

Only factors passing the gate are eligible for top-K selection.

## 5) Optional multi-iteration loop

Files:
- `src/agent/graph.py`
- `src/agent/agents/prepare_next_iteration_agent.py`
- `src/agent/agents/hypothesis_agent.py`
- `src/agent/state.py`

New optional loop controls:

- `--max-iterations`: run multiple research iterations in a single command
- `--target-ir`: stop early once the best factor reaches a target IR

This is intentionally small and avoids infinite loops by using a strict max-iteration cap.

## 6) Extra safety for legacy Python exec (still not recommended)

Files:
- `src/agent/research/factor_runner.py`
- `src/agent/research/code_safety.py`

If `--allow-python-exec` is enabled:

- Best-effort execution timeout: `--python-exec-timeout-sec`
- Static code safety check blocks common IO methods like `pd.read_csv`, `.to_csv`, etc.

The default path remains **DSL-first**.

## 7) Better CLI UX + JSON export

File: `main.py`

- Prints a concise summary of hypothesis + top factors.
- Optional `--output-json` to save the full payload (including daily metrics).

## 8) Tests + requirements

Files:
- `tests/unit_tests/test_dsl.py`
- `tests/unit_tests/test_alpha_eval.py`
- `tests/unit_tests/test_factor_runner.py`
- `requirements.txt`

- Added unit coverage for DSL evaluation, factor runner, and evaluator metrics.
- Added `pytest-asyncio` to support the existing async graph integration test.
