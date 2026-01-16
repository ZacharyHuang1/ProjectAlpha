# P0 Hardening Update: Guardrails for Execution and Data

Note: The project is now DSL-first by default (see `DSL_GUIDE.md`). The `exec()` guardrails remain useful for the optional legacy Python path.

This update adds lightweight guardrails to make the P0 loop safer and less brittle.

## What changed

### 1) Data validation (OHLCV schema + minimal NaN policy)
File: `src/agent/services/market_data.py`

- Enforces `MultiIndex(datetime, instrument)`
- Enforces required columns: `open, high, low, close, volume`
- Coerces numeric dtypes and replaces `inf` with `NaN`
- Minimal NaN policy:
  - drop rows with missing `close`
  - fill missing `open` with previous `close` (per instrument), then fall back to `close`
  - fill missing `high/low` with a conservative envelope around `open/close`
  - fill missing `volume` with `0`

### 2) Static denylist scan for factor code
File: `src/agent/research/code_safety.py`

Before running `exec()`, factor code is parsed and rejected if it contains:
- disallowed imports (anything outside `pandas`, `numpy`, `math`)
- obvious IO/network/system keywords (e.g., `os`, `subprocess`, `socket`, `requests`, `urllib`, ...)
- dangerous builtins (`eval`, `exec`, `open`, `globals`, `locals`, ...)
- while-loops (to reduce the risk of infinite loops)
- dunder access (e.g., `__class__`, `__subclasses__`) which is a common escape hatch
- common pandas IO patterns (`read_*`, `to_csv`, `to_parquet`, ...)

### 3) Runner-level constraints
File: `src/agent/research/factor_runner.py`

- Rejects datasets larger than `max_rows` (default: 2,000,000 rows)
- Rejects factor outputs with too many NaNs (`max_nan_ratio`, default: 0.95)
- Normalizes factor outputs:
  - aligns to the input index
  - enforces a single output column named `alpha_id`
  - converts to numeric and replaces `inf` with `NaN`

### 4) CLI knobs
File: `main.py`

New flags:
- `--max-nan-ratio`
- `--max-rows`
- `--max-code-chars`
- `--disable-code-safety` (unsafe)

## Notes / limitations

- This is still a best-effort sandbox. It reduces obvious failure modes but it is not a full security boundary.
- For production use, the recommended direction is to replace Python `exec()` with a controlled DSL/operator library.

## Legacy Python exec timeout

If you enable `--allow-python-exec`, the runner applies a best-effort timeout (`--python-exec-timeout-sec`).
This is only a guardrail; the recommended path is still DSL-first.

The static code safety check also blocks common pandas IO methods such as `read_csv`, `to_csv`, etc.
