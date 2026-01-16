# P0.5 Update: DSL-First Factor Execution (No `exec()` by Default)

This update improves safety and reproducibility by making factor execution **DSL-first**.

## Summary

- Added a small operator DSL (AST allow-list).
- Switched the default factor runner to evaluate **DSL expressions**.
- Kept the old Python `exec()` path behind an explicit flag (legacy / debugging only).
- Updated the alpha coder to output a DSL expression (`dsl`) instead of Python code.

## Files changed / added

### Added

- `src/agent/research/dsl.py`  
  DSL operators + safe evaluator (AST allow-list, no attributes/subscripts/strings/loops).

- `DSL_GUIDE.md`  
  Usage, syntax, and operator list.

### Updated

- `src/agent/research/factor_runner.py`  
  Runs DSL by default; Python execution is optional via `--allow-python-exec`.

- `src/agent/agents/alpha_coder_agent.py`  
  Produces `dsl` field (JSON-based). Includes deterministic stubs when `OPENAI_API_KEY` is not set.

- `src/agent/prompts/alpha_coder_prompts.py`  
  Prompt updated to request a single JSON object with `dsl`.

- `src/agent/agents/evaluate_alphas_agent.py`  
  Passes DSL/Python execution config into the runner.

- `main.py`  
  New CLI knobs:
  - `--prefer-dsl / --no-prefer-dsl`
  - `--allow-python-exec / --no-allow-python-exec`
  - `--max-dsl-chars`

- `src/agent/database/operations/alpha_operations.py`  
  Persists DSL into the `code` column (best-effort; schema remains unchanged).

## How to run

DSL-first (recommended):

```bash
pip install -r requirements.txt
python main.py --idea "Volume-conditioned momentum" --thread-id demo1
```

Legacy Python execution (not recommended):

```bash
python main.py --no-prefer-dsl --allow-python-exec
```

## Limitations

- The DSL is intentionally small. If the model outputs functions outside the allow-list, execution will fail with `DSLValidationError`.
- For production-grade performance, operators may need vectorization improvements (rolling operations are currently implemented with `groupby(...).rolling(...)` patterns).
