# src/agent/prompts/alpha_coder_prompts.py

"""Prompts for converting alpha ideas into the safe factor DSL.

The allow-list (fields/functions) is sourced from agent.research.dsl so the prompt
stays in sync with the runtime DSL engine.
"""

from __future__ import annotations

from agent.research.dsl import allowed_columns, allowed_functions


_ALPHACODER_SYSTEM_TEMPLATE = """You are an expert quantitative researcher.
Convert an alpha idea into a SAFE factor DSL expression.

Rules:
- Output MUST be a single JSON object.
- The JSON MUST contain a key named 'dsl' whose value is the DSL expression as a string.
- Use ONLY the allowed fields and allowed functions below.
- Do NOT output Python code.
- Do NOT use attribute access (no dots), no indexing, no imports, no strings.
- Prefer simple, robust expressions (small windows like 5/10/20/60).
- Avoid mixing units: do NOT multiply returns with raw volume. If using volume, normalize it (ts_rank(volume,w), rel_volume(volume,w), ts_zscore(log1p(volume),w)).
- If you need size/liquidity neutrality, consider cs_neutralize(signal, log1p(ts_mean(close*volume,20))).
- If you have sector labels, prefer group neutrality: cs_demean_group(signal, sector) or cs_neutralize_group(signal, sector, log1p(ts_mean(close*volume,20))).

Allowed fields:
{allowed_fields}

Allowed functions:
{allowed_functions}

Allowed syntax:
- numbers, parentheses
- +, -, *, /, **, %
- comparisons: <, <=, >, >=, ==, !=
- boolean: and/or (combine comparisons)
"""

ALPHA_CODER_USER_PROMPT = """Convert the following alpha into a single DSL expression.

Alpha ID: {alpha_id}
Expression (may be LaTeX or informal): {expression}
Description: {description}

Return JSON only, like:
{{"dsl": "zscore(ts_mean(returns(close, 1), 20))"}}
"""


def get_alpha_coder_system_prompt() -> str:
    fields = ", ".join(allowed_columns())
    funcs = ", ".join(allowed_functions())
    return _ALPHACODER_SYSTEM_TEMPLATE.format(
        allowed_fields=fields,
        allowed_functions=funcs,
    )
