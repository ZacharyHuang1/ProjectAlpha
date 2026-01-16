# src/agent/agents/alpha_generator_agent.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import State
from agent.prompts.alpha_prompts import (
    ALPHA_SYSTEM_PROMPT,
    ALPHA_INITIAL_PROMPT,
    ALPHA_ITERATION_PROMPT,
    ALPHA_OUTPUT_FORMAT,
)


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _safe_json_loads(content: str) -> Dict[str, Any]:
    json_start = content.find("{")
    json_end = content.rfind("}") + 1
    if json_start >= 0 and json_end > json_start:
        return json.loads(content[json_start:json_end])
    return json.loads(content)


async def alpha_generator_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    """Generate alpha factors based on the trading hypothesis.

    Output schema for each factor dict (normalized):
    - alpha_id: str
    - expression: str
    - description: str
    - variables: dict[str, str]

    Backward-compat keys are also included:
    - alphaID / expr / desc
    """
    sota_alphas = _get_attr_or_key(state, "sota_alphas", []) or []
    hypothesis = _get_attr_or_key(state, "hypothesis", "") or ""

    # Determine whether we're iterating
    is_first_iteration = not bool(sota_alphas)
    num_factors = 5 if is_first_iteration else 3

    # Dev/test stub (no API key)
    if not os.getenv("OPENAI_API_KEY"):
        factors = [
            {
                "alpha_id": "vol_cond_mom_5_20",
                "expression": r"\text{rank}(\text{ts\_mean}(r_{5}) - \text{ts\_mean}(r_{20})) \cdot \text{rank}(\Delta \log(\text{volume}))",
                "description": "Volume-conditioned momentum: short-term vs medium-term return spread scaled by volume expansion.",
                "variables": {
                    "r_5": "5-day return",
                    "r_20": "20-day return",
                    "volume": "trading volume",
                },
            },
            {
                "alpha_id": "price_range_breakout",
                "expression": r"\text{rank}\left(\frac{\text{close} - \text{ts\_min}(\text{low}, 20)}{\text{ts\_max}(\text{high}, 20) - \text{ts\_min}(\text{low}, 20)}\right)",
                "description": "20-day range position: where today's close sits within the 20-day high-low range.",
                "variables": {"close": "closing price", "high": "high price", "low": "low price"},
            },
        ][:num_factors]

        seed_alphas = []
        for f in factors:
            # We standardize on: alpha_id / expression / description.
            # The extra keys (alphaID/expr/desc) are kept for backwards-compat with older code paths.
            seed_alphas.append(
                {
                    **f,
                    "alphaID": f["alpha_id"],
                    "expr": f["expression"],
                    "desc": f["description"],
                }
            )
        return {"seed_alphas": seed_alphas}

    # Build prompt
    if is_first_iteration:
        user_prompt = ALPHA_INITIAL_PROMPT.format(
            hypothesis=hypothesis,
            num_factors=num_factors,
            output_format=ALPHA_OUTPUT_FORMAT,
        )
    else:
        user_prompt = ALPHA_ITERATION_PROMPT.format(
            hypothesis=hypothesis,
            sota_alphas=sota_alphas,
            num_factors=num_factors,
            output_format=ALPHA_OUTPUT_FORMAT,
        )

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    response = await llm.ainvoke(
        [
            {"role": "system", "content": ALPHA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    factor_json = _safe_json_loads(response.content)

    seed_alphas: List[Dict[str, Any]] = []
    for factor_name, factor_data in factor_json.items():
        normalized = {
            "alpha_id": factor_name,
            "expression": factor_data.get("formulation", ""),
            "description": factor_data.get("description", ""),
            "variables": factor_data.get("variables", {}),
        }
        seed_alphas.append(
            {
                **normalized,
                # Backward-compat keys
                "alphaID": normalized["alpha_id"],
                "expr": normalized["expression"],
                "desc": normalized["description"],
            }
        )

    return {"seed_alphas": seed_alphas}
