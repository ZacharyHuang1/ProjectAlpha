# src/agent/agents/alpha_coder_agent.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import State
from agent.prompts.alpha_coder_prompts import ALPHA_CODER_USER_PROMPT, get_alpha_coder_system_prompt

from agent.research.dsl import autofix_dsl, critique_dsl


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_alpha(alpha: Dict[str, Any]) -> Dict[str, Any]:
    # Standard schema: alpha_id / expression / description
    alpha_id = alpha.get("alpha_id") or alpha.get("alphaID") or alpha.get("id")
    expression = alpha.get("expression") or alpha.get("expr") or ""
    description = alpha.get("description") or alpha.get("desc") or ""
    variables = alpha.get("variables") or {}
    out: Dict[str, Any] = {
        "alpha_id": alpha_id,
        "expression": expression,
        "description": description,
        "variables": variables,
        # backward-compat keys
        "alphaID": alpha_id,
        "expr": expression,
        "desc": description,
    }
    for k, v in alpha.items():
        if k not in out:
            out[k] = v
    return out


def _extract_dsl_json(content: str) -> str:
    """Best-effort extraction of a JSON object with a 'dsl' field."""
    text = content.strip()
    # Try direct JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and isinstance(obj.get("dsl"), str):
            return obj["dsl"].strip()
    except Exception:
        pass

    # Fallback: find the first {...} block
    l = text.find("{")
    r = text.rfind("}")
    if l >= 0 and r > l:
        try:
            obj = json.loads(text[l : r + 1])
            if isinstance(obj, dict) and isinstance(obj.get("dsl"), str):
                return obj["dsl"].strip()
        except Exception:
            pass

    # Last resort: treat the whole content as a DSL expression
    return text


def _stub_dsl(alpha_id: str, expression: str, description: str) -> str:
    # A deterministic, safe DSL baseline for local runs / CI.
    # Uses only allowed operators and fields.
    expr = (expression + " " + description).lower()
    if "volume" in expr:
        # Volume is scale-dependent; standardize it and neutralize the liquidity proxy.
        return "zscore(cs_neutralize_group(ts_mean(returns(close, 1), 20) * ts_zscore(log1p(volume), 20), sector, log1p(ts_mean(close * volume, 20))))"
    return "zscore(ts_mean(returns(close, 1), 20))"


async def alpha_coder_agent(state: State, config: RunnableConfig) -> Dict[str, Any]:
    seed_alphas = _get_attr_or_key(state, "seed_alphas", []) or []
    if not seed_alphas:
        return {}

    coded_alphas: List[Dict[str, Any]] = []

    # Offline-friendly mode: no API key -> deterministic DSL stubs.
    if not os.getenv("OPENAI_API_KEY"):
        for raw in seed_alphas:
            alpha = _normalize_alpha(raw)
            alpha_id = alpha.get("alpha_id") or "unknown"
            dsl0 = _stub_dsl(alpha_id, alpha.get("expression", ""), alpha.get("description", ""))
            dsl, fixes = autofix_dsl(dsl0)
            warnings = (critique_dsl(dsl) or {}).get("warnings") or []
            coded_alpha = dict(alpha)
            coded_alpha["dsl"] = dsl
            if fixes:
                coded_alpha["dsl_original"] = dsl0
                coded_alpha["dsl_fixes"] = fixes
            if warnings:
                coded_alpha["dsl_warnings"] = warnings
            coded_alphas.append(coded_alpha)
        return {"coded_alphas": coded_alphas}

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    system_prompt = get_alpha_coder_system_prompt()

    for raw in seed_alphas:
        alpha = _normalize_alpha(raw)
        alpha_id = alpha.get("alpha_id")
        if not alpha_id:
            continue

        user_prompt = ALPHA_CODER_USER_PROMPT.format(
            alpha_id=alpha_id,
            expression=alpha.get("expression", ""),
            description=alpha.get("description", ""),
        )

        try:
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            dsl0 = _extract_dsl_json(response.content)
            dsl, fixes = autofix_dsl(dsl0)
            warnings = (critique_dsl(dsl) or {}).get("warnings") or []

            coded_alpha = dict(alpha)
            coded_alpha["dsl"] = dsl
            if fixes:
                coded_alpha["dsl_original"] = dsl0
                coded_alpha["dsl_fixes"] = fixes
            if warnings:
                coded_alpha["dsl_warnings"] = warnings
            coded_alphas.append(coded_alpha)

        except Exception as e:
            coded_alpha = dict(alpha)
            coded_alpha["dsl"] = ""
            coded_alpha["backtest_results"] = {"error": f"CoderError: {e.__class__.__name__}: {e}"}
            coded_alphas.append(coded_alpha)

    return {"coded_alphas": coded_alphas}
