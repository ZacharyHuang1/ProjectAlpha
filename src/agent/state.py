# src/agent/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class State:
    """Define the state for alpha generation workflow.

    Notes:
    - LangGraph state is typically a dict-like object. We keep this as a dataclass for clarity.
    - The repository's tests send an input key called ``changeme``; we keep it here to
      remain compatible with the starter-template test harness.
    """

    # Template/test compatibility
    changeme: Optional[Any] = None

    # Input/Output
    trading_idea: str = ""

    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 1
    target_information_ratio: float = 0.0


    # Hypothesis fields
    hypothesis: str = ""
    reason: str = ""
    concise_reason: str = ""
    concise_observation: str = ""
    concise_justification: str = ""
    concise_knowledge: str = ""

    # Alpha generation
    seed_alphas: List[Dict[str, Any]] = field(default_factory=list)
    coded_alphas: List[Dict[str, Any]] = field(default_factory=list)

    # For SOTA tracking and feedback loop
    sota_alphas: List[Dict[str, Any]] = field(default_factory=list)
    feedback: Optional[Dict[str, Any]] = None

    # Evaluation artifacts (optional, filled in eval agent)
    selection: Optional[Dict[str, Any]] = None
    alpha_correlation: Optional[Dict[str, Any]] = None
    ensemble: Optional[Dict[str, Any]] = None
    ensemble_holdings: Optional[Dict[str, Any]] = None
    ensemble_holdings_allocated: Optional[Dict[str, Any]] = None

    # Persistence metadata (optional)
    _persistence: Optional[Dict[str, Any]] = None

    # Historical data for iterations
    hypothesis_history: List[Dict[str, Any]] = field(default_factory=list)
    alpha_history: List[Dict[str, Any]] = field(default_factory=list)
