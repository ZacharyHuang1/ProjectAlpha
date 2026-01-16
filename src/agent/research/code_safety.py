"""agent.research.code_safety

Static safety checks for LLM-generated factor code.

This is a lightweight guardrail for P0:
- reject obvious IO/network/process usage
- restrict imports to a small allow-list
- disallow while-loops and dunder attribute access

This is NOT a perfect sandbox. For production, prefer a small DSL/operator library.
"""

from __future__ import annotations

import ast
import re
from typing import Optional, Set


_DEFAULT_ALLOWED_IMPORTS: Set[str] = {"pandas", "numpy", "math"}

_FORBIDDEN_ATTRS = {
    # pandas IO helpers
    "read_csv",
    "read_parquet",
    "read_pickle",
    "read_hdf",
    "read_excel",
    "read_sql",
    "to_csv",
    "to_parquet",
    "to_pickle",
    "to_hdf",
    "to_excel",
    "to_sql",
    "to_json",
}

_FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "dir",
    "help",
    "getattr",
    "setattr",
    "delattr",
}

_FORBIDDEN_WORDS_RE = re.compile(
    r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|http|ftplib|importlib|builtins|pickle|marshal|ctypes)\b",
    flags=re.IGNORECASE,
)


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, allowed_imports: Set[str]):
        self.allowed_imports = allowed_imports
        self.issue: Optional[str] = None

    def _set_issue(self, msg: str) -> None:
        if self.issue is None:
            self.issue = msg

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in self.allowed_imports:
                self._set_issue(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            self._set_issue("ImportFrom with no module is not allowed")
            return
        root = node.module.split(".")[0]
        if root not in self.allowed_imports:
            self._set_issue(f"Import not allowed: {node.module}")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._set_issue("While loops are not allowed in factor code")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self._set_issue("Dunder attribute access is not allowed")

        if node.attr in _FORBIDDEN_ATTRS:
            self._set_issue(f"Forbidden IO method detected: {node.attr}")

        if isinstance(node.value, ast.Name) and node.value.id == "pd" and node.attr == "io":
            self._set_issue("pandas internal IO is not allowed (pd.io)")

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id != "__name__":
            self._set_issue("Dunder names are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            self._set_issue(f"Call not allowed: {node.func.id}()")

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr.startswith("read_"):
                self._set_issue("pandas IO is not allowed (read_*)")
            if attr in {"to_csv", "to_parquet", "to_pickle", "to_sql", "to_json"}:
                self._set_issue("pandas IO is not allowed (to_*)")

        self.generic_visit(node)


def check_factor_code_safety(
    code: str,
    *,
    allowed_imports: Optional[Set[str]] = None,
    max_chars: int = 20_000,
) -> Optional[str]:
    """Return an error string if code is unsafe, otherwise None."""

    if code is None or not str(code).strip():
        return "Missing code"

    if len(code) > max_chars:
        return f"Code too long ({len(code)} chars)"

    if _FORBIDDEN_WORDS_RE.search(code):
        return "Forbidden module/keyword detected"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg}"

    visitor = _SafetyVisitor(set(allowed_imports or _DEFAULT_ALLOWED_IMPORTS))
    visitor.visit(tree)
    return visitor.issue
