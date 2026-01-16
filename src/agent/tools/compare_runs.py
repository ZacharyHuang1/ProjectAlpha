"""agent.tools.compare_runs

Compare two local runs produced by P2.3 experiment tracking.

Usage:
  python -m agent.tools.compare_runs runs/<runA> runs/<runB>
  python -m agent.tools.compare_runs runs/<runA> runs/<runB> --output /tmp/compare.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _best_row(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {}
    return df.iloc[0].to_dict()


def _fmt(v: Any) -> str:
    try:
        fv = float(v)
        return f"{fv:.4f}" if np.isfinite(fv) else "na"
    except Exception:
        return "na"


def _run_label(run_dir: Path, cfg: Dict[str, Any]) -> str:
    rid = run_dir.name
    mode = cfg.get("eval_mode") or ""
    data = cfg.get("data_path") or "synthetic"
    return f"{rid} (mode={mode}, data={data})"


def build_report(run_a: Path, run_b: Path) -> str:
    cfg_a = _load_json(run_a / "config.json")
    cfg_b = _load_json(run_b / "config.json")

    m_a = _load_metrics(run_a / "alpha_metrics.csv")
    m_b = _load_metrics(run_b / "alpha_metrics.csv")

    if not m_a.empty and "information_ratio" in m_a.columns:
        m_a = m_a.sort_values("information_ratio", ascending=False, na_position="last")
    if not m_b.empty and "information_ratio" in m_b.columns:
        m_b = m_b.sort_values("information_ratio", ascending=False, na_position="last")

    best_a = _best_row(m_a)
    best_b = _best_row(m_b)

    lines = []
    lines.append("# Run comparison\n\n")
    lines.append("## Runs\n\n")
    lines.append(f"- A: `{_run_label(run_a, cfg_a)}`\n")
    lines.append(f"- B: `{_run_label(run_b, cfg_b)}`\n\n")

    lines.append("## Best alpha (A vs B)\n\n")
    cmp = pd.DataFrame(
        [
            {
                "run": "A",
                "alpha_id": best_a.get("alpha_id"),
                "IR": _fmt(best_a.get("information_ratio")),
                "AnnRet": _fmt(best_a.get("annualized_return")),
                "MDD": _fmt(best_a.get("max_drawdown")),
                "TO": _fmt(best_a.get("turnover_mean")),
                "COV": _fmt(best_a.get("coverage_mean")),
            },
            {
                "run": "B",
                "alpha_id": best_b.get("alpha_id"),
                "IR": _fmt(best_b.get("information_ratio")),
                "AnnRet": _fmt(best_b.get("annualized_return")),
                "MDD": _fmt(best_b.get("max_drawdown")),
                "TO": _fmt(best_b.get("turnover_mean")),
                "COV": _fmt(best_b.get("coverage_mean")),
            },
        ]
    )
    lines.append(cmp.to_markdown(index=False))
    lines.append("\n\n")

    # Top tables
    def _top(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        cols = [
            c
            for c in [
                "alpha_id",
                "mode",
                "information_ratio",
                "annualized_return",
                "max_drawdown",
                "turnover_mean",
                "coverage_mean",
            ]
            if c in df.columns
        ]
        out = df[cols].head(10).copy()
        for c in ["information_ratio", "annualized_return", "max_drawdown", "turnover_mean", "coverage_mean"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    top_a = _top(m_a)
    top_b = _top(m_b)

    lines.append("## Top 10 (A)\n\n")
    lines.append(top_a.to_markdown(index=False) if not top_a.empty else "(no metrics table found)")
    lines.append("\n\n")

    lines.append("## Top 10 (B)\n\n")
    lines.append(top_b.to_markdown(index=False) if not top_b.empty else "(no metrics table found)")
    lines.append("\n")

    return "".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_a", type=str, help="Path to run directory A")
    p.add_argument("run_b", type=str, help="Path to run directory B")
    p.add_argument("--output", type=str, default="", help="Optional output markdown path")
    args = p.parse_args()

    run_a = Path(args.run_a).expanduser().resolve()
    run_b = Path(args.run_b).expanduser().resolve()

    report = build_report(run_a, run_b)
    print(report)

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print("Saved report to:", str(out))


if __name__ == "__main__":
    main()
