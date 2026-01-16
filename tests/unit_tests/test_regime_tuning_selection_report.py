from agent.research.regime_tuning_selection_report import (
    build_regime_tuning_selection_report,
    build_stage_selection_report,
    render_regime_tuning_report_md,
)


def test_stage_report_marks_selected_and_includes_slack():
    rows = [
        {
            "config_id": "c1",
            "mode": "vol",
            "window": 10,
            "buckets": 2,
            "smoothing": 0.0,
            "objective": 1.0,
            "alpha_weight_turnover_mean": 0.10,
            "is_pareto": True,
        },
        {
            "config_id": "c2",
            "mode": "vol",
            "window": 10,
            "buckets": 3,
            "smoothing": 0.0,
            "objective": 1.2,
            "alpha_weight_turnover_mean": 0.30,
            "is_pareto": True,
        },
        {
            "config_id": "c3",
            "mode": "vol",
            "window": 10,
            "buckets": 4,
            "smoothing": 0.0,
            "objective": 0.9,
            "alpha_weight_turnover_mean": 0.05,
            "is_pareto": False,
        },
    ]

    summary = {
        "enabled": True,
        "constraints": {"max_alpha_weight_turnover_mean": 0.2},
        "selection": {
            "selection_method": "best_objective",
            "prefer_pareto": True,
            "objectives": [("objective", "max"), ("alpha_weight_turnover_mean", "min")],
        },
        "pareto_objectives": [("objective", "max"), ("alpha_weight_turnover_mean", "min")],
        "chosen": dict(rows[0]),
    }

    rep = build_stage_selection_report(
        stage="proxy",
        rows=rows,
        summary=summary,
        objective_key="objective",
        top_n=5,
    )

    assert rep["counts"]["total_rows"] == 3
    assert rep["counts"]["candidate_count"] == 2  # pareto rows
    assert rep["counts"]["feasible_count"] == 1  # c2 violates max turnover
    assert rep["counts"]["base_set_size"] == 1

    slack = rep.get("chosen_constraint_slack") or {}
    assert "max_alpha_weight_turnover_mean" in slack
    assert float(slack["max_alpha_weight_turnover_mean"]) > 0.0

    top = rep.get("top_candidates") or []
    assert len(top) == 1
    assert bool(top[0].get("is_selected")) is True


def test_combined_report_renders_markdown():
    rows = [
        {
            "config_id": "c1",
            "mode": "vol",
            "window": 10,
            "buckets": 2,
            "smoothing": 0.0,
            "objective": 1.0,
            "alpha_weight_turnover_mean": 0.10,
            "is_pareto": True,
        }
    ]

    proxy_summary = {
        "enabled": True,
        "constraints": {},
        "selection": {"selection_method": "best_objective", "prefer_pareto": True, "objectives": [("objective", "max")]},
        "pareto_objectives": [("objective", "max")],
        "chosen": dict(rows[0]),
    }

    cfg = {
        "configurable": {
            "alpha_allocation_regime_tune_preset": "low_turnover",
            "alpha_allocation_regime_tune_utility_weights_source": "cli",
            "alpha_allocation_regime_tune_utility_weights": "objective=1",
        }
    }

    rep = build_regime_tuning_selection_report(config=cfg, proxy_rows=rows, proxy_summary=proxy_summary, top_n=5)
    md = render_regime_tuning_report_md(rep)

    assert "# Regime Tuning Report" in md
    assert "Proxy-stage selection" in md