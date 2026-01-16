from agent.research.alpha_selection_report import build_alpha_selection_report, render_alpha_selection_report_md


def test_alpha_selection_report_build_and_render() -> None:
    config = {
        "top_k": 2,
        "diverse_min_periods": 2,
    }

    a1 = {
        "alpha_id": "a1",
        "backtest_results": {
            "information_ratio": 1.1,
            "turnover_mean": 0.2,
            "coverage_mean": 0.9,
            "walk_forward": {
                "oos_daily": [
                    {"datetime": "2020-01-01", "net_return": 0.01},
                    {"datetime": "2020-01-02", "net_return": -0.005},
                    {"datetime": "2020-01-03", "net_return": 0.003},
                ],
                "stability": {"n_splits": 2, "test_ir_mean": 1.0, "test_ir_positive_frac": 1.0},
            },
            "quality_gate": {"passed": True, "reasons": []},
        },
    }
    a2 = {
        "alpha_id": "a2",
        "backtest_results": {
            "information_ratio": 1.0,
            "turnover_mean": 0.25,
            "coverage_mean": 0.95,
            "walk_forward": {
                "oos_daily": [
                    {"datetime": "2020-01-01", "net_return": 0.009},
                    {"datetime": "2020-01-02", "net_return": -0.004},
                    {"datetime": "2020-01-03", "net_return": 0.002},
                ],
                "stability": {"n_splits": 2, "test_ir_mean": 0.9, "test_ir_positive_frac": 1.0},
            },
            "quality_gate": {"passed": True, "reasons": []},
        },
    }
    a3 = {
        "alpha_id": "a3",
        "backtest_results": {
            "information_ratio": 0.8,
            "turnover_mean": 0.1,
            "coverage_mean": 0.85,
            "walk_forward": {
                "oos_daily": [
                    {"datetime": "2020-01-01", "net_return": -0.002},
                    {"datetime": "2020-01-02", "net_return": 0.001},
                    {"datetime": "2020-01-03", "net_return": 0.002},
                ],
                "stability": {"n_splits": 2, "test_ir_mean": 0.7, "test_ir_positive_frac": 0.5},
            },
            "quality_gate": {"passed": True, "reasons": []},
        },
    }
    a4 = {
        "alpha_id": "a4",
        "backtest_results": {
            "information_ratio": 1.2,
            "turnover_mean": 2.0,
            "coverage_mean": 0.9,
            "walk_forward": {"oos_daily": [{"datetime": "2020-01-01", "net_return": 0.001}]},
            "quality_gate": {"passed": False, "reasons": ["turnover"]},
        },
    }

    result = {
        "coded_alphas": [a1, a2, a3, a4],
        "sota_alphas": [a1, a2],
        "selection": {
            "method": "diverse_greedy",
            "metric": "information_ratio",
            "top_k": 2,
            "diversity_lambda": 0.5,
            "use_abs_corr": True,
            "candidate_pool": 3,
            "selected_alpha_ids": ["a1", "a2"],
            "selection_table": [
                {"step": 1, "alpha_id": "a1", "base_score": 1.1, "avg_corr_to_selected": 0.0, "diversity_score": 1.1},
                {"step": 2, "alpha_id": "a2", "base_score": 1.0, "avg_corr_to_selected": 0.9, "diversity_score": 0.55},
            ],
            "correlation_summary": {"n": 2, "avg_abs_corr": 0.9, "max_abs_corr": 0.9},
            "min_periods": 2,
        },
    }

    rep = build_alpha_selection_report(config=config, result=result, top_n=10, corr_pool_max=10)
    assert rep.get("enabled") is True
    assert (rep.get("selection") or {}).get("method") == "diverse_greedy"
    assert "diagnostics" in rep

    md = render_alpha_selection_report_md(rep)
    assert "# Alpha selection report" in md
    assert "## Selected alphas" in md
    assert "alpha_selection_report.json" in md
