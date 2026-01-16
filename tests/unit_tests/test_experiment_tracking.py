import tempfile
from pathlib import Path

from agent.services.experiment_tracking import extract_ablation_results_table, make_run_id, save_run_artifacts


def test_save_run_artifacts_writes_files():
    result = {
        "coded_alphas": [
            {
                "alpha_id": "a1",
                "dsl": "ts_mean(returns(close,1),5)",
                "backtest_results": {
                    "mode": "p2",
                    "information_ratio": 1.23,
                    "annualized_return": 0.10,
                    "max_drawdown": -0.05,
                    "turnover_mean": 0.2,
                    "coverage_mean": 0.9,
                    "walk_forward": {
                        "oos_daily": [{"datetime": "2020-01-01", "net_return": 0.001}],
                        "stability": {"n_splits": 2, "test_ir_positive_frac": 1.0},
                    },
                },
            }
        ],
        "sota_alphas": [
            {
                "alpha_id": "a1",
                "dsl": "ts_mean(returns(close,1),5)",
                "backtest_results": {
                    "mode": "p2",
                    "information_ratio": 1.23,
                    "walk_forward": {"oos_daily": [{"datetime": "2020-01-01", "net_return": 0.001}]},
                },
            }
        ],
        "trading_idea": "demo",
    }

    cfg = {"thread_id": "t1", "eval_mode": "p2", "data_path": ""}

    with tempfile.TemporaryDirectory() as td:
        run_id = make_run_id("t1", cfg)
        run_dir = save_run_artifacts(
            runs_root=Path(td),
            run_id=run_id,
            thread_id="t1",
            config=cfg,
            result=result,
            save_daily_top=1,
        )

        assert (run_dir / "config.json").exists()
        assert (run_dir / "result.json").exists()
        assert (run_dir / "alpha_metrics.csv").exists()
        assert (run_dir / "sota_alphas.json").exists()
        assert (run_dir / "SUMMARY.md").exists()
        assert (run_dir / "REPORT.md").exists()
        assert (run_dir / "ALPHA_SELECTION_REPORT.md").exists()
        assert (run_dir / "alpha_selection_report.json").exists()
        assert (run_dir / "alpha_selection_top_candidates.csv").exists()
        assert (run_dir / "daily" / "a1_oos_daily.csv").exists()


def test_extract_ablation_results_table_supports_both_modes() -> None:
    result = {
        "coded_alphas": [
            {
                "alpha_id": "a1",
                "backtest_results": {
                    "tuning": {
                        "ablation": {
                            "end_to_end": {
                                "scenarios": [
                                    {"scenario": "full", "information_ratio": 1.0, "total_cost_bps": 5.0},
                                ]
                            },
                            "execution_only": {
                                "scenarios": [
                                    {"scenario": "full", "information_ratio": 1.2, "mean_cost_drag_bps": 6.0},
                                ]
                            },
                        }
                    }
                },
            }
        ]
    }

    df = extract_ablation_results_table(result)
    assert not df.empty
    assert set(df["ablation_mode"].unique()) == {"end_to_end", "execution_only"}
