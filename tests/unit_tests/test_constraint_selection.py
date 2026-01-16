from agent.research.constraint_selection import annotate_pareto, compute_pareto_front, select_best_row


def test_select_best_row_constrained_feasible():
    rows = [
        {"config_id": "a", "objective": 1.0, "alpha_weight_turnover_mean": 0.50, "is_pareto": True},
        {"config_id": "b", "objective": 0.8, "alpha_weight_turnover_mean": 0.10, "is_pareto": True},
    ]

    chosen, meta = select_best_row(
        rows,
        objective_key="objective",
        constraints={"max_alpha_weight_turnover_mean": 0.2},
        prefer_pareto=False,
    )

    assert chosen is not None
    assert chosen.get("config_id") == "b"
    assert meta.get("selected_by", "").startswith("constrained_best")
    assert rows[0]["is_feasible"] is False
    assert rows[1]["is_feasible"] is True


def test_select_best_row_constrained_fallback_when_none_feasible():
    rows = [
        {"config_id": "a", "objective": 1.0, "alpha_weight_turnover_mean": 0.50, "is_pareto": True},
        {"config_id": "b", "objective": 0.8, "alpha_weight_turnover_mean": 0.10, "is_pareto": True},
    ]

    chosen, meta = select_best_row(
        rows,
        objective_key="objective",
        constraints={"max_alpha_weight_turnover_mean": 0.01},
        prefer_pareto=False,
    )

    assert chosen is not None
    assert chosen.get("config_id") == "a"
    assert meta.get("selected_by") == "fallback_unconstrained_no_feasible"
    assert meta.get("feasible_count") == 0


def test_select_best_row_prefer_pareto():
    rows = [
        {"config_id": "a", "objective": 1.0, "alpha_weight_turnover_mean": 0.10, "is_pareto": False},
        {"config_id": "b", "objective": 0.9, "alpha_weight_turnover_mean": 0.20, "is_pareto": True},
    ]

    chosen, meta = select_best_row(
        rows,
        objective_key="objective",
        constraints=None,
        prefer_pareto=True,
    )

    assert chosen is not None
    assert chosen.get("config_id") == "b"
    assert meta.get("selected_by") == "best_objective_pareto"


def test_select_best_row_knee_picks_balanced_point():
    rows = [
        {"config_id": "a", "objective": 0.9, "turnover": 1.0},
        {"config_id": "b", "objective": 0.6, "turnover": 0.0},
        {"config_id": "c", "objective": 0.75, "turnover": 0.5},
    ]

    # Annotate a Pareto front so the selector can operate on it.
    annotate_pareto(rows, objectives=[("objective", "max"), ("turnover", "min")])

    chosen, meta = select_best_row(
        rows,
        objective_key="objective",
        constraints=None,
        prefer_pareto=False,
        selection_method="knee",
        objectives=[("objective", "max"), ("turnover", "min")],
    )

    assert chosen is not None
    assert chosen.get("config_id") == "c"
    assert "knee" in str(meta.get("selected_by") or "")


def test_select_best_row_utility_respects_weights():
    rows = [
        {"config_id": "a", "objective": 0.9, "turnover": 1.0},
        {"config_id": "b", "objective": 0.6, "turnover": 0.0},
        {"config_id": "c", "objective": 0.75, "turnover": 0.5},
    ]

    annotate_pareto(rows, objectives=[("objective", "max"), ("turnover", "min")])

    chosen_hi_obj, _ = select_best_row(
        rows,
        objective_key="objective",
        constraints=None,
        prefer_pareto=False,
        selection_method="utility",
        objectives=[("objective", "max"), ("turnover", "min")],
        utility_weights={"objective": 1.0, "turnover": 0.1},
    )
    assert chosen_hi_obj is not None
    assert chosen_hi_obj.get("config_id") == "a"

    chosen_lo_to, _ = select_best_row(
        rows,
        objective_key="objective",
        constraints=None,
        prefer_pareto=False,
        selection_method="utility",
        objectives=[("objective", "max"), ("turnover", "min")],
        utility_weights={"objective": 0.1, "turnover": 1.0},
    )
    assert chosen_lo_to is not None
    assert chosen_lo_to.get("config_id") == "b"


def test_compute_pareto_front_multi_objective():
    rows = [
        {"id": "a", "objective": 1.0, "alpha_weight_turnover_mean": 0.50, "regime_switch_rate_mean": 0.10},
        {"id": "b", "objective": 0.9, "alpha_weight_turnover_mean": 0.20, "regime_switch_rate_mean": 0.05},
        # Dominated by b: lower objective, same turnover, worse switch.
        {"id": "c", "objective": 0.85, "alpha_weight_turnover_mean": 0.20, "regime_switch_rate_mean": 0.06},
    ]

    front = compute_pareto_front(
        rows,
        objectives=[
            ("objective", "max"),
            ("alpha_weight_turnover_mean", "min"),
            ("regime_switch_rate_mean", "min"),
        ],
    )

    assert 0 in front
    assert 1 in front
    assert 2 not in front


def test_annotate_pareto_adds_flags_and_rank():
    rows = [
        {"id": "a", "objective": 1.0, "alpha_weight_turnover_mean": 0.50},
        {"id": "b", "objective": 0.9, "alpha_weight_turnover_mean": 0.20},
        {"id": "c", "objective": 0.85, "alpha_weight_turnover_mean": 0.20},
    ]

    meta = annotate_pareto(rows, objectives=[("objective", "max"), ("alpha_weight_turnover_mean", "min")])
    assert isinstance(meta, dict)
    assert meta.get("pareto_count") is not None
    assert "is_pareto" in rows[0]
    assert "pareto_rank" in rows[0]
