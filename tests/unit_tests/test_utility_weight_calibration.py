from agent.research.utility_weight_calibration import calibrate_utility_weights, weights_to_kv_string


def test_calibrate_utility_weights_is_deterministic_and_nonnegative():
    w1, meta1 = calibrate_utility_weights(
        turnover_cost_bps=0.2,
        constraints={"max_alpha_weight_turnover_mean": 0.2},
        include_stability=True,
    )
    w2, meta2 = calibrate_utility_weights(
        turnover_cost_bps=0.2,
        constraints={"max_alpha_weight_turnover_mean": 0.2},
        include_stability=True,
    )

    assert w1 == w2
    assert meta1.get("method") == "heuristic_v1"
    assert meta2.get("method") == "heuristic_v1"

    assert float(w1.get("objective") or 0.0) > 0.0
    assert float(w1.get("alpha_weight_turnover_mean") or 0.0) >= 0.0

    s = weights_to_kv_string(w1)
    assert "objective=" in s