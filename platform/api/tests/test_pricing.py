from evalforge.pricing import cost_usd


def test_known_model_cost():
    # claude-sonnet-5: $3/M input, $15/M output
    assert cost_usd("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0) == 3.0
    assert cost_usd("claude-sonnet-5", input_tokens=0, output_tokens=1_000_000) == 15.0


def test_unknown_model_costs_zero():
    assert cost_usd("llama3.2", input_tokens=5000, output_tokens=5000) == 0.0


def test_fractional_cost_rounds_to_six_places():
    cost = cost_usd("gpt-4o-mini", input_tokens=1234, output_tokens=567)
    assert cost == round(cost, 6)
    assert cost > 0
