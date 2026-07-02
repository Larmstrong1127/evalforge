"""Per-model pricing in USD per million tokens.

Unknown models (local Ollama models) cost 0. Prices are a point-in-time
snapshot; the benchmark report records the date they were captured.
"""

# (input $/M, output $/M) — snapshot 2026-07
_PRICES: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gemini-2.0-flash": (0.1, 0.4),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in _PRICES:
        return 0.0
    input_price, output_price = _PRICES[model]
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return round(cost, 6)
