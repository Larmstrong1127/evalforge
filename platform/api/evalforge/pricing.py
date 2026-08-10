"""Per-model pricing in USD per million tokens.

Unknown models (local Ollama models) cost 0. Prices are a point-in-time
snapshot; the benchmark report records the date they were captured. Model
identifiers age out — Google retired `gemini-2.0-flash` (returns 404 "no
longer available") sometime between this table's original snapshot and
2026-07-03, discovered when a real benchmark run against it failed outright.
`gemini-2.5-flash-lite` is its direct successor at the same price point and
is the current entry here.
"""

# (input $/M, output $/M) — snapshot 2026-07
_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic list prices, snapshot 2026-08-09. claude-sonnet-5 carries an
    # introductory $2.00/$10.00 rate through 2026-08-31; the standard $3/$15
    # is used here so a benchmark run does not silently under-report cost once
    # the promotion lapses.
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gemini-2.5-flash-lite": (0.1, 0.4),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in _PRICES:
        return 0.0
    input_price, output_price = _PRICES[model]
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return round(cost, 6)
