"""Compares the locally fine-tuned judge against LLM-as-judge (Claude, GPT,
Gemini) on the RAGTruth held-out set: agreement with ground truth, cost per
1K evaluations, and p50/p95 latency.

Reuses evalforge.providers directly from platform/api — install it into this
venv with `pip install -e ../platform/api` before running for real.

Running this against real cloud APIs costs money (per the design doc's
~$10-15 budget) and is NOT invoked automatically by any test in this file —
tests here only exercise the pure aggregation logic and validate input shape.
"""
import statistics
from dataclasses import dataclass

from evalforge.config import Settings


@dataclass
class BenchmarkResult:
    predictions: list[int]
    costs_usd: list[float]
    latencies_ms: list[float]

    def __post_init__(self) -> None:
        if not (len(self.predictions) == len(self.costs_usd) == len(self.latencies_ms)):
            raise ValueError(
                "predictions, costs_usd, and latencies_ms must all be the same length"
            )


def aggregate_benchmark_results(
    ground_truth: list[int], judges: dict[str, BenchmarkResult]
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for name, result in judges.items():
        if len(result.predictions) != len(ground_truth):
            raise ValueError(
                f"judge '{name}' has {len(result.predictions)} predictions, "
                f"expected {len(ground_truth)} to match ground_truth"
            )
        correct = sum(
            1 for gt, pred in zip(ground_truth, result.predictions, strict=True) if gt == pred
        )
        agreement = correct / len(ground_truth)
        sorted_latencies = sorted(result.latencies_ms)
        if len(sorted_latencies) > 1:
            p95_index = max(0, int(len(sorted_latencies) * 0.95) - 1)
            p95_latency_ms = sorted_latencies[p95_index]
        else:
            p95_latency_ms = sorted_latencies[0]
        report[name] = {
            "agreement": agreement,
            "total_cost_usd": sum(result.costs_usd),
            "cost_per_1k_usd": (sum(result.costs_usd) / len(result.costs_usd)) * 1000,
            "p50_latency_ms": statistics.median(sorted_latencies),
            "p95_latency_ms": p95_latency_ms,
        }
    return report


async def score_with_local_judge(examples, checkpoint_path: str) -> BenchmarkResult:
    """Real implementation: loads the checkpoint once, scores every example,
    records per-item latency. Cost is always 0.0 for a local model."""
    import time

    import torch
    from transformers import AutoModelForSequenceClassification

    from training.hallucination_encoding import encode_qca
    from training.models.classifier import build_tokenizer

    tokenizer = build_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[int] = []
    latencies_ms: list[float] = []
    with torch.no_grad():
        for ex in examples:
            start = time.perf_counter()
            inputs = encode_qca(
                tokenizer, ex.question, ex.context, ex.answer, 512, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            predictions.append(int(logits.argmax(dim=-1).item()))
            latencies_ms.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        # Cost is 0.0 for a local model because we only count paid API spend here;
        # this ignores GPU/electricity amortization, which is a real but separate cost.
        predictions=predictions, costs_usd=[0.0] * len(examples), latencies_ms=latencies_ms
    )


def _strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = lines[1:]  # drop opening fence (with optional language tag)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


async def score_with_llm_judge(
    examples,
    provider_name: str,
    model_name: str,
    settings: Settings,
    request_delay_s: float = 0.5,
    max_retries: int = 4,
    retry_base_delay_s: float = 2.0,
) -> BenchmarkResult:
    """Real implementation: uses evalforge.providers to score each example,
    parsing a strict-JSON faithful/hallucinated verdict from the response.

    This hits real paid APIs one example at a time with no checkpointing, so a
    single failed example must not lose all prior progress and spent cost. A
    retryable ProviderError (e.g. HTTP 429) is retried with exponential
    backoff up to max_retries before being treated as a real failure; a small
    fixed delay between every request (request_delay_s) additionally avoids
    bursting past rate limits in the first place rather than reacting only
    after they're hit. Any example that still fails after retries (a
    non-retryable provider error, or a malformed/non-JSON response) is
    skipped with a printed warning rather than raising — as a result, the
    returned BenchmarkResult may contain fewer entries than len(examples).
    """
    import asyncio
    import json
    import time

    from evalforge.pricing import cost_usd
    from evalforge.providers import ProviderError, get_provider

    provider = get_provider(provider_name, settings)
    predictions: list[int] = []
    costs_usd: list[float] = []
    latencies_ms: list[float] = []

    for ex in examples:
        prompt = (
            "Classify whether the ANSWER is faithful to the CONTEXT or hallucinated.\n"
            f"CONTEXT: {ex.context}\nQUESTION: {ex.question}\nANSWER: {ex.answer}\n"
            'Reply with strict JSON only: {"label": "faithful"} or {"label": "hallucinated"}'
        )
        attempt = 0
        completion = None
        while True:
            start = time.perf_counter()
            try:
                completion = await provider.generate(model=model_name, prompt=prompt)
                break
            except ProviderError as exc:
                if not exc.retryable or attempt >= max_retries:
                    print(f"warning: failed to score example, skipping: {exc}")
                    break
                await asyncio.sleep(retry_base_delay_s * (2**attempt))
                attempt += 1

        if request_delay_s:
            await asyncio.sleep(request_delay_s)

        if completion is None:
            continue

        latency_ms = (time.perf_counter() - start) * 1000
        try:
            data = json.loads(_strip_markdown_fence(completion.text))
            label = data["label"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"warning: failed to score example, skipping: {exc}")
            continue

        predictions.append(1 if label == "hallucinated" else 0)
        costs_usd.append(cost_usd(model_name, completion.input_tokens, completion.output_tokens))
        latencies_ms.append(latency_ms)

    return BenchmarkResult(predictions=predictions, costs_usd=costs_usd, latencies_ms=latencies_ms)
