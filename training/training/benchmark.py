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
        report[name] = {
            "agreement": agreement,
            "total_cost_usd": sum(result.costs_usd),
            "cost_per_1k_usd": (sum(result.costs_usd) / len(result.costs_usd)) * 1000,
            "p50_latency_ms": statistics.median(sorted_latencies),
            "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)]
            if len(sorted_latencies) > 1
            else sorted_latencies[0],
        }
    return report


async def score_with_local_judge(examples, checkpoint_path: str) -> BenchmarkResult:
    """Real implementation: loads the checkpoint once, scores every example,
    records per-item latency. Cost is always 0.0 for a local model."""
    import time

    import torch
    from transformers import AutoModelForSequenceClassification

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
            text = f"Q: {ex.question} C: {ex.context} A: {ex.answer}"
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            predictions.append(int(logits.argmax(dim=-1).item()))
            latencies_ms.append((time.perf_counter() - start) * 1000)

    return BenchmarkResult(
        predictions=predictions, costs_usd=[0.0] * len(examples), latencies_ms=latencies_ms
    )


async def score_with_llm_judge(
    examples, provider_name: str, model_name: str, settings: Settings
) -> BenchmarkResult:
    """Real implementation: uses evalforge.providers to score each example,
    parsing a strict-JSON faithful/hallucinated verdict from the response."""
    import json
    import time

    from evalforge.pricing import cost_usd
    from evalforge.providers import get_provider

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
        start = time.perf_counter()
        completion = await provider.generate(model=model_name, prompt=prompt)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        data = json.loads(completion.text)
        predictions.append(1 if data["label"] == "hallucinated" else 0)
        costs_usd.append(cost_usd(model_name, completion.input_tokens, completion.output_tokens))

    return BenchmarkResult(predictions=predictions, costs_usd=costs_usd, latencies_ms=latencies_ms)
