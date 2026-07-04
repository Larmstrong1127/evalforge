"""One-off script: runs the real benchmark against 200 RAGTruth examples,
comparing the local lr-2e5 judge against Claude/GPT/Gemini as LLM judges.

This is a manually-invoked operational script, not part of the tested
package — it makes real, paid API calls. Not covered by the test suite by
design (see training/training/benchmark.py's module docstring).
"""
import asyncio
import json
import random

from evalforge.config import Settings

from training.benchmark import (
    aggregate_benchmark_results,
    score_with_llm_judge,
    score_with_local_judge,
)
from training.data.ragtruth import load_ragtruth_examples

SAMPLE_SIZE = 200
SEED = 42
CHECKPOINT = "checkpoints/lr-2e5"


async def main() -> None:
    # pydantic-settings resolves env_file relative to the CURRENT WORKING
    # DIRECTORY at construction time, not relative to config.py's location.
    # Since this script is meant to be run from training/, but the .env
    # file lives in platform/api/, pass the path explicitly rather than
    # relying on the model's default relative ".env" (which silently loads
    # empty keys if the cwd doesn't happen to be platform/api/ — exactly
    # what happened on the first real run: a clean 401, not a hang or a
    # billed request, but wasted time nonetheless).
    settings = Settings(_env_file="../platform/api/.env")  # type: ignore[call-arg]
    all_examples = load_ragtruth_examples()
    sample = random.Random(SEED).sample(all_examples, SAMPLE_SIZE)
    ground_truth = [ex.label for ex in sample]

    print(f"Scoring {SAMPLE_SIZE} RAGTruth examples with local judge (lr-2e5)...")
    local_result = await score_with_local_judge(sample, CHECKPOINT)
    print(f"  local judge: {len(local_result.predictions)} scored")

    print("Scoring with Claude (claude-sonnet-5)...")
    claude_result = await score_with_llm_judge(sample, "anthropic", "claude-sonnet-5", settings)
    print(f"  claude: {len(claude_result.predictions)} scored")

    print("Scoring with GPT (gpt-4o)...")
    gpt_result = await score_with_llm_judge(sample, "openai", "gpt-4o", settings)
    print(f"  gpt-4o: {len(gpt_result.predictions)} scored")

    print("Scoring with Gemini (gemini-2.5-flash-lite)...")
    gemini_result = await score_with_llm_judge(sample, "gemini", "gemini-2.5-flash-lite", settings)
    print(f"  gemini: {len(gemini_result.predictions)} scored")

    # aggregate_benchmark_results requires predictions length == ground_truth
    # length; per-example failures in score_with_llm_judge produce shorter
    # results, so trim ground_truth per-judge to match what actually scored.
    judges = {
        "local-deberta-lr2e5": local_result,
        "claude-sonnet-5": claude_result,
        "gpt-4o": gpt_result,
        "gemini-2.5-flash-lite": gemini_result,
    }

    report = {}
    for name, result in judges.items():
        n = len(result.predictions)
        gt_for_judge = ground_truth[:n] if n == len(ground_truth) else ground_truth
        # if a judge dropped examples, we can't safely align by position;
        # report a warning instead of silently mismatching ground truth.
        if n != len(ground_truth):
            print(f"WARNING: {name} scored {n}/{len(ground_truth)} examples "
                  f"(some were skipped on failure) — excluding from aggregate report")
            continue
        report.update(aggregate_benchmark_results(gt_for_judge, {name: result}))

    print()
    print(json.dumps(report, indent=2))

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print()
    print("Results written to benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
