"""One-off script: reruns a single named judge and merges its result into
benchmark_results.json IMMEDIATELY after that judge finishes — not batched
with any other judge. This is a deliberate fix after a prior version of this
script lost an already-completed, already-paid-for gpt-4o result because it
only wrote output after a SECOND, unrelated judge (Gemini) also finished, and
the process was killed while Gemini was stuck retrying against an exhausted
quota. Never again: each provider's spend is persisted the moment it's done.

Usage: python rerun_one_judge.py <provider_name> <model_name> <report_key>
Example: python rerun_one_judge.py openai gpt-4o gpt-4o
"""
import asyncio
import json
import random
import sys

from evalforge.config import Settings

from run_benchmark import SAMPLE_SIZE, SEED
from training.benchmark import aggregate_benchmark_results, score_with_llm_judge
from training.data.ragtruth import load_ragtruth_examples


async def main() -> None:
    if len(sys.argv) != 4:
        print("usage: python rerun_one_judge.py <provider_name> <model_name> <report_key>")
        sys.exit(1)
    provider_name, model_name, report_key = sys.argv[1], sys.argv[2], sys.argv[3]

    settings = Settings(_env_file="../platform/api/.env")  # type: ignore[call-arg]
    all_examples = load_ragtruth_examples()
    sample = random.Random(SEED).sample(all_examples, SAMPLE_SIZE)
    ground_truth = [ex.label for ex in sample]

    with open("benchmark_results.json", encoding="utf-8") as f:
        report = json.load(f)

    print(f"Scoring with {report_key} ({provider_name}:{model_name})...")
    result = await score_with_llm_judge(sample, provider_name, model_name, settings)
    print(f"  {report_key}: {len(result.predictions)} scored")

    if len(result.predictions) != len(ground_truth):
        print(
            f"  incomplete ({len(result.predictions)}/{len(ground_truth)}) — NOT merged, "
            f"NOT overwriting benchmark_results.json"
        )
        sys.exit(1)

    report.update(aggregate_benchmark_results(ground_truth, {report_key: result}))
    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  merged and saved immediately: {json.dumps(report[report_key], indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
