"""Generate the benchmark results table (markdown) from benchmark_results.json.

The local-judge-vs-cloud-judge numbers quoted in the READMEs are produced by
this script rather than hand-typed, so the tables stay in sync with the raw
results file. Run it after re-running the benchmark to refresh the tables.

Usage:
    python scripts/gen_results_table.py                 # print to stdout
    python scripts/gen_results_table.py --out table.md   # also write a file

The source of truth is training/benchmark_results.json (produced by
run_benchmark.py). This script does no I/O against models or the network.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Human-facing labels for the raw keys in benchmark_results.json, in the
# display order used in the READMEs (local judge first, then cloud judges by
# descending agreement).
DISPLAY = [
    ("local-deberta-lr2e5", "**local-deberta (ours)**"),
    ("claude-sonnet-5", "claude-sonnet-5"),
    ("gpt-4o", "gpt-4o"),
    ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "benchmark_results.json"


def _bold_best(value: str, is_best: bool) -> str:
    return f"**{value}**" if is_best else value


def render_table(results: dict[str, dict[str, float]]) -> str:
    keys = [k for k, _ in DISPLAY if k in results]

    best_agreement = max(results[k]["agreement"] for k in keys)
    min_cost = min(results[k]["cost_per_1k_usd"] for k in keys)
    min_latency = min(results[k]["p50_latency_ms"] for k in keys)

    lines = [
        "| Judge | Agreement | Cost / 1K evals | p50 latency | p95 latency |",
        "|---|---|---|---|---|",
    ]
    for key, label in DISPLAY:
        if key not in results:
            continue
        r = results[key]
        agreement = _bold_best(f"{r['agreement'] * 100:.1f}%", r["agreement"] == best_agreement)
        cost = _bold_best(f"${r['cost_per_1k_usd']:.2f}", r["cost_per_1k_usd"] == min_cost)
        p50 = _bold_best(f"{r['p50_latency_ms']:.0f} ms", r["p50_latency_ms"] == min_latency)
        p95 = f"{r['p95_latency_ms']:.0f} ms"
        lines.append(f"| {label} | {agreement} | {cost} | {p50} | {p95} |")

    return "\n".join(lines) + "\n"


def render_summary(results: dict[str, dict[str, float]]) -> str:
    local = results["local-deberta-lr2e5"]
    cloud_p50 = [
        results[k]["p50_latency_ms"] for k, _ in DISPLAY[1:] if k in results
    ]
    speedup = min(cloud_p50) / local["p50_latency_ms"]
    return (
        f"Local judge: free (${local['cost_per_1k_usd']:.2f}/1K), "
        f"{speedup:.0f}x faster than the fastest cloud judge, "
        f"{local['agreement'] * 100:.1f}% agreement on out-of-distribution data.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--out", type=Path, default=None, help="also write the table here")
    parser.add_argument("--summary", action="store_true", help="print the one-line summary too")
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    table = render_table(results)
    print(table, end="")
    if args.summary:
        print("\n" + render_summary(results), end="")
    if args.out is not None:
        args.out.write_text(table, encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
