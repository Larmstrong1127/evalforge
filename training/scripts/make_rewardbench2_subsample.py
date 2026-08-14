"""Build a domain-stratified subsample of RewardBench 2 as a local JSONL.

Why this exists: the full benchmark is 8,977 completions. On CPU that is ~2.5 h
for a 184M encoder and ~8 h for the 435M OpenAssistant baseline. Eight hours of
single-threaded patience is not a good trade for one comparison row, so the
baseline is measured on a fixed, seeded, domain-stratified subsample instead —
and *our* model is re-scored on the identical subsample from its own full-set
per-row scores, so the head-to-head is exactly like-for-like.

The full-set number remains the headline; the subsample number is only ever
quoted as a paired comparison, never as a leaderboard-comparable score.

The harness reads a local JSONL directly (`--dataset path.jsonl`), so no
harness code is involved in the subsampling.

Usage:
    python training/scripts/make_rewardbench2_subsample.py \
        --prompts 450 --seed 20260813 --out subsample.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DATASET = "allenai/reward-bench-2"


def stratified_indices(subsets: list[str], n: int, seed: int) -> list[int]:
    """Pick `n` row indices, allocating each domain a share of `n` proportional
    to its size in the full set (largest-remainder, so the counts sum exactly).
    Selection within a domain is uniform under `seed`."""
    by_domain: dict[str, list[int]] = collections.defaultdict(list)
    for i, s in enumerate(subsets):
        by_domain[s].append(i)
    total = len(subsets)

    exact = {d: len(idx) * n / total for d, idx in by_domain.items()}
    quota = {d: int(v) for d, v in exact.items()}
    remainder = n - sum(quota.values())
    for d, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if remainder <= 0:
            break
        quota[d] += 1
        remainder -= 1

    rng = random.Random(seed)
    chosen: list[int] = []
    for domain in sorted(by_domain):
        pool = sorted(by_domain[domain])
        chosen.extend(rng.sample(pool, min(quota[domain], len(pool))))
    return sorted(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=450)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from datasets import load_dataset

    data = load_dataset(DATASET, split="test")
    picked = stratified_indices(list(data["subset"]), args.prompts, args.seed)
    subset = data.select(picked)

    with args.out.open("w", encoding="utf-8") as handle:
        for row in subset:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = collections.Counter(subset["subset"])
    completions = sum(subset["total_completions"])
    print(f"wrote {len(subset)} prompts / {completions} completions -> {args.out}")
    print(f"seed={args.seed}  ids={len(set(subset['id']))} unique")
    for domain, count in sorted(counts.items()):
        print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
