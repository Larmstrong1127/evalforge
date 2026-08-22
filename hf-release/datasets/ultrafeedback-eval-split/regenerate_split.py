"""Regenerate the exact 1,987-pair UltraFeedback evaluation split used by
`DantheMan124/deberta-preference-reward`.

This repository ships NO data rows. It ships this script plus `manifest.json`,
which records a SHA-256 over the regenerated split so you can prove your copy
is byte-identical to the one the published metrics were measured on. See
README.md for why (upstream prompt provenance is mixed and we do not have a
clean grant to redistribute the rows).

Usage:
    pip install -r requirements.txt
    python regenerate_split.py --out ultrafeedback_eval_split.jsonl --verify

Deterministic: no sampling, no shuffling, no seed. The split is
`HuggingFaceH4/ultrafeedback_binarized` split `test_prefs` in its native row
order, with two filters applied in order:

  1. Drop any row whose last chosen turn equals its last rejected turn
     (no preference signal at all). This filter removes 0 rows on test_prefs.
  2. Truncation audit: encode (prompt, chosen) and (prompt, rejected) as a
     two-segment DeBERTa sequence pair at max_length=512 and drop the pair if
     the cap was actually hit AND the two encodings came out identical.
     Training on such a pair asks the model to separate two identical inputs.
     This filter removes exactly 1 row: 1,988 loaded -> 1,987 kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

TOKENIZER_ID = "microsoft/deberta-v3-base"
DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
SPLIT = "test_prefs"
MAX_LENGTH = 512

EXPECTED_LOADED = 1988
EXPECTED_KEPT = 1987
EXPECTED_DROPPED = 1


def flatten_messages(messages: list[dict]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def load_pairs() -> list[dict]:
    """Flatten the binarized rows into (prompt, chosen, rejected) triples.

    Multi-turn prompts are flattened to a single "role: content" transcript --
    DeBERTa has no chat template, and the reward head only needs the textual
    context, not the turn structure. Single-turn rows use the dataset's own
    `prompt` column verbatim.
    """
    import datasets

    rows = datasets.load_dataset(DATASET_ID, split=SPLIT)
    pairs: list[dict] = []
    for row in rows:
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        if chosen == rejected:
            continue
        prompt_messages = row["chosen"][:-1]
        prompt = (
            flatten_messages(prompt_messages) if len(prompt_messages) > 1 else row["prompt"]
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def audit_and_filter(pairs: list[dict], tokenizer, max_length: int) -> tuple[list[dict], list[int]]:
    kept: list[dict] = []
    dropped_indices: list[int] = []
    for i, pair in enumerate(pairs):
        chosen_ids = tokenizer(
            pair["prompt"], pair["chosen"], truncation=True, max_length=max_length
        )["input_ids"]
        rejected_ids = tokenizer(
            pair["prompt"], pair["rejected"], truncation=True, max_length=max_length
        )["input_ids"]
        # Only a real problem if the cap was actually hit: if both encodings
        # come in under max_length nothing was cut off, so equal encodings are
        # impossible here (identical completions were filtered out above).
        truncated = len(chosen_ids) == max_length or len(rejected_ids) == max_length
        if truncated and chosen_ids == rejected_ids:
            dropped_indices.append(i)
            continue
        kept.append(pair)
    return kept, dropped_indices


def split_digest(pairs: list[dict]) -> str:
    """Order-sensitive SHA-256 over the split's text content.

    Field order and the NUL separators are part of the definition; do not
    change them or every previously published digest becomes unverifiable.
    """
    h = hashlib.sha256()
    for pair in pairs:
        for field in ("prompt", "chosen", "rejected"):
            h.update(pair[field].encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\x01")
    return h.hexdigest()


def build(max_length: int = MAX_LENGTH) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    pairs = load_pairs()
    kept, dropped_indices = audit_and_filter(pairs, tokenizer, max_length)
    stats = {
        "dataset": DATASET_ID,
        "split": SPLIT,
        "tokenizer": TOKENIZER_ID,
        "max_length": max_length,
        "n_pairs_loaded": len(pairs),
        "n_pairs_kept": len(kept),
        "n_dropped_identical_after_truncation": len(dropped_indices),
        "dropped_indices": dropped_indices,
        "sha256": split_digest(kept),
    }
    return kept, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ultrafeedback_eval_split.jsonl"))
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("manifest.json"))
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare counts and digest against manifest.json and exit non-zero on mismatch.",
    )
    parser.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    kept, stats = build(args.max_length)
    print(json.dumps({k: v for k, v in stats.items() if k != "dropped_indices"}, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for pair in kept:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} pairs -> {args.out}")

    if args.write_manifest:
        args.manifest.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"wrote manifest -> {args.manifest}")
        return 0

    if args.verify:
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))
        problems = [
            f"{key}: expected {expected[key]!r}, got {stats[key]!r}"
            for key in (
                "n_pairs_loaded",
                "n_pairs_kept",
                "n_dropped_identical_after_truncation",
                "sha256",
            )
            if expected[key] != stats[key]
        ]
        if problems:
            print("VERIFY FAILED:")
            for problem in problems:
                print("  " + problem)
            return 1
        print(f"VERIFY OK: {stats['n_pairs_kept']} pairs, sha256 {stats['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
