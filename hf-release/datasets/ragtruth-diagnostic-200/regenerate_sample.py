"""Regenerate the exact 200-example RAGTruth diagnostic sample used to measure
the encoding defect in `DantheMan124/deberta-hallucination-judge`.

Unlike the UltraFeedback split in this collection, the rows ARE shipped here
(`data/diagnostic-200.jsonl`) -- RAGTruth carries an explicit MIT LICENSE from
Particle Media. This script exists so the sample can be rebuilt from source and
verified against `manifest.json`, and so the per-encoding truncation columns are
reproducible rather than asserted.

Usage:
    pip install -r requirements.txt
    python regenerate_sample.py --verify                  # rebuild + check digest
    python regenerate_sample.py --out data/diagnostic-200.jsonl

Determinism: `random.Random(42).sample(examples, 200)` over the upstream rows in
their native order. That is the identical call made by
`training/run_benchmark.py` and `training/scripts/diagnose_ragtruth_agreement.py`
in the EvalForge repository, so this sample is the same 200 examples the
published diagnostic numbers were measured on. The sample is a function of the
SOURCE ROW ORDER AND COUNT: `manifest.json` records `n_source_examples` so an
upstream reorder or reupload is caught by --verify instead of silently producing
a different 200.

Truncation columns are computed with the `microsoft/deberta-v3-base` tokenizer
at max_length=512 under both encodings:

  legacy            -- one flat "Q: .. C: .. A: .." string right-truncated. The
                       answer sits at the tail, so truncation eats the very span
                       being classified.
  answer-preserving -- budgets the context and keeps question and answer whole.

The finding this sample is keyed to: fixing the encoding moved ROC-AUC from
0.444 (below chance) to 0.603, while headline accuracy at threshold 0.5 FELL
from 0.475 to 0.385. See README.md -- both numbers are real and the pair has to
travel together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DATASET_ID = "wandb/RAGTruth-processed"
SPLIT = "test"
TOKENIZER_ID = "microsoft/deberta-v3-base"
SAMPLE_SIZE = 200
SEED = 42
MAX_LENGTH = 512

QUESTION_PREFIX = "Q: "
CONTEXT_PREFIX = " C: "
ANSWER_PREFIX = " A: "


def legacy_encode_text(question: str, context: str, answer: str) -> str:
    return f"{QUESTION_PREFIX}{question}{CONTEXT_PREFIX}{context}{ANSWER_PREFIX}{answer}"


def load_examples() -> list[dict]:
    """Load and label the upstream rows.

    `hallucination_labels` is a JSON-encoded STRING (e.g. "[]"), not a list --
    `bool("[]")` is True in Python, so testing the raw field for truthiness
    would label every single row as hallucinated. It must be json.loads()'d
    first. This is the original bug the EvalForge loader docstring warns about.
    """
    import datasets

    rows = datasets.load_dataset(DATASET_ID, split=SPLIT)
    return [
        {
            "question": row["query"],
            "context": row["context"],
            "answer": row["output"],
            "label": 1 if json.loads(row["hallucination_labels"]) else 0,
        }
        for row in rows
    ]


def budget_segments(
    question_ids: list[int], context_ids: list[int], answer_ids: list[int], budget: int
) -> tuple[list[int], list[int], list[int]]:
    """Fit three token-id segments into `budget`, sacrificing context first and
    the answer last. Mirrors training/hallucination_encoding.py exactly."""
    if budget <= 0:
        return [], [], []
    fixed = len(question_ids) + len(answer_ids)
    if fixed + len(context_ids) <= budget:
        return question_ids, context_ids, answer_ids
    if fixed <= budget:
        return question_ids, context_ids[: budget - fixed], answer_ids
    if len(answer_ids) < budget:
        return question_ids[: budget - len(answer_ids)], [], answer_ids
    return [], [], answer_ids[:budget]


def encoding_stats(tokenizer, ex: dict, max_length: int) -> dict:
    """Per-example truncation accounting under both encodings."""
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length - n_special

    flat_ids = tokenizer(
        legacy_encode_text(ex["question"], ex["context"], ex["answer"]),
        add_special_tokens=False,
    )["input_ids"]
    answer_text = ANSWER_PREFIX + ex["answer"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    question_text = QUESTION_PREFIX + ex["question"] + CONTEXT_PREFIX
    question_ids = tokenizer(question_text, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(ex["context"], add_special_tokens=False)["input_ids"]

    legacy_overflow = max(0, len(flat_ids) - budget)
    legacy_answer_lost = min(legacy_overflow, len(answer_ids))

    kept_q, kept_c, kept_a = budget_segments(question_ids, context_ids, answer_ids, budget)
    preserving_answer_lost = len(answer_ids) - len(kept_a)

    return {
        "n_tokens_flat": len(flat_ids) + n_special,
        "n_tokens_answer": len(answer_ids),
        "truncated": len(flat_ids) > budget,
        "legacy": {
            "n_tokens_kept": min(len(flat_ids), budget) + n_special,
            "answer_tokens_lost": legacy_answer_lost,
            "answer_fully_dropped": legacy_answer_lost == len(answer_ids),
            "answer_partially_dropped": 0 < legacy_answer_lost < len(answer_ids),
        },
        "answer_preserving": {
            "n_tokens_kept": len(kept_q) + len(kept_c) + len(kept_a) + n_special,
            "context_tokens_lost": len(context_ids) - len(kept_c),
            "answer_tokens_lost": preserving_answer_lost,
            "answer_fully_dropped": preserving_answer_lost == len(answer_ids),
            "answer_partially_dropped": 0 < preserving_answer_lost < len(answer_ids),
        },
    }


def row_digest(rows: list[dict]) -> str:
    """Order-sensitive SHA-256 over the sample's text content and labels.

    Deliberately covers only the source fields, not the derived encoding
    columns: the sample identity must not change when a tokenizer version
    shifts a token count by one.
    """
    h = hashlib.sha256()
    for row in rows:
        for field in ("question", "context", "answer"):
            h.update(row[field].encode("utf-8"))
            h.update(b"\x00")
        h.update(str(row["label"]).encode("ascii"))
        h.update(b"\x01")
    return h.hexdigest()


def build(max_length: int = MAX_LENGTH) -> tuple[list[dict], dict]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    examples = load_examples()
    sample = random.Random(SEED).sample(examples, SAMPLE_SIZE)

    rows = []
    for i, ex in enumerate(sample):
        rows.append(
            {"sample_index": i, **ex, "encoding": encoding_stats(tokenizer, ex, max_length)}
        )

    legacy = [r["encoding"]["legacy"] for r in rows]
    preserving = [r["encoding"]["answer_preserving"] for r in rows]
    stats = {
        "dataset": DATASET_ID,
        "split": SPLIT,
        "tokenizer": TOKENIZER_ID,
        "max_length": max_length,
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "sampler": "random.Random(SEED).sample(examples, SAMPLE_SIZE)",
        "n_source_examples": len(examples),
        "n_rows": len(rows),
        "label_balance": {
            "hallucinated_1": sum(r["label"] for r in rows),
            "faithful_0": sum(1 - r["label"] for r in rows),
        },
        "n_truncated": sum(1 for r in rows if r["encoding"]["truncated"]),
        "legacy_answer_fully_dropped": sum(1 for e in legacy if e["answer_fully_dropped"]),
        "legacy_answer_partially_dropped": sum(
            1 for e in legacy if e["answer_partially_dropped"]
        ),
        "answer_preserving_answer_fully_dropped": sum(
            1 for e in preserving if e["answer_fully_dropped"]
        ),
        "answer_preserving_answer_partially_dropped": sum(
            1 for e in preserving if e["answer_partially_dropped"]
        ),
        "sha256": row_digest(rows),
    }
    return rows, stats


def main() -> int:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=here / "data" / "diagnostic-200.jsonl")
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    rows, stats = build(args.max_length)
    print(json.dumps(stats, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")

    if args.write_manifest:
        args.manifest.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        print(f"wrote manifest -> {args.manifest}")
        return 0

    if args.verify:
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))
        problems = [
            f"{key}: expected {expected[key]!r}, got {stats[key]!r}"
            for key in ("n_source_examples", "n_rows", "label_balance", "sha256")
            if expected[key] != stats[key]
        ]
        if problems:
            print("VERIFY FAILED:")
            for problem in problems:
                print("  " + problem)
            return 1
        print(f"VERIFY OK: {stats['n_rows']} rows, sha256 {stats['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
