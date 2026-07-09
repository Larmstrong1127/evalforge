"""Evaluate a trained reward checkpoint: ID pairwise accuracy (UltraFeedback
test_prefs) and, if provided, the rating-room OOD probe (tiny N — report it
as a probe, never as a benchmark; the printed output includes N explicitly).

Usage:
    python eval_reward.py --checkpoint checkpoints/reward-lr2e5 \
        --probe data/rating_pairs.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.data.preference import (
    PreferencePair,
    audit_and_filter_pairs,
    load_ultrafeedback_pairs,
)
from training.train_reward import PairDataset, evaluate_pairs, pair_collate


def load_jsonl_pairs(path: Path) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pairs.append(
            PreferencePair(prompt=row["prompt"], chosen=row["chosen"], rejected=row["rejected"])
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32
    ).to(device)

    def acc_for(pairs: list[PreferencePair]) -> float:
        loader = DataLoader(
            PairDataset(pairs, tokenizer, args.max_length),
            batch_size=args.batch_size,
            collate_fn=lambda items: pair_collate(items, tokenizer.pad_token_id),
        )
        return evaluate_pairs(model, loader, device)

    id_pairs = load_ultrafeedback_pairs(split="test_prefs")
    id_pairs, _ = audit_and_filter_pairs(id_pairs, tokenizer, args.max_length)
    print(
        f"ID (UltraFeedback test_prefs, N={len(id_pairs)}): "
        f"pairwise accuracy = {acc_for(id_pairs):.4f}"
    )

    if args.probe is not None:
        probe_pairs = load_jsonl_pairs(args.probe)
        if not probe_pairs:
            print("OOD probe: no pairs found — skipping")
        else:
            print(
                f"OOD probe (rating room, N={len(probe_pairs)} — a probe, not a benchmark): "
                f"pairwise accuracy = {acc_for(probe_pairs):.4f}"
            )


if __name__ == "__main__":
    main()
