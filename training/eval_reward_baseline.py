"""Evaluate a public off-the-shelf reward model on OUR held-out split, with
OUR harness, so the headline number has something to stand next to.

The Phase 5 card reported 0.7026 pairwise accuracy with no comparator, which
makes it unreadable: a reviewer cannot tell whether 70% is respectable for a
base-size single-epoch Bradley-Terry reproduction or embarrassing. Two rows fix
that — the chance floor (0.500 by construction, since the pairwise task is a
balanced binary choice) and one public model run under identical conditions.

"Identical conditions" is the whole point, so this script deliberately reuses
`load_ultrafeedback_pairs` / `audit_and_filter_pairs` / `evaluate_pairs` rather
than reimplementing them. The only per-model difference is the tokenizer and
the sequence budget, both of which are taken from the model being evaluated
(see `training/reward_metadata.py` for why the budget is never restated).

Usage:
    python eval_reward_baseline.py \
        --model OpenAssistant/reward-model-deberta-v3-large-v2
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.data.preference import (
    PreferencePair,
    audit_and_filter_pairs,
    load_ultrafeedback_pairs,
)
from training.reward_metadata import resolve_max_length
from training.train_reward import PairDataset, evaluate_pairs, pair_collate

DEFAULT_BASELINE = "OpenAssistant/reward-model-deberta-v3-large-v2"

# A pairwise preference choice is a balanced binary decision (the harness
# scores `r_chosen > r_rejected`), so an uninformative model lands at exactly
# 0.5. Stated as a constant rather than left implicit: the card previously
# gestured at "the 5e-5 run collapsed to 0.51" without ever naming the floor.
CHANCE_FLOOR = 0.5


def evaluate_baseline(model_id: str, split: str, batch_size: int) -> tuple[float, int, int]:
    """Return (pairwise accuracy, N evaluated, sequence budget used)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, dtype=torch.float32
    ).to(device)
    if model.config.num_labels != 1:
        raise ValueError(
            f"{model_id} has num_labels={model.config.num_labels}; this harness scores "
            f"single-head reward models (num_labels=1) only"
        )

    max_length = resolve_max_length(model.config.to_dict())
    pairs: list[PreferencePair] = load_ultrafeedback_pairs(split=split)
    pairs, stats = audit_and_filter_pairs(pairs, tokenizer, max_length)
    print(
        f"truncation audit: dropped {stats.dropped_identical}/{stats.total} pairs "
        f"at {max_length} tokens"
    )

    loader = DataLoader(
        PairDataset(pairs, tokenizer, max_length),
        batch_size=batch_size,
        collate_fn=lambda items: pair_collate(items, tokenizer.pad_token_id),
    )
    return evaluate_pairs(model, loader, device), len(pairs), max_length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_BASELINE)
    parser.add_argument("--split", default="test_prefs")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print(f"chance floor (balanced pairwise choice): {CHANCE_FLOOR:.4f}")
    accuracy, n, max_length = evaluate_baseline(args.model, args.split, args.batch_size)
    print(
        f"baseline {args.model} on {args.split} "
        f"(N={n}, {max_length}-token budget): pairwise accuracy = {accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
