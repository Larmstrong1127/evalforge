"""Fit a calibration temperature for a trained reward checkpoint and write it
into the checkpoint's config.json (key: "reward_temperature"), so the platform
judge can read it alongside the weights. The sequence budget it was fit under
is written alongside it ("reward_train_max_length") for the same reason.

The budget defaults to the checkpoint's own recorded value rather than a
literal — see training/reward_metadata.py for why.

Usage: python calibrate_reward.py --checkpoint checkpoints/reward-lr2e5
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.calibrate import fit_temperature
from training.data.preference import audit_and_filter_pairs, load_ultrafeedback_pairs
from training.reward_metadata import TRAIN_MAX_LENGTH_KEY, load_checkpoint_max_length
from training.train_reward import PairDataset, _rewards, pair_collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    # Default derived from the checkpoint, never hardcoded: a temperature fit
    # at one sequence budget does not transfer to another.
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    max_length = args.max_length or load_checkpoint_max_length(args.checkpoint)
    print(f"sequence budget: {max_length} tokens (from checkpoint config)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    pairs = load_ultrafeedback_pairs(split="test_prefs")
    pairs, stats = audit_and_filter_pairs(pairs, tokenizer, max_length)
    print(f"calibration set: {len(pairs)} pairs (dropped {stats.dropped_identical})")

    loader = DataLoader(
        PairDataset(pairs, tokenizer, max_length),
        batch_size=args.batch_size,
        collate_fn=lambda items: pair_collate(items, tokenizer.pad_token_id),
    )
    margins: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            r_c = _rewards(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            r_r = _rewards(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            margins.append((r_c - r_r).cpu())

    temperature = fit_temperature(torch.cat(margins))
    config_path = args.checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["reward_temperature"] = temperature
    # Record the regime next to the constant fit under it, so downstream
    # consumers (the platform judge, eval_reward.py) derive the sequence
    # budget instead of restating it.
    config[TRAIN_MAX_LENGTH_KEY] = max_length
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"fitted temperature {temperature:.4f} at {max_length} tokens -> {config_path}")


if __name__ == "__main__":
    main()
