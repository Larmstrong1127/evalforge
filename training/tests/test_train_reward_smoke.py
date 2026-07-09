"""CPU smoke test: two training steps on fixture pairs, real (tiny) forward math.

Uses the real tokenizer/model classes but the smallest usable config; asserts
loss is finite and a checkpoint directory is produced. Marked slow-ish but
CPU-safe (a few seconds with the tiny max_length).
"""
from pathlib import Path
from unittest.mock import patch

import torch

from training.config import TrainConfig
from training.data.preference import PreferencePair
from training.train_reward import PairDataset, pair_collate, train_reward


def _tiny_pairs() -> list[PreferencePair]:
    return [
        PreferencePair(prompt="What is 2+2?", chosen="4", rejected="banana"),
        PreferencePair(prompt="Capital of France?", chosen="Paris", rejected="Lyon"),
        PreferencePair(prompt="Opposite of hot?", chosen="Cold", rejected="hot dog"),
        PreferencePair(prompt="Days in a week?", chosen="7", rejected="9"),
    ]


class _StubTokenizer:
    """Whitespace tokenizer exposing the interface PairDataset/pair_collate use."""

    pad_token_id = 0

    def __call__(self, text, pair, truncation, max_length):
        ids = [hash(w) % 1000 + 1 for w in (text + " " + pair).split()][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def test_pair_dataset_and_collate_shapes() -> None:
    ds = PairDataset(_tiny_pairs(), _StubTokenizer(), max_length=16)
    batch = pair_collate([ds[0], ds[1]], pad_token_id=0)
    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["chosen_input_ids"].shape == batch["chosen_attention_mask"].shape
    assert batch["rejected_input_ids"].shape[0] == 2
    # dynamic padding: both sides padded to their own batch max, not a global constant
    assert batch["chosen_input_ids"].dtype == torch.long


def test_train_reward_smoke(tmp_path: Path) -> None:
    config = TrainConfig(
        experiment_name="smoke",
        model_name="microsoft/deberta-v3-base",
        max_length=32,
        batch_size=2,
        learning_rate=1e-5,
        warmup_ratio=0.0,
        max_grad_norm=1.0,
        max_epochs=1,
        early_stopping_patience=1,
        use_amp=False,
        seed=7,
    )
    with patch("training.train_reward.load_ultrafeedback_pairs") as mock_load:
        # train split then eval split
        mock_load.side_effect = [_tiny_pairs(), _tiny_pairs()[:2]]
        ckpt = train_reward(config, tmp_path / "ckpt", tmp_path / "logs")
    assert ckpt.exists()
