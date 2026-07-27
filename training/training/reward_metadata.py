"""Single source of truth for a reward checkpoint's training sequence budget.

A reward model must be *evaluated and served* at the sequence length it was
*trained* at. Phase 5 shipped with the budget written down in three
independent places — the training YAML (512), the eval/calibration script
defaults (1024), and the platform judge constant (1024) — which is three
chances to drift and no mechanism to notice. The fix is to stop restating
the number and start deriving it from the artifact that travels with the
weights: the checkpoint's own `config.json`.

Resolution order:

1. `reward_train_max_length` — written by `calibrate_reward.py` so the
   temperature and the regime it was fit under ship together explicitly.
2. `max_position_embeddings` — the base model's own budget (512 for
   `deberta-v3-base`), present in every HF config, so this works for
   checkpoints exported before key 1 existed.

There is deliberately no numeric default: a checkpoint carrying neither key
is not a checkpoint we know how to evaluate honestly, and guessing is what
caused the original mismatch.
"""
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRAIN_MAX_LENGTH_KEY = "reward_train_max_length"
POSITION_LIMIT_KEY = "max_position_embeddings"


def resolve_max_length(config: Mapping[str, Any]) -> int:
    """Derive the training sequence budget from a checkpoint config mapping."""
    for key in (TRAIN_MAX_LENGTH_KEY, POSITION_LIMIT_KEY):
        value = config.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError(
        f"checkpoint config carries neither {TRAIN_MAX_LENGTH_KEY!r} nor "
        f"{POSITION_LIMIT_KEY!r}; cannot determine the training sequence "
        f"budget — pass --max-length explicitly if you know it"
    )


def load_checkpoint_max_length(checkpoint: Path) -> int:
    """Read the training sequence budget from `<checkpoint>/config.json`."""
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint config not found: {config_path}")
    config: Mapping[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    return resolve_max_length(config)
