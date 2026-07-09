# Phase 5: Preference Reward Model Implementation Plan

**Goal:** Train a DeBERTa-v3-base reward model on UltraFeedback preference pairs with a hand-written Bradley-Terry loop, calibrate and honestly evaluate it (ID split + rating-room OOD probe), and register it as the platform's `reward` judge.

**Architecture:** New modules in the existing `training/` package (pure loss module, data loader with truncation audit, model builders, dual-forward training loop, post-hoc temperature calibration, DB pair exporter, eval script) plus one new judge plugin in `platform/api` behind an optional `reward` extra. Spec: `docs/design/specs/2026-07-08-phase5-reward-model-design.md`.

**Tech Stack:** PyTorch (AMP, dynamic padding, TF32), transformers (`AutoModelForSequenceClassification`, num_labels=1), datasets (`HuggingFaceH4/ultrafeedback_binarized`), SQLAlchemy (export), pytest.

**Working conventions (apply to every task):** run commands from `C:\Users\cobra\ClaudeProjects\evalforge\training` (or `platform/api` for Task 8) with that package's venv active; conventional commits; NO AI attribution in commit messages, ever; after each task run `ruff check .` and fix anything it flags before committing.

---

### Task 1: Bradley-Terry loss + pairwise accuracy (pure module)

**Files:**
- Create: `training/training/reward_loss.py`
- Test: `training/tests/test_reward_loss.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the Bradley-Terry pairwise loss and pairwise accuracy."""
import math

import torch

from training.reward_loss import bradley_terry_loss, pairwise_accuracy


def test_loss_zero_margin_is_log2() -> None:
    # r_chosen == r_rejected -> sigma(0) = 0.5 -> loss = -log(0.5) = log(2)
    chosen = torch.tensor([1.0, -2.0])
    rejected = torch.tensor([1.0, -2.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert math.isclose(loss.item(), math.log(2.0), rel_tol=1e-6)


def test_loss_hand_computed_value() -> None:
    # margin = 2.0 -> loss = -log(sigma(2.0)) = softplus(-2.0) = 0.126928...
    chosen = torch.tensor([3.0])
    rejected = torch.tensor([1.0])
    loss = bradley_terry_loss(chosen, rejected)
    assert math.isclose(loss.item(), math.log(1 + math.exp(-2.0)), rel_tol=1e-6)


def test_loss_decreases_with_larger_margin() -> None:
    rejected = torch.tensor([0.0])
    small = bradley_terry_loss(torch.tensor([1.0]), rejected)
    large = bradley_terry_loss(torch.tensor([5.0]), rejected)
    assert large.item() < small.item()


def test_flip_symmetry() -> None:
    # Swapping chosen/rejected must mirror the margin around zero:
    # loss(flipped) = softplus(+m) where loss(orig) = softplus(-m).
    # Guards against ordering bias in the implementation.
    chosen = torch.tensor([2.0, 0.5, -1.0])
    rejected = torch.tensor([-1.0, 0.5, 3.0])
    margins = chosen - rejected
    orig = bradley_terry_loss(chosen, rejected)
    flipped = bradley_terry_loss(rejected, chosen)
    expected_orig = torch.nn.functional.softplus(-margins).mean()
    expected_flipped = torch.nn.functional.softplus(margins).mean()
    assert torch.isclose(orig, expected_orig, rtol=1e-6)
    assert torch.isclose(flipped, expected_flipped, rtol=1e-6)


def test_loss_requires_grad_flows() -> None:
    chosen = torch.tensor([1.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    bradley_terry_loss(chosen, rejected).backward()
    assert chosen.grad is not None and rejected.grad is not None


def test_pairwise_accuracy() -> None:
    chosen = torch.tensor([2.0, 0.0, 1.0, 5.0])
    rejected = torch.tensor([1.0, 1.0, 1.0, -5.0])
    # correct: idx 0 (2>1), idx 3 (5>-5). idx 1 wrong (0<1), idx 2 tie counts wrong.
    assert pairwise_accuracy(chosen, rejected) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward_loss.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.reward_loss'`

- [ ] **Step 3: Write the implementation**

```python
"""Bradley-Terry pairwise reward loss and metrics.

Pure tensor functions, no model or GPU dependency, so the exact math is
unit-testable with hand-computed values.

L = -log sigma(r_chosen - r_rejected), implemented via softplus(-(margin))
for numerical stability (equivalent identity: -log(sigmoid(x)) == softplus(-x),
avoids log(0) when the margin is very negative).
"""
import torch


def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Mean Bradley-Terry loss over a batch of reward pairs (1-D tensors)."""
    margins = chosen_rewards - rejected_rewards
    return torch.nn.functional.softplus(-margins).mean()


def pairwise_accuracy(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> float:
    """Fraction of pairs where the chosen completion outscores the rejected.

    Ties count as wrong: a reward model that cannot separate a pair has not
    learned the preference.
    """
    correct = (chosen_rewards > rejected_rewards).sum().item()
    return correct / len(chosen_rewards)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward_loss.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add training/reward_loss.py tests/test_reward_loss.py
git commit -m "feat: add Bradley-Terry pairwise loss with flip-symmetry guarantee"
```

---

### Task 2: Preference data loading with truncation audit

**Files:**
- Create: `training/training/data/preference.py`
- Test: `training/tests/test_preference_data.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for UltraFeedback preference-pair loading and the truncation audit."""
from training.data.preference import (
    PreferencePair,
    audit_and_filter_pairs,
    flatten_messages,
    load_ultrafeedback_pairs,
)


def _fake_row(prompt: str, chosen: str, rejected: str) -> dict:
    # ultrafeedback_binarized stores chosen/rejected as message lists whose
    # last entry is the assistant completion.
    return {
        "prompt": prompt,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }


def test_load_pairs_extracts_last_assistant_message() -> None:
    rows = [_fake_row("What is 2+2?", "4", "5")]
    pairs = load_ultrafeedback_pairs(load_fn=lambda split: rows)
    assert pairs == [PreferencePair(prompt="What is 2+2?", chosen="4", rejected="5")]


def test_load_pairs_skips_identical_completions() -> None:
    rows = [_fake_row("q", "same", "same"), _fake_row("q2", "a", "b")]
    pairs = load_ultrafeedback_pairs(load_fn=lambda split: rows)
    assert len(pairs) == 1
    assert pairs[0].prompt == "q2"


def test_flatten_messages_multi_turn() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and 2+2?"},
    ]
    assert flatten_messages(messages) == "user: hi\nassistant: hello\nuser: and 2+2?"


class _FakeTokenizer:
    """Encodes 1 token per whitespace-separated word; truncation keeps the
    first max_length words. Mirrors the only tokenizer behavior the audit
    relies on."""

    def __call__(self, text: str, pair: str, truncation: bool, max_length: int) -> dict:
        ids = list(range(len((text + " " + pair).split())))[:max_length]
        return {"input_ids": ids}


def test_audit_drops_pairs_identical_after_truncation() -> None:
    # Long shared prompt + differing tails: with a tiny budget both sides
    # truncate to the same prefix -> dropped. With a big budget -> kept.
    long_prompt = " ".join(["word"] * 10)
    pair = PreferencePair(prompt=long_prompt, chosen="ending one", rejected="different two")
    kept_small, stats_small = audit_and_filter_pairs([pair], _FakeTokenizer(), max_length=5)
    assert kept_small == []
    assert stats_small.dropped_identical == 1
    kept_large, stats_large = audit_and_filter_pairs([pair], _FakeTokenizer(), max_length=50)
    assert kept_large == [pair]
    assert stats_large.dropped_identical == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preference_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.data.preference'`

- [ ] **Step 3: Write the implementation**

```python
"""UltraFeedback-binarized preference pairs: loading, flattening, truncation audit.

Dataset identifier note: verify ULTRAFEEDBACK_DATASET_ID against the Hub
before the first real run (same caveat as prepare.py's HaluEval id).

Multi-turn prompts are flattened to a single "role: content" transcript
string — DeBERTa has no chat template, and the reward head only needs the
textual context, not the turn structure.

Truncation audit: right-truncating completions under a token budget can
leave a pair's chosen and rejected encodings IDENTICAL (long shared prompt,
differing tails cut off). Training on such pairs asks the model to separate
two identical inputs — pure gradient noise. `audit_and_filter_pairs` drops
them and reports how many, so the run log records the data loss honestly.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ULTRAFEEDBACK_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class AuditStats:
    total: int
    dropped_identical: int


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _default_load_fn(split: str) -> Any:
    import datasets

    return datasets.load_dataset(ULTRAFEEDBACK_DATASET_ID, split=split)


def load_ultrafeedback_pairs(
    split: str = "train_prefs", load_fn: Callable[[str], Any] = _default_load_fn
) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for row in load_fn(split):
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        if chosen == rejected:
            continue  # no preference signal at all
        prompt_messages = row["chosen"][:-1]
        prompt = flatten_messages(prompt_messages) if prompt_messages else row["prompt"]
        pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))
    return pairs


def audit_and_filter_pairs(
    pairs: list[PreferencePair], tokenizer: Any, max_length: int
) -> tuple[list[PreferencePair], AuditStats]:
    kept: list[PreferencePair] = []
    dropped = 0
    for pair in pairs:
        chosen_ids = tokenizer(
            pair.prompt, pair.chosen, truncation=True, max_length=max_length
        )["input_ids"]
        rejected_ids = tokenizer(
            pair.prompt, pair.rejected, truncation=True, max_length=max_length
        )["input_ids"]
        if chosen_ids == rejected_ids:
            dropped += 1
            continue
        kept.append(pair)
    return kept, AuditStats(total=len(pairs), dropped_identical=dropped)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preference_data.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add training/data/preference.py tests/test_preference_data.py
git commit -m "feat: add UltraFeedback preference loading with truncation audit"
```

---

### Task 3: Reward model builders

**Files:**
- Create: `training/training/models/reward.py`
- Test: `training/tests/test_reward_model.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the reward-model factory functions (mocked, no downloads)."""
from unittest.mock import patch

from training.models.reward import build_reward_model, build_reward_tokenizer


@patch("training.models.reward.AutoModelForSequenceClassification")
def test_build_reward_model_single_scalar_head(mock_cls) -> None:
    build_reward_model("microsoft/deberta-v3-base")
    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["num_labels"] == 1
    import torch

    assert kwargs["torch_dtype"] is torch.float32


@patch("training.models.reward.AutoTokenizer")
def test_build_reward_tokenizer(mock_tok) -> None:
    build_reward_tokenizer("microsoft/deberta-v3-base")
    mock_tok.from_pretrained.assert_called_once_with("microsoft/deberta-v3-base")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
"""Factory functions for the reward model: DeBERTa-v3 + 1-dim scalar head.

num_labels=1 makes AutoModelForSequenceClassification a regression head —
outputs.logits has shape (batch, 1) and is used directly as the scalar
reward. torch_dtype=torch.float32 for the same GradScaler reason documented
in models/classifier.py.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def build_reward_model(model_name: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.float32,
    )


def build_reward_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward_model.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add training/models/reward.py tests/test_reward_model.py
git commit -m "feat: add reward model factory (DeBERTa scalar head)"
```

---

### Task 4: Reward training loop + configs

**Files:**
- Create: `training/training/train_reward.py` (inside the package, sibling of the existing `training/training/train.py`)
- Create: `training/configs/reward-lr2e5.yaml`, `training/configs/reward-lr5e5.yaml`
- Test: `training/tests/test_train_reward_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_train_reward_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'training.train_reward'`

- [ ] **Step 3: Write the implementation**

```python
"""Bradley-Terry reward-model training loop.

Mirrors training/train.py's structure (TF32, AMP, dynamic padding, linear
warmup schedule, best-checkpoint-by-eval-metric, real-checkpoint-path
discipline) but with a dual forward pass per batch: the chosen and rejected
encodings each go through the SAME model, and the loss is
bradley_terry_loss(r_chosen, r_rejected). Selection metric is pairwise
accuracy on the held-out test_prefs split.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from training.config import TrainConfig, load_config
from training.data.preference import (
    PreferencePair,
    audit_and_filter_pairs,
    load_ultrafeedback_pairs,
)
from training.models.reward import build_reward_model, build_reward_tokenizer
from training.reward_loss import bradley_terry_loss, pairwise_accuracy


class PairDataset(Dataset):
    def __init__(self, pairs: list[PreferencePair], tokenizer, max_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        chosen = self.tokenizer(
            pair.prompt, pair.chosen, truncation=True, max_length=self.max_length
        )
        rejected = self.tokenizer(
            pair.prompt, pair.rejected, truncation=True, max_length=self.max_length
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


def _pad(seqs: list[list[int]], pad_value: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_value] * (width - len(s)) for s in seqs], dtype=torch.long)


def pair_collate(items: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Dynamic padding, chosen and rejected padded independently to their own
    batch max — same VRAM lesson as Phase 2 (fixed-width padding caused OOM)."""
    return {
        "chosen_input_ids": _pad([i["chosen_input_ids"] for i in items], pad_token_id),
        "chosen_attention_mask": _pad([i["chosen_attention_mask"] for i in items], 0),
        "rejected_input_ids": _pad([i["rejected_input_ids"] for i in items], pad_token_id),
        "rejected_attention_mask": _pad([i["rejected_attention_mask"] for i in items], 0),
    }


def _rewards(model, input_ids, attention_mask) -> torch.Tensor:
    return model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)


def evaluate_pairs(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct_chosen: list[torch.Tensor] = []
    correct_rejected: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            correct_chosen.append(
                _rewards(model, batch["chosen_input_ids"], batch["chosen_attention_mask"]).cpu()
            )
            correct_rejected.append(
                _rewards(model, batch["rejected_input_ids"], batch["rejected_attention_mask"]).cpu()
            )
    return pairwise_accuracy(torch.cat(correct_chosen), torch.cat(correct_rejected))


def train_reward(config: TrainConfig, checkpoint_dir: Path, log_dir: Path) -> Path:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    tokenizer = build_reward_tokenizer(config.model_name)
    model = build_reward_model(config.model_name).to(device)

    train_pairs = load_ultrafeedback_pairs(split="train_prefs")
    eval_pairs = load_ultrafeedback_pairs(split="test_prefs")
    train_pairs, train_stats = audit_and_filter_pairs(train_pairs, tokenizer, config.max_length)
    eval_pairs, eval_stats = audit_and_filter_pairs(eval_pairs, tokenizer, config.max_length)
    print(
        f"truncation audit: train dropped {train_stats.dropped_identical}/{train_stats.total}, "
        f"eval dropped {eval_stats.dropped_identical}/{eval_stats.total}"
    )

    def make_loader(pairs: list[PreferencePair], shuffle: bool) -> DataLoader:
        return DataLoader(
            PairDataset(pairs, tokenizer, config.max_length),
            batch_size=config.batch_size,
            shuffle=shuffle,
            collate_fn=lambda items: pair_collate(items, tokenizer.pad_token_id),
        )

    train_loader = make_loader(train_pairs, shuffle=True)
    eval_loader = make_loader(eval_pairs, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(enabled=config.use_amp)

    writer = SummaryWriter(log_dir=str(log_dir / config.experiment_name))
    checkpoint_path = checkpoint_dir / config.experiment_name
    best_acc = 0.0
    epochs_without_improvement = 0
    checkpoint_saved = False

    for epoch in range(config.max_epochs):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=config.use_amp):
                r_chosen = _rewards(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
                r_rejected = _rewards(
                    model, batch["rejected_input_ids"], batch["rejected_attention_mask"]
                )
                loss = bradley_terry_loss(r_chosen, r_rejected)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        eval_acc = evaluate_pairs(model, eval_loader, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("metrics/eval_pairwise_acc", eval_acc, epoch)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} eval_pairwise_acc={eval_acc:.4f}")

        if eval_acc > best_acc:
            best_acc = eval_acc
            epochs_without_improvement = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            checkpoint_saved = True
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    writer.close()
    if not checkpoint_saved:
        raise RuntimeError(
            "training completed but eval pairwise accuracy never improved past 0.0 — "
            "no checkpoint was saved"
        )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    path = train_reward(load_config(args.config), args.checkpoint_dir, args.log_dir)
    print(f"best checkpoint: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the two configs**

`training/configs/reward-lr2e5.yaml`:
```yaml
experiment_name: reward-lr2e5
model_name: microsoft/deberta-v3-base
max_length: 1024
batch_size: 8
learning_rate: 2.0e-5
warmup_ratio: 0.06
max_grad_norm: 1.0
max_epochs: 3
early_stopping_patience: 1
use_amp: true
seed: 42
```

`training/configs/reward-lr5e5.yaml`: identical except `experiment_name: reward-lr5e5` and `learning_rate: 5.0e-5`.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_train_reward_smoke.py -v`
Expected: 2 passed (the smoke test downloads deberta-v3-base once; subsequent runs hit the HF cache)

- [ ] **Step 6: Run full suite + ruff**

Run: `pytest -q && ruff check .`
Expected: all green

- [ ] **Step 7: Commit**

```bash
git add training/train_reward.py configs/reward-lr2e5.yaml configs/reward-lr5e5.yaml tests/test_train_reward_smoke.py
git commit -m "feat: add Bradley-Terry reward training loop with lr-sweep configs"
```

---

### Task 5: Temperature calibration

**Files:**
- Create: `training/training/calibrate.py` (pure fitting logic) and `training/calibrate_reward.py` (CLI script at package root, sibling of run_benchmark.py)
- Test: `training/tests/test_calibrate.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for post-hoc temperature fitting on reward margins."""
import math

import torch

from training.calibrate import fit_temperature


def test_temperature_widens_for_overconfident_margins() -> None:
    # Margins are huge but 25% of preferences are actually violated ->
    # NLL-optimal temperature must be large (>1) to soften the sigmoid.
    margins = torch.tensor([8.0, 8.0, 8.0, -8.0])  # last one: chosen LOST
    t = fit_temperature(margins)
    assert t > 1.0


def test_temperature_identity_when_calibrated() -> None:
    # Margins already near-calibrated -> fitted T should be close to 1.
    torch.manual_seed(0)
    # simulate margins drawn so that sigmoid(m) matches empirical win rate
    margins = torch.randn(2000) * 0.5 + 0.4
    t = fit_temperature(margins)
    assert 0.2 < t < 5.0  # sane, finite, positive


def test_temperature_positive_and_finite() -> None:
    t = fit_temperature(torch.tensor([1.0, 2.0, -0.5]))
    assert t > 0 and math.isfinite(t)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_calibrate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the fitting logic**

```python
"""Post-hoc temperature calibration for Bradley-Terry reward margins.

Bradley-Terry logits are arbitrarily scaled: a confident model can emit
rewards like +/-8, which a bare sigmoid flattens to 0.999/0.001 and
destroys score granularity. We fit a single scalar T on the ID validation
split by minimizing the NLL of sigma(margin / T) — every margin in the
calibration set is a (chosen - rejected) pair, so the "label" is always 1
(chosen preferred) and miscalibration shows up when large margins coexist
with preference violations (negative margins).

Fitting T on the same split used for ID evaluation is safe for the
headline pairwise-accuracy metric: accuracy is invariant under any
positive temperature (monotone transform).
"""
import torch


def fit_temperature(margins: torch.Tensor, steps: int = 200, lr: float = 0.05) -> float:
    """Fit scalar T minimizing mean softplus(-margins / T) via LBFGS-free Adam.

    Optimizes log_t for positivity. Margins: r_chosen - r_rejected on the
    calibration split.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.nn.functional.softplus(-margins / log_t.exp()).mean()
        loss.backward()
        optimizer.step()
    return float(log_t.exp().item())
```

- [ ] **Step 4: Write the CLI script** (`training/calibrate_reward.py`)

```python
"""Fit a calibration temperature for a trained reward checkpoint and write it
into the checkpoint's config.json (key: "reward_temperature"), so the platform
judge can read it alongside the weights.

Usage: python calibrate_reward.py --checkpoint checkpoints/reward-lr2e5
"""
import argparse
import json
from pathlib import Path

import torch

from training.data.preference import audit_and_filter_pairs, load_ultrafeedback_pairs
from training.calibrate import fit_temperature
from training.train_reward import PairDataset, pair_collate, _rewards
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    pairs = load_ultrafeedback_pairs(split="test_prefs")
    pairs, stats = audit_and_filter_pairs(pairs, tokenizer, args.max_length)
    print(f"calibration set: {len(pairs)} pairs (dropped {stats.dropped_identical})")

    loader = DataLoader(
        PairDataset(pairs, tokenizer, args.max_length),
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
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"fitted temperature {temperature:.4f} -> {config_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests + ruff, commit**

Run: `pytest tests/test_calibrate.py -v && ruff check .`
Expected: 3 passed, ruff clean

```bash
git add training/calibrate.py calibrate_reward.py tests/test_calibrate.py
git commit -m "feat: add post-hoc temperature calibration for reward margins"
```

---

### Task 6: Rating-pair export from the platform DB

**Files:**
- Create: `platform/api/evalforge/export_rating_pairs.py`
- Test: `platform/api/tests/test_export_rating_pairs.py`

Placement rationale: the exporter reads platform tables with the platform's
ORM models, so it lives in the platform package (run with the platform venv);
the training package only consumes its JSONL output. Zero new dependencies
for `training/`. This task runs from `platform/api`, not `training`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for exporting HumanRating rows as (prompt, chosen, rejected) JSONL."""
import json

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from evalforge.db.engine import init_db
from evalforge.db.models import (
    Base,
    CandidateModel,
    HumanRating,
    Prompt,
    PromptVersion,
    Result,
    ResultStatus,
    Run,
    RunStatus,
    Suite,
)
from evalforge.export_rating_pairs import export_pairs


@pytest.mark.asyncio
async def test_export_pairs_orientation_and_skip_filtering(tmp_path) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await init_db(engine)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        suite = Suite(name="s")
        prompt = Prompt(suite=suite)
        version = PromptVersion(
            prompt=prompt, version_number=1, input_text="What is 2+2?", expected_output="4"
        )
        model = CandidateModel(name="m", provider="demo")
        run = Run(suite=suite, status=RunStatus.COMPLETED, concurrency_limit=1,
                  completed_steps=2, total_steps=2)
        res_a = Result(run=run, prompt_version=version, candidate_model=model,
                       status=ResultStatus.OK, generated_text="4", latency_ms=1,
                       input_tokens=1, output_tokens=1, cost_usd=0.0)
        res_b = Result(run=run, prompt_version=version, candidate_model=model,
                       status=ResultStatus.OK, generated_text="5", latency_ms=1,
                       input_tokens=1, output_tokens=1, cost_usd=0.0)
        session.add_all([suite, prompt, version, model, run, res_a, res_b])
        await session.flush()
        # vote where B was shown first but A won: orientation must follow chosen_result_id
        session.add(HumanRating(prompt_version_id=version.id, result_a_id=res_b.id,
                                result_b_id=res_a.id, chosen_result_id=res_a.id))
        # skipped vote: must be excluded
        session.add(HumanRating(prompt_version_id=version.id, result_a_id=res_a.id,
                                result_b_id=res_b.id, chosen_result_id=None, skipped=True))
        await session.commit()

        out = tmp_path / "pairs.jsonl"
        count = await export_pairs(session, out)

    assert count == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}]
    await engine.dispose()
```

- [ ] **Step 2: Run to verify failure**

Run (from `platform/api`): `pytest tests/test_export_rating_pairs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evalforge.export_rating_pairs'`

- [ ] **Step 3: Write the implementation**

```python
"""Export rating-room preference pairs as JSONL for the training package.

Each non-skipped HumanRating becomes one line:
    {"prompt": <input_text>, "chosen": <winning text>, "rejected": <losing text>}
oriented by chosen_result_id (NOT by the a/b display order, which is
shuffled per rater). Skipped votes (chosen_result_id IS NULL) are excluded —
a skip is "no preference", not a preference.

Run with the platform venv (needs the evalforge package):
    cd platform/api
    .venv/Scripts/python.exe -m evalforge.export_rating_pairs --out ../../training/data/rating_pairs.jsonl
"""
import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evalforge.config import Settings
from evalforge.db.engine import make_engine, make_session_factory
from evalforge.db.models import HumanRating, PromptVersion, Result


async def export_pairs(session: AsyncSession, out_path: Path) -> int:
    ratings = (
        (await session.execute(select(HumanRating).where(HumanRating.chosen_result_id.is_not(None))))
        .scalars()
        .all()
    )
    # batch-fetch referenced rows (avoid per-rating N+1, same pattern as runs.py)
    result_ids = {r.result_a_id for r in ratings} | {r.result_b_id for r in ratings}
    results = {
        res.id: res
        for res in (await session.execute(select(Result).where(Result.id.in_(result_ids))))
        .scalars()
        .all()
    }
    version_ids = {r.prompt_version_id for r in ratings}
    versions = {
        v.id: v
        for v in (
            await session.execute(select(PromptVersion).where(PromptVersion.id.in_(version_ids)))
        )
        .scalars()
        .all()
    }

    lines: list[str] = []
    for rating in ratings:
        chosen_id = rating.chosen_result_id
        rejected_id = rating.result_b_id if chosen_id == rating.result_a_id else rating.result_a_id
        chosen = results.get(chosen_id)
        rejected = results.get(rejected_id)
        version = versions.get(rating.prompt_version_id)
        if chosen is None or rejected is None or version is None:
            continue  # dangling reference; skip rather than crash the export
        if chosen.generated_text is None or rejected.generated_text is None:
            continue
        lines.append(
            json.dumps(
                {
                    "prompt": version.input_text,
                    "chosen": chosen.generated_text,
                    "rejected": rejected.generated_text,
                },
                ensure_ascii=False,
            )
        )
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


async def _main(out: Path) -> None:
    settings = Settings()
    engine = make_engine(settings)
    factory = make_session_factory(engine)
    async with factory() as session:
        count = await export_pairs(session, out)
    print(f"exported {count} preference pairs -> {out}")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    asyncio.run(_main(parser.parse_args().out))
```

- [ ] **Step 4: Run tests + full platform suite + lint/type**

Run: `pytest tests/test_export_rating_pairs.py -v && pytest -q && ruff check . && mypy --strict evalforge`
Expected: new test passes, 73 total, ruff/mypy clean

- [ ] **Step 5: Commit**

```bash
git add evalforge/export_rating_pairs.py tests/test_export_rating_pairs.py
git commit -m "feat: export rating-room preference pairs as JSONL"
```

---

### Task 7: Reward evaluation script (ID + OOD probe)

**Files:**
- Create: `training/eval_reward.py` (script, package root)
- Test: `training/tests/test_eval_reward.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for reward-model evaluation report assembly (JSONL probe loading)."""
import json
from pathlib import Path

from training.data.preference import PreferencePair
from eval_reward import load_jsonl_pairs


def test_load_jsonl_pairs(tmp_path: Path) -> None:
    p = tmp_path / "probe.jsonl"
    p.write_text(
        json.dumps({"prompt": "q", "chosen": "a", "rejected": "b"}) + "\n",
        encoding="utf-8",
    )
    assert load_jsonl_pairs(p) == [PreferencePair(prompt="q", chosen="a", rejected="b")]


def test_load_jsonl_pairs_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_jsonl_pairs(p) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_eval_reward.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

```python
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
from training.train_reward import PairDataset, pair_collate, evaluate_pairs


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
    print(f"ID (UltraFeedback test_prefs, N={len(id_pairs)}): "
          f"pairwise accuracy = {acc_for(id_pairs):.4f}")

    if args.probe is not None:
        probe_pairs = load_jsonl_pairs(args.probe)
        if not probe_pairs:
            print("OOD probe: no pairs found — skipping")
        else:
            print(f"OOD probe (rating room, N={len(probe_pairs)} — a probe, not a benchmark): "
                  f"pairwise accuracy = {acc_for(probe_pairs):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests + ruff, commit**

Run: `pytest tests/test_eval_reward.py -v && pytest -q && ruff check .`
Expected: green

```bash
git add eval_reward.py tests/test_eval_reward.py
git commit -m "feat: add reward evaluation script with honest OOD probe reporting"
```

---

### Task 8: `reward` judge plugin in the platform

**Files:**
- Create: `platform/api/evalforge/judges/reward_judge.py`
- Modify: `platform/api/evalforge/judges/__init__.py` (register in `get_judge`, same lazy-import pattern as "deberta-hallucination")
- Modify: `platform/api/pyproject.toml` (add `reward` extra + mypy override module list entry if needed)
- Test: `platform/api/tests/test_reward_judge.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the reward judge (mocked model/tokenizer, no downloads)."""
from unittest.mock import MagicMock, patch

import pytest
import torch

from evalforge.config import Settings


def _make_judge():
    from evalforge.judges.reward_judge import RewardJudge

    return RewardJudge(Settings(database_url="sqlite+aiosqlite:///:memory:"))


@pytest.mark.asyncio
async def test_score_returns_calibrated_sigmoid_with_raw_in_justification() -> None:
    judge = _make_judge()
    with patch("evalforge.judges.reward_judge.AutoTokenizer") as mock_tok, patch(
        "evalforge.judges.reward_judge.AutoModelForSequenceClassification"
    ) as mock_model_cls:
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tok.from_pretrained.return_value = tokenizer
        model = MagicMock()
        model.config.reward_temperature = 2.0
        output = MagicMock()
        output.logits = torch.tensor([[4.0]])
        model.return_value = output
        model.to.return_value = model
        mock_model_cls.from_pretrained.return_value = model

        judgment = await judge.score("What is 2+2?", None, "4")

    assert judgment is not None
    expected = torch.sigmoid(torch.tensor(4.0 / 2.0)).item()
    assert abs(judgment.score - expected) < 1e-6
    assert "raw_reward=4.000" in judgment.justification
    assert "temperature=2.00" in judgment.justification


@pytest.mark.asyncio
async def test_score_empty_output_returns_none_without_loading() -> None:
    judge = _make_judge()
    with patch("evalforge.judges.reward_judge.AutoTokenizer") as mock_tok:
        judgment = await judge.score("prompt", None, "")
    assert judgment is None
    mock_tok.from_pretrained.assert_not_called()


def test_get_judge_registry() -> None:
    from evalforge.judges import get_judge

    # Import may fail if the extra isn't installed — but the registry must
    # know the name. With the extra installed (dev env), it constructs.
    judge = get_judge("reward", Settings(database_url="sqlite+aiosqlite:///:memory:"))
    assert judge.name == "reward"
```

- [ ] **Step 2: Run to verify failure**

Run (from `platform/api`): `pytest tests/test_reward_judge.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the judge**

```python
"""Preference reward judge: a fine-tuned DeBERTa reward model (Bradley-Terry
trained on UltraFeedback-binarized) scoring how preferred a response is,
published to Hugging Face Hub and run locally.

Requires the optional `reward` extra (same torch/transformers set as the
`deberta` extra). Module-level imports + lazy weight loading: identical
pattern and rationale as deberta_judge.py (see ADR-004).

Unlike every other judge, this one needs NO expected output — it scores any
(prompt, output) pair, so it works on suites without golden answers.
`expected` is accepted and ignored (protocol uniformity).

Score = sigmoid(raw_reward / T) where T is the calibration temperature fit
post-hoc on the ID validation split and stored in the checkpoint config as
`reward_temperature` (fallback 1.0 for uncalibrated checkpoints). The raw
reward is preserved in the justification string — Bradley-Terry logits are
the fine-grained signal; the sigmoid is a UI-friendly squash.
"""
import asyncio

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from evalforge.config import Settings
from evalforge.judges import Judgment

MODEL_ID = "DantheMan124/deberta-preference-reward"
MAX_LENGTH = 1024


class RewardJudge:
    name = "reward"

    def __init__(self, settings: Settings) -> None:
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._device: torch.device | None = None
        self._temperature: float = 1.0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self._model = model.to(self._device)  # type: ignore[arg-type]
        self._model.eval()
        self._temperature = float(getattr(self._model.config, "reward_temperature", 1.0))

    def _score_sync(self, prompt: str, output: str) -> tuple[float, float]:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        encoding = self._tokenizer(
            prompt, output, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            raw = self._model(**encoding).logits.squeeze().item()
        calibrated = torch.sigmoid(torch.tensor(raw / self._temperature)).item()
        return calibrated, raw

    async def score(self, prompt: str, expected: str | None, output: str) -> Judgment | None:
        if not output.strip():
            return None  # nothing to score; don't load the model for this
        calibrated, raw = await asyncio.to_thread(self._score_sync, prompt, output)
        return Judgment(
            score=calibrated,
            justification=f"raw_reward={raw:.3f}, temperature={self._temperature:.2f}",
        )
```

- [ ] **Step 4: Register in `judges/__init__.py`**

Mirror the existing "deberta-hallucination" branch in `get_judge` exactly — add:

```python
    if name == "reward":
        from evalforge.judges.reward_judge import RewardJudge

        return RewardJudge(settings)
```

(read the file first; place it adjacent to the deberta branch, matching its style.)

- [ ] **Step 5: Add the extra in `platform/api/pyproject.toml`**

Under `[project.optional-dependencies]`:

```toml
reward = ["torch>=2.3", "transformers>=4.42"]
```

- [ ] **Step 6: Run tests + full gates**

Run: `pytest tests/test_reward_judge.py -v && pytest -q && ruff check . && mypy --strict evalforge`
Expected: all green (mypy: reward_judge.py may need adding to the pyproject warn_unused_ignores override list — the module override currently names only deberta_judge; change it to `module = ["evalforge.judges.deberta_judge", "evalforge.judges.reward_judge"]` if the same both-worlds ignore situation arises)

- [ ] **Step 7: Commit**

```bash
git add evalforge/judges/reward_judge.py evalforge/judges/__init__.py pyproject.toml tests/test_reward_judge.py
git commit -m "feat: add preference reward judge behind optional reward extra"
```

---

### Task 9: Operational run + publish (controller/user, not subagent)

No new code — execution of the trained artifacts. Requires GPU time and user awareness.

- [ ] **Step 1: Dry run** (per the dry-run-before-real-run rule): temporarily slice `train_pairs = train_pairs[:100]`, `eval_pairs = eval_pairs[:50]` via a quick local edit, run `python -m training.train_reward --config configs/reward-lr2e5.yaml`, confirm: dataset downloads, truncation audit prints, loss decreases, checkpoint saves, eval accuracy prints. Revert the slice.
- [ ] **Step 2: Full lr sweep**: run both configs (each ~2–4h on the 3090; watch first-epoch VRAM — if OOM at batch 8/1024 tokens, drop to batch 4 and document).
- [ ] **Step 3: Calibrate** the best checkpoint: `python calibrate_reward.py --checkpoint checkpoints/<best>`.
- [ ] **Step 4: Export probe + evaluate**: run the platform exporter, then `python eval_reward.py --checkpoint checkpoints/<best> --probe data/rating_pairs.jsonl`. Record ID accuracy and probe result (with N) for the README.
- [ ] **Step 5: Publish** best checkpoint to `DantheMan124/deberta-preference-reward` (mirror publish_to_hub.py; honest model card: trained on AI-feedback data, ID accuracy, OOD probe N and result, temperature).
- [ ] **Step 6: Live smoke test** the reward judge through the platform (real run with `--judge reward` against an Ollama model).
- [ ] **Step 7: Update** root README (roadmap item → shipped; honest numbers) and training/README.md ("what didn't work" additions from the real runs).
- [ ] **Step 8: Merge** `feat/phase5-reward-model` → master after final whole-branch review; push.

---

## Self-review notes

- Spec coverage: decisions 1–7 map to Tasks 1 (loss), 2 (data/audit/flattening), 3 (model), 4 (loop/1024/AMP), 5 (calibration), 6 (export), 7 (eval/probe), 8 (judge/extra); operational plan = Task 9. ✓
- MODEL_ID `deberta-preference-reward` is used consistently in Task 8 and Task 9 Step 5. ✓
- `PairDataset`/`pair_collate`/`_rewards`/`evaluate_pairs` defined in Task 4, reused by Tasks 5 and 7 via import — signatures match. ✓
- Task 6 contains an inline path correction (exporter belongs to the platform package); the corrected file list is authoritative. ✓
