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
    all_chosen: list[torch.Tensor] = []
    all_rejected: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            all_chosen.append(
                _rewards(model, batch["chosen_input_ids"], batch["chosen_attention_mask"]).cpu()
            )
            all_rejected.append(
                _rewards(
                    model, batch["rejected_input_ids"], batch["rejected_attention_mask"]
                ).cpu()
            )
    return pairwise_accuracy(torch.cat(all_chosen), torch.cat(all_rejected))


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
                r_chosen = _rewards(
                    model, batch["chosen_input_ids"], batch["chosen_attention_mask"]
                )
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
