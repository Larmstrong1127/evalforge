"""Hand-written PyTorch training loop.

Not HuggingFace `Trainer`, not `accelerate` — every step (zero_grad,
forward, backward, clip, optimizer step, scheduler step) is explicit, so the
training mechanics are fully inspectable and debuggable. Mixed precision uses
the modern `torch.amp` API (`torch.cuda.amp.*` is deprecated in PyTorch 2.x
and must not be used).

Interrupted runs (process killed mid-training) are not resumed — this is an
accepted v1 tradeoff; there is no checkpoint-of-optimizer-state resumption,
only the best-validation-F1 model weights are saved.
"""
import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorWithPadding, get_linear_schedule_with_warmup

from training.config import TrainConfig, load_config
from training.data.prepare import Example, load_halueval_examples, split_train_val
from training.metrics import ClassificationMetrics, compute_classification_metrics
from training.models.classifier import build_model, build_tokenizer


def run_training_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[dict],
    device: torch.device,
    max_grad_norm: float,
    use_amp: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> list[float]:
    """Runs one optimizer step per batch. Returns the loss for each step.

    Factored out from the full `train()` orchestrator so it can be exercised
    directly in tests without touching real data loading, checkpointing, or
    TensorBoard logging.
    """
    model.to(device)
    model.train()
    # device.type (not a hardcoded "cuda") so this is correct whether the
    # caller passes a CPU or CUDA device — enabled=False makes both a no-op
    # regardless, but the device_type argument must still match reality.
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    losses: list[float] = []

    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())

    return losses


class ExampleDataset(Dataset):
    """Tokenizes with truncation but WITHOUT padding — padding happens
    per-batch via DataCollatorWithPadding instead (see `_make_loader`).

    DeBERTa's disentangled attention computes content-to-position AND
    position-to-content matrices in addition to standard content-to-content
    attention, making it substantially more memory-hungry than plain BERT at
    a given sequence length. Padding every example to a fixed `max_length`
    regardless of its actual length (most HaluEval examples are far shorter
    than 512 tokens) wastes enormous activation memory and was the direct
    cause of a real CUDA OOM at batch_size=64/max_length=512 on a 24GB GPU
    during the first live training run. Dynamic per-batch padding (each
    batch padded only to its own longest example) is the standard fix.
    """

    def __init__(self, examples: list[Example], tokenizer, max_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        text = f"Q: {ex.question} C: {ex.context} A: {ex.answer}"
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": ex.label,
        }


def _make_loader(
    examples: list[Example], tokenizer, max_length: int, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        ExampleDataset(examples, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer, return_tensors="pt"),
    )


def evaluate_loader(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> ClassificationMetrics:
    """Returns a training.metrics.ClassificationMetrics (dataclass, not dict)."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return compute_classification_metrics(y_true, y_pred)


def train(config: TrainConfig, checkpoint_dir: Path, log_dir: Path) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    tokenizer = build_tokenizer(config.model_name)
    model = build_model(config.model_name)

    all_examples = load_halueval_examples()
    train_examples, val_examples = split_train_val(all_examples, val_ratio=0.1, seed=config.seed)
    train_loader = _make_loader(
        train_examples, tokenizer, config.max_length, config.batch_size, shuffle=True
    )
    val_loader = _make_loader(
        val_examples, tokenizer, config.max_length, config.batch_size, shuffle=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Scheduler is built for the full max_epochs run. If early stopping fires
    # before max_epochs, the LR schedule will not reach its planned decay
    # endpoint — accepted v1 simplification, not a bug.
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    writer = SummaryWriter(log_dir=str(log_dir / config.experiment_name))
    checkpoint_path = checkpoint_dir / config.experiment_name
    best_f1 = 0.0
    epochs_without_improvement = 0
    checkpoint_saved = False

    for epoch in range(config.max_epochs):
        losses = run_training_steps(
            model, optimizer, train_loader, device, config.max_grad_norm, config.use_amp, scheduler
        )
        train_loss = sum(losses) / len(losses)
        val_metrics = evaluate_loader(model, val_loader, device)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("metrics/val_f1", val_metrics.f1, epoch)
        writer.add_scalar("metrics/val_precision", val_metrics.precision, epoch)
        writer.add_scalar("metrics/val_recall", val_metrics.recall, epoch)

        if val_metrics.f1 > best_f1:
            best_f1 = val_metrics.f1
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
            "training completed but validation F1 never improved past 0.0 — "
            "no checkpoint was saved"
        )

    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.checkpoint_dir, args.log_dir)


if __name__ == "__main__":
    main()
