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
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from training.config import TrainConfig, load_config
from training.data.prepare import Example, load_halueval_examples, split_train_val
from training.metrics import compute_classification_metrics
from training.models.classifier import build_model, build_tokenizer


def run_training_steps(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[dict],
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
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.label, dtype=torch.long),
        }


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device):
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
    train_loader = DataLoader(
        ExampleDataset(train_examples, tokenizer, config.max_length),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        ExampleDataset(val_examples, tokenizer, config.max_length),
        batch_size=config.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    writer = SummaryWriter(log_dir=str(log_dir / config.experiment_name))
    checkpoint_path = checkpoint_dir / f"{config.experiment_name}_best.pt"
    best_f1 = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.max_epochs):
        batches = [
            {
                "input_ids": b["input_ids"],
                "attention_mask": b["attention_mask"],
                "labels": b["labels"],
            }
            for b in train_loader
        ]
        losses = run_training_steps(
            model, optimizer, batches, device, config.max_grad_norm, config.use_amp, scheduler
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
            model.save_pretrained(checkpoint_dir / config.experiment_name)
            tokenizer.save_pretrained(checkpoint_dir / config.experiment_name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    writer.close()
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
