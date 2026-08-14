"""Evaluates a trained checkpoint against both the held-out HaluEval
validation split (in-distribution) and RAGTruth (out-of-distribution).
Reporting both side by side is the point: the gap between them is the
real generalization signal.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from training.data.prepare import Example, load_halueval_examples, split_train_val
from training.data.ragtruth import load_ragtruth_examples
from training.hallucination_encoding import encode_qca
from training.metrics import compute_classification_metrics, expected_calibration_error
from training.models.classifier import build_tokenizer


@dataclass(frozen=True)
class EvaluationResult:
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    ece: float


def _score_examples(
    model,
    tokenizer,
    examples: list[Example],
    device: torch.device,
    max_length: int = 512,
) -> EvaluationResult:
    model.to(device)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []
    correct: list[bool] = []

    with torch.no_grad():
        for ex in examples:
            inputs = encode_qca(
                tokenizer, ex.question, ex.context, ex.answer, max_length, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            pred = int(probs.argmax().item())
            confidence = float(probs[pred].item())

            y_true.append(ex.label)
            y_pred.append(pred)
            confidences.append(confidence)
            correct.append(pred == ex.label)

    classification_metrics = compute_classification_metrics(y_true, y_pred)
    ece = expected_calibration_error(confidences, correct)
    return EvaluationResult(
        precision=classification_metrics.precision,
        recall=classification_metrics.recall,
        f1=classification_metrics.f1,
        confusion_matrix=classification_metrics.confusion_matrix,
        ece=ece,
    )


def evaluate_checkpoint(
    checkpoint_path: str, val_examples: list[Example], max_length: int = 512
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

    ragtruth_examples = load_ragtruth_examples()

    return {
        "in_distribution": _score_examples(
            model, tokenizer, val_examples, device, max_length=max_length
        ),
        "out_of_distribution": _score_examples(
            model, tokenizer, ragtruth_examples, device, max_length=max_length
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"error: checkpoint not found: {args.checkpoint}")

    all_examples = load_halueval_examples()
    _, val_examples = split_train_val(all_examples, val_ratio=0.1, seed=42)

    results = evaluate_checkpoint(str(args.checkpoint), val_examples)
    for distribution, result in results.items():
        print(f"\n{distribution}:")
        print(f"  precision: {result.precision}")
        print(f"  recall: {result.recall}")
        print(f"  f1: {result.f1}")
        print(f"  ece: {result.ece}")
        print(f"  confusion_matrix: {result.confusion_matrix}")


if __name__ == "__main__":
    main()
