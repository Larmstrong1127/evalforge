"""Diagnostic: audits the published RAGTruth benchmark agreement figure.

Re-runs the local DeBERTa judge over the EXACT same 200-example RAGTruth
sample used by run_benchmark.py (same SEED, same SAMPLE_SIZE, same
`random.Random(SEED).sample` call), and reports the measurement-bug
diagnostics that a bare accuracy number cannot distinguish:

* accuracy under both label polarities (inversion check)
* prediction and label class balance (degenerate-predictor check)
* ROC-AUC and a full threshold sweep on P(hallucinated) (operating-point
  check — AUC is polarity-symmetric around 0.5 and threshold-free, so it
  separates "no signal" from "wrong threshold")
* fraction of examples truncated at the checkpoint's trained max_length

Read-only: loads the checkpoint, writes nothing. Costs nothing (no API).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from training.data.prepare import Example
from training.data.ragtruth import load_ragtruth_examples
from training.hallucination_encoding import ANSWER_PREFIX, encode_qca, legacy_encode_text
from training.models.classifier import build_tokenizer

SAMPLE_SIZE = 200
SEED = 42
DEFAULT_CHECKPOINT = "checkpoints/lr-2e5"

LEGACY_ENCODING = "legacy"
ANSWER_PRESERVING_ENCODING = "answer-preserving"
ENCODINGS = (LEGACY_ENCODING, ANSWER_PRESERVING_ENCODING)


def encode_text(ex: Example) -> str:
    """The flat encoding the judge originally shipped with. Delegates to the
    shared module so this diagnostic cannot drift from what is served."""
    return legacy_encode_text(ex.question, ex.context, ex.answer)


def resolve_device(requested: str) -> torch.device:
    """`auto` picks CUDA when present; `cpu`/`cuda` force the choice.

    Forcing CPU is a real operating mode, not a debugging hack: this
    diagnostic is a 200-example single-sample sweep over a 184M-param model,
    which runs comfortably on CPU, and the GPU is frequently held by other
    work. The device does not affect the reported metrics.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def answer_tokens_lost(tokenizer: object, ex: Example, max_length: int, encoding: str) -> int:
    """How many tokens of the ANSWER the encoding drops.

    Under the legacy flat encoding the answer sits at the tail of the string,
    so right-truncation eats it from the end: everything past the budget is
    lost, and any of that overflow beyond the non-answer prefix is answer.
    The answer-preserving encoding sacrifices context first, so this is 0
    unless question+answer alone overflow the budget.
    """
    n_special = tokenizer.num_special_tokens_to_add(pair=False)  # type: ignore[attr-defined]
    budget = max_length - n_special
    answer_ids = tokenizer(  # type: ignore[operator]
        f"{ANSWER_PREFIX}{ex.answer}", add_special_tokens=False
    )["input_ids"]
    if encoding == LEGACY_ENCODING:
        flat_ids = tokenizer(  # type: ignore[operator]
            encode_text(ex), add_special_tokens=False
        )["input_ids"]
        overflow = max(0, len(flat_ids) - budget)
        return min(overflow, len(answer_ids))
    question_ids = tokenizer(  # type: ignore[operator]
        f"Q: {ex.question} C: ", add_special_tokens=False
    )["input_ids"]
    if len(question_ids) + len(answer_ids) <= budget:
        return 0
    if len(answer_ids) < budget:
        return 0
    return len(answer_ids) - max(0, budget)


def checkpoint_max_length(checkpoint: Path, fallback: int = 512) -> int:
    """Derive the trained sequence length from the checkpoint config rather
    than hardcoding it (same principle as training/reward_metadata.py)."""
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        return fallback
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("train_max_length", "max_position_embeddings"):
        value = config.get(key)
        if isinstance(value, int) and 0 < value <= 4096:
            return min(value, fallback) if key == "max_position_embeddings" else value
    return fallback


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """Rank-based AUC with tie correction. No sklearn dependency."""
    paired = sorted(zip(scores, labels, strict=True), key=lambda p: p[0])
    ranks: list[float] = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    positives = sum(1 for _, label in paired if label == 1)
    negatives = len(paired) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    positive_rank_sum = sum(r for r, (_, label) in zip(ranks, paired, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@dataclass(frozen=True)
class ThresholdPoint:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def sweep_thresholds(
    labels: list[int], scores: list[float], thresholds: list[float] | None = None
) -> list[ThresholdPoint]:
    """Sweep operating points. Defaults to the original fixed 0.01 grid so the
    historical numbers stay reproducible; pass `thresholds` for a data-driven
    sweep (see `sweep_score_thresholds`)."""
    if thresholds is None:
        thresholds = [step / 100.0 for step in range(1, 100)]
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        preds = [1 if s >= threshold else 0 for s in scores]
        tp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p == 1)
        fp = sum(1 for y, p in zip(labels, preds, strict=True) if y == 0 and p == 1)
        fn = sum(1 for y, p in zip(labels, preds, strict=True) if y == 1 and p == 0)
        accuracy = sum(1 for y, p in zip(labels, preds, strict=True) if y == p) / len(labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        points.append(ThresholdPoint(threshold, accuracy, precision, recall, f1))
    return points


def sweep_score_thresholds(labels: list[int], scores: list[float]) -> list[ThresholdPoint]:
    """Threshold sweep over the EMPIRICAL score values rather than a fixed
    0.01 grid.

    The fixed grid silently under-reports whenever a model's scores bunch
    into a narrow band: the answer-preserving encoding pushes almost every
    P(hallucinated) above 0.99, so a 0.01 grid has exactly one usable cut
    point inside the region where all the ranking information lives, and the
    best operating point it can find looks far worse than the ROC-AUC says it
    should be. Every distinct score is a candidate cut point, which is what
    an ROC curve is actually made of; this reports the true best achievable
    operating point.
    """
    candidates = sorted({s for s in scores})
    return sweep_thresholds(labels, scores, thresholds=candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default="ragtruth_diagnostic.json")
    parser.add_argument(
        "--encoding",
        choices=ENCODINGS,
        default=ANSWER_PRESERVING_ENCODING,
        help=(
            "how each (question, context, answer) triple is packed into the "
            "sequence budget. 'legacy' reproduces the original right-truncated "
            "flat string (which deletes the answer on long inputs); "
            "'answer-preserving' budgets the context and is what now ships."
        ),
    )
    parser.add_argument(
        "--merge-key",
        default=None,
        help=(
            "instead of replacing --out, load it and store this run under this "
            "top-level key, leaving every existing key untouched. This is how a "
            "re-measurement is recorded without destroying the historical "
            "numbers it is meant to be compared against."
        ),
    )
    parser.add_argument(
        "--protocol-note",
        default=None,
        help="free-text note stored alongside the run (what/when/where it was measured).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device; 'cpu' is fine for this 200-example sweep.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    max_length = checkpoint_max_length(checkpoint)

    all_examples = load_ragtruth_examples()
    sample = random.Random(SEED).sample(all_examples, SAMPLE_SIZE)
    labels = [ex.label for ex in sample]

    tokenizer = build_tokenizer(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    preds: list[int] = []
    scores: list[float] = []
    untruncated_lengths: list[int] = []
    answer_tokens_dropped: list[int] = []
    answer_token_counts: list[int] = []
    with torch.no_grad():
        for ex in sample:
            text = encode_text(ex)
            untruncated_lengths.append(len(tokenizer(text, truncation=False)["input_ids"]))
            answer_tokens_dropped.append(
                answer_tokens_lost(tokenizer, ex, max_length, args.encoding)
            )
            answer_token_counts.append(
                len(tokenizer(f"{ANSWER_PREFIX}{ex.answer}", add_special_tokens=False)["input_ids"])
            )
            if args.encoding == LEGACY_ENCODING:
                inputs = tokenizer(
                    text, truncation=True, max_length=max_length, return_tensors="pt"
                )
            else:
                inputs = encode_qca(
                    tokenizer, ex.question, ex.context, ex.answer, max_length,
                    return_tensors="pt",
                )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            probs = torch.softmax(model(**inputs).logits, dim=-1).squeeze(0)
            preds.append(int(probs.argmax().item()))
            scores.append(float(probs[1].item()))

    n = len(labels)
    accuracy = sum(1 for y, p in zip(labels, preds, strict=True) if y == p) / n
    accuracy_flipped = sum(1 for y, p in zip(labels, preds, strict=True) if y != p) / n
    auc = roc_auc(labels, scores)
    sweep = sweep_thresholds(labels, scores)
    best_acc = max(sweep, key=lambda p: p.accuracy)
    best_f1 = max(sweep, key=lambda p: p.f1)
    score_sweep = sweep_score_thresholds(labels, scores)
    best_acc_scores = max(score_sweep, key=lambda p: p.accuracy)
    best_f1_scores = max(score_sweep, key=lambda p: p.f1)
    truncated = sum(1 for length in untruncated_lengths if length > max_length)

    fully_lost = sum(
        1 for lost, total in zip(answer_tokens_dropped, answer_token_counts, strict=True)
        if total and lost >= total
    )
    partially_lost = sum(
        1 for lost, total in zip(answer_tokens_dropped, answer_token_counts, strict=True)
        if total and 0 < lost < total
    )

    report = {
        "checkpoint": str(checkpoint),
        "encoding": args.encoding,
        "device": str(device),
        "n": n,
        "max_length_used": max_length,
        "accuracy_as_published": accuracy,
        "accuracy_under_label_inversion": accuracy_flipped,
        "label_balance": {
            "hallucinated_1": sum(labels),
            "faithful_0": n - sum(labels),
            "positive_rate": sum(labels) / n,
        },
        "prediction_balance": {
            "predicted_hallucinated_1": sum(preds),
            "predicted_faithful_0": n - sum(preds),
            "positive_rate": sum(preds) / n,
        },
        "score_stats": {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / n,
            "n_distinct_rounded_2dp": len({round(s, 2) for s in scores}),
        },
        "roc_auc": auc,
        "best_accuracy": asdict(best_acc),
        "best_f1": asdict(best_f1),
        "best_accuracy_score_grid": asdict(best_acc_scores),
        "best_f1_score_grid": asdict(best_f1_scores),
        "at_threshold_0_5": asdict(next(p for p in sweep if abs(p.threshold - 0.5) < 1e-9)),
        "truncation": {
            "n_truncated": truncated,
            "fraction_truncated": truncated / n,
            "median_token_length": sorted(untruncated_lengths)[n // 2],
            "max_token_length": max(untruncated_lengths),
        },
        "answer_truncation": {
            "n_answer_fully_dropped": fully_lost,
            "fraction_answer_fully_dropped": fully_lost / n,
            "n_answer_partially_dropped": partially_lost,
            "fraction_answer_partially_dropped": partially_lost / n,
        },
    }

    # Subgroup check: if truncation were the cause of the lost signal, the
    # examples that fit entirely inside max_length should show a materially
    # higher AUC than the truncated ones.
    for group_name, keep in (
        ("untruncated", lambda length: length <= max_length),
        ("truncated", lambda length: length > max_length),
    ):
        idx = [i for i, length in enumerate(untruncated_lengths) if keep(length)]
        group_labels = [labels[i] for i in idx]
        if len(idx) < 2 or len(set(group_labels)) < 2:
            report[f"subgroup_{group_name}"] = {"n": len(idx), "roc_auc": None}
            continue
        group_scores = [scores[i] for i in idx]
        group_preds = [preds[i] for i in idx]
        report[f"subgroup_{group_name}"] = {
            "n": len(idx),
            "roc_auc": roc_auc(group_labels, group_scores),
            "accuracy_as_published": sum(
                1 for y, p in zip(group_labels, group_preds, strict=True) if y == p
            )
            / len(idx),
            "positive_rate": sum(group_labels) / len(idx),
            "majority_class_baseline": max(
                sum(group_labels), len(idx) - sum(group_labels)
            )
            / len(idx),
            "best_accuracy": asdict(
                max(sweep_thresholds(group_labels, group_scores), key=lambda p: p.accuracy)
            ),
            "best_f1": asdict(
                max(sweep_thresholds(group_labels, group_scores), key=lambda p: p.f1)
            ),
        }

    # Is the score driven by input length rather than by faithfulness?
    # Spearman-style rank correlation between token length and P(hallucinated),
    # computed as 2*AUC-1 over a median split of length.
    median_length = sorted(untruncated_lengths)[n // 2]
    long_flags = [1 if length > median_length else 0 for length in untruncated_lengths]
    report["length_confound"] = {
        "auc_length_predicts_label": roc_auc(labels, [float(x) for x in untruncated_lengths]),
        "auc_score_predicts_long": roc_auc(long_flags, scores),
        "majority_class_baseline_overall": max(sum(labels), n - sum(labels)) / n,
    }

    if args.protocol_note:
        report["protocol_note"] = args.protocol_note

    print(json.dumps(report, indent=2))
    out_path = Path(args.out)
    if args.merge_key:
        existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
        existing[args.merge_key] = report
        out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
