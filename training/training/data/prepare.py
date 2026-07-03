"""HaluEval loading and preprocessing.

Builds two labeled examples per source row: the faithful answer (label 0)
and the hallucinated answer (label 1), so the classifier sees a balanced
faithful/hallucinated distribution by construction.

Dataset identifier note: verify `HALUEVAL_DATASET_ID` against the current
Hugging Face Hub listing before the first real run — dataset repo names on
the Hub can be renamed or reorganized between when this was written and when
it's actually executed.
"""
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

HALUEVAL_DATASET_ID = "pminervini/HaluEval"
HALUEVAL_CONFIG = "qa"


@dataclass(frozen=True)
class Example:
    question: str
    context: str
    answer: str
    label: int  # 0 = faithful, 1 = hallucinated


def _default_load_fn(*args: Any, **kwargs: Any) -> Any:
    import datasets

    return datasets.load_dataset(HALUEVAL_DATASET_ID, HALUEVAL_CONFIG, split="data")


def load_halueval_examples(load_fn: Callable[..., Any] = _default_load_fn) -> list[Example]:
    rows = load_fn()
    examples: list[Example] = []
    for row in rows:
        examples.append(
            Example(
                question=row["question"],
                context=row["knowledge"],
                answer=row["right_answer"],
                label=0,
            )
        )
        examples.append(
            Example(
                question=row["question"],
                context=row["knowledge"],
                answer=row["hallucinated_answer"],
                label=1,
            )
        )
    return examples


def split_train_val(
    examples: list[Example], val_ratio: float, seed: int
) -> tuple[list[Example], list[Example]]:
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    return shuffled[val_size:], shuffled[:val_size]
