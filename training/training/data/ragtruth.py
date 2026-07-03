"""RAGTruth loading — used exclusively as a held-out, out-of-distribution
evaluation set. Never touched during training or hyperparameter selection.

Dataset identifier note: verify `RAGTRUTH_DATASET_ID` against the current
Hugging Face Hub listing before the first real run.
"""
from collections.abc import Callable
from typing import Any

from training.data.prepare import Example

RAGTRUTH_DATASET_ID = "wandb/RAGTruth-processed"


def _default_load_fn(*args: Any, **kwargs: Any) -> Any:
    import datasets

    return datasets.load_dataset(RAGTRUTH_DATASET_ID, split="test")


def load_ragtruth_examples(load_fn: Callable[..., Any] = _default_load_fn) -> list[Example]:
    rows = load_fn()
    examples: list[Example] = []
    for row in rows:
        label = 1 if row["labels"] else 0
        examples.append(
            Example(
                question=row["prompt"],
                context=row["source_info"],
                answer=row["response"],
                label=label,
            )
        )
    return examples
