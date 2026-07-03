"""RAGTruth loading — used exclusively as a held-out, out-of-distribution
evaluation set. Never touched during training or hyperparameter selection.

Verified against the live `wandb/RAGTruth-processed` schema on 2026-07-02:
columns are id/query/context/output/task_type/quality/model/temperature/
hallucination_labels/hallucination_labels_processed/input_str. Critically,
`hallucination_labels` is a JSON-encoded STRING (e.g. "[]"), not a real list
— `bool("[]")` is True in Python, so checking truthiness on the raw field
would silently label every single row as hallucinated. It must be parsed
with json.loads() first.
"""
import json
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
        spans = json.loads(row["hallucination_labels"])
        label = 1 if spans else 0
        examples.append(
            Example(
                question=row["query"],
                context=row["context"],
                answer=row["output"],
                label=label,
            )
        )
    return examples
