from training.data.prepare import Example
from training.data.ragtruth import load_ragtruth_examples


def fake_load_fn(*args, **kwargs):
    """Stands in for datasets.load_dataset — returns raw RAGTruth-shaped rows,
    matching the real wandb/RAGTruth-processed schema (verified 2026-07-02):
    hallucination_labels is a JSON-ENCODED STRING, not a real list."""
    return [
        {
            "context": "The Eiffel Tower is located in Paris, France.",
            "query": "Where is the Eiffel Tower?",
            "output": "The Eiffel Tower is in Paris.",
            "hallucination_labels": "[]",  # no hallucination spans -> faithful
        },
        {
            "context": "The Eiffel Tower is located in Paris, France.",
            "query": "Where is the Eiffel Tower?",
            "output": "The Eiffel Tower is in Berlin.",
            "hallucination_labels": '[{"label_type": "Evident Conflict"}]',  # has spans
        },
    ]


def test_load_ragtruth_returns_examples():
    examples = load_ragtruth_examples(load_fn=fake_load_fn)
    assert len(examples) == 2
    assert all(isinstance(e, Example) for e in examples)


def test_load_ragtruth_labels_by_presence_of_hallucination_spans():
    examples = load_ragtruth_examples(load_fn=fake_load_fn)
    assert examples[0].label == 0  # no labels -> faithful
    assert examples[1].label == 1  # has labels -> hallucinated


def test_load_ragtruth_maps_fields_correctly():
    examples = load_ragtruth_examples(load_fn=fake_load_fn)
    assert examples[0].question == "Where is the Eiffel Tower?"
    assert examples[0].answer == "The Eiffel Tower is in Paris."
    assert "Eiffel Tower" in examples[0].context


def test_load_ragtruth_does_not_treat_empty_json_string_as_truthy():
    """Regression test: hallucination_labels is a JSON string, and Python's
    bool("[]") is True (non-empty string) even though the parsed list is
    empty. Every row must be correctly labeled faithful if the parsed JSON
    is an empty list, not mislabeled hallucinated by raw string truthiness."""

    def only_empty_string_fn(*args, **kwargs):
        return [
            {
                "context": "c",
                "query": "q",
                "output": "a",
                "hallucination_labels": "[]",
            }
        ]

    examples = load_ragtruth_examples(load_fn=only_empty_string_fn)
    assert examples[0].label == 0
