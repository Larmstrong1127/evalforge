from training.data.prepare import Example
from training.data.ragtruth import load_ragtruth_examples


def fake_load_fn(*args, **kwargs):
    """Stands in for datasets.load_dataset — returns raw RAGTruth-shaped rows."""
    return [
        {
            "source_info": "The Eiffel Tower is located in Paris, France.",
            "prompt": "Where is the Eiffel Tower?",
            "response": "The Eiffel Tower is in Paris.",
            "labels": [],  # no hallucination spans -> faithful
        },
        {
            "source_info": "The Eiffel Tower is located in Paris, France.",
            "prompt": "Where is the Eiffel Tower?",
            "response": "The Eiffel Tower is in Berlin.",
            "labels": [{"label_type": "Evident Conflict"}],  # has hallucination spans
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
