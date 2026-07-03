from training.data.prepare import Example, load_halueval_examples, split_train_val


def fake_load_fn(*args, **kwargs):
    """Stands in for datasets.load_dataset — returns raw HaluEval-shaped rows."""
    return [
        {
            "question": "What is the capital of France?",
            "knowledge": "France is a country in Europe. Its capital is Paris.",
            "right_answer": "Paris",
            "hallucinated_answer": "Lyon",
        },
        {
            "question": "Who wrote Hamlet?",
            "knowledge": "Hamlet is a tragedy written by William Shakespeare.",
            "right_answer": "William Shakespeare",
            "hallucinated_answer": "Christopher Marlowe",
        },
    ]


def test_load_halueval_produces_two_examples_per_row():
    examples = load_halueval_examples(load_fn=fake_load_fn)
    assert len(examples) == 4  # 2 rows x (faithful + hallucinated)


def test_load_halueval_labels_are_correct():
    examples = load_halueval_examples(load_fn=fake_load_fn)
    faithful = [e for e in examples if e.label == 0]
    hallucinated = [e for e in examples if e.label == 1]
    assert len(faithful) == 2
    assert len(hallucinated) == 2
    assert faithful[0].answer == "Paris"
    assert hallucinated[0].answer == "Lyon"


def test_load_halueval_preserves_question_and_context():
    examples = load_halueval_examples(load_fn=fake_load_fn)
    assert examples[0].question == "What is the capital of France?"
    assert "Paris" in examples[0].context


def test_split_train_val_ratio_and_determinism():
    examples = [
        Example(question=f"q{i}", context="c", answer="a", label=i % 2) for i in range(100)
    ]
    train_a, val_a = split_train_val(examples, val_ratio=0.1, seed=42)
    train_b, val_b = split_train_val(examples, val_ratio=0.1, seed=42)
    assert len(val_a) == 10
    assert len(train_a) == 90
    assert train_a == train_b  # same seed -> same split
    assert val_a == val_b


def test_split_train_val_different_seeds_differ():
    examples = [
        Example(question=f"q{i}", context="c", answer="a", label=i % 2) for i in range(100)
    ]
    train_a, _ = split_train_val(examples, val_ratio=0.1, seed=1)
    train_b, _ = split_train_val(examples, val_ratio=0.1, seed=2)
    assert train_a != train_b
