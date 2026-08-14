"""Tests for the answer-preserving encoding used by the hallucination judge.

The defect these lock down: the judge used to encode `Q: … C: … A: …` as one
string and truncate it from the right, which deletes the ANSWER — the only
span being classified — on long inputs (51% of the RAGTruth benchmark
sample, entirely). The invariant is therefore blunt: **the answer is always
present in the encoding**, and it is the last segment to lose anything.
"""
import pytest

pytest.importorskip("transformers")

from evalforge.judges.hallucination_encoding import (  # noqa: E402
    budget_segments,
    encode_qca,
    legacy_encode_text,
)
from tests.fake_tokenizer import CLS, SEP, FakeTokenizer  # noqa: E402


def _words(tokenizer: FakeTokenizer, encoding: object) -> list[str]:
    ids = encoding["input_ids"]  # type: ignore[index]
    return tokenizer.decode_words([i for i in ids if i not in (CLS, SEP)])


# --- budget_segments: the pure budgeting rule -------------------------------


def test_everything_kept_when_it_all_fits() -> None:
    assert budget_segments([1, 2], [3, 4], [5, 6], budget=10) == ([1, 2], [3, 4], [5, 6])


def test_context_is_the_segment_that_gets_cut_first() -> None:
    question, context, answer = budget_segments([1, 2], [3, 4, 5, 6], [7, 8], budget=6)
    assert question == [1, 2]
    assert answer == [7, 8]
    assert context == [3, 4]  # trimmed from the right to fit, nothing else touched


def test_context_is_dropped_entirely_before_the_question_is_touched() -> None:
    question, context, answer = budget_segments([1, 2, 3], [4, 5, 6], [7, 8], budget=5)
    assert (question, context, answer) == ([1, 2, 3], [], [7, 8])


def test_question_is_trimmed_only_after_context_is_gone() -> None:
    question, context, answer = budget_segments([1, 2, 3, 4], [5, 6], [7, 8], budget=5)
    assert answer == [7, 8]  # answer survives whole
    assert context == []
    assert question == [1, 2, 3]  # question absorbed the remaining shortfall


def test_answer_is_only_truncated_as_a_last_resort() -> None:
    question, context, answer = budget_segments([1, 2, 3], [4, 5], [6, 7, 8, 9], budget=3)
    assert (question, context) == ([], [])
    assert answer == [6, 7, 8]  # the answer alone overflows; only then is it cut


def test_budget_is_never_exceeded_across_regimes() -> None:
    for budget in range(0, 12):
        question, context, answer = budget_segments([1, 2, 3], [4, 5, 6, 7], [8, 9], budget)
        assert len(question) + len(context) + len(answer) <= max(budget, 0)


def test_non_positive_budget_yields_nothing() -> None:
    assert budget_segments([1], [2], [3], budget=0) == ([], [], [])
    assert budget_segments([1], [2], [3], budget=-5) == ([], [], [])


# --- encode_qca: end-to-end through a tokenizer -----------------------------


def test_short_input_is_encoded_exactly_like_the_original_flat_string() -> None:
    tokenizer = FakeTokenizer()
    encoded = encode_qca(tokenizer, "who", "ctx here", "paris", max_length=64)
    flat = tokenizer(legacy_encode_text("who", "ctx here", "paris"), add_special_tokens=True)
    assert encoded["input_ids"] == flat["input_ids"]


def test_answer_survives_a_context_that_would_have_evicted_it() -> None:
    tokenizer = FakeTokenizer()
    context = " ".join(f"filler{i}" for i in range(500))
    encoded = encode_qca(tokenizer, "who is it", context, "the answer is paris", max_length=64)
    words = _words(tokenizer, encoded)
    # The whole point: every answer word is present despite a 500-word context.
    for word in ["the", "answer", "is", "paris"]:
        assert word in words
    assert "A:" in words


def test_legacy_encoding_would_have_dropped_that_same_answer() -> None:
    """Guards the premise, not just the fix: without this, a bug that made the
    context short would make the test above pass vacuously."""
    tokenizer = FakeTokenizer()
    context = " ".join(f"filler{i}" for i in range(500))
    flat_ids = tokenizer(
        legacy_encode_text("who is it", context, "the answer is paris"), add_special_tokens=False
    )["input_ids"]
    truncated = flat_ids[:62]
    assert "paris" not in tokenizer.decode_words(truncated)


def test_encoding_respects_max_length_including_special_tokens() -> None:
    tokenizer = FakeTokenizer()
    context = " ".join(f"filler{i}" for i in range(2000))
    for max_length in (8, 16, 64, 512):
        encoded = encode_qca(tokenizer, "q " * 20, context, "a " * 20, max_length=max_length)
        assert len(encoded["input_ids"]) <= max_length


def test_absurdly_long_answer_is_truncated_but_still_fills_the_window() -> None:
    tokenizer = FakeTokenizer()
    answer = " ".join(f"word{i}" for i in range(300))
    encoded = encode_qca(tokenizer, "q", "some context", answer, max_length=32)
    assert len(encoded["input_ids"]) == 32
    assert "word0" in _words(tokenizer, encoded)


def test_empty_context_is_handled() -> None:
    tokenizer = FakeTokenizer()
    encoded = encode_qca(tokenizer, "who", "", "paris", max_length=64)
    assert "paris" in _words(tokenizer, encoded)


def test_return_tensors_produces_a_batch_dimension() -> None:
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    encoded = encode_qca(tokenizer, "who", "ctx", "paris", max_length=64, return_tensors="pt")
    assert isinstance(encoded["input_ids"], torch.Tensor)
    assert encoded["input_ids"].shape[0] == 1
    assert encoded["attention_mask"].shape == encoded["input_ids"].shape
