"""Tests for the canonical answer-preserving encoding, plus the drift guard
that keeps the platform's serving mirror honest.

The invariant under test is blunt, because the defect was blunt: the ANSWER —
the only span the classifier is judging — must always be present in the
encoding, and it must be the last segment to lose anything.
"""
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from training.hallucination_encoding import budget_segments, encode_qca, legacy_encode_text

CLS = 101
SEP = 102

PLATFORM_MIRROR = (
    Path(__file__).resolve().parents[2]
    / "platform"
    / "api"
    / "evalforge"
    / "judges"
    / "hallucination_encoding.py"
)


class FakeTokenizer:
    """Whitespace tokenizer: one id per word, wrapped in [CLS] … [SEP]."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, 1000 + len(self.vocab))

    def decode_words(self, ids: list[int]) -> list[str]:
        reverse = {v: k for k, v in self.vocab.items()}
        return [reverse.get(i, "<special>") for i in ids]

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs: object) -> dict:
        ids = [self._id(w) for w in text.split()]
        if add_special_tokens:
            ids = [CLS, *ids, SEP]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _words(tokenizer: FakeTokenizer, encoding) -> list[str]:
    return tokenizer.decode_words([i for i in encoding["input_ids"] if i not in (CLS, SEP)])


# --- the budgeting rule -----------------------------------------------------


def test_everything_kept_when_it_all_fits() -> None:
    assert budget_segments([1, 2], [3, 4], [5, 6], budget=10) == ([1, 2], [3, 4], [5, 6])


def test_context_is_cut_first_and_the_answer_survives_whole() -> None:
    question, context, answer = budget_segments([1, 2], [3, 4, 5, 6], [7, 8], budget=6)
    assert (question, context, answer) == ([1, 2], [3, 4], [7, 8])


def test_question_is_trimmed_only_after_the_context_is_gone() -> None:
    question, context, answer = budget_segments([1, 2, 3, 4], [5, 6], [7, 8], budget=5)
    assert (question, context, answer) == ([1, 2, 3], [], [7, 8])


def test_answer_is_truncated_only_as_a_last_resort() -> None:
    assert budget_segments([1, 2], [3], [4, 5, 6, 7], budget=3) == ([], [], [4, 5, 6])


def test_budget_is_never_exceeded() -> None:
    for budget in range(0, 12):
        segments = budget_segments([1, 2, 3], [4, 5, 6, 7], [8, 9], budget)
        assert sum(len(s) for s in segments) <= max(budget, 0)


# --- end to end through a tokenizer ----------------------------------------


def test_input_that_fits_is_encoded_identically_to_the_original() -> None:
    tokenizer = FakeTokenizer()
    encoded = encode_qca(tokenizer, "who", "some context", "paris", max_length=64)
    flat = tokenizer(legacy_encode_text("who", "some context", "paris"), add_special_tokens=True)
    assert encoded["input_ids"] == flat["input_ids"]


def test_answer_survives_a_context_that_used_to_evict_it() -> None:
    tokenizer = FakeTokenizer()
    context = " ".join(f"filler{i}" for i in range(1000))
    encoded = encode_qca(tokenizer, "who is it", context, "the answer is paris", max_length=64)
    words = _words(tokenizer, encoded)
    for word in ["the", "answer", "is", "paris"]:
        assert word in words


def test_the_legacy_encoding_really_did_drop_that_answer() -> None:
    tokenizer = FakeTokenizer()
    context = " ".join(f"filler{i}" for i in range(1000))
    flat = tokenizer(
        legacy_encode_text("who is it", context, "the answer is paris"), add_special_tokens=False
    )["input_ids"]
    assert "paris" not in tokenizer.decode_words(flat[:62])


@pytest.mark.parametrize("max_length", [8, 16, 64, 512])
def test_max_length_is_respected(max_length: int) -> None:
    tokenizer = FakeTokenizer()
    context = " ".join(f"filler{i}" for i in range(2000))
    encoded = encode_qca(tokenizer, "q " * 30, context, "a " * 30, max_length=max_length)
    assert len(encoded["input_ids"]) <= max_length


# --- drift guard ------------------------------------------------------------


def _load_platform_mirror() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_platform_mirror", PLATFORM_MIRROR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_platform_mirror_exists() -> None:
    assert PLATFORM_MIRROR.exists(), (
        "the serving copy of the encoding is missing; the judge must not fall "
        "back to a hand-rolled encoding"
    )


def test_platform_mirror_agrees_on_the_budgeting_rule() -> None:
    mirror = _load_platform_mirror()
    for budget in range(0, 20):
        assert mirror.budget_segments([1, 2, 3], [4, 5, 6, 7], [8, 9], budget) == budget_segments(
            [1, 2, 3], [4, 5, 6, 7], [8, 9], budget
        )


def test_platform_mirror_produces_identical_encodings() -> None:
    """The platform package cannot import this one (separate distributions),
    so the two copies are kept in step by measurement rather than by trust —
    the same reason reward_judge mirrors reward_metadata."""
    mirror = _load_platform_mirror()
    cases = [
        ("who", "short context", "paris", 64),
        ("who is it", " ".join(f"w{i}" for i in range(1000)), "the answer is paris", 64),
        ("q " * 40, " ".join(f"w{i}" for i in range(300)), "a " * 40, 32),
        ("q", "", "a", 16),
        ("q", "c " * 200, " ".join(f"a{i}" for i in range(200)), 24),
    ]
    for question, context, answer, max_length in cases:
        assert (
            mirror.encode_qca(FakeTokenizer(), question, context, answer, max_length)["input_ids"]
            == encode_qca(FakeTokenizer(), question, context, answer, max_length)["input_ids"]
        )


def test_platform_mirror_shares_the_prefix_constants() -> None:
    mirror = _load_platform_mirror()
    assert mirror.legacy_encode_text("q", "c", "a") == legacy_encode_text("q", "c", "a")
