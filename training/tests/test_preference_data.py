"""Tests for UltraFeedback preference-pair loading and the truncation audit."""
from training.data.preference import (
    PreferencePair,
    audit_and_filter_pairs,
    flatten_messages,
    load_ultrafeedback_pairs,
)


def _fake_row(prompt: str, chosen: str, rejected: str) -> dict:
    # ultrafeedback_binarized stores chosen/rejected as message lists whose
    # last entry is the assistant completion.
    return {
        "prompt": prompt,
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected},
        ],
    }


def test_load_pairs_extracts_last_assistant_message() -> None:
    rows = [_fake_row("What is 2+2?", "4", "5")]
    pairs = load_ultrafeedback_pairs(load_fn=lambda split: rows)
    assert pairs == [PreferencePair(prompt="What is 2+2?", chosen="4", rejected="5")]


def test_load_pairs_skips_identical_completions() -> None:
    rows = [_fake_row("q", "same", "same"), _fake_row("q2", "a", "b")]
    pairs = load_ultrafeedback_pairs(load_fn=lambda split: rows)
    assert len(pairs) == 1
    assert pairs[0].prompt == "q2"


def test_flatten_messages_multi_turn() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and 2+2?"},
    ]
    assert flatten_messages(messages) == "user: hi\nassistant: hello\nuser: and 2+2?"


class _FakeTokenizer:
    """Encodes 1 token per whitespace-separated word; truncation keeps the
    first max_length words. Mirrors the only tokenizer behavior the audit
    relies on."""

    def __call__(self, text: str, pair: str, truncation: bool, max_length: int) -> dict:
        ids = list(range(len((text + " " + pair).split())))[:max_length]
        return {"input_ids": ids}


def test_audit_drops_pairs_identical_after_truncation() -> None:
    # Long shared prompt + differing tails: with a tiny budget both sides
    # truncate to the same prefix -> dropped. With a big budget -> kept.
    long_prompt = " ".join(["word"] * 10)
    pair = PreferencePair(prompt=long_prompt, chosen="ending one", rejected="different two")
    kept_small, stats_small = audit_and_filter_pairs([pair], _FakeTokenizer(), max_length=5)
    assert kept_small == []
    assert stats_small.dropped_identical == 1
    kept_large, stats_large = audit_and_filter_pairs([pair], _FakeTokenizer(), max_length=50)
    assert kept_large == [pair]
    assert stats_large.dropped_identical == 0
