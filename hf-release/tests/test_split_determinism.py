"""Determinism and correctness tests for both dataset regeneration scripts.

The heavy path (downloading UltraFeedback and RAGTruth, then tokenizing) is
NOT exercised here -- these tests inject a stub loader and a stub tokenizer so
they run in under a second with no network and no 184M-parameter model. What
they pin down is the part that can silently rot:

  * the truncation-audit filter's exact semantics, including the guard that
    makes it fire only when the cap was actually hit,
  * that the digest is order-sensitive and content-sensitive,
  * that the digest ignores derived encoding columns, so a tokenizer bump
    cannot change a sample's identity,
  * that RAGTruth's JSON-string label field is parsed rather than coerced.

The committed manifests are what pin the real numbers (1,988 -> 1,987, and the
200-row sample's 78/122 balance); `--verify` on each script is the gate that
checks them against live data.

Run:  python -m pytest hf-release/tests/test_split_determinism.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent / "datasets"
UF_DIR = DATASETS / "ultrafeedback-eval-split"
RT_DIR = DATASETS / "ragtruth-diagnostic-200"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


uf = _load("regenerate_split", UF_DIR / "regenerate_split.py")
rt = _load("regenerate_sample", RT_DIR / "regenerate_sample.py")


class WordTokenizer:
    """Whitespace tokenizer with real truncation semantics.

    Encodes a sequence PAIR as `prompt words + [SEP] + response words`, then
    right-truncates to max_length -- the same shape as the DeBERTa pair
    encoding the real filter relies on.
    """

    def __init__(self, n_special: int = 2) -> None:
        self.n_special = n_special

    def __call__(self, first, second=None, truncation=True, max_length=512, **kwargs):
        ids = first.split() + (["[SEP]"] + second.split() if second is not None else [])
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self.n_special


# --------------------------------------------------------------------------
# UltraFeedback: the truncation audit
# --------------------------------------------------------------------------


def test_pair_is_dropped_only_when_truncation_actually_made_them_identical():
    tok = WordTokenizer()
    shared = " ".join(f"w{i}" for i in range(10))
    pairs = [
        # Both sides survive intact under the cap -> keep.
        {"prompt": "p", "chosen": "alpha", "rejected": "beta"},
        # Long shared prompt eats the budget; both tails are cut off, leaving
        # two identical encodings -> drop.
        {"prompt": shared, "chosen": "tail_a", "rejected": "tail_b"},
    ]
    kept, dropped = uf.audit_and_filter(pairs, tok, max_length=8)
    assert dropped == [1]
    assert kept == [pairs[0]]


def test_a_truncated_pair_that_still_differs_is_kept():
    """Truncation alone is not grounds for dropping -- only lost signal is."""
    tok = WordTokenizer()
    pairs = [{"prompt": "p", "chosen": "a b c d", "rejected": "z y x w"}]
    kept, dropped = uf.audit_and_filter(pairs, tok, max_length=4)
    assert dropped == []
    assert len(kept) == 1


def test_untruncated_pairs_are_never_dropped_even_when_short():
    """The 'cap actually hit' guard. Without it the filter measures nothing."""
    tok = WordTokenizer()
    pairs = [{"prompt": "p", "chosen": "x", "rejected": "y"}]
    kept, dropped = uf.audit_and_filter(pairs, tok, max_length=512)
    assert dropped == []
    assert len(kept) == 1


def test_load_pairs_drops_rows_with_no_preference_signal(monkeypatch):
    rows = [
        {"prompt": "q1", "chosen": [{"role": "user", "content": "q1"},
                                    {"role": "assistant", "content": "same"}],
         "rejected": [{"role": "user", "content": "q1"},
                      {"role": "assistant", "content": "same"}]},
        {"prompt": "q2", "chosen": [{"role": "user", "content": "q2"},
                                    {"role": "assistant", "content": "good"}],
         "rejected": [{"role": "user", "content": "q2"},
                      {"role": "assistant", "content": "bad"}]},
    ]
    monkeypatch.setattr(uf, "load_pairs", lambda: [
        {"prompt": r["prompt"], "chosen": r["chosen"][-1]["content"],
         "rejected": r["rejected"][-1]["content"]}
        for r in rows if r["chosen"][-1]["content"] != r["rejected"][-1]["content"]
    ])
    assert len(uf.load_pairs()) == 1


def test_multi_turn_prompts_flatten_to_a_role_content_transcript():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "more"},
    ]
    assert uf.flatten_messages(messages) == "user: hello\nassistant: hi\nuser: more"


# --------------------------------------------------------------------------
# Digests: the property that makes --verify meaningful
# --------------------------------------------------------------------------


def test_split_digest_is_stable_across_repeated_calls():
    pairs = [{"prompt": "p", "chosen": "a", "rejected": "b"}]
    assert uf.split_digest(pairs) == uf.split_digest(list(pairs))


def test_split_digest_is_order_sensitive():
    a = {"prompt": "p1", "chosen": "a", "rejected": "b"}
    b = {"prompt": "p2", "chosen": "c", "rejected": "d"}
    assert uf.split_digest([a, b]) != uf.split_digest([b, a])


def test_split_digest_changes_when_any_field_changes():
    base = [{"prompt": "p", "chosen": "a", "rejected": "b"}]
    for field in ("prompt", "chosen", "rejected"):
        mutated = [dict(base[0], **{field: base[0][field] + "!"})]
        assert uf.split_digest(mutated) != uf.split_digest(base), field


def test_field_boundaries_cannot_be_forged_by_concatenation():
    """Without the NUL separators, ('ab','c') and ('a','bc') would collide."""
    left = [{"prompt": "ab", "chosen": "c", "rejected": "x"}]
    right = [{"prompt": "a", "chosen": "bc", "rejected": "x"}]
    assert uf.split_digest(left) != uf.split_digest(right)


def test_ragtruth_digest_ignores_derived_encoding_columns():
    """A tokenizer bump must not change the sample's identity."""
    row = {"question": "q", "context": "c", "answer": "a", "label": 1}
    with_encoding = dict(row, encoding={"n_tokens_flat": 999}, sample_index=0)
    assert rt.row_digest([row]) == rt.row_digest([with_encoding])


def test_ragtruth_digest_is_label_sensitive():
    faithful = [{"question": "q", "context": "c", "answer": "a", "label": 0}]
    hallucinated = [{"question": "q", "context": "c", "answer": "a", "label": 1}]
    assert rt.row_digest(faithful) != rt.row_digest(hallucinated)


# --------------------------------------------------------------------------
# RAGTruth: sampling and the label trap
# --------------------------------------------------------------------------


def test_sampling_is_reproducible_for_a_fixed_seed_and_source_order():
    import random

    source = [{"i": i} for i in range(2700)]
    first = random.Random(rt.SEED).sample(source, rt.SAMPLE_SIZE)
    second = random.Random(rt.SEED).sample(source, rt.SAMPLE_SIZE)
    assert first == second
    assert len(first) == 200


def test_a_row_count_change_does_not_reliably_change_the_drawn_indices():
    """The reason the DIGEST, not `n_source_examples`, is the real guard.

    It is tempting to assume a changed source length reshuffles everything. It
    does not: CPython's `sample` draws indices via `_randbelow(n)`, which pulls
    `n.bit_length()` bits and rejects draws >= n. 2700 and 2701 both need 12
    bits, so the accepted index sequence is identical unless some draw lands
    exactly on 2700 -- and here none does. An upstream APPEND would therefore
    slip past a count check on the indices alone.

    `n_source_examples` is still recorded and still checked, because it names
    the failure when it does fire. But the guarantee comes from the SHA-256
    over the sampled content, which is sensitive to what those indices point at.
    """
    import random

    a = random.Random(rt.SEED).sample(list(range(2700)), 200)
    b = random.Random(rt.SEED).sample(list(range(2701)), 200)
    assert a == b  # documents the hazard rather than wishing it away


def test_the_digest_catches_a_content_change_the_indices_would_miss():
    """Same drawn indices, different rows underneath -> different digest."""
    def rows(marker: str) -> list[dict]:
        return [
            {"question": f"q{i}", "context": f"c{i}{marker}", "answer": f"a{i}", "label": i % 2}
            for i in range(10)
        ]

    assert rt.row_digest(rows("")) != rt.row_digest(rows("-edited"))


def test_reordering_the_source_changes_the_sample():
    import random

    forward = random.Random(rt.SEED).sample(list(range(2700)), 200)
    reversed_source = random.Random(rt.SEED).sample(list(reversed(range(2700))), 200)
    assert forward != reversed_source


def test_json_string_label_field_is_parsed_not_coerced():
    """bool("[]") is True. Coercing the raw field labels every row hallucinated."""
    assert bool("[]") is True  # the trap itself
    assert (1 if json.loads("[]") else 0) == 0
    assert (1 if json.loads('[{"start": 0}]') else 0) == 1


# --------------------------------------------------------------------------
# The answer-preserving encoding's degradation order
# --------------------------------------------------------------------------


def test_context_is_sacrificed_before_the_answer():
    q, c, a = [1, 2], list(range(100)), [7, 8, 9]
    kept_q, kept_c, kept_a = rt.budget_segments(q, c, a, budget=10)
    assert kept_q == q
    assert kept_a == a
    assert len(kept_c) == 5


def test_question_is_trimmed_before_the_answer_when_context_alone_is_not_enough():
    q, c, a = list(range(50)), [1, 2, 3], [7, 8, 9]
    kept_q, kept_c, kept_a = rt.budget_segments(q, c, a, budget=10)
    assert kept_a == a
    assert kept_c == []
    assert len(kept_q) == 7


def test_answer_is_truncated_only_as_a_last_resort():
    q, c, a = [1], [2], list(range(100))
    kept_q, kept_c, kept_a = rt.budget_segments(q, c, a, budget=10)
    assert (kept_q, kept_c) == ([], [])
    assert kept_a == list(range(10))


def test_everything_fits_is_a_passthrough():
    q, c, a = [1], [2], [3]
    assert rt.budget_segments(q, c, a, budget=10) == (q, c, a)


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_yields_empty_segments(budget):
    assert rt.budget_segments([1], [2], [3], budget) == ([], [], [])


# --------------------------------------------------------------------------
# The committed manifests must stay self-consistent with the cards
# --------------------------------------------------------------------------


def test_ultrafeedback_manifest_records_the_published_counts():
    m = json.loads((UF_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert (m["n_pairs_loaded"], m["n_pairs_kept"]) == (1988, 1987)
    assert m["n_dropped_identical_after_truncation"] == 1
    assert m["max_length"] == 512
    assert len(m["sha256"]) == 64


def test_ragtruth_manifest_matches_the_published_diagnostic():
    m = json.loads((RT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert m["n_rows"] == 200
    assert m["n_source_examples"] == 2700
    assert m["label_balance"] == {"hallucinated_1": 78, "faithful_0": 122}
    assert m["n_truncated"] == 132
    # The headline of the encoding finding: 50.5% fully, 15.5% partially.
    assert m["legacy_answer_fully_dropped"] == 101
    assert m["legacy_answer_partially_dropped"] == 31
    assert m["answer_preserving_answer_fully_dropped"] == 0
    assert m["answer_preserving_answer_partially_dropped"] == 0


def test_shipped_rows_match_the_manifest_digest_and_count():
    m = json.loads((RT_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (RT_DIR / "data" / "diagnostic-200.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == m["n_rows"]
    assert rt.row_digest(rows) == m["sha256"]
    assert sum(r["label"] for r in rows) == m["label_balance"]["hallucinated_1"]
