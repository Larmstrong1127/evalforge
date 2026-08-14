"""Single source of truth for how a (question, context, answer) triple is
encoded for the hallucination classifier — training, eval, and serving.

THE DEFECT THIS FIXES
---------------------
The original encoding was one string truncated from the right:

    tokenizer(f"Q: {q} C: {c} A: {a}", truncation=True, max_length=512)

Right-truncation drops the TAIL of that string, and the tail is the answer —
the only span actually being classified. Measured on the 200-example
RAGTruth benchmark sample (2026-08-09,
`training/scripts/diagnose_ragtruth_agreement.py`): the answer was removed
entirely on 51% of examples and partially on a further 15%. The model was
being asked to classify an answer it had never seen.

THE FIX
-------
Budget the CONTEXT — the one segment that is legitimately compressible,
because a retrieval context is a bag of evidence and losing its tail costs
some evidence — and keep the question and the answer whole. This mirrors
`training/training/reward_metadata.py`: the number that must agree between
train and serve is derived in one place, never restated at each call site.

BACKWARD COMPATIBILITY
----------------------
When the whole triple fits inside the budget, this function takes the
single-string path and produces an encoding byte-identical to the original.
The budgeted assembly only engages once truncation would otherwise happen.
So the published checkpoint's in-distribution behaviour (HaluEval, almost
all of which fits in 512 tokens) is unchanged; only the previously-broken
long-input path moves.

DEGRADATION ORDER when even question+answer will not fit:

1. context is dropped to empty,
2. then the question is truncated from the right,
3. and only as a last resort, when the answer alone exceeds the budget, is
   the answer itself truncated.

The answer is the last thing to go, never the first.

NOTE ON PLACEMENT: `platform/api/evalforge/judges/hallucination_encoding.py`
is a deliberate mirror of this module — the platform package cannot import
the training package (separate distributions, separate virtualenvs), the
same constraint that made `reward_judge._resolve_max_length` a mirror of
`reward_metadata.resolve_max_length`. Drift between the two copies is caught
by `training/tests/test_hallucination_encoding.py::test_platform_mirror_*`,
which loads the platform copy from disk and asserts identical output.
"""
from __future__ import annotations

from typing import Any

from transformers.tokenization_utils_base import BatchEncoding

QUESTION_PREFIX = "Q: "
CONTEXT_PREFIX = " C: "
ANSWER_PREFIX = " A: "


def legacy_encode_text(question: str, context: str, answer: str) -> str:
    """The original flat encoding. Kept because it defines the fast path and
    because the diagnostic re-measures against it."""
    return f"{QUESTION_PREFIX}{question}{CONTEXT_PREFIX}{context}{ANSWER_PREFIX}{answer}"


def budget_segments(
    question_ids: list[int],
    context_ids: list[int],
    answer_ids: list[int],
    budget: int,
) -> tuple[list[int], list[int], list[int]]:
    """Fit three token-id segments into `budget` tokens, sacrificing context
    first and the answer last. Pure — no tokenizer, no model, no I/O.

    `budget` is the room left for content after the tokenizer's special
    tokens have been accounted for. A non-positive budget yields three empty
    segments (there is nothing to keep).
    """
    if budget <= 0:
        return [], [], []

    fixed = len(question_ids) + len(answer_ids)
    if fixed + len(context_ids) <= budget:
        return question_ids, context_ids, answer_ids

    # Context is the compressible segment: give it whatever the question and
    # the answer do not need.
    if fixed <= budget:
        return question_ids, context_ids[: budget - fixed], answer_ids

    # Question + answer alone overflow. Drop the context entirely and trim
    # the question; the answer is the thing being classified, so it is the
    # last segment to lose anything.
    if len(answer_ids) < budget:
        return question_ids[: budget - len(answer_ids)], [], answer_ids

    # Last resort: even the answer alone does not fit.
    return [], [], answer_ids[:budget]


def _special_token_wrapper(tokenizer: Any) -> tuple[list[int], list[int]]:
    """The special tokens the tokenizer wraps a single sequence in, split into
    (prefix, suffix).

    Derived by encoding the empty string rather than hardcoded, because recent
    transformers releases removed `build_inputs_with_special_tokens` and
    `prepare_for_model` from the slow tokenizer classes, and the wrapper
    differs by model family ([CLS]/[SEP] here, <s>/</s> elsewhere). Encoding
    "" yields exactly the wrapper with no content between the halves.
    """
    wrapper: list[int] = tokenizer("", add_special_tokens=True)["input_ids"]
    half = len(wrapper) // 2
    return wrapper[:half], wrapper[half:]


def _finalize(tokenizer: Any, content_ids: list[int], return_tensors: str | None) -> Any:
    prefix, suffix = _special_token_wrapper(tokenizer)
    input_ids = prefix + content_ids + suffix
    return BatchEncoding(
        {
            "input_ids": [input_ids],
            "token_type_ids": [[0] * len(input_ids)],
            "attention_mask": [[1] * len(input_ids)],
        }
        if return_tensors
        else {
            "input_ids": input_ids,
            "token_type_ids": [0] * len(input_ids),
            "attention_mask": [1] * len(input_ids),
        },
        tensor_type=return_tensors,
    )


def encode_qca(
    tokenizer: Any,
    question: str,
    context: str,
    answer: str,
    max_length: int,
    return_tensors: str | None = None,
) -> Any:
    """Answer-preserving encoding of one (question, context, answer) triple.

    Returns a BatchEncoding with `input_ids`, `token_type_ids` and
    `attention_mask`, so callers can use it exactly like a plain
    `tokenizer(...)` call.
    """
    prefix, suffix = _special_token_wrapper(tokenizer)
    budget = max_length - len(prefix) - len(suffix)

    flat = legacy_encode_text(question, context, answer)
    flat_ids: list[int] = tokenizer(flat, add_special_tokens=False)["input_ids"]
    if len(flat_ids) <= budget:
        # Fits: identical to the original encoding, byte for byte.
        return _finalize(tokenizer, flat_ids, return_tensors)

    question_ids: list[int] = tokenizer(
        f"{QUESTION_PREFIX}{question}{CONTEXT_PREFIX}", add_special_tokens=False
    )["input_ids"]
    context_ids: list[int] = tokenizer(context, add_special_tokens=False)["input_ids"]
    answer_ids: list[int] = tokenizer(
        f"{ANSWER_PREFIX}{answer}", add_special_tokens=False
    )["input_ids"]

    kept_question, kept_context, kept_answer = budget_segments(
        question_ids, context_ids, answer_ids, budget
    )
    return _finalize(tokenizer, kept_question + kept_context + kept_answer, return_tensors)
