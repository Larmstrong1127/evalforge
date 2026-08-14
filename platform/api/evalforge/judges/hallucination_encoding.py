"""Answer-preserving encoding for the hallucination judge — the SERVING copy.

Mirrors `training/training/hallucination_encoding.py`, which carries the full
rationale. The short version: the original encoding was one string truncated
from the right, and the tail of that string is the answer — the only span
being classified. On the 200-example RAGTruth benchmark sample the answer was
removed entirely on 51% of examples. This module budgets the CONTEXT instead
and keeps question and answer whole.

Why a mirror rather than an import: `evalforge` (the platform package) and
`training` are separate distributions installed into separate virtualenvs,
and the platform must not take a dependency on the training pipeline. This is
the same constraint that makes `reward_judge._resolve_max_length` a mirror of
`training/training/reward_metadata.resolve_max_length`. Drift between the two
copies is not left to vigilance: `training/tests/test_hallucination_encoding.py`
loads THIS file from disk and asserts it produces identical output to the
training copy across a battery of length regimes.

When the whole triple fits inside the budget this takes the single-string
path and is byte-identical to the original encoding, so the published
checkpoint's in-distribution behaviour is untouched.
"""
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

QUESTION_PREFIX = "Q: "
CONTEXT_PREFIX = " C: "
ANSWER_PREFIX = " A: "


def legacy_encode_text(question: str, context: str, answer: str) -> str:
    """The original flat encoding, kept because it defines the fast path."""
    return f"{QUESTION_PREFIX}{question}{CONTEXT_PREFIX}{context}{ANSWER_PREFIX}{answer}"


def budget_segments(
    question_ids: list[int],
    context_ids: list[int],
    answer_ids: list[int],
    budget: int,
) -> tuple[list[int], list[int], list[int]]:
    """Fit three token-id segments into `budget` tokens, sacrificing context
    first and the answer last. Pure — no tokenizer, no model, no I/O."""
    if budget <= 0:
        return [], [], []

    fixed = len(question_ids) + len(answer_ids)
    if fixed + len(context_ids) <= budget:
        return question_ids, context_ids, answer_ids

    if fixed <= budget:
        return question_ids, context_ids[: budget - fixed], answer_ids

    if len(answer_ids) < budget:
        return question_ids[: budget - len(answer_ids)], [], answer_ids

    return [], [], answer_ids[:budget]


def _special_token_wrapper(tokenizer: PreTrainedTokenizerBase) -> tuple[list[int], list[int]]:
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


def _finalize(
    tokenizer: PreTrainedTokenizerBase,
    content_ids: list[int],
    return_tensors: str | None,
) -> BatchEncoding:
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
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    context: str,
    answer: str,
    max_length: int,
    return_tensors: str | None = None,
) -> BatchEncoding:
    """Answer-preserving encoding of one (question, context, answer) triple."""
    prefix, suffix = _special_token_wrapper(tokenizer)
    budget = max_length - len(prefix) - len(suffix)

    flat = legacy_encode_text(question, context, answer)
    flat_ids: list[int] = tokenizer(flat, add_special_tokens=False)["input_ids"]
    if len(flat_ids) <= budget:
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
