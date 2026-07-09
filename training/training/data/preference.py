"""UltraFeedback-binarized preference pairs: loading, flattening, truncation audit.

Dataset identifier note: verify ULTRAFEEDBACK_DATASET_ID against the Hub
before the first real run (same caveat as prepare.py's HaluEval id).

Multi-turn prompts are flattened to a single "role: content" transcript
string — DeBERTa has no chat template, and the reward head only needs the
textual context, not the turn structure.

Truncation audit: right-truncating completions under a token budget can
leave a pair's chosen and rejected encodings IDENTICAL (long shared prompt,
differing tails cut off). Training on such pairs asks the model to separate
two identical inputs — pure gradient noise. `audit_and_filter_pairs` drops
them and reports how many, so the run log records the data loss honestly.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

ULTRAFEEDBACK_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


@dataclass(frozen=True)
class AuditStats:
    total: int
    dropped_identical: int


def flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _default_load_fn(split: str) -> Any:
    import datasets

    return datasets.load_dataset(ULTRAFEEDBACK_DATASET_ID, split=split)


def load_ultrafeedback_pairs(
    split: str = "train_prefs", load_fn: Callable[[str], Any] = _default_load_fn
) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    for row in load_fn(split):
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        if chosen == rejected:
            continue  # no preference signal at all
        prompt_messages = row["chosen"][:-1]
        prompt = (
            flatten_messages(prompt_messages) if len(prompt_messages) > 1 else row["prompt"]
        )
        pairs.append(PreferencePair(prompt=prompt, chosen=chosen, rejected=rejected))
    return pairs


def audit_and_filter_pairs(
    pairs: list[PreferencePair], tokenizer: Any, max_length: int
) -> tuple[list[PreferencePair], AuditStats]:
    kept: list[PreferencePair] = []
    dropped = 0
    for pair in pairs:
        chosen_ids = tokenizer(
            pair.prompt, pair.chosen, truncation=True, max_length=max_length
        )["input_ids"]
        rejected_ids = tokenizer(
            pair.prompt, pair.rejected, truncation=True, max_length=max_length
        )["input_ids"]
        # Only a real problem if the cap was actually hit: if both encodings
        # come in under max_length, nothing was cut off, so equal encodings
        # are impossible here (identical completions were already filtered
        # out in load_ultrafeedback_pairs).
        truncated = len(chosen_ids) == max_length or len(rejected_ids) == max_length
        if truncated and chosen_ids == rejected_ids:
            dropped += 1
            continue
        kept.append(pair)
    return kept, AuditStats(total=len(pairs), dropped_identical=dropped)
