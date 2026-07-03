"""Thin wrappers around the pretrained DeBERTa classifier and its tokenizer.

Kept as one-line factory functions so tests can mock `from_pretrained`
entirely (no network access, no download) while asserting the exact
label configuration passed to the model.
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LABEL_NAMES = {0: "faithful", 1: "hallucinated"}
LABEL_IDS = {name: idx for idx, name in LABEL_NAMES.items()}


def build_model(model_name: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id=LABEL_IDS,
    )


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
