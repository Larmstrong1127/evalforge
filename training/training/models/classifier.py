"""Thin wrappers around the pretrained DeBERTa classifier and its tokenizer.

Kept as one-line factory functions so tests can mock `from_pretrained`
entirely (no network access, no download) while asserting the exact
label configuration passed to the model.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

LABEL_NAMES = {0: "faithful", 1: "hallucinated"}
LABEL_IDS = {name: idx for idx, name in LABEL_NAMES.items()}


def build_model(model_name: str) -> PreTrainedModel:
    # torch_dtype=torch.float32 is required, not cosmetic: some recent
    # transformers versions load weights in the checkpoint's stored dtype by
    # default when no dtype is specified. If that happens to be fp16, mixing
    # fp16 model weights with torch.amp.GradScaler (which assumes fp32
    # master weights during autocast) fails with "Attempting to unscale FP16
    # gradients." Forcing fp32 here is what makes AMP training actually work.
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id=LABEL_IDS,
        torch_dtype=torch.float32,
    )


def build_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
