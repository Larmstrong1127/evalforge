"""Factory functions for the reward model: DeBERTa-v3 + 1-dim scalar head.

num_labels=1 makes AutoModelForSequenceClassification a regression head —
outputs.logits has shape (batch, 1) and is used directly as the scalar
reward. torch_dtype=torch.float32 for the same GradScaler reason documented
in models/classifier.py.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def build_reward_model(model_name: str) -> PreTrainedModel:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.float32,
    )


def build_reward_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_name)
