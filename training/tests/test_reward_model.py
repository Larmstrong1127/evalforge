"""Tests for the reward-model factory functions (mocked, no downloads)."""
from unittest.mock import patch

from training.models.reward import build_reward_model, build_reward_tokenizer


@patch("training.models.reward.AutoModelForSequenceClassification")
def test_build_reward_model_single_scalar_head(mock_cls) -> None:
    build_reward_model("microsoft/deberta-v3-base")
    _, kwargs = mock_cls.from_pretrained.call_args
    assert kwargs["num_labels"] == 1
    import torch

    assert kwargs["torch_dtype"] is torch.float32


@patch("training.models.reward.AutoTokenizer")
def test_build_reward_tokenizer(mock_tok) -> None:
    build_reward_tokenizer("microsoft/deberta-v3-base")
    mock_tok.from_pretrained.assert_called_once_with("microsoft/deberta-v3-base")
