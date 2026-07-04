"""Tests for the local DeBERTa hallucination judge.

torch/transformers are an optional extra (`pip install -e ".[deberta]"`),
not installed in the base dev environment — these tests skip gracefully
(pytest.importorskip) rather than fail when the extra isn't present.
"""
import pytest

transformers = pytest.importorskip("transformers")

from unittest.mock import MagicMock, patch  # noqa: E402

from evalforge.config import Settings  # noqa: E402
from evalforge.judges import get_judge  # noqa: E402
from evalforge.judges.deberta_judge import DebertaJudge  # noqa: E402

SETTINGS = Settings()


def test_registry_includes_deberta():
    assert isinstance(get_judge("deberta-hallucination", SETTINGS), DebertaJudge)


@patch("evalforge.judges.deberta_judge.AutoModelForSequenceClassification")
@patch("evalforge.judges.deberta_judge.AutoTokenizer")
async def test_score_without_context_returns_none(mock_tok_cls, mock_model_cls):
    judge = DebertaJudge(SETTINGS)
    judgment = await judge.score(prompt="q", expected=None, output="a")
    assert judgment is None
    # model/tokenizer should never even be loaded if there's nothing to check against
    mock_model_cls.from_pretrained.assert_not_called()


@patch("evalforge.judges.deberta_judge.AutoModelForSequenceClassification")
@patch("evalforge.judges.deberta_judge.AutoTokenizer")
async def test_score_returns_faithfulness_not_hallucination_probability(
    mock_tok_cls, mock_model_cls
):
    import torch

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    mock_tok_cls.from_pretrained.return_value = fake_tokenizer

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_output = MagicMock()
    # logits favor class 1 (hallucinated) strongly: softmax gives P(hallucinated) close to 1
    fake_output.logits = torch.tensor([[0.0, 5.0]])
    fake_model.return_value = fake_output
    mock_model_cls.from_pretrained.return_value = fake_model

    judge = DebertaJudge(SETTINGS)
    judgment = await judge.score(
        prompt="What is the capital of France?",
        expected="France is in Europe. Its capital is Paris.",
        output="The capital of France is Berlin.",
    )

    assert judgment is not None
    # P(hallucinated) is high here, so score (= P(faithful)) must be LOW,
    # not high — this is the score-direction convention every other judge uses
    # (1.0 = good/correct, 0.0 = bad), and it's the inverse of the model's
    # raw output.
    assert judgment.score < 0.1


@patch("evalforge.judges.deberta_judge.AutoModelForSequenceClassification")
@patch("evalforge.judges.deberta_judge.AutoTokenizer")
async def test_model_loaded_once_and_reused_across_calls(mock_tok_cls, mock_model_cls):
    import torch

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    mock_tok_cls.from_pretrained.return_value = fake_tokenizer

    fake_model = MagicMock()
    fake_model.to.return_value = fake_model
    fake_output = MagicMock()
    fake_output.logits = torch.tensor([[5.0, 0.0]])
    fake_model.return_value = fake_output
    mock_model_cls.from_pretrained.return_value = fake_model

    judge = DebertaJudge(SETTINGS)
    await judge.score(prompt="q1", expected="c1", output="a1")
    await judge.score(prompt="q2", expected="c2", output="a2")

    # loaded once at construction/first use, not once per score() call
    assert mock_model_cls.from_pretrained.call_count == 1
