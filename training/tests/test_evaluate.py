from unittest.mock import MagicMock, patch

import torch

from training.evaluate import EvaluationResult, evaluate_checkpoint


@patch("training.evaluate.build_tokenizer")
@patch("training.evaluate.AutoModelForSequenceClassification")
@patch("training.evaluate.load_ragtruth_examples")
def test_evaluate_checkpoint_reports_both_distributions(
    mock_ragtruth, mock_model_cls, mock_tokenizer
):
    from training.data.prepare import Example

    mock_ragtruth.return_value = [
        Example(question="q", context="c", answer="a", label=0),
        Example(question="q2", context="c2", answer="a2", label=1),
    ]

    fake_model = MagicMock()

    def fake_call(**kwargs):
        result = MagicMock()
        # logits favor class 0 for the first call pattern used below
        result.logits = torch.tensor([[2.0, 0.1]])
        return result

    fake_model.side_effect = fake_call
    fake_model.to.return_value = fake_model
    mock_model_cls.from_pretrained.return_value = fake_model

    fake_tok = MagicMock()
    fake_tok.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    mock_tokenizer.return_value = fake_tok

    result = evaluate_checkpoint(
        checkpoint_path="fake/path",
        val_examples=[Example(question="qv", context="cv", answer="av", label=0)],
    )

    assert "in_distribution" in result
    assert "out_of_distribution" in result
    assert isinstance(result["in_distribution"], EvaluationResult)
    assert isinstance(result["out_of_distribution"], EvaluationResult)
    assert 0.0 <= result["in_distribution"].f1 <= 1.0
    assert 0.0 <= result["in_distribution"].ece <= 1.0


@patch("training.evaluate.build_tokenizer")
@patch("training.evaluate.AutoModelForSequenceClassification")
@patch("training.evaluate.load_ragtruth_examples")
def test_evaluate_checkpoint_threads_custom_max_length(
    mock_ragtruth, mock_model_cls, mock_tokenizer
):
    from training.data.prepare import Example

    mock_ragtruth.return_value = [Example(question="q", context="c", answer="a", label=0)]

    fake_model = MagicMock()

    def fake_call(**kwargs):
        result = MagicMock()
        result.logits = torch.tensor([[2.0, 0.1]])
        return result

    fake_model.side_effect = fake_call
    fake_model.to.return_value = fake_model
    mock_model_cls.from_pretrained.return_value = fake_model

    fake_tok = MagicMock()
    fake_tok.return_value = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    mock_tokenizer.return_value = fake_tok

    evaluate_checkpoint(
        checkpoint_path="fake/path",
        val_examples=[Example(question="qv", context="cv", answer="av", label=0)],
        max_length=128,
    )

    for call in fake_tok.call_args_list:
        assert call.kwargs["max_length"] == 128
