"""Tests for the reward judge (mocked model/tokenizer, no downloads).

torch/transformers are an optional extra (`pip install -e ".[reward]"`),
not installed in the base dev environment — these tests skip gracefully
(pytest.importorskip) rather than fail when the extra isn't present. Same
pattern as test_deberta_judge.py.
"""
import pytest

torch = pytest.importorskip("torch")

from unittest.mock import MagicMock, patch  # noqa: E402

from evalforge.config import Settings  # noqa: E402


def _make_judge():
    from evalforge.judges.reward_judge import RewardJudge

    return RewardJudge(Settings(database_url="sqlite+aiosqlite:///:memory:"))


async def test_score_returns_calibrated_sigmoid_with_raw_in_justification() -> None:
    judge = _make_judge()
    with patch("evalforge.judges.reward_judge.AutoTokenizer") as mock_tok, patch(
        "evalforge.judges.reward_judge.AutoModelForSequenceClassification"
    ) as mock_model_cls:
        tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.to.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        tokenizer.return_value = encoding
        mock_tok.from_pretrained.return_value = tokenizer
        model = MagicMock()
        model.config.reward_temperature = 2.0
        output = MagicMock()
        output.logits = torch.tensor([[4.0]])
        model.return_value = output
        model.to.return_value = model
        mock_model_cls.from_pretrained.return_value = model

        judgment = await judge.score("What is 2+2?", None, "4")

    assert judgment is not None
    expected = torch.sigmoid(torch.tensor(4.0 / 2.0)).item()
    assert abs(judgment.score - expected) < 1e-6
    assert judgment.justification is not None
    assert "raw_reward=4.000" in judgment.justification
    assert "temperature=2.00" in judgment.justification


async def test_score_empty_output_returns_none_without_loading() -> None:
    judge = _make_judge()
    with patch("evalforge.judges.reward_judge.AutoTokenizer") as mock_tok:
        judgment = await judge.score("prompt", None, "")
    assert judgment is None
    mock_tok.from_pretrained.assert_not_called()


def test_get_judge_registry() -> None:
    from evalforge.judges import get_judge

    judge = get_judge("reward", Settings(database_url="sqlite+aiosqlite:///:memory:"))
    assert judge.name == "reward"
