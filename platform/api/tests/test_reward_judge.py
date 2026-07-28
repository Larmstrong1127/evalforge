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


async def test_score_returns_temperature_scaled_reward_with_raw_in_justification() -> None:
    """The score is `raw / T`, NOT `sigmoid(raw / T)`.

    Bradley-Terry rewards have an arbitrary additive offset and T was fit on
    pairwise margins, so squashing a bare logit produced a number that looked
    like a calibrated probability and wasn't. Keeping the scaled logit means
    a difference of two scores is the calibrated margin.
    """
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
        model.config.reward_train_max_length = 512
        output = MagicMock()
        output.logits = torch.tensor([[4.0]])
        model.return_value = output
        model.to.return_value = model
        mock_model_cls.from_pretrained.return_value = model

        judgment = await judge.score("What is 2+2?", None, "4")

    assert judgment is not None
    assert abs(judgment.score - 2.0) < 1e-6  # 4.0 / T=2.0, unsquashed
    assert judgment.justification is not None
    assert "raw_reward=4.000" in judgment.justification
    assert "temperature=2.00" in judgment.justification
    # The justification is the only place a dashboard reader learns the score
    # is relative; losing this wording silently restores the old misreading.
    assert "not an absolute quality probability" in judgment.justification
    # The sequence budget must come from the checkpoint, not a literal: the
    # judge shipped with a 1024 constant against a 512-token checkpoint.
    assert tokenizer.call_args.kwargs["max_length"] == 512


async def test_score_is_unbounded_not_a_probability() -> None:
    """A negative reward must stay negative.

    Guards the semantics change: any future re-squashing through a sigmoid
    would silently clamp this into (0, 1) and the score would start reading as
    a probability again.
    """
    judge = _make_judge()
    with patch("evalforge.judges.reward_judge.AutoTokenizer") as mock_tok, patch(
        "evalforge.judges.reward_judge.AutoModelForSequenceClassification"
    ) as mock_model_cls:
        tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.to.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.return_value = encoding
        mock_tok.from_pretrained.return_value = tokenizer
        model = MagicMock()
        model.config.reward_temperature = 1.0
        model.config.reward_train_max_length = 512
        output = MagicMock()
        output.logits = torch.tensor([[-2.5]])
        model.return_value = output
        model.to.return_value = model
        mock_model_cls.from_pretrained.return_value = model

        judgment = await judge.score("prompt", None, "a plausible but weak answer")

    assert judgment is not None
    assert abs(judgment.score - -2.5) < 1e-6


def test_resolve_max_length_prefers_explicit_training_key() -> None:
    from evalforge.judges.reward_judge import _resolve_max_length

    config = MagicMock()
    config.reward_train_max_length = 512
    config.max_position_embeddings = 1024
    assert _resolve_max_length(config) == 512


def test_resolve_max_length_falls_back_to_position_limit() -> None:
    """Checkpoints exported before the explicit key still resolve correctly."""

    class _Config:
        max_position_embeddings = 512

    from evalforge.judges.reward_judge import _resolve_max_length

    assert _resolve_max_length(_Config()) == 512


def test_resolve_max_length_raises_when_undeterminable() -> None:
    from evalforge.judges.reward_judge import _resolve_max_length

    with pytest.raises(ValueError, match="training sequence budget"):
        _resolve_max_length(object())


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
