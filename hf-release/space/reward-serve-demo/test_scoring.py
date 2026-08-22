"""Space handler tests against a stub model.

No Hub call, no checkpoint download, no Gradio process. torch is still imported
(the scorer uses it for sigmoid and no_grad), but nothing 184M-parameter-shaped
is ever constructed.

Run:  python -m pytest test_scoring.py -q
"""
from __future__ import annotations

import math

import pytest
import torch

from scoring import PairScore, RewardScorer, ScoringError, format_result, validate


class _StubConfig:
    reward_temperature = 2.0
    reward_train_max_length = 8


class _StubOutput:
    def __init__(self, value: float) -> None:
        self.logits = torch.tensor([[value]])


class _StubModel:
    """Returns a fixed score per response text, so margins are exact."""

    config = _StubConfig()

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, str]] = []

    def __call__(self, **kwargs):
        return _StubOutput(kwargs["_score"])


class _StubEncoding(dict):
    pass


class _StubTokenizer:
    def __init__(self, scores: dict[str, float], length: int = 4) -> None:
        self.scores = scores
        self.length = length

    def __call__(self, prompt, response, truncation=True, max_length=8, return_tensors=None):
        n = min(len(response.split()), max_length) if self.length is None else self.length
        return _StubEncoding(
            {"input_ids": torch.zeros(1, n, dtype=torch.long), "_score": self.scores[response]}
        )


def _scorer(scores: dict[str, float], length: int = 4) -> RewardScorer:
    s = RewardScorer(repo="stub")
    s._tokenizer = _StubTokenizer(scores, length)
    s._model = _StubModel(scores)
    s._temperature = _StubConfig.reward_temperature
    s._max_length = _StubConfig.reward_train_max_length
    return s


def test_margin_and_probability_are_computed_from_the_pair():
    s = _scorer({"good": 1.0, "bad": -1.0})
    result = s.score("q", "good", "bad")
    assert result.margin == pytest.approx(2.0)
    assert result.prob_a_preferred == pytest.approx(1 / (1 + math.exp(-2.0 / 2.0)))
    assert result.winner == "A"


def test_probability_is_symmetric_under_swapping_the_responses():
    """The single property that would break the ranking-only claim if violated."""
    s = _scorer({"good": 1.0, "bad": -1.0})
    forward = s.score("q", "good", "bad").prob_a_preferred
    reverse = s.score("q", "bad", "good").prob_a_preferred
    assert forward + reverse == pytest.approx(1.0)


def test_adding_a_constant_to_every_reward_leaves_the_verdict_unchanged():
    """Bradley-Terry invariance: the arbitrary zero point must not reach the output."""
    base = _scorer({"good": 1.0, "bad": -1.0}).score("q", "good", "bad")
    shifted = _scorer({"good": 101.0, "bad": 99.0}).score("q", "good", "bad")
    assert shifted.margin == pytest.approx(base.margin)
    assert shifted.prob_a_preferred == pytest.approx(base.prob_a_preferred)


def test_identical_scores_report_a_tie_not_a_winner():
    s = _scorer({"x": 0.5, "y": 0.5})
    assert s.score("q", "x", "y").winner == "tie"
    assert "Tie" in format_result(s.score("q", "x", "y"))


def test_temperature_and_budget_come_from_the_config_not_a_constant():
    s = _scorer({"a": 1.0, "b": 0.0})
    result = s.score("q", "a", "b")
    assert result.temperature == 2.0
    assert result.max_length == 8


def test_truncation_is_reported_when_the_budget_is_hit():
    s = _scorer({"a": 1.0, "b": 0.0}, length=_StubConfig.reward_train_max_length)
    result = s.score("q", "a", "b")
    assert result.truncated == {"a": True, "b": True}
    assert "Truncated" in format_result(result)


def test_short_inputs_are_not_flagged_as_truncated():
    s = _scorer({"a": 1.0, "b": 0.0}, length=3)
    assert s.score("q", "a", "b").truncated == {"a": False, "b": False}


@pytest.mark.parametrize(
    "args",
    [("", "a", "b"), ("q", "", "b"), ("q", "a", ""), ("q", "   ", "b")],
)
def test_empty_fields_are_rejected(args):
    with pytest.raises(ScoringError):
        validate(*args)


def test_oversized_input_is_rejected_before_the_model_is_touched():
    s = RewardScorer(repo="stub")  # never loaded
    with pytest.raises(ScoringError, match="limit is"):
        s.score("q", "x" * 20_001, "b")
    assert not s.loaded


def test_model_is_not_loaded_at_import_or_construction():
    """A CPU-tier Space that loads during import fails its own health check."""
    assert not RewardScorer(repo="stub").loaded


def test_format_result_never_presents_a_lone_score_as_a_quality_measure():
    body = format_result(
        PairScore(
            reward_a=-1.0902,
            reward_b=-2.3133,
            margin=1.2231,
            prob_a_preferred=0.740,
            temperature=1.1668,
            max_length=512,
        )
    )
    assert "arbitrary zero point" in body
    assert "Margin" in body
    assert "sign carries no meaning" in body
