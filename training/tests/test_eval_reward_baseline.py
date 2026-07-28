"""Tests for the public-baseline harness (no model downloads).

`evaluate_baseline` itself needs 435M weights and the UltraFeedback split, so
it is exercised by hand (see the numbers recorded in
MODEL_CARD_preference_reward.md). What is worth pinning down in CI is the
guard that keeps the comparison honest.
"""
import pytest

from eval_reward_baseline import CHANCE_FLOOR, DEFAULT_BASELINE, evaluate_baseline


def test_chance_floor_is_the_balanced_binary_floor() -> None:
    """The pairwise task scores `r_chosen > r_rejected` — a coin flip is 0.5.

    Hardcoded deliberately: the card reported 0.7026 with no floor stated, so a
    reader had no way to tell a good number from a collapsed one.
    """
    assert CHANCE_FLOOR == 0.5


def test_default_baseline_is_the_declared_public_comparator() -> None:
    assert DEFAULT_BASELINE == "OpenAssistant/reward-model-deberta-v3-large-v2"


def test_multi_label_models_are_rejected_not_silently_argmaxed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A classifier head is not a reward head.

    Scoring a multi-label model by squeezing its logits would produce a
    plausible-looking accuracy from a meaningless quantity — exactly the class
    of error this whole baseline exists to guard against.
    """

    class _FakeConfig:
        num_labels = 3

        def to_dict(self) -> dict[str, object]:
            return {"max_position_embeddings": 512}

    class _FakeModel:
        config = _FakeConfig()

        def to(self, device: object) -> "_FakeModel":
            return self

    monkeypatch.setattr(
        "eval_reward_baseline.AutoTokenizer",
        type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}),
    )
    monkeypatch.setattr(
        "eval_reward_baseline.AutoModelForSequenceClassification",
        type("M", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeModel())}),
    )

    with pytest.raises(ValueError, match="num_labels=3"):
        evaluate_baseline("some/classifier", "test_prefs", 8)
